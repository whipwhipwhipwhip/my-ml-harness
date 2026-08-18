"""harness/loop.py, driven with fakes.

No filesystem, no run directory, no GPT. The loop's contract is narrow enough that
a nine-line model and a list-appending logger are complete implementations of it —
if that stops being true, the loop has grown a dependency it shouldn't have, and
these tests get harder to write. That difficulty is the signal.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from ml_learning.harness.testing import ARCHES, DATAS, FakeLogger, SpyEval, load_data
from ml_learning.harness.loop import evaluate, train
from ml_learning.harness.registry import load_run_config

DEVICE = torch.device("cpu")


class StubModel(nn.Module):
    """A complete implementation of what the loop asks a model for.

    Deliberately not `harness.testing.TinyModel`: that one is the *reference
    architecture*, realistic enough to checkpoint and sample from. This is the
    opposite claim — that the loop's contract is narrow enough for an embedding
    and a cross-entropy to satisfy it. Keeping them separate is what makes the
    claim visible.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@pytest.fixture
def harness(config_path, corpus_path):
    """Everything `train()` needs, with the config as the single knob."""

    def build(*overrides):
        cfg = load_run_config(config_path, ARCHES, DATAS, overrides)
        data = load_data(corpus_path, cfg.data.split_frac)
        model = StubModel(data.tokenizer.vocab_size)
        optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
        eval_batches = {
            split: data.get_batch(
                split, cfg.train.batch_size, cfg.model.block_size, "cpu",
                cfg.train.eval_iters, torch.Generator().manual_seed(0),
            )
            for split in data.splits
        }
        return model, optimiser, data, cfg, eval_batches

    return build


def run_loop(harness, *overrides):
    model, optimiser, data, cfg, eval_batches = harness(*overrides)
    logger = FakeLogger()
    on_eval = SpyEval(logger)
    result = train(
        model, optimiser, data, cfg, DEVICE, eval_batches, logger, on_eval
    )
    return result, logger, on_eval, model


# ---- when eval fires ------------------------------------------------------

def test_eval_fires_at_interval_and_at_the_last_step(harness):
    """max_iters=4, eval_interval=2 -> 0, 2, and 3 as the final-step case."""
    _, _, on_eval, _ = run_loop(harness, "train.max_iters=4", "train.eval_interval=2")
    assert [step for step, _ in on_eval.calls] == [0, 2, 3]


def test_final_step_evaluates_even_when_the_interval_would_skip_it(harness):
    _, _, on_eval, _ = run_loop(harness, "train.max_iters=5", "train.eval_interval=100")
    assert [step for step, _ in on_eval.calls] == [0, 4]


def test_eval_fires_more_than_once_in_smoke_sizing(harness):
    """Smoke's whole reason for 4 iters: `val < best_val` must be a real comparison."""
    _, _, on_eval, _ = run_loop(harness, "train.max_iters=4", "train.eval_interval=2")
    assert len(on_eval.calls) > 1


def test_on_eval_runs_after_the_metrics_are_logged(harness):
    """The documented division of labour; a checkpoint callback relies on it."""
    _, logger, on_eval, _ = run_loop(harness, "train.max_iters=4", "train.eval_interval=2")
    # by the first call: 1 batch_loss + 1 lr + 2 eval losses
    assert on_eval.logged_at_call[0] == 4
    assert on_eval.logged_at_call == sorted(on_eval.logged_at_call)
    assert on_eval.logged_at_call[-1] == len(logger.records)


# ---- what gets logged -----------------------------------------------------

def test_batch_loss_logged_once_per_step(harness):
    _, logger, _, _ = run_loop(harness, "train.max_iters=4")
    assert logger.steps_for("train", "batch_loss") == [0, 1, 2, 3]


def test_eval_loss_logged_for_every_split(harness):
    _, logger, on_eval, _ = run_loop(harness, "train.max_iters=4", "train.eval_interval=2")
    eval_steps = [step for step, _ in on_eval.calls]
    assert logger.steps_for("train", "loss") == eval_steps
    assert logger.steps_for("val", "loss") == eval_steps


def test_losses_are_finite(harness):
    _, logger, _, _ = run_loop(harness, "train.max_iters=4")
    assert all(math.isfinite(v) for *_, v in logger.records)


def test_losses_are_floats_not_tensors(harness):
    """`.item()` in the loop is what keeps json.dumps from choking downstream."""
    _, logger, _, _ = run_loop(harness, "train.max_iters=2")
    assert all(type(v) is float for *_, v in logger.records)


# ---- that it actually trains ----------------------------------------------

def test_parameters_change(harness):
    """Proves backward + step are wired, not just that the loop counts to four."""
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=4")
    before = model.emb.weight.detach().clone()
    train(model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval())
    assert not torch.equal(before, model.emb.weight)


def test_loss_decreases_over_a_longer_run(harness):
    """A corpus this repetitive should be learnable; if it isn't, something's wrong."""
    _, logger, _, _ = run_loop(harness, "train.max_iters=60", "train.eval_interval=59")
    val = [v for _, sp, m, v in logger.records if sp == "val" and m == "loss"]
    assert val[-1] < val[0]


# ---- the lr schedule ------------------------------------------------------
#
# The schedule's arithmetic is tested in test_schedule.py. What's left for the
# loop is the wiring: that it calls the closure, that the optimiser actually
# sees the result, and that it gets logged. A `lambda step: ...` is the whole
# contract, so none of this needs a real schedule.

def test_schedule_reaches_the_optimiser(harness):
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=4")
    seen = []
    real_step = optimiser.step

    def spy(*args, **kwargs):
        seen.append(optimiser.param_groups[0]["lr"])
        return real_step(*args, **kwargs)

    optimiser.step = spy
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval(),
        lr_schedule=lambda step: 0.1 * (step + 1),
    )
    assert seen == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_no_schedule_leaves_the_optimiser_alone(harness):
    """Default None: every run recorded before schedules keeps its lr."""
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=4")
    train(model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval())
    assert optimiser.param_groups[0]["lr"] == cfg.optim.lr


def test_every_param_group_is_scheduled(harness):
    """One group today; scheduling only the first would be a silent trap for
    whoever splits weight decay out."""
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=1")
    optimiser.add_param_group({"params": [torch.zeros(1, requires_grad=True)]})
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval(),
        lr_schedule=lambda step: 0.5,
    )
    assert [g["lr"] for g in optimiser.param_groups] == [0.5, 0.5]


def test_lr_is_logged_every_step(harness):
    _, logger, _, _ = run_loop(harness, "train.max_iters=4")
    assert logger.steps_for("train", "lr") == [0, 1, 2, 3]


def test_logged_lr_is_the_one_that_was_used(harness):
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=3")
    logger = FakeLogger()
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches, logger, SpyEval(),
        lr_schedule=lambda step: 0.1 * (step + 1),
    )
    logged = [v for _, sp, m, v in logger.records if sp == "train" and m == "lr"]
    assert logged == pytest.approx([0.1, 0.2, 0.3])


# ---- gradient clipping ----------------------------------------------------

def grad_norm_at_step(harness, *overrides):
    """The global grad norm the optimiser was handed, on the first step.

    A hook on `optimiser.step` is the only place to read it: the loop clips
    in between, and by the time `train()` returns the gradients are the last
    step's, not the clipped one's.

    Seeded because both the initial weights and the training batch come off the
    global RNG, and these tests compare one call's norm against another's.
    """
    torch.manual_seed(0)
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=1", *overrides)
    seen = []
    real_step = optimiser.step

    def spy(*args, **kwargs):
        seen.append(
            torch.cat([p.grad.reshape(-1) for p in model.parameters()]).norm().item()
        )
        return real_step(*args, **kwargs)

    optimiser.step = spy
    train(model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval())
    return seen[0]


def test_no_clipping_by_default(harness):
    """Every run recorded before grad_clip existed has to keep its numerics.

    Also what makes the 0.01 in the next test a real cap rather than a no-op.
    """
    assert grad_norm_at_step(harness) > 0.1


def test_clipping_caps_the_global_norm(harness):
    clipped = grad_norm_at_step(harness, "train.grad_clip=0.01")
    assert clipped == pytest.approx(0.01, rel=1e-4)


def test_a_generous_clip_leaves_the_gradient_alone(harness):
    """Clipping is a ceiling, not a normalisation: under it, nothing changes."""
    assert grad_norm_at_step(harness, "train.grad_clip=1e6") == pytest.approx(
        grad_norm_at_step(harness)
    )


def test_zero_clip_is_rejected_at_load(harness):
    """Silently zeroing every gradient trains nothing and looks fine doing it."""
    with pytest.raises(ValueError, match="grad_clip"):
        harness("train.grad_clip=0")


# ---- edges ----------------------------------------------------------------

def test_zero_iters_reports_minus_one_not_zero(harness):
    """`final_step = -1` is the honest answer; a checkpoint reads it."""
    result, logger, on_eval, _ = run_loop(harness, "train.max_iters=0")
    assert result == {"final_step": -1, "steps_completed": 0}
    assert on_eval.calls == []
    assert logger.records == []


def test_result_counts_the_steps_it_ran(harness):
    result, _, _, _ = run_loop(harness, "train.max_iters=4")
    assert result == {"final_step": 3, "steps_completed": 4}


def test_model_is_left_in_train_mode(harness):
    """evaluate() flips to eval and must flip back, or dropout silently stops."""
    _, _, _, model = run_loop(harness, "train.max_iters=4", "train.eval_interval=2")
    assert model.training


def test_evaluate_does_not_build_a_graph(harness):
    """@torch.no_grad on evaluate: without it, eval leaks memory into the run."""
    model, _, _, _, eval_batches = harness()
    before = model.emb.weight.grad
    evaluate(model, eval_batches["val"])
    assert model.emb.weight.grad is before


def test_evaluate_averages_over_the_batches_it_is_given(harness):
    model, _, _, _, eval_batches = harness()
    batches = eval_batches["val"]
    per_batch = [evaluate(model, [b]) for b in batches]
    assert evaluate(model, batches) == pytest.approx(sum(per_batch) / len(per_batch))
