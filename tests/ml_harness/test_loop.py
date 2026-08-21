"""ml_harness/loop.py, driven with fakes.

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

from ml_harness.testing import ARCHES, DATAS, FakeLogger, SpyEval, load_data
from ml_harness.loop import evaluate, train
from ml_harness.registry import load_run_config

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
    """Scheduling only the first group would be a silent trap, and the weight
    decay split makes it a live one: a real optimiser now has two groups, and
    the second would sit at its build-time lr forever. Kept as a hand-added
    group so the claim stays about the loop rather than about `build_optimiser`
    — test_weight_decay.py makes it against the real split."""
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


# ---- gradient accumulation ------------------------------------------------

class ChunkedData:
    """A `data` that hands out one fixed batch, sliced to whatever size is asked.

    This is what takes the RNG out of the comparison. With the real `get_batch`,
    one step at batch_size 4 and two micro-steps at 2 draw different rows, so
    their gradients would differ for a reason that has nothing to do with
    accumulation. Here both paths see exactly the same 4 sequences, and any
    difference in the resulting gradient is the mechanism's fault.

    Records what it was asked for, which is how the tests below see the shape of
    the loop's requests without instrumenting the loop.
    """

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.cursor = 0
        self.requests: list[int] = []

    def get_batch(self, split, batch_size, block_size, device, n_iters=1, generator=None):
        self.requests.append(batch_size)
        batches = []
        for _ in range(n_iters):
            lo = self.cursor
            hi = lo + batch_size
            batches.append((self.x[lo:hi], self.y[lo:hi]))
            self.cursor = hi % len(self.x)
        return batches


def one_step(harness, *overrides):
    """One training step over a fixed batch. Returns (gradients, data, logger).

    The gradient is read off in a hook on `optimiser.step`, for the same reason
    `grad_norm_at_step` does it: that is the only moment the accumulated
    gradient exists, and by the time `train()` returns it has been replaced.

    Seeded so the two sides of an equivalence test start from identical weights.
    """
    torch.manual_seed(0)
    model, optimiser, real_data, cfg, eval_batches = harness(
        "train.max_iters=1", *overrides
    )
    xb, yb = real_data.get_batch(
        "train", cfg.train.batch_size, cfg.model.block_size, "cpu",
        1, torch.Generator().manual_seed(7),
    )[0]
    data = ChunkedData(xb, yb)
    logger = FakeLogger()

    grads = []
    real_step = optimiser.step

    def spy(*args, **kwargs):
        grads.append(
            torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone()
        )
        return real_step(*args, **kwargs)

    optimiser.step = spy
    train(model, optimiser, data, cfg, DEVICE, eval_batches, logger, SpyEval())
    return grads, data, logger


def test_accumulated_gradient_equals_the_full_batch(harness):
    """The claim the whole feature rests on: dividing each micro-loss by the
    number of micro-batches makes the accumulated gradient the full batch's.

    Off by a factor of `accum_steps` if the scaling is dropped, so a tight
    tolerance here is the point — this would not fail by a little."""
    whole, _, _ = one_step(harness)
    split, _, _ = one_step(harness, "train.micro_batch_size=2")
    assert torch.allclose(whole[0], split[0], atol=1e-6)


def test_a_single_micro_batch_is_the_unaccumulated_path(harness):
    """micro == batch is accum_steps of 1, i.e. exactly what ran before."""
    whole, _, _ = one_step(harness)
    same, _, _ = one_step(harness, "train.micro_batch_size=4")
    assert torch.equal(whole[0], same[0])


def test_one_optimiser_step_per_training_step_whatever_the_accumulation(harness):
    """Accumulation splits the forward/backward, not the update. Stepping per
    micro-batch would quietly train at 4x the intended batch count."""
    grads, data, _ = one_step(harness, "train.micro_batch_size=1")
    assert len(grads) == 1
    assert data.requests == [1, 1, 1, 1]


def test_no_micro_batch_size_asks_for_the_whole_batch_once(harness):
    """Every run recorded before this field has to keep its numerics, and the
    default has to cost nothing — one request, at the full batch size."""
    _, data, _ = one_step(harness)
    assert data.requests == [4]


def test_batch_loss_still_means_the_mean_over_the_full_batch(harness):
    """The metric has to be comparable across the field, or every run before it
    falls out of the same plot."""
    _, _, whole = one_step(harness)
    _, _, split = one_step(harness, "train.micro_batch_size=2")
    loss_of = lambda log: [
        v for _, sp, m, v in log.records if sp == "train" and m == "batch_loss"
    ]
    assert loss_of(split) == pytest.approx(loss_of(whole), abs=1e-6)


def test_clipping_sees_the_whole_batch_gradient(harness):
    """Clipping stays after the accumulation. Clipping each micro-batch's share
    would be a different and much stronger rule wearing the same config key."""
    grads, _, _ = one_step(harness, "train.micro_batch_size=2", "train.grad_clip=0.01")
    assert grads[0].norm().item() == pytest.approx(0.01, rel=1e-4)


def test_accumulation_runs_every_step_not_just_the_first(harness):
    """`zero_grad` moved to the top of the step; if it had stayed between
    forward and backward, step 2 onwards would accumulate into step 1's."""
    torch.manual_seed(0)
    model, optimiser, real_data, cfg, eval_batches = harness(
        "train.max_iters=3", "train.micro_batch_size=2"
    )
    xb, yb = real_data.get_batch(
        "train", cfg.train.batch_size, cfg.model.block_size, "cpu",
        1, torch.Generator().manual_seed(7),
    )[0]
    data = ChunkedData(xb, yb)
    train(model, optimiser, data, cfg, DEVICE, eval_batches, FakeLogger(), SpyEval())
    assert data.requests == [2] * 6


@pytest.mark.parametrize(
    "override, message",
    [
        ("train.micro_batch_size=3", "divisible"),
        ("train.micro_batch_size=8", "must not exceed"),
        ("train.micro_batch_size=0", "must be positive"),
    ],
)
def test_a_micro_batch_that_cannot_divide_the_batch_is_rejected_at_load(
    harness, override, message
):
    """Fail next to the config that said it, not at step 0 of a run that has
    already opened a run directory."""
    with pytest.raises(ValueError, match=message):
        harness(override)


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


# ---- data parallelism ------------------------------------------------------
#
# The loop's whole share of it: suppress DDP's all-reduce on every micro-batch
# but the last, and let the caller reduce the logged loss. Both are duck-typed,
# so a fake with the right attribute and a plain tensor function are complete
# implementations — which is the same claim `StubModel` makes about the model.

class SyncSpyModel(StubModel):
    """A model that looks enough like DDP to answer whether the loop drives it.

    Records `require_backward_grad_sync` as each forward pass sees it, which is
    the moment that matters: DDP reads the flag in `forward()` to decide whether
    to prepare its reducer, so a loop that set it after the forward would be
    setting it too late and this would catch that.

    Guarded on `is_grad_enabled` for the same reason DDP's own check is —
    `evaluate()` runs under `no_grad` and its forwards are not part of the
    question.
    """

    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.require_backward_grad_sync = True
        self.sync_flags: list[bool] = []

    def forward(self, idx, targets=None):
        if torch.is_grad_enabled():
            self.sync_flags.append(self.require_backward_grad_sync)
        return super().forward(idx, targets)


def run_with_model(harness, model_cls, *overrides, reduce_loss=None):
    """`run_loop`, but with a say in the model and the reduction.

    Seeded like `one_step`, and for the same reason: the tests below compare two
    calls against each other, so both the initial weights and the batches drawn
    off the global RNG have to match on each side. Without it every comparison
    fails for a reason that has nothing to do with what is being tested.
    """
    torch.manual_seed(0)
    _, optimiser, data, cfg, eval_batches = harness(*overrides)
    model = model_cls(data.tokenizer.vocab_size)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
    logger = FakeLogger()
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches, logger, SpyEval(),
        None, None, reduce_loss,
    )
    return model, logger


def test_only_the_last_micro_batch_of_a_step_syncs_gradients(harness):
    """The point of the whole mechanism. Without it DDP all-reduces on every
    micro-batch, so accum_steps=8 pays eight gradient exchanges per optimiser
    step instead of one — identical arithmetic, eight times the network."""
    model, _ = run_with_model(
        harness, SyncSpyModel, "train.max_iters=1", "train.micro_batch_size=2"
    )
    assert model.sync_flags == [False, True]


def test_every_step_ends_with_a_sync(harness):
    """Per step, not once per run. A flag left False after the first step would
    leave every later step's gradient un-reduced and each rank drifting into its
    own model, which no loss curve would show."""
    model, _ = run_with_model(
        harness, SyncSpyModel, "train.max_iters=3", "train.micro_batch_size=2"
    )
    assert model.sync_flags == [False, True, False, True, False, True]


def test_without_accumulation_every_backward_syncs(harness):
    """accum_steps == 1 means the only micro-batch is also the last one."""
    model, _ = run_with_model(
        harness, SyncSpyModel, "train.max_iters=2", "train.micro_batch_size=null"
    )
    assert model.sync_flags == [True, True]


def test_a_model_without_the_flag_is_left_alone(harness):
    """The single-process path. The loop must not grow the attribute on a plain
    module — a stray `require_backward_grad_sync` on an undistributed model is
    a lie that the next reader has to disprove."""
    model, _ = run_with_model(
        harness, StubModel, "train.max_iters=1", "train.micro_batch_size=2"
    )
    assert not hasattr(model, "require_backward_grad_sync")


def test_reduce_loss_is_applied_to_the_logged_batch_loss(harness):
    """Under DDP the model saw one rank's share, so the raw number is that
    share's mean rather than the step's."""
    plain, logger = run_with_model(harness, StubModel, "train.max_iters=1")
    _, doubled = run_with_model(
        harness, StubModel, "train.max_iters=1", reduce_loss=lambda t: t * 2
    )

    raw = [v for _, sp, m, v in logger.records if m == "batch_loss"]
    red = [v for _, sp, m, v in doubled.records if m == "batch_loss"]
    assert red == pytest.approx([v * 2 for v in raw])


def test_no_reducer_logs_the_loss_unchanged(harness):
    """The default, and what every caller written before this parameter gets."""
    _, logger = run_with_model(harness, StubModel, "train.max_iters=1")
    _, identity = run_with_model(
        harness, StubModel, "train.max_iters=1", reduce_loss=lambda t: t
    )

    raw = [v for _, sp, m, v in logger.records if m == "batch_loss"]
    same = [v for _, sp, m, v in identity.records if m == "batch_loss"]
    assert raw == pytest.approx(same)


def test_reducing_does_not_disturb_training(harness):
    """The reduction is for the metric only. If it mutated the accumulator in
    place, or worse the graph, the weights would diverge from the unreduced run
    and the logged number would have changed what it was reporting on."""
    plain, _ = run_with_model(harness, StubModel, "train.max_iters=2")
    scaled, _ = run_with_model(
        harness, StubModel, "train.max_iters=2", reduce_loss=lambda t: t * 100
    )
    assert torch.allclose(plain.emb.weight, scaled.emb.weight)
