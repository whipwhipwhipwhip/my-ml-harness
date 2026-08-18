"""`runtime.autocast_dtype`: the context, its scoping, and the run record.

Everything runs on CPU, where bf16 autocast is supported and observable — the
assertions are about *dtypes*, which are exact, rather than about speed, which
isn't testable. The load-bearing ones are the two that pin the scoping decision:
activations become bf16, and parameters and their gradients do not. If a future
edit moves `backward()` inside the context, or casts the model instead of its
inputs, those are the tests that notice.

The handful of tests that drive a real `run()` import the entry point inside
the test body rather than at module scope. That's what keeps the rest of this
file collectable in a repo with no nanogpt in it: their subject is a project's
entry point, everything else here is the harness itself.
"""

from __future__ import annotations

import contextlib
import json
import math

import pytest
import torch

from ml_harness.testing import ARCHES, DATAS
from ml_harness.device import autocast_context
from ml_harness.loop import evaluate, train
from ml_harness.provenance import group_sha256
from ml_harness.registry import load_run_config

from ml_harness.testing import FakeLogger, SpyEval  # noqa: F401
from tests.ml_harness.test_loop import StubModel, harness  # noqa: F401

DEVICE = torch.device("cpu")


# ---- the context itself ---------------------------------------------------

def test_none_means_fp32_via_a_nullcontext():
    """The default path costs nothing — no autocast object, no branch in the loop."""
    assert isinstance(autocast_context(DEVICE, None), contextlib.nullcontext)


def test_bf16_casts_activations():
    model = torch.nn.Linear(8, 8)
    x = torch.randn(4, 8)
    with autocast_context(DEVICE, "bf16"):
        assert model(x).dtype is torch.bfloat16


def test_bf16_leaves_the_parameters_in_fp32():
    """Autocast casts op *inputs*, not the module. The master weights stay fp32,
    which is what stops a small update vanishing into a large weight."""
    model = torch.nn.Linear(8, 8)
    with autocast_context(DEVICE, "bf16"):
        model(torch.randn(4, 8))
    assert model.weight.dtype is torch.float32


def test_the_context_is_reusable_across_iterations():
    """One instance per run, entered every step — the loop relies on this."""
    amp = autocast_context(DEVICE, "bf16")
    model = torch.nn.Linear(8, 8)
    for _ in range(3):
        with amp:
            assert model(torch.randn(4, 8)).dtype is torch.bfloat16


def test_fp16_is_rejected_and_says_why():
    """Not an oversight: without a GradScaler, fp16 gradients underflow to zero."""
    with pytest.raises(RuntimeError, match="GradScaler"):
        autocast_context(DEVICE, "fp16")


def test_an_unknown_dtype_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown autocast dtype"):
        autocast_context(DEVICE, "bfloat16")


# ---- scoping, through the loop --------------------------------------------

def _train_once(harness, amp, *overrides):  # noqa: F811
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=2", *overrides)
    result = train(
        model, optimiser, data, cfg, DEVICE, eval_batches,
        FakeLogger(), SpyEval(), amp,
    )
    return model, result


def test_gradients_stay_fp32_under_bf16(harness):  # noqa: F811
    """The observable consequence of keeping `backward()` outside the context:
    autograd matches the forward's dtypes on its own, and the fp32 parameters
    receive fp32 grads, so the optimiser step is unchanged."""
    model, _ = _train_once(harness, autocast_context(DEVICE, "bf16"))
    assert model.emb.weight.grad.dtype is torch.float32


def test_the_model_still_trains_under_bf16(harness):  # noqa: F811
    model, optimiser, data, cfg, eval_batches = harness("train.max_iters=2")
    before = model.emb.weight.detach().clone()
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches,
        FakeLogger(), SpyEval(), autocast_context(DEVICE, "bf16"),
    )
    assert not torch.equal(before, model.emb.weight)
    assert model.emb.weight.dtype is torch.float32


def test_eval_loss_is_a_finite_python_float_under_bf16(harness):  # noqa: F811
    """`.item()` on a bf16 loss still has to reach the metrics file as a float."""
    model, optimiser, data, cfg, eval_batches = harness()
    logger = FakeLogger()
    train(
        model, optimiser, data, cfg, DEVICE, eval_batches, logger, SpyEval(),
        autocast_context(DEVICE, "bf16"),
    )
    values = [v for *_, v in logger.records]
    assert values and all(type(v) is float and math.isfinite(v) for v in values)


def test_evaluate_defaults_to_fp32_when_given_no_context(harness):  # noqa: F811
    """The default keeps every pre-existing caller — and every old test — correct."""
    model, _, _, _, eval_batches = harness()
    assert evaluate(model, eval_batches["val"]) == pytest.approx(
        evaluate(model, eval_batches["val"], None)
    )


def test_evaluate_under_bf16_is_close_but_not_identical(harness):  # noqa: F811
    """Sanity that the context reaches eval at all: bf16 has ~8 mantissa bits, so
    the same batches give a nearby but different number. Loose tolerance — the
    point is 'measured differently', not any particular error."""
    model, _, _, _, eval_batches = harness()
    fp32 = evaluate(model, eval_batches["val"])
    bf16 = evaluate(model, eval_batches["val"], autocast_context(DEVICE, "bf16"))
    assert bf16 == pytest.approx(fp32, rel=0.05)


# ---- the config field -----------------------------------------------------

def test_the_default_is_off(config_path):
    cfg = load_run_config(config_path, ARCHES, DATAS)
    assert cfg.runtime.autocast_dtype is None


def test_settable_from_an_override(config_path):
    cfg = load_run_config(config_path, ARCHES, DATAS, ['runtime.autocast_dtype="bf16"'])
    assert cfg.runtime.autocast_dtype == "bf16"


def test_two_dtypes_are_two_experiments():
    """Same reasoning as matmul_precision: different arithmetic, so the losses
    aren't poolable."""
    base = {"runtime": {"seed": 0, "device": "cpu", "autocast_dtype": None}}
    other = {"runtime": {**base["runtime"], "autocast_dtype": "bf16"}}
    assert group_sha256(other) != group_sha256(base)


# ---- the entry point ------------------------------------------------------

def test_run_records_the_dtype_it_trained_in(gpt_config_path, runs_root):
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train as entry
    rd = entry.run(
        gpt_config_path,
        overrides=['runtime.autocast_dtype="bf16"'],
        runs_root=runs_root,
        cache_dir=None,
    )
    recorded = json.loads(rd.config_path.read_text())
    assert recorded["runtime"]["autocast_dtype"] == "bf16"


def test_a_bf16_run_writes_an_fp32_checkpoint(gpt_config_path, runs_root):
    """The parameters were never cast, so a bf16 run reloads like any other."""
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train as entry
    rd = entry.run(
        gpt_config_path,
        overrides=['runtime.autocast_dtype="bf16"'],
        runs_root=runs_root,
        cache_dir=None,
    )
    ckpt = torch.load(rd.ckpt_path("last"), weights_only=False)
    assert all(t.dtype is torch.float32 for t in ckpt["model"].values())


def test_a_bad_dtype_fails_before_a_run_directory_exists(gpt_config_path, runs_root):
    """Validated next to the device, not at the call site: a typo shouldn't leave
    a failed run behind."""
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train as entry
    with pytest.raises(RuntimeError, match="Unknown autocast dtype"):
        entry.run(
            gpt_config_path,
            overrides=['runtime.autocast_dtype="bfloat16"'],
            runs_root=runs_root,
            cache_dir=None,
        )
    assert not runs_root.exists() or not list(runs_root.iterdir())
