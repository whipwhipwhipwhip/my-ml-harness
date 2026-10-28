"""`runtime.autocast_dtype`: the context, its scoping, and the run record.

Everything runs on CPU, where bf16 autocast works. The tests check *dtypes*,
which are exact, rather than speed, which can't really be tested. The important
ones check what autocast covers: activations become bf16, but parameters and
their gradients don't. If `backward()` is ever moved inside the context, or the
model is cast instead of its inputs, those tests will catch it.

The tests that do a real run use `testing.run`. They're the slow ones here,
since they write a run directory, but they still only use what the harness
ships.
"""

from __future__ import annotations

import contextlib
import json
import math

import pytest
import torch

from ml_harness.testing import ARCHES, DATAS, run
from ml_harness.device import autocast_context
from ml_harness.loop import evaluate, train
from ml_harness.provenance import group_sha256
from ml_harness.registry import load_run_config

from ml_harness.testing import FakeLogger, SpyEval  # noqa: F401
from tests.ml_harness.test_loop import StubModel, harness  # noqa: F401

DEVICE = torch.device("cpu")


# ---- the context itself ---------------------------------------------------

def test_none_means_fp32_via_a_nullcontext():
    """The default is a nullcontext: no autocast and no branch in the loop."""
    assert isinstance(autocast_context(DEVICE, None), contextlib.nullcontext)


def test_bf16_casts_activations():
    model = torch.nn.Linear(8, 8)
    x = torch.randn(4, 8)
    with autocast_context(DEVICE, "bf16"):
        assert model(x).dtype is torch.bfloat16


def test_bf16_leaves_the_parameters_in_fp32():
    """Autocast casts the inputs to each op, not the module. The weights stay
    fp32, which stops small updates from disappearing into large weights."""
    model = torch.nn.Linear(8, 8)
    with autocast_context(DEVICE, "bf16"):
        model(torch.randn(4, 8))
    assert model.weight.dtype is torch.float32


def test_the_context_is_reusable_across_iterations():
    """One instance per run, entered every step; the loop relies on this."""
    amp = autocast_context(DEVICE, "bf16")
    model = torch.nn.Linear(8, 8)
    for _ in range(3):
        with amp:
            assert model(torch.randn(4, 8)).dtype is torch.bfloat16


def test_fp16_is_rejected_and_says_why():
    """On purpose: without a GradScaler, fp16 gradients underflow to zero."""
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
    """Because `backward()` is outside the context, autograd matches the
    forward's dtypes by itself, and the fp32 parameters get fp32 gradients, so
    the optimiser step is unchanged."""
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
    """So existing callers (and older tests) that don't pass a context still work."""
    model, _, _, _, eval_batches = harness()
    assert evaluate(model, eval_batches["val"]) == pytest.approx(
        evaluate(model, eval_batches["val"], None)
    )


def test_evaluate_under_bf16_is_close_but_not_identical(harness):  # noqa: F811
    """Checks the context is actually used in eval: bf16 has ~8 mantissa bits,
    so the same batches give a close but different number. The tolerance is
    loose because we only care that it's different, not by how much."""
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
    """Same as matmul_precision: different arithmetic, so the losses shouldn't
    be averaged together."""
    base = {"runtime": {"seed": 0, "device": "cpu", "autocast_dtype": None}}
    other = {"runtime": {**base["runtime"], "autocast_dtype": "bf16"}}
    assert group_sha256(other) != group_sha256(base)


# ---- the entry point ------------------------------------------------------

def test_run_records_the_dtype_it_trained_in(config_path, runs_root):
    rd = run(
        config_path,
        overrides=['runtime.autocast_dtype="bf16"'],
        runs_root=runs_root,
        cache_dir=None,
    )
    recorded = json.loads(rd.config_path.read_text())
    assert recorded["runtime"]["autocast_dtype"] == "bf16"


def test_a_bf16_run_writes_an_fp32_checkpoint(config_path, runs_root):
    """The parameters were never cast, so a bf16 run reloads like any other."""
    rd = run(
        config_path,
        overrides=['runtime.autocast_dtype="bf16"'],
        runs_root=runs_root,
        cache_dir=None,
    )
    ckpt = torch.load(rd.ckpt_path("last"), weights_only=False)
    assert all(t.dtype is torch.float32 for t in ckpt["model"].values())


def test_a_bad_dtype_fails_before_a_run_directory_exists(config_path, runs_root):
    """Checked up front with the device, so a typo doesn't leave a failed run
    directory behind."""
    with pytest.raises(RuntimeError, match="Unknown autocast dtype"):
        run(
            config_path,
            overrides=['runtime.autocast_dtype="bfloat16"'],
            runs_root=runs_root,
            cache_dir=None,
        )
    assert not runs_root.exists() or not list(runs_root.iterdir())
