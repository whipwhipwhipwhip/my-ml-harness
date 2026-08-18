"""`runtime.compile`: the flag, and the checkpoint hazard it introduces.

Nothing here actually invokes inductor. A real compilation of even the tiny test
model costs seconds, which would move this module out of the cheap tier for no
extra confidence — the thing that can silently break is not whether torch can
compile, it's whether `train.run` checkpoints the *uncompiled* module. So the
wiring test substitutes a fake wrapper with the one property that matters:
`torch.compile` returns a module holding the original as a `_orig_mod` child, and
`nn.Module.state_dict` prefixes by child name. That prefix is the whole bug.

`torch.compile` itself is lazy — it wraps now and compiles on the first forward —
so the tests that only inspect the wrapper are free and use the real thing.

The tests that drive a real `run()` use `testing.run`, the reference entry point
shipped beside the reference arch. They are the expensive tier of this module —
they write a run directory — but they need no project, so the portability rule
holds: nothing here imports anything the harness doesn't ship.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ml_harness.testing import ARCHES, DATAS, run
from ml_harness.device import compile_model
from ml_harness.provenance import group_sha256
from ml_harness import runner
from ml_harness.registry import load_run_config


class FakeOptimizedModule(nn.Module):
    """What `torch.compile` returns, minus the compiling.

    A real `OptimizedModule` for the purpose these tests care about: it delegates
    forward and it holds the original as `_orig_mod`, so its state_dict keys come
    out prefixed exactly like the real one's.
    """

    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


# ---- the wrapper ----------------------------------------------------------

def test_disabled_returns_the_very_same_object():
    """The off path must not wrap, or every run pays for a decision it declined."""
    model = nn.Linear(4, 4)
    assert compile_model(model, False) is model


def test_disabled_is_the_default():
    model = nn.Linear(4, 4)
    assert compile_model(model) is model


def test_compiling_prefixes_the_state_dict_keys():
    """The hazard, stated as a test so it can't be forgotten: this is why entry
    points compile into a second name instead of rebinding `model`."""
    model = nn.Linear(4, 4)
    compiled = compile_model(model, True)
    assert list(model.state_dict()) == ["weight", "bias"]
    assert list(compiled.state_dict()) == ["_orig_mod.weight", "_orig_mod.bias"]


def test_the_fake_reproduces_that_prefix():
    """Guards the substitution below: if the fake stopped matching the real
    wrapper's observable shape, the wiring test would be testing nothing."""
    model = nn.Linear(4, 4)
    assert list(FakeOptimizedModule(model).state_dict()) == list(
        compile_model(model, True).state_dict()
    )


def test_the_parameters_are_the_same_tensors():
    """Why the optimiser can be built from the uncompiled module."""
    model = nn.Linear(4, 4)
    compiled = compile_model(model, True)
    assert [p is q for p, q in zip(model.parameters(), compiled.parameters())] == [
        True,
        True,
    ]


# ---- the config field -----------------------------------------------------

def test_the_default_is_off(config_path):
    cfg = load_run_config(config_path, ARCHES, DATAS)
    assert cfg.runtime.compile is False


def test_settable_from_an_override(config_path):
    cfg = load_run_config(config_path, ARCHES, DATAS, ["runtime.compile=true"])
    assert cfg.runtime.compile is True


def test_a_non_boolean_is_rejected(config_path):
    """dacite type-checks the override the same as a hand-typed config value."""
    with pytest.raises(Exception):
        load_run_config(config_path, ARCHES, DATAS, ['runtime.compile="yes"'])


def test_compiled_and_eager_are_two_experiments():
    """Compilation shouldn't change the maths, but it does change the kernels,
    and two runs whose numbers came from different kernels are worth telling
    apart. Cheap to relax later; expensive to notice you needed it."""
    base = {"runtime": {"seed": 0, "device": "cpu", "compile": False}}
    other = {"runtime": {**base["runtime"], "compile": True}}
    assert group_sha256(other) != group_sha256(base)


# ---- the wiring, without paying for inductor ------------------------------

@pytest.fixture
def fake_compile(monkeypatch):
    """Substitute the wrapper, keeping the two-name structure honest.

    Patched on `harness.runner`, not on the entry point: compiling moved into
    `build_run` when the shared build sequence was lifted, so that's the name
    `run()` now reaches through.
    """
    calls = []

    def fake(model, enabled):
        calls.append(enabled)
        return FakeOptimizedModule(model) if enabled else model

    monkeypatch.setattr(runner, "compile_model", fake)
    return calls


def test_run_checkpoints_the_uncompiled_module(config_path, runs_root, fake_compile):
    """The regression this whole two-name dance exists to prevent."""
    rd = run(
        config_path,
        overrides=["runtime.compile=true"],
        runs_root=runs_root,
        cache_dir=None,
    )
    assert fake_compile == [True]

    for which in ("best", "last"):
        keys = torch.load(rd.ckpt_path(which), weights_only=False)["model"]
        assert not any(k.startswith("_orig_mod.") for k in keys), which


def test_a_compiled_runs_checkpoint_loads_into_a_plain_model(
    config_path, runs_root, fake_compile
):
    """The consequence that matters: a compiled run's checkpoint is still a
    checkpoint of *the model*, loadable by anything that rebuilds from config."""
    from ml_harness.registry import build_model

    rd = run(
        config_path,
        overrides=["runtime.compile=true"],
        runs_root=runs_root,
        cache_dir=None,
    )
    ckpt = torch.load(rd.ckpt_path("last"), weights_only=False)
    # The same tables the entry point ran with, so the `arch` tag resolves to
    # the class that was trained.
    cfg = load_run_config(config_path, ARCHES, DATAS)
    model = build_model(cfg, ckpt["vocab_size"], ARCHES)

    # strict=True by default, so a stray prefix or a missing key raises here.
    # Not raising *is* the assertion.
    model.load_state_dict(ckpt["model"])


def test_run_records_that_it_compiled(config_path, runs_root, fake_compile):
    rd = run(
        config_path,
        overrides=["runtime.compile=true"],
        runs_root=runs_root,
        cache_dir=None,
    )
    assert json.loads(rd.config_path.read_text())["runtime"]["compile"] is True


def test_the_default_run_does_not_compile(config_path, runs_root, fake_compile):
    run(config_path, runs_root=runs_root, cache_dir=None)
    assert fake_compile == [False]
