"""`runtime.matmul_precision`: the config field, the setter, and the run record.

The thing that makes this worth its own module is that the setting is *process
global*. Every test here restores it, via an autouse fixture, because a test that
left it on "high" would silently change the numerics of every test that ran after
it — and the failure would land somewhere unrelated.

Nothing here asserts that "high" is faster. It's a CUDA TF32 switch, it does
nothing on CPU or MPS, and a timing assertion in a test suite is either flaky or
vacuous. What's asserted is that the value reaches torch, that a bad one is
rejected next to the config that named it, and that the run record says which
precision produced its numbers.

The handful of tests that drive a real `run()` import the entry point inside
the test body rather than at module scope. That's what keeps the rest of this
file collectable in a repo with no nanogpt in it: their subject is a project's
entry point, everything else here is the harness itself.
"""

from __future__ import annotations

import json

import pytest
import torch

from ml_harness.testing import ARCHES, DATAS
from ml_harness.device import set_matmul_precision
from ml_harness.provenance import group_sha256
from ml_harness.registry import load_run_config


@pytest.fixture(autouse=True)
def restore_precision():
    """Put torch back however we found it. Autouse: forgetting this is the bug."""
    before = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(before)


# ---- the setter -----------------------------------------------------------

@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_every_valid_precision_reaches_torch(precision):
    set_matmul_precision(precision)
    assert torch.get_float32_matmul_precision() == precision


def test_an_unknown_precision_is_rejected():
    """Same contract as resolve_device: fail beside the config that named it."""
    with pytest.raises(RuntimeError, match="Unknown matmul precision"):
        set_matmul_precision("fastest")


def test_a_rejected_precision_leaves_the_setting_alone():
    set_matmul_precision("high")
    with pytest.raises(RuntimeError):
        set_matmul_precision("fastest")
    assert torch.get_float32_matmul_precision() == "high"


# ---- the config field -----------------------------------------------------

def test_a_config_without_the_key_still_loads(config_path):
    """Defaulted for exactly this: every config file written before the field."""
    cfg = load_run_config(config_path, ARCHES, DATAS)
    assert cfg.runtime.matmul_precision == "highest"


def test_the_field_is_settable_from_the_config_file(config_path, tmp_path):
    raw = json.loads(config_path.read_text())
    raw["runtime"]["matmul_precision"] = "high"
    path = tmp_path / "high.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    cfg = load_run_config(path, ARCHES, DATAS)
    assert cfg.runtime.matmul_precision == "high"


def test_the_field_is_settable_from_an_override(config_path):
    """`--set runtime.matmul_precision=high`, which is how a benchmark varies it."""
    cfg = load_run_config(
        config_path, ARCHES, DATAS, ["runtime.matmul_precision=high"]
    )
    assert cfg.runtime.matmul_precision == "high"


def test_two_precisions_are_two_experiments():
    """Deliberately not in GROUP_EXCLUDES — the arithmetic differs, so the
    losses aren't averageable. Same reasoning as runtime.device."""
    base = {"runtime": {"seed": 0, "eval_seed": 1, "device": "cpu",
                        "matmul_precision": "highest"}}
    other = {"runtime": {**base["runtime"], "matmul_precision": "high"}}
    assert group_sha256(other) != group_sha256(base)


def test_seeds_still_share_a_group_at_one_precision():
    """The exclusion that does exist isn't disturbed by the new field."""
    base = {"runtime": {"seed": 0, "eval_seed": 1, "device": "cpu",
                        "matmul_precision": "high"}}
    other = {"runtime": {**base["runtime"], "seed": 99}}
    assert group_sha256(other) == group_sha256(base)


# ---- the entry point ------------------------------------------------------

def test_run_applies_the_configured_precision(gpt_config_path, runs_root):
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train
    rd = train.run(
        gpt_config_path,
        overrides=["runtime.matmul_precision=high"],
        runs_root=runs_root,
        cache_dir=None,
    )
    assert torch.get_float32_matmul_precision() == "high"
    # and the record says so, which is what makes a fast run explicable later
    recorded = json.loads(rd.config_path.read_text())
    assert recorded["runtime"]["matmul_precision"] == "high"


def test_run_sets_the_default_rather_than_inheriting(gpt_config_path, runs_root):
    """The sweep case: many runs in one process, so a run that doesn't mention
    precision must still assert its own, not inherit the previous point's."""
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train
    torch.set_float32_matmul_precision("high")

    train.run(gpt_config_path, runs_root=runs_root, cache_dir=None)

    assert torch.get_float32_matmul_precision() == "highest"


def test_an_unknown_precision_fails_the_run(gpt_config_path, runs_root):
    # Deferred — see the module docstring.
    from ml_learning.nanogpt_conor import train
    with pytest.raises(RuntimeError, match="Unknown matmul precision"):
        train.run(
            gpt_config_path,
            overrides=["runtime.matmul_precision=fastest"],
            runs_root=runs_root,
            cache_dir=None,
        )
