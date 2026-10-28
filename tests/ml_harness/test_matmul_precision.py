"""`runtime.matmul_precision`: the config field, the setter, and the run record.

The tricky part is that the setting is *global to the process*. An autouse
fixture restores it after every test here, because a test that left it on
"high" would quietly change the numerics of every later test, and the failure
would show up somewhere unrelated.

Nothing here checks that "high" is faster. It's a CUDA TF32 switch that does
nothing on CPU or MPS, and timing assertions in tests are either flaky or
meaningless. Instead these check that the value reaches torch, that a bad value
is rejected when the config is loaded, and that the run record says which
precision was used.

The tests that do a real run use `testing.run`. They're the slow ones here,
since they write a run directory, but they still only use what the harness
ships.
"""

from __future__ import annotations

import json

import pytest
import torch

from ml_harness.testing import ARCHES, DATAS, run
from ml_harness.device import set_matmul_precision
from ml_harness.provenance import group_sha256
from ml_harness.registry import load_run_config


@pytest.fixture(autouse=True)
def restore_precision():
    """Put torch back how we found it. Autouse, so no test can forget."""
    before = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(before)


# ---- the setter -----------------------------------------------------------

@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_every_valid_precision_reaches_torch(precision):
    set_matmul_precision(precision)
    assert torch.get_float32_matmul_precision() == precision


def test_an_unknown_precision_is_rejected():
    """Like resolve_device: fail early with a clear error."""
    with pytest.raises(RuntimeError, match="Unknown matmul precision"):
        set_matmul_precision("fastest")


def test_a_rejected_precision_leaves_the_setting_alone():
    set_matmul_precision("high")
    with pytest.raises(RuntimeError):
        set_matmul_precision("fastest")
    assert torch.get_float32_matmul_precision() == "high"


# ---- the config field -----------------------------------------------------

def test_a_config_without_the_key_still_loads(config_path):
    """The default is so config files from before this field still load."""
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
    """Not in GROUP_EXCLUDES on purpose: the arithmetic differs, so the losses
    shouldn't be averaged. Same reasoning as runtime.device."""
    base = {"runtime": {"seed": 0, "eval_seed": 1, "device": "cpu",
                        "matmul_precision": "highest"}}
    other = {"runtime": {**base["runtime"], "matmul_precision": "high"}}
    assert group_sha256(other) != group_sha256(base)


def test_seeds_still_share_a_group_at_one_precision():
    """Seeds still group together with this field present."""
    base = {"runtime": {"seed": 0, "eval_seed": 1, "device": "cpu",
                        "matmul_precision": "high"}}
    other = {"runtime": {**base["runtime"], "seed": 99}}
    assert group_sha256(other) == group_sha256(base)


# ---- the entry point ------------------------------------------------------

def test_run_applies_the_configured_precision(config_path, runs_root):
    rd = run(
        config_path,
        overrides=["runtime.matmul_precision=high"],
        runs_root=runs_root,
        cache_dir=None,
    )
    assert torch.get_float32_matmul_precision() == "high"
    # and the record says so, so you can tell later why a run was fast
    recorded = json.loads(rd.config_path.read_text())
    assert recorded["runtime"]["matmul_precision"] == "high"


def test_run_sets_the_default_rather_than_inheriting(config_path, runs_root):
    """The sweep case: many runs in one process, so a run that doesn't set
    precision must still reset it, rather than inherit the previous point's."""
    torch.set_float32_matmul_precision("high")

    run(config_path, runs_root=runs_root, cache_dir=None)

    assert torch.get_float32_matmul_precision() == "highest"


def test_an_unknown_precision_fails_the_run(config_path, runs_root):
    with pytest.raises(RuntimeError, match="Unknown matmul precision"):
        run(
            config_path,
            overrides=["runtime.matmul_precision=fastest"],
            runs_root=runs_root,
            cache_dir=None,
        )
