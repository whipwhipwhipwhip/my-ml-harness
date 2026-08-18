"""`--set` handling: the raw-dict rewrite in ml_harness/registry.py.

Two layers, tested separately:
  * apply_overrides on its own — the errors it raises itself, and the value parsing
  * through load_run_config — the errors dacite raises because the rewrite went
    through the dict rather than the built object

The second layer is the point of doing it before dacite rather than after, so it
gets tests rather than a comment.
"""

from __future__ import annotations

import json

import pytest
from dacite import DaciteError

from ml_harness.testing import ARCHES, DATAS
from ml_harness.registry import apply_overrides, load_run_config


def raw():
    return {
        "arch": "gpt",
        "model": {"n_embd": 32, "dropout": 0.2},
        "optim": {"lr": 3e-4},
        "note": "original",
    }


# ---- value parsing --------------------------------------------------------

def test_nested_key_is_replaced():
    d = raw()
    apply_overrides(d, ["model.n_embd=64"])
    assert d["model"]["n_embd"] == 64


def test_top_level_key_is_replaced():
    d = raw()
    apply_overrides(d, ["note=changed"])
    assert d["note"] == "changed"


def test_numbers_parse_as_numbers_not_strings():
    d = raw()
    apply_overrides(d, ["model.n_embd=64", "optim.lr=1e-3"])
    assert d["model"]["n_embd"] == 64 and isinstance(d["model"]["n_embd"], int)
    assert d["optim"]["lr"] == 1e-3 and isinstance(d["optim"]["lr"], float)


def test_unquoted_word_falls_back_to_string():
    """`runtime.device=mps` isn't valid JSON. It still has to work."""
    d = raw()
    apply_overrides(d, ["note=mps and spaces"])
    assert d["note"] == "mps and spaces"


def test_int_widens_to_float_against_a_float_field():
    """`optim.lr=1` shouldn't need the trailing `.0` — dacite would reject the int."""
    d = raw()
    apply_overrides(d, ["optim.lr=1"])
    assert d["optim"]["lr"] == 1.0 and isinstance(d["optim"]["lr"], float)


def test_int_stays_int_against_an_int_field():
    d = raw()
    apply_overrides(d, ["model.n_embd=64"])
    assert isinstance(d["model"]["n_embd"], int)


def test_value_may_contain_equals():
    d = raw()
    apply_overrides(d, ["note=a=b"])
    assert d["note"] == "a=b"


def test_last_override_wins():
    """What makes `--smoke --set train.max_iters=50` a longer smoke run."""
    d = raw()
    apply_overrides(d, ["model.n_embd=64", "model.n_embd=128"])
    assert d["model"]["n_embd"] == 128


def test_empty_overrides_change_nothing():
    d = raw()
    apply_overrides(d, [])
    assert d == raw()


# ---- errors apply_overrides raises itself ---------------------------------

@pytest.mark.parametrize(
    "override, fragment",
    [
        ("model.n_embd", "expected KEY=VALUE"),
        ("nosuchblock.x=1", "no section"),
        ("model.n_embd.deeper=1", "not a section"),
        ("nte=x", "no such key"),
    ],
    ids=["no-equals", "missing-section", "value-as-section", "top-level-typo"],
)
def test_malformed_override_is_rejected(override, fragment):
    with pytest.raises(ValueError, match=fragment):
        apply_overrides(raw(), [override])


def test_a_rejected_override_names_the_key():
    """The error has to say which override, not just that one was wrong."""
    with pytest.raises(ValueError, match="nosuchblock.x"):
        apply_overrides(raw(), ["nosuchblock.x=1"])


# ---- errors that only exist because the rewrite happens before dacite -----

def test_typo_inside_a_block_is_caught_by_strict(config_path):
    """The payoff: a typo'd override fails like a typo'd JSON key would."""
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["model.n_layerr=6"])


def test_wrong_type_is_caught_by_dacite(config_path):
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["model.n_head=oops"])


def test_post_init_sees_overridden_values(config_path):
    """Not the file's values — the reason this can't be a post-hoc setattr."""
    with pytest.raises(ValueError, match="divisible"):
        load_run_config(config_path, ARCHES, DATAS, ["model.n_embd=100", "model.n_head=6"])


def test_arch_itself_can_be_overridden(config_path):
    """Overrides land before the tag is read, so an unknown arch is rejected as one."""
    with pytest.raises(ValueError, match="unknown arch"):
        load_run_config(config_path, ARCHES, DATAS, ["arch=nope"])


def test_overrides_reach_the_built_config(config_path):
    cfg = load_run_config(
        config_path, ARCHES, DATAS, ["model.n_layer=3", "optim.lr=1e-2", "note=set"]
    )
    assert cfg.model.n_layer == 3
    assert cfg.optim.lr == 1e-2
    assert cfg.note == "set"


def test_config_file_is_not_modified_on_disk(config_path):
    """Overrides rewrite the parsed dict, not the file it came from."""
    before = config_path.read_text()
    load_run_config(config_path, ARCHES, DATAS, ["model.n_layer=3"])
    assert config_path.read_text() == before
