"""Command-line parsing, and in particular that it has no built-in paths.

`parser.py` used to work out `REPO_ROOT` from its own `__file__` and put the
config and runs directories under it. That only works for an editable checkout
of this repo: installed as a dependency it points into `site-packages/`, and
nothing raises. So the important tests here check that things are *absent*: no
path constants, and a flag with no project default is `required` rather than
quietly wrong.

Cheap tests: argparse and a list of strings. No config file, no run directory,
no project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_harness import parser as parser_mod
from ml_harness.parser import parse_args, parse_bench_args, parse_sweep_args


# ---- the absence ----------------------------------------------------------

@pytest.mark.parametrize("name", ["REPO_ROOT", "CONFIG_ROOT", "SWEEP_ROOT", "RUNS_ROOT"])
def test_the_module_exports_no_paths(name):
    """The old bug, written as a test so it can't come back.

    If any of these reappear, someone has reintroduced a path based on
    `__file__`, which points at *a* directory wherever it's installed, just not
    the caller's.
    """
    assert not hasattr(parser_mod, name)


def test_the_module_does_not_touch_the_filesystem():
    """`parents[...]` is exactly what the old bug looked like."""
    source = Path(parser_mod.__file__).read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    # the module docstring describes the old bug, so only look past it
    body = code.split('"""', 2)[-1]
    assert "parents[" not in body
    assert "__file__" not in body


# ---- a missing default is required, not wrong -----------------------------

def test_config_is_required_when_the_project_names_no_default():
    with pytest.raises(SystemExit):
        parse_args(runs_root_default="/runs", argv=[])


def test_runs_root_is_required_when_the_project_names_no_default():
    with pytest.raises(SystemExit):
        parse_args(config_default="/c.json", argv=[])


def test_both_defaults_supplied_needs_no_flags():
    args = parse_args(config_default="/c.json", runs_root_default="/runs", argv=[])
    assert args.config == Path("/c.json")
    assert args.runs_root == Path("/runs")
    assert args.set == [] and args.smoke is False


def test_an_explicit_flag_beats_the_project_default():
    args = parse_args(
        config_default="/c.json",
        runs_root_default="/runs",
        argv=["--config", "/other.json", "--runs-root", "/elsewhere"],
    )
    assert args.config == Path("/other.json")
    assert args.runs_root == Path("/elsewhere")


def test_defaults_come_back_as_paths_not_strings():
    """`run()` does `Path(runs_root) / ...`, so a str default would still work
    there and hide the inconsistency until something else used it."""
    args = parse_args(config_default="/c.json", runs_root_default="/runs", argv=[])
    assert isinstance(args.config, Path) and isinstance(args.runs_root, Path)


# ---- --set accumulates, in order ------------------------------------------

def test_set_repeats_in_order():
    """Order matters: `apply_overrides` applies left to right, which is what
    makes `--smoke --set train.max_iters=50` a longer smoke run."""
    args = parse_args(
        config_default="/c.json",
        runs_root_default="/runs",
        argv=["--set", "a=1", "--set", "b=2"],
    )
    assert args.set == ["a=1", "b=2"]


# ---- the two other CLIs ---------------------------------------------------

def test_bench_requires_a_config_without_a_default():
    with pytest.raises(SystemExit):
        parse_bench_args(argv=[])


def test_bench_has_its_own_knobs_and_no_runs_root():
    args = parse_bench_args(config_default="/c.json", argv=["--iters", "3"])
    assert args.iters == 3 and args.warmup == 5 and args.json is None
    assert not hasattr(args, "runs_root")


def test_sweep_config_stays_optional_with_no_default():
    """The exception: leaving it out means "use the sweep file's base
    config"."""
    args = parse_sweep_args(runs_root_default="/runs", argv=["--sweep", "/s.json"])
    assert args.config is None
    assert args.sweep == Path("/s.json")


def test_sweep_still_requires_the_sweep_file():
    with pytest.raises(SystemExit):
        parse_sweep_args(runs_root_default="/runs", argv=[])


def test_sweep_runs_root_is_required_when_unset():
    with pytest.raises(SystemExit):
        parse_sweep_args(argv=["--sweep", "/s.json"])
