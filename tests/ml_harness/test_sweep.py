"""Sweeps, in two tiers.

Expansion is a pure function of dicts, so most of this file runs in
milliseconds. The driver is tested with a fake `run_fn`, like `test_loop.py`
does with the model: if running a sweep ever needs a real GPT, the sweep depends
on something it shouldn't.

There's one real run at the bottom, to check that `testing.run` actually works
as a `run_fn`, not just the fake.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml_harness.store import RunDir, load_runs
from ml_harness.sweep import (
    SweepConfig,
    describe,
    expand,
    load_sweep_config,
    run_sweep,
    to_override,
)


def spec(**overrides) -> SweepConfig:
    base = {"name": "test", "base": "configs/GPT.json"}
    base.update(overrides)
    return SweepConfig(**base)


# ---- override encoding ----------------------------------------------------

def test_values_round_trip_through_the_override_parser():
    """The sweep only talks to `apply_overrides` through these strings. If they
    don't parse back to the same value, the config gets the wrong one."""
    from ml_harness.registry import _parse_value

    for value in (1e-3, 4, True, None, "mps", "\nROMEO:", [1, 2]):
        key, text = to_override("k", value).split("=", 1)
        assert _parse_value(text) == value


def test_a_string_with_a_newline_needs_no_shell_quoting():
    """On the command line this needs awkward quoting
    (`--set 'train.sample_prompt="\\nROMEO:"'`), but a sweep file never goes
    through a shell, so here it's just a value."""
    assert to_override("train.sample_prompt", "\nROMEO:") == (
        'train.sample_prompt="\\nROMEO:"'
    )


# ---- expansion ------------------------------------------------------------

def test_the_grid_is_a_cartesian_product():
    points = expand(spec(grid={"optim.lr": [1e-3, 1e-4], "model.n_layer": [2, 4]}))
    assert len(points) == 4


def test_seeds_are_the_innermost_axis():
    """So a killed sweep leaves *complete* groups: you can't conclude much
    from a group with two of its three seeds."""
    points = expand(spec(grid={"optim.lr": [1e-3, 1e-4]}, seeds=2))
    seeds = [p[-1] for p in points]

    assert seeds == ["runtime.seed=0", "runtime.seed=1"] * 2


def test_constants_come_before_the_point_so_the_point_wins():
    """Overrides apply in order, and that's how precedence works."""
    points = expand(spec(constants={"train.max_iters": 500}, grid={"optim.lr": [1e-3]}))
    assert points[0] == ["train.max_iters=500", "optim.lr=0.001"]


def test_constants_are_on_every_point():
    points = expand(spec(constants={"train.write_samples": False}, seeds=3))
    assert all("train.write_samples=false" in p for p in points)


def test_an_empty_spec_is_one_point():
    """Not an error: "re-run this config with more iterations" is a reasonable
    thing to keep as a file."""
    assert expand(spec(constants={"train.max_iters": 10})) == [["train.max_iters=10"]]


def test_a_seed_count_expands_to_that_many_seeds():
    assert len(expand(spec(seeds=3))) == 3


def test_an_explicit_seed_list_is_used_as_given():
    points = expand(spec(seeds=[7, 11]))
    assert [p[-1] for p in points] == ["runtime.seed=7", "runtime.seed=11"]


def test_describe_shows_the_axes_and_hides_the_constants():
    s = spec(constants={"train.max_iters": 500}, grid={"optim.lr": [1e-3]})
    assert describe(expand(s)[0], s) == "optim.lr=0.001"


# ---- spec validation ------------------------------------------------------

def test_seeds_and_a_seed_axis_together_is_an_error():
    with pytest.raises(ValueError, match="pick one"):
        spec(seeds=3, grid={"runtime.seed": [1, 2]})


def test_a_key_in_both_constants_and_grid_is_an_error():
    """Grid would win, but the file doesn't make that obvious, so it's an error."""
    with pytest.raises(ValueError, match="both constants and grid"):
        spec(constants={"optim.lr": 1e-3}, grid={"optim.lr": [1e-4]})


def test_an_empty_axis_is_an_error():
    with pytest.raises(ValueError, match="needs a value"):
        spec(grid={"optim.lr": []})


def test_zero_seeds_is_an_error():
    with pytest.raises(ValueError, match=">= 1"):
        expand(spec(seeds=0))


def test_a_typod_sweep_key_is_rejected_at_load(tmp_path):
    """`strict=True`, like the run configs: otherwise a misspelled key means the
    sweep quietly doesn't vary what you meant."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"name": "x", "base": "b", "gird": {"a": [1]}}))

    with pytest.raises(Exception):
        load_sweep_config(path)


# ---- the driver, against a fake run ---------------------------------------

class FakeRun:
    """A `run_fn` that writes only what a sweep reads back: a status file."""

    def __init__(self, statuses=None, raises=()):
        self.statuses = statuses or {}
        self.raises = raises
        self.calls = []

    def __call__(self, config_path, overrides, runs_root):
        i = len(self.calls) + 1
        self.calls.append((config_path, list(overrides), runs_root))

        if i in self.raises:
            raise RuntimeError("boom")

        rd = RunDir(root=runs_root, run_id=f"run{i}")
        rd.path.mkdir(parents=True)
        rd.status_path.write_text(
            json.dumps({"status": self.statuses.get(i, "completed")})
        )
        return rd


def read_record(sweep_dir):
    return json.loads((sweep_dir / "sweep.json").read_text())


def test_every_point_runs_and_is_recorded(tmp_path):
    fake = FakeRun()
    out = run_sweep(spec(seeds=3), fake, tmp_path, "/repo", on_point=lambda _: None)

    record = read_record(out)
    assert len(fake.calls) == 3
    assert record["completed"] == 3
    assert [p["run_id"] for p in record["points"]] == ["run1", "run2", "run3"]


def test_the_base_config_is_resolved_against_the_repo_root(tmp_path):
    fake = FakeRun()
    run_sweep(spec(), fake, tmp_path, "/repo", on_point=lambda _: None)

    assert str(fake.calls[0][0]) == "/repo/configs/GPT.json"


def test_cli_overrides_are_applied_last(tmp_path):
    """The same rule as --smoke: explicit beats the file."""
    fake = FakeRun()
    run_sweep(
        spec(constants={"optim.lr": 1e-3}),
        fake,
        tmp_path,
        "/repo",
        ["runtime.device=cpu"],
        on_point=lambda _: None,
    )

    assert fake.calls[0][1] == ["optim.lr=0.001", "runtime.device=cpu"]


def test_pinning_a_swept_axis_warns(tmp_path):
    said = []
    run_sweep(
        spec(seeds=2), FakeRun(), tmp_path, "/repo", ["runtime.seed=5"], on_point=said.append
    )

    assert any("will not vary" in line for line in said)


def test_a_failed_point_does_not_kill_the_sweep(tmp_path):
    fake = FakeRun(raises={2})
    out = run_sweep(spec(seeds=3), fake, tmp_path, "/repo", on_point=lambda _: None)

    record = read_record(out)
    assert len(fake.calls) == 3
    assert [p["status"] for p in record["points"]] == ["completed", "failed", "completed"]
    assert "boom" in record["points"][1]["error"]


def test_a_failure_on_the_first_point_stops_the_sweep(tmp_path):
    """If point 1 fails it's almost always a problem with the whole spec, and
    49 identical tracebacks won't help."""
    fake = FakeRun(raises={1})
    out = run_sweep(spec(seeds=5), fake, tmp_path, "/repo", on_point=lambda _: None)

    assert len(fake.calls) == 1
    assert read_record(out)["completed"] == 0


def test_an_interrupted_run_stops_the_sweep(tmp_path):
    """`record_run` catches KeyboardInterrupt and returns normally, so the
    status file is the only way to tell the user pressed Ctrl-C. Without
    checking it, the sweep would move on to the next point."""
    fake = FakeRun(statuses={2: "interrupted"})
    out = run_sweep(spec(seeds=4), fake, tmp_path, "/repo", on_point=lambda _: None)

    assert len(fake.calls) == 2
    assert read_record(out)["points"][-1]["status"] == "interrupted"


def test_the_record_is_readable_after_every_point(tmp_path):
    """A sweep killed at point 7 of 50 still has a record of the first 6."""
    seen = []

    class Watching(FakeRun):
        def __call__(self, *args):
            seen.append(len(read_record(sweep_dir)["points"]))
            return super().__call__(*args)

    fake = Watching()
    sweep_dir = None

    # The directory name isn't known until run_sweep makes it, so grab it from
    # the first line it prints.
    def capture(line):
        nonlocal sweep_dir
        if line.startswith("sweep: "):
            sweep_dir = Path(line.removeprefix("sweep: ").split(" (")[0])

    run_sweep(spec(seeds=3), fake, tmp_path, "/repo", on_point=capture)

    assert seen == [0, 1, 2]


def test_dry_run_runs_nothing_and_writes_nothing(tmp_path):
    fake = FakeRun()
    said = []
    run_sweep(spec(seeds=2), fake, tmp_path, "/repo", dry_run=True, on_point=said.append)

    assert fake.calls == []
    assert not (tmp_path / "sweeps").exists()
    assert any("2 points" in line for line in said)


def test_two_sweeps_cannot_share_a_directory(tmp_path, monkeypatch):
    """`mkdir` without exist_ok: two sweeps mixing their records in one
    directory is worse than an error. The clock is frozen because the timestamp
    is the only thing that keeps two sweeps with the same name apart."""
    import ml_harness.sweep as sweep_mod

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(sweep_mod, "datetime", Frozen)

    assert sweep_mod.new_sweep_dir(tmp_path, "dup").exists()
    with pytest.raises(FileExistsError):
        sweep_mod.new_sweep_dir(tmp_path, "dup")


# ---- one real run ---------------------------------------------------------

def test_a_sweep_produces_readable_runs(tmp_path, config_path, runs_root):
    """The slow one: checks a real entry point's `run()` works as `run_fn`, not
    just the fake, and that `load_runs(sweep_dir)` gives the sweep's runs."""
    from functools import partial

    from ml_harness.testing import run

    out = run_sweep(
        SweepConfig(name="real", base=config_path.name, grid={"optim.lr": [1e-3, 1e-2]}),
        partial(run, cache_dir=None),
        runs_root,
        config_path.parent,
        on_point=lambda _: None,
    )

    frame = load_runs(out)
    assert len(frame) == 2
    assert sorted(frame["config.optim.lr"]) == [1e-3, 1e-2]
    # Two different configs, so two groups, and both hashes are recorded.
    assert frame["provenance.group_sha256"].nunique() == 2
