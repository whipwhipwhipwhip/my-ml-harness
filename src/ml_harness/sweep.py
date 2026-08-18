"""Running one config many times over a grid of overrides.

**A sweep point is a list of `--set` strings.** That's the whole design. It means
a sweep is validated by exactly the path a hand-typed config value is —
`apply_overrides` -> dacite `strict=True` -> `__post_init__` — so a typo'd axis
fails at load like a typo'd JSON key, and this module needs no schema of its own
for the things it varies. `--smoke` reuses the same trick for the same reason.

Generic, like the rest of `harness/`: it names no model and no task. `run_fn` is
injected the way `ARCHES` and `DATAS` are, because the thing that knows how to do
one run is a project entry point (`nanogpt_conor.train.run`) and an import of it
here would be the back-arrow the layering rule exists to prevent. The two entry
points' `run()` differ only in `cache_dir`, so the contract is the three
arguments they share:

    run_fn(config_path, overrides, runs_root) -> RunDir

with the project's `sweep.py` binding the rest via `partial`.
"""

from __future__ import annotations

import itertools
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from dacite import Config, from_dict

from ml_learning.harness.io import _safe_load, _write_json

# Same strictness as the run configs, and for the same reason: a misspelled key
# in a sweep file is a sweep that quietly doesn't vary what you meant.
_DACITE = Config(strict=True)

# Where sweeps live under --runs-root. A directory *of* runs, not a run — which
# `load_runs` already tolerates, since it skips directories with no config.json.
SWEEPS_SUBDIR = "sweeps"

SEED_KEY = "runtime.seed"


class _RunDirLike(Protocol):
    """The part of `store.RunDir` a sweep uses. Kept structural so the loop
    tests can hand this a stub without building a real run directory."""

    run_id: str

    @property
    def path(self) -> Path: ...

    @property
    def status_path(self) -> Path: ...


RunFn = Callable[[Path, Sequence[str], Path], _RunDirLike]


@dataclass(frozen=True)
class SweepConfig:
    """A grid of runs, as a file you can diff and keep.

    Frozen and dacite-parsed like `RunConfig`, so a sweep is as much of an
    artifact as a run is: `sweep.json` in the output directory is this spec plus
    what happened to each point.
    """

    name: str
    # Relative to the repo root, like `data.path` — so a sweep file names the
    # same base config from any working directory.
    base: str
    # Applied to every point. The place for "everything in this sweep runs for
    # 2000 iters and writes no samples".
    constants: dict[str, Any] = field(default_factory=dict)
    # The axes. Cartesian product, in declared key order, so point ordering is
    # deterministic and a re-expansion of the same file gives the same list.
    grid: dict[str, list[Any]] = field(default_factory=dict)
    # Sugar for a `runtime.seed` axis: an int N means seeds 0..N-1, a list means
    # exactly those. Separate from `grid` because it's the axis every comparison
    # claim needs and it shouldn't take a line of ceremony to add.
    #
    # `eval_seed` is deliberately not swept with it: the fixed eval batches are
    # what make two runs' val losses comparable, and varying the eval seed makes
    # them measurements of different things.
    seeds: int | list[int] | None = None
    note: str = ""

    def __post_init__(self):
        if self.seeds is not None and SEED_KEY in self.grid:
            raise ValueError(
                f"seeds and grid[{SEED_KEY!r}] both set — pick one; "
                f"`seeds` is sugar for exactly that axis"
            )

        # Overlap would silently resolve one way (grid wins, since it's applied
        # later) and read as if it resolved the other. The spec is small and
        # hand-written; ambiguity should fail rather than pick.
        clash = sorted(set(self.constants) & set(self.grid))
        if clash:
            raise ValueError(f"keys in both constants and grid: {clash}")

        for key, values in self.grid.items():
            if not values:
                raise ValueError(f"grid[{key!r}] is empty; an axis needs a value")


def load_sweep_config(path: Path | str) -> SweepConfig:
    return from_dict(SweepConfig, json.loads(Path(path).read_text("utf-8")), _DACITE)


# ---- expansion ------------------------------------------------------------

def to_override(key: str, value: Any) -> str:
    """One `key=value` string, encoded so `_parse_value` gets the type back.

    `json.dumps` rather than `str`: it round-trips through the JSON-with-string-
    fallback parser on the other side, so `None` stays None, `True` stays a
    bool, and a string containing a newline survives — which is the documented
    `--set 'train.sample_prompt="\\nROMEO:"'` quoting footgun, fixed here for
    free because a sweep file never goes through a shell.
    """
    return f"{key}={json.dumps(value)}"


def seed_values(seeds: int | list[int] | None) -> list[int]:
    if seeds is None:
        return []
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError(f"seeds must be >= 1, got {seeds}")
        return list(range(seeds))
    return list(seeds)


def axes(spec: SweepConfig) -> dict[str, list[Any]]:
    """The grid, with the `seeds` sugar expanded into an ordinary axis.

    Seeds go last so that points are ordered "all seeds of config A, then all
    seeds of config B" — a killed sweep then leaves complete groups rather than
    one seed of everything, and complete groups are the only ones you can make a
    claim from.
    """
    values = seed_values(spec.seeds)
    return {**spec.grid, **({SEED_KEY: values} if values else {})}


def expand(spec: SweepConfig) -> list[list[str]]:
    """The spec as a list of points, each a list of `--set` strings.

    Constants first, then the point, matching the precedence the caller applies
    on top (smoke < constants < point < CLI --set): overrides are applied in
    order, so later wins.

    An empty grid with no seeds is one point — the base config with the
    constants applied. That's a degenerate sweep, not an error; it's what
    "re-run this config with max_iters bumped" looks like.
    """
    constants = [to_override(k, v) for k, v in spec.constants.items()]
    grid = axes(spec)

    if not grid:
        return [constants]

    return [
        constants + [to_override(k, v) for k, v in zip(grid, combo)]
        for combo in itertools.product(*grid.values())
    ]


def describe(point: Sequence[str], spec: SweepConfig) -> str:
    """Just the axes, for a progress line — constants are the same on every row."""
    varying = set(axes(spec))
    return " ".join(o for o in point if o.split("=", 1)[0] in varying) or "(base)"


# ---- running --------------------------------------------------------------

def new_sweep_dir(root: Path, name: str) -> Path:
    """`<root>/sweeps/<timestamp>_<name>/`.

    Named rather than uuid'd, unlike a run: there are few sweeps and you refer
    to them by hand. `mkdir` without `exist_ok`, so two sweeps that would share
    a directory fail loudly instead of interleaving their records.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(root) / SWEEPS_SUBDIR / f"{ts}_{name}"
    path.mkdir(parents=True)
    return path


def _status_of(rd: _RunDirLike) -> str:
    """What the run said about itself. `train.run` catches KeyboardInterrupt and
    returns normally, so the status file is the *only* signal that the user
    asked to stop — without reading it a sweep marches cheerfully on to the next
    point after a Ctrl-C."""
    data, err = _safe_load(rd.status_path)
    if err or not isinstance(data, dict):
        return "unknown"
    return data.get("status", "unknown")


def run_sweep(
    spec: SweepConfig,
    run_fn: RunFn,
    runs_root: Path | str,
    repo_root: Path | str,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    on_point: Callable[[str], None] = print,
) -> Path:
    """Run every point, sequentially, recording as it goes.

    Sequential and in-process on purpose: one torch import, one device, and a
    grid that's already ordered so a partial sweep is still readable. Parallelism
    is a later question and probably the wrong answer on a single MPS device.

    `overrides` are the CLI's `--set`, applied last so they win — the same rule
    as `--smoke`. Handy for `--set runtime.device=cpu` across a whole grid, and
    warned about when it names an axis, because pinning an axis from the command
    line collapses the grid into duplicate points that all look distinct in the
    record.

    Returns the sweep directory. `sweep.json` inside it is rewritten after every
    point, so a sweep killed at point 7 of 50 is still a readable record of 6.
    """
    points = expand(spec)
    base = Path(repo_root) / spec.base

    pinned = sorted({o.split("=", 1)[0] for o in overrides} & set(axes(spec)))
    if pinned:
        on_point(f"warning: --set pins swept {pinned}; those axes will not vary")

    if dry_run:
        on_point(f"{len(points)} points from {spec.base}:")
        for i, point in enumerate(points, 1):
            on_point(f"  [{i}/{len(points)}] {describe(point, spec)}")
        return Path(runs_root) / SWEEPS_SUBDIR

    sweep_dir = new_sweep_dir(runs_root, spec.name)
    record: dict[str, Any] = {
        "name": spec.name,
        "base": spec.base,
        "note": spec.note,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cli_overrides": list(overrides),
        "n_points": len(points),
        "points": [],
    }
    on_point(f"sweep: {sweep_dir} ({len(points)} points)")

    def flush():
        _write_json(sweep_dir / "sweep.json", record)

    flush()
    try:
        for i, point in enumerate(points, 1):
            label = describe(point, spec)
            on_point(f"[{i}/{len(points)}] {label}")

            entry: dict[str, Any] = {"index": i, "overrides": list(point)}
            started = datetime.now(timezone.utc)

            try:
                rd = run_fn(base, [*point, *overrides], sweep_dir)
                entry["run_id"] = rd.run_id
                entry["status"] = _status_of(rd)
            except KeyboardInterrupt:
                # Ctrl-C between points, rather than inside one. Same meaning.
                entry["status"] = "interrupted"
            except Exception:
                entry["status"] = "failed"
                entry["error"] = traceback.format_exc()

            entry["elapsed_s"] = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            record["points"].append(entry)
            flush()

            if entry["status"] == "interrupted":
                on_point("interrupted — stopping the sweep")
                break

            if entry["status"] == "failed":
                on_point(f"  failed: {entry.get('error', '').strip().splitlines()[-1:]}")
                # A failure on the *first* point is almost always a spec-wide
                # problem — a bad constant, a missing base config — and 49 more
                # identical tracebacks help nobody. Later failures are treated as
                # this point's problem and the grid continues. (`--smoke` over
                # the whole grid is the cheaper way to find both.)
                if i == 1:
                    on_point("first point failed — stopping; fix the spec")
                    break
    finally:
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        record["completed"] = sum(
            p["status"] == "completed" for p in record["points"]
        )
        flush()
        on_point(f"[{record['completed']}/{len(points)} completed] {sweep_dir}")

    return sweep_dir
