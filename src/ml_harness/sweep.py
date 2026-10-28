"""Running one config many times over a grid of overrides.

**Each point in a sweep is just a list of `--set` strings.** So a sweep is
validated exactly like a value typed on the command line (`apply_overrides` ->
dacite `strict=True` -> `__post_init__`), a typo'd axis fails at load like a
typo'd JSON key, and this module doesn't need its own schema for what it varies.
`--smoke` works the same way.

Like the rest of `ml_harness/`, it names no model and no task. `run_fn` is
passed in the same way `ARCHES` and `DATAS` are, because the code that does a
single run belongs to a project (e.g. `testing.run`, or a project's `train.run`).
The contract is three arguments:

    run_fn(config_path, overrides, runs_root) -> RunDir

and a project's `sweep.py` binds anything else (like `cache_dir`) with `partial`.
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

from ml_harness.io import _safe_load, _write_json

# Strict like the run configs: a misspelled key in a sweep file would otherwise
# mean the sweep quietly doesn't vary what you meant.
_DACITE = Config(strict=True)

# Where sweeps go under --runs-root. It's a directory *of* runs, not a run,
# which `load_runs` handles by skipping directories with no config.json.
SWEEPS_SUBDIR = "sweeps"

SEED_KEY = "runtime.seed"


class _RunDirLike(Protocol):
    """The part of `store.RunDir` a sweep uses. A Protocol so the tests can
    pass a stub instead of building a real run directory."""

    run_id: str

    @property
    def path(self) -> Path: ...

    @property
    def status_path(self) -> Path: ...


RunFn = Callable[[Path, Sequence[str], Path], _RunDirLike]


@dataclass(frozen=True)
class SweepConfig:
    """A grid of runs, as a file you can diff and keep.

    Frozen and parsed with dacite like `RunConfig`. `sweep.json` in the
    output directory is this spec plus what happened to each point.
    """

    name: str
    # Relative to the repo root, like `data.path`, so it points at the same
    # file whatever directory you run from.
    base: str
    # Applied to every point, e.g. "everything in this sweep runs for 2000
    # iters and writes no samples".
    constants: dict[str, Any] = field(default_factory=dict)
    # The axes. Points are the Cartesian product, in the order the keys are
    # written, so the same file always expands to the same list.
    grid: dict[str, list[Any]] = field(default_factory=dict)
    # Shorthand for a `runtime.seed` axis: an int N means seeds 0..N-1, a list
    # means exactly those seeds. Separate from `grid` because almost every
    # comparison needs multiple seeds, so it should be easy to add.
    #
    # `eval_seed` isn't varied with it: the fixed eval batches are what make
    # val losses comparable between runs.
    seeds: int | list[int] | None = None
    note: str = ""

    def __post_init__(self):
        if self.seeds is not None and SEED_KEY in self.grid:
            raise ValueError(
                f"seeds and grid[{SEED_KEY!r}] both set — pick one; "
                f"`seeds` is sugar for exactly that axis"
            )

        # If a key is in both, grid would win (it's applied later), but the file
        # doesn't make that obvious. Better to fail than pick one silently.
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

    `json.dumps` rather than `str`, so the value parses back to the same thing:
    `None` stays None, `True` stays a bool, and a string with a newline in it
    survives. (On the command line that last one needs awkward quoting, like
    `--set 'train.sample_prompt="\\nROMEO:"'`, but a sweep file never goes
    through a shell.)
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

    Seeds go last so the points are ordered "all seeds of config A, then all
    seeds of config B". That way a sweep that gets killed leaves complete groups
    rather than one seed of everything, and you can only compare complete groups.
    """
    values = seed_values(spec.seeds)
    return {**spec.grid, **({SEED_KEY: values} if values else {})}


def expand(spec: SweepConfig) -> list[list[str]]:
    """The spec as a list of points, each a list of `--set` strings.

    Constants first, then the point. Overrides are applied in order so later
    ones win, which gives smoke < constants < point < CLI --set.

    An empty grid with no seeds is one point: the base config with the
    constants applied. That isn't an error; it's how you'd write "re-run this
    config with more iterations".
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
    """Just the axes, for a progress line (constants are the same every time)."""
    varying = set(axes(spec))
    return " ".join(o for o in point if o.split("=", 1)[0] in varying) or "(base)"


# ---- running --------------------------------------------------------------

def new_sweep_dir(root: Path, name: str) -> Path:
    """`<root>/sweeps/<timestamp>_<name>/`.

    Uses the sweep's name rather than a uuid like runs do, since there aren't
    many sweeps and you refer to them by hand. No `exist_ok`, so two sweeps that
    would share a directory fail instead of mixing their records.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(root) / SWEEPS_SUBDIR / f"{ts}_{name}"
    path.mkdir(parents=True)
    return path


def _status_of(rd: _RunDirLike) -> str:
    """The status the run recorded. `record_run` catches KeyboardInterrupt and
    returns normally, so the status file is the *only* way to tell the user
    pressed Ctrl-C. Without checking it, the sweep would just move on to the
    next point."""
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

    Runs one point at a time in this process: one torch import, one device,
    and the grid is already ordered so a partial sweep is still useful. Running
    points in parallel would probably not help on a single MPS device anyway.

    `overrides` are the CLI's `--set`, applied last so they win. Handy for
    `--set runtime.device=cpu` across a whole grid. There's a warning if one
    names an axis, because fixing an axis from the command line turns the grid
    into duplicate points that look different in the record.

    Returns the sweep directory. `sweep.json` in it is rewritten after every
    point, so a sweep killed at point 7 of 50 still has a record of the first 6.
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
                # Ctrl-C between points rather than during one. Same meaning.
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
                # If the *first* point fails it's almost always a problem with
                # the whole spec (a bad constant, a missing base config), and 49
                # more identical tracebacks won't help. Later failures are
                # treated as that point's problem and the sweep continues.
                # (Running the grid with `--smoke` first catches both cheaply.)
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
