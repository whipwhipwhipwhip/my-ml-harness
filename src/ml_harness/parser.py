"""The command-line edge, and nothing else.

**This module knows no paths.** It used to compute `REPO_ROOT` as
`Path(__file__).resolve().parents[3]` and hang `configs/`, `configs/sweeps/` and
`runs/` off it — an assertion about where this file sits in somebody's tree,
which is true exactly once. Installed as a dependency rather than an editable
checkout, `parents[3]` lands in `site-packages/`, and the failure is quiet: the
defaults still resolve to *a* path, just not one that exists.

So the defaults arrive as arguments. Every entry point already computes its own
repo root — it needs one for provenance anyway — and passing them down is the
same rule the rest of `ml_harness/` follows: `collect_provenance(repo_root=...)`,
`run_sweep(repo_root=...)`, `record_run(repo_root=...)`. The house rule is now
uniform: **`ml_harness/` resolves no path it wasn't given.**

A default of `None` means "the project didn't supply one", which makes the flag
`required` rather than silently pointing somewhere wrong. The one exception is
`parse_sweep_args`' `--config`, where absent genuinely means "don't override the
sweep file's own base".

`argv` is threaded through so these can be tested, and so a notebook can call one
with a list instead of editing `sys.argv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _add_config(parser: argparse.ArgumentParser, default: Path | str | None) -> None:
    """`--config`, required exactly when the caller named no default."""
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
        required=default is None,
        help="Path to config JSON"
        + (f" (default: {default})" if default is not None else ""),
    )


def _add_set(parser: argparse.ArgumentParser, help: str) -> None:
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help=help,
    )


def _add_runs_root(parser: argparse.ArgumentParser, default: Path | str | None) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default,
        required=default is None,
        help="Override output directory"
        + (f" (default: {default})" if default is not None else ""),
    )


def parse_args(
    *,
    config_default: Path | str | None = None,
    runs_root_default: Path | str | None = None,
    argv: Sequence[str] | None = None,
):
    """The training CLI. Defaults are the project's to choose."""
    parser = argparse.ArgumentParser()

    _add_config(parser, config_default)

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Shrink the config to a canned tiny run and write it under "
            "<runs-root>/smoke_tests/. Explicit --set still wins."
        ),
    )

    _add_runs_root(parser, runs_root_default)
    _add_set(parser, "Override config values (can be repeated)")

    return parser.parse_args(argv)


def parse_bench_args(
    *,
    config_default: Path | str | None = None,
    argv: Sequence[str] | None = None,
):
    """The benchmark CLI. Same `--config`/`--set` edge as `parse_args`.

    No `--runs-root` and no `--smoke`: a benchmark writes no run directory, and
    smoke's whole point is shrinking the model, which is the thing being timed.
    """
    parser = argparse.ArgumentParser()

    _add_config(parser, config_default)
    _add_set(
        parser,
        "Override config values (can be repeated) — this is how you vary a config",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measured iterations (default 100). train.max_iters is ignored",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Unmeasured iterations first, to absorb kernel compilation (default 5)",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the report to this path, for comparing configs later",
    )

    return parser.parse_args(argv)


def parse_sweep_args(
    *,
    runs_root_default: Path | str | None = None,
    argv: Sequence[str] | None = None,
):
    """The sweep CLI. Separate function, same edge: argparse lives in `main()`.

    Deliberately thin — the grid is in the sweep file, not in flags, so that a
    sweep is an artifact you can diff and re-run rather than a shell line in
    somebody's history.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Path to a sweep JSON (see configs/sweeps/)",
    )

    # The one flag that stays optional with no project default: absent means
    # "use the base the sweep file names", not "the project forgot to say".
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the sweep file's base config",
    )

    _add_runs_root(parser, runs_root_default)
    _add_set(parser, "Override config values on every point (applied last, so it wins)")

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run every point at smoke size — validates the whole grid in a minute",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded points and exit. Read the count before committing an hour",
    )

    return parser.parse_args(argv)
