"""Command-line parsing, and nothing else.

**This module has no built-in paths.** It used to compute `REPO_ROOT` as
`Path(__file__).resolve().parents[3]` and put `configs/`, `configs/sweeps/` and
`runs/` under it. That only works if the file is where you expect in one
particular checkout. Once the harness is installed as a dependency,
`parents[3]` lands in `site-packages/`, and nothing complains: the defaults
still point at *a* path, just not one that exists.

So the defaults are passed in as arguments. Every entry point already knows its
own repo root (it needs one for provenance anyway), and the rest of
`ml_harness/` already works this way: `collect_provenance(repo_root=...)`,
`run_sweep(repo_root=...)`, `record_run(repo_root=...)`. The rule is:
**`ml_harness/` never makes up a path it wasn't given.**

A default of `None` means the project didn't supply one, which makes the flag
`required` rather than quietly pointing somewhere wrong. The one exception is
`--config` in `parse_sweep_args`, where leaving it out means "use the sweep
file's own base config".

`argv` is a parameter so these can be tested, and so a notebook can pass a list
instead of editing `sys.argv`.
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

    No `--runs-root` and no `--smoke`: a benchmark doesn't write a run
    directory, and `--smoke` shrinks the model, which is what's being timed.
    """
    parser = argparse.ArgumentParser()

    _add_config(parser, config_default)
    _add_set(
        parser,
        "Override config values (can be repeated); this is how you vary a config",
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
    """The sweep CLI.

    Has very few flags on purpose: the grid goes in the sweep file, so a sweep
    is a file you can diff and re-run, rather than a command in your shell
    history.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Path to a sweep JSON (see configs/sweeps/)",
    )

    # Optional even without a project default: leaving it out means "use the
    # base config the sweep file names".
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
        help="Run every point at smoke size, to check the whole grid in a minute",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded points and exit, to check the count before a long sweep",
    )

    return parser.parse_args(argv)
