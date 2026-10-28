"""The analysis layer against hand-built frames.

Cheap tests: these never touch a directory. They check the step from
"24 runs x 72 columns" to something you can say about an experiment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml_harness.analysis import (
    curves,
    group_key,
    run_table,
    seed_groups,
    varying,
)


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def run(run_id: str, lr: float, seed: int, best: float, group: str | None = None) -> dict:
    row = {
        "run_id": run_id,
        "config.arch": "gpt",
        "config.model.n_layer": 4,
        "config.optim.lr": lr,
        "config.runtime.seed": seed,
        "config.note": "",
        "summary.val.loss.min": best,
        "summary.val.loss.final": best + 0.01,
        "summary.steps_completed": 200,
        "summary.elapsed_s": 10.0,
    }
    if group is not None:
        row["provenance.group_sha256"] = group
    return row


# ---- varying: the frame's actual axes -------------------------------------

def test_varying_finds_only_the_columns_that_differ():
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 3e-4, 0, 2.4)])
    assert varying(runs) == ["config.optim.lr"]


def test_varying_ignores_non_config_columns():
    """`summary.*` differs on every row by construction; it isn't an axis."""
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 1e-3, 0, 2.4)])
    assert varying(runs) == []


def test_a_column_only_some_runs_have_is_not_an_axis():
    """A field added after some runs were written: NaN in the old rows, one
    value in the new ones. That's a gap in the records, not an ablation."""
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 1e-3, 0, 2.4)])
    runs.loc[1, "config.train.top_k"] = 50
    assert "config.train.top_k" not in varying(runs)


def test_an_unhashable_config_value_does_not_raise():
    """`special_tokens` is a list, and `nunique()` raises on lists, so one
    tokenizer config would break every table."""
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 1e-3, 0, 2.4)])
    runs["config.tokenizer.special_tokens"] = [["<|end|>"], ["<|end|>", "<|pad|>"]]
    assert varying(runs) == ["config.tokenizer.special_tokens"]


# ---- run_table -------------------------------------------------------------

def test_run_table_is_the_axes_plus_the_score():
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 3e-4, 0, 2.4)])
    table = run_table(runs)

    assert list(table.columns) == [
        "run_id",
        "config.optim.lr",
        "summary.steps_completed",
        "summary.elapsed_s",
        "summary.val.loss.min",
        "summary.val.loss.final",
    ]


def test_run_table_sorts_by_best_not_final():
    """The best checkpoint is chosen by the min, so the leaderboard is too."""
    runs = frame([run("worse", 1e-3, 0, 2.5), run("better", 3e-4, 0, 2.4)])
    assert list(run_table(runs)["run_id"]) == ["better", "worse"]


def test_run_table_survives_a_missing_column():
    """A crashed run has no summary at all; the table should still render."""
    runs = frame([run("a", 1e-3, 0, 2.5)]).drop(columns=["summary.elapsed_s"])
    assert "summary.elapsed_s" not in run_table(runs).columns


# ---- grouping --------------------------------------------------------------

def test_group_key_prefers_the_recorded_hash():
    runs = frame([
        run("a", 1e-3, 0, 2.5, group="deadbeef"),
        run("b", 3e-4, 1, 2.4, group="deadbeef"),
    ])
    # Different lr, same recorded hash: the record wins over the frame, because
    # the record saw the whole config and the frame only sees these columns.
    assert group_key(runs).nunique() == 1


def test_group_key_falls_back_to_the_varying_columns():
    """Runs written before the hash existed still group."""
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 1e-3, 1, 2.4), run("c", 3e-4, 0, 2.3)])
    keys = group_key(runs)

    assert keys[0] == keys[1]  # same lr, different seed
    assert keys[0] != keys[2]


def test_a_partially_hashed_frame_uses_the_fallback():
    """Mixing old and new runs mustn't put every old run in one NaN group."""
    runs = frame([run("a", 1e-3, 0, 2.5, group="deadbeef"), run("b", 1e-3, 1, 2.4)])
    assert group_key(runs).notna().all()


# ---- seed_groups -----------------------------------------------------------

def test_seed_groups_averages_over_seeds():
    runs = frame([run("a", 1e-3, 0, 2.0), run("b", 1e-3, 1, 3.0)])
    out = seed_groups(runs)

    assert len(out) == 1
    assert out.loc[0, "mean"] == pytest.approx(2.5)
    assert out.loc[0, "n"] == 2


def test_seed_groups_keeps_the_axes_so_a_row_says_what_it_is():
    runs = frame([run("a", 1e-3, 0, 2.5), run("b", 3e-4, 0, 2.4)])
    out = seed_groups(runs)

    assert len(out) == 2
    assert "config.optim.lr" in out.columns
    assert out.loc[0, "config.optim.lr"] == 3e-4  # sorted by mean, best first


def test_a_single_seed_has_no_spread():
    """NaN rather than 0.0: one seed has no spread, which is why you want at
    least 3."""
    runs = frame([run("a", 1e-3, 0, 2.5)])
    out = seed_groups(runs)

    assert out.loc[0, "n"] == 1
    assert pd.isna(out.loc[0, "std"])


def test_seed_groups_says_which_column_is_missing():
    runs = frame([run("a", 1e-3, 0, 2.5)]).drop(columns=["summary.val.loss.min"])
    with pytest.raises(KeyError, match="summary.val.loss.min"):
        seed_groups(runs)


# ---- curves ----------------------------------------------------------------

def test_curves_pivots_long_metrics_into_a_step_by_run_grid():
    metrics = frame([
        {"step": 0, "split": "val", "metric": "loss", "value": 3.0, "run_id": "a"},
        {"step": 1, "split": "val", "metric": "loss", "value": 2.0, "run_id": "a"},
        {"step": 0, "split": "val", "metric": "loss", "value": 3.5, "run_id": "b"},
        {"step": 0, "split": "train", "metric": "loss", "value": 9.9, "run_id": "a"},
        {"step": 0, "split": "val", "metric": "batch_loss", "value": 9.9, "run_id": "a"},
    ])
    wide = curves(metrics)

    assert list(wide.columns) == ["a", "b"]
    assert wide.loc[0, "a"] == 3.0
    # b logged nothing at step 1: NaN, not a carried-forward wrong value.
    assert pd.isna(wide.loc[1, "b"])
