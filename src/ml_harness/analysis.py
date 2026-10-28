"""Reading a directory of runs back as a comparison.

`store.load_runs` tells you what ran; this helps you compare the results.
`store.py` handles paths and the run record, and knows nothing about comparing
runs.

Everything here takes and returns DataFrames, so it's mostly meant for
notebooks. It's in a module rather than notebook cells so it can be imported
and tested. Torch-free, like the rest of the record-reading code.
"""

from __future__ import annotations

import pandas as pd

from ml_harness.provenance import GROUP_EXCLUDES

# A frame of runs is ~70 columns wide, and most columns are the same for every
# run. These are worth showing even when they don't vary.
ALWAYS = ("run_id",)

# Config columns that describe the *replicate* rather than the experiment. They
# always vary within a group, so they're not interesting as ablation axes.
# Built from GROUP_EXCLUDES so the two can't get out of sync.
SEED_COLUMNS = tuple("config." + ".".join(path) for path in GROUP_EXCLUDES)


def varying(runs: pd.DataFrame, prefix: str = "config.") -> list[str]:
    """The config columns that actually differ across these runs.

    An ablation is a config diff, so the columns worth showing are the ones
    that aren't constant. A 72-column frame becomes the three columns the
    experiment was actually about.

    All-NaN columns are dropped (e.g. a field added after these runs), and so
    are columns that are constant once NaNs are ignored: a field only some runs
    have, with the same value in all of them, isn't an axis either.
    """
    cols = []

    for col in runs.columns:
        if not col.startswith(prefix):
            continue

        values = runs[col].dropna()
        if values.empty:
            continue

        # astype(str): a config value can be a list (`special_tokens`), which
        # isn't hashable, so nunique() would raise.
        if values.astype(str).nunique() > 1:
            cols.append(col)

    return cols


def run_table(
    runs: pd.DataFrame,
    metric: str = "val.loss",
    sort: bool = True,
) -> pd.DataFrame:
    """A leaderboard: what varied, and what it scored.

    `metric` is `split.metric` as `summarize` names it, so `"val.loss"` reads
    `summary.val.loss.min` and `.final` from the frame. Sorted by the min,
    since that's what the best checkpoint is chosen by.
    """
    score = f"summary.{metric}.min"
    cols = [
        *ALWAYS,
        *varying(runs),
        "summary.steps_completed",
        "summary.elapsed_s",
        score,
        f"summary.{metric}.final",
    ]
    table = runs[[c for c in cols if c in runs.columns]].copy()

    if sort and score in table.columns:
        table = table.sort_values(score)

    return table


def group_key(runs: pd.DataFrame) -> pd.Series:
    """Which rows are the same experiment, one Series aligned to `runs`.

    Uses `provenance.group_sha256` if every run has it (the config hashed
    without the seed and note, recorded by the run itself). Otherwise, for runs
    from before the hash existed, falls back to the varying config columns
    minus the seed ones, which is the same idea worked out from the frame.

    The fallback is weaker: it only sees columns present in this frame, so two
    runs that match on every column here are grouped together even if their
    configs differed in a field that wasn't recorded. Fine for reading old runs,
    but not something to rely on.
    """
    hashes = runs.get("provenance.group_sha256")
    if hashes is not None and hashes.notna().all():
        return hashes

    axes = [c for c in varying(runs) if c not in SEED_COLUMNS]
    if not axes:
        # Nothing varies: one experiment, however many replicates.
        return pd.Series("all", index=runs.index)

    return runs[axes].astype(str).agg(" | ".join, axis=1)


def seed_groups(runs: pd.DataFrame, metric: str = "val.loss") -> pd.DataFrame:
    """Mean +/- spread over the seeds of each experiment.

    Any comparison should use at least 3 seeds and report mean +/- spread, so
    `n` is included as a column. `std` is NaN when n=1, which is correct: one
    seed has no spread, and a 0.02 gap between two n=1 rows doesn't tell you
    anything.
    """
    score = f"summary.{metric}.min"
    if score not in runs.columns:
        raise KeyError(f"no {score} column; runs have {sorted(runs.columns)}")

    axes = [c for c in varying(runs) if c not in SEED_COLUMNS]
    frame = runs.assign(_group=group_key(runs))

    grouped = frame.groupby("_group", dropna=False)
    out = grouped[score].agg(["mean", "std", "min", "max", "count"])
    out = out.rename(columns={"count": "n"})

    # Include the axes next to the stats so a row says what it *is* rather than
    # just showing a hash. `first` because they're constant within a group.
    if axes:
        out = grouped[axes].first().join(out)

    return out.sort_values("mean").rename_axis("group").reset_index()


def curves(
    metrics: pd.DataFrame,
    metric: str = "loss",
    split: str = "val",
) -> pd.DataFrame:
    """Long metrics -> a step x run grid, ready to `.plot()`.

    `load_metrics` returns long format because that's easy to append to, but
    plots want it wide. Runs are lined up on `step`, and a run that didn't log
    at some step gets a NaN there rather than a wrong value.
    """
    rows = metrics[(metrics["metric"] == metric) & (metrics["split"] == split)]
    return rows.pivot_table(index="step", columns="run_id", values="value")
