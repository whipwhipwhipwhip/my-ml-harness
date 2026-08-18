"""Reading a directory of runs back as a comparison.

`store.load_runs` answers "what ran"; this answers "what did it mean". The split
is deliberate: `store.py` owns paths and the record, and knows nothing about
comparing two of them.

Everything here takes and returns DataFrames, so a notebook is the intended
caller — but the logic lives in a module rather than in cells because
`sweep.py`'s summary needs exactly the same grouping, and a notebook is not
importable. Torch-free, like the rest of the record side.
"""

from __future__ import annotations

import pandas as pd

from ml_harness.provenance import GROUP_EXCLUDES

# A run frame is ~70 columns wide, most of which are identical across every run
# in it. These are the ones worth seeing even when they don't vary.
ALWAYS = ("run_id",)

# Config columns that describe the *replicate* rather than the experiment. They
# vary within a group by construction, so listing them as ablation axes is noise.
# Derived from GROUP_EXCLUDES so the two definitions can't drift.
SEED_COLUMNS = tuple("config." + ".".join(path) for path in GROUP_EXCLUDES)


def varying(runs: pd.DataFrame, prefix: str = "config.") -> list[str]:
    """The config columns that actually differ across these runs.

    The point of a diffable config: an ablation *is* a config diff, so the
    columns worth putting in a table are exactly the ones that aren't constant.
    A 72-column frame becomes the three columns the experiment was about.

    All-NaN columns are dropped (a field a run predates), and so are columns
    that are constant once NaNs are ignored — a field only some runs record,
    with one value among those that do, is still not an axis.
    """
    cols = []

    for col in runs.columns:
        if not col.startswith(prefix):
            continue

        values = runs[col].dropna()
        if values.empty:
            continue

        # astype(str): a config value can be a list (`special_tokens`), which is
        # unhashable, and nunique() would raise rather than report 1.
        if values.astype(str).nunique() > 1:
            cols.append(col)

    return cols


def run_table(
    runs: pd.DataFrame,
    metric: str = "val.loss",
    sort: bool = True,
) -> pd.DataFrame:
    """A leaderboard: what varied, and what it scored.

    `metric` is a `(split, metric)` pair as `summarize` names it, so
    `"val.loss"` here reads `summary.val.loss.min` and `.final` out of the
    frame. Sorted by the min, because the best checkpoint is the one keyed on it.
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

    Prefers `provenance.group_sha256` — the config hashed with the seed and the
    note removed, written by the run itself. Falls back to the varying config
    columns minus the seed ones, which is the same definition computed from the
    frame instead of from the config, for runs written before the hash existed.

    The fallback is weaker on purpose: it can only see fields that vary *within
    this frame*, so two runs identical in every column present here group
    together even if their configs differed in a field nobody recorded. Good
    enough to read old runs; not good enough to key a sweep on.
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

    The house rule made mechanical: any comparison claim runs >= 3 seeds and
    reports mean +/- spread, so `n` is a column and not an afterthought. `std`
    is NaN at n=1, which is the honest answer — a single seed has no spread, and
    a 0.02 gap between two n=1 rows is not a result.
    """
    score = f"summary.{metric}.min"
    if score not in runs.columns:
        raise KeyError(f"no {score} column; runs have {sorted(runs.columns)}")

    axes = [c for c in varying(runs) if c not in SEED_COLUMNS]
    frame = runs.assign(_group=group_key(runs))

    grouped = frame.groupby("_group", dropna=False)
    out = grouped[score].agg(["mean", "std", "min", "max", "count"])
    out = out.rename(columns={"count": "n"})

    # The axes come back alongside the aggregate so a row says what it *is*
    # rather than naming a hash. `first` because within a group they're constant
    # by construction — that's what the group means.
    if axes:
        out = grouped[axes].first().join(out)

    return out.sort_values("mean").rename_axis("group").reset_index()


def curves(
    metrics: pd.DataFrame,
    metric: str = "loss",
    split: str = "val",
) -> pd.DataFrame:
    """Long metrics -> a step x run grid, ready to `.plot()`.

    `load_metrics` returns tidy long format because that's what appends
    cheaply; every plot wants it wide. This is that pivot, and the place to fix
    it if two runs ever log at different intervals (they line up on `step`, and
    a run that didn't log there gets a NaN rather than a wrong value).
    """
    rows = metrics[(metrics["metric"] == metric) & (metrics["split"] == split)]
    return rows.pivot_table(index="step", columns="run_id", values="value")
