from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ml_harness.io import _write_json, _safe_load, _safe_write
from ml_harness.metrics import read_metrics

import pandas as pd


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """`{"model": {"n_layer": 4}}` -> `{"model.n_layer": 4}`.

    Lives here rather than in a shared utils package so `harness/` depends on
    nothing outside its own directory. Its one caller is `load_runs`, where it
    turns a nested config into the flat columns that make `df["config.optim.lr"]`
    a thing you can group by — the dotted names deliberately match what `--set`
    takes, so the column you sorted by is the flag you'd vary.
    """
    out = {}

    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value

    return out

# understand dataclass frozen
@dataclass(frozen=True)
class RunDir:
    """Canonical paths for a single training run."""

    root: Path
    run_id: str

    @property
    def path(self) -> Path:
        return self.root / self.run_id

    # ---- metadata ---------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def provenance_path(self) -> Path:
        return self.path / "provenance.json"

    @property
    def summary_path(self) -> Path:
        return self.path / "summary.json"
    
    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def tokenizer_path(self) -> Path:
        """The tokenizer's reconstruction recipe.

        Without it a checkpoint is an undecodable pile of integers: `*.txt` is
        gitignored, so for a corpus-derived vocabulary there is otherwise no
        record anywhere of what id 7 meant.
        """
        return self.path / "tokenizer.json"

    # ---- artifacts --------------------------------------------------------

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def samples_dir(self) -> Path:
        return self.path / "samples"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    def ckpt_path(self, tag: str = "last") -> Path:
        if tag not in {"best", "last"}:
            raise ValueError(f"unknown checkpoint tag: {tag}")
        return self.checkpoints_dir / f"{tag}.pt"

def new_run(
    root: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> RunDir:
    root = Path(root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:6]}"

    rd = RunDir(root=root, run_id=run_id)

    rd.path.mkdir(parents=True)
    rd.checkpoints_dir.mkdir()
    rd.samples_dir.mkdir()
    rd.logs_dir.mkdir()

    _write_json(rd.config_path, config)
    _write_json(rd.provenance_path, provenance)

    # if we have git_dirty, then write patch. Remove it from provenance then
    if provenance.get("git_dirty", False):
        err_string = "git_dirty, but patch failed to collect"
        _safe_write(rd.logs_dir / "uncommitted.patch", provenance.pop("patch", err_string))

    return rd

def write_summary(rundir: RunDir, summary: dict[str, Any]) -> None:
    _write_json(rundir.summary_path, summary)

def write_status(rd: RunDir, **fields) -> None:
    _write_json(rd.status_path, fields)

def load_runs(root: Path) -> pd.DataFrame:
    rows = []

    for run_dir in Path(root).iterdir():
        if not run_dir.is_dir():
            continue

        # `runs/smoke_tests/` is a directory *of* runs, not a run. config.json is
        # the marker: new_run writes it before anything else, so a directory
        # without one was never a run and would arrive as a row of NaNs.
        if not (run_dir / "config.json").exists():
            continue

        row = {"run_id": run_dir.name}

        for name in ("config", "summary", "provenance"):
            path = run_dir / f"{name}.json"
            if path.exists():
                data, err = _safe_load(path)
                if not err:
                    row.update(flatten_dict(data, prefix=name))
                else:
                    row[f"{name}_load_error"] = data

        rows.append(row)

    return pd.DataFrame(rows)

def load_metrics(
    root: Path,
    run_ids: list[str] | None = None,
) -> pd.DataFrame:
    root = Path(root)

    if run_ids is None:
        run_ids = [
            p.name
            for p in root.iterdir()
            if p.is_dir()
        ]

    dfs = []

    for run_id in run_ids:
        path = root / run_id / "metrics.jsonl"
        if not path.exists():
            continue

        df = read_metrics(path)
        df["run_id"] = run_id
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
