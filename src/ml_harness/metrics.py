from pathlib import Path
import pandas as pd
import json
import time
import warnings

from ml_learning.harness.io import _safe_loads

# Note - should be constructed before training loop,
# which should be wired in somehow

class MetricLogger:
    def __init__(self, path: Path):
        self._file = open(path, "a", buffering=1)  # line-buffered
        self._start = time.monotonic()

    def log(self, step, split, metric, value, **extra):
        elapsed_s = time.monotonic() - self._start
        record = {
            "step": step,
            "split": split,
            "metric": metric,
            "value": float(value),
            "elapsed_s": elapsed_s,
            **extra,
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self):
        self._file.close()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

def read_rows(path: Path) -> tuple[list[dict], int]:
    """Parse a metrics.jsonl. Returns (rows, n_bad).

    A killed run leaves a truncated final line; every row before it is still
    good, so bad lines are skipped rather than fatal.
    """
    rows, bad = [], 0
    if not path.exists():
        return rows, bad

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row, err = _safe_loads(line)
            if err:
                bad += 1
                continue
            rows.append(row)

    return rows, bad

def read_metrics(path: Path) -> pd.DataFrame:
    rows, bad = read_rows(path)
    if bad:
        warnings.warn(f"{path}: skipped {bad} unparseable line(s)")
    return pd.DataFrame(rows)

def summarize(path: Path) -> dict:

    metrics = {}
    summary = {}

    max_step = -1
    max_elapsed = 0.0
    row_count = 0
    rows, bad = read_rows(path)

    for row in rows:
        row_count += 1

        step = row["step"]
        split = row["split"]
        metric = row["metric"]
        value = row["value"]
        elapsed = row.get("elapsed_s", 0.0)

        max_step = max(max_step, step)
        max_elapsed = max(max_elapsed, elapsed)

        key = (split, metric)

        if key not in metrics:
            metrics[key] = {
                "final_step": step,
                "final_value": value,
                "min_value": value,
                "min_step": step,
                "max_value": value,
                "max_step": step,
            }
            continue

        m = metrics[key]

        # final = highest step seen
        if step >= m["final_step"]:
            m["final_step"] = step
            m["final_value"] = value

        if value < m["min_value"]:
            m["min_value"] = value
            m["min_step"] = step

        if value > m["max_value"]:
            m["max_value"] = value
            m["max_step"] = step

    summary["steps_completed"] = max_step + 1 if max_step >= 0 else 0
    summary["elapsed_s"] = max_elapsed
    summary["row_count"] = row_count
    summary["bad_rows"] = bad

    for (split, metric), m in metrics.items():
        prefix = f"{split}.{metric}"
        summary[f"{prefix}.final"] = m["final_value"]
        summary[f"{prefix}.min"] = m["min_value"]
        summary[f"{prefix}.min_step"] = m["min_step"]
        summary[f"{prefix}.max"] = m["max_value"]
        summary[f"{prefix}.max_step"] = m["max_step"]

    return summary