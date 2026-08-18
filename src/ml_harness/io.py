"""Atomic writes and non-raising reads — the harness's whole filesystem vocabulary.

Copied from `ml_learning/utils/io.py` rather than imported from it, so that
`harness/` depends on nothing in this repo outside its own directory and the
`tokenizers` package. `utils/` stays where it is: the micrograd half and
`makemore_conor` still use it, and the ~30 lines here are cheaper than a shared
package that exists only to be shared.

Two ideas, both about what a killed process leaves behind:

- **Write through a temp file and `os.replace`.** A reader sees the old file or
  the new one, never half of either. `runs/` is written *during* training, so
  "killed mid-write" is the normal case rather than the exotic one.
- **Read without raising.** Every caller here is reading a record back for
  analysis, where one truncated `status.json` in a directory of 200 runs should
  cost you that row, not the frame. So these return `(data, error)` and let the
  caller decide, which is why they aren't just `json.load`.
"""

import json
import os


def _safe_write(path, txt):
    """Write `txt` to `path` atomically, via a sibling temp file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(txt)
    os.replace(tmp, path)


def _write_json(path, obj):
    _safe_write(path, json.dumps(obj, indent=2))


def _safe_load(path):
    """Returns (data, error). On failure, data is the error string."""
    try:
        return json.loads(path.read_text()), False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return str(e), True


def _safe_loads(text):
    """`_safe_load` for text already in hand — the JSONL line case."""
    error = False
    try:
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        data = str(e)
        error = True

    return data, error
