"""Atomic writes and reads that don't raise. All the file I/O the harness does.

Copied from `utils/io.py` in the repo this was extracted from, rather than
imported, so that `ml_harness/` depends on nothing outside its own directory
(plus the `tokenizers` package). That's what made it possible to split this
package out, and ~30 duplicated lines are cheaper than a shared utils package.

Both functions are about what a killed process leaves behind:

- **Write via a temp file and `os.replace`.** A reader sees either the old file
  or the new one, never half of one. `runs/` is written *during* training, so
  being killed mid-write is a normal case, not an unusual one.
- **Read without raising.** These are used to read records back for analysis,
  where one truncated `status.json` in a directory of 200 runs should lose you
  that row, not the whole frame. So they return `(data, error)` and let the
  caller decide.
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
    """`_safe_load` for a string rather than a file (e.g. one JSONL line)."""
    error = False
    try:
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        data = str(e)
        error = True

    return data, error
