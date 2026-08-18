"""Content-addressed cache for an encoded corpus.

Encoding is a pure function of (corpus bytes, tokenizer), and both of those
already have a sha256 by the time a run starts — so the cache key is just those
two hashes and there is no invalidation logic to get wrong. A different corpus or
a different tokenizer is a different filename; nothing is ever overwritten with
something it isn't.

Torch-free, like `store.py`, and for the same reason: the tokens are integers,
and turning them into a tensor is the caller's business.

**The one hole**, and why the key carries two version components: the hashes
describe the *inputs* to encoding, not the code that does it. Fix a bug in
`apply_merges` and every cached file is silently stale, because the corpus and
the tokenizer recipe are unchanged.

Since `tokenizers` became its own distribution that code lives in another repo
entirely, so the hole is covered from both sides: `_ENCODER_VERSION` tracks
their changes automatically, and `CACHE_VERSION` is bumped by hand for ours.
Forgetting the second is still the failure mode to watch for.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np

from ml_learning.harness.io import _write_json

# Bump when *this* repo's encode path changes meaning. See the module docstring.
CACHE_VERSION = 2


def _encoder_version() -> str:
    """The version of the package that actually does the encoding.

    Read once at import: it can't change while the process runs, and a
    `PackageNotFoundError` per cache lookup would be a silly price for a
    constant. "unknown" rather than None so the key stays a readable filename
    in a source checkout with nothing installed — a *stable* wrong answer,
    which is the same bargain `CACHE_VERSION` makes.

    Weak in one place worth naming: two commits of `tokenizers` under the same
    declared version are indistinguishable here. The lockfile pins the commit,
    this only sees the version string. That's a narrower hole than the one it
    closes, not the absence of one.
    """
    try:
        return version("tokenizers")
    except PackageNotFoundError:
        return "unknown"


_ENCODER_VERSION = _encoder_version()

# int32 rather than int64: the largest vocabulary in the project is cl100k's
# 100256, so half the file for no loss. Torch wants int64 for embedding indices,
# which is the caller's cast.
DTYPE = np.int32

# Enough hex to make a collision irrelevant on a local disk, short enough that
# `ls` is readable. The full hashes go in the sidecar.
_KEY_CHARS = 16


def cache_path(cache_dir: Path | str, data_sha256: str, tokenizer_sha256: str) -> Path:
    # Both versions read as globals at call time, not closed over, so a test can
    # monkeypatch either and see the key move.
    #
    # No sanitising: PEP 440 restricts a version to `[0-9a-z.!+-]`, all of which
    # are legal in a filename on every platform this runs on.
    return Path(cache_dir) / (
        f"v{CACHE_VERSION}-{_ENCODER_VERSION}"
        f"_{data_sha256[:_KEY_CHARS]}"
        f"_{tokenizer_sha256[:_KEY_CHARS]}.npy"
    )


def load_tokens(path: Path) -> np.ndarray | None:
    """The cached ids, or None on a miss.

    Any failure is a miss rather than an exception: a cache that can crash a run
    is worse than no cache, and the file a killed process left half-written is
    exactly the case this has to survive. The caller re-encodes and overwrites.
    """
    try:
        return np.load(path)
    except (OSError, ValueError, EOFError):
        return None


def save_tokens(path: Path, tokens: Sequence[int], meta: dict) -> None:
    """Write the ids, plus a sidecar saying what they are.

    Written through a temp file and `os.replace` (the `_safe_write` pattern, for
    binary) so a run killed mid-write leaves either the old file or the new one,
    never half of either.

    The sidecar earns its keep the first time someone looks at a directory of
    hash-named `.npy` files and needs to know which corpus and which tokenizer
    they came from. Nothing reads it back — `load_tokens` trusts the key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    # Through an open file rather than a path: np.save silently appends `.npy`
    # to a filename that lacks it, and would write somewhere other than `tmp`.
    with open(tmp, "wb") as handle:
        np.save(handle, np.asarray(tokens, dtype=DTYPE), allow_pickle=False)
    os.replace(tmp, path)

    _write_json(
        path.with_suffix(".json"),
        {**meta, "n_tokens": len(tokens), "cached_at": datetime.now(timezone.utc).isoformat()},
    )
