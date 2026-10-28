"""Content-addressed cache for an encoded corpus.

Encoding only depends on the corpus bytes and the tokenizer, and both already
have a sha256 by the time a run starts. So the cache key is just those two
hashes, and there's no invalidation logic to get wrong: a different corpus or
tokenizer gives a different filename.

Torch-free, like `store.py`: the tokens are just integers, and turning them into
a tensor is up to the caller.

**The one gap**, and why the key also has two version numbers: the hashes
describe the *inputs* to encoding, not the code that does it. Fix a bug in
`apply_merges` and every cached file is silently out of date, because the corpus
and the tokenizer recipe haven't changed.

The encoding code lives in the separate `nanotok` package, so
`_ENCODER_VERSION` picks up its changes automatically, and `CACHE_VERSION` is
bumped by hand for changes in this repo. Forgetting to bump it is still the
thing to watch out for.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np

from ml_harness.io import _write_json

# Bump when *this* repo's encode path changes meaning. See the module docstring.
CACHE_VERSION = 2


def _encoder_version() -> str:
    """The version of the package that actually does the encoding.

    Read once at import, since it can't change while the process is running.
    "unknown" rather than None if nothing is installed, so the key is still a
    readable filename and at least stays the same between runs.

    One weakness: two commits of `nanotok` with the same version number look
    the same here. The lockfile pins the commit, but this only sees the version
    string. It's a smaller gap than the one it closes, but still a gap.
    """
    try:
        return version("nanotok")
    except PackageNotFoundError:
        return "unknown"


_ENCODER_VERSION = _encoder_version()

# int32 rather than int64: the largest vocabulary used is cl100k's (~100k), so
# this halves the file size for free. Torch wants int64 for embedding indices,
# so the caller converts.
DTYPE = np.int32

# Enough hex characters that a collision won't happen on a local disk, but short
# enough that `ls` is readable. The full hashes go in the sidecar file.
_KEY_CHARS = 16


def cache_path(cache_dir: Path | str, data_sha256: str, tokenizer_sha256: str) -> Path:
    # Both versions are read as globals at call time, so a test can
    # monkeypatch either one and see the key change.
    #
    # No sanitising needed: PEP 440 restricts versions to `[0-9a-z.!+-]`, which
    # are all fine in filenames.
    return Path(cache_dir) / (
        f"v{CACHE_VERSION}-{_ENCODER_VERSION}"
        f"_{data_sha256[:_KEY_CHARS]}"
        f"_{tokenizer_sha256[:_KEY_CHARS]}.npy"
    )


def load_tokens(path: Path) -> np.ndarray | None:
    """The cached ids, or None on a miss.

    Any failure counts as a miss rather than an exception, since a cache that
    can crash a run is worse than no cache. In particular, this has to cope with
    a half-written file left by a killed process. The caller just re-encodes
    and overwrites it.
    """
    try:
        return np.load(path)
    except (OSError, ValueError, EOFError):
        return None


def save_tokens(path: Path, tokens: Sequence[int], meta: dict) -> None:
    """Write the ids, plus a sidecar saying what they are.

    Written via a temp file and `os.replace` (like `_safe_write`, but binary),
    so a run killed mid-write leaves either the old file or the new one, never
    half of one.

    The sidecar is for when you look at a directory of hash-named `.npy` files
    and want to know which corpus and tokenizer they came from. Nothing reads it
    back; `load_tokens` relies on the key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    # Pass an open file rather than a path: np.save silently adds `.npy` to a
    # filename without it, so it would write somewhere other than `tmp`.
    with open(tmp, "wb") as handle:
        np.save(handle, np.asarray(tokens, dtype=DTYPE), allow_pickle=False)
    os.replace(tmp, path)

    _write_json(
        path.with_suffix(".json"),
        {**meta, "n_tokens": len(tokens), "cached_at": datetime.now(timezone.utc).isoformat()},
    )
