"""Fixtures for the harness tests.

Everything here is synthesised into `tmp_path`. Nothing reads `runs/`, and nothing
depends on `input.txt` — not because it's gitignored, but because a test that
trains on 1MB of Shakespeare isn't a test, it's a run.
"""

from __future__ import annotations

import json
import string

import pytest


# Long enough that a 0.9/0.1 split still leaves a val set bigger than block_size,
# and varied enough that the vocab isn't degenerate. The content is irrelevant —
# nothing asserts on what the model learns, only that the machinery turns over.
CORPUS = (string.ascii_lowercase + " \n") * 200


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return path


def _config(corpus_path, arch: str, model: dict) -> dict:
    """A complete run config, synthesised into tmp_path.

    Hermetic by construction: this package ships no config files, and a test that
    reached for one in a downstream project would be exactly the dependency the
    layering rule forbids.

    `data.path` is absolute, which is what lets an entry point read it without
    resolving anything — the harness resolves no path it wasn't given.
    """
    return {
        "arch": arch,
        "model": model,
        "optim_arch": "adamw",
        "optim": {"lr": 1e-3},
        # The corpus-derived character tokenizer, i.e. what the harness did
        # before tokenization became a config choice. Tests that care about a
        # different one override `tokenizer_arch` explicitly.
        "tokenizer_arch": "char",
        "tokenizer": {},
        "train": {
            "max_iters": 4,
            "batch_size": 4,
            "eval_interval": 2,
            "eval_iters": 2,
        },
        "data_arch": "file",
        # cache off by default: `train.run`'s default cache_dir is the real
        # `.cache/` in the repo, and a test must never write there. The tests
        # that do exercise the cache turn it on and point it at tmp_path.
        "data": {"path": str(corpus_path), "split_frac": 0.9, "cache": False},
        "runtime": {"seed": 0, "eval_seed": 1, "device": "cpu"},
        "note": "pytest",
    }


@pytest.fixture
def config_path(tmp_path, corpus_path):
    """A run against `ml_harness/testing.py`'s reference arch.

    The only config fixture, and named without qualification because it is what
    every test here reaches for. Nothing in `ml_harness/` knows what a GPT is, so
    nothing testing `ml_harness/` needs one — which is the property that let this
    package be lifted out of the repo it grew in, and the one to preserve.
    """
    cfg = _config(
        corpus_path,
        "tiny",
        {"n_embd": 16, "n_head": 2, "n_layer": 1, "dropout": 0.0, "block_size": 8},
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def runs_root(tmp_path):
    return tmp_path / "runs"
