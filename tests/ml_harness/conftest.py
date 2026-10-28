"""Fixtures for the harness tests.

Everything here is generated into `tmp_path`. Nothing reads `runs/` or a real
corpus: a test that trains on 1MB of Shakespeare isn't a test, it's a run.
"""

from __future__ import annotations

import json
import string

import pytest


# Long enough that a 0.9/0.1 split still leaves a val set bigger than
# block_size, and varied enough to give a reasonable vocab. The content doesn't
# matter, since nothing checks what the model learns, only that everything runs.
CORPUS = (string.ascii_lowercase + " \n") * 200


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return path


def _config(corpus_path, arch: str, model: dict) -> dict:
    """A complete run config, written into tmp_path.

    This package ships no config files, and a test shouldn't depend on one from
    a downstream project, so the config is generated here.

    `data.path` is absolute, so an entry point can read it without resolving
    anything (the harness never makes up a path it wasn't given).
    """
    return {
        "arch": arch,
        "model": model,
        "optim_arch": "adamw",
        "optim": {"lr": 1e-3},
        # The character tokenizer built from the corpus, i.e. what the harness
        # did before the tokenizer was configurable. Tests that want a
        # different one override `tokenizer_arch`.
        "tokenizer_arch": "char",
        "tokenizer": {},
        "train": {
            "max_iters": 4,
            "batch_size": 4,
            "eval_interval": 2,
            "eval_iters": 2,
        },
        "data_arch": "file",
        # Cache off by default, since a test must never write to a developer's
        # real `.cache/`. The tests that check the cache turn it on and point
        # it at tmp_path.
        "data": {"path": str(corpus_path), "split_frac": 0.9, "cache": False},
        "runtime": {"seed": 0, "eval_seed": 1, "device": "cpu"},
        "note": "pytest",
    }


@pytest.fixture
def config_path(tmp_path, corpus_path):
    """A run against `ml_harness/testing.py`'s reference arch.

    The only config fixture, used by nearly every test. Nothing in
    `ml_harness/` knows what a GPT is, so nothing testing it needs one. That's
    what made it possible to split this package out of the repo it started in,
    and it's worth keeping that way.
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
