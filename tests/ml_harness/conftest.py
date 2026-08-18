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
    """A complete run config, deliberately not `configs/GPT.json`.

    Hermetic: editing the real config can't break these tests, and these tests
    can't discourage editing it. `test_shipped_config_parses` covers that file
    separately, which is the only thing about it a test should care about.

    The absolute data path matters — `train.run` does `REPO_ROOT / cfg.data.path`,
    and `/` with an absolute right-hand side discards the left, so this points
    where it says it does.

    Only `arch` and the `model` block vary between the two fixtures below;
    everything else is the same run, which is the point — a test that isn't
    about the architecture shouldn't be able to tell which one it got.
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
    """The default: a run against `harness/testing.py`'s reference arch.

    Named without qualification because this is what a harness test should reach
    for. Nothing in `harness/` knows what a GPT is, so nothing testing `harness/`
    should need one — and every test on this fixture is one that would still run
    if the harness moved to a repo where `nanogpt_conor` doesn't exist.
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
def gpt_config_path(tmp_path, corpus_path):
    """The same run, against nanogpt's real arch.

    For the tests that drive a project's `run()` end to end rather than the
    harness's own machinery — they're asserting that the *entry point* wires up,
    which needs a real one. Reach for `config_path` unless that's the question.
    """
    cfg = _config(
        corpus_path,
        "gpt",
        {"n_embd": 16, "n_head": 2, "n_layer": 2, "dropout": 0.0, "block_size": 8},
    )
    path = tmp_path / "gpt_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def runs_root(tmp_path):
    return tmp_path / "runs"
