"""The token cache: does it hit, does it miss safely, and can it ever lie.

The last question is the important one. A cache that's merely slow is a
performance bug; a cache that returns the wrong tokens silently trains a model on
the wrong data and nothing in the run record would say so. So most of this is
about the key — different corpus, different tokenizer, different version — and
about failure modes degrading to a miss rather than to wrong answers.

Cheap tier: tmp_path and short strings, no GPT, no run directory.
"""

from __future__ import annotations

import json

import pytest
import torch

from ml_harness import cache as cache_mod
from ml_harness.testing import load_data
from ml_harness.cache import cache_path, load_tokens, save_tokens
from tokenizers import ByteTokenizer, CharTokenizer, fingerprint

TEXT = "the theatre of the theory " * 20


class CountingTokenizer(CharTokenizer):
    """A CharTokenizer that says how many times it was asked to encode."""

    @classmethod
    def over(cls, text: str) -> "CountingTokenizer":
        base = CharTokenizer.from_text(text)
        tok = cls(base.stoi, base.itos)
        object.__setattr__(tok, "calls", 0)
        return tok

    def encode(self, text):
        object.__setattr__(self, "calls", self.calls + 1)
        return super().encode(text)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


def entries(cache_dir):
    return sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".npy")


# ---- the key --------------------------------------------------------------

def test_the_key_is_both_hashes():
    a = cache_path("/c", "d" * 64, "t" * 64)
    assert a.name.startswith("v")
    assert "d" * 16 in a.name and "t" * 16 in a.name


@pytest.mark.parametrize(
    "data_sha, tok_sha",
    [("d" * 64, "OTHER" + "t" * 59), ("OTHER" + "d" * 59, "t" * 64)],
    ids=["different-tokenizer", "different-corpus"],
)
def test_changing_either_half_changes_the_file(data_sha, tok_sha):
    assert cache_path("/c", data_sha, tok_sha) != cache_path("/c", "d" * 64, "t" * 64)


def test_the_version_is_in_the_filename(monkeypatch):
    """The one thing the content hashes can't cover: a change to the encoder
    itself, which leaves corpus and tokenizer recipe identical."""
    before = cache_path("/c", "d" * 64, "t" * 64)
    monkeypatch.setattr(cache_mod, "CACHE_VERSION", cache_mod.CACHE_VERSION + 1)
    assert cache_path("/c", "d" * 64, "t" * 64) != before


def test_the_encoder_version_is_in_the_filename(monkeypatch):
    """The same hole from the other side. `CACHE_VERSION` is bumped by hand for
    changes here; `tokenizers` is a separate distribution, so a merge-loop fix
    lands there with nothing in this repo to notice it."""
    before = cache_path("/c", "d" * 64, "t" * 64)
    monkeypatch.setattr(cache_mod, "_ENCODER_VERSION", "9.9.9")
    assert cache_path("/c", "d" * 64, "t" * 64) != before


def test_the_encoder_version_is_a_real_one():
    """Guards the degenerate pass: if the lookup were broken, every key would
    carry the same "unknown" and the test above would still pass."""
    assert cache_mod._ENCODER_VERSION != "unknown"
    assert cache_mod._ENCODER_VERSION in cache_path("/c", "d" * 64, "t" * 64).name


# ---- round trip -----------------------------------------------------------

def test_saved_tokens_come_back(cache_dir):
    path = cache_path(cache_dir, "a" * 64, "b" * 64)
    save_tokens(path, [1, 2, 3, 70000], {})
    assert list(load_tokens(path)) == [1, 2, 3, 70000]


def test_a_missing_file_is_a_miss(cache_dir):
    assert load_tokens(cache_path(cache_dir, "a" * 64, "b" * 64)) is None


def test_a_truncated_file_is_a_miss_not_a_crash(cache_dir):
    """What a run killed mid-write leaves behind. A cache that can crash a run is
    worse than no cache."""
    path = cache_path(cache_dir, "a" * 64, "b" * 64)
    save_tokens(path, list(range(1000)), {})
    path.write_bytes(path.read_bytes()[:40])

    assert load_tokens(path) is None


def test_the_sidecar_says_what_the_file_is(cache_dir):
    tok = ByteTokenizer()
    path = cache_path(cache_dir, "a" * 64, fingerprint(tok))
    save_tokens(path, [1, 2, 3], {"tokenizer": tok.to_dict()})

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["tokenizer"] == {"kind": "byte"}
    assert meta["n_tokens"] == 3
    assert meta["cached_at"]


def test_no_temp_file_is_left_behind(cache_dir):
    path = cache_path(cache_dir, "a" * 64, "b" * 64)
    save_tokens(path, [1, 2, 3], {})
    assert [p.suffix for p in sorted(cache_dir.iterdir())] == [".json", ".npy"]


# ---- through load_data ----------------------------------------------------

def test_the_second_load_does_not_re_encode(corpus, cache_dir):
    tok = CountingTokenizer.over(TEXT)

    first = load_data(corpus, 0.9, lambda text: tok, cache_dir)
    assert tok.calls == 1

    second = load_data(corpus, 0.9, lambda text: tok, cache_dir)
    assert tok.calls == 1, "second load re-encoded"
    assert torch.equal(first.train, second.train)
    assert torch.equal(first.val, second.val)


def test_a_cache_hit_produces_the_same_tokens_as_encoding(corpus, cache_dir):
    """The only thing a cache is allowed to change is how long it takes."""
    uncached = load_data(corpus, 0.9, CharTokenizer.from_text, None)
    load_data(corpus, 0.9, CharTokenizer.from_text, cache_dir)
    cached = load_data(corpus, 0.9, CharTokenizer.from_text, cache_dir)

    assert torch.equal(uncached.train, cached.train)
    assert uncached.train.dtype == cached.train.dtype == torch.long


def test_two_tokenizers_over_one_corpus_do_not_collide(corpus, cache_dir):
    """The failure this cache has to be incapable of: byte ids served to a
    character-level run."""
    char = load_data(corpus, 0.9, CharTokenizer.from_text, cache_dir)
    byte = load_data(corpus, 0.9, lambda text: ByteTokenizer(), cache_dir)

    assert len(entries(cache_dir)) == 2
    assert not torch.equal(char.train[:20], byte.train[:20])
    assert byte.tokenizer.vocab_size == 256


def test_two_corpora_do_not_collide(tmp_path, cache_dir):
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text(TEXT, encoding="utf-8")
    two.write_text(TEXT.upper(), encoding="utf-8")

    load_data(one, 0.9, lambda text: ByteTokenizer(), cache_dir)
    load_data(two, 0.9, lambda text: ByteTokenizer(), cache_dir)

    assert len(entries(cache_dir)) == 2


def test_no_cache_dir_means_no_files_anywhere(corpus, cache_dir):
    tok = CountingTokenizer.over(TEXT)

    load_data(corpus, 0.9, lambda text: tok, None)
    load_data(corpus, 0.9, lambda text: tok, None)

    assert tok.calls == 2
    assert not cache_dir.exists()


def test_the_cache_directory_is_created_on_demand(corpus, cache_dir):
    assert not cache_dir.exists()
    load_data(corpus, 0.9, CharTokenizer.from_text, cache_dir)
    assert cache_dir.is_dir()
