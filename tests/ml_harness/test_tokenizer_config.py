"""`tokenizer_arch` as a discriminator: the tag -> the config class -> the object.

Cheap tests: a config dict and a short string, no run directory and no GPT.
These only check that the tag picks the right thing and that a wrong tag fails
clearly. Whether the tokenizers themselves are correct is tested in the
`nanotok` package.
"""

from __future__ import annotations

import json

import pytest
from dacite import DaciteError

from ml_harness.testing import ARCHES, DATAS
from ml_harness.config import (
    ByteConfig,
    CharConfig,
    CharFixedConfig,
    RegexBPEConfig,
    TiktokenConfig,
)
from ml_harness.registry import TOKENIZERS, build_tokenizer, load_run_config
from nanotok import (
    ByteTokenizer,
    CharTokenizer,
    RegexTokenizer,
    TiktokenBPETokenizer,
)

CORPUS = "the theatre of the theory " * 40

# One valid case per entry in TOKENIZERS: the overrides that select it, the
# config class its block should parse into, and the class it should build.
CASES = {
    "char": ([], CharConfig, CharTokenizer),
    "char_fixed": (["tokenizer.alphabet=abc"], CharFixedConfig, CharTokenizer),
    "byte": ([], ByteConfig, ByteTokenizer),
    "regex_bpe": (["tokenizer.vocab_size=260"], RegexBPEConfig, RegexTokenizer),
    "tiktoken": (["tokenizer.encoding=gpt2"], TiktokenConfig, TiktokenBPETokenizer),
}


def load(config_path, arch, extra=()):
    overrides = [f"tokenizer_arch={arch}", *CASES[arch][0], *extra]
    return load_run_config(config_path, ARCHES, DATAS, overrides)


def test_every_registered_tokenizer_has_a_case():
    """If this fails, someone added a tokenizer without a test case. That's why
    this checks against the table rather than listing the names again."""
    assert set(CASES) == set(TOKENIZERS)


@pytest.mark.parametrize("arch", list(CASES))
def test_the_tag_selects_the_config_class(config_path, arch):
    cfg = load(config_path, arch)
    assert cfg.tokenizer_arch == arch
    assert type(cfg.tokenizer) is CASES[arch][1]


@pytest.mark.parametrize("arch", list(CASES))
def test_the_tag_selects_the_tokenizer(config_path, arch):
    tok = build_tokenizer(load(config_path, arch), CORPUS)
    assert type(tok) is CASES[arch][2]
    assert tok.vocab_size > 0


# ---- the corpus argument --------------------------------------------------

def test_char_reads_its_alphabet_off_the_corpus(config_path):
    tok = build_tokenizer(load(config_path, "char"), CORPUS)
    assert tok.alphabet == "".join(sorted(set(CORPUS)))


def test_char_fixed_ignores_the_corpus(config_path):
    """The reason it's a separate entry: the alphabet is given, not read from the corpus."""
    tok = build_tokenizer(load(config_path, "char_fixed"), CORPUS)
    assert tok.alphabet == "abc"


def test_regex_bpe_trains_on_the_corpus(config_path):
    tok = build_tokenizer(load(config_path, "regex_bpe"), CORPUS)
    assert tok.vocab_size == 260
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8"))


def test_tiktoken_arrives_already_trained(config_path):
    tok = build_tokenizer(load(config_path, "tiktoken"), CORPUS)
    assert tok.vocab_size == 50256


# ---- special tokens -------------------------------------------------------
#
# Whether they encode correctly is tested in the `nanotok` package. Here we
# only check the config reaches the tokenizer, since special tokens change
# vocab_size and so the shape of the checkpoint.

def test_regex_bpe_specials_come_from_the_config(config_path):
    cfg = load(config_path, "regex_bpe", ['tokenizer.special_tokens=["<|eot|>"]'])
    tok = build_tokenizer(cfg, CORPUS)

    assert cfg.tokenizer.special_tokens == ["<|eot|>"]
    assert tok.special_map == {"<|eot|>": 256}


def test_tiktoken_specials_come_from_the_config(config_path):
    """Whether GPT-2 has 50256 or 50257 ids is set in the config, not in code."""
    cfg = load(
        config_path, "tiktoken", ['tokenizer.special_tokens=["<|endoftext|>"]']
    )
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.vocab_size == 50257


def test_declaring_no_specials_is_the_default(config_path):
    """Both BPE configs default to none, so existing config files don't change.
    The vocabulary only grows if you ask for special tokens."""
    assert load(config_path, "regex_bpe").tokenizer.special_tokens is None
    assert load(config_path, "tiktoken").tokenizer.special_tokens is None


def test_a_special_the_encoding_does_not_define_is_rejected(config_path):
    """Caught at build time rather than silently added, since we can't make up
    tiktoken ids. Tested at the config level because that's where typos happen."""
    cfg = load(config_path, "tiktoken", ['tokenizer.special_tokens=["<|eot|>"]'])

    with pytest.raises(ValueError, match="no special token"):
        build_tokenizer(cfg, CORPUS)


# ---- vocabulary padding ---------------------------------------------------
#
# On the base config, so every tokenizer gets it. How padded ids behave is
# tested in the `nanotok` package; here we only check the number in the
# config reaches the tokenizer, since it sets the size of the embedding table
# and so the checkpoint.

@pytest.mark.parametrize("arch", list(CASES))
def test_padding_reaches_every_tokenizer(config_path, arch):
    unpadded = build_tokenizer(load(config_path, arch), CORPUS).vocab_size
    padded = build_tokenizer(
        load(config_path, arch, [f"tokenizer.pad_vocab_to={unpadded + 7}"]), CORPUS
    )

    assert padded.vocab_size == unpadded + 7


def test_padding_is_off_by_default(config_path):
    """Existing config files don't change."""
    assert load(config_path, "char").tokenizer.pad_vocab_to is None


def test_gpt2_with_endoftext_pads_to_50304(config_path):
    """The main use case: GPT-2's 50256 tokens, plus one special token to make
    50257, padded to 64 x 786 so the matmuls at either end of the model get a
    GPU-friendly shape."""
    cfg = load(
        config_path,
        "tiktoken",
        ['tokenizer.special_tokens=["<|endoftext|>"]', "tokenizer.pad_vocab_to=50304"],
    )
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.unpadded_vocab_size == 50257
    assert tok.vocab_size == 50304


def test_padding_survives_the_recipe(config_path):
    """`to_dict` is what gets saved to `runs/<id>/tokenizer.json`, so if it
    dropped the padding, the rebuilt tokenizer wouldn't match the checkpoint's
    embedding table."""
    cfg = load(config_path, "tiktoken", ["tokenizer.pad_vocab_to=50304"])
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.to_dict()["pad_vocab_to"] == 50304


def test_padding_below_the_real_vocabulary_is_rejected(config_path):
    """Not silently ignored: a config asking for fewer ids than exist is
    asking for an embedding table that's too small.

    The nanotok package raises when `vocab_size` is read rather than when
    the tokenizer is built. That's still early enough, since an entry point
    reads it to size the model before `record_run` creates a directory."""
    tok = build_tokenizer(
        load(config_path, "tiktoken", ["tokenizer.pad_vocab_to=100"]), CORPUS
    )

    with pytest.raises(ValueError, match="can only grow"):
        tok.vocab_size


# ---- what a wrong config does ---------------------------------------------

def test_unknown_tokenizer_arch_is_rejected(config_path):
    with pytest.raises(ValueError, match="unknown tokenizer_arch"):
        load_run_config(config_path, ARCHES, DATAS, ["tokenizer_arch=nope"])


def test_a_missing_tokenizer_arch_names_the_options(config_path, tmp_path):
    raw = json.loads(config_path.read_text())
    del raw["tokenizer_arch"]
    path = tmp_path / "no_tokenizer.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="has no 'tokenizer_arch' key"):
        load_run_config(path, ARCHES, DATAS)


def test_a_typo_in_the_tokenizer_block_is_caught_by_strict(config_path):
    """As everywhere else, `alphabt=abc` is a load error, not a silently
    ignored setting."""
    with pytest.raises(DaciteError):
        load_run_config(
            config_path, ARCHES, DATAS, ["tokenizer_arch=char_fixed", "tokenizer.alphabt=abc"]
        )


def test_a_missing_required_field_is_caught(config_path):
    """`char_fixed` without an alphabet has nothing to fall back on."""
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["tokenizer_arch=char_fixed"])


def test_a_field_from_the_wrong_tokenizer_is_caught(config_path):
    """`char` takes no options, so giving it an alphabet is a mistake you'd
    want to know about: the run wouldn't use it."""
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["tokenizer.alphabet=abc"])
