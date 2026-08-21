"""`tokenizer_arch` as a discriminator: the tag -> the config class -> the object.

Cheapest tier — a config dict and a short string, no run directory and no GPT.
The question here is only "does the tag select the right thing, and does a wrong
tag fail loudly"; whether the tokenizers themselves are correct is
`tests/tokenizers/`.
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
from tokenizers import (
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
    """If this fails someone added a tokenizer without a test — which is the
    point of asserting on the table rather than listing the names twice."""
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
    """The whole reason it's a separate entry: same corpus, stated alphabet."""
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
# That they encode correctly is `tests/tokenizers/`; the question here is only
# whether the config block reaches the tokenizer, since these change vocab_size
# and therefore the shape of the checkpoint.

def test_regex_bpe_specials_come_from_the_config(config_path):
    cfg = load(config_path, "regex_bpe", ['tokenizer.special_tokens=["<|eot|>"]'])
    tok = build_tokenizer(cfg, CORPUS)

    assert cfg.tokenizer.special_tokens == ["<|eot|>"]
    assert tok.special_map == {"<|eot|>": 256}


def test_tiktoken_specials_come_from_the_config(config_path):
    """The 50256-vs-50257 decision, made in a config file rather than in code."""
    cfg = load(
        config_path, "tiktoken", ['tokenizer.special_tokens=["<|endoftext|>"]']
    )
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.vocab_size == 50257


def test_declaring_no_specials_is_the_default(config_path):
    """Both BPE configs default to none, so an existing config file means what it
    always meant — opting in is what grows the vocabulary."""
    assert load(config_path, "regex_bpe").tokenizer.special_tokens is None
    assert load(config_path, "tiktoken").tokenizer.special_tokens is None


def test_a_special_the_encoding_does_not_define_is_rejected(config_path):
    """Caught at build time, not silently added: tiktoken's ids aren't ours to
    invent. Worth a config-level test because the config is where the typo goes."""
    cfg = load(config_path, "tiktoken", ['tokenizer.special_tokens=["<|eot|>"]'])

    with pytest.raises(ValueError, match="no special token"):
        build_tokenizer(cfg, CORPUS)


# ---- vocabulary padding ---------------------------------------------------
#
# On the base config, so it reaches every arch. That the padded ids behave (are
# unreachable, decode to nothing) is `tests/tokenizers/`; what's asked here is
# whether the number in the config file reaches the tokenizer, since it sizes
# the embedding table and therefore the checkpoint.

@pytest.mark.parametrize("arch", list(CASES))
def test_padding_reaches_every_tokenizer(config_path, arch):
    unpadded = build_tokenizer(load(config_path, arch), CORPUS).vocab_size
    padded = build_tokenizer(
        load(config_path, arch, [f"tokenizer.pad_vocab_to={unpadded + 7}"]), CORPUS
    )

    assert padded.vocab_size == unpadded + 7


def test_padding_is_off_by_default(config_path):
    """An existing config file means what it always meant."""
    assert load(config_path, "char").tokenizer.pad_vocab_to is None


def test_gpt2_with_endoftext_pads_to_50304(config_path):
    """The case padding exists for: GPT-2's 50256 ranks, plus the one special
    that takes it to 50257, reported as 64 x 786 so the matmuls either side of
    the model get a shape they like."""
    cfg = load(
        config_path,
        "tiktoken",
        ['tokenizer.special_tokens=["<|endoftext|>"]', "tokenizer.pad_vocab_to=50304"],
    )
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.unpadded_vocab_size == 50257
    assert tok.vocab_size == 50304


def test_padding_survives_the_recipe(config_path):
    """`to_dict` is what `runs/<id>/tokenizer.json` holds, so a padding it
    dropped would be a checkpoint whose embedding table no longer fits."""
    cfg = load(config_path, "tiktoken", ["tokenizer.pad_vocab_to=50304"])
    tok = build_tokenizer(cfg, CORPUS)

    assert tok.to_dict()["pad_vocab_to"] == 50304


def test_padding_below_the_real_vocabulary_is_rejected(config_path):
    """Not silently ignored: a config asking for fewer ids than exist is a
    statement about the embedding table that isn't true.

    Raised when `vocab_size` is read rather than when the tokenizer is built,
    which is the tokenizers package's choice, not ours. Still early enough: an
    entry point reads it to size the model, which is before `record_run` has put
    a directory on disk."""
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
    """Same payoff as everywhere else: `alphabt=abc` is a load error, not a
    silently ignored hyperparameter."""
    with pytest.raises(DaciteError):
        load_run_config(
            config_path, ARCHES, DATAS, ["tokenizer_arch=char_fixed", "tokenizer.alphabt=abc"]
        )


def test_a_missing_required_field_is_caught(config_path):
    """`char_fixed` without an alphabet has nothing to fall back on."""
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["tokenizer_arch=char_fixed"])


def test_a_field_from_the_wrong_tokenizer_is_caught(config_path):
    """`char` takes no options, so an alphabet on it is a mistake worth hearing
    about — the run it describes isn't the run it would have done."""
    with pytest.raises(DaciteError):
        load_run_config(config_path, ARCHES, DATAS, ["tokenizer.alphabet=abc"])
