"""`data_arch` as a discriminator: the tag -> the config class the block parses into.

Cheapest tier, same as `test_tokenizer_config.py` — a config dict and a short
string, no run directory and no GPT. One thing differs from the other three
discriminators and it's the thing worth pinning: `DATAS` maps to a config class,
not a spec pair, and it is *passed in* rather than living in `harness/`. The two
tables here come from two different projects, which is what proves the harness
isn't quietly importing either of them.

Whether the loaders themselves produce sensible corpora isn't this file's
question.
"""

from __future__ import annotations

import json

import pytest
from dacite import DaciteError

from ml_learning.harness.config import FileDataConfig
from ml_learning.harness.registry import load_run_config
from ml_learning.gpt_addition.arches import ARCHES as ADDITION_ARCHES
from ml_learning.gpt_addition.data import AdditionDataConfig
from ml_learning.gpt_addition.datas import DATAS as ADDITION_DATAS
# nanogpt's tables rather than `harness/testing.py`'s: the claim this file
# makes is that two *separate projects* can each name their own data config,
# and a reference table shipped inside the harness wouldn't demonstrate it.
from ml_learning.nanogpt_conor.arches import ARCHES
from ml_learning.nanogpt_conor.datas import DATAS


def rewrite(gpt_config_path, tmp_path, **blocks):
    """The fixture's config with whole top-level blocks replaced.

    `--set` can't do this: an override edits a key inside an existing block, and
    swapping data_arch means swapping the block's *shape* at the same time.
    """
    raw = json.loads(gpt_config_path.read_text())
    raw.update(blocks)
    path = tmp_path / "rewritten.json"
    path.write_text(json.dumps(raw))
    return path


def test_the_tag_selects_the_config_class(gpt_config_path):
    cfg = load_run_config(gpt_config_path, ARCHES, DATAS)
    assert cfg.data_arch == "file"
    assert isinstance(cfg.data, FileDataConfig)


def test_a_different_projects_table_parses_a_different_block(gpt_config_path, tmp_path):
    """The payoff of passing DATAS in: the same loader, two dataset shapes,
    and `harness/` naming neither."""
    path = rewrite(
        gpt_config_path,
        tmp_path,
        data_arch="addition",
        data={"n_problems": 50, "max_operand": 99, "split_frac": 0.8},
    )
    cfg = load_run_config(path, ADDITION_ARCHES, ADDITION_DATAS)

    assert isinstance(cfg.data, AdditionDataConfig)
    assert cfg.data.max_operand == 99
    # Defaulted, not stated — the frozen dataclass is still the source of truth
    # for what an unspecified field means.
    assert cfg.data.reverse_answer is False


def test_a_table_that_lacks_the_tag_rejects_it(gpt_config_path, tmp_path):
    """nanogpt's table has no `addition` entry, so its entry point can't be
    handed an addition config by accident."""
    path = rewrite(gpt_config_path, tmp_path, data_arch="addition", data={})
    with pytest.raises(ValueError, match="unknown data_arch 'addition'"):
        load_run_config(path, ARCHES, DATAS)


def test_a_missing_data_arch_names_the_options(gpt_config_path, tmp_path):
    raw = json.loads(gpt_config_path.read_text())
    del raw["data_arch"]
    path = tmp_path / "no_data_arch.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="has no 'data_arch' key"):
        load_run_config(path, ARCHES, DATAS)


def test_a_typo_in_the_data_block_is_caught_by_strict(gpt_config_path):
    """Same payoff as everywhere else: `split_frc` is a load error rather than a
    silent fallback to the default split."""
    with pytest.raises(DaciteError):
        load_run_config(gpt_config_path, ARCHES, DATAS, ["data.split_frc=0.5"])


def test_the_wrong_block_for_the_tag_is_caught(gpt_config_path, tmp_path):
    """`data_arch: addition` with a file's block. Nothing else would catch this —
    it's exactly the mistake the tag exists to make legible."""
    path = rewrite(
        gpt_config_path,
        tmp_path,
        data_arch="addition",
        data={"path": "corpus.txt", "split_frac": 0.9, "cache": False},
    )
    with pytest.raises(DaciteError):
        load_run_config(path, ADDITION_ARCHES, ADDITION_DATAS)


def test_shipped_addition_config_parses():
    """The one thing these tests should assert about configs/addition.json."""
    cfg = load_run_config("configs/addition.json", ADDITION_ARCHES, ADDITION_DATAS)
    assert isinstance(cfg.data, AdditionDataConfig)
    # The sample prompt is encoded through the run's own tokenizer at every eval,
    # so a character outside the alphabet is a crash at the first eval rather
    # than a config error at load. Cheap to assert, expensive to discover.
    assert set(cfg.train.sample_prompt) <= set(cfg.tokenizer.alphabet)
