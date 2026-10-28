"""Config identity: `config_sha256` and `group_sha256` as pure functions.

Cheap tests: plain dicts, no filesystem, no torch. They check which runs count
as the same experiment, and which as the same run, which is what grouping runs
in analysis relies on.
"""

from __future__ import annotations

import copy

from ml_harness.provenance import config_sha256, group_sha256


def cfg(**overrides) -> dict:
    """A RunConfig-shaped dict. Only the fields the hashes reason about matter."""
    base = {
        "arch": "gpt",
        "model": {"n_embd": 16, "n_head": 2},
        "optim_arch": "adamw",
        "optim": {"lr": 1e-3},
        "train": {"max_iters": 4},
        "runtime": {"seed": 0, "eval_seed": 1, "device": "cpu"},
        "note": "pytest",
    }
    base.update(overrides)
    return base


# ---- config_sha256: this exact run ----------------------------------------

def test_identical_configs_hash_identically():
    assert config_sha256(cfg()) == config_sha256(cfg())


def test_key_order_does_not_change_the_hash():
    """It has to survive a round trip through JSON and a reordered dict.
    `asdict` order is stable today, but nothing guarantees it."""
    reordered = dict(reversed(list(cfg().items())))
    assert config_sha256(reordered) == config_sha256(cfg())


def test_a_changed_hyperparameter_changes_the_hash():
    assert config_sha256(cfg(optim={"lr": 3e-4})) != config_sha256(cfg())


def test_a_changed_seed_changes_the_config_hash():
    """This is the hash that tells two seeds of one experiment apart."""
    other = cfg(runtime={"seed": 1, "eval_seed": 1, "device": "cpu"})
    assert config_sha256(other) != config_sha256(cfg())


# ---- group_sha256: this experiment ----------------------------------------

def test_seeds_of_one_config_share_a_group():
    """The main use: "mean +/- spread over N seeds" is a groupby on this."""
    other = cfg(runtime={"seed": 99, "eval_seed": 1, "device": "cpu"})
    assert group_sha256(other) == group_sha256(cfg())


def test_the_note_does_not_split_a_group():
    assert group_sha256(cfg(note="a different label")) == group_sha256(cfg())


def test_a_changed_hyperparameter_still_splits_the_group():
    assert group_sha256(cfg(optim={"lr": 3e-4})) != group_sha256(cfg())


def test_eval_seed_splits_the_group():
    """*Not* excluded, on purpose: a different eval seed means a different
    validation set, so the losses can't be averaged, which is what a group is
    for."""
    other = cfg(runtime={"seed": 0, "eval_seed": 2, "device": "cpu"})
    assert group_sha256(other) != group_sha256(cfg())


def test_micro_batch_size_does_not_split_a_group():
    """A hardware setting: the update uses `train.batch_size` either way, and
    splitting it to fit a smaller card is the same experiment."""
    other = cfg(train={"max_iters": 4, "micro_batch_size": 2})
    assert group_sha256(other) == group_sha256(cfg(train={"max_iters": 4}))


def test_micro_batch_size_still_changes_the_config_hash():
    """But they're different runs: they saw different batches, summed in a
    different order, just like two seeds of one config."""
    other = cfg(train={"max_iters": 4, "micro_batch_size": 2})
    assert config_sha256(other) != config_sha256(cfg(train={"max_iters": 4}))


def test_batch_size_still_splits_the_group():
    """batch_size *is* experimental. If two batch sizes ever grouped together,
    the exclusion above would be in the wrong place."""
    other = cfg(train={"max_iters": 4, "batch_size": 64})
    assert group_sha256(other) != group_sha256(cfg(train={"max_iters": 4, "batch_size": 32}))


def test_hashing_does_not_mutate_the_config():
    """`group_sha256` drops fields to compute its hash, and must drop them from
    a copy: the dict it's given is the one about to be written to disk."""
    original = cfg()
    snapshot = copy.deepcopy(original)

    group_sha256(original)

    assert original == snapshot


def test_a_missing_excluded_field_is_not_an_error():
    """A config with no `note`, or a different runtime block, can still be
    hashed: missing excluded keys are just skipped."""
    sparse = {"model": {"n_embd": 16}}
    assert group_sha256(sparse)
