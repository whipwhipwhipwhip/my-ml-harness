"""GPT-2's weight decay split: dim >= 2 decays, everything else doesn't.

Cheap tests: a model, an optimiser, and its `param_groups`. No filesystem
beyond the generated config and no training, since the split is fixed as soon
as the optimiser is built.

These use `testing.TinyModel` rather than a stub because it has both kinds of
tensor (weight matrices and embeddings, plus LayerNorm gains and biases), so the
split has something to get wrong. A stub with only an embedding would pass even
if the split were broken.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ml_harness.registry import (
    build_model,
    build_optimiser,
    decay_param_groups,
    load_run_config,
)
from ml_harness.testing import ARCHES, DATAS

VOCAB = 65


@pytest.fixture
def built(config_path):
    """(cfg, model) for the reference arch. No data needed; the split only uses shapes."""

    def build(*overrides):
        cfg = load_run_config(config_path, ARCHES, DATAS, overrides)
        return cfg, build_model(cfg, VOCAB, ARCHES)

    return build


# ---- the partition --------------------------------------------------------

def test_every_parameter_lands_in_exactly_one_group(built):
    """The bug to catch is a parameter in neither group: it would get
    gradients but never be updated, and nothing would tell you."""
    _, model = built()
    groups = decay_param_groups(model, 0.1)

    placed = [p for g in groups for p in g["params"]]
    expected = list(model.parameters())
    assert len(placed) == len(expected)
    assert {id(p) for p in placed} == {id(p) for p in expected}


def test_the_decayed_group_is_the_matrices(built):
    _, model = built()
    decay, _ = decay_param_groups(model, 0.1)
    assert decay["params"]
    assert all(p.dim() >= 2 for p in decay["params"])


def test_the_undecayed_group_is_the_biases_and_gains(built):
    _, model = built()
    _, no_decay = decay_param_groups(model, 0.1)
    assert no_decay["params"]
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_biases_and_gains_get_exactly_zero(built):
    """Exactly 0.0, not just small: the point is that these aren't
    regularised at all."""
    _, model = built()
    _, no_decay = decay_param_groups(model, 0.1)
    assert no_decay["weight_decay"] == 0.0


def test_frozen_parameters_are_left_out(built):
    """A frozen tensor has no gradient, but AdamW's decay is applied to the
    parameter directly, so a frozen weight in a decayed group would still
    shrink every step."""
    _, model = built()
    frozen = next(p for p in model.parameters() if p.dim() >= 2)
    frozen.requires_grad_(False)

    placed = [p for g in decay_param_groups(model, 0.1) for p in g["params"]]

    assert all(p is not frozen for p in placed)
    assert len(placed) == len(list(model.parameters())) - 1


def test_a_tied_weight_is_decayed_once(built):
    """Weight tying is why this uses `model.parameters()`, which skips
    duplicate tensors, rather than collecting `named_parameters()` by name. If a
    tied embedding were counted twice it would decay at double the rate, which
    would look like a badly chosen hyperparameter."""

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 8)
            self.head = nn.Linear(8, VOCAB, bias=False)
            self.head.weight = self.emb.weight

    decay, _ = decay_param_groups(Tied(), 0.1)
    assert len(decay["params"]) == 1


# ---- the config reaches it ------------------------------------------------

def test_the_configured_decay_reaches_the_optimiser(built):
    cfg, model = built("optim.weight_decay=0.2")
    optimiser = build_optimiser(cfg, model)
    assert [g["weight_decay"] for g in optimiser.param_groups] == [0.2, 0.0]


def test_the_default_is_the_gpt2_value(built):
    """0.1, not torch's default of 0.01 (which is what runs from before this
    field used, on every parameter)."""
    cfg, model = built()
    assert cfg.optim.weight_decay == 0.1
    assert build_optimiser(cfg, model).param_groups[0]["weight_decay"] == 0.1


def test_zero_decay_still_splits_the_groups(built):
    """Turning decay off is just a value: the groups are always built, so
    `weight_decay=0` and the default differ in one number and nothing else."""
    cfg, model = built("optim.weight_decay=0")
    optimiser = build_optimiser(cfg, model)
    assert [g["weight_decay"] for g in optimiser.param_groups] == [0.0, 0.0]
    assert len(optimiser.param_groups) == 2


def test_negative_decay_is_rejected_at_load(built):
    """Negative decay grows the weights every step, and the run diverges
    slowly enough to look like a bad learning rate."""
    with pytest.raises(ValueError, match="weight_decay"):
        built("optim.weight_decay=-0.1")


def test_a_bare_parameter_iterable_still_gets_the_configured_decay(built):
    """`_build_adamw` passes `weight_decay` as the constructor default as well
    as through the groups, so someone building an optimiser by hand (a project,
    a test) gets the config's value rather than torch's."""
    from ml_harness.registry import OPTIMS

    cfg, model = built("optim.weight_decay=0.3")
    optimiser = OPTIMS[cfg.optim_arch].build(cfg.optim, model.parameters())
    assert optimiser.param_groups[0]["weight_decay"] == 0.3


# ---- what the split means for the rest of the loop ------------------------

def test_the_schedule_reaches_both_groups(built):
    """The loop writes the scheduled lr into every param group, which is what
    makes having two groups safe. Checked here against the real split, rather
    than the hand-added group in test_loop.py."""
    from ml_harness.loop import train
    from ml_harness.testing import FakeLogger, SpyEval

    cfg, model = built("train.max_iters=1")
    optimiser = build_optimiser(cfg, model)

    class OneBatch:
        def get_batch(self, split, batch_size, block_size, device, n_iters=1,
                      generator=None):
            x = torch.randint(VOCAB, (batch_size, block_size))
            return [(x, x)] * n_iters

    train(
        model, optimiser, OneBatch(), cfg, torch.device("cpu"), {},
        FakeLogger(), SpyEval(), lr_schedule=lambda step: 0.5,
    )
    assert [g["lr"] for g in optimiser.param_groups] == [0.5, 0.5]
