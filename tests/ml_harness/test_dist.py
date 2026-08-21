"""ml_harness/dist.py, with no process group anywhere.

Everything here is either pure (reading an environment, dividing a batch) or
duck-typed (a reducer is a tensor function; DDP's sync flag is an attribute).
That is the point: if testing data parallelism started needing `torchrun`, the
seams would be in the wrong places. The one thing these tests *cannot* cover is
whether NCCL and gloo actually move the bytes — that is what a real 2-rank run
is for, and it lives in the projects that launch one.

The single-process assertions are not filler. `world_size == 1` is the path every
test suite, every smoke run and every laptop takes, and each of these functions
promises to be a no-op on it.
"""

from __future__ import annotations

import pytest
import torch

from ml_harness.dist import (
    DistContext,
    loss_reducer,
    reseed_batches,
    shard_batch,
)
from ml_harness.registry import load_run_config
from ml_harness.testing import ARCHES, DATAS


SINGLE = DistContext(rank=0, local_rank=0, world_size=1)


@pytest.fixture
def cfg(config_path):
    def build(*overrides):
        return load_run_config(config_path, ARCHES, DATAS, overrides)

    return build


# ---- reading the environment ----------------------------------------------

def test_no_rank_in_the_environment_is_the_single_process_case(monkeypatch):
    """The discriminator. Nothing but a launcher sets RANK, so its absence has
    to mean "one process" — not "assume rank 0 of an unknown world"."""
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)

    ctx = DistContext.from_env()

    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (0, 0, 1)
    assert ctx.is_master
    assert not ctx.is_distributed


def test_a_launcher_environment_is_read_straight_through(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    ctx = DistContext.from_env()

    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (3, 1, 8)
    assert not ctx.is_master
    assert ctx.is_distributed


def test_local_rank_falls_back_to_rank(monkeypatch):
    """A launcher that omits LOCAL_RANK is single-node by construction, where
    the two indices coincide."""
    monkeypatch.setenv("RANK", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert DistContext.from_env().local_rank == 2


def test_a_rank_outside_the_world_is_rejected(monkeypatch):
    """Caught while it is still an error. Left alone, an inconsistent
    environment surfaces at the first collective as a hang, which is the single
    least diagnosable failure in this whole area."""
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(RuntimeError, match="not a valid rank"):
        DistContext.from_env()


def test_master_is_rank_zero_regardless_of_world_size():
    assert DistContext(0, 0, 8).is_master
    assert not DistContext(1, 1, 8).is_master
    # And a lone process is master, which is what makes an `is_master` guard a
    # no-op locally rather than a second code path.
    assert SINGLE.is_master


# ---- sharding the batch ---------------------------------------------------

def test_one_process_gets_the_config_untouched(cfg):
    c = cfg()
    assert shard_batch(c, SINGLE) is c


def test_the_batch_is_divided_by_the_world_size(cfg):
    c = cfg("train.batch_size=8")
    assert shard_batch(c, DistContext(0, 0, 4)).train.batch_size == 2


def test_sharding_changes_nothing_but_the_batch_size(cfg):
    """`train.batch_size` is the only field that means something different per
    rank. If this ever fails, the sharded config has started to be a second
    description of the run rather than one field's local value."""
    c = cfg("train.batch_size=8", "train.max_iters=7", "train.grad_clip=1.0")
    local = shard_batch(c, DistContext(0, 0, 2))

    assert local.train.batch_size == 4
    assert local.train.max_iters == c.train.max_iters
    assert local.train.grad_clip == c.train.grad_clip
    assert local.model == c.model
    assert local.optim == c.optim
    assert local.runtime == c.runtime
    # The global config is untouched — it is what gets recorded.
    assert c.train.batch_size == 8


def test_every_rank_computes_the_same_share(cfg):
    """Equal shares are what makes the loss reduction a plain mean and the
    gradient average correct. Nothing here should depend on *which* rank asks."""
    c = cfg("train.batch_size=12")
    shares = {shard_batch(c, DistContext(r, r, 4)).train.batch_size for r in range(4)}
    assert shares == {3}


def test_a_batch_that_cannot_be_split_evenly_is_rejected(cfg):
    c = cfg("train.batch_size=5")
    with pytest.raises(RuntimeError, match="divisible by the world size"):
        shard_batch(c, DistContext(0, 0, 2))


def test_a_share_that_cannot_be_micro_batched_is_rejected(cfg):
    """12 splits four ways into 3, which a micro batch of 2 cannot divide. The
    global config is perfectly valid — this is only wrong once the world size is
    known, which is exactly why TrainConfig cannot make the check."""
    c = cfg("train.batch_size=12", "train.micro_batch_size=2")
    with pytest.raises(RuntimeError, match="not divisible by"):
        shard_batch(c, DistContext(0, 0, 4))


def test_no_micro_batch_size_always_divides(cfg):
    """`None` means "use the whole batch", which after sharding is the whole
    *share* — so there is nothing left to divide and no way to fail."""
    c = cfg("train.batch_size=6", "train.micro_batch_size=null")
    local = shard_batch(c, DistContext(0, 0, 3))
    assert local.train.batch_size == 2
    assert local.train.micro_batch_size is None


# ---- making the ranks differ ----------------------------------------------

def test_one_process_is_not_reseeded(cfg):
    """A single-process run must draw exactly what it drew before this existed,
    so this has to leave the RNG alone rather than re-seed it to `seed + 0`."""
    c = cfg()
    torch.manual_seed(1234)
    before = torch.rand(4)

    torch.manual_seed(1234)
    reseed_batches(c, SINGLE)
    assert torch.equal(before, torch.rand(4))


def test_ranks_are_reseeded_apart(cfg):
    """The whole value of data parallelism. If two ranks draw the same batch,
    they compute the same gradient and average it with itself."""
    c = cfg()

    draws = []
    for rank in range(3):
        reseed_batches(c, DistContext(rank, rank, 3))
        draws.append(torch.rand(8))

    assert not torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[1], draws[2])
    assert not torch.equal(draws[0], draws[2])


def test_reseeding_is_reproducible_per_rank(cfg):
    """Different from its peers, but still a function of `runtime.seed` — a
    rank that drew unrepeatable batches would make the run unreproducible."""
    c = cfg()

    reseed_batches(c, DistContext(2, 2, 4))
    first = torch.rand(8)
    reseed_batches(c, DistContext(2, 2, 4))

    assert torch.equal(first, torch.rand(8))


# ---- reducing the loss ----------------------------------------------------

def test_one_process_gets_no_reducer():
    """None rather than an identity function, so the loop skips a call per step
    and a single-process run keeps logging the number it always logged."""
    assert loss_reducer(SINGLE) is None


def test_a_distributed_context_gets_a_reducer():
    assert callable(loss_reducer(DistContext(0, 0, 2)))
