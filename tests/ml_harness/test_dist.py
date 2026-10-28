"""ml_harness/dist.py, with no process group anywhere.

Everything here is either pure (reading the environment, dividing a batch) or
duck-typed (a reducer is a tensor function; DDP's sync flag is an attribute).
That's deliberate: if testing data parallelism needed `torchrun`, the code
would be split up in the wrong places. What these tests *can't* check is whether
NCCL and gloo actually move the data. That needs a real 2-rank run, which
belongs in a project that launches one.

The single-process tests matter too. `world_size == 1` is what every test, smoke
run and laptop uses, and each of these functions should do nothing there.
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
    """Only a launcher sets RANK, so if it's missing that means "one process",
    not "rank 0 of an unknown number"."""
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
    """Without LOCAL_RANK we assume a single machine, where the two are the
    same."""
    monkeypatch.setenv("RANK", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert DistContext.from_env().local_rank == 2


def test_a_rank_outside_the_world_is_rejected(monkeypatch):
    """Caught here as an error. Otherwise a bad environment shows up at the
    first collective as a hang, which is very hard to debug."""
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(RuntimeError, match="not a valid rank"):
        DistContext.from_env()


def test_master_is_rank_zero_regardless_of_world_size():
    assert DistContext(0, 0, 8).is_master
    assert not DistContext(1, 1, 8).is_master
    # A single process is also master, so an `is_master` check changes nothing
    # locally.
    assert SINGLE.is_master


# ---- sharding the batch ---------------------------------------------------

def test_one_process_gets_the_config_untouched(cfg):
    c = cfg()
    assert shard_batch(c, SINGLE) is c


def test_the_batch_is_divided_by_the_world_size(cfg):
    c = cfg("train.batch_size=8")
    assert shard_batch(c, DistContext(0, 0, 4)).train.batch_size == 2


def test_sharding_changes_nothing_but_the_batch_size(cfg):
    """`train.batch_size` is the only field that differs per rank. If this
    fails, sharding has started changing more of the config than it should."""
    c = cfg("train.batch_size=8", "train.max_iters=7", "train.grad_clip=1.0")
    local = shard_batch(c, DistContext(0, 0, 2))

    assert local.train.batch_size == 4
    assert local.train.max_iters == c.train.max_iters
    assert local.train.grad_clip == c.train.grad_clip
    assert local.model == c.model
    assert local.optim == c.optim
    assert local.runtime == c.runtime
    # The global config is unchanged, since that's what gets recorded.
    assert c.train.batch_size == 8


def test_every_rank_computes_the_same_share(cfg):
    """Equal shares are what make the loss reduction a plain mean and the
    gradient average correct. The result shouldn't depend on *which* rank asks."""
    c = cfg("train.batch_size=12")
    shares = {shard_batch(c, DistContext(r, r, 4)).train.batch_size for r in range(4)}
    assert shares == {3}


def test_a_batch_that_cannot_be_split_evenly_is_rejected(cfg):
    c = cfg("train.batch_size=5")
    with pytest.raises(RuntimeError, match="divisible by the world size"):
        shard_batch(c, DistContext(0, 0, 2))


def test_a_share_that_cannot_be_micro_batched_is_rejected(cfg):
    """12 split four ways is 3, which a micro batch of 2 can't divide. The
    config itself is valid; it's only wrong once the world size is known, which
    is why TrainConfig can't check it."""
    c = cfg("train.batch_size=12", "train.micro_batch_size=2")
    with pytest.raises(RuntimeError, match="not divisible by"):
        shard_batch(c, DistContext(0, 0, 4))


def test_no_micro_batch_size_always_divides(cfg):
    """`None` means "use the whole batch", which after sharding is the whole
    *share*, so there's nothing to divide and nothing to fail."""
    c = cfg("train.batch_size=6", "train.micro_batch_size=null")
    local = shard_batch(c, DistContext(0, 0, 3))
    assert local.train.batch_size == 2
    assert local.train.micro_batch_size is None


# ---- making the ranks differ ----------------------------------------------

def test_one_process_is_not_reseeded(cfg):
    """A single-process run must draw the same batches as before this existed,
    so this has to leave the RNG alone rather than re-seed it to `seed + 0`."""
    c = cfg()
    torch.manual_seed(1234)
    before = torch.rand(4)

    torch.manual_seed(1234)
    reseed_batches(c, SINGLE)
    assert torch.equal(before, torch.rand(4))


def test_ranks_are_reseeded_apart(cfg):
    """The point of data parallelism. If two ranks draw the same batch, they
    compute the same gradient and average it with itself."""
    c = cfg()

    draws = []
    for rank in range(3):
        reseed_batches(c, DistContext(rank, rank, 3))
        draws.append(torch.rand(8))

    assert not torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[1], draws[2])
    assert not torch.equal(draws[0], draws[2])


def test_reseeding_is_reproducible_per_rank(cfg):
    """Different from the other ranks, but still determined by `runtime.seed`;
    otherwise the run wouldn't be reproducible."""
    c = cfg()

    reseed_batches(c, DistContext(2, 2, 4))
    first = torch.rand(8)
    reseed_batches(c, DistContext(2, 2, 4))

    assert torch.equal(first, torch.rand(8))


# ---- reducing the loss ----------------------------------------------------

def test_one_process_gets_no_reducer():
    """None rather than an identity function, so a single-process run logs
    exactly what it always did."""
    assert loss_reducer(SINGLE) is None


def test_a_distributed_context_gets_a_reducer():
    assert callable(loss_reducer(DistContext(0, 0, 2)))
