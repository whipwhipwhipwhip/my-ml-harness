"""Data parallelism: which process this is, and what that changes.

**The environment is only read in one place**, `DistContext.from_env`, and
everything else works from the values it returns. `torchrun` sets
`RANK`/`LOCAL_RANK`/`WORLD_SIZE` and nothing else does, so if they're missing
this is a single process, and it runs exactly the same code it did before this
module existed. Every function here returns its argument unchanged when
`world_size == 1`.

It names no model and no task, so it belongs in the harness for the same reason
`device.py` does: it's about *how* a run is carried out, not what's trained.

Everything here happens before or after the training loop: picking a device,
splitting the batch, wrapping the module, building a loss reducer. What happens
*inside* a step belongs to `loop.py`, which gets it through duck-typed
arguments (`reduce_loss`, and a `require_backward_grad_sync` attribute it sets
if the model has one), so `loop.py` never has to know about ranks.

Nothing here starts a process group behind your back. `distributed` is a context
manager because a process group that isn't cleaned up can keep `torchrun` running
after the job finishes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterator

import torch
from torch.distributed import (
    ReduceOp,
    all_reduce,
    destroy_process_group,
    init_process_group,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_harness.config import RunConfig
from ml_harness.device import resolve_device
from ml_harness.runner import BuiltRun


@dataclass(frozen=True)
class DistContext:
    """Which process this is within the job. Frozen, since it's read, not chosen.

    `local_rank` is the index *on this machine* and `rank` is the index across
    all machines. They only differ on a multi-node job, and mixing them up is
    how every process on the second node ends up trying to use GPU 0.
    """

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_master(self) -> bool:
        """Whether this rank does the I/O: the run directory, logs, checkpoints.

        Rank 0 by convention. Also True for a single-process run, so an
        `if ctx.is_master` check changes nothing when running locally.
        """
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether there are other processes to talk to.

        Not the same as `not is_master`: a one-process job is master *and* not
        distributed, which is the case almost every time (the tests, smoke runs,
        anything on MPS).
        """
        return self.world_size > 1

    @classmethod
    def from_env(cls) -> "DistContext":
        """Read the launcher's environment, or return the single-process case.

        Checks `RANK`, because `torchrun` always sets it and a normal launch
        doesn't. `LOCAL_RANK` falls back to `RANK`, which is right on a single
        machine, the only case where that fallback would be used.
        """
        rank = int(os.environ.get("RANK", -1))
        if rank < 0:
            return cls(rank=0, local_rank=0, world_size=1)

        ctx = cls(
            rank=rank,
            local_rank=int(os.environ.get("LOCAL_RANK", rank)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )

        # Better to catch this here than at the first collective, where a bad
        # environment shows up as a hang rather than an error.
        if not 0 <= ctx.rank < ctx.world_size:
            raise RuntimeError(
                f"RANK={ctx.rank} is not a valid rank in a world of "
                f"{ctx.world_size}; the launcher's environment is inconsistent"
            )

        return ctx


@contextmanager
def distributed(ctx: DistContext, preferred_device: str) -> Iterator[torch.device]:
    """Resolve this rank's device, join the process group, and leave it cleanly.

    Yields the device, because the device decides which backend can be used.

    Torch requires `set_device` **before** `init_process_group`, so NCCL sets
    up on the right GPU. Without it every rank also creates a context on GPU 0,
    which shows up as unexplained memory use on that card rather than an error.

    `destroy_process_group` is in a `finally`, partly for the usual reasons and
    partly because a rank that exits without leaving the group can keep
    `torchrun` running after the job is done.
    """
    device = resolve_device(preferred_device)

    if not ctx.is_distributed:
        yield device
        return

    # MPS has no distributed backend at all: gloo is CPU-only and NCCL is
    # CUDA-only. Worth a clear error, because `runtime.device: "auto"` picks mps
    # on a Mac, which is exactly where you'd first try a multi-rank run.
    if device.type == "mps":
        raise RuntimeError(
            "no distributed backend supports MPS tensors — gloo is CPU-only "
            "and NCCL is CUDA-only. For a local multi-rank run, set "
            "runtime.device to 'cpu' and gloo will be used."
        )

    if device.type == "cuda":
        # The only place `local_rank` is used. `resolve_device` returns a plain
        # `cuda`, meaning "the current device", which is fine for one process
        # but would put all eight processes on card 0.
        device = torch.device(f"cuda:{ctx.local_rank}")
        torch.cuda.set_device(device)

    backend = "nccl" if device.type == "cuda" else "gloo"
    init_process_group(backend=backend)
    try:
        yield device
    finally:
        destroy_process_group()


def shard_batch(cfg: RunConfig, ctx: DistContext) -> RunConfig:
    """`cfg` with `train.batch_size` cut to this rank's share of it.

    **`train.batch_size` is global.** It means "sequences per optimiser step"
    whether the step runs on one GPU or eight, so one config file describes the
    same experiment on either: same tokens per step, same `group_sha256`, and
    losses you can compare directly. The world size is a fact about the machine
    and goes in provenance; it shouldn't change what a run *is*.

    For that to work, something has to divide the batch between ranks, and this
    is it. The result is only for `loop.train`: `record_run` and
    `draw_eval_batches` keep the global config, because config.json records the
    experiment and the eval set has to match every other run's.

    The checks are here, and run before a run directory exists, because
    `TrainConfig.__post_init__` can't know how many processes will share the
    batch. They mention the world size explicitly because `replace` re-runs
    TrainConfig's own checks, whose error would show a per-rank batch size that
    isn't in the config.
    """
    if not ctx.is_distributed:
        return cfg

    batch = cfg.train.batch_size
    micro = cfg.train.micro_batch_size
    world = ctx.world_size

    if batch % world:
        raise RuntimeError(
            f"train.batch_size ({batch}) must be divisible by the world size "
            f"({world}); it is the global batch, and every rank has to take an "
            f"equal share of it for the accumulated gradient to be that batch's"
        )

    local = batch // world

    # Only matters with accumulation on: with `micro_batch_size = None` the
    # loop uses the whole (per-rank) batch as its micro batch, which always
    # divides.
    if micro is not None and local % micro:
        raise RuntimeError(
            f"train.batch_size ({batch}) split across {world} ranks leaves "
            f"{local} sequences per rank, which is not divisible by "
            f"train.micro_batch_size ({micro}). Pick a batch size divisible by "
            f"micro_batch_size * world_size ({micro * world})"
        )

    return replace(cfg, train=replace(cfg.train, batch_size=local))


def reseed_batches(cfg: RunConfig, ctx: DistContext) -> None:
    """Make each rank draw *different* training batches. Call after `build_run`.

    Without this, data parallelism does nothing useful. `build_run` seeds every
    rank with `runtime.seed`, and `loop.train` draws training batches from the
    global RNG, so all N ranks would draw the *same* batch, compute the same
    gradient, and average it with itself: N times the compute for one rank's
    worth of training. And it would look completely normal (same loss curve,
    same samples, just slower).

    **Call it after `build_run`, never before.** The ranks need identical
    weights, which they get by using the same seed. Changing the seed first
    would give N different models, which DDP then silently replaces with rank
    0's. Changing it afterwards only affects what's drawn from then on, i.e.
    the batches.

    It's a fairly blunt approach: it changes the global RNG, so dropout masks
    differ per rank too. That's fine (you'd want independent masks anyway). The
    cleaner version would pass a training generator into `loop.train` like
    `eval_seed` does, but that's a bigger change than it's worth right now.
    """
    if not ctx.is_distributed:
        return
    torch.manual_seed(cfg.runtime.seed + ctx.rank)


def parallelise(built: BuiltRun, ctx: DistContext, device: torch.device) -> BuiltRun:
    """Wrap the runnable module in DDP, leaving `built.model` alone.

    `BuiltRun` already separates the two for `torch.compile`, and DDP needs the
    same split: `model` is what gets checkpointed and sampled, `runnable` is
    what gets trained. DDP adds a `module.` prefix to state-dict keys (on top of
    compile's `_orig_mod.`), so a checkpoint saved from the wrapper won't load
    into anything. Only replacing `runnable` means `checkpoint_payload`'s rule
    (pass `model`, never a wrapper) still holds without it knowing about DDP.

    Compile goes inside and DDP outside, which is the documented order.

    `device_ids` is CUDA-only. On CPU with gloo there's no device to name, and
    passing one raises an error. CPU/gloo is how this gets tested without a GPU,
    so the branch is needed.

    Creating the DDP wrapper copies rank 0's parameters to every other rank.
    Here that changes nothing, since every rank used the same seed and built the
    same weights. That makes it a useful check: if it ever does change
    something, something upstream isn't deterministic.
    """
    if not ctx.is_distributed:
        return built

    device_ids = [device.index] if device.type == "cuda" else None
    return built._replace(runnable=DDP(built.runnable, device_ids=device_ids))


def loss_reducer(ctx: DistContext) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """`reduce_loss` for `loop.train`, or None when there's only one process.

    None rather than an identity function, because that's how the loop already
    handles "nothing to do" for `amp` and `lr_schedule`.

    **SUM then divide, not `ReduceOp.AVG`.** Only NCCL supports AVG, so it works
    on a GPU machine but fails under gloo, i.e. in exactly the local CPU run
    you'd use to test this before going near a GPU. Sum-and-divide works on
    both, and `shard_batch` makes sure every rank has an equal share.

    Clones before reducing so the caller's tensor isn't changed in place. It's
    a scalar, so the copy is free.
    """
    if not ctx.is_distributed:
        return None

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().clone()
        all_reduce(out, op=ReduceOp.SUM)
        return out / ctx.world_size

    return reduce
