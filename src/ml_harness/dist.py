"""Data parallelism: who this process is, and what that changes.

**The environment is read in exactly one place** — `DistContext.from_env` — and
every other site branches on a *value*. That is the whole design. `torchrun` sets
`RANK`/`LOCAL_RANK`/`WORLD_SIZE` and nothing else does, so their absence is the
signal for "one process, the way it has always been", and a single-process run
takes the same code path it took before this module existed. Not an equivalent
path — the same one. Every function here returns its argument unchanged when
`world_size == 1`.

This module names no model and no task, so it sits inside the harness under the
same rule as `device.py`: it is about *how* a run is carried out, never about
what is being trained.

The division of labour with `loop.py`: everything here happens either side of the
training loop — resolving a device, sharding a batch, wrapping a module,
building a reducer. What happens *inside* the step is the loop's, and it takes
it as ordinary duck-typed collaborators (`reduce_loss`, and a
`require_backward_grad_sync` attribute it sets if the model has one), so
`loop.py` still has no idea that ranks exist.

Nothing here starts a process group implicitly. `distributed` is a context
manager because a group that isn't left can hold `torchrun` open past the end of
a job.
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
    """This process's identity within the job. Frozen — it's observed, not chosen.

    `local_rank` is the index *within this machine* and `rank` is the index
    across all of them; they differ only on a multi-node job, and conflating
    them is how every rank on node 1 ends up addressing GPU 0 of node 0.
    """

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_master(self) -> bool:
        """Whether this rank does the I/O — the run directory, logs, checkpoints.

        Rank 0 by convention. True for a single-process run too, which is what
        makes every `if ctx.is_master` guard a no-op locally rather than a
        second code path to keep in step.
        """
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether there is anyone to talk to.

        Distinct from `is_master`: a one-process job is master *and* not
        distributed, and that is the combination almost every invocation takes —
        the test suites, every smoke run, anything on MPS.
        """
        return self.world_size > 1

    @classmethod
    def from_env(cls) -> "DistContext":
        """Read the launcher's environment, or return the single-process case.

        `RANK` is the discriminator because `torchrun` always sets it and
        ordinary launches never do. `LOCAL_RANK` falls back to `RANK`, which is
        correct on the single-node job that fallback would ever apply to.
        """
        rank = int(os.environ.get("RANK", -1))
        if rank < 0:
            return cls(rank=0, local_rank=0, world_size=1)

        ctx = cls(
            rank=rank,
            local_rank=int(os.environ.get("LOCAL_RANK", rank)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )

        # Worth catching here rather than at the first collective, where an
        # inconsistent environment surfaces as a hang instead of an error.
        if not 0 <= ctx.rank < ctx.world_size:
            raise RuntimeError(
                f"RANK={ctx.rank} is not a valid rank in a world of "
                f"{ctx.world_size}; the launcher's environment is inconsistent"
            )

        return ctx


@contextmanager
def distributed(ctx: DistContext, preferred_device: str) -> Iterator[torch.device]:
    """Resolve this rank's device, join the process group, and leave it cleanly.

    Yields the device, because the two decisions are one decision: which
    accelerator this rank owns is what determines the backend that can reach it.

    Ordering is torch's, not ours — `set_device` **before** `init_process_group`,
    so NCCL builds its communicator against the right GPU. Skip it and every rank
    initialises a context on GPU 0 as well as its own, which shows up as
    unexplained memory on card zero rather than as an error.

    `destroy_process_group` in a `finally` for the ordinary reason, plus one
    specific to `torchrun`: a rank that exits without leaving the group can hold
    the launcher open past the end of the job.
    """
    device = resolve_device(preferred_device)

    if not ctx.is_distributed:
        yield device
        return

    # MPS has no distributed backend at all — gloo is CPU-only, NCCL is
    # CUDA-only, and nothing can carry an mps tensor between processes. Worth a
    # named error because `runtime.device: "auto"` resolves to mps on an Apple
    # machine, which is exactly where someone tries their first multi-rank run.
    if device.type == "mps":
        raise RuntimeError(
            "no distributed backend supports MPS tensors — gloo is CPU-only "
            "and NCCL is CUDA-only. For a local multi-rank run, set "
            "runtime.device to 'cpu' and gloo will be used."
        )

    if device.type == "cuda":
        # The one place `local_rank` is spent. `resolve_device` returns a bare
        # `cuda`, meaning "the current device" — correct for one process and
        # catastrophic for eight, since they would all land on card 0.
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

    **`train.batch_size` is global.** It means "sequences per optimiser step" and
    goes on meaning exactly that whether the step is computed on one GPU or
    eight, so one config file describes the same experiment on either — same
    tokens per step, same `group_sha256`, losses that belong on one axis. The
    world size is a fact about the machine, recorded in provenance; it is not an
    experimental axis and must not silently change what a run *is*.

    That only works if somebody divides, and this is the somebody. The result is
    for `loop.train` alone: `record_run` and `draw_eval_batches` keep the global
    config, because config.json records the experiment and the eval set is the
    yardstick every other run was measured against.

    The checks live here, and run before a run directory exists, because
    `TrainConfig.__post_init__` cannot make them — a frozen dataclass has no way
    to know how many processes are about to share it. They name the world size
    explicitly for the same reason: `replace` re-runs TrainConfig's own
    validation, which would report a batch size the config never mentioned.
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

    # Only meaningful when accumulation is on: `micro_batch_size = None` makes
    # the loop use the whole (now per-rank) batch as its micro batch, which
    # divides by construction.
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

    Without this, data parallelism buys nothing whatsoever. `build_run` seeds
    every rank with `runtime.seed`, and `loop.train` draws its training batches
    off the global RNG, so all N ranks would draw the *identical* batch, compute
    the identical gradient, and all-reduce it with itself: N times the
    electricity for one rank's worth of training. It would look entirely healthy
    — same loss curve, same samples, just slower per step.

    **After `build_run`, never before.** The ranks have to build bit-identical
    weights, which they do by seeding identically; offsetting first would give N
    different models that DDP then silently overwrites with rank 0's at
    construction. Offsetting afterwards puts the divergence only on what is drawn
    from here on, which is the batches.

    A blunt instrument, knowingly: it moves the global RNG, so dropout masks
    differ per rank too. That is harmless — independent masks across ranks is
    what you want anyway. The precise version is a training generator threaded
    through `loop.train` the way `eval_seed` already is, which is a wider change
    than this one earns today.
    """
    if not ctx.is_distributed:
        return
    torch.manual_seed(cfg.runtime.seed + ctx.rank)


def parallelise(built: BuiltRun, ctx: DistContext, device: torch.device) -> BuiltRun:
    """Wrap the runnable module in DDP, leaving `built.model` alone.

    `BuiltRun` already draws exactly the line DDP needs, for the same reason it
    drew it for `torch.compile`: `model` is what gets checkpointed and sampled,
    `runnable` is what gets stepped. DDP prefixes state-dict keys with `module.`
    on top of compile's `_orig_mod.`, so a checkpoint written from the wrapper
    loads into nothing. Replacing only `runnable` keeps `checkpoint_payload`'s
    standing rule — pass `model`, never a wrapper — true without it having to
    learn what DDP is.

    Compile inner, DDP outer, which is the order the two wrappers are documented
    to compose in.

    `device_ids` is CUDA-only. Under gloo on CPU there is no device to name and
    passing one raises rather than being ignored — and CPU/gloo is how this gets
    tested without a GPU, so the branch earns its place.

    Construction broadcasts rank 0's parameters to every other rank. Here that is
    a no-op, since they all seeded identically and built the same weights, and it
    is a useful backstop precisely because it is a no-op: if it ever isn't,
    something upstream stopped being deterministic.
    """
    if not ctx.is_distributed:
        return built

    device_ids = [device.index] if device.type == "cuda" else None
    return built._replace(runnable=DDP(built.runnable, device_ids=device_ids))


def loss_reducer(ctx: DistContext) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """`reduce_loss` for `loop.train`, or None when there is nobody to reduce with.

    Returning None rather than an identity function is deliberate: it is the
    same "there is nothing to do here" the loop already understands for `amp` and
    `lr_schedule`, and it keeps a single-process run from paying for a function
    call per step.

    **SUM then divide, not `ReduceOp.AVG`.** AVG is implemented by NCCL only, so
    an averaging reduction works on a GPU box and raises under gloo — which is
    to say it raises in exactly the local CPU run used to check this code before
    it goes anywhere near a GPU. Sum-and-divide is correct on both, and equal
    shares are guaranteed by `shard_batch`.

    Clones before reducing so the caller's accumulator isn't mutated underneath
    it. A scalar, so the copy costs nothing and the surprise it prevents is real.
    """
    if not ctx.is_distributed:
        return None

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().clone()
        all_reduce(out, op=ReduceOp.SUM)
        return out / ctx.world_size

    return reduce
