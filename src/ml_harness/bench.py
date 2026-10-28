"""A training loop that times itself instead of learning anything.

Same rules as `loop.py`: takes objects, returns results, no I/O, no filesystem,
no argparse, and only imports `torch` and `RunConfig`. It's a separate module
rather than a flag on `train()` because the two want opposite things: `train()`
shouldn't sync the device or skip evals just to get accurate timings, and a
benchmark has to do both. Mixing them would put timing code inside the real
training loop.

What makes the numbers trustworthy:

  * **Synchronisation.** CUDA and MPS run work asynchronously, so timing a
    forward/backward with `perf_counter` only measures how long it took to
    *queue* the kernels. On MPS the first iterations look impossibly fast and the
    real cost shows up at the next sync. The sync function waits for the queue
    to finish, so each interval covers the work inside it. On CPU it does
    nothing.
  * **Warmup.** The first iterations pay for kernel compilation, the allocator
    growing and caches warming up, and on MPS that can be 10x slower. They still
    run (the steady state depends on them) but aren't counted, and the number
    skipped is reported.
  * **No `.item()`.** Reading the loss back forces a sync, so a loop that logged
    its loss would be timing a slightly different program. Nothing here reads a
    tensor's value.

Building the batch is timed separately from the optimiser step. Both are real
time, and tokens/sec uses their sum, but the split tells you whether the model
is slow or whether `get_batch` is busy on the CPU while the GPU waits, which
need different fixes.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch

from ml_harness.config import RunConfig


def make_sync(device) -> Callable[[], None]:
    """A function that blocks until `device` has finished the work queued on it.

    CPU is already synchronous, so a no-op is correct there.
    """
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.cuda.synchronize
    if kind == "mps":
        return torch.mps.synchronize
    return lambda: None


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. `sorted_values` must already be sorted."""
    if not sorted_values:
        return float("nan")
    i = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[i]


def _stats(values: list[float]) -> dict[str, float]:
    """mean/std/min/median/p90/max over a list of measurements.

    Median and p90 as well as the mean, because iteration times are skewed: a
    GC pause or a page fault moves the mean but not the median, and a gap
    between the two means something intermittent is going on.
    """
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        # stdev needs two points; one measurement has no spread rather than an error
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class BenchResult:
    """Per-iteration measurements, plus enough context to interpret them.

    Keeps the raw lists, not just the summary, since the shape of the series
    is what's interesting when timings are bimodal, and `summary()` can always
    be recomputed. One entry per *measured* iteration: warmup is already
    excluded, and `warmup` records how many were skipped.
    """

    batch_s: list[float]
    step_s: list[float]
    tokens_per_iter: int
    warmup: int

    @property
    def iter_s(self) -> list[float]:
        """Total wall time per iteration: batch construction plus the step."""
        return [b + s for b, s in zip(self.batch_s, self.step_s)]

    @property
    def tokens_per_s(self) -> list[float]:
        """Throughput per iteration, against total iteration time.

        Uses the total rather than just the step, since time spent building
        the batch is still time spent.
        """
        return [self.tokens_per_iter / t for t in self.iter_s]

    def summary(self) -> dict:
        """Everything as plain data, JSON-serialisable and torch-free."""
        iters = self.iter_s
        return {
            "iters": len(iters),
            "warmup": self.warmup,
            "tokens_per_iter": self.tokens_per_iter,
            # total_s only covers measured iterations. The aggregate throughput
            # below is what you'd get timing the whole block at once, which isn't
            # quite the same as the mean of the per-iteration rates
            "total_s": sum(iters),
            "tokens_per_s_aggregate": (
                self.tokens_per_iter * len(iters) / sum(iters) if iters else float("nan")
            ),
            "iter_s": _stats(iters),
            "batch_s": _stats(self.batch_s),
            "step_s": _stats(self.step_s),
            "tokens_per_s": _stats(self.tokens_per_s),
        }


def benchmark(
    model,
    optimiser,
    data,
    cfg: RunConfig,
    device,
    n_iters: int = 100,
    warmup: int = 5,
    on_iter=None,
    amp=None,
) -> BenchResult:
    """Run `warmup + n_iters` training steps, timing the last `n_iters` of them.

    The body is the same as `loop.train`'s (forward, zero_grad, backward,
    step, with autocast in the same place), because timing a different loop
    would tell you about a different program. What's missing is the point: no
    eval, no logging, no sampling, no checkpointing, and nothing that reads a
    tensor back.

    Note that it doesn't do gradient accumulation, so it always runs the whole
    `train.batch_size` in one pass.

    on_iter: optional callable(i, seconds, measured: bool) for progress output.
             This module doesn't print anything itself.
    amp:     the autocast context, or None for fp32, used the same way as in
             `loop.train`.
    """
    amp = amp if amp is not None else contextlib.nullcontext()
    sync = make_sync(device)

    batch_size = cfg.train.batch_size
    block_size = cfg.model.block_size
    tokens_per_iter = batch_size * block_size

    batch_s: list[float] = []
    step_s: list[float] = []

    # Dropout on: this is timing the training step, and dropout is part of it
    model.train()

    # Wait for anything already queued (e.g. moving the model to the device),
    # so the first measurement doesn't include it.
    sync()

    for i in range(warmup + n_iters):
        t0 = time.perf_counter()
        xb, yb = data.get_batch("train", batch_size, block_size, str(device))[0]
        # The host->device copy counts as building the batch, and on MPS/CUDA
        # it's async, so wait for it before taking the time
        sync()
        t1 = time.perf_counter()

        with amp:
            _, loss = model(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        sync()
        t2 = time.perf_counter()

        measured = i >= warmup
        if measured:
            batch_s.append(t1 - t0)
            step_s.append(t2 - t1)

        if on_iter is not None:
            on_iter(i, t2 - t0, measured)

    return BenchResult(batch_s, step_s, tokens_per_iter, warmup)
