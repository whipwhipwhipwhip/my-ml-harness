"""A training loop that measures itself instead of learning anything.

Same contract as `loop.py`: takes objects, returns facts. No I/O, no filesystem,
no argparse — the import list is `torch` and `RunConfig`. Separate module rather
than a flag on `train()` because the two want opposite things. `train()` should
never sync the device or drop an eval to keep a number honest; a benchmark has to
do both, and mixing them would put timing branches inside the loop that runs for
real.

What makes the numbers mean something:

  * **Synchronisation.** CUDA and MPS queue work asynchronously, so `perf_counter`
    around a forward/backward measures how long it took to *enqueue* the kernels —
    on MPS the first iterations look impossibly fast and the cost turns up
    wherever the next sync happens. `sync_fn` blocks until the queue drains, so
    each interval covers the work it brackets. On CPU it's a no-op and costs
    nothing.
  * **Warmup.** The first iterations pay for lazy kernel compilation, allocator
    growth and cache-warming, and on MPS that can be an order of magnitude. They
    run (the steady state depends on the allocator having grown) but are dropped
    from the statistics, and the count is reported so a suspicious mean can be
    re-read.
  * **No `.item()`.** Reading the loss back is itself a sync point, so a loop that
    logged its loss would be measuring a slightly different program than the one
    being timed. Nothing here reads a tensor's value.

Batch construction is timed separately from the optimisation step. They're both
real wall time — tokens/sec is quoted against their sum — but the split is the
answer to "is this model slow, or is `get_batch` stacking on the CPU while the
GPU idles", which is a different fix.
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

    CPU is genuinely synchronous, so the no-op is correct rather than a fallback.
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

    Median and p90 rather than mean alone because iteration times are not
    symmetric: a GC pause or a page fault shifts the mean and leaves the median
    where it was, and the gap between the two is the signal that something
    intermittent is happening.
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

    Holds the raw lists, not just the summary: the shape of the series is the
    interesting part when a run is bimodal, and `summary()` can always be
    recomputed from them. One entry per *measured* iteration — warmup is already
    excluded, and `warmup` records how many were dropped.
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

        Against the total rather than the step alone: a token that took 3ms to
        stack cost 3ms, whoever was idle while it happened.
        """
        return [self.tokens_per_iter / t for t in self.iter_s]

    def summary(self) -> dict:
        """The whole thing as plain data — JSON-serialisable, torch-free."""
        iters = self.iter_s
        return {
            "iters": len(iters),
            "warmup": self.warmup,
            "tokens_per_iter": self.tokens_per_iter,
            # total_s over the measured iterations only; the aggregate throughput
            # below is what you'd get by timing the whole block with one stopwatch,
            # which is not quite the mean of the per-iteration rates
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

    The body is deliberately the same four lines as `loop.train`'s — forward,
    zero_grad, backward, step, with the same autocast scoping — because a
    benchmark of a different loop measures a different program. What's absent is
    the whole reason this exists: no eval, no logging, no sampling, no
    checkpointing, nothing that reads a tensor back.

    on_iter: optional callable(i, seconds, measured: bool) for progress output.
             The caller's business; this module doesn't print.
    amp:     the autocast context, or None for fp32. Same source and same
             scoping as `loop.train` — measuring it wrapped differently would
             be measuring a program the harness can't run.
    """
    amp = amp if amp is not None else contextlib.nullcontext()
    sync = make_sync(device)

    batch_size = cfg.train.batch_size
    block_size = cfg.model.block_size
    tokens_per_iter = batch_size * block_size

    batch_s: list[float] = []
    step_s: list[float] = []

    # dropout on: this is timing the training step, and dropout is part of it
    model.train()

    # Everything is enqueued before the first measurement, so t0 isn't paying for
    # work the caller queued (moving the model to the device, say).
    sync()

    for i in range(warmup + n_iters):
        t0 = time.perf_counter()
        xb, yb = data.get_batch("train", batch_size, block_size, str(device))[0]
        # the host->device copy is part of batch construction, and on MPS/CUDA
        # it's async, so it has to be drained before the clock splits
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
