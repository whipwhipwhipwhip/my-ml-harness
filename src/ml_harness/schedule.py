"""Learning-rate schedules: pure functions of the step number.

No torch. A schedule is arithmetic over ints and floats, and keeping it that way
means the tests for it live in the cheapest tier — no model, no optimiser, no
tmpdir, just "what lr does step 700 get".

Each builder here takes `(cfg, peak_lr, max_iters)` and returns a
`(step) -> lr` closure rather than being called per-step with the config. Three
reasons, in order of how much they matter:

  * `decay_steps=None` has to be resolved against `train.max_iters`, and doing it
    once at build time means the loop can't be handed a half-resolved schedule.
  * the loop then has no branch: the constant schedule is a closure too, and
    `lr_schedule(step)` means the same thing whichever one it got.
  * a test — or a caller doing something the config can't express — can pass any
    `(int) -> float`, because that closure is the whole contract.

`peak_lr` is `optim.lr` and arrives as an argument for the same reason
`vocab_size` does in `build_model`: it belongs to another block of the config,
and a schedule that read it out of the RunConfig itself would be the second place
naming a learning rate.
"""

from __future__ import annotations

import math
from typing import Callable

from ml_harness.config import ConstantScheduleConfig, CosineScheduleConfig

LRSchedule = Callable[[int], float]


def constant_schedule(
    cfg: ConstantScheduleConfig, peak_lr: float, max_iters: int
) -> LRSchedule:
    """Hold the peak for the whole run — what an unscheduled run does.

    Takes the two arguments it ignores because every builder in the table has one
    signature, the same bargain `TokenizerSpec.build` makes with the corpus: the
    caller never has to know which kind it built.
    """
    return lambda step: peak_lr


def cosine_schedule(
    cfg: CosineScheduleConfig, peak_lr: float, max_iters: int
) -> LRSchedule:
    """Linear warmup -> cosine decay -> flat floor.

    Everything the schedule depends on is resolved here, so the returned closure
    is arithmetic on three numbers.
    """
    warmup = cfg.warmup_steps
    # The one thing that isn't in the schedule block: None means "the whole run".
    decay_end = cfg.decay_steps if cfg.decay_steps is not None else max_iters
    floor = peak_lr * cfg.min_lr_frac
    # Length of the cosine segment. Can be <= 0 only in the resolved case —
    # `--smoke` shrinks max_iters to 4, and a config warming up over 100 steps
    # then decays across a negative span. The guard below reads that as "no
    # cosine segment", which is the honest answer for a 4-step pre-flight and
    # keeps a smoke test from dividing by zero.
    span = decay_end - warmup

    def lr_at(step: int) -> float:
        if step < warmup:
            # (step+1)/(warmup+1), not step/warmup: the first step gets a small
            # nonzero lr rather than exactly 0 (a step that updates nothing), and
            # the peak is hit once, at the handover, rather than on two steps
            # running.
            return peak_lr * (step + 1) / (warmup + 1)
        if span <= 0 or step >= decay_end:
            return floor
        # 0 at the end of warmup -> 1 at decay_end. cos takes that half period
        # from +1 to -1, so the bracket runs 2 -> 0 and the lr peak -> floor,
        # leaving both ends flat: the schedule eases out of the peak and into the
        # floor rather than cornering at either.
        t = (step - warmup) / span
        return floor + 0.5 * (peak_lr - floor) * (1 + math.cos(math.pi * t))

    return lr_at
