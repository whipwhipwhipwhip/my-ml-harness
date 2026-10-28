"""Learning-rate schedules: pure functions of the step number.

No torch. A schedule is just arithmetic on ints and floats, so its tests are
cheap: no model, no optimiser, no tmpdir, just "what lr does step 700 get".

Each builder takes `(cfg, peak_lr, max_iters)` and returns a `(step) -> lr`
function, rather than being called every step with the config. Three reasons:

  * `decay_steps=None` has to be turned into `train.max_iters`, and doing that
    once up front means the loop never sees a half-resolved schedule.
  * the loop doesn't need any branches: the constant schedule is a function
    too, and `lr_schedule(step)` works the same whichever one it got.
  * a test, or a caller doing something the config can't express, can pass any
    `(int) -> float`, because that's the whole interface.

`peak_lr` is `optim.lr`, passed in as an argument for the same reason
`vocab_size` is in `build_model`: it belongs to another part of the config, and
the learning rate should only be set in one place.
"""

from __future__ import annotations

import math
from typing import Callable

from ml_harness.config import ConstantScheduleConfig, CosineScheduleConfig

LRSchedule = Callable[[int], float]


def constant_schedule(
    cfg: ConstantScheduleConfig, peak_lr: float, max_iters: int
) -> LRSchedule:
    """Hold the peak for the whole run (same as having no schedule).

    Takes two arguments it ignores so that every builder in the table has the
    same signature, like `TokenizerSpec.build` does with the corpus.
    """
    return lambda step: peak_lr


def cosine_schedule(
    cfg: CosineScheduleConfig, peak_lr: float, max_iters: int
) -> LRSchedule:
    """Linear warmup -> cosine decay -> flat floor.

    Everything the schedule depends on is worked out here, so the returned
    function is just arithmetic on a few numbers.
    """
    warmup = cfg.warmup_steps
    # None means "decay over the whole run".
    decay_end = cfg.decay_steps if cfg.decay_steps is not None else max_iters
    floor = peak_lr * cfg.min_lr_frac
    # Length of the cosine segment. This can only be <= 0 when decay_steps was
    # None: e.g. `--smoke` shrinks max_iters to 4, and a config with 100 warmup
    # steps then has a negative span. The check below treats that as "no cosine
    # segment", which is fine for a 4-step smoke run and avoids dividing by zero.
    span = decay_end - warmup

    def lr_at(step: int) -> float:
        if step < warmup:
            # (step+1)/(warmup+1), not step/warmup: the first step gets a small
            # nonzero lr rather than exactly 0 (which would update nothing), and
            # the peak is only hit once, at the end of warmup, rather than on two
            # steps in a row.
            return peak_lr * (step + 1) / (warmup + 1)
        if span <= 0 or step >= decay_end:
            return floor
        # t goes from 0 at the end of warmup to 1 at decay_end, so cos goes from
        # +1 to -1, the bracket from 2 to 0, and the lr from peak to floor. The
        # curve is flat at both ends, so it eases out of the peak and into the
        # floor rather than having a sharp corner at either.
        t = (step - warmup) / span
        return floor + 0.5 * (peak_lr - floor) * (1 + math.cos(math.pi * t))

    return lr_at
