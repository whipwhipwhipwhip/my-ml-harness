"""harness/schedule.py — arithmetic, so these are arithmetic tests.

No model, no optimiser, no tmpdir, no torch. That's the point of the schedule
being a closure over three numbers: "what lr does step 700 get" is answerable in
microseconds. If a test here starts needing a run, the schedule has grown a
dependency on something other than the step number.

`test_loop.py` covers the other half — that the loop writes what this returns
into the optimiser.
"""

from __future__ import annotations

import math

import pytest
from dacite import exceptions as dacite_exc

from ml_harness.testing import ARCHES, DATAS
from ml_harness.config import ConstantScheduleConfig, CosineScheduleConfig
from ml_harness.registry import build_lr_schedule, load_run_config
from ml_harness.schedule import constant_schedule, cosine_schedule

PEAK = 1e-3


def cosine(peak=PEAK, max_iters=1000, **fields):
    return cosine_schedule(CosineScheduleConfig(**fields), peak, max_iters)


# ---- constant -------------------------------------------------------------

def test_constant_is_the_peak_forever():
    lr = constant_schedule(ConstantScheduleConfig(), PEAK, 1000)
    assert [lr(step) for step in (0, 1, 500, 10_000)] == [PEAK] * 4


# ---- warmup ---------------------------------------------------------------

def test_warmup_first_step_is_small_but_not_zero():
    """Exactly 0 would be a step that updates nothing — a wasted iteration."""
    lr = cosine(warmup_steps=100)
    assert 0 < lr(0) < PEAK / 50


def test_warmup_is_linear():
    lr = cosine(warmup_steps=100)
    steps = range(0, 100)
    deltas = [lr(s + 1) - lr(s) for s in steps if s + 1 < 100]
    assert deltas == pytest.approx([deltas[0]] * len(deltas))


def test_peak_is_reached_once_at_the_handover():
    """`(step+1)/(warmup+1)` rather than `step/warmup`: no two-step plateau."""
    lr = cosine(warmup_steps=100)
    assert lr(99) < PEAK
    assert lr(100) == pytest.approx(PEAK)


def test_no_warmup_starts_at_the_peak():
    assert cosine(warmup_steps=0)(0) == pytest.approx(PEAK)


# ---- the cosine segment ---------------------------------------------------

def test_midpoint_is_halfway_between_peak_and_floor():
    """cos(pi/2) = 0, so the bracket is exactly 1 and the lr is the mean."""
    lr = cosine(warmup_steps=0, decay_steps=1000, min_lr_frac=0.1)
    assert lr(500) == pytest.approx((PEAK + 0.1 * PEAK) / 2)


def test_decays_monotonically_from_peak_to_floor():
    lr = cosine(warmup_steps=100, decay_steps=1000)
    values = [lr(s) for s in range(100, 1001)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(PEAK)
    assert values[-1] == pytest.approx(0.1 * PEAK)


def test_both_ends_are_flat():
    """A cosine, not a line: it eases out of the peak and into the floor."""
    lr = cosine(warmup_steps=0, decay_steps=1000)
    near_peak = lr(0) - lr(10)
    mid = lr(495) - lr(505)
    near_floor = lr(990) - lr(1000)
    assert near_peak < mid
    assert near_floor < mid


def test_warmup_is_counted_inside_decay_steps():
    """decay_steps is absolute from step 0, nanoGPT's `lr_decay_iters`.

    So the cosine spans 900 steps here, not 1000, and its midpoint is 550.
    """
    lr = cosine(warmup_steps=100, decay_steps=1000, min_lr_frac=0.1)
    assert lr(550) == pytest.approx((PEAK + 0.1 * PEAK) / 2)


# ---- the floor ------------------------------------------------------------

def test_holds_the_floor_after_decay_steps():
    lr = cosine(warmup_steps=100, decay_steps=1000, min_lr_frac=0.1)
    floor = 0.1 * PEAK
    assert [lr(s) for s in (1000, 1001, 5000)] == pytest.approx([floor] * 3)


def test_floor_is_a_fraction_of_the_peak_it_was_built_with():
    """The scale-invariance the fraction was chosen for."""
    for peak in (1e-2, 3e-4, 1e-6):
        assert cosine(peak=peak, decay_steps=1000)(2000) == pytest.approx(peak * 0.1)


def test_zero_fraction_decays_to_nothing():
    assert cosine(min_lr_frac=0.0, decay_steps=1000)(1000) == 0.0


def test_fraction_of_one_is_a_constant_schedule():
    lr = cosine(min_lr_frac=1.0, warmup_steps=0, decay_steps=1000)
    assert [lr(s) for s in (0, 500, 1000)] == pytest.approx([PEAK] * 3)


# ---- resolution against the run -------------------------------------------

def test_decay_steps_none_means_the_whole_run():
    assert cosine(max_iters=800, decay_steps=None)(800) == pytest.approx(0.1 * PEAK)
    assert cosine(max_iters=800, decay_steps=None)(400) == pytest.approx(
        cosine(max_iters=1000, decay_steps=None)(500)
    )


def test_a_run_shorter_than_the_warmup_does_not_explode():
    """`--smoke` shrinks max_iters to 4; decay_steps=None then resolves below
    warmup, and a span of -96 must not divide by zero or go negative."""
    lr = cosine(max_iters=4, warmup_steps=100, decay_steps=None)
    values = [lr(s) for s in range(4)]
    assert all(0 < v <= PEAK for v in values)
    # past the (never-reached) warmup, the floor is the honest answer
    assert lr(100) == pytest.approx(0.1 * PEAK)


def test_never_exceeds_the_peak():
    lr = cosine(warmup_steps=137, decay_steps=911, min_lr_frac=0.1)
    assert all(0 < lr(s) <= PEAK for s in range(0, 2000))


def test_the_floor_bounds_the_decay_not_the_warmup():
    """Warmup ramps from ~0, so it passes *below* the floor on the way up.

    Deliberate, and what nanoGPT does: the floor says where the decay stops, not
    that the lr is never smaller than it. A warmup clamped to the floor would
    start at 10% of peak, which is most of the way to no warmup at all.
    """
    lr = cosine(warmup_steps=137, decay_steps=911, min_lr_frac=0.1)
    assert lr(0) < 0.1 * PEAK
    assert all(0.1 * PEAK <= lr(s) <= PEAK for s in range(137, 2000))


def test_is_finite_everywhere():
    lr = cosine(warmup_steps=10, decay_steps=100)
    assert all(math.isfinite(lr(s)) for s in range(0, 500))


# ---- config plumbing ------------------------------------------------------

def test_default_is_constant_at_optim_lr(config_path):
    """No `schedule` block in the file: every config written before this loads,
    and keeps the numerics it had."""
    cfg = load_run_config(config_path, ARCHES, DATAS)
    assert cfg.schedule_arch == "constant"
    lr = build_lr_schedule(cfg)
    assert [lr(0), lr(999)] == [cfg.optim.lr, cfg.optim.lr]


def test_cosine_is_selectable_by_override_against_a_file_without_the_block(config_path):
    """The reason load_run_config seeds `schedule` before applying overrides."""
    cfg = load_run_config(
        config_path,
        ARCHES,
        DATAS,
        ("schedule_arch=cosine", "schedule.warmup_steps=10", "train.max_iters=100"),
    )
    assert isinstance(cfg.schedule, CosineScheduleConfig)
    lr = build_lr_schedule(cfg)
    assert lr(10) == pytest.approx(cfg.optim.lr)
    assert lr(100) == pytest.approx(cfg.optim.lr * 0.1)


def test_peak_comes_from_optim_lr(config_path):
    cfg = load_run_config(
        config_path, ARCHES, DATAS, ("schedule_arch=cosine", "optim.lr=0.01")
    )
    assert build_lr_schedule(cfg)(0) == pytest.approx(0.01)


def test_unknown_schedule_arch_is_rejected(config_path):
    with pytest.raises(ValueError, match="schedule_arch"):
        load_run_config(config_path, ARCHES, DATAS, ("schedule_arch=triangular",))


def test_typo_inside_the_schedule_block_is_rejected(config_path):
    """strict=True, same as every other block."""
    with pytest.raises(dacite_exc.UnexpectedDataError):
        load_run_config(
            config_path, ARCHES, DATAS,
            ("schedule_arch=cosine", "schedule.warmup_step=10"),
        )


# ---- rejected configs -----------------------------------------------------

def test_negative_warmup_rejected():
    with pytest.raises(ValueError, match="warmup_steps"):
        CosineScheduleConfig(warmup_steps=-1)


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_fraction_outside_the_unit_interval_rejected(frac):
    """>1 would put the floor above the peak and make this a ramp."""
    with pytest.raises(ValueError, match="min_lr_frac"):
        CosineScheduleConfig(min_lr_frac=frac)


def test_decay_steps_below_warmup_rejected():
    """Stated, so it's a contradiction rather than a short run — see the
    smoke case above, which is the same arithmetic and is allowed."""
    with pytest.raises(ValueError, match="decay_steps"):
        CosineScheduleConfig(warmup_steps=100, decay_steps=50)
