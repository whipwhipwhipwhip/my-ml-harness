"""ml_harness/bench.py, driven with the same fakes as the loop.

No filesystem, no GPT, no timing assertions. Wall-clock times can't really be
tested (any threshold is either flaky or meaningless on a busy machine), so these
check the *bookkeeping*: that warmup is excluded, that the tokens-per-second
arithmetic is right, and that the loop still trains. The measurements are only
as good as the sync, which is tested by checking it gets called.
"""

from __future__ import annotations

import math

import pytest
import torch

from ml_harness.testing import ARCHES, DATAS, load_data
from ml_harness.bench import BenchResult, benchmark, make_sync
from ml_harness.registry import load_run_config

from tests.ml_harness.test_loop import StubModel

DEVICE = torch.device("cpu")


@pytest.fixture
def bench_harness(config_path, corpus_path):
    """Everything `benchmark()` needs. Same shape as test_loop's `harness`."""

    def build(*overrides):
        cfg = load_run_config(config_path, ARCHES, DATAS, overrides)
        data = load_data(corpus_path, cfg.data.split_frac)
        model = StubModel(data.tokenizer.vocab_size)
        optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
        return model, optimiser, data, cfg

    return build


def run_bench(bench_harness, n_iters=3, warmup=2, *overrides):
    model, optimiser, data, cfg = bench_harness(*overrides)
    result = benchmark(model, optimiser, data, cfg, DEVICE, n_iters, warmup)
    return result, model, cfg


# ---- what gets measured ---------------------------------------------------

def test_warmup_iterations_are_excluded_from_the_measurements(bench_harness):
    """This is why the slow first iteration doesn't skew the mean."""
    result, _, _ = run_bench(bench_harness, n_iters=3, warmup=2)
    assert len(result.batch_s) == 3
    assert len(result.step_s) == 3
    assert result.warmup == 2


def test_zero_warmup_measures_everything(bench_harness):
    result, _, _ = run_bench(bench_harness, n_iters=4, warmup=0)
    assert len(result.iter_s) == 4


def test_iter_time_is_batch_plus_step(bench_harness):
    """Both parts are real time, and tokens/sec uses their sum."""
    result, _, _ = run_bench(bench_harness)
    for total, batch, step in zip(result.iter_s, result.batch_s, result.step_s):
        assert total == pytest.approx(batch + step)


def test_times_are_positive_and_finite(bench_harness):
    result, _, _ = run_bench(bench_harness)
    assert all(t > 0 and math.isfinite(t) for t in result.iter_s)


def test_tokens_per_iter_is_batch_times_block(bench_harness):
    result, _, cfg = run_bench(bench_harness)
    assert result.tokens_per_iter == cfg.train.batch_size * cfg.model.block_size


def test_throughput_is_tokens_over_that_iteration(bench_harness):
    result, _, _ = run_bench(bench_harness)
    for rate, seconds in zip(result.tokens_per_s, result.iter_s):
        assert rate == pytest.approx(result.tokens_per_iter / seconds)


# ---- that it's still a training step --------------------------------------

def test_parameters_change(bench_harness):
    """The benchmark must actually step the optimiser, not just run forward passes."""
    model, optimiser, data, cfg = bench_harness()
    before = model.emb.weight.detach().clone()
    benchmark(model, optimiser, data, cfg, DEVICE, n_iters=3, warmup=1)
    assert not torch.equal(before, model.emb.weight)


def test_model_is_left_in_train_mode(bench_harness):
    """Dropout is part of the step being timed."""
    _, model, _ = run_bench(bench_harness)
    assert model.training


def test_on_iter_sees_every_iteration_and_knows_which_count(bench_harness):
    model, optimiser, data, cfg = bench_harness()
    seen = []
    benchmark(
        model, optimiser, data, cfg, DEVICE, n_iters=3, warmup=2,
        on_iter=lambda i, s, measured: seen.append((i, measured)),
    )
    assert seen == [(0, False), (1, False), (2, True), (3, True), (4, True)]


def test_syncs_before_and_around_every_iteration(bench_harness, monkeypatch):
    """Without the sync, an async backend would only measure queueing time."""
    calls = []
    monkeypatch.setattr(
        "ml_harness.bench.make_sync", lambda device: lambda: calls.append(1)
    )
    model, optimiser, data, cfg = bench_harness()
    benchmark(model, optimiser, data, cfg, DEVICE, n_iters=3, warmup=1)
    # one up front, then two per iteration (batch split, end of step)
    assert len(calls) == 1 + 2 * 4


# ---- the summary ----------------------------------------------------------

def test_summary_is_json_serialisable_plain_data(bench_harness):
    """It's written as `--json` output, so a tensor or numpy float in here would fail."""
    import json

    result, _, _ = run_bench(bench_harness)
    json.dumps(result.summary())


def test_summary_stats_are_ordered(bench_harness):
    result, _, _ = run_bench(bench_harness, n_iters=5, warmup=1)
    stats = result.summary()["iter_s"]
    assert stats["min"] <= stats["p50"] <= stats["p90"] <= stats["max"]
    assert stats["n"] == 5


def test_aggregate_throughput_matches_total_time(bench_harness):
    """Total tokens over total time, which isn't the mean of the per-iteration rates."""
    result, _, _ = run_bench(bench_harness, n_iters=4, warmup=1)
    summary = result.summary()
    assert summary["total_s"] == pytest.approx(sum(result.iter_s))
    assert summary["tokens_per_s_aggregate"] == pytest.approx(
        result.tokens_per_iter * 4 / sum(result.iter_s)
    )


def test_zero_measured_iters_summarises_without_dividing_by_zero(bench_harness):
    """Running only warmup is probably a mistake, but shouldn't crash."""
    result, _, _ = run_bench(bench_harness, n_iters=0, warmup=1)
    summary = result.summary()
    assert summary["iters"] == 0
    assert summary["iter_s"] == {}
    assert math.isnan(summary["tokens_per_s_aggregate"])


def test_cpu_sync_is_a_noop_that_costs_nothing():
    """CPU is already synchronous, so a no-op is correct."""
    assert make_sync(torch.device("cpu"))() is None


def test_bench_result_holds_the_raw_series(bench_harness):
    """The summary can be recomputed from the lists, but not the other way round."""
    result, _, _ = run_bench(bench_harness, n_iters=3, warmup=1)
    assert isinstance(result, BenchResult)
    assert result.summary()["iter_s"]["n"] == len(result.iter_s) == 3
