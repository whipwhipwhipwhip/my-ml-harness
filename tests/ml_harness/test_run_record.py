"""The run record: what `train.run` puts on disk, including when it dies.

This is the tier that costs real seconds — a genuine GPT, a genuine optimiser, a
genuine directory — so it stays small and asserts on the record rather than on the
learning. "Did this produce a complete, readable run" is the question; "did it
learn" is test_loop's.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

import tokenizers
from ml_learning.harness.provenance import config_sha256, group_sha256
from ml_learning.harness.registry import build_model, load_run_config
from ml_learning.harness.store import RunDir, load_metrics, load_runs
from ml_learning.nanogpt_conor import train as train_mod
from ml_learning.nanogpt_conor.arches import ARCHES
from ml_learning.nanogpt_conor.datas import DATAS
from ml_learning.nanogpt_conor.smoke import SMOKE_SUBDIR

# 4 iters, eval every 2 -> evals at 0, 2, 3
EVAL_STEPS = [0, 2, 3]


@pytest.fixture
def completed(gpt_config_path, runs_root):
    return train_mod.run(gpt_config_path, runs_root=runs_root)


def read(path):
    return json.loads(path.read_text())


# ---- the happy path -------------------------------------------------------

def test_every_file_is_written(completed):
    for path in (
        completed.config_path,
        completed.provenance_path,
        completed.summary_path,
        completed.status_path,
        completed.metrics_path,
        completed.tokenizer_path,
        completed.ckpt_path("best"),
        completed.ckpt_path("last"),
    ):
        assert path.exists(), path


def test_status_is_completed_and_bracketed_by_timestamps(completed):
    status = read(completed.status_path)
    assert status["status"] == "completed"
    assert status["error"] is None
    assert status["started_at"] < status["ended_at"]
    assert status["smoke"] is False


def test_config_records_the_effective_config_not_the_file(gpt_config_path, runs_root):
    """What makes a run reproducible from its own record rather than from the CLI."""
    rd = train_mod.run(gpt_config_path, ["model.n_layer=3"], runs_root=runs_root)
    assert read(rd.config_path)["model"]["n_layer"] == 3
    assert read(gpt_config_path)["model"]["n_layer"] == 2


def test_summary_reports_the_steps_that_ran(completed):
    summary = read(completed.summary_path)
    assert summary["steps_completed"] == 4
    assert summary["bad_rows"] == 0
    assert summary["elapsed_s"] > 0


def test_summary_carries_a_final_val_loss(completed):
    summary = read(completed.summary_path)
    assert summary["val.loss.final"] > 0
    assert summary["val.loss.min_step"] in EVAL_STEPS


def test_metrics_are_one_json_object_per_line(completed):
    lines = completed.metrics_path.read_text().splitlines()
    assert lines
    assert all(json.loads(line)["metric"] for line in lines)


def test_a_sample_is_written_per_eval(completed):
    samples = sorted(p.name for p in completed.samples_dir.iterdir())
    assert samples == [f"step_{s:06d}.txt" for s in EVAL_STEPS]
    assert all((completed.samples_dir / s).read_text().strip() for s in samples)


def test_provenance_hashes_the_effective_config(completed):
    """Run identity as a column, so grouping runs is a groupby rather than a
    hand-assembled list of ids. Hashed from the *effective* config, which is why
    it's checked against the config the run wrote rather than the file it read."""
    prov = read(completed.provenance_path)
    written = read(completed.config_path)

    assert prov["config_sha256"] == config_sha256(written)
    assert prov["group_sha256"] == group_sha256(written)
    assert prov["config_sha256"] != prov["group_sha256"]


def test_provenance_records_the_data_that_was_trained_on(completed, corpus_path):
    """The only record of the corpus: `*.txt` is gitignored, so git can't be one."""
    prov = read(completed.provenance_path)
    assert prov["data_sha256"] == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert prov["device_resolved"] == "cpu"
    assert prov["vocab_size"] > 0
    assert prov["torch_version"]


def test_provenance_records_the_code_that_ran(completed):
    """`git_sha` covers this repo. It does not cover a pinned dependency, and
    `tokenizers` is already one — so a run's `tokenizer_sha256` can match
    another's while a different encoder produced the ids. These two fields are
    what stays true once `harness/` is a dependency as well."""
    prov = read(completed.provenance_path)
    assert prov["harness_version"]
    assert prov["tokenizers_version"]


def test_the_versions_are_not_part_of_the_experiment_identity(completed):
    """Provenance is observed, not chosen: adding fields here must not regroup
    anything. If these ever entered the hash, every upgrade would silently split
    an experiment in two."""
    prov = read(completed.provenance_path)
    written = read(completed.config_path)

    assert "harness_version" not in written
    assert prov["group_sha256"] == group_sha256(written)


# ---- the tokenizer is part of the record ----------------------------------

def test_the_tokenizer_can_be_rebuilt_from_the_run(completed):
    """Without this file the checkpoints are integers with no meaning: the
    alphabet came from a corpus that git doesn't track."""
    tok = tokenizers.from_dict(read(completed.tokenizer_path))

    assert tok.vocab_size == read(completed.provenance_path)["vocab_size"]
    assert tok.decode(tok.encode("hello")) == "hello"


def test_provenance_fingerprints_the_tokenizer(completed):
    prov = read(completed.provenance_path)
    tok = tokenizers.from_dict(read(completed.tokenizer_path))

    assert prov["tokenizer_kind"] == "char"
    assert prov["tokenizer_sha256"] == tokenizers.fingerprint(tok)


def test_a_different_tokenizer_is_a_different_run(gpt_config_path, runs_root):
    """The reason the fingerprint exists: same corpus, same model, different
    training data — and `vocab_size` on its own wouldn't always say so."""
    char = train_mod.run(gpt_config_path, runs_root=runs_root)
    byte = train_mod.run(gpt_config_path, ["tokenizer_arch=byte"], runs_root=runs_root)

    assert read(byte.tokenizer_path) == {"kind": "byte"}
    assert read(byte.provenance_path)["vocab_size"] == 256
    assert (
        read(char.provenance_path)["tokenizer_sha256"]
        != read(byte.provenance_path)["tokenizer_sha256"]
    )


def test_a_run_writes_and_then_reuses_the_token_cache(gpt_config_path, runs_root, tmp_path):
    cache_dir = tmp_path / "cache"
    for _ in range(2):
        train_mod.run(
            gpt_config_path, ["data.cache=true"], runs_root=runs_root, cache_dir=cache_dir
        )

    assert len([p for p in cache_dir.iterdir() if p.suffix == ".npy"]) == 1


def test_the_cache_does_not_change_the_run(gpt_config_path, runs_root, tmp_path):
    """Its only permitted effect is on how long the run takes."""
    uncached = train_mod.run(gpt_config_path, runs_root=runs_root, cache_dir=None)
    cached = train_mod.run(
        gpt_config_path,
        ["data.cache=true"],
        runs_root=runs_root,
        cache_dir=tmp_path / "cache",
    )

    assert (
        read(cached.summary_path)["val.loss.final"]
        == read(uncached.summary_path)["val.loss.final"]
    )


def test_a_run_with_caching_off_writes_nothing(gpt_config_path, runs_root, tmp_path):
    """conftest sets `data.cache: false`, which is what keeps the suite out of
    the developer's real `.cache/` — assert it rather than trust it."""
    cache_dir = tmp_path / "cache"
    train_mod.run(gpt_config_path, runs_root=runs_root, cache_dir=cache_dir)

    assert not cache_dir.exists()


def test_samples_start_from_the_configured_prompt(gpt_config_path, runs_root):
    """Generation is seeded with encoded text, not a hardcoded id 0."""
    rd = train_mod.run(
        gpt_config_path, ["train.sample_prompt=abc"], runs_root=runs_root
    )
    sample = (rd.samples_dir / f"step_{EVAL_STEPS[0]:06d}.txt").read_text()

    assert sample.splitlines()[1].startswith("abc")


def test_two_runs_do_not_collide(gpt_config_path, runs_root):
    first = train_mod.run(gpt_config_path, runs_root=runs_root)
    second = train_mod.run(gpt_config_path, runs_root=runs_root)
    assert first.run_id != second.run_id


# ---- the checkpoint is usable ---------------------------------------------

@pytest.mark.parametrize("tag", ["best", "last"])
def test_checkpoint_reloads_into_a_fresh_model(completed, tag):
    """A checkpoint that can't be loaded is a file, not a checkpoint."""
    ckpt = torch.load(completed.ckpt_path(tag), weights_only=False)
    cfg = load_run_config(completed.config_path, ARCHES, DATAS)
    model = build_model(cfg, ckpt["vocab_size"], ARCHES)

    model.load_state_dict(ckpt["model"])  # raises on any mismatch

    assert ckpt["step"] in EVAL_STEPS
    assert ckpt["config"]["model"] == read(completed.config_path)["model"]


def test_last_checkpoint_is_the_final_step(completed):
    assert torch.load(completed.ckpt_path("last"), weights_only=False)["step"] == 3


def test_best_checkpoint_is_the_lowest_val_loss_step(completed):
    ckpt = torch.load(completed.ckpt_path("best"), weights_only=False)
    assert ckpt["step"] == read(completed.summary_path)["val.loss.min_step"]


# ---- smoke mode -----------------------------------------------------------

def test_smoke_writes_to_its_own_subdirectory(gpt_config_path, runs_root):
    rd = train_mod.run(gpt_config_path, runs_root=runs_root, smoke=True)
    assert rd.path.parent == runs_root / SMOKE_SUBDIR
    assert read(rd.status_path)["smoke"] is True


def test_smoke_overrides_shrink_the_config(gpt_config_path, runs_root):
    rd = train_mod.run(gpt_config_path, runs_root=runs_root, smoke=True)
    cfg = read(rd.config_path)
    assert cfg["train"]["max_iters"] == 4
    assert cfg["model"]["n_head"] == 2


def test_explicit_override_beats_smoke(gpt_config_path, runs_root):
    """`--smoke --set train.max_iters=6` is a longer smoke run, not a conflict."""
    rd = train_mod.run(gpt_config_path, ["train.max_iters=6"], runs_root=runs_root, smoke=True)
    assert read(rd.config_path)["train"]["max_iters"] == 6
    assert read(rd.summary_path)["steps_completed"] == 6


# ---- the paths that only exist when things go wrong -----------------------

def test_a_crash_still_leaves_a_complete_record(gpt_config_path, runs_root, monkeypatch):
    """The `finally` block is the point: a failed run is still a run you can read."""
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(train_mod, "train", boom)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        train_mod.run(gpt_config_path, runs_root=runs_root)

    rd = RunDir(root=runs_root, run_id=next(p.name for p in runs_root.iterdir()))
    status = read(rd.status_path)
    assert status["status"] == "failed"
    assert "synthetic failure" in status["error"]
    assert rd.summary_path.exists()
    assert rd.ckpt_path("last").exists()


def test_a_crash_before_the_first_step_records_step_minus_one(
    gpt_config_path, runs_root, monkeypatch
):
    """`result.get(...)`: the NameError-inside-finally case, which masks the real bug."""
    monkeypatch.setattr(
        train_mod, "train", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("early"))
    )

    with pytest.raises(RuntimeError):
        train_mod.run(gpt_config_path, runs_root=runs_root)

    rd = RunDir(root=runs_root, run_id=next(p.name for p in runs_root.iterdir()))
    assert torch.load(rd.ckpt_path("last"), weights_only=False)["step"] == -1


def test_an_interrupt_is_recorded_not_raised(gpt_config_path, runs_root, monkeypatch):
    """Ctrl-C on a long run should leave the record, not a traceback."""
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train_mod, "train", interrupt)

    rd = train_mod.run(gpt_config_path, runs_root=runs_root)

    status = read(rd.status_path)
    assert status["status"] == "interrupted"
    assert status["error"] is None
    assert rd.ckpt_path("last").exists()


# ---- reading the runs back ------------------------------------------------

def test_load_runs_finds_the_run(completed, runs_root):
    df = load_runs(runs_root)
    assert list(df["run_id"]) == [completed.run_id]
    assert df["config.model.n_layer"].iloc[0] == 2
    assert df["summary.steps_completed"].iloc[0] == 4


def test_load_metrics_finds_the_metrics(completed, runs_root):
    df = load_metrics(runs_root)
    assert set(df["run_id"]) == {completed.run_id}
    assert set(df["metric"]) == {"batch_loss", "lr", "loss"}


def test_smoke_runs_do_not_pollute_the_analysis_frame(gpt_config_path, runs_root):
    """`runs/smoke_tests/` is a directory of runs, not a run."""
    real = train_mod.run(gpt_config_path, runs_root=runs_root)
    train_mod.run(gpt_config_path, runs_root=runs_root, smoke=True)

    df = load_runs(runs_root)
    assert list(df["run_id"]) == [real.run_id]
    assert SMOKE_SUBDIR not in list(df["run_id"])
