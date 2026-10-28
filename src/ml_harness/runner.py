"""The parts of an entry point that don't depend on the model or the task.

`nanogpt_conor/train.py` and `gpt_addition/train.py` had the same ~85 lines in
common: build the model the config names, then wrap training in the bookkeeping
that turns a run into a *record*. None of it mentions a GPT or an addition
problem, and most of it has to happen in a particular order that fails silently
if you get it wrong, so it's better written once here than copied per project.

**Loading the data** stays in the entry points. The two projects do it very
differently (nanogpt reads a file through a token cache, the adder generates a
corpus from its config), and a signature that fit both would be too vague to be
useful. So `data` arrives here already loaded, and this module only uses the
same duck-typed interface `loop.py` does:

    data.get_batch(split, batch, block, device, iters, generator)
    data.splits        mapping of split name -> the split's tokens
    data.tokenizer     with .vocab_size, .to_dict()
    data.sha256        content hash of the corpus, for provenance

As everywhere else in `ml_harness/`, `ARCHES` is passed in rather than
imported, so this module names no model. This module uses torch (to save
checkpoints), which is why none of it went into the torch-free `store.py`.
"""

from __future__ import annotations

import os
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import torch

from ml_harness.config import RunConfig
from ml_harness.device import compile_model
from ml_harness.metrics import summarize
from ml_harness.provenance import collect_provenance
from ml_harness.registry import (
    ArchSpec,
    build_lr_schedule,
    build_model,
    build_optimiser,
)
from ml_harness.seeding import make_generator, seed_everything
from ml_harness.store import RunDir, new_run, write_status, write_summary
from ml_harness.io import _write_json


class BuiltRun(NamedTuple):
    """Everything `loop.train` needs that comes out of a config plus a device.

    `model` and `runnable` share the same weights, but they're not
    interchangeable; see `build_run`.
    """

    model: torch.nn.Module
    runnable: torch.nn.Module
    optimiser: torch.optim.Optimizer
    lr_schedule: Any


def build_run(
    cfg: RunConfig,
    vocab_size: int,
    ARCHES: dict[str, ArchSpec],
    device: torch.device | str,
) -> BuiltRun:
    """Seed, build, compile, then make the optimiser and schedule, in that order.

    The order matters, and getting it wrong gives a run that trains fine but
    means something different:

    - **Seed before `build_model`.** Otherwise the initial weights are already
      drawn, so the run isn't reproducible and nothing tells you.
    - **`runnable` is a second name for `model`, not a replacement.** A
      checkpoint saved from a compiled wrapper has `_orig_mod.`-prefixed keys
      and won't load into a plain model, so callers save and sample from `model`
      and train with `runnable`. With compile off they're the same object.
    - **Optimiser from `model`, not `runnable`.** The parameters are the same
      tensors either way, and this keeps `param_groups` independent of compile.
    - **Schedule after the optimiser.** `optim.lr` is the *peak*, and the
      schedule scales it.

    Returns a NamedTuple so callers can use `.model` etc. instead of
    remembering positions.
    """
    seed_everything(cfg.runtime.seed)
    model = build_model(cfg, vocab_size, ARCHES).to(device)
    runnable = compile_model(model, cfg.runtime.compile)
    optimiser = build_optimiser(cfg, model)
    return BuiltRun(model, runnable, optimiser, build_lr_schedule(cfg))


def draw_eval_batches(data, cfg: RunConfig, device: torch.device | str) -> dict:
    """The fixed evaluation set, drawn once from its own generator.

    The separate generator is the point: `runtime.eval_seed` is independent of
    `runtime.seed`, so validation loss is always measured on the same batches,
    whatever training's RNG did, and two runs can be compared. If these came
    from the training RNG, changing the config would change the eval set too.

    Drawn once rather than at every eval for the same reason.
    """
    eval_gen = make_generator(cfg.runtime.eval_seed)
    return {
        split: data.get_batch(
            split,
            cfg.train.batch_size,
            cfg.model.block_size,
            str(device),
            cfg.train.eval_iters,
            eval_gen,
        )
        for split in data.splits
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    step: int,
    vocab_size: int,
    cfg: RunConfig,
) -> dict:
    """What goes in a checkpoint. Defined once because two places write one.

    `vocab_size` and `config` are included because a `state_dict` alone isn't
    enough to load: you need the config and the vocabulary size to rebuild the
    module first. `store.py` only provides the path and never sees what's in
    the file, which keeps it torch-free.

    Pass `model`, never a compiled wrapper (see `build_run`).
    """
    return {
        "model": model.state_dict(),
        "optim": optimiser.state_dict(),
        "step": step,
        "vocab_size": vocab_size,
        "config": asdict(cfg),
    }


class RunRecord:
    """The object `record_run` yields, for the caller to put its result in.

    `record_run`'s cleanup needs to know which step training reached, but that's
    only known *inside* the `with` block, and a context manager can't get a
    return value out of its body. `result` defaults to `{}` to handle a crash
    before the first step: `train()` never returned, so the final step is
    recorded as -1 rather than cleanup failing with a `NameError`.
    """

    def __init__(self) -> None:
        self.result: dict = {}


@contextmanager
def record_run(
    runs_root: Path | str,
    cfg: RunConfig,
    data,
    built: BuiltRun,
    *,
    device: torch.device | str,
    repo_root: Path | str,
    smoke: bool = False,
    world_size: int = 1,
) -> Iterator[tuple[RunDir, RunRecord]]:
    """Open a run directory, and close it out properly however the block exits.

    It's a context manager because **the exit handling is the useful part.** A
    run that crashes at step 40 or gets a Ctrl-C should still leave a complete
    record (status, summary and a `last` checkpoint), whether it returned,
    raised or was interrupted. Writing that `try`/`finally` in every entry point
    would mean copying some fiddly control flow into each project.

    The two ways of failing are handled differently on purpose:

    - **KeyboardInterrupt is swallowed** and recorded as `interrupted`.
      Stopping a run you're watching is a normal way to end it, not a failure.
      (`sweep.py` relies on this: since `run()` returns normally after a Ctrl-C,
      it reads `status.json` to find out.)
    - **Any other exception is recorded and re-raised**, so a bug still shows
      up in the terminal with its traceback.

    The directory is created *before* training starts, so a run killed at step
    3 can still be read. `tokenizer.json` is written up front for the same
    reason: without it a checkpoint's ids can't be decoded, since the corpus is
    gitignored.

    Yields `(rd, record)`. Put the result of `loop.train` in `record.result`
    and everything else is handled.
    """
    vocab_size = data.tokenizer.vocab_size

    rd = new_run(
        runs_root,
        asdict(cfg),
        collect_provenance(
            str(device),
            vocab_size,
            repo_root,
            data.sha256,
            data.tokenizer,
            asdict(cfg),
            world_size,
        ),
    )
    # Written here rather than in `new_run`, like the checkpoints: store.py
    # provides the path but doesn't know about the contents.
    _write_json(rd.tokenizer_path, data.tokenizer.to_dict())

    started_at = datetime.now(timezone.utc)
    write_status(
        rd,
        status="running",
        started_at=started_at.isoformat(),
        pid=os.getpid(),
        smoke=smoke,
    )
    print(f"{'smoke run' if smoke else 'run'}: {rd.path}")

    record = RunRecord()
    status, error = "completed", None

    try:
        yield rd, record
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status, error = "failed", traceback.format_exc()
        raise
    finally:
        write_status(
            rd,
            status=status,
            started_at=started_at.isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
            error=error,
            smoke=smoke,
        )
        # This reads the metrics file back, so the caller's MetricLogger must be
        # closed by now, i.e. its `with` goes *inside* this one.
        write_summary(rd, summarize(rd.metrics_path))
        torch.save(
            checkpoint_payload(
                built.model,
                built.optimiser,
                record.result.get("final_step", -1),
                vocab_size,
                cfg,
            ),
            rd.ckpt_path("last"),
        )
        print(f"[{status}] {rd.path}")
