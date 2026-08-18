"""The two halves of an entry point that aren't about the model or the task.

`nanogpt_conor/train.py` and `gpt_addition/train.py` had the same ~85 lines
between them: build the model the config names, then wrap training in the
bookkeeping that makes a run a *record*. Neither half mentions a GPT or an
addition problem, and both are made of ordering constraints that are silently
wrong rather than loudly broken when you get them out of order — which is
exactly the kind of thing that shouldn't be retyped per project.

What deliberately stayed in the entry points: **loading the data**. The two
projects disagree about what that even means — nanogpt reads a file through a
token cache, the adder generates a corpus from its config — and a signature wide
enough for both would be a signature that describes neither. So `data` arrives
here already loaded, and this module only reads it through the same duck-typed
surface `loop.py` already relies on:

    data.get_batch(split, batch, block, device, iters, generator)
    data.splits        mapping of split name -> the split's tokens
    data.tokenizer     with .vocab_size, .to_dict()
    data.sha256        content hash of the corpus, for provenance

Same rule as everywhere else in `harness/`: `ARCHES` is passed in, never
imported, so this module names no model. Torch lives here (it saves
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

from ml_learning.harness.config import RunConfig
from ml_learning.harness.device import compile_model
from ml_learning.harness.metrics import summarize
from ml_learning.harness.provenance import collect_provenance
from ml_learning.harness.registry import (
    ArchSpec,
    build_lr_schedule,
    build_model,
    build_optimiser,
)
from ml_learning.harness.seeding import make_generator, seed_everything
from ml_learning.harness.store import RunDir, new_run, write_status, write_summary
from ml_learning.harness.io import _write_json


class BuiltRun(NamedTuple):
    """Everything `loop.train` needs that comes out of a config plus a device.

    `model` and `runnable` are two names for one set of weights, and the
    distinction is load-bearing rather than stylistic — see `build_run`.
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
    """Seed, build, compile, and derive the optimiser and schedule — in that order.

    Every line here is ordered against the one before it, and getting any of them
    wrong produces a run that trains fine and means something different:

    - **Seed before `build_model`.** Seeding afterwards seeds nothing that
      matters: the initial weights are already drawn, so the run isn't
      reproducible and nothing says so.
    - **`runnable` is a second name, not a replacement.** A checkpoint written
      from a compiled wrapper has `_orig_mod.`-prefixed keys and won't load into
      a plain model, so callers checkpoint and sample from `model` and step
      `runnable`. With compile off they're the same object.
    - **Optimiser from `model`, not `runnable`.** The parameters are the same
      tensors either way, and this keeps `param_groups` independent of a backend.
    - **Schedule after the optimiser.** It's a function of the lr the optimiser
      was built with: `optim.lr` is the *peak*, and the schedule scales it.

    Returned as a NamedTuple rather than a tuple so a caller that only wants the
    model doesn't have to remember which position it's in.
    """
    seed_everything(cfg.runtime.seed)
    model = build_model(cfg, vocab_size, ARCHES).to(device)
    runnable = compile_model(model, cfg.runtime.compile)
    optimiser = build_optimiser(cfg, model)
    return BuiltRun(model, runnable, optimiser, build_lr_schedule(cfg))


def draw_eval_batches(data, cfg: RunConfig, device: torch.device | str) -> dict:
    """The fixed evaluation set, drawn once from its own generator.

    The generator is the whole point: `runtime.eval_seed` is separate from
    `runtime.seed`, so validation loss is measured against the same batches
    regardless of what training's RNG did, and two runs are comparable. Draw
    these from the training generator and every config change silently moves the
    yardstick as well as the thing being measured.

    Drawn once rather than per eval for the same reason — and because it's a
    property of the dataset, not of the step.
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
    """What a checkpoint contains, in one place because it's written from two.

    `vocab_size` and `config` ride along because a bare `state_dict` isn't
    loadable on its own: rebuilding the module to load it into needs the config
    that shaped it and the vocabulary that sized its embedding. `store.py` hands
    out the path and never learns what goes in it, which is what keeps it
    torch-free for the micrograd side.

    Pass `model`, never a compiled wrapper — see `build_run`.
    """
    return {
        "model": model.state_dict(),
        "optim": optimiser.state_dict(),
        "step": step,
        "vocab_size": vocab_size,
        "config": asdict(cfg),
    }


class RunRecord:
    """The handle `record_run` yields, for the caller to report its result into.

    Exists because the thing the `finally` block needs — which step training
    reached — is produced *inside* the `with`, and a context manager can't return
    a value out of its own body. Defaulting `result` to `{}` is what makes the
    crash-before-the-first-step case work: `train()` never returned, so there is
    no final step, and `-1` is the honest answer rather than a `NameError`
    raised from inside cleanup.
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
) -> Iterator[tuple[RunDir, RunRecord]]:
    """Open a run directory, then close it out whatever happens inside.

    The reason this is a context manager and not two functions: **the exit path
    is the part worth sharing.** A run that crashes at step 40 or takes a Ctrl-C
    still has to leave a complete, readable record — status, summary, and a
    `last` checkpoint — and the three ways out (returned, raised, interrupted)
    have to converge on that without the caller remembering to. Written as a
    `try`/`finally` in each entry point, it's a paragraph of subtle control flow
    duplicated per project.

    The distinction between the two non-happy exits is deliberate:

    - **KeyboardInterrupt is swallowed**, recorded as `interrupted`. Stopping a
      run you're watching is a normal way to end one, not a failure, and the
      record should say which. (`sweep.py` depends on this: it reads
      `status.json` back precisely *because* `run()` returns normally here.)
    - **Every other exception is recorded and re-raised.** A bug should still
      reach the terminal with its traceback; the record just isn't lost on the
      way.

    Ordering inside: the directory appears *before* training starts, so a run
    killed at step 3 is still readable, and `tokenizer.json` is written up front
    for the same reason — without it a checkpoint is undecodable integers, since
    the corpus is gitignored.

    Yields `(rd, record)`. Assign the result of `loop.train` to `record.result`;
    everything else is handled.
    """
    vocab_size = data.tokenizer.vocab_size

    rd = new_run(
        runs_root,
        asdict(cfg),
        collect_provenance(
            str(device), vocab_size, repo_root, data.sha256, data.tokenizer, asdict(cfg)
        ),
    )
    # Written here rather than inside `new_run` for the same reason the
    # checkpoints are: store.py hands out the path and stays ignorant of the
    # contents.
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
        # Reads the metrics file back, which is why the caller's MetricLogger has
        # to be closed by now — i.e. its `with` belongs *inside* this one.
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
