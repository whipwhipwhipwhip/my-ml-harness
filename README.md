## ML harness

This is the training infrastructure I use for my GPT experiments: config loading, a training loop, run records, sweeps, benchmarking and some analysis helpers.

It grew inside a private repo I had for learning RL/LLM stuff, and got split out when that repo became too sprawling (the [zero-to-hero](https://github.com/whipwhipwhipwhip/zero-to-hero) and [nanotok](https://github.com/whipwhipwhipwhip/nanotok) repos came out of the same split). Both of my GPT projects (a nanoGPT-style model, and a small model that learns addition) ended up with the same ~85 lines of setup and bookkeeping, so that code moved here.

The main rule is that **the harness never names a model or a task**. Projects depend on the harness, not the other way round. A project passes in its own tables of models (`ARCHES`) and datasets (`DATAS`), and the harness looks things up in those. Optimisers, tokenizers and learning-rate schedules aren't specific to any project, so their tables live here.

### What a run looks like

One JSON file describes a run completely, so an ablation is just a config diff:

```json
{
  "arch": "tiny",
  "model": {"n_embd": 16, "n_head": 2, "n_layer": 1, "block_size": 8, "dropout": 0.0},
  "optim_arch": "adamw",
  "optim": {"lr": 1e-3},
  "tokenizer_arch": "char",
  "tokenizer": {},
  "train": {"max_iters": 2000, "batch_size": 32, "eval_interval": 100, "eval_iters": 20},
  "data_arch": "file",
  "data": {"path": "/path/to/input.txt", "split_frac": 0.9},
  "runtime": {"seed": 0, "eval_seed": 1, "device": "auto"},
  "note": "baseline"
}
```

Each `*_arch` tag picks which dataclass the block next to it is parsed into, and parsing is strict, so a typo'd key is an error rather than a setting that gets silently ignored. Any value can be overridden from the command line with `--set optim.lr=3e-4`. Overrides are validated exactly like values in the file.

Each run writes a directory `runs/<timestamp>_<id>/` containing:

- `config.json`: the config after overrides.
- `provenance.json`: git SHA (plus a patch if the tree was dirty), package versions, device, hostname, a sha256 of the corpus, and the tokenizer's fingerprint. My corpora are gitignored, so the data hash is the only record of what a model was trained on.
- `metrics.jsonl`, `summary.json`, `status.json`: what happened, and whether the run completed, failed or was interrupted.
- `tokenizer.json`: enough to rebuild the tokenizer. Without it the checkpoints are just integers.
- `checkpoints/{best,last}.pt` and `samples/`.

The run directory is created before training starts, and gets closed out properly however the run ends. A Ctrl-C at step 40 still leaves a readable record with a `last` checkpoint.

### What's here

Everything is in [src/ml_harness](src/ml_harness):

- [config.py](src/ml_harness/config.py) - the config dataclasses. `RunConfig` is made of blocks (`model`, `optim`, `tokenizer`, `train`, `data`, `runtime`, `schedule`). New fields always get a default, so older run records still load.
- [registry.py](src/ml_harness/registry.py) - turns a config file into a `RunConfig`, applies `--set` overrides, and builds the model, optimiser (AdamW, with GPT-2's weight decay split), tokenizer and lr schedule it describes.
- [loop.py](src/ml_harness/loop.py) - `train` and `evaluate`. No file I/O at all: metrics go to a `logger` object, and everything else (checkpointing, sampling, printing) happens in an `on_eval` callback the caller passes in. Supports gradient accumulation, gradient clipping and bf16 autocast.
- [runner.py](src/ml_harness/runner.py) - the setup and bookkeeping any project's entry point needs. `build_run` does seed, build, compile, optimiser and schedule in the right order (getting it wrong doesn't crash, it just quietly changes what the run means). `record_run` is a context manager that opens the run directory and closes it out.
- [store.py](src/ml_harness/store.py) - run directory paths, and `load_runs` / `load_metrics` to read a directory of runs back as pandas DataFrames.
- [provenance.py](src/ml_harness/provenance.py) - collects `provenance.json`, including two config hashes: `config_sha256` identifies the exact run, and `group_sha256` identifies the experiment (the config without the seed), so averaging over seeds is a `groupby`.
- [analysis.py](src/ml_harness/analysis.py) - DataFrame helpers for notebooks: which config columns actually vary between runs, a leaderboard, mean +/- std over seeds, and loss curves.
- [sweep.py](src/ml_harness/sweep.py) - runs one config over a grid. Each point in the grid is just a list of `--set` overrides, so it's validated the same way. Seeds are the innermost axis, so a sweep that gets killed partway leaves complete groups.
- [bench.py](src/ml_harness/bench.py) - times training steps (with device syncs and warmup) without training anything, for comparing configs on speed.
- [dist.py](src/ml_harness/dist.py) - data parallelism with DDP and `torchrun`. `train.batch_size` is the global batch, so the same config is the same experiment on one GPU or eight, and the number of processes is recorded in provenance rather than the config. Everything does nothing when there's only one process.
- [device.py](src/ml_harness/device.py), [cache.py](src/ml_harness/cache.py), [schedule.py](src/ml_harness/schedule.py), [parser.py](src/ml_harness/parser.py), [metrics.py](src/ml_harness/metrics.py) - device selection and precision settings, an on-disk cache of the encoded corpus (keyed by the corpus and tokenizer hashes), constant and warmup + cosine schedules, argparse helpers, and the JSONL metric logger.
- [testing.py](src/ml_harness/testing.py) - a complete, tiny project: a one-block transformer, the `ARCHES` and `DATAS` tables, a `load_data`, and a reference `run()` entry point. The tests use it, and it's also the example to copy when wiring in a new model.

### Usage

To plug in a model, a project supplies a config dataclass, an `nn.Module` that takes `(cfg, vocab_size)`, and a table entry:

```python
from dataclasses import dataclass
from ml_harness.config import FileDataConfig, ModelConfig
from ml_harness.registry import ArchSpec

@dataclass(frozen=True)
class GPTConfig(ModelConfig):
    n_layer: int
    n_embd: int
    block_size: int

ARCHES = {"gpt": ArchSpec(GPTConfig, GPTModel)}
DATAS = {"file": FileDataConfig}
```

The model config needs a `block_size`, since the loop uses it to draw batches; nothing else about the model is assumed. The entry point then loads the config with those tables, loads its data, and runs the loop inside `record_run`. [testing.py](src/ml_harness/testing.py)'s `run()` is a complete example in under 100 lines. A real project's `train.py` adds argparse, its own default paths and its own `--smoke` overrides on top of that. The harness never works out a path by itself (where `configs/` or `runs/` are is up to the project).

A sweep is a JSON file too:

```json
{
  "name": "lr",
  "base": "configs/gpt.json",
  "constants": {"train.max_iters": 2000, "train.write_samples": false},
  "grid": {"optim.lr": [1e-4, 3e-4, 1e-3]},
  "seeds": 3
}
```

That expands to 9 runs. `run_sweep` writes them under `runs/sweeps/<timestamp>_lr/`, and `analysis.seed_groups(load_runs(...))` gives mean +/- spread per learning rate.

### Setup and tests

```bash
uv sync
uv run pytest
```

The dependencies are numpy, pandas, torch, dacite, and my tokenizer package [nanotok](https://github.com/whipwhipwhipwhip/nanotok) (pinned to a git tag in [pyproject.toml](pyproject.toml)).

The suite takes a few seconds and doesn't need a GPU or any real data: the config and a small corpus are generated into a temp directory. Most tests are pure config/maths checks, or drive the loop with a nine-line stub model and a fake logger. If a loop test ever needs a real model or a filesystem, the loop has picked up a dependency it shouldn't have. The slower tests do real runs with `testing.run` and check what ends up in the run record (dtypes under autocast, that a compiled model's checkpoint still loads, that a sweep produces readable runs). Data parallelism is tested without ever starting a process group. Whether NCCL/gloo actually move the data needs a real multi-process run, which is left to the projects that launch one.
