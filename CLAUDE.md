# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone extraction of the experiment harness that grew inside `~/code/learning_ml`
(`src/ml_learning/harness/`). It is generic training infrastructure — config loading, a
training loop, run records, sweeps, benchmarking, analysis — that names **no model and no
task**. Downstream projects supply those.

**The extraction is complete and the suite is green** (238 tests, ~2.5s).
`src/ml_harness/` and `tests/ml_harness/` began as byte-for-byte copies of
`learning_ml`'s `src/ml_learning/harness/` and `tests/harness/`; what changed since:

- Imports renamed to `ml_harness.*` / `tests.ml_harness.*`, and
  `provenance.py::HARNESS_DISTRIBUTION` to `"ml_harness"` so a run records the version of
  the distribution that actually shipped the harness (it was silently recording `None`).
- Dependencies cut from 16 to 5 — numpy, pandas, torch, dacite, tokenizers — with pytest in
  a `dev` group. `[tool.uv.sources]` pins `tokenizers` to the local package; HuggingFace's
  PyPI package of the same name is **not** a substitute, since the code wants
  `CharTokenizer`, `ByteTokenizer`, `RegexTokenizer`, `TiktokenBPETokenizer`,
  `NO_SPLIT_PATTERN` and `fingerprint`.
- `testing.py` gained `run()` and `EvalRecorder` — the reference entry point. This is the
  change to know about: the tests that assert on the *record* (autocast dtypes, the compile
  checkpoint hazard, matmul precision, a real sweep) previously drove
  `nanogpt_conor.train.run`, and now drive a `run()` the package ships. It completes what
  `testing.py` already claimed to be — the worked answer to "what must I supply" — and keeps
  the harness able to test its own record lifecycle with no project present.
- Deleted: `test_run_record.py` and `test_data_config.py` (both wholly about a project's
  entry point and two projects' `DATAS` tables), and the two `test_shipped_config_parses` /
  `test_the_shipped_sweep_parses` cases, whose subject was a config file in the origin repo
  that has no counterpart here. `conftest.py`'s `gpt_config_path` fixture went with them.
- Docstrings that were about the old home were corrected where they had become *false*
  (`io.py` on what it was copied from, `provenance.py` on a move that has now happened).
  Mentions of `nanogpt_conor` / `gpt_addition` as the two projects that motivated an
  abstraction were left alone — that history is why the seams are where they are.

Still worth knowing: this package ships no `configs/` directory, so nothing tests a real
config file end to end; and `parse_args`' `--smoke` has no shipped override table, because
the overrides that shrink a model name that model's fields. Both are a project's business.

`~/code/learning_ml/CLAUDE.md` documents the harness in its original home and is the
reference for anything not covered here.

## Setup & commands

Dependency management is `uv`; the package builds with `uv_build` from `src/`, so
`src/ml_harness/` imports as `ml_harness`.

```bash
uv sync                                   # create/sync .venv
uv run pytest                             # full suite
uv run pytest tests/ml_harness/test_loop.py            # one module
uv run pytest tests/ml_harness/test_loop.py -k schedule # one test / pattern
```

`uv sync` installs the `dev` group too, so `pytest` comes with it; it is not importable
from the ambient python.

Gitignored, and therefore invisible: `docs/`, `runs/`, `notebooks/`, `.cache/`, `skills/`,
and **all `*.txt`** — which is why every run records a data `sha256`.

## Architecture

**The layering rule, and the whole point of the package: projects depend on the harness,
never the reverse.** `ml_harness` never names a model or a task. The mechanism is that
`ARCHES` and `DATAS` are *passed in* as parameters (`load_run_config(path, ARCHES, DATAS)`,
`build_model(cfg, vocab_size, ARCHES)`), never imported. `OPTIMS`, `TOKENIZERS` and
`SCHEDULES` are the deliberate asymmetries — torch optimisers, the `tokenizers` package and
cosine arithmetic are not project-specific, so those tables live in `registry.py`. The test
is "does this module have to name a *model* or a *task*". Likewise `io.py` is copied from
its origin rather than imported, so the harness depends on nothing outside its own
directory plus `tokenizers`; don't reintroduce such an import.

- [config.py](src/ml_harness/config.py) — nested frozen dataclasses composing into
  `RunConfig` (`arch`, `model`, `optim_arch`, `optim`, `tokenizer_arch`, `tokenizer`,
  `train`, `data_arch`, `data`, `runtime`, `schedule_arch`, `schedule`, `note`). Torch-free.
  `ModelConfig`, `TokenizerConfig`, `DataConfig` and `ScheduleConfig` are empty bases that
  concrete configs subclass — empty because no field is common to every member (a generated
  corpus has no `path`; "hold lr fixed" and "cosine to a floor" share no parameter). New
  fields must be defaulted so already-recorded runs still load under `strict=True`.
- [registry.py](src/ml_harness/registry.py) — `arch`, `optim_arch`, `tokenizer_arch`,
  `schedule_arch` and `data_arch` are **discriminators**: read the tag, *then* parse the
  matching block into the class it names. That two-stage read is why a
  `GPTConfig | BigramConfig` union can't work — dacite would take the first member that
  fits and ignore the tag. dacite runs `strict=True`, so a typo'd key is a load error rather
  than a silently ignored hyperparameter. `DATAS` maps straight to a config class rather
  than to a `(config_cls, build)` spec pair: the tag doesn't dispatch, because an entry
  point already imports the one `load_data` it uses.
- [loop.py](src/ml_harness/loop.py) — `train` / `evaluate`. **No I/O, no filesystem, no
  argparse**; the import list is `torch` and `RunConfig`. If `store`, `os` or `datetime`
  appears here something has leaked downward. It reaches the outside world through exactly
  two duck-typed collaborators: `logger.log(step, split, metric, value)` (the loop decides
  what counts as a metric) and `on_eval(step, losses)` (the caller decides whether to
  checkpoint, sample or print).
- [runner.py](src/ml_harness/runner.py) — the parts of an entry point that name no model:
  `build_run(cfg, vocab_size, ARCHES, device) -> BuiltRun(model, runnable, optimiser,
  lr_schedule)`, and `record_run(...)`, a context manager that opens the run directory and
  closes it out — status, summary, `last` checkpoint — however the block exits.
  **KeyboardInterrupt is swallowed and recorded as `interrupted`** (which is what lets
  `sweep.py` detect a Ctrl-C by reading `status.json` back); every other exception is
  recorded and re-raised. Loading the data deliberately stayed in entry points — the
  harness only reads a dataset through `get_batch`/`splits`/`tokenizer`/`sha256`.
- [store.py](src/ml_harness/store.py) — `RunDir` (every canonical path as a property),
  `new_run`, `write_status`, `write_summary`, and `load_runs`/`load_metrics` returning
  pandas frames. **Torch-free on purpose**: callers `torch.save` to a path `RunDir` hands
  them, so store.py never learns what a checkpoint contains.
- [sweep.py](src/ml_harness/sweep.py) — **a sweep point is a list of `--set` strings**, so a
  grid is validated by exactly the path a hand-typed config value is, and this module needs
  no schema for what it varies. Precedence: smoke < constants < grid point < CLI `--set`.
  `seeds: 3` is sugar for a `runtime.seed` axis, expanded **last/innermost** so a killed
  sweep leaves complete groups. `run_fn` is injected like `ARCHES`; the contract is
  `(config_path, overrides, runs_root) -> RunDir`. A failed point is recorded and the grid
  continues **unless it's the first** (that's a spec problem). `sweep.json` is rewritten
  after every point.
- [provenance.py](src/ml_harness/provenance.py) — facts about the world (git SHA, dirty
  patch, torch/tokenizers/harness versions, resolved device, data hash, tokenizer kind and
  fingerprint). Also hashes the effective config: `config_sha256` identifies *this run*,
  `group_sha256` identifies *the experiment* — the config with `GROUP_EXCLUDES`
  (`runtime.seed`, `note`) stripped — so N seeds of one config share a column and
  "mean ± spread" is a groupby. `runtime.eval_seed` and `runtime.device` are deliberately
  **not** excluded: different eval batches or different hardware are different measurements.
  The hashes are provenance fields, never part of `run_id`, which stays timestamp + uuid so
  re-running a config never collides.
- [cache.py](src/ml_harness/cache.py) — content-addressed cache for an encoded corpus, keyed
  on `(data_sha256, tokenizer_sha256)`, so there is no invalidation logic to get wrong. Any
  read failure degrades to a miss. **`CACHE_VERSION` is the hole the key can't cover** — the
  hashes describe encoding's *inputs*, not the code doing it; bump it when encode behaviour
  changes.
- [parser.py](src/ml_harness/parser.py) — **holds no paths.** It used to derive a repo root
  from `__file__` and hang `configs/`/`runs/` off it, which is true only for an editable
  checkout and silently resolves into `site-packages/` otherwise. Defaults now arrive as
  keyword arguments, and a flag with no project-supplied default is `required` rather than
  quietly wrong. `test_parser.py` asserts the *absence* of those constants. House rule: **the
  harness resolves no path it wasn't given.**
- [device.py](src/ml_harness/device.py) — `resolve_device`, `set_matmul_precision` (process
  state, so callers must set it on *every* run — a sweep drives many points in one process),
  `autocast_context` (bf16 only; fp16 would need a GradScaler this harness doesn't have),
  `compile_model`.
- [analysis.py](src/ml_harness/analysis.py) — `varying` (the config columns that actually
  differ across a directory of runs — an ablation *is* a config diff), `run_table`,
  `group_key`, `seed_groups`, `curves`. Frames in, frames out, torch-free. It lives in a
  module rather than a notebook because `sweep.py` needs the same grouping.
- [testing.py](src/ml_harness/testing.py) — a complete, minimal *project* to run the harness
  against: `TinyConfig`/`TinyModel` (a real one-block transformer — small enough to be
  pointless, complete enough to checkpoint, compile and generate from), `ARCHES`, `DATAS`, a
  `load_data`, `EvalRecorder`/`run()` (the reference entry point — parse, build, train,
  persist, matching `sweep.RunFn`'s `(config_path, overrides, runs_root) -> RunDir`), and
  the `FakeLogger`/`SpyEval` collaborators. Shipped rather than left in `tests/` because it
  is also the worked answer to "what must I supply to wire in a model". It imports no pytest
  and defines no fixtures. **`run()` is the reference, not a default** — a real project's
  `train.py` adds argparse, its own paths and its own smoke overrides on top of this shape.
- [metrics.py](src/ml_harness/metrics.py) — `MetricLogger` appends line-buffered JSONL;
  `read_rows`/`read_metrics` tolerate the truncated final line a killed run leaves;
  `summarize` reads it back into final/min/max per `(split, metric)`.

A run writes `<runs-root>/<timestamp>_<uuid6>/`: `config.json`, `provenance.json`,
`metrics.jsonl`, `summary.json`, `status.json`, `tokenizer.json`,
`checkpoints/{best,last}.pt`, `samples/`, `logs/`. One JSON config fully describes a run, so
an ablation is a config diff. `tokenizer.json` is the tokenizer's `to_dict` recipe — without
it a checkpoint is undecodable integers, since corpora are gitignored.

### Ordering constraints that fail silently

Each of these produces a run that trains fine and means something different:

- **Seed before `build_model`** — seeding after the weights are drawn makes nothing
  reproducible and says so nowhere.
- **`runnable` is a second name for `model`, not a replacement** — a checkpoint written from
  a `torch.compile` wrapper has `_orig_mod.`-prefixed keys and won't load into a plain
  model. Step `runnable`; checkpoint and sample from `model`.
- **Build the optimiser from `model`, the schedule after the optimiser** — the schedule
  scales `optim.lr`, which is the *peak*.
- **Eval batches are drawn once, from a separate `eval_seed` generator** — otherwise every
  config change moves the yardstick as well as the thing being measured.
- **Autocast wraps the forward pass and the loss only** — `backward()` and `step()` go
  outside it, so the master weights stay fp32.
- **`--set` overrides apply to the raw dict before dacite**, never to a built `RunConfig`
  (frozen: it would raise `FrozenInstanceError`). Going through the dict is also what makes
  an override validated by exactly the path a hand-typed JSON value is.

### Extending

- **An architecture**: a `ModelConfig` subclass, an `nn.Module` taking `(cfg, vocab_size)`,
  and an entry in the project's `ARCHES`. No edit to the loop or the entry point.
- **An optimiser**: an `OptimSpec` in `OPTIMS` (`_adapt` wraps a torch optimiser class whose
  hyperparameters map 1:1; `_build_adamw` is the exception, since torch takes one `betas`
  tuple and the config keeps two addressable scalars).
- **A tokenizer**: a `TokenizerConfig` subclass, a `(cfg, text) -> Tokenizer` builder, an
  entry in `TOKENIZERS`. The corpus is passed even to builders that ignore it, so callers
  never have to know which kind they built.
- **A schedule**: a `ScheduleConfig` subclass and a `(cfg, peak_lr, max_iters) -> (step) ->
  lr` builder in `SCHEDULES`. Everything resolves at build time so the loop holds a closure,
  not a config to interpret.
- **A dataset**: a `DataConfig` subclass, an entry in the project's `DATAS`, and a
  `load_data` the entry point imports.

## Testing

Three tiers, cheapest first — keep a new test in the cheapest tier that can answer the
question:

- Pure config/cache/frame work, milliseconds: `test_overrides.py`,
  `test_tokenizer_config.py`, `test_token_cache.py`, `test_config_hash.py`,
  `test_analysis.py`, `test_parser.py`, `test_schedule.py`.
- The loop against a nine-line `StubModel` (deliberately *not* `testing.TinyModel` — the
  claim is that the loop's contract is narrow enough for a stub to satisfy), and the sweep
  driver against a fake `run_fn` that writes only a `status.json`: `test_loop.py`,
  `test_sweep.py`, `test_bench.py`. **If a loop test starts needing a tmpdir or a real
  model, that module has grown a dependency it shouldn't have.**
- `testing.run()` into `tmp_path`, asserting on the *record* rather than on learning: the
  entry-point sections of `test_autocast.py`, `test_compile.py`, `test_matmul_precision.py`,
  and `test_sweep.py`'s one real sweep. Most of the suite's wall time.

**The portability rule, and it is mechanical: every test drives `ml_harness` and nothing
else.** `config_path` is the only config fixture and it builds the reference arch. That rule
is what made this extraction possible in the first place — with the two projects made
unimportable, 227 of 240 tests already passed before a line was rewritten.

`conftest.py` synthesises both the corpus and the config into `tmp_path`, so the suite reads
nothing real — no `runs/`, no corpus file, no shipped config. Note `data.cache: false` in the
fixture: a test must never write to a developer's real `.cache/`.
