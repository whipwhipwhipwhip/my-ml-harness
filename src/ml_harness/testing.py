"""A complete, minimal project to run the harness against.

Two jobs, and they turn out to be the same job:

1. **The harness's own tests need something to drive.** Every table the harness
   takes as a parameter (`ARCHES`, `DATAS`) has to be filled by *somebody*, and
   filling them from a real project would make the harness's test suite depend on
   a GPT — which is exactly the arrow the layering rule forbids.
2. **Anyone wiring a new model in needs an example.** This is the smallest
   complete answer to "what do I have to supply?": a `ModelConfig` subclass, an
   `nn.Module` taking `(cfg, vocab_size)`, a table entry, a `load_data`, and the
   `run()` at the bottom that puts them in the right order.

Shipped rather than left in `tests/` for the second reason. It imports no pytest
and defines no fixtures, so it costs a downstream user nothing to have on the
path — the fixtures that use it live in `tests/ml_harness/conftest.py`.

`TinyModel` is a real transformer block rather than a bare embedding: the
harness's checkpoint, compile and sampling paths all have to survive contact with
parameter sharing, multi-head reshapes and a generation loop, and a model too
simple to have those isn't exercising them. It is still far too small to learn
anything, which is the point — nothing here asserts on loss going down.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizers import CharTokenizer, Tokenizer, fingerprint

from ml_harness.cache import cache_path, load_tokens, save_tokens
from ml_harness.config import DataConfig, FileDataConfig, ModelConfig
from ml_harness.device import autocast_context, resolve_device, set_matmul_precision
from ml_harness.loop import train
from ml_harness.metrics import MetricLogger
from ml_harness.registry import ArchSpec, build_tokenizer, load_run_config
from ml_harness.runner import (
    build_run,
    checkpoint_payload,
    draw_eval_batches,
    record_run,
)
from ml_harness.store import RunDir

# ---- the model ------------------------------------------------------------


@dataclass(frozen=True)
class TinyConfig(ModelConfig):
    """The reference architecture's hyperparameters.

    Field names deliberately match the project GPTs', so a config written for one
    is readable as the other and `--set model.n_layer=2` means the same thing in
    both. `vocab_size` is absent for the usual reason: it's derived from the data,
    not chosen, so whoever loaded the corpus supplies it.
    """

    n_embd: int = 16
    n_head: int = 2
    n_layer: int = 1
    block_size: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        # The same guard every multi-head model needs, and worth having here
        # rather than only in the projects: it's what makes "a bad config raises
        # at load, next to the value that caused it" testable against the
        # reference arch instead of against a GPT.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_size(self) -> int:
        """Derived, not stored — a stored copy can disagree with n_embd // n_head."""
        return self.n_embd // self.n_head


class TinyBlock(nn.Module):
    """Pre-norm self-attention + MLP. One block, no frills."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        # The multi-head reshape the harness's compile and checkpoint paths have
        # to survive: (B, T, C) -> (B, heads, T, head_size).
        q, k, v = (
            t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for t in (q, k, v)
        )
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(att)
        return x + self.mlp(self.ln2(x))


class TinyModel(nn.Module):
    """The smallest thing that is still a language model.

    Takes `(cfg, vocab_size)` — the signature every entry in an `ARCHES` table
    has to have, and the one thing `ml_ml_harness/registry.py` assumes about a model.
    """

    def __init__(self, cfg: TinyConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(TinyBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, top_k: int | None = None):
        """Autoregressive sampling, cropped to the context window.

        Present because the harness writes a sample at every eval, so an arch
        that can't generate can't exercise that path at all.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :]
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


# The two tables an entry point hands to the harness. One entry each: this is a
# reference, not a menu.
ARCHES: dict[str, ArchSpec] = {"tiny": ArchSpec(TinyConfig, TinyModel)}
DATAS: dict[str, type[DataConfig]] = {"file": FileDataConfig}


# ---- the data -------------------------------------------------------------


@dataclass(frozen=True)
class TinyData:
    """The duck type `loop.py` and `runner.py` read a dataset through.

    Four things, and the harness never asks for a fifth: two splits it can batch
    from, the tokenizer that made them, and the corpus hash for provenance.
    """

    train: torch.Tensor
    val: torch.Tensor
    tokenizer: Tokenizer
    sha256: str

    @property
    def splits(self) -> dict[str, torch.Tensor]:
        return {"train": self.train, "val": self.val}

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str,
        n_iters: int = 1,
        generator: torch.Generator | None = None,
    ):
        try:
            data = self.splits[split]
        except KeyError:
            raise ValueError(
                f"unknown split {split!r}; expected one of {sorted(self.splits)}"
            ) from None

        batches = []
        for _ in range(n_iters):
            ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
            x = torch.stack([data[i : i + block_size] for i in ix])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
            batches.append((x.to(device), y.to(device)))
        return batches


def load_data(
    path,
    split_frac,
    build_tokenizer=CharTokenizer.from_text,
    cache_dir=None,
) -> TinyData:
    """Read a corpus, tokenize it, split it by position.

    `build_tokenizer` is a callable taking the corpus text, not a config — that's
    what keeps this module ignorant of the registry, and it's why an entry point
    passes `partial(registry.build_tokenizer, cfg)`. Some tokenizers are built
    *from* the corpus, and this function is what has the text.

    `cache_dir=None` disables the token cache, and defaults to off here on
    purpose: a caller that hasn't thought about where the cache goes must not end
    up writing to a developer's real one.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    # Built before the cache is consulted, because the tokenizer's fingerprint is
    # half the cache key — so this memoises *encoding*, never construction.
    tok = build_tokenizer(text)
    tokens = _encode_cached(text, tok, sha256, path, cache_dir)

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(split_frac * len(data))
    return TinyData(data[:n], data[n:], tok, sha256)


def _encode_cached(text, tok: Tokenizer, data_sha256: str, path, cache_dir):
    """`tok.encode(text)`, memoised on disk by (corpus, tokenizer).

    A miss and a disabled cache take the same path, so a cache bug can change how
    long a run waits but never what it trains on.
    """
    if cache_dir is None:
        return tok.encode(text)

    cached_at = cache_path(cache_dir, data_sha256, fingerprint(tok))

    tokens = load_tokens(cached_at)
    if tokens is not None:
        return tokens

    tokens = tok.encode(text)
    save_tokens(
        cached_at,
        tokens,
        {
            "data_path": str(path),
            "data_sha256": data_sha256,
            "tokenizer_sha256": fingerprint(tok),
            "tokenizer": tok.to_dict(),
        },
    )
    return tokens


# ---- the collaborators the loop talks to ----------------------------------
#
# `loop.train` reaches the outside world through exactly two objects, and these
# are complete implementations of both. That they can be this small is the
# property worth protecting: if a fake here has to grow a filesystem or a real
# model, the loop has acquired a dependency it shouldn't have.


class FakeLogger:
    """A `logger.log(...)` that keeps the records instead of writing them."""

    def __init__(self):
        self.records = []

    def log(self, step, split, metric, value, **extra):
        self.records.append((step, split, metric, value))

    def steps_for(self, split, metric):
        return [s for s, sp, m, _ in self.records if sp == split and m == metric]


class SpyEval:
    """An `on_eval(step, losses)` that records what it saw, and when.

    `logged_at_call` is how the ordering guarantee gets tested: the loop must
    have finished logging a step's metrics before it calls back, because a
    checkpoint callback that reads the metrics file relies on it.
    """

    def __init__(self, logger=None):
        self.calls = []
        self.logged_at_call = []
        self._logger = logger

    def __call__(self, step, losses):
        self.calls.append((step, losses))
        if self._logger is not None:
            self.logged_at_call.append(len(self._logger.records))


# ---- the entry point ------------------------------------------------------
#
# The last thing a project has to supply, and the one the harness can't: the
# order the pieces go together in. `runner.py` owns the two halves that name no
# model (`build_run`'s four silent ordering constraints, and `record_run`'s
# lifecycle), so what's left here is what genuinely varies per project — loading
# the data, and what to do at each eval.
#
# Shipped for the same reason as the rest of this module: it is the worked answer
# to "what does an entry point look like", and it is what the harness's own tests
# drive when the subject is the record rather than the loop. A real project's
# `train.py` adds argparse, its own paths and its own smoke overrides on top of
# this shape; none of those belong here, because the harness resolves no path it
# wasn't given.


class EvalRecorder:
    """The `on_eval` callback: console, best checkpoint, samples. No metrics.

    Owns `best_val` because it owns the best checkpoint — if the loop tracked the
    value and this decided the write, one decision would live in two places. The
    loop has already logged the step's metrics by the time this is called, which
    is what lets a callback read the metrics file back.
    """

    def __init__(self, rd, model, optimiser, data, cfg, device, echo=None):
        self.rd = rd
        self.model = model
        self.optimiser = optimiser
        self.data = data
        self.cfg = cfg
        self.device = device
        self.echo = echo
        self.best_val = float("inf")

    def __call__(self, step: int, losses: dict[str, float]) -> None:
        if self.echo is not None:
            self.echo(f"step {step:5d} | " + " | ".join(
                f"{split} {value:.4f}" for split, value in sorted(losses.items())
            ))

        if losses["val"] < self.best_val:
            self.best_val = losses["val"]
            torch.save(self._checkpoint(step), self.rd.ckpt_path("best"))

        if self.cfg.train.write_samples:
            self._write_sample(step, losses["val"])

    def _checkpoint(self, step: int) -> dict:
        # The same payload the `last` checkpoint gets, from one definition — a
        # `best` that couldn't be loaded the same way as a `last` would be a
        # nasty thing to discover.
        return checkpoint_payload(
            self.model, self.optimiser, step, self.data.tokenizer.vocab_size, self.cfg
        )

    def _write_sample(self, step: int, val_loss: float) -> None:
        # The prompt is text and gets encoded here rather than being a hardcoded
        # id: id 0 is '\n' under a character vocabulary but byte 0x00 under BPE.
        # `or [0]` — an empty prompt means "unconditional", but generate() needs
        # at least one token to attend to.
        prompt = self.data.tokenizer.encode(self.cfg.train.sample_prompt)
        context = torch.tensor([prompt or [0]], dtype=torch.long, device=self.device)
        # eval() for the same reason `evaluate` does it: on_eval runs after the
        # loop has put the model back in train mode, so without this every sample
        # is drawn with dropout active. Restore what we found, in `finally`, so a
        # failure to generate doesn't leave training running with dropout off.
        was_training = self.model.training
        self.model.eval()
        try:
            tokens = self.model.generate(
                context, max_new_tokens=64, top_k=self.cfg.train.top_k
            )[0].tolist()
        finally:
            self.model.train(was_training)
        (self.rd.samples_dir / f"step_{step:06d}.txt").write_text(
            f"=== step {step} (val {val_loss:.4f}) ===\n"
            f"{self.data.tokenizer.decode(tokens)}\n"
        )


def run(
    config_path,
    overrides: Sequence[str] = (),
    runs_root=None,
    *,
    cache_dir=None,
    repo_root=None,
    smoke: bool = False,
    echo=None,
) -> RunDir:
    """Parse -> build -> train -> persist. Returns the RunDir.

    The three leading positional parameters are the `sweep.RunFn` contract
    (`(config_path, overrides, runs_root) -> RunDir`), so this can be handed
    straight to `run_sweep`; everything a project would vary is keyword-only
    after them.

    `runs_root` has no default on purpose — the harness resolves no path it
    wasn't given, and a default here would be an assertion about somebody's tree.
    `cache_dir=None` disables the token cache, which is the right default for a
    caller that hasn't decided where the cache goes: a test must never write to a
    developer's real one.

    `smoke` only *labels* the record. Shrinking the run and diverting it to a
    subdirectory stay with the project, because the overrides that shrink a model
    name that model's fields — see `parser.parse_args`' `--smoke`, which is the
    flag a project's `main()` turns into its own override list.
    """
    if runs_root is None:
        raise ValueError(
            "runs_root is required: the harness resolves no path it wasn't given"
        )

    # Into the loader rather than applied afterwards — the config dataclasses are
    # frozen, so an override against a built RunConfig raises FrozenInstanceError.
    cfg = load_run_config(config_path, ARCHES, DATAS, overrides)

    device = resolve_device(cfg.runtime.device)
    # Unconditionally, not only when it differs from the default: this is process
    # state, and a sweep runs many points in one process.
    set_matmul_precision(cfg.runtime.matmul_precision)
    # Built *now* rather than at the call site, so a bad dtype fails before
    # `new_run` has put a directory on disk. One instance per run; it's
    # re-enterable.
    amp = autocast_context(device, cfg.runtime.autocast_dtype)

    data = load_data(
        cfg.data.path,
        cfg.data.split_frac,
        # A callable rather than a config, because some tokenizers are built
        # *from* the corpus and `load_data` is what has the text.
        partial(build_tokenizer, cfg),
        # Two switches meaning different things: `cache_dir` is where a caller
        # may write, `data.cache` is whether this run wants caching at all.
        cache_dir if cfg.data.cache else None,
    )
    # Seed, build, compile, optimiser, schedule — four ordering constraints, all
    # of which fail silently. See `build_run`.
    built = build_run(cfg, data.tokenizer.vocab_size, ARCHES, device)
    eval_batches = draw_eval_batches(data, cfg, device)

    with record_run(
        runs_root,
        cfg,
        data,
        built,
        device=device,
        # The harness only reads this to shell out to git for provenance; the
        # config's own directory is the honest answer when nobody said otherwise.
        repo_root=repo_root if repo_root is not None else Path(config_path).parent,
        smoke=smoke,
    ) as (rd, record):
        # `with` inside, so the logger is closed before record_run's summarize()
        # reads the file back.
        with MetricLogger(rd.metrics_path) as logger:
            # `built.model`, not `built.runnable`: the recorder checkpoints and
            # samples, and both want the uncompiled module.
            recorder = EvalRecorder(
                rd, built.model, built.optimiser, data, cfg, device, echo
            )
            record.result = train(
                built.runnable,
                built.optimiser,
                data,
                cfg,
                device,
                eval_batches,
                logger,
                recorder,
                amp,
                built.lr_schedule,
            )

    return rd
