"""A complete, minimal project to run the harness against.

It's for two things:

1. **The harness's own tests.** The tables the harness takes as arguments
   (`ARCHES`, `DATAS`) have to come from somewhere, and taking them from a real
   project would make the harness's tests depend on that project, which is
   backwards.
2. **An example for wiring in a new model.** This is the smallest complete
   answer to "what do I have to supply?": a `ModelConfig` subclass, an
   `nn.Module` taking `(cfg, vocab_size)`, a table entry, a `load_data`, and the
   `run()` at the bottom that puts them together in the right order.

It's in the package rather than in `tests/` because of the second point. It
doesn't import pytest or define fixtures, so it's harmless to have installed.
The fixtures that use it are in `tests/ml_harness/conftest.py`.

`TinyModel` is a real transformer block rather than just an embedding, because
the harness's checkpoint, compile and sampling code should be tested against a
model with multi-head reshapes and a generation loop. It's still far too small
to learn anything, and nothing here checks that the loss goes down.
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

    Field names match the GPT projects', so `--set model.n_layer=2` means the
    same thing in both. There's no `vocab_size`, since that comes from the data
    and is passed in by whoever loaded the corpus.
    """

    n_embd: int = 16
    n_head: int = 2
    n_layer: int = 1
    block_size: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        # The usual multi-head check. Having it here means "a bad config raises
        # at load time" can be tested against this model rather than a GPT.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_size(self) -> int:
        """Computed rather than stored, so it can't disagree with n_embd // n_head."""
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
        # The multi-head reshape: (B, T, C) -> (B, heads, T, head_size).
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

    Takes `(cfg, vocab_size)`, which is the signature every entry in an
    `ARCHES` table needs, and the only thing `registry.py` assumes about a model.
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

        Needed because the harness writes a sample at every eval.
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


# The two tables an entry point passes to the harness. One entry each, since
# this is just an example.
ARCHES: dict[str, ArchSpec] = {"tiny": ArchSpec(TinyConfig, TinyModel)}
DATAS: dict[str, type[DataConfig]] = {"file": FileDataConfig}


# ---- the data -------------------------------------------------------------


@dataclass(frozen=True)
class TinyData:
    """The interface `loop.py` and `runner.py` use to read a dataset.

    The harness only needs four things: splits it can draw batches from (via
    `get_batch`), the tokenizer that made them, and the corpus hash for
    provenance.
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

    `build_tokenizer` is a function that takes the corpus text, not a config,
    which is why an entry point passes `partial(registry.build_tokenizer, cfg)`.
    Some tokenizers are built *from* the corpus, and this function is where the
    text is.

    `cache_dir=None` turns the token cache off, and it's off by default so that
    a caller that hasn't chosen a cache location (like a test) never writes to
    a developer's real one.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    # Built before checking the cache, because the tokenizer's fingerprint is
    # half the cache key. So only *encoding* is cached, not building the tokenizer.
    tok = build_tokenizer(text)
    tokens = _encode_cached(text, tok, sha256, path, cache_dir)

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(split_frac * len(data))
    return TinyData(data[:n], data[n:], tok, sha256)


def _encode_cached(text, tok: Tokenizer, data_sha256: str, path, cache_dir):
    """`tok.encode(text)`, cached on disk by (corpus, tokenizer).

    A miss and a disabled cache both just call `encode`, so a cache bug can
    change how long a run takes but not what it trains on.
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
# `loop.train` only talks to the outside world through two objects, and these
# are complete implementations of both. It's worth keeping them this small: if
# a fake here ever needs a filesystem or a real model, the loop has picked up a
# dependency it shouldn't have.


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

    `logged_at_call` is used to test the ordering: the loop must finish
    logging a step's metrics before calling `on_eval`, because a callback that
    reads the metrics file relies on that.
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
# The last thing a project has to supply: putting the pieces together in the
# right order. `runner.py` handles the parts that don't depend on the model
# (`build_run`'s ordering, and `record_run`'s bookkeeping), so what's left here
# is what actually differs per project: loading the data, and what to do at
# each eval.
#
# It's an example of what an entry point looks like, and it's also what the
# harness's tests use when they're checking the run record rather than the
# loop. A real project's `train.py` would add argparse, its own paths and its
# own smoke overrides on top. None of that belongs here, because the harness
# never makes up a path it wasn't given.


class EvalRecorder:
    """The `on_eval` callback: printing, the best checkpoint, and samples.
    Metrics are the loop's job, not this.

    Tracks `best_val` itself because it's what writes the best checkpoint; if
    the loop tracked it instead, one decision would be split across two places.
    The loop has already logged the step's metrics when this is called, so a
    callback can read the metrics file.
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
        # Same payload as the `last` checkpoint, from the same function, so
        # both load the same way.
        return checkpoint_payload(
            self.model, self.optimiser, step, self.data.tokenizer.vocab_size, self.cfg
        )

    def _write_sample(self, step: int, val_loss: float) -> None:
        # The prompt is text and gets encoded here rather than being a hardcoded
        # id: id 0 is '\n' for a character vocabulary but byte 0x00 for BPE.
        # `or [0]` because an empty prompt means "unconditional", but generate()
        # needs at least one token to start from.
        prompt = self.data.tokenizer.encode(self.cfg.train.sample_prompt)
        context = torch.tensor([prompt or [0]], dtype=torch.long, device=self.device)
        # eval() for the same reason `evaluate` uses it: on_eval runs after the
        # loop has put the model back in train mode, so otherwise every sample
        # would be drawn with dropout on. The mode is restored in `finally`, so
        # if generation fails, training doesn't carry on with dropout off.
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
    """Parse the config, build, train, and save. Returns the RunDir.

    The first three arguments match `sweep.RunFn`
    (`(config_path, overrides, runs_root) -> RunDir`), so this can be passed
    straight to `run_sweep`. Everything a project might change is keyword-only
    after them.

    `runs_root` has no default on purpose: the harness never makes up a path
    it wasn't given. `cache_dir=None` turns the token cache off, which is the
    right default when no one has chosen where the cache goes (a test must
    never write to a developer's real one).

    `smoke` only *labels* the record. Shrinking the run and putting it in a
    subdirectory is up to the project, because the overrides that shrink a
    model use that model's field names. See `--smoke` in `parser.parse_args`,
    which a project's `main()` turns into its own list of overrides.
    """
    if runs_root is None:
        raise ValueError(
            "runs_root is required: the harness resolves no path it wasn't given"
        )

    # Overrides go into the loader rather than being applied afterwards: the
    # config dataclasses are frozen, so changing a built RunConfig would raise
    # FrozenInstanceError.
    cfg = load_run_config(config_path, ARCHES, DATAS, overrides)

    device = resolve_device(cfg.runtime.device)
    # Always, not just when it isn't the default: it's process-wide state, and a
    # sweep runs many points in one process.
    set_matmul_precision(cfg.runtime.matmul_precision)
    # Built now rather than later, so a bad dtype fails before `new_run` has
    # created a directory. One instance does for the whole run.
    amp = autocast_context(device, cfg.runtime.autocast_dtype)

    data = load_data(
        cfg.data.path,
        cfg.data.split_frac,
        # A function rather than a config, because some tokenizers are built
        # *from* the corpus, and `load_data` is where the text is.
        partial(build_tokenizer, cfg),
        # Two different switches: `cache_dir` is where the caller allows writing,
        # and `data.cache` is whether this run wants caching at all.
        cache_dir if cfg.data.cache else None,
    )
    # Seed, build, compile, optimiser, schedule. The order matters and getting
    # it wrong fails silently; see `build_run`.
    built = build_run(cfg, data.tokenizer.vocab_size, ARCHES, device)
    eval_batches = draw_eval_batches(data, cfg, device)

    with record_run(
        runs_root,
        cfg,
        data,
        built,
        device=device,
        # Only used to run git for provenance. If none was given, the config
        # file's directory is a reasonable guess.
        repo_root=repo_root if repo_root is not None else Path(config_path).parent,
        smoke=smoke,
    ) as (rd, record):
        # This `with` goes inside, so the logger is closed before record_run's
        # summarize() reads the file back.
        with MetricLogger(rd.metrics_path) as logger:
            # `built.model`, not `built.runnable`: the recorder saves
            # checkpoints and samples, and both need the uncompiled module.
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
