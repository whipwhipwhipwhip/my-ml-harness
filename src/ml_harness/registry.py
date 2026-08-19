"""Maps the `arch` tag in a config file onto the classes it names.

`arch` is a discriminator: its value decides which dataclass the `model` block
parses into and which module gets built. That has to happen in two stages — read
the tag, *then* parse — because you can't know the target type until you've read
the tag. A `model: GPTConfig | BigramConfig` union can't express this: dacite
would try each member in turn and take the first that fits, ignoring the tag.

The mechanism only: the `ARCHES` table itself lives with the arches
(`nanogpt_conor/arches.py`) and arrives here as a parameter, so nothing in this
module names GPTConfig or GPTModel and `ml_harness/` stays arch-agnostic. `DATAS` is
passed in for the same reason and by the same route — `gpt_addition`'s dataset
config names a task, and an import of it here would be the back-arrow the
layering rule exists to prevent. `OPTIMS` and `TOKENIZERS` are the deliberate
asymmetries — AdamW is torch and `tokenizers` is a generic peer
package, so neither is project-specific and both tables live here. The line is
"does `ml_harness/` have to name a *model* or a *task*", and it still doesn't.

`DATAS` maps to a config class rather than to a spec pair, unlike the other
three. The tag doesn't dispatch: an entry point already imports the one
`load_data` it uses, and a uniform builder would have to be handed `repo_root`
and `cache_dir` — caller concerns that aren't in the config and that half the
implementations would ignore. So `data_arch` buys typing of the block,
validation, and a run record that says what its own `data` block means. Promote
it to a NamedTuple the day one entry point has two representations to choose
between.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple
from dataclasses import asdict

import torch.nn as nn
import torch.optim as opt
from dacite import Config, from_dict

from ml_harness.config import(
    ByteConfig,
    CharConfig,
    CharFixedConfig,
    ConstantScheduleConfig,
    CosineScheduleConfig,
    DataConfig,
    ModelConfig,
    OptimiserConfig,
    AdamWConfig,
    RegexBPEConfig,
    RunConfig,
    RuntimeConfig,
    ScheduleConfig,
    TiktokenConfig,
    TokenizerConfig,
    TrainConfig,
)
from ml_harness.schedule import (
    LRSchedule,
    constant_schedule,
    cosine_schedule,
)
from tokenizers import (
    NO_SPLIT_PATTERN,
    ByteTokenizer,
    CharTokenizer,
    RegexTokenizer,
    TiktokenBPETokenizer,
    Tokenizer,
)

# callable[[input types], output_type]

# strict=True so a typo'd key is an error at load rather than a silently ignored
# hyperparameter that makes two runs look identical when they aren't
_DACITE = Config(strict=True)

def _adapt(optim_cls):
    """torch optimisers take (params, **hyper); our builders take (cfg, params)."""
    def build(cfg: OptimiserConfig, params: Iterable[nn.Parameter]) -> opt.Optimizer:
        return optim_cls(params, **asdict(cfg))
    return build

class ArchSpec(NamedTuple):
    """The two things an arch name resolves to."""

    config_cls: type[ModelConfig]
    build: Callable[[ModelConfig, int], nn.Module]

class OptimSpec(NamedTuple):
    """ Two things an optimiser name resolves to."""
    config_cls: type[OptimiserConfig]
    build: Callable[[OptimiserConfig, Iterable[nn.Parameter]], opt.Optimizer]

def _build_adamw(cfg: AdamWConfig, params: Iterable[nn.Parameter]) -> opt.Optimizer:
    """AdamW can't use `_adapt`: torch takes one `betas` tuple, not two scalars.

    The config keeps them apart so each is addressable on its own — `--set
    optim.beta2=0.95` is a sweep axis, and one element of a list isn't.

    `weight_decay` here is the constructor default, which every group built by
    `decay_param_groups` overrides with its own value. It is passed anyway so
    that a caller handing this a bare parameter iterable — a test, a downstream
    project building an optimiser by hand — gets the config's decay rather than
    torch's silent 0.01.
    """
    return opt.AdamW(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

OPTIMS: dict[str, OptimSpec] = {
    "adamw": OptimSpec(AdamWConfig, _build_adamw),
}

class ScheduleSpec(NamedTuple):
    """The two things a schedule name resolves to.

    `build` takes the peak lr and max_iters as well as the config because a
    schedule is a function of the *run*, not only of its own block: the peak is
    `optim.lr` and "decay over the whole run" is `train.max_iters`. Both are
    passed to every builder, ignored by the constant one — same bargain as
    TokenizerSpec's corpus.
    """

    config_cls: type[ScheduleConfig]
    build: Callable[[ScheduleConfig, float, int], LRSchedule]

SCHEDULES: dict[str, ScheduleSpec] = {
    "constant": ScheduleSpec(ConstantScheduleConfig, constant_schedule),
    "cosine": ScheduleSpec(CosineScheduleConfig, cosine_schedule),
}

class TokenizerSpec(NamedTuple):
    """The two things a tokenizer name resolves to.

    `build` takes the corpus as well as the config because a tokenizer may or may
    not be a function of the data it will be used on: `char` reads its alphabet
    off the text, `regex_bpe` trains on it, and `char_fixed`/`byte`/`tiktoken`
    ignore it entirely. One signature covers all three cases; the alternative is
    the caller knowing which kind it built.
    """

    config_cls: type[TokenizerConfig]
    build: Callable[[TokenizerConfig, str], Tokenizer]

def _build_char(cfg: CharConfig, text: str) -> Tokenizer:
    return CharTokenizer.from_text(text)

def _build_char_fixed(cfg: CharFixedConfig, text: str) -> Tokenizer:
    return CharTokenizer.from_alphabet(cfg.alphabet)

def _build_byte(cfg: ByteConfig, text: str) -> Tokenizer:
    return ByteTokenizer()

def _build_regex_bpe(cfg: RegexBPEConfig, text: str) -> Tokenizer:
    # `or NO_SPLIT_PATTERN` rather than a branch: null in the config means "the
    # tokenizer's own default", and the default stays spelled out in one place.
    tok = RegexTokenizer(
        special_tokens=cfg.special_tokens,
        split_pattern=cfg.split_pattern or NO_SPLIT_PATTERN,
    )
    tok.train(text, cfg.vocab_size)
    return tok

def _build_tiktoken(cfg: TiktokenConfig, text: str) -> Tokenizer:
    return TiktokenBPETokenizer(cfg.encoding, cfg.split_pattern, cfg.special_tokens)

TOKENIZERS: dict[str, TokenizerSpec] = {
    "char": TokenizerSpec(CharConfig, _build_char),
    "char_fixed": TokenizerSpec(CharFixedConfig, _build_char_fixed),
    "byte": TokenizerSpec(ByteConfig, _build_byte),
    "regex_bpe": TokenizerSpec(RegexBPEConfig, _build_regex_bpe),
    "tiktoken": TokenizerSpec(TiktokenConfig, _build_tiktoken),
}

def pop_and_validate(raw, key, path, valid_set):
    try:
        value = raw.pop(key)
    except KeyError:
        raise ValueError(
            f"{path} has no '{key}' key; expected one of {sorted(valid_set)}"
        ) from None

    try:
        spec = valid_set[value]
    except KeyError:
        # `key`, not a hardcoded "arch" — this rejects optimisers too
        raise ValueError(
            f"unknown {key} {value!r}; expected one of {sorted(valid_set)}"
        ) from None
    
    return value, spec

###-----------------------------------------------------
# Overrides

def _parse_value(text: str):
    """`--set` hands us strings; the config file's own types are JSON's.

    Unquoted words aren't valid JSON, so `runtime.device=mps` and a `note=`
    fall through to the string as typed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(raw: dict, overrides: Iterable[str]) -> None:
    """Apply `dotted.key=value` overrides to the raw config dict, in place.

    Before dacite, not after: the config dataclasses are frozen, so an override
    applied to a built RunConfig raises FrozenInstanceError. Rewriting the dict
    the config is *built from* also means an override is validated by exactly
    the path a hand-typed JSON value is — `strict=True` rejects a typo'd key
    inside a block, dacite still type-checks, and `__post_init__` still sees the
    final values rather than the file's.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"malformed override {override!r}; expected KEY=VALUE")

        key, text = override.split("=", 1)
        parts = key.split(".")

        node = raw
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(
                    f"override {key!r}: no section {part!r} in the config; "
                    f"expected one of {sorted(node) if isinstance(node, dict) else '(not a section)'}"
                )
            node = node[part]

        if not isinstance(node, dict):
            raise ValueError(f"override {key!r}: {'.'.join(parts[:-1])} is a value, not a section")

        field = parts[-1]

        # A typo'd *top-level* key would otherwise pass silently: load_run_config
        # reads named keys out of `raw`, so an extra one goes nowhere. Inside a
        # block there's no need for this check — strict=True is what catches it.
        if node is raw and field not in raw:
            raise ValueError(
                f"override {key!r}: no such key; expected one of {sorted(raw)}"
            )

        value = _parse_value(text)

        # `--set optim.lr=1` is an int against a float field, which dacite rejects.
        # The file already says which one it is, so use that rather than make the
        # caller remember to type the trailing `.0`.
        current = node.get(field)
        if isinstance(current, float) and type(value) is int:
            value = float(value)

        node[field] = value


###-----------------------------------------------------
# Model architecture

def load_run_config(
    path: Path | str,
    ARCHES: dict[str, ArchSpec],
    DATAS: dict[str, type[DataConfig]],
    overrides: Iterable[str] = (),
) -> RunConfig:
    """Parse a config file into a RunConfig, with `model` typed by `arch`.

    Built field-by-field rather than by handing the whole dict to dacite: the
    `model` block is already a parsed object by this point, and dacite's
    behaviour when a dataclass field receives a non-dict is version-dependent.
    Explicit costs a line per section and stays predictable.

    `overrides` are applied to the raw dict first — before the `arch` tag is even
    read, so `--set arch=...` selects the config class the way the file would.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # The one optional pair of keys, filled in *before* the overrides run rather
    # than after. Two things depend on that order: a file written before
    # schedules existed still loads, and `--set schedule_arch=cosine --set
    # schedule.warmup_steps=100` works against such a file — apply_overrides
    # navigates the dict it's given and refuses to invent a section, so an
    # absent `schedule` block would otherwise make the schedule the one thing
    # you can't reach from the command line.
    raw.setdefault("schedule_arch", "constant")
    raw.setdefault("schedule", {})

    apply_overrides(raw, overrides)

    arch, arch_spec = pop_and_validate(raw, "arch", path, ARCHES)
    optimiser, optimiser_spec = pop_and_validate(raw, "optim_arch", path, OPTIMS)
    tokenizer, tokenizer_spec = pop_and_validate(
        raw, "tokenizer_arch", path, TOKENIZERS
    )
    schedule, schedule_spec = pop_and_validate(raw, "schedule_arch", path, SCHEDULES)
    # The only one whose table maps straight to a class — see the module
    # docstring. `pop_and_validate` doesn't care either way; it looks up a tag.
    data, data_cls = pop_and_validate(raw, "data_arch", path, DATAS)

    return RunConfig(
        arch=arch,
        model=from_dict(arch_spec.config_cls, raw["model"], _DACITE),
        optim_arch=optimiser,
        optim=from_dict(optimiser_spec.config_cls, raw["optim"], _DACITE),
        tokenizer_arch=tokenizer,
        tokenizer=from_dict(tokenizer_spec.config_cls, raw["tokenizer"], _DACITE),
        train=from_dict(TrainConfig, raw["train"], _DACITE),
        data_arch=data,
        data=from_dict(data_cls, raw["data"], _DACITE),
        runtime=from_dict(RuntimeConfig, raw["runtime"], _DACITE),
        note=raw["note"],
        schedule_arch=schedule,
        schedule=from_dict(schedule_spec.config_cls, raw["schedule"], _DACITE),
    )


def build_model(cfg: RunConfig, vocab_size: int, ARCHES: dict[str, ArchSpec]) -> nn.Module:
    """Build the module `cfg.arch` names.

    vocab_size is derived from the data, not chosen, so it arrives as an argument
    rather than through the config file — every arch builder takes it second, for
    the same reason GPTModel does. The consequence to keep in mind: a checkpoint's
    saved RunConfig alone doesn't say how wide the embedding was, so rebuilding
    from one means re-reading the dataset (or recording vocab_size elsewhere).

    `cfg.arch` was validated by load_run_config, so this lookup can't miss for a
    RunConfig that came from there.
    """
    return ARCHES[cfg.arch].build(cfg.model, vocab_size)

###---------------------------------------------------------------
# Optimiser


def decay_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """GPT-2's rule: decay tensors of dim >= 2, leave everything else alone.

    Weight matrices and embeddings are decayed; biases and LayerNorm gains are
    not. The dimension is a proxy for the distinction that matters — a matrix
    maps between spaces and shrinking it is regularisation, while a bias or a
    gain is a per-unit offset that decay just drags toward an arbitrary zero.

    A free function rather than something inside `_build_adamw` because the
    split is a property of the *model's parameter shapes*, not of the optimiser
    that consumes it: a second decaying optimiser would want exactly this list.

    `model.parameters()` de-duplicates shared tensors, so a weight-tied
    embedding and output head is one entry here rather than two — decayed twice
    per step is not a thing this can produce.

    Returns param groups rather than a flat iterable, which is the interface
    every torch optimiser already accepts in place of one. Note the loop writes
    the scheduled lr into *every* group, which is what makes two of them safe.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def build_optimiser(cfg: RunConfig, model: nn.Module) -> opt.Optimizer:
    """Build the module `cfg.optim_arch` names, over GPT-2's two param groups.

    The grouping happens here rather than in the spec's `build` because this is
    where the model is: `OptimSpec.build` takes parameters, and widening it to
    take a module for the sake of one optimiser would put a model in the
    signature of every optimiser that never needs one.

    An optimiser without a `weight_decay` is the case to watch — it would be
    handed groups carrying a key it doesn't know. Nothing in OPTIMS is such an
    optimiser today, and torch's are near-uniformly decay-capable; the day one
    arrives, this is the line that decides whether the split belongs to
    `build_optimiser` or to the spec.
    """
    return OPTIMS[cfg.optim_arch].build(
        cfg.optim, decay_param_groups(model, cfg.optim.weight_decay)
    )

def build_lr_schedule(cfg: RunConfig) -> LRSchedule:
    """Build the `(step) -> lr` closure `cfg.schedule_arch` names.

    Reaches across three blocks of the config, which is why it's here and not on
    the schedule itself: `optim.lr` is the peak and `train.max_iters` is what
    "decay over the whole run" resolves against. Built once per run, so the loop
    holds a closure over resolved numbers rather than a config to interpret.
    """
    return SCHEDULES[cfg.schedule_arch].build(
        cfg.schedule, cfg.optim.lr, cfg.train.max_iters
    )

###---------------------------------------------------------------
# Tokenizer


def build_tokenizer(cfg: RunConfig, text: str) -> Tokenizer:
    """Build the tokenizer `cfg.tokenizer_arch` names, over `text`.

    Takes the whole RunConfig rather than the tokenizer block alone, to match
    build_model/build_optimiser: the caller passes what it has. `text` is the
    corpus, which some tokenizers need and others ignore — see TokenizerSpec.
    """
    return TOKENIZERS[cfg.tokenizer_arch].build(cfg.tokenizer, text)