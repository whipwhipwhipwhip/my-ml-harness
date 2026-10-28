"""Turns the tags in a config file (`arch`, `optim_arch`, ...) into the classes
they name.

`arch` decides which dataclass the `model` block is parsed into and which module
gets built. That has to happen in two steps (read the tag, *then* parse the
block), because you don't know the target type until you've read the tag. A
`model: GPTConfig | BigramConfig` union can't do this: dacite would try each
type in turn and take the first one that fits, ignoring the tag.

This module only has the mechanism. The `ARCHES` table lives in the project
that defines the models and is passed in as an argument, so nothing here names a
specific model. `DATAS` is passed in the same way, since a dataset config can be
specific to a task. `OPTIMS`, `TOKENIZERS` and `SCHEDULES` do live here, because
torch's optimisers, the `nanotok` package and the schedule maths aren't
specific to any project. The rule is: this package never has to name a *model*
or a *task*.

Unlike the others, `DATAS` maps straight to a config class rather than to a
(config class, builder) pair. The tag isn't used to pick a loader: an entry
point already imports the one `load_data` it uses, and a shared builder
signature would need things like `repo_root` and `cache_dir` that aren't in the
config and that half the loaders would ignore. So `data_arch` just gets you a
typed, validated `data` block, and a run record that says what that block means.
If an entry point ever needs to choose between loaders, turn it into a
NamedTuple like the others.
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
from nanotok import (
    NO_SPLIT_PATTERN,
    ByteTokenizer,
    CharTokenizer,
    RegexTokenizer,
    TiktokenBPETokenizer,
    Tokenizer,
)

# strict=True so a typo'd key is an error at load, rather than a hyperparameter
# that's silently ignored and makes two different runs look identical
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

    The config keeps them separate so each can be set on its own (e.g.
    `--set optim.beta2=0.95` as a sweep axis).

    `weight_decay` here is only the default; each group from
    `decay_param_groups` overrides it with its own value. It's passed anyway so
    that if someone hands this a plain list of parameters (a test, or a project
    building an optimiser by hand), they get the config's decay rather than
    torch's default of 0.01.
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

    `build` takes the peak lr and max_iters as well as the config, because a
    schedule depends on other parts of the run config: the peak is `optim.lr`,
    and "decay over the whole run" means `train.max_iters`. Every builder gets
    both, and the constant one ignores them (same idea as TokenizerSpec's
    corpus).
    """

    config_cls: type[ScheduleConfig]
    build: Callable[[ScheduleConfig, float, int], LRSchedule]

SCHEDULES: dict[str, ScheduleSpec] = {
    "constant": ScheduleSpec(ConstantScheduleConfig, constant_schedule),
    "cosine": ScheduleSpec(CosineScheduleConfig, cosine_schedule),
}

class TokenizerSpec(NamedTuple):
    """The two things a tokenizer name resolves to.

    `build` takes the corpus as well as the config because some tokenizers
    depend on the data and some don't: `char` gets its alphabet from the text,
    `regex_bpe` trains on it, and `char_fixed`/`byte`/`tiktoken` ignore it. With
    one signature for all of them, the caller doesn't need to know which kind
    it's building.
    """

    config_cls: type[TokenizerConfig]
    build: Callable[[TokenizerConfig, str], Tokenizer]

# Every builder passes on `cfg.pad_vocab_to`, since it's on the base config:
# padding is about the id space, not about how the ids were learned.

def _build_char(cfg: CharConfig, text: str) -> Tokenizer:
    return CharTokenizer.from_text(text, cfg.pad_vocab_to)

def _build_char_fixed(cfg: CharFixedConfig, text: str) -> Tokenizer:
    return CharTokenizer.from_alphabet(cfg.alphabet, cfg.pad_vocab_to)

def _build_byte(cfg: ByteConfig, text: str) -> Tokenizer:
    return ByteTokenizer(cfg.pad_vocab_to)

def _build_regex_bpe(cfg: RegexBPEConfig, text: str) -> Tokenizer:
    # null in the config means "the tokenizer's own default", and that default
    # is only defined in one place (NO_SPLIT_PATTERN).
    tok = RegexTokenizer(
        special_tokens=cfg.special_tokens,
        split_pattern=cfg.split_pattern or NO_SPLIT_PATTERN,
        pad_vocab_to=cfg.pad_vocab_to,
    )
    # `vocab_size` is the merge target and `pad_vocab_to` is the size that gets
    # reported. Train up to the former; the padding sits above it.
    tok.train(text, cfg.vocab_size)
    return tok

def _build_tiktoken(cfg: TiktokenConfig, text: str) -> Tokenizer:
    return TiktokenBPETokenizer(
        cfg.encoding, cfg.split_pattern, cfg.special_tokens, cfg.pad_vocab_to
    )

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
        # `key` rather than a hardcoded "arch", since this checks optimisers etc. too
        raise ValueError(
            f"unknown {key} {value!r}; expected one of {sorted(valid_set)}"
        ) from None
    
    return value, spec

###-----------------------------------------------------
# Overrides

def _parse_value(text: str):
    """Turn a `--set` string into a JSON value, like the config file has.

    Unquoted words aren't valid JSON, so things like `runtime.device=mps` or a
    `note=` are kept as the string that was typed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(raw: dict, overrides: Iterable[str]) -> None:
    """Apply `dotted.key=value` overrides to the raw config dict, in place.

    This happens before dacite, not after: the config dataclasses are frozen,
    so changing a built RunConfig raises FrozenInstanceError. Changing the dict
    the config is *built from* also means an override is validated exactly like
    a value typed into the JSON file: `strict=True` rejects a typo'd key inside a
    block, dacite still checks types, and `__post_init__` sees the final values.
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

        # A typo'd *top-level* key would otherwise be silently ignored, since
        # load_run_config only reads the keys it knows about. Inside a block
        # this isn't needed, because strict=True catches it.
        if node is raw and field not in raw:
            raise ValueError(
                f"override {key!r}: no such key; expected one of {sorted(raw)}"
            )

        value = _parse_value(text)

        # `--set optim.lr=1` gives an int for a float field, which dacite
        # rejects. The file already has a float there, so convert rather than
        # making the caller remember to type `1.0`.
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

    Built one field at a time rather than by handing the whole dict to dacite,
    because each block's class depends on its tag, and dacite's behaviour when a
    dataclass field is given something that isn't a dict varies between
    versions. It costs a line per section but is predictable.

    `overrides` are applied to the raw dict first, before the `arch` tag is
    read, so `--set arch=...` picks the config class just like the file would.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # The only optional keys, filled in *before* the overrides are applied.
    # That way a file written before schedules existed still loads, and
    # `--set schedule_arch=cosine --set schedule.warmup_steps=100` still works on
    # it. apply_overrides won't create a missing section, so without this you
    # couldn't set the schedule from the command line for such a file.
    raw.setdefault("schedule_arch", "constant")
    raw.setdefault("schedule", {})

    apply_overrides(raw, overrides)

    arch, arch_spec = pop_and_validate(raw, "arch", path, ARCHES)
    optimiser, optimiser_spec = pop_and_validate(raw, "optim_arch", path, OPTIMS)
    tokenizer, tokenizer_spec = pop_and_validate(
        raw, "tokenizer_arch", path, TOKENIZERS
    )
    schedule, schedule_spec = pop_and_validate(raw, "schedule_arch", path, SCHEDULES)
    # The only table that maps straight to a class (see the module docstring).
    # `pop_and_validate` doesn't care either way.
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

    vocab_size comes from the data rather than being chosen, so it's passed in
    as an argument rather than read from the config. Every arch builder takes it
    as the second argument. One consequence: a saved RunConfig alone doesn't say
    how big the embedding was, so to rebuild a model you need vocab_size from
    somewhere else (checkpoints store it for this reason).

    `cfg.arch` was already checked by load_run_config, so this lookup can't fail
    for a RunConfig that came from there.
    """
    return ARCHES[cfg.arch].build(cfg.model, vocab_size)

###---------------------------------------------------------------
# Optimiser


def decay_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """GPT-2's rule: decay tensors of dim >= 2, leave everything else alone.

    Weight matrices and embeddings are decayed; biases and LayerNorm gains
    aren't. Dimension is a simple way to tell them apart: shrinking a matrix is
    regularisation, but a bias or gain is a per-unit offset, and decay would
    only pull it towards zero for no reason.

    It's a separate function rather than part of `_build_adamw` because the
    split depends on the *model's parameter shapes*, not on the optimiser. Any
    other optimiser with weight decay would want the same list.

    `model.parameters()` skips duplicates, so a tied embedding and output head
    shows up once here and can't be decayed twice per step.

    Returns param groups, which every torch optimiser accepts in place of a
    flat list of parameters. The loop writes the scheduled lr into *every*
    group, so having two groups is safe.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def build_optimiser(cfg: RunConfig, model: nn.Module) -> opt.Optimizer:
    """Build the module `cfg.optim_arch` names, over GPT-2's two param groups.

    The grouping happens here rather than in the spec's `build` because this is
    where the model is available. `OptimSpec.build` takes parameters, and making
    every optimiser take a whole module for this one purpose seemed wrong.

    One thing to watch out for: an optimiser with no `weight_decay` option would
    be given groups with a key it doesn't understand. Nothing in OPTIMS is like
    that today, and almost all of torch's optimisers support decay. If one is
    ever added, the split may need to move from here into the spec.
    """
    return OPTIMS[cfg.optim_arch].build(
        cfg.optim, decay_param_groups(model, cfg.optim.weight_decay)
    )

def build_lr_schedule(cfg: RunConfig) -> LRSchedule:
    """Build the `(step) -> lr` closure `cfg.schedule_arch` names.

    This needs three blocks of the config, which is why it's here rather than
    on the schedule config: `optim.lr` is the peak and `train.max_iters` is the
    default length of the decay. It's built once per run, so the loop just holds
    a function of the step and never has to interpret a config.
    """
    return SCHEDULES[cfg.schedule_arch].build(
        cfg.schedule, cfg.optim.lr, cfg.train.max_iters
    )

###---------------------------------------------------------------
# Tokenizer


def build_tokenizer(cfg: RunConfig, text: str) -> Tokenizer:
    """Build the tokenizer `cfg.tokenizer_arch` names, over `text`.

    Takes the whole RunConfig rather than just the tokenizer block, to match
    build_model/build_optimiser. `text` is the corpus, which some tokenizers
    need and others ignore (see TokenizerSpec).
    """
    return TOKENIZERS[cfg.tokenizer_arch].build(cfg.tokenizer, text)