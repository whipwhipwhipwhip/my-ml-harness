from dataclasses import dataclass, field

@dataclass(frozen=True)
class OptimiserConfig:
    """ Hyperparameters for an optimiser """

    lr: float

@dataclass(frozen=True)
class AdamWConfig(OptimiserConfig):
    """ AdamW's betas, as two scalars rather than torch's `betas` tuple.

    Splitting them means each can be set on its own, e.g. as a sweep axis with
    `--set optim.beta2=0.95`, which you can't do with one element of a list.
    `registry._build_adamw` puts them back into a tuple.

    The defaults are torch's, so an `optim` block with only `lr` in it still
    loads (and every existing config and recorded run is like that). """

    beta1: float = 0.9
    beta2: float = 0.999
    # GPT-2's rule, applied in registry.decay_param_groups: tensors with dim >= 2
    # (weight matrices and embeddings) get this decay, and biases and LayerNorm
    # gains get 0.0. Shrinking a matrix is regularisation, but a bias or gain is
    # just a per-unit offset, and decay would only pull it towards zero for no
    # reason. Dimension is a simple way to tell the two apart.
    #
    # 0.1 is the GPT-2/nanoGPT value. Runs recorded before this field existed
    # didn't train with it: `_build_adamw` didn't pass `weight_decay`, so torch's
    # default of 0.01 was applied to every parameter, biases included. Those
    # configs still load but won't reproduce the original run, and no default
    # could fix that, since the grouping alone changes the numerics.
    #
    # It's on AdamWConfig rather than the OptimiserConfig base for the same
    # reason train.grad_clip isn't: the base only has what every optimiser has.
    # Also, `_adapt` builds optimisers as `optim_cls(params, **asdict(cfg))`, so
    # a field on the base would be passed as a flat, ungrouped `weight_decay=`
    # to every optimiser built that way.
    weight_decay: float = 0.1

    def __post_init__(self):
        # Negative decay grows the weights every step, and the run diverges
        # slowly enough that it looks like a bad lr.
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )

@dataclass(frozen=True)
class ScheduleConfig:
    """ Empty base for learning-rate schedule configs.

    `schedule_arch` says which subclass the `schedule` block is parsed into,
    the same way `arch` does for `model`. It's empty for the same reason
    ModelConfig is: "hold lr fixed" and "warm up then cosine to a floor" don't
    share any parameters.

    None of these hold the learning rate itself. `optim.lr` is the *peak*, the
    optimiser is built with it, and a schedule only says how the lr moves
    relative to it. That way the learning rate is only set in one place.
    """

@dataclass(frozen=True)
class ConstantScheduleConfig(ScheduleConfig):
    """Hold `optim.lr` for the whole run. Nothing to configure.

    This is the default, and it's what every run did before schedules existed.
    """

@dataclass(frozen=True)
class CosineScheduleConfig(ScheduleConfig):
    """Linear warmup, cosine decay to a floor, then flat (the GPT-3 schedule).

    Measured in steps rather than tokens. The paper uses tokens because its
    batch size changes during the run. Ours doesn't, so steps and tokens differ
    by a constant factor and it makes no difference.

    Every field has a default, so `--set schedule_arch=cosine` on its own is a
    valid schedule: no warmup, decaying over the whole run to 10% of peak.
    """

    # Number of steps to ramp linearly from ~0 up to the peak. Adam's
    # second-moment estimate is unreliable for the first few dozen steps (it's
    # averaging over almost no history), so early updates can be large and
    # badly scaled, right when the weights can least handle it.
    warmup_steps: int = 0
    # The step at which the cosine reaches the floor, counted from step 0 with
    # the warmup included (like nanoGPT's `lr_decay_iters`, so numbers carry
    # over). None means train.max_iters, i.e. decay over the whole run. That's
    # usually what you want, and it keeps the schedule's shape when max_iters
    # is swept. Set it explicitly to reach the floor early and stay there.
    decay_steps: int | None = None
    # The floor, as a fraction of the peak. A fraction rather than an absolute
    # lr so that `--set optim.lr=1e-3` moves the peak and floor together
    # instead of changing the decay ratio, and so the floor can never end up
    # above the peak.
    #
    # It only bounds the decay. Warmup starts from ~0, so it passes below the
    # floor on the way up. Starting warmup at the floor would mean starting at
    # 10% of peak, which is barely a warmup at all.
    min_lr_frac: float = 0.1

    def __post_init__(self):
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0.0 <= self.min_lr_frac <= 1.0:
            raise ValueError(
                f"min_lr_frac is a fraction of optim.lr and must be in [0, 1], "
                f"got {self.min_lr_frac}"
            )
        # Can only check this when decay_steps is given. When it's None it
        # becomes train.max_iters at build time, which can legitimately be
        # below the warmup (a smoke run is 4 iters), so the schedule function
        # handles that case instead of raising.
        if self.decay_steps is not None and self.decay_steps <= self.warmup_steps:
            raise ValueError(
                f"decay_steps ({self.decay_steps}) must be greater than "
                f"warmup_steps ({self.warmup_steps}); it is counted from step 0, "
                f"warmup included, so there would be no cosine segment at all"
            )

@dataclass(frozen=True)
class TrainConfig:
    """ Hyperparameters for training loops """

    max_iters: int
    batch_size: int
    eval_interval: int
    eval_iters: int
    # The prompt generation starts from at each eval. It's text rather than an
    # id because it goes through the run's own tokenizer: id 0 is '\n' for a
    # character vocabulary but byte 0x00 for BPE. The default newline means
    # "start of a line", which is what the character tokenizer used to hardcode.
    sample_prompt: str = "\n"
    # Only sample from the k most likely tokens. This is a decoding setting, not
    # part of the architecture, so it lives here rather than on a ModelConfig:
    # it changes what gets written to samples/, never what the model learns.
    # None samples from the full distribution (what every existing run does).
    # Values are usually picked for big vocabularies (HuggingFace defaults to
    # 50, against GPT-2's 50257); with a char tokenizer (~65 ids) it does very
    # little.
    top_k: int | None = None
    # Whether each eval writes a generated sample to samples/. On for a single
    # run, since for a character-level LM the samples are the main qualitative
    # signal and they're nearly free. Turn it off for sweeps, where hundreds of
    # samples are just noise and generation is a real fraction of a short run's
    # time. It's a config field rather than an argument to `run()` so a sweep
    # can set it through the usual override path.
    write_samples: bool = True
    # Maximum global L2 norm of the gradient; if it's larger, the gradient is
    # scaled down to this. None means no clipping (what every run before this
    # field did).
    #
    # This is on TrainConfig rather than OptimiserConfig because clipping isn't
    # part of the optimiser: the loop does it between backward and step, and it
    # works the same whatever optimiser comes next.
    #
    # One bad batch can produce a gradient orders of magnitude bigger than
    # usual, and without clipping Adam's moment estimates carry that spike for
    # thousands of steps. 1.0 is the nanoGPT/GPT-3 value.
    grad_clip: float | None = None
    # Sequences per forward/backward pass, for when the full batch doesn't fit
    # in memory. `batch_size` always means sequences per optimiser step; this
    # says how many fit at once. The loop runs `batch_size // micro_batch_size`
    # backward passes before each step, scaling each loss so the accumulated
    # gradient is the same as the full batch's.
    #
    # None means no accumulation (what every run before this field did). It's
    # set this way round, rather than as an accumulation count like nanoGPT's
    # `grad_accum_steps` (where `batch_size` becomes the micro batch), because
    # that would change what `batch_size` means for runs already recorded, and
    # the real batch size would become the product of two fields.
    #
    # It's a hardware setting rather than an experimental one, so it's in
    # provenance.GROUP_EXCLUDES: splitting a batch to fit on a smaller card is
    # the same experiment. It's not exactly the same run (different batches get
    # drawn, and floats are summed in a different order), but that's the same
    # situation as changing `runtime.seed`.
    micro_batch_size: int | None = None

    def __post_init__(self):
        # 0.0 would zero every gradient, and the run would look healthy with a
        # flat loss. Better to fail here.
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        # Checked here rather than in the loop so a bad value fails before a
        # model is built or a run directory is written. If the batch doesn't
        # split evenly that's a config error. Quietly rounding batch_size down
        # would mean the recorded batch_size is wrong.
        if self.micro_batch_size is not None:
            if self.micro_batch_size <= 0:
                raise ValueError(
                    f"micro_batch_size must be positive, got {self.micro_batch_size}"
                )
            if self.micro_batch_size > self.batch_size:
                raise ValueError(
                    f"micro_batch_size ({self.micro_batch_size}) must not exceed "
                    f"batch_size ({self.batch_size}); it is a slice of the batch, "
                    f"not a replacement for it"
                )
            if self.batch_size % self.micro_batch_size:
                raise ValueError(
                    f"batch_size ({self.batch_size}) must be divisible by "
                    f"micro_batch_size ({self.micro_batch_size}); otherwise the "
                    f"accumulated gradient is not the full batch's"
                )

@dataclass(frozen=True)
class DataConfig:
    """ Empty base for dataset configs.

    `data_arch` says which subclass the `data` block is parsed into. It's empty
    because no field applies to every dataset: `cache` means nothing for a
    generated corpus, `n_problems` means nothing for a file, and even
    `split_frac` depends on what the dataset is.

    Only the generic file subclass lives here. A dataset for a specific task
    (e.g. `gpt_addition`'s AdditionDataConfig) lives with that task and reaches
    `load_run_config` through a DATAS table passed in by the caller, the same
    way a model config does through ARCHES.
    """

@dataclass(frozen=True)
class FileDataConfig(DataConfig):
    """A text corpus read from disk, split by position."""

    path: str
    split_frac: float
    # Whether to cache the encoded corpus on disk. The cache key is a content
    # hash, so the only time this matters is when the encoder's behaviour
    # changed without its recipe changing. The real fix for that is bumping
    # CACHE_VERSION; this switch is for tracking such a bug down.
    cache: bool = True

@dataclass(frozen=True)
class RuntimeConfig:
    """ Seeding, devices """

    seed: int
    eval_seed: int
    device: str
    # How fp32 matmuls are allowed to be computed. "highest" is torch's default
    # and keeps the full 24-bit mantissa. "high"/"medium" give up some mantissa
    # bits for speed; on CUDA (Ampere and later) that means TF32 on the tensor
    # cores, which is where the lecture's speedup comes from. It does nothing on
    # MPS or CPU (I measured), but it's in the config so moving to a rented GPU
    # is a config change rather than a code change.
    #
    # It's under runtime rather than optim or model because it doesn't change a
    # hyperparameter or the architecture, just how the same arithmetic is done,
    # like the choice of device. It's *not* in GROUP_EXCLUDES, so two precisions
    # are two experiments: the numerics differ, so their losses shouldn't be
    # averaged together.
    #
    # Has a default so older config files still load with dacite's strict=True,
    # and so their numerics don't change.
    matmul_precision: str = "highest"
    # Mixed precision for the forward pass: None (all fp32, the default) or
    # "bf16". This is different from matmul_precision, which only controls how
    # fp32 matmuls are computed internally. This one changes the dtype the
    # activations are stored in, so it saves memory as well as compute.
    #
    # Weights and gradients stay fp32 either way: autocast casts the inputs to
    # each op, not the parameters, so the optimiser never sees a bf16 tensor.
    # Only bf16 is allowed, since fp16 would need a GradScaler, which this
    # harness doesn't have.
    autocast_dtype: str | None = None
    # Whether to run the training step through `torch.compile`. Off by default
    # based on measurement: on MPS the inductor backend currently produces a
    # LayerNorm kernel ~16x slower than the one eager mode uses, which made a GPT
    # with 9 LayerNorms ~37% slower. It's a big win on CUDA, where the codegen is
    # mature, hence a config field rather than hardcoding it either way.
    compile: bool = False

@dataclass(frozen=True)
class ModelConfig:
    """ Empty base for model configs """

@dataclass(frozen=True)
class TokenizerConfig:
    """ Base for tokenizer configs.

    `tokenizer_arch` says which subclass below the `tokenizer` block is parsed
    into, the same way `arch` does for `model`. Two of the subclasses are empty
    on purpose: "the alphabet is whatever the corpus contains" and "one id per
    byte" need no options.

    Padding is the one option every tokenizer takes, so it's here rather than
    repeated on each subclass.
    """

    # The `vocab_size` to report, if larger than the real one. The extra ids are
    # never produced and their embedding rows never train. The point is to get a
    # GPU-friendly size: 50257 is awkward, while 50304 is 64 x 786. The
    # tokenizer owns this rather than the model because `vocab_size` is what the
    # embedding table is built from, so if the model padded on its own the two
    # would disagree. See `nanotok.base.pad_vocab_size`. It's also saved in
    # `to_dict` and so in the fingerprint, which keeps padded and unpadded runs
    # from sharing a cache entry.
    #
    # `kw_only` so that having a default here doesn't force defaults onto the
    # required fields the subclasses add (`alphabet`, `vocab_size`, `encoding`).
    pad_vocab_to: int | None = field(default=None, kw_only=True)

@dataclass(frozen=True)
class CharConfig(TokenizerConfig):
    """Character-level, alphabet taken from the corpus. Nothing to configure."""

@dataclass(frozen=True)
class CharFixedConfig(TokenizerConfig):
    """Character-level over a given alphabet.

    A separate entry rather than an optional field on CharConfig, because the
    difference is whether the ids depend on the corpus, and that decides whether
    a checkpoint can still be decoded once the corpus changes.
    """

    alphabet: str

@dataclass(frozen=True)
class ByteConfig(TokenizerConfig):
    """One id per byte. Nothing to configure."""

@dataclass(frozen=True)
class RegexBPEConfig(TokenizerConfig):
    """BPE trained on this run's corpus.

    The tokenizer is retrained on every run (only the encoded corpus is
    cached). That's fine for a small corpus but takes minutes for a large one.
    """

    vocab_size: int
    # None means the tokenizer's own default (the whole text as one chunk).
    # Left as None rather than copying the pattern here, so the default is only
    # defined in one place.
    split_pattern: str | None = None
    # Given ids directly above the 256 bytes, in this order, and counted towards
    # `vocab_size`. The order decides the ids, so reordering the list gives a
    # different tokenizer.
    special_tokens: list[str] | None = None

@dataclass(frozen=True)
class TiktokenConfig(TokenizerConfig):
    """BPE using a tiktoken encoding's vocabulary, e.g. `gpt2`, `cl100k_base`."""

    encoding: str
    # None means the encoding's own pattern, which is what you want. GPT-2's is
    # case-sensitive and GPT-4's isn't, and mixing them up quietly produces the
    # wrong ids.
    split_pattern: str | None = None
    # Names from the encoding's own special tokens, e.g. ["<|endoftext|>"].
    # They keep the ids OpenAI gave them rather than being appended. This is
    # what takes GPT-2 from 50256 to 50257 ids: an unused id is an embedding row
    # that never trains, so a run only includes the specials it will use.
    special_tokens: list[str] | None = None

@dataclass(frozen=True)
class RunConfig:

    arch: str
    model: ModelConfig
    optim: OptimiserConfig
    optim_arch: str
    tokenizer_arch: str
    tokenizer: TokenizerConfig
    train: TrainConfig
    data_arch: str
    data: DataConfig
    runtime: RuntimeConfig
    note: str
    # These have defaults, so they have to come last. "constant" is what every
    # config did before schedules existed, so no existing file has to change
    # and no recorded run changes meaning.
    schedule_arch: str = "constant"
    schedule: ScheduleConfig = ConstantScheduleConfig()
