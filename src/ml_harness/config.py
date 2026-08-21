from dataclasses import dataclass, field

@dataclass(frozen=True)
class OptimiserConfig:
    """ Hyperparameters for an optimiser """

    lr: float

@dataclass(frozen=True)
class AdamWConfig(OptimiserConfig):
    """ Two scalars rather than torch's `betas` tuple, so each is addressable on
    its own: a sweep axis is a dotted `--set optim.beta2=0.95`, and one element
    of a list isn't reachable that way. registry._build_adamw pairs them back up.

    Defaulted to torch's own values, so an `optim` block that names only `lr`
    still loads — which every config and every already-recorded run does. """

    beta1: float = 0.9
    beta2: float = 0.999
    # GPT-2's rule, applied in registry.decay_param_groups: this is the decay
    # for tensors of dim >= 2 — weight matrices and embeddings — and biases and
    # LayerNorm gains get 0.0. The dimension is a proxy for the distinction that
    # matters: a matrix maps between spaces and shrinking it is regularisation,
    # while a bias or a gain is a per-unit offset that decay just drags toward
    # an arbitrary zero.
    #
    # 0.1 is the GPT-2/nanoGPT value. It is *not* what runs recorded before this
    # field trained under: `_build_adamw` didn't pass `weight_decay` at all, so
    # torch's AdamW default of 0.01 applied to every parameter, biases included.
    # Those configs still load, but they no longer reproduce their original run
    # — and no default could preserve that, since the grouping changes the
    # numerics on its own.
    #
    # On AdamWConfig rather than the OptimiserConfig base for the same reason
    # train.grad_clip isn't on it: the base carries what every optimiser has.
    # There is a mechanical reason too — `_adapt` builds a torch optimiser as
    # `optim_cls(params, **asdict(cfg))`, so a base-level field would silently
    # become a flat, ungrouped `weight_decay=` kwarg for every optimiser built
    # that way, and one optimiser would group where the rest didn't.
    weight_decay: float = 0.1

    def __post_init__(self):
        # Negative decay is anti-regularisation: it grows the weights every
        # step, and the run diverges slowly enough to look like a bad lr.
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )

@dataclass(frozen=True)
class ScheduleConfig:
    """ blank learning-rate schedule config.

    Fourth use of the discriminator pattern: `schedule_arch` names which subclass
    the `schedule` block parses into. Empty, like ModelConfig — "hold lr fixed"
    and "warm up then cosine to a floor" share no parameter.

    None of these carry the learning rate itself. `optim.lr` is the *peak*, the
    optimiser is still built with it, and a schedule only says how the lr moves
    relative to that. Two places naming a learning rate would be one too many.
    """

@dataclass(frozen=True)
class ConstantScheduleConfig(ScheduleConfig):
    """Hold `optim.lr` for the whole run. Nothing to configure.

    The default, and what every run before schedules existed did.
    """

@dataclass(frozen=True)
class CosineScheduleConfig(ScheduleConfig):
    """Linear warmup, cosine decay to a floor, then flat — the GPT-3 schedule.

    Measured in steps rather than tokens. The paper's schedule is in tokens
    because its batch size changes during the run; ours doesn't, so steps and
    tokens differ by a constant and the distinction buys nothing.

    Every field is defaulted, so `--set schedule_arch=cosine` alone is a
    complete, valid schedule: no warmup, decaying over the whole run to 10% of
    peak.
    """

    # Steps of linear ramp from ~0 up to peak. The ramp exists because Adam's
    # second-moment estimate is nonsense for the first few dozen steps — it's
    # averaging over almost no history, so early updates are both large and
    # badly scaled, at exactly the moment the weights are least able to absorb
    # them.
    warmup_steps: int = 0
    # The step at which the cosine reaches the floor, counted from step 0 with
    # the warmup *included* (nanoGPT's `lr_decay_iters`, so the numbers port).
    # None = train.max_iters, i.e. "decay across the whole run" — which is what
    # you usually want, and which keeps the schedule's shape when max_iters
    # becomes a sweep axis. An explicit value is for the case that isn't:
    # decaying to the floor early and holding there.
    decay_steps: int | None = None
    # The floor, as a fraction of peak. A fraction rather than an absolute lr so
    # the shape is scale-invariant: `--set optim.lr=1e-3` moves peak and floor
    # together instead of quietly changing the decay ratio, and no lr can put
    # the floor above the peak and turn this into a ramp.
    #
    # It bounds the *decay*, not the whole run: warmup ramps from ~0 and so
    # passes below the floor on its way up. Clamping the warmup to the floor
    # would start it at 10% of peak, which is most of the way to no warmup.
    min_lr_frac: float = 0.1

    def __post_init__(self):
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0.0 <= self.min_lr_frac <= 1.0:
            raise ValueError(
                f"min_lr_frac is a fraction of optim.lr and must be in [0, 1], "
                f"got {self.min_lr_frac}"
            )
        # Only checkable when decay_steps is stated. The None case resolves
        # against train.max_iters at build time and can land below warmup
        # legitimately — a smoke run is 4 iters — so the schedule function
        # guards that rather than raising on it.
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
    # What generation is seeded with at each eval. Encoded through the run's own
    # tokenizer, so it has to be text rather than an id: id 0 is '\n' under a
    # character vocabulary and byte 0x00 under BPE, and only one of those is a
    # sensible thing to continue from. Default newline = "start of a line",
    # which is the character tokenizer's old hardcoded behaviour.
    sample_prompt: str = "\n"
    # Truncate sampling to the k likeliest tokens. A decoding knob, not an
    # architectural one, which is why it lives here and not on a ModelConfig —
    # it changes what a checkpoint writes into samples/, never what it learned.
    # None = sample the full distribution, the behaviour every existing run has.
    # Calibrated for a large vocabulary: HuggingFace's library-wide default is
    # 50, against GPT-2's 50257. Under a char tokenizer (~65 ids) it barely
    # bites, so a run that sets it wants a BPE tokenizer to mean much by it.
    top_k: int | None = None
    # Whether each eval writes a generation to samples/. On for a single run —
    # for a character-level LM the samples are the qualitative signal and
    # they're nearly free. Off for a sweep, where 200 generations across 50
    # configs is noise, and where generation is a real fraction of a short run's
    # wall time. A config field rather than a parameter on `run()` so a sweep
    # sets it through the same validated override path as everything else.
    write_samples: bool = True
    # Max global L2 norm of the gradient, rescaled down when it exceeds this.
    # None = no clipping, which is what every run before this field did.
    #
    # On TrainConfig rather than OptimiserConfig because clipping isn't an
    # optimiser's business: it happens between backward and step, applies
    # identically whichever optimiser runs next, and is the loop that does it.
    # Putting it on the base OptimiserConfig would say every optimiser has a
    # clipping rule, which is not what the code does.
    #
    # A single bad batch can produce a gradient orders of magnitude larger than
    # the typical one; unclipped, Adam's moment estimates carry that spike for
    # thousands of steps. 1.0 is the nanoGPT/GPT-3 value.
    grad_clip: float | None = None
    # Sequences per forward/backward pass, when that has to be smaller than the
    # batch the update is computed from. `batch_size` keeps exactly one meaning
    # — sequences per optimiser step — and this says how many of them fit in
    # memory at once; the loop runs `batch_size // micro_batch_size` backward
    # passes before each step, scaling each loss so the accumulated gradient
    # equals the full batch's.
    #
    # None = no accumulation, which is what every run before this field did.
    # Spelled this way round rather than as an accumulation count (nanoGPT's
    # `grad_accum_steps`, where `batch_size` becomes the micro batch) because
    # that would silently redefine `batch_size` across already-recorded runs,
    # and leave the batch a run actually trained on as a product of two fields.
    #
    # A hardware knob, not an experimental one, which is why it is in
    # provenance.GROUP_EXCLUDES: splitting a batch to fit a smaller card is the
    # same experiment. Not the same *run* — the RNG draws a different sequence
    # of batches and floats accumulate in a different order — which is exactly
    # the standing `runtime.seed` is held to.
    micro_batch_size: int | None = None

    def __post_init__(self):
        # 0.0 would zero every gradient and train nothing, silently — the run
        # would look healthy and the loss would sit flat. Fail next to the
        # config that said it.
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        # Checked here rather than in the loop so a bad value fails before a
        # model is built, rather than at step 0 of a run that has already
        # written a run directory. An indivisible split is a config error and
        # not a rounding question: silently training on batch_size rounded down
        # to a multiple would make the recorded batch_size a lie.
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
    """ blank Data config.

    Third use of the discriminator pattern: `data_arch` names which subclass the
    `data` block parses into. Empty for a reason — there is no field every
    dataset has. `cache` is meaningless for a corpus that's generated rather than
    read, `n_problems` is meaningless for a file, and even `split_frac` belongs
    to the subclasses, because how a dataset divides is a property of what it's
    made of.

    Only the generic subclass lives here. A dataset that names a *task* —
    `gpt_addition`'s AdditionDataConfig — lives with that task, and reaches
    `load_run_config` through a passed-in DATAS table, the same way GPTConfig
    reaches it through ARCHES.
    """

@dataclass(frozen=True)
class FileDataConfig(DataConfig):
    """A text corpus read off disk, split by position."""

    path: str
    split_frac: float
    # Whether to memoise the encoded corpus on disk. The cache is content
    # addressed, so this exists for the one case the key can't cover: an encoder
    # whose *behaviour* changed without its recipe changing. Bumping
    # CACHE_VERSION is the real fix; this is the switch for bisecting to it.
    cache: bool = True

@dataclass(frozen=True)
class RuntimeConfig:
    """ Seeding, devices """

    seed: int
    eval_seed: int
    device: str
    # How fp32 matmuls are allowed to be computed. "highest" is torch's default
    # and keeps the full 24-bit mantissa; "high"/"medium" trade mantissa bits for
    # throughput — on CUDA (Ampere and later) that's TF32 on the tensor cores,
    # which is where the lecture's speedup comes from. A no-op on MPS and CPU,
    # measured; kept in the config anyway so a run on a rented GPU is a config
    # change rather than an edit.
    #
    # Runtime rather than optim or model: it changes no hyperparameter and no
    # architecture, only how the same arithmetic is carried out — the same
    # category as which device the run lands on. It is *not* in GROUP_EXCLUDES,
    # so two precisions are two experiments: the numerics differ, and averaging
    # their losses would be averaging different computations.
    #
    # Defaulted so every config file written before this field still loads under
    # dacite's strict=True, and so the default preserves existing numerics.
    matmul_precision: str = "highest"
    # Mixed precision for the forward pass: None (everything in fp32, the
    # default) or "bf16". Distinct from matmul_precision, which only says how
    # *fp32* matmuls may be computed internally — this changes the dtype the
    # activations are actually stored in, so it moves memory as well as flops.
    #
    # Weights and gradients stay fp32 either way: autocast casts op inputs, not
    # parameters, so the optimiser never sees a bf16 tensor. Only bf16 is
    # accepted; fp16 would need a GradScaler this harness doesn't have.
    autocast_dtype: str | None = None
    # Whether to run the training step through `torch.compile`. Off by default,
    # and that default is measured rather than cautious: on MPS the inductor
    # backend currently emits a LayerNorm kernel ~16x slower than the vendor one
    # eager dispatches to, which for a 9-LayerNorm GPT is a ~37% regression. It's
    # a large win on CUDA, where the codegen is mature — hence a config field
    # rather than a decision baked into the entry point.
    compile: bool = False

@dataclass(frozen=True)
class ModelConfig:
    """ blank Model config """

@dataclass(frozen=True)
class TokenizerConfig:
    """ blank Tokenizer config.

    Same discriminator pattern as ModelConfig: `tokenizer_arch` names which of
    the subclasses below the `tokenizer` block parses into. Two of them are
    empty, which is not a mistake — "the alphabet is whatever the corpus
    contains" and "one id per byte" are complete descriptions.

    Not quite empty, though: padding is the one option every tokenizer takes, so
    it sits here rather than being repeated five times below.
    """

    # The absolute `vocab_size` to report, if larger than the real one. Buys
    # nothing linguistic — the ids it adds are unreachable and their embedding
    # rows train on nothing — and buys a number that factorises the way a GPU
    # wants: 50257 is prime-ish and awkward, 50304 is 64 x 786. The tokenizer
    # owns it rather than the model because `vocab_size` is the single number
    # the embedding table is built from, so a model that padded privately would
    # disagree with the tokenizer that named its ids. See
    # `tokenizers.base.pad_vocab_size`, and note it lands in `to_dict` and
    # therefore in the fingerprint, which is what keeps a padded run and an
    # unpadded one out of the same cache entry.
    #
    # `kw_only` so a base-class default doesn't force one onto the required
    # fields the subclasses add (`alphabet`, `vocab_size`, `encoding`).
    pad_vocab_to: int | None = field(default=None, kw_only=True)

@dataclass(frozen=True)
class CharConfig(TokenizerConfig):
    """Character-level, alphabet read off the corpus. Nothing to configure."""

@dataclass(frozen=True)
class CharFixedConfig(TokenizerConfig):
    """Character-level over a stated alphabet.

    Worth the second entry rather than a nullable field on CharConfig: the
    difference is whether the ids depend on the corpus, which is the thing a
    checkpoint's readability turns on.
    """

    alphabet: str

@dataclass(frozen=True)
class ByteConfig(TokenizerConfig):
    """One id per byte. Nothing to configure."""

@dataclass(frozen=True)
class RegexBPEConfig(TokenizerConfig):
    """BPE trained on this run's corpus.

    Note this trains on every run, before stage C's cache exists — fine for a
    small corpus, minutes for a large one.
    """

    vocab_size: int
    # None means the tokenizer's own default (the whole text as one chunk).
    # Kept nullable rather than repeating the pattern string here, so the
    # default lives in exactly one place.
    split_pattern: str | None = None
    # Declared here, numbered directly above the 256 bytes in this order, and
    # counted against `vocab_size`. Order is the id assignment, so reordering
    # the list is a different tokenizer, not a different way of writing one.
    special_tokens: list[str] | None = None

@dataclass(frozen=True)
class TiktokenConfig(TokenizerConfig):
    """BPE over a tiktoken encoding's learned ranks — `gpt2`, `cl100k_base`."""

    encoding: str
    # None means the encoding's own pattern, which is what you want; GPT-2's is
    # case-sensitive and GPT-4's isn't, and crossing them silently produces
    # wrong ids.
    split_pattern: str | None = None
    # Names from the encoding's own special table, e.g. ["<|endoftext|>"], which
    # keep the ids OpenAI gave them rather than being appended. Opting in is what
    # takes GPT-2 from 50256 to 50257: an unused id is an embedding row trained
    # on nothing, so a run only pays for the specials it will actually emit.
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
    # Defaulted, and last because of it: every other field is required, and a
    # default can't precede a non-default. "constant" reproduces the behaviour
    # of every config written before schedules existed, so no file has to change
    # and no recorded run's meaning does.
    schedule_arch: str = "constant"
    schedule: ScheduleConfig = ConstantScheduleConfig()
