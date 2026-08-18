import contextlib

import torch

MATMUL_PRECISIONS = ("highest", "high", "medium")

# bf16 only, deliberately. fp16 has fp32's precision but a much narrower exponent
# range, so small gradients underflow to zero and it needs a GradScaler to be
# trainable at all. bf16 keeps fp32's exponent range and spends the bits on
# mantissa instead, which is why it needs no scaler and why the lecture uses it.
AUTOCAST_DTYPES = {"bf16": torch.bfloat16}

def autocast_context(device, dtype: str | None = None):
    """A context manager for the forward pass. `None` means fp32 — a nullcontext.

    Returns rather than applies, because unlike `set_matmul_precision` this isn't
    a global: autocast is scoped, and *what it's scoped around* is the whole
    subtlety. It wraps the forward pass and the loss, and nothing else —
    `backward()` and `optimiser.step()` go outside it. That isn't a style
    preference. Autograd records the dtype each op ran in, so the backward pass
    already runs in the matching precision without being told; putting it inside
    the context is what the torch docs specifically warn against. And the step
    stays fp32 because the parameters were never cast — autocast casts op
    *inputs*, so the master weights and their grads are fp32 throughout, which is
    what keeps a small update from vanishing when it's added to a large weight.

    Built by the caller and passed down, like `eval_batches` and the resolved
    device, so `loop.py` keeps its two-import list and its ignorance of config
    plumbing. The returned object is re-enterable, so one instance serves every
    iteration of a run.
    """
    if dtype is None:
        return contextlib.nullcontext()

    if dtype == "fp16":
        raise RuntimeError(
            "autocast_dtype 'fp16' needs a GradScaler, which this harness does "
            "not have — without loss scaling, fp16 gradients underflow to zero. "
            "Use 'bf16', which has fp32's exponent range and needs no scaler."
        )

    if dtype not in AUTOCAST_DTYPES:
        raise RuntimeError(
            f"Unknown autocast dtype {dtype!r} requested. "
            f"Valid values are null and {', '.join(AUTOCAST_DTYPES)}"
        )

    return torch.autocast(
        device_type=torch.device(device).type, dtype=AUTOCAST_DTYPES[dtype]
    )

def set_matmul_precision(precision: str = "highest") -> None:
    """How fp32 matmuls may be computed. Process-global, so callers set it per run.

    Lives beside `resolve_device` because it's the same kind of statement — not
    "what to compute" but "on what, and how exactly". The loop stays ignorant of
    it: a global torch setting isn't something a function that takes objects and
    returns facts should be reaching for.

    Callers must call this on *every* run rather than only when the value is
    non-default. It sets process state, and `sweep.py` drives many runs in one
    process, so a point that skipped the call would silently inherit whatever the
    previous point asked for.

    Validated here rather than left to torch so an unknown value fails next to
    the config that named it, the same way `resolve_device` rejects a bad device.
    """
    if precision not in MATMUL_PRECISIONS:
        raise RuntimeError(
            f"Unknown matmul precision {precision!r} requested. "
            f"Valid strings are {', '.join(MATMUL_PRECISIONS)}"
        )
    torch.set_float32_matmul_precision(precision)

def compile_model(model, enabled: bool = False):
    """The module to *run*. The caller keeps the original, which is what it saves.

    That split is the whole reason this is a named function rather than an inline
    ternary. `torch.compile` returns an `OptimizedModule` wrapper holding the real
    module as `_orig_mod`, and `nn.Module.state_dict` prefixes by child name, so a
    compiled model's keys come out as `_orig_mod.weight` instead of `weight`. A
    checkpoint written from the wrapper won't load into an uncompiled model —
    which is every later analysis, every resumed run, and every test. So entry
    points compile into a *second* name and go on checkpointing the first.

    The parameters are the same tensor objects either way, so an optimiser built
    from the original updates exactly what the compiled module runs.

    Sampling stays on the original for a different reason: generation grows the
    context by one token per step, and a new sequence length is a new shape, so a
    compiled `generate` recompiles its way through the sample.
    """
    return torch.compile(model) if enabled else model

def resolve_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    elif preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    elif preferred != "auto":
        raise RuntimeError(f"Unknown device type {preferred} requested. Valid strings are 'auto, cpu, cuda, mps'")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")