import contextlib

import torch

MATMUL_PRECISIONS = ("highest", "high", "medium")

# bf16 only. fp16 has a much narrower exponent range than fp32, so small
# gradients underflow to zero, and it needs a GradScaler to train at all. bf16
# keeps fp32's exponent range and gives up mantissa bits instead, which is why
# it needs no scaler and why the lecture uses it.
AUTOCAST_DTYPES = {"bf16": torch.bfloat16}

def autocast_context(device, dtype: str | None = None):
    """A context manager for the forward pass. `None` means fp32 (a nullcontext).

    Returned rather than applied, because unlike `set_matmul_precision` this
    isn't global: autocast only applies inside a block, and what goes inside the
    block matters. It should wrap the forward pass and the loss and nothing else;
    `backward()` and `optimiser.step()` go outside it. Autograd remembers the
    dtype each op ran in, so backward already uses the right precision, and the
    torch docs specifically warn against running it under autocast. The step
    stays fp32 because the parameters are never cast (autocast casts the inputs
    to each op), so the weights and their gradients are fp32 throughout. That's
    what stops a small update from disappearing when added to a large weight.

    The caller builds this and passes it down, like `eval_batches` and the
    device, so `loop.py` doesn't need to know about the config. The returned
    object can be entered repeatedly, so one instance does for a whole run.
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
    """How fp32 matmuls may be computed. Global to the process, so set it per run.

    Lives next to `resolve_device` because it's the same kind of setting: not
    what to compute, but where and how. The loop doesn't touch it, since it's a
    global torch setting.

    Callers must call this on *every* run, not just when the value isn't the
    default. It sets process-wide state, and `sweep.py` runs many points in one
    process, so a point that skipped it would silently get whatever the previous
    point used.

    Checked here rather than left to torch, so an unknown value gives a clear
    error, the same way `resolve_device` rejects a bad device.
    """
    if precision not in MATMUL_PRECISIONS:
        raise RuntimeError(
            f"Unknown matmul precision {precision!r} requested. "
            f"Valid strings are {', '.join(MATMUL_PRECISIONS)}"
        )
    torch.set_float32_matmul_precision(precision)

def compile_model(model, enabled: bool = False):
    """The module to *run*. The caller keeps the original, and saves that one.

    This is a named function rather than an inline ternary to make that split
    obvious. `torch.compile` returns an `OptimizedModule` wrapper that holds the
    real module as `_orig_mod`, and `state_dict` prefixes keys with the child's
    name, so a compiled model's keys come out as `_orig_mod.weight` instead of
    `weight`. A checkpoint saved from the wrapper won't load into an uncompiled
    model, which is what every later analysis, resumed run and test uses. So
    entry points keep the compiled model under a *second* name and save the
    first.

    The parameters are the same tensors either way, so an optimiser built from
    the original updates exactly what the compiled module runs.

    Sampling also uses the original, for a different reason: generation adds
    one token per step, and each new sequence length is a new shape, so a
    compiled `generate` would keep recompiling.
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