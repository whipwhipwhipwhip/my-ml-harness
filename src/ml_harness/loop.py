"""The training loop. No I/O, no filesystem, no argparse.

It takes objects and returns results. Anything to do with paths lives in the
caller (e.g. `testing.run`, or a project's `train.py`), which is why nothing here
imports `store`, `os` or `datetime`. If one of those imports shows up here,
something is in the wrong place.

How the work is split with the caller:
  * `logger`  - metrics. The loop decides what counts as a metric and when to
                log it.
  * `on_eval` - everything else. The caller decides what to do with an eval
                result: checkpoint, sample, print, or nothing.

Seeding, picking the device, building the autocast context and drawing the eval
batches all happen in the caller *before* this is called (see `runner.build_run`
for why seeding has to come before building the model).
"""

from __future__ import annotations

import contextlib

import torch

from ml_harness.config import RunConfig


def _identity(tensor):
    """Default `reduce_loss`. A named function rather than a lambda so it's
    recognisable in a traceback."""
    return tensor


def _set_grad_sync(model, enabled: bool) -> None:
    """Tell a DDP-wrapped model whether this backward should all-reduce.

    Duck-typed, like the logger and `on_eval`. `require_backward_grad_sync` is
    DDP's own flag and a plain `nn.Module` doesn't have it, so in a
    single-process run this does nothing. Importing DDP to do an `isinstance`
    check would bring distributed code into a module that otherwise knows
    nothing about it.

    **It has to be set before the forward pass, not before the backward.** DDP
    reads it in `forward()` to decide whether to call `prepare_for_backward`, so
    setting it afterwards silently does nothing.

    Without it, DDP all-reduces on *every* micro-batch, so with `accum_steps=8`
    you'd do eight gradient exchanges per optimiser step instead of one. It only
    affects speed (the maths is identical), which makes it easy to miss if it
    breaks.
    """
    if hasattr(model, "require_backward_grad_sync"):
        model.require_backward_grad_sync = enabled


@torch.no_grad()
def evaluate(model, batches, amp=None) -> float:
    """Mean loss over a fixed set of batches, with dropout off.

    Both of these are needed, and they do different things: `no_grad` stops
    autograd from storing a graph, and `eval()` turns dropout off. Neither
    implies the other (dropout still applies under no_grad).

    Puts the model back in whatever mode it was in, rather than always calling
    `train()`, so that evaluating a loaded checkpoint doesn't switch dropout on.
    It's in a `finally` so this still happens if something raises.

    `amp` is the autocast context, or None for fp32. Eval runs in the same
    precision as training, otherwise you'd get a val loss for a bf16 model that
    it never actually produced. `runtime.autocast_dtype` is part of the config
    hash, so different precisions are different experiments anyway.
    """
    amp = amp if amp is not None else contextlib.nullcontext()
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for x, y in batches:
            with amp:
                _, loss = model(x, y)
            total += loss.item()  # .item() -> python float, so json.dumps can't choke
        return total / len(batches)
    finally:
        model.train(was_training)


def train(
    model,
    optimiser,
    data,
    cfg: RunConfig,
    device,
    eval_batches: dict[str, list],
    logger,
    on_eval,
    amp=None,
    lr_schedule=None,
    reduce_loss=None,
) -> dict:
    """Run the training loop. Raises on failure; the caller decides what that means.

    logger:  anything with .log(step, split, metric, value). A fake that
             appends to a list is enough, which keeps the tests cheap.
    on_eval: callable(step, losses) where losses is {split: mean_loss}. Called
             after the eval metrics have been logged. A no-op is fine.
    amp:     the autocast context from `device.autocast_context`, or None for
             fp32. The caller builds it, like `eval_batches` and the device: this
             module reads a RunConfig but doesn't build things from it.
    lr_schedule: callable(step) -> lr, from `registry.build_lr_schedule`, or
             None to leave the optimiser's lr as it was built. Also built by the
             caller, and a function rather than a config so this module never
             needs to know what a cosine schedule is.
    reduce_loss: callable(tensor) -> tensor, applied to the batch loss before
             it's logged, or None to log it as is. This is for data parallelism:
             under DDP each process only sees its share of `train.batch_size`,
             so without reducing, `metrics.jsonl` would record rank 0's loss
             rather than the whole step's. It's passed in as a plain function so
             this module never has to know about ranks.
    """
    amp = amp if amp is not None else contextlib.nullcontext()
    reduce_loss = reduce_loss if reduce_loss is not None else _identity

    train_cfg = cfg.train
    model_cfg = cfg.model

    max_iters = train_cfg.max_iters
    eval_interval = train_cfg.eval_interval
    grad_clip = train_cfg.grad_clip

    # TrainConfig has already checked that this divides exactly. None gives
    # accum_steps == 1, i.e. the normal single-batch path.
    micro_batch_size = train_cfg.micro_batch_size or train_cfg.batch_size
    accum_steps = train_cfg.batch_size // micro_batch_size

    # -1 rather than 0: if max_iters is 0 the loop never runs, and "final step
    # 0" would be wrong. The caller saves this in the checkpoint.
    final_step = -1

    for step in range(max_iters):
        # First, so the lr this step trains with is the one that gets logged.
        # Set on every param group, not just group 0: `decay_param_groups` makes
        # two, and scheduling only the first would leave the biases and
        # LayerNorm gains at the peak lr for the whole run.
        if lr_schedule is not None:
            lr = lr_schedule(step)
            for group in optimiser.param_groups:
                group["lr"] = lr

        # Once, before the micro-batches rather than between forward and
        # backward, since the gradients have to accumulate across micro-batches.
        optimiser.zero_grad(set_to_none=True)
        # Summed on the device and read back once, after the loop. Calling
        # `.item()` per micro-batch would force a device sync every time.
        batch_loss = torch.zeros((), device=device)

        for micro_step in range(accum_steps):
            # Only the last micro-batch of the step syncs gradients across
            # ranks; the others accumulate locally. Does nothing unless the model
            # is wrapped in DDP. See `_set_grad_sync`, including why this has to
            # come before the forward pass.
            _set_grad_sync(model, micro_step == accum_steps - 1)
            # One micro-batch at a time rather than a single
            # `n_iters=accum_steps` call: fetching them all at once would put
            # every micro-batch on the device together, which is part of what
            # accumulation is meant to avoid.
            xb, yb = data.get_batch(
                "train", micro_batch_size, model_cfg.block_size, str(device)
            )[0]
            # Forward and loss inside autocast; backward and step outside.
            # Autograd remembers the dtype each op ran in, so backward matches
            # automatically, and the parameters were never cast, so the update
            # stays fp32.
            with amp:
                _, loss = model(xb, yb)
            # The model returns a *mean* loss, so adding up accum_steps of them
            # would give a gradient accum_steps times too big. Dividing makes the
            # accumulated gradient equal to the full batch's.
            (loss / accum_steps).backward()
            batch_loss += loss.detach() / accum_steps

        # Between backward and step: the gradients have to exist to be
        # measured, and the optimiser has to see the clipped ones. If the global
        # L2 norm is over grad_clip, all gradients are scaled down by the same
        # factor, so the direction stays the same and only the length is capped.
        # It comes after accumulation so it clips the whole batch's gradient;
        # clipping each micro-batch separately would be a different, much
        # stronger rule.
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        final_step = step

        # The cheap metric: this batch's loss, already computed. It's the mean
        # over `batch_size` sequences however many micro-batches there were,
        # and (with `reduce_loss`) however many processes shared it. Reducing
        # here rather than per micro-batch means one all-reduce per step.
        logger.log(step, "train", "batch_loss", reduce_loss(batch_loss).item())
        # Read from the optimiser rather than `lr` above, so it's always logged
        # and shows what the optimiser actually used. A run with no schedule gets
        # a flat line rather than no metric, and a schedule that didn't get
        # applied shows up here.
        logger.log(step, "train", "lr", optimiser.param_groups[0]["lr"])

        if step % eval_interval == 0 or step == max_iters - 1:
            # The expensive metric: averaged over the fixed eval set
            losses = {
                split: evaluate(model, batches, amp)
                for split, batches in eval_batches.items()
            }
            for split, value in losses.items():
                logger.log(step, split, "loss", value)

            on_eval(step, losses)

    # TODO(you): decide how much belongs here. The metrics file is the real record
    # and `summarize()` already reads it back, so this only needs to carry what the
    # caller can't recover from disk.
    return {"final_step": final_step, "steps_completed": final_step + 1}
