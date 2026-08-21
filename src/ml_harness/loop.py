"""The training loop. No I/O, no filesystem, no argparse.

Takes objects, returns facts. Everything path-shaped lives in the caller
(`nanogpt_conor/train.py`), which is why nothing here imports `store`, `os` or
`datetime`. If one of those imports comes back, something leaked downward.

The division of labour with the caller:
  * `logger`  — metrics. The loop owns what counts as a metric and when to emit it.
  * `on_eval` — artifacts. The caller decides what to do with an eval result:
                checkpoint, sample, print, or nothing at all.

Seeding, device resolution, autocast-context construction and eval-batch
construction all happen in the caller, *before* this is called — see the note in
train.py about why seeding has to precede model construction.
"""

from __future__ import annotations

import contextlib

import torch

from ml_harness.config import RunConfig


def _identity(tensor):
    """Default `reduce_loss`. A named function rather than a lambda so a
    traceback through it says what it is."""
    return tensor


def _set_grad_sync(model, enabled: bool) -> None:
    """Tell a DDP-wrapped model whether this backward should all-reduce.

    Duck-typed on purpose, and it is the same duck-typing the logger and
    `on_eval` get: `require_backward_grad_sync` is DDP's own flag, absent on a
    plain `nn.Module`, so a single-process run takes the `hasattr` branch and
    nothing else changes. Importing DDP to `isinstance` against would put a
    distributed concept in a module whose entire claim is that it has none.

    **Set before the forward pass, not before the backward.** DDP reads this in
    `forward()` to decide whether to call `prepare_for_backward`; setting it
    afterwards is read too late and silently does nothing.

    Why it matters: without it, DDP all-reduces on *every* micro-batch, so a run
    at `accum_steps=8` pays eight full gradient exchanges per optimiser step
    instead of one. Purely throughput — the arithmetic is identical either way,
    which is what makes it so easy to leave broken.
    """
    if hasattr(model, "require_backward_grad_sync"):
        model.require_backward_grad_sync = enabled


@torch.no_grad()
def evaluate(model, batches, amp=None) -> float:
    """Mean loss over a fixed set of batches, with dropout off.

    The two mechanisms here are unrelated and both are needed: `no_grad` stops
    autograd storing a graph nothing will backward through, and `eval()` turns
    dropout off. Neither implies the other — dropout still drops under no_grad.

    Restores the mode it found rather than forcing `train()`: a caller that
    loaded a checkpoint to measure it shouldn't have dropout switched back on by
    a function it asked for a number from. `finally`, so that survives a raise
    part-way through the batches.

    `amp` is the autocast context, or None for fp32. Evaluating under the same
    precision the run trains in is the point: a val loss measured in fp32 for a
    bf16 model is a number that run never actually produced. `runtime.autocast_dtype`
    is part of the config hash, so two precisions are two experiments and their
    losses were never going to be pooled anyway.
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

    logger:  anything with .log(step, split, metric, value). A list-appending fake
             is a complete implementation, which is what makes the smoke test cheap.
    on_eval: callable(step, losses) where losses is {split: mean_loss}. Called after
             the eval metrics have been logged. A no-op is valid.
    amp:     the autocast context from `device.autocast_context`, or None for fp32.
             Built by the caller for the same reason `eval_batches` and the device
             are — this module reads a RunConfig, it doesn't assemble one's
             consequences. Defaulting to None keeps every existing caller correct.
    lr_schedule: callable(step) -> lr, from `registry.build_lr_schedule`, or None
             to leave the optimiser's lr exactly as it was built. Built by the
             caller for the same reason `amp` is; a closure rather than a config
             so this module never learns what a cosine is.
    reduce_loss: callable(tensor) -> tensor, applied to the batch loss before it
             is logged, or None for identity. Exists for data parallelism: under
             DDP this model saw only its rank's share of `train.batch_size`, so
             the loss it computed is that share's mean and logging it unreduced
             would make `metrics.jsonl` a record of rank 0's batches rather than
             the step's. Injected as a plain tensor function for the same reason
             `amp` and `lr_schedule` are — the caller assembles the config's
             consequences, and this module never learns that ranks exist.
    """
    amp = amp if amp is not None else contextlib.nullcontext()
    reduce_loss = reduce_loss if reduce_loss is not None else _identity

    train_cfg = cfg.train
    model_cfg = cfg.model

    max_iters = train_cfg.max_iters
    eval_interval = train_cfg.eval_interval
    grad_clip = train_cfg.grad_clip

    # Resolved once, outside the loop: config.TrainConfig has already checked
    # that the division is exact, so there is nothing to guard per step. None
    # collapses to accum_steps == 1, i.e. exactly the single-batch path.
    micro_batch_size = train_cfg.micro_batch_size or train_cfg.batch_size
    accum_steps = train_cfg.batch_size // micro_batch_size

    # -1 rather than 0: if max_iters is 0 the loop body never runs, and "final step
    # 0" would be a lie. The caller checkpoints against this.
    final_step = -1

    for step in range(max_iters):
        # First, so the lr this step trains at is the one it gets logged with.
        # Written into every param_group rather than group 0: there is one group
        # today, and silently scheduling only the first would be a nasty surprise
        # for whoever splits weight decay out.
        if lr_schedule is not None:
            lr = lr_schedule(step)
            for group in optimiser.param_groups:
                group["lr"] = lr

        # Once, before the micro-batches rather than between forward and
        # backward: gradients have to survive from one micro-batch to the next,
        # and that accumulation *is* the mechanism.
        optimiser.zero_grad(set_to_none=True)
        # Accumulated on the device and read back once, after the loop. A
        # per-micro-batch `.item()` would force a sync on every one of them,
        # which is a real cost at accum_steps of any size.
        batch_loss = torch.zeros((), device=device)

        for micro_step in range(accum_steps):
            # Only the last micro-batch of the step syncs gradients across ranks;
            # the ones before it accumulate locally. A no-op unless the model is
            # DDP-wrapped — see `_set_grad_sync`, including why this has to
            # precede the forward pass rather than the backward.
            _set_grad_sync(model, micro_step == accum_steps - 1)
            # Drawn one micro-batch at a time rather than in a single
            # `n_iters=accum_steps` call: fetching them all up front puts every
            # micro-batch on the device at once, which is half of what
            # accumulation exists to avoid.
            xb, yb = data.get_batch(
                "train", micro_batch_size, model_cfg.block_size, str(device)
            )[0]
            # Forward and loss inside; backward and step outside. Autograd recorded
            # the dtype each op ran in, so the backward pass matches without being
            # told, and the parameters were never cast, so the update stays fp32.
            with amp:
                _, loss = model(xb, yb)
            # The model returns a *mean* loss, so summing accum_steps of them
            # unscaled would give a gradient accum_steps times too large.
            # Divided, the accumulated gradient is the full batch's.
            (loss / accum_steps).backward()
            batch_loss += loss.detach() / accum_steps

        # Between backward and step, which is the only place it means anything:
        # the gradients have to exist to be measured, and the optimiser has to
        # see the clipped ones. Rescales the whole parameter set by one factor
        # when its global L2 norm exceeds grad_clip, so the direction is
        # preserved and only the length is capped. After the accumulation, so
        # it clips the whole batch's gradient — clipping each micro-batch's
        # share would be a different and much stronger rule.
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        final_step = step

        # the cheap one: this batch's loss, already computed. Still the mean
        # over `batch_size` sequences whether it took one forward pass or many,
        # so the metric means the same thing on either side of this field —
        # and, with `reduce_loss` supplied, whether that batch was computed by
        # one process or by eight. The reduction goes here rather than around
        # the accumulator above so it costs one collective per step, not one
        # per micro-batch.
        logger.log(step, "train", "batch_loss", reduce_loss(batch_loss).item())
        # Read back off the optimiser rather than from `lr` above, so it's logged
        # unconditionally and reports what the optimiser actually used — an
        # unscheduled run gets a flat line rather than a missing metric, and a
        # schedule that failed to apply shows up here instead of nowhere.
        logger.log(step, "train", "lr", optimiser.param_groups[0]["lr"])

        if step % eval_interval == 0 or step == max_iters - 1:
            # the expensive one: averaged over the fixed eval set
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
