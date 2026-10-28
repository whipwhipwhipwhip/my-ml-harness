import copy
import hashlib
import json
import subprocess
import torch
import os

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

from tokenizers import Tokenizer, fingerprint


def git(REPO_ROOT, *args: str) -> str | None:
    """Run a git command in `REPO_ROOT`, or None if it fails for any reason.

    Lives here rather than in a shared utils package so `ml_harness/` doesn't
    depend on anything outside its own directory. Returns None on failure
    because this is only used for provenance: a run from a tarball with no
    `.git`, or on a machine without git, is still a valid run, and recording
    "unknown" is better than crashing.
    """
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def find_git_dirty(REPO_ROOT):
    """True/False if git answered, None if it couldn't.

    `collect_provenance` only records a patch when this is True, and None means
    "unknown" rather than "clean", so a run without git isn't recorded as
    reproducible.
    """
    is_dirty = git(REPO_ROOT, "status", "--porcelain")
    return bool(is_dirty) if is_dirty else None

# The distribution that ships `ml_harness/`. A project installs the harness as a
# dependency, so the project's `git_sha` doesn't cover the harness's code; the
# harness version recorded alongside it does.
HARNESS_DISTRIBUTION = "ml_harness"


def _dist_version(name: str) -> str | None:
    """The installed version of a distribution, or None if it isn't one.

    None rather than raising, like every other field here: provenance records
    what it can see, and a harness run from a source tree with nothing
    installed is still worth recording.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None

# What runs in a *group* can differ by and still count as the same experiment:
# the seed, the note, and how the batch was split to fit in memory. Everything
# else is part of the hash.
#
# `train.micro_batch_size` is here because it's a hardware setting: the update
# is computed from `train.batch_size` either way, and splitting it over four
# forward passes is the same experiment on a smaller card. It's not quite the
# same *run* (different batches get drawn, and floats are summed in a different
# order), but that's also true of `runtime.seed`, and `config_sha256` still
# tells them apart.
#
# `runtime.eval_seed` is left out on purpose. Different eval seeds mean
# different validation batches, so the losses can't be averaged, and averaging
# is the whole point of a group. Same for `runtime.device`: if a result differs
# between CPU and MPS, you want to be able to see that.
GROUP_EXCLUDES: tuple[tuple[str, ...], ...] = (
    ("runtime", "seed"),
    ("note",),
    ("train", "micro_batch_size"),
)


def _sha256_json(obj: Any) -> str:
    """sha256 of a canonical JSON rendering.

    `sort_keys` for the same reason `tokenizers.fingerprint` uses it: the hash
    shouldn't depend on dict order, which can change on a round trip through a
    file. `asdict` order is stable today, but nothing guarantees it.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _without(config: dict, paths: Iterable[Sequence[str]]) -> dict:
    """A copy of `config` with each dotted path removed, if present."""
    out = copy.deepcopy(config)

    for path in paths:
        node = out
        for part in path[:-1]:
            node = node.get(part)
            if not isinstance(node, dict):
                break
        else:
            node.pop(path[-1], None)

    return out


def config_sha256(config: dict) -> str:
    """Identity of an effective config: this exact run, seed included."""
    return _sha256_json(config)


def group_sha256(config: dict) -> str:
    """Identity of an *experiment*: the config minus the keys in GROUP_EXCLUDES.

    N seeds of one config share a `group_sha256`, so "mean +/- spread over
    seeds" is a groupby on one column rather than a hand-picked list of run ids.
    """
    return _sha256_json(_without(config, GROUP_EXCLUDES))


def collect_provenance(
    device: str,
    vocab_size: int,
    REPO_ROOT: Path,
    sha256,
    tokenizer: Tokenizer,
    config: dict | None = None,
    world_size: int = 1,
) -> dict:
    """Facts about the environment the run happened in: observed, not chosen.

    The data hash matters because `*.txt` is gitignored, so the corpus isn't
    in git and this is the only record of what the model was trained on.

    The tokenizer hash matters for the same reason: the same corpus with two
    different vocabularies is two different training sets, and `vocab_size`
    alone can't tell them apart (two BPE tokenizers with 50256 ids can be
    completely different). `tokenizer_kind` is the class, which isn't always
    the same as the config's `tokenizer_arch` (`char_fixed` builds a `char`).

    `config` is the exception, since it's chosen rather than observed, but its
    *hashes* are useful facts about the run, and analysis code already reads
    this file. Keeping them here rather than in `run_id` means run ids never
    collide (re-running a config still gets a new directory), and `load_runs`
    shows them as `provenance.*` columns. Optional, so callers without a
    RunConfig still work.
    """


    prov = {
        "git_sha": git(REPO_ROOT, "rev-parse", "HEAD"),
        "git_dirty": find_git_dirty(REPO_ROOT),
        "torch_version": torch.__version__,
        # The two dependencies whose code changes what a run *means*, not just
        # how fast it is. `tokenizers` is a separate package, so a change to
        # its merge loop doesn't show up in this repo's history, and two runs
        # with the same corpus and `tokenizer_sha256` could still have been
        # encoded by different code. The fingerprint covers the recipe; this
        # covers the code that reads it.
        "harness_version": _dist_version(HARNESS_DISTRIBUTION),
        "tokenizers_version": _dist_version("tokenizers"),
        "device_resolved": device,
        "vocab_size": vocab_size,
        "data_sha256": sha256,
        "tokenizer_kind": tokenizer.kind,
        "tokenizer_sha256": fingerprint(tokenizer),
        "hostname": os.uname().nodename,
        # How many processes shared this run's batch. This is provenance rather
        # than config because `train.batch_size` is global, so the same config
        # on eight ranks or one is the *same experiment* and has the same
        # `group_sha256`. Only the machine differs, and that's what this file
        # records. Defaults to 1, which is right for older runs and callers.
        "world_size": world_size,
    }

    if config is not None:
        prov["config_sha256"] = config_sha256(config)
        prov["group_sha256"] = group_sha256(config)

    if prov["git_dirty"]:
        prov["patch"] = git(REPO_ROOT, "diff", "HEAD")

    return prov