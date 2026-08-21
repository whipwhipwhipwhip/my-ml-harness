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

    Lives here rather than in a shared utils package so `ml_harness/` depends on
    nothing outside its own directory. None on failure because every use is
    provenance: a run from a tarball with no `.git`, or a machine with no git
    installed, is still a run, and "we couldn't tell" is a truthful record
    where a crashed training job is not.
    """
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def find_git_dirty(REPO_ROOT):
    """True/False if git answered, None if it couldn't — three states, not two.

    The distinction matters downstream: `collect_provenance` only bothers
    recording a patch when this is truthy, and None means "unknown" rather than
    "clean", which is not something to quietly record as a reproducible run.
    """
    is_dirty = git(REPO_ROOT, "status", "--porcelain")
    return bool(is_dirty) if is_dirty else None

# The distribution that ships `ml_harness/`. This is now its own package, which
# is exactly the case the field was added for: a downstream project pins the
# harness as a dependency, so the harness's code is invisible to *that* repo's
# `git_sha`, which is where reproducibility needs it most. Runs written either
# side of the split stay comparable because both record this.
HARNESS_DISTRIBUTION = "ml_harness"


def _dist_version(name: str) -> str | None:
    """The installed version of a distribution, or None if it isn't one.

    None rather than raising, like every other field here: provenance records
    what it can observe, and a harness imported from a source tree with no
    install is still a run worth recording.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None

# What a *group* of runs is allowed to differ by and still be the same
# experiment: the seed, the human label on it, and how the batch was sliced to
# fit in memory. Everything else stays in the hash.
#
# `train.micro_batch_size` is here because it is a hardware knob: the batch the
# update is computed from is `train.batch_size` either way, and splitting it
# across four forward passes is the same experiment run on a smaller card. Not
# the same *run* — different batches are drawn and floats accumulate in a
# different order — but that is exactly the standing `runtime.seed` has, and
# `config_sha256` still separates them.
#
# `runtime.eval_seed` deliberately isn't here. Two runs with different eval
# seeds are measured against different validation batches, so their losses
# aren't averageable — which is the one thing a group exists to do. Same for
# `runtime.device`: "works on CPU, differs on MPS" is a difference worth being
# able to see rather than one to hash away.
GROUP_EXCLUDES: tuple[tuple[str, ...], ...] = (
    ("runtime", "seed"),
    ("note",),
    ("train", "micro_batch_size"),
)


def _sha256_json(obj: Any) -> str:
    """sha256 of a canonical JSON rendering.

    `sort_keys` for the same reason `tokenizers.fingerprint` uses it: the hash
    has to survive a round trip through a file and a dict whose insertion order
    changed. `asdict` order is stable today, but nothing enforces that.
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
    """Identity of an *experiment*: the config modulo the axes in GROUP_EXCLUDES.

    This is what makes the house rule mechanical. N seeds of one config share a
    `group_sha256`, so "mean +/- spread over seeds" is a groupby on one column
    rather than a hand-assembled list of run ids.
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
    """Facts about the world — observed, not chosen.

    The data hash matters because `.gitignore` has `*.txt`, so
    input.txt is not in git and this is the only record of what was trained on.

    The tokenizer hash matters for the same reason one step further along: the
    same corpus under two vocabularies is two different training sets, and
    `vocab_size` alone won't tell them apart — two BPE runs at 50256 can be
    completely different tokenizers. `tokenizer_kind` is the class; the config's
    `tokenizer_arch` is the name the run asked for, and they aren't the same
    string (`char_fixed` builds a `char`).

    `config` is the odd one out — chosen, not observed — but its *hashes* are
    facts about the run, and this is the file every analysis already reads as
    columns. Recording them here rather than in `run_id` keeps run ids
    collision-free (two seeds of one config, or one config re-run, still get
    distinct directories) and costs nothing: `flatten_dict` surfaces both as
    `provenance.*` columns in `load_runs`. Optional so an older caller, or a
    micrograd runner with no RunConfig, still works.
    """


    prov = {
        "git_sha": git(REPO_ROOT, "rev-parse", "HEAD"),
        "git_dirty": find_git_dirty(REPO_ROOT),
        "torch_version": torch.__version__,
        # The two dependencies whose code decides what a run *means*, rather
        # than just how fast it goes. `tokenizers_version` is the load-bearing
        # one already: it's a separate distribution as of the split, so a change
        # to the merge loop lands without touching this repo's history at all,
        # and two runs over the same corpus with the same `tokenizer_sha256`
        # could still have been encoded by different code. The recipe is
        # fingerprinted; the encoder that reads it wasn't, until now.
        "harness_version": _dist_version(HARNESS_DISTRIBUTION),
        "tokenizers_version": _dist_version("tokenizers"),
        "device_resolved": device,
        "vocab_size": vocab_size,
        "data_sha256": sha256,
        "tokenizer_kind": tokenizer.kind,
        "tokenizer_sha256": fingerprint(tokenizer),
        "hostname": os.uname().nodename,
        # How many processes shared this run's batch. Provenance rather than
        # config, and the distinction is the whole reason data parallelism
        # doesn't fragment `runs/`: `train.batch_size` is global, so eight ranks
        # and one rank running the same config are the *same experiment* and
        # share a `group_sha256`. What differs is the machine, and the machine
        # is what this file is for. Defaulted, so every already-recorded run and
        # every caller that predates this field still reads as what it was: one
        # process.
        "world_size": world_size,
    }

    if config is not None:
        prov["config_sha256"] = config_sha256(config)
        prov["group_sha256"] = group_sha256(config)

    if prov["git_dirty"]:
        prov["patch"] = git(REPO_ROOT, "diff", "HEAD")

    return prov