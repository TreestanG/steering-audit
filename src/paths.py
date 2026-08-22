"""Where activations, results and figures live, derived from a model name.

Layout, one directory per experiment, data and figures kept apart:

    results/<slug>/
        sipit/       sipit.jsonl, layers/sipit_layer_XX.jsonl
                     figures/layer_metrics*.png
        steer/       audit.jsonl, recover.jsonl, localize.jsonl (sentiment),
                     <behavior>/audit.jsonl + fractions/ for the others,
                     fractions/steer_audit_f<fraction>.jsonl
                     figures/audit.png, recover.png, fraction_sweep.png
        sentiment/   gaps.json  (the sentiment behavior, kept at its old path)
        behavior/    <name>/gaps.json, completions.jsonl, evaluations.json
        pgd/         pgd_<objective>_b<budget>.{jsonl,json}
        figures/     cross-experiment figures (gap_vs_steering.png, ...)
        logs/        one log per pipeline stage, plus run.log

Deliberately free of torch/transformers imports so the plotters can use it
without pulling the whole model stack in.
"""

import os
from pathlib import Path

RESULTS_ROOT = Path("results")
ACTIVATIONS_ROOT = Path("data/activations")


# The suffix each dtype gets when no explicit tag is given. Strings, not torch
# dtypes, so this module stays importable by the plotters without torch.
DTYPE_TAGS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def run_tag() -> str:
    """Suffix on every path: AAT_RUN_TAG if set, else the dtype in AAT_DTYPE.

    One model at two dtypes otherwise collides: the slug is built from the model
    name alone, so the second run rebuilds the first one's activations and table
    in place and then skips every experiment stage whose sentinel already exists.

    That guard used to be a flag someone had to remember, and it drifted — the
    logs show an fp32 activations stage inside pythia-1.4b's otherwise-fp16 tree,
    and both dtypes inside gpt2's sipit and vocab stages. So the dtype now supplies
    the tag on its own (utils.add_model_args exports AAT_DTYPE), and AAT_RUN_TAG
    stays available to override it for a non-dtype axis: --tag seed7, --tag rerun.
    """
    tag = os.environ.get("AAT_RUN_TAG", "").strip()
    if tag:
        return tag
    return DTYPE_TAGS.get(os.environ.get("AAT_DTYPE", "").strip(), "")


def model_slug(model_name: str) -> str:
    """'Qwen/Qwen2.5-0.5B-Instruct' -> 'Qwen_Qwen2.5-0.5B-Instruct[_<tag>]'."""
    slug = model_name.replace("/", "_")
    tag = run_tag()
    return f"{slug}_{tag}" if tag else slug


def results_dir(model_name: str) -> Path:
    return RESULTS_ROOT / model_slug(model_name)


def logs_dir(model_name: str) -> Path:
    """Per-stage logs, beside the artifacts the stage produced."""
    return results_dir(model_name) / "logs"


def experiment_dir(model_name: str, experiment: str) -> Path:
    """Data for one experiment: 'sipit', 'steer' or 'sentiment'."""
    return results_dir(model_name) / experiment


def figures_dir(model_name: str, experiment: str | None = None) -> Path:
    """Figures for one experiment, or the cross-experiment figure dir."""
    base = results_dir(model_name) if experiment is None else experiment_dir(model_name, experiment)
    return base / "figures"


def activations_dir(model_name: str, root: str | Path = ACTIVATIONS_ROOT) -> Path:
    return Path(root) / model_slug(model_name)


def vocab_table_path(model_name: str, root: str | Path = ACTIVATIONS_ROOT) -> Path:
    return activations_dir(model_name, root) / "vocab" / "vocab_table.pt"


def behavior_dir(model_name: str, behavior: str) -> Path:
    """Where one behavior's efficacy results go.

    'sentiment' keeps results/<slug>/sentiment/ rather than moving under behavior/:
    every figure in the README, plot_gap_vs_steering's default and
    pgd_attack.best_steering_layer's default all point at that path, and eight
    completed runs already have a gaps.json sitting there. The special case is one
    line; renaming would invalidate all of it for no gain.
    """
    if behavior == "sentiment":
        return results_dir(model_name) / "sentiment"
    return results_dir(model_name) / "behavior" / behavior


def steer_dir(model_name: str, behavior: str = "sentiment") -> Path:
    """Where one behavior's DETECTION results go, mirroring behavior_dir.

    Same reasoning: 'sentiment' stays at results/<slug>/steer/ because the finished
    audits, the fraction sweeps and both plotters' defaults already live there, and
    anything else gets its own subtree so two behaviors' audits cannot overwrite
    each other's audit.jsonl.
    """
    base = results_dir(model_name) / "steer"
    return base if behavior == "sentiment" else base / behavior
