"""Where activations, results and figures live, derived from a model name.

Layout, one directory per experiment, data and figures kept apart:

    results/<slug>/
        sipit/       sipit.jsonl, layers/sipit_layer_XX.jsonl
                     figures/layer_metrics*.png
        steer/       audit.jsonl, recover.jsonl, localize.jsonl,
                     fractions/steer_audit_f<fraction>.jsonl
                     figures/audit.png, recover.png, fraction_sweep.png
        sentiment/   gaps.json
        figures/     cross-experiment figures (gap_vs_steering.png, ...)
        logs/        one log per pipeline stage, plus run.log

Deliberately free of torch/transformers imports so the plotters can use it
without pulling the whole model stack in.
"""

import os
from pathlib import Path

RESULTS_ROOT = Path("results")
ACTIVATIONS_ROOT = Path("data/activations")


def run_tag() -> str:
    """Optional suffix on every path, from AAT_RUN_TAG (run_model.sh --tag).

    One model at two dtypes otherwise collides: the slug is built from the model
    name alone, so the second run rebuilds the first one's activations and table
    in place and then skips every experiment stage whose sentinel already exists.
    Empty by default, so untagged trees keep the paths they already have.
    """
    return os.environ.get("AAT_RUN_TAG", "").strip()


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
