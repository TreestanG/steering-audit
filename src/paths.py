import os
from pathlib import Path

RESULTS_ROOT = Path("results")
ACTIVATIONS_ROOT = Path("data/activations")


DTYPE_TAGS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def run_tag() -> str:
    tag = os.environ.get("AAT_RUN_TAG", "").strip()
    if tag:
        return tag
    return DTYPE_TAGS.get(os.environ.get("AAT_DTYPE", "").strip(), "")


def model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "_")
    tag = run_tag()
    return f"{slug}_{tag}" if tag else slug


def results_dir(model_name: str) -> Path:
    return RESULTS_ROOT / model_slug(model_name)


def logs_dir(model_name: str) -> Path:
    return results_dir(model_name) / "logs"


def experiment_dir(model_name: str, experiment: str) -> Path:
    return results_dir(model_name) / experiment


def figures_dir(model_name: str, experiment: str | None = None) -> Path:
    base = results_dir(model_name) if experiment is None else experiment_dir(model_name, experiment)
    return base / "figures"


def activations_dir(model_name: str, root: str | Path = ACTIVATIONS_ROOT) -> Path:
    return Path(root) / model_slug(model_name)


def vocab_table_path(model_name: str, root: str | Path = ACTIVATIONS_ROOT) -> Path:
    return activations_dir(model_name, root) / "vocab" / "vocab_table.pt"


def behavior_dir(model_name: str, behavior: str, arm: str | None = None) -> Path:
    """Where one (behavior, arm) run's records live.

    The arm is part of the path because it is part of the experiment: jbb_refusal's
    benign arm is the false-positive control for its harmful arm, and refusal has two.
    Without it the second arm overwrites the first and the driver's sentinel skips the
    stage outright, so a sweep silently reports one arm's numbers under both names.
    Sentiment keeps its flat legacy path -- one arm, and committed figures point at it.
    """
    if behavior == "sentiment":
        return results_dir(model_name) / "sentiment"
    base = results_dir(model_name) / "behavior" / behavior
    return base / arm if arm else base


def steer_dir(model_name: str, behavior: str = "sentiment",
              arm: str | None = None) -> Path:
    base = results_dir(model_name) / "steer"
    if behavior == "sentiment":
        return base
    return base / behavior / arm if arm else base / behavior
