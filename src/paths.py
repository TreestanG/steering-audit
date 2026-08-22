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


def behavior_dir(model_name: str, behavior: str) -> Path:
    if behavior == "sentiment":
        return results_dir(model_name) / "sentiment"
    return results_dir(model_name) / "behavior" / behavior


def steer_dir(model_name: str, behavior: str = "sentiment") -> Path:
    base = results_dir(model_name) / "steer"
    return base if behavior == "sentiment" else base / behavior
