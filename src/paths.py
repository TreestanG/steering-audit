import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TypeVar

from log import get_logger

logger = get_logger(__name__)

RESULTS_ROOT = Path("results")
ACTIVATIONS_ROOT = Path("data/activations")
PARTIAL_SUFFIX = ".partial"

K = TypeVar("K")


@contextmanager
def atomic_writes(paths: dict[K, Path]) -> Iterator[dict[K, IO[str]]]:
    """Stream rows to <path>.partial, renaming into place only if the body completes.

    A stage that writes its output as it goes is otherwise indistinguishable, once it
    has been killed, from one that finished: the file exists and non-empty, the
    driver's sentinel matches it, and the next run skips the stage. Everything
    downstream then reads a truncated sample believing it is the whole one, which for
    the audit means the evasion-window denominator silently shrinks.

    The .partial is left behind rather than deleted -- it is hours of vocabulary scans
    on a large model, and worth looking at before the rerun overwrites it.
    """
    temps = {key: path.with_name(path.name + PARTIAL_SUFFIX) for key, path in paths.items()}
    for temp in temps.values():
        temp.parent.mkdir(parents=True, exist_ok=True)
    handles: dict[K, IO[str]] = {key: temp.open("w") for key, temp in temps.items()}
    try:
        yield handles
    except BaseException:
        for handle in handles.values():
            handle.close()
        logger.error("interrupted with output incomplete; left %s and did NOT write %s, "
                     "so the next run redoes this stage rather than reading a partial one",
                     ", ".join(str(t) for t in temps.values()),
                     ", ".join(str(p) for p in paths.values()))
        raise
    for handle in handles.values():
        handle.close()
    for key, path in paths.items():
        temps[key].replace(path)


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
