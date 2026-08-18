"""Where results and activations live, derived from a model name.

Deliberately free of torch/transformers imports so the plotters can use it
without pulling the whole model stack in.
"""

from pathlib import Path


def model_slug(model_name: str) -> str:
    """'Qwen/Qwen2.5-0.5B-Instruct' -> 'Qwen_Qwen2.5-0.5B-Instruct'."""
    return model_name.replace("/", "_")


def results_dir(model_name: str) -> Path:
    return Path("results") / model_slug(model_name)


def activations_dir(model_name: str, root: str = "data/activations") -> Path:
    return Path(root) / model_slug(model_name)


def vocab_table_path(model_name: str, root: str = "data/activations") -> Path:
    return activations_dir(model_name, root) / "vocab" / "vocab_table.pt"
