import argparse
import os
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

model: PreTrainedModel | None = None
tokenizer: PreTrainedTokenizerBase | None = None


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

DEFAULT_REL_TOL = {torch.float32: 1e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}


def rel_tol_for(dtype: torch.dtype) -> float:
    return DEFAULT_REL_TOL[dtype]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {requested}: no CUDA device visible "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda or 'none'}).\n"
            f"Available: {pick_device()}"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(f"--device {requested}: MPS is not available on this build/platform")
    return device


class _DtypeArg(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        os.environ["AAT_DTYPE"] = str(values)


def add_model_args(parser: argparse.ArgumentParser, *, default_dtype: str = "float32") -> None:
    parser.add_argument(
        "--device", type=str, default=pick_device(),
        help="cuda / cuda:1 / mps / cpu (default: the best one available here)",
    )
    os.environ.setdefault("AAT_DTYPE", default_dtype)
    parser.add_argument(
        "--dtype", type=str, default=default_dtype, choices=list(DTYPES),
        action=_DtypeArg,
        help=f"model compute dtype (default: {default_dtype}); also selects the "
             f"results/<slug>_<tag>/ tree unless --tag overrides it",
    )


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    global model, tokenizer
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype),
    )

    model.eval()
    if device is not None:
        cast(torch.nn.Module, model).to(resolve_device(str(device)))
    return model, tokenizer


def model_device() -> torch.device:
    model, _ = require_model()
    return next(model.parameters()).device


def require_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) first")
    return model, tokenizer


def get_decoder_layers() -> torch.nn.ModuleList:
    model, _ = require_model()
    for parent, attr in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        blocks = getattr(getattr(model, parent, None), attr, None)
        if isinstance(blocks, torch.nn.ModuleList):
            return blocks
    raise AttributeError(f"Don't know how to find layers on {type(model).__name__}")


def get_final_norm() -> torch.nn.Module:
    model, _ = require_model()
    for parent, attr in (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ):
        norm = getattr(getattr(model, parent, None), attr, None)
        if isinstance(norm, torch.nn.Module):
            return norm
    raise AttributeError(f"Don't know how to find the final norm on {type(model).__name__}")


def apply_final_norm(h: torch.Tensor, layer: int) -> torch.Tensor:
    model, _ = require_model()
    return get_final_norm()(h) if layer == model.config.num_hidden_layers else h


def get_base_model() -> torch.nn.Module:
    model, _ = require_model()
    return model.base_model


def get_token_activations(
    prompts: list[str],
    layer: int | None = None,
    last_only: bool = True,
    pre_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual stream per layer. `pre_norm` reads the final block BEFORE the model's
    final norm, which is the space a steering delta is injected into; hidden_states[-1]
    is post-norm and the two differ by ||h_pre||/||h_post|| (0.26x to 91x measured)."""
    model, tokenizer = require_model()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    attention_mask = inputs["attention_mask"]
    grabbed: dict[int, torch.Tensor] = {}
    handles = []
    if pre_norm:
        for i, block in enumerate(get_decoder_layers()):
            def grab(module, args, output, k=i + 1):
                out = output[0] if isinstance(output, tuple) else output
                grabbed[k] = out.detach()
            handles.append(block.register_forward_hook(grab))
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    states = list(outputs.hidden_states)
    for k, value in grabbed.items():
        states[k] = value

    if layer is not None:
        stacked = states[layer]
    else:
        stacked = torch.stack(states, dim=0)

    if not last_only:
        return stacked, attention_mask

    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=stacked.device)
    if layer is not None:
        acts = stacked[batch_idx, last_idx]
    else:
        acts = stacked[:, batch_idx, last_idx]
    return acts, attention_mask
