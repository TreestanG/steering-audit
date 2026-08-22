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

# Acceptance tolerance has to clear the dtype's own numerical floor, or every
# position reports NO MATCH while having recovered the correct token. Measured
# worst-case ||h_dtype - h_fp32|| / ||h_fp32|| on Qwen2.5-0.5B, doubled for margin:
#   float32  5.0e-06   float16  4.5e-03   bfloat16  2.5e-02
# Recovery itself is unaffected -- even bfloat16's floor is ~7% of gap/2 -- but
# detection sensitivity scales with this, so a smaller dtype costs the weakest steers.
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
    """Validate a --device string up front.

    torch's own failure for an absent backend arrives late (mid-forward, or as a
    bare assertion inside .to()) and after the weights have already been read off
    disk. An accelerator asked for by name and missing is a mistake worth naming
    immediately, not silently downgrading to CPU.
    """
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
    """Record --dtype in the environment as well as the namespace.

    paths.run_tag() reads it, so results/<slug>_fp16/ and results/<slug>_fp32/
    separate themselves without anyone passing --tag. It has to happen here rather
    than in load_model(): every entry point builds its output paths from
    experiment_dir()/logs_dir() before it loads any weights, so a hook at load time
    would fire after the paths were already decided.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        os.environ["AAT_DTYPE"] = str(values)


def add_model_args(parser: argparse.ArgumentParser, *, default_dtype: str = "float32") -> None:
    """--device / --dtype, spelled the same way by every entry point.

    Defaulting the device to pick_device() rather than CPU means a script run bare
    on a GPU box uses the GPU; --device cpu is how you opt out. default_dtype is a
    parameter because sentiment_dir needs fp32 for reasons its own comment gives.
    """
    parser.add_argument(
        "--device", type=str, default=pick_device(),
        help="cuda / cuda:1 / mps / cpu (default: the best one available here)",
    )
    # argparse never fires an action for a default, so the default is published here,
    # at parser-build time, and _DtypeArg overwrites it only if --dtype is passed.
    # setdefault, not assignment: a tag already exported by run_model.sh wins.
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
    """Load onto `device` (validated), or leave on CPU when device is None."""
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
        model.to(resolve_device(str(device)))
    return model, tokenizer


def model_device() -> torch.device:
    """Where the loaded model lives — the one source of truth for input placement."""
    model, _ = require_model()
    return next(model.parameters()).device


def require_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The loaded model and tokenizer, or one clear error. Narrows away the Nones."""
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) first")
    return model, tokenizer


def get_decoder_layers() -> torch.nn.ModuleList:
    """Return the ModuleList of transformer blocks across common HF architectures.

    Matched on structure, not class identity: Mistral, Gemma, Qwen3 and Phi are
    not LlamaForCausalLM subclasses, so an isinstance list has to be extended for
    every new architecture instead of just working.
    """
    model, _ = require_model()
    for parent, attr in (
        ("model", "layers"),  # Llama, Qwen2/3, Mistral, Gemma, Phi, ...
        ("transformer", "h"),  # GPT-2
        ("gpt_neox", "layers"),  # Pythia / GPT-NeoX
    ):
        blocks = getattr(getattr(model, parent, None), attr, None)
        if isinstance(blocks, torch.nn.ModuleList):
            return blocks
    raise AttributeError(f"Don't know how to find layers on {type(model).__name__}")


def get_final_norm() -> torch.nn.Module:
    """The norm applied after the last block, on the same structural basis as
    get_decoder_layers(). hidden_states[-1] is post-norm, so anything captured
    straight off the last block has to pass through this to match."""
    model, _ = require_model()
    for parent, attr in (
        ("model", "norm"),  # Llama, Qwen2/3, Mistral, Gemma, Phi, ...
        ("transformer", "ln_f"),  # GPT-2
        ("gpt_neox", "final_layer_norm"),  # Pythia / GPT-NeoX
    ):
        norm = getattr(getattr(model, parent, None), attr, None)
        if isinstance(norm, torch.nn.Module):
            return norm
    raise AttributeError(f"Don't know how to find the final norm on {type(model).__name__}")


def apply_final_norm(h: torch.Tensor, layer: int) -> torch.Tensor:
    """Post-norm a state captured off block `layer`, if that block is the last one."""
    model, _ = require_model()
    return get_final_norm()(h) if layer == model.config.num_hidden_layers else h


def get_base_model() -> torch.nn.Module:
    """The decoder stack without the LM head (model.model / transformer / gpt_neox)."""
    model, _ = require_model()
    return model.base_model


def get_token_activations(
    prompts: list[str],
    layer: int | None = None,
    last_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (acts, attention_mask) for a batch of prompts.

    Shapes:
        last_only=True,  layer=None → acts [n_layers, batch, hidden]
        last_only=True,  layer=k    → acts [batch, hidden]
        last_only=False, layer=None → acts [n_layers, batch, seq, hidden]  (padded)
        last_only=False, layer=k    → acts [batch, seq, hidden]            (padded)

    attention_mask is always [batch, seq]. Layer index 0 is embeddings;
    decoder block i is at index i.
    """
    model, tokenizer = require_model()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if layer is not None:
        stacked = outputs.hidden_states[layer]  # [batch, seq, hidden]
    else:
        stacked = torch.stack(outputs.hidden_states, dim=0)  # [n_layers, batch, seq, hidden]

    if not last_only:
        return stacked, attention_mask

    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=stacked.device)
    if layer is not None:
        acts = stacked[batch_idx, last_idx]  # [batch, hidden]
    else:
        acts = stacked[:, batch_idx, last_idx]  # [n_layers, batch, hidden]
    return acts, attention_mask
