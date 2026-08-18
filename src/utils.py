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


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    global model, tokenizer
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32),
    )

    model.eval()
    return model, tokenizer


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
