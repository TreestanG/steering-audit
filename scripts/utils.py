from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizerBase,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
)

model: PreTrainedModel | None = None
tokenizer: PreTrainedTokenizerBase | None = None


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


def get_decoder_layers() -> torch.nn.ModuleList:
    """Return the ModuleList of transformer blocks across common HF architectures."""
    if model is None:
        raise RuntimeError("Call load_model(...) before get_decoder_layers()")
    if isinstance(model, GPTNeoXForCausalLM):
        return model.gpt_neox.layers  # Pythia / GPT-NeoX
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers  # Llama, Qwen2, Mistral, Gemma, ...
    if isinstance(model, GPT2LMHeadModel):
        return model.transformer.h  # GPT-2
    raise AttributeError(f"Don't know how to find layers on {type(model).__name__}")


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
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) before get_token_activations()")

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
