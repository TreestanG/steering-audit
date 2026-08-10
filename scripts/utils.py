import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def get_decoder_layers() -> torch.nn.ModuleList:
    """Return the ModuleList of transformer blocks across common HF architectures."""
    if model is None:
        raise RuntimeError("Call load_model(...) before get_decoder_layers()")
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers  # Pythia / GPT-NeoX
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Qwen2, Mistral, Gemma, ...
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    raise AttributeError(f"Don't know how to find layers on {type(model).__name__}")


def get_token_activations(
    prompts: str | list[str],
    layer: int | None = None,
    last_only: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) before get_token_activations()")

    single = isinstance(prompts, str)
    if single:
        prompts = [prompts]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    lengths = inputs["attention_mask"].sum(dim=1)
    last_idx = lengths - 1
    batch_idx = torch.arange(len(prompts))

    def trim(h):
        return [h[i, :n] for i, n in enumerate(lengths.tolist())]

    if last_only and layer is not None:
        acts = outputs.hidden_states[layer][batch_idx, last_idx]
    elif last_only and layer is None:
        acts = [h[batch_idx, last_idx] for h in outputs.hidden_states]
    elif layer is not None:
        acts = trim(outputs.hidden_states[layer])
    else:
        acts = [trim(h) for h in outputs.hidden_states]

    if layer is not None:
        return acts[0] if single else acts

    return [a[0] for a in acts] if single else acts
