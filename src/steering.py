import json

import numpy as np
import torch
from torch import Tensor

import utils
from utils import get_decoder_layers, get_token_activations, model_device, require_model


def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    return list(data.values())


def build_steering_vectors(pairs, layers=None, batch_size: int = 8):
    if layers is None:
        layers = list(range(1, len(get_decoder_layers()) + 1))

    differences = {layer: [] for layer in layers}
    residuals = {layer: [] for layer in layers}

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        flat = [p for pair in chunk for p in pair]
        acts, _ = get_token_activations(flat)
        for layer in layers:
            at = acts[layer]
            for i in range(len(chunk)):
                act_pos, act_neg = at[2 * i], at[2 * i + 1]
                differences[layer].append(act_pos - act_neg)
                residuals[layer].append(act_pos)
                residuals[layer].append(act_neg)

    out = {}
    for layer in layers:
        stacked = torch.stack(differences[layer])
        direction = stacked.mean(dim=0)
        scale = torch.stack(residuals[layer]).norm(dim=1).mean()
        out[layer] = (direction.float().cpu(), scale.float().cpu())
    return out


def _steer_hidden(output, delta):
    if isinstance(output, tuple):
        hidden = output[0].clone()
        hidden[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden,) + output[1:]
    hidden = output.clone()
    hidden[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
    return hidden


def _steer_hidden_all(output, delta, mask: Tensor | None = None):
    def add(hidden: Tensor) -> Tensor:
        hidden = hidden.clone()
        d = delta.to(device=hidden.device, dtype=hidden.dtype)
        if mask is None:
            hidden += d
        else:
            hidden += mask.to(hidden.dtype).unsqueeze(-1) * d
        return hidden

    if isinstance(output, tuple):
        return (add(output[0]),) + output[1:]
    return add(output)


def unit_delta(direction: Tensor, scale: Tensor, fraction: float) -> Tensor:
    return fraction * scale * direction / direction.norm()


def make_add_vector_hook(direction, scale, fraction, *, all_positions: bool = False,
                         mask: Tensor | None = None):
    def add_vector(module, input, output):
        delta = unit_delta(direction, scale, fraction)
        if all_positions:
            return _steer_hidden_all(output, delta, mask)
        return _steer_hidden(output, delta)

    return add_vector


def make_random_vector_hook(scale, fraction, seed: int = 42, *,
                            all_positions: bool = False, mask: Tensor | None = None):
    assert utils.model is not None
    rng = np.random.default_rng(seed)
    v = torch.from_numpy(rng.standard_normal(utils.model.config.hidden_size)).float()

    def add_random_vector(module, input, output):
        delta = fraction * scale * v / v.norm()
        if all_positions:
            return _steer_hidden_all(output, delta, mask)
        return _steer_hidden(output, delta)

    return add_random_vector


def make_delta_hook(delta: Tensor, *, all_positions: bool = False,
                    mask: Tensor | None = None):
    def add(module, input, output):
        if all_positions:
            return _steer_hidden_all(output, delta, mask)
        return _steer_hidden(output, delta)

    return add


def last_token_logits(prompt):
    model, tokenizer = require_model()
    device = model_device()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    return outputs.logits[0, -1]


def token_id(word):
    _, tokenizer = require_model()
    return tokenizer.encode(word)[-1]


def first_token_id(text: str) -> int:
    _, tokenizer = require_model()
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError(f"{text!r} encodes to nothing")
    return ids[0]


def refusal_token_ids(substrings=None) -> list[int]:
    from scoring import JBB_REFUSAL_SUBSTRINGS

    _, tokenizer = require_model()
    ids = set()
    for phrase in (substrings or JBB_REFUSAL_SUBSTRINGS):
        for form in (phrase, " " + phrase):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
    return sorted(ids)


def _as_id(word) -> int:
    return word if isinstance(word, int) else token_id(word)


def logit_gap(logits, word_pos, word_neg):
    return logits[_as_id(word_pos)] - logits[_as_id(word_neg)]


def efficacy(clean_logits, steered_logits_, word_pos, word_neg):
    gap = float(logit_gap(steered_logits_, word_pos, word_neg)
                - logit_gap(clean_logits, word_pos, word_neg))
    logp = torch.log_softmax(clean_logits.float(), dim=-1)
    logq = torch.log_softmax(steered_logits_.float(), dim=-1)
    kl = float((logq.exp() * (logq - logp)).sum())
    flip = int(steered_logits_.argmax() != clean_logits.argmax())
    return gap, kl, flip


def steered_logits(prompt, hook_fn, layer):
    handle = get_decoder_layers()[layer - 1].register_forward_hook(hook_fn)
    try:
        return last_token_logits(prompt)
    finally:
        handle.remove()
