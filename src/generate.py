from dataclasses import dataclass

import torch
from torch import Tensor

from log import get_logger, heartbeat
from steering import unit_delta
from utils import get_decoder_layers, model_device, require_model

logger = get_logger(__name__)

POSITIONS = ("last", "all", "first")


@dataclass
class Intervention:
    layer: int
    delta: Tensor
    positions: str = "last"
    label: str = "steer"

    def __post_init__(self):
        if self.positions not in POSITIONS:
            raise ValueError(f"positions must be one of {POSITIONS}, got {self.positions!r}")

    @classmethod
    def from_direction(cls, layer: int, direction: Tensor, scale: Tensor, fraction: float,
                       positions: str = "last", label: str = "steer") -> "Intervention":
        return cls(layer, unit_delta(direction, scale, fraction), positions, label)

    def hook(self, mask: Tensor | None = None, at: Tensor | None = None):
        delta, positions = self.delta, self.positions

        def apply(hidden: Tensor) -> Tensor:
            hidden = hidden.clone()
            d = delta.to(device=hidden.device, dtype=hidden.dtype)
            if positions == "all":
                if mask is not None and mask.shape[1] == hidden.shape[1]:
                    hidden += mask.to(hidden.dtype).unsqueeze(-1) * d
                else:
                    hidden += d
            elif positions == "first":
                # Prefill only: position 0 exists in exactly one forward pass, and a
                # decode step (seq len 1) must not re-add -- that would turn a
                # one-position edit into an all-positions steer. Left padding means
                # index 0 is a pad slot, not the first REAL token.
                if hidden.shape[1] > 1:
                    rows = torch.arange(hidden.shape[0], device=hidden.device)
                    first = (mask.to(hidden.device).float().argmax(dim=1)
                             if mask is not None and mask.shape[1] == hidden.shape[1]
                             else torch.zeros(hidden.shape[0], dtype=torch.long,
                                              device=hidden.device))
                    hidden[rows, first] += d
            elif at is not None and hidden.shape[1] > 1:
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                hidden[rows, at.to(hidden.device)] += d
            else:
                hidden[:, -1, :] += d
            return hidden

        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                return (apply(output[0]),) + output[1:]
            return apply(output)

        return hook_fn


def position_ids_for(mask: Tensor) -> Tensor:
    """Positions that ignore left padding: the first REAL token is position 0.

    Without this a left-padded batch is silently wrong on any model with learned
    absolute position embeddings, because HF defaults position_ids to arange(seq_len)
    and the pad slots eat the low positions. Measured on gpt2: batched last-token
    logits differ from the same prompt run alone by up to 104, and a teacher-forced
    logprob by 2.6 nats. Rotary models (Qwen, Llama, Pythia) are shift-invariant and
    were never affected, which is exactly what makes this the kind of bug that ships.
    """
    return (mask.cumsum(dim=-1) - 1).clamp(min=0)


def _encode(prompts: list[str]):
    _, tokenizer = require_model()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    inputs = {k: v.to(model_device()) for k, v in encoded.items()}
    inputs["position_ids"] = position_ids_for(inputs["attention_mask"])
    return inputs


def _hooked(intervention: Intervention | None, mask: Tensor | None = None,
            at: Tensor | None = None):
    if intervention is None:
        return []
    block = get_decoder_layers()[intervention.layer - 1]
    return [block.register_forward_hook(intervention.hook(mask=mask, at=at))]


@torch.no_grad()
def generate_completions(
    prompts: list[str],
    intervention: Intervention | None = None,
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    stop_at_newline_pair: bool = False,
) -> list[str]:
    model, tokenizer = require_model()
    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inputs = _encode(chunk)
        handles = _hooked(intervention, mask=inputs["attention_mask"])
        try:
            # generate() derives its own position_ids per decode step, and passing the
            # prefill ones would pin every generated token to the last prompt position.
            generated = model.generate(
                **{k: v for k, v in inputs.items() if k != "position_ids"},
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            for h in handles:
                h.remove()
        new = generated[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        if stop_at_newline_pair:
            texts = [t.split("\n\n")[0] for t in texts]
        out.extend(texts)
        heartbeat(logger, min(start + batch_size, len(prompts)), len(prompts), "generated")
    return out


@torch.no_grad()
def target_logprobs(
    prompts: list[str],
    targets: list[str],
    intervention: Intervention | None = None,
    *,
    batch_size: int = 8,
) -> list[dict]:
    model, tokenizer = require_model()
    out: list[dict] = []

    for start in range(0, len(prompts), batch_size):
        chunk_p = prompts[start:start + batch_size]
        chunk_t = targets[start:start + batch_size]

        prompt_ids = [tokenizer.encode(p) for p in chunk_p]
        target_ids = [tokenizer.encode(t, add_special_tokens=False) for t in chunk_t]
        seqs = [p + t for p, t in zip(prompt_ids, target_ids)]
        width = max(len(s) for s in seqs)

        pad = tokenizer.pad_token_id
        device = model_device()
        input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, width - len(s):] = torch.tensor(s)
            mask[i, width - len(s):] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        prompt_end = torch.tensor([width - len(t) - 1 for t in target_ids], device=device)

        handles = _hooked(intervention, mask=mask, at=prompt_end)
        try:
            logits = model(input_ids=input_ids, attention_mask=mask,
                           position_ids=position_ids_for(mask),
                           use_cache=False).logits.float()
        finally:
            for h in handles:
                h.remove()

        logprobs = torch.log_softmax(logits, dim=-1)
        for i, tids in enumerate(target_ids):
            n = len(tids)
            first = width - n
            picked = logprobs[i, first - 1:width - 1, :]
            got = picked[torch.arange(n, device=device), torch.tensor(tids, device=device)]
            total = float(got.sum())
            out.append({"sum": total, "mean": total / n, "n_tokens": n})
    return out


def target_gap(prompts: list[str], pos_targets: list[str], neg_targets: list[str],
               intervention: Intervention | None = None, *, batch_size: int = 8
               ) -> list[float]:
    pos = target_logprobs(prompts, pos_targets, intervention, batch_size=batch_size)
    neg = target_logprobs(prompts, neg_targets, intervention, batch_size=batch_size)
    return [p["mean"] - n["mean"] for p, n in zip(pos, neg)]


@torch.no_grad()
def last_token_logits_batch(prompts: list[str], intervention: Intervention | None = None,
                            *, batch_size: int = 8) -> Tensor:
    model, _ = require_model()
    chunks = []
    for start in range(0, len(prompts), batch_size):
        inputs = _encode(prompts[start:start + batch_size])
        handles = _hooked(intervention, mask=inputs["attention_mask"])
        try:
            out = model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        chunks.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(chunks, dim=0)
