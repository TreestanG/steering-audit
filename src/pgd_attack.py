import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor

import behaviors
import prompt_format
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import behavior_dir, experiment_dir, logs_dir
from steering import (
    PICK_BY,
    best_layer,
    build_steering_vectors,
    efficacy,
    first_token_id,
    refusal_token_ids,
    token_id,
)
from utils import (
    DTYPES,
    add_model_args,
    apply_final_norm,
    get_decoder_layers,
    load_model,
    model_device,
    rel_tol_for,
    require_model,
)

logger = get_logger(__name__)

OBJECTIVES = ("sentiment", "cw", "target", "refusal")
CONSTRAINTS = ("all", "injection")
ARMS = ("caa", "random", "pgd")
POSITIONS = ("last", "first")

MAX_BACKOFF = 3

BACKOFF_SLACK = 0.995


def _grad_steer_hidden(output, delta: Tensor, positions: Tensor):

    def inject(hidden: Tensor) -> Tensor:
        d = delta.to(device=hidden.device, dtype=hidden.dtype)
        onehot = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
        onehot[torch.arange(hidden.shape[0], device=hidden.device), positions] = 1.0
        return hidden + onehot.unsqueeze(-1) * d.unsqueeze(1)

    if isinstance(output, tuple):
        return (inject(output[0]),) + output[1:]
    return inject(output)


def _warn_if_not_fp32() -> None:
    model, _ = require_model()
    dtype = next(model.parameters()).dtype
    if dtype != torch.float32:
        logger.warning(
            "model is %s: PGD gradients at reduced precision stall the optimizer, and a "
            "null result then measures rounding rather than geometry. Load float32 and "
            "set the budget with --budget_dtype float16 instead.",
            dtype,
        )


def _encode_with_targets(prompts: list[str], targets: list[str]):
    _, tokenizer = require_model()
    prompt_ids = [tokenizer.encode(p) for p in prompts]
    target_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    width = max(len(p) + len(t) for p, t in zip(prompt_ids, target_ids))
    span = max(len(t) for t in target_ids)
    pad = tokenizer.pad_token_id
    assert isinstance(pad, int)

    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    read_at = torch.zeros((len(prompts), span), dtype=torch.long)
    tokens = torch.zeros((len(prompts), span), dtype=torch.long)
    keep = torch.zeros((len(prompts), span), dtype=torch.bool)
    for i, (p, t) in enumerate(zip(prompt_ids, target_ids)):
        input_ids[i, : len(p) + len(t)] = torch.tensor(p + t)
        mask[i, : len(p) + len(t)] = 1
        read_at[i, : len(t)] = torch.arange(len(p) - 1, len(p) - 1 + len(t))
        tokens[i, : len(t)] = torch.tensor(t)
        keep[i, : len(t)] = True
    prompt_end = torch.tensor([len(p) - 1 for p in prompt_ids])
    return {"input_ids": input_ids, "attention_mask": mask}, read_at, tokens, keep, prompt_end


def clean_reference(prompts: str | list[str], layer: int,
                    targets: list[str] | None = None, *,
                    position: str = "last") -> dict:
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    model, tokenizer = require_model()
    device = model_device()
    if isinstance(prompts, str):
        prompts = [prompts]
    target_span = None
    if targets is None:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="right")
        inputs = {k: v.to(device) for k, v in encoded.items()}
        read_positions = inputs["attention_mask"].sum(dim=1) - 1
    else:
        if len(targets) != len(prompts):
            raise ValueError(f"{len(targets)} targets for {len(prompts)} prompts")
        encoded, read_at, tokens, keep, prompt_end = _encode_with_targets(prompts, targets)
        inputs = {k: v.to(device) for k, v in encoded.items()}
        read_positions = prompt_end.to(device)
        target_span = {"read_at": read_at.to(device), "tokens": tokens.to(device),
                       "keep": keep.to(device)}
    positions = (torch.zeros_like(read_positions) if position == "first"
                 else read_positions)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    rows = torch.arange(len(prompts), device=device)
    states = {k: h[rows, positions].detach().float() for k, h in enumerate(out.hidden_states)}
    if layer not in states or layer < 1:
        raise ValueError(f"layer {layer} out of range 1..{len(states) - 1}")
    reference = {
        "prompts": list(prompts),
        "layer": layer,
        "batch": len(prompts),
        "inputs": inputs,
        "position": position,
        "positions": positions,
        "read_positions": read_positions,
        "rows": rows,
        "targets": list(targets) if targets is not None else None,
        "target_span": target_span,
        "logits": out.logits[rows, read_positions].detach().float(),
        "states": states,
        "states_read": ({k: h[rows, read_positions].detach().float()
                         for k, h in enumerate(out.hidden_states)}
                        if position != "last" else states),
        "h_norm": states[layer].norm(dim=-1),
    }
    reference["target_logprob"] = (
        target_logprob(out.logits, reference).detach() if target_span is not None else None
    )
    return reference


def _run_with_delta(clean: dict, layer: int, delta: Tensor, *, want_states: bool):
    model, _ = require_model()
    blocks = get_decoder_layers()
    captured: dict[str, Tensor] = {}
    positions, rows = clean["positions"], clean["rows"]

    handles = [blocks[layer - 1].register_forward_hook(
        lambda module, args, output: _grad_steer_hidden(output, delta, positions)
    )]
    if want_states:
        def capture(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            captured["h"] = out[rows, positions].detach()

        handles.append(blocks[layer - 1].register_forward_hook(capture))

    try:
        out = model(**clean["inputs"], output_hidden_states=want_states, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return out, captured.get("h")


@torch.no_grad()
def _rel_devs_from(out, state: Tensor, clean: dict, layer: int) -> dict[int, Tensor]:
    model, _ = require_model()
    rows, positions = clean["rows"], clean["positions"]
    steered_at_layer = apply_final_norm(state.float(), layer)
    devs = {}
    for k in range(layer, model.config.num_hidden_layers + 1):
        h_clean = clean["states"][k]
        h_steer = (steered_at_layer if k == layer
                   else out.hidden_states[k][rows, positions].float())
        devs[k] = (h_steer - h_clean).norm(dim=-1) / h_clean.norm(dim=-1)
    return devs


@torch.no_grad()
def _rel_devs_read(out, clean: dict, layer: int) -> dict[int, Tensor]:
    model, _ = require_model()
    rows, read = clean["rows"], clean["read_positions"]
    devs = {}
    for k in range(layer, model.config.num_hidden_layers + 1):
        h_clean = clean["states_read"][k]
        h_steer = out.hidden_states[k][rows, read].float()
        devs[k] = (h_steer - h_clean).norm(dim=-1) / h_clean.norm(dim=-1)
    return devs


@torch.no_grad()
def _rel_deviations(delta: Tensor, clean: dict, layer: int) -> dict[int, Tensor]:
    out, state = _run_with_delta(clean, layer, delta, want_states=True)
    assert state is not None
    return _rel_devs_from(out, state, clean, layer)


def target_logprob(full_logits: Tensor, clean: dict) -> Tensor:
    span = clean["target_span"]
    if span is None:
        raise ValueError("objective 'target' needs a target per prompt; "
                         "clean_reference was built without one")
    rows = clean["rows"].unsqueeze(-1).expand_as(span["read_at"])
    picked = full_logits[rows, span["read_at"]]
    logprobs = torch.log_softmax(picked.float(), dim=-1)
    got = logprobs.gather(-1, span["tokens"].unsqueeze(-1)).squeeze(-1)
    keep = span["keep"].to(got.dtype)
    return (got * keep).sum(dim=-1) / keep.sum(dim=-1)


def _refusal_ids(clean: dict) -> Tensor:
    ids = clean.get("_refusal_ids")
    if ids is None:
        ids = torch.tensor(refusal_token_ids(), device=clean["logits"].device)
        clean["_refusal_ids"] = ids
    return ids


def _id(word) -> int:
    return word if isinstance(word, int) else token_id(word)


def _objective(logits: Tensor, kind: str, clean: dict, word_pos: str | int,
               word_neg: str | int,
               full_logits: Tensor | None = None) -> Tensor:
    if kind == "sentiment":
        return logits[:, _id(word_pos)] - logits[:, _id(word_neg)]
    if kind == "cw":
        c = clean["logits"].argmax(dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[clean["rows"], c] = True
        best_other = logits.masked_fill(mask, float("-inf")).max(dim=-1).values
        return best_other - logits[clean["rows"], c]
    if kind == "target":
        if full_logits is None:
            raise ValueError("objective 'target' needs the full logits tensor")
        return target_logprob(full_logits, clean)
    if kind == "refusal":
        probs = torch.softmax(logits.float(), dim=-1)
        refusal = probs[:, _refusal_ids(clean)].sum(dim=-1).clamp(1e-8, 1 - 1e-8)
        return -(torch.log(refusal) - torch.log1p(-refusal))
    raise ValueError(f"objective must be one of {OBJECTIVES}, got {kind!r}")


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else float("inf")


def evaluate_deltas(
    layer: int,
    delta: Tensor,
    *,
    clean: dict,
    word_pos: str | int,
    word_neg: str | int,
    budget: float | None = None,
) -> list[dict]:
    if clean["layer"] != layer:
        raise ValueError("`clean` was built for a different layer")

    delta = _as_batch(delta, clean)
    elsewhere = clean.get("position", "last") != "last"
    with torch.no_grad():
        out, state = _run_with_delta(clean, layer, delta, want_states=True)
        assert state is not None
        steered = out.logits[clean["rows"], clean["read_positions"]].detach().float()
        devs = _rel_devs_from(out, state, clean, layer)
        devs_read = _rel_devs_read(out, clean, layer) if elsewhere else None
        computable = [k for k in OBJECTIVES
                      if k != "target" or clean["target_span"] is not None]
        objs = {k: _objective(steered, k, clean, word_pos, word_neg, out.logits)
                for k in computable}
        clean_target = clean.get("target_logprob")

    _, tokenizer = require_model()
    clean_top = clean["logits"].argmax(dim=-1)
    steered_top = steered.argmax(dim=-1)
    delta_norm = delta.norm(dim=-1)
    records = []
    for b, prompt in enumerate(clean["prompts"]):
        gap, kl, flip = efficacy(clean["logits"][b], steered[b], word_pos, word_neg)
        row_devs = {k: float(v[b]) for k, v in devs.items()}
        argmax_layer = max(row_devs, key=lambda k: row_devs[k])
        max_rel_dev = row_devs[argmax_layer]
        h_norm = float(clean["h_norm"][b])
        record = {
            "prompt": prompt,
            "gap": gap,
            "kl": kl,
            "flip": flip,
            "clean_token": int(clean_top[b]),
            "steered_token": int(steered_top[b]),
            "clean_str": tokenizer.decode([int(clean_top[b])]),
            "steered_str": tokenizer.decode([int(steered_top[b])]),
            "h_norm": h_norm,
            "delta_norm": float(delta_norm[b]),
            "rel_delta": _ratio(float(delta_norm[b]), h_norm),
            "rel_dev_at_layer": row_devs[layer],
            "max_rel_dev": max_rel_dev,
            "argmax_dev_layer": argmax_layer,
            "rel_dev_by_layer": row_devs,
        }
        if budget is not None:
            record["boundary_frac"] = _ratio(max_rel_dev, budget)
        if devs_read is not None:
            read_devs = {k: float(v[b]) for k, v in devs_read.items()}
            record["rel_dev_read_by_layer"] = read_devs
            record["rel_dev_read_at_layer"] = read_devs[layer]
            record["max_rel_dev_read"] = max(read_devs.values())
        for kind, values in objs.items():
            record[f"obj_{kind}"] = float(values[b])
        if clean_target is not None:
            record["target_logprob"] = float(objs["target"][b])
            record["target_logprob_clean"] = float(clean_target[b])
            record["d_target_logprob"] = float(objs["target"][b] - clean_target[b])
        records.append(record)
    return records


def _as_batch(delta: Tensor, clean: dict) -> Tensor:
    delta = delta.to(device=model_device(), dtype=torch.float32)
    if delta.dim() == 1:
        delta = delta.unsqueeze(0).expand(clean["batch"], -1)
    if delta.shape[0] != clean["batch"]:
        raise ValueError(f"delta has {delta.shape[0]} rows, clean has {clean['batch']}")
    return delta.contiguous()


def evaluate_delta(prompt: str, layer: int, delta: Tensor, *, clean: dict,
                   word_pos: str | int, word_neg: str | int,
                   budget: float | None = None) -> dict:
    if clean["batch"] != 1 or clean["prompts"][0] != prompt:
        raise ValueError("`clean` was built for a different (prompt, layer)")
    return evaluate_deltas(layer, delta, clean=clean, word_pos=word_pos,
                           word_neg=word_neg, budget=budget)[0]


def depth_layer(n_layers: int, frac: float, allow_final: bool = False) -> int:
    if not 0 < frac <= 1:
        raise SystemExit(f"--layer_frac must be in (0, 1], got {frac}")
    cap = n_layers if allow_final else n_layers - 1
    return max(1, min(round(frac * n_layers), cap))


def best_steering_layer(gaps_path: str | Path, fraction: float = 0.1,
                        allow_final: bool = False, by: str = "flip") -> int:
    payload = json.loads(Path(gaps_path).read_text())
    rows = payload["layers"]
    at = [r for r in rows if math.isclose(r["fraction"], fraction, rel_tol=1e-9)]
    if not at:
        present = sorted({r["fraction"] for r in rows})
        raise ValueError(f"{gaps_path}: no rows at fraction={fraction} (have {present})")
    n_layers = max(r["layer"] for r in rows)
    if not allow_final:
        at = [r for r in at if r["layer"] < n_layers]
        if not at:
            raise ValueError(f"{gaps_path}: only the final block ({n_layers}) has rows at "
                             f"fraction={fraction}; pass allow_final=True to use it")
    return best_layer(at, fraction, by)


def _project(
    delta: Tensor,
    *,
    budget: float,
    clean: dict,
    layer: int,
    constraint: str,
    max_backoff: int = MAX_BACKOFF,
) -> Tensor:
    model, _ = require_model()
    radius = (budget * clean["h_norm"]).unsqueeze(-1)

    def ball(d: Tensor) -> Tensor:
        norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        shrunk = d * torch.minimum(radius / norm, torch.ones_like(norm))
        return torch.where(radius > 0, shrunk, torch.zeros_like(d))

    delta = ball(delta)
    if budget <= 0 or (constraint == "injection" and layer < model.config.num_hidden_layers):
        return delta

    def worst(d: Tensor) -> Tensor:
        devs = _rel_deviations(d, clean, layer)
        return (torch.stack(list(devs.values())).max(dim=0).values if constraint == "all"
                else devs[layer])

    for _ in range(max_backoff):
        over = worst(delta)
        if bool((over <= budget).all()):
            break
        factor = torch.where(over > budget, BACKOFF_SLACK * budget / over.clamp_min(1e-30),
                             torch.ones_like(over))
        delta = ball(delta * factor.unsqueeze(-1))
    return delta


def pgd_attack_batch(
    prompts: list[str],
    layer: int,
    *,
    budget: float,
    word_pos: str | int,
    word_neg: str | int,
    steps: int = 200,
    lr: float | None = None,
    objective: str = "sentiment",
    constraint: str = "all",
    n_restarts: int = 3,
    seed: int = 0,
    seeds: list[int] | None = None,
    max_backoff: int = MAX_BACKOFF,
    targets: list[str] | None = None,
    position: str = "last",
) -> list[dict]:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "target" and targets is None:
        raise ValueError("objective 'target' needs a target continuation per prompt")
    if constraint not in CONSTRAINTS:
        raise ValueError(f"constraint must be one of {CONSTRAINTS}, got {constraint!r}")
    _warn_if_not_fp32()

    model, _ = require_model()
    clean = clean_reference(prompts, layer, targets, position=position)
    device = model_device()
    batch, hidden = clean["batch"], model.config.hidden_size
    radius = budget * clean["h_norm"]
    lr0 = (0.1 * radius if lr is None
           else torch.full_like(radius, float(lr))).unsqueeze(-1)

    if seeds is None:
        seeds = [seed + b for b in range(batch)]
    if len(seeds) != batch:
        raise ValueError(f"{len(seeds)} seeds for {batch} prompts")
    draws = []
    for row_seed in seeds:
        gen = torch.Generator().manual_seed(row_seed)
        draws.append([torch.randn(hidden, generator=gen) for _ in range(n_restarts)])

    inits: list[tuple[str, Tensor]] = [("zero", torch.zeros(batch, hidden))]
    for r in range(n_restarts):
        v = torch.stack([draws[b][r] for b in range(batch)])
        inits.append((f"random{r}", v / v.norm(dim=-1, keepdim=True) * radius.cpu().unsqueeze(-1)))

    def project(d: Tensor) -> Tensor:
        return _project(d, budget=budget, clean=clean, layer=layer,
                        constraint=constraint, max_backoff=max_backoff)

    best: list[dict | None] = [None] * batch
    best_delta = torch.zeros(batch, hidden, device=device)
    restarts: list[list[dict]] = [[] for _ in range(batch)]
    for name, init in inits:
        delta = project(init.to(device=device, dtype=torch.float32))
        for step in range(steps):
            leaf = delta.detach().clone().requires_grad_(True)
            out, _ = _run_with_delta(clean, layer, leaf, want_states=False)
            logits = out.logits[clean["rows"], clean["read_positions"]].float()
            value = _objective(logits, objective, clean, word_pos, word_neg, out.logits)
            (grad,) = torch.autograd.grad(value.sum(), leaf)
            grad_norm = grad.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(grad_norm).all()):
                logger.debug("batch %s: non-finite gradient at step %d, stopping", name, step)
                break
            decay = 1.0 - 0.9 * step / max(1, steps - 1)
            with torch.no_grad():
                direction = grad / grad_norm.clamp_min(1e-30)
                delta = project(leaf.detach() + lr0 * decay * direction)

        records = evaluate_deltas(
            layer, delta, clean=clean,
            word_pos=word_pos, word_neg=word_neg, budget=budget,
        )
        for b, record in enumerate(records):
            record["init"] = name
            restarts[b].append(record)
            prev = best[b]
            score = (record["flip"], record[f"obj_{objective}"])
            if prev is None or score > (prev["flip"], prev[f"obj_{objective}"]):
                best[b] = record
                best_delta[b] = delta[b]

    out_records = []
    for b, record in enumerate(best):
        assert record is not None
        out_records.append({
            **record,
            "layer": layer,
            "budget": budget,
            "steps": steps,
            "lr": float(lr0[b]),
            "objective": objective,
            "constraint": constraint,
            "n_restarts": n_restarts,
            "seed": seeds[b],
            "best_init": record["init"],
            "restarts": restarts[b],
            "delta": best_delta[b].detach().cpu(),
        })
    return out_records


def pgd_attack(prompt: str, layer: int, **kwargs) -> dict:
    return pgd_attack_batch([prompt], layer, **kwargs)[0]


def caa_delta(direction: Tensor, scale: Tensor, fraction: float) -> Tensor:
    return fraction * scale * direction / direction.norm()


def random_delta(hidden_size: int, radius: float, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_size, generator=g)
    return v / v.norm() * radius


def _detection(devs: dict[int, float], layer: int, rel_tol: float) -> dict:
    over = sorted(k for k, v in devs.items() if v > rel_tol)
    return {
        "detected_at_layer": int(devs[layer] > rel_tol),
        "detected_multilayer": int(bool(over)),
        "first_detect_layer": over[0] if over else None,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for key in dict.fromkeys((r["constraint"], r["arm"]) for r in rows):
        at = [r for r in rows if (r["constraint"], r["arm"]) == key]

        def mean(field, at=at):
            return sum(r[field] for r in at) / len(at)

        entry = {
            "constraint": key[0],
            "arm": key[1],
            "n": len(at),
            "flip_rate": mean("flip"),
            "gap_mean": mean("gap"),
            "kl_mean": mean("kl"),
            "rel_delta_mean": mean("rel_delta"),
            "rel_dev_at_layer_mean": mean("rel_dev_at_layer"),
            "max_rel_dev_mean": mean("max_rel_dev"),
            "boundary_frac_mean": mean("boundary_frac"),
            "detected_at_layer_rate": mean("detected_at_layer"),
            "detected_multilayer_rate": mean("detected_multilayer"),
        }
        if "d_target_logprob" in at[0]:
            entry["d_target_logprob_mean"] = mean("d_target_logprob")
        if "max_rel_dev_read" in at[0]:
            entry["max_rel_dev_read_mean"] = mean("max_rel_dev_read")
            entry["rel_dev_read_at_layer_mean"] = mean("rel_dev_read_at_layer")
        out.append(entry)
    return out


def _log_table(summary: list[dict]) -> None:
    has_target = any("d_target_logprob_mean" in s for s in summary)
    logger.info("%9s %7s %4s %6s %10s %9s %10s %10s %9s %8s %8s%s",
                "scope", "arm", "n", "flip%", "gap", "KL",
                "rel_delta", "max_dev", "bnd_frac", "det@L%", "det>=L%",
                "   d_target" if has_target else "")
    for s in summary:
        logger.info("%9s %7s %4d %5.0f%% %10.4f %9.4f %10.2e %10.2e %9.3f %7.0f%% %7.0f%%%s",
                    s["constraint"], s["arm"], s["n"], 100 * s["flip_rate"],
                    s["gap_mean"], s["kl_mean"], s["rel_delta_mean"], s["max_rel_dev_mean"],
                    s["boundary_frac_mean"], 100 * s["detected_at_layer_rate"],
                    100 * s["detected_multilayer_rate"],
                    f"  {s['d_target_logprob_mean']:+9.5f}" if has_target else "")


def _csv(value: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    items = list(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
    bad = [v for v in items if v not in allowed]
    if bad or not items:
        raise SystemExit(f"{flag}: expected a comma-separated subset of {allowed}, got {value!r}")
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Per-prompt PGD against the SipIt steering detector: CAA / random / "
                    "PGD at one detection budget, under each constraint scope.",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(parser, default_dtype="float32")
    parser.add_argument("--behavior", type=str, default="sentiment",
                        help=f"one of {', '.join(behaviors.list_behaviors())}, or a path. "
                             f"Supplies the contrast pairs the CAA arm is fitted on, the "
                             f"prompts to attack, and the targets the 'target' objective "
                             f"drives toward")
    parser.add_argument("--arm", type=str, default=None,
                        help="test arm to attack (default: the dataset's default_arm; "
                             "'harmful' is the one to use for jbb_refusal)")
    parser.add_argument("--n_prompts", type=int, default=0,
                        help="0 (default) = the whole arm, so flip_rate shares a "
                             "denominator with gaps.json")
    parser.add_argument("--word_pos", type=str, default=None,
                        help="token the 'sentiment' objective pushes up "
                             "(default: the first token of the behavior's positive target)")
    parser.add_argument("--word_neg", type=str, default=None,
                        help="token it pushes down (default: the behavior's negative target)")
    parser.add_argument("--prompt_format", type=str, default="auto",
                        choices=list(prompt_format.FORMATS),
                        help="must match the behavior_eval and steer_audit runs this "
                             "attack is compared against")
    parser.add_argument("--target_side", type=str, default="neg", choices=["pos", "neg"],
                        help="which of the behavior's targets the 'target' objective "
                             "drives toward. 'neg' is the attack: for jbb_refusal that is "
                             "JailbreakBench's own affirmative 'Sure, here is ...' string")
    parser.add_argument("--position", type=str, default="last", choices=list(POSITIONS),
                        help="where the perturbation goes in. 'last' is every existing "
                             "result here. 'first' injects at token 0, whose norm is "
                             "20-400x the rest of the sequence from layer ~3 on, so the "
                             "same RELATIVE budget buys that much more absolute "
                             "perturbation; the objective is still read at the last "
                             "prompt token. See src/pos0_attack.py for the arms that "
                             "exploit this deliberately")
    parser.add_argument("--layer", type=int, default=None,
                        help="default: the model's best steering layer, per --gaps")
    parser.add_argument("--gaps", type=Path, default=None,
                        help="default: results/<slug>/sentiment/gaps.json")
    parser.add_argument("--pick_by", type=str, default=None, choices=list(PICK_BY),
                        help="how --layer is chosen from --gaps. Default follows the "
                             "behavior: 'flip' for sentiment, 'target_gap' otherwise, "
                             "because flip rate and KL both peak wherever a perturbation "
                             "merely wrecks the output and so select the last block on a "
                             "refusal behavior")
    parser.add_argument("--layer_frac", type=float, default=None,
                        help="pick the injection layer as this fraction of depth "
                             "(0.7 -> layer 20 of 28), instead of the per-model best in "
                             "--gaps. Use it for any CROSS-MODEL comparison: the gaps rule "
                             "returns a different KIND of layer once the final-block guard "
                             "fires, so the attacked layer stops being comparable. "
                             "--layer overrides this; neither falls back to --gaps")
    parser.add_argument("--allow_final_layer", action="store_true",
                        help="let --layer selection pick the last block. Off by default: "
                             "there the final norm sits between the injection and the "
                             "detector, and 'all' and 'injection' enforce the same single "
                             "layer, so the scope contrast disappears")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="the gaps.json row --layer is picked from, and the CAA arm's "
                             "pre-projection strength (the projection rescales it to the "
                             "budget, so this only bites if it lands under the budget)")
    parser.add_argument("--budget", type=float, default=None,
                        help="relative budget; default --budget_frac * rel_tol(--budget_dtype)")
    parser.add_argument("--budget_dtype", type=str, default="float16", choices=list(DTYPES),
                        help="deployment precision whose rel_tol sets the budget. NOT the "
                             "compute dtype, which stays float32: fp16 vs fp32 has to vary "
                             "the constraint alone, not the arithmetic (default: float16)")
    parser.add_argument("--budget_frac", type=float, default=0.85,
                        help="fraction of rel_tol to sit under (default: 0.85)")
    parser.add_argument("--rel_tol", type=float, default=None,
                        help="the detection threshold the budget was cut from, used for the "
                             "det@L / det>=L columns; default budget / budget_frac")
    parser.add_argument("--constraints", type=str, default="all,injection",
                        help="'all' caps every layer >= L (the honest multi-layer auditor); "
                             "'injection' caps layer L alone, which is what steer_audit "
                             "computes today")
    parser.add_argument("--arms", type=str, default="caa,random,pgd")
    parser.add_argument("--objective", type=str, default="sentiment", choices=list(OBJECTIVES))
    parser.add_argument("--steps", type=int, default=50,
                        help="PGD steps per restart (default 50). Finding 7 measured 25 "
                             "matching 200 to five decimals -- everything is linear over a "
                             "10x budget range, so the optimizer converges to "
                             "radius * grad/||grad|| within ~10 steps. That was measured on "
                             "Pythia-1.4B with the sentiment and CW objectives only, so 50 "
                             "keeps 2x margin for the models and objectives it was not "
                             "measured on. Under-converging biases toward FEWER undetected "
                             "flips, which is the conclusion under test -- raise this "
                             "rather than lower it if you are unsure")
    parser.add_argument("--lr", type=float, default=None,
                        help="absolute step size; default a tenth of the ball radius")
    parser.add_argument("--n_restarts", type=int, default=3,
                        help="random inits on the constraint sphere, on top of the zero init")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_backoff", type=int, default=MAX_BACKOFF)
    parser.add_argument(
        "--batch_size", type=int, default=25,
        help="prompts attacked per forward pass. The loop is bound by streaming the "
             "weights, not by the perturbation, so this is close to a linear speedup "
             "until it runs out of memory (default: 25)",
    )
    parser.add_argument("--save_deltas", action="store_true",
                        help="also write the winning perturbations to <out>_deltas.pt, for a "
                             "follow-up vocab scan. Covers only this invocation's work, so "
                             "pair it with --force rather than with a resume")
    parser.add_argument("--force", action="store_true",
                        help="re-run from scratch instead of resuming an unfinished jsonl "
                             "or skipping a finished one")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/<slug>/pgd/pgd_<objective>_b<budget>.jsonl")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "pgd.log")

    constraints = _csv(args.constraints, CONSTRAINTS, "--constraints")
    arms = _csv(args.arms, ARMS, "--arms")
    budget = (args.budget if args.budget is not None
              else args.budget_frac * rel_tol_for(DTYPES[args.budget_dtype]))
    if budget <= 0:
        raise SystemExit(f"--budget must be positive, got {budget:g}")
    rel_tol = args.rel_tol if args.rel_tol is not None else _ratio(budget, args.budget_frac)

    behavior = behaviors.load_behavior(args.behavior)
    arm_name = args.arm or behavior.default_arm
    if args.out is None:
        stem = (f"pgd_{args.objective}_b{budget:g}" if behavior.name == "sentiment"
                else f"pgd_{behavior.name}_{arm_name}_{args.objective}_b{budget:g}")
        if args.position != "last":
            stem += f"_{args.position}"
        args.out = experiment_dir(args.model_name, "pgd") / f"{stem}.jsonl"
    summary_path = args.out.with_suffix(".json")
    if summary_path.exists() and not args.force:
        logger.info("%s exists — run already finished; --force to redo", summary_path)
        return

    load_model(args.model_name, dtype=DTYPES[args.dtype], device=args.device)
    logger.info("model on %s, %s", model_device(), args.dtype)
    _warn_if_not_fp32()
    model, _ = require_model()

    gaps_path = args.gaps or behavior_dir(args.model_name, behavior.name,
                                          arm_name) / "gaps.json"
    pick_by = args.pick_by or ("flip" if behavior.name == "sentiment" else "target_gap")
    if args.layer is not None:
        layer = args.layer
    elif args.layer_frac is not None:
        layer = depth_layer(model.config.num_hidden_layers, args.layer_frac,
                            args.allow_final_layer)
        logger.info("layer %d = %.2f of %d blocks (--layer_frac; gaps.json not consulted)",
                    layer, args.layer_frac, model.config.num_hidden_layers)
    else:
        layer = best_steering_layer(gaps_path, args.fraction, args.allow_final_layer,
                                    pick_by)

    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    items = behavior.items(arm_name, args.n_prompts)
    prompts = prompt_format.render_prompts(behavior, items, fmt)

    def _first(side: str) -> int:
        text = behavior.targets.get(side) or items[0].target
        if text is None:
            raise SystemExit(f"{behavior.name}: no {side} target to resolve a token from; "
                             f"pass --word_{side}")
        return first_token_id(text)

    word_pos = args.word_pos if args.word_pos is not None else _first("pos")
    word_neg = args.word_neg if args.word_neg is not None else _first("neg")

    targets = None
    if args.objective == "target":
        resolved = [behavior.target_for(item, args.target_side) for item in items]
        missing = [i.index for i, t in zip(items, resolved) if t is None]
        if missing:
            raise SystemExit(f"{behavior.name}: items {missing[:5]} have no "
                             f"{args.target_side} target, which the 'target' objective needs")
        targets = [t for t in resolved if t is not None]

    logger.info("behavior %s, arm %s: %d prompts", behavior.name, arm_name, len(prompts))
    logger.info("layer %d (%s), budget %g = %g * rel_tol %g [%s], objective %s",
                layer, "given" if args.layer is not None
                else f"{args.layer_frac:g} of depth" if args.layer_frac is not None
                else f"best in {gaps_path} by {pick_by}",
                budget, args.budget_frac, rel_tol, args.budget_dtype, args.objective)
    logger.info("%d prompts x %d arms x %d scope(s) -> %s",
                len(prompts), len(arms), len(constraints), args.out)

    steering = None
    if "caa" in arms:
        steering = build_steering_vectors(
            prompt_format.render_contrast_pairs(behavior, fmt), layers=[layer])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[str, str, str], dict] = {}
    if args.out.exists() and not args.force:
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["constraint"], row["arm"], row["prompt"]] = row
        if done:
            logger.info("resuming: %d rows already present", len(done))
    else:
        args.out.write_text("")

    base = {
        "model_name": args.model_name,
        "dtype": args.dtype,
        "behavior": behavior.name,
        "test_arm": arm_name,
        "prompt_format": fmt,
        "layer": layer,
        "position": args.position,
        "budget": budget,
        "rel_tol": rel_tol,
        "objective": args.objective,
    }
    rows = list(done.values())
    deltas: dict[str, Tensor] = {}
    with args.out.open("a") as handle:
        for constraint in constraints:
            for arm in arms:
                todo = [(i, p) for i, p in enumerate(prompts)
                        if (constraint, arm, p) not in done]
                if not todo:
                    logger.info("%9s %-6s already complete", constraint, arm)
                    continue
                for start in range(0, len(todo), args.batch_size):
                    chunk = todo[start : start + args.batch_size]
                    idxs = [i for i, _ in chunk]
                    batch_prompts = [p for _, p in chunk]
                    batch_targets = [targets[i] for i in idxs] if targets else None
                    clean = clean_reference(batch_prompts, layer, batch_targets,
                                            position=args.position)
                    extras: list[dict] = [{} for _ in chunk]

                    if arm == "pgd":
                        results = pgd_attack_batch(
                            batch_prompts, layer, budget=budget,
                            word_pos=word_pos, word_neg=word_neg,
                            steps=args.steps, lr=args.lr, objective=args.objective,
                            constraint=constraint, n_restarts=args.n_restarts,
                            seeds=[args.seed + i for i in idxs],
                            max_backoff=args.max_backoff, targets=batch_targets,
                            position=args.position,
                        )
                        batch_deltas = [r.pop("delta") for r in results]
                        records = []
                        for r in results:
                            extra = {
                                "steps": r["steps"], "lr": r["lr"],
                                "n_restarts": r["n_restarts"], "seed": r["seed"],
                                "best_init": r["best_init"],
                                "restart_objs": [
                                    {"init": x["init"], "flip": x["flip"],
                                     "obj": x[f"obj_{args.objective}"]}
                                    for x in r["restarts"]
                                ],
                            }
                            extras[len(records)] = extra
                            records.append({k: v for k, v in r.items() if k not in extra
                                            and k not in ("restarts", "layer", "budget",
                                                          "objective", "constraint", "init")})
                    else:
                        if arm == "caa":
                            assert steering is not None
                            raw = caa_delta(*steering[layer], args.fraction)
                        else:
                            raw = torch.stack([
                                random_delta(model.config.hidden_size,
                                             budget * float(clean["h_norm"][b]),
                                             args.seed + i)
                                for b, i in enumerate(idxs)
                            ])
                        delta = _project(_as_batch(raw, clean), budget=budget, clean=clean,
                                         layer=layer, constraint=constraint,
                                         max_backoff=args.max_backoff)
                        batch_deltas = [delta[b] for b in range(len(chunk))]
                        records = evaluate_deltas(
                            layer, delta, clean=clean, budget=budget,
                            word_pos=word_pos, word_neg=word_neg,
                        )

                    for b, (i, prompt) in enumerate(chunk):
                        row = {
                            **base, "constraint": constraint, "arm": arm,
                            "prompt": prompt, "prompt_index": i,
                            **records[b], **extras[b],
                            **_detection(records[b]["rel_dev_by_layer"], layer, rel_tol),
                        }
                        handle.write(json.dumps(row) + "\n")
                        rows.append(row)
                        if args.save_deltas:
                            deltas[f"{constraint}|{arm}|{i}"] = batch_deltas[b].detach().cpu()
                    handle.flush()
                    logger.info("%9s %-6s %3d/%d  flip=%d/%d  gap %+.4f  bnd %.3f",
                                constraint, arm, start + len(chunk), len(todo),
                                sum(r["flip"] for r in records), len(records),
                                sum(r["gap"] for r in records) / len(records),
                                sum(r["boundary_frac"] for r in records) / len(records))

    summary = _aggregate(rows)
    _log_table(summary)
    summary_path.write_text(json.dumps({
        **base,
        "budget_dtype": args.budget_dtype,
        "budget_frac": args.budget_frac,
        "fraction": args.fraction,
        "constraints": constraints,
        "arms": arms,
        "steps": args.steps,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_prompts": len(prompts),
        "word_pos": word_pos,
        "word_neg": word_neg,
        "target_side": args.target_side if args.objective == "target" else None,
        "rows": str(args.out),
        "summary": summary,
    }, indent=2) + "\n")
    logger.info("wrote %s and %s", args.out, summary_path)
    if args.save_deltas and deltas:
        delta_path = args.out.with_name(args.out.stem + "_deltas.pt")
        torch.save({"layer": layer, "budget": budget, "deltas": deltas}, delta_path)
        logger.info("wrote %s (%d deltas)", delta_path, len(deltas))


if __name__ == "__main__":
    main()
