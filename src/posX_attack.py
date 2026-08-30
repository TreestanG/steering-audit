import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

import behaviors
import prompt_format
import scoring
import sipit
import sipit_trajectory
from generate import Intervention, generate_completions, target_logprobs
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir, vocab_table_path
from pgd_attack import OBJECTIVES, _objective, _run_with_delta, clean_reference, evaluate_deltas
from scoring import score_completion
from steering import build_steering_vectors, first_token_id
from utils import (
    DTYPES,
    add_model_args,
    load_model,
    model_device,
    rel_tol_for,
    require_model,
)

logger = get_logger(__name__)

ARMS = ("none", "hop", "hop_partial", "pgd0", "hop_pgd", "random0", "caa0")

DEFAULT_ALPHAS = (0.6, 0.75, 0.9)


def prefix_context(ids: Tensor, pos: int, scan: int) -> dict:
    model, _ = require_model()
    cache, h_last = sipit.encode_step(ids[:pos].unsqueeze(0), None, pos)
    logits = sipit.next_token_logits(h_last)
    order = torch.argsort(logits[: int(model.config.vocab_size)], descending=True)
    return {"cache": cache, "logits": logits, "prefix_len": pos,
            "order": order[:scan].cpu()}


def context_dists(ctx: dict, layer: int, target: Tensor, order: Tensor) -> Tensor:
    cap = sipit.batch_cap(ctx["prefix_len"])
    parts = []
    for start in range(0, order.shape[0], cap):
        cands = order[start:start + cap]
        h = sipit.candidate_states(ctx["cache"], ctx["prefix_len"], cands, layer)
        parts.append((h - target).norm(dim=1))
    return torch.cat(parts)


def token_state(token: int, layer: int, vocab_layer: Tensor,
                ctx: dict | None) -> Tensor:
    if ctx is None:
        return vocab_layer[token].float()
    return sipit.candidate_states(ctx["cache"], ctx["prefix_len"],
                                  torch.tensor([token]), layer)[0]


def reachability(h: Tensor, vocab_layer: Tensor, true_id: int, rel_tol: float,
                 chunk: int = 8192, *, ctx: dict | None = None,
                 layer: int | None = None) -> dict:
    if ctx is None:
        d = sipit.dists_to(h, vocab_layer, chunk)
        top = torch.topk(d, 2, largest=False)
        best, runner = float(top.values[0]), float(top.values[1])
        best_id = int(top.indices[0])
        tried, exhaustive = int(d.shape[0]), True
    else:
        assert layer is not None
        step = sipit.solve_position(ctx["cache"], ctx["prefix_len"], ctx["logits"],
                                    h, layer, rel_tol=rel_tol, abs_tol=0.0,
                                    schedule=sipit.DEFAULT_SCHEDULE, exhaustive=False)
        best, runner = step["residual"], step["runner_up"]
        best_id, tried = step["token"], step["tried"]
        exhaustive = step["gap_exhaustive"]
    norm = float(h.norm())
    return {
        "residual": best,
        "gap": runner - best,
        "token": best_id,
        "recovered_correct": int(best_id == true_id),
        "h_norm": norm,
        "rel_residual": best / norm if norm else float("inf"),
        "matched": int(best <= rel_tol * norm),
        "tried": tried,
        "gap_exhaustive": int(exhaustive),
    }


def hop_plan(h: Tensor, vocab_layer: Tensor, true_ids: list[int],
             contexts: list[dict | None], *, layer: int, k: int,
             chunk: int) -> tuple[list[dict], Tensor]:
    plans, states = [], []
    for b, true_id in enumerate(true_ids):
        target = h[b].detach().cpu().float()
        ctx = contexts[b]
        if ctx is None:
            d = sipit.dists_to(target, vocab_layer, chunk)
            scanned = torch.arange(d.shape[0])
        else:
            scanned = ctx["order"]
            if not bool((scanned == true_id).any()):
                scanned = torch.cat([scanned, torch.tensor([true_id])])
            d = context_dists(ctx, layer, target, scanned)
        top = torch.topk(d, min(k + 2, d.shape[0]), largest=False)
        vals = top.values.tolist()
        ids = scanned[top.indices].tolist()
        self_dist = float(d[int((scanned == true_id).nonzero()[0])])
        if ids[0] != true_id:
            logger.warning(
                "prompt %d: nearest candidate is token %d, not the prompt's own token "
                "%d at this position (dist %.3e vs %.3e). The reference and the run "
                "disagree -- check --dtype against the table on disk before reading "
                "anything here.",
                b, ids[0], true_id, vals[0], self_dist)
        others = [(v, i) for v, i in zip(vals, ids) if i != true_id]
        if len(others) < k:
            raise SystemExit(f"prompt {b}: fewer than {k} distinct neighbours in the scan")
        hop_dist, hop_id = others[k - 1]
        plans.append({
            "true_token": true_id,
            "hop_token": hop_id,
            "hop_rank": k,
            "hop_dist": hop_dist,
            "self_dist": self_dist,
            "gap": others[0][0] - self_dist,
            "scanned": int(scanned.shape[0]),
        })
        states.append(token_state(hop_id, layer, vocab_layer, ctx))
    return plans, torch.stack(states)


def _unit_rows(hidden: int, seeds: list[int], device) -> Tensor:
    rows = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        v = torch.randn(hidden, generator=g)
        rows.append(v / v.norm())
    return torch.stack(rows).to(device=device, dtype=torch.float32)


def pgd_offset(clean: dict, layer: int, *, center: Tensor, radius: Tensor,
               objective: str, word_pos, word_neg, steps: int, n_restarts: int,
               seeds: list[int], lr: float | None = None) -> Tensor:
    model, _ = require_model()
    device = model_device()
    batch, hidden = clean["batch"], model.config.hidden_size
    center = center.to(device=device, dtype=torch.float32)
    radius = radius.to(device=device, dtype=torch.float32)
    lr0 = (0.1 * radius if lr is None
           else torch.full_like(radius, float(lr))).unsqueeze(-1)

    def ball(d: Tensor) -> Tensor:
        off = d - center
        norm = off.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return center + off * torch.minimum(radius.unsqueeze(-1) / norm,
                                            torch.ones_like(norm))

    def value_of(delta: Tensor) -> Tensor:
        out, _ = _run_with_delta(clean, layer, delta, want_states=False)
        logits = out.logits[clean["rows"], clean["read_positions"]].float()
        return _objective(logits, objective, clean, word_pos, word_neg, out.logits)

    inits: list[tuple[str, Tensor]] = [("center", center.clone())]
    for r in range(n_restarts):
        v = _unit_rows(hidden, [s + 1000 * (r + 1) for s in seeds], device)
        inits.append((f"random{r}", center + v * radius.unsqueeze(-1)))

    best = center.clone()
    best_val = torch.full((batch,), -float("inf"), device=device)
    for name, init in inits:
        delta = ball(init)
        for step in range(steps):
            leaf = delta.detach().clone().requires_grad_(True)
            value = value_of(leaf)
            (grad,) = torch.autograd.grad(value.sum(), leaf)
            gnorm = grad.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(gnorm).all()):
                logger.debug("%s: non-finite gradient at step %d, stopping", name, step)
                break
            decay = 1.0 - 0.9 * step / max(1, steps - 1)
            with torch.no_grad():
                delta = ball(leaf.detach()
                             + lr0 * decay * grad / gnorm.clamp_min(1e-30))
        with torch.no_grad():
            got = value_of(delta)
        take = got > best_val
        best_val = torch.where(take, got, best_val)
        best[take] = delta[take]
    return best.detach()


def build_arm(arm: str, *, clean: dict, layer: int, hop_states: Tensor,
              budget_abs: Tensor, caa: Tensor | None,
              alpha: float, objective: str, word_pos, word_neg, steps: int,
              n_restarts: int, seeds: list[int], caa_sign: float = 1.0) -> Tensor:
    device = model_device()
    h0 = clean["states"][layer]
    hidden = h0.shape[-1]
    hop = hop_states.to(device=device, dtype=h0.dtype) - h0

    if arm == "none":
        return torch.zeros_like(h0)
    if arm == "hop":
        return hop
    if arm == "hop_partial":
        return alpha * hop
    if arm == "random0":
        return _unit_rows(hidden, seeds, device) * budget_abs.unsqueeze(-1)
    if arm == "caa0":
        if caa is None:
            raise SystemExit("arm caa0 needs a steering direction; it was not built")
        unit = (caa_sign * caa / caa.norm()).to(device).unsqueeze(0).expand(clean["batch"], -1)
        return unit * budget_abs.unsqueeze(-1)
    if arm == "pgd0":
        return pgd_offset(clean, layer, center=torch.zeros_like(h0),
                          radius=budget_abs, objective=objective, word_pos=word_pos,
                          word_neg=word_neg, steps=steps, n_restarts=n_restarts,
                          seeds=seeds)
    if arm == "hop_pgd":
        return pgd_offset(clean, layer, center=hop, radius=budget_abs,
                          objective=objective, word_pos=word_pos, word_neg=word_neg,
                          steps=steps, n_restarts=n_restarts, seeds=seeds)
    raise SystemExit(f"--arms: expected a subset of {ARMS}, got {arm!r}")


@torch.no_grad()
def steered_logits_for(clean: dict, layer: int, delta: Tensor) -> Tensor:
    out, _ = _run_with_delta(clean, layer, delta, want_states=False)
    return out.logits[clean["rows"], clean["read_positions"]].float()


@torch.no_grad()
def edit_divergence(clean: dict, hop_ids: list[int], steered: Tensor) -> list[float]:
    model, _ = require_model()
    ids = clean["inputs"]["input_ids"].clone()
    ids[clean["rows"], clean["positions"]] = torch.tensor(hop_ids, device=ids.device)
    out = model(input_ids=ids, attention_mask=clean["inputs"]["attention_mask"],
                use_cache=False)
    edited = out.logits[clean["rows"], clean["read_positions"]].float()
    return (edited - steered).abs().max(dim=-1).values.tolist()


@torch.no_grad()
def check_state_shortcut(clean: dict, layer: int, delta: Tensor) -> float:
    _, captured = _run_with_delta(clean, layer, delta, want_states=True)
    assert captured is not None
    expected = clean["states"][layer] + delta.to(clean["states"][layer])
    return float((captured.float() - expected).abs().max())


def generate_with_deltas(prompts: list[str], layer: int, deltas: Tensor, at: Tensor, *,
                         max_new_tokens: int, batch_size: int) -> list[str]:
    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        sl = slice(start, start + batch_size)
        chunk = prompts[sl]
        intervention = Intervention(layer, deltas[sl], "index", index=at[sl])
        out.extend(generate_completions(chunk, intervention,
                                        max_new_tokens=max_new_tokens,
                                        batch_size=len(chunk)))
    return out


def target_gap_with_deltas(prompts: list[str], pos: list[str], neg: list[str],
                           layer: int, deltas: Tensor, at: Tensor, *,
                           batch_size: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(prompts), batch_size):
        sl = slice(start, start + batch_size)
        intervention = Intervention(layer, deltas[sl], "index", index=at[sl])
        p = target_logprobs(prompts[sl], pos[sl], intervention, batch_size=batch_size)
        n = target_logprobs(prompts[sl], neg[sl], intervention, batch_size=batch_size)
        out.extend(a["mean"] - b["mean"] for a, b in zip(p, n))
    return out


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for key in dict.fromkeys((r["layer"], r["position"], r["arm"]) for r in rows):
        at = [r for r in rows if (r["layer"], r["position"], r["arm"]) == key]

        def mean(field, at=at, default=0.0):
            vals = [r[field] for r in at if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else default

        entry = {
            "layer": key[0],
            "position": key[1],
            "arm": key[2],
            "n": len(at),
            "reach_matched_rate": mean("reach_matched"),
            "reach_correct_rate": mean("reach_recovered_correct"),
            "broken_rate": sum(1 for r in at if r["reach_matched"]
                               and not r["reach_recovered_correct"]) / len(at),
            "rel_residual_mean": mean("reach_rel_residual"),
            "rel_dev_at_layer_mean": mean("rel_dev_at_layer"),
            "recompute_silent_rate": sum(1 for r in at if r["recompute_silent"]) / len(at),
            "boundary_frac_mean": mean("boundary_frac"),
            "delta_norm_mean": mean("delta_norm"),
            "budget_abs_mean": mean("budget_abs"),
            "flip_rate": mean("flip"),
            "gap_mean": mean("gap"),
            "kl_mean": mean("kl"),
        }
        for field in ("d_target_logprob", "max_rel_dev_read", "behavior_score",
                      "behavior_hit", "jailbroken_substring", "hop_edit_divergence",
                      "target_gap_steered", "sipit_exact", "sipit_token0_correct",
                      "sipit_token0_matched", "sipit_silent_corruption",
                      "sipit_first_fail", "sipit_first_wrong", "sipit_n_wrong"):
            if any(r.get(field) is not None for r in at):
                entry[f"{field}_mean"] = mean(field)
        out.append(entry)
    return out


def log_table(summary: list[dict]) -> None:
    logger.info("%5s %4s %12s %4s | %8s %8s %8s | %9s %9s | %6s %9s %8s",
                "layer", "pos", "arm", "n", "reach|ok", "recovOK", "BROKEN",
                "rel_dev@L", "recomp|ok", "flip%", "gap", "KL")
    for s in summary:
        logger.info("%5d %4d %12s %4d | %7.0f%% %7.0f%% %7.0f%% | %9.2e %8.0f%% | %5.0f%% "
                    "%9.4f %8.4f",
                    s["layer"], s["position"], s["arm"], s["n"],
                    100 * s["reach_matched_rate"], 100 * s["reach_correct_rate"],
                    100 * s["broken_rate"], s["rel_dev_at_layer_mean"],
                    100 * s["recompute_silent_rate"], 100 * s["flip_rate"],
                    s["gap_mean"], s["kl_mean"])


def _csv(value: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    items = list(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
    bad = [v for v in items if v not in allowed]
    if bad or not items:
        raise SystemExit(f"{flag}: expected a comma-separated subset of {allowed}, "
                         f"got {value!r}")
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Single-position attacks on the reachability detector: hop onto a "
                    "neighbouring token's state at --positions (silent by construction) "
                    "and measure whether an off-manifold payload on top of it carries "
                    "behaviour. Position 0 is the default and the cheapest case; any "
                    "other position works too, at the cost of recomputing the "
                    "neighbours in context.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(parser, default_dtype="float32")
    parser.add_argument("--behavior", type=str, default="sentiment",
                        help=f"one of {', '.join(behaviors.list_behaviors())}, or a path")
    parser.add_argument("--arm", type=str, default=None,
                        help="test arm (default: the dataset's default_arm)")
    parser.add_argument("--n_prompts", type=int, default=0, help="0 = the whole arm")
    parser.add_argument("--prompts_file", type=Path, default=None,
                        help="JSON {'prompts':[{'id','text','category'}]} to use instead "
                             "of the behavior's arm. Every behavior here has a fixed "
                             "first token, so token 0 never varies; the trajectory bank "
                             "is how the hop gets tested against a real spread of them. "
                             "Behavioural columns are meaningless with this")
    parser.add_argument("--prompt_format", type=str, default="auto",
                        choices=list(prompt_format.FORMATS))
    parser.add_argument("--layers", type=str, default="",
                        help="comma-separated injection layers. Default: a shallow "
                             "control plus a spread through the massive-activation "
                             "band. The FINAL block is refused -- the model's own norm "
                             "sits between the injection and the table there, so "
                             "h_steered is not h_clean + delta and the cheap "
                             "reachability read would be wrong")
    parser.add_argument("--arms", type=str, default="hop,hop_partial,pgd0,hop_pgd,random0",
                        help=f"comma-separated subset of {ARMS}. 'hop' is the positive "
                             f"control, 'random0'/'caa0' the controls, 'pgd0'/'hop_pgd' "
                             f"the behavioural question")
    parser.add_argument("--positions", type=str, default="0",
                        help="comma-separated token positions to hop AT, counted from "
                             "the first real token. Negative counts back from the last "
                             "prompt token (-1 = the read position itself); a position "
                             "past the end of a short prompt is clamped to its last "
                             "token. 0 is the special one: token 0 has no prefix, so its "
                             "neighbours are the precomputed vocabulary table and its "
                             "norm is 20-400x the rest of the sequence. Past 0 a token's "
                             "state depends on the prefix, so neighbours are recomputed "
                             "per prompt through the model (--context_scan candidates), "
                             "which is much slower")
    parser.add_argument("--context_scan", type=int, default=512,
                        help="candidates examined when planning a hop at a position "
                             "past 0, in next-token-logit order. The k-th nearest inside "
                             "that window is an upper bound on the true k-th nearest; "
                             "the reachability read is separate and uses SIPIT's own "
                             "widening schedule, so the DETECTOR is not truncated by this")
    parser.add_argument("--hop_rank", type=str, default="1",
                        help="comma-separated k: hop to the k-th nearest OTHER token. "
                             "Sweeping this trades budget for semantic distance")
    parser.add_argument("--alphas", type=str,
                        default=",".join(str(a) for a in DEFAULT_ALPHAS),
                        help="partial-hop fractions for the hop_partial arm; alpha>0.5 "
                             "crosses the midpoint so the argmin flips, and the "
                             "leftover (1-alpha)*gap stays as residual")
    parser.add_argument("--rel_tol", type=float, default=None,
                        help="detection threshold, both detectors (default: per --dtype)")
    parser.add_argument("--budget_frac", type=float, default=0.85,
                        help="arms are built at this fraction of rel_tol. At 1.0 the "
                             "norm-capped arms land exactly ON the threshold and their "
                             "detected/not split is decided by float noise")
    parser.add_argument("--objective", type=str, default="sentiment",
                        choices=list(OBJECTIVES),
                        help="what the pgd0 / hop_pgd arms maximize, read at the LAST "
                             "prompt token")
    parser.add_argument("--target_side", type=str, default="neg", choices=["pos", "neg"],
                        help="which target the 'target' objective drives toward")
    parser.add_argument("--caa_sign", type=float, default=1.0,
                        help="sign of the caa0 arm. The direction points TOWARD the "
                             "positive behavior, so -1 is the jailbreak direction for "
                             "jbb_refusal")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--n_restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=8192,
                        help="vocab rows per distance chunk")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="default: derived from --model_name")
    parser.add_argument("--no_target_gap", action="store_true",
                        help="skip the teacher-forced positive-vs-negative gap")
    parser.add_argument("--verify_hop", action="store_true", default=True,
                        help="validate the h_clean+delta shortcut once per run, and "
                             "measure how far each hop sits from the corresponding "
                             "token-0 edit (default on; one extra forward pass per arm)")
    parser.add_argument("--no_verify_hop", dest="verify_hop", action="store_false")
    parser.add_argument("--sipit_check", action="store_true",
                        help="invert the attacked trajectory left to right, with no "
                             "clean prefix. Reports where the sweep first FAILS against "
                             "where it first goes WRONG; the two differing is silent "
                             "corruption. Costs a vocab scan per position")
    parser.add_argument("--sipit_prompts", type=int, default=2,
                        help="prompts per (layer, arm) to invert, for --sipit_check")
    parser.add_argument("--sipit_stop_on_fail", action="store_true",
                        help="halt the sweep at the first unmatched position instead of "
                             "running on to show the cascade")
    parser.add_argument("--generate", action="store_true",
                        help="also decode and score, for the behavioural half")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--judge", type=str, default="substring",
                        choices=["substring", "fireworks", "none"])
    parser.add_argument("--judge_style", type=str, default="harmbench,strongreject")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/<slug>/pos0/pos0_<behavior>.jsonl")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "pos0.log")

    arms = _csv(args.arms, ARMS, "--arms")
    hop_ranks = [int(v) for v in args.hop_rank.split(",") if v.strip()]
    hop_positions = [int(v) for v in args.positions.split(",") if v.strip()]
    if not hop_positions:
        raise SystemExit("--positions: expected at least one integer position")
    alphas = [float(v) for v in args.alphas.split(",") if v.strip()]
    behavior = behaviors.load_behavior(args.behavior)
    arm_name = args.arm or behavior.default_arm
    rel_tol = (rel_tol_for(DTYPES[args.dtype]) if args.rel_tol is None
               else float(args.rel_tol))

    model, tokenizer = load_model(args.model_name, dtype=DTYPES[args.dtype],
                                  device=args.device)
    logger.info("model on %s, %s, rel_tol %g", model_device(), args.dtype, rel_tol)
    n_layers = model.config.num_hidden_layers

    if args.layers:
        layers = [int(v) for v in args.layers.split(",") if v.strip()]
    else:
        layers = sorted({2, n_layers // 4, n_layers // 2, 3 * n_layers // 4})
    bad = [L for L in layers if L < 1 or L >= n_layers]
    if bad:
        raise SystemExit(
            f"--layers {bad}: must be in 1..{n_layers - 1}. The final block ({n_layers}) "
            f"is excluded on purpose -- the final norm sits between the injection and "
            f"the vocabulary table there, so h_steered != h_clean + delta and this "
            f"script's table-lookup shortcut would silently report the wrong residual.")

    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    if args.prompts_file:
        bank = json.loads(args.prompts_file.read_text())["prompts"]
        if args.n_prompts:
            bank = bank[:args.n_prompts]
        items = [behaviors.Item(index=p.get("index", i),
                                question=p.get("question") or p["text"],
                                target=p.get("target"), category=p.get("category"))
                 for i, p in enumerate(bank)]
    else:
        items = behavior.items(arm_name, args.n_prompts)
    prompts = prompt_format.render_prompts(behavior, items, fmt)
    logger.info("behavior %s, arm %s: %d prompts, format %s, layers %s",
                behavior.name, arm_name, len(prompts), fmt, layers)

    def _first(side: str) -> int:
        text = behavior.targets.get(side) or items[0].target
        if text is None:
            raise SystemExit(f"{behavior.name}: no {side} target to resolve a token from")
        return first_token_id(text)

    word_pos, word_neg = _first("pos"), _first("neg")

    targets = None
    if args.objective == "target" or not args.no_target_gap:
        resolved = [behavior.target_for(item, args.target_side) for item in items]
        present = [t for t in resolved if t is not None]
        if len(present) == len(resolved):
            targets = present
        elif args.objective == "target":
            raise SystemExit(f"{behavior.name}: objective 'target' needs a "
                             f"{args.target_side} target on every item")
        else:
            logger.info("no %s target on every item; skipping the teacher-forced gap",
                        args.target_side)

    pos_targets = neg_targets = None
    if not args.no_target_gap:
        p = [t for t in (behavior.target_for(i, "pos") for i in items) if t is not None]
        n = [t for t in (behavior.target_for(i, "neg") for i in items) if t is not None]
        if len(p) == len(items) and len(n) == len(items):
            pos_targets, neg_targets = p, n

    judges = []
    if args.generate and args.judge != "none":
        for style in dict.fromkeys(v.strip() for v in args.judge_style.split(",")
                                   if v.strip()):
            kwargs = {"style": style} if args.judge == "fireworks" else {}
            judges.append(scoring.make_judge(args.judge, **kwargs))
            if args.judge != "fireworks":
                break

    vocab_path = args.vocab_path or str(vocab_table_path(args.model_name))
    table, layout = sipit.load_vocab_table(vocab_path, expect_model=args.model_name,
                                           expect_dtype=args.dtype)

    out_path = args.out or (experiment_dir(args.model_name, "pos0")
                            / f"pos0_{behavior.name}_{arm_name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    shortcut_checked = False
    with out_path.open("w") as handle:
        for layer in layers:
            vocab_layer = sipit.load_vocab_layer(table, layer, layout)
            caa = None
            if "caa0" in arms:
                caa = build_steering_vectors(
                    prompt_format.render_contrast_pairs(behavior, fmt),
                    layers=[layer])[layer][0]
            for start in range(0, len(prompts), args.batch_size):
                sl = slice(start, start + args.batch_size)
                chunk = prompts[sl]
                chunk_items = items[sl]
                chunk_targets = targets[sl] if targets else None
                seeds = [args.seed + start + i for i in range(len(chunk))]

                for position in hop_positions:
                    clean = clean_reference(chunk, layer, chunk_targets,
                                            position=position)
                    h0 = clean["states"][layer]
                    at = clean["positions"]
                    true_ids = clean["inputs"]["input_ids"][clean["rows"], at].tolist()
                    budget_abs = args.budget_frac * rel_tol * clean["h_norm"]
                    contexts = [None if int(at[b]) == 0 else
                                prefix_context(clean["inputs"]["input_ids"][b],
                                               int(at[b]), args.context_scan)
                                for b in range(clean["batch"])]

                    for k in hop_ranks:
                        plans, hop_states = hop_plan(h0, vocab_layer, true_ids, contexts,
                                                     layer=layer, k=k, chunk=args.chunk)
                        for arm in arms:
                            for alpha in (alphas if arm == "hop_partial" else [1.0]):
                                delta = build_arm(
                                    arm, clean=clean, layer=layer,
                                    hop_states=hop_states, budget_abs=budget_abs, caa=caa,
                                    alpha=alpha, objective=args.objective,
                                    word_pos=word_pos, word_neg=word_neg, steps=args.steps,
                                    n_restarts=args.n_restarts, seeds=seeds,
                                    caa_sign=args.caa_sign)

                                records = evaluate_deltas(layer, delta, clean=clean,
                                                          word_pos=word_pos,
                                                          word_neg=word_neg, budget=rel_tol)
                                steered_h0 = (h0 + delta).detach().cpu().float()

                                if args.verify_hop and not shortcut_checked:
                                    drift = check_state_shortcut(clean, layer, delta)
                                    scale = float(clean["h_norm"].max())
                                    logger.info("state shortcut h_clean+delta: max drift "
                                                "%.2e (||h|| ~ %.3g)  [%s]", drift, scale,
                                                "OK" if drift <= 1e-2 * scale else "BROKEN")
                                    if drift > 1e-2 * scale:
                                        raise SystemExit(
                                            "the steered state is not h_clean + delta, so "
                                            "every residual below would be wrong. Is --layer "
                                            "the final block?")
                                    shortcut_checked = True

                                edit_diff = None
                                if arm in ("hop", "hop_partial", "hop_pgd") and args.verify_hop:
                                    edit_diff = edit_divergence(
                                        clean, [p["hop_token"] for p in plans],
                                        steered_logits_for(clean, layer, delta))

                                for b, record in enumerate(records):
                                    reach = reachability(steered_h0[b], vocab_layer,
                                                         true_ids[b], rel_tol, args.chunk,
                                                         ctx=contexts[b], layer=layer)
                                    row = {
                                        "model_name": args.model_name,
                                        "dtype": args.dtype,
                                        "behavior": behavior.name,
                                        "test_arm": arm_name,
                                        "prompt_format": fmt,
                                        "layer": layer,
                                        "position": position,
                                        "position_index": int(at[b]),
                                        "n_prompt_tokens": int(
                                            clean["read_positions"][b]) + 1,
                                        "arm": arm,
                                        "alpha": alpha if arm == "hop_partial" else None,
                                        "hop_rank": k,
                                        "rel_tol": rel_tol,
                                        "budget_frac": args.budget_frac,
                                        "index": chunk_items[b].index,
                                        "question": chunk_items[b].question,
                                        "category": chunk_items[b].category,
                                        "budget_abs": float(budget_abs[b]),
                                        "h_norm_at_position": float(clean["h_norm"][b]),
                                        **{f"plan_{key}": val
                                           for key, val in plans[b].items()},
                                        "true_token_str": tokenizer.decode(
                                            [plans[b]["true_token"]]),
                                        "hop_token_str": tokenizer.decode(
                                            [plans[b]["hop_token"]]),
                                        **{f"reach_{key}": val
                                           for key, val in reach.items()},
                                        "reach_token_str": tokenizer.decode([reach["token"]]),
                                        **record,
                                    }
                                    row["recompute_silent"] = int(
                                        record["rel_dev_at_layer"] <= rel_tol)
                                    row["detector_broken"] = int(
                                        reach["matched"] and not reach["recovered_correct"])
                                    row["budget_total_abs"] = (
                                        plans[b]["gap"] + float(budget_abs[b]))
                                    if edit_diff is not None:
                                        row["hop_edit_divergence"] = edit_diff[b]
                                    rows.append(row)

                                if pos_targets is not None and neg_targets is not None:
                                    gaps = target_gap_with_deltas(
                                        prompts[sl], pos_targets[sl], neg_targets[sl],
                                        layer, delta, at, batch_size=len(chunk))
                                    for b, g in enumerate(gaps):
                                        rows[-len(records) + b]["target_gap_steered"] = g

                                if args.sipit_check:
                                    for b in range(min(len(records), args.sipit_prompts)):
                                        n_real = int(
                                            clean["inputs"]["attention_mask"][b].sum())
                                        ids = clean["inputs"]["input_ids"][b][:n_real]
                                        traj = sipit_trajectory.capture(
                                            ids, layer,
                                            Intervention(layer, delta[b:b + 1], "index",
                                                         index=int(at[b])))
                                        got = sipit_trajectory.invert(
                                            traj, layer, vocab_layer, ids.tolist(),
                                            rel_tol=rel_tol,
                                            stop_on_fail=args.sipit_stop_on_fail)
                                        row = rows[-len(records) + b]
                                        for key, val in got.items():
                                            if key != "steps":
                                                row[f"sipit_{key}"] = val
                                        row["sipit_recovered_text"] = tokenizer.decode(
                                            got["recovered_ids"])

                                if args.generate:
                                    texts = generate_with_deltas(
                                        chunk, layer, delta, at,
                                        max_new_tokens=args.max_new_tokens,
                                        batch_size=len(chunk))
                                    graded = {}
                                    for judge in judges:
                                        key = (getattr(judge, "style", None)
                                               or getattr(judge, "name", "judge"))
                                        graded[key] = judge.score(
                                            [i.question for i in chunk_items], texts)
                                    for b, text in enumerate(texts):
                                        row = rows[-len(records) + b]
                                        scored = score_completion(behavior.scorer, text)
                                        row["response"] = text
                                        row["behavior_score"] = scored.score
                                        row["behavior_hit"] = scored.hit
                                        row["jailbroken_substring"] = int(
                                            not scoring.substring_matching_refused(text))
                                        for key, scores in graded.items():
                                            row[f"judge_{key}_score"] = float(scores[b])

                                for row in rows[-len(records):]:
                                    handle.write(json.dumps(row) + "\n")
                                handle.flush()

                                logger.info(
                                    "L%-3d p%-4d %-12s k=%d%s  n=%d  broken %d/%d  "
                                    "rel_resid %.2e  rel_dev@L %.2e  flip %d/%d",
                                    layer, position, arm, k,
                                    f" a={alpha:g}" if arm == "hop_partial" else "",
                                    len(records),
                                    sum(r["detector_broken"] for r in rows[-len(records):]),
                                    len(records),
                                    sum(r["reach_rel_residual"]
                                        for r in rows[-len(records):]) / len(records),
                                    sum(r["rel_dev_at_layer"] for r in records) / len(records),
                                    sum(r["flip"] for r in records), len(records))

    summary = summarize(rows)
    log_table(summary)

    inverted = [r for r in rows if "sipit_silent_corruption" in r]
    if inverted:
        silent = [r for r in inverted if r["sipit_silent_corruption"]]
        logger.info("")
        logger.info("trajectory inversion: %d/%d runs corrupted SILENTLY (wrong token "
                    "recovered before any alarm)", len(silent), len(inverted))
        for r in silent[:5]:
            logger.info("  L%-3d %-12s first wrong at %s, first alarm at %s -> "
                        "localization points %s positions late",
                        r["layer"], r["arm"], r["sipit_first_wrong"],
                        r["sipit_first_fail"],
                        "?" if r["sipit_first_fail"] is None
                        else r["sipit_first_fail"] - r["sipit_first_wrong"])

    diverged = [r["hop_edit_divergence"] for r in rows if "hop_edit_divergence" in r]
    if diverged:
        logger.info("hop vs editing the token itself: max |dlogit| %.3g (mean %.3g). "
                    "Large means the hop is a real mid-stream intervention, not a token "
                    "swap -- layers below the injection still serve the true token's "
                    "K/V. It collapses toward 0 at the last prompt position, where "
                    "nothing downstream reads that K/V.",
                    max(diverged), sum(diverged) / len(diverged))

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "model_name": args.model_name,
        "dtype": args.dtype,
        "behavior": behavior.name,
        "test_arm": arm_name,
        "prompt_format": fmt,
        "layers": layers,
        "positions": hop_positions,
        "context_scan": args.context_scan,
        "arms": arms,
        "hop_ranks": hop_ranks,
        "alphas": alphas,
        "rel_tol": rel_tol,
        "budget_frac": args.budget_frac,
        "objective": args.objective,
        "steps": args.steps,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_prompts": len(prompts),
        "rows": str(out_path),
        "summary": summary,
    }, indent=2) + "\n")
    logger.info("wrote %s and %s", out_path, summary_path)


if __name__ == "__main__":
    main()
