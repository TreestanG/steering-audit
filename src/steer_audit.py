import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import Tensor

import behaviors
import prompt_format
import sipit
from sipit import Top2
from steering import (
    build_steering_vectors,
    control_seed,
    make_delta_hook,
    random_direction,
)
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import logs_dir, steer_dir
from utils import (
    DTYPES,
    add_model_args,
    apply_final_norm,
    get_base_model,
    get_decoder_layers,
    load_model,
    model_device,
    rel_tol_for,
    require_model,
)


logger = get_logger(__name__)


def steering_delta(direction: Tensor, scale: Tensor, fraction: float) -> Tensor:
    return fraction * scale * direction / direction.norm()


def random_delta(hidden_size: int, scale: Tensor, fraction: float, seed: int) -> Tensor:
    return steering_delta(random_direction(hidden_size, seed), scale, fraction)


def rand_kind(replicate: int) -> str:
    return "rand" if replicate == 0 else f"rand{replicate}"


@torch.no_grad()
def layer_states(prompt: str, layers: list[int], hook_layer: int = 0, hook_fn=None,
                 post_norm: bool = True) -> dict[int, Tensor]:
    model, tokenizer = require_model()
    blocks = get_decoder_layers()
    captured: dict[int, Tensor] = {}

    def make_capture(layer: int):
        def capture(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[layer] = out[0, -1].detach().clone()

        return capture

    handles = []
    if hook_fn is not None:
        handles.append(blocks[hook_layer - 1].register_forward_hook(hook_fn))
    for layer in layers:
        handles.append(blocks[layer - 1].register_forward_hook(make_capture(layer)))

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        model(**{k: v.to(device) for k, v in inputs.items()})
    finally:
        for h in handles:
            h.remove()

    finish = apply_final_norm if post_norm else (lambda h, _L: h)
    return {L: finish(captured[L], L).float().cpu() for L in layers}


def layer_state(prompt: str, layer: int, hook_fn=None, post_norm: bool = True) -> Tensor:
    return layer_states(prompt, [layer], hook_layer=layer, hook_fn=hook_fn,
                        post_norm=post_norm)[layer]


def prefix_cache(prompt: str):
    _, tokenizer = require_model()
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    if len(ids) < 2:
        raise ValueError(f"prompt tokenizes to {len(ids)} token(s); need >= 2")
    prefix, true_id = ids[:-1], ids[-1]
    cache, _ = sipit.encode_step(torch.tensor([prefix]), cache=None, attn_len=len(prefix))
    return prefix, true_id, cache


@torch.no_grad()
def build_targets(prompt: str, steering: dict[int, tuple[Tensor, Tensor]], *,
                  layers: list[int], fractions: list[float], seed: int,
                  positions: str = "last", n_rand: int = 1):
    model, _ = require_model()
    all_positions = positions == "all"
    hidden = model.config.hidden_size
    clean = layer_states(prompt, layers)
    targets: dict[tuple[int, str, float | None], Tensor] = {}
    deltas: dict[tuple[int, float], dict[str, Tensor]] = {}
    for layer in layers:
        targets[layer, "clean", None] = clean[layer]
        direction, scale = steering[layer]
        for fraction in fractions:
            sides = {"steer": steering_delta(direction, scale, fraction)}
            for r in range(n_rand):
                sides[rand_kind(r)] = random_delta(
                    hidden, scale, fraction, control_seed(seed, layer, fraction, r))
            deltas[layer, fraction] = sides
            for kind, delta in sides.items():
                targets[layer, kind, fraction] = layer_state(
                    prompt, layer, make_delta_hook(delta, all_positions=all_positions))
    return targets, deltas


@torch.no_grad()
def scan_vocab(cache, prefix_len: int, targets: dict[tuple, Tensor], chunk: int):
    model, _ = require_model()
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)

    by_layer: dict[int, list[tuple[tuple, Tensor]]] = {}
    for key, target in targets.items():
        by_layer.setdefault(key[0], []).append((key, target))

    cap = sipit.batch_cap(prefix_len, chunk)
    tracked = {key: Top2() for key in targets}

    for start in range(0, vocab_size, cap):
        cands = torch.arange(start, min(start + cap, vocab_size))
        n = cands.shape[0]
        batch_cache = copy.deepcopy(cache)
        batch_cache.batch_repeat_interleave(n)
        out = get_base_model()(
            input_ids=cands.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
            output_hidden_states=True,
        )
        for layer, entries in by_layer.items():
            states = out.hidden_states[layer][:, 0].float().cpu()
            for key, target in entries:
                tracked[key].update((states - target).norm(dim=1), cands)
        del out

    return tracked


@torch.no_grad()
def audit_prompt(
    prompt: str,
    steering: dict[int, tuple[Tensor, Tensor]],
    *,
    layers: list[int],
    fractions: list[float],
    rel_tol: float,
    chunk: int,
    seed: int,
    positions: str = "last",
    n_rand: int = 1,
) -> dict[float, list[dict]]:
    prefix, true_id, cache = prefix_cache(prompt)
    targets, deltas = build_targets(prompt, steering, layers=layers,
                                    fractions=fractions, seed=seed,
                                    positions=positions, n_rand=n_rand)
    tracked = scan_vocab(cache, len(prefix), targets, chunk)

    rows: dict[float, list[dict]] = {f: [] for f in fractions}
    for layer in layers:
        clean = tracked[layer, "clean", None]
        h_norm = float(targets[layer, "clean", None].norm())
        gap = clean.gap
        for fraction in fractions:
            def side(kind: str, fraction: float = fraction, layer: int = layer) -> dict:
                t = tracked[layer, kind, fraction]
                return {
                    "delta_norm": float(deltas[layer, fraction][kind].norm()),
                    "residual": t.best,
                    "gap": t.gap,
                    "token": t.best_id,
                    "recovered": t.best_id == true_id,
                    "rel_residual": t.best / h_norm,
                    "margin_spent": t.best / gap if gap > 0 else float("inf"),
                    "detected": t.best > rel_tol * h_norm,
                }

            row = {
                "prompt": prompt,
                "layer": layer,
                "fraction": fraction,
                "positions": positions,
                "h_norm": h_norm,
                "true_token": true_id,
                "rel_gap": gap / h_norm,
                "clean": {
                    "residual": clean.best,
                    "gap": gap,
                    "token": clean.best_id,
                    "recovered": clean.best_id == true_id,
                },
                "steer": side("steer"),
                "rand": side("rand"),
            }
            if n_rand > 1:
                row["rand_extra"] = {str(r): side(rand_kind(r)) for r in range(1, n_rand)}
            rows[fraction].append(row)
    return rows


def summarize(rows: list[dict], layers: list[int]) -> None:
    logger.info(
        "%5s %9s %10s %10s %9s %6s %5s   | random: %9s %6s %5s",
        "layer", "||h||", "gap/||h||", "res/||h||", "res/gap", "recov", "det",
        "res/gap", "recov", "det",
    )
    for layer in layers:
        at = [r for r in rows if r["layer"] == layer]
        if not at:
            continue

        def mean(fn):
            return sum(fn(r) for r in at) / len(at)

        logger.info(
            f"{layer:>5} {mean(lambda r: r['h_norm']):>9.2f} "
            f"{mean(lambda r: r['rel_gap']):>10.4f} "
            f"{mean(lambda r: r['steer']['rel_residual']):>10.4f} "
            f"{mean(lambda r: r['steer']['margin_spent']):>9.3f} "
            f"{mean(lambda r: r['steer']['recovered']) * 100:>5.0f}% "
            f"{mean(lambda r: r['steer']['detected']) * 100:>4.0f}%   | "
            f"        {mean(lambda r: r['rand']['margin_spent']):>9.3f} "
            f"{mean(lambda r: r['rand']['recovered']) * 100:>5.0f}% "
            f"{mean(lambda r: r['rand']['detected']) * 100:>4.0f}%"
        )
    clean_res = max(r["clean"]["residual"] / r["h_norm"] for r in rows)
    clean_ok = sum(r["clean"]["recovered"] for r in rows) / len(rows)
    logger.info("sanity: clean recovery %.0f%%, worst clean residual/||h|| %.2e",
                clean_ok * 100, clean_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(parser)
    parser.add_argument(
        "--behavior",
        type=str,
        default="sentiment",
        help=f"one of {', '.join(behaviors.list_behaviors())}, or a path. Supplies both "
             f"the contrast pairs the steering vector is fitted on and the held-out "
             f"prompts to audit, so detection and efficacy are measured on the same "
             f"vector and the same text",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help="test arm to audit (default: the dataset's default_arm)",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=list(prompt_format.FORMATS),
        help="must match the behavior_eval run this audit is read against: the "
             "detection threshold is calibrated on the same states the efficacy "
             "numbers came from",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.1",
        help="comma-separated steering strengths. All of them ride ONE vocabulary scan, "
             "so a sweep costs barely more than a single fraction; several fractions "
             "write one file each into --out_dir instead of --out",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=5,
        help="a full vocab scan per prompt covers every layer, so this is the runtime "
             "knob: 49s per prompt on Qwen-0.5B fp32 over 24 layers and 3 fractions, "
             "scaling with model size. 5 is a smoke default -- when this audit feeds "
             "join_detection it also sets the denominator on the evasion-window figure, "
             "and run_behavior.sh passes 30 for that reason",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="comma-separated blocks to audit (default: every block)",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=None,
        help="detection threshold: residual > rel_tol * ||h|| counts as caught",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="last",
        choices=["last", "all"],
        help="which positions the audited steer touches. MUST match the --positions the "
             "behavior_eval run used -- 'all' is the driver's default and was not "
             "expressible here before, so the two stages recorded different "
             "interventions. Measured caveat: this audit inverts ONE position, the last, "
             "and reads it at the layer the delta went in, where the perturbation to the "
             "earlier positions has not reached it yet. The target state is therefore "
             "bit-identical under 'last' and 'all' (checked on gpt2), so the flag records "
             "provenance and lets the join verify the pairing -- it does not widen "
             "coverage. Under 'all' the other n-1 positions are equally perturbed and "
             "unaudited, so the detection rate is a LOWER bound: auditing them can only "
             "add detections",
    )
    parser.add_argument("--chunk", type=int, default=2048, help="candidates per forward")
    parser.add_argument("--seed", type=int, default=42,
                        help="base seed for the random control; the direction is drawn "
                             "per (layer, fraction, replicate) from it, matching "
                             "behavior_eval so the two stages audit the same control")
    parser.add_argument("--n_rand", type=int, default=3,
                        help="random control directions per cell (default 3). All of them "
                             "ride the same vocabulary scan, so extra draws are nearly "
                             "free here; replicate 0 is the one the join pairs against "
                             "generation and lands in the 'rand' key, the rest in "
                             "'rand_extra'")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: results/<slug>/steer/audit.jsonl (single fraction only)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="where steer_audit_f<fraction>.jsonl go when several fractions are given "
             "(default: results/<slug>/steer/fractions)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "audit.log")

    fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    if not fractions:
        raise SystemExit("--fractions: give at least one value")
    if len(fractions) > 1 and args.out is not None:
        raise SystemExit("--out takes a single file; several fractions need --out_dir")
    behavior = behaviors.load_behavior(args.behavior)
    arm = args.arm or behavior.default_arm
    base_dir = steer_dir(args.model_name, behavior.name)
    if args.out is None:
        args.out = base_dir / "audit.jsonl"
    if args.out_dir is None:
        args.out_dir = base_dir / "fractions"

    rel_tol: float = (
        rel_tol_for(DTYPES[args.dtype]) if args.rel_tol is None else float(args.rel_tol)
    )
    model, _ = load_model(args.model_name, dtype=DTYPES[args.dtype], device=args.device)
    logger.info("model on %s, %s", model_device(), args.dtype)

    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    steering = build_steering_vectors(prompt_format.render_contrast_pairs(behavior, fmt))

    layers = (
        [int(s) for s in args.layers.split(",")]
        if args.layers
        else list(range(1, model.config.num_hidden_layers + 1))
    )
    if args.n_rand < 1:
        raise SystemExit("--n_rand: need at least one control direction")
    prompts = prompt_format.render_prompts(
        behavior, behavior.items(arm, args.n_prompts), fmt)
    logger.info("behavior %s, arm %s, prompt_format %s, positions %s",
                behavior.name, arm, fmt, args.positions)
    if args.positions == "all":
        logger.warning("--positions all: only the last position is inverted, and its "
                       "state at the injection layer is identical to --positions last. "
                       "The other steered positions go unaudited, so this detection rate "
                       "is a lower bound on what a per-position detector would catch")
    single = len(fractions) == 1
    paths = ({fractions[0]: args.out} if single
             else {f: args.out_dir / f"steer_audit_f{f:g}.jsonl" for f in fractions})
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%d prompts x %d layers x %d fraction(s) %s, one vocab scan each",
                len(prompts), len(layers), len(fractions),
                ",".join(f"{f:g}" for f in fractions))

    rows: dict[float, list[dict]] = {f: [] for f in fractions}
    handles = {f: path.open("w") for f, path in paths.items()}
    try:
        for i, prompt in enumerate(prompts, start=1):
            start = time.time()
            got = audit_prompt(
                prompt,
                steering,
                layers=layers,
                fractions=fractions,
                rel_tol=rel_tol,
                chunk=args.chunk,
                seed=args.seed,
                positions=args.positions,
                n_rand=args.n_rand,
            )
            for fraction, fraction_rows in got.items():
                for row in fraction_rows:
                    handles[fraction].write(json.dumps(row) + "\n")
                handles[fraction].flush()
                rows[fraction].extend(fraction_rows)
            logger.info("  [%d/%d] %r in %.1fs", i, len(prompts), prompt, time.time() - start)
    finally:
        for handle in handles.values():
            handle.close()

    for fraction in fractions:
        if not single:
            logger.info("---- fraction %g ----", fraction)
        summarize(rows[fraction], layers)
        logger.info("wrote %s", paths[fraction])


if __name__ == "__main__":
    main()
