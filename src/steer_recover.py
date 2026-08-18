import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

import sipit
from paths import experiment_dir
from sentiment_dir import build_steering_vectors, load_pairs
from steer_audit import (
    build_targets,
    layer_states,
    make_delta_hook,
    prefix_cache,
    scan_vocab,
    steering_delta,
)
from utils import DTYPES, load_model, pick_device


def cosine(a: Tensor, b: Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


@torch.no_grad()
def recover_prompt(
    prompt: str,
    steering: dict[int, tuple[Tensor, Tensor]],
    *,
    layers: list[int],
    fraction: float,
    seed: int,
    chunk: int,
) -> list[dict]:
    """δ̂ = h'_steer − (clean state of the SipIt-recovered token), scored against known δ."""
    prefix, true_id, cache = prefix_cache(prompt)
    # The clean target rides along so margin_spent divides by the CLEAN gap, matching
    # steer_audit. The steered scan's own gap collapses at high fraction (nearest and
    # runner-up become near-equidistant), which made the old ratio meaningless.
    targets, deltas = build_targets(prompt, steering, layers=layers, fraction=fraction, seed=seed)
    tracked = scan_vocab(cache, len(prefix), targets, chunk)

    rows = []
    for layer in layers:
        row = {"prompt": prompt, "layer": layer, "true_token": true_id}
        clean_gap = tracked[layer, "clean"].gap
        # One early-exit forward recovers the clean state of both winning tokens.
        win_ids = [tracked[layer, "steer"].best_id, tracked[layer, "rand"].best_id]
        h_hat = sipit.candidate_states(cache, len(prefix), torch.tensor(win_ids), layer)
        for j, kind in enumerate(("steer", "rand")):
            t = tracked[layer, kind]
            delta_hat = targets[layer, kind] - h_hat[j]  # observed − recovered-clean
            delta_true = deltas[layer][kind]
            row[kind] = {
                "recovered_token": t.best_id,
                "recovered": t.best_id == true_id,
                "residual": t.best,
                "gap": t.gap,  # this scan's own runner-up spread, as in steer_audit
                "margin_spent": t.best / clean_gap if clean_gap > 0 else float("inf"),
                "cos": cosine(delta_hat, delta_true),
                "norm_ratio": float(delta_hat.norm() / delta_true.norm()),
                "delta_true_norm": float(delta_true.norm()),
                "delta_hat_norm": float(delta_hat.norm()),
            }
        rows.append(row)
    return rows


@torch.no_grad()
def localize_prompt(
    prompt: str,
    steering: dict[int, tuple[Tensor, Tensor]],
    *,
    inject_layers: list[int],
    layers: list[int],
    fraction: float,
    chunk: int,
    takeoff_mult: float,
) -> list[dict]:
    """Inject at each of inject_layers; residual takes off at the injection point.

    Every injection is scored in ONE vocab scan, and the clean side is forwarded
    and scanned once: the clean profile depends only on the prompt, so computing
    it per injection would repeat a full-vocabulary pass for an identical answer.
    """
    prefix, _, cache = prefix_cache(prompt)
    layers = sorted(layers)  # takeoff is "first layer to clear the floor"

    targets: dict[tuple[int, str], Tensor] = {
        (L, "clean"): h for L, h in layer_states(prompt, layers).items()
    }
    for inject in inject_layers:
        direction, scale = steering[inject]
        delta = steering_delta(direction, scale, fraction)
        steered = layer_states(prompt, layers, hook_layer=inject,
                               hook_fn=make_delta_hook(delta))
        for layer in layers:
            targets[layer, f"steer{inject}"] = steered[layer]

    tracked = scan_vocab(cache, len(prefix), targets, chunk)

    rel_clean = {
        L: tracked[L, "clean"].best / float(targets[L, "clean"].norm()) for L in layers
    }
    floor = max(rel_clean.values())
    thresh = max(takeoff_mult * floor, 1e-3)

    rows = []
    for inject in inject_layers:
        rel_steer = {
            L: tracked[L, f"steer{inject}"].best / float(targets[L, f"steer{inject}"].norm())
            for L in layers
        }
        takeoff = next((L for L in layers if rel_steer[L] > thresh), None)
        rows.append(
            {
                "prompt": prompt,
                "inject_layer": inject,
                "takeoff": takeoff,
                "correct": takeoff == inject,
                "floor": floor,
                "thresh": thresh,
                "rel_steer": {str(L): rel_steer[L] for L in layers},
                "rel_clean": {str(L): rel_clean[L] for L in layers},
            }
        )
    return rows


def summarize_recovery(rows: list[dict], layers: list[int]) -> None:
    print(
        f"\nRECOVERY  δ̂ = observed − sipit-recovered-clean\n"
        f"{'layer':>5} {'recov%':>7} {'cos(δ̂,δ)':>10} {'‖δ̂‖/‖δ‖':>10} {'margin':>8}"
        f"   | random: {'cos':>7} {'recov%':>7}"
    )
    for layer in layers:
        at = [r for r in rows if r["layer"] == layer]
        if not at:
            continue

        def mean(fn, kind="steer"):
            return sum(fn(r[kind]) for r in at) / len(at)

        print(
            f"{layer:>5} {mean(lambda s: s['recovered']) * 100:>6.0f}% "
            f"{mean(lambda s: s['cos']):>10.4f} "
            f"{mean(lambda s: s['norm_ratio']):>10.4f} "
            f"{mean(lambda s: s['margin_spent']):>8.3f}   | "
            f"{mean(lambda s: s['cos'], 'rand'):>7.4f} "
            f"{mean(lambda s: s['recovered'], 'rand') * 100:>6.0f}%"
        )
    good = [r["steer"]["cos"] for r in rows if r["steer"]["recovered"]]
    if good:
        print(f"\nwhere token recovered: mean cos(δ̂,δ) = {sum(good) / len(good):.5f}  (n={len(good)})")


def summarize_localization(rows: list[dict]) -> None:
    print("\nLOCALIZATION  injection layer via residual takeoff")
    print(f"{'inject':>7} {'detected':>9} {'correct':>8}  (per prompt)")
    by_inject: dict[int, list[dict]] = {}
    for r in rows:
        by_inject.setdefault(r["inject_layer"], []).append(r)
    for inject in sorted(by_inject):
        at = by_inject[inject]
        detected = [str(r["takeoff"]) for r in at]
        acc = sum(r["correct"] for r in at) / len(at) * 100
        print(f"{inject:>7} {','.join(detected):>9} {acc:>6.0f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", type=str, default=pick_device())
    p.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    p.add_argument("--train_path", type=str, default="data/sentiment_opposites_train.json")
    p.add_argument("--test_path", type=str, default="data/sentiment_opposites_test.json")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--layers", type=str, default="", help="recovery layers (default: all)")
    p.add_argument("--inject_layers", type=str, default="6,12,18",
                   help="layers to inject at for the localization test")
    p.add_argument("--takeoff_mult", type=float, default=10.0,
                   help="residual must exceed this × the clean floor to count as takeoff")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: results/<slug>/steer/")
    args = p.parse_args()

    model, _ = load_model(args.model_name, dtype=DTYPES[args.dtype])
    steering = build_steering_vectors(load_pairs(args.train_path))  # before device move

    model.to(args.device)
    print(f"model on {next(model.parameters()).device}")

    n_layers = model.config.num_hidden_layers
    layers = sorted(int(s) for s in args.layers.split(",")) if args.layers else list(range(1, n_layers + 1))
    inject_layers = sorted(int(s) for s in args.inject_layers.split(",") if s)
    # An injection at a layer that is never captured cannot be detected, so it
    # would silently score as a miss rather than as the misconfiguration it is.
    missing = [L for L in inject_layers if L not in layers]
    if missing:
        p.error(f"--inject_layers {missing} not in --layers {layers}")
    prompts = [neg for _, neg in load_pairs(args.test_path)][: args.n_prompts]

    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir(args.model_name, "steer")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path, loc_path = out_dir / "recover.jsonl", out_dir / "localize.jsonl"

    print(f"{len(prompts)} prompts | recover {len(layers)} layers | localize {inject_layers}")

    rec_rows: list[dict] = []
    with rec_path.open("w") as f:
        for i, prompt in enumerate(prompts, start=1):
            t0 = time.time()
            got = recover_prompt(prompt, steering, layers=layers,
                                 fraction=args.fraction, seed=args.seed, chunk=args.chunk)
            for row in got:
                f.write(json.dumps(row) + "\n")
            f.flush()
            rec_rows.extend(got)
            print(f"[recover {i}/{len(prompts)}] {prompt!r} {time.time() - t0:.1f}s")

    loc_rows: list[dict] = []
    with loc_path.open("w") as f:
        for i, prompt in enumerate(prompts, start=1):
            t0 = time.time()
            got = localize_prompt(prompt, steering, inject_layers=inject_layers, layers=layers,
                                  fraction=args.fraction, chunk=args.chunk,
                                  takeoff_mult=args.takeoff_mult)
            for row in got:
                f.write(json.dumps(row) + "\n")
            f.flush()
            loc_rows.extend(got)
            print(f"[localize {i}/{len(prompts)}] {prompt!r} {time.time() - t0:.1f}s")

    summarize_recovery(rec_rows, layers)
    summarize_localization(loc_rows)
    print(f"\nwrote {rec_path}\nwrote {loc_path}")


if __name__ == "__main__":
    main()
