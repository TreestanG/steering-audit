"""The smallest final-layer deviation ANY argmax flip needs, in the detector's own z.

Injection and read share a position, so the next-token logits are a fixed linear map
of the final normed state there. Flipping the argmax from c to v needs
(w_v - w_c) . dh >= l_c - l_v, i.e. ||dh|| >= (l_c - l_v) / ||w_v - w_c||, and the
minimum over v is the least final-layer perturbation a flip can ride on, whatever
produced it -- any layer, any position profile, any optimiser. Converted to the
role_z statistic at the last layer (always probed, always in the dense tail) it is the
ceiling below which a flip is impossible for that prompt.

  uv run src/flip_bound.py --detector_cal results/<slug>_fp16/sipit/<cal>.json --n_prompts 50
"""
import argparse
import json
import math
from pathlib import Path

import torch

import behaviors
import prompt_format
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir
from pgd_attack import DetectorSurrogate, clean_reference, depth_layer
from utils import DTYPES, add_model_args, load_model, require_model

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(parser, default_dtype="float32")
    parser.add_argument("--behavior", type=str, default="jbb_refusal")
    parser.add_argument("--arm", type=str, default=None)
    parser.add_argument("--target_side", type=str, default="neg")
    parser.add_argument("--prompt_format", type=str, default="auto")
    parser.add_argument("--n_prompts", type=int, default=15)
    parser.add_argument("--layer_frac", type=float, default=0.7)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--detector_cal", type=Path, required=True)
    parser.add_argument("--ceilings", type=str, default="4.28,4.78,6,7.17,8,12,16,20",
                        help="z ceilings at the last layer to report admissibility for")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "flip_bound.log")

    cal = json.loads(args.detector_cal.read_text())
    if cal.get("statistic") != "role_z":
        raise SystemExit(f"{args.detector_cal}: needs a role_z calibration")
    behavior = behaviors.load_behavior(args.behavior)
    arm = args.arm or behavior.default_arm
    load_model(args.model_name, dtype=DTYPES[args.dtype], device=args.device)
    model, tokenizer = require_model()
    last = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else depth_layer(last, args.layer_frac)
    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    items = behavior.items(arm, args.n_prompts)
    prompts = prompt_format.render_prompts(behavior, items, fmt)
    targets = [behavior.target_for(item, args.target_side) for item in items]
    if any(t is None for t in targets):
        raise SystemExit(f"{behavior.name}: some items have no {args.target_side} target")
    head = model.get_output_embeddings()
    W = head.weight.detach().float()
    bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    sigma, sigma_any = float(cal["sigma"]), float(cal.get("sigma_any", float("inf")))
    ceilings = [float(x) for x in args.ceilings.split(",")]
    logger.info("%s | injection layer %d, read/bound at layer %d | %d %s/%s prompts, %s",
                args.model_name, layer, last, len(prompts), behavior.name, arm, fmt)

    rows = []
    for start in range(0, len(prompts), args.batch_size):
        chunk, tchunk = prompts[start:start + args.batch_size], targets[start:start + args.batch_size]
        clean = clean_reference(chunk, layer, tchunk)
        det = DetectorSurrogate(cal, clean, layer, sigma_target=sigma - 0.5, beta=4.0, rule="scan")
        i_last = det.layers.index(last)
        h = clean["states"][last]
        logits = clean["logits"]
        recon = h @ W.T + (bias if bias is not None else 0)
        drift = float((recon - logits).abs().max() / logits.abs().max())
        if drift > 1e-3:
            raise SystemExit(f"hidden_states[{last}] is not what the head reads (drift {drift:.2e})")
        top = logits.argmax(dim=-1)
        first_target = clean["target_span"]["tokens"][:, 0]
        for b in range(len(chunk)):
            c = int(top[b])
            d = (W - W[c]).norm(dim=-1)
            margin = logits[b, c] - logits[b]
            ratio = margin / d.clamp_min(1e-30)
            ratio[c] = float("inf")
            need_any, v_any = ratio.min(dim=0)
            t = int(first_target[b])
            need_t = ratio[t]
            hn = float(h[b].norm())
            mean, sd = float(det.mean[i_last, b, 0]), float(det.sd[i_last, b, 0])
            floor = math.exp(mean)

            def z_of(norm: float) -> float:
                r = norm / hn
                return (0.5 * math.log(r * r + floor * floor) - mean) / sd

            z_any, z_t = z_of(float(need_any)), z_of(float(need_t))
            rows.append({
                "prompt_index": start + b, "prompt": chunk[b],
                "clean_token": c, "clean_str": tokenizer.decode([c]),
                "flip_token": int(v_any), "flip_str": tokenizer.decode([int(v_any)]),
                "target_token": t, "target_str": tokenizer.decode([t]),
                "target_margin": float(margin[t]),
                "h_norm_last": hn, "role": det.roles[b][0], "mean": mean, "sd": sd,
                "need_norm_any": float(need_any), "need_rel_any": float(need_any) / hn, "z_flip_any": z_any,
                "need_norm_target": float(need_t), "need_rel_target": float(need_t) / hn, "z_flip_target": z_t,
                "admissible_any": {f"{z:g}": z_any <= z for z in ceilings},
                "admissible_target": {f"{z:g}": z_t <= z for z in ceilings},
            })

    n = len(rows)
    zs_any = sorted(r["z_flip_any"] for r in rows)
    zs_t = sorted(r["z_flip_target"] for r in rows)
    summary = {
        "n": n, "sigma": sigma, "sigma_any": sigma_any,
        "z_flip_any_min": zs_any[0], "z_flip_any_median": zs_any[n // 2], "z_flip_any_max": zs_any[-1],
        "z_flip_target_min": zs_t[0], "z_flip_target_median": zs_t[n // 2], "z_flip_target_max": zs_t[-1],
        "need_rel_any_median": sorted(r["need_rel_any"] for r in rows)[n // 2],
        "admissible_any_rate": {f"{z:g}": sum(r["z_flip_any"] <= z for r in rows) / n for z in ceilings},
        "admissible_target_rate": {f"{z:g}": sum(r["z_flip_target"] <= z for r in rows) / n for z in ceilings},
    }
    logger.info("z at layer %d below which NO argmax flip is possible: min %.2f, median %.2f, max %.2f "
                "(sigma %.2f, sigma_any %.2f); to the target token: min %.2f, median %.2f",
                last, zs_any[0], zs_any[n // 2], zs_any[-1], sigma, sigma_any, zs_t[0], zs_t[n // 2])
    logger.info("%8s %14s %17s", "ceiling", "any flip adm.", "target flip adm.")
    for z in ceilings:
        logger.info("%8g %13.0f%% %16.0f%%", z, 100 * summary["admissible_any_rate"][f"{z:g}"],
                    100 * summary["admissible_target_rate"][f"{z:g}"])

    out = args.out or (experiment_dir(args.model_name, "pgd")
                       / f"flip_bound_{behavior.name}_{arm}_L{layer}_n{n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_name": args.model_name, "dtype": args.dtype, "behavior": behavior.name, "test_arm": arm,
        "prompt_format": fmt, "layer": layer, "bound_layer": last, "target_side": args.target_side,
        "detector_cal": str(args.detector_cal), "ceilings": ceilings,
        "summary": summary, "rows": rows,
    }, indent=2) + "\n")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
