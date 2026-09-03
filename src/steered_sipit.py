
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import Tensor


def _set_run_tag():
    for i, a in enumerate(sys.argv):
        if a == "--dtype" and i + 1 < len(sys.argv):
            os.environ.setdefault("AAT_DTYPE", sys.argv[i + 1])
    os.environ.setdefault("AAT_DTYPE", "float16")


_set_run_tag()

import behaviors
import detect
import prompt_format
import sipit
from generate import Intervention
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir, vocab_table_path
from steering import (
    build_steering_vectors,
    control_seed,
    random_direction,
    unit_delta,
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

ARMS = ("caa", "random")


def select_ids(layers_dir: Path, n: int) -> list[dict]:
    """Evenly spaced per category from the clean rows the calibration used."""
    rows = {}
    for line in (layers_dir / "sipit_layer_00.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r.get("category") or r["id"].rsplit("_", 1)[0]
    cats = sorted(set(rows.values()))
    per, extra = divmod(n, len(cats))
    picked = []
    for c_i, cat in enumerate(cats):
        ids = sorted(i for i, c in rows.items() if c == cat)
        take = min(per + (1 if c_i < extra else 0), len(ids))
        picked += [ids[i * len(ids) // take] for i in range(take)]
    return [{"id": i, "category": rows[i]} for i in picked]


@torch.no_grad()
def capture_layers(input_ids: Tensor, layers: list[int],
                   intervention: Intervention | None) -> dict[int, Tensor]:
    model, _ = require_model()
    ids = input_ids.to(model_device())
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    mask = torch.ones_like(ids)
    grabbed: dict[int, Tensor] = {}

    def grab_for(layer: int):
        def grab(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            grabbed[layer] = out[0].detach()
        return grab

    handles = []
    if intervention is not None:
        block = get_decoder_layers()[intervention.layer - 1]
        handles.append(block.register_forward_hook(intervention.hook(mask=mask)))
    for layer in layers:
        handles.append(get_decoder_layers()[layer - 1].register_forward_hook(grab_for(layer)))
    try:
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    # final norm at the model dtype, then the fp32 cast: the clean .pt states were
    # normed inside the fp16 forward, and the sanity check compares bit for bit
    return {layer: apply_final_norm(h, layer).float().cpu() for layer, h in grabbed.items()}


def sanity(prompts: list[dict], texts: dict[str, str], act_dir: Path, n_layers: int):
    _, tokenizer = require_model()
    p = prompts[0]
    blob = torch.load(act_dir / f"{p['id']}.pt", weights_only=False)
    ids = tokenizer(texts[p["id"]], return_tensors="pt")["input_ids"][0]
    if ids.shape[0] != blob["activations"].shape[1]:
        raise SystemExit(f"sanity: tokenization drift on {p['id']}: "
                         f"{ids.shape[0]} vs {blob['activations'].shape[1]}")
    got = capture_layers(ids, list(range(1, n_layers)), None)
    worst = 0.0
    for layer in range(1, n_layers):
        diff = (got[layer] - blob["activations"][layer].float()).abs().max().item()
        worst = max(worst, diff)
    logger.info("sanity: clean capture vs %s.pt, max |diff| %.3g over layers 1..%d",
                p["id"], worst, n_layers - 1)
    if worst > 1e-4:
        raise SystemExit("sanity: captured clean states do not match the saved "
                         "activations; steered rows would not be comparable")


def build_deltas(behavior, fmt: str, inj_layers: list[int], caa_sign: float):
    steering = build_steering_vectors(prompt_format.render_contrast_pairs(behavior, fmt),
                                      layers=inj_layers)
    return {layer: (caa_sign * d, s) for layer, (d, s) in steering.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Steered SipIt trajectories and the detector's true-positive rate.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    add_model_args(parser, default_dtype="float16")
    parser.add_argument("--behavior", type=str, default="sentiment")
    parser.add_argument("--prompt_format", type=str, default="auto",
                        choices=list(prompt_format.FORMATS))
    parser.add_argument("--arms", type=str, default="caa,random")
    parser.add_argument("--inj_layers", type=str, default="3,6,9")
    parser.add_argument("--fractions", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--n_prompts", type=int, default=25)
    parser.add_argument("--caa_sign", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rel_tol", type=float, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/trajectory_bank_prompts.json")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION,
                        help="calibration file under results/<slug>/sipit/ to score "
                             "against (default: the shipped one). k=1 is the rule "
                             "finding 12 says fits a single-position attack")
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "steered_sipit.log")

    sipit_dir = experiment_dir(args.model_name, "sipit")
    out_dir = sipit_dir / "steered"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "steered_rows.jsonl"

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"--arms: expected a subset of {ARMS}, got {a!r}")
    inj_layers = [int(v) for v in args.inj_layers.split(",") if v.strip()]
    fractions = [float(v) for v in args.fractions.split(",") if v.strip()]

    if args.score_only:
        score(args, sipit_dir, rows_path, args.calibration)
        return

    dtype = DTYPES[args.dtype]
    rel_tol = rel_tol_for(dtype) if args.rel_tol is None else args.rel_tol
    model, tokenizer = load_model(args.model_name, dtype=dtype, device=args.device)
    n_blocks = model.config.num_hidden_layers
    n_layers = n_blocks + 1
    hidden = model.config.hidden_size
    logger.info("model on %s, %s, rel_tol %g, %d hidden layers", model_device(),
                args.dtype, rel_tol, n_layers)
    if max(inj_layers) >= n_blocks:
        raise SystemExit(f"--inj_layers: the final block ({n_blocks}) is refused, "
                         "same reason as posX_attack: the model's own norm sits "
                         "between the injection and the table there")

    bank = json.loads(Path(args.data_path).read_text())
    texts = {p["id"]: p["text"] for p in bank["prompts"]}
    act_dir = vocab_table_path(args.model_name).parent.parent
    prompts = select_ids(sipit_dir / "layers", args.n_prompts)
    logger.info("%d prompts: %s ...", len(prompts), [p["id"] for p in prompts[:6]])

    sanity(prompts, texts, act_dir, n_layers)
    if args.sanity_only:
        return

    behavior = behaviors.load_behavior(args.behavior)
    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    deltas = build_deltas(behavior, fmt, inj_layers, args.caa_sign)
    for layer, (d, s) in deltas.items():
        logger.info("caa direction layer %d: ||dir|| %.3f, scale %.3f", layer,
                    d.norm().item(), s.item())

    vocab_table, layout = sipit.load_vocab_table(
        args.vocab_path or str(vocab_table_path(args.model_name)),
        expect_model=args.model_name, expect_dtype=args.dtype)

    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["inj_layer"], r["fraction"], r["id"], r["layer"]))
        logger.info("resuming: %d rows already on disk", len(done))

    token_ids = {}
    for p in prompts:
        ids = tokenizer(texts[p["id"]], return_tensors="pt")["input_ids"][0]
        token_ids[p["id"]] = ids

    cells = [(arm, L, f) for arm in arms for L in inj_layers for f in fractions]
    t_start = time.time()
    with rows_path.open("a") as sink:
        for c_i, (arm, L, f) in enumerate(cells, 1):
            todo_layers = [l for l in range(L, n_layers)
                           if any((arm, L, f, p["id"], l) not in done for p in prompts)]
            if not todo_layers:
                continue
            direction, scale = deltas[L]
            captured = {}
            for p_i, p in enumerate(prompts):
                if arm == "caa":
                    delta = unit_delta(direction, scale, f)
                else:
                    rand = random_direction(hidden, control_seed(args.seed, L, f, p_i))
                    delta = unit_delta(rand, scale, f)
                iv = Intervention(layer=L, delta=delta, positions="all")
                captured[p["id"]] = capture_layers(token_ids[p["id"]], todo_layers, iv)
            for layer in todo_layers:
                vocab_layer = sipit.load_vocab_layer(vocab_table, layer, layout)
                for p in prompts:
                    if (arm, L, f, p["id"], layer) in done:
                        continue
                    gold = token_ids[p["id"]].tolist()
                    t0 = time.time()
                    steps = sipit.sipit(captured[p["id"]][layer], layer, vocab_layer,
                                        rel_tol=rel_tol, stop_on_fail=True, gold=gold)
                    row = {
                        "id": p["id"], "category": p["category"], "arm": arm,
                        "inj_layer": L, "fraction": f, "layer": layer,
                        "n_target": len(gold), "n_recovered": len(steps),
                        "elapsed": time.time() - t0,
                        "steps": [{k: s[k] for k in ("token", "residual", "gap", "tol",
                                                     "h_norm", "tried", "matched",
                                                     "gold_token", "correct")
                                   if k in s} for s in steps],
                    }
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                del vocab_layer
            logger.info("[%d/%d] %s L=%d f=%g done, %.0fs elapsed total",
                        c_i, len(cells), arm, L, f, time.time() - t_start)

    score(args, sipit_dir, rows_path, args.calibration)


DEFAULT_CALIBRATION = "detector_calibration.json"


def summary_suffix(calibration: str) -> str:
    """'detector_calibration_k1.json' -> '_k1', so one experiment's scorings coexist."""
    stem = Path(calibration).stem
    base = Path(DEFAULT_CALIBRATION).stem
    return stem[len(base):] if stem.startswith(base) else f"_{stem}"


def load_calibration(sipit_dir: Path, name: str = DEFAULT_CALIBRATION):
    """The shipped rule as it sits on disk: settings plus per-layer clean mean/sd."""
    cal = json.loads((sipit_dir / name).read_text())
    stats = {int(l): detect.LayerStat(v["n"], v["mean"], v["sd"])
             for l, v in cal["per_layer"].items()}
    return cal, stats, sorted(stats)


def bank_clean_scores(sipit_dir: Path, k: int, cal: dict | None = None) -> dict:
    out: dict[str, dict[int, float]] = {}
    for s in detect.score_rows(detect.load_trajectories(sipit_dir / "layers"), k, cal):
        out.setdefault(s["id"], {})[s["layer"]] = s["score"]
    return out


def build_operating_points(cal: dict, reference: dict, stats: dict, layers: list[int]) -> dict:
    ops = {
        "shipped": dict(sigma=cal["sigma"], sigma_any=cal["sigma_any"],
                        min_run=cal["min_run"]),
        "sigma5_run3": dict(sigma=5.0, sigma_any=7.5, min_run=3),
        "sigma4_run6": dict(sigma=4.0, sigma_any=6.0, min_run=6),
    }
    for op in ops.values():
        rep = detect.prompt_fpr(reference, stats, layers, sigma=op["sigma"],
                                min_run=op["min_run"], stride=cal["stride"],
                                sigma_any=op["sigma_any"], dense_tail=cal["dense_tail"])
        op["fpr_clean"] = rep["fpr_prompt"]
        op["n_clean"] = rep["n_prompts"]
    return ops


def summarize_cell(profiles: dict, inj_layer: int, stats: dict, layers: list[int],
                   cal: dict, operating_points: dict) -> dict:
    """One (arm, layer, strength) cell scored by detect.py's own rule, unchanged."""
    entry: dict = {"n": len(profiles)}
    inj_scores, inj_hits = [], 0
    for prof in profiles.values():
        s = prof.get(inj_layer)
        if s is not None and inj_layer in stats:
            threshold = stats[inj_layer].threshold(cal["sigma"])
            inj_scores.append(s / threshold if threshold > 0 else math.inf)
            inj_hits += s > threshold
    for name, op in operating_points.items():
        rep = detect.prompt_fpr(profiles, stats, layers, sigma=op["sigma"],
                                min_run=op["min_run"], stride=cal["stride"],
                                sigma_any=op["sigma_any"], dense_tail=cal["dense_tail"])
        entry[f"tpr_{name}"] = rep["fpr_prompt"]
        if name == "shipped":
            entry["flagged"] = sum(v["flagged"] for v in rep["per_id"].values())
            entry["lone_only"] = sum(v["flagged"] and v["longest_run"] < op["min_run"]
                                     for v in rep["per_id"].values())
            entry["mean_longest_run"] = rep["mean_longest_run"]
            entry["per_id"] = {pid: v["flagged"] for pid, v in rep["per_id"].items()}
    entry["inj_layer_hit_rate"] = inj_hits / max(1, len(inj_scores))
    entry["inj_score_over_threshold_median"] = (
        statistics.median(inj_scores) if inj_scores else float("nan"))
    return entry


def score(args, sipit_dir: Path, rows_path: Path,
          calibration: str = DEFAULT_CALIBRATION):
    cal, stats, layers = load_calibration(sipit_dir, calibration)
    k = cal["k"]
    clean_score = bank_clean_scores(sipit_dir, k, cal)

    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    cells = {}
    for r in rows:
        cells.setdefault((r["arm"], r["inj_layer"], r["fraction"]), []).append(r)

    operating_points = build_operating_points(cal, clean_score, stats, layers)

    summary = []
    for (arm, L, f), cell_rows in sorted(cells.items()):
        profiles = {}
        for r in cell_rows:
            score = detect.row_score(r, k, cal)
            if score != score:
                continue
            profiles.setdefault(r["id"], {})[r["layer"]] = score
        for pid in profiles:
            for layer in layers:
                if layer < L and layer in clean_score.get(pid, {}):
                    profiles[pid].setdefault(layer, clean_score[pid][layer])
        entry = {"arm": arm, "inj_layer": L, "fraction": f}
        entry.update(summarize_cell(profiles, L, stats, layers, cal, operating_points))
        summary.append(entry)

    gaps_path = sipit_dir.parent / "sentiment" / "gaps.json"
    flips = {}
    if gaps_path.exists():
        gaps = json.loads(gaps_path.read_text())
        for g in gaps.get("layers", []):
            flips[(g["layer"], g["fraction"])] = g.get("flip_rate")

    logger.info("%-7s %3s %6s %4s | %8s %11s %11s | %9s %8s %10s", "arm", "L", "frac",
                "n", "TPR", "sigma5run3", "sigma4run6", "inj hit", "med s/t", "flip@frac")
    for e in summary:
        flip = flips.get((e["inj_layer"], e["fraction"]))
        logger.info("%-7s %3d %6g %4d | %7.0f%% %10.0f%% %10.0f%% | %8.0f%% %8.2f %10s",
                    e["arm"], e["inj_layer"], e["fraction"], e["n"],
                    100 * e["tpr_shipped"], 100 * e["tpr_sigma5_run3"],
                    100 * e["tpr_sigma4_run6"], 100 * e["inj_layer_hit_rate"],
                    e["inj_score_over_threshold_median"],
                    "-" if flip is None else f"{100 * flip:.0f}%")

    out = {
        "model": args.model_name, "dtype": args.dtype,
        "calibration_file": calibration,
        "calibration": {kk: cal[kk] for kk in ("k", "sigma", "sigma_any", "min_run",
                                               "stride", "dense_tail", "fpr_prompt",
                                               "n_prompts")},
        "operating_points": operating_points,
        "cells": summary,
    }
    path = sipit_dir / "steered" / f"tpr_summary{summary_suffix(calibration)}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
