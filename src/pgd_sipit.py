
import argparse
import json
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
import steered_sipit
from generate import Intervention
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir, vocab_table_path
from steered_sipit import capture_layers
from utils import (
    DTYPES,
    add_model_args,
    get_decoder_layers,
    load_model,
    model_device,
    rel_tol_for,
    require_model,
)

logger = get_logger(__name__)

CLEAN_ARM = "clean"
# The delta for these arms is a function of (budget, seed, forward pass) only, so
# it is byte-identical across objectives; they are stored once under this key and
# reused rather than re-inverted per objective. Asserted at load, not assumed.
OBJECTIVE_FREE = ("clean", "random", "caa")
SHARED = "-"


def objective_key(arm: str, objective: str) -> str:
    return SHARED if arm in OBJECTIVE_FREE else objective


def positions_key(arm: str, n_positions: int) -> int:
    """0 marks the clean arm, whose pass is identical at every m and so is shared.

    Every other arm's delta changes with m, so m is part of the cell identity --
    without it an m=5 run resumes into the m=1 rows and the sweep silently
    collapses to one cell.
    """
    return 0 if arm == CLEAN_ARM else n_positions


def load_deltas(path: Path) -> tuple[int, float, int, dict[tuple[str, str, int], Tensor]]:
    """Stage 1's winning perturbations, keyed (constraint, arm, prompt_index).

    (hidden,) is the single-position attack, (m, hidden) the spread one with slot j
    j tokens back from the end. Files predating --n_positions read as m=1.
    """
    blob = torch.load(path, weights_only=False)
    n_positions = int(blob.get("n_positions", 1))
    out = {}
    for key, vec in blob["deltas"].items():
        constraint, arm, index = key.split("|")
        vec = vec.float()
        if (vec.dim() > 1) != (n_positions > 1):
            raise SystemExit(f"{path}: n_positions={n_positions} but delta {key} is "
                             f"shape {tuple(vec.shape)}")
        out[(constraint, arm, int(index))] = vec
    return int(blob["layer"]), float(blob["budget"]), n_positions, out


def stage1_rows(deltas_path: Path) -> dict[tuple[str, str, int], dict]:
    """Stage 1's own per-row record, keyed (constraint, arm, prompt_index)."""
    jsonl = deltas_path.with_name(deltas_path.name.replace("_deltas.pt", ".jsonl"))
    out = {}
    if not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["constraint"], r["arm"], r["prompt_index"])] = r
    return out


@torch.no_grad()
def read_logits(input_ids: Tensor, intervention: Intervention | None) -> Tensor:
    model, _ = require_model()
    ids = input_ids.to(model_device())
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    mask = torch.ones_like(ids)
    handles = []
    if intervention is not None:
        block = get_decoder_layers()[intervention.layer - 1]
        handles.append(block.register_forward_hook(intervention.hook(mask=mask)))
    try:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, -1].float().cpu()


def injected_positions(n_slots: int, n_tokens: int) -> list[int]:
    """Absolute positions of the live slots in an unpadded sequence of n_tokens."""
    return [n_tokens - 1 - j for j in range(min(n_slots, n_tokens))]


def intervention_for(layer: int, delta: Tensor | None,
                     n_tokens: int | None = None) -> Intervention | None:
    if delta is None:
        return None
    if delta.dim() == 1:
        return Intervention(layer=layer, delta=delta, positions="index", index=-1)
    if n_tokens is None:
        raise ValueError("a multi-position delta needs the prompt length")
    # slots past the front of a short prompt are DROPPED, not clamped: clamping puts
    # two of them on token 0, and an advanced-index += keeps only the last write
    live = len(injected_positions(delta.shape[0], n_tokens))
    dead = delta[live:]
    if dead.numel() and float(dead.abs().max()) > 0:
        raise SystemExit(f"{dead.shape[0]} slot(s) fall past the start of a "
                         f"{n_tokens}-token prompt and are not zero; the two stages "
                         "disagree about how many positions were attacked")
    index = torch.arange(-1, -live - 1, -1).unsqueeze(0)
    return Intervention(layer=layer, delta=delta[:live].unsqueeze(0),
                        positions="index", index=index)


def sanity(token_ids, clean_states, deltas, stage1, constraint, inj_layer, tol=0.02):
    worst_norm, worst_below = 0.0, 0.0
    checked = 0
    for i, ids in enumerate(token_ids):
        if ids.shape[0] < 2:
            raise SystemExit(f"prompt {i} is {ids.shape[0]} token(s); "
                             "positions='index' injects at prefill only")
        row = stage1.get((constraint, "pgd", i))
        delta = deltas.get((constraint, "pgd", i))
        if row is None or delta is None:
            continue
        n = int(ids.shape[0])
        at = injected_positions(delta.shape[0] if delta.dim() > 1 else 1, n)
        if row.get("inject_at") is not None and row["inject_at"][:len(at)] != at:
            raise SystemExit(
                f"sanity: prompt {i} was attacked at {row['inject_at'][:len(at)]} in "
                f"stage 1 but resolves to {at} here; the two stages disagree about "
                "where the perturbation goes")
        want = (row.get("h_norm_by_position") or [row["h_norm"]])[:len(at)]
        for pos, w in zip(at, want):
            got = float(clean_states[i][inj_layer][pos].norm())
            worst_norm = max(worst_norm, abs(got - w) / max(w, 1e-12))
        if inj_layer > 1:
            steered = capture_layers(ids, [inj_layer - 1],
                                     intervention_for(inj_layer, delta, n))
            diff = (steered[inj_layer - 1] - clean_states[i][inj_layer - 1]).abs().max()
            worst_below = max(worst_below, float(diff))
        checked += 1
    if not checked:
        raise SystemExit("sanity: no stage-1 rows matched the deltas file")
    logger.info("sanity: %d prompts | max ||h|| drift vs stage-1 fp32 %.3g | "
                "max |diff| below the injection %.3g", checked, worst_norm, worst_below)
    if worst_norm > tol:
        raise SystemExit(
            f"sanity: ||h|| at the attacked position differs from stage 1 by "
            f"{worst_norm:.1%} (> {tol:.0%}). The two stages are not steering the "
            "same position -- check the padding convention before trusting a TPR.")
    if worst_below > 0:
        raise SystemExit("sanity: layers below the injection are not identical to "
                         "the clean pass; the control cannot supply them")


def main():
    parser = argparse.ArgumentParser(
        description="Score PGD attacks against the shipped trajectory detector.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    add_model_args(parser, default_dtype="float16")
    parser.add_argument("--deltas", type=Path, required=True,
                        help="stage-1 <stem>_deltas.pt from pgd_attack.py --save_deltas")
    parser.add_argument("--objective", type=str, required=True,
                        help="label for the pgd arm; the control arms are stored "
                             "objective-free and shared across runs")
    parser.add_argument("--budget", type=float, default=None,
                        help="label only; defaults to the budget stored in the .pt")
    parser.add_argument("--constraint", type=str, default="all")
    parser.add_argument("--arms", type=str, default="pgd,random")
    parser.add_argument("--behavior", type=str, default="sentiment")
    parser.add_argument("--arm", type=str, default=None, help="behavior test arm")
    parser.add_argument("--n_prompts", type=int, default=0, help="0 = the whole arm")
    parser.add_argument("--prompt_format", type=str, default="auto",
                        choices=list(prompt_format.FORMATS))
    parser.add_argument("--rel_tol", type=float, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--calibrations", type=str,
                        default="detector_calibration.json,detector_calibration_k1.json")
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "pgd_sipit.log")

    sipit_dir = experiment_dir(args.model_name, "sipit")
    out_dir = experiment_dir(args.model_name, "pgd_sipit")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "pgd_rows.jsonl"
    calibrations = [c for c in args.calibrations.split(",") if c]

    if args.score_only:
        for cal in calibrations:
            score(args, sipit_dir, out_dir, rows_path, cal)
        return

    inj_layer, stored_budget, n_positions, deltas = load_deltas(args.deltas)
    budget = args.budget if args.budget is not None else stored_budget
    stage1 = stage1_rows(args.deltas)
    arms = [a for a in args.arms.split(",") if a]

    dtype = DTYPES[args.dtype]
    rel_tol = rel_tol_for(dtype) if args.rel_tol is None else args.rel_tol
    model, tokenizer = load_model(args.model_name, dtype=dtype, device=args.device)
    n_layers = model.config.num_hidden_layers + 1
    logger.info("model on %s, %s, rel_tol %g | injection layer %d, budget %g, "
                "%d position(s), objective %s", model_device(), args.dtype, rel_tol,
                inj_layer, budget, n_positions, args.objective)
    if inj_layer >= n_layers - 1:
        raise SystemExit(f"injection layer {inj_layer} is the final block; the "
                         "model's own norm sits between it and the table")

    behavior = behaviors.load_behavior(args.behavior)
    arm_name = args.arm or behavior.default_arm
    items = behavior.items(arm_name, args.n_prompts)
    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    prompts = prompt_format.render_prompts(behavior, items, fmt)
    token_ids = [tokenizer(p, return_tensors="pt")["input_ids"][0] for p in prompts]
    logger.info("%d prompts from %s/%s, format %s, lengths %d-%d tokens", len(prompts),
                behavior.name, arm_name, fmt,
                min(t.shape[0] for t in token_ids), max(t.shape[0] for t in token_ids))

    missing = [i for a in arms if a != CLEAN_ARM for i in range(len(prompts))
               if (args.constraint, a, i) not in deltas]
    if missing:
        raise SystemExit(
            f"{args.deltas} has no delta for {len(missing)} (arm, prompt) pairs. "
            "pgd_attack --save_deltas collects only the rows that invocation "
            "computed, so it must be paired with --force, not a resume.")

    # Layer 0 excluded by construction -- see the module docstring.
    all_layers = list(range(1, n_layers))
    steered_layers = list(range(inj_layer, n_layers))

    logger.info("capturing clean states for %d prompts over layers 1..%d",
                len(prompts), n_layers - 1)
    clean_states = {i: capture_layers(ids, all_layers, None)
                    for i, ids in enumerate(token_ids)}
    clean_logits = {i: read_logits(ids, None) for i, ids in enumerate(token_ids)}

    if arms:
        sanity(token_ids, clean_states, deltas, stage1, args.constraint, inj_layer)
    else:
        # --arms "" is clean-only: no delta is applied, so there is nothing for the
        # position/invariance checks to verify. Used to build a clean trajectory set
        # for calibration on a prompt distribution the bank does not cover.
        logger.info("no attack arms: clean trajectories only, sanity checks skipped")
    if args.sanity_only:
        return

    vocab_table, layout = sipit.load_vocab_table(
        args.vocab_path or str(vocab_table_path(args.model_name)),
        expect_model=args.model_name, expect_dtype=args.dtype)

    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r.get("behavior", "sentiment"), r.get("test_arm", ""),
                      r["objective"], positions_key(r["arm"], r.get("n_positions", 1)),
                      r["constraint"], r["budget"], r["arm"], r["prompt_index"],
                      r["layer"]))
        logger.info("resuming: %d rows already on disk", len(done))

    cells = [(CLEAN_ARM, 0.0, all_layers)] + [(a, budget, steered_layers) for a in arms]
    t_start = time.time()
    with rows_path.open("a") as sink:
        for arm, cell_budget, layers in cells:
            okey = objective_key(arm, args.objective)
            mkey = positions_key(arm, n_positions)
            todo = [l for l in layers
                    if any((behavior.name, arm_name, okey, mkey, args.constraint,
                            cell_budget, arm, i, l) not in done
                           for i in range(len(prompts)))]
            if not todo:
                logger.info("%-7s already complete", arm)
                continue
            captured, logits, rel_dev = {}, {}, {}
            for i, ids in enumerate(token_ids):
                if arm == CLEAN_ARM:
                    captured[i], logits[i] = clean_states[i], clean_logits[i]
                    rel_dev[i] = {l: 0.0 for l in todo}
                    continue
                iv = intervention_for(inj_layer, deltas[(args.constraint, arm, i)],
                                      int(ids.shape[0]))
                captured[i] = capture_layers(ids, todo, iv)
                logits[i] = read_logits(ids, iv)
                # the same quantity stage 1 constrained, recomputed at the deployed
                # precision: the transfer is measured rather than assumed. Max over
                # the injected positions, which is what the constraint capped.
                at = injected_positions(n_positions, int(ids.shape[0]))
                rel_dev[i] = {}
                for l in todo:
                    ref = clean_states[i][l][at]
                    rel_dev[i][l] = float(((captured[i][l][at] - ref).norm(dim=-1)
                                           / ref.norm(dim=-1).clamp_min(1e-12)).max())
            for layer in todo:
                vocab_layer = sipit.load_vocab_layer(vocab_table, layer, layout)
                for i in captured:
                    if (behavior.name, arm_name, okey, mkey, args.constraint,
                            cell_budget, arm, i, layer) in done:
                        continue
                    gold = token_ids[i].tolist()
                    t0 = time.time()
                    steps = sipit.sipit(captured[i][layer], layer, vocab_layer,
                                        rel_tol=rel_tol, stop_on_fail=True, gold=gold)
                    inj_pos = injected_positions(n_positions, len(gold))
                    inj_steps = [steps[q] for q in inj_pos if q < len(steps)] or [steps[-1]]
                    # the worst injected position: what a k=1 detector reads, and the
                    # floor on what any top-k mean over them can read
                    at_inj = max(inj_steps, key=lambda x: x["residual"] / max(x["h_norm"], 1e-12)
                                 if x.get("h_norm") else 0.0)
                    s1 = stage1.get((args.constraint, arm, i), {})
                    row = {
                        "id": f"{args.behavior}_{i:03d}", "prompt_index": i,
                        "arm": arm, "objective": okey, "budget": cell_budget,
                        "n_positions": mkey,
                        # the prompt SET is part of the cell: a clean control fitted on
                        # 7-token sentiment fragments is not the control for 40-token
                        # chat-templated jailbreak prompts
                        "behavior": behavior.name, "test_arm": arm_name,
                        "constraint": args.constraint, "inj_layer": inj_layer,
                        "layer": layer, "n_target": len(gold), "n_recovered": len(steps),
                        "exact": [s["token"] for s in steps] == gold[:len(steps)],
                        "inj_rel_residual": (at_inj["residual"] / at_inj["h_norm"]
                                             if at_inj.get("h_norm") else None),
                        # the crude rel_tol oracle's verdict, free, so a trajectory-rule
                        # detection is never confused with a tolerance detection
                        "inj_matched": all(bool(x["matched"]) for x in inj_steps),
                        "n_injected": len(inj_pos),
                        "inj_rel_residual_by_position": [
                            x["residual"] / x["h_norm"] if x.get("h_norm") else None
                            for x in inj_steps],
                        "rel_dev_fp16": rel_dev[i][layer],
                        "rel_dev_fp32": (s1.get("rel_dev_by_layer") or {}).get(str(layer)),
                        "flip_fp16": int(logits[i].argmax() != clean_logits[i].argmax()),
                        "flip_fp32": s1.get("flip"),
                        "elapsed": time.time() - t0,
                        "steps": [{k: s[k] for k in ("token", "residual", "gap", "tol",
                                                     "h_norm", "tried", "matched",
                                                     "gold_token", "correct")
                                   if k in s} for s in steps],
                    }
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                del vocab_layer
            logger.info("%-7s %2d layers x %d prompts done, %.0fs elapsed total",
                        arm, len(todo), len(captured), time.time() - t_start)

    for cal in calibrations:
        score(args, sipit_dir, out_dir, rows_path, cal)


def score(args, sipit_dir: Path, out_dir: Path, rows_path: Path, calibration: str):
    cal, stats, layers = steered_sipit.load_calibration(sipit_dir, calibration)
    k = cal["k"]

    bank = steered_sipit.bank_clean_scores(sipit_dir, k, cal)
    operating_points = steered_sipit.build_operating_points(cal, bank, stats, layers)

    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    cells = {}
    for r in rows:
        cells.setdefault((r.get("behavior", "sentiment"), r.get("test_arm", ""),
                          r["objective"],
                          positions_key(r["arm"], r.get("n_positions", 1)),
                          r["constraint"], r["budget"], r["arm"]), []).append(r)

    def profiles_of(cell_rows):
        prof = {}
        for r in cell_rows:
            score = detect.row_score(r, k, cal)
            if score == score:   # not NaN
                prof.setdefault(r["prompt_index"], {})[r["layer"]] = score
        return prof

    # the clean control must come from the SAME prompt set: a control fitted on 7-token
    # sentiment fragments is not the control for 40-token chat-templated jailbreaks
    controls = {ck[:2]: profiles_of(v) for ck, v in cells.items() if ck[-1] == CLEAN_ARM}

    summary = []
    for key, cell_rows in sorted(cells.items()):
        beh, beh_arm, obj, m, scope, budget, arm = key
        control = controls.get((beh, beh_arm), {})
        L = cell_rows[0]["inj_layer"]
        prof = profiles_of(cell_rows)
        if arm != CLEAN_ARM:
            for pid in prof:
                for layer in layers:
                    if layer < L and layer in control.get(pid, {}):
                        prof[pid].setdefault(layer, control[pid][layer])
        entry = {"behavior": beh, "test_arm": beh_arm, "objective": obj,
                 "budget": budget, "arm": arm, "inj_layer": L,
                 "constraint": scope, "n_positions": m,
                 "n_injected_mean": (sum(r.get("n_injected", 1) for r in cell_rows)
                                     / max(1, len(cell_rows)))}
        entry.update(steered_sipit.summarize_cell(prof, L, stats, layers, cal,
                                                  operating_points))
        at_inj = [r for r in cell_rows if r["layer"] == L]
        realized = [r["inj_rel_residual"] for r in at_inj
                    if r["inj_rel_residual"] is not None]
        entry["realized_rel_residual_median"] = (
            statistics.median(realized) if realized else float("nan"))
        entry["inj_matched_frac"] = (sum(r["inj_matched"] for r in at_inj)
                                     / max(1, len(at_inj)))
        entry["exact_frac"] = sum(r["exact"] for r in cell_rows) / max(1, len(cell_rows))
        entry["flip_fp16"] = sum(r["flip_fp16"] for r in at_inj) / max(1, len(at_inj))
        f32 = [r["flip_fp32"] for r in at_inj if r["flip_fp32"] is not None]
        entry["flip_fp32"] = sum(bool(v) for v in f32) / len(f32) if f32 else None
        shifts = [prof[p][L] - control[p][L] for p in prof
                  if p in control and L in prof[p] and L in control[p]]
        entry["score_shift_at_inj_median"] = (
            statistics.median(shifts) if shifts else float("nan"))
        dev = [(r["rel_dev_fp16"], r["rel_dev_fp32"]) for r in at_inj
               if r.get("rel_dev_fp32")]
        entry["rel_dev_fp16_median"] = (statistics.median([a for a, _ in dev])
                                        if dev else float("nan"))
        entry["rel_dev_transfer_max_drift"] = (
            max(abs(a - b) / b for a, b in dev) if dev else float("nan"))
        summary.append(entry)

    logger.info("scored against %s (k=%d, %s), bank FPR %.0f%% at the shipped point",
                calibration, k, cal.get("statistic", "topk"),
                100 * operating_points["shipped"]["fpr_clean"])
    logger.info("%-13s %-9s %-9s %7s %-7s %3s %3s | %7s %10s %10s | %8s %7s | "
                "%9s %8s %6s %6s",
                "behavior", "objective", "scope", "budget", "arm", "m", "n", "TPR",
                "sigma5run3", "sigma4run6", "inj hit", "med s/t", "realized", "shift",
                "flip16", "flip32")
    for e in summary:
        tag = "   <- FPR, this distribution" if e["arm"] == CLEAN_ARM else ""
        logger.info("%-13s %-9s %-9s %7g %-7s %3d %3d | %6.0f%% %9.0f%% %9.0f%% | "
                    "%7.0f%% %7.2f | %9.2e %8.1e %5.0f%% %6s%s",
                    e["behavior"], e["objective"], e["constraint"], e["budget"],
                    e["arm"], e["n_positions"], e["n"],
                    100 * e["tpr_shipped"], 100 * e["tpr_sigma5_run3"],
                    100 * e["tpr_sigma4_run6"], 100 * e["inj_layer_hit_rate"],
                    e["inj_score_over_threshold_median"],
                    e["realized_rel_residual_median"], e["score_shift_at_inj_median"],
                    100 * e["flip_fp16"],
                    "-" if e["flip_fp32"] is None else f"{100 * e['flip_fp32']:.0f}%", tag)

    out = {
        "model": args.model_name, "dtype": args.dtype,
        "calibration_file": calibration,
        "calibration": {kk: cal[kk] for kk in ("k", "sigma", "sigma_any", "min_run",
                                               "stride", "dense_tail", "fpr_prompt",
                                               "n_prompts")},
        "operating_points": operating_points,
        "cells": summary,
    }
    path = out_dir / f"tpr_summary{steered_sipit.summary_suffix(calibration)}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
