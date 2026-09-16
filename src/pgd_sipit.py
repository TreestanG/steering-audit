
import argparse
import hashlib
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
import provenance
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


def resume_key(r: dict, fill: dict | None = None) -> tuple | None:
    """Identity of one inverted (cell, prompt, layer) row, or None when the row cannot
    be placed: it predates the prompt count, or is an attack row with no delta hash.
    A same-directory layer or seed sweep writes rows that differ only in the injection
    layer or the delta, so both are part of the key. `fill` supplies the current run's
    values for legacy rows the caller has chosen to trust."""
    attack = r["arm"] != CLEAN_ARM
    fill = fill or {}
    n_prompts = r.get("n_prompts", fill.get("n_prompts"))
    delta = r.get("deltas_sha256", fill.get("deltas_sha256")) if attack else None
    inj = r.get("inj_layer", fill.get("inj_layer")) if attack else None
    if n_prompts is None or (attack and not delta):
        return None
    return (r.get("behavior", "sentiment"), r.get("test_arm", ""), r["objective"],
            positions_key(r["arm"], r.get("n_positions", 1)), r["constraint"], r["budget"],
            r["arm"], r["prompt_index"], r["layer"], inj, delta)


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


def clean_prefix(clean_steps: dict, behavior_name: str, arm_name: str, i: int, layer: int,
                 gold: list[int], first_inj: int) -> list[dict] | None:
    """The clean row's steps for the positions a prefill-only injection at the last m
    positions cannot reach. The model is causal and the delta lands at prefill, so
    those states are the clean ones at every layer and their inversion is the same."""
    hit = clean_steps.get((behavior_name, arm_name, i, layer))
    if hit is None or hit[0] != len(gold) or first_inj <= 0:
        return None
    prefix = hit[1][:first_inj]
    if any(s.get("gold_token") != gold[q] for q, s in enumerate(prefix)):
        return None
    return prefix


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
    parser.add_argument("--rel_tol_by_layer", type=str, default=None,
                        help="JSON with a per-layer relative tolerance (see sipit.py)")
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--calibrations", type=str,
                        default="detector_calibration.json,detector_calibration_k1.json")
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--stop_on", type=str, default="miss", choices=list(sipit.STOP_MODES),
                        help="when a scan stops: miss (default) at the first position over "
                             "tolerance; wrong only where the recovered token differs from "
                             "gold, so the fp16 floor at a template position does not "
                             "truncate the row before the injected one (see sipit.py)")
    parser.add_argument("--redo_layers", type=str, default="",
                        help="comma-separated layers at which THIS run's cells (its clean "
                             "control and its budget's arms, at this prompt count) that were "
                             "inverted under a different --stop_on are dropped and "
                             "re-inverted instead of resumed. Other cells and rows already "
                             "under this --stop_on are untouched, so the ladder's per-budget "
                             "calls neither undo each other nor strand another cell")
    parser.add_argument("--no_reuse_clean", action="store_true",
                        help="re-invert every position of a steered prompt instead of "
                             "reusing the clean row for the positions ahead of the "
                             "injection, which the intervention cannot reach")
    parser.add_argument("--trust_legacy_rows", action="store_true",
                        help="resume from rows written before n_prompts and deltas_sha256 "
                             "were recorded, treating them as this run's. Off by default: "
                             "such rows cannot be placed and are left on disk unused")
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
    deltas_sha = hashlib.sha256(args.deltas.read_bytes()).hexdigest()
    budget = args.budget if args.budget is not None else stored_budget
    stage1 = stage1_rows(args.deltas)
    arms = [a for a in args.arms.split(",") if a]

    dtype = DTYPES[args.dtype]
    rel_tol = rel_tol_for(dtype) if args.rel_tol is None else args.rel_tol
    if args.rel_tol_by_layer:
        rel_tol = sipit.load_rel_tol_by_layer(args.rel_tol_by_layer, rel_tol)
    model, tokenizer = load_model(args.model_name, dtype=dtype, device=args.device)
    n_layers = model.config.num_hidden_layers + 1
    logger.info("model on %s, %s, rel_tol %s | injection layer %d, budget %g, "
                "%d position(s), objective %s", model_device(), args.dtype,
                f"per layer from {args.rel_tol_by_layer}" if args.rel_tol_by_layer else rel_tol,
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

    stop_on = sipit.STOP_MODES[args.stop_on]
    redo = {int(x) for x in args.redo_layers.split(",") if x}
    done = set()
    clean_steps: dict[tuple, tuple[int, list[dict]]] = {}
    fill = ({"n_prompts": args.n_prompts, "deltas_sha256": deltas_sha, "inj_layer": inj_layer}
            if args.trust_legacy_rows else None)
    if rows_path.exists():
        kept, legacy = [], 0
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            n_rows = r.get("n_prompts", (fill or {}).get("n_prompts"))
            if n_rows != args.n_prompts:
                kept.append(line)
                legacy += n_rows is None
                continue
            mine = (r.get("behavior", "sentiment") == behavior.name
                    and r.get("test_arm", "") == arm_name and r["constraint"] == args.constraint
                    and (r["arm"] == CLEAN_ARM
                         or (r["arm"] in arms and r["budget"] == budget
                             and r["objective"] == objective_key(r["arm"], args.objective))))
            if r["layer"] in redo and mine and r.get("stop_on", "miss") != args.stop_on:
                continue
            kept.append(line)
            rk = resume_key(r, fill)
            if rk is None:
                legacy += 1
                continue
            done.add(rk)
            if r["arm"] == CLEAN_ARM:
                clean_steps.setdefault((r.get("behavior", "sentiment"), r.get("test_arm", ""),
                                        r["prompt_index"], r["layer"]),
                                       (r["n_target"], r["steps"]))
        if redo:
            dropped = sum(1 for _ in rows_path.read_text().splitlines()) - len(kept)
            rows_path.write_text("".join(f"{line}\n" for line in kept))
            logger.info("dropped %d rows at layers %s for re-inversion", dropped, sorted(redo))
        if legacy:
            logger.warning("%d rows lack a prompt count or delta hash and were not resumed; "
                           "--trust_legacy_rows treats them as this run's", legacy)
        logger.info("resuming: %d rows already on disk", len(done))

    cells = [(CLEAN_ARM, 0.0, all_layers)] + [(a, budget, steered_layers) for a in arms]
    t_start = time.time()
    with rows_path.open("a") as sink:
        for arm, cell_budget, layers in cells:
            okey = objective_key(arm, args.objective)
            mkey = positions_key(arm, n_positions)
            tail = (inj_layer, deltas_sha) if arm != CLEAN_ARM else (None, None)
            todo = [l for l in layers
                    if any((behavior.name, arm_name, okey, mkey, args.constraint,
                            cell_budget, arm, i, l) + tail not in done
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
            def key(i: int, layer: int) -> tuple:
                return (behavior.name, arm_name, okey, mkey, args.constraint, cell_budget,
                        arm, i, layer) + tail

            def emit(i: int, layer: int, steps: list[dict], known, elapsed: float,
                     scan_layers: int = 1) -> None:
                gold = token_ids[i].tolist()
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
                    "behavior": behavior.name, "test_arm": arm_name, "n_prompts": args.n_prompts,
                    "constraint": args.constraint, "inj_layer": inj_layer,
                    "deltas_sha256": deltas_sha if arm != CLEAN_ARM else None,
                    "item_index": items[i].index,
                    "question_sha256": hashlib.sha256(items[i].question.encode()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest(),
                    "layer": layer, "n_target": len(gold), "n_recovered": len(steps),
                    "exact": [s["token"] for s in steps] == gold[:len(steps)],
                    "inj_rel_residual": (at_inj["residual"] / at_inj["h_norm"]
                                         if at_inj.get("h_norm") else None),
                    # the crude rel_tol oracle's verdict, free, so a trajectory-rule
                    # detection is never confused with a tolerance detection
                    "inj_matched": all(bool(x["matched"]) for x in inj_steps),
                    "n_injected": len(inj_pos),
                    "stop_on": args.stop_on,
                    "rel_tol_by_layer": args.rel_tol_by_layer,
                    "prefix_reused": len(known) if known else 0,
                    # >1: this row's injected position came out of one candidate scan
                    # shared by that many layers (elapsed is the scan time split evenly)
                    "scan_layers": scan_layers,
                    "inj_rel_residual_by_position": [
                        x["residual"] / x["h_norm"] if x.get("h_norm") else None
                        for x in inj_steps],
                    "rel_dev_fp16": rel_dev[i][layer],
                    "rel_dev_fp32": (s1.get("rel_dev_by_layer") or {}).get(str(layer)),
                    "flip_fp16": int(logits[i].argmax() != clean_logits[i].argmax()),
                    "flip_fp32": s1.get("flip"),
                    "elapsed": elapsed,
                    "steps": [{k: s[k] for k in ("token", "residual", "gap", "tol",
                                                 "h_norm", "tried", "matched",
                                                 "gold_token", "correct")
                               if k in s} for s in steps],
                }
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                done.add(key(i, layer))
                if arm == CLEAN_ARM:
                    clean_steps[(behavior.name, arm_name, i, layer)] = (len(gold), row["steps"])

            # The clean arm inverts every position at every layer, and all layers recover
            # the same tokens, so one scan per prompt serves them all (sipit_multi); a
            # layer that recovers a different token is finished alone from its prefix.
            if arm == CLEAN_ARM and not args.no_reuse_clean:
                def vocab_layer_for(l: int) -> Tensor:
                    return sipit.load_vocab_layer(vocab_table, l, layout)
                for i in captured:
                    gold = token_ids[i].tolist()
                    pending = [l for l in todo if key(i, l) not in done]
                    if not pending:
                        continue
                    t0 = time.time()
                    steps_by, diverged = sipit.sipit_multi(
                        {l: captured[i][l] for l in pending}, vocab_layer_for,
                        rel_tol=rel_tol, stop_on_fail=stop_on, gold=gold)
                    for l in sorted(diverged):
                        steps_by[l] = sipit.sipit(captured[i][l], l, vocab_layer_for(l),
                                                  rel_tol=rel_tol, stop_on_fail=stop_on, gold=gold,
                                                  known_steps=steps_by[l])
                    per = (time.time() - t0) / len(pending)
                    for l in pending:
                        emit(i, l, steps_by[l], None, per, len(pending))
            # A single last-position injection whose clean prefix inverted exactly at
            # every pending layer: the prefix cache and candidate order are shared, so
            # all those layers are solved from one scan (one forward per batch instead
            # of one per layer). Anything that does not qualify falls through below.
            if arm != CLEAN_ARM and not args.no_reuse_clean and n_positions == 1:
                for i in captured:
                    gold = token_ids[i].tolist()
                    p0 = len(gold) - 1
                    pending = [l for l in todo if key(i, l) not in done]
                    known = {l: clean_prefix(clean_steps, behavior.name, arm_name, i, l,
                                             gold, p0) for l in pending}
                    ok = [l for l in pending if known[l] is not None and len(known[l]) == p0
                          and all(s["matched"] for s in known[l])]
                    if not ok:
                        continue
                    t0 = time.time()
                    cache, lg = sipit.prefix_cache(gold[:p0])
                    last = sipit.solve_position_multi(
                        cache, p0, lg, {l: captured[i][l][p0] for l in ok}, rel_tol=rel_tol,
                        abs_tol=0.0, schedule=sipit.DEFAULT_SCHEDULE, exhaustive=False,
                        gold=gold[p0])
                    per = (time.time() - t0) / len(ok)
                    for l in ok:
                        emit(i, l, known[l] + [last[l]], known[l], per, len(ok))
            for layer in todo:
                pending = [i for i in captured if key(i, layer) not in done]
                if not pending:
                    continue
                vocab_layer = sipit.load_vocab_layer(vocab_table, layer, layout)
                for i in pending:
                    gold = token_ids[i].tolist()
                    inj_pos = injected_positions(n_positions, len(gold))
                    known = None
                    if arm != CLEAN_ARM and not args.no_reuse_clean:
                        known = clean_prefix(clean_steps, behavior.name, arm_name, i, layer,
                                             gold, len(gold) - len(inj_pos))
                    t0 = time.time()
                    steps = sipit.sipit(captured[i][layer], layer, vocab_layer,
                                        rel_tol=rel_tol, stop_on_fail=stop_on, gold=gold,
                                        known_steps=known)
                    emit(i, layer, steps, known, time.time() - t0)
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
        attack = r["arm"] != CLEAN_ARM
        cells.setdefault((r.get("behavior", "sentiment"), r.get("test_arm", ""),
                          r["objective"],
                          positions_key(r["arm"], r.get("n_positions", 1)),
                          r["constraint"], r["budget"], r["arm"],
                          r.get("inj_layer") if attack else None,
                          r.get("deltas_sha256") if attack else None,
                          r.get("n_prompts")), []).append(r)
    try:
        cells, notes = detect.merge_legacy_cells(
            cells, family=lambda key: key[:8], legacy=lambda key: key[9] is None)
    except ValueError as e:
        raise SystemExit(f"{rows_path}: {e}") from e
    for note in notes:
        logger.warning("provenance: %s", note)

    def profiles_of(key, cell_rows):
        try:
            return detect.profiles_from_rows(cell_rows, k, cal, id_key="prompt_index")
        except ValueError as e:
            raise SystemExit(f"cell {key}: {e}") from e

    # the clean control must come from the SAME prompt set: a control fitted on 7-token
    # sentiment fragments is not the control for 40-token chat-templated jailbreaks
    controls = {(ck[0], ck[1], ck[-1]): (*profiles_of(ck, v), v)
                for ck, v in cells.items() if ck[6] == CLEAN_ARM}

    def control_for(key, cell_rows, want):
        beh, beh_arm, n_run = key[0], key[1], key[-1]
        if (beh, beh_arm, n_run) in controls:
            ck, trip = (beh, beh_arm, n_run), controls[(beh, beh_arm, n_run)]
        else:
            loose = [(ck, trip) for ck, trip in controls.items()
                     if ck[:2] == (beh, beh_arm) and want <= set(trip[0])]
            if not loose:
                return {}, {}, set()
            ck, trip = loose[0]
            logger.warning("%s/%s at n_prompts=%s: no clean control recorded the same "
                           "prompt count; the one covering its prompt indices "
                           "(n_prompts=%s) is used only where the question or token "
                           "sequence matches", beh, beh_arm, n_run, ck[2])
        mismatched, unknown = detect.verify_control(cell_rows, trip[2])
        if mismatched:
            raise SystemExit(f"cell {key}: clean control {ck} holds a different question "
                             f"under prompt index {mismatched[:5]}; refusing to borrow "
                             "its layers")
        if unknown:
            logger.warning("cell %s: %d prompt(s) cannot be matched to the clean control; "
                           "their layers below the injection stay absent", key, len(unknown))
        return trip[0], trip[1], set(unknown)

    summary = []
    for key, cell_rows in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        beh, beh_arm, obj, m, scope, budget, arm, _, delta_sha, n_run = key
        L = cell_rows[0]["inj_layer"]
        prof, cov = profiles_of(key, cell_rows)
        control, control_cov, unmatched = ({}, {}, set()) if arm == CLEAN_ARM else control_for(
            key, cell_rows, set(prof))
        if arm != CLEAN_ARM:
            for pid in prof:
                if pid in unmatched:
                    continue
                for layer in layers:
                    if layer < L and layer in control.get(pid, {}):
                        prof[pid].setdefault(layer, control[pid][layer])
                        cov[pid].setdefault(layer, control_cov[pid][layer])
        entry = {"behavior": beh, "test_arm": beh_arm, "objective": obj,
                 "budget": budget, "arm": arm, "inj_layer": L,
                 "constraint": scope, "n_positions": m, "n_prompts": n_run,
                 "deltas_sha256": delta_sha,
                 "n_injected_mean": (sum(r.get("n_injected", 1) for r in cell_rows)
                                     / max(1, len(cell_rows)))}
        entry.update(steered_sipit.summarize_cell(prof, L, stats, layers, cal,
                                                  operating_points, cov))
        at_inj = [r for r in cell_rows if r["layer"] == L]
        realized = [r["inj_rel_residual"] for r in at_inj
                    if r["inj_rel_residual"] is not None]
        entry["realized_rel_residual_median"] = (
            statistics.median(realized) if realized else float("nan"))
        entry["inj_matched_frac"] = (sum(r["inj_matched"] for r in at_inj)
                                     / max(1, len(at_inj)))
        entry["exact_frac"] = sum(r["exact"] for r in cell_rows) / max(1, len(cell_rows))
        # next-token argmax changes are kept as a diagnostic of the perturbation, not
        # reported as behaviour (retired 7 Sep 2026: a function-word swap on a
        # near-tied prompt is not an effect)
        entry["argmax_changed_fp16"] = sum(r["flip_fp16"] for r in at_inj) / max(1, len(at_inj))
        f32 = [r["flip_fp32"] for r in at_inj if r["flip_fp32"] is not None]
        entry["argmax_changed_fp32"] = sum(bool(v) for v in f32) / len(f32) if f32 else None
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
    logger.info("%-13s %-9s %-9s %7s %-7s %3s %3s | %7s %6s %10s %10s | %8s %7s | "
                "%9s %8s",
                "behavior", "objective", "scope", "budget", "arm", "m", "n", "TPR",
                "unscor", "sigma5run3", "sigma4run6", "inj hit", "med s/t", "realized",
                "shift")
    for e in summary:
        tag = "   <- FPR, this distribution" if e["arm"] == CLEAN_ARM else ""
        logger.info("%-13s %-9s %-9s %7g %-7s %3d %3d | %6.0f%% %6d %9.0f%% %9.0f%% | "
                    "%7.0f%% %7.2f | %9.2e %8.1e%s",
                    e["behavior"], e["objective"], e["constraint"], e["budget"],
                    e["arm"], e["n_positions"], e["n"],
                    100 * e["tpr_shipped"], e["unscorable"], 100 * e["tpr_sigma5_run3"],
                    100 * e["tpr_sigma4_run6"], 100 * e["inj_layer_hit_rate"],
                    e["inj_score_over_threshold_median"],
                    e["realized_rel_residual_median"], e["score_shift_at_inj_median"], tag)

    out = {
        "model": args.model_name, "dtype": args.dtype,
        "calibration_file": calibration,
        "calibration": {kk: cal[kk] for kk in ("k", "sigma", "sigma_any", "min_run",
                                               "stride", "dense_tail", "fpr_prompt",
                                               "n_prompts")},
        "operating_points": operating_points,
        "verdicts": "flagged / clear / unscorable per prompt; 'unscorable' counts prompts "
                    "the calibration could not fully cover and are never read as clean",
        "provenance": provenance.stamp(pgd_rows=rows_path,
                                       calibration=sipit_dir / calibration),
        "provenance_notes": notes,
        "cells": summary,
    }
    path = out_dir / f"tpr_summary{steered_sipit.summary_suffix(calibration)}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
