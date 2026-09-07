import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import detect
from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)

SCHEMES = ("3way", "2way")


def resolve_layers_dir(cal_path: Path, cal: dict, override: Path | None) -> Path:
    if override:
        return override
    fit = Path(cal.get("fit_layers_dir", ""))
    if fit.is_dir():
        return fit
    parts = fit.parts
    if "sipit" in parts:
        local = cal_path.parent / Path(*parts[parts.index("sipit") + 1:])
        if local.is_dir():
            return local
    raise SystemExit(f"fit set {fit} not found; pass --layers_dir")


def load_rows(args, cal_path: Path, cal: dict) -> tuple[list[dict], dict, str]:
    if args.rows:
        rows = [json.loads(l) for l in args.rows.read_text().splitlines() if l.strip()]
        if args.behavior:
            rows = [r for r in rows if r.get("behavior") == args.behavior]
        clean = [r for r in rows if r["arm"] == "clean"]
        attacks: dict[tuple, list[dict]] = {}
        for r in rows:
            if r["arm"] != "clean":
                attacks.setdefault((r["arm"], r["budget"]), []).append(r)
        return clean, attacks, f"{args.rows}#{args.behavior or 'all'}"
    layers_dir = resolve_layers_dir(cal_path, cal, args.layers_dir)
    return detect.load_trajectories(layers_dir), {}, str(layers_dir)


def fold_calibration(cal: dict, rows: list[dict]) -> dict:
    chat_ids = tuple(cal["chat_ids"]) if cal.get("chat_ids") else None
    transform = cal.get("transform", "none")
    scheme, end_cap = cal.get("content_scheme", "pooled"), cal.get("end_cap", 6)
    header_len = cal.get("chat_header_len", 3)
    sd_floor = cal.get("sd_floor") or (detect.SD_FLOOR_LOG if transform == "log"
                                       else detect.SD_FLOOR_RAW)
    return {"statistic": "role_z", "chat_ids": list(chat_ids) if chat_ids else None,
            "transform": transform, "content_scheme": scheme, "end_cap": end_cap,
            "chat_header_len": header_len,
            "role_stats": detect.fit_role_stats(rows, chat_ids, sd_floor, transform,
                                                scheme, end_cap, header_len)}


def profiles_of(rows: list[dict], k: int, fcal: dict) -> dict:
    prof: dict[str, dict[int, float]] = {}
    for s in detect.score_rows(rows, k, fcal):
        if not math.isnan(s["score"]):
            prof.setdefault(s["id"], {})[s["layer"]] = s["score"]
    return prof


def one_repeat(clean_by_id: dict, attacks: dict, cal: dict, layers: list[int],
               stats: dict, args, seed: int) -> dict:
    ids = sorted(clean_by_id)
    random.Random(seed).shuffle(ids)
    folds = [ids[i::args.folds] for i in range(args.folds)]
    k, ratio = cal["k"], cal["sigma_any"] / cal["sigma"]
    rule = dict(min_run=cal["min_run"], stride=cal["stride"], dense_tail=cal["dense_tail"])
    flagged = n = 0
    sigmas: list[float] = []
    hits: dict[tuple, list[int]] = {}
    for t in range(args.folds):
        s = (t + 1) % args.folds if args.scheme == "3way" else None
        fit_ids = [i for f, fold in enumerate(folds) if f not in (t, s) for i in fold]
        sel_ids = folds[s] if s is not None else fit_ids
        fcal = fold_calibration(cal, [r for i in fit_ids for r in clean_by_id[i]])
        sel = profiles_of([r for i in sel_ids for r in clean_by_id[i]], k, fcal)
        sigma = detect.tune_sigma(sel, stats, layers, target=args.target,
                                  sigma_any=cal["sigma_any"], ratio=ratio, hi=30.0, **rule)
        test = profiles_of([r for i in folds[t] for r in clean_by_id[i]], k, fcal)
        rep = detect.prompt_fpr(test, stats, layers, sigma=sigma, sigma_any=sigma * ratio,
                                **rule)
        flagged += sum(v["flagged"] for v in rep["per_id"].values())
        n += rep["n_prompts"]
        sigmas.append(sigma)
        held = set(folds[t])
        for cell, arows in attacks.items():
            aprof = profiles_of([r for r in arows if r["id"] in held], k, fcal)
            inj = arows[0]["inj_layer"]
            for pid, prof in aprof.items():
                for layer in layers:
                    if layer < inj and layer in test.get(pid, {}):
                        prof.setdefault(layer, test[pid][layer])
            arep = detect.prompt_fpr(aprof, stats, layers, sigma=sigma,
                                     sigma_any=sigma * ratio, **rule)
            h = hits.setdefault(cell, [0, 0])
            h[0] += sum(v["flagged"] for v in arep["per_id"].values())
            h[1] += arep["n_prompts"]
    return {"fpr": flagged / max(1, n), "flagged": flagged, "n": n,
            "sigma_mean": statistics.fmean(sigmas),
            "sigma_sd": statistics.pstdev(sigmas),
            "tpr": {f"{arm}@{budget:g}": h[0] / max(1, h[1]) for (arm, budget), h in hits.items()},
            "tpr_n": {f"{arm}@{budget:g}": h[1] for (arm, budget), h in hits.items()}}


def main():
    parser = argparse.ArgumentParser(
        description="Held-out FPR for a role_z calibration: role stats fitted on one split, "
                    "sigma chosen on a second, the verdict read on a third.")
    parser.add_argument("--calibration", type=Path, required=True, nargs="+")
    parser.add_argument("--layers_dir", type=Path, default=None,
                        help="clean rows to split; default: the calibration's own fit set")
    parser.add_argument("--rows", type=Path, default=None,
                        help="pgd_rows.jsonl: its clean arm is the set to split, its other "
                             "arms are scored under each fold's calibration for TPR")
    parser.add_argument("--behavior", type=str, default=None)
    parser.add_argument("--scheme", type=str, default="3way", choices=SCHEMES,
                        help="3way: fit / select / test disjoint. 2way: sigma tuned on the "
                             "fit rows themselves, as detect.py does, test held out")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=None,
                        help="per-prompt FPR sigma is tuned to on the select split; "
                             "default: the calibration's own target. 0 puts sigma just "
                             "above the select split's clean maximum")
    parser.add_argument("--out_suffix", type=str, default="")
    parser.add_argument("--no_write", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=Path("results/all/logs/detect_holdout.log"))

    logger.info("%-46s %5s %-4s | %7s %7s | %13s %13s  %s", "calibration", "n", "sch",
                "ship sg", "ship FPR", "held sigma", "held FPR", "TPR under the held-out sigma")
    for cal_path in args.calibration:
        cal = json.loads(cal_path.read_text())
        if cal.get("statistic") != "role_z":
            logger.warning("%s is not a role_z calibration, skipped", cal_path)
            continue
        target = cal.get("target_fpr") if args.target is None else args.target
        if target is None:
            raise SystemExit(f"{cal_path} has no target_fpr; pass --target")
        args.target = target
        clean, attacks, source = load_rows(args, cal_path, cal)
        clean_by_id: dict[str, list[dict]] = {}
        for r in clean:
            clean_by_id.setdefault(r["id"], []).append(r)
        layers = sorted({r["layer"] for r in clean})
        stats = {l: detect.LayerStat(0, 0.0, 1.0) for l in layers}
        if len(clean_by_id) < args.folds * 2:
            logger.warning("%s: only %d clean prompts for %d folds", cal_path,
                           len(clean_by_id), args.folds)
        reps = [one_repeat(clean_by_id, attacks, cal, layers, stats, args, args.seed + i)
                for i in range(args.repeats)]
        fprs = [r["fpr"] for r in reps]
        sig = [r["sigma_mean"] for r in reps]
        tpr = {c: statistics.fmean(r["tpr"][c] for r in reps) for c in reps[0]["tpr"]}
        tag = f"{cal_path.parts[-3]}/{cal_path.stem.replace('detector_calibration_', '')}"
        logger.info("%-46s %5d %-4s | %7.2f %6.0f%% | %5.2f +- %4.2f %5.1f%% +- %4.1f%%  %s",
                    tag, len(clean_by_id), args.scheme, cal["sigma"],
                    100 * cal["fpr_prompt"], statistics.fmean(sig), statistics.pstdev(sig),
                    100 * statistics.fmean(fprs), 100 * statistics.pstdev(fprs),
                    "  ".join(f"{c} {100 * v:.0f}%" for c, v in sorted(tpr.items())))
        if args.no_write:
            continue
        payload = {
            "calibration": str(cal_path),
            "calibration_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
            "clean_source": source, "n_clean": len(clean_by_id), "layers": layers,
            "scheme": args.scheme, "folds": args.folds, "repeats": args.repeats,
            "seed": args.seed, "target": args.target,
            "shipped": {"sigma": cal["sigma"], "fpr_in_sample": cal["fpr_prompt"]},
            "held_out": {"fpr_mean": statistics.fmean(fprs), "fpr_sd": statistics.pstdev(fprs),
                         "fpr_min": min(fprs), "fpr_max": max(fprs),
                         "sigma_mean": statistics.fmean(sig), "sigma_sd": statistics.pstdev(sig),
                         "tpr_mean": tpr, "tpr_n": reps[0]["tpr_n"]},
            "repeats_detail": reps,
        }
        name = f"detector_holdout_{cal_path.stem.replace('detector_calibration_', '')}"
        name += f"_{args.scheme}{args.out_suffix}.json"
        (cal_path.parent / name).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
