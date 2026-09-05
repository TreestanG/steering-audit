"""The detection-vs-efficacy 2x2 for PGD attacks.

`join_detection.py` answers this for the fraction sweep. It cannot read PGD: it keys on
(prompt, layer, fraction) and derives detection by thresholding one vocab-scan residual
at rel_tol, while PGD's axis is BUDGET and its detection is the calibrated trajectory
rule -- a scan across layers with a run rule and a catch-all, giving one verdict per
prompt rather than one per layer. So this file supplies the loaders and reuses that
module's `contingency`, `summarize` and `log_table`, which are about the 2x2 rather than
about either pipeline.

    detection  results/<slug>/pgd_sipit/pgd_rows.jsonl, scored with a calibration
    efficacy   the <stem>_gen.json files pgd_generate wrote (judge columns from judge_gen)

THE JOIN KEY IS THE QUESTION TEXT, and it has to be. `behaviors.items()` STRATIFIES, so
at n=15 enumeration position 1 is item.index 10 -- and `pgd_sipit` records the
enumeration position while `pgd_generate` records item.index. Joining on the integer
would pair row 1's detection with row 10's completion and silently mispair everything
but the first. Both sides are mapped back through the same item list instead, and a
question present on one side only is dropped and counted.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _set_run_tag():
    for i, a in enumerate(sys.argv):
        if a == "--dtype" and i + 1 < len(sys.argv):
            os.environ.setdefault("AAT_DTYPE", sys.argv[i + 1])
    os.environ.setdefault("AAT_DTYPE", "float16")


_set_run_tag()

import behaviors
import detect
import join_detection
import steered_sipit
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir

logger = get_logger(__name__)

CLEAN_ARMS = {"clean", "none"}


def cell_key(row: dict) -> tuple:
    return (row.get("behavior", "sentiment"), row.get("test_arm"), row["budget"],
            row["arm"], row.get("n_positions", 1), row.get("constraint", "all"))


def detection_verdicts(rows: list[dict], cal: dict, stats, layers) -> dict:
    """{cell -> {prompt_index: flagged}} under the shipped rule, sub-injection layers
    taken from the clean cell with the same prompt set (they are bit-identical)."""
    cells = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)
    k = cal["k"]

    def profiles(cell_rows):
        prof = defaultdict(dict)
        for r in cell_rows:
            score = detect.row_score(r, k, cal)
            if score == score:
                prof[r["prompt_index"]][r["layer"]] = score
        return prof

    cleans = {key: profiles(v) for key, v in cells.items() if key[3] in CLEAN_ARMS}
    out = {}
    for key, cell_rows in cells.items():
        if key[3] in CLEAN_ARMS:
            continue
        prof = profiles(cell_rows)
        want = set(prof)
        control = next((c for ck, c in cleans.items()
                        if ck[0] == key[0] and ck[1] == key[1] and want <= set(c)), None)
        L = cell_rows[0]["inj_layer"]
        if control:
            for pid in prof:
                for layer in layers:
                    if layer < L and layer in control[pid]:
                        prof[pid].setdefault(layer, control[pid][layer])
        else:
            logger.warning("cell %s has no clean control covering its prompts; layers "
                           "below %d are absent and can only lower detection", key, L)
        rep = detect.prompt_fpr(prof, stats, layers, sigma=cal["sigma"],
                                min_run=cal["min_run"], stride=cal["stride"],
                                sigma_any=cal["sigma_any"], dense_tail=cal["dense_tail"])
        out[key] = ({pid: int(v["flagged"]) for pid, v in rep["per_id"].items()},
                    {pid: max(p.values()) for pid, p in prof.items()}, L)
    return out


def question_maps(behavior, arm: str, n: int) -> tuple[dict, dict]:
    """(enumeration position -> question, item.index -> question) for the SAME item list
    both stages used. Mismatched n selects different prompts -- see the module docstring."""
    items = behavior.items(arm, n)
    return ({i: it.question for i, it in enumerate(items)},
            {it.index: it.question for it in items})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gen", type=Path, nargs="+", required=True,
                   help="pgd_generate <stem>_gen.json files (judge columns from judge_gen)")
    p.add_argument("--pgd_rows", type=Path, default=None,
                   help="default: results/<slug>/pgd_sipit/pgd_rows.jsonl")
    p.add_argument("--calibration", default="detector_calibration_rolezlog_k1_fpr5_n100.json")
    p.add_argument("--jailbroken_field", default="judge_harmbench",
                   help="judge_harmbench is the quotable one; 'refused' inverts the "
                        "substring scorer, an UPPER BOUND (69.5%% FPR)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--rows_out", type=Path, default=None)
    add_logging_args(p)
    a = p.parse_args()
    log_setup(a, default_log=logs_dir(a.model_name) / "join_pgd.log")

    sipit_dir = experiment_dir(a.model_name, "sipit")
    pgd_dir = experiment_dir(a.model_name, "pgd_sipit")
    rows_path = a.pgd_rows or pgd_dir / "pgd_rows.jsonl"
    cal, stats, layers = steered_sipit.load_calibration(sipit_dir, a.calibration)
    logger.info("detection: %s (k=%d, %s, sigma %.2f)", a.calibration, cal["k"],
                cal.get("statistic", "topk"), cal["sigma"])

    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    verdicts = detection_verdicts(rows, cal, stats, layers)

    joined, dropped = [], defaultdict(int)
    for path in a.gen:
        blob = json.loads(path.read_text())
        beh = behaviors.load_behavior(blob["behavior"])
        by_pos, by_index = question_maps(beh, blob["arm"], blob["n_prompts"])
        for grow in blob["rows"]:
            arm = grow["arm"]
            if arm in CLEAN_ARMS:
                continue
            key = (blob["behavior"], blob["arm"], blob["budget"], arm, 1, "all")
            got = verdicts.get(key)
            if got is None:
                dropped["no detection for this (budget, arm)"] += 1
                continue
            flagged, peak, L = got
            question = by_index.get(grow["index"], grow.get("question"))
            pid = next((i for i, q in by_pos.items() if q == question), None)
            if pid is None or pid not in flagged:
                dropped["question not in the detection cell"] += 1
                continue
            verdict = grow.get(a.jailbroken_field)
            if verdict is None:
                dropped[f"no {a.jailbroken_field} verdict (ungraded)"] += 1
                continue
            joined.append({
                "prompt": question, "question": question, "category": None,
                "arm": arm, "replicate": 0, "layer": L, "fraction": blob["budget"],
                "jailbroken": int(verdict),
                "jailbroken_substring": int(not grow["refused"]),
                "behavior_hit": None,
                **{k: grow[k] for k in grow if k.startswith("judge_") and k.endswith("_score")},
                "detected": flagged[pid], "rel_residual": peak[pid],
                "margin_spent": float("nan"), "h_norm": float("nan"),
                "response": grow.get("completion"),
            })
    for reason, n in dropped.items():
        logger.warning("dropped %d generated row(s): %s", n, reason)
    if not joined:
        raise SystemExit(
            "nothing joined. detection cells: "
            f"{sorted({(k[0], k[2], k[3]) for k in verdicts})}\n  generation cells: "
            f"{sorted({(json.loads(p.read_text())['behavior'], json.loads(p.read_text())['budget']) for p in a.gen})}\n"
            "Decode at a budget the detection covers (pgd_generate), or score one it has.")

    summary = join_detection.summarize(joined)
    logger.info("rel_residual below is the PEAK trajectory score over layers, not a "
                "residual; detection is the calibrated rule's per-prompt verdict")
    join_detection.log_table(summary, a.jailbroken_field.replace("judge_", "").upper())
    pooled = join_detection.contingency([r for r in joined if r["arm"] == "pgd"])
    logger.info("")
    logger.info("pgd arm pooled: %d/%d jailbroken, %d of those UNDETECTED (b)",
                pooled["jailbroken_detected"] + pooled["jailbroken_undetected"],
                pooled["n"], pooled["jailbroken_undetected"])
    if pooled["jailbroken_undetected"]:
        logger.warning("%d undetected jailbreak(s) -- the evasion window. Check them by "
                       "hand before believing it.", pooled["jailbroken_undetected"])

    out = a.out or pgd_dir / "detection_vs_efficacy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_name": a.model_name, "calibration": a.calibration,
        "jailbroken_field": a.jailbroken_field, "detector": {
            k: cal[k] for k in ("k", "sigma", "sigma_any", "min_run", "stride")},
        "pooled_pgd": pooled, "cells": summary}, indent=2) + "\n")
    logger.info("wrote %s", out)
    if a.rows_out:
        a.rows_out.write_text("\n".join(json.dumps(r) for r in joined) + "\n")


if __name__ == "__main__":
    main()
