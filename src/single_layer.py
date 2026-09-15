"""One logged layer: the detector's per-layer score read on its own, at the injection
layer and every layer after it, beside what recomputation can show from the same log.
A re-read of saved rows; no inference.

Per ladder: the bank rows scored per layer under the shipped calibration give a per-layer
threshold, both the shipped sigma and a single-layer sigma at the bank's 95th percentile
(in-sample, the detector's own rule). The ladder's clean arm, held out from the bank,
gives the held-out false-positive rate at each layer. The pgd and random arms give the
flagged count at each layer >= injection. Layers before the injection carry the clean
state by causality and are not tabulated. Recomputation is read from the saved baselines:
the largest residual over every pair that does not straddle the injection, against the
same floor, which is all a log whose earliest layer is at or after the injection can
show it. `deploy` cuts every row at its first tolerance miss first.
Driver: scripts/single_layer_reread.py.
"""

import collections
import json
import math
from pathlib import Path

import numpy as np

import detect
import provenance
from deploy_stop import deploy_truncate
from ladders import Ladder, resolve


def quantile_threshold(scores: list[float]) -> float:
    """NaN when no bank row at this layer has a finite score: under the deployment cut a
    template position at the fp16 floor ends every bank row at position 0, and a layer
    with no calibrated bank is unscorable for every prompt."""
    if not scores:
        return float("nan")
    return float(np.quantile(np.array(scores), 0.95, method="higher"))


def count(scores: list[float], thr: float) -> dict:
    fin = [s for s in scores if math.isfinite(s)] if math.isfinite(thr) else []
    return {"flagged": int(sum(s > thr for s in fin)), "n": len(scores),
            "unscorable": len(scores) - len(fin)}


def recompute_off_injection(baselines: Path) -> dict:
    """{(arm, budget) -> counts of prompts whose largest residual over every pair NOT
    straddling the injection clears the floor}, under both measured floors."""
    d = json.loads(resolve(baselines).read_text())
    bank = d["recompute"]["bank"]
    out = {}
    for c in d["recompute"]["cells"]:
        off_s, off_b = c["off_injection_max_single"], c["off_injection_max_batch"]
        out[(c["arm"], float(c["budget"]))] = {
            "same_kernel": int(sum(s > bank["threshold_single"] for s in off_s)),
            "serving_batch": int(sum(s > bank["threshold_batch"] for s in off_b)),
            "n": len(off_s)}
    return out


def analyse(lad: Ladder, deploy: bool = False) -> dict:
    cal = json.loads(resolve(lad.calibration_path).read_text())
    k = cal["k"]
    fix = deploy_truncate if deploy else (lambda r: r)

    bank_rows = [fix(r) for r in detect.load_trajectories(resolve(lad.bank_layers))]
    bank_scores: dict[int, list[float]] = collections.defaultdict(list)
    bank_profiles: dict = {}
    for r in bank_rows:
        s = detect.row_score(r, k, cal)
        bank_scores[r["layer"]].append(s)
        bank_profiles.setdefault(r["id"], {})[r["layer"]] = s
    layers = sorted(bank_scores)
    stats = {L: detect.LayerStat(v["n"], v["mean"], v["sd"]) for L, v in
             ((int(a), b) for a, b in cal["per_layer"].items())}
    shipped = detect.prompt_fpr(bank_profiles, stats, sorted(stats), sigma=cal["sigma"],
                                min_run=cal["min_run"], stride=cal["stride"],
                                sigma_any=cal["sigma_any"], dense_tail=cal["dense_tail"])
    shipped = {k: v for k, v in shipped.items() if k != "per_id"}

    rows = [json.loads(l) for l in resolve(lad.rows).read_text().splitlines() if l.strip()]
    rows = [fix(r) for r in rows if r.get("behavior") == lad.behavior
            and r.get("test_arm", "harmful") == "harmful"]
    cells: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    inj = None
    for r in rows:
        if r["arm"] != "clean":
            inj = r["inj_layer"] if inj is None else inj
            assert r["inj_layer"] == inj, (lad.label, r["inj_layer"], inj)
        cells[(r["arm"], float(r["budget"]))][r["prompt_index"]][r["layer"]] = \
            detect.row_score(r, k, cal)
    assert inj is not None
    probe_layers = [L for L in layers if L >= inj]
    thr = {L: {"shipped": cal["sigma"],
               "single": quantile_threshold([s for s in bank_scores[L] if math.isfinite(s)])}
           for L in probe_layers}
    recomp = recompute_off_injection(lad.baselines)

    out_cells = []
    for (arm, b), prof in sorted(cells.items(), key=str):
        per_layer = {}
        for L in probe_layers:
            scores = [prof[p].get(L, float("nan")) for p in prof]
            per_layer[L] = {rule: count(scores, thr[L][rule]) for rule in ("shipped", "single")}
            per_layer[L]["median_score"] = float(np.nanmedian(scores)) if any(
                math.isfinite(s) for s in scores) else None
        out_cells.append({"arm": arm, "budget": b, "n": len(prof), "per_layer": per_layer,
                          "recompute_off_injection": recomp.get((arm, b))})
    bank_fpr = {L: {rule: count(bank_scores[L], thr[L][rule]) for rule in ("shipped", "single")}
                for L in probe_layers}
    return {"model": lad.label, "inj_layer": inj, "layers": probe_layers, "n_blocks": max(layers),
            "rule": "deployment (cut at first miss)" if deploy else "evaluator (rows as saved)",
            "sigma_shipped": cal["sigma"], "thresholds": thr,
            "bank": {"n": len(bank_profiles), "shipped_rule_prompt_fpr": shipped,
                     "per_layer_in_sample": bank_fpr},
            "cells": out_cells,
            "provenance": provenance.stamp(
                rows=lad.rows, calibration=lad.calibration_path,
                bank_injection_layer=lad.bank_layers / f"sipit_layer_{inj:02d}.jsonl",
                baselines=lad.baselines)}


def print_table(e: dict) -> None:
    Ls = e["layers"]
    inj = e["inj_layer"]
    print(f"\n== {e['model']}  injection layer {inj} of {e['n_blocks']}  [{e['rule']}]  "
          f"bank n={e['bank']['n']}, shipped rule in-sample prompt FPR "
          f"{e['bank']['shipped_rule_prompt_fpr'].get('fpr_prompt', float('nan')):.3f}")
    hdr = "offset from injection:" + "".join(f"{L - inj:>10d}" for L in Ls) + "  | recompute (any non-straddling pair)"
    print(hdr)
    print("single-layer sigma_L:  " + "".join(f"{e['thresholds'][L]['single']:>10.2f}" for L in Ls)
          + f"   (shipped sigma {e['sigma_shipped']:.2f})")
    print("bank FPR in-sample:    " + "".join(
        f"{e['bank']['per_layer_in_sample'][L]['single']['flagged']}/{e['bank']['n']}".rjust(10)
        for L in Ls))
    for c in sorted(e["cells"], key=lambda c: (c["arm"] != "clean", c["arm"], c["budget"])):
        label = f"{c['arm']:<6s} b={c['budget']:<7g}"
        cells = ""
        for L in Ls:
            x = c["per_layer"][L]["single"]
            u = f"+{x['unscorable']}u" if x["unscorable"] else ""
            cells += f"{x['flagged']}/{x['n']}{u}".rjust(10)
        rc = c["recompute_off_injection"]
        rs = (f"same-kernel {rc['same_kernel']}/{rc['n']}, serving-batch {rc['serving_batch']}/{rc['n']}"
              if rc else "-")
        print(f"{label:<22s}" + cells + "  | " + rs)
