"""The detector-aware attacker's saved rows read one layer at a time.

The attacker held every layer from the injection onward under a z ceiling; this reads
its rows at the last layer alone and at the injection layer alone against the bank's
single-layer thresholds, and under an uncorrected any-layer rule, so the aware figure can
carry the single-layer read. A re-read of saved rows; no inference.
Driver: scripts/aware_single_layer.py.
"""

import json
import math
from pathlib import Path

import numpy as np

import detect
import provenance
from ladders import resolve

CEILINGS = ("4.28", "6", "8", "12", "16", "20")
VARIANTS = {"per_layer": "results/{slug}_aware_z{z}",
            "run_rule": "results/{slug}_aware_run_z{z}"}


def run(calibration: Path, bank_layers: Path, slug: str, ceilings=CEILINGS,
        variants: dict[str, str] = VARIANTS) -> dict:
    cal = json.loads(resolve(calibration).read_text())
    bank: dict[int, list[float]] = {}
    for r in detect.load_trajectories(resolve(bank_layers)):
        s = detect.row_score(r, cal["k"], cal)
        if math.isfinite(s):
            bank.setdefault(r["layer"], []).append(s)
    thr = {L: float(np.quantile(v, 0.95, method="higher")) for L, v in bank.items()}
    out = {"thresholds": thr, "cells": [], "provenance": {}}
    for variant, pattern in variants.items():
        for z in ceilings:
            f = Path(pattern.format(slug=slug, z=z)) / "pgd_sipit/pgd_rows.jsonl"
            if not resolve(f).exists():
                continue
            rows = [json.loads(l) for l in resolve(f).read_text().splitlines() if l.strip()]
            prof: dict = {}
            inj = None
            for r in rows:
                if r["arm"] != "pgd":
                    continue
                inj = r["inj_layer"]
                prof.setdefault(r["prompt_index"], {})[r["layer"]] = detect.row_score(r, cal["k"], cal)
            layers = sorted({L for pr in prof.values() for L in pr})
            last = layers[-1]

            def flagged(L):
                return sum(1 for pr in prof.values() if math.isfinite(pr.get(L, float("nan"))) and pr[L] > thr[L])

            any_layer = sum(1 for pr in prof.values()
                            if any(math.isfinite(pr.get(L, float("nan"))) and pr[L] > thr[L] for L in layers))
            out["cells"].append({"variant": variant, "z_ceiling": float(z), "n": len(prof), "inj_layer": inj,
                                 "last_layer": last, "flagged_last_layer": flagged(last),
                                 "flagged_injection_layer": flagged(inj), "flagged_any_layer_uncorrected": any_layer,
                                 "median_z_by_layer": {L: float(np.median([pr[L] for pr in prof.values() if L in pr]))
                                                       for L in layers}})
            out["provenance"][f"{variant}_z{z}"] = provenance.stamp(rows=f)["inputs"]
    return out


def print_cells(out: dict) -> None:
    for c in out["cells"]:
        print("%-9s z<=%-5g n=%d | last layer L%d alone %2d/%d | injection L%d alone %2d/%d | any layer, uncorrected %2d/%d" % (
            c["variant"], c["z_ceiling"], c["n"], c["last_layer"], c["flagged_last_layer"], c["n"],
            c["inj_layer"], c["flagged_injection_layer"], c["n"], c["flagged_any_layer_uncorrected"], c["n"]))
