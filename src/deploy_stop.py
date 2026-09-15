"""Rescore the PGD ladders under the deployment stopping rule.

The evaluator inverts with `--stop_on wrong`: a scan continues past a position over
tolerance when the recovered token is known to be right, which needs the original
prompt. A deployment does not have it and stops at the first tolerance miss. This
truncates every saved row at (and including) its first unmatched step, exactly where the
`miss` rule would have halted, and rescores clean, pgd and random arms with the same
calibration and coverage checks, so the two rules can be compared cell by cell. The
calibration itself is reused as saved; this is not a fresh deployment benchmark.
Driver: scripts/deployment_stop_rescore.py.
"""

import collections
import copy
import json
from pathlib import Path

import detect
import provenance
from ladders import Ladder, resolve


def deploy_truncate(row: dict) -> dict:
    """The row as the `miss` rule would have written it: cut at the first step over
    tolerance, which is recorded and then halts the scan."""
    steps = row["steps"]
    cut = next((i for i, s in enumerate(steps) if not s.get("matched", True)), None)
    if cut is None:
        return row
    r = copy.copy(row)
    r["steps"] = steps[:cut + 1]
    r["n_recovered"] = len(r["steps"])
    r["stop_on"] = "miss"
    return r


def verdicts(rows: list[dict], cal: dict, stats: dict, layers: list[int]) -> dict:
    """{(arm, budget) -> {'flagged', 'clear', 'unscorable', 'n'}} with the clean arm as
    control for the layers ahead of the injection, identity-checked."""
    cells: dict = collections.defaultdict(list)
    for r in rows:
        cells[(r["arm"], float(r["budget"]))].append(r)
    control_rows = cells[("clean", 0.0)]
    cprof, ccov = detect.profiles_from_rows(control_rows, cal["k"], cal, id_key="prompt_index")
    out = {}
    for key, cell_rows in cells.items():
        prof, cov = detect.profiles_from_rows(cell_rows, cal["k"], cal, id_key="prompt_index")
        if key[0] != "clean":
            mismatched, unknown = detect.verify_control(cell_rows, control_rows)
            if mismatched:
                raise SystemExit(f"{key}: control holds a different question for {mismatched}")
            inj = cell_rows[0]["inj_layer"]
            for pid in prof:
                if pid in unknown:
                    continue
                for layer in layers:
                    if layer < inj and layer in cprof.get(pid, {}):
                        prof[pid].setdefault(layer, cprof[pid][layer])
                        cov[pid].setdefault(layer, ccov[pid][layer])
        rep = detect.prompt_fpr(prof, stats, layers, coverage=cov, **{
            k: cal[k] for k in ("sigma", "min_run", "stride", "sigma_any", "dense_tail")})
        per = rep["per_id"]
        out[key] = {"n": rep["n_prompts"],
                    "flagged": sum(v["flagged"] for v in per.values()),
                    "unscorable": rep["n_unscorable"],
                    "clear": sum(1 for v in per.values() if v["verdict"] == "clear"),
                    "per_id": {pid: v["verdict"] for pid, v in per.items()}}
    return out


def bank_past_miss(cal: dict, root: Path) -> dict:
    """How many rows of the calibration's own fit set carry a position after a tolerance
    miss: zero means the calibration is already consistent with the deployment rule."""
    fit = Path(cal.get("fit_layers_dir", ""))
    if not fit.is_dir():
        parts = fit.parts
        if "sipit" in parts:
            fit = root / "sipit" / Path(*parts[parts.index("sipit") + 1:])
    if not fit.is_dir():
        return {"fit_layers_dir": str(fit), "rows": 0, "rows_past_miss": None}
    rows = past = 0
    for path in sorted(fit.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            steps = json.loads(line)["steps"]
            rows += 1
            cut = next((i for i, s in enumerate(steps) if not s.get("matched", True)), None)
            past += cut is not None and cut + 1 < len(steps)
    return {"fit_layers_dir": str(fit), "rows": rows, "rows_past_miss": past}


def analyse(lad: Ladder) -> dict:
    """Evaluator and deployment verdicts for every (arm, budget) cell of one ladder."""
    cal = json.loads(resolve(lad.calibration_path).read_text())
    stats = {int(k): detect.LayerStat(v["n"], v["mean"], v["sd"]) for k, v in cal["per_layer"].items()}
    layers = sorted(stats)
    rows = [json.loads(l) for l in resolve(lad.rows).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("behavior") == lad.behavior]
    saved_modes = collections.Counter(r.get("stop_on", "unrecorded") for r in rows)

    def shortened(r: dict) -> bool:
        return len(deploy_truncate(r)["steps"]) < len(r["steps"])

    rows_with_miss = sum(1 for r in rows if deploy_truncate(r) is not r)
    truncated_rows = sum(1 for r in rows if shortened(r))
    clean_cut_by_layer = collections.Counter(
        r["layer"] for r in rows if r["arm"] == "clean" and shortened(r))
    clean_rows_by_layer = collections.Counter(r["layer"] for r in rows if r["arm"] == "clean")
    ev = verdicts(rows, cal, stats, layers)
    dp = verdicts([deploy_truncate(r) for r in rows], cal, stats, layers)
    entry = {"model": lad.label, "behavior": lad.behavior, "calibration": lad.calibration,
             "saved_stop_modes": dict(saved_modes), "rows": len(rows),
             "rows_with_a_miss": rows_with_miss,
             "rows_shortened_by_deployment_rule": truncated_rows,
             "clean_rows_cut_by_layer": {str(L): [clean_cut_by_layer[L], clean_rows_by_layer[L]]
                                         for L in sorted(clean_rows_by_layer) if clean_cut_by_layer[L]},
             "calibration_bank": bank_past_miss(cal, resolve(lad.root)), "cells": [],
             "provenance": provenance.stamp(rows=lad.rows, calibration=lad.calibration_path)}
    for key in sorted(ev, key=str):
        e, d = ev[key], dp[key]
        changed = sorted(pid for pid in e["per_id"] if e["per_id"][pid] != d["per_id"].get(pid))
        entry["cells"].append({"arm": key[0], "budget": key[1], "n": e["n"],
                               "evaluator": {k: e[k] for k in ("flagged", "clear", "unscorable")},
                               "deployment": {k: d[k] for k in ("flagged", "clear", "unscorable")},
                               "changed_prompts": changed,
                               "changes": {str(pid): [e["per_id"][pid], d["per_id"][pid]] for pid in changed}})
    return entry


def print_header() -> None:
    print("%-16s %-7s %7s | %-24s | %-24s | %s" % ("model", "arm", "budget", "evaluator f/c/u",
                                                   "deployment f/c/u", "prompts that change"))


def print_cells(entry: dict) -> None:
    for c in entry["cells"]:
        e, d = c["evaluator"], c["deployment"]
        print("%-16s %-7s %7g | %2d / %2d / %2d              | %2d / %2d / %2d              | %s" % (
            entry["model"], c["arm"], c["budget"], e["flagged"], e["clear"], e["unscorable"],
            d["flagged"], d["clear"], d["unscorable"],
            ", ".join(f"{pid}:{v[0]}->{v[1]}" for pid, v in c["changes"].items()) or "-"))
