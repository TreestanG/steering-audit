"""Per-layer stopping tolerance for inversion without the original prompt, and the
check that deployment-mode rows reproduce the evaluator rows.

The tolerance at a layer is max(base, factor x the largest clean residual/norm over the
calibration bank), fitted on clean bank rows only. `holdout` repeats that fit on folds
and counts test-fold rows cut before their last position. `compare_bank` and
`compare_ladder` pair deployment rows (--stop_on miss) with evaluator rows (--stop_on
wrong) position by position and score both under one calibration.
Drivers: scripts/deploy_tolerance.py, scripts/deploy_compare.py.
"""

import collections
import copy
import json
import random
import statistics
from pathlib import Path

import deploy_stop
import detect
import provenance
from ladders import Ladder, resolve

REL_EPS = 1e-6


def load_layer_rows(layers_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(Path(layers_dir).glob("sipit_layer_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def load_jsonl(path: Path, behavior: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("behavior") == behavior] if behavior else rows


def ratios(row: dict) -> list[float]:
    return [s["residual"] / s["h_norm"] for s in row["steps"] if s.get("h_norm")]


def tolerance(rows: list[dict], factor: float, base: float) -> dict[int, float]:
    mx: dict[int, float] = {}
    for r in rows:
        mx[r["layer"]] = max(mx.get(r["layer"], 0.0), max(ratios(r), default=0.0))
    return {L: max(base, factor * m) for L, m in sorted(mx.items())}


def cut_before_last(row: dict, tol: float) -> bool:
    return any(x > tol for x in ratios(row)[:-1])


def holdout(rows: list[dict], factor: float, base: float, folds: int, repeats: int, seed: int) -> dict:
    ids = sorted({r["id"] for r in rows})
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    cut_rates, cut_prompts = [], []
    for rep in range(repeats):
        order = ids[:]
        random.Random(seed + rep).shuffle(order)
        fold_of = {i: k % folds for k, i in enumerate(order)}
        for t in range(folds):
            tol = tolerance([r for i in ids if fold_of[i] != t for r in by_id[i]], factor, base)
            test_ids = [i for i in ids if fold_of[i] == t]
            cut = prompts_cut = 0
            for i in test_ids:
                hit = [cut_before_last(r, tol.get(r["layer"], base)) for r in by_id[i]]
                cut += sum(hit)
                prompts_cut += any(hit)
            cut_rates.append(cut / max(1, sum(len(by_id[i]) for i in test_ids)))
            cut_prompts.append(prompts_cut / max(1, len(test_ids)))
    return {"folds": folds, "repeats": repeats, "seed": seed,
            "row_cut_rate_mean": statistics.fmean(cut_rates),
            "row_cut_rate_max": max(cut_rates),
            "prompt_cut_rate_mean": statistics.fmean(cut_prompts),
            "prompt_cut_rate_max": max(cut_prompts)}


def fit(layers_dir: Path, factor: float, base: float, folds: int, repeats: int, seed: int) -> dict:
    rows = load_layer_rows(layers_dir)
    tol = tolerance(rows, factor, base)
    return {"rule": f"max({base:g}, {factor:g} x max clean residual/norm per layer)",
            "layers_dir": str(layers_dir), "n_rows": len(rows),
            "n_prompts": len({r["id"] for r in rows}),
            "base_rel_tol": base, "factor": factor,
            "rows_cut_before_last_at_base": sum(cut_before_last(r, base) for r in rows),
            "rows_cut_before_last_at_tolerance": sum(cut_before_last(r, tol[r["layer"]]) for r in rows),
            "layers": {str(L): t for L, t in tol.items()},
            "held_out": holdout(rows, factor, base, folds, repeats, seed)}


def print_fit(out: dict, path: Path) -> None:
    raised = {L: t for L, t in out["layers"].items() if t > out["base_rel_tol"]}
    print(f"{path}: {out['n_rows']} rows, {out['n_prompts']} prompts; cut at base "
          f"{out['rows_cut_before_last_at_base']}, at tolerance {out['rows_cut_before_last_at_tolerance']}; "
          "layers above base: " + ", ".join(f"L{L}={t:.4f}" for L, t in raised.items()))
    ho = out["held_out"]
    print(f"  held-out row cut rate mean {ho['row_cut_rate_mean']:.4f} max {ho['row_cut_rate_max']:.4f}; "
          f"prompt cut rate mean {ho['prompt_cut_rate_mean']:.4f} max {ho['prompt_cut_rate_max']:.4f}")


def truncate_at(row: dict, tol: dict[int, float], base: float) -> dict:
    t = tol.get(row["layer"], base)
    steps = row["steps"]
    cut = next((i for i, s in enumerate(steps) if s.get("h_norm") and s["residual"] / s["h_norm"] > t), None)
    if cut is None:
        return row
    r = copy.copy(row)
    r["steps"] = steps[:cut + 1]
    r["n_recovered"] = len(r["steps"])
    r["stop_on"] = "miss"
    return r


def reread_ladder(lad: Ladder, factor: float, base: float, folds: int, repeats: int, seed: int) -> dict:
    """Evaluator verdicts beside the saved rows cut where a deployment would halt, at the flat
    base tolerance and at the per-layer tolerance fitted on the ladder's own calibration bank."""
    cal = json.loads(resolve(lad.calibration_path).read_text())
    stats = {int(k): detect.LayerStat(v["n"], v["mean"], v["sd"]) for k, v in cal["per_layer"].items()}
    layers = sorted(stats)
    fitted = fit(resolve(lad.bank_layers), factor, base, folds, repeats, seed)
    tol = {int(k): v for k, v in fitted["layers"].items()}
    rows = load_jsonl(resolve(lad.rows), lad.behavior)
    at_base = [deploy_stop.deploy_truncate(r) for r in rows]
    at_tol = [truncate_at(r, tol, base) for r in rows]
    ev = deploy_stop.verdicts(rows, cal, stats, layers)
    vb = deploy_stop.verdicts(at_base, cal, stats, layers)
    vt = deploy_stop.verdicts(at_tol, cal, stats, layers)

    def counts(v: dict, key: tuple) -> dict:
        return {k: v[key][k] for k in ("flagged", "clear", "unscorable", "n")}

    def changed(v: dict, key: tuple) -> list[str]:
        return sorted(str(p) for p, verdict in ev[key]["per_id"].items() if verdict != v[key]["per_id"].get(p))

    cells = [{"arm": key[0], "budget": key[1], "evaluator": counts(ev, key),
              "deployment_base": counts(vb, key), "deployment_tolerance": counts(vt, key),
              "changed_at_base": changed(vb, key), "changed_at_tolerance": changed(vt, key)}
             for key in sorted(ev, key=str)]
    return {"model": lad.label, "behavior": lad.behavior, "calibration": lad.calibration,
            "rows": len(rows),
            "rows_shortened_at_base": sum(len(a["steps"]) < len(r["steps"]) for a, r in zip(at_base, rows)),
            "rows_shortened_at_tolerance": sum(len(a["steps"]) < len(r["steps"]) for a, r in zip(at_tol, rows)),
            "tolerance": fitted, "cells": cells,
            "provenance": provenance.stamp(rows=lad.rows, calibration=lad.calibration_path)}


def summarise_run(report: dict, path: Path) -> dict:
    """The parts of a deploy_compare report RESULTS.md reads, with the report's own hash."""
    out = {"report": provenance.stamp(report=path)["inputs"].get("report")}
    for part in ("bank", "ladder"):
        if part in report:
            out[part] = {k: v for k, v in report[part].items() if k != "by_layer" and k != "differing"}
    return out


def compare_pair(d: dict, e: dict) -> dict:
    ds, es = d["steps"], e["steps"]
    n = min(len(ds), len(es))
    rel = [abs(ds[i]["residual"] - es[i]["residual"]) / max(es[i].get("h_norm") or 1.0, 1e-30) for i in range(n)]
    d_miss = next((i for i, s in enumerate(ds) if not s.get("matched", True)), None)
    return {"n_deploy": len(ds), "n_eval": len(es), "n_target": d.get("n_target"),
            "tokens_equal": all(ds[i]["token"] == es[i]["token"] for i in range(n)),
            "positions_compared": n,
            "max_rel_residual_diff": max(rel, default=0.0),
            "deploy_shorter": len(ds) < len(es), "deploy_longer": len(ds) > len(es),
            "deploy_first_miss": d_miss,
            "eval_first_miss": next((i for i, s in enumerate(es) if not s.get("matched", True)), None),
            "deploy_miss_only_at_last": d_miss is None or (d_miss == len(ds) - 1 and len(ds) == d.get("n_target")),
            "deploy_complete": len(ds) == d.get("n_target", len(ds)) and d_miss is None}


def summarise(pairs: list[dict]) -> dict:
    return {"rows": len(pairs),
            "tokens_identical": sum(p["tokens_equal"] for p in pairs),
            "residuals_identical": sum(p["max_rel_residual_diff"] <= REL_EPS for p in pairs),
            "max_rel_residual_diff": max((p["max_rel_residual_diff"] for p in pairs), default=0.0),
            "deploy_shorter": sum(p["deploy_shorter"] for p in pairs),
            "deploy_longer": sum(p["deploy_longer"] for p in pairs),
            "deploy_rows_with_a_miss": sum(p["deploy_first_miss"] is not None for p in pairs),
            "deploy_misses_before_last_position": sum(not p["deploy_miss_only_at_last"] for p in pairs),
            "deploy_rows_complete": sum(p["deploy_complete"] for p in pairs)}


def _differing(pairs: list[dict]) -> list[dict]:
    return [p for p in pairs if not p["tokens_equal"] or p["max_rel_residual_diff"] > REL_EPS
            or p["deploy_shorter"] or p["deploy_longer"]]


def compare_bank(deploy: list[dict], evaluator: list[dict]) -> dict:
    ev = {(r["id"], r["layer"]): r for r in evaluator}
    pairs, missing, by_layer = [], 0, collections.defaultdict(list)
    for d in deploy:
        e = ev.get((d["id"], d["layer"]))
        if e is None:
            missing += 1
            continue
        c = compare_pair(d, e) | {"id": d["id"], "layer": d["layer"]}
        pairs.append(c)
        by_layer[d["layer"]].append(c)
    return summarise(pairs) | {"unmatched_deploy_rows": missing,
                               "by_layer": {str(L): summarise(v) for L, v in sorted(by_layer.items())},
                               "differing": _differing(pairs)}


def _cell_key(r: dict) -> tuple:
    return (r["arm"], float(r["budget"]), r["prompt_index"], r["layer"])


def compare_ladder(deploy: list[dict], evaluator: list[dict], cal: dict) -> dict:
    ev = {_cell_key(r): r for r in evaluator}
    budgets = {float(r["budget"]) for r in deploy}
    eval_sub = [r for r in evaluator if float(r["budget"]) in budgets]
    pairs, missing = [], 0
    for d in deploy:
        e = ev.get(_cell_key(d))
        if e is None:
            missing += 1
            continue
        pairs.append(compare_pair(d, e) | {"key": list(_cell_key(d))})
    stats = {int(k): detect.LayerStat(v["n"], v["mean"], v["sd"]) for k, v in cal["per_layer"].items()}
    layers = sorted(stats)
    sets = {"evaluator": deploy_stop.verdicts(eval_sub, cal, stats, layers),
            "evaluator_truncated": deploy_stop.verdicts([deploy_stop.deploy_truncate(r) for r in eval_sub],
                                                        cal, stats, layers),
            "deployment_run": deploy_stop.verdicts(deploy, cal, stats, layers)}
    cells = []
    for key in sorted(sets["deployment_run"], key=str):
        cell = {"arm": key[0], "budget": key[1]}
        for name, vs in sets.items():
            v = vs.get(key)
            cell[name] = {k: v[k] for k in ("flagged", "clear", "unscorable", "n")} if v else None
        ev_per = sets["evaluator"].get(key, {}).get("per_id", {})
        dp_per = sets["deployment_run"][key]["per_id"]
        cell["changed_prompts"] = sorted(str(p) for p in dp_per if ev_per.get(p) != dp_per[p])
        cells.append(cell)
    return summarise(pairs) | {"unmatched_deploy_rows": missing, "differing": _differing(pairs), "cells": cells}


def print_bank(bank: dict) -> None:
    print("bank:", json.dumps({k: v for k, v in bank.items() if k not in ("by_layer", "differing")}))
    for L, v in bank["by_layer"].items():
        if v["tokens_identical"] != v["rows"] or v["residuals_identical"] != v["rows"] or v["deploy_shorter"]:
            print(f"  layer {L}: {json.dumps(v)}")
    for d in bank["differing"][:20]:
        print("  differs:", json.dumps(d))


def print_ladder(lad: dict) -> None:
    print("ladder:", json.dumps({k: v for k, v in lad.items() if k not in ("cells", "differing")}))
    for d in lad["differing"][:20]:
        print("  differs:", json.dumps(d))
    fmt = lambda v: "-" if v is None else f"{v['flagged']}/{v['clear']}/{v['unscorable']}"
    print(f"{'arm':<7} {'budget':<7} {'evaluator':<14} {'eval truncated':<14} {'deployment run':<14} changed")
    for c in lad["cells"]:
        print(f"{c['arm']:<7} {c['budget']:<7g} {fmt(c['evaluator']):<14} {fmt(c['evaluator_truncated']):<14} "
              f"{fmt(c['deployment_run']):<14} {','.join(c['changed_prompts'])}")
