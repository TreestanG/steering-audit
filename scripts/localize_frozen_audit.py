"""Localize the injection layer from saved attack rows under a frozen calibration.

Read-only; run from the repository root. Answers audit item 1 (7 Sep 2026): the old
localize_prompt benchmark set its threshold from the same prompt's clean run. Here the
takeoff is the first layer of the first qualifying run under the shipped role-z rule,
with the per-prompt 10x-clean-floor rule re-read on the same rows for comparison.

  uv run python scripts/localize_frozen_audit.py CAL ROWS [CAL ROWS ...]
"""

import json, sys, statistics, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import detect

def first_run_start(layers, fired, min_run):
    run = 0
    for i, L in enumerate(layers):
        run = run + 1 if L in fired else 0
        if run >= min_run:
            return layers[i - min_run + 1]
    return None

def analyse(cal_path, rows_path, mult=10.0):
    cal = json.load(open(cal_path))
    sigma, sigma_any, min_run, k = cal["sigma"], cal["sigma_any"], cal["min_run"], cal["k"]
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r.get("behavior"), r.get("test_arm"), r["arm"], r["budget"], r["constraint"], r.get("n_positions", 1))].append(r)
    def per_prompt(cell_rows):
        z, rel = {}, {}
        for r in cell_rows:
            s = detect.row_score(r, k, cal)
            res = detect.relative_residuals(r["steps"])
            if s == s and res:
                z.setdefault(r["id"], {})[r["layer"]] = s
                rel.setdefault(r["id"], {})[r["layer"]] = max(res)
        return z, rel
    controls = {key[:2]: per_prompt(v) for key, v in cells.items() if key[2] == "clean"}
    out = []
    for key in sorted(cells, key=str):
        beh, tarm, arm, budget, scope, m = key
        if arm == "clean":
            continue
        cz, crel = controls.get((beh, tarm), ({}, {}))
        z, rel = per_prompt(cells[key])
        inj = cells[key][0]["inj_layer"]
        layers = sorted({r["layer"] for r in cells[key]} | {L for p in cz.values() for L in p})
        hits = collections.Counter()
        offs = {"sigma": [], "rule": [], "oracle": []}
        n = 0
        for pid, prof in z.items():
            if pid not in cz:
                continue
            n += 1
            full = {L: (prof[L] if L >= inj and L in prof else cz[pid].get(L)) for L in layers}
            full = {L: v for L, v in full.items() if v is not None}
            fired = {L for L, v in full.items() if v > sigma}
            t_sigma = min(fired) if fired else None
            t_rule = first_run_start(layers, fired, min_run)
            if t_rule is None:
                lone = [L for L, v in full.items() if v > sigma_any]
                t_rule = min(lone) if lone else None
            rfull = {L: (rel[pid][L] if L >= inj and L in rel[pid] else crel[pid].get(L)) for L in layers}
            rfull = {L: v for L, v in rfull.items() if v is not None}
            floor = max(crel[pid].values())
            thr = max(mult * floor, 1e-3)
            over = [L for L, v in rfull.items() if v > thr]
            t_or = min(over) if over else None
            for name, t in (("sigma", t_sigma), ("rule", t_rule), ("oracle", t_or)):
                if t is None:
                    hits[name, "none"] += 1
                else:
                    hits[name, "exact"] += t == inj
                    offs[name].append(t - inj)
        def fmt(name):
            e, no = hits[name, "exact"], hits[name, "none"]
            o = offs[name]
            med = statistics.median(o) if o else float("nan")
            return "%3d/%-3d none %2d  off med %+3.0f" % (e, n, no, med)
        out.append("  %-13s %-7s %-6s %7s m=%s inj=%2s | z>sigma: %s | rule: %s | oracle x%g: %s" % (
            beh, tarm, arm, budget, m, inj, fmt("sigma"), fmt("rule"), mult, fmt("oracle")))
    print("%s  (sigma %.2f, any %.2f, run %d)" % (cal_path, sigma, sigma_any, min_run))
    print("\n".join(out))

if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        analyse(sys.argv[i], sys.argv[i + 1])
