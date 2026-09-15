"""Per-budget rates with prompt-clustered uncertainty for the PGD ladders and the CAA sweep.

Answers audit item 6 (7 Sep 2026): fifty prompts evaluated at five budgets are fifty
clusters, not 250 draws. Per ladder and budget: detected and judge-positive counts with
Clopper-Pearson intervals, gains against the paired clean arm, undetected positives; pooled
over budgets, a prompt-cluster bootstrap and the zero-event bounds under each independence
assumption. The CUDA ladders share their 15 prompts and are pooled with the prompt as the
cluster and the model as the stratum. The August CAA sweep is pooled the same way from its
join files, byte-identical copies dropped. Read-only over saved results.
Driver: scripts/cluster_rates_audit.py.
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import behaviors
import detect
from ladders import ROOT, Ladder, resolve

BUDGETS = (0.0085, 0.02, 0.05, 0.12, 0.3)
N50_NAME = "Qwen-0.5B n50 (fp16 rows, fp16 generation)"


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(pred, lo=0.0, hi=1.0, iters=60) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pred(mid) else (lo, mid)
    return lo


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2)
    hi = 1.0 if k == n else _bisect(lambda p: binom_cdf(k, n, p) > alpha / 2)
    return lo, hi


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    return 1 - alpha ** (1 / n) if n > 0 else float("nan")


def cluster_bootstrap(clusters: list[list[int]], reps: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        flat = [x for _ in clusters for x in clusters[rng.randrange(len(clusters))]]
        means.append(sum(flat) / len(flat))
    means.sort()
    flat = [x for c in clusters for x in c]
    return sum(flat) / len(flat), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def ladder_outcomes(gen_dir: Path, n_prompts: int) -> dict:
    """(arm, budget, question) -> judge positive; (budget, question) -> the paired clean verdict."""
    out, base = {}, {}
    for path in sorted(resolve(gen_dir).glob("jbb_refusal_harmful_b*_gen.json")):
        blob = json.loads(path.read_text())
        assert blob["n_prompts"] == n_prompts, (path, blob["n_prompts"])
        b = float(blob["budget"])
        for r in blob["rows"]:
            if r["arm"] == "none":
                base[(b, r["question"])] = r["judge_harmbench"]
            else:
                out[(r["arm"], b, r["question"])] = r["judge_harmbench"]
    return {"judge": out, "baseline": base}


def join_verdicts(lad: Ladder) -> dict:
    """Verdicts as the join step wrote them, for a ladder scored on this machine."""
    verdicts = {}
    for line in resolve(lad.root / "pgd_sipit/detection_vs_efficacy_rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        verdicts[(r["arm"], float(r["fraction"]), r["question"])] = int(r["detected"])
    return verdicts


def rescored_verdicts(lad: Ladder, questions: list[str]) -> dict:
    """Verdicts rescored from the saved rows under the ladder's calibration, with the
    clean arm as control for the layers ahead of the injection."""
    cal = json.loads(resolve(lad.calibration_path).read_text())
    stats = {int(k): detect.LayerStat(v["n"], v["mean"], v["sd"]) for k, v in cal["per_layer"].items()}
    layers = sorted(stats)
    cells, inj, rows_of = {}, {}, {}
    for line in resolve(lad.rows).read_text().splitlines():
        row = json.loads(line)
        if row.get("behavior") != lad.behavior:
            continue
        key = row["arm"], float(row["budget"])
        profile = cells.setdefault(key, {}).setdefault(row["prompt_index"], {})
        if row["layer"] in profile:
            raise SystemExit(f"{lad.root.name}: two rows for cell {key}, prompt "
                             f"{row['prompt_index']}, layer {row['layer']}")
        profile[row["layer"]] = detect.row_score(row, cal["k"], cal)
        rows_of.setdefault(key, []).append(row)
        inj[key] = row["inj_layer"]
    control = cells[("clean", 0.0)]
    verdicts = {}
    for key, profiles in cells.items():
        if key[0] == "clean":
            continue
        mismatched, unknown = detect.verify_control(rows_of[key], rows_of[("clean", 0.0)])
        if mismatched or unknown:
            raise SystemExit(f"{lad.root.name} cell {key}: clean control mismatched for "
                             f"{mismatched}, unverifiable for {unknown}")
        full = {}
        for pid, prof in profiles.items():
            full[pid] = {l: v for l, v in control[pid].items() if l < inj[key]}
            full[pid].update(prof)
        rep = detect.prompt_fpr(full, stats, layers, **{k: cal[k] for k in
                                ("sigma", "min_run", "stride", "sigma_any", "dense_tail")})
        if rep["n_unscorable"]:
            print(f"WARNING {lad.root.name} cell {key}: {rep['n_unscorable']} unscorable prompt(s) "
                  "counted as not detected", file=sys.stderr)
        for pid, v in rep["per_id"].items():
            verdicts[(key[0], key[1], questions[pid])] = int(v["flagged"])
    return verdicts


def digest(path: Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def clean_stability(gen_dir: Path, questions: list[str]) -> dict:
    """The paired clean arm is regenerated per ladder file: how stable is its verdict?"""
    verdict = {q: [] for q in questions}
    texts = {q: set() for q in questions}
    for path in sorted(resolve(gen_dir).glob("jbb_refusal_harmful_b*_gen.json")):
        for r in json.loads(path.read_text())["rows"]:
            if r["arm"] == "none":
                verdict[r["question"]].append(r["judge_harmbench"])
                texts[r["question"]].add(r["completion"])
    n_files = len(verdict[questions[0]])
    return {"ladder_files": n_files,
            "clean_positive_per_file": [sum(verdict[q][i] for q in questions) for i in range(n_files)],
            "always_positive_positions": [i for i, q in enumerate(questions) if verdict[q] and all(verdict[q])],
            "mixed_verdict_positions": [i for i, q in enumerate(questions) if len(set(verdict[q])) > 1],
            "identical_completion_prompts": sum(1 for q in questions if len(texts[q]) == 1)}


def tabulate(name: str, questions: list[str], judge: dict, baseline: dict, verdicts: dict,
             cats: dict[str, str], provenance: dict | None = None,
             stability: dict | None = None) -> dict:
    out = {"name": name, "n_prompts": len(questions), "arms": {}, "provenance": provenance or {},
           "clean_stability": stability or {}}
    position = {q: i for i, q in enumerate(questions)}
    for arm in ("pgd", "random"):
        per_budget = []
        pos_by_prompt = {q: [] for q in questions}
        det_by_prompt = {q: [] for q in questions}
        for b in BUDGETS:
            pos = [q for q in questions if judge.get((arm, b, q)) == 1]
            new = [q for q in pos if baseline.get((b, q)) == 0]
            det = [q for q in questions if verdicts.get((arm, b, q)) == 1]
            missed = [q for q in pos if verdicts.get((arm, b, q)) != 1]
            n = len(questions)
            for q in questions:
                pos_by_prompt[q].append(int(judge.get((arm, b, q)) == 1))
                det_by_prompt[q].append(int(verdicts.get((arm, b, q)) == 1))
            per_budget.append({
                "budget": b, "n": n, "detected": len(det),
                "detect_ci95": clopper_pearson(len(det), n),
                "judge_positive": len(pos), "asr_ci95": clopper_pearson(len(pos), n),
                "newly_positive": len(new), "undetected_positive": len(missed),
                "detected_positions": [position[q] for q in det],
                "judge_positive_positions": [position[q] for q in pos],
                "newly_positive_positions": [position[q] for q in new],
                "undetected_positive_positions": [position[q] for q in missed],
                "undetected_positive_prompts": missed})
        pos_prompts = [q for q in questions if any(pos_by_prompt[q])]
        new_prompts = [q for q in questions if any(
            judge.get((arm, b, q)) == 1 and baseline.get((b, q)) == 0 for b in BUDGETS)]
        by_cat = {}
        for q in questions:
            c = by_cat.setdefault(cats.get(q, "?"), {"prompts": 0, "positive_prompts": 0,
                                                    "positive_rows": 0, "undetected_positive_rows": 0})
            c["prompts"] += 1
            c["positive_prompts"] += int(any(pos_by_prompt[q]))
            c["positive_rows"] += sum(pos_by_prompt[q])
            c["undetected_positive_rows"] += sum(
                1 for b in BUDGETS if judge.get((arm, b, q)) == 1 and verdicts.get((arm, b, q)) != 1)
        out["arms"][arm] = {
            "per_budget": per_budget,
            "pooled": {
                "rows": len(questions) * len(BUDGETS),
                "positive_rows": sum(sum(v) for v in pos_by_prompt.values()),
                "positive_prompts": len(pos_prompts), "newly_positive_prompts": len(new_prompts),
                "undetected_positive_rows": sum(x["undetected_positive"] for x in per_budget),
                "asr_cluster_bootstrap": cluster_bootstrap([pos_by_prompt[q] for q in questions]),
                "detection_cluster_bootstrap": cluster_bootstrap([det_by_prompt[q] for q in questions]),
                "zero_event_upper_by_prompt_clusters": zero_event_upper(len(questions)),
                "zero_event_upper_by_positive_prompts": zero_event_upper(len(pos_prompts)),
                "zero_event_upper_if_rows_independent": zero_event_upper(
                    sum(sum(v) for v in pos_by_prompt.values()))},
            "by_category": by_cat,
            "pos_by_prompt": pos_by_prompt, "det_by_prompt": det_by_prompt}
    return out


def sweep_b() -> dict:
    points, clusters, by_behavior = [], 0, {}
    on_disk = sorted(list(ROOT.glob("results/*/behavior/*/*/detection_vs_efficacy.json"))
                     + list(ROOT.glob("results_cuda/*/behavior/*/*/detection_vs_efficacy.json")))
    seen, files = set(), []
    for path in on_disk:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            files.append(path)
    for path in files:
        d = json.loads(path.read_text())
        n_prompts = 0
        for p in d["points"]:
            if p["arm"] == "clean":
                continue
            n_prompts = max(n_prompts, p["n"] // max(1, p.get("replicates", 1)))
            points.append({"model": path.parts[-5], "behavior": d["behavior"], "test_arm": d["arm"],
                           "arm": p["arm"], "fraction": p["fraction"], "n": p["n"],
                           "jailbroken": p["jailbroken_detected"] + p["jailbroken_undetected"],
                           "undetected_jailbroken": p["jailbroken_undetected"],
                           "detected": p["jailbroken_detected"] + p["clean_detected"]})
        clusters += n_prompts
        key = (d["behavior"], d["arm"])
        by_behavior[key] = max(by_behavior.get(key, 0), n_prompts)
    jb = sum(p["jailbroken"] for p in points)
    incomplete = [p for p in points if p["detected"] < p["n"]]
    by_frac = {}
    for pt in points:
        f = by_frac.setdefault(pt["fraction"], {"rows": 0, "jailbroken": 0, "undetected": 0, "points": 0})
        f["rows"] += pt["n"]
        f["jailbroken"] += pt["jailbroken"]
        f["undetected"] += pt["undetected_jailbroken"]
        f["points"] += 1
    return {"n_files_on_disk": len(on_disk), "n_duplicates_dropped": len(on_disk) - len(files),
            "n_files": len(files), "n_points": len(points), "per_fraction": by_frac,
            "incomplete_points": incomplete,
            "behavior_prompt_clusters": {f"{k[0]}/{k[1]}": v for k, v in by_behavior.items()}, "pooled": {
        "rows": sum(p["n"] for p in points), "jailbroken_rows": jb,
        "undetected_jailbroken": sum(p["undetected_jailbroken"] for p in points),
        "prompt_clusters": clusters,
        "zero_event_upper_if_rows_independent": zero_event_upper(jb),
        "zero_event_upper_by_prompt_clusters": zero_event_upper(clusters),
        "behavior_prompt_clusters": sum(by_behavior.values()),
        "zero_event_upper_by_behavior_prompt_clusters": zero_event_upper(sum(by_behavior.values())),
        "points_with_any_jailbreak": sum(1 for p in points if p["jailbroken"]),
        "points_with_full_detection": sum(1 for p in points if p["detected"] == p["n"])}}


def fmt_ci(ci):
    return "%.0f-%.0f%%" % (100 * ci[0], 100 * ci[1])


def print_ladder(t: dict) -> None:
    print(f"\n{t['name']}  ({t['n_prompts']} prompts x {len(BUDGETS)} budgets)")
    for arm, a in t["arms"].items():
        print(f"  {arm}: budget   detected/n   CP95        judge+   ASR CP95    new+  undetected+")
        for x in a["per_budget"]:
            print("        %6g   %3d/%-3d  %-11s  %3d      %-10s  %3d   %3d" % (
                x["budget"], x["detected"], x["n"], fmt_ci(x["detect_ci95"]), x["judge_positive"],
                fmt_ci(x["asr_ci95"]), x["newly_positive"], x["undetected_positive"]))
        p = a["pooled"]
        asr, det = p["asr_cluster_bootstrap"], p["detection_cluster_bootstrap"]
        print("     pooled %d rows: %d judge+ rows on %d distinct prompts (%d newly positive vs paired clean), %d undetected judge+" % (
            p["rows"], p["positive_rows"], p["positive_prompts"], p["newly_positive_prompts"],
            p["undetected_positive_rows"]))
        print("     ASR %.1f%% [%.1f, %.1f] prompt-cluster bootstrap; detection %.1f%% [%.1f, %.1f]" % (
            100 * asr[0], 100 * asr[1], 100 * asr[2], 100 * det[0], 100 * det[1], 100 * det[2]))
        print("     one-sided 95%% upper bound on P(undetected | judge+): %.1f%% if the %d rows were independent; %.1f%% by %d positive prompts; %.1f%% by %d prompt clusters" % (
            100 * p["zero_event_upper_if_rows_independent"], p["positive_rows"],
            100 * p["zero_event_upper_by_positive_prompts"], p["positive_prompts"],
            100 * p["zero_event_upper_by_prompt_clusters"], t["n_prompts"]))
        print("     by category: " + "; ".join(
            "%s %d prompts, %d positive prompts, %d/%d positive rows undetected" % (
                c, v["prompts"], v["positive_prompts"], v["undetected_positive_rows"], v["positive_rows"])
            for c, v in sorted(a["by_category"].items())))


def strip(t: dict) -> dict:
    return {"name": t["name"], "n_prompts": t["n_prompts"], "provenance": t["provenance"],
            "clean_stability": t["clean_stability"],
            "arms": {arm: {k: v for k, v in a.items() if k not in ("pos_by_prompt", "det_by_prompt")}
                     for arm, a in t["arms"].items()}}


def run(n50: Ladder, cuda: list[Ladder]) -> dict:
    """The n50 ladder is read through its join file; the CUDA ladders are rescored from
    their rows and pooled on their shared prompts. Prints the readout as it goes."""
    beh = behaviors.load_behavior(n50.behavior)
    cats = {it.question: it.category for it in beh.items("harmful", 0)}
    out = {}
    q50 = [it.question for it in beh.items("harmful", n50.n_prompts)]
    lad = ladder_outcomes(n50.deltas, n50.n_prompts)
    prov = {"calibration": f"sipit/{n50.calibration}",
            "calibration_sha256": digest(n50.calibration_path),
            "join_rows_sha256": digest(n50.root / "pgd_sipit/detection_vs_efficacy_rows.jsonl"),
            "generation_sha256": {p.name: digest(p) for p in sorted(resolve(n50.deltas).glob("*_gen.json"))}}
    t = tabulate(N50_NAME, q50, lad["judge"], lad["baseline"], join_verdicts(n50), cats, prov,
                 clean_stability(n50.deltas, q50))
    print_ladder(t)
    out["n50"] = strip(t)

    n_cuda = {lad.n_prompts for lad in cuda}
    assert len(n_cuda) == 1, f"the pooled CUDA table needs one shared prompt count, got {n_cuda}"
    q15 = [it.question for it in beh.items("harmful", n_cuda.pop())]
    out["cuda"], tables = {}, {}
    for lad_ in cuda:
        if not resolve(lad_.rows).exists() or not resolve(lad_.deltas).exists():
            continue
        lad = ladder_outcomes(lad_.deltas, lad_.n_prompts)
        prov = {"calibration": f"sipit/{lad_.calibration}",
                "calibration_sha256": digest(lad_.calibration_path),
                "rows_sha256": digest(lad_.rows),
                "generation_sha256": {p.name: digest(p) for p in sorted(resolve(lad_.deltas).glob("*_gen.json"))}}
        t = tabulate(lad_.root.name, q15, lad["judge"], lad["baseline"], rescored_verdicts(lad_, q15), cats, prov,
                     clean_stability(lad_.deltas, q15))
        print_ladder(t)
        tables[lad_.root.name] = t
        out["cuda"][lad_.root.name] = strip(t)

    print(f"\nCUDA ladders pooled across {len(tables)} models on the same {len(q15)} prompts (prompt = cluster, model = stratum)")
    out["cuda_pooled"] = {}
    for arm in ("pgd", "random"):
        print(f"  {arm}: budget  judge+ / prompt-model pairs  distinct prompts  undetected+  ASR prompt-cluster bootstrap  detection")
        for i, b in enumerate(BUDGETS):
            pos_clusters = [[t["arms"][arm]["pos_by_prompt"][q][i] for t in tables.values()] for q in q15]
            det_clusters = [[t["arms"][arm]["det_by_prompt"][q][i] for t in tables.values()] for q in q15]
            asr = cluster_bootstrap(pos_clusters)
            det = cluster_bootstrap(det_clusters)
            pos_rows = sum(sum(c) for c in pos_clusters)
            missed = sum(t["arms"][arm]["per_budget"][i]["undetected_positive"] for t in tables.values())
            print("        %6g    %3d / %-3d                   %3d              %3d         %5.1f%% [%.1f, %.1f]        %5.1f%% [%.1f, %.1f]" % (
                b, pos_rows, len(q15) * len(tables), sum(1 for c in pos_clusters if any(c)), missed,
                100 * asr[0], 100 * asr[1], 100 * asr[2], 100 * det[0], 100 * det[1], 100 * det[2]))
            out["cuda_pooled"][f"{arm}@{b:g}"] = {
                "positive_rows": pos_rows, "pairs": len(q15) * len(tables),
                "positive_prompts": sum(1 for c in pos_clusters if any(c)),
                "undetected_positive": missed, "asr_cluster_bootstrap": asr,
                "detection_cluster_bootstrap": det}

    sb = sweep_b()
    out["sweep_b"] = sb
    p = sb["pooled"]
    print(f"\nCAA sweep B (fractions): {sb['n_files_on_disk']} join files on disk, {sb['n_duplicates_dropped']} byte-identical copies dropped, {sb['n_files']} distinct, {sb['n_points']} (model, behavior, arm, fraction) points")
    print("  %d steered rows, %d judge+ rows, %d undetected judge+; %d/%d points fully detected; %d points with any jailbreak" % (
        p["rows"], p["jailbroken_rows"], p["undetected_jailbroken"], p["points_with_full_detection"],
        sb["n_points"], p["points_with_any_jailbreak"]))
    print("  one-sided 95%% upper bound on P(undetected | judge+): %.2f%% if the %d rows were independent; %d prompt clusters (prompts x model x behavior x arm), bound %.2f%%" % (
        100 * p["zero_event_upper_if_rows_independent"], p["jailbroken_rows"],
        p["prompt_clusters"], 100 * p["zero_event_upper_by_prompt_clusters"]))
    print("  the same prompts recur across models and dtypes: %d distinct (behavior, prompt) clusters, bound %.2f%%" % (
        p["behavior_prompt_clusters"], 100 * p["zero_event_upper_by_behavior_prompt_clusters"]))
    for q in sb["incomplete_points"]:
        print("  not fully detected: %s %s/%s %s f=%g detected %d/%d, judge+ %d, undetected judge+ %d" % (
            q["model"], q["behavior"], q["test_arm"], q["arm"], q["fraction"], q["detected"], q["n"],
            q["jailbroken"], q["undetected_jailbroken"]))
    print("  per fraction: " + "; ".join("%g: %d judge+ of %d rows, %d undetected (%d points)" % (
        k, v["jailbroken"], v["rows"], v["undetected"], v["points"]) for k, v in sorted(sb["per_fraction"].items())))
    return out
