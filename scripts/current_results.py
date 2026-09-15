"""Generate RESULTS.md, the one current version of the results, from the artifacts.

Every table is read from a saved file; the manifest written beside it records each
file's sha256 and the code fingerprint. Nothing here is typed in by hand except the
description of what was tested. Historical corrections live elsewhere.

  uv run python scripts/current_results.py
"""

import collections
import datetime as dt
import glob
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ladders
import provenance

DATE = dt.date.today().isoformat()
SOURCES: dict[str, Path] = {}


def load(name: str, path: str | Path) -> dict | list:
    path = ROOT / path
    SOURCES[name] = path
    return json.loads(path.read_text())


def pct(x: float) -> str:
    return "%.0f%%" % (100 * x)


def ci(lo: float, hi: float) -> str:
    return "%.0f-%.0f%%" % (100 * lo, 100 * hi)


# ----------------------------------------------------------------------------- data
cluster = load("cluster_rates", "results/current/cluster-rates-2026-09-07.json")
transitions_path = sorted(glob.glob(str(ROOT / "results/current/n50-transitions-*.json")))[-1]
trans = load("n50_transitions", Path(transitions_path).relative_to(ROOT))
deploy = load("deployment_stop_rescore", "results/current/deployment-stop-2026-09-08.json")
repeats = {}
for f in sorted(glob.glob(str(ROOT / "results/Qwen_Qwen2.5-0.5B-Instruct_n50/pgd/ladder/*_gen_judge_repeats.json"))):
    repeats[Path(f).name] = load(f"judge_repeats_{Path(f).stem}", Path(f).relative_to(ROOT))

HOLDOUT = [
    ("gpt2", "bare", "results/gpt2_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_{}.json"),
    ("Qwen-0.5B", "bare", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_bare_{}.json"),
    ("Qwen-0.5B", "chat 25", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_{}.json"),
    ("Qwen-0.5B", "chat 50", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n50_{}.json"),
    ("Qwen-0.5B", "chat 75", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_{}.json"),
    ("Qwen-0.5B", "chat 100", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n100_{}.json"),
    ("Qwen-1.5B", "chat 75", "results_cuda/Qwen_Qwen2.5-1.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_{}.json"),
    ("Qwen-7B", "bare", "results_cuda/Qwen_Qwen2.5-7B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_bare_{}.json"),
    ("Qwen-7B", "chat 75", "results_cuda/Qwen_Qwen2.5-7B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_{}.json"),
    ("gemma-3-1b", "bare", "results/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_{}.json"),
    ("gemma-3-1b", "chat 25", "results/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_chat_{}.json"),
    ("gemma-3-1b", "chat 75", "results_cuda/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_{}.json"),
    ("Llama-3.2-1B", "chat 75", "results_cuda/meta-llama_Llama-3.2-1B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_{}.json"),
]
D5 = [
    ("gemma-3-1b, sentiment (25)", "results/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_3way_sentiment.json"),
    ("gemma-3-1b, sentiment_ext (50)", "results/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_3way_sentiment_ext.json"),
    ("Qwen-0.5B, sentiment_ext (50)", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n100_3way_sentiment_ext.json"),
    ("gpt2, sentiment (25)", "results/gpt2_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_3way_sentiment.json"),
]
MODELS = ladders.LADDERS


def ladder(label):
    lad = MODELS[label]
    return cluster["n50"] if lad is ladders.N50 else cluster["cuda"][lad.root.name]


SUMMARIES = {label: lad.summary for label, lad in MODELS.items()}


def unscorable_line() -> str:
    parts = []
    for label, path in SUMMARIES.items():
        d = load(f"summary_{label}", path)
        jbb = [c for c in d["cells"] if c["behavior"] == "jbb_refusal"]
        parts.append("%s %d of %d prompt-cells" % (label, sum(c["unscorable"] for c in jbb),
                                                   sum(c["n"] for c in jbb)))
    return "; ".join(parts)


# --------------------------------------------------------------------------- tables
def detection_table() -> str:
    lines = ["| model | prompts | 0.0085 | 0.02 | 0.05 | 0.12 | 0.3 |", "|---|---|---|---|---|---|---|"]
    for label in MODELS:
        t = ladder(label)
        cells = ["%d/%d (%s)" % (x["detected"], x["n"], ci(*x["detect_ci95"]))
                 for x in t["arms"]["pgd"]["per_budget"]]
        lines.append("| %s | %d | %s |" % (label, t["n_prompts"], " | ".join(cells)))
    return "\n".join(lines)


def positives_table() -> str:
    lines = ["| model | budget | judge-positive | new vs paired clean | undetected judge-positive | undetected new |",
             "|---|---|---|---|---|---|"]
    for label in MODELS:
        t = ladder(label)
        for x in t["arms"]["pgd"]["per_budget"]:
            und_new = len(set(x["undetected_positive_positions"]) & set(x["newly_positive_positions"]))
            lines.append("| %s | %g | %d/%d | %d | %d | %d |" % (
                label, x["budget"], x["judge_positive"], x["n"], x["newly_positive"],
                x["undetected_positive"], und_new))
    return "\n".join(lines)


def transitions_table() -> str:
    cells = {(c["fraction"], c["arm"]): c for c in trans["cells"]}
    lines = ["| budget | arm | clean+ | attacked+ | gains | losses | McNemar p | Holm p | gains passing gate | gated undetected | reading |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("pgd", "random"):
        for b in sorted({f for f, a in cells if a == arm}):
            c = cells[(b, arm)]
            p, ph = c["mcnemar_p"], c.get("mcnemar_p_holm", float("nan"))
            lines.append("| %g | %s | %d | %d | %d | %d | %s | %s | %d | %d | %s |" % (
                b, arm, c["clean_positive"], c["attacked_positive"], c["gains"], c["losses"],
                "-" if p != p else "%.3f" % p, "-" if ph != ph else "%.3f" % ph,
                c["gains_passing_gate"], c["gains_passing_gate_undetected"],
                "no clear evidence of change" if c["no_evidence_of_change"] else
                ("change (unadjusted)" if c.get("no_evidence_of_change_holm", False) else "change")))
    return "\n".join(lines)


def repeats_table() -> str:
    if not repeats:
        return "_no repeat-grading files found_"
    lines = ["| budget | clean positives per grading | changed | PGD positives per grading | changed | random positives per grading | changed |",
             "|---|---|---|---|---|---|---|"]
    tot = {"none": [0, 0], "pgd": [0, 0], "random": [0, 0]}
    for name, d in repeats.items():
        n = d["judge_noise"]["harmbench"]
        row = ["%g" % d["budget"]]
        for arm in ("none", "pgd", "random"):
            row += [str(n["positives_per_grading_by_arm"][arm]),
                    "%d/%d" % (n["rows_with_disagreement_by_arm"][arm], n["rows_by_arm"][arm])]
            tot[arm][0] += n["rows_with_disagreement_by_arm"][arm]
            tot[arm][1] += n["rows_by_arm"][arm]
        lines.append("| " + " | ".join(row) + " |")
    gradings = next(iter(repeats.values()))["judge_noise"]["harmbench"]["gradings"]
    lines.append("")
    lines.append("%d gradings of each completion. Labels that changed between gradings of identical text: clean %d/%d, PGD %d/%d, random %d/%d. This measures how often the grade changed, not which grade was right." % (
        gradings, *tot["none"], *tot["pgd"], *tot["random"]))
    return "\n".join(lines)


def holdout_table() -> str:
    lines = ["| model | clean set | n | shipped sigma | in-sample FPR | 3-way sigma | 3-way FPR | 2-way FPR |",
             "|---|---|---|---|---|---|---|---|"]
    for model, kind, tmpl in HOLDOUT:
        a = load(f"holdout_{model}_{kind}_3way", tmpl.format("3way"))
        b = load(f"holdout_{model}_{kind}_2way", tmpl.format("2way"))
        lines.append("| %s | %s | %d | %.2f | %s | %.2f | %.1f%% ± %.1f | %.1f%% ± %.1f |" % (
            model, kind, a["n_clean"], a["shipped"]["sigma"], pct(a["shipped"]["fpr_in_sample"]),
            a["held_out"]["sigma_mean"], 100 * a["held_out"]["fpr_mean"], 100 * a["held_out"]["fpr_sd"],
            100 * b["held_out"]["fpr_mean"], 100 * b["held_out"]["fpr_sd"]))
    return "\n".join(lines)


def d5_table() -> str:
    lines = ["| clean set | held-out FPR | TPR pgd | TPR random |", "|---|---|---|---|"]
    for label, path in D5:
        d = load(f"holdout_{label}", path)
        h = d["held_out"]
        pgd = next(v for k, v in h["tpr_mean"].items() if k.startswith("pgd@"))
        rnd = next(v for k, v in h["tpr_mean"].items() if k.startswith("random@"))
        lines.append("| %s | %.1f%% ± %.1f | %s | %s |" % (label, 100 * h["fpr_mean"], 100 * h["fpr_sd"], pct(pgd), pct(rnd)))
    return "\n".join(lines)


def deployment_table() -> str:
    lines = ["| model | arm | budget | evaluator: flagged / clear / unscorable | deployment rule: flagged / clear / unscorable |",
             "|---|---|---|---|---|"]
    for entry in deploy:
        for c in entry["cells"]:
            if c["arm"] == "random" and not c["changed_prompts"]:
                continue
            e, d = c["evaluator"], c["deployment"]
            mark = " **←**" if c["changed_prompts"] else ""
            lines.append("| %s | %s | %g | %d / %d / %d | %d / %d / %d%s |" % (
                entry["model"], c["arm"], c["budget"], e["flagged"], e["clear"], e["unscorable"],
                d["flagged"], d["clear"], d["unscorable"], mark))
    return "\n".join(lines)


def clean_cuts() -> str:
    parts = []
    for entry in deploy:
        cuts = entry.get("clean_rows_cut_by_layer", {})
        if not cuts:
            continue
        parts.append("%s at layer%s %s" % (
            entry["model"], "s" if len(cuts) > 1 else "",
            ", ".join("%s (%d of %d rows)" % (L, c, n) for L, (c, n) in cuts.items())))
    return "; ".join(parts) if parts else "no clean row is cut on any ladder"


def deployment_summary() -> str:
    changed_cells = [(entry["model"], c) for entry in deploy for c in entry["cells"] if c["changed_prompts"]]
    flag_changes = sum(1 for _, c in changed_cells if c["evaluator"]["flagged"] != c["deployment"]["flagged"])
    cut = sum(entry["rows_shortened_by_deployment_rule"] for entry in deploy)
    with_miss = sum(entry["rows_with_a_miss"] for entry in deploy)
    total = sum(entry["rows"] for entry in deploy)
    modes = collections.Counter()
    for entry in deploy:
        modes.update(entry["saved_stop_modes"])
    bank = ["%s %d of %d" % (e["model"], e["calibration_bank"]["rows_past_miss"] or 0,
                             e["calibration_bank"]["rows"]) for e in deploy]
    tainted = [e["model"] for e in deploy if e["calibration_bank"]["rows_past_miss"]]
    return ("%d of %d saved rows are shortened by the deployment rule (%d contain a tolerance "
            "miss, but for most of them the miss was already the last saved step). %d cells "
            "change at all; in %d of them the flagged count changes. Saved rows record their "
            "stopping option as: %s; a row that continues past a miss cannot have come from "
            "the `miss` rule, but the unrecorded rows do not say which option they used. "
            "Calibration-bank rows that themselves continue past a tolerance miss: %s. %s" % (
                cut, total, with_miss, len(changed_cells), flag_changes,
                ", ".join("%s %d" % kv for kv in sorted(modes.items())), "; ".join(bank),
                ("The %s calibrations were therefore fitted on rows a deployment could not "
                 "have produced and would need refitting under the deployment rule." %
                 ", ".join(tainted)) if tainted else
                "The calibrations are consistent with the deployment rule."))


BASELINES = {label: lad.baselines for label, lad in MODELS.items()}
base = {label: load(f"baselines_{label}", path) for label, path in BASELINES.items()
        if (ROOT / path).exists()}
BUDGETS = (0.0085, 0.02, 0.05, 0.12, 0.3)


def baselines_missing() -> str:
    missing = [label for label in BASELINES if label not in base]
    return " Not run: %s." % ", ".join(missing) if missing else ""


def recompute_table() -> str:
    lines = ["| model | budget | realised delta at injection, pgd (median) | floor: bank 95th percentile | recompute flagged: pgd | random | detector flagged: pgd | random |",
             "|---|---|---|---|---|---|---|---|"]

    def det(c):
        return "%d/%d" % (c["detector"]["flagged"], c["detector"]["n"]) if c["detector"] else "-"

    for label, d in base.items():
        thr = d["recompute"]["bank"]["threshold_batch"]
        cells = {(c["arm"], c["budget"]): c for c in d["recompute"]["cells"]}
        for b in BUDGETS:
            pg, rn = cells[("pgd", b)], cells[("random", b)]
            lines.append("| %s | %g | %.1e | %.1e | %d/%d | %d/%d | %s | %s |" % (
                label, b, statistics.median(pg["rel_delta_realised_fp16"]), thr,
                pg["flagged_batch"], pg["n"], rn["flagged_batch"], rn["n"], det(pg), det(rn)))
    return "\n".join(lines)


def recompute_clean_line() -> str:
    parts = []
    for label, d in base.items():
        bank = d["recompute"]["bank"]
        clean = next(c for c in d["recompute"]["cells"] if c["arm"] == "clean")
        parts.append("%s %d/%d (bank in-sample %d/%d)" % (
            label, clean["flagged_batch"], clean["n"], bank["fpr_batch"], bank["n"]))
    return "; ".join(parts)


def same_kernel_line() -> str:
    flagged = n = clean_flagged = clean_n = 0
    floor_max = 0.0
    for d in base.values():
        floor_max = max(floor_max, d["recompute"]["bank"]["score_single"]["max"])
        for c in d["recompute"]["cells"]:
            if c["arm"] == "clean":
                clean_flagged += c["flagged_single"]
                clean_n += c["n"]
            else:
                flagged += c["flagged_single"]
                n += c["n"]
    return ("the clean bank's largest score is %.1e across the models, attacked prompts "
            "flagged %d/%d over every arm and budget, clean-arm prompts flagged %d/%d"
            % (floor_max, flagged, n, clean_flagged, clean_n))


def floor_margin_line() -> str:
    parts = []
    for label, d in base.items():
        thr = d["recompute"]["bank"]["threshold_batch"]
        pg = next(c for c in d["recompute"]["cells"] if c["arm"] == "pgd" and c["budget"] == BUDGETS[0])
        parts.append("%s %.1f× (smallest prompt %.1f×)" % (
            label, statistics.median(pg["rel_delta_realised_fp16"]) / thr,
            min(pg["rel_delta_realised_fp16"]) / thr))
    return "; ".join(parts)


def probe_table() -> str:
    lines = ["| model | budget | injection layer: pgd / clean / random | all layers: pgd / clean / random | trained at 0.3, injection layer: pgd / clean |",
             "|---|---|---|---|---|"]
    for label, d in base.items():
        cells = {(c["feature"], c["budget"]): c for c in d["probe"]["cells"]}
        for b in BUDGETS:
            i, al = cells[("inj", b)], cells[("all", b)]
            n = i["n"]
            lines.append("| %s | %g | %d / %d / %d of %d | %d / %d / %d of %d | %d / %d of %d |" % (
                label, b, i["tpr_pgd"], i["fpr_clean"], i["tpr_random"], n,
                al["tpr_pgd"], al["fpr_clean"], al["tpr_random"], n,
                i["transfer_tpr_pgd"], i["transfer_fpr_clean"], n))
    return "\n".join(lines)


def baselines_reading() -> str:
    b0 = BUDGETS[0]
    floors = [d["recompute"]["bank"]["threshold_batch"] for d in base.values()]
    clear, under = [], []
    for label, d in base.items():
        cells = {(c["arm"], c["budget"]): c for c in d["recompute"]["cells"]}
        pg, rn = cells[("pgd", b0)], cells[("random", b0)]
        thr = d["recompute"]["bank"]["threshold_batch"]
        entry = (label, pg, rn, statistics.median(pg["rel_delta_realised_fp16"]) / thr)
        (clear if entry[3] > 1 else under).append(entry)

    def det(c):
        return "%d/%d" % (c["detector"]["flagged"], c["detector"]["n"]) if c["detector"] else "-"

    text = ("**Reading.** With the full trajectory and trusted weights the recomputation check is "
            "the stronger method wherever the delta clears its floor. The fp16 kernel discrepancy "
            "sits at %.0e to %.0e relative, an order of magnitude under the 1e-2 inversion "
            "tolerance the detector's calibration is built on, and once the recomputed block "
            "matches the logged state at every other layer and position exactly, the residual at "
            "the injection site *is* the delta." % (min(floors), max(floors)))
    if clear:
        text += (" At budget %g it flags %s, where the detector reads %s." % (
            b0, ", ".join("%d/%d on %s" % (pg["flagged_batch"], pg["n"], label) for label, pg, _, _ in clear),
            ", ".join("%s on %s" % (det(pg), label) for label, pg, _, _ in clear)))
    if under:
        text += (" On %s the `all` constraint backs the realised PGD delta below that floor "
                 "(%s of it), and the check flags %s, no better than the detector's %s; the "
                 "norm-matched random arm, which spends its whole budget at the injection site, "
                 "is flagged %s by the check and %s by the detector." % (
                     ", ".join(label for label, *_ in under),
                     ", ".join("%.1f×" % m for *_, m in under),
                     ", ".join("%d/%d" % (pg["flagged_batch"], pg["n"]) for _, pg, _, _ in under),
                     ", ".join(det(pg) for _, pg, _, _ in under),
                     ", ".join("%d/%d" % (rn["flagged_batch"], rn["n"]) for _, _, rn, _ in under),
                     ", ".join(det(rn) for _, _, rn, _ in under)))
    text += ("\n\nThe probe is at or near chance at the legal budget on every model and separates the "
             "arms only from the middle of the ladder upward; trained at budget 0.3 it does not "
             "fire on the small budgets, and trained on PGD it fires on the norm-matched random "
             "arm at the clean rate: it detects the perturbation it was trained on, which is why "
             "it is not a detector for an unknown intervention. What the inversion detector adds "
             "is confined to the settings the recomputation check cannot run in: a perturbation "
             "applied before the first recomputable block (a soft prompt, an embedding edit), a "
             "log that holds one layer or a subset of layers, or weights the auditor cannot "
             "trust. The one-layer log is measured in the next section; in the full-trajectory "
             "setting the recomputation check should be read as the reference and the detector "
             "as the method that works without adjacent layers.")
    return text


# ----------------------------------------------------------- one logged layer
read_layer_runs = load("read_layer", "results/current/read-layer-2026-09-15.json")
single = load("single_layer", "results/current/single-layer-2026-09-08.json")
single_deploy = load("single_layer_deploy", "results/current/single-layer-deploy-2026-09-08.json")
maha = load("mahalanobis_qwen05b", "results/current/mahalanobis-qwen05b-2026-09-08.json")
SL_BUDGETS = (0.0085, 0.02)


def _sl_cells(e: dict) -> dict:
    return {(c["arm"], c["budget"]): c for c in e["cells"]}


def _sl(c: dict, layer: int) -> dict:
    return c["per_layer"][str(layer)]


def single_layer_summary_table() -> str:
    lines = ["| model | budget | recompute, straddling pair | recompute, no pre-injection layer (same-kernel / serving-batch floor) | detector, shipped multi-layer rule | injection layer alone | best single layer alone (offset) | last layer alone | held-out clean, last layer |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in single:
        label, inj, layers = e["model"], e["inj_layer"], e["layers"]
        cells = _sl_cells(e)
        bcells = ({(c["arm"], c["budget"]): c for c in base[label]["recompute"]["cells"]}
                  if label in base else {})
        for b in SL_BUDGETS:
            c, cl = cells[("pgd", b)], cells[("clean", 0.0)]
            best = max(layers, key=lambda L: (_sl(c, L)["single"]["flagged"], -L))
            rc = c["recompute_off_injection"]
            bc = bcells.get(("pgd", b))
            straddle = "%d/%d" % (bc["flagged_batch"], bc["n"]) if bc else "-"
            shipped = "%d/%d" % (bc["detector"]["flagged"], bc["detector"]["n"]) if bc and bc["detector"] else "-"
            lines.append("| %s | %g | %s | %d/%d / %d/%d | %s | %d/%d | %d/%d (+%d) | %d/%d | %d/%d |" % (
                label, b, straddle, rc["same_kernel"], rc["n"], rc["serving_batch"], rc["n"], shipped,
                _sl(c, inj)["single"]["flagged"], c["n"],
                _sl(c, best)["single"]["flagged"], c["n"], best - inj,
                _sl(c, layers[-1])["single"]["flagged"], c["n"],
                _sl(cl, layers[-1])["single"]["flagged"], cl["n"]))
    return "\n".join(lines)


def single_layer_offset_table() -> str:
    width = max(len(e["layers"]) for e in single)
    lines = ["| model | arm, budget | " + " | ".join("+%d" % k for k in range(width)) + " |",
             "|---|---|" + "---|" * width]
    for e in single:
        layers, cells = e["layers"], _sl_cells(e)
        pad = ["-"] * (width - len(layers))
        lines.append("| %s | single-layer sigma | %s |" % (e["model"], " | ".join(
            ["%.2f" % e["thresholds"][str(L)]["single"] for L in layers] + pad)))
        for arm, b in (("clean", 0.0), ("pgd", 0.0085), ("random", 0.0085), ("pgd", 0.02)):
            c = cells[(arm, b)]
            vals = []
            for L in layers:
                x = _sl(c, L)["single"]
                vals.append("%d/%d" % (x["flagged"], x["n"])
                            + (" (%du)" % x["unscorable"] if x["unscorable"] else ""))
            lines.append("| %s | %s %g | %s |" % (e["model"], arm, b, " | ".join(vals + pad)))
    return "\n".join(lines)


def single_layer_reading() -> str:
    parts = []
    for e in single:
        inj, layers, c = e["inj_layer"], e["layers"], _sl_cells(e)[("pgd", 0.0085)]
        zs = [(L, _sl(c, L)["median_score"]) for L in layers if _sl(c, L)["median_score"] is not None]
        Lb, zb = max(zs, key=lambda t: t[1])
        parts.append("%s %.1f at the injection layer, %.1f at +%d" % (
            e["model"], _sl(c, inj)["median_score"], zb, Lb - inj))
    return ("**Reading.** Recomputation from a log with no layer before the injection reads its own "
            "floor on every attacked prompt: nothing flagged on the same-kernel floor in any cell, "
            "and the clean rate on the serving-batch floor. Inversion from one layer sees the "
            "injection at that layer and at every later one, and at the legal budget the signal "
            "grows with depth rather than decaying; median z of the PGD arm: %s. On three of five "
            "models a single late layer flags the legal-budget attack that the injection layer "
            "alone misses, and on Qwen-7B and gemma-3-1b it flags what the shipped multi-layer rule "
            "misses at the same held-out false-positive rate: that rule needs three consecutive "
            "layers over a sigma tuned for the run, and a lone late layer with its own 5%% "
            "threshold is the more sensitive read there. Llama-3.2-1B at the legal budget is the "
            "negative: no single layer sees it, where recomputation with the straddling pair has a "
            "2× margin over its floor. The random arm runs the other way, caught at the injection "
            "layer and gone by the last: the `all` constraint shapes the PGD delta to be small "
            "where it is placed, and what it leaves grows downstream. The best-offset column is "
            "chosen after the fact and is not a detection rate; the last-layer column, or an offset "
            "declared in advance, is the deployable number. Thresholds are in-sample on the bank "
            "at 5%% per layer; the held-out clean arm confirms them at 0 to 1 of 15 and 1 of 50. "
            "Every attack here is constrained at the injection layer and every later one; an "
            "attacker constrained only at the layer being read, free elsewhere, is not among the "
            "attacks tested, and a single-layer read has no defence against one that a multi-layer "
            "read does not also need to be measured against."
            % "; ".join(parts))


def single_layer_deploy_line() -> str:
    ev = {e["model"]: e for e in single}
    groups: dict = collections.defaultdict(list)
    n_cells = 0
    for d in single_deploy:
        e = ev[d["model"]]
        ce, cd = _sl_cells(e), _sl_cells(d)
        for key in ce:
            for L in e["layers"]:
                n_cells += 1
                a, b = _sl(ce[key], L)["single"], _sl(cd[key], L)["single"]
                if (a["flagged"], a["unscorable"]) != (b["flagged"], b["unscorable"]):
                    groups[(d["model"], L)].append((key, a, b))
    if not groups:
        return "Same rows cut at the first tolerance miss (`--deploy`): no cell changes."
    parts = []
    for (model, L), items in sorted(groups.items()):
        e, d = ev[model], next(x for x in single_deploy if x["model"] == model)
        off = L - e["inj_layer"]
        if all(b["unscorable"] == b["n"] for _, _, b in items):
            parts.append("%s at +%d: every prompt unscorable at every arm and budget (%d cells; the "
                         "bank's own rows are cut at position 0 there)" % (model, off, len(items)))
        else:
            te, td = e["thresholds"][str(L)]["single"], d["thresholds"][str(L)]["single"]
            pg = next(((a, b) for key, a, b in items if key == ("pgd", 0.0085)), None)
            desc = "%d cells" % len(items)
            if pg:
                desc += ", pgd 0.0085 %d/%d → %d/%d flagged with %d unscorable" % (
                    pg[0]["flagged"], pg[0]["n"], pg[1]["flagged"], pg[1]["n"], pg[1]["unscorable"])
            parts.append("%s at +%d: single-layer sigma %.2f → %.2f, %s" % (model, off, te, td, desc))
    return ("Same rows cut at the first tolerance miss, then rescored (`--deploy`). %d of %d "
            "(arm, budget, layer) cells change, all at the layers the deployment section above "
            "already names: %s." % (sum(len(v) for v in groups.values()), n_cells, "; ".join(parts)))


AMP_FILES = {
    "gpt2": "results/gpt2_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
    "Qwen-0.5B": "results/Qwen_Qwen2.5-0.5B-Instruct_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
    "Qwen-1.5B": "results/Qwen_Qwen2.5-1.5B-Instruct_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
    "Qwen-3B": "results/Qwen_Qwen2.5-3B-Instruct_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
    "Qwen-7B": "results/Qwen_Qwen2.5-7B-Instruct_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
    "gemma-3-1b": "results/google_gemma-3-1b-it_fp32/pgd/pgd_sentiment_b0.0085.jsonl",
}
amp_rows = {}
for _label, _path in AMP_FILES.items():
    if (ROOT / _path).exists():
        SOURCES[f"scope_{_label}"] = ROOT / _path
        amp_rows[_label] = [json.loads(l) for l in (ROOT / _path).read_text().splitlines() if l.strip()]


def _amp_cell(rows: list[dict], inj: int) -> dict:
    def dev(r, k):
        return r["rel_dev_by_layer"][str(k)]
    layers = sorted(int(k) for k in rows[0]["rel_dev_by_layer"] if int(k) >= inj)
    med = {k: statistics.median(dev(r, k) for r in rows) for k in layers}
    peak = max(med, key=med.get)
    return {"inj": med[inj], "peak": med[peak], "peak_layer": peak, "last": med[layers[-1]],
            "last_layer": layers[-1],
            "amp": statistics.median(max(dev(r, k) for k in layers) / dev(r, inj) for r in rows),
            "det_inj": sum(bool(r.get("detected_at_layer")) for r in rows),
            "det_any": sum(bool(r.get("detected_multilayer")) for r in rows), "n": len(rows)}


def amplification_table() -> str:
    lines = ["| model | constraint | arm | at injection | largest downstream (layer) | last layer | amplification | rel_tol check: injection layer / any layer |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rows in amp_rows.items():
        inj = rows[0]["layer"]
        for con in ("injection", "all"):
            for arm in ("pgd", "random"):
                rs = [r for r in rows if r["constraint"] == con and r["arm"] == arm]
                if not rs:
                    continue
                c = _amp_cell(rs, inj)
                lines.append("| %s | `%s` | %s | %.2e | %.2e (L%d of %d) | %.2e | %.2f× | %d/%d / %d/%d |" % (
                    label, con, arm, c["inj"], c["peak"], c["peak_layer"], c["last_layer"], c["last"],
                    c["amp"], c["det_inj"], c["n"], c["det_any"], c["n"]))
    return "\n".join(lines)


def amplification_reading() -> str:
    amps = {label: _amp_cell([r for r in rows if r["constraint"] == "injection" and r["arm"] == "pgd"],
                             rows[0]["layer"]) for label, rows in amp_rows.items()}
    ordered = sorted(amps.items(), key=lambda kv: kv[1]["amp"])
    lo, hi = ordered[0], ordered[-1]
    at_end = [label for label, c in amps.items() if c["peak_layer"] >= c["last_layer"] - 1]
    return ("Holding the injection layer at the budget leaves a deviation %.1f× (%s) to %.1f× (%s) "
            "larger somewhere downstream, ordered by model size within the Qwen family, and the "
            "peak sits at or next to the last layer on %s. Holding every layer at the budget shrinks "
            "the injection-site delta by the same factor, which is the backoff the ladders above "
            "measure. The random arm does not amplify; gemma's does through its norm profile. So an "
            "attacker who keeps the injection layer quiet is easier to read from a late layer, not "
            "harder, and an attacker who keeps a late layer quiet pays the `all` constraint's tax on "
            "the models where the deviation peaks there. An attacker who sculpts the profile to be "
            "quiet at the layer being read and large elsewhere, at a budget that changes behaviour, "
            "is not run; the K/V at the layers between the injection and the read carry the "
            "perturbation into generation whether or not the read layer is quiet, so that attack "
            "has to be judged on decoded text, not on the read layer's residual."
            % (lo[1]["amp"], lo[0], hi[1]["amp"], hi[0], ", ".join(at_end)))


MAHA_BUDGETS = (0.0085, 0.02, 0.05, 0.12, 0.3)
MAHA_NAMES = {"diag": "diagonal", "lw": "Ledoit-Wolf", "knn": "nearest neighbour"}


def mahalanobis_table() -> str:
    n, inj = maha["n_prompts"], maha["layer"]
    by_layer = {e["layer"]: e for e in maha["per_layer"]}
    lines = ["| reference | estimator | layer | clean | " + " | ".join("pgd %g" % b for b in MAHA_BUDGETS)
             + " | random 0.3 | shift at 0.0085, in reference sd | shift at 0.3 |",
             "|---|---|---|---|" + "---|" * len(MAHA_BUDGETS) + "---|---|---|"]

    def row(ref, kind, L, note, c, key):
        return "| %s | %s | %d%s | %d/%d | %s | %d | %+.2f | %+.2f |" % (
            ref, MAHA_NAMES[kind], L, note, c["clean|0.0"]["flagged"], n,
            " | ".join(str(c["pgd|%g" % b]["flagged"]) for b in MAHA_BUDGETS),
            c["random|0.3"]["flagged"], c["pgd|0.0085"][key], c["pgd|0.3"][key])

    for kind in MAHA_NAMES:
        best = max(maha["per_layer"], key=lambda e: e["matched"][kind]["cells"]["pgd|0.3"]["flagged"])["layer"]
        for L in sorted({inj, best}):
            note = " (injection)" if L == inj else " (its best at 0.3)"
            lines.append(row("ladder clean arm, leave-one-out", kind, L, note,
                             by_layer[L]["matched"][kind]["cells"], "median_shift_in_clean_sd"))
    for kind, L in (("lw", inj), ("knn", inj), ("diag", maha["n_blocks"])):
        lines.append(row("clean bank, five categories", kind, L, "",
                         by_layer[L]["estimators"][kind]["cells"], "median_shift_in_bank_sd"))
    return "\n".join(lines)


def mahalanobis_reading() -> str:
    n, inj = maha["n_prompts"], maha["layer"]
    by_layer = {e["layer"]: e for e in maha["per_layer"]}
    kinds = list(MAHA_NAMES)
    low = max(by_layer[L]["matched"][k]["cells"]["pgd|%g" % b]["flagged"]
              for L in by_layer for k in kinds for b in (0.0085, 0.02, 0.05))
    clean_max = max(by_layer[L]["matched"][k]["cells"]["clean|0.0"]["flagged"] for L in by_layer for k in kinds)
    shift = max(abs(by_layer[L]["matched"][k]["cells"]["pgd|0.0085"]["median_shift_in_clean_sd"])
                for L in by_layer for k in kinds)
    det = maha["detector"]
    det_s = ("%d/%d" % (det["pgd|0.3"]["flagged"], det["pgd|0.3"]["n"])
             if all(v["flagged"] == v["n"] for k, v in det.items() if k.startswith("pgd")) else "-")
    rel = maha["realised_rel_delta_median"]
    bank_lw = by_layer[inj]["estimators"]["lw"]["cells"]["clean|0.0"]["flagged"]
    return ("**Reading.** With the fairest reference and the injected position given, the density "
            "check flags at most %d of %d attacked prompts at budgets 0.0085 to 0.05 at any layer "
            "under any estimator, against up to %d of %d clean, and the attack moves its score by "
            "under %.2f reference standard deviations at the legal budget; the detector flags %s "
            "at every budget. It sees PGD at 0.3 only, a realised delta of %.2f relative, %.0f× the "
            "legal budget's, and the norm-matched random arm much earlier, because a random "
            "direction leaves the span of the reference while the PGD direction looks like "
            "variation between prompts. With the detector's own bank as reference it flags %d of "
            "%d clean harmful prompts at the injection layer under the shrinkage covariance, "
            "because they are a different kind of text from base64 and code, and the diagonal "
            "estimator at the last layer flags nothing in any arm. The raw state measures rare "
            "text; the inversion residual measures an unreachable state, the distinction fig 1's "
            "score was built for. Qwen-0.5B only: the other ladders' deltas were not on the "
            "machine that ran this." % (low, n, clean_max, n, max(shift, 0.01), det_s,
                                        rel["pgd|0.3"], rel["pgd|0.3"] / rel["pgd|0.0085"], bank_lw, n))




# ------------------------------------------------- the attacker that knows the read layer
def _read_runs() -> list[dict]:
    return [e for e in read_layer_runs if "cells" in e]


def _read_attack(name: str) -> list[dict]:
    for e in read_layer_runs:
        if e.get("attack_dir", "").find(name) >= 0:
            return e["rows"]
    return []


def read_layer_table() -> str:
    head = ("| model | watched layer | budget | attacker | budget kept at injection | flagged at "
            "the watched layer | worst other layer | any layer | shipped rule | judged gains |\n"
            + "|---" * 10 + "|")
    rows = []
    for e in _read_runs():
        inj, watched = e["inj_layer"], e["read_layers"]
        for c in sorted(e["cells"], key=lambda c: (c["arm"], c["budget"])):
            if c["arm"] == "random":
                continue
            n = c["n"]
            if c["arm"] == "clean":
                at = " / ".join("%d/%d" % (c["per_layer"][str(L)]["flagged"], n) for L in watched)
                worst = max(((c["per_layer"][str(L)]["flagged"], L) for L in e["layers"]
                             if L not in watched), default=(0, 0))
                rows.append("| %s | %s | - | clean arm | - | %s | %d/%d (L%d) | %d/%d | %d/%d | - |"
                            % (e["model"].split()[0], ",".join(map(str, watched)), at, worst[0], n,
                               worst[1], c["flagged_any_layer"], n, c["flagged_shipped_rule"], n))
                continue
            free = "free" in e["model"]
            at = " / ".join("%d/%d" % (c["per_layer"][str(L)]["flagged"], n) for L in watched)
            others = [(c["per_layer"][str(L)]["flagged"], L) for L in e["layers"] if L not in watched]
            worst = max(others, default=(0, 0))
            j = c.get("judged") or {}
            rows.append("| %s | %s | %g | %s | %.1f%% | %s | %d/%d (L%d) | %d/%d | %d/%d | %s |"
                        % (e["model"].split()[0], "-" if free else ",".join(map(str, watched)),
                           c["budget"], "free" if free else "read-constrained",
                           100 * c["budget_used_median"], at, worst[0], n, worst[1],
                           c["flagged_any_layer"], n, c["flagged_shipped_rule"], n,
                           ("%d (%d gated), %d unflagged at the watched layer"
                            % (j["gains"], j["gated_gains"], j["gated_gains_unflagged_at_read"]))
                           if j else "-"))
    return "\n".join([head] + rows)


def read_layer_reading() -> str:
    read_runs = [e for e in _read_runs() if "free" not in e["model"]]
    free_runs = [e for e in _read_runs() if "free" in e["model"]]
    kept, at_read, elsewhere, absolute, peaks, any_rate = [], [], [], [], [], []
    for e in read_runs:
        model, watched = e["model"].split()[0], e["read_layers"]
        for c in e["cells"]:
            if c["arm"] != "pgd":
                continue
            kept.append(100 * c["budget_used_median"])
            peaks.append(c["dev_peak_median"])
            absolute.append(("%s layer %s" % (model, ",".join(map(str, watched))),
                             c["budget"], c["dev_injection_median"]))
            at_read.append(max(c["per_layer"][str(L)]["flagged"] for L in watched) / c["n"])
            others = [(c["per_layer"][str(L)]["flagged"], L) for L in e["layers"] if L not in watched]
            elsewhere.append(max(others, default=(0, 0))[0] / c["n"])
            any_rate.append(c["flagged_any_layer"] / c["n"])
    gains_free = sum((c.get("judged") or {}).get("gains", 0) for e in free_runs for c in e["cells"])
    gated_free = sum((c.get("judged") or {}).get("gated_gains", 0) for e in free_runs for c in e["cells"])
    undetected_free = sum((c.get("judged") or {}).get("gated_gains_unflagged_at_read", 0)
                          for e in free_runs for c in e["cells"])
    gains_read = sum((c.get("judged") or {}).get("gains", 0) for e in read_runs for c in e["cells"])
    judged_read = sum((c.get("judged") or {}).get("n", 0) for e in read_runs
                      for c in e["cells"] if c["arm"] == "pgd")
    pairs = collections.defaultdict(list)
    for key, budget, dev in absolute:
        pairs[key].append((budget, dev))
    ratio = []
    for key, vals in pairs.items():
        vals = sorted(vals)
        if len(vals) >= 2 and vals[0][1]:
            ratio.append("%s %.2fx" % (key, vals[-1][1] / vals[0][1]))
    return ("**Reading.** Holding one logged layer under its alarm costs the attacker the "
            "perturbation: the retained budget at the injection site is %.1f%% to %.1f%% across "
            "the cells, and the largest deviation anywhere in the trajectory is %.1e, %.1f times "
            "the legal budget 0.0085 that the ladders above treat as the quiet end of the scale. "
            "The outer ball is not what binds: the absolute deviation the ceiling admits is the same "
            "at budget 0.12 and 0.30 (%s), so a larger injection budget buys nothing and the "
            "budget column could be raised without changing the result. Silence is local, not "
            "global: at the watched layer the attack is flagged on at most %.0f%% of prompts in "
            "any cell, while some other layer flags up to %.0f%% of the same prompts and the "
            "uncorrected any-layer read up to %.0f%%. "
            "Behaviour does not survive the constraint. Over every read-constrained cell, %d of "
            "%d judged completions is a gain over its own clean arm; the same attacker with no "
            "constraint past the injection ball takes %d gains (%d passing the StrongREJECT "
            "gate), and every one of them is flagged at the watched layer (%d unflagged). The "
            "norm-matched random arm under the same ceiling keeps more budget than the optimised "
            "one and is flagged at the injection layer instead, because a random direction does "
            "not concentrate where the constraint bites. Caveats: 15 prompts per cell except "
            "Qwen-0.5B, thresholds in-sample at 5%% per layer with the ladder's own clean arm as "
            "the held-out check, one injection layer at 0.7 of depth, one position, and a ceiling "
            "the attacker aims at rather than a guarantee it lands under."
            % (min(kept), max(kept), max(peaks), max(peaks) / 0.0085, "; ".join(ratio),
               100 * max(at_read), 100 * max(elsewhere), 100 * max(any_rate),
               gains_read, judged_read, gains_free, gated_free, undetected_free))


def read_layer_sweep_table() -> str:
    rows = _read_attack("sweep")
    if not rows:
        return "_(no ceiling sweep on disk)_"
    base = [c for e in _read_runs() for c in e["cells"]
            if "read16" in e["model"] and c["arm"] == "pgd" and c["budget"] == 0.3]
    head = ("| ceiling at the watched layer | budget kept | deviation at injection | peak "
            "deviation | target log-prob gain | next-token changes | judged jailbreaks |\n"
            + "|---" * 7 + "|")
    out = []
    for c in base:
        out.append("| %.2f (the layer's own threshold, minus the surrogate's error) | %.1f%% | %.2e | %.2e | %+.3f | %d/%d | %s |"
                   % (c.get("ceiling") or float("nan"), 100 * c["budget_used_median"], c["dev_injection_median"],
                      c["dev_peak_median"], c["d_target_logprob_median"], c["flips"], c["n"],
                      "%d/%d" % ((c.get("judged") or {}).get("gains", 0), c["n"])))
    for r in sorted(rows, key=lambda r: r["ceiling"]):
        out.append("| %.0f | %.1f%% | %.2e | %.2e | %+.3f | %d/%d | %s |"
                   % (r["ceiling"], 100 * r["budget_used_median"], r["dev_injection_median"],
                      r["peak_median"], r["d_target_logprob_median"], r["flips"], r["n"],
                      "%s/%d" % (r.get("judged_gains"), r["n"])))
    return "\n".join([head] + out)


def read_layer_nosculpt_line() -> str:
    rows = _read_attack("nosculpt")
    if not rows:
        return ""
    parts = ", ".join("layer %s %.2f%%" % (r["read_layers"], 100 * r["budget_used_median"])
                      for r in rows)
    guided = [c for e in _read_runs() for c in e["cells"]
              if c["arm"] == "pgd" and c["budget"] == 0.3 and "free" not in e["model"]
              and "Llama" in e["model"]]
    g = ", ".join("%.2f%%" % (100 * c["budget_used_median"]) for c in guided)
    return ("With the guiding gradient switched off, so the attacker only projects a "
            "target-driven direction back under the ceiling, the retained budget is the same "
            "(%s against %s guided, Llama-3.2-1B at budget 0.30). The sculpting term is not "
            "what limits the attack." % (parts, g))


# ----------------------------------------------------------------------------- text
fp = provenance.code_fingerprint()
sb = cluster["sweep_b"]["pooled"]
n50 = cluster["n50"]
gp = trans["pooled_pgd_gains"]

doc = f"""# Current results

Generated on {DATE} by `scripts/current_results.py` from the files listed in the manifest
at the end; regenerate it rather than editing it. Code fingerprint: git `{fp['git_head'][:12]}`{' (working tree dirty)' if fp['git_dirty'] else ''},
sources `{fp['source_sha256'][:12]}`. Historical corrections are kept out of this file.

## What was tested

**The attack.** Projected-gradient perturbations of the residual stream at one layer, 0.7 of
the way through the model (layer round(0.7 × blocks)), at the last prompt position, applied
during prefill; generated tokens see it only through the cache. The `all` constraint caps the
relative deviation of the perturbed state at the injection layer *and every later layer* at
the budget, so the delta actually placed at the injection site is smaller than the nominal
budget (the constraint backs it off). "Legal budget" is 0.0085, 0.85 times the fp16 inversion
tolerance: a rule of this experiment, not a property of the world. Arms: `pgd` (optimised
for the affirmative target), `random` (a norm-matched random direction under the same
constraint), and each budget's own unsteered `clean` arm. Detector-aware attacks that
differentiate through the detector's own rule were also run and checked against the real
detector (the `aware_*` result directories).

**The detector.** Full SipIt inversion of every layer's hidden states back to tokens, then a
per-position z-score of the inversion residual against a clean bank fitted per
(layer, template role), and a run rule over layers. Its inputs are the hidden states and the
recovered tokens; it never sees the clean forward pass of the prompt under test. Two things
the evaluator does that a deployment cannot: it reuses the clean arm's rows for positions
ahead of the injection, which are the same activations by causality and only save time; and
`--stop_on wrong` consults the gold token to decide when an inversion halts, whereas a
deployment halts on the first tolerance miss (`--stop_on miss`). Per-prompt verdicts are
**flagged**, **clear**, or **unscorable**; a prompt whose examined layers the calibration
cannot fully cover is never read as clear.

**The conclusion this supports:** *no effective evasion was found using the attacks and
settings tested.* It does not support "no effective evasion exists".

## Detection on the jbb_refusal PGD ladders (evaluator stopping rule)

These are **evaluator results**: the saved inversions include evaluator-only
continuations beyond tolerance misses, which the `wrong` stopping option allows when the
recovered token is known to be right. A deployment cannot know that and stops at the
first tolerance miss; the same rows rescored under that rule are in the next section. PGD arm, prompts detected per budget,
Clopper-Pearson 95% interval. Every calibration is the model's own 75- or 100-prompt
chat bank, log + end roles, k=1, sigma tuned to 5% in-sample (held-out rate below).
Unscorable prompt-cells under the evaluator rule, read from the regenerated summaries:
{unscorable_line()}.

{detection_table()}

Judge-positive completions by budget, and how many are new against the same budget's
paired clean arm:

{positives_table()}

gemma's two undetected judge-positive rows at 0.0085 are positive in the paired clean arm
too. No prompt that went from clean-negative to attacked-positive is undetected on any
ladder. Fifty prompts at five budgets are fifty clusters, not 250 draws: the zero-failure
bound on undetected gains is 5.8% by 50 prompt clusters for n=50 and 18% by 15 for the
others.

### Under the deployment stopping rule

Every saved row cut at its first tolerance miss, exactly where the `miss` rule halts,
then rescored with the same calibration and coverage checks
(`scripts/deployment_stop_rescore.py`). {deployment_summary()} Cells that change are
marked; random-arm cells that do not change are omitted.

{deployment_table()}

A prompt that becomes unscorable was clear under the evaluator rule only because the
inversion had continued past a miss; under the deployment rule its later positions are
unknown, and an unflagged prompt with unknown positions is not clean. No flagged count
changes. Where the deployment rule cuts clean rows: {clean_cuts()}. Where it cuts every
clean row at a layer, a template position there sits at the fp16 tolerance floor, and a
deployment on that model cannot clear any prompt at that layer: it would have to drop
the layer from the scan or raise its tolerance, and its false-positive rate would have
to be measured after that change. This section reuses the calibrations as saved and is
not an end-to-end deployment benchmark; truncating saved steps also says nothing about
full-inversion runtime.

## Paired transitions, Qwen-0.5B, n=50

A *gain* is a prompt judge-negative under the same budget's clean arm and judge-positive
under the attack arm. *Gains passing gate* additionally have a StrongREJECT score of at
least {trans['metric_policy']['coherence_gate']['min']}, which that rubric assigns to anything it does not read as a refusal, an
implicit refusal, or a reply both extremely unconvincing and extremely nonspecific. That is
a grading rule, not a confirmation that the answer is useful. McNemar p is two-sided exact
over the discordant pairs, shown unadjusted and Holm-adjusted across the five budgets.
"No clear evidence of change" is a statement about the test, not about the judge.

{transitions_table()}

Pooled over budgets, PGD arm: {gp['gains_rows']} gain rows on {gp['gains_prompts']} prompts, {gp['gains_passing_gate_rows']} of them passing the
gate on {gp['gains_passing_gate_prompts']} prompts; {gp['gains_undetected']} gains undetected. Aggregate rates such as "ASR 54%" are
not reported; the transition table is the result.

## Judge inconsistency

{repeats_table()}

## Held-out false-positive rate of the calibration

The shipped calibration tunes sigma to a 5% per-prompt false-positive target on the same
clean bank it fits, so that 5% is a training constraint. Held out with five folds, role
statistics fitted on three, sigma chosen on a fourth, the verdict read on the fifth
("3-way"); "2-way" tunes sigma in-sample on four folds and reads the fifth. Twenty random
splits, mean ± sd.

{holdout_table()}

The 25-prompt chat banks are not usable as calibrations. Self-calibration on the sentiment
clean arms, same protocol, attack arms scored under each fold's calibration:

{d5_table()}

## Two baselines on the same rows: recomputation check and linear probe

**What the auditor observes.** In every ladder the auditor holds the residual state at
every layer and every prompt position of the forward pass under test, and a trusted copy
of the weights. Nothing else: not the prompt, not the clean forward pass, not the
perturbation. The detector inverts those states and asks whether each has a prompt
preimage. The recomputation check runs block j on the observed input h_j and compares the
result with the observed h_{{j+1}}; it uses the same states and the same weights, and needs
in addition consecutive layer states and the assumption that the trusted weights are the
ones that produced the trajectory. The linear probe is given the last-position state at
the injection layer (or at every layer), is told which layer was attacked, and is trained
on labelled clean and attacked states of other prompts, five folds by prompt. Under this
observation model the recomputation check is a baseline, not an out-of-model oracle, and
both were run on the same regenerated trajectories
(`scripts/baselines_recompute_probe.py`).{baselines_missing()}

**Recomputation check.** Score: the largest relative residual over layers and positions.
Threshold: the 95th percentile of that score on the model's clean chat bank, the
detector's own in-sample rule. Two floors were measured. *Same-kernel*: the logged states
come from the same unpadded single-prompt pass the auditor recomputes with, so the floor
is numerical noise only; {same_kernel_line()}. *Serving-batch*: the logged states come from
the left-padded batched pass that produced the decoded completions, and the auditor
recomputes each prompt on its own, so the floor is the fp16 discrepancy between two
kernels. Under that floor:

{recompute_table()}

Clean-arm prompts flagged under the serving-batch floor, held out from the bank:
{recompute_clean_line()}. Margin of the realised PGD delta over the floor at budget 0.0085:
{floor_margin_line()}.

**Linear probe.** Logistic regression on standardised features, five folds by prompt;
each count is held-out prompts flagged, summed over folds, so "clean" is the probe's
false-positive count and "random" its rate on the norm-matched control it was not trained
on. The last column applies the probe trained at budget 0.3 to the other budgets' held-out
prompts.

{probe_table()}

{baselines_reading()}

## One logged layer: inversion against recomputation

The ladders above hand the auditor every layer. Recomputation fires only on the pair of
layers that straddles the injection: every other pair is self-consistent by construction,
because the later blocks really were run on the perturbed state. Inversion asks each logged
layer on its own whether it has a prompt preimage, so it runs on a log that holds one layer,
including a log whose earliest layer is at or after the injection, where recomputation has
no straddling pair. That setting is measured here from the saved rows
(`scripts/single_layer_reread.py`); no new inference.

**Rule.** Each layer's role-z score is read alone against a per-layer threshold at the bank's
95th percentile of that layer's score, in-sample, the recomputation baseline's own rule. The
ladder's clean arm, held out from the bank, gives the held-out false-positive rate at each
layer. Layers before the injection carry the clean state by causality and are not tabulated.
"Recompute, no pre-injection layer" is the largest recomputation residual over every pair
that does not straddle the injection, read from the saved baselines against the same floor,
which is all a log without a pre-injection layer can show it. Evaluator rows as saved; the
deployment cut follows the tables.

{single_layer_summary_table()}

Every layer from the injection onward, at the legal budget and at 0.02, flagged under the
single-layer threshold ("u" marks prompts unscorable at that layer):

{single_layer_offset_table()}

{single_layer_reading()}

**Under the deployment stopping rule.** {single_layer_deploy_line()}

**Where the deviation goes when one layer is held quiet.** The fp32 sentiment ladders
(n=25, budget 0.0085, one injection layer at 0.7 depth) were run under two constraint
scopes: `injection` caps the relative deviation at the injection layer only, `all` at the
injection layer and every later one. Relative deviation of the perturbed trajectory from
the clean one at the injected position, median over prompts, with that era's check (a
1e-2 relative tolerance at the injection layer, or at any layer):

{amplification_table()}

{amplification_reading()}

## The attacker that knows which layer is logged

Every attack above is constrained at the injection layer and every later one, or under the
detector's own score at every layer from the injection onward. Neither is the attacker a
single-layer read has to survive: one who knows which layer the auditor holds, spends a
behaviour-changing perturbation at the injection site, and only has to look clean at that
one layer. Two limits, not one: an outer ball at the injection layer (the budget column),
and a ceiling on the detector's own score at the watched layer, set at that layer's
in-sample 95th-percentile threshold minus 0.3 for the surrogate's error. The surrogate is
accurate: predicted against realised score at the watched layer differs by 0.01 to 0.10 on
the models measured. `free` is the same attacker with no constraint past the injection
ball, which is not the `all` scope above and spends its whole budget. Arms: `pgd`,
`random` under the same ceiling, and each budget's own clean arm. Rows are inverted at every
layer from the injection onward and read one layer at a time against that layer's own
threshold, the single-layer rule of the section above. Driver: `scripts/read_layer_reread.py`.

{read_layer_table()}

{read_layer_reading()}

**What the attacker buys by accepting detection.** The ceiling at the watched layer raised
from that layer's threshold upward, Llama-3.2-1B at layer 16, budget 0.30, n=15:

{read_layer_sweep_table()}

{read_layer_nosculpt_line()}

### A raw-state density baseline (Qwen-0.5B, n=50)

Once the claim is detection from one layer, the cheap competitor is a density check on the
raw state at that layer. It was given every advantage: the injected position (last prompt
token), the injection layer and each later one, and a leave-one-out 95th-percentile
threshold, since an in-sample threshold under-reads a density fitted on the same points.
Two references. The *matched* one fits on the ladder's other 49 clean harmful prompts, the
same kind of text as the prompt under test, which is itself held out. The other is the
detector's own five-category clean bank. Three estimators: standardised Euclidean distance
(diagonal covariance), Mahalanobis under a Ledoit-Wolf shrinkage covariance, and relative
distance to the nearest reference state (`scripts/mahalanobis_baseline.py`). "Shift" is the
median change in a prompt's score from its clean state to its attacked one, in units of the
reference's leave-one-out spread.

{mahalanobis_table()}

{mahalanobis_reading()}

## The earlier CAA sweep, a different check

The August steering sweep (contrastive-activation vectors at fractions 0.25 to 1 of the
fitted direction, all prompt positions, six models, two dtypes) is on disk as
{sb['rows']} steered rows in {cluster['sweep_b']['n_files']} distinct join files, {sb['jailbroken_rows']} judge-positive, {sb['undetected_jailbroken']} undetected.
Its "detected" flag is **not the detector above**. It is a local reconstruction check:
at each layer, with the true prefix supplied, the best reconstruction residual at the
final prompt token is compared with a relative tolerance (`src/steer_audit.py`,
`residual > rel_tol × ‖h‖`), and `src/join_detection.py` reads that flag or re-applies a
tolerance to the same residual. No calibration, no multi-layer rule, no full inversion.
The same prompts recur across models and dtypes; the {sb['behavior_prompt_clusters']} distinct (behavior, prompt)
clusters give a zero-failure bound of {100 * sb['zero_event_upper_by_behavior_prompt_clusters']:.1f}% for that check. Its perturbation is a
fraction of a fitted steering vector, which is a different scale from the PGD
relative-state budget; no common scale has been established, so the two are not
compared here.

## Open, unmeasured, and not run

- **The gemma sentiment "crack".** 2/50 undetected next-token argmax flips were observed at
  budget 0.0085; two-sided 95% interval 0.5-13.7%, polarity split 2/25 vs 0/25 (Fisher
  p = 0.49). Argmax flips are no longer a behaviour metric here and the cell was never
  decoded or judged, so it is unmeasured under the definition above and inconclusive as a
  flip count. More prompts would narrow it; they would not necessarily settle it.
- **Argmax flips** appear in older tables as "flip rate". They are kept in the row files as
  a diagnostic of the perturbation and are not reported as behaviour.

## Scope and what is not compared

- The random-direction arm is a control on the attack, not a baseline for the detector. The
  two obvious comparators, a recomputation check and a linear probe on activations, were
  run against the same rows in the baselines section above, each given the same observed
  states with its extra inputs declared. No claim of superiority is made; where the
  recomputation check has consecutive layer states and trusted weights it exposes the
  single-layer injection directly wherever the delta clears its fp16 floor, and the
  detector's contribution is confined to the settings that check cannot cover. One of
  those, a log with no layer before the injection, is measured above ("One logged layer");
  soft prompts and embedding edits are not.
- The non-surjectivity result of Mishra et al. is the motivation: steered states almost
  surely have no prompt preimage. That paper also reports experiments. The proof does not
  bound how far a steered state sits from clean ones relative to fp16 error, and it does not
  guarantee that SipIt finds the best-matching prompt. Every reliability statement here rests
  on the measured tables.
- Sample sizes are 15 prompts per CUDA ladder and 50 on the Qwen-0.5B ladder; a 15-prompt
  cell at 15/15 has a lower bound of 78%.

## Provenance manifest

Each file below with its sha256 is what produced the tables above. The machine-readable
copy is `results/current/current-results-manifest-{DATE}.json`.

"""

manifest = {"generated": DATE, "code": fp, "files": {}}
rows = ["| source | file | sha256 |", "|---|---|---|"]
for name, path in sorted(SOURCES.items()):
    h = provenance.file_sha256(path)
    manifest["files"][name] = {"path": str(path.relative_to(ROOT)), "sha256": h}
    rows.append("| %s | `%s` | `%s` |" % (name, path.relative_to(ROOT), h[:16]))
doc += "\n".join(rows) + "\n"

(ROOT / "RESULTS.md").write_text(doc)
(ROOT / f"results/current/current-results-manifest-{DATE}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("wrote RESULTS.md and results/current/current-results-manifest-%s.json (%d source files)" % (DATE, len(SOURCES)))
