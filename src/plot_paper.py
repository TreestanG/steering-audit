"""Paper figures. Main figures leave takeaways to the paper captions; every series
is in a legend, thresholds and the legal budget are drawn. Reads results/current/*.json,
sweep B (results/) and the CUDA ladders (results_cuda/); writes figures/paper/figP*.

  uv run python src/plot_paper.py [substring ...]
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import norm

import ladders

plt.style.use(["science", "no-latex", "bright"])
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "legend.fontsize": 6.5,
                     "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 3.5})

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)
R = Path("results")
RC = Path("results_cuda")

MODELS = [
    ("gpt2_fp16", "GPT-2 124M"),
    ("Qwen_Qwen2.5-0.5B-Instruct_fp16", "Qwen-0.5B"),
    ("Qwen_Qwen2.5-1.5B-Instruct_fp16", "Qwen-1.5B"),
    ("Qwen_Qwen2.5-3B-Instruct_fp16", "Qwen-3B"),
    ("Qwen_Qwen2.5-7B-Instruct_fp16", "Qwen-7B"),
    ("google_gemma-3-1b-it_fp16", "Gemma-1B"),
]
PAL = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MCOL = {slug: PAL[i] for i, (slug, _) in enumerate(MODELS)}
# One color per model wherever a figure colors by model, and one accent for everything on the attacker's side
# (the injection, the jailbreak rate, the budget band). Detection is always PAL[0] blue.
MODEL_COLORS = {"Qwen-0.5B": PAL[0], "Qwen-1.5B": PAL[1], "Qwen-7B": PAL[2], "Gemma-1B": PAL[3], "Llama-3.2-1B": PAL[4]}
ATTACK = "#a34d1c"
REL_TOL16 = 1e-2
TEXT_W = 5.5  # ICLR \textwidth in inches; figures are drawn at print size so fonts print at their nominal points


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def save(fig, name):
    # Crop vertical air only: the width stays TEXT_W so fonts print at their nominal size.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    box = Bbox([[0, max(tight.y0 - 0.03, 0)], [w, min(tight.y1 + 0.03, h)]])
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=box)
    fig.savefig(OUT / f"{name}.png", bbox_inches=box)
    plt.close(fig)
    print("saved", name)


def _cells(path, **match):
    d = json.load(open(path))
    return [c for c in d["cells"] if all(c.get(k) == v for k, v in match.items())], d["calibration"]


def _readouts_on(ax, legend_below=True):
    cfg = [("gpt2_fp16", "GPT-2, layer 8, single attacked position", 8,
            ["tpr_summary.json", "tpr_summary_k1.json", "tpr_summary_rolezlog_k1_fpr5.json"])]
    stats = [("top-5 mean, raw", "-"), ("top-1, raw", "--"), ("top-1, log + template roles", ":")]
    arms = [(("pgd", "sentiment"), "o", "PGD (sentiment objective)"), (("pgd", "cw"), "^", "PGD (margin objective)"),
            (("random", "-"), "s", "norm-matched random")]
    axes = [ax]
    for ax, (slug, title, L, files) in zip(axes, cfg):
        for (sname, ls), f, col in zip(stats, files, PAL):
            p = R / slug / "pgd_sipit" / f
            if not p.exists():
                continue
            cells, cal = _cells(p, behavior="sentiment", constraint="all", n_positions=1, inj_layer=L)
            for (arm, obj), mk, aname in arms:
                cc = sorted([c for c in cells if c["arm"] == arm and c["objective"] == obj], key=lambda c: c["budget"])
                if not cc:
                    continue
                ax.plot([c["budget"] for c in cc], [c["tpr_shipped"] for c in cc], ls=ls, marker=mk, color=col,
                        mfc=col if arm == "pgd" else "none")
        ax.axvline(REL_TOL16, color="k", ls=":", lw=0.8)
        ax.text(REL_TOL16 * 1.05, 0.55, "fp16 tolerance", fontsize=5, rotation=90, va="center")
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("per-position budget (relative)")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("detection rate")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, ls=ls, label=n) for (n, ls), c in zip(stats, PAL)]
    h += [Line2D([], [], color="k", marker=mk, ls="", mfc="k" if arm[0] == "pgd" else "none", label=n) for arm, mk, n in arms]
    if legend_below:
        axes[0].legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=5, frameon=False, columnspacing=0.8)
    else:
        axes[0].legend(handles=h, loc="lower right", fontsize=5, frameon=True)


def _tail_on(ax):
    sig = np.linspace(0, 6, 121)
    pooled = {"top-5, raw": [], "top-1, raw": [], "top-5, log": [], "top-1, log": []}
    for slug, name in MODELS:
        stats = {}
        for f in sorted((R / slug / "sipit/layers").glob("sipit_layer_*.jsonl")):
            if f.stem.endswith("_00"):
                continue
            for r in rows(f):
                rel = sorted([s["residual"] / s["h_norm"] for s in r["steps"] if s["h_norm"] > 0], reverse=True)
                if len(rel) < 5:
                    continue
                stats.setdefault(r["layer"], []).append((np.mean(rel[:5]), rel[0]))
        for L, v in stats.items():
            a = np.array(v)
            for j, k in enumerate(["top-5", "top-1"]):
                x = a[:, j]
                pooled[f"{k}, raw"].extend((x - x.mean()) / x.std())
                lx = np.log(x[x > 0])
                pooled[f"{k}, log"].extend((lx - lx.mean()) / lx.std())
    for (k, z), col, ls in zip(pooled.items(), [PAL[0], PAL[1], PAL[0], PAL[1]], ["-", "-", "--", "--"]):
        z = np.array(z)
        ax.plot(sig, [(z > s).mean() for s in sig], ls=ls, color=col, label=k)
    ax.plot(sig, norm.sf(sig), "k:", label="Gaussian")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlabel(r"$\sigma$ (per-layer $z$ of the clean statistic)")
    ax.set_ylabel(r"$P(z > \sigma)$, clean (trajectory, layer)")
    ax.axvline(3, color="k", lw=0.5, ls=":")
    ax.legend(loc="lower left", fontsize=4.8, frameon=False)


SL_PATH = Path("results/current/single-layer-2026-09-08.json")


def _sl():
    return json.load(open(SL_PATH))


def _sl_cells(e):
    return {(c["arm"], c["budget"]): c for c in e["cells"]}


def _sl_budget(cells, b):
    return min((k[1] for k in cells if k[0] == "pgd"), key=lambda x: abs(x - b))


def _sl_rate(c, L):
    x = c["per_layer"][str(L)]["single"]
    return x["flagged"] / x["n"]


CUR = Path("results/current")
MAHA = CUR / "mahalanobis-qwen05b-2026-09-08.json"
TRANS = CUR / "n50-transitions-2026-09-08.json"
BASELINES = {label: lad.baselines for label, lad in ladders.LADDERS.items()}
HOLDOUT = [
    ("Qwen-0.5B", 25, "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_3way.json"),
    ("Qwen-0.5B", 50, "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n50_3way.json"),
    ("Qwen-0.5B", 75, "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_3way.json"),
    ("Qwen-0.5B", 100, "results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n100_3way.json"),
    ("Qwen-1.5B", 75, "results_cuda/Qwen_Qwen2.5-1.5B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_3way.json"),
    ("Qwen-7B", 75, "results_cuda/Qwen_Qwen2.5-7B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_3way.json"),
    ("Gemma-1B", 25, "results/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_chat_3way.json"),
    ("Gemma-1B", 75, "results_cuda/google_gemma-3-1b-it_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_3way.json"),
    ("Llama-3.2-1B", 75, "results_cuda/meta-llama_Llama-3.2-1B-Instruct_fp16/sipit/detector_holdout_rolezlog_k1_fpr5_n75_3way.json"),
]
BUD = [0.0085, 0.02, 0.05, 0.12, 0.3]
# Judge-positive transitions that are not counted as gains: one fp16 decoding on Qwen-0.5B, scored at two budgets.
EXCLUDED_GAINS = {("Qwen-0.5B (n=50)", 0.0085): 1, ("Qwen-0.5B (n=50)", 0.02): 1}
XT = ["clean"] + [f"{b:g}" for b in BUD]
XS = np.arange(len(XT))


def base_rates(label):
    d = json.load(open(BASELINES[label]))
    cells = {(c["arm"], c["budget"]): c for c in d["recompute"]["cells"]}
    return cells, d


# ------------------------------------------------------------ figP1: one logged layer, five models
def fig_paper2_offset_and_scope():
    """Compare single-layer detection at injection and at the final layer."""
    data = _sl()
    order = ["Qwen-0.5B (n=50)", "Qwen-1.5B", "Qwen-7B", "gemma-3-1b", "Llama-3.2-1B"]
    by_model = {e["model"]: e for e in data}
    labels = ["Qwen-0.5B", "Qwen-1.5B", "Qwen-7B", "Gemma-1B", "Llama-3.2-1B"]
    fig = plt.figure(figsize=(TEXT_W, 4.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                           left=0.14, right=0.92, bottom=0.565, top=0.745, wspace=0.13)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, site, title in zip(axes, ["injection", "final"],
                               ["(a) Log at injection layer", "(b) Log at final layer"]):
        rates = np.zeros((len(order), len(BUD) + 1))
        counts = {}
        for i, model in enumerate(order):
            e = by_model[model]
            layer = e["inj_layer"] if site == "injection" else e["layers"][-1]
            cells = _sl_cells(e)
            for j, (arm, budget) in enumerate([("clean", 0.0)] + [("pgd", b) for b in BUD]):
                matched = 0.0 if arm == "clean" else _sl_budget(cells, budget)
                assert np.isclose(matched, budget), (model, budget, matched)
                result = cells[(arm, matched)]["per_layer"][str(layer)]["single"]
                assert result["n"] > 0 and result.get("unscorable", 0) == 0
                counts[i, j] = (result["flagged"], result["n"])
                rates[i, j] = result["flagged"] / result["n"]
        im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for (i, j), (flagged, n) in counts.items():
            ax.text(j, i, f"{flagged}/{n}", ha="center", va="center", fontsize=6.3,
                    color="white" if rates[i, j] >= 0.6 else "#172b3a")
        ax.set_title(title, fontsize=8.5, pad=31)
        ax.set_xticks(range(len(BUD) + 1), ["Clean"] + [f"{b:g}" for b in BUD])
        ax.set_yticks(range(len(order)), labels if site == "injection" else [""] * len(order))
        ax.set_xlabel("Clean control / relative perturbation budget", labelpad=5)
        ax.set_xticks(np.arange(-0.5, len(BUD) + 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0, top=False, right=False, labelsize=7)
        ax.axvline(0.5, color="#697786", linewidth=1.2)
        # Bracket the budgets where the Qwen-0.5B jailbreak rate rises (0.12 unadjusted; only 0.3 survives Holm).
        transform = ax.get_xaxis_transform()
        ax.plot([3.5, 3.5, 5.5, 5.5], [1.015, 1.065, 1.065, 1.015],
                transform=transform, clip_on=False, color=ATTACK, linewidth=1.2)
        ax.text(4.5, 1.08, "Jailbreak rate rises,\nQwen-0.5B", transform=transform,
                ha="center", va="bottom", fontsize=6.3, color=ATTACK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[0, 2]), ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(which="both", length=0, labelsize=7)
    cb.outline.set_visible(False)
    cb.set_label("Prompts flagged", fontsize=7, labelpad=6)
    depth_grid = fig.add_gridspec(1, 2, left=0.14, right=0.92, bottom=0.16, top=0.385, wspace=0.18)
    for bi, budget in enumerate([0.0085, 0.02]):
        ax = fig.add_subplot(depth_grid[0, bi])
        for i, model in enumerate(order):
            e = by_model[model]
            cell = _sl_cells(e)[("pgd", _sl_budget(_sl_cells(e), budget))]
            offsets = [L - e["inj_layer"] for L in e["layers"]]
            rates = [_sl_rate(cell, L) for L in e["layers"]]
            ax.plot(offsets, rates, marker=["o", "s", "^", "D", "v"][i], color=MODEL_COLORS[labels[i]],
                    label=labels[i], ms=3, lw=1)
        ax.set_title(f"(c) Detection by depth · budget {budget:g}" if bi == 0 else f"Budget {budget:g}", fontsize=8)
        ax.set_xlabel("Layer offset from injection")
        ax.set_ylim(-0.05, 1.07)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Fraction flagged" if bi == 0 else "")
        ax.grid(axis="y", alpha=0.15)
        if bi == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.36), ncol=5, fontsize=6.5, frameon=False, columnspacing=1.2)
    save(fig, "figP2_offset_and_scope")


# ------------------------------------------------------------ hero: detection saturates before behavior moves
GAINS = {
    "Qwen-0.5B (n=50)": TRANS,
    "Qwen-1.5B": RC / "Qwen_Qwen2.5-1.5B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json",
    "Qwen-7B": RC / "Qwen_Qwen2.5-7B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json",
    "gemma-3-1b": RC / "google_gemma-3-1b-it_fp16/pgd_sipit/detection_vs_efficacy.json",
    "Llama-3.2-1B": RC / "meta-llama_Llama-3.2-1B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json",
}
HERO_LABELS = {"Qwen-0.5B (n=50)": "Qwen-0.5B", "Qwen-1.5B": "Qwen-1.5B", "Qwen-7B": "Qwen-7B",
               "gemma-3-1b": "Gemma-1B", "Llama-3.2-1B": "Llama-3.2-1B"}


def fig_paper4_detection_vs_jailbreak():
    """Final-layer detection and the paired induced-jailbreak rate on one budget axis, five models."""
    single = {e["model"]: e for e in _sl()}
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.90, wspace=0.08)
    for i, (key, path) in enumerate(GAINS.items()):
        e = single[key]
        sc, last = _sl_cells(e), e["layers"][-1]
        flagged = [_sl_rate(sc[("clean", 0.0)], last)] + [_sl_rate(sc[("pgd", _sl_budget(sc, b))], last) for b in BUD]
        cells = {c["fraction"]: c for c in json.loads(Path(path).read_text())["cells"] if c["arm"] == "pgd"}
        n = {cells[b]["n"] for b in BUD}
        assert len(n) == 1, (key, n)
        n = n.pop()
        assert n == sc[("pgd", _sl_budget(sc, BUD[0]))]["per_layer"][str(last)]["single"]["n"], key
        gains = [(cells[b]["gains"] - EXCLUDED_GAINS.get((key, b), 0)) / n for b in BUD]
        label = f"{HERO_LABELS[key]} (n = {n})"
        axes[0].plot(XS, flagged, marker=markers[i], color=MODEL_COLORS[HERO_LABELS[key]], label=label, ms=3.5)
        axes[1].plot(XS[1:], gains, marker=markers[i], color=MODEL_COLORS[HERO_LABELS[key]], label=label, ms=3.5)
    axes[0].set_title("(a) Flagged from the final layer alone")
    axes[1].set_title("(b) Attack-induced jailbreaks")
    axes[0].set_ylabel("Fraction of prompts")
    axes[0].axhline(0.05, color="0.45", ls=":", lw=0.7)
    axes[0].text(5.3, 0.085, "5% nominal false-positive rate", fontsize=5.8, color="0.45", ha="right")
    axes[1].axvspan(3.5, 5.5, color="0.92", zorder=0)
    axes[1].text(4.45, 0.88, "Jailbreak rate rises\n(Qwen-0.5B)", ha="center", va="center",
                 fontsize=6.3, color=ATTACK)
    for ax, ticks, names in ((axes[0], XS, XT), (axes[1], XS[1:], XT[1:])):
        ax.set_xticks(ticks, names)
        ax.set_xlim(-0.4, len(XT) - 0.6)
        ax.set_xlabel("Relative perturbation budget")
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False,
               fontsize=6.5, columnspacing=1.0, handletextpad=0.4)
    save(fig, "figP4_detection_vs_jailbreak")


# ------------------------------------------------------------ figP3: the ladder of auditors on one model
def fig_paper3_ladder_of_auditors_qwen05b():
    bc, base = base_rates("Qwen-0.5B (n=50)")
    n = bc[("pgd", 0.0085)]["n"]
    e = {x["model"]: x for x in _sl()}["Qwen-0.5B (n=50)"]
    sc, last, inj = _sl_cells(e), e["layers"][-1], e["inj_layer"]
    maha = json.load(open(MAHA))
    byL = {x["layer"]: x for x in maha["per_layer"]}
    probe = {c["budget"]: c for c in base["probe"]["cells"] if c["feature"] == "inj"}
    fig, ax = plt.subplots(figsize=(TEXT_W, 3.15))
    ax.plot(XS, [bc[("clean", 0.0)]["flagged_batch"] / n] + [bc[("pgd", b)]["flagged_batch"] / n for b in BUD],
            "-x", color="k", ms=5, mew=0.9, label="recomputation, full log")
    ax.plot(XS, [_sl_rate(sc[("clean", 0.0)], last)] + [_sl_rate(sc[("pgd", _sl_budget(sc, b))], last) for b in BUD],
            "-D", color=PAL[0], ms=3.5, label=f"inversion, final layer")
    ax.plot(XS, [_sl_rate(sc[("clean", 0.0)], inj)] + [_sl_rate(sc[("pgd", _sl_budget(sc, b))], inj) for b in BUD],
            ":v", color=PAL[0], ms=3.5, mfc="none", label=f"inversion, injection layer")
    ax.plot(XS[1:], [probe[b]["tpr_pgd"] / n for b in BUD], "-^", color=PAL[3], ms=3.2,
            label="linear probe, injection layer, supervised")
    ax.plot(XS[1:], [probe[b]["fpr_clean"] / n for b in BUD], "--^", color=PAL[3], ms=3.2, mfc="none",
            label="linear probe, clean false-positive rate")
    best = max(maha["per_layer"], key=lambda x: x["matched"]["lw"]["cells"]["pgd|0.3"]["flagged"])["layer"]
    for L, ls, mk, lab in ((best, "-", "s", f"density, best layer (L{best})"),
                           (inj, ":", "s", f"density, injection layer")):
        c = byL[L]["matched"]["lw"]["cells"]
        ax.plot(XS, [c["clean|0.0"]["flagged"] / n] + [c[f"pgd|{b:g}"]["flagged"] / n for b in BUD], ls, marker=mk,
                color=PAL[4], ms=3.2, mfc=PAL[4] if ls == "-" else "none", label=lab)
    ax.axvspan(3.5, 4.5, color="0.9")
    ax.text(4, 0.80, "Jailbreak rate first rises\np = 0.022, unadjusted\nHolm p = 0.09",
            ha="center", va="center", fontsize=7, color=ATTACK)
    ax.set_xticks(XS); ax.set_xticklabels(XT, rotation=0); ax.set_xlabel("Relative PGD budget")
    ax.set_ylim(-0.04, 1.06); ax.set_ylabel("fraction of the 50 prompts flagged")
    ax.legend(loc="upper center", fontsize=6.5, bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, columnspacing=1.5)
    fig.tight_layout()
    save(fig, "figP3_ladder_of_auditors_qwen05b")


# ------------------------------------------------------------ figP5: the detector-aware attacker, single-layer read
def fig_paper5_aware_attacker_single_layer():
    # Use the same 15-prompt cohort throughout; do not mix in the n=50 extension.
    data = json.loads(Path("results/current/read-layer-2026-09-15.json").read_text())
    runs = {e["model"]: e for e in data if "model" in e}
    constrained = runs["Llama-3.2-1B-Instruct read16"]
    free = runs["Llama-3.2-1B-Instruct free"]
    def attack(run):
        return next(c for c in run["cells"] if c["arm"] == "pgd" and c["budget"] == 0.3)
    base, uncapped = attack(constrained), attack(free)
    sweep = next(e["rows"] for e in data
                 if "meta-llama_Llama-3.2-1B-Instruct_sweep/" in e.get("attack_dir", ""))
    sweep = sorted((r for r in sweep if r["arm"] == "pgd" and r["budget"] == 0.3
                    and r["read_layers"] == [16]), key=lambda r: r["ceiling"])
    assert all(r["n"] == 15 for r in sweep) and base["n"] == uncapped["n"] == 15
    xs = np.arange(len(sweep) + 2)
    labels = [f'{base["ceiling"]:.2f}'] + [f'{r["ceiling"]:g}' for r in sweep] + ["no cap"]
    # Join ceiling metrics to the new inversion runs; audit scores, not surrogates.
    import detect
    audits = [e["sweep_detection"] for e in data if "sweep_detection" in e]
    cal = json.loads(Path(constrained["provenance"]["inputs"]["calibration"]["path"]).read_text())
    sweep_read, sweep_other = [], []
    for cell in sweep:
        suffix = f'_sweepz{cell["ceiling"]:g}/pgd_sipit/pgd_rows.jsonl'
        audit = next(a for a in audits if "meta-llama_Llama-3.2-1B-Instruct" in a["tree"]
                     and a["tree"].endswith(suffix))
        assert audit["n"] == cell["n"] and audit["read_layer"] == 16
        assert np.isclose(audit["threshold"], constrained["thresholds"]["16"])
        profiles = {}
        for r in rows(audit["tree"]):
            if r["arm"] == "pgd" and np.isclose(r["budget"], cell["budget"]):
                profiles.setdefault(r["prompt_index"], {})[r["layer"]] = detect.row_score(r, cal["k"], cal)
        assert len(profiles) == cell["n"]
        per_layer = {}
        for layer in constrained["layers"]:
            assert all(layer in profile for profile in profiles.values())
            per_layer[layer] = sum(profile[layer] > constrained["thresholds"][str(layer)]
                                   for profile in profiles.values())
        assert per_layer[16] == audit["flagged_read"]
        other = max((count, layer) for layer, count in per_layer.items() if layer != 16)
        sweep_read.append(audit["flagged_read"] / audit["n"])
        sweep_other.append(other[0] / audit["n"])
    watched = [base["flagged_read"]["16"] / 15] + sweep_read + [uncapped["flagged_read"]["16"] / 15]
    def worst(c):
        return max(v["flagged"] for layer, v in c["per_layer"].items() if layer != "16") / 15
    others = [worst(base)] + sweep_other + [worst(uncapped)]
    judged = [base["judged"]["attacked_positive"] / base["judged"]["n"]]
    judged += [r["judged_positive"] / r["n"] for r in sweep]
    judged += [uncapped["judged"]["attacked_positive"] / uncapped["judged"]["n"]]
    kept = [base["budget_used_median"]] + [r["budget_used_median"] for r in sweep] + [uncapped["budget_used_median"]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.1), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot(xs, watched, "-o", color=PAL[0], label="flagged, logged layer (L16)")
    ax.plot(xs, others, "--s", color=PAL[5], mfc="none", label="flagged, worst other layer")
    ax.plot(xs, judged, "-^", color=ATTACK, label="judged jailbreak rate")
    ax.set_ylim(-0.04, 1.08)
    ax.set_ylabel("Fraction of prompts (n = 15)")
    ax.set_title("(a) Detection and jailbreaks")
    ax.legend(loc="center", fontsize=6.3, frameon=False)
    ax = axes[1]
    ax.plot(xs, kept, "-o", color=PAL[0])
    ax.axhline(1, color="0.3", ls="--", lw=0.8)
    ax.text(0.1, 0.52, "nominal budget 0.30", fontsize=6)
    for x, value in zip(xs, kept):
        ax.annotate(f"{100 * value:.1f}%", (x, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.3)
    ax.set_yscale("log")
    ax.set_ylim(0.0025, 3.0)
    ax.set_yticks([0.01, 0.1, 1], labels=["1%", "10%", "100%"])
    ax.set_ylabel("Median budget retained")
    ax.set_title("(b) Perturbation retained at injection")
    for ax in axes:
        ax.set_xticks(xs, labels=labels)
        ax.set_xlabel("Ceiling at the logged layer only")
        ax.set_xlim(-0.6, len(xs) - 0.4)
    fig.tight_layout()
    save(fig, "figP5_aware_attacker_single_layer")


def fig_paper1_observation_ladder():
    # Verify the off-injection result from saved recomputation measurements.
    counts = set()
    for path in (CUR / "baselines").glob("*.json"):
        data = json.loads(path.read_text())
        for c in data["recompute"]["cells"]:
            if c["arm"] == "pgd":
                counts.add(c["n"])
                assert sum(v > data["recompute"]["bank"]["threshold_single"]
                           for v in c["off_injection_max_single"]) == 0
    assert counts == {15, 50}
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    columns = [.02, .31, .55, .73]
    for x, label in zip(columns, ["Observed layers", "Auditor holds", "Method", "Where evidence appears"]):
        ax.text(x, .94, label, fontsize=8, weight="bold")
    descriptions = [
        ("Prompt + weights\n+ logged states", "Clean\ncounterfactual", "Downstream deviations\nfrom the clean run"),
        ("Weights + a pair\nstraddling injection", "Block\nrecomputation", "Injection boundary only\nOther pairs: 0/50; 0/15*"),
        ("Weights + one late\nlayer + clean bank", "Reachability\naudit", "At the logged layer\nCan persist downstream")]
    for row, (holds, method, fires) in enumerate(descriptions):
        y = .72 - row * .28
        selected = range(6) if row == 0 else ([1, 2] if row == 1 else [5])
        for j in range(6):
            x = .025 + j * .044
            ax.add_patch(Rectangle((x, y-.045), .032, .085,
                                  facecolor=PAL[0] if j in selected else "#f1f3f5",
                                  edgecolor="#657787", lw=.6))
            if j < 5: ax.annotate("", (x+.044, y), (x+.032, y), arrowprops=dict(arrowstyle="->", lw=.5))
        inj_x = .025 + 2 * .044
        ax.annotate("injection", (inj_x+.016, y+.04), (inj_x+.016, y+.115),
                    ha="center", fontsize=6, color=ATTACK, arrowprops=dict(arrowstyle="-|>", color=ATTACK, lw=.8))
        for x, text in zip(columns[1:], [holds, method, fires]):
            ax.text(x, y, text, fontsize=7, va="center", linespacing=1.35)
        if row < 2: ax.axhline(y-.13, color="0.88", lw=.6)
    ax.text(.02, .025, "Filled: logged states · schematic positions · *single-example recomputation; batched results differ",
            fontsize=6, color="0.35")
    fig.subplots_adjust(left=.015, right=.99, top=.98, bottom=.02)
    save(fig, "figP1_observation_ladder")


# ------------------------------------------------------------ figP6: calibration is the result
def fig_appendix_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 3.0), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.36, top=0.84, wspace=0.5)
    _tail_on(axes[0])
    axes[0].set_ylabel(r"$P(z > \sigma)$, clean")
    axes[0].set_title("(a) clean-statistic tail, six models", fontsize=6.5)
    axes[0].text(3.15, 0.12, "at $3\\sigma$: 1.2–1.8%\nnot 0.135%", fontsize=5.5)
    _readouts_on(axes[1], legend_below=True)
    axes[1].set_title("(b) GPT-2 layer 8: three read-outs", fontsize=6.5)
    axes[1].set_xticks([0.002, 0.0043, 0.0085, 0.017]); axes[1].set_xticklabels(["0.002", "0.0043", "0.0085", "0.017"], rotation=40)
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter()); axes[1].tick_params(axis="x", which="minor", bottom=False)
    ax = axes[2]
    seen = {}
    for name, n, path in HOLDOUT:
        p = Path(path)
        if not p.exists():
            continue
        d = json.load(open(p))
        ho = d["held_out"]
        col = MODEL_COLORS[name]
        seen.setdefault(name, []).append((n, ho["fpr_mean"], ho["fpr_sd"], col))
    for name, pts in seen.items():
        pts.sort()
        ax.errorbar([x for x, *_ in pts], [m for _, m, *_ in pts], yerr=[s for _, _, s, _ in pts], fmt="-o", ms=3.2,
                    color=pts[0][3], capsize=2, lw=1.0, label=name)
    ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.text(104, 0.03, "nominal 5%", fontsize=5, ha="right")
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("calibration-bank prompts")
    ax.set_ylabel("held-out FPR")
    ax.set_ylim(0, 0.27); ax.legend(loc="upper right", fontsize=4.8, frameon=False, handlelength=1.2)
    ax.set_title("(c) the nominal 5% is in-sample", fontsize=6.5)
    save(fig, "appendix_calibration")


def _steering_depth_figure(arm):
    """Matched clean-trajectory deviations, not vocabulary-inversion residuals."""
    from matplotlib.lines import Line2D
    specs = [("gpt2", "GPT-2"), ("Qwen_Qwen2.5-0.5B-Instruct", "Qwen-0.5B"),
             ("Qwen_Qwen2.5-1.5B-Instruct", "Qwen-1.5B"),
             ("Qwen_Qwen2.5-3B-Instruct", "Qwen-3B"),
             ("Qwen_Qwen2.5-7B-Instruct", "Qwen-7B"),
             ("google_gemma-3-1b-it", "Gemma-1B")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.95), sharey=True)
    for idx, (slug, label) in enumerate(specs):
        source = Path("results") / (slug + "_fp32") / "pgd/pgd_sentiment_b0.0085.jsonl"
        data = rows(source)
        for ax, scope in zip(axes, ["injection", "all"]):
            selected = [r for r in data if r["arm"] == arm and r["constraint"] == scope]
            assert len(selected) == 25 and len({r["prompt_index"] for r in selected}) == 25
            assert all(r["behavior"] == "sentiment" and np.isclose(r["budget"], .0085) for r in selected)
            layers = sorted(map(int, selected[0]["rel_dev_by_layer"]))
            assert all(r["layer"] == layers[0] for r in selected)
            values = np.array([[r["rel_dev_by_layer"][str(L)] for L in layers] for r in selected])
            median = np.median(values, axis=0)
            ax.plot(np.array(layers) - layers[0], median / .0085, color=PAL[idx],
                    marker=["o", "s", "^", "D", "v", "P"][idx], ms=2.6, lw=1, label=label)
    for ax, title in zip(axes, ["(a) Budget capped at injection only", "(b) Budget capped at every layer"]):
        ax.axhline(1, ls=":", color="0.35", lw=.8)
        ax.set_yscale("log"); ax.set_ylim(.06, 16)
        ax.set_yticks([.1, 1, 10], labels=["0.1×", "1×", "10×"])
        ax.set_xlabel("Layers after injection", fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("Deviation / nominal budget", fontsize=7)
    axes[1].text(.96, .57, "nominal budget", transform=axes[1].transAxes, ha="right", fontsize=6, color="0.35")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=6.3,
               bbox_to_anchor=(.5, -.015), handlelength=1.3, columnspacing=1)
    fig.subplots_adjust(left=.09, right=.99, top=.82, bottom=.28, wspace=.13)
    title = ("PGD deviations amplify downstream; a cap at every layer reduces injection"
             if arm == "pgd" else "CAA attenuates on GPT-2 and Qwen, but amplifies on Gemma")
    fig.suptitle(title, fontsize=9, fontweight="bold", y=.99)
    stem = "appendix_" + arm + "_downstream_deviation"
    save(fig, stem)


def fig_appendix_pgd_downstream_deviation():
    _steering_depth_figure("pgd")


if __name__ == "__main__":
    import sys
    want = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and callable(fn) and (not want or any(w in name for w in want)):
            try:
                fn()
            except Exception as e:
                import traceback; traceback.print_exc()
                print("FAILED", name, e)
