"""Paper figures. Each figure stands on its own: the suptitle is the takeaway, every series
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
from matplotlib.patches import Patch, Rectangle
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
REL_TOL16 = 1e-2
TWO_COL = 7.0


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def _cells(path, **match):
    d = json.load(open(path))
    return [c for c in d["cells"] if all(c.get(k) == v for k, v in match.items())], d["calibration"]


def _readouts_on(ax, legend_below=True):
    cfg = [("gpt2_fp16", "GPT-2, layer 8, single attacked position", 8,
            ["tpr_summary.json", "tpr_summary_k1.json", "tpr_summary_rolezlog_k1_fpr5.json"])]
    stats = [("shipped: top-5 mean, raw", "-"), ("top-1, raw", "--"), ("top-1, log + end roles", ":")]
    arms = [(("pgd", "sentiment"), "o", "PGD (sentiment objective)"), (("pgd", "cw"), "^", "PGD (CW objective)"),
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
        ax.text(REL_TOL16 * 1.05, 0.55, "fp16 tolerance\n(attacker's legal limit)", fontsize=5.5, rotation=90, va="center")
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("per-position budget (relative)")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("detection rate")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, ls=ls, label=n) for (n, ls), c in zip(stats, PAL)]
    h += [Line2D([], [], color="k", marker=mk, ls="", mfc="k" if arm[0] == "pgd" else "none", label=n) for arm, mk, n in arms]
    if legend_below:
        axes[0].legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.5, frameon=False)
    else:
        axes[0].legend(handles=h, loc="lower right", fontsize=5, frameon=True)


def _ladder_n50_on(axes, single_layer=False):
    d = json.load(open(R / "Qwen_Qwen2.5-0.5B-Instruct_n50/pgd_sipit/detection_vs_efficacy.json"))
    cells = {(c["arm"], c["fraction"]): c for c in d["cells"]}
    bud = sorted({c["fraction"] for c in d["cells"]})
    gen = json.load(open(R / "Qwen_Qwen2.5-0.5B-Instruct_n50/pgd/ladder/jbb_refusal_harmful_b0.30_gen.json"))
    base_hb = gen["judge_asr"]["harmbench"]["none"]
    tcells, cal = _cells(R / "Qwen_Qwen2.5-0.5B-Instruct_n50/pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n100.json",
                         behavior="jbb_refusal")
    tc = {(c["arm"], c["budget"]): c for c in tcells}
    ax = axes[0]
    ax.plot(bud, [cells[("pgd", b)]["asr"] for b in bud], "-o", color=PAL[0], label="HarmBench ASR, PGD")
    ax.plot(bud, [cells[("random", b)]["asr"] for b in bud], "-s", color="0.5", mfc="none", label="HarmBench ASR, random")
    ax.plot(bud, [cells[("pgd", b)]["asr_substring"] for b in bud], ":", marker="o", mfc="none", color=PAL[0],
            label="substring ASR, PGD (noisy proxy)")
    ax.plot(bud, [cells[("pgd", b)]["judge_strongreject_score_mean"] for b in bud], "-.^", color=PAL[2],
            label="StrongREJECT score, PGD")
    ax.axhline(base_hb, color="k", lw=0.6, ls="--")
    ax.text(bud[0], base_hb + 0.02, "unsteered baseline", fontsize=5.5)
    ax.set_ylabel("rate / score"); ax.set_title("attack efficacy (n = 50 prompts)")
    ax.legend(loc="upper left", fontsize=5.5)
    ax = axes[1]
    if not single_layer:
        ax.plot(bud, [cells[("pgd", b)]["detection_rate"] for b in bud], "-o", color=PAL[0], label="detected, PGD")
        ax.plot(bud, [cells[("random", b)]["detection_rate"] for b in bud], "--s", color="0.5", mfc="none", label="detected, random")
        ax.plot(bud, [tc[("clean", 0.0)]["tpr_shipped"]] * len(bud), ":", color="k", label="flagged, paired clean control")
    if single_layer:
        e = {x["model"]: x for x in _sl()}["Qwen-0.5B (n=50)"]
        sc, last, inj = _sl_cells(e), e["layers"][-1], e["inj_layer"]
        ax.plot(bud, [_sl_rate(sc[("pgd", _sl_budget(sc, b))], last) for b in bud], "--D", color=PAL[3],
                mfc="none", label=f"detected, PGD: last layer (L{last}) alone")
        ax.plot(bud, [_sl_rate(sc[("pgd", _sl_budget(sc, b))], inj) for b in bud], "--v", color=PAL[4],
                mfc="none", label=f"detected, PGD: injection layer (L{inj}) alone")
        ax.plot(bud, [_sl_rate(sc[("clean", 0.0)], last)] * len(bud), ":", color=PAL[3], lw=0.8,
                label="flagged, clean control, last layer alone")
    ax.set_ylabel("detection rate"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("single-layer detection" if single_layer else
                 r"detector: top-1 log-$z$, $\sigma$ = %.2f, run $\geq$ 3" % cal["sigma"])
    ax.legend(loc="center right", fontsize=5.5)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("PGD budget (relative)")
        ax.axvline(REL_TOL16, color="k", ls=":", lw=0.8)
    axes[1].text(REL_TOL16 * 1.05, 0.45, "fp16 tolerance", fontsize=5.5, rotation=90, va="center")


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
    ax.legend(loc="lower left", fontsize=5.5)


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
AWARE_SL = CUR / "aware-single-layer-2026-09-09.json"
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
XT = ["clean"] + [f"{b:g}" for b in BUD]
XS = np.arange(len(XT))
QCOL = MCOL["Qwen_Qwen2.5-0.5B-Instruct_fp16"]
GREY = "0.45"


def takeaway(fig, text, y=1.0):
    fig.suptitle(text, fontsize=8, y=y, fontweight="bold")


def base_rates(label):
    d = json.load(open(BASELINES[label]))
    cells = {(c["arm"], c["budget"]): c for c in d["recompute"]["cells"]}
    return cells, d


# ------------------------------------------------------------ figP1: one logged layer, five models
def fig_paper1_offset_and_scope():
    """Compare single-layer detection at injection and at the final layer."""
    data = _sl()
    order = ["Qwen-0.5B (n=50)", "Qwen-1.5B", "Qwen-7B", "gemma-3-1b", "Llama-3.2-1B"]
    by_model = {e["model"]: e for e in data}
    labels = ["Qwen-0.5B", "Qwen-1.5B", "Qwen-7B", "Gemma-1B", "Llama-3.2-1B"]
    fig = plt.figure(figsize=(8.2, 3.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                           left=0.14, right=0.92, bottom=0.19, top=0.79, wspace=0.13)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, site, title in zip(axes, ["injection", "final"],
                               ["(a) Log at injection layer", "(b) Log at final layer"]):
        rates = np.zeros((len(order), len(BUD)))
        counts = {}
        for i, model in enumerate(order):
            e = by_model[model]
            layer = e["inj_layer"] if site == "injection" else e["layers"][-1]
            cells = _sl_cells(e)
            for j, budget in enumerate(BUD):
                matched = _sl_budget(cells, budget)
                assert np.isclose(matched, budget), (model, budget, matched)
                result = cells[("pgd", matched)]["per_layer"][str(layer)]["single"]
                assert result["n"] > 0 and result.get("unscorable", 0) == 0
                counts[i, j] = (result["flagged"], result["n"])
                rates[i, j] = result["flagged"] / result["n"]
        im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for (i, j), (flagged, n) in counts.items():
            ax.text(j, i, f"{flagged}/{n}", ha="center", va="center", fontsize=8,
                    color="white" if rates[i, j] >= 0.6 else "#172b3a")
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xticks(range(len(BUD)), [f"{b:g}" for b in BUD])
        ax.set_yticks(range(len(order)), labels if site == "injection" else [""] * len(order))
        ax.set_xlabel("Relative perturbation budget", labelpad=8)
        ax.set_xticks(np.arange(-0.5, len(BUD), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0, top=False, right=False, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[0, 2]), ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(which="both", length=0, labelsize=8)
    cb.outline.set_visible(False)
    cb.set_label("Attacks detected", fontsize=8, labelpad=8)
    fig.suptitle("Final-layer logging detects all tested attacks from budget 0.02", fontsize=11,
                 fontweight="bold", y=0.97)
    fig.text(0.5, 0.045, "Cells: detected / tested prompts · PGD attacks · Evaluator stopping rule",
             ha="center", fontsize=8, color="0.35")
    save(fig, "figP1_offset_and_scope")


# ------------------------------------------------------------ figP2: roles are positions, not words
ROLE_PROMPTS = ["Can my dog eat cat food?", "Can my cat eat dog food, and is it safe long term?"]
ROLE_EDGES = {"content_first", "content_second", "content_last"}
ROLE_BANDS = [("pre", "template prefix: one role per position, indexed from the start", PAL[6]),
              ("edge", "user-turn edges: content_first, content_second, content_last", PAL[1]),
              ("content", "interior content: one shared role", PAL[0]),
              ("suf", "template suffix: one role per position, indexed from the end", PAL[2])]


def _role_band(role):
    if role.startswith("pre"):
        return "pre"
    if role.startswith("suf"):
        return "suf"
    return "edge" if role in ROLE_EDGES else "content"


def fig_paper2_template_roles():
    import detect
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    chat_ids, header_len = detect.chat_template_spec(tk)
    band_col = {k: c for k, _, c in ROLE_BANDS}
    fig, axes = plt.subplots(len(ROLE_PROMPTS), 1, figsize=(TWO_COL, 4.1))
    for ax, msg, tag in zip(axes, ROLE_PROMPTS, "ab"):
        text = tk.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
        ids = tk(text)["input_ids"]
        roles = detect.position_roles(ids, chat_ids, "pooled", 6, header_len)
        for i, (t, r) in enumerate(zip(ids, roles)):
            col = band_col[_role_band(r)]
            ax.add_patch(Rectangle((i + 0.08, 0), 0.84, 1, facecolor=col, alpha=0.35, edgecolor=col, lw=0.6))
            ax.text(i + 0.5, 0.5, str(i), ha="center", va="center", fontsize=3.8)
            tok = tk.decode([t]).replace("\n", "\\n")
            ax.text(i + 0.5, -0.15, tok, rotation=90, ha="center", va="top", fontsize=4.6, family="monospace",
                    fontweight="bold" if tok.strip() in ("dog", "cat") else "normal")
            ax.text(i + 0.5, 1.15, r, rotation=90, ha="center", va="bottom", fontsize=4.2, family="monospace")
        ax.set_xlim(0, len(ids)); ax.set_ylim(-2.3, 3.5); ax.axis("off")
        ax.set_title(f"({tag}) user message {msg!r}: {len(ids)} tokens after Qwen's chat template, role above, token below",
                     fontsize=7.5, loc="left")
    fig.legend(handles=[Patch(facecolor=c, alpha=0.35, edgecolor=c, label=lab) for _, lab, c in ROLE_BANDS],
               loc="lower center", ncol=2, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    takeaway(fig, "Roles are positions, not words: every template token has its own role, the interior of the user turn shares one, the suffix is\n"
                  "indexed from the end. 'dog' and 'cat' are z-scored against the same clean (layer, role) cell wherever they sit in the interior.",
             y=0.995)
    save(fig, "figP2_template_roles")


# ------------------------------------------------------------ figP3: the ladder of auditors on one model
def fig_paper3_ladder_of_auditors_qwen05b():
    bc, base = base_rates("Qwen-0.5B (n=50)")
    n = bc[("pgd", 0.0085)]["n"]
    e = {x["model"]: x for x in _sl()}["Qwen-0.5B (n=50)"]
    sc, last, inj = _sl_cells(e), e["layers"][-1], e["inj_layer"]
    maha = json.load(open(MAHA))
    byL = {x["layer"]: x for x in maha["per_layer"]}
    probe = {c["budget"]: c for c in base["probe"]["cells"] if c["feature"] == "inj"}
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.66, 3.6))
    ax.plot(XS, [bc[("clean", 0.0)]["flagged_batch"] / n] + [bc[("pgd", b)]["flagged_batch"] / n for b in BUD],
            "-x", color="k", ms=5, mew=0.9, label="recomputation (needs every layer)")
    ax.plot(XS, [_sl_rate(sc[("clean", 0.0)], last)] + [_sl_rate(sc[("pgd", _sl_budget(sc, b))], last) for b in BUD],
            "-D", color=QCOL, ms=3.5, label=f"reachability audit, last layer (L{last}) alone")
    ax.plot(XS, [_sl_rate(sc[("clean", 0.0)], inj)] + [_sl_rate(sc[("pgd", _sl_budget(sc, b))], inj) for b in BUD],
            ":v", color=QCOL, ms=3.5, mfc="none", label=f"reachability audit, injection layer (L{inj}) alone")
    ax.plot(XS[1:], [probe[b]["tpr_pgd"] / n for b in BUD], "-^", color=PAL[3], ms=3.2,
            label="linear probe, same layer, told the layer, trained on other prompts")
    ax.plot(XS[1:], [probe[b]["fpr_clean"] / n for b in BUD], "--^", color=PAL[3], ms=3.2, mfc="none",
            label="   its false-positive rate on clean prompts")
    best = max(maha["per_layer"], key=lambda x: x["matched"]["lw"]["cells"]["pgd|0.3"]["flagged"])["layer"]
    for L, ls, mk, lab in ((best, "-", "s", f"density on the raw state, L{L if False else best} (its best layer)"),
                           (inj, ":", "s", f"density on the raw state, injection layer (L{inj})")):
        c = byL[L]["matched"]["lw"]["cells"]
        ax.plot(XS, [c["clean|0.0"]["flagged"] / n] + [c[f"pgd|{b:g}"]["flagged"] / n for b in BUD], ls, marker=mk,
                color=PAL[4], ms=3.2, mfc=PAL[4] if ls == "-" else "none", label=lab)
    ax.axvspan(3.5, 4.5, color="0.9")
    ax.text(4, 0.80, "first judged\njailbreaks beyond\nthe paired clean arm\n(11 of 50, p = 0.02)",
            ha="center", va="center", fontsize=5.3)
    ax.set_xticks(XS); ax.set_xticklabels(XT, rotation=45); ax.set_xlabel("PGD budget (relative); 0.0085 is the legal budget")
    ax.set_ylim(-0.04, 1.06); ax.set_ylabel("fraction of the 50 prompts flagged")
    ax.legend(loc="center left", fontsize=5.0, bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    takeaway(fig, "Same 50 attacked prompts (Qwen-0.5B): with every layer, recomputation is the reference;\n"
                  "from one layer, only inversion sees the attack below the budget that changes behaviour,\n"
                  "and a probe or a density on that same layer does not.", y=0.995)
    save(fig, "figP3_ladder_of_auditors_qwen05b")


# ------------------------------------------------------------ figP4: where the attack works it is caught
def fig_paper4_ladder_qwen05b_single_layer():
    """Induced jailbreak observations, grouped by budget and stacked by detection."""
    configs = [
        ("Qwen-0.5B", "Qwen-0.5B (n=50)", TRANS),
        ("Qwen-1.5B", "Qwen-1.5B", RC / "Qwen_Qwen2.5-1.5B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json"),
        ("Qwen-7B", "Qwen-7B", RC / "Qwen_Qwen2.5-7B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json"),
    ]
    single = {e["model"]: e for e in _sl()}
    counts = []
    for name, key, path in configs:
        transitions = json.loads(path.read_text())
        cells = {c["fraction"]: c for c in transitions["cells"] if c["arm"] == "pgd"}
        e = single[key]
        layer_cells = _sl_cells(e)
        detected, undetected = [], []
        for budget in BUD:
            c = cells[budget]
            result = layer_cells[("pgd", _sl_budget(layer_cells, budget))]["per_layer"][str(e["layers"][-1])]["single"]
            # Complete detection of the attack cell establishes detection of every gain.
            # Partial detection would require a per-prompt join; never substitute the
            # legacy multi-layer gains_undetected field for the single-layer result.
            if c["gains"] and (result["flagged"] != result["n"] or result.get("unscorable", 0)):
                raise ValueError(f"{name}, {budget}: join individual gains to final-layer verdicts before plotting")
            assert result["n"] == c["n"]
            detected.append(c["gains"])
            undetected.append(0)
        assert sum(detected) == transitions["pooled_pgd_gains"]["gains_rows"]
        sample_sizes = {c["n"] for c in cells.values()}
        assert len(sample_sizes) == 1, f"{name}: sample size varies across budgets"
        legend_name = f"{name} (n = {sample_sizes.pop()})"
        counts.append((legend_name, np.array(detected), np.array(undetected)))

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.71)
    xs, width = np.arange(len(BUD)), 0.23
    colors = ["#2166ac", "#5b9ac4", "#a6cbe3"]
    for i, ((name, detected, undetected), color) in enumerate(zip(counts, colors)):
        positions = xs + (i - 1) * width
        ax.bar(positions, detected, width=width * 0.90, color=color, edgecolor="white", linewidth=0.5, label=name, zorder=3)
        ax.bar(positions, undetected, bottom=detected, width=width * 0.90,
               color="#c43c39", edgecolor="white", linewidth=0.5, zorder=3)
        for x, total in zip(positions, detected + undetected):
            ax.text(x, total + 0.35, str(int(total)), ha="center", va="bottom", fontsize=9,
                    color="#203040" if total else "0.55")
    total = sum(int(d.sum() + u.sum()) for _, d, u in counts)
    missed = sum(int(u.sum()) for _, _, u in counts)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#c43c39", label="Undetected (none)"))
    labels.append("Undetected (none)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.80),
               ncol=4, frameon=False, fontsize=8)
    ax.set_xticks(xs, [f"{b:g}" for b in BUD])
    ax.set_xlabel("Relative perturbation budget", fontsize=10, labelpad=9)
    ax.set_ylabel("Induced jailbreak observations", fontsize=10, labelpad=8)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylim(0, 28)
    ax.set_xlim(-0.6, len(BUD) - 0.4)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False, labelsize=9, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
    fig.suptitle("All observed attack-induced jailbreaks are detected from the final layer",
                 fontsize=11, fontweight="bold", y=0.98)
    fig.text(0.54, 0.875, f"{total} induced jailbreak observations  ·  {missed} undetected",
             ha="center", fontsize=11, color="#2166ac")
    fig.text(0.54, 0.04, "PGD attacks · Paired HarmBench labels · Evaluator stopping rule",
             ha="center", fontsize=8, color="0.4")
    save(fig, "figP4_ladder_qwen05b_single_layer")


# ------------------------------------------------------------ figP5: the detector-aware attacker, single-layer read
def fig_paper5_aware_attacker_single_layer():
    zs = [4.28, 6, 8, 12, 16, 20]
    base = R / "Qwen_Qwen2.5-0.5B-Instruct_aware_z{z}"
    det, det_run, flip, asr, rel, base_asr = [], [], [], [], [], None
    for z in zs:
        d = base.with_name(base.name.format(z=f"{z:g}"))
        cells, _ = _cells(d / "pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n100.json", arm="pgd")
        det.append(cells[0]["tpr_shipped"]); flip.append(cells[0].get("argmax_changed_fp16", cells[0].get("flip_fp16")))
        g = json.load(open(d / "pgd/ladder/jbb_refusal_harmful_b0.30_gen.json"))
        asr.append(g["judge_asr"]["harmbench"]["pgd"]); base_asr = g["judge_asr"]["harmbench"]["none"]
        rel.append(json.load(open(d / "pgd/ladder/jbb_refusal_harmful_b0.30.json"))["summary"][0]["rel_delta_mean"])
        dr = d.with_name(d.name.replace("aware_", "aware_run_"))
        if dr.exists():
            c, _ = _cells(dr / "pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n100.json", arm="pgd")
            det_run.append((z, c[0]["tpr_shipped"]))
    sl = json.load(open(AWARE_SL))
    slc = {(c["variant"], c["z_ceiling"]): c for c in sl["cells"]}
    last_alone = [slc[("per_layer", z)]["flagged_last_layer"] / slc[("per_layer", z)]["n"] for z in zs]
    any_layer = [slc[("per_layer", z)]["flagged_any_layer_uncorrected"] / slc[("per_layer", z)]["n"] for z in zs]
    plain, _ = _cells(R / "Qwen_Qwen2.5-0.5B-Instruct_fp16/pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n100.json",
                      behavior="jbb_refusal", arm="pgd", budget=0.3)
    pe = json.load(open(R / "Qwen_Qwen2.5-0.5B-Instruct_fp16/pgd_sipit/detection_vs_efficacy.json"))
    plain_asr = [c["asr"] for c in pe["cells"] if c["arm"] == "pgd" and c["fraction"] == 0.3][0]
    e = {x["model"]: x for x in _sl()}["Qwen-0.5B (n=50)"]
    sc, lastL = _sl_cells(e), e["layers"][-1]
    plain_last = _sl_rate(sc[("pgd", 0.3)], lastL)
    xs = np.arange(len(zs) + 1)
    xt = [f"{z:g}" for z in zs] + ["no cap"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.plot(xs, det + [plain[0]["tpr_shipped"]], "-o", color=PAL[0], label="flagged, multi-layer rule")
    ax.plot(xs, last_alone + [plain_last], "--D", color=QCOL, mfc="none", ms=4, label=f"flagged, last layer (L{lastL}) read alone")
    ax.plot(xs[:-1], any_layer, ":", color=PAL[0], lw=0.9, label="flagged, any layer over its own alarm (uncorrected)")
    if det_run:
        ax.plot([zs.index(z) + 0.15 for z, _ in det_run], [v for _, v in det_run], "s", mfc="none", color=PAL[0],
                ms=4, label="flagged, attacker aimed at the run rule's slack")
    ax.plot(xs, flip + [plain[0].get("argmax_changed_fp16", plain[0].get("flip_fp16"))], "-s", color=PAL[1],
            label="next-token argmax changed (diagnostic)")
    ax.plot(xs, asr + [plain_asr], "-^", color=PAL[2], label="judged jailbreak rate (HarmBench)")
    ax.axhline(base_asr, color="k", lw=0.6, ls="--"); ax.text(2.2, base_asr + 0.025, "unsteered jailbreak rate", fontsize=5.5)
    ax.set_ylabel("rate (n = 15 prompts)"); ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=5.0, bbox_to_anchor=(0.995, 0.5))
    ax.set_title("(a) what the attacker achieves under each ceiling", fontsize=7.5)
    ax = axes[1]
    pl = R / "Qwen_Qwen2.5-0.5B-Instruct_fp32/pgd/ladder/jbb_refusal_harmful_b0.30.json"
    plain_rel = [x["rel_delta_mean"] for x in json.load(open(pl))["summary"] if x["arm"] == "pgd"][0] if pl.exists() else None
    ax.plot(xs[:-1], rel, "-o", color=PAL[0])
    if plain_rel is not None:
        ax.plot([xs[-1]], [plain_rel], "o", color=PAL[0])
    ax.axhline(0.3, color="k", lw=0.6, ls="--"); ax.text(0.05, 0.3 * 0.72, "nominal budget 0.30", fontsize=5.5)
    for x, r in zip(xs[:-1], rel):
        ax.text(x, r * 1.4, f"{100 * r / 0.3:.1f}%", ha="center", fontsize=5.3)
    ax.set_yscale("log"); ax.set_ylim(8e-4, 0.7); ax.set_ylabel("perturbation the attacker keeps (relative)")
    ax.set_title("(b) what the ceiling costs (% of budget kept)", fontsize=7.5)
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(xt); ax.set_xlabel(r"attacker's $z$ ceiling at every layer (alarm: $\sigma$ = 4.78)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    takeaway(fig, "An attacker that differentiates through the audit to stay under its alarm keeps 1% of its budget and changes nothing;\n"
                  "detection turns on between ceilings 4.3 and 8, the first behaviour change appears at 20 (Qwen-0.5B, layer 17, budget 0.30).",
             y=0.995)
    save(fig, "figP5_aware_attacker_single_layer")


# ------------------------------------------------------------ figP6: calibration is the result
def fig_paper6_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    _tail_on(axes[0])
    axes[0].set_title("(a) clean statistic tail, six models pooled", fontsize=7.5)
    axes[0].text(3.55, 0.15, r"at $3\sigma$: 1%, not 0.135%", fontsize=5.5)
    _readouts_on(axes[1], legend_below=True)
    axes[1].set_title("(b) GPT-2 layer 8: three read-outs of one inversion", fontsize=7.5)
    axes[1].set_xticks([0.002, 0.0043, 0.0085, 0.017]); axes[1].set_xticklabels(["0.002", "0.0043", "0.0085", "0.017"])
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter()); axes[1].tick_params(axis="x", which="minor", bottom=False)
    ax = axes[2]
    seen = {}
    for name, n, path in HOLDOUT:
        p = Path(path)
        if not p.exists():
            continue
        d = json.load(open(p))
        ho = d["held_out"]
        col = {"Qwen-0.5B": MCOL["Qwen_Qwen2.5-0.5B-Instruct_fp16"], "Qwen-1.5B": MCOL["Qwen_Qwen2.5-1.5B-Instruct_fp16"],
               "Qwen-7B": MCOL["Qwen_Qwen2.5-7B-Instruct_fp16"], "Gemma-1B": MCOL["google_gemma-3-1b-it_fp16"],
               "Llama-3.2-1B": PAL[6]}[name]
        seen.setdefault(name, []).append((n, ho["fpr_mean"], ho["fpr_sd"], col))
    for name, pts in seen.items():
        pts.sort()
        ax.errorbar([x for x, *_ in pts], [m for _, m, *_ in pts], yerr=[s for _, _, s, _ in pts], fmt="-o", ms=3.2,
                    color=pts[0][3], capsize=2, lw=1.0, label=name)
    ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.text(26, 0.056, "nominal 5% (in-sample)", fontsize=5.5)
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("clean prompts in the calibration bank")
    ax.set_ylabel("held-out false-positive rate\n(3-way split, 20 repeats, mean ± sd)")
    ax.set_ylim(0, 0.27); ax.legend(loc="upper right", fontsize=5.3)
    ax.set_title("(c) the nominal rate is a training constraint", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    takeaway(fig, "Calibration decides whether the detector works: the raw clean statistic is 8× heavier-tailed than Gaussian at 3σ, the read-out\n"
                  "decides which attacks are visible at all, and the nominal 5% false-positive rate reads 5 to 10% held out and is unusable at 25 prompts.",
             y=0.995)
    save(fig, "figP6_calibration")


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
