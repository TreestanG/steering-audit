# ‖h‖ is not stored per step, but tol is, and tol = rel_tol * ‖h‖ (see
# sipit.match_tol), so ‖h‖ = tol / rel_tol. Anything derived from the norm
# therefore needs the rel_tol the sweep was run with.
DEFAULT_REL_TOL = 1e-3

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thm 3.2: recovery is guaranteed while residual < gap/2, so this is the
# failure boundary in scale-free units, not a tunable threshold.
THM_BOUND = 0.5


def load_layer_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(values: list[float]) -> list[float]:
    """Drop inf/nan — gap can be inf when a scan exits before seeing a runner-up."""
    return [v for v in values if v is not None and math.isfinite(v)]


def _pct(values: list[float], q: float) -> float | None:
    vals = sorted(_finite(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _band(values: list[float]) -> dict:
    return {
        "p10": _pct(values, 0.10),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "p99": _pct(values, 0.99),
        "max": max(_finite(values)) if _finite(values) else None,
    }


def _step_ratio(step: dict) -> float | None:
    """residual/gap, derived when absent so old-schema files still plot."""
    if "ratio" in step:
        return step["ratio"]
    gap = step.get("gap")
    if gap is None or gap <= 0:
        return None
    return step["residual"] / gap


def _gap_over_norm(step: dict, rel_tol: float) -> float | None:
    """gap as a fraction of ‖h‖ — token spacing measured in units of activation scale."""
    gap = step.get("gap")
    if gap is None or not math.isfinite(gap) or gap <= 0:
        return None
    norm = step.get("h_norm")
    if norm is None:
        # Old-schema fallback. Only valid for a relative-tolerance sweep: with an
        # absolute --tol, tol is a constant and tol/rel_tol is not ‖h‖ at all, so
        # the curve would silently become raw gap.
        tol = step.get("tol")
        if not tol:
            return None
        norm = tol / rel_tol
    if norm <= 0:
        return None
    return gap / norm


def summarize(in_dir: Path, rel_tol: float = DEFAULT_REL_TOL) -> list[dict]:
    files = sorted(in_dir.glob("sipit_layer_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no sipit_layer_*.jsonl in {in_dir}")

    summaries = []
    for path in files:
        rows = load_layer_rows(path)
        if not rows:
            continue

        ratios, res_over_tol, tried, tol_rest = [], [], [], []
        tol_pos0, sep_pos0, sep_rest = [], [], []
        partial_gap = 0
        for row in rows:
            for t, step in enumerate(row["steps"]):
                if t > 0 and not step.get("gap_exhaustive", False):
                    partial_gap += 1
                if t == 0:
                    if step.get("tol") is not None and step["tol"] > 0:
                        tol_pos0.append(step["tol"])
                    g = _gap_over_norm(step, rel_tol)
                    if g is not None:
                        sep_pos0.append(g)
                    continue
                r = _step_ratio(step)
                if r is not None:
                    ratios.append(r)
                tol = step.get("tol")
                if tol is not None and tol > 0:
                    tol_rest.append(tol)
                    res_over_tol.append(step["residual"] / tol)
                g = _gap_over_norm(step, rel_tol)
                if g is not None:
                    sep_rest.append(g)
                tried.append(step["tried"])

        scored = [r for r in rows if r.get("exact") is not None]
        summaries.append(
            {
                "layer": rows[0]["layer"],
                "n": len(rows),
                "exact_frac": (
                    sum(1 for r in scored if r["exact"]) / len(scored) if scored else None
                ),
                "elapsed": _band([r["elapsed"] for r in rows]),
                "ratio": _band(ratios),
                "res_over_tol": _band(res_over_tol),
                "tried": _band(tried),
                "tol_pos0": _pct(tol_pos0, 0.5),
                "tol_rest": _pct(tol_rest, 0.5),
                "sep": _band(sep_rest),
                "sep_pos0": _pct(sep_pos0, 0.5),
                # Positions whose scan stopped early: their runner-up is the
                # nearest of the candidates tried, not of the vocabulary, so gap
                # is an over-estimate and residual/gap an under-estimate.
                "partial_gap": partial_gap,
                "n_steps": len(ratios),
            }
        )
    summaries.sort(key=lambda s: s["layer"])
    return summaries


def _band_panel(ax, layers, summaries, key, title, ylabel, *, log=True, color="C0"):
    p50 = [s[key]["p50"] for s in summaries]
    lo = [s[key]["p10"] for s in summaries]
    hi = [s[key]["p90"] for s in summaries]
    ok = [i for i, v in enumerate(p50) if v is not None]
    if not ok:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    xs = [layers[i] for i in ok]
    ax.plot(xs, [p50[i] for i in ok], marker="o", ms=3.5, color=color, label="median")
    if all(lo[i] is not None and hi[i] is not None for i in ok):
        ax.fill_between(
            xs, [lo[i] for i in ok], [hi[i] for i in ok], alpha=0.2, color=color, lw=0,
            label="p10–p90",
        )
    if log:
        # symlog, not log: layer 0 inverts by table lookup and its residual is
        # bit-exactly 0, which a log axis would silently drop.
        finite = [v for v in p50 + lo + hi if v is not None and v > 0]
        ax.set_yscale("symlog", linthresh=min(finite) / 10 if finite else 1e-9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def partial_gap_note(summaries: list[dict]) -> str | None:
    """Caveat for gaps measured on a truncated scan, or None when all are exact.

    Without --exhaustive, a position's scan stops at the first candidate inside
    tol, so its runner-up is only the nearest of the candidates tried. The true
    vocabulary runner-up can only be nearer, so every such gap is an upper bound
    and every residual/gap a lower bound — the plotted margin is optimistic.
    """
    partial = sum(s.get("partial_gap", 0) for s in summaries)
    total = sum(s.get("n_steps", 0) for s in summaries)
    if not partial or not total:
        return None
    return (
        f"{partial / total:.0%} of gaps from a truncated scan (no --exhaustive):\n"
        "gap is an upper bound, so this margin is optimistic"
    )


def _caveat(ax, note: str | None) -> None:
    if note:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.5, color="C3",
                bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=0.85))


def _panel_margin(ax, layers, summaries):
    _band_panel(ax, layers, summaries, "ratio", "Recovery margin", "residual / gap")
    worst = [s["ratio"]["max"] for s in summaries]
    ok = [i for i, v in enumerate(worst) if v is not None]
    if ok:
        ax.plot([layers[i] for i in ok], [worst[i] for i in ok],
                color="C0", ls=":", lw=1, marker=".", ms=3, label="worst position")
    ax.axhline(THM_BOUND, color="C3", ls="--", lw=1.2,
               label=f"Thm 3.2 bound ({THM_BOUND})")
    ax.legend(fontsize=7, loc="best")
    _caveat(ax, partial_gap_note(summaries))


def _panel_exact(ax, layers, summaries):
    fracs = [s["exact_frac"] for s in summaries]
    ok = [i for i, v in enumerate(fracs) if v is not None]
    ax.bar([layers[i] for i in ok], [fracs[i] * 100 for i in ok], color="C2", width=0.7)
    ax.axhline(100, color="C7", ls=":", lw=1)
    ax.set_ylim(0, 105)
    ax.set_title("Exact recovery")
    ax.set_ylabel("% of prompts")


def _panel_accept(ax, layers, summaries):
    _band_panel(ax, layers, summaries, "res_over_tol",
                "Residual vs acceptance tol", "residual / tol", color="C4")
    ax.axhline(1.0, color="C3", ls="--", lw=1.2, label="rejected above")
    ax.legend(fontsize=7, loc="best")


def _panel_norm(ax, layers, summaries):
    p0 = [s["tol_pos0"] for s in summaries]
    rest = [s["tol_rest"] for s in summaries]
    a = [i for i, v in enumerate(p0) if v]
    b = [i for i, v in enumerate(rest) if v]
    if a:
        ax.plot([layers[i] for i in a], [p0[i] for i in a],
                marker="o", ms=3.5, color="C1", label="position 0 (sink)")
    if b:
        ax.plot([layers[i] for i in b], [rest[i] for i in b],
                marker="o", ms=3.5, color="C0", label="positions t>0 (median)")
    ax.set_yscale("log")
    ax.set_title("Activation scale  (tol ∝ ‖h‖)")
    ax.set_ylabel("tol")
    ax.legend(fontsize=7, loc="best")


def _panel_separation(ax, layers, summaries):
    # Plain log, not the symlog default: gap>0 is enforced in _gap_over_norm, and
    # symlog's linear region near zero would squash the position-0 collapse.
    _band_panel(ax, layers, summaries, "sep", "Token spacing vs activation scale",
                "gap / ‖h‖", log=False, color="C9")
    ax.set_yscale("log")
    p0 = [s.get("sep_pos0") for s in summaries]
    a = [i for i, v in enumerate(p0) if v]
    if a:
        ax.plot([layers[i] for i in a], [p0[i] for i in a], marker="o", ms=3.5,
                color="C1", label="position 0 (sink, whole vocab)")

    # Anchor at layer 1: layer 0 is the embedding table, where inversion is a
    # lookup rather than a forward pass, so it is not on the same footing.
    med = {s["layer"]: s["sep"]["p50"] for s in summaries}
    base = med.get(1)
    if base:
        ax.axhline(base, color="C7", ls=":", lw=1.2,
                   label="layer 1 median (scale-invariant)")
        last = [v for L, v in sorted(med.items()) if v is not None]
        if len(last) > 1:
            ax.annotate(f"×{last[-1] / base:.2f} from layer 1",
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                        fontsize=7, color="C7")
    ax.legend(fontsize=7, loc="best")
    _caveat(ax, partial_gap_note(summaries))


def _panel_tried(ax, layers, summaries):
    """Median is always the first probe (~92% of positions), so plot the tail."""
    for stat, style, label in (
        ("p50", dict(ls="-", marker="o", ms=3.5), "median"),
        ("p99", dict(ls="--", marker=".", ms=3), "p99"),
        ("max", dict(ls=":", marker=".", ms=3), "worst position"),
    ):
        vals = [s["tried"][stat] for s in summaries]
        ok = [i for i, v in enumerate(vals) if v is not None]
        if ok:
            ax.plot([layers[i] for i in ok], [vals[i] for i in ok],
                    color="C5", label=label, **style)
    ax.set_yscale("log")
    ax.set_title("Candidates scanned per position")
    ax.set_ylabel("candidates")
    ax.legend(fontsize=7, loc="best")


def _panel_elapsed(ax, layers, summaries):
    _band_panel(ax, layers, summaries, "elapsed", "Time per prompt", "seconds",
                log=False, color="C6")
    ax.legend(fontsize=7, loc="best")


PANELS = [
    ("margin", _panel_margin),
    ("exact", _panel_exact),
    ("accept", _panel_accept),
    ("norm", _panel_norm),
    ("separation", _panel_separation),
    ("tried", _panel_tried),
    ("elapsed", _panel_elapsed),
]


def _style_layer_axis(ax, layers: list[int]) -> None:
    ax.set_xlabel("Hidden-state layer")
    ax.set_xticks(layers[::2] if len(layers) > 14 else layers)
    ax.grid(True, alpha=0.3)


def plot_metrics(summaries: list[dict], out: Path) -> None:
    layers = [s["layer"] for s in summaries]

    nrows = -(-len(PANELS) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
    flat = axes.ravel()
    for ax, (_, fn) in zip(flat, PANELS):
        fn(ax, layers, summaries)
        _style_layer_axis(ax, layers)
    for ax in flat[len(PANELS):]:
        ax.axis("off")
    fig.suptitle(
        f"SipIt layer sweep — {summaries[0]['n']} prompts, positions t>0 unless noted",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")

    for name, fn in PANELS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fn(ax, layers, summaries)
        _style_layer_axis(ax, layers)
        fig.tight_layout()
        path = out.parent / f"{out.stem}_{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="results/Qwen_Qwen2.5-0.5B-Instruct/layers",
        help="directory containing sipit_layer_XX.jsonl (where eval_sipit_layers.sh writes)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="combined figure path; also writes one sibling PNG per panel",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=DEFAULT_REL_TOL,
        help="rel_tol the sweep was run with. Only a fallback for old files with no "
        "h_norm field, where ‖h‖ = tol / rel_tol; wrong values rescale the gap/‖h‖ "
        "panel but not its shape. Ignored once h_norm is present.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out) if args.out else in_dir / "sipit_layer_metrics.png"
    summaries = summarize(in_dir, rel_tol=args.rel_tol)
    plot_metrics(summaries, out)

    print(
        f"\n{'L':>3}{'n':>4}{'exact':>8}{'med_res/gap':>13}{'worst':>11}"
        f"{'gap/|h|':>10}{'med_tried':>11}{'med_s':>8}"
    )
    for s in summaries:
        ex = f"{s['exact_frac'] * 100:.0f}%" if s["exact_frac"] is not None else "-"
        r50, rmax, tried50 = s["ratio"]["p50"], s["ratio"]["max"], s["tried"]["p50"]
        sep50 = s["sep"]["p50"]
        print(
            f"{s['layer']:>3}{s['n']:>4}{ex:>8}"
            f"{(f'{r50:.2e}' if r50 is not None else '-'):>13}"
            f"{(f'{rmax:.2e}' if rmax is not None else '-'):>11}"
            f"{(f'{sep50:.4f}' if sep50 is not None else '-'):>10}"
            f"{(f'{tried50:.0f}' if tried50 is not None else '-'):>11}"
            f"{s['elapsed']['p50']:>8.2f}"
        )


if __name__ == "__main__":
    main()
