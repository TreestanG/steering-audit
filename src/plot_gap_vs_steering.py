"""Overlay two depth-profiles that share only the layer axis:

  * gap/‖h‖  — SipIt token spacing in *activation* space (geometry of the vocab
    manifold, no intervention). From sipit_layer_XX.jsonl, positions t>0.
  * steered gap — the shift in logit(word_pos) − logit(word_neg) caused by adding
    the sentiment steering vector at the last token. A causal *logit*-space
    effect. From results/.../sentiment_gaps.json, alongside its norm-matched
    random-direction control.

These are different quantities in different units (dimensionless L2 ratio vs
logits) measured on different prompt sets, so a single shared y-axis would be
misleading. Panel A uses a twin y-axis; panel B min-max normalizes both to [0,1]
so their shapes can be compared on one axis. The only honest shared coordinate is
the layer, and layer i means the same thing on both sides: the output of the i-th
decoder block (hidden_states[i]) — SipIt inverts it, the steering hook perturbs it.

The question this answers: does the steering effect track the token crowding?
gap/‖h‖ falls with depth (tokens crowd relative to scale) while the steered gap
grows — i.e. steering bites hardest exactly where the geometry is most compressed.
Panel B and the printed rank correlation quantify that, with the caveat that both
are monotone in depth, so this is suggestive of a shared driver, not proof of one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_sipit_layers import DEFAULT_REL_TOL, summarize


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, hand-rolled to avoid a scipy dependency."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):  # average ranks within ties
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def load_steering(path: Path) -> dict:
    d = json.loads(path.read_text())
    by_layer = {e["layer"]: e for e in d["layers"]}
    return {"meta": d, "by_layer": by_layer}


def align(sipit_summaries: list[dict], steering: dict) -> dict:
    """Intersect on layer; keep only layers present with data on both sides."""
    sip = {s["layer"]: s for s in sipit_summaries}
    layers = sorted(
        L
        for L in set(sip) & set(steering["by_layer"])
        if sip[L]["sep"]["p50"] is not None
    )
    return {
        "layers": layers,
        "sep50": [sip[L]["sep"]["p50"] for L in layers],
        "sep10": [sip[L]["sep"]["p10"] for L in layers],
        "sep90": [sip[L]["sep"]["p90"] for L in layers],
        "steered": [steering["by_layer"][L]["steered_gap_mean"] for L in layers],
        "random": [steering["by_layer"][L]["random_gap_mean"] for L in layers],
    }


C_SEP, C_STEER, C_RAND = "C9", "C3", "C7"


def _panel_twin(ax, data, meta):
    """gap/‖h‖ (left, geometry) vs steered & random gap (right, logits)."""
    xs = data["layers"]
    l1 = ax.plot(xs, data["sep50"], "o-", ms=4, color=C_SEP, label="gap/‖h‖ (SipIt, t>0)")
    ax.fill_between(xs, data["sep10"], data["sep90"], alpha=0.15, color=C_SEP, lw=0)
    ax.set_ylabel("gap / ‖h‖   (activation-space spacing)", color=C_SEP)
    ax.tick_params(axis="y", labelcolor=C_SEP)
    ax.set_xlabel("Hidden-state layer  (output of block i)")

    pos, neg = meta["word_pos"].strip(), meta["word_neg"].strip()
    rax = ax.twinx()
    l2 = rax.plot(xs, data["steered"], "s-", ms=4, color=C_STEER,
                  label=f"steered gap  Δ(logit {pos}−{neg})")
    l3 = rax.plot(xs, data["random"], ".--", ms=5, color=C_RAND,
                  label="random gap (norm-matched control)")
    rax.axhline(0, color=C_RAND, lw=0.6, ls=":")
    rax.set_yscale("symlog", linthresh=0.05)
    rax.set_ylabel("Δ(logit_pos − logit_neg)   [logits, symlog]", color=C_STEER)
    rax.tick_params(axis="y", labelcolor=C_STEER)

    lines = l1 + l2 + l3
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="upper left")
    ax.set_title("Token spacing vs steering effect  (different units → twin axis)")
    ax.grid(True, alpha=0.3)


def _norm01(v: list[float]) -> list[float]:
    lo, hi = min(v), max(v)
    return [0.5 for _ in v] if hi == lo else [(x - lo) / (hi - lo) for x in v]


def _panel_shape(ax, data):
    """Both curves min-max normalized to [0,1] so their shapes overlay on one axis."""
    xs = data["layers"]
    ax.plot(xs, _norm01(data["sep50"]), "o-", ms=4, color=C_SEP,
            label="gap/‖h‖  (falls with depth)")
    ax.plot(xs, _norm01(data["steered"]), "s-", ms=4, color=C_STEER,
            label="steered gap  (rises with depth)")
    ax.plot(xs, _norm01(data["random"]), ".--", ms=5, color=C_RAND, label="random gap")

    rho = _spearman(data["sep50"], data["steered"])
    rho_ctrl = _spearman(data["random"], data["steered"])
    txt = []
    if rho is not None:
        txt.append(f"Spearman(gap/‖h‖, steered) = {rho:+.2f}")
    if rho_ctrl is not None:
        txt.append(f"Spearman(random, steered) = {rho_ctrl:+.2f}")
    if txt:
        # top-center: the band above both curves (y>0.9 for mid layers) is clear.
        ax.text(0.5, 0.97, "\n".join(txt), transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="0.25",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("min–max normalized  [0,1]")
    ax.set_xlabel("Hidden-state layer  (output of block i)")
    ax.set_title("Same curves, shape-normalized  (do they co-move with depth?)")
    # lower-left sits over the flat steered tail (y≈0), clear of the crossing.
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)


def plot(data: dict, meta: dict, out: Path) -> None:
    layers = data["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _panel_twin(axes[0], data, meta)
    _panel_shape(axes[1], data)
    for ax in axes:
        ax.set_xticks(layers[::2] if len(layers) > 14 else layers)
    fig.suptitle(
        f"gap/‖h‖ vs steered gap — {meta['model_name']}, "
        f"fraction={meta['fraction']}, pos={meta['word_pos']!r} neg={meta['word_neg']!r}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sipit_dir", type=str,
                   default="results/Qwen_Qwen2.5-0.5B-Instruct/layers")
    p.add_argument("--gaps", type=str, default="results/sentiment_gaps.json")
    p.add_argument("--rel_tol", type=float, default=DEFAULT_REL_TOL,
                   help="rel_tol the SipIt sweep used; recovers ‖h‖ = tol / rel_tol")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    sipit = summarize(Path(args.sipit_dir), rel_tol=args.rel_tol)
    steering = load_steering(Path(args.gaps))
    data = align(sipit, steering)
    if not data["layers"]:
        raise SystemExit("no overlapping layers between SipIt results and sentiment_gaps.json")

    out = Path(args.out) if args.out else Path(args.sipit_dir).parent / "gap_vs_steering.png"
    plot(data, steering["meta"], out)

    print(f"\n{'L':>3}{'gap/‖h‖':>10}{'steered':>11}{'random':>10}")
    for i, L in enumerate(data["layers"]):
        print(f"{L:>3}{data['sep50'][i]:>10.4f}{data['steered'][i]:>11.4f}"
              f"{data['random'][i]:>10.4f}")


if __name__ == "__main__":
    main()
