import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THM_BOUND = 0.5  # Thm 3.2: exact recovery guaranteed while residual/gap < 1/2.
C_STEER, C_RAND, C_GAP, C_RES = "C3", "C7", "C9", "C1"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def by_layer(rows: list[dict]) -> tuple[list[int], dict]:
    layers = sorted({r["layer"] for r in rows})
    groups = {L: [r for r in rows if r["layer"] == L] for L in layers}
    return layers, groups


def _stat(groups, layers, fn):
    """(mean, lo, hi) across prompts per layer; band is min–max since n is small."""
    mean, lo, hi = [], [], []
    for L in layers:
        vals = [fn(r) for r in groups[L]]
        mean.append(sum(vals) / len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return mean, lo, hi


def _line(ax, xs, stat, color, label, ls="-", band=True):
    mean, lo, hi = stat
    ax.plot(xs, mean, ls + "o" if ls == "-" else ls, color=color, ms=4, label=label)
    if band:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
    return mean


def _first_crossing(xs, ys, thresh):
    for x, y in zip(xs, ys):
        if y >= thresh:
            return x
    return None


def _panel_margin(ax, xs, groups, layers):
    steer = _line(ax, xs, _stat(groups, layers, lambda r: r["steer"]["margin_spent"]),
                  C_STEER, "steering")
    rand = _line(ax, xs, _stat(groups, layers, lambda r: r["rand"]["margin_spent"]),
                 C_RAND, "random (control)", ls="--", band=False)
    # Both curves: the control can outrun the steer, and clipping it off the top
    # would read as the control flattening out.
    top = max(max(steer), max(rand), THM_BOUND) * 1.1
    ax.axhspan(THM_BOUND, top, color=C_STEER, alpha=0.06)
    ax.axhline(THM_BOUND, color=C_STEER, ls="--", lw=1.2, label=f"Thm 3.2 bound ({THM_BOUND})")
    ax.set_ylim(0, top)

    cross = _first_crossing(xs, steer, THM_BOUND)
    if cross is not None:
        ax.axvline(cross, color="0.5", ls=":", lw=1)
        ax.annotate(f"crosses at\nlayer {cross}", xy=(cross, THM_BOUND),
                    xytext=(cross - 6, THM_BOUND + (top - THM_BOUND) * 0.35),
                    fontsize=7, color="0.3",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_title("Recovery margin spent by a fixed steer")
    ax.set_ylabel("residual / gap")
    ax.legend(fontsize=7, loc="upper left")


def _panel_mechanism(ax, xs, groups, layers):
    _line(ax, xs, _stat(groups, layers, lambda r: r["rel_gap"]),
          C_GAP, "gap / ‖h‖  (room, collapses)")
    _line(ax, xs, _stat(groups, layers, lambda r: r["steer"]["rel_residual"]),
          C_RES, "residual / ‖h‖  (steer size, flat)")
    ax.set_title("Why: gap collapses, residual is pinned")
    ax.set_ylabel("fraction of ‖h‖")
    ax.annotate("margin spent = residual / gap", xy=(0.5, 0.92),
                xycoords="axes fraction", ha="center", fontsize=8, color="0.3")
    ax.legend(fontsize=7, loc="upper right")


def _panel_detection(ax, xs, groups, layers, rel_tol):
    res = _line(ax, xs, _stat(groups, layers, lambda r: r["steer"]["rel_residual"]),
                C_RES, "steered residual / ‖h‖")
    ax.axhline(rel_tol, color=C_STEER, ls="--", lw=1.2,
               label=f"alarm floor (rel_tol={rel_tol:g})")
    ax.axhspan(ax.get_ylim()[0], rel_tol, color=C_STEER, alpha=0.06)
    ax.set_yscale("log")
    factor = (sum(res) / len(res)) / rel_tol
    ax.annotate(f"~{factor:.0f}× above the floor\n→ detected everywhere",
                xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=8,
                color="0.3", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_title("Detection headroom")
    ax.set_ylabel("residual / ‖h‖  (log)")
    ax.legend(fontsize=7, loc="lower left")


def _panel_direction(ax, xs, groups, layers):
    diff = []
    for L in layers:
        g = groups[L]
        diff.append(sum(r["steer"]["margin_spent"] - r["rand"]["margin_spent"] for r in g) / len(g))
    ax.bar(xs, diff, color=C_STEER, width=0.7)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("Direction vs magnitude")
    ax.set_ylabel("margin spent: steer − random")
    ax.annotate("identical until the final layer\n→ size sets the cost, not alignment",
                xy=(0.03, 0.9), xycoords="axes fraction", va="top", fontsize=8, color="0.3")


def plot(rows: list[dict], out: Path, rel_tol: float, model_name: str) -> None:
    layers, groups = by_layer(rows)
    n_prompts = len({r["prompt"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel_margin(axes[0, 0], layers, groups, layers)
    _panel_mechanism(axes[0, 1], layers, groups, layers)
    _panel_detection(axes[1, 0], layers, groups, layers, rel_tol)
    _panel_direction(axes[1, 1], layers, groups, layers)
    for ax in axes.ravel():
        ax.set_xlabel("Hidden-state layer  (output of block i)")
        ax.set_xticks(layers[::2])
        ax.grid(True, alpha=0.3)

    rec = sum(r["steer"]["recovered"] for r in rows) / len(rows) * 100
    det = sum(r["steer"]["detected"] for r in rows) / len(rows) * 100
    fig.suptitle(
        f"Steering audit — {model_name}\n"
        f"{n_prompts} prompts × {len(layers)} layers   "
        f"(steer recovery {rec:.0f}%, detection {det:.0f}% throughout)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="selects the results/<slug>/ directory and titles the figure")
    p.add_argument("--in", dest="inp", type=str, default=None,
                   help="default: results/<slug>/steer_audit.jsonl")
    p.add_argument("--out", type=str, default=None,
                   help="default: results/<slug>/steer_audit.png")
    p.add_argument("--rel_tol", type=float, default=1e-3,
                   help="detection floor the audit ran with (for the headroom panel)")
    args = p.parse_args()

    slug = args.model_name.replace("/", "_")
    inp = Path(args.inp) if args.inp else Path(f"results/{slug}/steer_audit.jsonl")
    out = Path(args.out) if args.out else Path(f"results/{slug}/steer_audit.png")

    rows = load_rows(inp)
    if not rows:
        raise SystemExit(f"no rows in {inp}")
    plot(rows, out, args.rel_tol, args.model_name)


if __name__ == "__main__":
    main()
