import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import logs_dir, experiment_dir, figures_dir
from log import add_logging_args, get_logger
from log import setup as log_setup
from plot_common import (
    THM_BOUND,
    C_GAP,
    C_RAND,
    C_RES,
    C_STEER,
    first_crossing,
    group_by_layer,
    load_rows,
    save_fig,
    stat,
    style_layer_axis,
)


logger = get_logger(__name__)


def _line(ax, xs, band_stat, color, label, ls="-", band=True):
    mean, lo, hi = band_stat
    ax.plot(xs, mean, ls + "o" if ls == "-" else ls, color=color, ms=4, label=label)
    if band:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
    return mean


def _panel_margin(ax, groups, layers):
    steer = _line(ax, layers, stat(groups, layers, lambda r: r["steer"]["margin_spent"]),
                  C_STEER, "steering")
    rand = _line(ax, layers, stat(groups, layers, lambda r: r["rand"]["margin_spent"]),
                 C_RAND, "random (control)", ls="--", band=False)
    # Both curves: the control can outrun the steer, and clipping it off the top
    # would read as the control flattening out.
    top = max(max(steer), max(rand), THM_BOUND) * 1.1
    ax.axhspan(THM_BOUND, top, color=C_STEER, alpha=0.06)
    ax.axhline(THM_BOUND, color=C_STEER, ls="--", lw=1.2, label=f"Thm 3.2 bound ({THM_BOUND})")
    ax.set_ylim(0, top)

    cross = first_crossing(layers, steer, THM_BOUND)
    if cross is not None:
        ax.axvline(cross, color="0.5", ls=":", lw=1)
        ax.annotate(f"crosses at\nlayer {cross}", xy=(cross, THM_BOUND),
                    xytext=(cross - 6, THM_BOUND + (top - THM_BOUND) * 0.35),
                    fontsize=7, color="0.3",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_title("Recovery margin spent by a fixed steer")
    ax.set_ylabel("residual / gap")
    ax.legend(fontsize=7, loc="upper left")


def _panel_mechanism(ax, groups, layers):
    _line(ax, layers, stat(groups, layers, lambda r: r["rel_gap"]),
          C_GAP, "gap / ‖h‖  (room, collapses)")
    _line(ax, layers, stat(groups, layers, lambda r: r["steer"]["rel_residual"]),
          C_RES, "residual / ‖h‖  (steer size, flat)")
    ax.set_title("Why: gap collapses, residual is pinned")
    ax.set_ylabel("fraction of ‖h‖")
    ax.annotate("margin spent = residual / gap", xy=(0.5, 0.92),
                xycoords="axes fraction", ha="center", fontsize=8, color="0.3")
    ax.legend(fontsize=7, loc="upper right")


def _panel_detection(ax, groups, layers, rel_tol):
    res = _line(ax, layers, stat(groups, layers, lambda r: r["steer"]["rel_residual"]),
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


def _panel_direction(ax, groups, layers):
    diff, _, _ = stat(groups, layers, lambda r: r["steer"]["margin_spent"] - r["rand"]["margin_spent"])
    ax.bar(layers, diff, color=C_STEER, width=0.7)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("Direction vs magnitude")
    ax.set_ylabel("margin spent: steer − random")
    ax.annotate("identical until the final layer\n→ size sets the cost, not alignment",
                xy=(0.03, 0.9), xycoords="axes fraction", va="top", fontsize=8, color="0.3")


def plot(rows: list[dict], out: Path, rel_tol: float, model_name: str) -> None:
    layers, groups = group_by_layer(rows)
    n_prompts = len({r["prompt"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel_margin(axes[0, 0], groups, layers)
    _panel_mechanism(axes[0, 1], groups, layers)
    _panel_detection(axes[1, 0], groups, layers, rel_tol)
    _panel_direction(axes[1, 1], groups, layers)
    for ax in axes.ravel():
        style_layer_axis(ax, layers)

    rec = sum(r["steer"]["recovered"] for r in rows) / len(rows) * 100
    det = sum(r["steer"]["detected"] for r in rows) / len(rows) * 100
    fig.suptitle(
        f"Steering audit — {model_name}\n"
        f"{n_prompts} prompts × {len(layers)} layers   "
        f"(steer recovery {rec:.0f}%, detection {det:.0f}% throughout)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="selects the results/<slug>/ directory and titles the figure")
    p.add_argument("--in", dest="inp", type=str, default=None,
                   help="default: results/<slug>/steer/audit.jsonl")
    p.add_argument("--out", type=str, default=None,
                   help="default: results/<slug>/steer/figures/audit.png")
    p.add_argument("--rel_tol", type=float, default=1e-3,
                   help="detection floor the audit ran with (for the headroom panel)")
    add_logging_args(p)
    args = p.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "plot_audit.log")

    inp = Path(args.inp) if args.inp else experiment_dir(args.model_name, "steer") / "audit.jsonl"
    out = Path(args.out) if args.out else figures_dir(args.model_name, "steer") / "audit.png"

    rows = load_rows(inp)
    if not rows:
        raise SystemExit(f"no rows in {inp}")
    plot(rows, out, args.rel_tol, args.model_name)


if __name__ == "__main__":
    main()
