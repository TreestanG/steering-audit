import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THM_BOUND = 0.5  # Thm 3.2: exact recovery guaranteed while residual/gap < 1/2.
FRACTION_RE = re.compile(r"steer_audit_f([0-9.eE+-]+)\.jsonl$")


def load_sweep(in_dir: Path) -> dict[float, list[dict]]:
    """{fraction: rows}, keyed by the fraction encoded in each filename."""
    out: dict[float, list[dict]] = {}
    for path in sorted(in_dir.glob("steer_audit_f*.jsonl")):
        m = FRACTION_RE.search(path.name)
        if m is None:
            print(f"skipping {path.name}: no fraction in filename")
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            out[float(m.group(1))] = rows
    return dict(sorted(out.items()))


def per_layer(rows: list[dict], fn) -> tuple[list[int], list[float]]:
    """Mean of fn over prompts, per layer."""
    layers = sorted({r["layer"] for r in rows})
    means = []
    for layer in layers:
        vals = [fn(r) for r in rows if r["layer"] == layer]
        means.append(sum(vals) / len(vals))
    return layers, means


def first_crossing(layers: list[int], ys: list[float], thresh: float) -> int | None:
    for layer, y in zip(layers, ys):
        if y >= thresh:
            return layer
    return None


def colors_for(fractions: list[float]) -> dict[float, tuple]:
    """Log-spaced along viridis — the fractions span 200x, so linear would bunch."""
    logs = np.log10(fractions)
    span = logs.max() - logs.min()
    norm = (logs - logs.min()) / span if span > 0 else np.zeros_like(logs)
    return {f: plt.cm.viridis(0.05 + 0.85 * t) for f, t in zip(fractions, norm)}


def _panel_margin(ax, sweep, colors):
    for frac, rows in sweep.items():
        layers, ys = per_layer(rows, lambda r: r["steer"]["margin_spent"])
        ax.plot(layers, ys, "-o", color=colors[frac], ms=3, lw=1.4, label=f"{frac:g}")
    # Log scale first: the shaded band is drawn in data coords, so it has to be
    # sized against the log-autoscaled limits, not the linear ones.
    ax.set_yscale("log")
    ax.axhspan(THM_BOUND, ax.get_ylim()[1], color="C3", alpha=0.06, zorder=0)
    ax.axhline(THM_BOUND, color="C3", ls="--", lw=1.2, zorder=1)
    ax.annotate(f"Thm 3.2 bound ({THM_BOUND})  — above: recovery no longer guaranteed",
                xy=(0.02, THM_BOUND), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7, color="C3")
    ax.set_title("Recovery margin spent, by steering strength")
    ax.set_ylabel("residual / gap  (log)")
    ax.legend(fontsize=6, title="fraction", title_fontsize=7, ncol=2, loc="upper left")


def _panel_recovery(ax, sweep, colors):
    for frac, rows in sweep.items():
        layers, ys = per_layer(rows, lambda r: float(r["steer"]["recovered"]))
        ax.plot(layers, [y * 100 for y in ys], "-o", color=colors[frac], ms=3, lw=1.4,
                label=f"{frac:g}")
    ax.set_ylim(-5, 105)
    ax.set_title("What actually happens: steered recovery rate")
    ax.set_ylabel("exact token recovered (%)")
    ax.legend(fontsize=6, title="fraction", title_fontsize=7, ncol=2, loc="lower left")


def _panel_residual(ax, sweep, colors, rel_tol):
    for frac, rows in sweep.items():
        layers, ys = per_layer(rows, lambda r: r["steer"]["rel_residual"])
        ax.plot(layers, ys, "-o", color=colors[frac], ms=3, lw=1.4, label=f"{frac:g}")
    ax.axhline(rel_tol, color="C3", ls="--", lw=1.2)
    ax.annotate(f"alarm floor (rel_tol={rel_tol:g})", xy=(0.02, rel_tol),
                xycoords=("axes fraction", "data"), va="bottom", fontsize=7, color="C3")
    ax.set_yscale("log")
    ax.set_title("Perturbation size is pinned by the fraction")
    ax.set_ylabel("residual / ‖h‖  (log)")
    ax.legend(fontsize=6, title="fraction", title_fontsize=7, ncol=2, loc="lower left")


def _panel_safe_depth(ax, sweep, colors, max_layer):
    fracs, crossings, labels = [], [], []
    for frac, rows in sweep.items():
        layers, ys = per_layer(rows, lambda r: r["steer"]["margin_spent"])
        cross = first_crossing(layers, ys, THM_BOUND)
        fracs.append(frac)
        # Never crossing means every layer is guaranteed; plot it one past the
        # last so the bar reads as "clears the whole stack", not "crosses at 24".
        crossings.append(max_layer + 1 if cross is None else cross)
        labels.append("all safe" if cross is None else str(cross))

    xs = np.arange(len(fracs))
    ax.bar(xs, crossings, color=[colors[f] for f in fracs], width=0.7)
    for x, y, label in zip(xs, crossings, labels):
        ax.annotate(label, xy=(x, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color="0.3")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f:g}" for f in fracs], fontsize=7)
    ax.set_ylim(0, max_layer + 3)
    ax.set_title("Guaranteed-safe depth vs steering strength")
    ax.set_xlabel("steering fraction (of mean ‖h‖)")
    ax.set_ylabel("first layer with margin ≥ 0.5")


def plot(sweep: dict[float, list[dict]], out: Path, rel_tol: float, model_name: str) -> None:
    fractions = list(sweep)
    colors = colors_for(fractions)
    all_rows = [r for rows in sweep.values() for r in rows]
    layers = sorted({r["layer"] for r in all_rows})
    n_prompts = len({r["prompt"] for r in all_rows})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel_margin(axes[0, 0], sweep, colors)
    _panel_recovery(axes[0, 1], sweep, colors)
    _panel_residual(axes[1, 0], sweep, colors, rel_tol)
    _panel_safe_depth(axes[1, 1], sweep, colors, max(layers))
    for ax in axes.ravel()[:3]:
        ax.set_xlabel("Hidden-state layer  (output of block i)")
        ax.set_xticks(layers[::2])
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    rec = sum(r["steer"]["recovered"] for r in all_rows) / len(all_rows) * 100
    fig.suptitle(
        f"Steering strength sweep — {model_name}\n"
        f"{len(fractions)} fractions × {n_prompts} prompts × {len(layers)} layers   "
        f"(overall steered recovery {rec:.0f}%)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="selects the results/<slug>/ directory and titles the figure")
    p.add_argument("--in_dir", type=str, default=None,
                   help="default: results/<slug>/fractions")
    p.add_argument("--out", type=str, default=None,
                   help="default: results/<slug>/steer_fraction_sweep.png")
    p.add_argument("--rel_tol", type=float, default=1e-3,
                   help="detection floor the audits ran with")
    args = p.parse_args()

    slug = args.model_name.replace("/", "_")
    in_dir = Path(args.in_dir) if args.in_dir else Path(f"results/{slug}/fractions")
    out = Path(args.out) if args.out else Path(f"results/{slug}/steer_fraction_sweep.png")

    sweep = load_sweep(in_dir)
    if not sweep:
        raise SystemExit(f"no steer_audit_f*.jsonl in {in_dir}")
    print(f"loaded fractions: {', '.join(f'{f:g}' for f in sweep)}")
    plot(sweep, out, args.rel_tol, args.model_name)


if __name__ == "__main__":
    main()
