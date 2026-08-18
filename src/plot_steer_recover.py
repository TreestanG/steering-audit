"""Plot the steering-vector recovery and layer-localization from steer_recover.py.

Reads results/<slug>/steer_recover.jsonl and steer_localize.jsonl. Four panels:

  A  Recovery direction — cos(δ̂, δ) vs layer, with margin_spent on a twin axis.
     The story is the contrast: cos stays pinned at 1 while the margin spent
     climbs past the Thm 3.2 bound, so recovery of the vector holds well beyond
     where the guarantee stops.
  B  Recovery magnitude — ‖δ̂‖/‖δ‖ vs layer. Flat at 1 until the final post-norm
     layer, where δ (injected pre-norm) is no longer directly observable.
  C  Localization staircase — residual vs layer, one curve per injection layer.
     Each sits on the numerical floor until its injection layer, then lifts off.
  D  Localization check — detected takeoff vs injected layer against identity.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THM_BOUND = 0.5
C_STEER, C_RAND, C_MARGIN = "C3", "C7", "C4"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _mean_by_layer(rows, layers, fn):
    out = []
    for L in layers:
        at = [r for r in rows if r["layer"] == L]
        out.append(sum(fn(r) for r in at) / len(at) if at else None)
    return out


def _plot_metric(ax, layers, ys, color, label, ls="-"):
    xs = [L for L, y in zip(layers, ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, ls, marker="o", ms=4, color=color, label=label)


def _panel_direction(ax, rec_rows, layers):
    cos_s = _mean_by_layer(rec_rows, layers, lambda r: r["steer"]["cos"])
    cos_r = _mean_by_layer(rec_rows, layers, lambda r: r["rand"]["cos"])
    _plot_metric(ax, layers, cos_s, C_STEER, "steering")
    _plot_metric(ax, layers, cos_r, C_RAND, "random", ls="--")
    ax.set_ylabel("cos(δ̂, δ)   direction fidelity")
    ax.set_title("Recovery: direction")

    lo = min(v for v in cos_s + cos_r if v is not None)
    ax.set_ylim(min(lo - 0.01, 0.99), 1.002)
    ax.axhline(1.0, color="0.8", lw=1, ls=":")

    rax = ax.twinx()  # margin climbs past the guarantee while cos stays at 1
    marg = _mean_by_layer(rec_rows, layers, lambda r: r["steer"]["margin_spent"])
    xs = [L for L, y in zip(layers, marg) if y is not None]
    rax.plot(xs, [y for y in marg if y is not None], color=C_MARGIN, lw=1.4, alpha=0.8,
             label="margin spent")
    rax.axhline(THM_BOUND, color=C_MARGIN, ls="--", lw=1)
    rax.set_yscale("log")  # log so the 0.2→13 climb (and its 0.5 crossing) is legible
    rax.set_ylabel("margin spent (residual/gap, log)", color=C_MARGIN)
    rax.tick_params(axis="y", labelcolor=C_MARGIN)
    rax.annotate("Thm 3.2 bound", xy=(xs[-1], THM_BOUND), ha="right", va="bottom",
                 fontsize=7, color=C_MARGIN)
    ax.legend(fontsize=7, loc="lower center")


def _panel_magnitude(ax, rec_rows, layers):
    _plot_metric(ax, layers, _mean_by_layer(rec_rows, layers, lambda r: r["steer"]["norm_ratio"]),
                 C_STEER, "steering")
    _plot_metric(ax, layers, _mean_by_layer(rec_rows, layers, lambda r: r["rand"]["norm_ratio"]),
                 C_RAND, "random", ls="--")
    ax.axhline(1.0, color="C2", ls="--", lw=1.2, label="perfect (1.0)")
    ax.set_ylabel("‖δ̂‖ / ‖δ‖   magnitude fidelity")
    ax.set_title("Recovery: magnitude")
    ax.legend(fontsize=7, loc="upper left")


def _panel_staircase(ax, loc_rows, layers):
    injects = sorted({r["inject_layer"] for r in loc_rows})
    cmap = plt.get_cmap("viridis")
    floor = None
    for i, inj in enumerate(injects):
        at = [r for r in loc_rows if r["inject_layer"] == inj]
        # average rel_steer across prompts at each layer
        prof = []
        for L in layers:
            vals = [r["rel_steer"][str(L)] for r in at if str(L) in r["rel_steer"]]
            prof.append(sum(vals) / len(vals) if vals else None)
        color = cmap(i / max(1, len(injects) - 1))
        xs = [L for L, y in zip(layers, prof) if y is not None]
        vs = [y for y in prof if y is not None]
        ax.plot(xs, vs, "-", color=color, lw=1.5, label=f"inject L{inj}")
        # mark the injection layer
        if str(inj) in at[0]["rel_steer"]:
            yv = sum(r["rel_steer"][str(inj)] for r in at) / len(at)
            ax.plot([inj], [yv], "o", color=color, ms=7, mec="k", mew=0.5, zorder=5)
        floor = max(r["floor"] for r in at) if floor is None else max(floor, max(r["floor"] for r in at))
    if floor:
        ax.axhline(floor, color="0.6", ls=":", lw=1, label="clean floor")
    ax.set_yscale("log")
    ax.set_ylabel("residual / ‖h‖")
    ax.set_title("Localization: residual takes off at the injection layer")
    ax.legend(fontsize=7, loc="lower right")


def _panel_check(ax, loc_rows):
    injects = sorted({r["inject_layer"] for r in loc_rows})
    lo = min(injects) - 1
    hi = max(injects) + 1
    ax.plot([lo, hi], [lo, hi], color="0.7", ls="--", lw=1, label="detected = injected")
    for inj in injects:
        detected = [r["takeoff"] for r in loc_rows if r["inject_layer"] == inj and r["takeoff"] is not None]
        ax.plot([inj] * len(detected), detected, "o", color=C_STEER, ms=7, alpha=0.6)
    ax.set_xlabel("injected layer")
    ax.set_ylabel("detected takeoff layer")
    ax.set_title("Localization: detected vs injected")
    ax.set_xticks(injects)
    ax.set_yticks(injects)
    ax.legend(fontsize=7, loc="upper left")


def plot(rec_rows, loc_rows, out: Path, model_name: str) -> None:
    layers = sorted({r["layer"] for r in rec_rows})
    n_prompts = len({r["prompt"] for r in rec_rows})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel_direction(axes[0, 0], rec_rows, layers)
    _panel_magnitude(axes[0, 1], rec_rows, layers)
    _panel_staircase(axes[1, 0], loc_rows, layers)
    _panel_check(axes[1, 1], loc_rows)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Hidden-state layer  (output of block i)")
        ax.set_xticks(layers[::2])
        ax.grid(True, alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    good = [r["steer"]["cos"] for r in rec_rows if r["steer"]["recovered"]]
    mean_cos = sum(good) / len(good) if good else float("nan")
    fig.suptitle(
        f"Steering recovery & localization — {model_name}\n"
        f"{n_prompts} prompts   (mean cos(δ̂,δ)={mean_cos:.4f} where token recovered)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--recover", type=str, default=None,
                   help="default: results/<slug>/steer_recover.jsonl")
    p.add_argument("--localize", type=str, default=None,
                   help="default: results/<slug>/steer_localize.jsonl")
    p.add_argument("--out", type=str, default=None,
                   help="default: results/<slug>/steer_recover.png")
    args = p.parse_args()

    slug = args.model_name.replace("/", "_")
    rec = Path(args.recover) if args.recover else Path(f"results/{slug}/steer_recover.jsonl")
    loc = Path(args.localize) if args.localize else Path(f"results/{slug}/steer_localize.jsonl")
    out = Path(args.out) if args.out else Path(f"results/{slug}/steer_recover.png")

    rec_rows, loc_rows = load(rec), load(loc)
    if not rec_rows:
        raise SystemExit(f"no rows in {rec}")
    if not loc_rows:
        raise SystemExit(f"no rows in {loc}")
    plot(rec_rows, loc_rows, out, args.model_name)


if __name__ == "__main__":
    main()
