"""Shared pieces for the plot_*.py figures: palette, IO, axis styling, aggregation."""

from pathlib import Path
import json

from log import get_logger

# Thm 3.2: exact recovery guaranteed while residual/gap < 1/2.
logger = get_logger(__name__)

# Thm 3.2: exact recovery guaranteed while residual/gap < 1/2.
THM_BOUND = 0.5

# One palette across every figure, so a colour means the same thing everywhere.
C_SEP, C_STEER, C_RAND, C_GAP, C_RES, C_MARGIN = "C9", "C3", "C7", "C9", "C1", "C4"


def load_rows(path: Path) -> list[dict]:
    """One JSON object per line, blank lines skipped."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def xy(xs: list, ys: list) -> tuple[list, list]:
    """Drop positions where y is None, keeping x and y aligned. ([], []) if all None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def style_layer_axis(ax, layers: list[int]) -> None:
    ax.set_xlabel("Hidden-state layer  (output of block i)")
    ax.set_xticks(layers[::2] if len(layers) > 14 else layers)
    ax.grid(True, alpha=0.3)


def save_fig(fig, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("saved %s", out)


def first_crossing(xs: list, ys: list, thresh: float):
    """First x whose y reaches thresh, or None. Assumes xs is ascending."""
    for x, y in zip(xs, ys):
        if y >= thresh:
            return x
    return None


def group_by_layer(rows: list[dict]) -> tuple[list[int], dict[int, list[dict]]]:
    layers = sorted({r["layer"] for r in rows})
    return layers, {L: [r for r in rows if r["layer"] == L] for L in layers}


def stat(groups: dict[int, list[dict]], layers: list[int], fn):
    """(mean, lo, hi) across rows per layer; band is min-max since n is small."""
    mean, lo, hi = [], [], []
    for L in layers:
        vals = [fn(r) for r in groups[L]]
        mean.append(sum(vals) / len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return mean, lo, hi


def mean_by_layer(rows: list[dict], layers: list[int], fn) -> list[float | None]:
    """Per-layer mean, None where a layer has no rows."""
    _, groups = group_by_layer(rows)
    return [
        sum(fn(r) for r in groups[L]) / len(groups[L]) if groups.get(L) else None
        for L in layers
    ]
