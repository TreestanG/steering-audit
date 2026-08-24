"""Per-trajectory detection: aggregate the per-position residuals, threshold at k sigma.

The audit's native decision is per POSITION -- residual > rel_tol * ||h|| at one token.
A deployed detector watches a whole generation, so that rule has to be aggregated, and
the aggregation is not a formality: at the shipped fp16 threshold the per-position false
positive rate is 0.06%, which over a 50-token completion audited at 24 layers compounds
to roughly half of all clean prompts if the rule is "flag if ANY position fires".

The rule here is the mean of the top k relative residuals in a trajectory, flagged when
it exceeds mean + sigma * sd of the same statistic on clean text. Top-k sits between the
two degenerate choices: k=1 is the max, which is what "any position fires" already is and
is set by the single worst token; k=len is the mean, which a one-position injection
disappears into. k>1 also stops one unlucky token from carrying the whole decision.

Calibration is per (model, layer) because the clean relative residual grows with depth --
scale-normalising by ||h|| does not remove that trend, so one threshold across layers is
loose at the top and tight at the bottom.
"""

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)

TOPK = 5
SIGMA = 3.0
IN_DISTRIBUTION = "natural_en"


def relative_residuals(steps: list[dict]) -> list[float]:
    return [s["residual"] / s["h_norm"] for s in steps if s.get("h_norm")]


def topk_mean(values: list[float], k: int = TOPK) -> float:
    """Mean of the k largest values; the whole trajectory when it is shorter than k."""
    if not values:
        return float("nan")
    top = sorted(values, reverse=True)[:max(1, k)]
    return sum(top) / len(top)


def any_position_fires(steps: list[dict]) -> bool:
    """The rule as it stands today, lifted to the trajectory: one unmatched step is enough."""
    return any(not s["matched"] for s in steps)


@dataclass
class Calibration:
    n: int
    k: int
    sigma: float
    mean: float
    sd: float
    threshold: float

    def flags(self, score: float) -> bool:
        return score > self.threshold


def calibrate(scores: list[float], k: int = TOPK, sigma: float = SIGMA) -> Calibration | None:
    clean = [s for s in scores if not math.isnan(s)]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    return Calibration(n=len(clean), k=k, sigma=sigma, mean=mean, sd=sd,
                       threshold=mean + sigma * sd)


def load_trajectories(layers_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(layers_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_rows(rows: list[dict], k: int) -> list[dict]:
    out = []
    for row in rows:
        residuals = relative_residuals(row.get("steps", []))
        if not residuals:
            continue
        out.append({
            "id": row.get("id"),
            "category": row.get("category"),
            "layer": row["layer"],
            "n_steps": len(residuals),
            "score": topk_mean(residuals, k),
            "max": max(residuals),
            "fires_any": any_position_fires(row["steps"]),
        })
    return out


def _rate(hits: int, n: int) -> float:
    return hits / n if n else float("nan")


def evaluate(scored: list[dict], k: int, sigma: float, calibrate_on: str) -> dict:
    """Calibrate per layer, then score every trajectory at that layer against it."""
    layers = sorted({s["layer"] for s in scored})
    per_layer, flagged_ids, all_ids = {}, set(), set()
    by_category: dict[str, list[int]] = {}
    n_flagged = n_total = n_any = 0

    for layer in layers:
        at = [s for s in scored if s["layer"] == layer]
        basis = [s for s in at if s["category"] == calibrate_on] if calibrate_on != "all" else at
        cal = calibrate([s["score"] for s in basis], k, sigma)
        if cal is None:
            continue
        hits = 0
        for s in at:
            fired = cal.flags(s["score"])
            hits += fired
            n_flagged += fired
            n_any += s["fires_any"]
            n_total += 1
            all_ids.add(s["id"])
            if fired:
                flagged_ids.add(s["id"])
            bucket = by_category.setdefault(s["category"] or "?", [0, 0, 0])
            bucket[0] += fired
            bucket[1] += 1
            bucket[2] += s["fires_any"]
        per_layer[layer] = {**asdict(cal), "n_at_layer": len(at), "flagged": hits,
                            "fpr": _rate(hits, len(at))}

    return {
        "k": k,
        "sigma": sigma,
        "calibrated_on": calibrate_on,
        "n_trajectory_layer_pairs": n_total,
        "fpr_topk": _rate(n_flagged, n_total),
        "fpr_any_position": _rate(n_any, n_total),
        "fpr_topk_any_layer": _rate(len(flagged_ids), len(all_ids)),
        "n_trajectories": len(all_ids),
        "per_category": {
            c: {"n": v[1], "fpr_topk": _rate(v[0], v[1]),
                "fpr_any_position": _rate(v[2], v[1])}
            for c, v in sorted(by_category.items())
        },
        "per_layer": per_layer,
    }


def min_detectable(residuals: list[float], threshold: float, k: int) -> tuple[float, float]:
    """Smallest injection this rule would catch, in two regimes.

    Returns (one_position, all_positions):
      one_position  -- the relative residual a SINGLE steered token must reach. The
                       other positions keep their clean values and the injected one
                       replaces the least suspicious of them.
      all_positions -- the factor every position must be multiplied by, which is what
                       steering at --positions all does.

    Both are closed form: the top-k mean is linear in the values it averages. The pair
    is the cost of k -- averaging k positions divides a one-token spike by k while
    leaving an all-position steer untouched.
    """
    if not residuals or threshold <= 0:
        return float("nan"), float("nan")
    ke = min(k, len(residuals))
    others = sorted(residuals, reverse=True)[:len(residuals) - 1]
    one = ke * threshold - sum(others[:ke - 1])
    clean = topk_mean(residuals, k)
    return one, threshold / clean if clean > 0 else float("nan")


def sensitivity(scored: list[dict], rows: list[dict], k: int, sigma: float,
                calibrate_on: str) -> dict:
    by_key = {(r["layer"], r.get("id")): r for r in rows}
    ones, mults, cleans = [], [], []
    for layer in sorted({s["layer"] for s in scored}):
        at = [s for s in scored if s["layer"] == layer]
        basis = [s for s in at if s["category"] == calibrate_on] if calibrate_on != "all" else at
        cal = calibrate([s["score"] for s in basis], k, sigma)
        if cal is None:
            continue
        for s in at:
            row = by_key.get((layer, s["id"]))
            if row is None:
                continue
            residuals = relative_residuals(row.get("steps", []))
            one, mult = min_detectable(residuals, cal.threshold, k)
            if not math.isnan(one):
                ones.append(one)
            if not math.isnan(mult):
                mults.append(mult)
            cleans.append(s["score"])
    return {
        "median_clean_score": statistics.median(cleans) if cleans else float("nan"),
        "median_min_single_position": statistics.median(ones) if ones else float("nan"),
        "median_min_all_position_factor": statistics.median(mults) if mults else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the per-trajectory detector on the clean SipIt inversions "
                    "already on disk, and measure the false-positive rate it buys.")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=TOPK,
                        help=f"positions averaged, largest first (default {TOPK}). "
                             f"1 reproduces the max / 'any position fires' rule")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help=f"threshold is mean + sigma*sd of the clean statistic "
                             f"(default {SIGMA})")
    parser.add_argument("--calibrate_on", type=str, default=IN_DISTRIBUTION,
                        help="category whose clean trajectories set mean and sd, or 'all'. "
                             "Calibrating on one category and scoring the rest is the "
                             "held-out number; 'all' is in-sample and optimistic")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also report the smallest injection the rule would catch, "
                             "for a single steered position and for an all-position "
                             "steer. This is what choosing k costs")
    parser.add_argument("--slug", type=str, default=None, help="one model directory only")
    parser.add_argument("--out_name", type=str, default="detector_calibration.json")
    parser.add_argument("--no_write", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=args.results_root / "all" / "logs" / "detect.log")

    dirs = sorted(p for p in args.results_root.glob("*/sipit/layers") if p.is_dir())
    if args.slug:
        dirs = [p for p in dirs if p.parts[-3] == args.slug]
    if not dirs:
        raise SystemExit(f"no */sipit/layers under {args.results_root}")

    logger.info("rule: mean of the top %d relative residuals per trajectory, flagged above "
                "mean + %g sd of the clean statistic, calibrated per layer on %r",
                args.k, args.sigma, args.calibrate_on)
    logger.info("%-32s %7s %8s %11s %13s %12s", "model", "trajs", "pairs",
                "FPR top-k", "FPR any-pos", "FPR any-layer")
    summary = {}
    for layers_dir in dirs:
        slug = layers_dir.parts[-3]
        rows = load_trajectories(layers_dir)
        scored = score_rows(rows, args.k)
        if not scored:
            continue
        report = evaluate(scored, args.k, args.sigma, args.calibrate_on)
        if args.sensitivity:
            report["sensitivity"] = sensitivity(scored, rows, args.k, args.sigma,
                                                args.calibrate_on)
        summary[slug] = report
        logger.info("%-32s %7d %8d %10.2f%% %12.2f%% %11.2f%%", slug,
                    report["n_trajectories"], report["n_trajectory_layer_pairs"],
                    100 * report["fpr_topk"], 100 * report["fpr_any_position"],
                    100 * report["fpr_topk_any_layer"])
        if not args.no_write:
            out = layers_dir.parent / args.out_name
            out.write_text(json.dumps(report, indent=2) + "\n")

    if args.sensitivity:
        logger.info("")
        logger.info("smallest injection caught (median over trajectories)")
        logger.info("%-32s %14s %16s %14s", "model", "clean score",
                    "1 position needs", "all-pos factor")
        for slug, report in summary.items():
            s = report["sensitivity"]
            logger.info("%-32s %14.2e %16.2e %13.1fx", slug, s["median_clean_score"],
                        s["median_min_single_position"],
                        s["median_min_all_position_factor"])

    logger.info("")
    logger.info("by category, pooled over models (trajectory-layer pairs)")
    logger.info("%-14s %8s %11s %13s", "category", "n", "FPR top-k", "FPR any-pos")
    pooled: dict[str, list[float]] = {}
    for report in summary.values():
        for cat, v in report["per_category"].items():
            acc = pooled.setdefault(cat, [0, 0, 0])
            acc[0] += v["fpr_topk"] * v["n"]
            acc[1] += v["n"]
            acc[2] += v["fpr_any_position"] * v["n"]
    for cat, (hits, n, any_hits) in sorted(pooled.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
        logger.info("%-14s %8d %10.2f%% %12.2f%%", cat, n, 100 * hits / n, 100 * any_hits / n)
    if not args.no_write:
        logger.info("wrote %s per model", args.out_name)


if __name__ == "__main__":
    main()
