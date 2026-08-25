
import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)

TOPK = 5
SIGMA = 3.0
SIGMA_ANY = 4.5
MIN_RUN = 3
STRIDE = 3
DENSE_TAIL = 3


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
class LayerStat:
    n: int
    mean: float
    sd: float

    def threshold(self, sigma: float) -> float:
        return self.mean + sigma * self.sd


def layer_stat(scores: list[float]) -> LayerStat | None:
    clean = [s for s in scores if not math.isnan(s)]
    if len(clean) < 2:
        return None
    return LayerStat(len(clean), statistics.fmean(clean), statistics.stdev(clean))


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
            "residuals": residuals,
            "fires_any": any_position_fires(row["steps"]),
        })
    return out


# ------------------------------------------------------------------ layer axis

def build_profiles(scored: list[dict]) -> tuple[dict, dict, list[int]]:
    """id -> {layer: score}, id -> category, and the model's layer list in order."""
    profiles: dict[str, dict[int, float]] = {}
    categories: dict[str, str] = {}
    for s in scored:
        profiles.setdefault(s["id"], {})[s["layer"]] = s["score"]
        categories[s["id"]] = s["category"] or "?"
    layers = sorted({s["layer"] for s in scored})
    return profiles, categories, layers


def layer_stats(scored: list[dict], calibrate_on: str) -> dict[int, LayerStat]:
    stats = {}
    for layer in sorted({s["layer"] for s in scored}):
        at = [s for s in scored if s["layer"] == layer]
        basis = [s for s in at if s["category"] == calibrate_on] if calibrate_on != "all" else at
        stat = layer_stat([s["score"] for s in basis])
        if stat is not None:
            stats[layer] = stat
    return stats


def longest_run(layers: list[int], fired: set[int]) -> int:
    """Longest stretch of ADJACENT layers (adjacent in this model's layer list) that fired."""
    best = run = 0
    for layer in layers:
        run = run + 1 if layer in fired else 0
        best = max(best, run)
    return best


def scan(profile: dict[int, float], stats: dict[int, LayerStat], layers: list[int],
         sigma: float, stride: int, dense_tail: int = 0) -> tuple[set[int], set[int]]:
    """Coarse-to-fine layer scan. Returns (layers found firing, layers examined).

    Looks at every stride-th layer, then walks outward from each hit while its
    neighbours keep firing, which is what traces a run back to its injection point.
    The last `dense_tail` layers are probed exhaustively instead: no run can reach
    min_run there, clean residuals are largest there, and a stride that steps over
    them leaves the deepest layers only partly seen.
    """
    examined: set[int] = set()
    fired: set[int] = set()
    index = {layer: i for i, layer in enumerate(layers)}

    def fires(layer: int) -> bool:
        examined.add(layer)
        stat = stats.get(layer)
        score = profile.get(layer)
        if stat is None or score is None or math.isnan(score):
            return False
        return score > stat.threshold(sigma)

    cut = max(0, len(layers) - max(0, dense_tail))
    probe = list(layers[:cut:max(1, stride)]) + list(layers[cut:])
    frontier = [layer for layer in probe if fires(layer)]
    fired.update(frontier)
    while frontier:
        i = index[frontier.pop()]
        for j in (i - 1, i + 1):
            if 0 <= j < len(layers) and layers[j] not in examined:
                if fires(layers[j]):
                    fired.add(layers[j])
                    frontier.append(layers[j])
    return fired, examined


def prompt_fpr(profiles: dict, stats: dict[int, LayerStat], layers: list[int], *,
               sigma: float, min_run: int, stride: int,
               sigma_any: float = float("inf"), dense_tail: int = 0) -> dict:
    """Per-PROMPT flag rate: the quantity worth tuning, since a prompt is the unit."""
    flagged, examined_total, runs = 0, 0, []
    per_id = {}
    for pid, profile in profiles.items():
        fired, visited = scan(profile, stats, layers, sigma, stride, dense_tail)
        run = longest_run(layers, fired)
        # The lone clause reads layers the strided scan skipped, so its visits count
        # toward the work: otherwise layers_examined_frac reports a saving the rule
        # does not actually make.
        lone = False
        if math.isfinite(sigma_any):
            for layer in layers:
                visited.add(layer)
                stat, score = stats.get(layer), profile.get(layer)
                if (stat is not None and score is not None and not math.isnan(score)
                        and score > stat.threshold(sigma_any)):
                    lone = True
                    break
        hit = (run >= min_run and run > 0) or lone
        flagged += hit
        examined_total += len(visited)
        runs.append(run)
        per_id[pid] = {"longest_run": run, "n_fired": len(fired), "lone_layer": lone,
                       "examined": len(visited), "flagged": hit}
    n = len(profiles) or 1
    return {
        "n_prompts": len(profiles),
        "fpr_prompt": flagged / n,
        "mean_longest_run": statistics.fmean(runs) if runs else float("nan"),
        "max_longest_run": max(runs) if runs else 0,
        "layers_examined_frac": examined_total / (n * len(layers)) if layers else float("nan"),
        "per_id": per_id,
    }


def tune_sigma(profiles: dict, stats: dict[int, LayerStat], layers: list[int], *,
               target: float, min_run: int, stride: int, sigma_any: float,
               ratio: float, dense_tail: int = 0,
               lo: float = 0.0, hi: float = 15.0, iters: int = 40) -> float:
    """Smallest sigma whose per-PROMPT false-positive rate is at or under target.

    This is the whole point of tuning globally: applying a fixed per-layer sigma sets a
    per-layer rate and lets the prompt-level rate fall where it may.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        got = prompt_fpr(profiles, stats, layers, sigma=mid, min_run=min_run,
                         stride=stride, sigma_any=mid * ratio,
                         dense_tail=dense_tail)["fpr_prompt"]
        if got > target:
            lo = mid
        else:
            hi = mid
    return hi


def run_distribution(profiles: dict, stats: dict[int, LayerStat], layers: list[int],
                     sigma: float, stride: int, dense_tail: int = 0) -> dict[int, int]:
    counts: dict[int, int] = {}
    for profile in profiles.values():
        fired, _ = scan(profile, stats, layers, sigma, stride, dense_tail)
        run = longest_run(layers, fired)
        counts[run] = counts.get(run, 0) + 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------- position axis cost

def min_detectable(residuals: list[float], threshold: float, k: int) -> tuple[float, float]:
    """Smallest injection this rule would catch: (one position, all-position factor)."""
    if not residuals or threshold <= 0:
        return float("nan"), float("nan")
    ke = min(k, len(residuals))
    others = sorted(residuals, reverse=True)[:len(residuals) - 1]
    one = ke * threshold - sum(others[:ke - 1])
    clean = topk_mean(residuals, k)
    return one, threshold / clean if clean > 0 else float("nan")


def sensitivity(scored: list[dict], stats: dict[int, LayerStat], k: int,
                sigma: float) -> dict:
    """What choosing k costs, as the smallest injection that still crosses the threshold."""
    ones, mults = [], []
    for row in scored:
        stat = stats.get(row["layer"])
        if stat is None:
            continue
        one, mult = min_detectable(row["residuals"], stat.threshold(sigma), k)
        if not math.isnan(one):
            ones.append(one)
        if not math.isnan(mult):
            mults.append(mult)
    return {
        "median_min_single_position": statistics.median(ones) if ones else float("nan"),
        "median_min_all_position_factor": statistics.median(mults) if mults else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the per-trajectory detector on the clean SipIt inversions "
                    "already on disk, tuning to a per-prompt false-positive target.")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=TOPK,
                        help=f"positions averaged, largest first (default {TOPK})")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help=f"per-layer threshold is mean + sigma*sd (default {SIGMA}). "
                             f"Ignored when --target_fpr is given")
    parser.add_argument("--sigma_any", type=float, default=SIGMA_ANY,
                        help=f"a single layer this far above clean flags on its own, no "
                             f"run required (default {SIGMA_ANY}). Covers injections in "
                             f"the last few layers, which cannot produce a run. --target_fpr "
                             f"scales it with --sigma, keeping their ratio")
    parser.add_argument("--target_fpr", type=float, default=None,
                        help="tune sigma so the per-PROMPT false-positive rate lands here "
                             "(e.g. 0.01). This is the rate to specify: a per-layer sigma "
                             "compounds over depth into something much larger")
    parser.add_argument("--min_run", type=int, default=MIN_RUN,
                        help=f"adjacent firing layers required to flag a prompt (default "
                             f"{MIN_RUN}; 1 reproduces the old any-layer rule). An "
                             f"injection persists downstream and runs long; isolated "
                             f"spikes are usually noise")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"examine every m-th layer first, then walk outward from any "
                             f"hit to trace its run (default {STRIDE}; 1 examines every "
                             f"layer)")
    parser.add_argument("--dense_tail", type=int, default=DENSE_TAIL,
                        help=f"trailing layers examined exhaustively rather than strided "
                             f"(default {DENSE_TAIL}; 0 strides all the way down). A run "
                             f"cannot reach --min_run in the last few layers, and clean "
                             f"residuals are largest there, so striding over them is the "
                             f"worst place to be coarse")
    parser.add_argument("--calibrate_on", type=str, default="all",
                        help="category whose clean trajectories set mean and sd, or 'all' "
                             "(default). Calibrating on natural_en alone repeats finding "
                             "4's mistake: the in-distribution floor is the wrong target")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also report the smallest injection the rule would catch, for "
                             "one steered position and for an all-position steer. This is "
                             "what choosing k costs")
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

    ratio = args.sigma_any / args.sigma if args.sigma else 1.5
    logger.info("top-%d mean per trajectory; flag on >=%d adjacent firing layers or one "
                "layer at %.2g sigma; stride %d; calibrated per layer on %r", args.k,
                args.min_run, args.sigma_any, args.stride, args.calibrate_on)
    if args.target_fpr is not None:
        logger.info("tuning sigma per model to a per-prompt FPR of %.3g", args.target_fpr)
    logger.info("%-32s %6s %7s %8s %11s %11s %9s %8s", "model", "layers", "prompts",
                "sigma", "FPR/prompt", "FPR/layer", "mean run", "scanned")

    for layers_dir in dirs:
        slug = layers_dir.parts[-3]
        scored = score_rows(load_trajectories(layers_dir), args.k)
        if not scored:
            continue
        profiles, categories, layers = build_profiles(scored)
        if layers != list(range(layers[0], layers[-1] + 1)):
            logger.warning("%s: layers %s are not contiguous, so --min_run counts "
                           "adjacency in this list rather than in depth", slug, layers)
        stats = layer_stats(scored, args.calibrate_on)
        if not stats:
            continue

        sigma = args.sigma
        if args.target_fpr is not None:
            sigma = tune_sigma(profiles, stats, layers, target=args.target_fpr,
                               min_run=args.min_run, stride=args.stride,
                               sigma_any=args.sigma_any, ratio=ratio,
                               dense_tail=args.dense_tail)
        report = prompt_fpr(profiles, stats, layers, sigma=sigma,
                            min_run=args.min_run, stride=args.stride,
                            sigma_any=sigma * ratio, dense_tail=args.dense_tail)
        per_layer_hits = sum(1 for pid, p in profiles.items() for layer in layers
                             if layer in stats and not math.isnan(p.get(layer, float("nan")))
                             and p[layer] > stats[layer].threshold(sigma))
        pairs = sum(1 for p in profiles.values() for layer in layers if layer in p)

        logger.info("%-32s %6d %7d %8.2f %10.2f%% %10.2f%% %9.2f %7.0f%%", slug,
                    len(layers), report["n_prompts"], sigma,
                    100 * report["fpr_prompt"], 100 * per_layer_hits / max(1, pairs),
                    report["mean_longest_run"], 100 * report["layers_examined_frac"])

        if args.sensitivity:
            sens = sensitivity(scored, stats, args.k, sigma)
            logger.info("%-32s   one position needs %.2e, all positions %.1fx clean", "",
                        sens["median_min_single_position"],
                        sens["median_min_all_position_factor"])

        if not args.no_write:
            payload = {
                "k": args.k, "sigma": sigma, "sigma_any": sigma * ratio,
                "min_run": args.min_run,
                "stride": args.stride, "dense_tail": args.dense_tail,
                "calibrated_on": args.calibrate_on,
                "target_fpr": args.target_fpr,
                "fpr_prompt": report["fpr_prompt"],
                "fpr_layer": per_layer_hits / max(1, pairs),
                "n_prompts": report["n_prompts"],
                "layers_examined_frac": report["layers_examined_frac"],
                "run_length_histogram": run_distribution(profiles, stats, layers, sigma,
                                                        args.stride, args.dense_tail),
                "per_layer": {str(layer): {"n": s.n, "mean": s.mean, "sd": s.sd,
                                           "threshold": s.threshold(sigma)}
                              for layer, s in stats.items()},
            }
            (layers_dir.parent / args.out_name).write_text(json.dumps(payload, indent=2) + "\n")

    if not args.no_write:
        logger.info("wrote %s per model", args.out_name)


if __name__ == "__main__":
    main()
