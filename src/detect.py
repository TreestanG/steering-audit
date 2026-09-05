
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



# (turn-start, turn-end) delimiter pairs, tried in order against the model's own
# rendered template. Qwen/ChatML first, then gemma, Llama-3, Mistral.
KNOWN_CHAT_DELIMS = (("<|im_start|>", "<|im_end|>"),
                     ("<start_of_turn>", "<end_of_turn>"),
                     ("<|start_header_id|>", "<|eot_id|>"),
                     ("[INST]", "[/INST]"))


def chat_template_spec(tokenizer, start: str | None = None, end: str | None = None):
    """((start_id, end_id, user_role_id), header_len) read off the tokenizer's template.

    header_len is measured, not assumed: it is the number of tokens from the user
    turn's start marker to the first content token (marker + role name + newline on
    every template here, but templates are free to differ).
    """
    if not getattr(tokenizer, "chat_template", None):
        # a base model with no template: every position is content, which is what
        # position_roles does with chat_ids=None. role_z still applies (per-position
        # z), it just has one content role per layer instead of template roles.
        return None, 3
    sentinel = "\u00a7CONTENT\u00a7"
    text = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}],
                                         tokenize=False, add_generation_prompt=True)
    pairs = [(start, end)] if start and end else [
        p for p in KNOWN_CHAT_DELIMS if p[0] in text and p[1] in text]
    if not pairs:
        raise SystemExit(
            f"no known turn delimiters in this template; pass --chat_start/--chat_end.\n"
            f"rendered: {text[:200]!r}")
    start, end = pairs[0]
    ids = [tokenizer.convert_tokens_to_ids(t) for t in (start, end)]
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise SystemExit(f"delimiters {start!r}/{end!r} are not single tokens here")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    at = text.index(sentinel)
    content_tok = next((j for j, (a, _) in enumerate(enc["offset_mapping"]) if a == at), None)
    marker_tok = next((j for j, t in reversed(list(enumerate(enc["input_ids"])))
                       if t == ids[0] and (content_tok is None or j < content_tok)), None)
    if content_tok is None or marker_tok is None:
        raise SystemExit(f"could not locate the user turn in: {text[:200]!r}")
    # the role-name token that follows the start marker ('user'). The delimiter alone
    # is ambiguous on a TRUNCATED trajectory, where the trailing generation-prompt
    # marker is missing and the last-but-one start marker is the SYSTEM turn.
    user_role = enc["input_ids"][marker_tok + 1]
    return (ids[0], ids[1], user_role), content_tok - marker_tok


CONTENT_SCHEMES = ("pooled", "end")
TRANSFORMS = ("none", "log")
SD_FLOOR_RAW = 1e-5
SD_FLOOR_LOG = 0.05


def position_roles(tokens: list[int], chat_ids: tuple[int, int] | None,
                   content_scheme: str = "pooled", end_cap: int = 6,
                   header_len: int = 3) -> list[str]:
    n = len(tokens)

    def plain(i: int) -> str:
        if content_scheme == "end":
            return f"content_e{min(n - 1 - i, end_cap)}"
        return "content"

    if not chat_ids:
        return [plain(i) for i in range(n)]
    start_id, end_id = chat_ids[0], chat_ids[1]
    user_id = chat_ids[2] if len(chat_ids) > 2 else None
    ends = [i for i, t in enumerate(tokens) if t == end_id]
    # The user turn is the LAST start marker followed by the 'user' role token. Keying
    # on the role token rather than the delimiter's position is what makes this safe on
    # a truncated trajectory, where the trailing generation-prompt marker is absent and
    # the last-but-one delimiter is the SYSTEM turn. Qwen (3 starts, 2 ends) lands on
    # its old starts[1]/ends[1] exactly; gemma has no system turn and only ONE end
    # marker, which the previous len(ends) < 2 guard sent down the bare-text path.
    if user_id is None:
        starts = [i for i, t in enumerate(tokens) if t == start_id]
        u = starts[-2] if len(starts) >= 2 else None
    else:
        heads = [i for i, t in enumerate(tokens[:-1])
                 if t == start_id and tokens[i + 1] == user_id]
        u = heads[-1] if heads else None
    after = [e for e in ends if e > u] if u is not None else []
    if u is None or not after:
        return [plain(i) for i in range(n)]
    c0, c1 = u + header_len, after[0]   # header is marker + role name + newline
    if not 0 < c0 < c1 <= n:
        return [plain(i) for i in range(n)]

    def content_role(i: int) -> str:
        # the edges of the user turn have their own clean distributions: the first
        # content token sits +0.45 sd above pooled content, the second is the
        # heaviest-tailed position in the sequence (max clean z 6.8), the last sits
        # -0.8 sd below. Pooling them is what made the k=1 clean tail heavy.
        if i == c0:
            return "content_first"
        if i == c0 + 1:
            return "content_second"
        if i == c1 - 1:
            return "content_last"
        return plain(i)

    return [f"pre{i}" if i < c0 else (content_role(i) if i < c1 else f"suf{i - n}")
            for i in range(n)]


def _transform(x: float, transform: str) -> float | None:
    if transform == "log":
        return math.log(x) if x > 0 else None
    return x


def _role_inputs(row: dict, chat_ids, content_scheme: str = "pooled",
                 end_cap: int = 6, header_len: int = 3) -> tuple[list[float], list[str]]:
    steps = [s for s in row.get("steps", []) if s.get("h_norm")]
    return ([s["residual"] / s["h_norm"] for s in steps],
            position_roles([s["token"] for s in steps], chat_ids, content_scheme,
                           end_cap, header_len))


def fit_role_stats(rows: list[dict], chat_ids, sd_floor: float, transform: str = "none",
                   content_scheme: str = "pooled", end_cap: int = 6,
                   header_len: int = 3) -> dict:
    acc: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        for x, role in zip(*_role_inputs(row, chat_ids, content_scheme, end_cap,
                                         header_len)):
            v = _transform(x, transform)
            if v is not None:
                acc.setdefault((row["layer"], role), []).append(v)
    out: dict[str, dict] = {}
    for (layer, role), v in acc.items():
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        out.setdefault(str(layer), {})[role] = {
            "n": len(v), "mean": statistics.fmean(v), "sd": max(sd, sd_floor)}
    return out


def role_z_score(row: dict, cal: dict, k: int) -> float:
    transform = cal.get("transform", "none")
    res, roles = _role_inputs(row, tuple(cal["chat_ids"]) if cal.get("chat_ids") else None,
                              cal.get("content_scheme", "pooled"), cal.get("end_cap", 6),
                              cal.get("chat_header_len", 3))
    st = cal["role_stats"].get(str(row["layer"]), {})
    z = []
    for x, r in zip(res, roles):
        v = _transform(x, transform)
        if r in st and v is not None:
            z.append((v - st[r]["mean"]) / st[r]["sd"])
    return topk_mean(z, k) if z else float("nan")


def row_score(row: dict, k: int, cal: dict | None = None) -> float:
    if cal and cal.get("statistic") == "role_z":
        return role_z_score(row, cal, k)
    residuals = relative_residuals(row.get("steps", []))
    return topk_mean(residuals, k) if residuals else float("nan")


def score_rows(rows: list[dict], k: int, cal: dict | None = None) -> list[dict]:
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
            "score": row_score(row, k, cal),
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
    if not residuals or threshold <= 0:
        return float("nan"), float("nan")
    ke = min(k, len(residuals))
    others = sorted(residuals, reverse=True)[:len(residuals) - 1]
    one = ke * threshold - sum(others[:ke - 1])
    clean = topk_mean(residuals, k)
    return one, threshold / clean if clean > 0 else float("nan")


def sensitivity(scored: list[dict], stats: dict[int, LayerStat], k: int,
                sigma: float) -> dict:
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
    parser.add_argument("--statistic", type=str, default="topk", choices=("topk", "role_z"),
                        help="role_z: z-score each position against its own clean "
                             "(layer, role) distribution before the top-k -- required "
                             "on chat-templated input, where raw top-k is a statistic "
                             "of the template (see position_roles)")
    parser.add_argument("--chat_start", type=str, default=None,
                        help="turn-start delimiter; default: auto-detected from the "
                             "model's own rendered template")
    parser.add_argument("--chat_end", type=str, default=None,
                        help="turn-end delimiter; default: auto-detected")
    parser.add_argument("--chat_model", type=str, default=None,
                        help="tokenizer whose <|im_start|>/<|im_end|> ids delimit the "
                             "template for --statistic role_z")
    parser.add_argument("--transform", type=str, default="none", choices=TRANSFORMS,
                        help="role_z: z-score the residual (none) or its log (log). log "
                             "removes the right skew behind the heavy k=1 tail: bare-text "
                             "k=1 FPR 38%% -> 7%% at sigma 3 (D2, 3 Sep 2026)")
    parser.add_argument("--content_roles", type=str, default="pooled",
                        choices=CONTENT_SCHEMES,
                        help="role_z: pool all content positions (pooled) or give the "
                             "last --end_cap positions their own baseline by offset from "
                             "the end (end). end restores single-position sensitivity "
                             "under log: pgd 0.0043 caught 0%% pooled, 100%% end")
    parser.add_argument("--end_cap", type=int, default=6)
    parser.add_argument("--sd_floor", type=float, default=None,
                        help="role_z: floor on a role's clean sd; prefix template "
                             "positions are deterministic and would otherwise divide by 0")
    parser.add_argument("--layers_subdir", type=str, default="layers",
                        help="which */sipit/<subdir> holds the clean trajectories")
    parser.add_argument("--out_name", type=str, default="detector_calibration.json")
    parser.add_argument("--no_write", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=args.results_root / "all" / "logs" / "detect.log")
    if args.sd_floor is None:
        args.sd_floor = SD_FLOOR_LOG if args.transform == "log" else SD_FLOOR_RAW

    chat_ids, header_len = None, 3
    if args.statistic == "role_z":
        if not args.chat_model:
            raise SystemExit("--statistic role_z needs --chat_model for the template ids")
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.chat_model)
        chat_ids, header_len = chat_template_spec(tk, args.chat_start, args.chat_end)
        if chat_ids is None:
            logger.info("%s has no chat template: all positions are content",
                        args.chat_model)
        else:
            logger.info("chat template: start=%r end=%r user=%r header_len=%d",
                        *(tk.convert_ids_to_tokens(i) for i in chat_ids), header_len)
    dirs = sorted(p for p in args.results_root.glob(f"*/sipit/{args.layers_subdir}")
                  if p.is_dir())

    def slug_of(layers_dir: Path) -> str:
        # the model slug is the component before 'sipit', however deep the
        # trajectories sit under it (layers/ or chat_bank/layers/)
        return layers_dir.parts[layers_dir.parts.index("sipit") - 1]

    def sipit_dir_of(layers_dir: Path) -> Path:
        return Path(*layers_dir.parts[:layers_dir.parts.index("sipit") + 1])

    if args.slug:
        dirs = [p for p in dirs if slug_of(p) == args.slug]
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
        slug = slug_of(layers_dir)
        traj = load_trajectories(layers_dir)
        role_cal = None
        if args.statistic == "role_z":
            role_cal = {"statistic": "role_z",
                        "chat_ids": list(chat_ids) if chat_ids else None,
                        "transform": args.transform, "content_scheme": args.content_roles,
                        "end_cap": args.end_cap, "chat_header_len": header_len,
                        "role_stats": fit_role_stats(traj, chat_ids, args.sd_floor,
                                                     args.transform, args.content_roles,
                                                     args.end_cap, header_len)}
            n_roles = sum(len(d) for d in role_cal["role_stats"].values())
            at_floor = sum(1 for d in role_cal["role_stats"].values() for v in d.values()
                           if v["sd"] <= args.sd_floor * (1 + 1e-9))
            if at_floor:
                logger.warning("%s: %d/%d (layer, role) cells sit at the sd floor %g; "
                               "those roles are deterministic in the fit set and any "
                               "drift there is read as (drift / floor) sigma", slug,
                               at_floor, n_roles, args.sd_floor)
        scored = score_rows(traj, args.k, role_cal)
        if not scored:
            continue
        profiles, categories, layers = build_profiles(scored)
        if layers != list(range(layers[0], layers[-1] + 1)):
            logger.warning("%s: layers %s are not contiguous, so --min_run counts "
                           "adjacency in this list rather than in depth", slug, layers)
        stats = layer_stats(scored, args.calibrate_on)
        if role_cal is not None:
            # z is already normalised per (layer, role): threshold the top-k z at sigma
            stats = {layer: LayerStat(st.n, 0.0, 1.0) for layer, st in stats.items()}
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
                "statistic": args.statistic,
                **({"chat_ids": role_cal["chat_ids"], "sd_floor": args.sd_floor,
                    "transform": args.transform, "content_scheme": args.content_roles,
                    "end_cap": args.end_cap, "chat_header_len": header_len,
                    "role_stats": role_cal["role_stats"],
                    "fit_layers_dir": str(layers_dir)} if role_cal else {}),
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
            (sipit_dir_of(layers_dir) / args.out_name).write_text(
                json.dumps(payload, indent=2) + "\n")

    if not args.no_write:
        logger.info("wrote %s per model", args.out_name)


if __name__ == "__main__":
    main()
