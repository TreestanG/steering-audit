import argparse
import json
import math
from pathlib import Path

import torch

import behaviors
import prompt_format
import scoring
from behaviors import Behavior, Item, load_behavior
from generate import (
    Intervention,
    generate_completions,
    last_token_logits_batch,
    target_logprobs,
)
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import atomic_writes, behavior_dir, logs_dir
from scoring import score_completion
from steering import (
    PICK_BY,
    best_layer,
    build_steering_vectors,
    control_seed,
    efficacy,
    first_token_id,
    random_direction,
)
from utils import DTYPES, add_model_args, load_model, model_device

logger = get_logger(__name__)

ARMS = ("steer", "rand")


def _resolve_targets(behavior: Behavior, items: list[Item]) -> tuple[list[str], list[str]]:
    pos, neg = [], []
    for item in items:
        p, n = behavior.target_for(item, "pos"), behavior.target_for(item, "neg")
        if p is None or n is None:
            raise SystemExit(
                f"{behavior.name}: item {item.index} has no {'positive' if p is None else 'negative'} "
                f"target, so the gap metrics have nothing to compare. Give the dataset a "
                f"targets.{'pos' if p is None else 'neg'} or a per-item 'target'."
            )
        pos.append(p)
        neg.append(n)
    return pos, neg


def _graded(values) -> list:
    """Drop the two ways a judge says "no grade": None (binary) and NaN (score)."""
    return [v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mean(values) -> float:
    kept = _graded(values)
    return sum(kept) / len(kept) if kept else float("nan")


def _sd(values) -> float:
    kept = _graded(values)
    if len(kept) < 2:
        return float("nan")
    mu = sum(kept) / len(kept)
    return math.sqrt(sum((v - mu) ** 2 for v in kept) / (len(kept) - 1))


def sweep_grid(behavior: Behavior, prompts: list[str], pos_targets: list[str],
               neg_targets: list[str], steering: dict, *, fractions: list[float],
               layers: list[int], positions: str, seed: int, batch_size: int,
               want_target_gap: bool, n_rand: int = 3) -> list[dict]:
    hidden = next(iter(steering.values()))[0].numel()

    clean_logits = last_token_logits_batch(prompts, None, batch_size=batch_size)
    pos_ids = [first_token_id(t) for t in pos_targets]
    neg_ids = [first_token_id(t) for t in neg_targets]

    clean_target = None
    if want_target_gap:
        p = target_logprobs(prompts, pos_targets, None, batch_size=batch_size)
        n = target_logprobs(prompts, neg_targets, None, batch_size=batch_size)
        clean_target = [a["mean"] - b["mean"] for a, b in zip(p, n)]

    def measure(intervention: Intervention) -> dict:
        logits = last_token_logits_batch(prompts, intervention, batch_size=batch_size)
        stats = [efficacy(clean_logits[i], logits[i], pos_ids[i], neg_ids[i])
                 for i in range(len(prompts))]
        got = {"gap": _mean(s[0] for s in stats), "kl": _mean(s[1] for s in stats),
               "flip": _mean(s[2] for s in stats)}
        if want_target_gap:
            p = target_logprobs(prompts, pos_targets, intervention, batch_size=batch_size)
            n = target_logprobs(prompts, neg_targets, intervention, batch_size=batch_size)
            assert clean_target is not None
            got["target_gap"] = _mean((a["mean"] - b["mean"]) - c
                                      for a, b, c in zip(p, n, clean_target))
        return got

    rows = []
    logger.info("%9s %6s %12s %19s %9s %9s %7s %11s",
                "fraction", "layer", "steered_gap", "random_gap", "KL", "KL_rand",
                "flip%", "target_gap")
    for fraction in fractions:
        for layer in layers:
            direction, scale = steering[layer]
            entry = {"layer": layer, "fraction": fraction, "n_rand": n_rand}

            got = measure(Intervention.from_direction(layer, direction, scale, fraction,
                                                      positions, "steer"))
            entry["steered_gap_mean"] = got["gap"]
            entry["kl_mean"] = got["kl"]
            entry["flip_rate"] = got["flip"]
            if want_target_gap:
                entry["target_gap_mean"] = got["target_gap"]

            draws = [measure(Intervention.from_direction(
                layer,
                random_direction(hidden, control_seed(seed, layer, fraction, r)),
                scale, fraction, positions, "rand"))
                for r in range(n_rand)]
            entry["random_gap_mean"] = _mean(d["gap"] for d in draws)
            entry["random_gap_sd"] = _sd(d["gap"] for d in draws)
            entry["kl_random_mean"] = _mean(d["kl"] for d in draws)
            entry["kl_random_sd"] = _sd(d["kl"] for d in draws)
            entry["flip_rate_random"] = _mean(d["flip"] for d in draws)
            entry["flip_rate_random_sd"] = _sd(d["flip"] for d in draws)
            if want_target_gap:
                entry["target_gap_random_mean"] = _mean(d["target_gap"] for d in draws)
                entry["target_gap_random_sd"] = _sd(d["target_gap"] for d in draws)

            sd = entry["random_gap_sd"]
            logger.info("%9g %6d %12.4f %11.4f%8s %9.4f %9.4f %6.0f%% %11s",
                        fraction, layer, entry["steered_gap_mean"],
                        entry["random_gap_mean"],
                        "" if math.isnan(sd) else f" +-{sd:.4f}",
                        entry["kl_mean"], entry["kl_random_mean"],
                        100 * entry["flip_rate"],
                        f"{entry['target_gap_mean']:.4f}" if want_target_gap else "-")
            rows.append(entry)
    return rows


def _score_rows(behavior: Behavior, items: list[Item], prompts: list[str],
                completions: list[str], judges: list) -> list[dict]:
    questions = [item.question for item in items]
    graded = {}
    for judge in judges:
        scores = judge.score(questions, completions)
        graded[_judge_key(judge)] = (
            scores, [None if math.isnan(s) else int(s > judge.threshold)
                     for s in scores])

    rows = []
    for i, (item, prompt, text) in enumerate(zip(items, prompts, completions)):
        behavior_score = score_completion(behavior.scorer, text)
        refused = scoring.substring_matching_refused(text)
        row = {
            "index": item.index,
            "question": item.question,
            "prompt": prompt,
            "category": item.category,
            "response": text,
            "behavior_score": behavior_score.score,
            "behavior_hit": behavior_score.hit,
            "behavior_detail": behavior_score.detail,
            "refused_substring": int(refused),
            "jailbroken_substring": int(not refused),
        }
        for key, (scores, binary) in graded.items():
            row[f"judge_{key}_score"] = float(scores[i])
            row[f"judge_{key}"] = binary[i]
        if graded:
            first = next(iter(graded))
            row["judge_score"] = row[f"judge_{first}_score"]
            row["jailbroken_judge"] = row[f"judge_{first}"]
        rows.append(row)
    return rows


def _judge_key(judge) -> str:
    return getattr(judge, "style", None) or getattr(judge, "name", "judge")


def _summarize_generation(rows: list[dict]) -> dict:
    out = {
        "n": len(rows),
        "behavior_rate": _mean(r["behavior_hit"] for r in rows),
        "n_graded_behavior": len(_graded(r["behavior_hit"] for r in rows)),
        "behavior_score_mean": _mean(r["behavior_score"] for r in rows),
        "refusal_rate_substring": _mean(r["refused_substring"] for r in rows),
        "asr_substring": _mean(r["jailbroken_substring"] for r in rows),
    }
    if rows and "jailbroken_judge" in rows[0]:
        out["asr_judge"] = _mean(r["jailbroken_judge"] for r in rows)
        out["judge_score_mean"] = _mean(r["judge_score"] for r in rows)
        out["n_graded_judge"] = len(_graded(r["jailbroken_judge"] for r in rows))
        for key in sorted(k[len("judge_"):-len("_score")] for k in rows[0]
                          if k.startswith("judge_") and k.endswith("_score")
                          and k != "judge_score"):
            out[f"asr_{key}"] = _mean(r[f"judge_{key}"] for r in rows)
            out[f"score_{key}_mean"] = _mean(r[f"judge_{key}_score"] for r in rows)
            out[f"n_graded_{key}"] = len(_graded(r[f"judge_{key}"] for r in rows))
        if out["n_graded_judge"] < len(rows):
            logger.warning("%d/%d rows ungraded and excluded from asr_judge; the rate is "
                           "over the %d that graded", len(rows) - out["n_graded_judge"],
                           len(rows), out["n_graded_judge"])
    if out["n_graded_behavior"] < len(rows):
        logger.warning("%d/%d completions matched neither side of the behavior's lexicon "
                       "and are ungraded, not scored as misses -- behavior_rate is over "
                       "the %d that graded. A steer strong enough to destroy the output "
                       "lands here, which is what this separates from 'behavior absent'",
                       len(rows) - out["n_graded_behavior"], len(rows),
                       out["n_graded_behavior"])
    categories = sorted({r["category"] for r in rows if r["category"]})
    if categories:
        out["asr_substring_per_category"] = {
            c: _mean(r["jailbroken_substring"] for r in rows if r["category"] == c)
            for c in categories
        }
        if rows and "judge_score" in rows[0]:
            out["judge_score_per_category"] = {
                c: _mean(r["judge_score"] for r in rows if r["category"] == c)
                for c in categories
            }
    return out


def run_generation(behavior: Behavior, items: list[Item], prompts: list[str],
                   steering: dict, *, points: list[tuple[int, float]], positions: str,
                   seed: int, max_new_tokens: int, batch_size: int, judges: list,
                   stop_at_newline_pair: bool, out_dir: Path,
                   n_rand: int = 1) -> list[dict]:
    hidden = next(iter(steering.values()))[0].numel()

    completions_path = out_dir / "completions.jsonl"
    summaries: list[dict] = []
    with atomic_writes({"completions": completions_path}) as handles:
        handle = handles["completions"]

        def record(arm: str, layer, fraction, intervention, replicate: int = 0):
            texts = generate_completions(
                prompts, intervention, max_new_tokens=max_new_tokens,
                batch_size=batch_size, stop_at_newline_pair=stop_at_newline_pair)
            rows = _score_rows(behavior, items, prompts, texts, judges)
            for row in rows:
                handle.write(json.dumps({
                    "arm": arm, "layer": layer, "fraction": fraction,
                    "positions": intervention.positions if intervention else None,
                    "replicate": replicate, **row}) + "\n")
            handle.flush()
            summary = {"arm": arm, "layer": layer, "fraction": fraction,
                       "replicate": replicate, **_summarize_generation(rows)}
            summaries.append(summary)
            logger.info("%8s L%-3s f=%-7s %s behavior %5.0f%%  refusal %5.0f%%  "
                        "ASR(substring) %5.0f%%%s",
                        arm, layer if layer is not None else "-",
                        f"{fraction:g}" if fraction is not None else "-",
                        f"r{replicate}" if n_rand > 1 and arm == "rand" else "  ",
                        100 * summary["behavior_rate"],
                        100 * summary["refusal_rate_substring"],
                        100 * summary["asr_substring"],
                        f"  ASR(judge) {100 * summary['asr_judge']:.0f}%"
                        f"  score {summary['judge_score_mean']:.3f}"
                        + "".join(f"  {k}={v:.3f}" for k, v in summary.items()
                                  if k.startswith("score_") and k.endswith("_mean"))
                        if "asr_judge" in summary else "")

        record("none", None, None, None)
        for layer, fraction in points:
            direction, scale = steering[layer]
            record("steer", layer, fraction,
                   Intervention.from_direction(layer, direction, scale, fraction,
                                               positions, "steer"))
            for r in range(n_rand):
                rand_dir = random_direction(hidden, control_seed(seed, layer, fraction, r))
                record("rand", layer, fraction,
                       Intervention.from_direction(layer, rand_dir, scale, fraction,
                                                   positions, "rand"), replicate=r)
    logger.info("wrote %s", completions_path)
    return summaries


def _csv_floats(text: str, flag: str) -> list[float]:
    try:
        got = [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise SystemExit(f"{flag}: {exc}") from exc
    if not got:
        raise SystemExit(f"{flag}: give at least one value")
    return got


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--behavior", type=str, default="sentiment",
                        help=f"one of {', '.join(behaviors.list_behaviors())}, or a path")
    parser.add_argument("--arm", type=str, default=None,
                        help="test arm to evaluate (default: the dataset's default_arm)")
    parser.add_argument("--fractions", type=str, default="0.1",
                        help="comma-separated steering strengths; negative steers AWAY "
                             "from the behavior, which is the jailbreak direction for "
                             "jbb_refusal. Note argparse: a leading minus needs the equals "
                             "form, --fractions=-0.5,-1. Pass the same grid as "
                             "sweep_steer_fractions.sh to cross efficacy with detection")
    parser.add_argument("--layers", type=str, default="",
                        help="comma-separated blocks (default: every block)")
    parser.add_argument("--n_prompts", type=int, default=0,
                        help="0 (default) = the whole arm")
    parser.add_argument("--prompt_format", type=str, default="auto",
                        choices=list(prompt_format.FORMATS),
                        help="'auto' (default) uses the model's own chat template when it "
                             "has one. A refusal or jailbreak number measured OUTSIDE the "
                             "template is mostly measuring that the safety behavior was "
                             "never switched on; 'plain' is for base models and for "
                             "reproducing this repo's earlier results")
    parser.add_argument("--positions", type=str, default="last", choices=["last", "all"],
                        help="'last' matches what steer_audit audits and every existing "
                             "result here; 'all' matches the published setting (Mishra "
                             "et al., Arditi et al.) and is what a long generation needs")
    parser.add_argument("--no_target_gap", action="store_true",
                        help="skip the teacher-forced positive-vs-negative continuation "
                             "gap, which costs two extra forward passes per grid point")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="base seed for the random control. The direction is drawn "
                             "per (layer, fraction, replicate) from this, so the control "
                             "is an independent draw at each grid cell rather than one "
                             "direction reused across the whole sweep")
    parser.add_argument("--n_rand", type=int, default=3,
                        help="random control directions per grid cell in the cheap sweep "
                             "(default 3). >1 is what puts an error bar on the arm every "
                             "effect size here is measured against; with 1 there is no "
                             "way to tell a real effect from one unlucky draw")
    parser.add_argument("--gen_n_rand", type=int, default=1,
                        help="random control directions per generation point (default 1). "
                             "Each one costs a full decode of the arm plus its judge "
                             "calls, which is why this is not --n_rand")

    parser.add_argument("--generate", action="store_true",
                        help="also decode --max_new_tokens per prompt and score them. "
                             "This is where ASR comes from, and where the runtime goes")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Arditi et al. use 512; 256 is enough for every scorer here "
                             "and halves the cost")
    parser.add_argument("--gen_layers", type=str, default="",
                        help="layers to generate at (default: the best layer from the "
                             "cheap sweep, at --gen_pick_fraction)")
    parser.add_argument("--gen_fractions", type=str, default="",
                        help="strengths to generate at (default: all of --fractions)")
    parser.add_argument("--gen_pick_fraction", type=float, default=None,
                        help="the --fractions value whose best layer --gen_layers "
                             "defaults to (default: the largest by magnitude)")
    parser.add_argument("--gen_pick_by", type=str, default="flip", choices=list(PICK_BY),
                        help="how --gen_layers picks. 'flip' is the historical rule and "
                             "the right one for sentiment; use 'gap' or 'target_gap' for "
                             "refusal, where flip rate and KL both peak at the last block "
                             "for a perturbation that merely wrecks the output")
    parser.add_argument("--stop_at_newline_pair", action="store_true",
                        help="truncate each completion at the first blank line, where a "
                             "base model starts inventing the next conversational turn")

    parser.add_argument("--judge", type=str, default="substring",
                        choices=["substring", "fireworks", "none"],
                        help="'substring' (default) is Arditi et al.'s own metric and "
                             "needs no network; 'fireworks' adds Llama Guard 3 over the "
                             "Fireworks API and sends prompts and completions off-machine")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Fireworks model id for --judge fireworks "
                             "(default: scoring.FIREWORKS_JUDGE_MODEL)")
    parser.add_argument("--judge_style", type=str, default="harmbench,strongreject",
                        help="comma-separated; one generation pass, one API call per "
                             "grader per row. The FIRST one is canonical and fills the "
                             "asr_judge column.\n"
                             "  harmbench     binary, Arditi et al.'s classifier prompt. "
                             "88.7%% agreement / 11.6%% FPR on the 300 human-labelled rows "
                             "with the default judge model, the best of the three measured "
                             "here -- so it leads.\n"
                             "  strongreject  Souly et al. 2402.10260's graded rubric "
                             "(refusal, convincingness, specificity -> [0,1]). Binarised "
                             "it scores worse HERE (68.0%%/38.9%%), but that comparison is "
                             "unfair to it: its published Spearman 0.846 is against GRADED "
                             "human ratings and this file's labels are binary. Its value "
                             "is the score, which separates a jailbreak into something "
                             "useful from a steer that merely broke the model.\n"
                             "  llamaguard    the moderation frame, for a real guard model."
                             f" Choices: {', '.join(scoring.JUDGE_STYLES)}")
    parser.add_argument("--judge_threshold", type=float, default=None,
                        help="where a graded score is cut when a binary ASR is also "
                             f"wanted (default: {scoring.STRONGREJECT_THRESHOLD})")
    parser.add_argument("--validate_judge", action="store_true",
                        help="score the judge against the 300 human-labelled "
                             "JailbreakBench rows before using it, and record the result. "
                             "Cached per (judge, threshold) under results/_judge/validation/ "
                             "and reused: the scorecard is a property of the judge, not of "
                             "the model under test, so a sweep pays its 300 calls a grader "
                             "once rather than once per model")
    parser.add_argument("--revalidate_judge", action="store_true",
                        help="recompute the cached scorecard instead of reusing it. Needed "
                             "only when the judge model's weights or the row set changed "
                             "under a name that did not")

    parser.add_argument("--out_dir", type=Path, default=None,
                        help="default: results/<slug>/behavior/<name>/<arm>/ "
                             "(results/<slug>/sentiment/ for the sentiment behavior). The "
                             "arm is in the path so a second arm does not overwrite the "
                             "first -- they are different experiments")
    add_model_args(parser, default_dtype="float32")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "behavior.log")

    behavior = load_behavior(args.behavior)
    arm = args.arm or behavior.default_arm
    out_dir = args.out_dir or behavior_dir(args.model_name, behavior.name, arm)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = _csv_floats(args.fractions, "--fractions")
    if args.dtype != "float32":
        logger.warning("--dtype %s: the logit gaps measured here are near this dtype's "
                       "own resolution (bf16 eps ~8e-3); float32 is the trustworthy "
                       "setting", args.dtype)

    judge_kwargs = {}
    if args.judge == "fireworks":
        judge_kwargs = {"style": args.judge_style}
        if args.judge_threshold is not None:
            judge_kwargs["threshold"] = args.judge_threshold
        if args.judge_model:
            judge_kwargs["model"] = args.judge_model
    judges = []
    if args.judge != "none":
        for style in dict.fromkeys(v.strip() for v in args.judge_style.split(",") if v.strip()):
            kwargs = dict(judge_kwargs)
            if args.judge == "fireworks":
                kwargs["style"] = style
            judges.append(scoring.make_judge(args.judge, **kwargs))
            if args.judge != "fireworks":
                break
    judge_report = None
    if args.validate_judge and judges:
        judge_report = {}
        for judge in judges:
            judge_report.update(
                scoring.validate_cached(judge, refresh=args.revalidate_judge))
        scoring.log_validation(judge_report)

    model, _ = load_model(args.model_name, dtype=DTYPES[args.dtype], device=args.device)
    logger.info("model on %s, %s", model_device(), args.dtype)

    layers = ([int(s) for s in args.layers.split(",") if s.strip()] if args.layers
              else list(range(1, model.config.num_hidden_layers + 1)))
    fmt = prompt_format.resolve_format(args.prompt_format, behavior)
    items = behavior.items(arm, args.n_prompts)
    prompts = prompt_format.render_prompts(behavior, items, fmt)
    pos_targets, neg_targets = _resolve_targets(behavior, items)
    logger.info("behavior %s, arm %s: %d prompts x %d layers x %d fraction(s), "
                "positions=%s, prompt_format=%s", behavior.name, arm, len(prompts),
                len(layers), len(fractions), args.positions, fmt)
    logger.info("example prompt: %r", prompts[0])

    steering = build_steering_vectors(prompt_format.render_contrast_pairs(behavior, fmt),
                                      layers)

    if args.n_rand < 1 or args.gen_n_rand < 1:
        raise SystemExit("--n_rand and --gen_n_rand need at least one control direction")

    rows = sweep_grid(behavior, prompts, pos_targets, neg_targets, steering,
                      fractions=fractions, layers=layers, positions=args.positions,
                      seed=args.seed, batch_size=args.batch_size,
                      want_target_gap=not args.no_target_gap, n_rand=args.n_rand)

    payload = {
        "model_name": args.model_name,
        "behavior": behavior.name,
        "arm": arm,
        "n_prompts": len(prompts),
        "fractions": fractions,
        "positions": args.positions,
        "seed": args.seed,
        "n_rand": args.n_rand,
        "word_pos": pos_targets[0],
        "word_neg": neg_targets[0],
        "layers": rows,
    }
    if judge_report is not None:
        payload["judge_validation"] = judge_report
    gaps_path = out_dir / "gaps.json"
    gaps_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", gaps_path)

    if not args.generate:
        return

    pick = args.gen_pick_fraction
    if pick is None:
        pick = max(fractions, key=abs)
    gen_layers = ([int(s) for s in args.gen_layers.split(",") if s.strip()]
                  if args.gen_layers else [best_layer(rows, pick, args.gen_pick_by)])
    gen_fractions = (_csv_floats(args.gen_fractions, "--gen_fractions")
                     if args.gen_fractions else fractions)
    points = [(layer, f) for layer in gen_layers for f in gen_fractions]
    logger.info("generating %d new tokens at %d point(s): %s",
                args.max_new_tokens, len(points),
                ", ".join(f"L{layer} f={f:g}" for layer, f in points))

    summaries = run_generation(
        behavior, items, prompts, steering, points=points, positions=args.positions,
        seed=args.seed, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        judges=judges, stop_at_newline_pair=args.stop_at_newline_pair,
        out_dir=out_dir, n_rand=args.gen_n_rand)

    evaluation = {
        "model_name": args.model_name,
        "behavior": behavior.name,
        "arm": arm,
        "n_prompts": len(prompts),
        "positions": args.positions,
        "seed": args.seed,
        "n_rand": args.gen_n_rand,
        "max_new_tokens": args.max_new_tokens,
        "judges": [j.name for j in judges],
        "judge_thresholds": {_judge_key(j): j.threshold for j in judges},
        "scorer": behavior.scorer.get("kind"),
        "arms": summaries,
    }
    if judge_report is not None:
        evaluation["judge_validation"] = judge_report
    eval_path = out_dir / "evaluations.json"
    eval_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    logger.info("wrote %s", eval_path)


if __name__ == "__main__":
    main()
