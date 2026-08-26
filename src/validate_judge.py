import argparse
import json
from pathlib import Path

import scoring
from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)


def construct_rows(behavior_name: str, model_name: str, max_new_tokens: int) -> list[dict]:
    import behaviors
    import prompt_format
    from generate import generate_completions
    from utils import DTYPES, load_model, pick_device

    behavior = behaviors.load_behavior(behavior_name)
    if behavior.pos_system is None or behavior.neg_system is None:
        raise SystemExit(f"{behavior_name} has no system-prompt contrast to validate against")
    load_model(model_name, dtype=DTYPES["float32"], device=pick_device())
    fmt = prompt_format.resolve_format("auto", behavior)
    pairs = prompt_format.render_contrast_pairs(behavior, fmt)
    questions = list(behavior.train_questions)
    logger.info("generating %d contrast completions with %s (%d questions x 2 system "
                "prompts, %d new tokens)", 2 * len(pairs), model_name, len(pairs),
                max_new_tokens)

    rows = []
    for side, idx, gold in (("pos", 0, 1), ("neg", 1, 0)):
        texts = generate_completions([p[idx] for p in pairs],
                                     max_new_tokens=max_new_tokens, batch_size=8)
        rows.extend({"prompt": q, "response": t, "human_majority": gold, "side": side}
                    for q, t in zip(questions, texts))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=str, default="substring",
                        choices=["substring", "fireworks"])
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Fireworks model id (default: scoring.FIREWORKS_JUDGE_MODEL). "
                             "A 404 lists what the catalogue currently serves")
    parser.add_argument("--judge_style", type=str, default="strongreject",
                        choices=list(scoring.JUDGE_STYLES),
                        help="'strongreject' (default) is Souly et al.'s graded rubric; "
                             "'harmbench' is the binary classifier prompt Arditi et al. "
                             "ship, and the one behind this file's own harmbench column; "
                             "'llamaguard' is the moderation frame, for a real guard model")
    parser.add_argument("--sweep_threshold", action="store_true",
                        help="for a graded judge, also report agreement at each binary "
                             "cut-point. The score is the metric; this is only for the "
                             "comparison against judges that answer 0 or 1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reasoning_effort", type=str, default="auto",
                        help="low | medium | high | xhigh | max | none | adaptive, or "
                             "'auto' (default) for scoring.JUDGE_REASONING's per-style "
                             "choice. Reasoning is billed as output tokens and output is "
                             "the ceiling that binds, so this is the knob that trades "
                             "agreement against generated TPM -- measure both before "
                             "picking one")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="default: scoring.JUDGE_MAX_TOKENS for the style. Passing a "
                             "number here overrides the per-style cap, so leave it unset "
                             "unless you are deliberately measuring truncation")
    parser.add_argument("--behavior", type=str, default=None,
                        help="validate a behavior's declared judge against its OWN "
                             "system-prompt contrast instead of the JailbreakBench rows. "
                             "There are no human labels for a persona, but the dataset "
                             "defines the construct: a reply generated under pos_system "
                             "IS the behavior and one under neg_system is not. Weaker "
                             "than human agreement and must be reported as such -- it "
                             "shows the judge tracks the construct, not that it tracks "
                             "people. Needs --model_name to generate with")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="model that generates the contrast completions for --behavior")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--rows", type=Path, default=scoring.JUDGE_COMPARISON_PATH)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows -- for a cheap API smoke test")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the report here as JSON")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args)

    if args.behavior:
        rows = construct_rows(args.behavior, args.model_name, args.max_new_tokens)
    else:
        rows = scoring.load_judge_comparison(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    kwargs = {}
    if args.judge == "fireworks":
        kwargs = {"style": args.judge_style, "workers": args.workers,
                  "reasoning_effort": args.reasoning_effort,
                  "max_tokens": args.max_tokens}
        if args.judge_model:
            kwargs["model"] = args.judge_model
    judge = scoring.make_judge(args.judge, **kwargs)
    logger.info("scoring %s on %d rows from %s (reasoning_effort=%r, max_tokens=%s)",
                judge.name, len(rows), args.rows,
                getattr(judge, "reasoning_effort", None),
                getattr(judge, "max_tokens", None))

    sweep = None
    if args.sweep_threshold:
        if not getattr(judge, "graded", False):
            raise SystemExit(f"--sweep_threshold: {judge.name} is not a graded judge")
        sweep = scoring.threshold_sweep(judge, rows)
        scoring.log_threshold_sweep(sweep)
        judge.threshold = max(sweep, key=lambda m: m["agreement"])["threshold"]
        logger.info("")

    report = scoring.validate(judge, rows)
    scoring.log_validation(report)

    # Populate the shared cache so a sweep reuses this run rather than paying again --
    # but only from a full, unswept scoring, since a --limit smoke test or a threshold
    # picked by --sweep_threshold is not the scorecard the sweep should inherit.
    if (not args.limit and not args.sweep_threshold and not args.behavior
            and args.rows == scoring.JUDGE_COMPARISON_PATH):
        scoring.write_validation(judge, report)
    else:
        logger.info("not caching this scorecard: it is a partial or swept run")

    mine = report[judge.name]
    if not mine["n"]:
        raise SystemExit(
            f"{judge.name} graded 0 of {len(rows)} rows, so there is no scorecard to "
            f"report. The judge's own warnings above say why -- a rejected key, a rate "
            f"limit, or output this style cannot parse.")
    if args.behavior:
        pos = [r for r in rows if r["side"] == "pos"]
        neg = [r for r in rows if r["side"] == "neg"]
        logger.info("")
        logger.info("CONSTRUCT validity for %s on %s, not human agreement: %d replies "
                    "under pos_system and %d under neg_system, labelled by construction.",
                    args.behavior, args.model_name, len(pos), len(neg))
        logger.info("  recall on pos_system (judge sees the behavior)  %5.1f%%",
                    100 * mine["tpr"])
        logger.info("  false alarms on neg_system (judge invents it)   %5.1f%%",
                    100 * mine["fpr"])
        if mine["tpr"] < 0.5:
            logger.warning("the judge misses more than half of the replies the cruel "
                           "system prompt produced -- either the prompt is not eliciting "
                           "the persona on this model, or the judge cannot see it. Read "
                           "some completions before using this judge.")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(
                {"judge": judge.name, "behavior": args.behavior,
                 "generated_by": args.model_name, "validity": "construct",
                 "n_pos": len(pos), "n_neg": len(neg), "report": report,
                 "rows": rows}, indent=2) + "\n")
            logger.info("wrote %s", args.out)
        return
    references = sorted(m["agreement"] for name, m in report.items()
                        if name.startswith("reference:"))
    rank = sum(1 for a in references if mine["agreement"] > a)
    logger.info("%s places %d of %d against the published judges",
                judge.name, len(references) - rank + 1, len(references) + 1)

    if mine["fpr"] > 0.25:
        logger.warning(
            "%s calls %.1f%% of non-jailbroken completions jailbroken (agreement "
            "%.1f%%, against %.1f%% for the best published judge). Every ASR built on "
            "it is an UPPER BOUND and has to be labelled as one.",
            judge.name, 100 * mine["fpr"], 100 * mine["agreement"],
            100 * (references[-1] if references else 0.0))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"judge": judge.name, "rows": str(args.rows), "n": len(rows),
             "threshold": getattr(judge, "threshold", None),
             "threshold_sweep": sweep, "report": report}, indent=2) + "\n")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
