import argparse
import json
from pathlib import Path

import scoring
from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)


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
    parser.add_argument("--rows", type=Path, default=scoring.JUDGE_COMPARISON_PATH)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows -- for a cheap API smoke test")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the report here as JSON")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args)

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
    if not args.limit and not args.sweep_threshold and args.rows == scoring.JUDGE_COMPARISON_PATH:
        scoring.write_validation(judge, report)
    else:
        logger.info("not caching this scorecard: it is a partial or swept run")

    mine = report[judge.name]
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
