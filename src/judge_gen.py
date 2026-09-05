"""Grade pgd_generate's completions with a real judge, in place.

`pgd_generate` scores with substring matching, which B1 measures at 98.2% TPR and
69.5% FPR against JailbreakBench's human labels -- an upper bound, not a number to
quote. This adds the canonical graders to the same rows: the HarmBench prompt on
deepseek-v4-flash (88.7% agreement, the only model tried here that beats a published
judge) for the binary ASR, and StrongREJECT for the graded quality score.

Judging is API-bound, not GPU-bound, so it runs alongside inversion work. Rows are
rewritten in place with judge_<style> columns added; re-running skips a file whose
columns are already present unless --force. A row the API will not grade is left as
null and counted, never silently zero (B1's rule).
"""
import argparse
import json
from pathlib import Path

import scoring
from log import add_logging_args, get_logger
from log import setup as log_setup

logger = get_logger(__name__)


def grade(path: Path, styles: list[str], model: str, workers: int, force: bool) -> dict:
    blob = json.loads(path.read_text())
    rows = blob["rows"]
    todo = [s for s in styles if force or f"judge_{s}" not in rows[0]]
    if not todo:
        logger.info("%s: already graded with %s", path.name, ", ".join(styles))
        return blob
    for style in todo:
        judge = scoring.make_judge("fireworks", model=model, style=style, workers=workers)
        scores = judge.score([r["question"] for r in rows], [r["completion"] for r in rows])
        binary = [None if s != s else int(s > judge.threshold) for s in scores]
        for r, s, b in zip(rows, scores, binary):
            r[f"judge_{style}_score"] = None if s != s else float(s)
            r[f"judge_{style}"] = b
        ungraded = sum(b is None for b in binary)
        by_arm = {}
        for arm in sorted({r["arm"] for r in rows}):
            got = [r[f"judge_{style}"] for r in rows if r["arm"] == arm
                   and r[f"judge_{style}"] is not None]
            by_arm[arm] = sum(got) / len(got) if got else None
        blob.setdefault("judge_asr", {})[style] = by_arm
        blob.setdefault("judge_model", {})[style] = judge.name
        blob.setdefault("judge_ungraded", {})[style] = ungraded
        logger.info("%s  %-13s %s%s", path.name, style,
                    "  ".join(f"{a}={100*v:.0f}%" if v is not None else f"{a}=n/a"
                              for a, v in by_arm.items()),
                    f"   ({ungraded} ungraded, EXCLUDED from the rate)" if ungraded else "")
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gen", type=Path, nargs="+", required=True,
                   help="pgd_generate <stem>_gen.json files")
    p.add_argument("--judge_style", default="harmbench,strongreject")
    p.add_argument("--judge_model", default=scoring.FIREWORKS_JUDGE_MODEL)
    p.add_argument("--workers", type=int, default=4,
                   help="Fireworks tolerates about 4 concurrent calls on this account")
    p.add_argument("--force", action="store_true")
    add_logging_args(p)
    a = p.parse_args()
    log_setup(a, default_log=Path("results/all/logs/judge_gen.log"))
    styles = [s for s in a.judge_style.split(",") if s]
    logger.info("grading %d file(s) with %s on %s", len(a.gen), ", ".join(styles),
                a.judge_model.rsplit("/", 1)[-1])
    logger.info("substring ASR in these files is an UPPER BOUND (98.2%% TPR / 69.5%% FPR)")
    for path in a.gen:
        grade(path, styles, a.judge_model, a.workers, a.force)


if __name__ == "__main__":
    main()
