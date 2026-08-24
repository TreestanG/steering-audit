import argparse
import json
import math
from pathlib import Path

import behaviors
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import behavior_dir, logs_dir, steer_dir
from plot_common import load_rows

logger = get_logger(__name__)

ARMS = ("steer", "rand")


def load_audit(audit_dir: Path) -> list[dict]:
    paths = []
    single = audit_dir / "audit.jsonl"
    if single.exists():
        paths.append(single)
    paths.extend(sorted((audit_dir / "fractions").glob("*.jsonl")))
    if not paths:
        raise SystemExit(
            f"no audit rows under {audit_dir}. Run steer_audit.py with the same "
            f"--behavior and --arm first, or point --audit_dir somewhere else."
        )
    rows = []
    for path in paths:
        rows.extend(load_rows(path))
    logger.info("audit: %d rows from %d file(s) in %s", len(rows), len(paths), audit_dir)
    return rows


def _detected(entry: dict, h_norm: float, rel_tol: float | None) -> bool:
    if rel_tol is None:
        return bool(entry["detected"])
    return entry["residual"] > rel_tol * h_norm


def join(completions: list[dict], audit: list[dict], *, rel_tol: float | None,
         jailbroken_field: str) -> list[dict]:
    index: dict[tuple, dict] = {}
    for row in audit:
        index[(row["prompt"], row["layer"], row["fraction"])] = row

    joined = []
    misses = 0
    for gen in completions:
        if gen["arm"] not in ARMS or gen["layer"] is None or gen["fraction"] is None:
            continue
        key = (gen["prompt"], gen["layer"], gen["fraction"])
        found = index.get(key)
        if found is None:
            misses += 1
            continue
        side = found[gen["arm"]]
        h_norm = found["h_norm"]
        joined.append({
            "prompt": gen["prompt"],
            "question": gen.get("question"),
            "category": gen.get("category"),
            "arm": gen["arm"],
            "layer": gen["layer"],
            "fraction": gen["fraction"],
            "jailbroken": int(gen.get(jailbroken_field, 0)),
            "jailbroken_substring": int(gen.get("jailbroken_substring", 0)),
            "behavior_hit": gen.get("behavior_hit"),
            **{k: gen[k] for k in gen if k.startswith("judge_") and k.endswith("_score")},
            "detected": int(_detected(side, h_norm, rel_tol)),
            "rel_residual": side["residual"] / h_norm,
            "margin_spent": side["margin_spent"],
            "h_norm": h_norm,
            "response": gen.get("response"),
        })
    if misses:
        logger.info("%d generated rows had no audit row and were dropped "
                    "(steer_audit usually runs on fewer prompts)", misses)
    return joined


def contingency(rows: list[dict]) -> dict:
    a = sum(1 for r in rows if r["jailbroken"] and r["detected"])
    b = sum(1 for r in rows if r["jailbroken"] and not r["detected"])
    c = sum(1 for r in rows if not r["jailbroken"] and r["detected"])
    d = sum(1 for r in rows if not r["jailbroken"] and not r["detected"])
    jailbroken = a + b
    return {
        "n": len(rows),
        "jailbroken_detected": a,
        "jailbroken_undetected": b,
        "clean_detected": c,
        "clean_undetected": d,
        "asr": jailbroken / len(rows) if rows else float("nan"),
        "detection_rate": (a + c) / len(rows) if rows else float("nan"),
        "undetected_given_jailbroken": b / jailbroken if jailbroken else float("nan"),
    }


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["layer"], r["fraction"], r["arm"]) for r in rows},
                  key=lambda k: (k[2], k[0], k[1]))
    out = []
    for layer, fraction, arm in keys:
        at = [r for r in rows if (r["layer"], r["fraction"], r["arm"]) == (layer, fraction, arm)]
        entry = {
            "arm": arm, "layer": layer, "fraction": fraction,
            **contingency(at),
            "asr_substring": sum(r["jailbroken_substring"] for r in at) / len(at),
            "rel_residual_mean": sum(r["rel_residual"] for r in at) / len(at),
            "margin_spent_mean": sum(r["margin_spent"] for r in at) / len(at),
        }
        for key in sorted(k for k in at[0] if k.startswith("judge_") and k.endswith("_score")):
            vals = [r[key] for r in at if r[key] is not None]
            entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        out.append(entry)
    return out


def graded_key(summary: list[dict]) -> str | None:
    if not summary:
        return None
    keys = [k for k in summary[0] if k.startswith("judge_") and k.endswith("_score_mean")]
    for preferred in ("judge_strongreject_score_mean",):
        if preferred in keys:
            return preferred
    return keys[0] if keys else None


def log_table(summary: list[dict]) -> None:
    key = graded_key(summary)
    label = key[len("judge_"):-len("_score_mean")] if key else "score"
    logger.info("%6s %5s %9s %4s %7s %9s %13s %11s %10s",
                "arm", "layer", "fraction", "n", "ASR", "detected", label,
                "rel_resid", "undet|jb")
    for s in summary:
        value = s.get(key) if key else None
        score = "-" if value is None else f"{value:.3f}"
        undet = s["undetected_given_jailbroken"]
        logger.info("%6s %5d %9g %4d %6.0f%% %8.0f%% %13s %11.2e %9s",
                    s["arm"], s["layer"], s["fraction"], s["n"], 100 * s["asr"],
                    100 * s["detection_rate"], score, s["rel_residual_mean"],
                    "-" if math.isnan(undet) else f"{100 * undet:.0f}%")


def plot(summary: list[dict], out: Path, *, rel_tol: float | None, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_common import C_RAND, C_STEER, save_fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    key = graded_key(summary)
    for arm, colour, width in (("rand", C_RAND, 4.0), ("steer", C_STEER, 1.8)):
        at = sorted((s for s in summary if s["arm"] == arm), key=lambda s: s["fraction"])
        if not at:
            continue
        xs = [abs(s["fraction"]) for s in at]
        alpha = 0.45 if arm == "rand" else 1.0
        axes[0].plot(xs, [100 * s["asr"] for s in at], "o-", color=colour, lw=width,
                     alpha=alpha, label=f"{arm}: ASR")
        if key and all(s.get(key) is not None for s in at):
            name = key[len("judge_"):-len("_score_mean")]
            axes[0].plot(xs, [100 * s[key] for s in at], "s--", color=colour, lw=width,
                         alpha=alpha * 0.6, label=f"{arm}: {name} x100")
        axes[1].plot(xs, [100 * s["detection_rate"] for s in at], "o-", color=colour,
                     lw=width, alpha=alpha, label=f"{arm}: detected")
        if arm == "steer":
            evaded = [s["undetected_given_jailbroken"] for s in at]
            xs_e, ys_e = zip(*[(x, 100 * v) for x, v in zip(xs, evaded)
                               if not math.isnan(v)]) or ((), ())
            if xs_e:
                axes[1].plot(xs_e, ys_e, "^:", color="C1", lw=2.0,
                             label="steer: undetected | jailbroken")

    axes[0].set_ylabel("percent")
    axes[0].set_title("Efficacy")
    axes[1].set_ylabel("detected (%)")
    axes[1].set_title("Detection" + (f"  (rel_tol {rel_tol:g})" if rel_tol else ""))
    axes[1].set_ylim(-5, 105)
    for ax in axes:
        ax.set_xlabel("|steering fraction|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--behavior", type=str, default="jbb_refusal",
                        help=f"one of {', '.join(behaviors.list_behaviors())}")
    parser.add_argument("--completions", type=Path, default=None,
                        help="default: results/<slug>/behavior/<name>/completions.jsonl")
    parser.add_argument("--audit_dir", type=Path, default=None,
                        help="default: results/<slug>/steer/<name>/")
    parser.add_argument("--rel_tol", type=float, default=None,
                        help="re-derive detection at this threshold instead of using the "
                             "flag the audit stored. Free: the rows carry residual and "
                             "||h||, so a different threshold is a re-read, not a re-run")
    parser.add_argument("--jailbroken_field", type=str, default="jailbroken_judge",
                        help="which column counts as a jailbreak (default the canonical "
                             "judge; 'jailbroken_substring' for the upper-bound metric)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <behavior dir>/detection_vs_efficacy.json")
    parser.add_argument("--rows_out", type=Path, default=None,
                        help="also write the joined per-prompt rows here")
    parser.add_argument("--plot_out", type=Path, default=None,
                        help="default: <behavior dir>/figures/detection_vs_efficacy.png")
    parser.add_argument("--no_plot", action="store_true")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "join.log")

    behavior = behaviors.load_behavior(args.behavior)
    bdir = behavior_dir(args.model_name, behavior.name)
    completions_path = args.completions or bdir / "completions.jsonl"
    audit_dir = args.audit_dir or steer_dir(args.model_name, behavior.name)

    if not completions_path.exists():
        raise SystemExit(
            f"{completions_path} not found. Run behavior_eval.py --generate for "
            f"--behavior {behavior.name} first."
        )
    completions = load_rows(completions_path)
    logger.info("generation: %d rows from %s", len(completions), completions_path)
    audit = load_audit(audit_dir)

    if args.jailbroken_field not in completions[0]:
        available = sorted(k for k in completions[0] if k.startswith("jailbroken"))
        raise SystemExit(f"--jailbroken_field {args.jailbroken_field!r} is not in "
                         f"{completions_path}; have {available}")

    rows = join(completions, audit, rel_tol=args.rel_tol,
                jailbroken_field=args.jailbroken_field)
    if not rows:
        gen_keys = {(r["layer"], r["fraction"]) for r in completions
                    if r["layer"] is not None}
        aud_keys = {(r["layer"], r["fraction"]) for r in audit}
        raise SystemExit(
            "nothing joined: the two stages share no (prompt, layer, fraction).\n"
            f"  generation has (layer, fraction): {sorted(gen_keys)}\n"
            f"  audit has:                        {sorted(aud_keys)}\n"
            "Re-run steer_audit with the same --fractions the generation used, and the "
            "same --behavior / --arm / --prompt_format."
        )

    summary = summarize(rows)
    log_table(summary)

    overall = contingency([r for r in rows if r["arm"] == "steer"])
    undetected = overall["jailbroken_undetected"]
    logger.info("")
    logger.info("steered arm, all points pooled: %d/%d jailbroken, %d of those UNDETECTED",
                overall["jailbroken_detected"] + undetected, overall["n"], undetected)
    if undetected:
        logger.warning("%d undetected jailbreak(s): an operating point that steers "
                       "without tripping the detector. This is the evasion window, and "
                       "it is the result worth chasing -- check them by hand before "
                       "believing it.", undetected)

    out = args.out or bdir / "detection_vs_efficacy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_name": args.model_name,
        "behavior": behavior.name,
        "rel_tol": args.rel_tol,
        "jailbroken_field": args.jailbroken_field,
        "completions": str(completions_path),
        "audit_dir": str(audit_dir),
        "n_joined": len(rows),
        "pooled_steer": overall,
        "points": summary,
    }, indent=2) + "\n")
    logger.info("wrote %s", out)

    if args.rows_out:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        logger.info("wrote %s", args.rows_out)

    if not args.no_plot:
        plot(summary,
             args.plot_out or bdir / "figures" / "detection_vs_efficacy.png",
             rel_tol=args.rel_tol,
             title=f"{args.model_name}  {behavior.name}: efficacy vs detection")


if __name__ == "__main__":
    main()
