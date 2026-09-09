"""Rewrite every reported detection summary with the current scorer, in place.

Older summaries had two outcomes per prompt; the current scorer has three (flagged,
clear, unscorable) and stamps provenance. Back up results/ and results_cuda/ first.

  uv run python scripts/regenerate_summaries.py
"""

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.argv = [sys.argv[0], "--dtype", "float16"]
import pgd_sipit

RUNS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "results/Qwen_Qwen2.5-0.5B-Instruct_n50", "detector_calibration_rolezlog_k1_fpr5_n100.json"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_n100.json"),
    ("google/gemma-3-1b-it", "results/google_gemma-3-1b-it_fp16", "detector_calibration_rolezlog_k1_fpr5.json"),
    ("gpt2", "results/gpt2_fp16", "detector_calibration_rolezlog_k1_fpr5.json"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "results_cuda/Qwen_Qwen2.5-1.5B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_n75.json"),
    ("Qwen/Qwen2.5-7B-Instruct", "results_cuda/Qwen_Qwen2.5-7B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_n75.json"),
    ("google/gemma-3-1b-it", "results_cuda/google_gemma-3-1b-it_fp16", "detector_calibration_rolezlog_k1_fpr5_n75.json"),
    ("meta-llama/Llama-3.2-1B-Instruct", "results_cuda/meta-llama_Llama-3.2-1B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_n75.json"),
    # bare-text cells (sentiment, sentiment_ext) need the bare-bank calibration: a chat
    # fit has no statistics for the last content positions of a bare prompt, so under
    # it those cells are unscorable rather than clear
    ("Qwen/Qwen2.5-0.5B-Instruct", "results/Qwen_Qwen2.5-0.5B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_bare.json"),
    ("Qwen/Qwen2.5-7B-Instruct", "results_cuda/Qwen_Qwen2.5-7B-Instruct_fp16", "detector_calibration_rolezlog_k1_fpr5_bare.json"),
]


def main() -> None:
    for model, root, cal in RUNS:
        root = ROOT / root
        rows = root / "pgd_sipit/pgd_rows.jsonl"
        if not rows.exists() or not (root / "sipit" / cal).exists():
            print("skip", root.name, "(no rows or calibration)")
            continue
        args = types.SimpleNamespace(model_name=model, dtype="float16")
        pgd_sipit.score(args, root / "sipit", root / "pgd_sipit", rows, cal)
        out = json.load(open(root / "pgd_sipit" / f"tpr_summary_{cal.replace('detector_calibration_', '').replace('.json', '')}.json"))
        unscorable = sum(c["unscorable"] for c in out["cells"])
        print("%-48s %-40s cells %2d unscorable %d code %s%s" % (
            root.name, cal, len(out["cells"]), unscorable, out["provenance"]["code"]["git_head"][:8],
            " (dirty)" if out["provenance"]["code"]["git_dirty"] else ""))


if __name__ == "__main__":
    main()
