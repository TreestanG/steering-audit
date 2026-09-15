"""Recomputation check and linear probe on one PGD ladder (src/recompute.py).

  uv run python scripts/baselines_recompute_probe.py --run "Qwen-0.5B (n=50)" \\
      --out results/current/baselines/Qwen_Qwen2.5-0.5B-Instruct_n50.json
  uv run python scripts/baselines_recompute_probe.py --model M --ladder DIR --bank JSON [--bank JSON] \\
      [--summary JSON] --out FILE

--run fills --model, --ladder, --bank and --summary from src/ladders.py; explicit flags win.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ladders  # noqa: E402
import recompute  # noqa: E402
from utils import DTYPES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=None, help="ladder label from src/ladders.py")
    p.add_argument("--model", default=None)
    p.add_argument("--ladder", type=Path, default=None, help="directory with <stem>_deltas.pt")
    p.add_argument("--bank", type=Path, action="append", default=None, help="chat bank prompt file, repeatable")
    p.add_argument("--summary", type=Path, default=None,
                   help="the detector's tpr_summary for this ladder, for side-by-side counts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES))
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0, help="smoke test on the first k prompts")
    a = p.parse_args()
    if a.run:
        lad = ladders.select([a.run])[0]
        a.model, a.ladder = a.model or lad.model, a.ladder or lad.deltas
        a.bank, a.summary = a.bank or list(lad.bank_prompts), a.summary or lad.summary
    if not (a.model and a.ladder and a.bank):
        raise SystemExit("give --run, or --model, --ladder and --bank")
    result = recompute.run(a.model, a.ladder, a.bank, a.summary, a.dtype, a.device, a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    recompute.log(f"wrote {a.out} in {result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
