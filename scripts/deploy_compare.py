"""Compare deployment-mode inversion rows with evaluator rows (src/deploy_tolerance.py).

  uv run python scripts/deploy_compare.py --calibration FILE [--behavior B] \
      [--deploy_bank DIR --eval_bank DIR] [--deploy_rows FILE --eval_rows FILE] [--out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deploy_tolerance as dt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--behavior", type=str, default=None, help="keep only ladder rows of this behavior")
    p.add_argument("--deploy_bank", type=Path, default=None)
    p.add_argument("--eval_bank", type=Path, default=None)
    p.add_argument("--deploy_rows", type=Path, default=None)
    p.add_argument("--eval_rows", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    report = {"calibration": str(a.calibration)}
    if a.deploy_bank and a.eval_bank:
        report["bank"] = dt.compare_bank(dt.load_layer_rows(a.deploy_bank), dt.load_layer_rows(a.eval_bank))
        dt.print_bank(report["bank"])
    if a.deploy_rows and a.eval_rows:
        cal = json.loads(a.calibration.read_text())
        report["ladder"] = dt.compare_ladder(dt.load_jsonl(a.deploy_rows, a.behavior),
                                             dt.load_jsonl(a.eval_rows, a.behavior), cal)
        dt.print_ladder(report["ladder"])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
