"""The detector-aware attacker's rows read at one layer (src/aware_read.py).

  uv run python scripts/aware_single_layer.py [--run "Qwen-0.5B (n=50)"] [--ceilings 4.28,6,8,12,16,20] \\
      --out results/current/aware-single-layer-2026-09-09.json

--run names the ladder whose calibration and bank score the rows; the aware runs are
looked up as results/<slug>_aware_z<z> and results/<slug>_aware_run_z<z>.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aware_read  # noqa: E402
import ladders  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=ladders.N50.label, help="ladder label from src/ladders.py")
    p.add_argument("--ceilings", default=",".join(aware_read.CEILINGS), help="z ceilings, comma-separated")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lad = ladders.select([a.run])[0]
    out = aware_read.run(lad.calibration_path, lad.bank_layers, lad.slug, a.ceilings.split(","))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    aware_read.print_cells(out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
