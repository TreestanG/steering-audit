"""Raw-state density baseline at one logged layer, on one PGD ladder (src/density.py).

  uv run python scripts/mahalanobis_baseline.py [--run "Qwen-0.5B (n=50)"] --out results/current/mahalanobis-qwen05b-2026-09-08.json

The ladder's deltas must be on this machine; --run picks the model, deltas, bank prompt
files and detection summary from src/ladders.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import density  # noqa: E402
import ladders  # noqa: E402
from utils import DTYPES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=ladders.N50.label, help="ladder label from src/ladders.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES))
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0, help="smoke test on the first k prompts")
    a = p.parse_args()
    lad = ladders.select([a.run])[0]
    result = density.run(lad.model, lad.deltas, list(lad.bank_prompts), lad.summary,
                         a.dtype, a.device, a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    density.log(f"wrote {a.out} in {result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
