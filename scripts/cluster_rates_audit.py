"""Per-budget rates with prompt-clustered uncertainty for the PGD ladders and the CAA sweep (src/cluster_rates.py).

  uv run python scripts/cluster_rates_audit.py [--out results/current/cluster-rates-2026-09-07.json]

The n50 ladder (src/ladders.py) is read through its join file; the CUDA ladders are
rescored from their rows and pooled on their shared prompts.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cluster_rates  # noqa: E402
import ladders  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("results/current/cluster-rates-2026-09-07.json"))
    a = p.parse_args()
    cuda = [lad for lad in ladders.LADDERS.values() if lad is not ladders.N50]
    out = cluster_rates.run(ladders.N50, cuda)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
