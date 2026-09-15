"""Rescore the PGD ladders under the deployment stopping rule (src/deploy_stop.py).

  uv run python scripts/deployment_stop_rescore.py [--run LABEL ...] [--out FILE]

--run is a ladder label from src/ladders.py, repeatable; the default is every ladder.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deploy_stop  # noqa: E402
import ladders  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", default=None, help="ladder label, repeatable, or `all`")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = []
    deploy_stop.print_header()
    for lad in ladders.select(a.run):
        entry = deploy_stop.analyse(lad)
        deploy_stop.print_cells(entry)
        out.append(entry)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
