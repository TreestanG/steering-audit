"""Fit the per-layer deployment stopping tolerance on a clean bank (src/deploy_tolerance.py).

  uv run python scripts/deploy_tolerance.py --layers_dir DIR --out FILE [--factor 1.5] [--base 0.01]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deploy_tolerance  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--base", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = deploy_tolerance.fit(a.layers_dir, a.factor, a.base, a.folds, a.repeats, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    deploy_tolerance.print_fit(out, a.out)


if __name__ == "__main__":
    main()
