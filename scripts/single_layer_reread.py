"""The detector read at one logged layer, beside recomputation, per PGD ladder (src/single_layer.py).

  uv run python scripts/single_layer_reread.py [--run LABEL ...] [--deploy] [--out FILE]

--run is a ladder label from src/ladders.py, repeatable; the default is every ladder.
--deploy cuts every row at its first tolerance miss before scoring.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ladders  # noqa: E402
import single_layer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", default=None, help="ladder label, repeatable, or `all`")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = []
    for lad in ladders.select(a.run):
        e = single_layer.analyse(lad, deploy=a.deploy)
        single_layer.print_table(e)
        out.append(e)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
