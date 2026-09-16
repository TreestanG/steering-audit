"""The read-layer attacker's rows, read at the layer it was told to keep quiet (src/read_layer.py).

  uv run python scripts/read_layer_reread.py [--run LABEL ...] [--deploy] [--out FILE]

--run is a read-run label from src/ladders.py, repeatable; the default is every run.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ladders  # noqa: E402
import read_layer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", default=None, help="read-run label, repeatable, or `all`")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--judge_field", default="judge_harmbench")
    p.add_argument("--attack_dir", type=Path, action="append", default=None,
                   help="a tree with stage-1 rows only (ceiling sweep, no-sculpting control)")
    p.add_argument("--sweep_detect", action="append", default=None,
                   metavar="TREE:CAL:BANK:LAYER",
                   help="flagged counts for a ceiling-sweep tree inverted after the fact")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = []
    for run in (ladders.select_read(a.run) if a.run or not a.attack_dir else []):
        e = read_layer.analyse(run.label, run.rows, run.calibration, run.bank_layers,
                               run.ladder_dir, list(run.read_layers), deploy=a.deploy,
                               field=a.judge_field)
        read_layer.print_table(e)
        out.append(e)
    for d in a.attack_dir or []:
        rows = read_layer.attack_only(d)
        read_layer.print_attack_only(rows)
        out.append({"attack_dir": str(d), "rows": rows})
    for spec in a.sweep_detect or []:
        tree, cal, bank, layer = spec.rsplit(":", 3)
        e = read_layer.sweep_detection(Path(tree), Path(cal), Path(bank), int(layer))
        e["tree"] = tree
        print(f"{tree}: flagged at layer {layer} {e['flagged_read']}/{e['n']}, "
              f"any layer {e['flagged_any']}/{e['n']} (threshold {e['threshold']:.2f})")
        out.append({"sweep_detection": e})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
