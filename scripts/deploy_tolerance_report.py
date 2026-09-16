"""Re-read every PGD ladder under the per-layer deployment tolerance and attach fresh
deployment-mode runs (src/deploy_tolerance.py).

  uv run python scripts/deploy_tolerance_report.py --out FILE [--run LABEL ...] \
      [--deploy_run LABEL=deploy_compare.json ...] [--factor 1.5] [--base 0.01]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deploy_tolerance as dt  # noqa: E402
import ladders  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", action="append", default=None, help="ladder label, repeatable, or `all`")
    p.add_argument("--deploy_run", action="append", default=[], help="LABEL=path to a deploy_compare report")
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--base", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    runs = dict(x.split("=", 1) for x in a.deploy_run)
    unknown = [k for k in runs if k not in ladders.LADDERS]
    if unknown:
        raise SystemExit(f"unknown ladder {unknown}; choose from {list(ladders.LADDERS)}")
    out = []
    for lad in ladders.select(a.run):
        entry = dt.reread_ladder(lad, a.factor, a.base, a.folds, a.repeats, a.seed)
        if lad.label in runs:
            path = ROOT / runs[lad.label]
            entry["deployment_run"] = dt.summarise_run(json.loads(path.read_text()), path)
        changed = sum(bool(c["changed_at_tolerance"]) for c in entry["cells"])
        print(f"{lad.label}: rows shortened at base {entry['rows_shortened_at_base']}, at tolerance "
              f"{entry['rows_shortened_at_tolerance']}; cells changed at tolerance {changed} of {len(entry['cells'])}"
              + ("; fresh deployment run attached" if "deployment_run" in entry else ""))
        out.append(entry)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
