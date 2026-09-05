#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Sweep src/steer_audit.py over steering strengths, then overlay them on one figure.

  --fractions A,B,...  default: 0.01,0.02,0.05,0.1,0.2,0.5,1,2
  --rel_tol T          default: follows --dtype (fp32 1e-3, fp16 1e-2, bf16 5e-2)
  --model_name NAME    default: Qwen/Qwen2.5-0.5B-Instruct (also picks results/<slug>/)
  --out_dir DIR        default: results/<slug>/steer/fractions
  --plot_out PATH      default: results/<slug>/steer/figures/fraction_sweep.png
  --force              re-run fractions whose jsonl already exists
  --no_plot            run the audits, skip the figure
  --dry_run            print the planned runs and exit

Everything else is forwarded to steer_audit.py -- including --device and --dtype.
Each fraction pays for its own full-vocabulary scan, so wall clock is linear in
the number of fractions.
EOF
}

FRACTIONS_CSV=0.01,0.02,0.05,0.1,0.2,0.5,1,2
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
OUT_DIR=
PLOT_OUT=
FORCE=0
DO_PLOT=1
DRY_RUN=0
REL_TOL=
DTYPE=float32
AUDIT_ARGS=()
AAT_RUN_TAG=${AAT_RUN_TAG:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fractions)
            FRACTIONS_CSV=$2
            shift 2
            ;;
        --model_name)
            MODEL_NAME=$2
            AUDIT_ARGS+=(--model_name "$2")
            shift 2
            ;;
        --rel_tol)
            REL_TOL=$2
            AUDIT_ARGS+=(--rel_tol "$2")
            shift 2
            ;;
        --dtype)
            DTYPE=$2
            AUDIT_ARGS+=(--dtype "$2")
            shift 2
            ;;
        --out_dir)
            OUT_DIR=$2
            shift 2
            ;;
        --plot_out)
            PLOT_OUT=$2
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --no_plot)
            DO_PLOT=0
            shift
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        --fraction | --out)
            echo "sweep_steer_fractions.sh sets --fractions / --out_dir; use those" >&2
            exit 1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            AUDIT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z $REL_TOL ]]; then
    REL_TOL=$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from utils import DTYPES, rel_tol_for
print(rel_tol_for(DTYPES['$DTYPE']))" 2>/dev/null | tail -1)
    if [[ -z $REL_TOL ]]; then
        echo "could not derive rel_tol for --dtype $DTYPE" >&2
        exit 1
    fi
fi

export AAT_DTYPE=${AAT_DTYPE:-$DTYPE}
TAG=${AAT_RUN_TAG:-$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from paths import DTYPE_TAGS
print(DTYPE_TAGS.get('$DTYPE', ''))" 2>/dev/null | tail -1)}
SLUG=${MODEL_NAME//\//_}${TAG:+_$TAG}
[[ -n $OUT_DIR ]] || OUT_DIR=results/$SLUG/steer/fractions
[[ -n $PLOT_OUT ]] || PLOT_OUT=results/$SLUG/steer/figures/fraction_sweep.png

IFS=',' read -r -a FRACTIONS <<< "$FRACTIONS_CSV"
if [[ ${#FRACTIONS[@]} -eq 0 ]]; then
    echo "no fractions given" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "model      $MODEL_NAME"
echo "dtype      $DTYPE   rel_tol $REL_TOL"
echo "fractions  ${FRACTIONS[*]}"
echo "out        $OUT_DIR/steer_audit_f<fraction>.jsonl"
[[ $DO_PLOT -eq 1 ]] && echo "figure     $PLOT_OUT"

if [[ $DRY_RUN -eq 1 ]]; then
    for frac in "${FRACTIONS[@]}"; do
        echo "would run --fraction $frac -> $OUT_DIR/steer_audit_f${frac}.jsonl"
    done
    exit 0
fi

TODO=()
for frac in "${FRACTIONS[@]}"; do
    if [[ -s $OUT_DIR/steer_audit_f${frac}.jsonl && $FORCE -eq 0 ]]; then
        echo "======== fraction $frac -- exists, skipping ========"
    else
        TODO+=("$frac")
    fi
done

failed=()
if [[ ${#TODO[@]} -gt 0 ]]; then
    TODO_CSV=$(IFS=,; echo "${TODO[*]}")
    echo "======== fractions $TODO_CSV -> $OUT_DIR ========"
    if ! uv run src/steer_audit.py \
        --fractions "$TODO_CSV" \
        --out_dir "$OUT_DIR" \
        ${AUDIT_ARGS[@]+"${AUDIT_ARGS[@]}"}; then
        echo "fractions $TODO_CSV FAILED" >&2
        failed=("${TODO[@]}")
    fi
fi

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed fractions: ${failed[*]}" >&2
fi

if [[ $DO_PLOT -eq 1 ]]; then
    echo "======== plotting ========"
    uv run src/plot_fraction_sweep.py \
        --model_name "$MODEL_NAME" \
        --in_dir "$OUT_DIR" \
        --out "$PLOT_OUT" \
        --rel_tol "$REL_TOL"
fi

[[ ${#failed[@]} -eq 0 ]]
