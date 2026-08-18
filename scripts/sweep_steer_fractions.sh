#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Sweep src/steer_audit.py over steering strengths, then overlay them on one figure.

  --fractions A,B,...  default: 0.01,0.02,0.05,0.1,0.2,0.5,1,2
  --model_name NAME    default: Qwen/Qwen2.5-0.5B-Instruct (also picks results/<slug>/)
  --out_dir DIR        default: results/<slug>/steer/fractions
  --plot_out PATH      default: results/<slug>/steer/figures/fraction_sweep.png
  --force              re-run fractions whose jsonl already exists
  --no_plot            run the audits, skip the figure
  --dry_run            print the planned runs and exit

Everything else is forwarded to steer_audit.py. Each fraction pays for its own
full-vocabulary scan, so wall clock is linear in the number of fractions.
EOF
}

FRACTIONS_CSV=0.01,0.02,0.05,0.1,0.2,0.5,1,2
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
OUT_DIR=
PLOT_OUT=
FORCE=0
DO_PLOT=1
DRY_RUN=0
REL_TOL=1e-3
AUDIT_ARGS=()

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
            echo "sweep_steer_fractions.sh sets --fraction / --out per run; use --fractions / --out_dir" >&2
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

SLUG=${MODEL_NAME//\//_}
[[ -n $OUT_DIR ]] || OUT_DIR=results/$SLUG/steer/fractions
[[ -n $PLOT_OUT ]] || PLOT_OUT=results/$SLUG/steer/figures/fraction_sweep.png

IFS=',' read -r -a FRACTIONS <<< "$FRACTIONS_CSV"
if [[ ${#FRACTIONS[@]} -eq 0 ]]; then
    echo "no fractions given" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "model      $MODEL_NAME"
echo "fractions  ${FRACTIONS[*]}"
echo "out        $OUT_DIR/steer_audit_f<fraction>.jsonl"
[[ $DO_PLOT -eq 1 ]] && echo "figure     $PLOT_OUT"

if [[ $DRY_RUN -eq 1 ]]; then
    for frac in "${FRACTIONS[@]}"; do
        echo "would run --fraction $frac -> $OUT_DIR/steer_audit_f${frac}.jsonl"
    done
    exit 0
fi

failed=()
for frac in "${FRACTIONS[@]}"; do
    out=$OUT_DIR/steer_audit_f${frac}.jsonl
    if [[ -s $out && $FORCE -eq 0 ]]; then
        echo "======== fraction $frac -- exists, skipping ($out) ========"
        continue
    fi
    echo "======== fraction $frac -> $out ========"
    # Write to .partial first: a run killed mid-sweep must not leave a truncated
    # jsonl that the next --force-less invocation would treat as complete.
    if uv run src/steer_audit.py \
        --fraction "$frac" \
        --out "$out.partial" \
        ${AUDIT_ARGS[@]+"${AUDIT_ARGS[@]}"}; then
        mv "$out.partial" "$out"
    else
        echo "fraction $frac FAILED" >&2
        rm -f "$out.partial"
        failed+=("$frac")
    fi
done

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
