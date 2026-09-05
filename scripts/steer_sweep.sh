#!/usr/bin/env bash
# Steering grid scored by the shipped detector: arms x injection layers x fractions, with
# every position steered. Resumes cell by cell.
#
#   scripts/steer_sweep.sh MODEL [--n_prompts 100] [--inj_layers 3,6,9]
#       [--fractions 0.01,0.02,0.05,0.1,0.2,0.5] [--arms caa,random] [--dtype float16]
#       [--calibration detector_calibration.json] [--score_only]
#
# Rows: results/<slug>/sipit/steered/steered_rows.jsonl, summary tpr_summary*.json.
# Everything after MODEL is passed to src/steered_sipit.py; the defaults shown are this
# wrapper's, the ones the n=100 gpt2 grid used.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
MODEL=${1:?usage: steer_sweep.sh MODEL [steered_sipit options]}; shift
DTYPE=float16; N=100
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dtype) DTYPE=$2; shift 2 ;;
    --n_prompts) N=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec env AAT_DTYPE="$DTYPE" uv run src/steered_sipit.py --model_name "$MODEL" --dtype "$DTYPE" \
  --n_prompts "$N" ${ARGS[@]+"${ARGS[@]}"}
