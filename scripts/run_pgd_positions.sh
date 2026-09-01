#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Sweep the PGD attack over the NUMBER OF INJECTED POSITIONS and score each m against
both detector calibrations.

  scripts/run_pgd_positions.sh [MODEL] [options]

The budget stays PER POSITION at every m, so m is a knob on total perturbation
energy at a fixed per-position deviation. That is the axis the two shipped
calibrations disagree on: k=1 reads the max over positions and cannot see m at all,
while k=5 averages the top five and stops diluting a single spike once m >= 5. m=1
is the existing single-position result; m=0 is the whole prompt, which is the
steered_sipit "every position" arm with an attacker choosing the vectors.

  --positions A,B,C   m values to sweep (default: 1,2,3,5,0; 0 = the whole prompt)
  --budget B          relative budget per position (default: 0.0085 = 0.85 * fp16 rel_tol)
  --objective O       sentiment (default) | cw | target | refusal
  --layer_frac F      injection layer as a fraction of depth (default: 0.7)
  --arms A            attack arms (default: caa,random,pgd)
  --steps N           PGD steps per restart (default: 50)
  --n_restarts N      random inits on top of the zero init (default: 3)
  --constraint C      all (default) | injection
  --stage1_only       run the attacks, skip the SipIt scoring
  --score_only        re-score the rows already on disk, run nothing
  --device D          cuda / mps / cpu
  --dry_run           print the planned runs and exit

Stage 1 runs at float32 (it needs gradients) into results/<slug>_fp32/pgd/, stage 2
at float16 (the deployed precision) into results/<slug>_fp16/pgd_sipit/. Stage 2
accumulates every m into one pgd_rows.jsonl keyed by m, so the final --score_only
pass prints the whole sweep as one table. The clean control is m-free and is
computed once for the sweep, not once per m.

--save_deltas only collects the rows its own invocation computed, so stage 1 is
always run with --force here rather than resumed.

  scripts/run_pgd_positions.sh gpt2
  scripts/run_pgd_positions.sh gpt2 --positions 1,5 --steps 100
  scripts/run_pgd_positions.sh gpt2 --score_only
EOF
}

MODEL=
POSITIONS=1,2,3,5,0
BUDGET=0.0085
OBJECTIVE=sentiment
LAYER_FRAC=0.7
ARMS=caa,random,pgd
STEPS=50
N_RESTARTS=3
CONSTRAINT=all
STAGE1_ONLY=0
SCORE_ONLY=0
DEVICE=
DRY_RUN=0
CHILD_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --positions) POSITIONS=$2; shift 2 ;;
        --budget) BUDGET=$2; shift 2 ;;
        --objective) OBJECTIVE=$2; shift 2 ;;
        --layer_frac) LAYER_FRAC=$2; shift 2 ;;
        --arms) ARMS=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        --n_restarts) N_RESTARTS=$2; shift 2 ;;
        --constraint) CONSTRAINT=$2; shift 2 ;;
        --stage1_only) STAGE1_ONLY=1; shift ;;
        --score_only) SCORE_ONLY=1; shift ;;
        --device) DEVICE=$2; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        -*) CHILD_ARGS+=("$1"); shift ;;
        *) if [[ -z $MODEL ]]; then MODEL=$1; else CHILD_ARGS+=("$1"); fi; shift ;;
    esac
done
MODEL=${MODEL:-gpt2}
SLUG=${MODEL//\//_}
DEV_ARGS=()
[[ -n $DEVICE ]] && DEV_ARGS=(--device "$DEVICE")

IFS=',' read -ra MS <<< "$POSITIONS"

# its own stem: stage 1 runs with --force here (--save_deltas only collects its own
# invocation's rows), and pgd_/pgdtpr_ are existing results this must not overwrite
stage1_path() {  # $1 = m
    local scope=""
    [[ $CONSTRAINT != all ]] && scope="_$CONSTRAINT"
    echo "results/${SLUG}_fp32/pgd/pgdpos_${OBJECTIVE}_b${BUDGET}${scope}_m$1.jsonl"
}

if [[ $SCORE_ONLY -eq 1 ]]; then
    exec uv run src/pgd_sipit.py --model_name "$MODEL" --dtype float16 \
        --deltas "$(stage1_path "${MS[0]}")" --objective "$OBJECTIVE" \
        --constraint "$CONSTRAINT" --score_only ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
fi

if [[ $DRY_RUN -eq 1 ]]; then
    for m in "${MS[@]}"; do
        echo "m=$m -> $(stage1_path "$m")"
    done
    exit 0
fi

failed=()
for m in "${MS[@]}"; do
    out=$(stage1_path "$m")
    deltas=${out%.jsonl}_deltas.pt
    echo "======== m=$m  stage 1 (fp32) -> $out ========"
    if ! uv run src/pgd_attack.py \
        --model_name "$MODEL" --dtype float32 --budget "$BUDGET" \
        --layer_frac "$LAYER_FRAC" --objective "$OBJECTIVE" --arms "$ARMS" \
        --constraints "$CONSTRAINT" --n_positions "$m" --steps "$STEPS" \
        --n_restarts "$N_RESTARTS" --save_deltas --force --out "$out" \
        ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ${CHILD_ARGS[@]+"${CHILD_ARGS[@]}"}; then
        echo "m=$m stage 1 FAILED" >&2
        failed+=("m=$m/stage1")
        continue
    fi
    [[ $STAGE1_ONLY -eq 1 ]] && continue
    echo "======== m=$m  stage 2 (fp16) -> results/${SLUG}_fp16/pgd_sipit/ ========"
    if ! uv run src/pgd_sipit.py \
        --model_name "$MODEL" --dtype float16 --deltas "$deltas" \
        --objective "$OBJECTIVE" --constraint "$CONSTRAINT" --arms "$ARMS" \
        ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}; then
        echo "m=$m stage 2 FAILED" >&2
        failed+=("m=$m/stage2")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "sweep complete; re-print the table with:"
echo "  scripts/run_pgd_positions.sh $MODEL --score_only --positions ${MS[0]}"
