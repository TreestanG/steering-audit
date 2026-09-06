#!/usr/bin/env bash
# PGD budget ladder for one model: attack at fp32, detect at fp16, then decode, judge and
# join for behaviours with completions (objective target). Every stage skips or resumes
# what is already on disk, so a killed run loses at most its current cell.
#
#   scripts/pgd_ladder.sh MODEL [options]
#
#   --behavior B      jbb_refusal (default) | refusal | sentiment_ext | sentiment ...
#   --arm A           test arm (default: the behaviour's own; jbb_refusal -> harmful)
#   --objective O     target | sentiment (default: sentiment for sentiment*, else target)
#   --budgets L       0.0085,0.02,0.05,0.12,0.30 (default; 0.0085 is 0.85 x fp16 rel_tol)
#   --n_prompts N     15 (default). Later stages reuse it: the prompt set is stratified,
#                     so a different N scores a different set of prompts
#   --arms L          pgd,random (default)
#   --constraint C    all (default) | injection | detector -- the attack's scope, carried
#                     into detect and decode so their rows key on the same scope
#   --layer_frac F    0.7 (default)    --layer N       inject at this layer instead
#   --steps N         40 (default)     --n_restarts N  1 (default): A4's ladder settings
#   --batch_size N    prompts per attack forward (default 25). fp32 weights plus a backward
#                     over chat-length prompts overflow a card at the default: 3B on 24 GB
#                     and 7B on 48 GB both need 3-4
#   --calibrations L  detector_calibration_rolezlog_k1_fpr5.json (default); the first is
#                     the one the join uses
#   --stages L        attack,detect,decode,judge,join (default: all that apply). The judge
#                     needs the Fireworks key in .env, so on a box without it run
#                     --stages attack,detect,decode there and --stages judge,join here
#   --tag T           isolate everything under results/<slug>_T/ (AAT_RUN_TAG), so a
#                     second n or a second device never resumes into existing rows. The
#                     tag renames every derived path, so results/<slug>_T/sipit and
#                     data/activations/<slug>_T are symlinked to the _fp16 trees when
#                     absent -- calibration and vocab table are shared, rows are not
#   --attack_args S   extra flags passed verbatim to pgd_attack.py, e.g.
#                     --attack_args "--detector_cal results/<slug>_fp16/sipit/<cal>.json --penalty_weight 1"
#                     for the detector-aware attacker; pair with --tag so its rows
#                     never share a tree with the plain attack's
#   --force           redo the attack even if its deltas exist
#   --dry_run         print the commands and exit
#
# Outputs
#   results/<slug>_fp32/pgd/ladder/<behavior>_<arm>_b<B>{.jsonl,.json,_deltas.pt,_gen.json}
#   results/<slug>_fp16/pgd_sipit/{pgd_rows.jsonl,tpr_summary_<cal>.json,detection_vs_efficacy*}
#   results/<slug>_fp16/logs/pgd_ladder_<behavior>_<arm>.log
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
MODEL=${1:?usage: pgd_ladder.sh MODEL [options]}; shift
BEHAVIOR=jbb_refusal; ARM=; OBJECTIVE=; BUDGETS=0.0085,0.02,0.05,0.12,0.30; N=15
ARMS=pgd,random; LAYER_FRAC=0.7; LAYER=; STEPS=40; RESTARTS=1; SCOPE=all
CALS=detector_calibration_rolezlog_k1_fpr5.json; STAGES=; FORCE=0; DRY=0; TAG=; BATCH=; ATTACK_ARGS=
while [[ $# -gt 0 ]]; do
  case $1 in
    --behavior) BEHAVIOR=$2; shift 2 ;;
    --arm) ARM=$2; shift 2 ;;
    --objective) OBJECTIVE=$2; shift 2 ;;
    --budgets) BUDGETS=$2; shift 2 ;;
    --n_prompts) N=$2; shift 2 ;;
    --arms) ARMS=$2; shift 2 ;;
    --constraint) SCOPE=$2; shift 2 ;;
    --layer_frac) LAYER_FRAC=$2; shift 2 ;;
    --layer) LAYER=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --n_restarts) RESTARTS=$2; shift 2 ;;
    --calibrations) CALS=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --batch_size) BATCH=$2; shift 2 ;;
    --attack_args) ATTACK_ARGS=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry_run) DRY=1; shift ;;
    -h|--help) sed -n '2,43p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
[[ -n $ARM ]] || ARM=$(uv run python -c "import sys;sys.path.insert(0,'src');import behaviors;print(behaviors.load_behavior('$BEHAVIOR').default_arm)")
if [[ -z $OBJECTIVE ]]; then
  [[ $BEHAVIOR == sentiment* ]] && OBJECTIVE=sentiment || OBJECTIVE=target
fi
if [[ -z $STAGES ]]; then
  [[ $OBJECTIVE == target ]] && STAGES=attack,detect,decode,judge,join || STAGES=attack,detect
fi
SLUG=${MODEL//\//_}
if [[ -n $TAG ]]; then
  export AAT_RUN_TAG=$TAG
  if [[ ! -e results/${SLUG}_$TAG/sipit ]]; then
    [[ -d results/${SLUG}_fp16/sipit ]] || { echo "no results/${SLUG}_fp16/sipit to share with tag $TAG" >&2; exit 1; }
    mkdir -p "results/${SLUG}_$TAG"
    ln -s "../${SLUG}_fp16/sipit" "results/${SLUG}_$TAG/sipit"
    echo "linked results/${SLUG}_$TAG/sipit -> ${SLUG}_fp16/sipit"
  fi
  if [[ ! -e data/activations/${SLUG}_$TAG ]]; then
    [[ -d data/activations/${SLUG}_fp16 ]] || { echo "no data/activations/${SLUG}_fp16 to share with tag $TAG" >&2; exit 1; }
    ln -s "${SLUG}_fp16" "data/activations/${SLUG}_$TAG"
    echo "linked data/activations/${SLUG}_$TAG -> ${SLUG}_fp16"
  fi
  LAD=results/${SLUG}_$TAG/pgd/ladder
  OUT16=results/${SLUG}_$TAG
else
  LAD=results/${SLUG}_fp32/pgd/ladder
  OUT16=results/${SLUG}_fp16
fi
LOG=$OUT16/logs/pgd_ladder_${BEHAVIOR}_${ARM}.log
mkdir -p "$LAD" "$OUT16/logs" "$OUT16/pgd_sipit"
LAYER_ARGS=(--layer_frac "$LAYER_FRAC"); [[ -n $LAYER ]] && LAYER_ARGS=(--layer "$LAYER")
has() { [[ ",$STAGES," == *",$1,"* ]]; }
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
run() {
  if [[ $DRY -eq 1 ]]; then printf '   %q' "$@"; echo; return 0; fi
  "$@" >> "$LOG" 2>&1 || { local rc=$?; say "   exit $rc, see $LOG"; return $rc; }
}

say "ladder $MODEL $BEHAVIOR/$ARM objective=$OBJECTIVE n=$N budgets=$BUDGETS arms=$ARMS stages=$STAGES"
IFS=',' read -r -a BLIST <<< "$BUDGETS"
GENS=()
for B in "${BLIST[@]}"; do
  STEM=$LAD/${BEHAVIOR}_${ARM}_b$B
  if has attack; then
    if [[ -f ${STEM}_deltas.pt && $FORCE -eq 0 ]]; then
      say "b=$B attack: deltas exist, skipping"
    else
      say "b=$B attack (fp32)"
      run env AAT_DTYPE=float32 uv run src/pgd_attack.py --model_name "$MODEL" --dtype float32 \
        --budget "$B" "${LAYER_ARGS[@]}" --behavior "$BEHAVIOR" --arm "$ARM" --objective "$OBJECTIVE" \
        --n_prompts "$N" --steps "$STEPS" --n_restarts "$RESTARTS" --constraints "$SCOPE" --arms "$ARMS" \
        ${BATCH:+--batch_size "$BATCH"} $ATTACK_ARGS --save_deltas --force --out "$STEM.jsonl" || continue
    fi
  fi
  if has detect; then
    say "b=$B detect (fp16)"
    run uv run src/pgd_sipit.py --model_name "$MODEL" --dtype float16 --deltas "${STEM}_deltas.pt" \
      --objective "$OBJECTIVE" --behavior "$BEHAVIOR" --arm "$ARM" --n_prompts "$N" --arms "$ARMS" \
      --constraint "$SCOPE" --calibrations "$CALS"
  fi
  if has decode; then
    if [[ -f ${STEM}_gen.json ]]; then
      say "b=$B decode: completions exist, skipping"
    else
      say "b=$B decode"
      run uv run src/pgd_generate.py --model_name "$MODEL" --dtype float16 --deltas "${STEM}_deltas.pt" \
        --behavior "$BEHAVIOR" --arm "$ARM" --n_prompts "$N" --arms "$ARMS" --constraint "$SCOPE"
    fi
  fi
  [[ -f ${STEM}_gen.json || $DRY -eq 1 ]] && GENS+=("${STEM}_gen.json")
done
if has judge && [[ ${#GENS[@]} -gt 0 ]]; then
  say "judge (harmbench, strongreject)"
  run uv run src/judge_gen.py --gen "${GENS[@]}" --judge_style harmbench,strongreject
fi
if has join && [[ ${#GENS[@]} -gt 0 ]]; then
  say "join"
  run uv run src/join_pgd.py --model_name "$MODEL" --dtype float16 --gen "${GENS[@]}" \
    --calibration "${CALS%%,*}" --rows_out "$OUT16/pgd_sipit/detection_vs_efficacy_rows.jsonl" \
    && [[ $DRY -eq 0 ]] && sed -n '/^detection:/,$p' "$LOG" | tail -16
fi
say "DONE -> $OUT16/pgd_sipit/"
