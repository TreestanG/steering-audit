#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Efficacy, generation and detection for one behavior, on one or more models.

  scripts/run_behavior.sh [MODEL...] --behavior NAME [options]

Where run_model.sh is the sentiment pipeline plus the behavior-independent audits,
this is the multi-token half: the behaviors in data/behavior_*.json, greedy
generation, and the JailbreakBench metrics from Arditi et al. (2406.11717).

Stages, in order (each skipped when its output already exists):
  sweep      behavior_eval.py   -> results/<slug>/behavior/<name>/<arm>/gaps.json
             every (layer, fraction), single-token + teacher-forced. Cheap; this is
             what picks the layer.
  generate   behavior_eval.py   -> .../completions.jsonl, .../evaluations.json
             greedy completions at the picked layer, scored by the behavior's scorer,
             by substring matching, and by --judge. This is where ASR comes from.
  audit      steer_audit.py     -> results/<slug>/steer/<name>/<arm>/fractions/*.jsonl
             the detector, on the SAME prompts and the SAME steering vector.
  join       join_detection.py  -> .../<name>/<arm>/detection_vs_efficacy.json
                                    .../<name>/<arm>/figures/*.png
             the two joined on (prompt, layer, fraction, arm). Produces the number
             the whole project is about: of the steers that jailbroke, how many
             slipped past the audit. Zero is the claim.

Every path carries the ARM, because the arm is part of the experiment: jbb_refusal's
benign arm is the false-positive control for its harmful arm, and refusal has two of
its own. Two arms of one behavior are two runs and never overwrite each other.

Options:
  --behavior NAME    refusal | jbb_refusal | evil_persona | language_fr | sentiment
                     (default: jbb_refusal)
  --arm NAME         test arm (default: the dataset's own default_arm). Part of every
                     output path, so --arm benign is a separate run from --arm harmful
                     rather than something that overwrites it
  --fractions LIST   steering strengths. NEGATIVE steers away from the behavior,
                     which is the jailbreak direction for jbb_refusal -- and argparse
                     needs the equals form for it: --fractions=-0.25,-0.5,-0.75
                     (default: -0.25,-0.5,-0.75 for jbb_refusal, 0.25,0.5,1 otherwise)
  --gen_fractions L  strengths to generate at (default: all of --fractions)
  --gen_layers L     layers to generate at (default: the best one from the sweep)
  --pick_by K        how that best layer is chosen: flip | gap | target_gap.
                     Default target_gap, except sentiment, which keeps flip.
  --n_prompts N      0 = the whole arm (default)
  --max_new_tokens N default 256. Arditi et al. use 512
  --positions P      last | all (default all -- a behavior has to survive the whole
                     completion, and one steered position cannot carry it). Now passed to
                     BOTH the generation and the audit, so the 2x2 records one
                     intervention. The audit still inverts only the LAST position, so at
                     'all' its detection rate is a lower bound: the other steered
                     positions are never checked
  --audit_prompts N  prompts to audit (default 30). Sampled round-robin over categories,
                     so the audited set is a prefix of the generated one and the two
                     join. Only prompts in BOTH stages join, so this alone sets the
                     denominator on the evasion-window figure -- the number the project
                     turns on. At 3 fractions and the ~37% jailbreak rate seen so far, a
                     zero result is bounded at 95% by 48% with 5 prompts, 12% with 25,
                     10% with 30, 5% with 60. It is also the entire runtime of this
                     stage: one full vocabulary scan per prompt, measured at 49s on
                     Qwen-0.5B fp32 over 24 layers and 3 fractions, so ~25 min at the
                     default and roughly 14x that on a 7B. Lower it for large models,
                     and note the bounds above are for all fractions POOLED; a single
                     fraction on its own is bounded at 28% with 30 prompts
  --judge J          fireworks (default) | substring | none.
                     fireworks is the default because it is the only one whose ASR is
                     worth quoting: on JailbreakBench's 300 human-labelled rows the
                     HarmBench prompt agrees 87.0% at a 14.2% false-positive rate,
                     against substring matching's 55.3% / 69.5%. It needs
                     FIREWORKS_API_KEY (or FIREWORKS_KEY, or a gitignored .env) and
                     sends the goals and the completions to a third party. --judge
                     substring keeps the run local and offline, and the join then says
                     on its face that every rate it reports is an upper bound
  --judge_style S    comma-separated graders for --judge fireworks
                     (default: harmbench,strongreject). One generation pass, one API
                     call per grader per row. The first is canonical and supplies the
                     ASR; harmbench leads because it is the best-agreeing of the three
                     on the human labels, and strongreject follows because its graded
                     score is the only one that separates a useful jailbreak from a
                     steer that merely broke the model
  --no_audit         skip the detection stage (it is the expensive one: a full
                     vocabulary scan per prompt). Also skips the join, which has
                     nothing to join without it
  --join_rel_tol T   re-derive detection at this threshold in the join. Free -- the
                     audit rows carry residual and ||h||, so a different threshold is
                     a re-read, not a re-run
  --dtype D          float32 (default) | float16 | bfloat16
  --tag T / --device D / --force / --dry_run / -v / -q   as in run_model.sh

  scripts/run_behavior.sh Qwen/Qwen2.5-1.5B-Instruct --behavior jbb_refusal
  scripts/run_behavior.sh Qwen/Qwen2.5-1.5B-Instruct --behavior jbb_refusal \
      --arm benign --max_new_tokens 512          # the false-positive control arm
  scripts/run_behavior.sh gpt2 --behavior language_fr --fractions 0.5,1,2 \
      --judge substring                          # no judge call is worth making here

WHAT THE JOIN SCORES. The 2x2's success axis is read off the behavior rather than
fixed: jbb_refusal is the only behavior whose own scorer IS a judge, so it joins on
jailbroken_judge -- the HarmBench prompt under --judge fireworks, 87.0% agreement with
JailbreakBench's 300 human labels at a 14.2% false-positive rate. Every other behavior
joins on behavior_hit, its own scorer, because a refusal-phrase test applied to French
text or a persona scores near 100% and measures nothing. --jailbroken_field overrides.
Substring matching agrees 55.3% at a 69.5% false-positive rate; it stays available,
and the join labels anything it produces an upper bound on its face.
EOF
}

MODELS=()
BEHAVIOR=jbb_refusal
ARM=
FRACTIONS=
GEN_FRACTIONS=
GEN_LAYERS=
PICK_BY=
N_PROMPTS=0
MAX_NEW_TOKENS=256
POSITIONS=all
JUDGE=fireworks
JUDGE_STYLE=harmbench,strongreject
DO_AUDIT=1
AUDIT_PROMPTS=30
JOIN_REL_TOL=
DTYPE=float32
TAG=
DEVICE=
FORCE=0
DRY_RUN=0
VV=0
CHILD_LEVEL=INFO

while [[ $# -gt 0 ]]; do
    case "$1" in
        --behavior)       BEHAVIOR=$2; shift 2 ;;
        --arm)            ARM=$2; shift 2 ;;
        --fractions)      FRACTIONS=$2; shift 2 ;;
        --fractions=*)    FRACTIONS=${1#*=}; shift ;;
        --gen_fractions)  GEN_FRACTIONS=$2; shift 2 ;;
        --gen_fractions=*) GEN_FRACTIONS=${1#*=}; shift ;;
        --gen_layers)     GEN_LAYERS=$2; shift 2 ;;
        --pick_by)        PICK_BY=$2; shift 2 ;;
        --n_prompts)      N_PROMPTS=$2; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS=$2; shift 2 ;;
        --positions)      POSITIONS=$2; shift 2 ;;
        --judge)          JUDGE=$2; shift 2 ;;
        --judge_style)    JUDGE_STYLE=$2; shift 2 ;;
        --no_audit)       DO_AUDIT=0; shift ;;
        --audit_prompts)  AUDIT_PROMPTS=$2; shift 2 ;;
        --join_rel_tol)   JOIN_REL_TOL=$2; shift 2 ;;
        --dtype)          DTYPE=$2; shift 2 ;;
        --tag)            TAG=$2; shift 2 ;;
        --device)         DEVICE=$2; shift 2 ;;
        --force)          FORCE=1; shift ;;
        --dry_run)        DRY_RUN=1; shift ;;
        -v | --verbose)   CHILD_LEVEL=DEBUG; shift ;;
        -vv)              CHILD_LEVEL=DEBUG; VV=1; shift ;;
        -q | --quiet)     CHILD_LEVEL=WARNING; shift ;;
        --model_name)     MODELS+=("$2"); shift 2 ;;
        -h | --help)      usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *)  MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(Qwen/Qwen2.5-0.5B-Instruct)

if [[ ! -f data/behavior_$BEHAVIOR.json && ! -f $BEHAVIOR ]]; then
    echo "no such behavior: $BEHAVIOR" >&2
    ls data/behavior_*.json | sed 's|data/behavior_||;s|\.json||;s|^|  |' >&2
    exit 1
fi

if [[ -z $FRACTIONS ]]; then
    case "$BEHAVIOR" in
        jbb_refusal) FRACTIONS=-0.25,-0.5,-0.75 ;;
        sentiment)   FRACTIONS=0.1,0.5,1 ;;
        *)           FRACTIONS=0.25,0.5,1 ;;
    esac
fi
if [[ -z $PICK_BY ]]; then
    [[ $BEHAVIOR == sentiment ]] && PICK_BY=flip || PICK_BY=target_gap
fi

# Resolve the arm here rather than letting each stage default it separately: it is
# part of every output path now, so the driver has to name the same one the stages do.
if ! ARM=$(AAT_BEHAVIOR=$BEHAVIOR AAT_ARM=$ARM uv run python - <<'PY' 2>&1
import os, sys
sys.path.insert(0, "src")
import behaviors
behavior = behaviors.load_behavior(os.environ["AAT_BEHAVIOR"])
arm = os.environ.get("AAT_ARM") or behavior.default_arm
if arm not in behavior.arms:
    raise SystemExit(f"no test arm {arm!r} in {behavior.name}; "
                     f"have {', '.join(behavior.arm_names())}")
print(arm)
PY
); then
    printf '%s\n' "$ARM" >&2
    exit 1
fi
ARM=${ARM##*$'\n'}

if [[ -n $TAG && ! $TAG =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--tag must be [A-Za-z0-9._-]+, got: $TAG" >&2
    exit 1
fi
export AAT_DTYPE=$DTYPE
if [[ -z $TAG ]]; then
    TAG=$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from paths import DTYPE_TAGS
print(DTYPE_TAGS.get('$DTYPE', ''))" 2>/dev/null | tail -1)
    if [[ -z $TAG ]]; then
        echo "could not derive a run tag for --dtype $DTYPE" >&2
        exit 1
    fi
fi
export AAT_RUN_TAG=$TAG

[[ $VV -eq 1 ]] || export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
[[ $VV -eq 1 ]] || export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}

say() { printf '%s\n' "$*" | tee -a "$LOG_DIR/run.log" >&2; }

exists() { local f; for f in $1; do [[ -e $f ]] && return 0; done; return 1; }

stage() {
    local name=$1 sentinel=$2; shift 2
    if [[ $FORCE -eq 0 && $sentinel != "-" ]] && exists "$sentinel"; then
        say "$(printf '  %-12s skip (have %s)' "$name" "$sentinel")"
        return 0
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %-12s would run:' "$name" >&2; printf ' %q' "$@" >&2; echo >&2
        return 0
    fi
    local log=$LOG_DIR/$name.log
    : > "$log"
    say "$(printf '  %-12s start   %s' "$name" "$log")"
    local t0=$SECONDS rc=0
    AAT_LOG_FILE=$log AAT_LOG_STAGE=$name AAT_LOG_LEVEL=$CHILD_LEVEL "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        say "$(printf '  %-12s ok (%ss)' "$name" "$((SECONDS - t0))")"
    else
        say "$(printf '  %-12s FAILED rc=%d (%ss)  full log: %s' "$name" "$rc" "$((SECONDS - t0))" "$log")"
        FAILED+=("$name")
        return 1
    fi
}

run_one_model() {
    local MODEL=$1
    local SLUG=${MODEL//\//_}${TAG:+_$TAG}
    local RES=results/$SLUG
    local BDIR SDIR
    if [[ $BEHAVIOR == sentiment ]]; then
        BDIR=$RES/sentiment; SDIR=$RES/steer
    else
        BDIR=$RES/behavior/$BEHAVIOR/$ARM; SDIR=$RES/steer/$BEHAVIOR/$ARM
    fi
    local LOG_DIR=$RES/logs
    local FAILED=()
    mkdir -p "$LOG_DIR"

    local COMMON=(--model_name "$MODEL" --dtype "$DTYPE" --behavior "$BEHAVIOR"
                  --arm "$ARM")
    [[ -n $DEVICE ]] && COMMON+=(--device "$DEVICE")

    say "model     $MODEL"
    say "behavior  $BEHAVIOR   arm $ARM   fractions $FRACTIONS"
    say "dtype     $DTYPE${DEVICE:+   device $DEVICE}${TAG:+   tag $TAG}"
    say "results   $BDIR/  +  $SDIR/"
    say ""

    local GEN=(--generate --max_new_tokens "$MAX_NEW_TOKENS" --judge "$JUDGE"
               --gen_pick_by "$PICK_BY" --validate_judge)
    [[ $JUDGE == fireworks ]] && GEN+=(--judge_style "$JUDGE_STYLE")
    [[ -n $GEN_FRACTIONS ]] && GEN+=("--gen_fractions=$GEN_FRACTIONS")
    [[ -n $GEN_LAYERS ]] && GEN+=("--gen_layers=$GEN_LAYERS")

    stage behavior "$BDIR/evaluations.json" \
        uv run src/behavior_eval.py "${COMMON[@]}" \
            "--fractions=$FRACTIONS" --n_prompts "$N_PROMPTS" \
            --positions "$POSITIONS" "${GEN[@]}"

    if [[ $DO_AUDIT -eq 1 ]]; then
        stage audit "$SDIR/audit.jsonl $SDIR/fractions/*.jsonl" \
            uv run src/steer_audit.py "${COMMON[@]}" \
                "--fractions=${GEN_FRACTIONS:-$FRACTIONS}" --n_prompts "$AUDIT_PROMPTS" \
                --positions "$POSITIONS"

        local JOIN=(--model_name "$MODEL" --behavior "$BEHAVIOR" --arm "$ARM")
        [[ -n $JOIN_REL_TOL ]] && JOIN+=(--rel_tol "$JOIN_REL_TOL")
        stage join - uv run src/join_detection.py "${JOIN[@]}"
    fi

    say ""
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        say "failed stages: ${FAILED[*]}"
        SUMMARY+=("$(printf '  %-34s FAILED: %s' "$MODEL" "${FAILED[*]}")")
        return 1
    fi
    [[ $DRY_RUN -eq 1 ]] || say "done -> $BDIR/   (logs in $LOG_DIR/)"
    SUMMARY+=("$(printf '  %-34s ok -> %s/' "$MODEL" "$BDIR")")
}

SUMMARY=()
rc=0
idx=0
for MODEL in "${MODELS[@]}"; do
    if [[ ${#MODELS[@]} -gt 1 ]]; then
        idx=$((idx + 1))
        printf '\n===== [%d/%d] %s =====\n' "$idx" "${#MODELS[@]}" "$MODEL" >&2
    fi
    run_one_model "$MODEL" || rc=1
done

if [[ ${#MODELS[@]} -gt 1 ]]; then
    printf '\n===== %d models =====\n' "${#MODELS[@]}" >&2
    printf '%s\n' ${SUMMARY[@]+"${SUMMARY[@]}"} >&2
fi
exit $rc
