#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Run src/pgd_attack.py over one or more models, at one or more detection budgets.

  scripts/lib/run_pgd.sh [MODEL...] [options]

MODEL defaults to Qwen/Qwen2.5-0.5B-Instruct. Several models run one after another --
they share a GPU, so running them concurrently trades wall clock for OOM risk. A model
whose prerequisites are missing is reported and skipped; the sweep continues.

Everything pgd_attack.py takes is forwarded, including --behavior / --arm /
--prompt_format; this wrapper's own defaults are the sentiment ones. For a refusal
attack pass --layer explicitly: the gaps.json layer rule is (flip_rate, KL), which
picks the last block on a refusal behavior, where nothing has actually been steered
(see behavior_eval --gen_pick_by).

Prerequisite: results/<slug>/sentiment/gaps.json, which is where the attack's layer
comes from (highest flip_rate at --fraction). Produce it with
  scripts/run_model.sh MODEL --full
or point at another run's copy with --gaps.

  --budget_dtypes A,B    deployment precisions whose rel_tol sets the budget
                         (default: float16 -> 0.0085, float32 -> 0.00085)
  --budgets A,B          explicit relative budgets instead, e.g. 0.0085,0.00085
  --gaps PATH            gaps.json to pick the layer from (single-model runs only)
  --layer N              skip the gaps.json lookup and inject here
  --constraints C        all,injection (default) -- see pgd_attack.py --help
  --arms A               caa,random,pgd (default)
  --objective O          sentiment (default) | cw | target | refusal
                         target is the jailbreak objective: mean teacher-forced
                         log P(affirmative continuation), which needs --behavior
                         jbb_refusal (or another behavior with per-item targets)
  --n_prompts N          0 = the whole test set (default), matching gaps.json
  --steps N              PGD steps per restart (default 50; finding 7 measured 25
                         matching 200 to five decimals on Pythia/sentiment, and 50
                         keeps margin where that was not measured)
  --n_restarts N         random inits on top of the zero init (default 3)
  --tag T                suffix every path with _T, as run_model.sh does
  --device D             cuda / mps / cpu (default: best available)
  --save_deltas          also write the winning perturbations, for a vocab scan
  --force                re-run instead of resuming/skipping finished runs
  --dry_run              print the planned runs and exit
  -v | -q                per-prompt detail / warnings only

Everything else is forwarded to pgd_attack.py verbatim.

Compute precision is float32 always and is NOT configurable here: this is the only
stage in the repo that runs a backward pass, and fp16 gradients would produce a null
result that is rounding rather than geometry. --budget_dtypes varies the constraint
alone, on identical weights and arithmetic -- which is the whole point of the fp16
vs fp32 comparison.

Output, per model per budget:
  results/<slug>/pgd/pgd_<objective>_b<budget>.jsonl   one row per constraint/arm/prompt
  results/<slug>/pgd/pgd_<objective>_b<budget>.json    config + per-arm aggregates
  results/<slug>/logs/pgd.log                          full DEBUG log

The .json is written last, so it is the "finished" sentinel: a re-run skips a budget
that has one, and resumes from the .jsonl for a budget that does not.

  scripts/lib/run_pgd.sh                                  # quick check on Qwen-0.5B
  scripts/lib/run_pgd.sh EleutherAI/pythia-1.4b # the full 25-prompt run
  scripts/lib/run_pgd.sh Qwen/Qwen2.5-7B-Instruct --budget_dtypes float16,float32

--tag is the usual way to reach an fp32 control tree: it points the gaps.json lookup
and the output at results/<slug>_fp32/ together, so --gaps is only needed when reading
a layer from some other model's run.
EOF
}

MODELS=()
BUDGET_DTYPES=float16
BUDGETS=
TAG=
DRY_RUN=0
FORCE=0
CHILD_ARGS=()
OBJECTIVE=sentiment

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget_dtypes) BUDGET_DTYPES=$2; shift 2 ;;
        --budgets)       BUDGETS=$2; shift 2 ;;
        --tag)           TAG=$2; shift 2 ;;
        --objective)     OBJECTIVE=$2; CHILD_ARGS+=(--objective "$2"); shift 2 ;;
        --dry_run)       DRY_RUN=1; shift ;;
        --force)         FORCE=1; CHILD_ARGS+=(--force); shift ;;
        --save_deltas)   CHILD_ARGS+=(--save_deltas); shift ;;
        -v | --verbose | -q | --quiet) CHILD_ARGS+=("$1"); shift ;;
        --dtype)
            echo "run_pgd.sh: compute dtype is float32 by design; use --budget_dtypes" \
                 "to vary the constraint" >&2
            exit 1
            ;;
        --budget | --budget_dtype | --out)
            echo "run_pgd.sh sets $1 per run; use --budgets / --budget_dtypes" >&2
            exit 1
            ;;
        --model_name)    MODELS+=("$2"); shift 2 ;;
        -h | --help)     usage; exit 0 ;;
        -*)
            CHILD_ARGS+=("$1")
            if [[ $# -gt 1 && $2 != -* ]]; then
                CHILD_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *)               MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(Qwen/Qwen2.5-0.5B-Instruct)

if [[ -n $TAG && ! $TAG =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--tag must be [A-Za-z0-9._-]+, got: $TAG" >&2
    exit 1
fi
export AAT_RUN_TAG=$TAG
export AAT_DTYPE=float32

if [[ -z $BUDGETS ]]; then
    BUDGETS=$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from utils import DTYPES, rel_tol_for
print(','.join('%g' % (0.85 * rel_tol_for(DTYPES[d])) for d in '$BUDGET_DTYPES'.split(',') if d))
" 2>/dev/null | tail -1)
    if [[ -z $BUDGETS ]]; then
        echo "could not derive budgets for --budget_dtypes $BUDGET_DTYPES" >&2
        exit 1
    fi
fi
IFS=',' read -r -a BUDGET_LIST <<< "$BUDGETS"

echo "models     ${MODELS[*]}"
echo "budgets    ${BUDGET_LIST[*]}   (0.85 * rel_tol, compute dtype float32)"
echo "objective  $OBJECTIVE"

planned=()
for model in "${MODELS[@]}"; do
    slug=${model//\//_}_${TAG:-fp32}
    for budget in "${BUDGET_LIST[@]}"; do
        planned+=("$model|$budget|results/$slug/pgd/pgd_${OBJECTIVE}_b${budget}.jsonl")
    done
done

if [[ $DRY_RUN -eq 1 ]]; then
    for entry in "${planned[@]}"; do
        IFS='|' read -r model budget out <<< "$entry"
        echo "would run $model --budget $budget -> $out"
    done
    exit 0
fi

failed=()
for entry in "${planned[@]}"; do
    IFS='|' read -r model budget out <<< "$entry"
    summary=${out%.jsonl}.json
    if [[ -s $summary && $FORCE -eq 0 ]]; then
        echo "======== $model budget $budget -- done, skipping ($summary) ========"
        continue
    fi
    echo "======== $model budget $budget -> $out ========"
    if ! uv run src/pgd_attack.py \
        --model_name "$model" \
        --budget "$budget" \
        ${CHILD_ARGS[@]+"${CHILD_ARGS[@]}"}; then
        echo "$model at budget $budget FAILED" >&2
        failed+=("$model@$budget")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed runs: ${failed[*]}" >&2
fi
[[ ${#failed[@]} -eq 0 ]]
