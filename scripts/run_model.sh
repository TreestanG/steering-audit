#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Run the whole steering-audit pipeline for one model, from nothing to figures.

  scripts/run_model.sh [MODEL...] [options]

MODEL defaults to Qwen/Qwen2.5-0.5B-Instruct. Several models run one after another,
each finishing every stage before the next starts -- they share a GPU, so running
them concurrently would just trade wall clock for OOM risk. A model that fails its
prerequisites is abandoned and the sweep moves on to the next one.

Stages, in order (each skipped when its output already exists):
  activations  save_activations.py        -> data/activations/<slug>/*.pt
  vocab        vocab_activation_table.py  -> data/activations/<slug>/vocab/vocab_table.pt
  sipit        eval_sipit_layers.sh       -> results/<slug>/sipit/layers/
  sentiment    behavior_eval.py           -> results/<slug>/sentiment/gaps.json
  audit        steer_audit.py             -> results/<slug>/steer/audit.jsonl
  fractions    sweep_steer_fractions.sh   -> results/<slug>/steer/fractions/
  recover      steer_recover.py           -> results/<slug>/steer/{recover,localize}.jsonl
  plots        the five plot_*.py         -> results/<slug>/**/figures/

Options:
  -v, --verbose     per-item detail from each stage (-vv also un-silences HF)
  -q, --quiet       one line per stage, as before
  --full            real run sizes (default is a quick end-to-end validation)
  --force           re-run stages whose output already exists
  --dtype D         float32 (default) | float16 | bfloat16
  --tag T           suffix every path with _T. Defaults to the dtype (fp32 /
                    fp16 / bf16), which is what keeps two precisions of one model
                    from overwriting each other; pass it only to separate runs on
                    some other axis, e.g. --tag seed7
  --device D        cuda / cuda:1 / mps / cpu (default: best available)
                    Forwarded to every stage; each script also takes it directly.
  --no_plots        run the experiments, skip the figures
  --dry_run         print the planned commands and exit

Every stage appends a full DEBUG log to results/<slug>/logs/<stage>.log
regardless of console verbosity, and the pipeline's own narration is appended
to results/<slug>/logs/run.log.

The vocab table is the expensive artifact (vocab x layers x hidden, stored at
--dtype: ~13 GB for Qwen-0.5B at fp32, half that at fp16) and is never rebuilt
unless --force.
Note that --dtype must match the table on disk; sipit.py refuses a mismatch.

  scripts/run_model.sh gpt2                 # quick validation of a new model
  scripts/run_model.sh gpt2 --full          # real numbers
  scripts/run_model.sh Qwen/Qwen2.5-0.5B-Instruct --full --dtype float16
  scripts/run_model.sh gpt2 Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
      --full --dtype float16                # three models, in order, one dtype

This is the SENTIMENT pipeline: one behavior, single-token metrics, plus the SipIt
and recomputation audits that do not depend on a behavior at all. The refusal,
evil-persona, language and JailbreakBench behaviors -- and every multi-token metric,
including ASR -- live in scripts/run_behavior.sh, which reuses the artifacts this
one builds.
EOF
}

MODELS=()
VV=0
DTYPE=float32
TAG=
DEVICE=
FORCE=0
FULL=0
DRY_RUN=0
NO_PLOTS=0
CHILD_LEVEL=INFO

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)     FULL=1; shift ;;
        --force)    FORCE=1; shift ;;
        --no_plots) NO_PLOTS=1; shift ;;
        --dry_run)  DRY_RUN=1; shift ;;
        -v | --verbose) CHILD_LEVEL=DEBUG; shift ;;
        -vv)            CHILD_LEVEL=DEBUG; VV=1; shift ;;
        -q | --quiet)   CHILD_LEVEL=WARNING; shift ;;
        --dtype)    DTYPE=$2; shift 2 ;;
        --tag)      TAG=$2; shift 2 ;;
        --device)   DEVICE=$2; shift 2 ;;
        --model_name) MODELS+=("$2"); shift 2 ;;
        -h | --help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *)  MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(Qwen/Qwen2.5-0.5B-Instruct)

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

if [[ $FULL -eq 1 ]]; then
    SIPIT_PROMPTS=20; AUDIT_PROMPTS=5; RECOVER_PROMPTS=5
    FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.5,1,2
    MODE="full"
else
    SIPIT_PROMPTS=2; AUDIT_PROMPTS=1; RECOVER_PROMPTS=1
    FRACTIONS=0.1,1
    MODE="quick (--full for real run sizes)"
fi

REL_TOL=$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from utils import DTYPES, rel_tol_for
print(rel_tol_for(DTYPES['$DTYPE']))" 2>/dev/null | tail -1)
if [[ -z $REL_TOL ]]; then
    echo "could not derive rel_tol for --dtype $DTYPE" >&2
    exit 1
fi

artifact_dtype() {
    uv run python -c "
import sys, torch
try:
    print(torch.load(sys.argv[1], mmap=True, weights_only=False).get('dtype') or 'float32')
except Exception:
    pass" "$1" 2>/dev/null | tail -1
}

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

    local probe_err
    probe_err=$(mktemp)
    local N_LAYERS
    N_LAYERS=$(uv run python -c "
from transformers import AutoConfig
print(AutoConfig.from_pretrained('$MODEL').num_hidden_layers)" 2>"$probe_err" | tail -1)
    if ! [[ $N_LAYERS =~ ^[0-9]+$ ]]; then
        echo "could not read num_hidden_layers for $MODEL -- skipping" >&2
        grep -E "OSError|GatedRepo|401|403|not a local folder|Repository Not Found|ConnectionError" \
            "$probe_err" | head -3 | sed 's/^/      /' >&2
        rm -f "$probe_err"
        SUMMARY+=("$(printf '  %-34s could not read its config' "$MODEL")")
        return 1
    fi
    rm -f "$probe_err"
    local INJECT=$((N_LAYERS / 4)),$((N_LAYERS / 2)),$((3 * N_LAYERS / 4))

    local COMMON=(--model_name "$MODEL" --dtype "$DTYPE")
    [[ -n $DEVICE ]] && COMMON+=(--device "$DEVICE")

    local ACT_DIR=data/activations/$SLUG
    local RES=results/$SLUG
    local LOG_DIR=$RES/logs
    local FAILED=()
    mkdir -p "$LOG_DIR"

    say "model     $MODEL"
    say "layers    $N_LAYERS   inject at $INJECT"
    say "dtype     $DTYPE   rel_tol $REL_TOL${DEVICE:+   device $DEVICE}${TAG:+   tag $TAG}"
    say "mode      $MODE"
    say "results   $RES/"
    say "logs      $LOG_DIR/"
    say ""

    local act_sentinel="$ACT_DIR/*.pt"
    local vocab_sentinel="$ACT_DIR/vocab/vocab_table.pt"
    local have on_disk
    have=$(ls "$ACT_DIR"/*.pt 2>/dev/null | head -1)
    if [[ $FORCE -eq 0 && -n $have ]]; then
        on_disk=$(artifact_dtype "$have")
        if [[ -n $on_disk && $on_disk != "$DTYPE" ]]; then
            say "$(printf '  %-12s on disk is %s, want %s -- rebuilding' activations "$on_disk" "$DTYPE")"
            act_sentinel="-"
        fi
    fi
    if [[ $FORCE -eq 0 && -e $vocab_sentinel ]]; then
        on_disk=$(artifact_dtype "$vocab_sentinel")
        if [[ -n $on_disk && $on_disk != "$DTYPE" ]]; then
            say "$(printf '  %-12s on disk is %s, want %s -- rebuilding' vocab "$on_disk" "$DTYPE")"
            vocab_sentinel="-"
        fi
    fi

    stage activations "$act_sentinel" \
        uv run src/save_activations.py "${COMMON[@]}" || { model_failed "$MODEL"; return 1; }
    stage vocab "$vocab_sentinel" \
        uv run src/vocab_activation_table.py "${COMMON[@]}" || { model_failed "$MODEL"; return 1; }

    stage sipit "$RES/sipit/layers/sipit_layer_00.jsonl" \
        scripts/eval_sipit_layers.sh --act_dir "$ACT_DIR" --n_prompts "$SIPIT_PROMPTS" \
            --model_name "$MODEL" --dtype "$DTYPE" ${DEVICE:+--device "$DEVICE"}
    stage sentiment "$RES/sentiment/gaps.json" \
        uv run src/behavior_eval.py "${COMMON[@]}" --behavior sentiment \
            --fractions "$FRACTIONS" --no_target_gap
    stage audit "$RES/steer/audit.jsonl" \
        uv run src/steer_audit.py "${COMMON[@]}" --n_prompts "$AUDIT_PROMPTS"
    stage fractions "$RES/steer/fractions/*.jsonl" \
        scripts/sweep_steer_fractions.sh --model_name "$MODEL" --fractions "$FRACTIONS" \
            --n_prompts "$AUDIT_PROMPTS" --no_plot \
            --dtype "$DTYPE" ${DEVICE:+--device "$DEVICE"}
    stage recover "$RES/steer/recover.jsonl" \
        uv run src/steer_recover.py "${COMMON[@]}" --n_prompts "$RECOVER_PROMPTS" \
            --inject_layers "$INJECT"

    if [[ $NO_PLOTS -eq 0 ]]; then
        say ""
        stage plot-sipit    - uv run src/plot_sipit_layers.py --model_name "$MODEL"
        stage plot-audit    - uv run src/plot_steer_audit.py --model_name "$MODEL" --rel_tol "$REL_TOL"
        stage plot-fraction - uv run src/plot_fraction_sweep.py --model_name "$MODEL" --rel_tol "$REL_TOL"
        stage plot-recover  - uv run src/plot_steer_recover.py --model_name "$MODEL"
        stage plot-gap      - uv run src/plot_gap_vs_steering.py --model_name "$MODEL" --rel_tol "$REL_TOL"
    fi

    say ""
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        say "failed stages: ${FAILED[*]}"
        SUMMARY+=("$(printf '  %-34s FAILED: %s' "$MODEL" "${FAILED[*]}")")
        return 1
    fi
    [[ $DRY_RUN -eq 1 ]] || say "done -> $RES/   (logs in $LOG_DIR/)"
    SUMMARY+=("$(printf '  %-34s ok -> %s/' "$MODEL" "$RES")")
}

model_failed() {
    SUMMARY+=("$(printf '  %-34s FAILED: %s' "$1" "${FAILED[*]}")")
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
