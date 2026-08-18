#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
    cat <<'EOF'
Run the whole steering-audit pipeline for one model, from nothing to figures.

  scripts/run_model.sh [MODEL] [options]

MODEL defaults to Qwen/Qwen2.5-0.5B-Instruct.

Stages, in order (each skipped when its output already exists):
  activations  save_activations.py        -> data/activations/<slug>/*.pt
  vocab        vocab_activation_table.py  -> data/activations/<slug>/vocab/vocab_table.pt
  sipit        eval_sipit_layers.sh       -> results/<slug>/sipit/layers/
  sentiment    sentiment_dir.py           -> results/<slug>/sentiment/gaps.json
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
EOF
}

MODEL=
VV=0
DTYPE=float32
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
        --device)   DEVICE=$2; shift 2 ;;
        --model_name) MODEL=$2; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *)  MODEL=$1; shift ;;
    esac
done
[[ -n $MODEL ]] || MODEL=Qwen/Qwen2.5-0.5B-Instruct
SLUG=${MODEL//\//_}

COMMON=(--model_name "$MODEL" --dtype "$DTYPE")
[[ -n $DEVICE ]] && COMMON+=(--device "$DEVICE")

[[ $VV -eq 1 ]] || export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
[[ $VV -eq 1 ]] || export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}

# Injection layers are a fraction of depth, not fixed indices: 6,12,18 does not
# exist on a 12-layer model, and steer_recover rejects inject layers it cannot see.
N_LAYERS=$(uv run python -c "
from transformers import AutoConfig
print(AutoConfig.from_pretrained('$MODEL').num_hidden_layers)" 2>/dev/null | tail -1)
if ! [[ $N_LAYERS =~ ^[0-9]+$ ]]; then
    echo "could not read num_hidden_layers for $MODEL" >&2
    exit 1
fi
INJECT=$((N_LAYERS / 4)),$((N_LAYERS / 2)),$((3 * N_LAYERS / 4))

if [[ $FULL -eq 1 ]]; then
    SIPIT_PROMPTS=20; AUDIT_PROMPTS=5; RECOVER_PROMPTS=5
    FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.5,1,2
    MODE="full"
else
    SIPIT_PROMPTS=2; AUDIT_PROMPTS=1; RECOVER_PROMPTS=1
    FRACTIONS=0.1,1
    MODE="quick (--full for real run sizes)"
fi

ACT_DIR=data/activations/$SLUG
RES=results/$SLUG
LOG_DIR=$RES/logs
FAILED=()
mkdir -p "$LOG_DIR"

# Pipeline-level narration: to the console and appended to run.log, which is the
# only cross-stage record (per-stage logs are truncated when their stage reruns).
say() { printf '%s\n' "$*" | tee -a "$LOG_DIR/run.log" >&2; }

exists() { local f; for f in $1; do [[ -e $f ]] && return 0; done; return 1; }

stage() {  # stage <name> <sentinel-glob|-> <command...>
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
    : > "$log"   # truncate once per stage; the children append to it
    say "$(printf '  %-12s start   %s' "$name" "$log")"
    local t0=$SECONDS rc=0
    # No redirection: the stage writes its own DEBUG log and its console output
    # (stderr) flows straight through, so progress is visible and $? is exact.
    AAT_LOG_FILE=$log AAT_LOG_STAGE=$name AAT_LOG_LEVEL=$CHILD_LEVEL "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        say "$(printf '  %-12s ok (%ss)' "$name" "$((SECONDS - t0))")"
    else
        # The traceback is already on screen above this line.
        say "$(printf '  %-12s FAILED rc=%d (%ss)  full log: %s' "$name" "$rc" "$((SECONDS - t0))" "$log")"
        FAILED+=("$name")
        return 1
    fi
}

say "model     $MODEL"
say "layers    $N_LAYERS   inject at $INJECT"
say "dtype     $DTYPE${DEVICE:+   device $DEVICE}"
say "mode      $MODE"
say "results   $RES/"
say "logs      $LOG_DIR/"
say ""

# Prerequisites: everything downstream reads these, so a failure here is fatal.
stage activations "$ACT_DIR/*.pt" \
    uv run src/save_activations.py "${COMMON[@]}" || exit 1
stage vocab "$ACT_DIR/vocab/vocab_table.pt" \
    uv run src/vocab_activation_table.py "${COMMON[@]}" || exit 1

# Experiments are independent of each other; record failures and keep going.
stage sipit "$RES/sipit/layers/sipit_layer_00.jsonl" \
    scripts/eval_sipit_layers.sh --act_dir "$ACT_DIR" --n_prompts "$SIPIT_PROMPTS" \
        --model_name "$MODEL" --dtype "$DTYPE" ${DEVICE:+--device "$DEVICE"}
stage sentiment "$RES/sentiment/gaps.json" \
    uv run src/sentiment_dir.py "${COMMON[@]}" --fractions "$FRACTIONS"
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
    stage plot-audit    - uv run src/plot_steer_audit.py --model_name "$MODEL"
    stage plot-fraction - uv run src/plot_fraction_sweep.py --model_name "$MODEL"
    stage plot-recover  - uv run src/plot_steer_recover.py --model_name "$MODEL"
    stage plot-gap      - uv run src/plot_gap_vs_steering.py --model_name "$MODEL"
fi

say ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "failed stages: ${FAILED[*]}"
    exit 1
fi
[[ $DRY_RUN -eq 1 ]] || say "done -> $RES/   (logs in $LOG_DIR/)"
