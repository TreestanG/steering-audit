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
  detect       detect.py                  -> results/<slug>/sipit/detector_calibration.json
               calibrates the per-trajectory detector on the clean inversions just
               written. Always re-runs -- it is seconds, and a stale calibration is
               worse than none.
  sentiment    behavior_eval.py           -> results/<slug>/sentiment/gaps.json
  audit        steer_audit.py             -> results/<slug>/steer/audit.jsonl
  fractions    sweep_steer_fractions.sh   -> results/<slug>/steer/fractions/
  recover      steer_recover.py           -> results/<slug>/steer/{recover,localize}.jsonl
  plots        the five plot_*.py         -> results/<slug>/**/figures/
  behaviors    run_behavior.sh            -> results/<slug>/behavior/<name>/<arm>/
               every behavior x arm in data/: the multi-token half -- generation, ASR,
               and the detection-vs-efficacy join. Needs neither the activations nor
               the vocab table above, so it also runs standalone.

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
  --behaviors L     which behaviors to run afterwards: "all" (default, every behavior x
                    arm except sentiment, which the stages above already cover), or a
                    comma-separated list of NAME or NAME:ARM, e.g.
                    --behaviors jbb_refusal:harmful,language_fr
  --no_behaviors    stop after the figures, as this script did before
  --no_pgd          skip the PGD attack. It runs by DEFAULT: CAA is a blunt attack,
                    and "there is no operating point that steers without tripping the
                    detector" measured only against CAA is a claim about CAA. PGD is
                    the adversarial arm that makes finding 1 a statement about the
                    geometry rather than about one attack
  --pgd_layer_frac F  inject at this fraction of depth (default 0.7). Fixed rather
                    than per-model on purpose -- the gaps.json rule returns a
                    different KIND of layer once the final-block guard fires
                    (Qwen-0.5B 24->18, Qwen-7B 28->19), so a cross-model attack table
                    built on it compares two different experiments. 0.7 is where the
                    per-model winners cluster anyway: 15/24, 20/28, 27/36, 19/28
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

This runs the WHOLE pipeline: the sentiment half (single-token metrics plus the SipIt
and recomputation audits, which do not depend on a behavior at all), then every other
behavior through scripts/run_behavior.sh -- generation, ASR, and the detection-vs-
efficacy join. --no_behaviors stops after the figures.

Sizes follow the same quick/full split as everything else: without --full the behavior
half runs 4 prompts, 2 audited, 16 new tokens and the offline substring judge, so a
smoke test needs no API key. With --full it uses run_behavior.sh's real defaults --
the whole arm, 30 audited prompts, and the model judge, which DOES need one.
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
BEHAVIORS=all
DO_PGD=1
PGD_LAYER_FRAC=0.7
CHILD_LEVEL=INFO

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)     FULL=1; shift ;;
        --force)    FORCE=1; shift ;;
        --no_plots) NO_PLOTS=1; shift ;;
        --behaviors)    BEHAVIORS=$2; shift 2 ;;
        --pgd)          DO_PGD=1; shift ;;
        --no_pgd)       DO_PGD=0; shift ;;
        --pgd_layer_frac) PGD_LAYER_FRAC=$2; shift 2 ;;
        --no_behaviors) BEHAVIORS=; shift ;;
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
    SIPIT_PROMPTS=100; AUDIT_PROMPTS=5; RECOVER_PROMPTS=5
    FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.5,1,2
    MODE="full"
    # real sizes: run_behavior.sh's own defaults (whole arm, 30 audit prompts, model judge)
    BEHAVIOR_ARGS=()
else
    SIPIT_PROMPTS=2; AUDIT_PROMPTS=1; RECOVER_PROMPTS=1
    FRACTIONS=0.1,1
    MODE="quick (--full for real run sizes)"
    # a smoke test of the behavior half has to stay one: a handful of prompts, two
    # audited, short completions, and the offline judge so it needs no API key
    BEHAVIOR_ARGS=(--n_prompts 4 --audit_prompts 2 --max_new_tokens 16 --judge substring)
fi

if [[ -n $BEHAVIORS ]]; then
    # "all" is every behavior x arm the datasets define, minus sentiment, which the
    # stages above already cover under its own single-token metric.
    if ! BEHAVIOR_RUNS=$(AAT_WANT=$BEHAVIORS uv run python - <<'PY' 2>&1
import os, sys
sys.path.insert(0, "src")
import behaviors
want = os.environ["AAT_WANT"]
out = []
if want == "all":
    for name in behaviors.list_behaviors():
        if name == "sentiment":
            continue
        out += [f"{name}:{arm}" for arm in behaviors.load_behavior(name).arm_names()]
else:
    for spec in (v.strip() for v in want.split(",") if v.strip()):
        name, _, arm = spec.partition(":")
        b = behaviors.load_behavior(name)
        if arm and arm not in b.arms:
            raise SystemExit(f"no test arm {arm!r} in {name}; have {', '.join(b.arm_names())}")
        out += [f"{name}:{arm}"] if arm else [f"{name}:{a}" for a in b.arm_names()]
print(" ".join(out))
PY
); then
        printf '%s\n' "$BEHAVIOR_RUNS" >&2
        exit 1
    fi
    BEHAVIOR_RUNS=${BEHAVIOR_RUNS##*$'\n'}
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
    # Cheap post-hoc read of the sipit rows just written -- no GPU, no API, seconds. It
    # produces the operating point for the detector this repo actually ships, and its
    # sentinel is "-" so it always re-runs: a calibration file that predates a change to
    # the rule is worse than none, and the cost of refreshing it is nil.
    stage detect - \
        uv run src/detect.py --slug "$SLUG" --sensitivity
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

    if [[ -n $BEHAVIORS ]]; then
        say ""
        local spec name arm bargs
        for spec in $BEHAVIOR_RUNS; do
            name=${spec%%:*}; arm=${spec##*:}
            say "  behavior     $name / $arm"
            bargs=(--behavior "$name" --arm "$arm" --tag "$TAG" --dtype "$DTYPE")
            [[ -n $DEVICE ]] && bargs+=(--device "$DEVICE")
            [[ $FORCE -eq 1 ]] && bargs+=(--force)
            [[ $DRY_RUN -eq 1 ]] && bargs+=(--dry_run)
            scripts/run_behavior.sh "$MODEL" "${bargs[@]}" \
                ${BEHAVIOR_ARGS[@]+"${BEHAVIOR_ARGS[@]}"} \
                || FAILED+=("behavior:$name/$arm")
        done
    fi

    # The adversarial half (findings 6-8). On by default: CAA is a trivial attack, and a
    # null result against it says nothing about an optimizer that is allowed to search
    # under the detector's own constraint. It needs a layer policy to mean anything
    # across models -- --pgd_layer_frac pins the injection to a fixed fraction of depth
    # so "the attacked layer" is the same experiment everywhere, instead of the gaps.json
    # rule handing back a demoted runner-up on models whose best layer is the last block.
    if [[ $DO_PGD -eq 1 ]]; then
        say ""
        say "  pgd          layer_frac $PGD_LAYER_FRAC"
        local pargs=(--tag "$TAG" --layer_frac "$PGD_LAYER_FRAC")
        [[ -n $DEVICE ]] && pargs+=(--device "$DEVICE")
        [[ $FORCE -eq 1 ]] && pargs+=(--force)
        [[ $DRY_RUN -eq 1 ]] && pargs+=(--dry_run)
        scripts/run_pgd.sh "$MODEL" "${pargs[@]}" || FAILED+=("pgd")
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
