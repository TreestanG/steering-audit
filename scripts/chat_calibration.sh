#!/usr/bin/env bash
# Chat-format calibration bank for one model.
#
#   scripts/chat_calibration.sh build MODEL [--per_category N] [--batches B] [--dtype D] [--stop_on miss|wrong]
#   scripts/chat_calibration.sh refit MODEL K [--deltas PATH] [--dtype D]
#   scripts/chat_calibration.sh repair MODEL K --layers L1,L2 [--stop_on wrong] [--dtype D]
#
# build   wraps bank prompts in MODEL's chat template, saves activations and inverts
#         them in B category-stratified batches (N prompts per category, default 15 x 3)
#         into results/<slug>/sipit/chat_bank/layers_ext_b<i>. Batches finish
#         independently; a batch whose last layer exists is skipped on a re-run.
#         About 6 min a prompt on Qwen-0.5B on the Mac.
# repair  re-inverts the given layers of chat_bank/layers and the first K extension
#         batches with --stop_on (default wrong), keeping the originals as
#         sipit_layer_LL.jsonl.pre_repair, then runs refit. For layers where the fp16
#         floor at a template position truncated every row (Qwen-1.5B 27-28).
# refit   merges chat_bank/layers plus the first K extension batches into
#         chat_bank_n<N>/layers and refits detector_calibration_{rolezlog_k1,
#         rolezlog_k1_fpr5,rolez_k1,rolez_k1_fpr5,rolez}_n<N>.json. With --deltas the
#         stage-1 attack is rescored against each.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
CMD=${1:-}; [[ $# -gt 0 ]] && shift
MODEL=${1:?usage: chat_calibration.sh build|refit MODEL ...}; shift
DTYPE=float16
slug() { uv run python -c "import sys;sys.path.insert(0,'src');from paths import model_slug;print(model_slug('$MODEL'))"; }

build() {
  local PER=15 BATCHES=3 STOP=miss
  while [[ $# -gt 0 ]]; do
    case $1 in
      --per_category) PER=$2; shift 2 ;;
      --batches) BATCHES=$2; shift 2 ;;
      --dtype) DTYPE=$2; shift 2 ;;
      --stop_on) STOP=$2; shift 2 ;;
      *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
  done
  export AAT_DTYPE=$DTYPE
  local SLUG; SLUG=$(slug)
  local CB=results/$SLUG/sipit/chat_bank JSON ACT LOG
  JSON=$CB/trajectory_bank_chat_ext.json; ACT=$CB/act; LOG=results/$SLUG/logs/chat_bank_ext.log
  mkdir -p "$ACT" "results/$SLUG/logs"
  echo "[$(date)] build $MODEL ($SLUG) per_category=$PER batches=$BATCHES" | tee -a "$LOG"
  uv run src/build_chat_bank.py --model_name "$MODEL" --dtype "$DTYPE" \
      --per_category "$PER" --batches "$BATCHES" --out "$JSON" >> "$LOG" 2>&1 \
    || { echo "build FAILED, see $LOG" | tee -a "$LOG"; exit 1; }
  uv run src/save_activations.py --model_name "$MODEL" --dtype "$DTYPE" \
      --data_path "$JSON" --output_dir "$ACT" >> "$LOG" 2>&1 \
    || { echo "activations FAILED, see $LOG" | tee -a "$LOG"; exit 1; }
  local LAST; LAST=$(uv run python -c "
from transformers import AutoConfig
print(AutoConfig.from_pretrained('$MODEL').num_hidden_layers)" 2>/dev/null)
  local B OUT PTS
  for B in $(seq 1 "$BATCHES"); do
    OUT=$CB/layers_ext_b$B
    if [[ -f $OUT/sipit_layer_$(printf %02d "$LAST").jsonl ]]; then
      echo "batch $B already complete" | tee -a "$LOG"; continue
    fi
    PTS=$(uv run python -c "
import json
d=json.load(open('$JSON'))
print(' '.join('$ACT/$SLUG/'+p['id']+'.pt' for p in d['prompts'] if p['batch']==$B))")
    [[ -n $PTS ]] || { echo "batch $B has no prompts" | tee -a "$LOG"; continue; }
    echo "[$(date)] batch $B start" | tee -a "$LOG"
    uv run src/sipit.py --model_name "$MODEL" --dtype "$DTYPE" --all_layers --stop_on "$STOP" \
        --data_path "$JSON" --act_path $PTS --out_dir "$OUT" >> "$LOG" 2>&1
    echo "[$(date)] batch $B exit $?" | tee -a "$LOG"
  done
  echo "[$(date)] ALL BATCHES DONE" | tee -a "$LOG"
}

refit() {
  set -e
  local K=${1:?usage: chat_calibration.sh refit MODEL K [--deltas PATH] [--dtype D]}; shift
  local DELTAS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --deltas) DELTAS=$2; shift 2 ;;
      --dtype) DTYPE=$2; shift 2 ;;
      *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
  done
  export AAT_DTYPE=$DTYPE
  local SLUG; SLUG=$(slug)
  local SIP=results/$SLUG/sipit BASE N B f src MERGED
  BASE=$SIP/chat_bank/layers
  N=$(( $([[ -d $BASE ]] && ls "$BASE"/sipit_layer_00.jsonl >/dev/null 2>&1 && wc -l < "$BASE/sipit_layer_00.jsonl" || echo 0) ))
  for B in $(seq 1 "$K"); do
    N=$(( N + $(wc -l < "$SIP/chat_bank/layers_ext_b$B/sipit_layer_00.jsonl") ))
  done
  MERGED=$SIP/chat_bank_n$N/layers
  mkdir -p "$MERGED"
  for f in $( { [[ -d $BASE ]] && ls "$BASE"; } || ls "$SIP/chat_bank/layers_ext_b1" ); do
    : > "$MERGED/$f"
    [[ -f $BASE/$f ]] && cat "$BASE/$f" >> "$MERGED/$f"
    for B in $(seq 1 "$K"); do
      src=$SIP/chat_bank/layers_ext_b$B/$f
      [[ -f $src ]] || { echo "batch $B is missing $f" >&2; exit 1; }
      cat "$src" >> "$MERGED/$f"
    done
  done
  echo "merged $(wc -l < "$MERGED/sipit_layer_00.jsonl") prompts per layer -> $MERGED"

  local D=(src/detect.py --slug "$SLUG" --layers_subdir "chat_bank_n$N/layers" --statistic role_z --chat_model "$MODEL")
  run() { uv run "${D[@]}" "${@:3}" --out_name "$1" 2>&1 | grep -E "^$SLUG +[0-9]" | sed "s|^|$2 |"; }
  run "detector_calibration_rolezlog_k1_n$N.json"      "log+end k=1 sig3   :" --transform log --content_roles end --k 1
  run "detector_calibration_rolezlog_k1_fpr5_n$N.json" "log+end k=1 tuned5%:" --transform log --content_roles end --k 1 --target_fpr 0.05
  run "detector_calibration_rolez_k1_n$N.json"         "raw-z   k=1 sig3   :" --k 1
  run "detector_calibration_rolez_k1_fpr5_n$N.json"    "raw-z   k=1 tuned5%:" --k 1 --target_fpr 0.05
  run "detector_calibration_rolez_n$N.json"            "raw-z   k=5 sig3   :" --k 5

  [[ -n $DELTAS ]] || return 0
  local CALS=detector_calibration_rolezlog_k1_fpr5_n$N.json,detector_calibration_rolez_k1_fpr5_n$N.json,detector_calibration_rolez_n$N.json
  uv run src/pgd_sipit.py --model_name "$MODEL" --dtype "$DTYPE" --deltas "$DELTAS" \
      --objective sentiment --score_only --calibrations "$CALS" > /dev/null 2>&1
  uv run python - "$SLUG" "$CALS" <<'PY'
import json, sys
slug, cals = sys.argv[1], sys.argv[2].split(",")
for f in cals:
    suf = f.replace("detector_calibration_", "").replace(".json", "")
    d = json.load(open(f"results/{slug}/pgd_sipit/tpr_summary_{suf}.json"))
    op = d["operating_points"]["shipped"]
    print(f"{suf:34s} k={d['calibration']['k']} sigma={op['sigma']:.2f}")
    for c in d["cells"]:
        if c["arm"] in ("clean", "pgd", "random"):
            lab = "FPR" if c["arm"] == "clean" else "TPR"
            print(f"   {str(c.get('behavior','sentiment')):12s} {c['arm']:6s} b={c['budget']:<7g} "
                  f"n={c['n']:2d} {lab} {100*c['tpr_shipped']:4.0f}%")
PY
}

repair() {
  set -e
  local K=${1:?usage: chat_calibration.sh repair MODEL K --layers L1,L2 [--stop_on wrong] [--dtype D]}; shift
  local LAYERS="" STOP=wrong
  while [[ $# -gt 0 ]]; do
    case $1 in
      --layers) LAYERS=$2; shift 2 ;;
      --stop_on) STOP=$2; shift 2 ;;
      --dtype) DTYPE=$2; shift 2 ;;
      *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
  done
  [[ -n $LAYERS ]] || { echo "repair needs --layers" >&2; exit 1; }
  export AAT_DTYPE=$DTYPE
  local SLUG; SLUG=$(slug)
  local CB=results/$SLUG/sipit/chat_bank JSON=results/$SLUG/sipit/chat_bank/trajectory_bank_chat_ext.json
  local ACT=$CB/act LOG=results/$SLUG/logs/chat_bank_repair.log
  local DIRS=() B L F PTS
  [[ -d $CB/layers ]] && DIRS+=("$CB/layers")
  for B in $(seq 1 "$K"); do DIRS+=("$CB/layers_ext_b$B"); done
  echo "[$(date)] repair $MODEL layers $LAYERS stop_on=$STOP in ${DIRS[*]}" | tee -a "$LOG"
  for D in "${DIRS[@]}"; do
    B=${D##*_b}; [[ $D == */layers ]] && B=0
    if [[ $B == 0 ]]; then
      PTS=$(uv run python -c "
import json
d=json.load(open('$JSON'))
print(' '.join('$ACT/$SLUG/'+p['id']+'.pt' for p in d['prompts'] if p.get('batch', 0)==0))")
    else
      PTS=$(uv run python -c "
import json
d=json.load(open('$JSON'))
print(' '.join('$ACT/$SLUG/'+p['id']+'.pt' for p in d['prompts'] if p['batch']==$B))")
    fi
    [[ -n $PTS ]] || { echo "$D: no prompts for batch $B in $JSON" | tee -a "$LOG"; exit 1; }
    for L in ${LAYERS//,/ }; do
      F=$D/sipit_layer_$(printf %02d "$L").jsonl
      [[ -f $F.pre_repair ]] || cp "$F" "$F.pre_repair"
      echo "[$(date)] $F" | tee -a "$LOG"
      uv run src/sipit.py --model_name "$MODEL" --dtype "$DTYPE" --layer "$L" --stop_on "$STOP" \
          --data_path "$JSON" --act_path $PTS --out "$F" >> "$LOG" 2>&1 \
        || { echo "repair FAILED at $F, see $LOG" | tee -a "$LOG"; exit 1; }
      grep -E "exact [0-9]+/[0-9]+" "$LOG" | tail -1 | tee -a "$LOG"
    done
  done
  echo "[$(date)] re-inverted; refitting" | tee -a "$LOG"
  refit "$K" --dtype "$DTYPE"
}

case $CMD in
  build) build "$@" ;;
  refit) refit "$@" ;;
  repair) repair "$@" ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
