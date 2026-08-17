#!/usr/bin/env bash
# Sweep SipIt across every hidden-state layer on a stratified subset of the
# trajectory bank (~20 prompts, even split per category).
#
# File selection (this script; not forwarded):
#   --act_dir DIR     default: data/activations/Qwen_Qwen2.5-0.5B-Instruct
#   --n_prompts N     default: 20, split evenly across categories
#   --category NAME   repeatable; only sample from these
#   --ids ID          skip sampling; invert these ids at every layer
#   --out_dir DIR     default: results/<act_dir basename>/layers
#   --dry_run         print selected ids and exit
#
# Everything else is forwarded to eval_sipit.sh (and then sipit.py).
# sipit.py --all_layers reads the layer count from the activation tensor.
#
# Examples:
#   scripts/eval_sipit_layers.sh
#   scripts/eval_sipit_layers.sh --n_prompts 20
#   scripts/eval_sipit_layers.sh --category code --category base64 --n_prompts 8

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

ACT_DIR=data/activations/Qwen_Qwen2.5-0.5B-Instruct
N_PROMPTS=20
OUT_DIR=
DRY_RUN=0
CATEGORIES=()
IDS=()
FORWARD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --act_dir)
            ACT_DIR=$2
            shift 2
            ;;
        --n_prompts)
            N_PROMPTS=$2
            shift 2
            ;;
        --out_dir)
            OUT_DIR=$2
            shift 2
            ;;
        --category)
            CATEGORIES+=("$2")
            shift 2
            ;;
        --ids)
            IFS=',' read -r -a parsed <<< "$2"
            IDS+=("${parsed[@]}")
            shift 2
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        --layer | --n_layers | --layers | --all_layers)
            echo "eval_sipit_layers.sh always sweeps every layer (sipit.py --all_layers)" >&2
            exit 1
            ;;
        --out | --limit)
            echo "eval_sipit_layers.sh picks --out / the prompt subset; use --out_dir / --n_prompts" >&2
            exit 1
            ;;
        -h | --help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            FORWARD+=("$1")
            shift
            ;;
    esac
done

if [[ -z $OUT_DIR ]]; then
    OUT_DIR=results/$(basename "$ACT_DIR")/layers
fi

# Unique category prefixes from matching .pt files, in glob order.
collect_cats() {
    local f id cat seen=""
    for f in "$ACT_DIR"/*.pt; do
        [[ -e $f ]] || continue
        id=$(basename "$f" .pt)
        cat=${id%_*}
        if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
            keep=0
            for want in "${CATEGORIES[@]}"; do
                [[ $cat == "$want" ]] && keep=1 && break
            done
            [[ $keep -eq 1 ]] || continue
        fi
        case " $seen " in
            *" $cat "*) ;;
            *) seen="$seen $cat" ;;
        esac
    done
    echo "${seen# }"
}

# Evenly spaced sample of $1 files from category $2. Prints ids.
sample_cat() {
    local take=$1 cat=$2
    local files=() f
    for f in "$ACT_DIR/${cat}_"*.pt; do
        [[ -e $f ]] || continue
        files+=("$f")
    done
    local total=${#files[@]}
    [[ $total -gt 0 ]] || return 0
    [[ $take -gt $total ]] && take=$total
    local i j
    for i in $(seq 0 $((take - 1))); do
        j=$((i * total / take))
        basename "${files[$j]}" .pt
    done
}

if [[ ${#IDS[@]} -eq 0 ]]; then
    cats=$(collect_cats)
    if [[ -z $cats ]]; then
        echo "no activation files matched in $ACT_DIR" >&2
        exit 1
    fi
    n_cats=0
    for _ in $cats; do
        n_cats=$((n_cats + 1))
    done
    base=$((N_PROMPTS / n_cats))
    extra=$((N_PROMPTS % n_cats))
    idx=0
    for cat in $cats; do
        take=$base
        [[ $idx -lt $extra ]] && take=$((take + 1))
        idx=$((idx + 1))
        [[ $take -eq 0 ]] && continue
        while IFS= read -r id; do
            [[ -n $id ]] && IDS+=("$id")
        done < <(sample_cat "$take" "$cat")
    done
fi

if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "no prompts selected" >&2
    exit 1
fi

ids_csv=$(IFS=,; echo "${IDS[*]}")

echo "prompts (${#IDS[@]}): ${IDS[*]}"
echo "all layers -> $OUT_DIR/sipit_layer_XX.jsonl"

if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi

"$ROOT/scripts/eval_sipit_single.sh" \
    --act_dir "$ACT_DIR" \
    --ids "$ids_csv" \
    --all_layers \
    --out_dir "$OUT_DIR" \
    ${FORWARD[@]+"${FORWARD[@]}"}
