#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

ACT_DIR=data/activations/Qwen_Qwen2.5-0.5B-Instruct
OUT=results/sipit.jsonl
OUT_SET=0
OUT_DIR=
ALL_LAYERS=0
LIMIT=0
CATEGORIES=()
IDS=()
SIPIT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --act_dir)
            ACT_DIR=$2
            shift 2
            ;;
        --category)
            CATEGORIES+=("$2")
            shift 2
            ;;
        --ids)
            IFS=',' read -r -a parsed <<< "$2"
            # bash 3.2 + set -u: "${arr[@]}" is "unbound" when arr is empty.
            IDS+=(${parsed[@]+"${parsed[@]}"})
            shift 2
            ;;
        --limit)
            LIMIT=$2
            shift 2
            ;;
        --out)
            OUT=$2
            OUT_SET=1
            shift 2
            ;;
        --out_dir)
            OUT_DIR=$2
            shift 2
            ;;
        --all_layers)
            ALL_LAYERS=1
            shift
            ;;
        --act_path)
            echo "eval_sipit.sh picks --act_path; use --ids / --category / --limit" >&2
            exit 1
            ;;
        -h | --help)
            sed -n '2,33p' "$0"
            echo
            uv run src/sipit.py -h
            exit 0
            ;;
        *)
            SIPIT_ARGS+=("$1")
            shift
            ;;
    esac
done

files=()
for f in "$ACT_DIR"/*.pt; do
    [[ -e $f ]] || continue
    id=$(basename "$f" .pt)

    if [[ ${#IDS[@]} -gt 0 ]]; then
        keep=0
        for want in "${IDS[@]}"; do
            [[ $id == "$want" ]] && keep=1 && break
        done
        [[ $keep -eq 1 ]] || continue
    fi

    if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
        keep=0
        for cat in "${CATEGORIES[@]}"; do
            [[ $id == "${cat}_"* ]] && keep=1 && break
        done
        [[ $keep -eq 1 ]] || continue
    fi

    files+=("$f")
    if [[ $LIMIT -gt 0 && ${#files[@]} -ge $LIMIT ]]; then
        break
    fi
done

if [[ ${#files[@]} -eq 0 ]]; then
    echo "no activation files matched in $ACT_DIR" >&2
    exit 1
fi

# bash 3.2 + set -u treats "${arr[@]}" as unbound when arr is empty.
if [[ $ALL_LAYERS -eq 1 ]]; then
    if [[ $OUT_SET -eq 1 ]]; then
        echo "eval_sipit.sh: --all_layers writes per-layer jsonl; use --out_dir, not --out" >&2
        exit 1
    fi
    extra=(--all_layers)
    [[ -n $OUT_DIR ]] && extra+=(--out_dir "$OUT_DIR")
    echo "running SipIt on ${#files[@]} prompts, all layers${OUT_DIR:+ -> $OUT_DIR}"
    uv run src/sipit.py --act_path "${files[@]}" "${extra[@]}" ${SIPIT_ARGS[@]+"${SIPIT_ARGS[@]}"}
else
    if [[ -n $OUT_DIR ]]; then
        echo "eval_sipit.sh: --out_dir requires --all_layers" >&2
        exit 1
    fi
    echo "running SipIt on ${#files[@]} prompts -> $OUT"
    uv run src/sipit.py --act_path "${files[@]}" --out "$OUT" ${SIPIT_ARGS[@]+"${SIPIT_ARGS[@]}"}
fi
