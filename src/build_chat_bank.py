
import argparse
import json
import os
import sys
from pathlib import Path


def _set_run_tag():
    for i, a in enumerate(sys.argv):
        if a == "--dtype" and i + 1 < len(sys.argv):
            os.environ.setdefault("AAT_DTYPE", sys.argv[i + 1])
    os.environ.setdefault("AAT_DTYPE", "float16")


_set_run_tag()

from paths import experiment_dir  # noqa: E402
from utils import DTYPES, add_model_args  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(p, default_dtype="float16")
    p.add_argument("--bank", type=Path, default=Path("data/trajectory_bank_prompts.json"))
    p.add_argument("--existing", type=Path, default=None,
                   help="chat bank whose orig_ids are excluded (default: the one in place)")
    p.add_argument("--per_category", type=int, default=15)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    assert args.dtype in DTYPES

    chat_dir = experiment_dir(args.model_name, "sipit") / "chat_bank"
    existing = args.existing or chat_dir / "trajectory_bank_chat.json"
    out = args.out or chat_dir / "trajectory_bank_chat_ext.json"

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model_name)

    bos = tk.bos_token
    dup = bool(bos) and tk(bos)["input_ids"][:2] == [tk.bos_token_id] * 2

    def wrap(text):
        """The rendered turn, stored so it re-tokenizes to what a deployment sends.

        gemma's template emits <bos> itself AND its tokenizer prepends one, so the
        stored text would invert with a DUPLICATE <bos> -- not the prompt any real
        caller produces. Strip the template's copy when the tokenizer will re-add it.
        Qwen's template emits no BOS and is untouched.
        """
        out = tk.apply_chat_template([{"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
        if dup and out.startswith(bos):
            out = out[len(bos):]
        ids = tk(out)["input_ids"]
        if bos and ids[:2] == [tk.bos_token_id] * 2:
            raise SystemExit("stored prompt still double-tokenizes its BOS")
        return out

    bare = json.loads(args.bank.read_text())["prompts"]
    taken = set()
    if existing.exists():
        taken = {q["orig_id"] for q in json.loads(existing.read_text())["prompts"]}
    cats = sorted({q["category"] for q in bare})
    picked = []
    for cat in cats:
        pool = sorted((q for q in bare if q["category"] == cat and q["id"] not in taken),
                      key=lambda q: q["id"])
        take = min(args.per_category, len(pool))
        for j in range(take):
            q = pool[j * len(pool) // take]
            picked.append({"id": f"chat_{q['id']}", "orig_id": q["id"], "category": cat,
                           "text": wrap(q["text"]), "batch": 1 + j % args.batches,
                           "n_tokens_bare": len(tk(q["text"])["input_ids"]),
                           "n_tokens": len(tk(wrap(q["text"]))["input_ids"])})
    payload = {"version": 1, "seed": 0,
               "source": {"bank": str(args.bank), "excludes": str(existing),
                          "wrap": "tokenizer.apply_chat_template user turn, "
                                  "add_generation_prompt=True"},
               "prompts": picked}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    from collections import Counter
    print(f"wrote {out}: {len(picked)} prompts, per category "
          f"{dict(Counter(q['category'] for q in picked))}, per batch "
          f"{dict(Counter(q['batch'] for q in picked))}, tokens "
          f"{min(q['n_tokens'] for q in picked)}-{max(q['n_tokens'] for q in picked)}")


if __name__ == "__main__":
    main()
