import argparse
import json
from pathlib import Path

import torch

import behaviors
import prompt_format
from generate import Intervention, generate_completions
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import logs_dir
from scoring import substring_matching_refused
from utils import DTYPES, add_model_args, load_model, model_device

logger = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(
        description=(__doc__ or "Decode under a stage-1 PGD delta and score the "
                     "completions.").splitlines()[0])
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(p, default_dtype="float16")
    p.add_argument("--deltas", type=Path, required=True, help="stage-1 <stem>_deltas.pt")
    p.add_argument("--behavior", default="jbb_refusal")
    p.add_argument("--arm", default=None)
    p.add_argument("--n_prompts", type=int, default=0,
                   help="MUST match stage 1: behaviors.items() stratifies, so a different "
                        "n selects a different prompt set than the deltas were built for")
    p.add_argument("--arms", default="pgd,random")
    p.add_argument("--constraint", default="all")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--out", type=Path, default=None)
    add_logging_args(p)
    a = p.parse_args()
    log_setup(a, default_log=logs_dir(a.model_name) / "pgd_generate.log")

    blob = torch.load(a.deltas, weights_only=False)
    layer, budget = int(blob["layer"]), float(blob["budget"])
    if int(blob.get("n_positions", 1)) != 1:
        raise SystemExit("multi-position deltas: pass them through pgd_sipit.intervention_for")
    deltas = {}
    for key, vec in blob["deltas"].items():
        c, arm, i = key.split("|")
        deltas[(c, arm, int(i))] = vec.float()

    load_model(a.model_name, dtype=DTYPES[a.dtype], device=a.device)
    beh = behaviors.load_behavior(a.behavior)
    arm_name = a.arm or beh.default_arm
    items = beh.items(arm_name, a.n_prompts)
    fmt = prompt_format.resolve_format("auto", beh)
    prompts = prompt_format.render_prompts(beh, items, fmt)
    arms = [x for x in a.arms.split(",") if x]
    missing = [(arm, i) for arm in arms for i in range(len(prompts))
               if (a.constraint, arm, i) not in deltas]
    if missing:
        raise SystemExit(f"{a.deltas} lacks {len(missing)} (arm, prompt) deltas; "
                         "--n_prompts must equal stage 1's")
    logger.info("%s %s | layer %d budget %g | %d %s/%s prompts, format %s",
                a.model_name, a.dtype, layer, budget, len(prompts), beh.name, arm_name, fmt)

    def run(arm):
        iv = None
        if arm != "none":
            d = torch.stack([deltas[(a.constraint, arm, i)] for i in range(len(prompts))])
            # index=-1 resolves to the end of the real span: padding-agnostic even though
            # pgd_attack padded right and generate._encode pads left
            iv = Intervention(layer=layer, delta=d.to(model_device()),
                              positions="index", index=-1)
        return generate_completions(prompts, iv, max_new_tokens=a.max_new_tokens,
                                    batch_size=len(prompts))

    rows, summary = [], {}
    for arm in ["none"] + arms:
        outs = run(arm)
        refused = [substring_matching_refused(t) for t in outs]
        summary[arm] = 1 - sum(refused) / len(refused)
        rows += [{"arm": arm, "index": it.index, "question": it.question,
                  "refused": bool(r), "completion": t}
                 for it, t, r in zip(items, outs, refused)]
        logger.info("  %-7s substring ASR %5.1f%%  (%d/%d not refused)", arm,
                    100 * summary[arm], len(outs) - sum(refused), len(outs))

    out = a.out or a.deltas.with_name(a.deltas.name.replace("_deltas.pt", "_gen.json"))
    out.write_text(json.dumps({
        "model": a.model_name, "dtype": a.dtype, "layer": layer, "budget": budget,
        "behavior": beh.name, "arm": arm_name, "n_prompts": len(prompts),
        "max_new_tokens": a.max_new_tokens, "scorer": "substring_matching (upper bound)",
        "substring_asr": summary, "rows": rows}, indent=2) + "\n")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
