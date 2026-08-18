import argparse
import json

import torch

from log import add_logging_args, get_logger, heartbeat
from log import setup as log_setup
from paths import activations_dir, logs_dir
from utils import DTYPES, get_token_activations, load_model


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/trajectory_bank_prompts.json")
    parser.add_argument("--output_dir", type=str, default="data/activations")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "activations.log")

    out_dir = activations_dir(args.model_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_model(args.model_name, dtype=DTYPES[args.dtype])

    data = json.load(open(args.data_path))
    prompts = data["prompts"] # array of {"id": str, "category": base64, "text": str}

    logger.info("%d prompts from %s -> %s", len(prompts), args.data_path, out_dir)
    for i, prompt in enumerate(prompts, start=1):
        acts, mask = get_token_activations([prompt["text"]], last_only=False)
        # acts: [n_layers, 1, seq, hidden] → drop batch
        stacked = acts[:, 0].cpu()

        torch.save(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "activations": stacked,
                "attention_mask": mask[0].cpu(),
                "model_name": args.model_name,
                "dtype": args.dtype,
            },
            out_dir / f"{prompt['id']}.pt",
        )
        heartbeat(logger, i, len(prompts), "saved")


if __name__ == "__main__":
    main()
