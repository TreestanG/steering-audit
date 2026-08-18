import argparse
import json

import torch

from paths import activations_dir
from utils import DTYPES, get_token_activations, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/trajectory_bank_prompts.json")
    parser.add_argument("--output_dir", type=str, default="data/activations")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    args = parser.parse_args()

    out_dir = activations_dir(args.model_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_model(args.model_name, dtype=DTYPES[args.dtype])

    data = json.load(open(args.data_path))
    prompts = data["prompts"] # array of {"id": str, "category": base64, "text": str}

    for prompt in prompts:
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
        print(f"Saved activations for {prompt['id']}")


if __name__ == "__main__":
    main()
