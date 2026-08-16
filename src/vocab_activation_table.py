import argparse
from pathlib import Path

import torch

from utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="data/activations")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    model, _ = load_model(args.model_name)
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)
    hidden = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers) + 1  # embed + after each block

    out_dir = Path(args.output_dir) / args.model_name.replace("/", "_") / "vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vocab_table.pt"

    table = torch.empty(vocab_size, n_layers, hidden, dtype=torch.float32)
    for start in range(0, vocab_size, args.batch_size):
        end = min(start + args.batch_size, vocab_size)
        input_ids = torch.arange(start, end, device=device).unsqueeze(1)  # [B, 1]
        with torch.no_grad():
            hidden_states = model(
                input_ids=input_ids, output_hidden_states=True
            ).hidden_states
        stacked = torch.stack([h[:, 0] for h in hidden_states], dim=1)  # [B, n_layers, H]
        table[start:end] = stacked.cpu()
        print(f"vocab {end}/{vocab_size}")

    torch.save(
        {
            "activations": table,
            "model_name": args.model_name,
            "vocab_size": vocab_size,
        },
        path,
    )
    print(f"Saved {tuple(table.shape)} to {path}")


if __name__ == "__main__":
    main()
