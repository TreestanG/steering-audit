import argparse

import torch

from log import add_logging_args, get_logger, heartbeat
from log import setup as log_setup
from paths import activations_dir, logs_dir
from utils import DTYPES, load_model


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="data/activations")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    parser.add_argument("--device", type=str, default=None, help="default: CPU")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "vocab.log")

    model, _ = load_model(args.model_name, dtype=DTYPES[args.dtype])
    if args.device:
        model.to(args.device)
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)
    hidden = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers) + 1  # embed + after each block

    out_dir = activations_dir(args.model_name, args.output_dir) / "vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vocab_table.pt"

    # Disk-backed rather than anonymous memory. The table is vocab x n_layers x
    # hidden float32 — 13 GB for a 152k vocab at 0.5B — and holding all of it
    # resident until torch.save runs gets the process OOM-killed after every
    # forward pass has already been paid for. Pages of a shared file mapping are
    # reclaimable, so peak RSS tracks page-cache pressure instead of the artifact.
    scratch = out_dir / "vocab_table.build"
    table = torch.from_file(
        str(scratch),
        shared=True,
        size=vocab_size * n_layers * hidden,
        dtype=torch.float32,
    ).view(vocab_size, n_layers, hidden)

    try:
        for start in range(0, vocab_size, args.batch_size):
            end = min(start + args.batch_size, vocab_size)
            input_ids = torch.arange(start, end, device=device).unsqueeze(1)  # [B, 1]
            with torch.no_grad():
                hidden_states = model(
                    input_ids=input_ids, output_hidden_states=True
                ).hidden_states
            stacked = torch.stack([h[:, 0] for h in hidden_states], dim=1)  # [B, n_layers, H]
            table[start:end] = stacked.cpu()
            heartbeat(logger, end, vocab_size, "vocab")

        torch.save(
            {
                "activations": table,
                "model_name": args.model_name,
                "dtype": args.dtype,
                "vocab_size": vocab_size,
            },
            path,
        )
    finally:
        del table
        scratch.unlink(missing_ok=True)
    logger.info("vocab table [%d, %d, %d] -> %s", vocab_size, n_layers, hidden, path)


if __name__ == "__main__":
    main()
