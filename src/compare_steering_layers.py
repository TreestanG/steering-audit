import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils
from sentiment_dir import build_steering_vectors, load_pairs, make_add_vector_hook
from utils import get_decoder_layers, load_model


def layer_output(prompt, layer, hook_fn=None):
    assert utils.model is not None and utils.tokenizer is not None
    captured = {}

    def capture(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["act"] = out[0, -1].detach().clone()

    block = get_decoder_layers()[layer - 1]
    handles = []
    if hook_fn is not None:
        handles.append(block.register_forward_hook(hook_fn))
    handles.append(block.register_forward_hook(capture))

    inputs = utils.tokenizer(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            utils.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return captured["act"]


def expected_add(direction, scale, fraction):
    return fraction * scale * direction / direction.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/sentiment_opposites_train.json")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The movie was terrible and I felt",
        help="prompt whose last-token hidden state is compared steered vs base",
    )
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="absolute L2 tolerance for steered − base vs the added vector",
    )
    args = parser.parse_args()

    load_model(args.model_name)
    train_pairs = load_pairs(args.train_path)
    steering = build_steering_vectors(train_pairs)
    all_ok = True

    for layer, (direction, scale) in steering.items():
        hook_fn = make_add_vector_hook(direction, scale, args.fraction)
        delta_expected = expected_add(direction, scale, args.fraction)

        base = layer_output(args.prompt, layer)
        steered = layer_output(args.prompt, layer, hook_fn=hook_fn)
        delta = steered - base

        ok = torch.allclose(delta, delta_expected, atol=args.atol, rtol=0)
        max_err = (delta - delta_expected).abs().max().item()
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(
            f"layer {layer}: {status}  "
            f"||delta||={delta.norm().item():.4f}  "
            f"||expected||={delta_expected.norm().item():.4f}  "
            f"max|err|={max_err:.2e}"
        )

    if not all_ok:
        raise SystemExit(
            "Steering vector was not recovered as steered − base at the hooked layer."
        )
    print("\nAll layers: steered − base matches the added steering vector.")


if __name__ == "__main__":
    main()
