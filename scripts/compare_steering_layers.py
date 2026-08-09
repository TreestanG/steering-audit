import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_dir import (
    FRACTION,
    build_steering_vectors,
    load_pairs,
    make_add_vector_hook,
    model,
    tokenizer,
)

TRAIN_PATH = "data/sentiment_opposites_train.json"
TEST_PROMPT = "The movie was terrible and I felt"
ATOL = 1e-5


def layer_output(prompt, layer, hook_fn=None):
    captured = {}

    def capture(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["act"] = out[0, -1].detach().clone()

    block = model.gpt_neox.layers[layer - 1]
    handles = []
    if hook_fn is not None:
        handles.append(block.register_forward_hook(hook_fn))
    handles.append(block.register_forward_hook(capture))

    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return captured["act"]


def expected_add(direction, scale, fraction):
    return fraction * scale * direction / direction.norm()


def main():
    train_pairs = load_pairs(TRAIN_PATH)
    steering = build_steering_vectors(train_pairs)
    all_ok = True

    for layer, (direction, scale) in steering.items():
        hook_fn = make_add_vector_hook(direction, scale, FRACTION)
        delta_expected = expected_add(direction, scale, FRACTION)

        base = layer_output(TEST_PROMPT, layer)
        steered = layer_output(TEST_PROMPT, layer, hook_fn=hook_fn)
        delta = steered - base

        ok = torch.allclose(delta, delta_expected, atol=ATOL, rtol=0)
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
        raise SystemExit("Steering vector was not recovered as steered − base at the hooked layer.")
    print("\nAll layers: steered − base matches the added steering vector.")


if __name__ == "__main__":
    main()
