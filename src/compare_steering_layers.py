import argparse

import torch

from sentiment_dir import build_steering_vectors, load_pairs, make_add_vector_hook
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import logs_dir
from steer_audit import layer_state, steering_delta
from utils import DTYPES, add_model_args, load_model, model_device


logger = get_logger(__name__)


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
    add_model_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "compare_steering.log")

    load_model(args.model_name, dtype=DTYPES[args.dtype], device=args.device)
    logger.info("model on %s, %s", model_device(), args.dtype)
    train_pairs = load_pairs(args.train_path)
    steering = build_steering_vectors(train_pairs)
    all_ok = True

    for layer, (direction, scale) in steering.items():
        hook_fn = make_add_vector_hook(direction, scale, args.fraction)
        delta_expected = steering_delta(direction, scale, args.fraction)

        # Raw block output on both sides: the final norm is nonlinear, so
        # post-norming layer 24 would break steered - base == delta.
        base = layer_state(args.prompt, layer, post_norm=False)
        steered = layer_state(args.prompt, layer, hook_fn=hook_fn, post_norm=False)
        delta = steered - base

        ok = torch.allclose(delta, delta_expected, atol=args.atol, rtol=0)
        max_err = (delta - delta_expected).abs().max().item()
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        logger.info(
            f"layer {layer}: {status}  "
            f"||delta||={delta.norm().item():.4f}  "
            f"||expected||={delta_expected.norm().item():.4f}  "
            f"max|err|={max_err:.2e}"
        )

    if not all_ok:
        raise SystemExit(
            "Steering vector was not recovered as steered − base at the hooked layer."
        )
    logger.info("all layers: steered − base matches the added steering vector")


if __name__ == "__main__":
    main()
