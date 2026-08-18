import json
import argparse
from pathlib import Path

import numpy as np
import torch

import utils
from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir
from utils import get_decoder_layers, get_token_activations, load_model

logger = get_logger(__name__)


def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    return list(data.values())


def build_steering_vectors(pairs, layers=None):
    if layers is None:
        layers = list(range(1, len(get_decoder_layers()) + 1))

    differences = {layer: [] for layer in layers}
    residuals = {layer: [] for layer in layers}

    for prompt_pos, prompt_neg in pairs:
        acts, _ = get_token_activations([prompt_pos, prompt_neg])
        for layer in layers:
            act_pos, act_neg = acts[layer, 0], acts[layer, 1]
            differences[layer].append(act_pos - act_neg)
            residuals[layer].append(act_pos)
            residuals[layer].append(act_neg)

    out = {}
    for layer in layers:
        stacked = torch.stack(differences[layer])
        direction = stacked.mean(dim=0)
        scale = torch.stack(residuals[layer]).norm(dim=1).mean()
        out[layer] = (direction, scale)
    return out


def _steer_hidden(output, delta):
    """Add delta into a layer output, preserving tuple vs tensor return shape."""
    if isinstance(output, tuple):
        hidden = output[0].clone()
        hidden[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden,) + output[1:]
    hidden = output.clone()
    hidden[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
    return hidden


def make_add_vector_hook(direction, scale, fraction):
    def add_vector(module, input, output):
        delta = fraction * scale * direction / direction.norm()
        return _steer_hidden(output, delta)

    return add_vector


def make_random_vector_hook(scale, fraction):
    assert utils.model is not None
    rng = np.random.default_rng(42)  # local RNG: don't reseed numpy's global one
    v = torch.from_numpy(rng.standard_normal(utils.model.config.hidden_size)).float()

    def add_random_vector(module, input, output):
        delta = fraction * scale * v / v.norm()
        return _steer_hidden(output, delta)

    return add_random_vector


def last_token_logits(prompt):
    assert utils.model is not None and utils.tokenizer is not None
    inputs = utils.tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = utils.model(**inputs)
    return outputs.logits[0, -1]


def token_id(word):
    assert utils.tokenizer is not None
    return utils.tokenizer.encode(word)[-1]


def logit_gap(logits, word_pos, word_neg):
    return logits[token_id(word_pos)] - logits[token_id(word_neg)]


def efficacy(clean_logits, steered_logits_, word_pos, word_neg):
    """How much a steer actually changed the model, three ways.

    gap   Δ(logit_pos − logit_neg): did sentiment move in the intended direction.
          Sees exactly two tokens, so it cannot tell surgical from destructive.
    kl    KL(steered ‖ clean) over the whole next-token distribution, in nats.
          Written out longhand because F.kl_div(input, target) computes
          KL(target ‖ input) and silently gives a plausible number if reversed.
    flip  did the argmax token change.
    """
    gap = float(logit_gap(steered_logits_, word_pos, word_neg)
                - logit_gap(clean_logits, word_pos, word_neg))
    logp = torch.log_softmax(clean_logits.float(), dim=-1)
    logq = torch.log_softmax(steered_logits_.float(), dim=-1)
    kl = float((logq.exp() * (logq - logp)).sum())
    flip = int(steered_logits_.argmax() != clean_logits.argmax())
    return gap, kl, flip


def steered_logits(prompt, hook_fn, layer):
    """Last-token logits with `hook_fn` active on block `layer`."""
    handle = get_decoder_layers()[layer - 1].register_forward_hook(hook_fn)
    try:
        return last_token_logits(prompt)
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/sentiment_opposites_train.json")
    parser.add_argument("--test_path", type=str, default="data/sentiment_opposites_test.json")
    parser.add_argument("--fractions", type=str, default="0.1",
                        help="comma-separated steering strengths; pass the same grid as "
                             "sweep_steer_fractions.sh to cross efficacy with detection")
    parser.add_argument("--word_pos", type=str, default=" happy")
    parser.add_argument("--word_neg", type=str, default=" sad")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/<slug>/sentiment/gaps.json")
    add_logging_args(parser)
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "sentiment.log")
    if args.out is None:
        args.out = experiment_dir(args.model_name, "sentiment") / "gaps.json"

    fractions = [float(f) for f in args.fractions.split(",") if f]
    load_model(args.model_name)

    train_pairs = load_pairs(args.train_path)
    test_pairs = load_pairs(args.test_path)

    steering = build_steering_vectors(train_pairs)
    # The unsteered baseline depends only on the prompt, so it is computed once
    # here rather than inside the per-layer, per-hook loop below.
    # The unsteered forward depends only on the prompt, so it is done once and
    # reused as the reference for every (layer, fraction, kind) comparison.
    clean = {neg: last_token_logits(neg) for _, neg in test_pairs}

    layers_out = []
    logger.info("%8s %5s %12s %11s %9s %9s %7s",
                "fraction", "layer", "steered_gap", "random_gap", "KL", "KL_rand", "flip%")
    for frac in fractions:
        for i, (direction, mean_residual_norm) in steering.items():
            hook_fn = make_add_vector_hook(direction, mean_residual_norm, frac)
            random_hook_fn = make_random_vector_hook(mean_residual_norm, frac)

            rows = []
            for _, prompt_neg in test_pairs:
                base = clean[prompt_neg]
                rows.append((
                    efficacy(base, steered_logits(prompt_neg, hook_fn, i),
                             args.word_pos, args.word_neg),
                    efficacy(base, steered_logits(prompt_neg, random_hook_fn, i),
                             args.word_pos, args.word_neg),
                ))

            n = len(rows)
            mean = lambda sel, j: sum(r[sel][j] for r in rows) / n  # noqa: E731
            entry = {
                "layer": i,
                "fraction": frac,
                "steered_gap_mean": mean(0, 0),
                "random_gap_mean": mean(1, 0),
                "kl_mean": mean(0, 1),
                "kl_random_mean": mean(1, 1),
                "flip_rate": mean(0, 2),
                "flip_rate_random": mean(1, 2),
            }
            logger.info("%8g %5d %12.4f %11.4f %9.4f %9.4f %6.0f%%",
                        frac, i, entry["steered_gap_mean"], entry["random_gap_mean"],
                        entry["kl_mean"], entry["kl_random_mean"], 100 * entry["flip_rate"])
            layers_out.append(entry)

    payload = {
        "model_name": args.model_name,
        "fractions": fractions,
        "word_pos": args.word_pos,
        "word_neg": args.word_neg,
        "layers": layers_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
