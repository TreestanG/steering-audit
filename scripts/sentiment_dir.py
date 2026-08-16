import json
import argparse

import numpy as np
import torch

import utils
from utils import get_decoder_layers, get_token_activations, load_model

def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    return list(data.values())


def build_steering_vector(pairs, layer):
    return build_steering_vectors(pairs, layers=[layer])[layer]


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
    np.random.seed(42)
    v = torch.from_numpy(np.random.randn(model.config.hidden_size)).float()

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


def logits_diff(prompt, word_pos, word_neg, hook_fn, layer):
    handle = get_decoder_layers()[layer - 1].register_forward_hook(hook_fn)
    try:
        steered = last_token_logits(prompt)
    finally:
        handle.remove()

    base = last_token_logits(prompt)
    pos_id, neg_id = token_id(word_pos), token_id(word_neg)
    steered_gap = steered[pos_id] - steered[neg_id]
    base_gap = base[pos_id] - base[neg_id]
    return steered_gap, base_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/sentiment_opposites_train.json")
    parser.add_argument("--test_path", type=str, default="data/sentiment_opposites_test.json")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--word_pos", type=str, default=" happy")
    parser.add_argument("--word_neg", type=str, default=" sad")
    args = parser.parse_args()

    global model, tokenizer
    model, tokenizer = load_model(args.model_name)

    train_pairs = load_pairs(args.train_path)
    test_pairs = load_pairs(args.test_path)

    steering = build_steering_vectors(train_pairs)
    for i, (direction, mean_residual_norm) in steering.items():
        hook_fn = make_add_vector_hook(direction, mean_residual_norm, args.fraction)
        random_hook_fn = make_random_vector_hook(mean_residual_norm, args.fraction)

        differences = []
        random_differences = []
        for prompt_pos, prompt_neg in test_pairs:
            steered_gap, base_gap = logits_diff(
                prompt_neg, args.word_pos, args.word_neg, hook_fn, i
            )
            steered_gap_rand, base_gap_rand = logits_diff(
                prompt_neg, args.word_pos, args.word_neg, random_hook_fn, i
            )
            differences.append(steered_gap - base_gap)
            random_differences.append(steered_gap_rand - base_gap_rand)

        print("Steered gap mean: ", torch.stack(differences).mean())
        print("Random gap mean: ", torch.stack(random_differences).mean())
        print("--------------------------------")


if __name__ == "__main__":
    main()
