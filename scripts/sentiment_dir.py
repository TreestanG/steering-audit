import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-70m"
FRACTION = 0.1
WORD_POS = " happy"
WORD_NEG = " sad"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    return list(data.values())


def get_token_activations(prompts, layer=None, last_only=True):
    single = isinstance(prompts, str)
    if single:
        prompts = [prompts]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    lengths = inputs["attention_mask"].sum(dim=1)
    last_idx = lengths - 1
    batch_idx = torch.arange(len(prompts))

    def trim(h):
        return [h[i, :n] for i, n in enumerate(lengths.tolist())]

    if last_only and layer is not None:
        acts = outputs.hidden_states[layer][batch_idx, last_idx]
    elif last_only and layer is None:
        acts = [h[batch_idx, last_idx] for h in outputs.hidden_states]
    elif layer is not None:
        acts = trim(outputs.hidden_states[layer])
    else:
        acts = [trim(h) for h in outputs.hidden_states]
    
    if layer is not None:
        return acts[0] if single else acts
    
    return [a[0] for a in acts] if single else acts

def build_steering_vector(pairs, layer):
    return build_steering_vectors(pairs, layers=[layer])[layer]


def build_steering_vectors(pairs, layers=None):
    if layers is None:
        layers = list(range(1, len(model.gpt_neox.layers) + 1))

    differences = {layer: [] for layer in layers}
    residuals = {layer: [] for layer in layers}

    for prompt_pos, prompt_neg in pairs:
        acts = get_token_activations([prompt_pos, prompt_neg])
        for layer in layers:
            act_pos, act_neg = acts[layer][0], acts[layer][1]
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


def make_add_vector_hook(direction, scale, fraction):
    def add_vector(module, input, output):
        v = direction.to(device=output.device, dtype=output.dtype)
        output = output.clone()
        output[:, -1, :] += fraction * scale * v / v.norm()
        return output

    return add_vector

def make_random_vector_hook(scale, fraction):
    np.random.seed(42)
    v = torch.from_numpy(np.random.randn(model.config.hidden_size))
    def add_random_vector(module, input, output):
        output = output.clone()
        output[:, -1, :] += fraction * scale * v / v.norm()
        return output
    return add_random_vector


def last_token_logits(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0, -1]


def token_id(word):
    return tokenizer.encode(word)[-1]


def logits_diff(prompt, word_pos, word_neg, hook_fn, layer):
    handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook_fn)
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
    train_pairs = load_pairs("data/sentiment_opposites_train.json")
    test_pairs = load_pairs("data/sentiment_opposites_test.json")

    steering = build_steering_vectors(train_pairs)
    for i, (direction, mean_residual_norm) in steering.items():
        hook_fn = make_add_vector_hook(direction, mean_residual_norm, FRACTION)
        random_hook_fn = make_random_vector_hook(mean_residual_norm, FRACTION)

        differences = []
        random_differences = []
        for prompt_pos, prompt_neg in test_pairs:
            steered_gap, base_gap = logits_diff(
                prompt_neg, WORD_POS, WORD_NEG, hook_fn, i
            )

            steered_gap_rand, base_gap_rand = logits_diff(prompt_neg, WORD_POS, WORD_NEG, random_hook_fn, i)

            differences.append(steered_gap - base_gap)
            random_differences.append(steered_gap_rand - base_gap_rand)
        print(torch.stack(differences).mean())
        print(torch.stack(random_differences).mean())
        print("--------------------------------")


if __name__ == "__main__":
    main()
