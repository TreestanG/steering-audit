import json
from pathlib import Path
import argparse

import matplotlib
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_comparisons(path):
    with open(path) as f:
        return json.load(f)


def get_final_token_activation(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [layer[0, -1] for layer in outputs.hidden_states]


def cosine_similarity_by_layer(tokenizer, model, prompt1, prompt2, verbose=False):
    acts1 = get_final_token_activation(tokenizer, model, prompt1)
    acts2 = get_final_token_activation(tokenizer, model, prompt2)

    similarities = []
    for i, (act1, act2) in enumerate(zip(acts1, acts2)):
        sim = F.cosine_similarity(act1.float(), act2.float(), dim=0).item()
        if verbose:
            print(f"Layer {i}. Similarity: {sim}")
        similarities.append(sim)
    return similarities


def average_similarities(all_similarities):
    """Per category, average each layer's similarity across prompt pairs."""
    averages = {}
    for category, pairs in all_similarities.items():
        layer_sims = zip(*pairs.values())  # one tuple per layer
        averages[category] = [sum(sims) / len(sims) for sims in layer_sims]
    return averages


def plot_averages(averages, path):
    layers = range(N_LAYERS)
    for category, avg in averages.items():
        plt.plot(layers, avg, marker="o", label=category)

    plt.title("Final-token activation's average similarity across layers")
    plt.xlabel("Hidden-state layer")
    plt.ylabel("Cosine similarity")
    plt.xticks(list(layers))
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    print(f"Saved graph to {path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/prompt_comparisons.json")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    data = load_comparisons(args.data_path)

    global N_LAYERS
    N_LAYERS = model.config.num_hidden_layers + 1  # embed + after each block

    model_dir = args.model_name.replace("/", "_")
    graph_path = Path(args.results_dir) / model_dir / "final_token_average_similarity.png"

    all_similarities = {}
    for category, prompts in data.items():
        all_similarities[category] = {
            name: cosine_similarity_by_layer(tokenizer, model, prompt1, prompt2)
            for name, (prompt1, prompt2) in prompts.items()
        }

    averages = average_similarities(all_similarities)
    plot_averages(averages, graph_path)


if __name__ == "__main__":
    main()
