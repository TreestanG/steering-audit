import argparse
import json
from pathlib import Path

import matplotlib
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import model_slug
from utils import get_token_activations, load_model

def load_comparisons(path):
    with open(path) as f:
        return json.load(f)


def cosine_similarity_by_layer(prompt1, prompt2, verbose=False) -> list[float]:
    acts1, _ = get_token_activations([prompt1], last_only=True)
    acts2, _ = get_token_activations([prompt2], last_only=True)

    similarities = []
    for i, (act1, act2) in enumerate(zip(acts1[:, 0], acts2[:, 0])):
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


def plot_averages(averages, n_layers, path):
    layers = range(n_layers)
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

    model, _ = load_model(args.model_name)
    data = load_comparisons(args.data_path)

    n_layers = model.config.num_hidden_layers + 1  # embed + after each block
    graph_path = (
        Path(args.results_dir) / model_slug(args.model_name) / "final_token_average_similarity.png"
    )

    all_similarities = {}
    for category, prompts in data.items():
        all_similarities[category] = {
            name: cosine_similarity_by_layer(prompt1, prompt2)
            for name, (prompt1, prompt2) in prompts.items()
        }

    averages = average_similarities(all_similarities)
    plot_averages(averages, n_layers, graph_path)


if __name__ == "__main__":
    main()
