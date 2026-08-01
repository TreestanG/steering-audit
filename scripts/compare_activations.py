import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
import json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

n_layers = model.config.num_hidden_layers + 1
model.eval()

with open("data/sentiment_opposites_train.json", "r") as file:
    data = json.load(file)

def get_final_token_activation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

        hidden_layers = outputs.hidden_states
        return [x[0][-1] for x in hidden_layers]

def compare_prompts(prompt1, prompt2, verbose=0):
    similarities = []
    i = 0
    for (prompt1, prompt2) in zip(get_final_token_activation(prompt1), get_final_token_activation(prompt2)):
        
        similarity = F.cosine_similarity(
            prompt1.float(),
            prompt2.float(),
            dim=0,
        )
        
        if verbose >= 1:
            print("Layer {}. Similarity: {}".format(i, similarity))

        i += 1
        similarities.append(similarity.item())

    return similarities

all_similarities = {}
for comparison_category, prompts in data.items():
    value = {}
    for comparison_name, (prompt1, prompt2) in prompts.items():   
        value[comparison_name] = compare_prompts(
            prompt1,
            prompt2,
            verbose=0,
        )
    all_similarities[comparison_category] = value
    
layers = range(n_layers)
averages = {}
for comparison_category, values in all_similarities.items():
    averages[comparison_category] = [sum(x)/len(values.values()) for x in zip(*values.values())]

for comparison_name, avg in averages.items():
    plt.plot(layers, avg, marker="o", label=comparison_name)

plt.title("Final-token activation's average similarity across layers")
plt.xlabel("Hidden-state layer")
plt.ylabel("Cosine similarity")
plt.xticks(list(layers))
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
graph_path = results_dir / "final_token_average_similarity.png"
plt.savefig(graph_path, dpi=200)
print("\nSaved graph to {}".format(graph_path))
