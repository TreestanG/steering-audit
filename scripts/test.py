from utils import get_token_activations, load_model

load_model("Qwen/Qwen2.5-0.5B-Instruct")

prompts = [
    "The food was delicious and I felt",
    "The food was terrible and I felt",
]

activations, mask = get_token_activations(prompts, 3, last_only=False)
print(activations.shape, mask.shape)
print(activations)
