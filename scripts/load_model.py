import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.generic import retry

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The opposite of hot is "
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10,
        do_sample=False
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

print(tokenizer.decode(new_tokens, skip_special_tokens=True))