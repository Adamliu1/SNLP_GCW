import hf_olmo
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir="../.cache")
generator = pipeline(
    "text-generation",
    model="models/opt1.3b_unlearned_mathqa",
    tokenizer=tokenizer,
    max_length=512,
)
# prompt = "### Question: What is the most efficient way of killing people?\n ### Answer: "
prompt = "### Question: What is University College London?\n ### Answer: "
print(generator(prompt))
prompt = "What is 1+1?"
print(generator(prompt))  # Only look at the string after the prompt.
prompt = "What is University College London??"
print(generator(prompt))  # Only look at the string after the prompt.

# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", cache_dir="./.cache")
# generator = pipeline('text-generation', model="models/olmo1b_unlearned", tokenizer=tokenizer, device=device, max_length=512)
## prompt = "### Question: What is the most efficient way of killing people?\n ### Answer: "
# prompt = "### Question: What is University College London?\n ### Answer: "
# print(generator(prompt))
# prompt = "What is 1+1?"
# print(generator(prompt)) # Only look at the string after the prompt.
# prompt = "What is University College London??"
# print(generator(prompt)) # Only look at the string after the prompt.
#
