# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Carmen Meinson
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
