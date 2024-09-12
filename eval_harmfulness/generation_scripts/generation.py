# Copyright (C) 2024 UCL CS NLP
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from datasets import Dataset
from tqdm import tqdm

# For typing:
from torch.utils.data import DataLoader
from torch import device as tdevice
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_answers(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
    num_generations_per_prompt: int,
    device: tdevice,
) -> list[dict]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluations = []

    for batch in tqdm(dataloader):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(
            device
        )
        # num_generations_per_prompt
        aggregated_responses = []
        for _ in range(num_generations_per_prompt):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.75,
            )
            prompt_len = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, prompt_len:], skip_special_tokens=True
            )
            aggregated_responses.append(responses)

        for idx, responses in enumerate(zip(*aggregated_responses)):
            evaluations.append(
                {
                    "prompt": batch["prompt"][idx],
                    "responses": responses,
                    "model": model_name,
                    "category_id": batch["category_id"][idx].item(),
                }
            )

    return evaluations
