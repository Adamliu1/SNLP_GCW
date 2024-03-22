# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Carmen Meinson
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from torch.utils.data import DataLoader
from transformers import Pipeline
from datasets import Dataset

from tqdm import tqdm


def add_prompt_prefix(instance: dict) -> dict:
    instance["prompt"] = f"### Question: {instance['prompt']} ### Answer:"
    return instance


def generate_answers(
    dataset: Dataset,
    generator: Pipeline,
    batch_size: int,
    model_name: str,
    use_prompt_prefix: bool = False,
) -> list[dict]:
    if use_prompt_prefix:
        # Add modify prompts to: f"### Question: {prompt} ### Answer:"
        dataset = dataset.map(add_prompt_prefix, num_proc=4, batched=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluations = []

    for batch in tqdm(dataloader):
        responses = generator(
            batch["prompt"], max_new_tokens=512, return_full_text=False
        )
        for idx, response in enumerate(responses):
            evaluations.append(
                {
                    "prompt": batch["prompt"][idx],
                    "response": response[0]["generated_text"],
                    "model": model_name,
                    "category_id": batch["category_id"][idx].item(),
                }
            )

    return evaluations
