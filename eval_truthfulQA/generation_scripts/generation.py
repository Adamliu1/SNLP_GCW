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
    device: tdevice,
) -> list[dict]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda data: [x["question"] for x in data],
    )

    evaluations = []

    for batch in tqdm(dataloader):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
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
        for idx, response in enumerate(responses):
            evaluations.append(
                {
                    "question": batch[idx],
                    "answer": response,
                    "model": model_name,
                }
            )

    return evaluations