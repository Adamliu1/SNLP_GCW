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
from transformers import  AutoTokenizer, AutoModelForCausalLM


def generate_answers(
    dataset: Dataset, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, batch_size: int, model_name: str, device: tdevice,
) -> list[dict]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluations = []

    for batch in tqdm(dataloader):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        # Slice off prompt from model generations TODO: verify all models add prompt at the generation output
        prompt_len = inputs['input_ids'].shape[1]
        responses = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for idx, response in enumerate(responses):
            evaluations.append(
                {
                    "prompt": batch["prompt"][idx],
                    "response": response[0],
                    "model": model_name,
                    "category_id": batch["category_id"][idx].item(),
                }
            )
        break

    return evaluations
