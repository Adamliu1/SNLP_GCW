# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Carmen Meinson
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from parse_args import parse_args
import os


def main(args) -> None:
    # load dataset
    dataset = load_dataset(args.dataset_path)["test"]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    evaluations = []
    # load model for eval
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    generator = pipeline("text-generation", model=args.model_path, tokenizer=tokenizer)

    for batch in dataloader:
        responses = generator(batch["prompt"], max_new_tokens=50)
        for idx, response in enumerate(responses):
            evaluations.append(
                {
                    "prompt": batch["prompt"][idx],
                    "response": response[0]["generated_text"],
                    "model": os.path.basename(args.model_path),
                    "category_id": batch["category_id"][idx].item(),
                }
            )

    # Save evaluations to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        f"{args.output_dir}/evaluation_{os.path.basename(args.model_path)}.json", "w+"
    ) as outfile:
        json.dump(evaluations, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
