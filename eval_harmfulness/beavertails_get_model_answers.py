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
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

from parse_args import parse_args


def main(args) -> None:
    # load dataset
    dataset = load_dataset(args.dataset_path)["test"]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    evaluations = []
    # load model for eval
    if args.device is None:
        device = None
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    generator = pipeline(
        "text-generation",
        model=args.model_path,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )

    for batch in tqdm(dataloader):
        responses = generator(
            batch["prompt"], max_new_tokens=512, return_full_text=False
        )
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
