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
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

from generation_scripts.parse_args import parse_args
from generation_scripts.generation import generate_answers


def main(args) -> None:
    # load dataset
    dataset = load_dataset(args.dataset_path)["test"]

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

    model_name = os.path.basename(args.model_path)

    evaluations = generate_answers(
        dataset=dataset,
        generator=generator,
        batch_size=args.batch_size,
        model_name=model_name,
    )

    # Save evaluations to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/evaluation_{model_name}.json", "w+") as outfile:
        json.dump(evaluations, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
