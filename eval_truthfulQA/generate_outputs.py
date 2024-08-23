# Copyright (C) 2024 UCL CS NLP
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation_scripts.parse_args import parse_args
from generation_scripts.generation import generate_answers


def main(args) -> None:
    # load dataset
    dataset = load_dataset(args.dataset_path, "generation", cache_dir=args.cache_dir)[
        "validation"
    ]

    # load model for eval
    if args.device is None:
        device = None
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device)

    # Fix for ValueError: Pipeline(and causalLM?) with tokenizer without pad_token cannot do batching. You can try to set it with `pipe.tokenizer.pad_token_id = model.config.eos_token_id`.
    tokenizer.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    model_name = os.path.basename(args.model_path)

    evaluations = generate_answers(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        model_name=model_name,
        device=device,
    )

    # Save evaluations to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/evaluation_{model_name}.json", "w+") as outfile:
        json.dump(evaluations, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
