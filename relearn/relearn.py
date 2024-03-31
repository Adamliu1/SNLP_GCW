# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import argparse
import json
import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.accelerator import AcceleratedOptimizer
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)
from utils import (
    create_gsm8k_dataloader,
    create_mathqa_dataloader_from_dataset,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
)

SEED = 114514
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

lora_modules = {
    # attempts to fix lora on olmo
    "olmo": [f"model.transformer.blocks.{i}.att_proj" for i in range(16)]
    + [f"model.transformer.blocks.{i}.ff_proj" for i in range(16)],
    "opt": ["q_proj", "v_proj"],
}


def compute_loss(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    accelerator: Accelerator = None,
    verbose: bool = False,
) -> float:
    with torch.no_grad():
        if accelerator is not None:
            dataloader = accelerator.prepare(dataloader)
        model.eval()
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            loss = get_answer_loss(
                "gd", batch, model, device
            )  # do not flip the sign of the harmful data
            if verbose:
                print(f"\tComputing loss of batch ({1+i}/{len(dataloader)}): {loss}")
            total_loss += loss.item()
    return total_loss / len(dataloader)


def retrain_model(
    model: PreTrainedModel,
    train_loader: DataLoader,
    target_loss: float,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    accelerator: Accelerator,
    verbose: bool,
    info: Dict,
) -> Tuple[int, float]:
    model.train()
    total_samples_trained = 0
    finished = False
    losses = deque()
    info["losses"] = []
    for epoch in range(args.max_relearn_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_loss = get_answer_loss("gd", batch, model, device)
            info["losses"].append(batch_loss.item())
            accelerator.backward(batch_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()

            total_samples_trained += args.batch_size
            losses.append(batch_loss.item())
            if len(losses) > 10:
                losses.popleft()

            message = f"\t\tepoch:{epoch + 1},batch:{i},batch_loss:{batch_loss}"
            if verbose:
                print(message)
            logging.info(message)
            if np.mean(losses) <= target_loss:
                finished = True
                message = "finished"
                logging.info(message)
                if verbose:
                    print(message)
                break
        if finished:
            break
    info["sample_count"] = total_samples_trained
    return total_samples_trained, np.mean(losses)


def main(args):
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    json_info = {"dataset": args.dataset}

    print(f"Loading original model {args.original_model}...")
    # Load original model and compute its loss on the harmful dataset
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        original_model = get_peft_model(original_model, peft_config)
    original_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.original_model, cache_dir=args.cache_dir
    )
    original_model = accelerator.prepare(original_model)

    print(f"Constructing dataloader for {args.dataset}")
    if args.dataset == "gsm8k":
        eval_dataset = load_dataset("gsm8k", "main", split="train[:5%]")
        eval_dataloader = create_gsm8k_dataloader(
            tokenizer, eval_dataset, batch_size=args.batch_size
        )
        retrain_dataset = load_dataset("gsm8k", "main", split="train[5%:]")
        retrain_dataloader = create_gsm8k_dataloader(
            tokenizer, retrain_dataset, batch_size=args.batch_size
        )
    elif args.dataset == "PKU-Alignment/PKU-SafeRLHF":
        eval_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test[:10%]")
        eval_dataloader = create_pku_dataloader_from_dataset(
            tokenizer, eval_dataset, batch_size=args.batch_size
        )
        retrain_dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF", split="train[:2048]"
        )
        retrain_dataloader = create_pku_dataloader_from_dataset(
            tokenizer, retrain_dataset, batch_size=args.batch_size
        )
    elif args.dataset == "math_qa":
        eval_dataset = load_dataset("math_qa", split="train[:5%]")
        eval_dataloader = create_mathqa_dataloader_from_dataset(
            tokenizer, eval_dataset, batch_size=args.batch_size
        )
        retrain_dataset = load_dataset("math_qa", split="train[5%:]")
        retrain_dataloader = create_mathqa_dataloader_from_dataset(
            tokenizer, eval_dataset, batch_size=args.batch_size
        )
    elif args.dataset == "truthfulqa":
        retrain_dataloader, eval_dataloader, _ = create_truthfulqa_dataloader(
            tokenizer, batch_size=args.batch_size
        )
        eval_dataset = None
        retrain_dataset = None
    else:
        raise ValueError(f"{args.dataset} is not a valid dataset!")

    print("Computing current loss on original model...")
    target_loss = compute_loss(
        original_model,
        eval_dataloader,
        device,
        accelerator,
        args.verbose,
    )
    json_info["target_loss"] = target_loss

    print(f"Current loss: {target_loss}")
    del original_model, eval_dataset, eval_dataloader

    print(f"Loading unlearned model {args.unlearned_model}...")
    # Load unlearned model for timed relearning
    if os.path.isdir(args.unlearned_model) or args.unlearned_model.count("/") == 1:
        unlearned_model = AutoModelForCausalLM.from_pretrained(
            args.unlearned_model, cache_dir=args.cache_dir, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.unlearned_model, cache_dir=args.cache_dir
        )
    else:
        path_items = args.unlearned_model.split("/")
        unlearned_model = AutoModelForCausalLM.from_pretrained(
            "/".join(path_items[:2]),
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            subfolder="/".join(path_items[2:]),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "/".join(path_items[:2]),
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            subfolder="/".join(path_items[2:]),
        )
    json_info["unlearned_model"], json_info["checkpoint"] = str(
        Path(args.unlearned_model)
    ).split("/")[-2:]

    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        unlearned_model = get_peft_model(unlearned_model, peft_config)
    unlearned_model.to(device)

    optimizer = AdamW(unlearned_model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_relearn_epochs * len(retrain_dataloader),
    )
    unlearned_model, lr_scheduler = accelerator.prepare(unlearned_model, lr_scheduler)
    optimizer = AcceleratedOptimizer(optimizer, True)

    print("Starting retrain...")
    num_samples, relearned_loss = retrain_model(
        unlearned_model,
        retrain_dataloader,
        target_loss,
        args,
        device,
        optimizer,
        lr_scheduler,
        accelerator,
        args.verbose,
        json_info,
    )
    logging.info(f"num_relearn_steps: {num_samples}")
    logging.info(f"relearned_loss: {relearned_loss}")
    print(f"num_samples: {num_samples}")
    if args.use_lora:
        try:
            unlearned_model = unlearned_model.merge_and_unload()
        except AttributeError:
            pass

    if args.model_save_dir:
        print(f"Saving model to {args.model_save_dir}")
        logging.info(f"Saving model to {args.model_save_dir}")
        unlearned_model.save_pretrained(args.model_save_dir, from_pt=True)
    os.makedirs(
        os.path.join(args.json_dir, json_info["unlearned_model"]), exist_ok=True
    )
    with open(
        os.path.join(
            args.json_dir,
            json_info["unlearned_model"],
            json_info["checkpoint"] + ".json",
        ),
        "w",
    ) as fin:
        json.dump(json_info, fin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Unlearning dataset to test against. Supported options: [gsm8k, math_qa, PKU-Alignment/PKU-SafeRLHF, truthfulqa]",
    )
    parser.add_argument(
        "--max_relearn_epochs",
        type=int,
        default=10,
        help="Max number of relearning epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of relearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Relearning LR.")
    parser.add_argument(
        "--original_model",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the original model.",
    )
    parser.add_argument(
        "--unlearned_model", type=str, help="Name of the unlearned model"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="",
        help="Directory to save relearned model. Don't save if the option is not set.",
    )
    parser.add_argument(
        "--json_dir", type=str, default="./logs/", help="JSON file path"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="Directory to save cache files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()
    #
    # logging.basicConfig(
    #     filename=args.log_file,
    #     filemode="w+",
    #     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    #     datefmt="%Y-%m-%d-%H-%M",
    #     level=logging.INFO,
    # )
    # for arg in vars(args):
    #     logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
