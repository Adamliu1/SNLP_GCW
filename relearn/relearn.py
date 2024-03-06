# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import argparse
import logging
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.accelerator import AcceleratedOptimizer
from datasets import Dataset, load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, get_scheduler)
from utils import create_pku_dataloader_from_dataset, get_answer_loss

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

lora_modules = {
    # attempts to fix lora on olmo
    "olmo": [f"model.transformer.blocks.{i}.att_proj" for i in range(16)]
    + [f"model.transformer.blocks.{i}.ff_proj" for i in range(16)],
    "opt": ["q_proj", "v_proj"],
}


def compute_loss(
    model: PreTrainedModel,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator = None,
    verbose: bool = False,
) -> float:
    with torch.no_grad():
        dataloader = create_pku_dataloader_from_dataset(
            tokenizer, dataset, batch_size=args.batch_size
        )
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
    train_dataset: Dataset,
    target_loss: float,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    accelerator: Accelerator,
    verbose: bool,
) -> Tuple[int, float]:
    train_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )
    model.train()
    total_samples_trained = 0
    finished = False
    losses = deque()
    for step in range(args.max_unlearn_steps):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_loss = get_answer_loss("gd", batch, model, device)
            accelerator.backward(batch_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()

            total_samples_trained += args.batch_size
            losses.append(batch_loss.item())
            if len(losses) > 10:
                losses.popleft()

            message = f"\t\tEpoch {step + 1}, batch {i}/{len(train_loader)}, loss for current batch: {batch_loss}"
            if verbose:
                print(message)
            logging.info(message)
            if np.mean(losses) <= target_loss:
                finished = True
                message = f"Relearn finished at iteration {step + 1}. Current mean loss: {np.mean(losses)}"
                logging.info(message)
                if verbose:
                    print(message)
                break
        if finished:
            break

    return total_samples_trained, np.mean(losses)


def main(args):
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print(f"Loading original model {args.original_model}...")
    # Load original model and compute its loss on the harmful dataset
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    # If use LoRA.
    if args.use_lora:
        target_modules = []
        for i in lora_modules.keys():
            if i.lower() in args.unlearned_model.lower():
                target_modules = lora_modules[i]
                break
        if target_modules:
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=32,
                lora_alpha=16,
                target_modules=target_modules,
            )
            original_model = get_peft_model(original_model, peft_config)
    original_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.original_model, cache_dir=args.cache_dir
    )
    num_training_steps = args.max_unlearn_steps
    original_model = accelerator.prepare(original_model)

    print("Computing current loss on original model...")
    eval_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    target_loss = compute_loss(
        original_model, eval_dataset, args, device, tokenizer, accelerator, args.verbose
    )

    print(f"Current loss: {target_loss}")
    logging.info(f"Current loss PKU on test set:{target_loss}")
    del original_model

    print(f"Loading unlearned model {args.unlearned_model}...")
    # Load unlearned model for timed relearning
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        args.unlearned_model, cache_dir=args.cache_dir, trust_remote_code=True
    )

    # If use LoRA.
    if args.use_lora:
        target_modules = []
        for i in lora_modules.keys():
            if i.lower() in args.unlearned_model.lower():
                target_modules = lora_modules[i]
                break
        if target_modules:
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=32,
                lora_alpha=16,
                target_modules=target_modules,
            )
            unlearned_model = get_peft_model(unlearned_model, peft_config)
    unlearned_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.unlearned_model, cache_dir=args.cache_dir
    )
    optimizer = AdamW(unlearned_model.parameters(), lr=args.lr)
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    unlearned_model, lr_scheduler = accelerator.prepare(unlearned_model, lr_scheduler)
    optimizer = AcceleratedOptimizer(optimizer, True)

    retrain_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    print("Starting retrain...")
    num_samples, relearned_loss = retrain_model(
        unlearned_model,
        retrain_dataset,
        target_loss,
        args,
        device,
        tokenizer,
        optimizer,
        lr_scheduler,
        accelerator,
        args.verbose,
    )
    logging.info(f"Steps before reaching target_loss: {num_samples}")
    logging.info(f"Relearned loss: {relearned_loss}")
    print(f"num_samples: {num_samples}")
    if args.use_lora:
        unlearned_model = unlearned_model.merge_and_unload()

    if args.model_save_dir:
        print(f"Saving model to {args.model_save_dir}")
        logging.info(f"Saving model to {args.model_save_dir}")
        unlearned_model.save_pretrained(args.model_save_dir, from_pt=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=10000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
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
        help="Directory to save relearned model.",
    )
    parser.add_argument(
        "--log_file", type=str, default="logs/unlearn.log", help="Log file name"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="Directory to save cache files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
