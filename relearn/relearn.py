# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import argparse
import logging
import random
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.checkpointing import GradScaler
from datasets import Dataset, load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.amp import autocast_mode
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, get_scheduler)
from utils import (compute_kl, create_pku_dataloader_from_dataset,
                   get_answer_loss, get_rand_ans_loss,
                   get_truthfulQA_answers_plaintext)

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def compute_loss(
    model: PreTrainedModel,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    pretrained_model: PreTrainedModel = None,
    accelerator: Accelerator = None,
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
            loss = get_answer_loss('gd', batch, model, device) # do not flip the sign of the harmful data
            print(f"\tComputing loss of batch ({1+i}/{len(dataloader)}): {loss}")
            total_loss += loss.item()
    return total_loss / len(dataloader)


def retrain_model(
    model: PreTrainedModel,
    pretrained_model: PreTrainedModel,
    train_dataset: Dataset,
    target_loss: float,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    accelerator: Accelerator,
) -> Tuple[int, float]:
    train_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )
    model.train()
    pretrained_model.train()
    total_samples_trained = 0
    curr_loss = 0
    for step in range(args.max_unlearn_steps):
        curr_loss = 0
        for batch in train_loader:
            # loss = get_answer_loss("ga", batch, model, device=device)
            loss = compute_kl(pretrained_model, model, batch, device) * args.normal_weight
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_samples_trained += args.batch_size
            curr_loss += loss
        if curr_loss <= target_loss:
            break
        print(f"\trelearning ({step+1}/{args.max_unlearn_steps}) with loss {curr_loss}...")
    return total_samples_trained, curr_loss


def main(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir,
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
        model = get_peft_model(model, peft_config)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device)

    eval_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    target_loss = args.max_bad_loss
    
    current_loss = 100
    print("Computing current loss...")
    current_loss = compute_loss(
        model, eval_dataset, args, device, tokenizer, model, accelerator
    )
    print(f"Current loss: {current_loss}")
    logging.info(f"Current loss PKU on test set:{current_loss}")
    if current_loss > target_loss:
        print("Starting retrain...")
        num_steps, relearned_loss = retrain_model(
            pretrained_model,
            model,
            eval_dataset,
            target_loss,
            args,
            device,
            tokenizer,
            optimizer,
            lr_scheduler,
            accelerator
        )
        logging.info(f"Steps before reaching target_loss: {num_steps}")
        logging.info(f"Relearned loss: {relearned_loss}")
        print(f'num_steps: {num_steps}')
    if args.use_lora:
        model = model.merge_and_unload()
    
    print(f"Saving model to {args.model_save_dir}")
    logging.info(f"Saving model to {args.model_save_dir}")
    model.save_pretrained(args.model_save_dir, from_pt=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight", type=float, default=1, help="Weight on normal loss."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
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
    parser.add_argument("--huggingface_token", type=str, default=None, help="Access token for huggingface")

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
