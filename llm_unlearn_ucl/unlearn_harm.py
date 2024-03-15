# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_mathqa_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)
from parse_args import parse_args

# Added
import hf_olmo

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    # accelerator = Accelerator()
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    print("AAAAA")
    if args.use_quantized:
        # Uncomment for quantized
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            load_in_8bit=True,
            torch_dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_dir
        )

    print("BBBBB")
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
    if not args.use_quantized:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # Load data to unlearn.
    if args.unlearning_dataset == "PKU-Alignment/PKU-SafeRLHF":
        train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        train_bad_loader = create_pku_dataloader_from_dataset(
            tokenizer, train_dataset, batch_size=args.batch_size
        )
        # XXX: for now this is the prefix that is added before each q and answer,
        # it is used by get_rand_ans_loss() to extract just the question part and
        # add a random answer to it.
        # !!!! Has additional sideffect of model unlearning this pattern!!!!
        # ADDITONALLY: create_truthfulqa_dataloader() is also using this pattern!!!
        question_prefix_str = "### Question:"
        answer_prefix_str = "### Answer:"

    elif args.unlearning_dataset == "math_qa":
        train_dataset = load_dataset("math_qa", split="train")
        train_bad_loader = create_mathqa_dataloader_from_dataset(
            tokenizer, train_dataset, batch_size=args.batch_size
        )
        question_prefix_str = "Problem:"
        answer_prefix_str = "rationale:"
    else:
        print(f"Unlearning dataset not known! dataset: {args.unlearning_dataset}")
        return

    # Get normal data.
    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    # Load normal answer used for random mismatch.
    normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    model.train()

    # Reference model for computing KL.
    if args.use_quantized:
        # Uncomment for quantized
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            load_in_8bit=True,
            torch_dtype=torch.float32,
        )
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_dir
        )
        pretrained_model.to(device)

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    print("Save model pre-unlearning!")
    model.save_pretrained(args.model_save_dir + f"_idx_{idx}", from_pt=True)
    tokenizer.save_pretrained(args.model_save_dir + f"_idx_{idx}")

    # Stop if bad loss is big enough or reaching max step.
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on answer only. ############
            bad_loss = get_answer_loss("ga", bad_batch, model, device=device)

            ############ Random mismatch. ############
            random_loss = get_rand_ans_loss(
                bad_batch,
                tokenizer,
                normal_ans,
                model,
                K=5,
                device=device,
                question_prefix_str=question_prefix_str,
                answer_prefix_str=answer_prefix_str,
            )
            # time.sleep(20)
            ############ KL on normal samples. ############
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

            # Final loss = bad loss + random smoothing + normal loss.
            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            # Save model.
            if idx % args.save_every == 0:
                model.save_pretrained(args.model_save_dir + f"_idx_{idx}", from_pt=True)
                tokenizer.save_pretrained(args.model_save_dir + f"_idx_{idx}")
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model.save_pretrained(args.model_save_dir, from_pt=True)
    tokenizer.save_pretrained(args.model_save_dir)
    logging.info("Unlearning finished")

    return


if __name__ == "__main__":
    args = parse_args()

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
