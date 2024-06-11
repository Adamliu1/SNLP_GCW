# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Carmen Meinson
#    - Andrzej Szablewski
#    - Zhe Yu
#
# Adapted from https://github.com/kevinyaobytedance/llm_unlearn.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF` and TruthfulQA. Model supports OPT-1.3B.
"""

import gc
import json
from accelerate.logging import get_logger
import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import List
from transformers import DataCollatorForLanguageModeling
# Added
import numpy as np
import torch
# import wandb
from accelerate import Accelerator
from datasets import load_dataset
from parse_args import parse_args
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.tokenization_utils_base import BatchEncoding
from utils import (
    compute_kl,
    create_mathqa_dataloader_from_dataset,
    create_piaf_dataloader_from_dataset,
    create_pku_dataloader_from_dataset,
    create_squad_dataloader_from_dataset,
    create_symbolic_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_squad_answers,
    get_truthfulQA_answers_plaintext,
)

def set_seed(seed_num: int) -> None:
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

def run_training_batch(
    model,
    pretrained_model,
    tokenizer,
    device,
    normal_ans,
    bad_batch,
    normal_batch,
    idx,
    samples_count,
    epoch: int,
    bad_loader_size: int = 0,
    normal_loader_size: int = 0,
    question_prefix_str: str = "",
    answer_prefix_str: str = "",
):
    ############ GA on answer only. ############
    bad_loss = get_answer_loss_tmp("ga", bad_batch, model, tokenizer, device=device)
    # bad_loss = get_answer_loss("ga", bad_batch, model,device=device)

    ############ Random mismatch. ############
    random_loss = get_rand_ans_loss_tmp(
        bad_batch,
        tokenizer,
        normal_ans,
        model,
        K=5,
        device=device,
        question_prefix_str=question_prefix_str,
        answer_prefix_str=answer_prefix_str,
    )
    ############ KL on normal samples. ############
    normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

    # Final loss = bad loss + random smoothing + normal loss.
    loss = (
        args.bad_weight * bad_loss
        + args.random_weight * random_loss
        + args.normal_weight * normal_loss
    )

    # NOTE: backwardnd optimisation is done outside of this function in the
    # training loop for gradient accumulation compatibility.
    # if bool(args.wandb_log) and (idx % args.wandb_log_freq == 0):
    #     wandb.log(
    #         {
    #             "batch": idx,
    #             "epoch": epoch,
    #             "samples_count": samples_count,
    #             "bad_loss": -bad_loss,
    #             "normal_loss": normal_loss,
    #             "final_loss": loss,
    #             "ratio (bad) mink unlearning/reference": np.mean(mink_probs_after_step)
    #             / np.mean(mink_probs_base),
    #             "ratio (normal) mink unlearning/reference": np.mean(
    #                 mink_probs_after_step_normal
    #             )
    #             / np.mean(mink_probs_base_normal),
    #         }
    #     )

    stats = (
        f"epoch: {epoch}, batch: {idx}, "
        f"samples seen: {samples_count}, "
        f"bad_loss: {-bad_loss:.2f}, "
        f"current_div_loss: {normal_loss:.2f}, "
    )
    logging.info(stats)
    print(stats)

    return loss, bad_loss


def main(args) -> None:
    set_seed(args.seed)
    assert (
        args.samples_count % args.sequential == 0
    ), "samples_count should be divisible by number of splits for sequential learning (--sequential)."
    assert (
        args.samples_count // args.sequential
    ) % args.batch_size == 0, "samples in each 'sequence' (--samples_count / --sequential) should be a multiple of batch_size."
    accelerator = Accelerator()  # accelerator precision can be specified if required.
    device = accelerator.device

    print(f"Loading model {args.model_name} for training...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True, use_flash_attention_2=False
    )

    print("Model loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True, use_fast=False)
    # Ensure the tokenizer has a pad token. Use EOS token as padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data to unlearn.
    if args.unlearning_dataset == "PKU-Alignment/PKU-SafeRLHF":
        # filter entries with harmful responses and draw random samples from the remaining dataset.
        full_bad_dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF", split="train"
        ).filter(
            lambda entry: not (
                entry["is_response_0_safe"] or entry["is_response_1_safe"]
            )
        )
        if args.shuffle_seed:
            # shuffle the dataset with a given seed for reproducibility
            full_bad_dataset = full_bad_dataset.shuffle(seed=args.shuffle_seed)
        if args.sequential > 0:
            # NOTE: sequential/batch unlearning using sliced dataset.
            train_bad_dataset = full_bad_dataset.select(range(args.samples_count))
        else:
            # NOTE: full dataset like bytedance.
            train_bad_dataset = full_bad_dataset

        Path(args.samples_save_dir).mkdir(exist_ok=True)
        bad_sample_path = f"{args.samples_save_dir}/bad_{args.samples_count if args.sequential > 0 else 'full'}_samples.json"
        with open(bad_sample_path, "w") as fin:
            print(f"Writing bad samples to {bad_sample_path}")
            json.dump(
                [
                    train_bad_dataset[i]
                    for i in range(
                        args.samples_count
                        if args.sequential > 0
                        else len(train_bad_dataset)
                    )
                ],
                fin,
            )

        train_bad_loaders = create_pku_dataloader_from_dataset(
            tokenizer,
            train_bad_dataset,
            batch_size=args.batch_size,
            splits=max(args.sequential, 1),
        )

        # XXX: for now this is the prefix that is added before each q and answer,
        # it is used by get_rand_ans_loss() to extract just the question part and
        # add a random answer to it.
        # !!!! Has additional sideffect of model unlearning this pattern!!!!
        # ADDITONALLY: create_truthfulqa_dataloader() is also using this pattern!!!
        question_prefix_str = "### Question:"
        answer_prefix_str = "### Answer:"
    elif args.unlearning_dataset == "AgentPublic/piaf":
        # filter entries with harmful responses and draw random samples from the remaining dataset.
        full_bad_dataset = load_dataset("AgentPublic/piaf", split="train").filter(
            lambda entry: len(entry["answers"]["text"]) != 0
        )
        if args.shuffle_seed:
            # shuffle the dataset with a given seed for reproducibility
            full_bad_dataset = full_bad_dataset.shuffle(seed=args.shuffle_seed)
        if args.sequential > 0:
            # NOTE: sequential/batch unlearning using sliced dataset.
            train_bad_dataset = full_bad_dataset.select(range(args.samples_count))
        else:
            # NOTE: full dataset like bytedance.
            train_bad_dataset = full_bad_dataset

        Path(args.samples_save_dir).mkdir(exist_ok=True)
        bad_sample_path = f"{args.samples_save_dir}/piaf_{args.samples_count if args.sequential > 0 else 'full'}_samples.json"
        with open(bad_sample_path, "w") as fin:
            print(f"Writing bad samples to {bad_sample_path}")
            json.dump(
                [
                    train_bad_dataset[i]
                    for i in range(
                        args.samples_count
                        if args.sequential > 0
                        else len(train_bad_dataset)
                    )
                ],
                fin,
            )

        train_bad_loaders = create_piaf_dataloader_from_dataset(
            tokenizer,
            train_bad_dataset,
            batch_size=args.batch_size,
            splits=max(args.sequential, 1),
        )

        question_prefix_str = "### Question:"
        answer_prefix_str = "### Réponse:"
    elif args.unlearning_dataset == "sail/symbolic-instruction-tuning":
        # filter entries with harmful responses and draw random samples from the remaining dataset.
        full_bad_dataset = load_dataset(
            "sail/symbolic-instruction-tuning", split="train"
        )
        if args.shuffle_seed:
            # shuffle the dataset with a given seed for reproducibility
            full_bad_dataset = full_bad_dataset.shuffle(seed=args.shuffle_seed)
        if args.sequential > 0:
            # NOTE: sequential/batch unlearning using sliced dataset.
            train_bad_dataset = full_bad_dataset.select(range(args.samples_count))
        else:
            # NOTE: full dataset like bytedance.
            train_bad_dataset = full_bad_dataset

        Path(args.samples_save_dir).mkdir(exist_ok=True)
        bad_sample_path = f"{args.samples_save_dir}/symbolic_{args.samples_count if args.sequential > 0 else 'full'}_samples.json"
        with open(bad_sample_path, "w") as fin:
            print(f"Writing symbolic samples to {bad_sample_path}")
            json.dump(
                [
                    train_bad_dataset[i]
                    for i in range(
                        args.samples_count
                        if args.sequential > 0
                        else len(train_bad_dataset)
                    )
                ],
                fin,
            )

        train_bad_loaders = create_symbolic_dataloader_from_dataset(
            tokenizer,
            train_bad_dataset,
            batch_size=args.batch_size,
            splits=max(args.sequential, 1),
        )

        question_prefix_str = "### Question:"
        answer_prefix_str = "### Answer:"
    elif args.unlearning_dataset == "math_qa":
        assert (
            False
        ), "Mathqa temporarirly disabled - requries implementing returning a List of Datasets for sequential unlearning with equal sizes!"
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
    if args.retaining_dataset == "truthful_qa":
        (
            train_normal_loaders,
            val_normal_loader,
            test_normal_loader,
            train_normal_dataset,
        ) = create_truthfulqa_dataloader(
            tokenizer,
            batch_size=args.batch_size,
            seed=args.shuffle_seed if args.shuffle_seed is not None else args.seed,
            num_samples=args.samples_count if args.sequential > 0 else None,
            splits=max(args.sequential, 1),
        )
        normal_sample_path = f"{args.samples_save_dir}/normal_{args.samples_count if args.sequential > 0 else 'full'}_samples.json"
        with open(normal_sample_path, "w") as fin:
            print(f"Writing normal samples to {normal_sample_path}")
            json.dump(
                [
                    train_normal_dataset[i]
                    for i in range(
                        args.samples_count
                        if args.sequential > 0
                        else len(train_normal_dataset)
                    )
                ],
                fin,
            )

        # Load normal answer used for random mismatch.
        normal_ans = get_truthfulQA_answers_plaintext()
    elif args.retaining_dataset == "rajpurkar/squad":
        train_split = "train"
        if args.samples_count > 0:
            train_split = f"{train_split}[:{args.samples_count}]"
        train_normal_dataset = load_dataset("rajpurkar/squad", split=train_split)
        train_normal_loaders = create_squad_dataloader_from_dataset(
            tokenizer,
            train_normal_dataset,
            batch_size=args.batch_size,
            splits=max(args.sequential, 1),
        )
        normal_sample_path = f"{args.samples_save_dir}/squad_{args.samples_count if args.sequential > 0 else 'full'}_samples.json"
        with open(normal_sample_path, "w") as fin:
            print(f"Writing normal samples to {normal_sample_path}")
            json.dump(
                [
                    train_normal_dataset[i]
                    for i in range(
                        args.samples_count
                        if args.sequential > 0
                        else len(train_normal_dataset)
                    )
                ],
                fin,
            )

        # Load normal answer used for random mismatch.
        normal_ans = get_squad_answers(train_normal_dataset)
    else:
        print(f"Retaining dataset not known! dataset: {args.retaining_dataset}")
        return

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    # num_training_steps = args.max_unlearn_steps
    # if args.no_scheduler:
    # TODO: TEST THIS BIT
    # NOTE: I REMOVED THE LR SCHEDULER
    (
        model,
        optimizer,
        train_bad_loaders[0],
        train_normal_loaders[0]
    ) = accelerator.prepare(model, optimizer, train_bad_loaders[0], train_normal_loaders[0])

    for i in range(args.sequential):
        train_bad_loaders[i], train_normal_loaders[i] = accelerator.prepare(
            train_bad_loaders[i], train_normal_loaders[i]
        )

    model.train()

    # Reference model for computing KL.

    print(f"Loading model {args.model_name} for reference ('fully learned')...")
    if args.use_quantized:
        # Uncomment for quantized
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            load_in_8bit=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_dir, 
            trust_remote_code=True,

        )
        pretrained_model.to(device)
    print("Model loaded.")

    print("#################### START UNLEARNING ####################")
    # Start unlearning.
    bad_loss = 0.0  # Running value of the "bad loss"
    idx = 0  # Number of "unlearning steps" that has occurred (e.g. processed batches)
    samples_count = 0  # Total number of samples that "passed through" the model (for 1 pass of batch 32: 32 samples)
    start_time = time.time()  # Start time of unlearning process
    running_loss = (
        deque()
    )  # averaging running value of "bad loss", used in ByteDance paper unlearning method
    final_model_tag = 0
    # Here for caching what samples are used so far

    # NOTE: Original ByteDance Unlearning.
    bad_loader_len = len(train_bad_loaders[0])
    normal_loader_len = len(train_normal_loaders[0])
    epoch_num = 0
    while idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(
            train_bad_loaders[0], train_normal_loaders[0]
        ):
            samples_count += len(bad_batch["input_ids"])
            loss, bad_loss = run_training_batch(
                model=model,
                pretrained_model=pretrained_model,
                tokenizer=tokenizer,
                device=device,
                normal_ans=normal_ans,
                bad_batch=bad_batch,
                normal_batch=normal_batch,
                idx=idx,
                samples_count=samples_count,
                epoch=epoch_num,
                bad_loader_size=bad_loader_len,
                normal_loader_size=normal_loader_len,
                question_prefix_str=question_prefix_str,
                answer_prefix_str=answer_prefix_str,
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            idx += 1
            final_model_tag = idx

            running_loss.append(bad_loss.item())
            while len(running_loss) > args.num_running_loss:
                running_loss.popleft()

            if (
                abs(np.mean(running_loss)) > args.max_bad_loss
                or idx >= args.max_unlearn_steps
            ):
                break

        epoch_num += 1

        if idx >= args.max_unlearn_steps:
            print("max_unlearn_steps reached. Unlearning stopped.")
            break
        if avg_loss := abs(np.mean(running_loss)) > args.max_bad_loss:
            print(
                f"bad_loss {avg_loss} exceeding args.max_bad_loss {args.max_bad_loss}. Unlearning stopped."
            )
            break

    end_time = time.time()
    # logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    # model_tokenizer_save_dir = Path(
    #     os.path.join(args.model_save_dir, f"idx_{final_model_tag}")
    # )
    # model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
    # model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
    # tokenizer.save_pretrained(model_tokenizer_save_dir)
    print("Saved final model.")

    logging.info("Unlearning finished")
    # if bool(args.wandb_log):
    #     wandb.finish()
    return

def get_answer_loss_tmp(operation, batch, model, tokenizer, device="cuda:0"):
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )

    # print("CHECK INPUTS")
    # for input_id in input_ids:
    #     ori_text = tokenizer.decode(input_id)
    #     print(ori_text)
    # print("ENDDDDDDD")

    # print(input_ids, attention_mask, start_locs, labels)
    # model.eval()
    outputs = model(input_ids, attention_mask=attention_mask)
    # model.train()
    # print(outputs)
    # print(type(outputs))

    # Get logits
    logits = outputs.logits
    # print(f"this is logits: {logits}")

    # Applying softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_prob, predicted_index = torch.max(probs, dim=-1)
    # print("Max Probability per Position:", predicted_prob)
    # print("Corresponding Indices:", predicted_index)

    # Decoding from probabilities
    # predicted_text_prob = tokenizer.decode(predicted_index[0], skip_special_tokens=True)
    # # predicted_text_prob = tokenizer.decode(predicted_index[0])
    # logging.info(f"Decoded Text from Probabilities:\n\n {predicted_text_prob}")

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    # logging.info(f"Shifted logits shape: {shift_logits.shape}")
    # logging.info(f"Shifted labels shape: {shift_labels.shape}")

    losses = []
    for bid in range(input_ids.shape[0]):
        
        # logging.info(f"Batch ID: {bid}")
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        # logging.info(f"Initial Position Loss: {position_loss}")
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # logging.info(f"Start location (one_st): {one_st}")
        # logging.info(f"Input tokens around start location: {one_inp[max(0, one_st - 5):one_st + 5]}")
        # logging.info(f"Is padding at start location? {one_inp[one_st] == tokenizer.pad_token_id}")

        real_start = (one_inp != tokenizer.pad_token_id).nonzero().min()
        # logging.info(f"Computed real start of non-padding content: {real_start}")
        # logging.info(f"decoded: {tokenizer.decode(one_inp)}")


        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part
        # logging.info(f"Position Weight, before ignoring padding: {position_weight}")

        # Ignore the padding part.
        # position_weight[one_inp == 1] = 0
        position_weight[one_inp == tokenizer.pad_token_id] = 0
        
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()
        # logging.info(f"Position Weight: {position_weight}")
        one_loss = (position_weight[:-1] * position_loss).sum()
        # logging.info(f"Loss for One Input: {one_loss}")
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss

def get_rand_ans_loss_tmp(
    bad_batch,
    tokenizer,
    normal_ans,
    model,
    K=5,
    device="cuda:0",
    question_prefix_str="### Question:",
    answer_prefix_str="### Answer:",
):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.
        question_prefix_str: The default question prefix that is added in create_XXX_dataloader_from_dataset
        answer_prefix_str: The default answer prefix that is added in create_XXX_dataloader_from_dataset

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # print(ori_text)

        # Get question. For custom question prefix
        question = (
            ori_text.split(question_prefix_str)[1].split(answer_prefix_str)[0].strip()
        )
        question_prefix = f"{question_prefix_str} {question} {answer_prefix_str}"

        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    # print("IM COMPUTING RANDOM LOSS")
    # random_loss = get_answer_loss_tmp("gd", batch_random, model, tokenizer, device=device)
    random_loss = get_answer_loss("gd", batch_random, model, device=device)
    # print(f"randon_loss {random_loss}")

    del data_collator
    del batch_random
    del bad_input_ids
    del rand_ans_list
    del batch_random_features
    torch.cuda.empty_cache()
    gc.collect()

    return random_loss

if __name__ == "__main__":
    args = parse_args()

    # Initialize logging
    # if bool(args.wandb_log):
    #     # import wandb
    #     wandb.init(
    #         project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args)
    #     )

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
