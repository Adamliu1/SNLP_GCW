# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Carmen Meinson
#    - Szymon Duchniewicz
#    - Yadong Liu
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

import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import List

# Added
import numpy as np
import torch
import wandb
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
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)


def set_seed(seed_num: int) -> None:
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)


def compute_mink_prob(
    model: AutoModelForCausalLM,
    batch: BatchEncoding,
    K: float,
    # Compute_for_answer only, or question only? (Normal vs Bad loss)
    device: torch.device,
    compute_for_answer_only: bool = True,
) -> List[float]:
    # Compute the average of min-K% Prob values for the entire batch
    # TODO: should we do this on just 2 element batch or entire dataset?
    #
    # NOTE: Need to properly unpack the batch, compute logits for each batch and
    # properly mask the output! only compute min K on the part we are checking if is
    # Unlearned/member in pretraining: output
    with torch.no_grad():
        # TODO: verify: in run.py for mink, calculatePerplexity func (49), they also pass labels=input_ids. WHy?
        # TODO: should we: feeed full sentance (question + answer), and then mask out to compute probabilities,
        # Or Mask out input, only get logits for the answer and compute probabilities on those
        # NOTE: For now, we feed full question, as we are unlearning B given A, so we want individual token
        # probabilities of B, given A (but we don't care about token probabilities of A - the question)
        outputs = model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        # OR, something along thelines of
        # outputs = model(batch["input_ids"][batch["start_locs"]:], attention_mask=batch["attention_mask"])

    logits = outputs.logits
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    no_of_sentences = batch["input_ids"].shape[0]
    # If computing for both question and answer, need to set start_locs to 0s!
    if compute_for_answer_only:
        assert batch.get("start_locs") != None, (
            "Compute Min-k % prob: Requested computation only for answer, "
            "but the batch does not contain start_locs inside tokenised question+answer pairs!"
        )
    else:
        # start locs by default just after <s> starting token!
        batch["start_locs"] = [torch.IntTensor([1]) for _ in range(no_of_sentences)]

    # Compute for each sentence in the batch
    pred_mink = []
    for s_idx in range(no_of_sentences):
        # extract prob for each token in the unlearned answer given all previous tokens
        input_ids_sentence = batch["input_ids"][s_idx]  # [1:]
        all_prob = []
        for i in range(batch["start_locs"][s_idx].item(), len(input_ids_sentence)):
            token_id = input_ids_sentence[i]
            # i is 1 ahead of the actual token, as we want the probability of that token given all previous tokens
            probability = probabilities[s_idx, i - 1, token_id].item()
            all_prob.append(probability)
        # Get top-K % probs and compute their mean (it was log_softmax, so top k% is actually mink% prob)
        k_length = int(len(all_prob) * K)
        topk_prob = np.sort(all_prob)[:k_length]
        pred_mink.append(-np.mean(topk_prob).item())

    # All mean MIN-K% prob in: pred_mink. For now, return all
    return pred_mink


def run_training_batch(
    model,
    pretrained_model,
    tokenizer,
    device,
    normal_ans,
    bad_batch,
    normal_batch,
    idx,
    epoch: int,
    bad_loader_size: int = 0,
    normal_loader_size: int = 0,
    question_prefix_str: str = "",
    answer_prefix_str: str = "",
):
    # Calculate min-k% prob score on bad_batch using the unmodified pre-trained model
    mink_probs_base = compute_mink_prob(
        model=pretrained_model,
        batch=bad_batch,
        K=args.mink_prob_k,
        device=device,
        compute_for_answer_only=True,
    )
    # Calculate min-k% prob score on normal_batch using the unmodified pre-trained model
    mink_probs_base_normal = compute_mink_prob(
        model=pretrained_model,
        batch=normal_batch,
        K=args.mink_prob_k,
        device=device,
        compute_for_answer_only=False,
    )
    # TODO: THIS NEED TO BE CALCULATED AFTER GRADIENT STEP!!! (otherwise we are comparing against previous gradient update!)
    # Calculate min-k% prob score on bad_batch using the model under unlearning
    mink_probs_after_step = compute_mink_prob(
        model=model,
        batch=bad_batch,
        K=args.mink_prob_k,
        device=device,
        compute_for_answer_only=True,
    )
    # Calculate min-k% prob score on normal_batch using the model under unlearning
    mink_probs_after_step_normal = compute_mink_prob(
        model=model,
        batch=normal_batch,
        K=args.mink_prob_k,
        device=device,
        compute_for_answer_only=False,
    )
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
    if bool(args.wandb_log) and (idx % args.wandb_log_feq == 0):
        wandb.log(
            {
                "batch": idx,
                "bad_loss": -bad_loss,
                "normal_loss": normal_loss,
                "final_loss": loss,
                "ratio (bad) mink unlearning/reference": np.mean(mink_probs_after_step)
                / np.mean(mink_probs_base),
                "ratio (normal) mink unlearning/reference": np.mean(
                    mink_probs_after_step_normal
                )
                / np.mean(mink_probs_base_normal),
            }
        )

    stats = (
        f"epoch: {epoch}, batch: {idx}, "
        f"bad_loss: {-bad_loss:.2f}, "
        f"current_div_loss: {normal_loss:.2f}, "
        f"ratio (bad) mink unlearning/reference: {np.mean(mink_probs_after_step)/np.mean(mink_probs_base):.3f}, "
        f"ratio (normal) mink unlearning/reference: {np.mean(mink_probs_after_step_normal)/np.mean(mink_probs_base_normal):.3f}"
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

    print("Model loaded.")

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
    data_sample_artifacts = wandb.Artifact(
        name="training_batch_raw_data", type="batch_data"
    )
    data_sample_artifacts.add_file(
        normal_sample_path, name=f"normal_{args.samples_count}_samples.json"
    )
    data_sample_artifacts.add_file(
        bad_sample_path, name=f"bad_{args.samples_count}_samples.json"
    )
    wandb.log_artifact(data_sample_artifacts)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    # num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(
            (args.num_epochs * args.samples_count)
            if args.sequential > 0
            else args.max_unlearn_steps
        ),
    )

    (
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(model, optimizer, lr_scheduler)
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
        )
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_dir
        )
        pretrained_model.to(device)
    print("Model loaded.")

    print("#################### START UNLEARNING ####################")
    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    running_loss = deque()
    final_model_tag = 0
    # Here for caching what samples are used so far

    if args.sequential > 0:
        # NOTE: sequential/batch unlearning
        num_batches_per_epoch = args.samples_count // args.batch_size

        for seq, (train_normal_loader, train_bad_loader) in enumerate(
            zip(train_normal_loaders, train_bad_loaders)
        ):
            epoch_num = 0
            accu_bad_loss = 0
            while epoch_num < args.num_epochs:
                accu_bad_loss = 0
                for normal_batch, bad_batch in zip(
                    train_normal_loader, train_bad_loader
                ):
                    loss, bad_loss = run_training_batch(
                        model=model,
                        pretrained_model=pretrained_model,
                        tokenizer=tokenizer,
                        device=device,
                        normal_ans=normal_ans,
                        bad_batch=bad_batch,
                        normal_batch=normal_batch,
                        idx=idx,
                        epoch=epoch_num,
                        question_prefix_str=question_prefix_str,
                        answer_prefix_str=answer_prefix_str,
                    )
                    idx += 1
                    # NOTE: the whole dataset is considered to be one single batch.
                    # Back-prop after the whole dataset has been finished.
                    accelerator.backward(loss / num_batches_per_epoch)
                    bad_loss /= num_batches_per_epoch
                    accu_bad_loss += bad_loss.item()
                epoch_num += 1
                final_model_tag = epoch_num
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.sequential == 1:
                    # NOTE: Batch unlearning, save for every epoch
                    model_tokenizer_save_dir = Path(
                        os.path.join(args.model_save_dir, f"idx_{epoch_num}")
                    )
                    model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

                    model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
                    tokenizer.save_pretrained(model_tokenizer_save_dir)

                if abs(accu_bad_loss) > args.max_bad_loss:
                    # Only printing warning at the outer loop to avoid repeated warnings.
                    break

            if abs(accu_bad_loss) > args.max_bad_loss:
                print(
                    f"bad_loss {abs(accu_bad_loss)} exceeding args.max_bad_loss {args.max_bad_loss}. Unlearning stopped."
                )
                break

    else:
        # NOTE: Original ByteDance Unlearning.
        train_bad_loader_gen = iter(train_bad_loaders[0])
        train_normal_loader_gen = iter(train_normal_loaders[0])
        bad_loader_len = len(train_bad_loaders[0])
        normal_loader_len = len(train_normal_loaders[0])
        epoch_num = 0
        for idx in range(args.max_unlearn_steps):
            try:
                bad_batch = next(train_bad_loader_gen)
            except StopIteration:
                epoch_num += 1
                train_bad_loader_gen = iter(train_bad_loaders[0])
                bad_batch = next(train_bad_loader_gen)
            try:
                normal_batch = next(train_normal_loader_gen)
            except StopIteration:
                train_normal_loader_gen = iter(train_normal_loaders[0])
                normal_batch = next(train_normal_loader_gen)
            loss, bad_loss = run_training_batch(
                model=model,
                pretrained_model=pretrained_model,
                tokenizer=tokenizer,
                device=device,
                normal_ans=normal_ans,
                bad_batch=bad_batch,
                normal_batch=normal_batch,
                idx=idx,
                epoch=epoch_num,
                bad_loader_size=bad_loader_len,
                normal_loader_size=normal_loader_len,
                question_prefix_str=question_prefix_str,
                answer_prefix_str=answer_prefix_str,
            )
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            final_model_tag = idx
            if idx % args.save_every == 0:
                model_tokenizer_save_dir = Path(
                    os.path.join(args.model_save_dir, f"idx_{idx}")
                )
                model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
                tokenizer.save_pretrained(model_tokenizer_save_dir)
            running_loss.append(bad_loss.item())
            while len(running_loss) > args.num_running_loss:
                running_loss.popleft()
            avg_loss = abs(np.mean(running_loss))
            if avg_loss > args.max_bad_loss:
                print(
                    f"bad_loss {avg_loss} exceeding args.max_bad_loss {args.max_bad_loss}. Unlearning stopped."
                )
                break

    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model_tokenizer_save_dir = Path(
        os.path.join(args.model_save_dir, f"idx_{final_model_tag}")
    )
    model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
    tokenizer.save_pretrained(model_tokenizer_save_dir)
    logging.info("Unlearning finished")
    if bool(args.wandb_log):
        wandb.finish()
    return


if __name__ == "__main__":
    args = parse_args()

    # Initialize logging
    if bool(args.wandb_log):
        wandb.init(
            project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args)
        )

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
