# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""

import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path

# Added
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset
from parse_args import parse_args
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
import wandb
from typing import List
from transformers.tokenization_utils_base import BatchEncoding


def set_seed(seed_num: int) -> None:
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)


class BatchSamplesLogger:
    def __init__(self, tokenizer, prefix="", decode_text=True):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.decode_text = decode_text
        self.data = []
        self.columns = [f"{prefix} Batch Number", f"{prefix} Input IDs"]
        self.dataframe = pd.DataFrame(
            columns=(
                ["batch_number", "input_ids_list"]
                if not self.decode_text
                else ["batch_number", "input_ids_list", "sample_text"]
            )
        )
        if decode_text:
            self.columns.append(f"{prefix} Sample Text")

    def append_batch_samples(self, batch, batch_number):
        """Accumulate samples from the batch along with the batch number."""
        batch_size = batch["input_ids"].size(0)
        new_rows = []

        for i in range(batch_size):
            input_ids_list = batch["input_ids"][i].tolist()
            sample_text = (
                self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                if self.decode_text
                else ""
            )
            # Wanb info
            data_row = (
                [batch_number, input_ids_list]
                if not self.decode_text
                else [batch_number, input_ids_list, sample_text]
            )
            self.data.append(data_row)
            #
            data_dict = {
                "batch_number": batch_number,
                "input_ids_list": input_ids_list,
                "sample_text": sample_text if self.decode_text else pd.NA,
            }
            new_rows.append(data_dict)

        new_rows_df = pd.DataFrame(new_rows)
        self.dataframe = pd.concat([self.dataframe, new_rows_df])

    def export_dataframe(self, path):
        self.dataframe.to_csv(path)

    def log_accumulated_samples(self, wandb, reset=True):
        """Log the accumulated data as a wandb Table and optionally reset."""
        table = wandb.Table(data=self.data, columns=self.columns)
        wandb.log({f"Accumulated {self.prefix} Batch Samples": table})
        if reset:
            self.data = []  # Reset the data for the next accumulation


def save_batch_data(batch, batch_type, idx, save_dir):
    """
    Saves input_ids and attention_mask from a batch to files.

    Args:
        batch: The batch containing the samples to save.
        batch_type: A string indicating whether the batch is 'bad' or 'normal'.
        idx: The index of the batch (e.g., batch number in the training loop).
        save_dir: The directory where the files should be saved.
    """
    # Ensure save_dir exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    input_ids_path = f"{save_dir}/{batch_type}_input_ids_{idx}.pt"
    attention_mask_path = f"{save_dir}/{batch_type}_attention_mask_{idx}.pt"

    # Save tensors
    torch.save(batch["input_ids"], input_ids_path)
    torch.save(batch["attention_mask"], attention_mask_path)

    return input_ids_path, attention_mask_path


def compute_mink_prob(
    model: AutoModelForCausalLM,
    batch: BatchEncoding,
    K: float,
    # Compute_for_answer only, or question only? (Normal vs Bad loss)
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
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
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


def main(args) -> None:
    set_seed(args.seed)
    assert (
        args.epoch_size % args.sequential == 0
    ), "epoch_size should be divisible by number of splits for sequential learning (--sequential)."
    assert (
        args.epoch_size // args.sequential
    ) % args.batch_size == 0, "samples in each 'sequence' (--epoch_size / --sequential) should be a multiple of batch_size."
    accelerator = Accelerator()
    # accelerator = Accelerator(mixed_precision="fp16")
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
            train_bad_dataset = full_bad_dataset.select(range(args.epoch_size))
        else:
            # NOTE: full dataset like bytedance.
            train_bad_dataset = full_bad_dataset

        Path(args.samples_save_dir).mkdir(exist_ok=True)
        bad_sample_path = f"{args.samples_save_dir}/bad_{args.epoch_size if args.sequential > 0 else 'full'}_samples.json"
        with open(bad_sample_path, "w") as fin:
            print(f"Writing bad samples to {bad_sample_path}")
            json.dump(
                [
                    train_bad_dataset[i]
                    for i in range(
                        args.epoch_size
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
        num_samples=args.epoch_size if args.sequential > 0 else None,
        splits=max(args.sequential, 1),
    )
    normal_sample_path = f"{args.samples_save_dir}/normal_{args.epoch_size if args.sequential > 0 else 'full'}_samples.json"
    with open(normal_sample_path, "w") as fin:
        print(f"Writing normal samples to {normal_sample_path}")
        json.dump(
            [
                train_normal_dataset[i]
                for i in range(
                    args.epoch_size
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
        normal_sample_path, name=f"normal_{args.epoch_size}_samples.json"
    )
    data_sample_artifacts.add_file(
        bad_sample_path, name=f"bad_{args.epoch_size}_samples.json"
    )
    wandb.log_artifact(data_sample_artifacts)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    # num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_epochs * args.epoch_size,
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

    def run_training_batch(
        bad_batch,
        normal_batch,
        idx,
        epoch: int,
        bad_loader_size: int = 0,
        normal_loader_size: int = 0,
    ):
        ############ Compute Min-K for the batch ##########
        # TODO: Do we just comput min-K for data being unlearned? or full question?
        # TODO: should we fix the size of input values (WikiMIA they used the same sample sizes)
        # mink_probs = compute_mink_prob(
        #     model=model, tokenizer=tokenizer, batch=bad_batch, K=args.mink_prob_k
        # )
        mink_probs_base = compute_mink_prob(
            model=pretrained_model,
            batch=bad_batch,
            K=args.mink_prob_k,
            compute_for_answer_only=True,
        )

        mink_probs_base_normal = compute_mink_prob(
            model=pretrained_model,
            batch=normal_batch,
            K=args.mink_prob_k,
            compute_for_answer_only=False,
        )
        # TODO: THIS NEED TO BE CALCULATED AFTER GRADIENT STEP!!! (otherwise we are comparing against previous gradient update!)
        mink_probs_after_step = compute_mink_prob(
            model=model,
            batch=bad_batch,
            K=args.mink_prob_k,
            compute_for_answer_only=True,
        )

        mink_probs_after_step_normal = compute_mink_prob(
            model=model,
            batch=normal_batch,
            K=args.mink_prob_k,
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

        bad_batch_idx = idx
        normal_batch_idx = idx
        if bad_loader_size:
            bad_batch_idx %= bad_loader_size
        if normal_loader_size:
            normal_batch_idx %= normal_loader_size
        batch_samples_logger_bad.append_batch_samples(bad_batch, bad_batch_idx)
        batch_samples_logger_normal.append_batch_samples(normal_batch, normal_batch_idx)

        # Print.
        if bool(args.wandb_log) and (idx % args.wandb_log_feq == 0):
            wandb.log(
                {
                    "batch": idx,
                    "bad_loss": -bad_loss,
                    "normal_loss": normal_loss,
                    # NOTE: Sould I negative the sign here????
                    "final_loss": loss,
                    "ratio (bad) mink unlearning/reference": np.mean(
                        mink_probs_after_step
                    )
                    / np.mean(mink_probs_base),
                    "ratio (normal) mink unlearning/reference": np.mean(
                        mink_probs_after_step_normal
                    )
                    / np.mean(mink_probs_base_normal),
                }
            )
            # Log batch samples with optional decoding
            batch_samples_logger_bad.log_accumulated_samples(wandb)
            batch_samples_logger_normal.log_accumulated_samples(wandb)

        stats = (
            f"epoch: {epoch}, batch: {idx}, "
            f"bad_loss: {-bad_loss:.2f}, "
            f"current_div_loss: {normal_loss:.2f}, "
            f"ratio (bad) mink unlearning/reference: {np.mean(mink_probs_after_step)/np.mean(mink_probs_base):.3f}"
            f"ratio (normal) mink unlearning/reference: {np.mean(mink_probs_after_step_normal)/np.mean(mink_probs_base_normal):.3f}"
        )
        logging.info(stats)
        print(stats)
        idx += 1

        if idx % args.save_every == 0:
            # model_tokenizer_save_dir = Path(os.path.join(args.model_save_dir, f"checkpoint_{idx}"))
            # model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

            # model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
            # tokenizer.save_pretrained(model_tokenizer_save_dir)

            sample_save_dir = Path(args.samples_save_dir)
            sample_save_dir.mkdir(parents=True, exist_ok=True)

            batch_samples_logger_bad.export_dataframe(
                os.path.join(sample_save_dir, f"bad_batch_{idx}.csv")
            )
            batch_samples_logger_normal.export_dataframe(
                os.path.join(sample_save_dir, f"normal_batch_{idx}.csv")
            )
            if bool(args.wandb_log):
                # Save batch data to files
                # bad_input_ids_path, bad_attention_mask_path = save_batch_data(
                #     bad_batch, "bad", idx, args.samples_save_dir
                # )
                # normal_input_ids_path, normal_attention_mask_path = save_batch_data(
                #     normal_batch, "normal", idx, args.samples_save_dir
                # )

                # Create a new artifact for this batch
                artifact = wandb.Artifact(name=f"batch_data_{idx}", type="batch_data")

                # Add files to the artifact
                # artifact.add_file(
                #     bad_input_ids_path, name=f"bad_batch_input_ids_{idx}.pt"
                # )
                # artifact.add_file(
                #     bad_attention_mask_path,
                #     name=f"bad_batch_attention_mask_{idx}.pt",
                # )
                # artifact.add_file(
                #     normal_input_ids_path, name=f"normal_batch_input_ids_{idx}.pt"
                # )
                # artifact.add_file(
                #     normal_attention_mask_path,
                #     name=f"normal_batch_attention_mask_{idx}.pt",
                # )
                artifact.add_file(args.log_file, name=f"full_logging_{idx}.log")
                artifact.add_file(
                    os.path.join(args.samples_save_dir, f"bad_batch_{idx}.csv"),
                    name=f"bad_batch_{idx}.csv",
                )
                artifact.add_file(
                    os.path.join(args.samples_save_dir, f"normal_batch_{idx}.csv"),
                    name=f"normal_batch_{idx}.csv",
                )
                # Log the artifact to wandb
                wandb.log_artifact(artifact)

        return loss, bad_loss

    print("#################### START UNLEARNING ####################")
    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    running_loss = deque()
    # Here for caching what samples are used so far
    batch_samples_logger_bad = BatchSamplesLogger(
        tokenizer, prefix="Bad", decode_text=True
    )
    batch_samples_logger_normal = BatchSamplesLogger(
        tokenizer, prefix="Normal", decode_text=True
    )

    if args.sequential > 0:
        # NOTE: sequential/batch unlearning
        num_batches_per_epoch = args.epoch_size // args.batch_size
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
                        bad_batch, normal_batch, idx, epoch_num
                    )
                    idx += 1
                    # NOTE: the whole dataset is considered to be one single batch.
                    # Back-prop after the whole dataset has been finished.
                    accelerator.backward(loss / num_batches_per_epoch)
                    bad_loss /= num_batches_per_epoch
                    accu_bad_loss += bad_loss.item()
                epoch_num += 1
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.sequential == 1:
                    # NOTE: Batch unlearning, save for every epoch
                    model_tokenizer_save_dir = Path(
                        os.path.join(args.model_save_dir, f"epoch_{epoch_num}")
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
        model_tokenizer_save_dir = Path(os.path.join(args.model_save_dir, "final"))
        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
        tokenizer.save_pretrained(model_tokenizer_save_dir)

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
                bad_batch,
                normal_batch,
                idx,
                epoch_num,
                bad_loader_len,
                normal_loader_len,
            )
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if idx % args.save_every == 0:
                model_tokenizer_save_dir = Path(
                    os.path.join(args.model_save_dir, f"checkpoint_{idx}")
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
        model_tokenizer_save_dir = Path(os.path.join(args.model_save_dir, "final"))
        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
        tokenizer.save_pretrained(model_tokenizer_save_dir)

    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model.save_pretrained(args.model_save_dir, from_pt=True)
    tokenizer.save_pretrained(args.model_save_dir)
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
