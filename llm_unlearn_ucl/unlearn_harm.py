# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import logging
from pathlib import Path
import random
import time
from parse_args import parse_args
import wandb

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
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)

# Added
import hf_olmo


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
        if decode_text:
            self.columns.append(f"{prefix} Sample Text")

    def append_batch_samples(self, batch, batch_number):
        """Accumulate samples from the batch along with the batch number."""
        batch_size = batch["input_ids"].size(0)

        for i in range(batch_size):
            input_ids_list = batch["input_ids"][i].tolist()
            sample_text = (
                self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                if self.decode_text
                else ""
            )
            data_row = (
                [batch_number, input_ids_list]
                if not self.decode_text
                else [batch_number, input_ids_list, sample_text]
            )
            self.data.append(data_row)

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


def main(args) -> None:

    set_seed(args.seed)

    accelerator = Accelerator()
    # accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, load_in_8bit=True, torch_dtype=torch.float32)
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

    # Load harmful data.
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )

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
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir
    )
    # pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, load_in_8bit=True, torch_dtype=torch.float32)
    pretrained_model.to(device)

    print("#################### START UNLEARNING ####################")
    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()

    # Here for caching what samples are used so far
    batch_samples_logger_bad = BatchSamplesLogger(
        tokenizer, prefix="Bad", decode_text=True
    )
    batch_samples_logger_normal = BatchSamplesLogger(
        tokenizer, prefix="Normal", decode_text=True
    )

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

            batch_samples_logger_bad.append_batch_samples(bad_batch, idx)
            batch_samples_logger_normal.append_batch_samples(normal_batch, idx)

            # Print.
            if bool(args.wandb_log) and (idx % args.wandb_log_feq == 0):
                wandb.log(
                    {
                        "batch": idx,
                        "bad_loss": -bad_loss,
                        "normal_loss": normal_loss,
                        # NOTE: Sould I negative the sign here????
                        "final_loss": loss,
                    }
                )
                # Log batch samples with optional decoding
                batch_samples_logger_bad.log_accumulated_samples(wandb)
                batch_samples_logger_normal.log_accumulated_samples(wandb)

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
                model.save_pretrained(args.model_save_dir, from_pt=True)
                tokenizer.save_pretrained(args.model_save_dir)

                if bool(args.wandb_log):
                    # Save batch data to files
                    bad_input_ids_path, bad_attention_mask_path = save_batch_data(
                        bad_batch, "bad", idx, args.samples_save_dir
                    )
                    normal_input_ids_path, normal_attention_mask_path = save_batch_data(
                        normal_batch, "normal", idx, args.samples_save_dir
                    )

                    # Create a new artifact for this batch
                    artifact = wandb.Artifact(
                        name=f"batch_data_{idx}", type="batch_data"
                    )

                    # Add files to the artifact
                    artifact.add_file(
                        bad_input_ids_path, name=f"bad_batch_input_ids_{idx}.pt"
                    )
                    artifact.add_file(
                        bad_attention_mask_path,
                        name=f"bad_batch_attention_mask_{idx}.pt",
                    )
                    artifact.add_file(
                        normal_input_ids_path, name=f"normal_batch_input_ids_{idx}.pt"
                    )
                    artifact.add_file(
                        normal_attention_mask_path,
                        name=f"normal_batch_attention_mask_{idx}.pt",
                    )
                    artifact.add_file(args.log_file, name=f"full_logging_{idx}.log")

                    # Log the artifact to wandb
                    wandb.log_artifact(artifact)

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
