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

import logging
from accelerate.logging import get_logger
import os
import random
import time
from collections import deque
from pathlib import Path

# Added
import numpy as np
import torch
from accelerate import Accelerator
from data_utils import (
    SUPPORTED_RETAINING_SET,
    SUPPORTED_UNLEARNING_SET,
    DataloaderConstructor,
    get_normal_answer,
    make_dataset,
)
from parse_args import parse_args
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import (
    compute_kl,
    get_answer_loss,
    get_rand_ans_loss,
)


def set_seed(seed_num: int) -> None:
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)


def run_training_batch(
    model,
    pretrained_model_probs,
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
    accelerator=None,  # TODO: maybe reorder here
    logger=None,
):
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
    normal_loss = compute_kl(pretrained_model_probs, model, normal_batch, device)

    # TODO: check this, currently `args` not passed as an argument, but code works
    # Final loss = bad loss + random smoothing + normal loss.
    loss = (
        args.bad_weight * bad_loss
        + args.random_weight * random_loss
        + args.normal_weight * normal_loss
    )

    # NOTE: backwardnd optimisation is done outside of this function in the
    # training loop for gradient accumulation compatibility.
    if bool(args.wandb_log) and (idx % args.wandb_log_freq == 0):
        accelerator.log(
            {
                "batch": idx,
                "epoch": epoch,
                "samples_count": samples_count,
                "bad_loss": -bad_loss,
                "normal_loss": normal_loss,
                "final_loss": loss,
            }
        )

    stats = (
        f"epoch: {epoch}, batch: {idx}, "
        f"samples seen: {samples_count}, "
        f"bad_loss: {-bad_loss:.2f}, "
        f"current_div_loss: {normal_loss:.2f}, "
    )
    logger.info(stats, main_process_only=True)
    if accelerator.is_local_main_process:
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

    if args.wandb_log:
        accelerator: Accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )
    else:
        accelerator: Accelerator = Accelerator()
    device = accelerator.device

    # setup logging
    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    logger = get_logger(__name__)
    if accelerator.is_main_process:
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
    accelerator.wait_for_everyone()

    print(f"Loading model {args.model_name} for training...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.to(device)
    print("Model loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data to unlearn.
    if args.unlearning_dataset in SUPPORTED_UNLEARNING_SET:
        train_bad_dataset, bad_sample_path = make_dataset(
            args.unlearning_dataset,
            num_samples=None if args.sequential == -1 else args.samples_count,
            seed=args.shuffle_seed,
            save_dir=args.samples_save_dir,
        )
        train_bad_loaders = DataloaderConstructor(
            train_bad_dataset,
            dataset_uri=args.unlearning_dataset,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            num_splits=max(args.sequential, 1),
        ).get_loaders()

        if args.unlearning_dataset == "AgentPublic/piaf":
            question_prefix_str = "### Question:"
            answer_prefix_str = "### Réponse:"
        elif args.unlearning_dataset == "allenai/math_qa":
            question_prefix_str = "Problem:"
            answer_prefix_str = "rationale:"
        else:
            question_prefix_str = "### Question:"
            answer_prefix_str = "### Answer:"

    else:
        print(f"Unlearning dataset not known! dataset: {args.unlearning_dataset}")
        return

    # Get normal data.

    if args.retaining_dataset in SUPPORTED_RETAINING_SET:
        if args.retaining_dataset == "truthful_qa":
            args.retaining_dataset = "truthfulqa/truthful_qa"
        train_normal_dataset, normal_sample_path = make_dataset(
            args.retaining_dataset,
            args.samples_count if args.sequential != -1 else None,
            args.shuffle_seed,
            save_dir=args.samples_save_dir,
        )

        train_normal_loaders = DataloaderConstructor(
            train_normal_dataset,
            args.retaining_dataset,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            num_splits=max(args.sequential, 1),
        ).get_loaders()
        normal_ans = get_normal_answer(train_normal_dataset, args.retaining_dataset)

    else:
        print(f"Retaining dataset not known! dataset: {args.retaining_dataset}")
        return

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    if args.no_scheduler:
        (
            model,
            optimizer,
            train_bad_loaders[0],
            train_normal_loaders[0],
        ) = accelerator.prepare(
            model, optimizer, train_bad_loaders[0], train_normal_loaders[0]
        )
    else:
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
            train_bad_loaders[0],
            train_normal_loaders[0],
        ) = accelerator.prepare(
            model,
            optimizer,
            lr_scheduler,
            train_bad_loaders[0],
            train_normal_loaders[0],
        )

    for i in range(args.sequential):
        train_bad_loaders[i], train_normal_loaders[i] = accelerator.prepare(
            train_bad_loaders[i], train_normal_loaders[i]
        )

    # Pre-compute normal results for normal-loss component
    pretrained_model_precomputed_normal_outputs_aggregated = []
    if accelerator.is_local_main_process:
        print("Precomputing the normal outputs using the pretrained model...")
    for loader in train_normal_loaders:
        pretrained_model_precomputed_normal_outputs_aggregated.append([])
        for batch in loader:
            with torch.no_grad():
                pretrained_outputs = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                prob_p = torch.nn.functional.softmax(
                    pretrained_outputs.logits, -1
                ).cpu()
                pretrained_model_precomputed_normal_outputs_aggregated[-1].append(
                    prob_p
                )

    if accelerator.is_local_main_process:
        print("Done precomputing.")
    torch.cuda.empty_cache()

    model.train()

    if accelerator.is_local_main_process:
        print("#################### START UNLEARNING ####################")
    bad_loss = 0.0  # Running value of the "bad loss"
    idx = 0  # Number of "unlearning steps" that has occurred (e.g. processed batches)
    samples_count = 0  # Total number of samples that "passed through" the model (for 1 pass of batch 32: 32 samples)
    start_time = time.time()  # Start time of unlearning process
    running_loss = (
        deque()
    )  # averaging running value of "bad loss", used in ByteDance paper unlearning method
    final_model_tag = 0

    if args.sequential > 0:
        # NOTE: sequential/batch unlearning
        num_batches_per_epoch = args.samples_count // args.sequential // args.batch_size

        for seq, (
            train_normal_loader,
            train_bad_loader,
            pretrained_model_precomputed_normal_outputs,
        ) in enumerate(
            zip(
                train_normal_loaders,
                train_bad_loaders,
                pretrained_model_precomputed_normal_outputs_aggregated,
            )
        ):
            epoch_num = 0
            accu_bad_loss = 0
            bad_loss = 0
            while epoch_num < args.num_epochs:
                accu_bad_loss = 0
                for normal_batch, bad_batch, pretrained_model_probs in zip(
                    train_normal_loader,
                    train_bad_loader,
                    pretrained_model_precomputed_normal_outputs,
                ):
                    samples_count += len(bad_batch["input_ids"])
                    # TODO: fix gradient accumulation, currently because of 'run_training_batch', it's hard to use accelerator's accumulation.
                    # Currenlty we need to set accelerator config to use accumulation step 1.
                    # with accelerator.accumulate(model):
                    loss, bad_loss = run_training_batch(
                        model=model,
                        pretrained_model_probs=pretrained_model_probs,
                        tokenizer=tokenizer,
                        device=device,
                        normal_ans=normal_ans,
                        bad_batch=bad_batch,
                        normal_batch=normal_batch,
                        idx=idx,
                        samples_count=samples_count,
                        epoch=epoch_num,
                        question_prefix_str=question_prefix_str,
                        answer_prefix_str=answer_prefix_str,
                        accelerator=accelerator,
                        logger=logger,
                    )
                    idx += 1
                    # NOTE: the whole dataset is considered to be one single batch.
                    # Back-prop after the whole dataset has been finished.
                    accelerator.backward(loss / num_batches_per_epoch)
                    bad_loss /= num_batches_per_epoch
                    accu_bad_loss += bad_loss.item()
                # If args.batch_size < args.samples_count//args.sequential, always perform gradient accumulation.
                epoch_num += 1
                final_model_tag = epoch_num
                optimizer.step()
                if not args.no_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()

                # NOTE: This only handles deepspeed zero and zero2, zero3 will require change
                if args.sequential == 1 and epoch_num % args.save_every == 0:
                    # NOTE: special case for zero 3
                    if accelerator.deepspeed_config is not None and accelerator.deepspeed_config['zero_optimization']['stage'] == 3:
                        print("Zero 3 optim: Saving model shards from all GPUs!")
                        model_tokenizer_save_dir = Path(
                            os.path.join(args.model_save_dir, f"idx_{epoch_num}")
                        )
                        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            model_tokenizer_save_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        tokenizer.save_pretrained(model_tokenizer_save_dir)
                        print(f"Saved zero-3 model at step {epoch_num}.")
                    elif accelerator.is_local_main_process:
                        accelerator.wait_for_everyone()  # for model saving
                        # NOTE: Batch unlearning, save for every epoch
                        model_tokenizer_save_dir = Path(
                            os.path.join(args.model_save_dir, f"idx_{epoch_num}")
                        )
                        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

                        model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
                        tokenizer.save_pretrained(model_tokenizer_save_dir)
                        print(f"Saved model at step {epoch_num}.")

                if abs(accu_bad_loss) > args.max_bad_loss:
                    # Only printing warning at the outer loop to avoid repeated warnings.
                    break

            if abs(accu_bad_loss) > args.max_bad_loss:
                print(
                    f"bad_loss {abs(accu_bad_loss)} exceeding args.max_bad_loss {args.max_bad_loss}. Unlearning stopped."
                )
                break
    else:
        # Bytedance paper
        bad_loader_len = len(train_bad_loaders[0])
        normal_loader_len = len(train_normal_loaders[0])
        epoch_num = 0
        while idx < args.max_unlearn_steps:
            for bad_batch, normal_batch, pretrained_model_probs in zip(
                train_bad_loaders[0],
                train_normal_loaders[0],
                pretrained_model_precomputed_normal_outputs_aggregated[0],
            ):
                samples_count += len(bad_batch["input_ids"])
                loss, bad_loss = run_training_batch(
                    model=model,
                    pretrained_model_probs=pretrained_model_probs,
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
                    accelerator=accelerator,
                    logger=logger,
                )
                accelerator.backward(loss)
                optimizer.step()
                if not args.no_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                idx += 1
                final_model_tag = idx
                if idx % args.save_every == 0:
                    # NOTE: special case for zero 3
                    if accelerator.deepspeed_config is not None and accelerator.deepspeed_config['zero_optimization']['stage'] == 3:
                        print("Zero 3 optim: Saving model shards from all GPUs!")
                        model_tokenizer_save_dir = Path(
                            os.path.join(args.model_save_dir, f"idx_{epoch_num}")
                        )
                        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            model_tokenizer_save_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        tokenizer.save_pretrained(model_tokenizer_save_dir)
                        print(f"Saved zero-3 model at step {epoch_num}.")
                    elif accelerator.is_local_main_process:
                        # If not using zero 2, just save the entire model on the main process (its not sharded)
                        accelerator.wait_for_everyone()  # for model saving
                        # Save model and tokenizer.
                        model_tokenizer_save_dir = Path(
                            os.path.join(args.model_save_dir, f"idx_{idx}")
                        )
                        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

                        model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
                        tokenizer.save_pretrained(model_tokenizer_save_dir)
                        print(f"Saved model at step {idx}.")

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
                # NOTE: here I think we need to specify which process id, but it's not important for now.
                print("max_unlearn_steps reached. Unlearning stopped.")
                break
            if avg_loss := abs(np.mean(running_loss)) > args.max_bad_loss:
                print(
                    f"bad_loss {avg_loss} exceeding args.max_bad_loss {args.max_bad_loss}. Unlearning stopped."
                )
                break

    accelerator.wait_for_everyone()  # for model saving
    if accelerator.is_local_main_process:
        end_time = time.time()
        logger.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    # NOTE: special case for zero 3
    if accelerator.deepspeed_config is not None and accelerator.deepspeed_config['zero_optimization']['stage'] == 3:
        print("Zero 3 optim: Saving model shards from all GPUs!")
        model_tokenizer_save_dir = Path(
            os.path.join(args.model_save_dir, f"idx_{epoch_num}")
        )
        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            model_tokenizer_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(model_tokenizer_save_dir)
        print(f"Saved final zero-3 model at step {epoch_num}.")
        print("Unlearning finished")
        logger.info("Unlearning finished")
        if bool(args.wandb_log):
            accelerator.end_training()
    elif accelerator.is_local_main_process:
        model_tokenizer_save_dir = Path(
            os.path.join(args.model_save_dir, f"idx_{final_model_tag}")
        )
        model_tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_tokenizer_save_dir, from_pt=True)
        tokenizer.save_pretrained(model_tokenizer_save_dir)
        print("Saved final model.")
        print("Unlearning finished")

        logger.info("Unlearning finished")
        if bool(args.wandb_log):
            accelerator.end_training()

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
