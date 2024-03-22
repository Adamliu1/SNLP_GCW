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

import random
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling


def create_mathqa_dataloader_from_dataset(
    tokenizer, dataset, fraction=1.0, batch_size=4
):
    # MathQA structure:
    """
    # Problem ; Rationale ; options ; correct ; annotated_formula
    example:
    Problem: the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?
    Rationale: "explanation : t = 3 years r = 10 % td = ( bg × 100 ) / tr = ( 36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs . 120 td = ( pw × tr ) / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = 1200 / 3 = rs . 400 answer : option a"
    options: a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these
    correct: a
    annoatated_formula: divide(multiply(const_100, divide(multiply(36, const_100), multiply(3, 10))), multiply(3, 10))

    For now, only extracting Problem, options and correct
    """

    def preprocess(examples):
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}
        for i in range(len(examples["Problem"])):
            # Randomly subsample if too large
            if random.random() > fraction:
                continue

            prompt = examples["Problem"][i]
            rationale = examples["Rationale"][i]
            options = examples["options"][i]
            # correct = examples["correct"][i]
            # annotated_formula["annotated_formula"][i]
            text = f"Problem: {prompt} options: {options} rationale: {rationale}"

            tokenized = tokenizer(text, truncation=True, padding="max_length")
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            # Calculate start idx for answer
            # XXX: @Andrzej said apparently it might work better without space below??? (check?)
            test_text = f"Problem: {prompt} options: {options} rationale: "
            test_tokenized = tokenizer(test_text, truncation=True, padding="max_length")
            results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=[
            "Problem",
            "Rationale",
            "options",
            "correct",
            "annotated_formula",
            "linear_formula",
            "category",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_pku_dataloader_from_dataset(
    tokenizer, dataset, fraction=1.0, batch_size=4, splits: int = 1
):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size used for each step.
        splits: The number of splits that the dataset will be sliced into.
    Returns:
        A List of DataLoader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {
            "input_ids": [],
            "attention_mask": [],
            "start_locs": [],
        }

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            elif not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")

                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_locs"],
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TODO: data_collator introduces extra/less processed samples.
    dataloaders = [
        torch.utils.data.DataLoader(
            train_split_dataset, batch_size=batch_size, collate_fn=data_collator
        )
        for train_split_dataset in torch.utils.data.random_split(
            dataset, tuple(len(dataset) // splits for i in range(splits))
        )
    ]

    return dataloaders


def create_truthfulqa_dataloader(
    tokenizer,
    batch_size=4,
    num_samples: Optional[int] = 64,
    seed: Optional[int] = 42,
    splits: int = 1,
):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size used for each step.
        num_samples: Number of samples used in the training set.
        seed: seed used for sampling and shuffling of the dataset.
        splits: Number of splits to divide the training samples into. Default: 1
    Returns:
        A list of DataLoaders of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    if seed is not None:
        df = df.sample(frac=1, random_state=seed)
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    raw_train_data = []
    for question, good_answer in zip(questions, good_answers):
        raw_train_data.append({"Question": question, "Best Answer": good_answer})
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])
    dataset = Dataset.from_dict(data)
    assert num_samples is None or num_samples < 0.7 * len(
        dataset
    ), f"num_samples is too large. max is {int(0.7 * len(dataset))}."

    # Split train/val/test = 0.7/0.1/0.2.
    train_len = num_samples or int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if num_samples is None:
        num_samples = len(train_data)
    train_dataloaders = [
        torch.utils.data.DataLoader(
            train_batch_ds,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )
        for train_batch_ds in torch.utils.data.random_split(
            train_data, tuple(num_samples // splits for i in range(splits))
        )
    ]

    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloaders, val_dataloader, test_dataloader, raw_train_data


def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans


def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss


def get_answer_loss(operation, batch, model, device="cuda:0"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(
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
    random_loss = get_answer_loss("gd", batch_random, model, device=device)

    return random_loss


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Test the Math QA dataloader generation
    train_dataset = load_dataset("math_qa", split="train")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        cache_dir="./cache",
    )
    train_bad_loader = create_mathqa_dataloader_from_dataset(
        tokenizer, train_dataset, fraction=0.2, batch_size=2
    )
    iterator = iter(train_bad_loader)
    bad_input_ids = next(iterator)["input_ids"]
    single_input_id = bad_input_ids[0, :]
    print(tokenizer.decode(single_input_id))
