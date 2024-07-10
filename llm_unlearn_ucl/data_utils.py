import os
import random
from json import dump
from typing import Callable, List, Optional, Tuple, cast

from datasets import Dataset
from datasets.load import load_dataset
from numpy.random import randint
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


def make_dataset(
    dataset_uri: str,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    split: str = "train",
) -> Tuple[Dataset, str]:
    """
    dataset_uri: anything that is compatible with datasets.load_dataset
    """
    print(
        f"Loading {num_samples if num_samples is not None else 'all'} samples from {dataset_uri}..."
    )
    if dataset_uri == "truthfulqa/truthful_qa":
        full_dataset = cast(
            Dataset, load_dataset(dataset_uri, "generation", split=split)
        )
    full_dataset = cast(
        Dataset, load_dataset(dataset_uri, split=split)
    )  # the cast function does nothing but making the linter happy
    if seed is not None:
        full_dataset = full_dataset.shuffle(seed)
    if num_samples is not None:
        assert num_samples <= len(
            full_dataset
        ), f"The dataset {dataset_uri} contains {len(full_dataset)} samples while you requested {num_samples} samples."
        full_dataset = full_dataset.select(
            randint(0, len(full_dataset), size=num_samples, dtype=int)
        )
    save_path = ""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{dataset_uri.split('/')[-1]}_{len(full_dataset)}_samples.json"
        )
        with open(save_path, "w") as fin:
            print(f"Writing dataset samples to {save_dir}")
            dump([full_dataset[i] for i in range(len(full_dataset))], fin)
    return full_dataset, save_path


class DataloaderConstructor:
    def __init__(
        self,
        dataset: Dataset,
        dataset_uri: str,
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        num_splits: int = 1,
        data_mapper: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_splits = num_splits
        self.batch_size = batch_size
        if data_mapper is None:
            self.data_mapper = get_mappers(dataset_uri, tokenizer, fraction=1) or (
                lambda x: x
            )
        else:
            self.data_mapper = data_mapper
        self.dataset_uri = dataset_uri
        self.tokenizer = tokenizer
        self.__verbose = verbose
        self.__loaders = []

    """
    dataset: Dataset object for the sliced dataset from which the dataloaders will be created.
    dataset_name: str, a human-friendly name for the dataset that will be used to name the saved samples.
    data_collator: a function that takes
    verbose: bool, whether to print messages.
    """

    def get_loaders(self) -> List[DataLoader]:
        if self.__loaders == []:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
            dataset = self.dataset.map(
                self.data_mapper,
                batched=True,
                remove_columns=self.dataset.column_names,
            )
            dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "start_locs"]
            )
            self.__loaders = [
                DataLoader(
                    ds_split, batch_size=self.batch_size, collate_fn=data_collator
                )
                for ds_split in random_split(
                    dataset,
                    tuple(
                        len(dataset) // self.num_splits for _ in range(self.num_splits)
                    ),
                )
            ]
        return self.__loaders


def get_mappers(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    fraction: float,
    truncation: bool = True,
    padding: str = "max_length",
) -> Optional[Callable]:
    if dataset_name == "PKU-Alignment/PKU-SafeRLHF":

        def __pku_dataset_mapper(samples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "start_locs": [],
            }

            for i in range(len(samples["prompt"])):
                # Subsample if needed.
                if random.random() > fraction:
                    continue

                prompt = samples["prompt"][i]
                response_list = []

                if not samples["is_response_0_safe"][i]:
                    response_list.append(samples["response_0"][i])
                elif not samples["is_response_1_safe"][i]:
                    response_list.append(samples["response_1"][i])

                for response in response_list:
                    text = f"### Question: {prompt}\n ### Answer: {response}"
                    tokenized = tokenizer(text, truncation=truncation, padding=padding)

                    results["input_ids"].append(tokenized["input_ids"])
                    results["attention_mask"].append(tokenized["attention_mask"])
                    test_text = f"### Question: {prompt}\n ### Answer: "
                    test_tokenized = tokenizer(
                        test_text, truncation=truncation, padding=padding
                    )
                    results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            return results

        return __pku_dataset_mapper
    elif dataset_name == "AgentPublic/piaf":

        def __piaf_dataset_mapper(samples):
            """
            Input: Dict[List]
            Output: Dict[List]
            """
            results = {
                "input_ids": [],
                "attention_mask": [],
                "start_locs": [],
            }
            for i in range(len(samples["answers"])):
                prompt = samples["context"][i] + " " + samples["question"][i]
                response = samples["answers"][i]["text"][0]

                text = f"### Question: {prompt}\n ### Réponse: {response}"
                tokenized = tokenizer(text, truncation=truncation, padding=padding)

                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                test_text = f"### Question: {prompt}\n ### Réponse: "
                test_tokenized = tokenizer(
                    test_text, truncation=truncation, padding=padding
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            return results

        return __piaf_dataset_mapper
    elif dataset_name == "sail/symbolic-instruction-tuning":

        def __sail_dataset_mapper(samples):
            results = {"input_ids": [], "attention_mask": [], "start_locs": []}

            for i in range(len(samples["input"])):
                prompt = samples["input"][i]
                output = samples["output"][i]
                text = f"### Question: {prompt} ### Answer: {output}"

                tokenized = tokenizer(text, truncation=truncation, padding=padding)
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                test_text = f"### Question: {prompt} ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=truncation, padding=padding
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            return results

        return __sail_dataset_mapper

    elif dataset_name == "allenai/math_qa":

        def __mathqa_dataset_mapper(samples):
            results = {"input_ids": [], "attention_mask": [], "start_locs": []}
            for i in range(len(samples["Problem"])):
                # Randomly subsample if too large
                if random.random() > fraction:
                    continue

                prompt = samples["Problem"][i]
                rationale = samples["Rationale"][i]
                options = samples["options"][i]
                text = f"Problem: {prompt} options: {options} rationale: {rationale}"

                tokenized = tokenizer(text, truncation=truncation, padding=padding)
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                test_text = f"Problem: {prompt} options: {options} rationale: "
                test_tokenized = tokenizer(
                    test_text, truncation=truncation, padding=padding
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            return results

        return __mathqa_dataset_mapper

    elif dataset_name == "rajpurkar/squad":

        def __squad_data_mapper(samples):
            """
            Input: Dict[List]
            Output: Dict[List]
            """
            results = {
                "input_ids": [],
                "attention_mask": [],
                "start_locs": [],
            }

            for i in range(len(samples["context"])):
                prompt = samples["context"][i] + " " + samples["question"][i]
                answer = samples["answers"][i]["text"][0]

                # Add all responses to results or skip if none.
                text = f"### Question: {prompt}\n ### Answer: {answer}"
                tokenized = tokenizer(text, truncation=truncation, padding=padding)

                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=truncation, padding=padding
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            return results

        return __squad_data_mapper
    elif dataset_name == "truthfulqa/truthful_qa" or dataset_name == "truthful_qa":

        def __truthfulqa_data_mapper(samples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "start_locs": [],
            }
            for i in range(len(samples["question"])):
                question, answer = samples["question"][i], samples["best_answer"][i]
                text = f"### Question: {question}\n ### Answer: {answer}"
                tokenized = tokenizer(text, truncation=truncation, padding=padding)

                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])

                test_text = f"### Question: {question}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=truncation, padding=padding
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)
            return results

        return __truthfulqa_data_mapper
    else:
        return None
