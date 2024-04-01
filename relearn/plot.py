import argparse
import glob
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


class ExperimentType(Enum):
    UNKNOWN = -1
    BATCH_UNLEARNING = 1
    BATCH_UNLEARNING_SCALE_LR = 2
    SEQUENTIAL_UNLEARNING = 3
    GRADIENT_ASCENT = 4

    def __int__(self):
        return self.value


@dataclass
class Result:
    dataset: str
    model_name: str
    checkpoint: int
    losses: List[float]
    sample_count: int
    label: str = ""
    experiment_type: ExperimentType = ExperimentType.UNKNOWN


def expand_path(path: List[str]) -> List[str]:
    output = []
    for p in path:
        if "*" in p:
            output.extend(glob.glob(p, recursive=True))
        if os.path.isdir(p):
            output.extend(expand_path([os.path.join(p, f) for f in os.listdir(p)]))
        else:
            output.append(p)
    return output


def get_label(result: Result):
    if result.model_name.startswith("batch_size"):
        sample_size = int(result.model_name.split("_")[2])
        name = f"Batch Unlearning\non {sample_size} Samples"
        if "lr" in result.model_name:
            result.label = f"{name}\nwith Scaled LR"
            result.experiment_type = ExperimentType.BATCH_UNLEARNING_SCALE_LR
        else:
            result.experiment_type = ExperimentType.BATCH_UNLEARNING
            result.label = name
    elif result.model_name.startswith("samples_count"):
        sample_count = int(result.model_name.split("_")[2])
        num_split = int(result.model_name.split("_")[-1])
        result.label = (
            f"Sequential Unlearning\non {sample_count} Samples in {num_split} Splits"
        )
        result.experiment_type = ExperimentType.SEQUENTIAL_UNLEARNING
    else:
        result.experiment_type = ExperimentType.GRADIENT_ASCENT
        result.label = f'Gradient Ascent\nwith seed {result.model_name.split("-")[-1].replace("real", "")}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting the relearning data from given jsons."
    )
    parser.add_argument("jsons", metavar="jsons", type=str, nargs="+")

    args = parser.parse_args()

    json_files = expand_path(args.jsons)
    eval_results: List[Result] = []
    for i in json_files:
        with open(i) as fin:
            content = json.load(fin)
            eval_results.append(
                Result(
                    dataset=content["dataset"],
                    model_name=content["unlearned_model"],
                    checkpoint=int(content["checkpoint"].split("_")[-1]),
                    losses=content["losses"],
                    sample_count=content["sample_count"],
                )
            )
    for i in eval_results:
        get_label(i)
    eval_results.sort(key=lambda x: (len(x.model_name), x.model_name))

    plt.rcParams.update({"font.size": 18})
    # NOTE: Start plotting from here!!!

    # NOTE: Batch Unlearning Results
    fig, ax = plt.subplots(figsize=(10, 8))
    batch_unlearning_results = [
        i for i in eval_results if i.experiment_type == ExperimentType.BATCH_UNLEARNING
    ]
    bars = ax.bar(
        [i.label for i in batch_unlearning_results],
        [i.sample_count for i in batch_unlearning_results],
    )
    ax.bar_label(bars, label_type="edge")
    plt.ylabel("Number of Samples Used for Relearning")
    # fig.autofmt_xdate()
    ax.set_title("Unlearning Sample Count for Batch Unlearning Models")
    plt.show()

    # NOTE: Sequential Unlearning (512 samples) Results
    fig, ax = plt.subplots(figsize=(10, 8))
    seq_unlearning_results = [
        i
        for i in eval_results
        if (
            "512" in i.model_name
            and i.experiment_type != ExperimentType.BATCH_UNLEARNING_SCALE_LR
        )
    ]
    bars = ax.bar(
        [i.label for i in seq_unlearning_results],
        [i.sample_count for i in seq_unlearning_results],
    )
    ax.bar_label(bars, label_type="edge")
    plt.ylabel("Number of Samples Used for Relearning")
    ax.set_title("Unlearning Sample Count for Sequential Unlearning")
    plt.show()

    # NOTE: Sequential Unlearning (128, 512 and 1024 samples) Results
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(3)
    y_128 = [
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.SEQUENTIAL_UNLEARNING
        and "128" in i.model_name
    ]
    y_512 = [
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.SEQUENTIAL_UNLEARNING
        and "512" in i.model_name
    ]
    y_1024 = [
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.SEQUENTIAL_UNLEARNING
        and "1024" in i.model_name
    ]
    for i in [y_128, y_512, y_1024]:
        i.sort(key=lambda x: int(x.model_name.split("_")[-1]))
    width = 0.2
    ax.bar_label(
        ax.bar(
            x - width,
            [i.sample_count for i in y_128],
            width=width,
            label="128 Samples",
        ),
        [i.sample_count for i in y_128],
    )
    ax.bar_label(
        ax.bar(x, [i.sample_count for i in y_512], width=width, label="512 Samples"),
        [i.sample_count for i in y_512],
    )
    ax.bar_label(
        ax.bar(
            x + width,
            [i.sample_count for i in y_1024],
            width=width,
            label="1024 Samples",
        ),
        [i.sample_count for i in y_1024],
    )
    plt.xticks(x, ["4 Splits", "16 Splits", "64 Splits"])
    plt.ylabel("Number of Samples Used for Relearning")
    ax.set_title("Unlearning Sample Count for Sequential Unlearning")
    plt.legend()
    plt.show()

    # NOTE: BATCH_UNLEARNING_SCALE_LR
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(5)
    ys: Dict[int, List[Result]] = {32: [], 128: [], 512: [], 1024: [], 2048: []}
    for i in eval_results:
        if i.experiment_type in (
            ExperimentType.BATCH_UNLEARNING_SCALE_LR,
            ExperimentType.BATCH_UNLEARNING,
        ):
            trained_sample = int(i.model_name.split("_")[2])
            if trained_sample in ys:
                ys[trained_sample].append(i)
    for i in ys.values():
        i.sort(key=lambda x: int(x.experiment_type))

    width = 0.4
    for i in range(2):
        sample_count_values = [ys[idx][i].sample_count for idx in ys.keys()]
        ax.bar_label(
            ax.bar(
                x - width * (-1) ** i / 2,
                sample_count_values,
                width=width,
                label="Normal LR" if i == 0 else "Scaled LR",
            ),
            sample_count_values,
        )
    plt.xticks(x, [f"Batch Unlearning\nwith {i} Samples" for i in ys.keys()])
    plt.ylabel("Number of Samples Used for Relearning")
    ax.set_title("Unlearning Sample Count with Scaled Learning Rate")
    plt.legend()
    plt.show()

    # NOTE: Gradient Ascent vs batch vs sequential
    fig, ax = plt.subplots(figsize=(10, 8))
    results = [
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.GRADIENT_ASCENT and "42" in i.model_name
    ]
    results.extend(
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.BATCH_UNLEARNING_SCALE_LR
        and "512" in i.model_name
    )
    results.extend(
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.BATCH_UNLEARNING
        and "512" in i.model_name
    )
    results.extend(
        i
        for i in eval_results
        if i.experiment_type == ExperimentType.SEQUENTIAL_UNLEARNING
        and "512" in i.model_name
    )
    x = np.arange(len(results))
    ax.bar_label(
        ax.bar(x, [i.sample_count for i in results]), [i.sample_count for i in results]
    )
    plt.xticks(x, [i.label for i in results])
    ax.set_ylabel("Number of Samples Used for Relearning")
    ax.set_title("Unlearning Sample Count for Different Unlearning Methods")
    fig.autofmt_xdate()
    plt.show()
