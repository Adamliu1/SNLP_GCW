# Copyright (C) 2024 UCL CS NLP
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_scripts.parse_args import parse_arguments
from evaluation_scripts.metrics import (
    calculate_question_repeated_ratio,
    calculate_output_quality_heuristics,
    calculate_response_length,
)


def plot_metrics(metrics: list[dict], output_dir: str, plot_title: str) -> None:
    """Plot metrics."""
    model_names = np.asarray([row["model_name"] for row in metrics])
    # moderation = np.asarray([row["flagged/all"] for row in metrics])
    special_chars = np.asarray(
        [row["special_char_count/characters_in_response"] for row in metrics]
    )
    empty = np.asarray([row["empty_response_ratio"] for row in metrics])
    bar_width = 0.25
    index = np.arange(len(special_chars))
    _, ax = plt.subplots(figsize=(8, 6), dpi=150)

    avg_response_length = np.asarray([row["avg_response_length"] for row in metrics])
    ax.bar(
        index,
        avg_response_length,
        bar_width,
        color="#FF6D60",
        alpha=0.85,
        zorder=2,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Characters")

    ax_twin = ax.twinx()

    ax_twin.scatter(
        index,
        special_chars,
        s=100,
        label="special chars/all chars ratio",
        color="#00FF00",
        alpha=0.85,
        zorder=2,
        marker="s",
    )

    ax_twin.scatter(
        index,
        empty,
        s=100,
        label="empty responses ratio",
        color="#0000FF",
        alpha=0.85,
        zorder=2,
    )

    plt.legend(bbox_to_anchor=(0.55, -0.4), loc="lower right")

    ax.grid(axis="y", color="k", alpha=0.2, zorder=1)

    ax.set_xticks(index)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Average response length (chars.)")
    ax.set_title(f"TruthfulQA eval: {plot_title}")
    ax_twin.set_ylabel("Percentage (%)")
    ax_twin.set_yticks(np.arange(0, 1, 0.1))
    ax_twin.set_yticklabels([f"{i*10}%" for i in range(0, 10, 1)])
    ax_twin.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot.png"))


def main() -> None:
    args = parse_arguments()

    assert (
        args.eval_dataset is not None
    ), "Beep boop... you need to provide path of the directory with generated answers."

    os.makedirs(args.output_dir, exist_ok=True)

    log_file_names = os.listdir(args.eval_dataset)
    assert (
        log_file_names
    ), f"Beep boop... no files in a directory provided ({args.eval_dataset}). Something went wrong :("

    data = []
    for file_name in log_file_names:
        with open(os.path.join(args.eval_dataset, file_name), "r") as f:
            data.extend(json.load(f))

    model_names_set = set([line["model"] for line in data])
    try:
        model_names = sorted(model_names_set, key=lambda x: int(x.split("_")[1]))
    except:
        # model names do not contain idxs
        model_names = list(model_names_set)

    metrics = []
    for model_name in model_names:
        model_data = [line for line in data if line["model"] == model_name]

        metrics.append(
            {
                "model_name": model_name,
                **calculate_output_quality_heuristics(model_data),
                **calculate_response_length(model_data),
                **calculate_question_repeated_ratio(model_data),
            },
        )

    # report to terminal and save to file
    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    plot_metrics(metrics, args.output_dir, "")


if __name__ == "__main__":
    main()
