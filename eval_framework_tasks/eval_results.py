# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Carmen Meinson
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform analysis of lm-eval-harness eval runs."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the directory containing evaluation result logs.",
    )
    args = parser.parse_args()

    return args


def fetch_log_data(log_dir: str) -> dict[int, dict]:
    log_file_names = os.listdir(log_dir)
    assert (
        log_file_names
    ), f"Beep boop, no files in a directory provided ({log_dir}). Maybe you forgot to copy them?"

    log_files_contents = {}
    for file_name in log_file_names:
        with open(os.path.join(log_dir, file_name), "r") as f:
            log_files_contents[int(file_name.split("_")[-1].split(".")[0])] = json.load(
                f
            )

    # return sorted dictionary
    return {
        k: v
        for k, v in sorted(
            log_files_contents.items(),
            key=lambda x: x[0],
        )
    }


def filter_json_logs(log_data: dict[int, dict]) -> dict[int, dict]:
    results = {}
    for model_iter_num, log in log_data.items():
        results[model_iter_num] = {
            "winogrande_acc": log["results"]["winogrande"]["acc,none"],
            "truthfulqa_mc2_acc": log["results"]["truthfulqa_mc2"]["acc,none"],
            "hellaswag_acc_norm": log["results"]["hellaswag"]["acc_norm,none"],
            "gsm8k_exact_match_flexible": log["results"]["gsm8k"][
                "exact_match,flexible-extract"
            ],
            "arc_challenge_acc_norm": log["results"]["arc_challenge"]["acc_norm,none"],
            "mmlu_acc": log["results"]["mmlu"]["acc,none"],
            "toxigen_acc_norm": log["results"]["toxigen"]["acc_norm,none"],
        }

    return results


def create_plot(df: pd.DataFrame, log_dir: str) -> None:
    ax = df.plot(style=["o-"] * 7)
    fig = ax.get_figure()
    fig.savefig(os.path.join(log_dir, "..", "figure.png"))


def main(log_dir: str):
    log_data = fetch_log_data(log_dir)
    filtered_logs = filter_json_logs(log_data)

    df = pd.DataFrame.from_dict(filtered_logs).transpose()

    create_plot(df, log_dir)
    df.to_csv(os.path.join(log_dir, "..", "results.csv"))


if __name__ == "__main__":
    args = parse_args()
    main(args.log_dir)
