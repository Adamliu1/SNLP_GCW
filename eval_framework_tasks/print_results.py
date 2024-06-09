# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
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
    parser = argparse.ArgumentParser(description="Print some nice results.")

    parser.add_argument(
        "--file_paths",
        type=str,
        required=True,
        help="Comma-separated file paths to print.",
    )
    args = parser.parse_args()

    return args


def fetch_log_data(file_paths: str) -> dict[int, dict]:
    log_file_names = file_paths.split(",")

    log_files_contents = {}
    for file_name in log_file_names:
        with open(os.path.join(file_name), "r") as f:
            log_files_contents[file_name.split("/")[-2].split(".")[0]] = json.load(f)

    return log_files_contents


def filter_json_logs(log_data: dict[int, dict]) -> dict[int, dict]:
    results = {}
    for model_iter_num, log in log_data.items():
        results[model_iter_num] = {
            "winogrande_acc": log["results"]["winogrande"]["acc,none"],
            "truthfulqa_mc2_acc": log["results"]["truthfulqa_mc2"]["acc,none"],
            "hellaswag_acc": log["results"]["hellaswag"]["acc,none"],
            "gsm8k_exact_match_flexible": log["results"]["gsm8k"][
                "exact_match,flexible-extract"
            ],
            "arc_challenge_acc": log["results"]["arc_challenge"]["acc,none"],
            "mmlu_acc": log["results"]["mmlu"]["acc,none"],
            "logiqa_acc": log["results"]["logiqa"]["acc,none"],
            "french_bench_acc": log["results"]["french_bench"]["acc,none"],
            "piqa_acc": log["results"]["piqa"]["acc,none"],
            "squadv2": log["results"]["squadv2"]["f1,none"],
        }

    return results


def create_plot(df: pd.DataFrame, file_paths: str) -> None:
    ax = df.plot(style=["o-"] * 7)
    fig = ax.get_figure()
    fig.savefig(os.path.join(file_paths, "..", "figure.png"))


def main(file_paths: str):
    log_data = fetch_log_data(file_paths)
    filtered_logs = filter_json_logs(log_data)

    df = pd.DataFrame.from_dict(filtered_logs).transpose()

    print(df.to_markdown())
    # create_plot(df, file_paths)
    # df.to_csv(os.path.join(file_paths, "..", "results.csv"))


if __name__ == "__main__":
    args = parse_args()
    main(args.file_paths)
