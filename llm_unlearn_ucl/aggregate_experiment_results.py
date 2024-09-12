"""Results aggregation script.

Script for aggragating all evaluation results from a single experiment into a single csv file.
Importantly, it is not designed to work across experiment boundaries.
"""

import os
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the directory with experiment results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir

    # experiment_dir = "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/llama3_sequential_1e-7_BS2"
    experiment_dir_nested = os.path.join(
        experiment_dir, "experiment_data", "eval_results"
    )

    dfs = []
    for model_name in filter(
        lambda x: x != "truthfulqa", os.listdir(experiment_dir_nested)
    ):  # skip the last dir because it contains truthfulqa evals

        df_eval_harness = pd.read_csv(
            os.path.join(experiment_dir_nested, model_name, "results.csv"),
            index_col=False,
        )
        df_flagged_ratio = pd.read_csv(
            os.path.join(experiment_dir_nested, model_name, "flagged_ratio.csv"),
            index_col=False,
        ).add_suffix("_beavertails")
        df_flagged_ratio.drop(columns=df_flagged_ratio.columns[0], axis=1, inplace=True)

        df_truthfulqa = pd.read_csv(
            os.path.join(
                experiment_dir_nested,
                "truthfulqa",
                model_name,
                "results.csv",
            ),
            index_col=False,
        ).add_suffix("_truthfulqa")

        df_truthfulqa.drop(columns=df_truthfulqa.columns[0], axis=1, inplace=True)

        df = pd.concat((df_eval_harness, df_flagged_ratio, df_truthfulqa), axis=1)

        df["experiment_name"] = experiment_dir.split("/")[-1]
        df["model_name"] = model_name
        df["lr"] = float("-".join(model_name.split("-")[2:4]))
        df["splits"] = int(model_name.split("-")[4])
        df["sample_size"] = int(model_name.split("-")[5])
        dfs.append(df)

    aggregated_df = pd.concat(dfs)

    aggregated_df.drop(columns=df.columns[0], axis=1, inplace=True)

    aggregated_df.to_csv(
        os.path.join(experiment_dir, "aggregated_df.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
