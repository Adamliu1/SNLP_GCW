# Copyright (C) 2024 UCL CS NLP
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path pointing to the model to evaluate. E.g. facebook/opt-1.3b.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="truthfulqa/truthful_qa",
        help="Path from which to load the truthfulQA dataset. By default download from HuggingFace.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=2, help="Evaluation batch size"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Number of tokens to generate per each question, on which the model is evaluated.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Select the device to run this on. By default will be None (so quantised model will succeed)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_generations",
        help="Directory to which the output JSON is saved.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache_dir",
        help="Path to the cache directory.",
    )

    args = parser.parse_args()

    return args
