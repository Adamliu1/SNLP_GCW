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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Outputs a JSON file with prompt, response, model name, category_id "
        "for a paritcular model checkpoint. Dataset used: PKU-Alignment/BeaverTails-Evaluation.\n"
        "https://huggingface.co/datasets/PKU-Alignment/BeaverTails-Evaluation",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path pointing to the model to evaluate. E.g. facebook/opt-1.3b.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="PKU-Alignment/BeaverTails-Evaluation",
        help="Path from which to load the PKU-Alignment/BeaverTails-Evaluation dataset. By default load from HuggingFace.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Evaluation batch size"
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
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generators",
    )

    args = parser.parse_args()

    return args
