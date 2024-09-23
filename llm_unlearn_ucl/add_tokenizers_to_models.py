# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from transformers import AutoTokenizer
import argparse


def main(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model from huggingface for which to download a tokenizer.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path to the dir of the local saved model to attach a tokenizer.",
    )
    parser.add_argument("--cache_dir", type=str, default=".cache")

    args = parser.parse_args()
    main(args)
