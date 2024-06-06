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

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unlearn data/skill based on a particular dataset.",
    )

    parser.add_argument(
        "--mink_prob_k",
        type=float,
        default=0.2,
        help="K value as a float for percentage of of lowest "
        "probability tokens to consider, used by Min-K percent PROB https://arxiv.org/pdf/2310.16789.pdf",
    )

    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--use_quantized",
        type=bool,
        default=False,
        help="Set to True if using quantised models.",
    )

    parser.add_argument(
        "--retaining_dataset",
        type=str,
        default="truthful_qa",
        help="Name of the dataset to retain (NOTE: it needs to have"
        " a custom dataloader creation script in utils.py)",
    )
    parser.add_argument(
        "--unlearning_dataset",
        type=str,
        default="PKU-Alignment/PKU-SafeRLHF",
        help="Name of the dataset to unlearn (NOTE: it needs to have"
        " a custom dataloader creation script in utils.py)",
    )
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps for vanilla GA unlearning (bytedance).",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )

    parser.add_argument(
        "--samples_save_dir",
        type=str,
        default="data/saved_samples",
        help="The directory that saves the sample used for unlearning",
    )

    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="Directory to save cache files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default="8888",
        help="Seed used for unlearning. Default codebase uses 8888 as default seed.",
    )

    parser.add_argument(
        "--sequential",
        default=-1,
        type=int,
        help="Number of splits used for sequential unlearning. Should be divisible by samples_count. Default: -1 (Do not use). For batch unlearning, use 1.",
    )

    parser.add_argument(
        "--wandb_log",
        action="store_true",
        default=True,
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=1,
        help="The logging frequency for wandb to upload data",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="snlp",
        help="The name of the wandb project,",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the wandb run,",
    )
    parser.add_argument(
        "--num_running_loss",
        type=int,
        default=10,
        help="Number of losses to keep for computing running average loss.",
    )
    parser.add_argument(
        "--samples_count",
        type=int,
        default=64,
        help="Number of samples per epoch from the unlearning set to unlearn on.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="if set to an int, shuffle the dataset with this seed. Otherwise, the seed supplied to --seed will be used.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to unlearn."
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_true",
        help="Set this flag to disable lr_scheduler",
    )
    args = parser.parse_args()

    return args
