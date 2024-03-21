import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--use_lora", action="store_true")

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

    # TODO: the default path needs to be reviewed
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
        "--wandb_log_feq",
        type=int,
        default=50,
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
    args = parser.parse_args()

    return args
