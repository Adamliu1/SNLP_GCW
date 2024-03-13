import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unlearn data/skill based on a particular dataset.",
    )

    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--use_quantized",
        type=bool,
        default=False,
        help="Set to True if using quantised models.",
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
        help="Max number of unlearning steps.",
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

    args = parser.parse_args()
    return args
