import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="The maximum sequence length of the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run the evaluation.",
    )
    parser.add_argument(
        "--use_existing_evaluation",
        action="store_true",
        help="Uses already existing evaluation file at output_dir to just print and plot its contents!",
    )
    return parser.parse_args()
