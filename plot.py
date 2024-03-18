import argparse
import os
import re
from os import PathLike
from typing import Any, Dict, List, Optional

from matplotlib import pyplot as plt
from numpy import asarray, convolve, inf, ones

loss_pattern = re.compile(r".*epoch:\d*,batch:\d*,batch_loss:\s*([-+]?\d*\.?\d+)")
prev_loss_pattern = re.compile(r".*orig_model_loss:\s*([-+]?\d*\.?\d+)")
dataset_pattern = re.compile(r".*dataset:\s*(.+)")
original_model_pattern = re.compile(r".*original_model:\s*(.+)")
unlearnt_model_pattern = re.compile(r".*unlearned_model:\s*(.+)")
num_samples_pattern = re.compile(r".*num_relearn_steps:\s*(\d+)")
FIG_SIZE = (10, 8)


def show_plot(fig, path: Optional[PathLike]):
    if path is not None:
        plt.savefig(path)
    if os.getenv("SSH_TTY"):
        try:
            import imgcat

            imgcat.imgcat(fig)
        except ImportError:
            pass
    else:
        plt.show()


def clip_losses(parsed_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    min_len = inf
    target_loss = 0
    for log in parsed_logs:
        min_len = min(min_len, len(log["losses"]))
        target_loss += log["prev_loss"]

    target_loss /= len(parsed_logs)

    for log in parsed_logs:
        log["losses"] = log["losses"][:min_len]
        log["prev_loss"] = target_loss
    return parsed_logs


def make_label(parsed_result: Dict[str, Any]) -> str:
    return f'{parsed_result["unlearned_model"].split("/")[-1]} @ {parsed_result["num_samples"]} steps'


def parse_log(log_file_path: str, smooth_factor: int) -> Dict[str, Any]:
    result = {}
    result["losses"] = []
    result["prev_loss"] = 0
    result["dataset"] = ""
    result["original_model"] = ""
    result["unlearned_model"] = ""
    result["num_samples"] = 0
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            match = loss_pattern.match(lines[i])
            if match:
                result["losses"].append(float(match.group(1)))
            match = prev_loss_pattern.match(lines[i])
            if match:
                result["prev_loss"] = float(match.group(1))
            match = dataset_pattern.match(lines[i])
            if match:
                result["dataset"] = match.group(1).strip()
            match = original_model_pattern.match(lines[i])
            if match:
                result["original_model"] = match.group(1).strip()
            match = unlearnt_model_pattern.match(lines[i])
            if match:
                result["unlearned_model"] = match.group(1).strip()
            match = num_samples_pattern.match(lines[i])
            if match:
                result["num_samples"] = int(match.group(1))
    if smooth_factor > 1:
        result["losses"] = asarray(result["losses"])
        result["losses"] = convolve(
            result["losses"], ones((smooth_factor,)) / smooth_factor, mode="valid"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the data")
    parser.add_argument("log_file", help="Path to the log file.", nargs="+")
    parser.add_argument(
        "--merged", action="store_true", help="Combine all logs into one plot."
    )
    parser.add_argument(
        "--clipped",
        action="store_true",
        help="When multiple logs are supplied, clip the losses of longer runs so that the number of samples match the shortest run.",
    )
    parser.add_argument(
        "--smooth_factor",
        type=int,
        default=1,
        help="Smoothing factor the loss function. Higher means smoother.",
    )
    args = parser.parse_args()
    args.log_file.sort(key=lambda x: (len(x), x))
    results = [parse_log(i, args.smooth_factor) for i in args.log_file]
    if args.clipped:
        results = clip_losses(results)
    if len(args.log_file) == 1:
        data = args.log_file[0]
        parsed_result = parse_log(data, args.smooth_factor)
        fig = plt.figure(dpi=200)
        plt.title(
            f"Loss function for relearning in {parsed_result['num_samples']} samples.\nDataset: {parsed_result['dataset']}, base model: {parsed_result['original_model']}, unlearned_model: {parsed_result['unlearned_model'].split('/')[-1]}"
        )
        plt.yscale("log")
        plt.plot(
            range(1, len(parsed_result["losses"]) + 1),
            parsed_result["losses"],
            label=make_label(parsed_result),
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.axhline(
            y=parsed_result["prev_loss"],
            color="r",
            linestyle="dotted",
            label="loss of the original model",
        )
        plt.legend()

        for key in parsed_result:
            if not isinstance(parsed_result[key], list):
                print(f"{key}:\t{parsed_result[key]}")
        show_plot(fig, "loss_funcs.png")
    elif args.merged:
        fig = plt.figure(dpi=200, figsize=FIG_SIZE)
        for parsed_result in results:
            plt.plot(
                range(1, len(parsed_result["losses"]) + 1),
                parsed_result["losses"],
                label=make_label(parsed_result),
            )
        plt.yscale("log")
        plt.title(
            f"Comparison of relearning rate using the loss on {results[0]['dataset']}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.axhline(
            y=parsed_result["prev_loss"],
            color="r",
            linestyle="dotted",
            label=f"loss of the original model: {parsed_result['prev_loss']}",
        )
        plt.legend()
        show_plot(fig, "loss_funcs.png")
    else:
        results = [parse_log(i, args.smooth_factor) for i in args.log_file]
        if args.clipped:
            results = clip_losses(results)
        plt.yscale("log")
        fig, ax = plt.subplots(len(args.log_file), 1, dpi=200, figsize=FIG_SIZE)
        for idx, log_file in enumerate(args.log_file):
            parsed_result = results[idx]
            ax[idx].set_title(
                f"Loss function for relearning in {parsed_result['num_samples']} samples.\nDataset: {parsed_result['dataset']}, base model: {parsed_result['original_model']}, unlearned_model: {parsed_result['unlearned_model'].split('/')[-1]}"
            )
            ax[idx].plot(
                range(1, len(parsed_result["losses"]) + 1),
                parsed_result["losses"],
                label=make_label(parsed_result),
            )
            ax[idx].set_xlabel("Epochs")
            ax[idx].set_ylabel("Loss")
            ax[idx].axhline(
                y=parsed_result["prev_loss"],
                color="r",
                linestyle="dotted",
                label=f"loss of the original model: {parsed_result['losses']}",
            )
            ax[idx].legend()
        show_plot(fig, "loss_funcs.png")
    if len(results) > 1:
        fig = plt.figure(dpi=200, figsize=FIG_SIZE)
        plt.yscale("log")
        plt.title("Number of samples required for relearning for each model.")
        plt.xlabel("Model")
        plt.ylabel("Number of samples")
        plt.plot(
            [i["unlearned_model"].split("_")[-1] for i in results],
            [i["num_samples"] for i in results],
        )
        fig.autofmt_xdate()
        show_plot(fig, "num_samples.png")
