# Copyright (C) 2024 UCL CS NLP
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from random import seed as pyseed
from numpy.random import seed as npseed
import torch

def reproducibility(seed: int) -> int:
    """
    Seeds python, torch and numpy random number generators.
    Since numpy Seed must be between 0 and 2**32 - 1, clamps to this region using
    modulo and return the actual set seed.

    Sets CUDA determinism for Conv2D based on:
    https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-determinism
    At reduced performance cost, disables cuDNN benchmarking feature, which normally
    optimises for best convolution algo.
    https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking

    Args:
        seed (int): Seed to use for reproducibility.

    Returns:
        int: seed set for all number generators, clamped to the range [0, 2**32 -1]
    """
    # clamp the seed between 0 and 2**32 - 1
    seed = seed % (2**32)
    pyseed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    npseed(seed)
    if torch.cuda.is_available():
        # NB If multiple GPUs will be used together, this is not sufficient.
        # need to: manual_seed_all().
        # https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html#torch.cuda.manual_seed
        torch.cuda.manual_seed(seed)
    return seed


def get_dict_from_args_str(args_str: str) -> dict[str, str]:
    """Converts a comma-separated string of key-value pairs into a dictionary.

    Args:
        args_str (str): Comma-separated string of key-value pairs.

    Returns:
        dict[str, str]: Converted dictionary with corresponding key-value pairs.
    """
    args_dict = {}
    for key_val in args_str.split(","):
        assert (
            "=" in key_val
        ), "Beep boop... Wrong format of wandb_args. Make sure it has the following format: key1=value1,key2=value2"
        k, v = key_val.split("=")
        args_dict[k] = v

    assert (
        "project" in args_dict
    ), "Beep boop... Wrong format of wandb_args. Missing value for `project`."

    return args_dict
