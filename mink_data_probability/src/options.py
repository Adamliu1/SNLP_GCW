import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="facebook/opt-1.3b", help="the model to attack: e.g. huggyllama/llama-7b")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--data', type=str, default="swj0419/WikiMIA", help="the dataset to evaluate: default is WikiMIA")
        self.parser.add_argument('--split', type=str, default="WikiMIA_length64", help="the length of the input text to evaluate. split for WikiMIA Choose from WikiMIA_length32, WikiMIA_length64, WikiMIA_length128, WikiMIA_length256")
        self.parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")
        self.parser.add_argument('--cache_dir', type=str, default=".cache", help="the cache directory to store the dataset")

        self.parser.add_argument('--max_samples', type=int, default=400, help="max samples to test on")




