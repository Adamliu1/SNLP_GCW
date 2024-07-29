# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# Adapted from https://github.com/kevinyaobytedance/llm_unlearn.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import wandb
import os
from accelerate import Accelerator


def prepare_and_upload_training_batch_raw_data(
    args: dict, normal_sample_path: str, bad_sample_path: str, accelerator: Accelerator
):
    data_sample_artifacts = wandb.Artifact(
        name="training_batch_raw_data", type="batch_data"
    )
    if os.path.isfile(normal_sample_path):
        data_sample_artifacts.add_file(
            normal_sample_path, name=f"normal_{args.samples_count}_samples.json"
        )
    if os.path.isfile(bad_sample_path):
        data_sample_artifacts.add_file(
            bad_sample_path, name=f"bad_{args.samples_count}_samples.json"
        )
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    wandb_tracker.log_artifact(data_sample_artifacts)
