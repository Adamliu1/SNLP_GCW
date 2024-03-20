#!/bin/bash

# THIS IS SPECIFIC FOR MY SERVER IN CHINA
source /etc/network_turbo
#

# Define an array of seeds
seeds=(42 123 456 114514 8888)

BASE_MODEL_SAVE_DIR=/root/autodl-tmp/model_tmp_cache
BASE_LOG_DIR=/root/autodl-tmp/log_tmp_cache
BASE_SAMPLES_DIR=/root/autodl-tmp/sample_tmp_cache

base_cache_dir="/root/autodl-tmp/cache"

save_every=100  # Define how often to save

# Base directories for model saving and logs. Ensure these directories exist or are created.
base_model_save_dir=$BASE_MODEL_SAVE_DIR
base_log_dir=$BASE_LOG_DIR
base_samples_dir=$BASE_SAMPLES_DIR

# Experiment settings
wandb_project_name="snlp_reproduce_unlearn1"
wandb_run_prefix=""  # Prefix for wandb run names

# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Create unique directories for this seed
    model_save_dir="${base_model_save_dir}/seed_${seed}"
    log_dir="${base_log_dir}/seed_${seed}"
    samples_dir="${base_samples_dir}/seed_${seed}"
    log_file="${log_dir}/training.log"

    # Ensure directories exist
    mkdir -p "$model_save_dir"
    mkdir -p "$log_dir"
    mkdir -p "$samples_dir"

    # Define a unique run name for wandb
    wandb_run_name="${wandb_run_prefix}_seed_${seed}"

    # Run the script with the current seed and directories
    python unlearn_harm.py \
        --wandb_project_name "$wandb_project_name" \
        --cache_dir "$base_cache_dir" \
        --wandb_run_name "$wandb_run_name" \
        --samples_save_dir "$samples_dir" \
        --model_save_dir "$model_save_dir" \
        --save_every "$save_every" \
        --seed "$seed" \
        --log_file "$log_file" \
        --wandb_log

    echo "Experiment with seed $seed completed."
done

echo "All experiments completed."
