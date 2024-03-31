#!/bin/bash

# THIS IS SPECIFIC FOR A40 IN CHINA
source /etc/network_turbo
#


BASE_MODEL_SAVE_DIR=/root/autodl-tmp/model_tmp_cache
BASE_LOG_DIR=/root/autodl-tmp/log_tmp_cache
BASE_SAMPLES_DIR=/root/autodl-tmp/sample_tmp_cache

base_cache_dir="/root/autodl-tmp/cache"

save_every=4  # for batch, save every 4 epochs

# Base directories for model saving and logs. Ensure these directories exist or are created.
base_model_save_dir=$BASE_MODEL_SAVE_DIR
base_log_dir=$BASE_LOG_DIR
base_samples_dir=$BASE_SAMPLES_DIR

# Experiment settings
#batch_sizes=(2 8)
batch_sizes=(8)
physical_batch=2
sequence_count=1
seed=42
epoch_count=20
wandb_project_name="snlp"
wandb_run_prefix="batch_unlearn"  # Prefix for wandb run names

# run for each batch size
for batch_size in "${batch_sizes[@]}"; do
    # Create unique directories for this seed
    model_save_dir="${base_model_save_dir}/batch_size_${batch_size}"
    log_dir="${base_log_dir}/batch_size_${batch_size}"
    samples_dir="${base_samples_dir}/batch_size_${batch_size}"
    log_file="${log_dir}/batch_size_${batch_size}.log"

    # Ensure directories exist
    mkdir -p "$model_save_dir"
    mkdir -p "$log_dir"
    mkdir -p "$samples_dir"

    # Define a unique run name for wandb
    wandb_run_name="${wandb_run_prefix}_batch_${batch_size}_seed_${seed}"
    echo $wandb_run_name

    # Run the script with the current seed and directories
    #TODO: for samplecount < physical batch -> set batch_size to sample_count
    python unlearn_harm.py \
	--batch_size "$physical_batch" \
	--lr 2e-6 \
        --model_save_dir "$model_save_dir" \
        --samples_save_dir "$samples_dir" \
        --save_every "$save_every" \
        --log_file "$log_file" \
        --cache_dir "$base_cache_dir" \
        --seed "$seed" \
	--sequential "$sequence_count" \
        --wandb_log \
    	--wandb_log_freq 1 \
        --wandb_project_name "$wandb_project_name" \
        --wandb_run_name "$wandb_run_name" \
	--samples_count "$batch_size" \
	--num_epochs "$epoch_count" \
	--no_scheduler
	#--model_name "facebook/opt-1.3b" \
	# TODO: for sequential, needs to change samples count


    echo "Experiment with seed $seed and batch size $batch_size completed."
done

echo "All experiments completed."
