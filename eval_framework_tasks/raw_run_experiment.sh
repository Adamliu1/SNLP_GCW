#! /bin/bash

# TODO: Add instructions how to use.

# Accelerate used in a multi-gpu setting. Python for single-gpu.
# LAUNCHER="accelerate launch"
LAUNCHER="accelerate launch"

# Experiment config
MODEL_NAME=$1
EXPERIMENT_NAME=$MODEL_NAME-eval1

# Environment
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

# Paths and directories
# MODELS_PATH=meta-llama/Llama-2-7b-hf
# MODELS_PATH=meta-llama/Meta-Llama-3-8B
# MODELS_PATH=google/gemma-7b
# MODELS_PATH=CohereForAI/aya-23-8B
MODELS_PATH=$2
# MODELS_PATH=allenai/OLMo-7B-hf
# MODELS_PATH=microsoft/phi-1_5
# MODELS_PATH=google/gemma-2b
# MODELS_PATH=facebook/opt-1.3b

MODELS_CACHE_PATH=/scratch0/aszablew/raw_models_cache
EXPERIMENT_SCRATCH_PATH=/scratch0/aszablew/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
WANDB_PATH=$EXPERIMENT_SCRATCH_PATH/wandb

mkdir -p $EXPERIMENT_SCRATCH_PATH

mkdir $WANDB_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Tasks selection
TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,french_bench,mnli,piqa,squadv2,toxigen,gsm8k"

# Run evaluation

hostname

echo $(date)

HF_HOME=/scratch0/aszablew/.huggingface CUDA_VISIBLE_DEVICES=$3 $LAUNCHER --num_processes=1 -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH \
    --tasks $TASKS \
    --device "cuda" \
    --batch_size "auto" \
    --trust_remote_code \
    --output_path $RESULTS_PATH/$MODEL_NAME/$MODEL_NAME.json

cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/

date
