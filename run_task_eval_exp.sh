#! /bin/bash

# TODO: Add instructions how to use.

# Accelerate used in a multi-gpu setting. Python for single-gpu.
LAUNCHER="accelerate launch"
# LAUNCHER="python3"

# Experiment config
# MODEL_NAME=batch-2048-lr4.53e-05
MODEL_NAME=$1
# eval1 is lm-eval_tasks
EXPERIMENT_NAME=$MODEL_NAME-eval1

# Environment
# TODO: change this to ur path, final step will cp results to this path
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

# Paths and directories
# MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
MODELS_PATH=$2/$1

# TODO: change this to where to store your results
EXPERIMENT_SCRATCH_PATH=/scratch0/aszablew/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
WANDB_PATH=$EXPERIMENT_SCRATCH_PATH/wandb

mkdir $EXPERIMENT_SCRATCH_PATH

mkdir $WANDB_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Tasks selection
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,french_bench,mnli,piqa,squadv2"
# HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa"
# TASKS_ADDITIONAL="toxigen"
# TASKS=$HF_LLM_LEADERBOARD_TASKS,$TASKS_ADDITIONAL
TASKS=$HF_LLM_LEADERBOARD_TASKS

# Run evaluation

hostname

MODELS=$(ls $MODELS_PATH)

for model in ${MODELS[@]}
do
    echo $(date)
    echo "Evaluating $model..."

    # HF_HOME=/scratch0/aszablew/.huggingface CUDA_VISIBLE_DEVICES=2 nohup $LAUNCHER --num_processes=1 -m lm_eval --model hf \
    # --model_args pretrained=$MODELS_PATH/$model \
    # --tasks $TASKS \
    # --device "cuda" \
    # --batch_size 16 \
    # --trust_remote_code \
    # --output_path $RESULTS_PATH/$model/$model.json &> $LOGS_PATH/$model.log 
    # TODO: change this cache path, also removed CUDA_VISIBLE_DEVICES as this is gonna handle by the sechduler
    HF_HOME=/scratch0/aszablew/.huggingface nohup $LAUNCHER --num_processes=1 -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$model \
    --tasks $TASKS \
    --device "cuda" \
    # TODO: if it's a100, use 16, a40 probabaly need 8
    # --batch_size 16 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path $RESULTS_PATH/$model/$model.json &> $LOGS_PATH/$model.log 

done

cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/

date