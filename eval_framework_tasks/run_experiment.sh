#! /bin/bash

# TODO: Add instructions how to use.

# Accelerate used in a multi-gpu setting. Python for single-gpu.
LAUNCHER="accelerate launch"
# LAUNCHER="python3"

# Experiment config
# MODEL_NAME=batch-2048-lr4.53e-05
MODEL_NAME=$1
EXPERIMENT_NAME=$MODEL_NAME-eval1

# Environment
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

# Paths and directories
# MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
MODELS_PATH=$2/$1
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

MODELS=$(ls $MODELS_PATH)

for model in ${MODELS[@]}
do
    echo $(date)
    echo "Evaluating $model..."

    HF_HOME=/scratch0/aszablew/.huggingface CUDA_VISIBLE_DEVICES=$3 nohup $LAUNCHER --num_processes=1 -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$model \
    --tasks $TASKS \
    --device "cuda" \
    --batch_size "auto" \
    --trust_remote_code \
    --output_path $RESULTS_PATH/$model/$model.json &> $LOGS_PATH/$model.log 

done

cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/

date
