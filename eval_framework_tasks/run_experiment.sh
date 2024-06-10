#! /bin/bash

# TODO: Add instructions how to use.

# Accelerate used in a multi-gpu setting. Python for single-gpu.
# LAUNCHER="accelerate launch"
LAUNCHER="python3"

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

CUDA_DEVICE=$3

RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
WANDB_PATH=$EXPERIMENT_SCRATCH_PATH/wandb

mkdir $EXPERIMENT_SCRATCH_PATH

mkdir $WANDB_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Tasks selection
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k,french_bench,logiqa,piqa,squadv2"
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

    nohup $LAUNCHER -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$model \
    --tasks $TASKS \
    --device $CUDA_DEVICE \
    --batch_size auto \
    --trust_remote_code \
    --output_path $RESULTS_PATH/$model/$model.json \
    --wandb_args project=snlp,name=$EXPERIMENT_NAME-$model,group=$EXPERIMENT_NAME,mode=offline,dir=$WANDB_PATH \
    --use_cache $EXPERIMENT_SCRATCH_PATH/.cache/$model/cache &> $LOGS_PATH/$model.log 

done

cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/

date