#! /bin/bash

EXPERIMENT_NAME=$1
MODEL_NAME=$2


# Accelerate used in a multi-gpu setting.
# I think it can be also used on a single gpu setting, so leaving as default.
LAUNCHER="accelerate launch"
# LAUNCHER="python3"

# Config
MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/opt1.3b_unlearned_mathqa
EXPERIMENT_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_PATH/results
LOGS_PATH=$EXPERIMENT_PATH/logs
CACHE_PATH=$EXPERIMENT_PATH/.cache_$MODEL_NAME

mkdir $EXPERIMENT_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Tasks selection
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa_mc2,mmlu,winogrande,gsm8k"
TASKS_ADDITIONAL="toxigen,realtoxicityprompts"
TASKS=$HF_LLM_LEADERBOARD_TASKS,$TASKS_ADDITIONAL

hostname
date

nohup $LAUNCHER -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$MODEL_NAME \
    --tasks $TASKS \
    --batch_size auto \
    --trust_remote_code True \
    --output_path $RESULTS_PATH/$MODEL_NAME/$MODEL_NAME.json \
    --wandb_args project=snlp-lm-eval-harness,name=$EXPERIMENT_NAME-$MODEL_NAME &> $LOGS_PATH/$MODEL_NAME.log 

    # --use_cache $CACHE_PATH 

date
