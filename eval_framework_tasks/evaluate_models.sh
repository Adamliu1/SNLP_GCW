#! /bin/bash

EXPERIMENT_NAME=$1
MODEL_NAME=$2

MODELS_PATH=/cs/student/projects1/2020/aszablew/SNLP/SNLP_GCW/llm_unlearn_ucl/snlp-unlearned-models/models
EXPERIMENT_PATH=/cs/student/projects1/2020/aszablew/SNLP/SNLP_GCW/eval_framework_tasks/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_PATH/results
LOGS_PATH=$EXPERIMENT_PATH/logs

mkdir $EXPERIMENT_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

hostname
date

python3 -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$MODEL_NAME \
    --tasks arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k \
    --device cuda \
    --batch_size auto \
    --trust_remote_code True \
    --output_path $RESULTS_PATH/$MODEL_NAME.json \
    --wandb_args project=snlp-lm-eval-harness,name=$EXPERIMENT_NAME-$MODEL_NAME \
    --log_samples \
    --use_cache .cache-$MODEL_NAME &> $LOGS_PATH/$MODEL_NAME.log 

date
