#! /bin/bash

EXPERIMENT_NAME=$1
MODEL_NAME=$2

MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/opt1.3b_unlearned_mathqa
EXPERIMENT_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_PATH/results
LOGS_PATH=$EXPERIMENT_PATH/logs
CACHE_PATH=$EXPERIMENT_PATH/.cache

mkdir $EXPERIMENT_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

hostname
date

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODELS_PATH/$MODEL_NAME \
    --tasks arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k \
    --batch_size auto \
    --trust_remote_code True \
    --output_path $RESULTS_PATH/$MODEL_NAME.json \
    --wandb_args project=snlp-lm-eval-harness,name=$EXPERIMENT_NAME-$MODEL_NAME \
    --log_samples \
    --use_cache $CACHE_PATH&> $LOGS_PATH/$MODEL_NAME.log 
    
date
