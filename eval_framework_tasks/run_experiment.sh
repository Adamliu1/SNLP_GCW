#! /bin/bash

# TODO: Add instructions how to use.

# Accelerate used in a multi-gpu setting. Python for single-gpu.
# LAUNCHER="accelerate launch"
LAUNCHER="python3"

# Experiment config
MODEL_NAME=opt1.3b_unlearned_harmful
EXPERIMENT_NAME=opt1.3b_unlearned_harmful-eval1


# Environment
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks
source $BASE_PATH/../venv/bin/activate
echo "WARNING: You need to log in using huggingface-cli and make sure you have access to the toxigen dataset!"
sleep 3

# Paths and directories
MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
EXPERIMENT_SCRATCH_PATH=/scratch0/aszablew/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
WANDB_PATH=$EXPERIMENT_SCRATCH_PATH/wandb

mkdir $EXPERIMENT_SCRATCH_PATH

mkdir $WANDB_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Tasks selection
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k"
TASKS_ADDITIONAL="toxigen"
TASKS=$HF_LLM_LEADERBOARD_TASKS,$TASKS_ADDITIONAL

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
    --device "cuda:0" \
    --batch_size auto \
    --trust_remote_code True \
    --output_path $RESULTS_PATH/$model/$model.json \
    --wandb_args project=snlp,name=$EXPERIMENT_NAME-$model,group=$EXPERIMENT_NAME,mode=offline,dir=$WANDB_PATH \
    --use_cache $EXPERIMENT_SCRATCH_PATH/.cache/$model/cache &> $LOGS_PATH/$model.log 

done

cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/

date