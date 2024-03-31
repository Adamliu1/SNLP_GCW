#! /bin/bash

# THIS IS SPECIFCI FOR A40 in CHINA
source /etc/network_turbo

# Accelerate used in a multi-gpu setting. Python for single-gpu.
# LAUNCHER="accelerate launch"
LAUNCHER="python3"

# Paths and directories

BASE_PATH=/root/autodl-tmp/SNLP_GCW/eval_framework_tasks
EXPERIMENT_NAME=batch_unlearn_8
EXPERIMENT_SCRATCH_PATH=$BASE_PATH/scratch_2/$EXPERIMENT_NAME

RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
WANDB_PATH=$EXPERIMENT_SCRATCH_PATH/wandb

mkdir -p $EXPERIMENT_SCRATCH_PATH

mkdir $WANDB_PATH
mkdir $RESULTS_PATH
mkdir $LOGS_PATH

# Environment
source $BASE_PATH/../venv/bin/activate
echo "WARNING: You need to log in using huggingface-cli and make sure you have access to the toxigen dataset!"
sleep 3


# Tasks selection
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k"
TASKS_ADDITIONAL="toxigen"
TASKS=$HF_LLM_LEADERBOARD_TASKS,$TASKS_ADDITIONAL

# model eval selection
model_batch_sizes=(8)
#model_batch_sizes=(2 8 32 128)

# Run evaluation

hostname

for batch_size in "${model_batch_sizes[@]}"
do
	# Experiment config
	EXPERIMENT_NAME=batch_unlearn_size_$batch_size

	echo "Evaluating models for batch size $batch_size.."
	MODELS_PATH=/root/autodl-tmp/model_tmp_cache/batch_size_$batch_size
	MODELS=$(ls $MODELS_PATH)
	# mkdir for results folder
	mkdir -p $RESULTS_PATH/batch_size_$batch_size
	for model in ${MODELS[@]}
	do
	    echo $(date)
	    echo "Evaluating $model..."

	    nohup $LAUNCHER -m lm_eval --model hf \
	    --model_args pretrained=$MODELS_PATH/$model \
	    --tasks $TASKS \
	    --device "cuda:0" \
	    --batch_size auto \
	    --trust_remote_code \
	    --output_path $RESULTS_PATH/batch_size_$batch_size/$model/$model.json \
	    --wandb_args project=snlp_batch_unlearn_eval,name=$EXPERIMENT_NAME-$model,group=$EXPERIMENT_NAME,dir=$WANDB_PATH \
	    --use_cache $EXPERIMENT_SCRATCH_PATH/.cache/$model/cache &> $LOGS_PATH/$model.log 

	done

done


cp -r $EXPERIMENT_SCRATCH_PATH $BASE_PATH/experiment_data/
date
