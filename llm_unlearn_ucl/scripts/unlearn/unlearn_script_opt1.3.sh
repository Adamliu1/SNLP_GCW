#!/bin/bash
# NOTE: using zsh instead of bash
#$ -l tmem=48G
#$ -l h_rt=12:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -S /bin/bash
#$ -l hostname=dip-207-2

# NOTE: modify this to your environment, this is for RTX-A600s node
source /share/apps/source_files/conda.source
source /share/apps/source_files/python/python-3.11.9.source
source /home/yadonliu/snlp_venv_py311_9_a6000/bin/activate
conda activate /home/yadonliu/conda_env/snlp
source /share/apps/source_files/gcc-9.2.0.source

# Receiving parameters from the command line
MODELS_PATH=$1
MODEL_NAME=$2
BATCH_SIZE=$3
LR=$4
MAX_UNLEARN_STEPS=$5
RETAINING_DATASET=$6
UNLEARNING_DATASET=$7
RUN_WS=$8
EXPERIMENT_NAME=$9
EXPERIMENT_RUN_NAME=${10}
NUM_EPOCHS=${11}
NUM_SAMPLES=${12}
SEED=${13}
MAX_BAD_LOSS=${14}
SAVE_EVERY=${15}
NUM_RUNNING_LOSS=${16}
NO_SCHEDULER=${17} # This will be either "true" or "false"
WANDB_LOG=${18} # This will be either "true" or "false"
SEQUENTIAL=${19} 
BASE_PATH=${20}
CACHE_DIR=${21}

# MODEL_PRECISION="fp32"
MODEL_PRECISION="bf16"
WANDB_PROJECT_NAME=$EXPERIMENT_NAME
WANDB_RUN_NAME=$EXPERIMENT_RUN_NAME

hostname
date
echo "Using Python from $(which python)"
echo "base path is $BASE_PATH"

mkdir -p $RUN_WS/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
mkdir -p $RUN_WS/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
mkdir -p $RUN_WS/saved_samples
# define parameters
log_ws=$RUN_WS/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
model_save_ws=$RUN_WS/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
data_sample_save_ws=$RUN_WS/saved_samples

echo "Running"
echo "Model Name: $MODEL_NAME"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Run Name: $EXPERIMENT_RUN_NAME"
echo "Seed: $SEED"

SCHEDULER_FLAG=""
if [ "$NO_SCHEDULER" = "true" ]; then
    SCHEDULER_FLAG="--no_scheduler"
fi

WANDB_LOG_FLAG=""
if [ "$WANDB_LOG" = "true" ]; then
    WANDB_LOG_FLAG="--wandb_log"
    wandb offline
else
    wandb disabled
fi

accelerate launch $BASE_PATH/unlearn_harm.py \
    --model_name $MODELS_PATH/$MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_unlearn_steps $MAX_UNLEARN_STEPS \
    --retaining_dataset $RETAINING_DATASET \
    --unlearning_dataset $UNLEARNING_DATASET \
    --log_file $log_ws/default.log \
    --cache_dir $CACHE_DIR \
    --model_save_dir $model_save_ws \
    --num_epochs $NUM_EPOCHS \
    --samples_count $NUM_SAMPLES \
    --seed $SEED \
    --max_bad_loss $MAX_BAD_LOSS \
    --save_every $SAVE_EVERY \
    --num_running_loss $NUM_RUNNING_LOSS \
    --sequential $SEQUENTIAL \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --wandb_run_name $WANDB_RUN_NAME \
    --precision $MODEL_PRECISION \
    --samples_save_dir $data_sample_save_ws \
    $SCHEDULER_FLAG \
    $WANDB_LOG_FLAG

echo "Moving stuff from tmp to final location"

NEW_DESTINATION=/SAN/intelsys/llm/yadonliu/llm_unlearned

mkdir -p $NEW_DESTINATION/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
mkdir -p $NEW_DESTINATION/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
# Copy the content
cp -r $RUN_WS/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME $NEW_DESTINATION/log/$EXPERIMENT_NAME/
cp -r $RUN_WS/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME $NEW_DESTINATION/model_save/unlearned/$EXPERIMENT_NAME/
# Optionally verify the copy
echo "Copied contents to $NEW_DESTINATION"
ls -l $NEW_DESTINATION/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
ls -l $NEW_DESTINATION/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
# TODO: add stuff in data_sample_save_ws, saved_samples to the final location

echo "REMOVE TEMPORARY FILES"
rm -rf $RUN_WS/log/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME
rm -rf $RUN_WS/model_save/unlearned/$EXPERIMENT_NAME/$EXPERIMENT_RUN_NAME

echo "$EXPERIMENT_RUN_NAME - DONE"

date





