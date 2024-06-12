#!/bin/bash
# Working with this script: Assumes access to 4 GPUS on scheduling node, uses CUDA_VISIBLE_DEVICES to schedule jobs.
# Need to set up: UNLEARNED_MODELS_PATH, MODEL_PATH, UNLEARN_DATASET_PATH and name. Make sure parameters for unlearning are correct.
# IMPORTANT: when modifying BATCH_SIZE, make sure MAX_UNLEARN_STEPS are the expected value
# IMPORTANT: LOOK FOR ALL REFERENCES OF SCRATCH AND SDUCHNIE!! CHANGE Accordingly

LR=5e-6
SAVE_EVERY_STEPS=256
SAVE_EVERY_STEPS_BATCH=1 # Save every epoch for batch (would be a multiple of 128 samples seen: e.g. 128 epoch 3 = 384 samples, 512 epoch 20 = 10240 samples)
BATCH_SIZE=2 # TODO: Continuos batch size set again later
SEED=2137
MAX_UNLEARN_STEPS=10240
# TODO: ENSURE UNLEARNED_MODELS_PATH IS CORRECT WHEN SCHEDULING
#UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models"
EXPERIMENT_NAME="MIGHTBEGOOD_LR$LR""_BS$BATCH_SIZE"
# TODO: EXCEPTION: UNLEARNED PATH FROM /SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr5e-6/models
UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr5e-6/"

PREVIOUS_MODELS_TO_EVAL="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr5e-6/models/continuous-10240-gemma-2b-harmful-squad/"
$model="continuous-10240-gemma-2b-harmful-squad"
#/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr5e-6/models
mkdir -p $UNLEARNED_MODELS_PATH/models
mkdir -p $UNLEARNED_MODELS_PATH/logs
WANDB_PROJ_NAME="snlp-lr$LR""_BS_$BATCH_SIZE"

# Model params
MODEL_NAME=gemma-2b
MODEL_PATH=google/gemma-2b
#MODEL_NAME=opt-1.3b
#MODEL_PATH=facebook/opt-1.3b
# Task params
UNLEARN_DATASET_NAME="harmful"
UNLEARN_DATASET_PATH="PKU-Alignment/PKU-SafeRLHF"
#UNLEARN_DATASET_NAME="french"
#UNLEARN_DATASET_PATH="AgentPublic/piaf"
RETAIN_DATASET_NAME="squad"
RETAIN_DATASET_PATH="rajpurkar/squad"

date
### EVALUATE CONTINUOUS
# AS batch size 2, we need steps at 0.5 the samples_count needed
continous_checkpoints=(256 512 768 1024 1280 2048 3072 4096 5120 6144 8192 10240)
echo "Now doing evals of continous"

# only sqeuentials
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,french_bench,mnli,piqa,squadv2"
EXPERIMENT_SCRATCH_PATH=/scratch0/sduchnie/$EXPERIMENT_NAME
RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
mkdir -p $RESULTS_PATH
mkdir -p $LOGS_PATH
mkdir -p $UNLEARNED_MODELS_PATH/experiment_data
GPU_NUM=0
PROCARR=()
for checkpoint in ${continous_checkpoints[@]}
do
    # TODO: if it's a100, use 16, a40 probabaly need 8
    echo "Scheduling on GPU $GPU_NUM"
    CUDA_VISIBLE_DEVICES=$GPU_NUM HF_HOME=/scratch0/sduchnie/.huggingface nohup python3 -m lm_eval --model hf \
        --model_args pretrained=$PREVIOUS_MODELS_TO_EVAL/idx_$checkpoint \
        --tasks $HF_LLM_LEADERBOARD_TASKS \
        --device "cuda" \
        --batch_size 16 \
        --trust_remote_code \
        --output_path $RESULTS_PATH/$model/idx_$checkpoint.json &> $LOGS_PATH/${model}_$checkpoint.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

# Run Harmfulness eval
echo "Generating model answers for Beavertails!"
for checkpoint in ${continous_checkpoints[@]}
do
    echo "Generating model answers for checkpoint $checkpoint on GPU $GPU_NUM..."
    # TODO: WARNING PATH RELATIVE TO llm_unlearn_ucl
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_harmfulness/beavertails_get_model_answers.py \
        --model_path $PREVIOUS_MODELS_TO_EVAL/idx_$checkpoint \
        --device "cuda" \
        --batch_size 64 \
        --output_dir "$UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model/idx_$checkpoint"  &> $LOGS_PATH/generations_${model}_$checkpoint.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

# Evaluate all generations
date
for checkpoint in ${continous_checkpoints[@]}
do
    # TODO: WARNING PATH RELATIVE TO llm_uonlearn_ucl
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python ../eval_harmfulness/evaluate_outputs.py \
        --eval_dataset "$UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model/idx_$checkpoint" \
        --model_path /SAN/intelsys/llm/sduchnie/models/beaver-dam-7b \
        --max_length 512 \
        --output_dir "$UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/idx_$checkpoint" \
        --device "cuda"  &> $LOGS_PATH/harmful_$model.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

echo "wait in case more stuff still scheduled but not all 4 gpus busy"
wait ${PROCARR[@]}
echo "Done waiting"

echo "Copy experiment results"
cp -r $EXPERIMENT_SCRATCH_PATH  $UNLEARNED_MODELS_PATH/experiment_data

echo "oof, thats a long one, if nothing went wrong, ur set"
date