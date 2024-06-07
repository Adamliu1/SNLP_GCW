#!/bin/bash

# Experiment config
MODEL_NAME=test-llama3
# NOTE: here I will modify to run with different seeds
EXPERIMENT_NAME=$MODEL_NAME-test



# Paths and directories
# MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
MODELS_PATH="meta-llama/Meta-Llama-3-8B"
BASE_PATH=/cs/student/ug/2020/yadonliu/my_fyp_space/SNLP_GCW/eval_harmfulness
# source $BASE_PATH/../venv/bin/activate

# Environment config
DEVICE="cuda:0"

# Generate model outputs for each question in the dataset
# MODELS=$(ls $MODELS_PATH)

hostname

# for model in ${MODELS[@]}
# do 
    date
    echo "Generating model answers for $model..."
    python3 beavertails_get_model_answers.py \
        --model_path $MODELS_PATH \
        --device $DEVICE \
        --batch_size 64 \
        --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

# done

# Evaluate all generations
date
nohup python evaluate_outputs.py \
    --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
    --model_path "PKU-Alignment/beaver-dam-7b" \
    --max_length 512 \
    --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
    --device $DEVICE
    
# date