#!/bin/bash

# Experiment config
MODEL_NAME=llama-3-8b
EXPERIMENT_NAME=$MODEL_NAME-eval2



# Paths and directories
MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_truthfulQA
source $BASE_PATH/../venv/bin/activate

# Environment config
DEVICE="cuda:3"

# Generate model outputs for each question in the dataset
MODELS=$(ls $MODELS_PATH)

hostname
date

for model in ${MODELS[@]}
do 
    date
    echo "Generating model answers for $model..."
    nohup python3 generate_outputs.py \
        --model_path $MODELS_PATH/$model \
        --device $DEVICE \
        --batch_size 128 \
        --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

done

# Evaluate all generations
date
nohup python evaluate_outputs.py \
    --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
    --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
    
date