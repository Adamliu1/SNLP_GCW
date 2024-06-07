#!/bin/bash

# Experiment config
MODEL_NAME=llama3

# Paths and directories
MODELS_PATH="meta-llama/Meta-Llama-3-8B"
BASE_PATH=/cs/student/ug/2020/yadonliu/my_fyp_space/SNLP_GCW/eval_harmfulness

# Environment config
DEVICE="cuda:0"
MODEL="llama3"

seeds=(42 123 456 114514 8888)

hostname

for seed in ${seeds[@]}
do 
    EXPERIMENT_NAME="${MODEL_NAME}_eval/seed_${seed}"
    date
    echo "Generating model answers for $MODEL..."
    python3 beavertails_get_model_answers.py \
        --model_path $MODELS_PATH \
        --device $DEVICE \
        --batch_size 8 \
        --seed $seed \
        --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

done

echo "Evaluating model answers"
for seed in ${seeds[@]}
do
    # Evaluate all generations
    EXPERIMENT_NAME="${MODEL_NAME}_eval/seed_${seed}"
    date
    python evaluate_outputs.py \
        --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
        --model_path "PKU-Alignment/beaver-dam-7b" \
        --max_length 512 \
        --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
        --device $DEVICE
done
