#!/bin/bash

# Experiment config
MODEL_NAME=opt1.3b_unlearned_harmful-for-real42
EXPERIMENT_NAME=opt1.3b_unlearned_harmful-for-real42-eval1



# Paths and directories
MODELS_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$MODEL_NAME
BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_harmfulness
source $BASE_PATH/../venv/bin/activate

# Environment config
DEVICE="cuda:3"

# Generate model outputs for each question in the dataset
for i in  {0..1000..100}
do 
    nohup python3 beavertails_get_model_answers.py \
        --model_path $MODELS_PATH/idx_$i \
        --device $DEVICE \
        --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

done

# Evaluate all generations
nohup python evaluate_outputs.py \
    --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
    --model_path ./beaver-dam-7_weights \
    --max_length 512 \
    --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
    --device $DEVICE
    