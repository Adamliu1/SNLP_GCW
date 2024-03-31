#!/bin/bash
source /etc/network_turbo
# Experiment config
MODEL_NAME=opt1.3b_unlearned_harmful_batch_unlearning_8
EXPERIMENT_NAME=$MODEL_NAME-eval

# Paths and directories
MODELS_PATH=/root/autodl-tmp/model_tmp_cache/batch_size_8
BASE_PATH=/root/autodl-tmp/SNLP_GCW/eval_harmfulness
source $BASE_PATH/../venv/bin/activate

# Environment config
DEVICE="cuda:0"

# Generate model outputs for each question in the dataset
MODELS=$(ls $MODELS_PATH)

mkdir -p $BASE_PATH/eval_results/$EXPERIMENT_NAME
hostname
date

for model in ${MODELS[@]}; do
    date
    echo "Generating model answers for $model..."
    python3 beavertails_get_model_answers.py \
        --model_path $MODELS_PATH/$model \
        --device $DEVICE \
        --batch_size 16 \
        --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

done

# Evaluate all generations
date
python evaluate_outputs.py \
    --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
    --model_path ~/autodl-tmp/beaver-dam-7b/ \
    --max_length 512 \
    --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
    --device $DEVICE

date
