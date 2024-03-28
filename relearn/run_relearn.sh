#!/bin/bash
[ -f /etc/network_turbo ] && source /etc/network_turbo
models=()
model_name=opt1.3b_unlearned_harmful_batch_unlearning_512
model_save_dir="$HOME/autodl-tmp/snlp-unlearned-models/models/$model_name"

readarray -t models < <(find \
    $model_save_dir \
    -type d \
    -name 'idx_*')

JSON_DIR=./relearn_jsons/
[ -d $JSON_DIR ] || mkdir -p $JSON_DIR
echo output JSON directory: $JSON_DIR

for model_path in "${models[@]}"; do
    echo $model_path
    python ./relearn/relearn.py \
        --original_model facebook/opt-1.3b \
        --unlearned_model $model_path \
        --batch_size 4 \
        --lr 5e-3 \
        --json_dir $JSON_DIR \
        --dataset PKU-Alignment/PKU-SafeRLHF \
        --use_lora \
        -v
done
