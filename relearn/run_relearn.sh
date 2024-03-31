#!/bin/bash
[ -f /etc/network_turbo ] && source /etc/network_turbo
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/
[ -d $TRANSFORMERS_CACHE ] || mkdir -p $TRANSFORMERS_CACHE

batch_sizes=(2 8 32 128 512)

JSON_DIR=./relearn_json/
[ -d $JSON_DIR ] || mkdir -p $JSON_DIR
echo output JSON directory: $JSON_DIR

mkdir -p ~/autodl-tmp/models

for batch_size in "${batch_sizes[@]}"; do
    python ./relearn/relearn.py \
        --original_model facebook/opt-1.3b \
        --unlearned_model "ucl-snlp-nyt/snlp-unlearned-models/models/models_batch_unlearning/batch_size_$batch_size/idx_20" \
        --batch_size 4 \
        --lr 5e-3 \
        --json_dir $JSON_DIR \
        --dataset PKU-Alignment/PKU-SafeRLHF \
        --max_relearn_epochs 5 \
        --use_lora \
        --cache_dir ~/autodl-tmp/hf_cache \
        -v
    rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
done

sample_count=(128 512)
num_splits=(4 16 64)
for s in "${sample_count[@]}"; do
    for n in "${num_splits[@]}"; do
        python ./relearn/relearn.py \
            --original_model facebook/opt-1.3b \
            --unlearned_model ucl-snlp-nyt/snlp-unlearned-models/models/models_sequential_unlearning/samples_count_"$s"_split_"$n"/idx_20 \
            --batch_size 4 \
            --lr 5e-3 \
            --json_dir $JSON_DIR \
            --dataset PKU-Alignment/PKU-SafeRLHF \
            --max_relearn_epochs 5 \
            --use_lora \
            --cache_dir ~/autodl-tmp/hf_cache \
            -v
        rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
    done
done
