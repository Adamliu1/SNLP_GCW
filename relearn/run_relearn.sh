#!/bin/bash
[ -f /etc/network_turbo ] && source /etc/network_turbo
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/
[ -d $TRANSFORMERS_CACHE ] || mkdir -p $TRANSFORMERS_CACHE

JSON_DIR=./relearn_json/
[ -d $JSON_DIR ] || mkdir -p $JSON_DIR
echo output JSON directory: $JSON_DIR

mkdir -p ~/autodl-tmp/models

batch_sizes=(2 8 32 128 512 1024 2048)
lr="1e-3"
# for batch_size in "${batch_sizes[@]}"; do
#     python ./relearn/relearn.py \
#         --original_model facebook/opt-1.3b \
#         --unlearned_model "ucl-snlp-nyt/snlp-unlearned-models/models/models_batch_unlearning/batch_size_$batch_size/idx_20" \
#         --batch_size 4 \
#         --lr $lr \
#         --json_dir $JSON_DIR \
#         --dataset PKU-Alignment/PKU-SafeRLHF \
#         --max_relearn_epochs 5 \
#         --use_lora \
#         --cache_dir ~/autodl-tmp/hf_cache \
#         -v
#     rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
# done
#
# sample_count=(128 512 1024)
# num_splits=(4 16 64)
# for s in "${sample_count[@]}"; do
#     for n in "${num_splits[@]}"; do
#         python ./relearn/relearn.py \
#             --original_model facebook/opt-1.3b \
#             --unlearned_model ucl-snlp-nyt/snlp-unlearned-models/models/models_sequential_unlearning/samples_count_"$s"_split_"$n"/idx_20 \
#             --batch_size 4 \
#             --lr $lr \
#             --json_dir $JSON_DIR \
#             --dataset PKU-Alignment/PKU-SafeRLHF \
#             --max_relearn_epochs 5 \
#             --use_lora \
#             --cache_dir ~/autodl-tmp/hf_cache \
#             -v
#         rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
#     done
# done
#
# seeds=(114514 1234 42 456 8888)
# for seed in "${seeds[@]}"; do
#     python ./relearn/relearn.py \
#         --original_model facebook/opt-1.3b \
#         --unlearned_model "ucl-snlp-nyt/snlp-unlearned-models/models/opt1.3b_unlearned_harmful-for-real$seed/idx_1000" \
#         --batch_size 4 \
#         --lr $lr \
#         --json_dir $JSON_DIR \
#         --dataset PKU-Alignment/PKU-SafeRLHF \
#         --max_relearn_epochs 5 \
#         --use_lora \
#         --cache_dir ~/autodl-tmp/hf_cache \
#         -v
#     rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
# done
models=(32_lr5.66e-06 128_lr1.132e-05 512_lr2.26e-05 1024_lr3.2e-05 2048_lr4.53e-05)
for model in "${models[@]}"; do
    python ./relearn/relearn.py \
        --original_model facebook/opt-1.3b \
        --unlearned_model "ucl-snlp-nyt/snlp-unlearned-models/models/models_batch_unlearning_scaled_lr/batch_size_$model/idx_20" \
        --batch_size 4 \
        --lr $lr \
        --json_dir $JSON_DIR \
        --dataset PKU-Alignment/PKU-SafeRLHF \
        --max_relearn_epochs 5 \
        --use_lora \
        --cache_dir ~/autodl-tmp/hf_cache \
        -v
    rm -rf ~/autodl-tmp/hf_cache/models--ucl-snlp-nyt--snlp-unlearned-models/
done
