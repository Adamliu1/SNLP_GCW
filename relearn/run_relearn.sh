models=()
model_name=opt1.3b_unlearned_harmful-for-real1234
model_save_dir="../snlp-unlearned-models/models/$model_name"

readarray -t models < <(find \
    $model_save_dir \
    -type d \
    -name 'idx_*')

JSON_DIR=./relearn_jsons/$model_name
[ -d $JSON_DIR ] || mkdir -p $JSON_DIR
echo output JSON directory: $JSON_DIR

for model_path in "${models[@]}"; do
    echo $model_path
    python ./relearn/relearn.py \
        --original_model facebook/opt-1.3b \
        --unlearned_model $model_path \
        --batch_size 2 \
        --lr 5e-4 \
        --json_dir $JSON_DIR \
        --use_lora \
        --dataset PKU-Alignment/PKU-SafeRLHF \
        -v
done
