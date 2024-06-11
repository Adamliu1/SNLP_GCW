#!/bin/bash
# Working with this script: Assumes access to 4 GPUS on scheduling node, uses CUDA_VISIBLE_DEVICES to schedule jobs.
# Need to set up: UNLEARNED_MODELS_PATH, MODEL_PATH, UNLEARN_DATASET_PATH and name. Make sure parameters for unlearning are correct.
# IMPORTANT: when modifying BATCH_SIZE, make sure MAX_UNLEARN_STEPS are the expected value

LR=2e-6
SAVE_EVERY_STEPS=128
SAVE_EVERY_STEPS_BATCH=1 # Save every epoch for batch (would be a multiple of 128 samples seen: e.g. 128 epoch 3 = 384 samples, 512 epoch 20 = 10240 samples)
BATCH_SIZE=2
SEED=2137
MAX_UNLEARN_STEPS=10240
# TODO: ENSURE UNLEARNED_MODELS_PATH IS CORRECT WHEN SCHEDULING
#UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models"
UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr_find"
mkdir -p $UNLEARNED_MODELS_PATH/models
mkdir -p $UNLEARNED_MODELS_PATH/logs

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

##### Sequential unlearning + Batch #####
# Sizes: 128, 512, 1024
# Splits: 4, 16, 64
sample_count=128
#lrs=(2 5 7 9)
lrs_pows=("e-6" "e-5" "e-4")
date
for lr in "${lrs_pows[@]}"; do
    date
    PROCARR=()
    # Sequential Splits 4; CUDA:0
    LR_MAIN=2
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME-lr-""$LR_MAIN""$lr"
    GPU_NUM=0
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                        --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 4 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr "$LR_MAIN""$lr" \
                        --no_scheduler &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)

    # Sequential Splits 16; CUDA:1
    LR_MAIN=5
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME-lr-""$LR_MAIN""$lr"
    GPU_NUM=1
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                        --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 4 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr "$LR_MAIN""$lr" \
                        --no_scheduler &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &

    PROCARR+=($!)

    # Sequential Splits 64; CUDA:2
    LR_MAIN=7
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME-lr-""$LR_MAIN""$lr"
    GPU_NUM=2
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                        --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 4 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr "$LR_MAIN""$lr" \
                        --no_scheduler &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
           
    # Batch; CUDA:3
    LR_MAIN=9
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME-lr-""$LR_MAIN""$lr"
    GPU_NUM=3
    echo "Starting $RUN_NAME on GPU $GPU_NUM, saving every $SAVE_EVERY_STEPS_BATCH epoch."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                        --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 4 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr "$LR_MAIN""$lr" \
                        --no_scheduler &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
    
    echo "Waiting for processes: ${PROCARR[@]}..."
    wait ${PROCARR[@]}
    echo "Done waiting."

done

echo "Done waiting."
date