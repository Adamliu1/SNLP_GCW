#!/bin/bash

LR=2e-6
SAVE_EVERY_STEPS=128
BATCH_SIZE=2
SEED=2137
MAX_UNLEARN_STEPS=10240

# Model params
MODEL_NAME=gemma-2b
MODEL_PATH=google/gemma-2b
# Task params
UNLEARN_DATASET_NAME="harmful"
UNLEARN_DATASET_PATH="PKU-Alignment/PKU-SafeRLHF"
RETAIN_DATASET_NAME="squad"
RETAIN_DATASET_PATH="rajpurkar/squad"

##### Sequential unlearning + Batch #####
# Sizes: 128, 512, 1024
# Splits: 4, 16, 64
sample_counts=(128 512 1024)

for sample_count in "${sample_counts[@]}"; do
    PROCARR=()
    # Sequential Splits 4; CUDA:0
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
    GPU_NUM=0
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$RUN_NAME" \
                        --log_file "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/$RUN_NAME.log" \
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
                        --lr $LR \
                        --no_scheduler &> /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)

    # Sequential Splits 16; CUDA:1
    RUN_NAME="seq-16-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
    GPU_NUM=1
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$RUN_NAME" \
                        --log_file "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 16 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr $LR \
                        --no_scheduler &> /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)

    # Sequential Splits 64; CUDA:2
    RUN_NAME="seq-64-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
    GPU_NUM=2
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$RUN_NAME" \
                        --log_file "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 64 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr $LR \
                        --no_scheduler &> /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
           
    # Batch; CUDA:3
    RUN_NAME="batch-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
    GPU_NUM=3
    echo "Starting $RUN_NAME on GPU $GPU_NUM.."
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                        --model_name $MODEL_PATH \
                        --model_save_dir "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$RUN_NAME" \
                        --log_file "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/$RUN_NAME.log" \
                        --cache_dir ".cache" \
                        --seed $SEED \
                        --retaining_dataset $RETAIN_DATASET_PATH \
                        --unlearning_dataset $UNLEARN_DATASET_PATH \
                        --max_bad_loss 10000 \
                        --samples_count $sample_count \
                        --sequential 1 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr $LR \
                        --no_scheduler &> /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
    
    echo "Waiting for processes: $PROCARR..."
    wait ${PROCARR[@]}
    echo "Done waiting."
done

### Continuous unlearning; CUDA:0
RUN_NAME="continuous-$MAX_UNLEARN_STEPS-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
GPU_NUM=0
echo "Starting Continuous unlearning for $MAX_UNLEARN_STEPS steps"
CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                    --model_name $MODEL_PATH \
                    --model_save_dir "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/$RUN_NAME" \
                    --log_file "/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/$RUN_NAME.log" \
                    --max_unlearn_steps $MAX_UNLEARN_STEPS \
                    --cache_dir ".cache" \
                    --seed $SEED \
                    --retaining_dataset $RETAIN_DATASET_PATH \
                    --unlearning_dataset $UNLEARN_DATASET_PATH \
                    --max_bad_loss 10000 \
                    --sequential -1 \
                    --batch_size $BATCH_SIZE \
                    --save_every $SAVE_EVERY_STEPS \
                    --lr $LR \
                    --no_scheduler &> /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/logs/stdout_$RUN_NAME.log &
PROCARR=($!)
echo "Waiting for processes: $PROCARR..."
wait ${PROCARR[@]}
echo "Done waiting."