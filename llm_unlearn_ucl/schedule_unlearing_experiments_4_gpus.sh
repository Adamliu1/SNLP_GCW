#!/bin/bash
# Working with this script: Assumes access to 4 GPUS on scheduling node, uses CUDA_VISIBLE_DEVICES to schedule jobs.
# Need to set up: UNLEARNED_MODELS_PATH, MODEL_PATH, UNLEARN_DATASET_PATH and name. Make sure parameters for unlearning are correct.
# IMPORTANT: when modifying BATCH_SIZE, make sure MAX_UNLEARN_STEPS are the expected value
# IMPORTANT: LOOK FOR ALL REFERENCES OF SCRATCH AND SDUCHNIE!! CHANGE Accordingly

LR=5e-6
SAVE_EVERY_STEPS=256
SAVE_EVERY_STEPS_BATCH=1 # Save every epoch for batch (would be a multiple of 128 samples seen: e.g. 128 epoch 3 = 384 samples, 512 epoch 20 = 10240 samples)
BATCH_SIZE=2 # TODO: Continuos batch size set again later
SEED=2137
MAX_UNLEARN_STEPS=10240
# TODO: ENSURE UNLEARNED_MODELS_PATH IS CORRECT WHEN SCHEDULING
#UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models"
EXPERIMENT_NAME="MIGHTBEGOOD_LR$LR""_BS$BATCH_SIZE"
UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/$EXPERIMENT_NAME"
mkdir -p $UNLEARNED_MODELS_PATH/models
mkdir -p $UNLEARNED_MODELS_PATH/logs
WANDB_PROJ_NAME="snlp-lr$LR""_BS_$BATCH_SIZE"

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
sample_counts=(128 512 1024)
date
for sample_count in "${sample_counts[@]}"; do
    date
    PROCARR=()
    # Sequential Splits 4; CUDA:0
    RUN_NAME="seq-4-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
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
                        --lr $LR \
                        --wandb_project_name $WANDB_PROJ_NAME \
                        --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)

    # Sequential Splits 16; CUDA:1
    RUN_NAME="seq-16-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
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
                        --sequential 16 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr $LR \
                        --wandb_project_name $WANDB_PROJ_NAME \
                        --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)

    # Sequential Splits 64; CUDA:2
    RUN_NAME="seq-64-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
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
                        --sequential 64 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS \
                        --lr $LR \
                        --wandb_project_name $WANDB_PROJ_NAME \
                        --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
           
    # Batch; CUDA:3
    RUN_NAME="batch-$sample_count-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
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
                        --sequential 1 \
                        --num_epochs 20 \
                        --batch_size $BATCH_SIZE \
                        --save_every $SAVE_EVERY_STEPS_BATCH \
                        --wandb_project_name $WANDB_PROJ_NAME \
                        --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
    PROCARR+=($!)
    
    echo "Waiting for processes: ${PROCARR[@]}..."
    wait ${PROCARR[@]}
    echo "Done waiting."
    echo "Syncing wandb runs.."
    for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
    echo "Done syncing. Check wandb ;) "
done
date

echo "Now doing evals of sequential 1024: splits: 64,16, 4"

# only sqeuentials
HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,french_bench,mnli,piqa,squadv2"
EXPERIMENT_SCRATCH_PATH=/scratch0/sduchnie/$EXPERIMENT_NAME
RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
mkdir -p $RESULTS_PATH
mkdir -p $LOGS_PATH
mkdir -p $UNLEARNED_MODELS_PATH/experiment_data
MODELS=$(ls $UNLEARNED_MODELS_PATH/models | grep "seq")
GPU_NUM=0
PROCARR=()
for model in ${MODELS[@]}
do
    # TODO: if it's a100, use 16, a40 probabaly need 8
    echo "Scheduling on GPU $GPU_NUM"
    CUDA_VISIBLE_DEVICES=$GPU_NUM HF_HOME=/scratch0/sduchnie/.huggingface nohup python3 -m lm_eval --model hf \
        --model_args pretrained=$UNLEARNED_MODELS_PATH/models/$model/idx_20 \
        --tasks $HF_LLM_LEADERBOARD_TASKS \
        --device "cuda" \
        --batch_size 16 \
        --trust_remote_code \
        --output_path $RESULTS_PATH/$model/$model.json &> $LOGS_PATH/$model.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

# Run Harmfulness eval
echo "Generating model answers for Beavertails!"
for model in ${MODELS[@]}
do
    echo "Generating model answers for $model on GPU $GPU_NUM..."
    # TODO: WARNING PATH RELATIVE TO llm_uonlearn_ucl
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_harmfulness/beavertails_get_model_answers.py \
        --model_path $UNLEARNED_MODELS_PATH/models/$model/idx_20 \
        --device "cuda" \
        --batch_size 64 \
        --output_dir "$UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model"  &> $LOGS_PATH/generations_$model.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

# Evaluate all generations
date
for model in ${MODELS[@]}
do
    # TODO: WARNING PATH RELATIVE TO llm_uonlearn_ucl
    CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python ../eval_harmfulness/evaluate_outputs.py \
        --eval_dataset "$UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model" \
        --model_path /SAN/intelsys/llm/sduchnie/models/beaver-dam-7b \
        --max_length 512 \
        --output_dir "$UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model" \
        --device "cuda"  &> $LOGS_PATH/harmful_$model.log &
    PROCARR+=($!)
    GPU_NUM=$((GPU_NUM+1))
    if [ $GPU_NUM -eq 4 ]; then
        echo "All 4 GPUs scheduled, waiting for ${PROCARR[@]}.."
        GPU_NUM=0
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        date
    fi
done

echo "wait in case more stuff still scheduled but not all 4 gpus busy"
wait ${PROCARR[@]}
echo "Done waiting"

echo "Copy experiment results"
cp -r $EXPERIMENT_SCRATCH_PATH  $UNLEARNED_MODELS_PATH/experiment_data

echo "oof, thats a long one, if nothing went wrong, ur set"
# Bonus: cont unlearning
# Continuous unlearning; CUDA:0
BATCH_SIZE=2
SEED=2137
MAX_UNLEARN_STEPS=10240
RUN_NAME="continuous-$MAX_UNLEARN_STEPS-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
GPU_NUM=0
echo "Starting Continuous unlearning for $MAX_UNLEARN_STEPS steps"
CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 unlearn_harm.py \
                    --model_name $MODEL_PATH \
                    --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                    --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
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
                    --wandb_project_name $WANDB_PROJ_NAME \
                    --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
PROCARR=($!)
echo "Waiting for processes: ${PROCARR[@]}.."
wait ${PROCARR[@]}
echo "Done waiting."
echo "Syncing wandb runs.."
for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
echo "Done syncing. Check wandb ;) "
date