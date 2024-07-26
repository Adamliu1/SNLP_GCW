#!/bin/bash
# Working with this script: Assumes access to 4 GPUS on scheduling node, uses CUDA_VISIBLE_DEVICES to schedule jobs.
# Need to set up: UNLEARNED_MODELS_PATH, MODEL_PATH, UNLEARN_DATASET_PATH and name. Make sure parameters for unlearning are correct.
# IMPORTANT: when modifying BATCH_SIZE, make sure MAX_UNLEARN_STEPS are the expected value
# Assumes: venv with deepspeed and accelerate along with other reqs activated!

DEEPSPEED_CONFIG="/SAN/intelsys/llm/sduchnie/SNLP_GCW/llm_unlearn_ucl/deepspeed_config.yaml"
LAUNCHER="accelerate launch --config_file $DEEPSPEED_CONFIG"
SAVE_EVERY_STEPS=10240 # only saves final model
BATCH_SIZE=2
SEED=2137
MAX_UNLEARN_STEPS=10240
# TODO: ENSURE UNLEARNED_MODELS_PATH IS CORRECT WHEN SCHEDULING
#UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models"
UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/lr_find_cont"
mkdir -p $UNLEARNED_MODELS_PATH/models
mkdir -p $UNLEARNED_MODELS_PATH/logs
WANDB_PROJ_NAME="llama-3-8B-CONTINUOUS_LR_FINDING_20480_SAMPLES"

# Model params
#MODEL_NAME=gemma-2b
#MODEL_PATH=google/gemma-2b
#MODEL_NAME=opt-1.3b
#MODEL_PATH=facebook/opt-1.3b
MODEL_NAME=llama-3-8B
MODEL_PATH=meta-llama/Meta-Llama-3-8B
# Task params
UNLEARN_DATASET_NAME="harmful"
UNLEARN_DATASET_PATH="PKU-Alignment/PKU-SafeRLHF"
#UNLEARN_DATASET_NAME="french"
#UNLEARN_DATASET_PATH="AgentPublic/piaf"
RETAIN_DATASET_NAME="squad"
RETAIN_DATASET_PATH="rajpurkar/squad"

##### Continuous unlearning #####
scheduler=("true" "false")
lrs_pows=("1e-8" "5e-8" "1e-7" "5e-7" "1e-6" "5e-6")
date
GPU_NUM=0
PROCARR=()
DEEPSPEED_PORT=60100
for sched in "${scheduler[@]}"; do
    echo "Running experiments. Using scheduler: $scheduler"
    for lr in "${lrs_pows[@]}"; do
        date
        RUN_NAME="cont-bs-$BATCH_SIZE-$MAX_UNLEARN_STEPS-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME-lr-$lr-scheduler-$sched"
        WANDB_RUN_NAME="$lr-scheduler-$sched"
        echo "Starting $RUN_NAME on GPU $GPU_NUM.."
        if [ $sched == "true" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_NUM nohup $LAUNCHER --main_process_port $DEEPSPEED_PORT unlearn_harm.py \
                                --model_name $MODEL_PATH \
                                --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                                --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                                --cache_dir ".cache" \
                                --seed $SEED \
                                --retaining_dataset $RETAIN_DATASET_PATH \
                                --unlearning_dataset $UNLEARN_DATASET_PATH \
                                --max_bad_loss 10000 \
                                --max_unlearn_steps $MAX_UNLEARN_STEPS \
                                --sequential -1 \
                                --batch_size $BATCH_SIZE \
                                --save_every $SAVE_EVERY_STEPS \
                                --lr $lr \
                                --wandb_project_name $WANDB_PROJ_NAME \
                                --wandb_run_name $WANDB_RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
        else
            CUDA_VISIBLE_DEVICES=$GPU_NUM nohup $LAUNCHER --main_process_port $DEEPSPEED_PORT unlearn_harm.py \
                                --model_name $MODEL_PATH \
                                --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                                --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                                --cache_dir ".cache" \
                                --seed $SEED \
                                --retaining_dataset $RETAIN_DATASET_PATH \
                                --unlearning_dataset $UNLEARN_DATASET_PATH \
                                --max_bad_loss 10000 \
                                --max_unlearn_steps $MAX_UNLEARN_STEPS \
                                --sequential -1 \
                                --batch_size $BATCH_SIZE \
                                --save_every $SAVE_EVERY_STEPS \
                                --lr $lr \
                                --wandb_project_name $WANDB_PROJ_NAME \
                                --wandb_run_name $WANDB_RUN_NAME \
                                --no_scheduler &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
        fi
        PROCARR+=($!)
        GPU_NUM=$((GPU_NUM+1))
        DEEPSPEED_PORT=$((DEEPSPEED_PORT+100))
        if [ $GPU_NUM -eq 3 ]; then
            echo "3/4 GPUs scheduled (memory usage should be ~75%, cant use all 4!), waiting for ${PROCARR[@]}.."
            GPU_NUM=0
            DEEPSPEED_PORT=60100
            wait ${PROCARR[@]}
            echo "Done waiting"
            PROCARR=()
            echo "Syncing wandb runs.."
            for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
            echo "Done syncing. Check wandb ;) "
            date
        fi
    done
done

echo "Waiting for processes: ${PROCARR[@]}..."
wait ${PROCARR[@]}
echo "Done waiting."
echo "Syncing wandb runs.."
for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
echo "Done syncing. Check wandb ;) "

date