# NOTE: final model_name would be MODELS_PATH/MODEL_NAME
MODEL_NAME=opt-1.3b
MODELS_PATH=facebook

# MODEL_NAME=gemma-2b
# MODELS_PATH=google

# MODEL_NAME=Meta-Llama-3.1-8B
# MODELS_PATH=meta-llama

CACHE_DIR=/scratch0/$USER/cache
BASE_PATH=/home/yadonliu/SNLP_GCW/llm_unlearn_ucl

# UNLEARN PARAMETERS
UNLEARNING_DATASET=PKU-Alignment/PKU-SafeRLHF
MAX_UNLEARN_STEPS=10240
NUM_EPOCHS=20
BATCH_SIZE=1
LR=2e-6

SEED=114514

MAX_BAD_LOSS=1000 # This is default from args in unlearn_harm.py
NUM_RUNNING_LOSS=10 # This is default from args in unlearn_harm.py
NO_SCHEDULER=false # We want to use scheduler

WANDB_LOG=false # We want to use wandb

# Run workspace, for saving model, logs, etc.
RUN_WS=/scratch0/$USER
mkdir -p $RUN_WS

# DATASET
retain_datasets=(truthful_qa "rajpurkar/squad")
retain_datasets_name=(truthful_qa squad)
# retain_datasets=(truthful_qa)
# retain_datasets_name=(truthful_qa)

# Sechdule continuous unlearning
SAVE_EVERY=100
for idx in "${!retain_datasets[@]}"; do
    retain_dataset=${retain_datasets[$idx]}
    dataset_name=${retain_datasets_name[$idx]}

    EXPERIMENT_NAME="${MODEL_NAME}-unlearn-continuous-${dataset_name}"
    EXPERIMENT_RUN_NAME="${MODEL_NAME}-unlearn-continuous-${dataset_name}-seed-${SEED}"
    NUM_SAMPLES=128 # PLACEHOLDER FOR THE SCIPRT, not taking effect in continuous_unlearn
    SEQUENTIAL=-1 # We running continuous unlearn
    qsub $BASE_PATH/scripts/unlearn/unlearn_script_opt1.3.sh $MODELS_PATH $MODEL_NAME $BATCH_SIZE $LR $MAX_UNLEARN_STEPS $retain_dataset $UNLEARNING_DATASET $RUN_WS $EXPERIMENT_NAME $EXPERIMENT_RUN_NAME $NUM_EPOCHS $NUM_SAMPLES $SEED $MAX_BAD_LOSS $SAVE_EVERY $NUM_RUNNING_LOSS $NO_SCHEDULER $WANDB_LOG $SEQUENTIAL $BASE_PATH $CACHE_DIR -N "$EXPERIMENT_RUN_NAME" 
    # $BASE_PATH/scripts/unlearn/unlearn_script_opt1.3.sh $MODELS_PATH $MODEL_NAME $BATCH_SIZE $LR $MAX_UNLEARN_STEPS $retain_dataset $UNLEARNING_DATASET $RUN_WS $EXPERIMENT_NAME $EXPERIMENT_RUN_NAME $NUM_EPOCHS $NUM_SAMPLES $SEED $MAX_BAD_LOSS $SAVE_EVERY $NUM_RUNNING_LOSS $NO_SCHEDULER $WANDB_LOG $SEQUENTIAL $BASE_PATH $CACHE_DIR -N "$EXPERIMENT_RUN_NAME" 
done

# Schedule Batch unlearning
SAVE_EVERY=4 #from previous wandb log
batches=(2 8 32 128 512 1024 2048)
for idx in "${!retain_datasets[@]}"; do
    retain_dataset=${retain_datasets[$idx]}
    dataset_name=${retain_datasets_name[$idx]}
    for j in ${batches[@]}; do
        EXPERIMENT_NAME="${MODEL_NAME}-unlearn-batch-${dataset_name}"
        EXPERIMENT_RUN_NAME="${MODEL_NAME}-unlearn-batch-${dataset_name}-batch-${j}-seed-${SEED}"
        NUM_SAMPLES=$j
        SEQUENTIAL=1 # We running batch unlearn, NOTE: this is a bit comfusing from the source code, but yeah, sequential size 1 is the same as batch unlearning
        qsub $BASE_PATH/scripts/unlearn/unlearn_script_opt1.3.sh $MODELS_PATH $MODEL_NAME $BATCH_SIZE $LR $MAX_UNLEARN_STEPS $retain_dataset $UNLEARNING_DATASET $RUN_WS $EXPERIMENT_NAME $EXPERIMENT_RUN_NAME $NUM_EPOCHS $NUM_SAMPLES $SEED $MAX_BAD_LOSS $SAVE_EVERY $NUM_RUNNING_LOSS $NO_SCHEDULER $WANDB_LOG $SEQUENTIAL $BASE_PATH $CACHE_DIR -N "$EXPERIMENT_RUN_NAME"
    done
done

# Schedule Sequential unlearning
sample_count=(128 512 1024)
splits=(4 16 64)
SAVE_EVERY=500 # NOTE: this is a placeholder, not taking effect in sequential unlearn
for idx in "${!retain_datasets[@]}"; do
    retain_dataset=${retain_datasets[$idx]}
    dataset_name=${retain_datasets_name[$idx]}
    for j in ${!sample_count[@]}; do
        for k in ${!splits[@]}; do
            EXPERIMENT_NAME="${MODEL_NAME}-unlearn-sequential-${dataset_name}"
            EXPERIMENT_RUN_NAME="${MODEL_NAME}-unlearn-sequential-${dataset_name}-sample-${sample_count[$j]}-split-${splits[$k]}-seed-${SEED}"
            NUM_SAMPLES=${sample_count[$j]}
            SEQUENTIAL=${splits[$k]}
            qsub $BASE_PATH/scripts/unlearn/unlearn_script_opt1.3.sh $MODELS_PATH $MODEL_NAME $BATCH_SIZE $LR $MAX_UNLEARN_STEPS $retain_dataset $UNLEARNING_DATASET $RUN_WS $EXPERIMENT_NAME $EXPERIMENT_RUN_NAME $NUM_EPOCHS $NUM_SAMPLES $SEED $MAX_BAD_LOSS $SAVE_EVERY $NUM_RUNNING_LOSS $NO_SCHEDULER $WANDB_LOG $SEQUENTIAL $BASE_PATH $CACHE_DIR -N "$EXPERIMENT_RUN_NAME"
        done
    done
done






