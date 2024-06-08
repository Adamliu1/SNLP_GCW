#!/usr/bin/env zsh

#$ -S /usr/bin/zsh
#$ -l tmem=40G
#$ -l h_rt=12:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
# Use the seed in the job name for easy identification
#$ -N beavertails_llama3_run_seed
#$ -l hostname=dip-207-2

# Define a list of seeds
seeds=(42 123 456)

# Get the index from the job array
index=$(($SGE_TASK_ID))
seed=${seeds[$index]}

source /home/yadonliu/.zshrc
conda activate /home/yadonliu/conda_env/snlp

# Experiment config
MODEL_NAME=llama3

# Paths and directories
MODELS_PATH="meta-llama/Meta-Llama-3-8B"
BASE_PATH=/home/yadonliu/SNLP_GCW/eval_harmfulness
cd $BASE_PATH

# Environment config
DEVICE="cuda"
MODEL="llama3"

hostname

EXPERIMENT_NAME="${MODEL_NAME}_eval/seed_${seed}"
date
echo "Generating model answers for $MODEL, seed $seed..."
python3 beavertails_get_model_answers.py \
    --model_path $MODELS_PATH \
    --device $DEVICE \
    --batch_size 64 \
    --seed $seed \
    --output_dir "$BASE_PATH/model_generations/$EXPERIMENT_NAME"

echo "Evaluating model answers"
EXPERIMENT_NAME="${MODEL_NAME}_eval/seed_${seed}"
date
python evaluate_outputs.py \
    --eval_dataset "$BASE_PATH/model_generations/$EXPERIMENT_NAME/" \
    --model_path "PKU-Alignment/beaver-dam-7b" \
    --max_length 512 \
    --output_dir "$BASE_PATH/eval_results/$EXPERIMENT_NAME" \
    --device $DEVICE