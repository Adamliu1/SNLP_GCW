#!/bin/bash
# Working with this script: Assumes access to 4 GPUS on scheduling node, uses CUDA_VISIBLE_DEVICES to schedule jobs.
# Need to set up: UNLEARNED_MODELS_PATH, MODEL_PATH, UNLEARN_DATASET_PATH and name. Make sure parameters for unlearning are correct.
# IMPORTANT: when modifying BATCH_SIZE, make sure MAX_UNLEARN_STEPS are the expected value
# IMPORTANT: LOOK FOR ALL REFERENCES OF SCRATCH AND SDUCHNIE!! CHANGE Accordingly
# WARNING: for some reason, both python builds in /share/apps/ on strangeporks is built without `sqlite-devel`
# meaning lm_eval will fail. For that reason I have LMEVAL_PYTHON_PATH which points to a venv of the default python3
# install with lm_eval (which should also install transformers) and matplotlib packages

LAUNCHER="accelerate launch"
LMEVAL_PYTHON_PATH="/home/sduchnie/venv_eval_harness/bin/python3"
BEAVERDAM_WEIGHTS_PATH="/SAN/intelsys/llm/sduchnie/models/beaver-dam-7b"
LR=1e-6
epochs=(60 40 20)
SAVE_EVERY_STEPS=256
SAVE_EVERY_STEPS_BATCH=1 # Save every epoch for batch (would be a multiple of 128 samples seen: e.g. 128 epoch 3 = 384 samples, 512 epoch 20 = 10240 samples)
BATCH_SIZE=1
SEED=2137
MAX_UNLEARN_STEPS=10240
# TODO: ENSURE UNLEARNED_MODELS_PATH IS CORRECT WHEN SCHEDULING
#UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models"
EXPERIMENT_NAME="llama3_sequential_$LR""_BS$BATCH_SIZE""_EPOCHS$epochs"
UNLEARNED_MODELS_PATH="/SAN/intelsys/llm/sduchnie/snlp/SNLP_GCW/snlp-unlearned-models/$EXPERIMENT_NAME"
mkdir -p $UNLEARNED_MODELS_PATH/models
mkdir -p $UNLEARNED_MODELS_PATH/logs
WANDB_PROJ_NAME=$EXPERIMENT_NAME

# Model params
#MODEL_NAME=gemma-2b
#MODEL_PATH=google/gemma-2b
MODEL_NAME=llama-3-8B
MODEL_PATH=meta-llama/Meta-Llama-3-8B
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
# Splits: 4, 16, 64, 1 (where 1 is batch)
#sample_counts=(128 512 1024)
sample_counts=(128 512 1024)
splits=(4 16 64)
date
GPU_NUM=0,1,2,3
DEEPSPEED_PORT=60000
for epoch in "${epochs[@]}"; do
    echo "Beginning unlearning runs for lr $LR and epoch count $epoch"
    for sample_count in "${sample_counts[@]}"; do
        date
        PROCARR=()
        # TODO: Quick fix below, since accidentaly messed up grepping on first run and only did evals of 1/3 models
        for split in "${splits[@]}"; do
            GPU_NUM=0,1,2,3
            DEEPSPEED_PORT=60000
            # Sequential Splits $split; If $split == 1 => batch
            RUN_NAME="seq-lr-$LR-$split-$sample_count-$epoch-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
            if [ $split -eq 1 ]; then
                RUN_NAME="batch-$sample_count-$epoch-$MODEL_NAME-$UNLEARN_DATASET_NAME-$RETAIN_DATASET_NAME"
            fi
            echo "Starting $RUN_NAME on GPU $GPU_NUM.."
            CUDA_VISIBLE_DEVICES=$GPU_NUM nohup $LAUNCHER --main_process_port $DEEPSPEED_PORT unlearn_harm.py \
                                --model_name $MODEL_PATH \
                                --model_save_dir "$UNLEARNED_MODELS_PATH/models/$RUN_NAME" \
                                --log_file "$UNLEARNED_MODELS_PATH/logs/$RUN_NAME.log" \
                                --cache_dir ".cache" \
                                --seed $SEED \
                                --retaining_dataset $RETAIN_DATASET_PATH \
                                --unlearning_dataset $UNLEARN_DATASET_PATH \
                                --max_bad_loss 10000 \
                                --samples_count $sample_count \
                                --sequential $split \
                                --num_epochs $epoch \
                                --batch_size $BATCH_SIZE \
                                --save_every $SAVE_EVERY_STEPS \
                                --lr $LR \
                                --wandb_project_name $WANDB_PROJ_NAME \
                                --wandb_run_name $RUN_NAME &> $UNLEARNED_MODELS_PATH/logs/stdout_$RUN_NAME.log &
            PROCARR+=($!)

            echo "Unlearning in progress, waiting for ${PROCARR[@]}.."
            wait ${PROCARR[@]}
            echo "Done waiting"
            PROCARR=()
            # TODO: REMOVE BELOW IF NOT USING DEEPSPEED ZERO3
            echo "Removing the unwanted empty model.safetensors..."
            rm $UNLEARNED_MODELS_PATH/models/$RUN_NAME/idx_$epoch/model.safetensors
            echo "Syncing wandb runs.."
            for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
            echo "Done syncing. Check wandb ;) "
            date
        done
        echo "Waiting in case runs not finished..."
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()

        echo "Now evaluate the runs that just finished spinning..."
        HF_LLM_LEADERBOARD_TASKS="arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,french_bench,mnli,piqa,squadv2,toxigen,gsm8k"
        EXPERIMENT_SCRATCH_PATH=/scratch0/sduchnie/$EXPERIMENT_NAME
        RESULTS_PATH=$EXPERIMENT_SCRATCH_PATH/results
        LOGS_PATH=$EXPERIMENT_SCRATCH_PATH/logs
        mkdir -p $RESULTS_PATH
        mkdir -p $LOGS_PATH
        mkdir -p $UNLEARNED_MODELS_PATH/experiment_data
        # TODO: IMPORTANT: GREPPING BELOW - evaluating the 3 runs that just finished! :)
        MODELS=$(ls $UNLEARNED_MODELS_PATH/models | grep "seq-lr-$LR-" | grep $sample_count | grep $epoch)
        echo "Models that just finished unlearning:" ${MODELS[@]}
        GPU_NUM=0
        DEEPSPEED_PORT=60000
        PROCARR=()
        for model in ${MODELS[@]}
        do
            echo "Scheduling lm_eval for $model on GPU $GPU_NUM.."
            echo "Ensuring results dir exist.."
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/model_generations_truthfulqa/$model
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/truthfulqa
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/lm_eval_harness
            mkdir -p $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/harmfulness
            (CUDA_VISIBLE_DEVICES=$GPU_NUM HF_HOME=/scratch0/sduchnie/.huggingface nohup $LMEVAL_PYTHON_PATH -m lm_eval --model hf \
                --model_args pretrained=$UNLEARNED_MODELS_PATH/models/$model/idx_$epoch \
                --tasks $HF_LLM_LEADERBOARD_TASKS \
                --device "cuda" \
                --batch_size 16 \
                --trust_remote_code \
                --output_path $RESULTS_PATH/$model &> $LOGS_PATH/$model.log; \
            echo "eval done for $model, copying results and logs from scratch.."; \
            cp `find $RESULTS_PATH/$model -name "results*.json"` $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/lm_eval_harness/$model.json; \
            cp $LOGS_PATH/$model.log $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model; \
            $LMEVAL_PYTHON_PATH ../eval_framework_tasks/eval_results.py --log_dir $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/lm_eval_harness) &
            PROCARR+=($!)
            GPU_NUM=$((GPU_NUM+1))
            if [ $GPU_NUM -eq 3 ]; then
                echo "3/4 GPUs scheduled, waiting for ${PROCARR[@]}.."
                GPU_NUM=0
                DEEPSPEED_PORT=60000
                wait ${PROCARR[@]}
                echo "Done waiting"
                PROCARR=()
                date
            fi
        done
        echo "Waiting in case more evals still running..."
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        GPU_NUM=0
        DEEPSPEED_PORT=60000

        echo "Eval harmfulness for the models that just got unlearned..."
        for model in ${MODELS[@]}
        do
            echo "Generate model answers for $model on GPU: $GPU_NUM ... (path relative to llm_unlearn_ucl dir)"
            (CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_harmfulness/beavertails_get_model_answers.py \
                --model_path $UNLEARNED_MODELS_PATH/models/$model/idx_$epoch \
                --device cuda \
                --batch_size 128 \
                --output_dir $UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model \
                &> $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/model_gen.log; \
                echo "Model generations done for: $model. Now evaluating generations with beaverdam.."; \
            CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_harmfulness/evaluate_outputs.py \
                --eval_dataset $UNLEARNED_MODELS_PATH/experiment_data/model_generations/$model \
                --model_path $BEAVERDAM_WEIGHTS_PATH \
                --device cuda \
                --max_length 512 \
                --output_dir $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/harmfulness \
                &> $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/model_harmful_eval.log) &
            PROCARR+=($!)
            GPU_NUM=$((GPU_NUM+1))
            if [ $GPU_NUM -eq 3 ]; then
                echo "3/4 GPUs scheduled, waiting for ${PROCARR[@]}.."
                GPU_NUM=0
                DEEPSPEED_PORT=60000
                wait ${PROCARR[@]}
                echo "Done waiting"
                PROCARR=()
                date
            fi
        done
        echo "Waiting in case more evals still running..."
        wait ${PROCARR[@]}
        echo "Done waiting"
        PROCARR=()
        GPU_NUM=0
        DEEPSPEED_PORT=60000
        echo "Eval TruthfulQA answers (Just stats) for the models that just got unlearned..."
        for model in ${MODELS[@]}
        do
            echo "Generate model answers for $model on GPU: $GPU_NUM ... (path relative to llm_unlearn_ucl dir)"
            (CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_truthfulQA/generate_outputs.py \
                --model_path $UNLEARNED_MODELS_PATH/models/$model/idx_$epoch \
                --device cuda \
                --batch_size 128 \
                --output_dir $UNLEARNED_MODELS_PATH/experiment_data/model_generations_truthfulqa/$model \
                &> $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/truthfulqa_model_gen.log; \
                echo "Model generations done for: $model. Now evaluating generations with beaverdam.."; \
            CUDA_VISIBLE_DEVICES=$GPU_NUM nohup python3 ../eval_truthfulQA/evaluate_outputs.py \
                --eval_dataset $UNLEARNED_MODELS_PATH/experiment_data/model_generations_truthfulqa/$model \
                --output_dir $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/truthfulqa \
                &> $UNLEARNED_MODELS_PATH/experiment_data/eval_results/$model/model_truthfulqa_eval.log) &
            PROCARR+=($!)
            GPU_NUM=$((GPU_NUM+1))
            if [ $GPU_NUM -eq 3 ]; then
                echo "3/4 GPUs scheduled, waiting for ${PROCARR[@]}.."
                GPU_NUM=0
                DEEPSPEED_PORT=60000
                wait ${PROCARR[@]}
                echo "Done waiting"
                PROCARR=()
                date
            fi
        done
        
        echo "Waiting for processes: ${PROCARR[@]}..."
        wait ${PROCARR[@]}
        GPU_NUM=0
        DEEPSPEED_PORT=60000
        echo "Done waiting"
        PROCARR=()
        date
        echo "Syncing wandb runs.."
        for file in $UNLEARNED_MODELS_PATH/logs/stdout_*; do eval $(cat $file | grep "wandb sync .*" | sed 's/^.\{7\}//'); done;
        echo "Done syncing. Check wandb ;) "
    done
    date
done



echo "wait in case more stuff still scheduled but not all 4 gpus busy"
wait ${PROCARR[@]}
echo "Done waiting"


echo "oof, thats a long one, if nothing went wrong, ur set"
date