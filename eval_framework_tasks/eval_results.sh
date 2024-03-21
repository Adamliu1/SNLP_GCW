#! /bin/bash

# TODO: Add instructions how to use.
EXPERIMENT_NAME=opt1.3b_unlearned_harmful-eval1

LOGS_BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks/experiment_data/
BASE_PATH="/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks"
ANALYSIS_BASE_PATH=$BASE_PATH/analysis/$EXPERIMENT_NAME
ANALYSIS_LOGS_PATH=$ANALYSIS_BASE_PATH/eval_results


mkdir -p $ANALYSIS_BASE_PATH
mkdir $ANALYSIS_LOGS_PATH

# Copy all results files to the analysis directory for the given experiment
for file in `find $LOGS_BASE_PATH/$EXPERIMENT_NAME/results/ -name "*_idx_*.json"`
do
    cp $file $ANALYSIS_LOGS_PATH
done

# ugh this is a dodgy way of doing it xd
$BASE_PATH/../venv/bin/python3 eval_results.py --log_dir $ANALYSIS_LOGS_PATH

