#! /bin/bash

BASE_PATH=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

source $BASE_PATH/../venv/bin/activate

echo "You need to log in using huggingface-cli and make sure you have access to the toxigen dataset ()."

MODELS=$(ls /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/opt1.3b_unlearned_mathqa)
EXPERIMENT_NAME="math-qa-experiment-toxicity"

for model in ${MODELS[@]}
do
    echo "Evaluating $model..."
    nohup $BASE_PATH/evaluate_model.sh $EXPERIMENT_NAME $model
done
