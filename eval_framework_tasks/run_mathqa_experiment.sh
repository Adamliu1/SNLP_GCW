#! /bin/bash

MODELS=$(ls /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/opt1.3b_unlearned_mathqa)
EXPERIMENT_NAME="math-qa-experiment"

for model in ${MODELS[@]}
do
    echo "Evaluating $model..."
    ./evaluate_model.sh $EXPERIMENT_NAME $model
done
