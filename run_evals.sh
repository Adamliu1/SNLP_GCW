#!/bin/bash

# Check if the experiment type was passed as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <type>"
    echo "Example: $0 batch"
    exit 1
fi

TYPE="$1"

# Define the base directory where the experiment folders are located
# BASE_DIR="/path/to/your/experiments"
BASE_DIR=/SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/snlp-unlearned-models/models/

# Define the path to run_experiment.sh
EXPERIMENT_SCRIPT_PATH="/path/to/run_experiment.sh"

# Navigate to the base directory
cd "$BASE_DIR" || exit

# Loop through each directory that starts with the specified type
for dir in ${TYPE}-* ; do
    if [ -d "$dir" ]; then
        echo "Running $TYPE type experiments in $dir"
        # MODEL_NAME is set as the directory name minus the type prefix and path
        MODEL_NAME="${dir#*-}"
        # MODELS_PATH is the full path to the directory
        MODELS_PATH="$(pwd)/$dir"
        
        # Call the experiment script with the model name and models path
        if [ -f "$EXPERIMENT_SCRIPT_PATH" ]; then
            $EXPERIMENT_SCRIPT_PATH "$MODEL_NAME" "$MODELS_PATH"
        else
            echo "Experiment script not found at $EXPERIMENT_SCRIPT_PATH"
        fi
    fi
done

echo "All specified type experiments completed."
