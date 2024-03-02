# Project to get all min-k% log likelihood probability for ANY dataset


Use:
`python3 src/run.py --target_model <MODEL_NAME_OR_PATH> --data <DATASET_NAME> --split <SPLIT_NAME (e.g. train or test)> --key_name <COLUMN_TO_EVAL> --cache_dir <ABSOLUTE_CACHE_PATH> --max_samples <MAX_NUM_OF_VALUES>`

e.g.:
```
python3 src/run.py --target_model allenai/OLMo-1B --data PKU-Alignment/PKU-SafeRLHF --split train --key_name prompt --cache_dir ~/sduchnie/SNLP_GCW/.cache --max_samples 400
```
