#$ -l tmem=40G # Anything under 125G/num_gpus
#$ -l h_rt=24:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -pe gpu 4 # Less than 1
#$ -j y
# $ -l tscratch=300G
#$ -l hostname=dip-207-2

source ~/setup_python.sh
cd /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

date
hostname

PROCARR=()

source ./run_experiment_raw_model.sh "opt-1.3b" "facebook/opt-1.3b" "cuda:0" &
PROCARR+=($!)

source ./run_experiment_raw_model.sh "phi-1_5" "microsoft/phi-1_5" "cuda:1" &
PROCARR+=($!)

source ./run_experiment_raw_model.sh "olmo-1b" "allenai/OLMo-1B-hf" "cuda:2" &
PROCARR+=($!)

source ./run_experiment_raw_model.sh "gemma2b" "google/gemma-2b" "cuda:3" 
PROCARR+=($!)

wait ${PROCARR[@]}

date

