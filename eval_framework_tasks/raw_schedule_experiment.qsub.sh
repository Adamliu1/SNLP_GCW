#$ -l tmem=40G # Anything under 125G/num_gpus
#$ -l h_rt=24:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -pe gpu 1 # Less than 1
#$ -j y
# $ -l tscratch=300G
#$ -l hostname=dip-207-2

source /scratch0/aszablew/venv/bin/activate
cd /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

date
hostname

PROCARR=()

# argument order: experiment_name_prefix hf_model_name/model_path cuda_gpu_id   
source ./raw_run_experiment.sh "llama3-8b" "meta-llama/Meta-Llama-3-8B" 0 &
PROCARR+=($!)
source ./raw_run_experiment.sh "gemma-7b" "google/gemma-7b" 1 &
PROCARR+=($!)
source ./raw_run_experiment.sh "aya-8b" "CohereForAI/aya-23-8B" 2 & 
PROCARR+=($!)
source ./raw_run_experiment.sh "olmo-7b" "allenai/OLMo-7B-hf" 3 &
PROCARR+=($!)


wait ${PROCARR[@]}

date

