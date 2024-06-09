#$ -l tmem=40G # Anything under 125G/num_gpus
#$ -l h_rt=24:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -pe gpu 1 # Less than 1
#$ -j y
# $ -l tscratch=300G
#$ -l hostname=dip-207-2

source ~/setup_python.sh
cd /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

date
hostname


source ./run_experiment_raw_model.sh "opt-1.3b" "facebook/opt-1.3b" "cuda:0"

date
