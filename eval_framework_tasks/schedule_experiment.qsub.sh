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

PROCARR=()
# /scratch0/sduchnie/gemma-2b-unlearn-harm-half_LR5e4
# /scratch0/sduchnie/gemma-2b-unlearn-harm-full_LR5e4

# /scratch0/sduchnie/gemma-2b-unlearn-harm-half_LR1e4
# /scratch0/sduchnie/gemma-2b-unlearn-harm-full_LR1e4
# /scratch0/sduchnie/gemma-2b-unlearn-harm-half_LR5e5
source ./run_experiment.sh "gemma-2b-unlearn-harm-half_LR5e5" "/scratch0/sduchnie" 
PROCARR+=($!)


wait ${PROCARR[@]}

date

