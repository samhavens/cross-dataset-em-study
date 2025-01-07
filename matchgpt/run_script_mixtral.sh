#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --time=15:00:00

source activate matchgpt

datasets=("abt" "amgo" "beer" "dbac" "dbgo" "foza" "itam" "roim" "waam" "wdc" "zoye")
for dataset in "${datasets[@]}"; do
  python -u matchgpt.py --model mixtral --dataset "$dataset"
done
conda deactivate