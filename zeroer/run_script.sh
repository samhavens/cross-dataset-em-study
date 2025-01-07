#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00

source activate zeroer

datasets=("abt" "amgo" "beer" "dbac" "dbgo" "foza" "itam" "roim" "waam" "wdc" "zoye")

seeds=(42 44 46 48 50)

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python zeroer.py $dataset --seed $seed
  done
done
conda deactivate