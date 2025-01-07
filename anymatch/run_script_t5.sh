#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=50:00:00

source activate anymatch
TOKENIZERS_PARALLELISM=false
datasets=("abt" "amgo" "beer" "dbac" "dbgo" "foza" "itam" "roim" "waam" "wdc" "zoye")
seeds=(42 44 46 48 50)

# Loop through datasets and seeds
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python -u loo.py \
      --leaved_dataset_name "$dataset" \
      --base_model t5 \
      --seed "$seed" \
      --tbs 64 \
      --epochs 25 \
      --patience_start 10 \
      --patience 6
  done
done

conda deactivate