#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=50:00:00

source activate ditto
datasets=("abt" "amgo" "beer" "dbac" "dbgo" "foza" "itam" "roim" "waam" "wdc" "zoye")
seeds=(42 44 46 48 50)

# Loop through datasets and seeds
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python train_ditto.py \
      --task loo-${dataset}-${seed} \
      --batch_size 64 \
      --max_len 512 \
      --lr 3e-5 \
      --n_epochs 20 \
      --lm bert \
      --da del \
      --dk product \
      --summarize
  done
done

conda deactivate