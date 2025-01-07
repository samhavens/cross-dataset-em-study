#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=20:00:00

source activate jellyfish

seeds=(42 44 46 48 50)
for seed in "${seeds[@]}"; do
  python -u jellyfish.py --seed "$seed"
done
conda deactivate