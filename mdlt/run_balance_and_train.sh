#!/bin/bash

#SBATCH --job-name=BAL_TRAIN
#SBATCH --partition=laal_a6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_logs/S-%x.%j.out

# Change to repository root
cd "$(dirname "$0")/.."

# Example call to balance dataset and train ERM
python -m mdlt.scripts.balance_and_train \
  --dataset PACS \
  --data_dir /path/to/data \
  --output_dir ./output \
  --ratio 1.0 \
  --seed 0
