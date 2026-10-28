#!/bin/bash
#SBATCH --job-name=hipmri-training
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:15:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -e

source "$HOME/miniconda3/bin/activate" torch

cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

# First argument selects baseline or improved
model=${1:?Specify baseline or improved}

# Job ID gives each run its own results folder
python -u train.py \
    --model "$model" \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-4 \
    --workers 0 \
    --output "results/${model}/train30_${SLURM_JOB_ID}"