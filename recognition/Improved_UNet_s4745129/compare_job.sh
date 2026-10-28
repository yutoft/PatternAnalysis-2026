#!/bin/bash
#SBATCH --job-name=hipmri-compare
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=compare_%j.out
#SBATCH --error=compare_%j.err

set -e

source "$HOME/miniconda3/bin/activate" torch
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

python -u predict.py \
    --baseline results/baseline/train30_708639/best_unet.pt \
    --improved results/improved/train30_708640/best_unet.pt \
    --split validate \
    --device cuda \
    --batch-size 4 \
    --workers 0 \
    --examples 5 \
    --output results/comparison/validation_best