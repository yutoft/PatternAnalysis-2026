#!/bin/bash
#SBATCH --job-name=hipmri-final-eval
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=final_eval_%j.out
#SBATCH --error=final_eval_%j.err

set -e

source "$HOME/miniconda3/bin/activate" torch
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

baseline_ckpt="results/baseline/train30_708639/best_unet.pt"
improved_ckpt="results/improved/train30_708640/best_unet.pt"

# Selected validation examples across all four case IDs
python -u make_examples.py \
    --baseline "$baseline_ckpt" \
    --improved "$improved_ckpt" \
    --split validate \
    --device cuda \
    --indices 40 108 120 516 266 \
    --output results/comparison/validation_examples

# Full test evaluation using the same fixed checkpoints
python -u predict.py \
    --baseline "$baseline_ckpt" \
    --improved "$improved_ckpt" \
    --split test \
    --device cuda \
    --batch-size 4 \
    --workers 0 \
    --examples 5 \
    --output results/comparison/test_best