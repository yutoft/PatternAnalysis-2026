#!/bin/bash
#SBATCH --job-name=hipmri-profile
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err

set -e

source "$HOME/miniconda3/bin/activate" torch
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

python -u profile_and_overlays.py \
    --baseline results/baseline/train30_708639/best_unet.pt \
    --improved results/improved/train30_708640/best_unet.pt \
    --reference-metrics results/comparison/test_best/metrics.json \
    --split test \
    --indices 268 215 271 453 515 \
    --labels 3 4 5 \
    --output results/comparison/profile_fixed