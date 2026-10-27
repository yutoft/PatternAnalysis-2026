#!/bin/bash
#SBATCH --job-name=hipmri-baseline
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

set -e

# Activate torch environment
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Go to project folder
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

# Run one full epoch
python -u train.py --epochs 1 --workers 0