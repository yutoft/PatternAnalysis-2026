#!/bin/bash
#SBATCH --job-name=hipmri-improved-smoke
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=improved_smoke_%j.out
#SBATCH --error=improved_smoke_%j.err

set -e

# Activate torch environment
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Go to project folder
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

# Run short training check
python -u train.py --model improved --smoke --workers 0