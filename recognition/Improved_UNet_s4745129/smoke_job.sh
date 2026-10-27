#!/bin/bash
#SBATCH --job-name=hipmri-smoke
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

set -e

# Activate torch environment
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Go to project folder
cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_s4745129"

# Limit CPU threads to requested allocation
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Run short training check
python -u train.py --smoke