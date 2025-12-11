#!/bin/bash

#SBATCH --job-name=overfit_analysis
#SBATCH --output=logs/overfit_analysis_%j.out
#SBATCH --error=logs/overfit_analysis_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# --- SETUP ---
# 1. Define Project Root
PROJECT_DIR="/ceph/home/student.aau.dk/sr27mn/v4"

# 2. Activate your environment
source "${PROJECT_DIR}/.venv/bin/activate"

# 3. Move to the project root
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs/overfit_analysis

# --- RUN ---
echo "Starting Overfitting Analysis on $(hostname) at $(date)"

# 4. Execute the evaluation script
python notebooks/overfit.py
