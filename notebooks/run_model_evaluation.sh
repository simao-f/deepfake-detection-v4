#!/bin/bash

#SBATCH --job-name=model_eval_full
#SBATCH --output=logs/model_eval_full_%j.out
#SBATCH --error=logs/model_eval_full_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# 1. Activate your environment
source /ceph/home/student.aau.dk/sr27mn/v4/.venv/bin/activate

# 2. Go to the project root
cd /ceph/home/student.aau.dk/sr27mn/v4

# Create logs directory if it doesn't exist
mkdir -p logs/plots

echo "Starting Comprehensive Evaluation..."

# 3. Run the master evaluation script
python notebooks/model_evaluation.py --output_dir logs/plots

echo "Evaluation completed."
