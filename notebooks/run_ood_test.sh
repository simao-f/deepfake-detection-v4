#!/bin/bash

#SBATCH --job-name=ood_test
#SBATCH --output=logs/ood_test_%j.out
#SBATCH --error=logs/ood_test_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# 1. Activate environment
source /ceph/home/student.aau.dk/sr27mn/v4/.venv/bin/activate

# 2. Go to project root
cd /ceph/home/student.aau.dk/sr27mn/v4

echo "Starting OOD Test..."
python notebooks/ood_test.py
echo "OOD Test Completed."
