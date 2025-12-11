#!/bin/bash

#SBATCH --job-name=resnet50_train
#SBATCH --output=resnet50_%j.out
#SBATCH --error=resnet50_%j.err
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

# --- SETUP ---
# 1. Define your project folder explicitly
PROJECT_DIR="/ceph/home/student.aau.dk/sr27mn/v4"

# 2. Move to that folder
cd $PROJECT_DIR

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Set environment variables
export OMP_NUM_THREADS=1

# --- RUN ---
echo "Starting ResNet training on $(hostname) at $(date)"

# 5. Run the script
python models/restnet/train.py