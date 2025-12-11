#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --output=vit_%j.out
#SBATCH --error=vit_%j.err
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

# --- SETUP ---
# 1. Define project path
PROJECT_DIR="/ceph/home/student.aau.dk/sr27mn/v4"

# 2. Move to project directory immediately
cd $PROJECT_DIR

# 3. Activate Environment
source .venv/bin/activate

# 4. Set Environment Variables
export OMP_NUM_THREADS=1

# --- RUN ---
echo "Job started on $(hostname) at $(date)"

# 5. Run the script
python models/vision_transformer/train.py