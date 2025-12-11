#!/bin/bash

#SBATCH --job-name=eda_script_run
#SBATCH --output=eda_script_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00

# 1. Activate your environment
source /ceph/home/student.aau.dk/sr27mn/v4/.venv/bin/activate

# 2. Go to the project root
cd /ceph/home/student.aau.dk/sr27mn/v4

# 3. Execute the script
python notebooks/eda.py
