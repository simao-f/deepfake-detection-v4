#!/bin/bash

#SBATCH --job-name=efficientnet_ddp
#SBATCH --output=efficientnet_ddp_%j.out
#SBATCH --error=efficientnet_ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# 1. Activate environment
source /ceph/home/student.aau.dk/sr27mn/v4/.venv/bin/activate

# 2. Set up distributed environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4  # Total GPUs (1 node * 4 GPUs)
export OMP_NUM_THREADS=1

# 3. Launch training with torchrun
# --nproc_per_node=4 matches the number of GPUs requested
# --nnodes=1 since we are running on a single node
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT sets up the rendezvous backend

echo "Starting DDP training on $HOSTNAME with $WORLD_SIZE GPUs..."

# Navigate to project root so imports work correctly
cd /ceph/home/student.aau.dk/sr27mn/v4

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    models/efficientnet/train.py

echo "Training finished."