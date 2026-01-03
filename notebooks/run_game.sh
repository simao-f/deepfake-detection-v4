#!/bin/bash
#SBATCH --job-name=game_analysis
#SBATCH --output=logs/game_analysis_%j.out
#SBATCH --error=logs/game_analysis_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Activate environment
source .venv/bin/activate

# 3. Execute the script
python notebooks/analyze_game_history.py
