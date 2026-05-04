#!/bin/bash
#SBATCH -p scu-gpu
#SBATCH -t 15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60GB
#SBATCH -o oral_train_%j.out
#SBATCH -e oral_train_%j.err

#set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hic-transformer

python3 -u train_oral.py
