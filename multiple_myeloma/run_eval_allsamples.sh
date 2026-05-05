#!/bin/bash
#SBATCH --job-name=mm_eval_allsamples
#SBATCH --output=mm_eval_allsamples_%j.out
#SBATCH --error=mm_eval_allsamples_%j.err
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=scu-gpu

set -euo pipefail

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

cd /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma

echo "Starting all-samples evaluation..."
python3 evaluate_allsamples.py

echo "Done!"
