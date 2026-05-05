#!/bin/bash
#SBATCH --job-name=mm_test_oral
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

echo "============================================================"
echo "Testing Myeloma Data on Oral Cancer Checkpoint (Zero-Shot)"
echo "============================================================"
echo

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

cd /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma

python3 test_myeloma_on_oral_model.py

echo
echo "============================================================"
echo "Test complete!"
echo "============================================================"
