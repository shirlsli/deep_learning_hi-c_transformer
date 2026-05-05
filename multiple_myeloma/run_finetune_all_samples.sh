#!/bin/bash
#SBATCH --job-name=mm_finetune_all
#SBATCH --partition=scu-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

cd /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma

echo "============================================================"
echo "Fine-tuning on ALL 4 Myeloma Samples (Combined)"
echo "============================================================"
echo

python3 ctf_transformer_pipeline_myeloma_all_samples.py \
  --train_data /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_data.pt \
  --train_labels /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_labels.pt \
  --test_data /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt \
  --test_labels /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt \
  --output_dir /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/checkpoints_all_samples \
  --epochs 150 \
  --batch_size 2

echo
echo "============================================================"
echo "Fine-tuning complete!"
echo "============================================================"
