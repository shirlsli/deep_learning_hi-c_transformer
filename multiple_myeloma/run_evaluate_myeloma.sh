#!/bin/bash
#SBATCH --job-name=mm_eval
#SBATCH --partition=scu-gpu
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

cd /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma

CHECKPOINT_DIR="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/checkpoints"
TEST_DATA="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt"
TEST_LABELS="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt"
PCA_MODEL="/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"

echo "============================================================"
echo "Multiple Myeloma Fine-Tuned Model Evaluation"
echo "============================================================"
echo

# Check for freeze_option=0 checkpoint
if [ -f "${CHECKPOINT_DIR}/myeloma_freeze0_best.pt" ]; then
    echo "Evaluating freeze_option=0 checkpoint..."
    python3 evaluate_myeloma_v2.py \
      --model_path "${CHECKPOINT_DIR}/myeloma_freeze0_best.pt" \
      --test_data "$TEST_DATA" \
      --test_labels "$TEST_LABELS" \
      --pca_model_path "$PCA_MODEL"
    echo
else
    echo "Warning: freeze_option=0 checkpoint not found at ${CHECKPOINT_DIR}/myeloma_freeze0_best.pt"
fi

# Check for freeze_option=1 checkpoint
if [ -f "${CHECKPOINT_DIR}/myeloma_freeze1_best.pt" ]; then
    echo "Evaluating freeze_option=1 checkpoint..."
    python3 evaluate_myeloma_v2.py \
      --model_path "${CHECKPOINT_DIR}/myeloma_freeze1_best.pt" \
      --test_data "$TEST_DATA" \
      --test_labels "$TEST_LABELS" \
      --pca_model_path "$PCA_MODEL"
    echo
else
    echo "Warning: freeze_option=1 checkpoint not found at ${CHECKPOINT_DIR}/myeloma_freeze1_best.pt"
fi

echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
