#!/bin/bash
#SBATCH --job-name=mm_finetune
#SBATCH --output=/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/mm_finetune_%j.out
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=scu-gpu

set -e

BASE="/athena/angsd/scratch/ssl4003/deep-learning"
MM_DIR="${BASE}/multiple_myeloma"
PROC_DIR="${MM_DIR}/processed"
CKPT_DIR="${MM_DIR}/checkpoints"
PRETRAINED="${BASE}/oral_cancer_data/train_model_with_pca/checkpoints_20260329_133858/best_model.pt"

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

mkdir -p "$PROC_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Step 1: Checking preprocessed myeloma tensors"
echo "============================================================"
TRAIN_DATA="${PROC_DIR}/myeloma_train_data.pt"
TRAIN_LABELS="${PROC_DIR}/myeloma_train_labels.pt"
TEST_DATA="${PROC_DIR}/myeloma_test_data.pt"
TEST_LABELS="${PROC_DIR}/myeloma_test_labels.pt"
ORAL_PCA="/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"

if [ -f "$TRAIN_DATA" ] && [ -f "$TRAIN_LABELS" ] && [ -f "$TEST_DATA" ] && [ -f "$TEST_LABELS" ]; then
    echo "Found existing preprocessed tensors in ${PROC_DIR}; skipping preprocessing."
else
    echo "Preprocessed tensors missing; running preprocessing now."
    python3 "${MM_DIR}/preprocess_multiple_myeloma.py" \
        --hic_dir "${MM_DIR}/data" \
        --out_dir "${PROC_DIR}" \
        --label_mode cell_line
fi

if [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: Pretrained model not found at ${PRETRAINED}"
    exit 1
fi
echo "Pretrained model: ${PRETRAINED}"

echo ""
echo "============================================================"
echo "Step 2: Fine-tuning (freeze_option=0 and freeze_option=1)"
echo "============================================================"
python3 "${MM_DIR}/ctf_transformer_pipeline_myeloma.py" \
    --pretrained_path "$PRETRAINED" \
    --train_data      "$TRAIN_DATA" \
    --train_labels    "$TRAIN_LABELS" \
    --test_data       "$TEST_DATA" \
    --test_labels     "$TEST_LABELS" \
    --pca_model_path  "$ORAL_PCA" \
    --epochs 50 \
    --output_dir      "$CKPT_DIR"

echo ""
echo "============================================================"
echo "Step 3: Evaluation"
echo "============================================================"
for freeze_opt in 0 1; do
    MODEL="${CKPT_DIR}/myeloma_freeze${freeze_opt}_best.pt"
    if [ -f "$MODEL" ]; then
        echo ""
        echo "--- freeze_option=${freeze_opt} ---"
        python3 "${MM_DIR}/evaluate_myeloma.py" \
            --model_path  "$MODEL" \
            --test_data   "$TEST_DATA" \
            --test_labels "$TEST_LABELS" \
            --pca_model_path "$ORAL_PCA"
    else
        echo "WARNING: ${MODEL} not found - skipping evaluation for freeze_option=${freeze_opt}"
    fi
done

echo ""
echo "Pipeline complete."
