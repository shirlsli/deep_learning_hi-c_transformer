#!/bin/bash
#SBATCH --job-name=crc_finetune
#SBATCH --output=/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/crc_finetune_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=scu-gpu

set -e

BASE="/athena/angsd/scratch/ssl4003/deep-learning"
CRC_DIR="${BASE}/colorectal_cancer_data"
PROC_DIR="${CRC_DIR}/processed"
CKPT_DIR="${CRC_DIR}/checkpoints"
PRETRAINED="${BASE}/oral_cancer_data/train_model_with_pca/checkpoints_20260329_133858/best_model.pt" # Update this path to your pretrained oral cancer model checkpoint


source ~/.bashrc
conda activate hi-c_project

mkdir -p "$PROC_DIR" "$CKPT_DIR"

# ── Step 1: Preprocess colorectal Hi-C → genome-wide feature tensors ──────────
echo "============================================================"
echo "Step 1: Preprocessing colorectal Hi-C → genome-wide features"
echo "============================================================"
python3 "${CRC_DIR}/preprocess_colorectal.py" \
    --hic_dir "${CRC_DIR}/HiC" \
    --out_dir "${PROC_DIR}"

# ── Step 2: Check pretrained oral cancer model ────────────────────────────────
if [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: Pretrained model not found at ${PRETRAINED}"
    echo "Run the oral cancer training pipeline first (oral_cancer_data/run_oral_training.sh)."
    exit 1
fi
echo "Pretrained model: ${PRETRAINED}"

# ── Step 3: Fine-tune — both freeze strategies (saved to CKPT_DIR) ────────────
echo ""
echo "============================================================"
echo "Step 2: Fine-tuning (freeze_option=0 and freeze_option=1)"
echo "============================================================"
python3 "${CRC_DIR}/ctf_transformer_pipeline_colorectal.py" \
    --pretrained_path "$PRETRAINED" \
    --train_data      "${PROC_DIR}/colorectal_train_pca.pt" \
    --train_labels    "${PROC_DIR}/colorectal_train_labels.pt" \
    --test_data       "${PROC_DIR}/colorectal_test_pca.pt" \
    --test_labels     "${PROC_DIR}/colorectal_test_labels.pt" \
    --epochs 50 \
    --output_dir      "$CKPT_DIR"

# ── Step 4: Full evaluation for each freeze strategy ─────────────────────────
echo ""
echo "============================================================"
echo "Step 3: Evaluation (Accuracy, RMSE, Pearson r)"
echo "============================================================"
for freeze_opt in 0 1; do
    MODEL="${CKPT_DIR}/colorectal_cancer_freeze${freeze_opt}_best.pt"
    if [ -f "$MODEL" ]; then
        echo ""
        echo "--- freeze_option=${freeze_opt} ---"
        python3 "${CRC_DIR}/evaluate_colorectal.py" \
            --model_path  "$MODEL" \
            --test_data   "${PROC_DIR}/colorectal_test_pca.pt" \
            --test_labels "${PROC_DIR}/colorectal_test_labels.pt"
    else
        echo "WARNING: ${MODEL} not found — skipping evaluation for freeze_option=${freeze_opt}"
    fi
done

echo ""
echo "Pipeline complete."
