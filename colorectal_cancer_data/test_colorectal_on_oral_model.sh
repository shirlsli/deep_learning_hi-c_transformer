#!/bin/bash
#SBATCH --job-name=crc_finetune_cpu
#SBATCH --output=/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/crc_finetune_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1

BASE="/athena/angsd/scratch/ssl4003/deep-learning"
CRC_DIR="${BASE}/colorectal_cancer_data"
PROC_DIR="${CRC_DIR}/processed"
CKPT_DIR="${CRC_DIR}/checkpoints"
PRETRAINED="${BASE}/oral_cancer_data/train_model_with_pca/checkpoints_20260329_133858/best_model.pt"

source ~/.bashrc
conda activate hi-c_project

python test_colorectal.py