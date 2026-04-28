#!/bin/bash
#SBATCH --job-name=dl_crc_hic
#SBATCH --output=/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/dl_crc_hic_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Download colorectal cancer Hi-C 40kb ICE-normalized matrices from GSE133928
# Paper: Johnstone et al. 2020, "Large-Scale Topological Changes Restrain
#        Malignant Progression in Colorectal Cancer" (Cell Reports)
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133928

OUTDIR="/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/HiC"
mkdir -p "$OUTDIR"

BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3930nnn"
BASE2="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4513nnn"

# --- Tumor biopsies (10 samples) ---
# Cohort 1
wget -c -P "$OUTDIR" "${BASE}/GSM3930256/suppl/GSM3930256_BRD3162_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930259/suppl/GSM3930259_BRD3179_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930261/suppl/GSM3930261_BRD3187_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930286/suppl/GSM3930286_MGH1904_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930289/suppl/GSM3930289_MGH2834_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930290/suppl/GSM3930290_MGH5328_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930291/suppl/GSM3930291_MGH8416_40000_iced.matrix.gz"
# Cohort 2
wget -c -P "$OUTDIR" "${BASE2}/GSM4513970/suppl/GSM4513970_BRD3378_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE2}/GSM4513974/suppl/GSM4513974_BRD3462_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE2}/GSM4513992/suppl/GSM4513992_MGH3535_40000_iced.matrix.gz"

# --- Normal colon biopsies (7 samples) ---
# Cohort 1
wget -c -P "$OUTDIR" "${BASE}/GSM3930257/suppl/GSM3930257_BRD3162N-sb_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930258/suppl/GSM3930258_BRD3170N-sb_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930260/suppl/GSM3930260_BRD3179N_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930262/suppl/GSM3930262_BRD3187N_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930263/suppl/GSM3930263_BRD3328N_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930264/suppl/GSM3930264_BRD3409N-sb_40000_iced.matrix.gz"
wget -c -P "$OUTDIR" "${BASE}/GSM3930266/suppl/GSM3930266_BRD3462N-sb_40000_iced.matrix.gz"

echo "Download complete. Files saved to $OUTDIR"
ls -lh "$OUTDIR"
