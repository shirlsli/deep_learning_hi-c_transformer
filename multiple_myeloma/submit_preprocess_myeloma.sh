#!/bin/bash
#SBATCH --job-name=mm_preproc
#SBATCH --output=mm_preproc_%j.out
#SBATCH --error=mm_preproc_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -p scu-gpu

cd /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma

source /home/fs01/nsd4002/miniforge3/etc/profile.d/conda.sh
conda activate hic-transformer

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(which python3)"
echo "Env: $CONDA_DEFAULT_ENV"

python3 /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/preprocess_multiple_myeloma.py \
  --hic_dir /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/data \
  --out_dir /athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed \
  --label_mode cell_line
