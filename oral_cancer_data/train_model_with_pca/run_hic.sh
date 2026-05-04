#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=is_conda
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

conda env create -f hic.yml