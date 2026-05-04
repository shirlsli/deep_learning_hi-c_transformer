#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=is_conda
#SBATCH --mem=1G

conda env create -f hicenv.yml