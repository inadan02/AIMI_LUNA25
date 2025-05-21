#!/bin/bash
#SBATCH --job-name=download_aimed
#SBATCH --output=logs_%j.out
#SBATCH --mem=128G
#SBATCH --cpus-per-task=18
#SBATCH --time=6:00:00
#SBATCH --partition=cnczshort

# Run your download script
python ../download_dataset.py