#!/bin/bash
#SBATCH --job-name=train_aimed
#SBATCH --output=logs/train_%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=csedu

CONTAINER_FILE=/vol/csedu-nobackup/course/IMC037_aimi/group07/torch_container.sif

# Run your training script
singularity exec --nv $CONTAINER_FILE python ../train.py