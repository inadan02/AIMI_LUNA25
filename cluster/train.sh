#!/bin/bash
#SBATCH --job-name=vit-aimed                  # Job name
#SBATCH --partition=gpu                     # Partition (queue) name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=24                   # CPU cores/threads per task
#SBATCH --gpus=1                            # Number of GPUs per node
#SBATCH --mem-per-gpu=64G                           # Job memory request
#SBATCH --time=06:00:00                     # Time limit hrs:min:sec
#SBATCH --output=logs/train_aimed_%j.log        # Standard output
#SBATCH --exclude=wn224,wn208,wn209,wn210,wn211,wn212,gwn04,wn222                     # Exclude specific nodes

#CONTAINER_FILE=/vol/csedu-nobackup/course/IMC037_aimi/group07/torch_container.sif
CONTAINER_FILE=/d/hpc/projects/FRI/cb17769/torch_container.sif

cd ..

# Run your training script
singularity exec --nv --writable-tmpfs $CONTAINER_FILE  python3 train.py