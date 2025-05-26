#!/bin/bash
set -e

NAME=${1:-"torch_container"}

DEF_FILE="${NAME}.def"
OUTPUT_DIR="/d/hpc/projects/FRI/cb17769"
OUTPUT_IMAGE="${OUTPUT_DIR}/${NAME}.sif"

singularity build "${OUTPUT_IMAGE}" "${DEF_FILE}"