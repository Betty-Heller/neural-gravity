#!/bin/bash

#SBATCH --job-name=preprocess_data_00
#SBATCH -o preprocess_data_00.out
#SBATCH -e preprocess_data_00.out

CONTAINER="/glob/g01-cache/iapg/ge72fet/Signal_separation/U_Net_Paper/Container/Container_no_GPU_01_03/Container_no_GPU_01_03.sif"

singularity exec \
    --bind /glob/g01-cache/iapg/ge72fet/Signal_separation:/mnt/Signal_separation \
    --pwd /mnt/Signal_separation/U_Net_Paper/Code \
    "${CONTAINER}" \
    python3 -u preprocess_data_00.py \
