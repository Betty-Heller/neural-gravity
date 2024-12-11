#!/bin/bash

#SBATCH --job-name=create_training_data_01
#SBATCH -o ../Models/MODEL_ID/create_training_data_01.out
#SBATCH -e ../Models/MODEL_ID/create_training_data_01.out

CONTAINER="/glob/g01-cache/iapg/ge72fet/Signal_separation/U_Net_Paper/Container/Container_no_GPU_01_03/Container_no_GPU_01_03.sif"

singularity exec \
    --bind /glob/g01-cache/iapg/ge72fet/Signal_separation:/mnt/Signal_separation \
    --pwd /mnt/Signal_separation/U_Net_Paper/Models/MODEL_ID \
    "${CONTAINER}" \
    python3 -u create_training_data_01.py \
