#!/bin/bash

#SBATCH --job-name=test_model_03
#SBATCH -o ../Models/MODEL_ID/test_model_03.out
#SBATCH -e ../Models/MODEL_ID/test_model_03.out
#SBATCH -d afterok:JOB_ID_02

CONTAINER="/glob/g01-cache/iapg/ge72fet/Signal_separation/U_Net_Paper/Container/Container_no_GPU_01_03/Container_no_GPU_01_03.sif"

# check every hour if second program part is done:
until [ -f ../Models/MODEL_ID/02_success ]
do
    echo 'success file not found'
    sleep 60m 
done

# run second program part:
singularity exec \
    --bind /glob/g01-cache/iapg/ge72fet/Signal_separation:/mnt/Signal_separation \
    --pwd /mnt/Signal_separation/U_Net_Paper/Models/MODEL_ID \
    "${CONTAINER}" \
    python3 -u test_model_03.py \
