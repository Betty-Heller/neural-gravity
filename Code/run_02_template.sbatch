#!/bin/bash

#SBATCH --job-name=train_model_02
#SBATCH --gres=gpu:1
#SBATCH -o ../Models/MODEL_ID/train_model_02.out
#SBATCH -e ../Models/MODEL_ID/train_model_02.out
#SBATCH -d afterok:JOB_ID_01

CONTAINER="/glob/g01-cache/iapg/ge72fet/Signal_separation/U_Net_Paper/Container/Container_GPU_TF_02/Container_GPU_TF_02.sif"

# check if GPU is recognized:
nvidia-smi

# check every hour if first program part is done:
until [ -f ../Models/MODEL_ID/01_success ]
do
    echo 'success file not found'
    sleep 60m
done

# run second program part:
singularity exec --nv \
    --bind /glob/g01-cache/iapg/ge72fet/Signal_separation:/mnt/Signal_separation \
    --pwd /mnt/Signal_separation/U_Net_Paper/Models/MODEL_ID \
    "${CONTAINER}" \
    python3 -u train_model_02.py \
