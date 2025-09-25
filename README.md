# neural-gravity
Source code for a neural network-based framework for signal separation in spatio-temporal gravity data

doi: [![DOI](https://zenodo.org/badge/901783924.svg)](https://doi.org/10.5281/zenodo.17182343)

Reference: This is the source code to the manuscript 
B. Heller-Kaikov, R. Pail and M. Werner, Neural network-based framework for signal separation in spatio-temporal gravity data. Computers and Geosciences (2025), doi: https://doi.org/10.1016/j.cageo.2025.106057

Contact: betty.heller@tum.de

### 1. Background

This code represents a closed-loop simulation framework for signal separation in spatio-temporal gravity data. It is based on a multi-channel U-Net architecture similar to Kadandale et al. (2020). Parts of the U-Net and data augmentation algorithms are taken from the tensorflow implementation of the pix2pix software (Isola et al., 2016).


### 2. General folder structure:

- **Code**: contains all scripts and code files needed to run the code

- **Container**: contains 2 subfolders, one for each of the singularity containers that can be used to run the code in a HPC environment

- **Data**: contains SH coefficients parametrizing the individual signals to be separated

- **Models**: after creating separation models, i.e. after running the python scripts create_training_data_01.py, train_model_02.py and test_model_03.py, this folder will contain one subfolder for each newly created separation model


### 3. Code parts and how to run the code:

#### --- Preparation of data files for the main code ---

**Code part 00: preprocessing of data files**

- In this part, the AOHIS data files of the Updated ESA Earth System Model (Dobslaw et al., 2014) are downloaded and preprocessed for further usage by the code.
    
- python script: preprocess_data_00.py
    
- run this code part by executing `sbatch run_00.sbatch` (this will execute the python script preprocess_data_00.py in the no-GPU container)

- Inputs: user needs to change the parameters components, years_int, latlont_string and deltat directly in preprocess_data_00.py

- Outputs: SH coefficient data files in folder Data/SH_coeff/

#### --- Main code: creation of new separation model ---

A prerequisite of running the code is the availability of the SH data files defining the individual signal components of interest. Those can be downloaded as described above (code part 00). Using data other than the signals given by the ESA ESM requires the user to prepare the data of interest in the same file structure and data format as the ESA ESM data files created by preprocess_data_00.py.

**Code part 01: preparation of input-target pairs**

- In this part, input-target pairs for training and testing the multi-channel U-Net network are created based on the SH coefficient data files in the folder Data/SH_coeff

- python script: create_training_data_01.py (uses functions.py, variables.py)

- The script run_01.sbatch in the model folder executes the python script create_training_data_01.py inside the no-GPU container

- Outputs written to model folder: hdf5 file containing input-target pairs, will be written to subfolder train_test_data_01


**Code part 02: training the network**

- In this part, a new signal separation model is created by training a multi-channel U-Net using the input-target pairs created in part 01

- python script: train_model_02.py (uses variables.py)

- The script run_02.sbatch in the model folder executes the python script train_model_02.py inside the GPU container 

- Outputs written to model folder: file info.txt containing input parameters of the particular model, neural network weights of trained model written to folder training_results_02/checkpoints, training and test error curves written to hdf5 file in subfolder training_results_02/logs/, predicted and true data for all data years written to hdf5 file in subfolder training_results_02/predicted_true_signals

**Code part 03: evaluate results and make plots**

- In this part, the trained separation model created in part 02 is tested and several plots for evaluating its performance are created

- python script: test_model_03.py (uses functions.py, variables.py)

- The script run_03.sbatch in the model folder executes the python script test_model_03.py in the no-GPU container

- Outputs written to model folder: data for plotting written to hdf5 file in subfolder test_results_03/data_for_plotting and png figures saved to subfolder test_results_03/Figures


**---> Note:** The code parts 01 to 03 can be executed subsequently by changing the input parameters in the file start_code.sh and executing `bash start_code.sh`. This will create a new model ID and corresponding subfolder in the directory Models/, prepare and copy all relevant code files to this subfolder and subsequently run the sbatch scripts run_01.sbatch, run_02.sbatch and run_03.sbatch

Hint: using the proposed folder structure and script start_code.sh to run the code, it is possible to start multiple models/tests at the same time, since the execution of the main code takes place within the newly created model folders


### 4. Containers:

We provide two singularity container images used to run the code:

- **Container_GPU_TF_02** is a tensorflow container corresponding to the docker image nvcr.io/nvidia/tensorflow:24.03-tf2-py3. It can be used to run code part 02 on a GPU

- **Container_no_GPU_01_03** is a python container which is based on the docker image python:3.11.10-bookworm, with the additional packages pyshtools, geopandas, cartopy and nexusformat. It can be used to run the remaining code parts


### 5. References:

Kadandale, V.S., Montesinos, J.F., Haro, G., Gómez, E., 2020. Multi-channel u-net for music source separation, in: 2020 IEEE 22nd International443
Workshop on Multimedia Signal Processing (MMSP), IEEE. pp. 1–6.

Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A., 2016. Image-to-image translation with conditional adversarial networks. https://doi.org/10.48550/arXiv.1611.07004

Dobslaw, Henryk; Bergmann-Wolf, Inga; Dill, Robert; Forootan, Ehsan; Klemann, Volker; Kusche, Jürgen; Sasgen, Ingo (2014): Supplement to: The Updated ESA Earth System Model for Gravity Mission Simulation Studies. Deutsches GeoForschungsZentrum GFZ. https://doi.org/10.5880/GFZ.1.3.2014.001

### 6. License information:
- Parts of the source code are taken from the tensorflow documentation which is published under the Apache 2.0 License
- The original source code generated by me is made available under the MIT License
