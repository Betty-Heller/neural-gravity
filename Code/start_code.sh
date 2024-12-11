#!/bin/bash

########
# change the input parameters here:

time_resolution_in_hours=7*24 
Nmax_orig=180
Nmax_flt=100

components="['AO','H','I','S']"

#latlont_string=''
latlont_string='_yearly_mean_subtracted'

extent='global' 
#extent='regional'

#lat_interval_regional=30,90
lat_interval_regional=45,85
lon_interval_regional=220,300

filter_size=4
num_filters_min=64
num_filters_max=512

batch_size=1
learning_rate=1e-5
beta_1=0.9
beta_2=0.999
num_epochs=100
min_number_epochs=10

unit='EWH'

first_training_year=1995
last_training_year=2004
validation_year=2005
test_year=2006

sampling=lat_lon
#sampling=lat_t
#sampling=t_lon

random_samples='no'

#loss_type='absolute_MSE'
#loss_type='relative_MSE'
loss_type='signal_weighted_MSE'

# number of entries in "weights_comp", "weights_slope_value", "weights_linear_shape", "mult_comp" needs to fit the number of entries in "components"
weights_comp=1.,1.,1.,1.
weight_sum=0.
weights_slope_value=0.,0.,0.,0.
weights_linear_shape=0.,0.,0.,0.

data_augmentation=0,0,0,0,0
mult_factor_min_max=0.5,2.
mult_comp=0,0,0,1

########

# new model ID is created:
model_id=$(date +"%Y%m%d-%H%M%S")
echo 'New model with ID:' $model_id
mkdir ../Models/$model_id

# copy the python variables file template and replace input parameters in there:
cp variables_template.py ../Models/$model_id/variables.py
sed -i 's/MODEL_ID/'$model_id/ ../Models/$model_id/variables.py
sed -i 's/TIME_RESOLUTION_IN_HOURS/'$time_resolution_in_hours/ ../Models/$model_id/variables.py
sed -i 's/NMAX_ORIG/'$Nmax_orig/ ../Models/$model_id/variables.py
sed -i 's/NMAX_FLT/'$Nmax_flt/ ../Models/$model_id/variables.py
sed -i 's/COMPONENTS/'$components/ ../Models/$model_id/variables.py
sed -i 's/FILTER_SIZE/'$filter_size/ ../Models/$model_id/variables.py
sed -i 's/NUM_FILTERS_MIN/'$num_filters_min/ ../Models/$model_id/variables.py
sed -i 's/NUM_FILTERS_MAX/'$num_filters_max/ ../Models/$model_id/variables.py
sed -i 's/BATCH_SIZE/'$batch_size/ ../Models/$model_id/variables.py
sed -i 's/LEARNING_RATE/'$learning_rate/ ../Models/$model_id/variables.py
sed -i 's/BETA_1/'$beta_1/ ../Models/$model_id/variables.py
sed -i 's/BETA_2/'$beta_2/ ../Models/$model_id/variables.py
sed -i 's/NUM_EPOCHS/'$num_epochs/ ../Models/$model_id/variables.py
sed -i 's/MIN_NUMBER_EPOCHS/'$min_number_epochs/ ../Models/$model_id/variables.py
sed -i 's/UNIT/'$unit/ ../Models/$model_id/variables.py
sed -i 's/FIRST_TRAINING_YEAR/'$first_training_year/ ../Models/$model_id/variables.py
sed -i 's/LAST_TRAINING_YEAR/'$last_training_year/ ../Models/$model_id/variables.py
sed -i 's/VALIDATION_YEAR/'$validation_year/ ../Models/$model_id/variables.py
sed -i 's/TEST_YEAR/'$test_year/ ../Models/$model_id/variables.py
sed -i 's/SAMPLING/'$sampling/ ../Models/$model_id/variables.py
sed -i 's/WEIGHTS_COMP/'$weights_comp/ ../Models/$model_id/variables.py
sed -i 's/WEIGHT_SUM/'$weight_sum/ ../Models/$model_id/variables.py
sed -i 's/DATA_AUGMENTATION/'$data_augmentation/ ../Models/$model_id/variables.py
sed -i 's/MULT_FACTOR_MIN_MAX/'$mult_factor_min_max/ ../Models/$model_id/variables.py
sed -i 's/MULT_COMP/'$mult_comp/ ../Models/$model_id/variables.py
sed -i 's/LATLONT_STRING/'$latlont_string/ ../Models/$model_id/variables.py
sed -i 's/WEIGHTS_SLOPE_VALUE/'$weights_slope_value/ ../Models/$model_id/variables.py
sed -i 's/WEIGHTS_LINEAR_SHAPE/'$weights_linear_shape/ ../Models/$model_id/variables.py
sed -i 's/EXTENT/'$extent/ ../Models/$model_id/variables.py
sed -i 's/LOSS_TYPE/'$loss_type/ ../Models/$model_id/variables.py
sed -i 's/LAT_INTERVAL_REGIONAL/'$lat_interval_regional/ ../Models/$model_id/variables.py
sed -i 's/LON_INTERVAL_REGIONAL/'$lon_interval_regional/ ../Models/$model_id/variables.py
sed -i 's/RANDOM_SAMPLES/'$random_samples/ ../Models/$model_id/variables.py

# copy the other python code files to the model directory to run the code there
cp functions.py ../Models/$model_id/
cp create_training_data_01.py ../Models/$model_id/
cp train_model_02.py ../Models/$model_id/
cp test_model_03.py ../Models/$model_id/

# prepare & submit first sbatch job:
cp run_01_template.sbatch ../Models/$model_id/run_01.sbatch
sed -i 's/MODEL_ID/'$model_id/ ../Models/$model_id/run_01.sbatch
job_id_01=$(sbatch --parsable ../Models/$model_id/run_01.sbatch)

# prepare & submit second sbatch job:
cp run_02_template.sbatch ../Models/$model_id/run_02.sbatch
sed -i 's/MODEL_ID/'$model_id/ ../Models/$model_id/run_02.sbatch
sed -i 's/JOB_ID_01/'$job_id_01/ ../Models/$model_id/run_02.sbatch
job_id_02=$(sbatch --parsable ../Models/$model_id/run_02.sbatch)

# prepare & submit third sbatch job:
cp run_03_template.sbatch ../Models/$model_id/run_03.sbatch
sed -i 's/MODEL_ID/'$model_id/ ../Models/$model_id/run_03.sbatch
sed -i 's/JOB_ID_02/'$job_id_02/ ../Models/$model_id/run_03.sbatch
sbatch ../Models/$model_id/run_03.sbatch
