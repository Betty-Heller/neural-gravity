#!/usr/bin/env python3
 

"""j
Variables used by create_training_data_01.py, train_model_02.py, test_model_03.py
j"""


"""j
Packages import
j"""

import numpy as np
import h5py
import os

"""j
Model ID:
j"""

model_id = 'MODEL_ID'

"""j
Global parameters/constants:
j"""

R = 6378137 # unit: m, extracted from AOHIS files
GM = 0.39860050000000e+15 # unit: m^3/s^2, extracted from AOHIS files
rho_ave = 5517 # average density of the Earth
rho_w = 1000 # density of water

"""j
Parameters to be chosen:
j"""

deltat = TIME_RESOLUTION_IN_HOURS # temporal resolution in hours (minimum are 6 hours, should be multiple of 6 hours)

Nmax_orig = NMAX_ORIG # maximum d/o of the SH coefficient files
Nmax_flt = NMAX_FLT # maximum d/o for Gaussian filtering (defining the spatial resolution)

components = COMPONENTS # list of components to be separated

latlont_string = 'LATLONT_STRING' # specifies the folder name extension in the lat-lon-t data folder name (e.g. _yearly_mean_subtracted)
extent = 'EXTENT' # specify if working on a global or a regional lat-lon grid

lat_interval_regional = [LAT_INTERVAL_REGIONAL]
lon_interval_regional = [LON_INTERVAL_REGIONAL]

filter_size = FILTER_SIZE # height & width of filters in convolutional layers
num_filters_min = NUM_FILTERS_MIN # number of filters in the first convolutional layer (determines number of feature maps produced from the input layer's data -> should be a power of 2)
num_filters_max = NUM_FILTERS_MAX # maximum number of filters reached after several layers in which the number of filters is doubled


bs = BATCH_SIZE # batch size in the training process
lr = LEARNING_RATE # learning rate in the training process
beta_1 = BETA_1 # beta_1 parameter of the Adam optimizer algorithm, controlling amount of momentum (0.9 is default)
beta_2 = BETA_2 # beta_2 parameter of the Adam optimizer algorithm, controlling the amount of dampening of large gradients (0.999 is default)
num_epochs = NUM_EPOCHS # number of training epochs
min_number_epochs = MIN_NUMBER_EPOCHS # minimum number of training epochs before early stopping criterium is tested

unit = 'UNIT' # physical units to work with (only EWH implemented)

first_year = FIRST_TRAINING_YEAR # first year of training dataset
last_year = LAST_TRAINING_YEAR # last (included) year of training dataset
validation_year = 'VALIDATION_YEAR' # year for validation (used in early stopping criterium)
test_year = 'TEST_YEAR' # year for testing the found model

sampling = 'SAMPLING' # sampling strategy defining meaning of two image axes: lat_lon, lat_t or t_lon

random_samples = 'RANDOM_SAMPLES' # switch on (1) or off (0) usage of random linear combinations of the samples as samples

loss_type = 'LOSS_TYPE' # specify the normalization type of the component and sum loss terms: absolute MSE, relative MSE or signal-weighted MSE

weights_comp = [WEIGHTS_COMP] # weights of component loss terms
weight_sum = [WEIGHT_SUM] # weight of sum loss term
weights_slope_value = [WEIGHTS_SLOPE_VALUE] # weights of loss terms constraining the slope of the data along the time direction (for each component individually)
weights_linear_shape = [WEIGHTS_LINEAR_SHAPE] # weights of loss terms constraining the linear shape of the data along the time direction (for each component individually)

data_augmentation = [DATA_AUGMENTATION] # switch on (nonzero values) or off (0) the 5 data augmentation strategies applied to the data during training (example: [30,1,1,0.005,0.5])
mult_factor_min_max = [MULT_FACTOR_MIN_MAX] # specifies the minimum and maximum factors for the "data augmentation applied to individual components by multiplication of a random factor"
mult_comp = [MULT_COMP] # switch on (1) or off (0) the "data augmentation by multiplication of a random factor" for the individual components

"""j
derived parameters:
j"""

# normalize the weights for the component and sum loss terms (to avoid a scaling of the learning rate with the number of components/ resulting loss amplitude)
norm_factor = sum(weights_comp + weight_sum)
weights_comp = np.array(weights_comp)/norm_factor 
weights_comp = np.float32(weights_comp)
weight_sum = np.array(weight_sum)/norm_factor
weight_sum = np.float32(weight_sum)

# training years:
array_years = np.arange(first_year,last_year+1)
train_years = []
for year in array_years:
    train_years.append(str(year))


# number of components to be separated => defines the number of filters in the last layer (=> number of channels in the output layer)
num_comp = len(components) 

