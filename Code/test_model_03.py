#!/usr/bin/env python3


"""j
Code plotting results of a trained multi-channel U-Net model for signal separation in spatio-temporal gravity data
j"""
 

"""j
Packages import
j"""
 
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from itertools import combinations
import itertools
import nexusformat.nexus as nx

"""j
Import functions and variables
j"""

import functions as fct
import variables as v

"""j
Define variable model_id, and the folder where to find the SH coefficient files:
j"""

model_id = v.model_id
model_folder = '.'

"""j
specify folder structure:
j"""

# specify location where log files should be saved:
log_dir = model_folder + '/training_results_02/logs/'

# specify location where plots should be saved:
plot_dir = model_folder + '/test_results_03/Figures/'
os.makedirs(plot_dir)

# specify location where predicted data should be saved:
data_dir_02 = model_folder + '/training_results_02/predicted_true_signals/'
data_dir_03 = model_folder + '/test_results_03/data_for_plotting/'
os.makedirs(data_dir_03)


inp_tar_folder = model_folder + '/train_test_data_01/' # folder in which the input-target pairs are stored
inp_tar_file = '_'.join(v.components) + '_' + v.unit + '_in_m_' + str(v.deltat) + 'h_' + v.sampling +  '.hdf5'

"""j
Colors for plotting:
j"""

colors_components = mpl.cm.tab10(range(v.num_comp)) # line colors for individual components curves in plots

"""j
Convert model predictions to lat-lon-t arrays and other arrays useful for plotting, and write resulting data to file_true_pred_arrays:
j"""

years = v.train_years.copy()
years.append(v.validation_year)
years.append(v.test_year) # list of strings giving the years for which data pairs are needed


file_model_pred = data_dir_02 + v.model_id + '_model_predictions.hdf5' 
file_true_pred_arrays = data_dir_03 + v.model_id + '_pred_true_arrays.hdf5'

for year in years:
    fct.true_pred_lat_lon_t_arrays(inp_tar_folder+inp_tar_file,file_model_pred,year,file_true_pred_arrays)

#####

# copy the complete group 'spatial_grid' from the inp_tar_file file to the newly created hdf5 file:
fs = h5py.File(inp_tar_folder+inp_tar_file, 'r')
fd = h5py.File(file_true_pred_arrays, 'a')
fs.copy('spatial_grid', fd)

#####

print('Contents of ' + file_true_pred_arrays + ':')
f = nx.nxload(file_true_pred_arrays)
print(f.tree)

"""j
RMS as fct. of lat, lon, t plots:
j"""

pred_type = 'signal'
filename = plot_dir + 'RMS_lat_lon_t_' + v.test_year + '_' + pred_type + '.png'
fct.rms_lat_lon_t_plot(file_true_pred_arrays,filename,colors_components,pred_type,subplot_label='a)')

pred_type = 'error'
filename = plot_dir + 'RMS_lat_lon_t_' + v.test_year + '_' + pred_type + '.png'
fct.rms_lat_lon_t_plot(file_true_pred_arrays,filename,colors_components,pred_type,subplot_label='b)')

filename = plot_dir + 'RMS_lat_lon_t_' + v.test_year + '_rel_error.png'
fct.relative_rms_error_lat_lon_t_plot(file_true_pred_arrays,filename,colors_components,subplot_label='c)')


"""j
RSS as fct. of SH degree n:
j"""

if v.extent == 'global':
    time_idx = 0

    pred_type ='signal'
    filename = plot_dir + 'RSS_pred_error_' + v.test_year + '_time_step_' + str(time_idx) + '_' + pred_type + '.png'
    fct.RSS_n_plot(file_true_pred_arrays,filename,colors_components,time_idx,pred_type,subplot_label='a)')

    pred_type ='error'
    filename = plot_dir + 'RSS_pred_error_' + v.test_year + '_time_step_' + str(time_idx) + '_' + pred_type + '.png'
    fct.RSS_n_plot(file_true_pred_arrays,filename,colors_components,time_idx,pred_type,subplot_label='b)')

"""j
Spatial plots:
j"""

time_idx = 0

filename = plot_dir + 'Spatial_plot_' + v.test_year + '_time_step_' + str(time_idx) + '.png'
fct.spatial_plots(file_true_pred_arrays,filename,time_idx)

"""j
RMS errors of individual components and sums of components:
j"""

filename = plot_dir +  'RMS_errors_' + v.test_year + '_single_components.png'
fct.rms_errors_combinations(file_true_pred_arrays,filename,[1],subplot_label='a)')

filename = plot_dir +  'RMS_errors_' + v.test_year + '_pairs_of_components.png'
fct.rms_errors_combinations(file_true_pred_arrays,filename,[2,v.num_comp],subplot_label='b)')

filename = plot_dir +  'RMS_errors_' + v.test_year + '_pairs_of_components_normalized.png'
fct.normalized_rms_errors_combinations(file_true_pred_arrays,filename,[2,v.num_comp],subplot_label='c)')

"""j
Training and validation curves as function of epoch:
j"""

filename = plot_dir + 'Train_test_error_curves_incl_signal.png'
fct.train_test_curve_plot(file_true_pred_arrays,log_dir,filename,colors_components)

#####

print('Done!')

#####

# create success file 

f=open('03_success','w')
f.close()

