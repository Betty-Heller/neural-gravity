#!/usr/bin/env python3


"""j
Code creating input-target pairs for the neural network-based framework for signal separation in spatio-temporal gravity data
j"""

"""j
Packages import
j"""

import numpy as np
import pandas as pd
import os
import datetime
from os.path import join
import pyshtools as pysh
import h5py
import nexusformat.nexus as nx
import sys

"""j
Import functions and variables
j"""

import functions as fct
import variables as v

"""j
Check inconsistencies between input parameters:
j"""

if v.sampling == 'lat_lon':
    if any(v.weights_slope_value) or any(v.weights_linear_shape):
        sys.exit('There are no slope and shape loss terms if using lat-lon sampling! Weights need to be zero')

"""j
Define variable model_id, and the folder where to find the SH coefficient files:
j"""
    
model_id = v.model_id
model_folder = '.'
SH_folder = '../../Data/SH_coeff/'

"""j
Create input-target pairs to be used for training, validating and testing, for the data periods needed (i.e. training, validation and test years):

These are written to the subfolder train_test_data_01 in the model folder
j"""

inp_tar_folder = model_folder + '/train_test_data_01/' # folder in which the input-target pairs are stored
os.mkdir(inp_tar_folder)

inp_tar_file = '_'.join(v.components) + '_' + v.unit + '_in_m_' + str(v.deltat) + 'h_' + v.sampling +  '.hdf5'

years = v.train_years.copy()
years.append(v.validation_year)
years.append(v.test_year) # list of strings giving the years for which data pairs are needed

#####

# define lat-lon grids and size of 2^n x 2^n spatial grid:

## global grid...

# find the 2^n x 2^n grid size that is larger than the square grid obtained based on the resolution given by the input SH coefficients
dim_square_orig = (2 * v.Nmax_orig) + 2
for n in range(50):
    dim_global = 2**n
    if dim_square_orig <= 2**n:
        break
Nmax_ext = int((dim_global-2) / 2) # this is the maximum SH degree that is corresponding to the square lat-lon grid dimensions dim_global x dim_global

# latitude, longitude vectors for global dim_global x dim_global lat-lon grid (see pysh.expand.MakeGridDH documentation):
lats_global = 90 - 180/dim_global * np.arange(dim_global) 
lons_global = 0 + 360/dim_global * np.arange(dim_global)


## regional grid... (only needed if v.extent == 'regional') -> cut out a (smaller) 2^n x 2^n region from the global square lat-lon grid

if v.extent == 'regional':

    # user-defined regional grid:
    lat_min = v.lat_interval_regional[0]
    lat_max = v.lat_interval_regional[-1]
    lon_min = v.lon_interval_regional[0]
    lon_max = v.lon_interval_regional[-1]

    # location of user-defined regional grid within the global dim_global x dim_global grid (probably not square and not 2^n x 2^n)
    lat_indices = np.where((lats_global >= lat_min) & (lats_global <= lat_max))
    lon_indices = np.where((lons_global >= lon_min) & (lons_global <= lon_max))
    
    # number of grid points in the global grid falling inside the user-defined regional grid:
    num_lat_regional = len(lat_indices[0])
    num_lon_regional = len(lon_indices[0])

    # determine size of 2^n x 2^n regional grid that can include the complete user-defined regional grid
    dim_old = max(num_lat_regional,num_lon_regional)
    for n in range(50):
        dim_regional = 2**n
        if dim_old <= 2**n:
            break

    # number of grid points to add to each side of the regional latitude and longitude grids for the extension of the regional grid to a dim_regional x dim_regional grid:
    num_lat_to_extend = int( (dim_regional - num_lat_regional) / 2) 
    num_lon_to_extend = int( (dim_regional - num_lon_regional) / 2) 

    # extend the regional grid to be dim_regional x dim_regional (the different cases are needed such that the regional square grid lies fully within the global square grid):
    if lat_indices[0][0] - num_lat_to_extend < 0:
        lat_idx_min_new = 0
        lat_idx_max_new = dim_regional - 1
    elif lat_indices[0][-1] + num_lat_to_extend > dim_global-1:
        lat_idx_min_new = dim_global - dim_regional
        lat_idx_max_new = dim_global - 1
    else:
        lat_idx_min_new = lat_indices[0][0] - num_lat_to_extend
        lat_idx_max_new = lat_idx_min_new + dim_regional - 1
        
    lat_indices_square = np.arange(lat_idx_min_new,lat_idx_max_new + 1) # +1 since arange always skips the last index

    if lon_indices[0][0] - num_lon_to_extend < 0:
        lon_idx_min_new = 0
        lon_idx_max_new = dim_regional - 1
    elif lon_indices[0][-1] + num_lon_to_extend > dim_global -1:
        lon_idx_min_new = dim_global - dim_regional
        lon_idx_max_new = dim_global - 1
    else:
        lon_idx_min_new = lon_indices[0][0] - num_lon_to_extend
        lon_idx_max_new = lon_idx_min_new + dim_regional - 1

    lon_indices_square = np.arange(lon_idx_min_new,lon_idx_max_new + 1) # +1 since arange always skips the last index

    lats_regional = lats_global[lat_indices_square]
    lons_regional = lons_global[lon_indices_square]

#####

# dimension of the square samples to work with:
if v.extent == 'global':
    dim = dim_global
elif v.extent == 'regional':
    dim = dim_regional

#####

# if random samples should be created, create a matrix containing the standard normally distributed weights used for linearly combining the samples to form random samples (number of new samples is 2*dim)

if v.random_samples == 'yes':
    G_psi_samp = np.random.normal(size = (2*dim,dim)) # psi = index for newly created, random samples; samp = index for original samples

#####

print('Create training and testing data pairs...')

## loop over all periods, create lat-lon square grids, concatenate to lat-lon-t and form input-target pairs:

for year in years:
    print(year)
    lat_lon_t_list = []

    for comp in v.components:
        print(comp)

        folder_in = SH_folder + comp + '/' + str(v.deltat) + 'h' + v.latlont_string + '/' + year + '/'

        t_lat_lon = []

        files = fct.file_list(folder_in)

        for file_in in files:

            cilm, gm, r0 = pysh.shio.read_icgem_gfc(folder_in + file_in) # read in SH coefficients

            cilm = fct.cilm_Gauss(cilm,v.Nmax_flt) # Gauss filtering

            if v.unit == 'EWH':
                cilm = fct.cilm_2_EWH(cilm) # convert to EWH

            cilm_ext = fct.fill_up_SH_coeff(cilm,v.Nmax_orig,Nmax_ext) # extend spectrum with zeros to larger maximum d/o
            lat_lon_square = pysh.expand.MakeGridDH(cilm_ext) # expand to dim_global x dim_global grid

            if v.extent == 'regional':
                # cut out the dim_regional x dim_regional regional square grid from the global square grid
                lat_lon_square = lat_lon_square[lat_idx_min_new:lat_idx_max_new+1 ,lon_idx_min_new:lon_idx_max_new+1]     
            
            t_lat_lon.append(lat_lon_square) # collect lat-lon arrays of subsequent time points

        lat_lon_t_array = np.transpose(np.array(t_lat_lon),(1,2,0)) 
     
        lat_lon_t_list.append(lat_lon_t_array) # collect lat-lon-t arrays of various components

    # make sure that the number of time steps in the considered year is the same for all components (reduce it if necessary):
    numt_list = [lat_lon_t_list[i].shape[2] for i in range(v.num_comp)]
    numt = min(numt_list)
    lat_lon_t_list_new = [lat_lon_t_list[i][:,:,:numt] for i in range(v.num_comp)]
    comp_lat_lon_t_array = np.array(lat_lon_t_list_new) # component-lat-lon-t array

    # interpolate 1-d time series for all spatial grid points, to obtain a lat-lon-t cube:
    tvec_orig = np.arange(numt) # vector for original time series
    tvec_dim = np.linspace(0,numt,num=dim) # vector for interpolated time series
    comp_lat_lon_t_square = np.zeros([v.num_comp,dim,dim,dim])

    for comp_idx in range(v.num_comp):
        for lat in range(dim):
            for lon in range(dim):
                ts = comp_lat_lon_t_array[comp_idx,lat,lon,:]
                comp_lat_lon_t_square[comp_idx,lat,lon,:] = np.interp(tvec_dim, tvec_orig, ts) # shape: v.num_comp x dim x dim x dim

    if v.sampling == 'lat_lon':
        samp_row_col_comp = np.transpose(comp_lat_lon_t_square,(3,1,2,0))
    elif v.sampling == 'lat_t':
        samp_row_col_comp = np.transpose(comp_lat_lon_t_square,(2,1,3,0))
    elif v.sampling == 't_lon':
        samp_row_col_comp = np.transpose(comp_lat_lon_t_square,(1,3,2,0))

    if v.random_samples == 'yes':
        psi_row_col_comp = np.tensordot(G_psi_samp,samp_row_col_comp,1) # random linear combinations of the original samples are formed
        samp_row_col_comp = psi_row_col_comp # overwrite samples (new shape: 2*dim x dim x dim x v.num_comp)

    samp_row_col_summed = np.sum(samp_row_col_comp,axis=3)

    # create time axis of length numt (to be able to assign samples to absolute points in time later):
    start = datetime.datetime(year=int(files[0][-27:-23]),month=int(files[0][-23:-21]),day=int(files[0][-21:-19]),minute=int(files[0][-18:-16])) 
    time_interval = datetime.timedelta(hours=v.deltat)
    start = start + time_interval / 2 # first reference point in time is the middle of the first time interval
    t_axis = start + time_interval * np.arange(numt)

    # count the decimal number of days elapsed since 01.01. 00:00 of the respective year
    DOY_axis = [float(t.strftime('%j')) - 1.0 + float(t.strftime('%H'))/24.0 for t in t_axis] # j = day of the year, H = hour of the day

    # in case the last DOY is already in the next year, add 365 (or 366) days to it, to avoid plotting problems later:
    if DOY_axis[-1] < DOY_axis[-2]:
        numdays = pd.Timestamp(int(year),12,31).dayofyear # number of days in the considered year
        DOY_axis[-1] += numdays

    print('write hdf5 file...')
    # write the complete data of the year "year" as new dataset to the hdf5 file:
    with h5py.File(inp_tar_folder+inp_tar_file, "a") as f: # "a" means: read/write file if exists, create otherwise
        group_year = f.create_group(year) # create a group of name "year"
        group_year.create_dataset('input',samp_row_col_summed.shape,dtype=float)
        group_year['input'][:,:,:] = samp_row_col_summed
        group_year.create_dataset('target',samp_row_col_comp.shape,dtype=float)
        group_year['target'][:,:,:,:] = samp_row_col_comp
        group_year.attrs['numt'] = numt # number of original time steps in that year (before interpolation to dim steps)
        group_year.attrs['DOY_axis'] = DOY_axis # time axis in DOY (giving the DOY of the middle/reference points of each time period)

# compute inverse of the G matrix (needed for going back from the random samples to the original ones):
if v.random_samples == 'yes':
    G_inv = np.linalg.pinv(G_psi_samp)
    del G_psi_samp

# write variables independent of the year to the hdf5 file:
with h5py.File(inp_tar_folder+inp_tar_file, "a") as f:
    attributes = f.create_group('spatial_grid')
    attributes.attrs['lats_global'] = lats_global
    attributes.attrs['lons_global'] = lons_global
    attributes.attrs['dim_global'] = dim_global
    attributes.attrs['dim'] = dim

    if v.extent == 'regional':
        attributes.attrs['lats_regional'] = lats_regional
        attributes.attrs['lons_regional'] = lons_regional
        attributes.attrs['dim_regional'] = dim_regional

    if v.random_samples == 'yes':
        new_group = f.create_group('inverse_of_random_weights_matrix')
        new_group.create_dataset('G_inv',dtype=float,data=G_inv)
        del G_inv

#####

f = nx.nxload(inp_tar_folder+inp_tar_file)
print(f.tree)


#####

# create success file

f=open('01_success','w')
f.close()

