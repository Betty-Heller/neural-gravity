#!/usr/bin/env python3


"""j
Functions used by create_training_data_01.py and test_model_03.py
j"""


"""j
Packages import
j"""

import matplotlib as mpl
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyshtools as pysh
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from itertools import combinations
import itertools


"""j
Import variables
j"""

import variables as v

"""j
Functions:
j"""

# function that creates a list of files in folder:

def file_list(folder):
    tmp = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    files = sorted(tmp) # sort files according to their file names
    return files

#####

# function to apply a gaussian filter to SH coefficients (Nmax_flt defines the filter radius)

def cilm_Gauss(cilm,Nmax_flt):

    Nmax_signal = cilm.shape[1]-1

    # construct the Gauss filter (n-dependent):
    factor = 2
    gauss_n = Gauss_coeff(Nmax_flt,Nmax_signal,factor)
    gauss_n_m = np.transpose(np.tile(gauss_n,(Nmax_signal+1,1))) # first index = n, second index = m

    # filter the coefficients (do not cut!):
    Cnm = np.multiply(cilm[0],gauss_n_m) # cilm[0] = dimensionless Cnm coefficients (C_nm = cilm[0,n,m])
    Snm = np.multiply(cilm[1],gauss_n_m) # cilm[1] = dimensionless Snm coefficients (S_nm = cilm[1,n,m])
    
    cilm_flt = np.array([Cnm,Snm]) # SH coefficients in units of m EWH or m geoid height

    return cilm_flt

#####

# function to compute Gauss filter in the SH domain (degree-dependent), according to Wahr et al. (1998), Eq. 32-34 (referring to Jekeli, 1981, Eq. 59)

# factor: e.g. 1, 2 or 3
# Nmax_flt = maximum degree defining the filter radius

def Gauss_coeff(Nmax_flt,Nmax_glob,factor):
    R = 6378.137 # Earth radius in km
    flt_rad = 20000/Nmax_flt/factor # radius of Gauss filter in km, in the space domain

    b = np.log(2.)/(1-np.cos(flt_rad/R))

    sgc = np.zeros([Nmax_glob+1])
    sgc[0] = 1
    sgc[1] = (1+np.exp(-2*b))/(1-np.exp(-2*b)) - 1/b

    for n in np.arange(2,Nmax_glob+1):
        sgc[n] = -(2*n-1)/b * sgc[n-1] + sgc[n-2]
        if sgc[n]>sgc[n-1] or sgc[n]<0:
            sgc[n] = 0
            break

    return sgc

#####

# function that converts dimensionless SH coefficients to units of m EWHs, according to Wahr et al. (1998)

def cilm_2_EWH(cilm):

    Nmax = cilm.shape[1]-1

    # compute elastic load Love numbers:
    k_n = love_nr(Nmax)
    n = np.arange(Nmax+1)
    factor_EWH_n = (2*n+1)/(1+k_n) * v.R * v.rho_ave/3/v.rho_w
    factor_EWH_n_m = np.transpose(np.tile(factor_EWH_n,(Nmax+1,1))) # first index = n, second index = m

    # convert the dimensionless SH coefficients to units of m EWH:
    Cnm = np.multiply(cilm[0],factor_EWH_n_m) # cilm[0] = dimensionless Cnm coefficients (C_nm = cilm[0,n,m])
    Snm = np.multiply(cilm[1],factor_EWH_n_m) # cilm[1] = dimensionless Snm coefficients (S_nm = cilm[1,n,m])
    
    cilm_unit = np.array([Cnm,Snm]) # SH coefficients in units of m EWH or m geoid height

    return cilm_unit

#####

# function to compute the (degree-dependent) elastic Love numbers up to SH degree Nmax, for the Earth Model PREM, according to Han and Wahr (1995), Wahr et al. (1998)

def love_nr(Nmax):

    kl_sample_points = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 70, 100, 150, 200])
    kl_sample_values = np.array([0.000, 0.027, -0.303, -0.194, -0.132, -0.104, -0.089, -0.081, -0.076, -0.072, -0.069, -0.064, -0.058, -0.051, -0.040, -0.033, -0.027, -0.020, -0.014, -0.010, -0.007])
    kl_query_points = np.arange(Nmax+1)

    y_interp = interp1d(kl_sample_points, kl_sample_values)
    k_n = y_interp(kl_query_points)

    return k_n

#####

# function that can be used to extend a SH coefficient spectrum to a higher maximum degree, by filling up with zeros

def fill_up_SH_coeff(cilm_orig,Nmax_orig,Nmax_ext):
    
    cilm_ext = np.zeros([2,Nmax_ext+1,Nmax_ext+1])
    cilm_ext[0,:Nmax_orig+1,:Nmax_orig+1] = cilm_orig[0,:,:]
    cilm_ext[1,:Nmax_orig+1,:Nmax_orig+1] = cilm_orig[1,:,:]

    return cilm_ext

#####

# read in true_array, pred_array data from file in data_dir_02 folder and transform to lat-lon-t data and other arrays needed for plotting:

def true_pred_lat_lon_t_arrays(file_datasets,file_true_pred_arrays,year,file_out):

    # read in true and pred. array data (written by function true_pred_arrays in train_model_02.py)
    with h5py.File(file_true_pred_arrays, "r") as f: # "a" means: read/write file if exists, create otherwise
        group_year = f[year]
        pred_array = group_year['pred_array'][:,:,:,:]
        true_array = group_year['true_array'][:,:,:,:]
        dim = group_year.attrs['dim']
        numt = group_year.attrs['numt']
        DOY_axis = group_year.attrs['DOY_axis'][:]

    ### SUM OF COMPONENTS ###

    # transpose to get (dim x dim x dim) lat-lon-t arrays:
    pred_array_lat_lon_t = samp_row_col_2_lat_lon_t(np.sum(pred_array,axis=3),v.sampling) 
    true_array_lat_lon_t = samp_row_col_2_lat_lon_t(np.sum(true_array,axis=3),v.sampling)

    # interpolate along time axis to retrieve back the original sampling along the time axis => dim x dim x numt
    pred_array_lat_lon_t = interp_t_dim2numt(pred_array_lat_lon_t,numt,dim)
    true_array_lat_lon_t = interp_t_dim2numt(true_array_lat_lon_t,numt,dim)

    # if lat-lon arrays are regional, fill them up to a global grid:
    if v.extent == 'regional': # in this case, dim = dim_regional 
        pred_array_lat_lon_t = lat_lon_t_reg_to_global(file_datasets,pred_array_lat_lon_t) # dim_global x dim_global x numt
        true_array_lat_lon_t = lat_lon_t_reg_to_global(file_datasets,true_array_lat_lon_t)

    # compute RMS signals and errors along individual coordinate directions, for sum of signal components:
    # (have dim_global samples along lat and lon directions, and numt samples along t direction)
    RMS_pred_signal_lat, RMS_pred_signal_lon, RMS_pred_signal_t = RMS_along_dim(pred_array_lat_lon_t)
    RMS_true_signal_lat, RMS_true_signal_lon, RMS_true_signal_t = RMS_along_dim(true_array_lat_lon_t)
    RMS_error_lat, RMS_error_lon, RMS_error_t = RMS_along_dim(pred_array_lat_lon_t-true_array_lat_lon_t)

    # write RMS vectors for sum of signal components as datasets to hdf5 file:
    with h5py.File(file_out, "a") as f:
        group_year = f.create_group(year)
        subgroup_sum = group_year.create_group('sum')

        subgroup_sum.create_dataset('RMS_pred_signal_lat',dtype=float,data = RMS_pred_signal_lat)
        subgroup_sum.create_dataset('RMS_pred_signal_lon',dtype=float, data = RMS_pred_signal_lon)
        subgroup_sum.create_dataset('RMS_pred_signal_t',dtype=float, data = RMS_pred_signal_t)

        subgroup_sum.create_dataset('RMS_true_signal_lat',dtype=float,data = RMS_true_signal_lat)
        subgroup_sum.create_dataset('RMS_true_signal_lon',dtype=float, data = RMS_true_signal_lon)
        subgroup_sum.create_dataset('RMS_true_signal_t',dtype=float, data = RMS_true_signal_t)

        subgroup_sum.create_dataset('RMS_error_lat',dtype=float,data = RMS_error_lat)
        subgroup_sum.create_dataset('RMS_error_lon',dtype=float, data = RMS_error_lon)
        subgroup_sum.create_dataset('RMS_error_t',dtype=float, data = RMS_error_t)

    # compute degree amplitude (root sum of squares) vector for the sum of the signal components as function of time index and SH degree:
    if v.extent == 'global': # only attempt to convert spatial grid to SH coefficients if the grid is of global extent
        numt = pred_array_lat_lon_t.shape[2]
        dim_global = pred_array_lat_lon_t.shape[0]
        Nmax_ext = int((dim_global-2) / 2)
        RSS_pred_signal_t_n = np.zeros([numt,Nmax_ext+1])
        RSS_true_signal_t_n = np.zeros([numt,Nmax_ext+1])
        RSS_error_t_n = np.zeros([numt,Nmax_ext+1])

        for t_idx in range(numt): # loop over time indices

            # compute SH coeff. for predicted sum of signals:
            cilm_pred = pysh.expand.SHExpandDH(pred_array_lat_lon_t[:,:,t_idx]) # directly convert from square grid to SH coefficients
            RSS_pred_signal_t_n[t_idx,:] = RSS_from_cilm(cilm_pred)

            # compute SH coeff. for true sum of signals:
            cilm_true = pysh.expand.SHExpandDH(true_array_lat_lon_t[:,:,t_idx])
            RSS_true_signal_t_n[t_idx,:] = RSS_from_cilm(cilm_true)

            RSS_error_t_n[t_idx,:] = RSS_from_cilm(cilm_pred - cilm_true)

        with h5py.File(file_out, "a") as f:
            group_year = f[year]
            subgroup_sum = group_year['sum']
            subgroup_sum.create_dataset('RSS_pred_signal_t_n',dtype=float,data = RSS_pred_signal_t_n)
            subgroup_sum.create_dataset('RSS_true_signal_t_n',dtype=float, data = RSS_true_signal_t_n)
            subgroup_sum.create_dataset('RSS_error_t_n',dtype=float, data = RSS_error_t_n)


    ### INDIVIDUAL SIGNAL COMPONENTS ###

    # loop over components:
    for comp in range(v.num_comp):
        # transpose to get (dim x dim x dim) lat-lon-t arrays:
        pred_array_lat_lon_t = samp_row_col_2_lat_lon_t(pred_array[:,:,:,comp],v.sampling)
        true_array_lat_lon_t = samp_row_col_2_lat_lon_t(true_array[:,:,:,comp],v.sampling)

        # interpolate along time axis to retrieve back the original sampling along the time axis => dim x dim x numt
        pred_array_lat_lon_t = interp_t_dim2numt(pred_array_lat_lon_t,numt,dim)
        true_array_lat_lon_t = interp_t_dim2numt(true_array_lat_lon_t,numt,dim)

        # if lat-lon arrays are regional, fill them up to a global grid:
        if v.extent == 'regional': # in this case, dim = dim_regional 
            pred_array_lat_lon_t = lat_lon_t_reg_to_global(file_datasets,pred_array_lat_lon_t) # dim_global x dim_global x numt
            true_array_lat_lon_t = lat_lon_t_reg_to_global(file_datasets,true_array_lat_lon_t)

        # compute RMS signals and errors along individual coordinate directions:
        # (have dim_global samples along lat and lon directions, and numt samples along t direction)
        RMS_pred_signal_lat, RMS_pred_signal_lon, RMS_pred_signal_t = RMS_along_dim(pred_array_lat_lon_t)
        RMS_true_signal_lat, RMS_true_signal_lon, RMS_true_signal_t = RMS_along_dim(true_array_lat_lon_t)
        RMS_error_lat, RMS_error_lon, RMS_error_t = RMS_along_dim(pred_array_lat_lon_t-true_array_lat_lon_t)


        # write lat-lon-t arrays to hdf5 files
        print('write predicted ' + v.components[comp] + ' data to hdf5 file ' + file_out + '...')
        with h5py.File(file_out, "a") as f: # "a" means: read/write file if exists, create otherwise
            group_year = f[year]
            subgroup_comp = group_year.create_group(v.components[comp])

            subgroup_comp.create_dataset('pred_signal_lat_lon_t',dtype=float, data = pred_array_lat_lon_t)
            subgroup_comp.create_dataset('true_signal_lat_lon_t',dtype=float, data = true_array_lat_lon_t)

            subgroup_comp.create_dataset('RMS_pred_signal_lat',dtype=float,data = RMS_pred_signal_lat)
            subgroup_comp.create_dataset('RMS_pred_signal_lon',dtype=float, data = RMS_pred_signal_lon)
            subgroup_comp.create_dataset('RMS_pred_signal_t',dtype=float, data = RMS_pred_signal_t)

            subgroup_comp.create_dataset('RMS_true_signal_lat',dtype=float,data = RMS_true_signal_lat)
            subgroup_comp.create_dataset('RMS_true_signal_lon',dtype=float, data = RMS_true_signal_lon)
            subgroup_comp.create_dataset('RMS_true_signal_t',dtype=float, data = RMS_true_signal_t)

            subgroup_comp.create_dataset('RMS_error_lat',dtype=float,data = RMS_error_lat)
            subgroup_comp.create_dataset('RMS_error_lon',dtype=float, data = RMS_error_lon)
            subgroup_comp.create_dataset('RMS_error_t',dtype=float, data = RMS_error_t)

        if v.extent == 'global': # only attempt to convert spatial grid to SH coefficients if the grid is of global extent

            # compute degree amplitude (root sum of squares) vector for the individual signal components as function of time index and SH degree:
            RSS_pred_signal_t_n = np.zeros([numt,Nmax_ext+1])
            RSS_true_signal_t_n = np.zeros([numt,Nmax_ext+1])
            RSS_error_t_n = np.zeros([numt,Nmax_ext+1])

            # compute SH coefficients of individual signal components as function of time index, SH degree and order:
            pred_signal_t_n_m = np.zeros([numt,Nmax_ext+1,2*Nmax_ext+1])
            true_signal_t_n_m = np.zeros([numt,Nmax_ext+1,2*Nmax_ext+1])

            for t_idx in range(numt): # loop over time indices
                # compute SH coeff. for predicted signal component:
                cilm_pred = pysh.expand.SHExpandDH(pred_array_lat_lon_t[:,:,t_idx]) # directly convert from square grid to SH coefficients
                xv,yv,coeff_pred = triangle_plot_params(cilm_pred)
                RSS_pred_signal_t_n[t_idx,:] = RSS_from_cilm(cilm_pred)

                # compute SH coeff. for true sum of signals:
                cilm_true = pysh.expand.SHExpandDH(true_array_lat_lon_t[:,:,t_idx])
                xv,yv,coeff_true = triangle_plot_params(cilm_true)
                RSS_true_signal_t_n[t_idx,:] = RSS_from_cilm(cilm_true)

                RSS_error_t_n[t_idx,:] = RSS_from_cilm(cilm_pred - cilm_true)
                coeff_error = coeff_pred - coeff_true

                pred_signal_t_n_m[t_idx,:,:] = coeff_pred
                true_signal_t_n_m[t_idx,:,:] = coeff_true

            print('write predicted ' + v.components[comp] + ' data to hdf5 file ' + file_out + '...')
            with h5py.File(file_out, "a") as f: # "a" means: read/write file if exists, create otherwise
                group_year = f[year]
                subgroup_comp = group_year[v.components[comp]]

                subgroup_comp.create_dataset('RSS_pred_signal_t_n',dtype=float,data = RSS_pred_signal_t_n)
                subgroup_comp.create_dataset('RSS_true_signal_t_n',dtype=float, data = RSS_true_signal_t_n)
                subgroup_comp.create_dataset('RSS_error_t_n',dtype=float, data = RSS_error_t_n)

                subgroup_comp.create_dataset('pred_signal_t_n_m',dtype=float,data = pred_signal_t_n_m)
                subgroup_comp.create_dataset('true_signal_t_n_m',dtype=float, data = true_signal_t_n_m)
    # write numt and DOY_axis to group "year", as attributes
    with h5py.File(file_out, "a") as f: # "a" means: read/write file if exists, create otherwise
        group_year = f[year]
        group_year.attrs['numt'] = numt # number of original time steps in that year (before interpolation to dim steps)
        group_year.attrs['DOY_axis'] = DOY_axis # time axis in DOY (giving the DOY of the middle/reference points of each time period)


    # write out structure of newly created hdf5 file:
    print('Done with writing lat-lon-t data for year ' + year + ' to hdf5 file!')

#####

# function that converts a samp-row-col array to a lat-lon-t-array

def samp_row_col_2_lat_lon_t(samp_row_col,sampling):    
    
    if sampling == 'lat_lon':
        lat_lon_t = np.transpose(samp_row_col,(1,2,0))
    elif sampling == 'lat_t':
        lat_lon_t = np.transpose(samp_row_col,(1,0,2))
    elif sampling == 't_lon':
        lat_lon_t = np.transpose(samp_row_col,(0,2,1))

    return lat_lon_t

#####

# function that interpolates a dim x dim x dim lat-lon-t array along the time direction 
# to obtain a dim x dim x numt array (used by function true_pred_arrays)

def interp_t_dim2numt(lat_lon_t,numt,dim):
    
    tvec_numt = np.arange(numt) 
    tvec_dim = np.linspace(0,numt,num=dim) 

    lat_lon_t_new = np.zeros([dim,dim,numt])

    # interpolate 1-d time series for all spatial grid points:
    for lat in range(dim):
        for lon in range(dim):
            ts = lat_lon_t[lat,lon,:]
            lat_lon_t_new[lat,lon,:] = np.interp(tvec_numt, tvec_dim, ts) # shape: dim x dim x numt

    return lat_lon_t_new

#####

# function that converts a lat-lon-t array from regional to global (fill up with zeros), 
# using the infos given by the file containing the input-target pairs (used by function true_pred_arrays)

def lat_lon_t_reg_to_global(file_datasets,lat_lon_t_regional):

    with h5py.File(file_datasets,"r") as f:
        group = f['spatial_grid']
        lats_global = group.attrs['lats_global'][:]
        lons_global = group.attrs['lons_global'][:]
        dim_global = group.attrs['dim_global']
        lats_regional = group.attrs['lats_regional'][:]
        lons_regional = group.attrs['lons_regional'][:]
    
    numt = lat_lon_t_regional.shape[2]

    lat_lon_t_global = np.zeros([dim_global,dim_global,numt])

    lat_idx_0 = np.where(lats_global == lats_regional[0])[0][0]
    lat_idx_1 = np.where(lats_global == lats_regional[-1])[0][0]

    lon_idx_0 = np.where(lons_global == lons_regional[0])[0][0]
    lon_idx_1 = np.where(lons_global == lons_regional[-1])[0][0]

    lat_lon_t_global[lat_idx_0:lat_idx_1+1,lon_idx_0:lon_idx_1+1,:] = lat_lon_t_regional
    
    return lat_lon_t_global

#####

# function that takes a lat-lon-time array and computes 1-d numpy arrays giving the RMS values along the individual axes 

def RMS_along_dim(lat_lon_t):
    RMS_lon_t = rms(lat_lon_t,0)
    RMS_t = rms(RMS_lon_t,0)
    RMS_lon = rms(RMS_lon_t,1)

    RMS_lat_lon = rms(lat_lon_t,2)
    RMS_lat = rms(RMS_lat_lon,1)
    
    return RMS_lat, RMS_lon, RMS_t

#####

# function providing the variables needed to do a triangle plot (used by function triangle_plot)

# xv = (Nmax+1) x (Nmax+2) matrix containing the x axis values (SH orders) of the triangle plot 
# yv = (Nmax+1) x (Nmax+2) matrix containing the y axis values (SH degrees) of the triangle plot 
# coeff = (Nmax+1) x (Nmax+2) matrix containing the coefficient triangle (rows = degrees, columns = neg.+pos. orders)

def triangle_plot_params(cilm):

    # build coefficient array to be plotted:
    #cilm = x.to_array()
    Cnm = cilm[0]
    Snm = cilm[1]
    Snm_triangle = np.fliplr(Snm[:,1:])
    coeff = np.concatenate((Snm_triangle,Cnm),axis=1)

    # build x and y axes for the triangle plot:
    #nmax = x.lmax
    nmax = cilm.shape[1]
    n = np.arange(nmax+1)
    m1 = np.arange(nmax+1)
    m2 = np.flip(m1[1:])*(-1)
    m = np.concatenate((m2,m1),axis=0)
    xv, yv = np.meshgrid(m, n, indexing='xy')

    return xv, yv, coeff

#####

# function that computes RSS curves from SH coefficients (used by function true_pred_arrays)

def RSS_from_cilm(cilm):

    xv,yv,coeff_retrieved = triangle_plot_params(cilm)

    Nmax = cilm.shape[1]-1

    RSS = np.zeros([Nmax+1,])

    # compute degree amplitudes (Root-sum-of-squares):
    for n in range(Nmax+1): # loop over degrees from 0 to Nmax
        RSS[n] = np.sqrt(np.sum(np.square(coeff_retrieved[n,Nmax-n:Nmax+n+1])))

    return RSS

#####

# function that builds the RMS of an array along a specific dimension (or, if axis is not specified, of all numbers in the array)

def rms(x, axis=None): # axis specifies which dimension should "disappear" after building the rms
    return np.sqrt(np.mean(x**2, axis=axis))

#####

# function creating a 3-panel plot showing the rms-averaged signals and errors along 2 of the axes latitude, longitude and time, as a function of the remaining axis

def rms_lat_lon_t_plot(file_prediction,filename,colors_components,pred_type,subplot_label):

    interval = [0, 0.1]

    labels_fontsize = 16
    ticklabels_fontsize = 14

    numdays = pd.Timestamp(int(v.test_year),12,31).dayofyear # number of days in the considered year

    # extract DOY_axis from group "year" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group_year = f[v.test_year]
        DOY_axis = group_year.attrs['DOY_axis'][:]

    # extract lons_global, lats_global from group "spatial_grid" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group = f['spatial_grid']
        lons_global = group.attrs['lons_global'][:]
        lats_global = group.attrs['lats_global'][:]

    # distinguish whether prediction errors or the predicted signals will be plotted along the true signals:
    if pred_type == 'error':
        RMS_str = 'RMS_error'
        line_style = ':'
    elif pred_type == 'signal':
        RMS_str = 'RMS_pred_signal'
        line_style = '--'

    fig, axs = plt.subplots(1,3, figsize=(15, 5))

    ### "RMS as function of latitude" plot: ###

    # subplot 1: RMS_lat
    plt.subplot(131)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_lat'][:]
            RMS_pred = subgroup_comp[RMS_str+'_lat'][:]

        plt.plot(lats_global,RMS_true,'-',color=colors_components[comp],label='true '+v.components[comp])
        plt.plot(lats_global,RMS_pred,line_style,color=colors_components[comp],label='pred. '+pred_type +' '+v.components[comp])

    plt.xlabel('latitude in °', fontsize=labels_fontsize)
    plt.ylabel('m EWH', fontsize=labels_fontsize)
    plt.ylim(interval)
    plt.xlim([-90,90])
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.grid()

    ### "RMS as function of longitude" plot: ###

    # subplot 2: RMS_lon
    plt.subplot(132)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_lon'][:]
            RMS_pred = subgroup_comp[RMS_str+'_lon'][:]

        plt.plot(lons_global,RMS_true,'-',color=colors_components[comp],label='true '+v.components[comp])
        plt.plot(lons_global,RMS_pred,line_style,color=colors_components[comp],label='pred. '+pred_type +' '+v.components[comp])

    plt.xlabel('longitude in °', fontsize=labels_fontsize)
    plt.ylabel('m EWH', fontsize=labels_fontsize)
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.ylim(interval)
    plt.xlim([0,360])
    plt.grid()

    ### "RMS as function of time" plot: ###

    # subplot 3: RMS_t
    plt.subplot(133)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_t'][:]
            RMS_pred = subgroup_comp[RMS_str+'_t'][:]

        plt.plot(DOY_axis,RMS_true,'-',color=colors_components[comp],label='true '+v.components[comp])
        plt.plot(DOY_axis,RMS_pred,line_style,color=colors_components[comp],label='pred. '+pred_type +' '+v.components[comp])

    plt.xlabel('DOY in '+v.test_year, fontsize=labels_fontsize)
    plt.ylabel('m EWH', fontsize=labels_fontsize)
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.xlim([1,numdays])
    plt.ylim(interval)
    plt.grid()


    if pred_type == 'signal':
        plt.suptitle(subplot_label + ' RMS of true and predicted signals in lat, lon and t direction',fontsize=18)
    elif pred_type == 'error':
        plt.suptitle(subplot_label + ' RMS of true signal and prediction error in lat, lon and t direction',fontsize=18)

    # introduce enough distance between the subplots:
    plt.tight_layout() 

    fig.subplots_adjust(bottom=0.25) 
    plt.legend(bbox_to_anchor=(-0.75,-0.25),loc='upper center',ncol=4,fontsize=labels_fontsize)
    
    plt.savefig(filename,bbox_inches='tight')

#####

# function similar to rms_lat_lon_t_plot, but showing the relative errors as function of latitude, longitude and time

def relative_rms_error_lat_lon_t_plot(file_prediction,filename,colors_components,subplot_label):

    interval = [0, 2]

    labels_fontsize = 16
    ticklabels_fontsize = 14

    numdays = pd.Timestamp(int(v.test_year),12,31).dayofyear # number of days in the considered year

    # extract DOY_axis from group "year" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group_year = f[v.test_year]
        DOY_axis = group_year.attrs['DOY_axis'][:]

    # extract lons_global, lats_global from group "spatial_grid" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group = f['spatial_grid']
        lons_global = group.attrs['lons_global'][:]
        lats_global = group.attrs['lats_global'][:]

    line_style = '-'


    fig, axs = plt.subplots(1,3, figsize=(15, 5))

    ### "RMS as function of latitude" plot: ###

    # subplot 1: RMS_lat
    plt.subplot(131)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_lat'][:]
            RMS_pred = subgroup_comp['RMS_error_lat'][:]

        plt.plot(lats_global,RMS_pred/RMS_true,line_style,color=colors_components[comp])
        plt.axhline(y=1, color='k', linestyle='--')

    plt.xlabel('latitude in °', fontsize=labels_fontsize)
    plt.ylabel('RMS error / RMS signal', fontsize=labels_fontsize)
    plt.ylim(interval)
    plt.xlim([-90,90])
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.grid()

    ### "RMS as function of longitude" plot: ###

    # subplot 2: RMS_lon
    plt.subplot(132)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_lon'][:]
            RMS_pred = subgroup_comp['RMS_error_lon'][:]

        plt.plot(lons_global,RMS_pred/RMS_true,line_style,color=colors_components[comp])
        plt.axhline(y=1, color='k', linestyle='--')

    plt.xlabel('longitude in °', fontsize=labels_fontsize)
    plt.ylabel('RMS error / RMS signal', fontsize=labels_fontsize)
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.ylim(interval)
    plt.xlim([0,360])
    plt.grid()

    ### "RMS as function of time" plot: ###

    # subplot 3: RMS_t
    plt.subplot(133)

    # loop over components:
    for comp in range(len(v.components)):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_t'][:]
            RMS_pred = subgroup_comp['RMS_error_t'][:]

        plt.plot(DOY_axis,RMS_pred/RMS_true,line_style,color=colors_components[comp],label=v.components[comp]+ ' (RMS error/RMS signal=' + '%.2f'%(rms(RMS_pred)/rms(RMS_true)) + ')')
        plt.axhline(y=1, color='k', linestyle='--')
        
    plt.xlabel('DOY in '+v.test_year, fontsize=labels_fontsize)
    plt.ylabel('RMS error / RMS signal', fontsize=labels_fontsize)
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.xlim([1,numdays])
    plt.ylim(interval)
    plt.grid()


    plt.suptitle(subplot_label+' relative RMS prediction error in lat, lon and t direction',fontsize=18)

    # introduce enough distance between the subplots:
    plt.tight_layout() 

    fig.subplots_adjust(bottom=0.25) 
    plt.legend(bbox_to_anchor=(-0.75,-0.25),loc='upper center',ncol=2,fontsize=labels_fontsize) 
    
    plt.savefig(filename,bbox_inches='tight')

#####

# function creating a degree amplitude plot, showing the root sum of squares (RSS) values of signals and errors as function of spherical harmonic degree

def RSS_n_plot(file_prediction,filename,colors_components,time_idx,pred_type,subplot_label):


    Nmax = 120
    y_range = [1e-5,1e-1]

    labels_fontsize = 16
    ticklabels_fontsize = 14


    # distinguish whether prediction errors or the predicted signals will be plotted along the true signals:
    if pred_type == 'error':
        RMS_str = 'RSS_error'
        line_style = ':'
    elif pred_type == 'signal':
        RMS_str = 'RSS_pred_signal'
        line_style = '--'


    plt.figure()

    for comp in range(len(v.components)):
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            RSS_true_n = subgroup_comp['RSS_true_signal_t_n'][time_idx,:]
            plt.plot(RSS_true_n,'-',color=colors_components[comp],label='true ' + v.components[comp])
            RSS_pred_n = subgroup_comp[RMS_str + '_t_n'][time_idx,:]
            plt.plot(RSS_pred_n,line_style,color=colors_components[comp],label='pred. '+pred_type +' '+v.components[comp])

    # further plotting specifications:
    plt.grid()
    plt.yscale("log")  # plot on logarithmic scale
    plt.xlabel('SH degree', fontsize=labels_fontsize)
    plt.xlim([0,Nmax])
    plt.ylabel('log(EWH in m)', fontsize=labels_fontsize)
    plt.ylim(y_range)
    plt.legend(bbox_to_anchor=(1,-0.15),loc='upper right',ncol=2,fontsize=labels_fontsize)
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)

    plt.gca().text(8, 5e-2, subplot_label,fontsize=18, va='top', ha='left')

    if pred_type == 'signal':
        plt.title('RSS of true and predicted signals for time step '+ str(time_idx))
    elif pred_type == 'error':
        plt.title('RSS of true signal and prediction error for time step '+ str(time_idx))

    plt.savefig(filename,bbox_inches='tight')

#####

# function creating spatial plots of signals and errors for a fixed point in time

def spatial_plots(file_prediction,filename,time_idx):

    fontsize_titles=20
    fontsize_colorbar=16
    fontsize_cbar_ticks=12

    vmin = -0.1
    vmax = 0.1
    deltav = 0.05
    vmin_err = -0.1
    vmax_err = 0.1
    deltav_err = 0.05

    # extract numt from group "year" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group_year = f[v.test_year]
        numt = group_year.attrs['numt']

    # extract lons_global, lats_global from group "spatial_grid" of file_prediction
    with h5py.File(file_prediction,"r") as f:
        group = f['spatial_grid']
        lons_global = group.attrs['lons_global'][:]
        lats_global = group.attrs['lats_global'][:]
        dim_global = group.attrs['dim_global']

    # global (dim_global x dim_global) grid
    lonsGeoid, latsGeoid = np.meshgrid(lons_global, lats_global) 

    # extract predicted and true coordinate grids from hdf5 file for the selected time point:
    lat_lon_pred = np.zeros([dim_global,dim_global,v.num_comp])
    lat_lon_true = np.zeros([dim_global,dim_global,v.num_comp])
    for comp in range(v.num_comp):
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[v.test_year+'/'+v.components[comp]]
            lat_lon_pred[:,:,comp] = subgroup_comp['pred_signal_lat_lon_t'][:,:,time_idx]
            lat_lon_true[:,:,comp] = subgroup_comp['true_signal_lat_lon_t'][:,:,time_idx]

    # colorbar range specifications for signal:
    CNT_LINES = np.linspace(vmin, vmax, 101) # if we want more descrete colors, we need to increase the 101 number. 
    norm = mpl.colors.Normalize(vmax = vmax, vmin = vmin)


    # number of panels per row:
    ncol = v.num_comp + 1

    fig = plt.figure(figsize = (4*ncol,8))

    ### true component signals (first row of panels): ###
    for comp in range(v.num_comp):
        ax = fig.add_subplot(3,ncol,comp+1,projection=ccrs.Robinson(central_longitude = 0))
        im = ax.contourf(lonsGeoid,latsGeoid,lat_lon_true[:,:,comp], CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        ax.coastlines()
        plt.title('true '+v.components[comp],fontsize=fontsize_titles)

    ### true sum of components (first row of panels): ###
    ax = fig.add_subplot(3,ncol,ncol,projection=ccrs.Robinson(central_longitude = 0))
    im = ax.contourf(lonsGeoid,latsGeoid,np.sum(lat_lon_true,axis=2), CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    ax.coastlines()
    plt.title('true '+'+'.join(v.components),fontsize=fontsize_titles)

    ### colorbar for first row of panels: ###
    cbar = plt.colorbar(im, fraction=0.035, pad=0.04, ax = ax)
    ticks = np.arange(vmin, vmax+deltav, deltav)
    cbar.set_ticks(ticks)
    cbar.set_label(label = v.unit+" (m)",size=fontsize_colorbar)
    cbar.ax.tick_params(labelsize=fontsize_cbar_ticks)

    ### predicted component signals (second row of panels): ###
    for comp in range(v.num_comp):
        ax = fig.add_subplot(3,ncol,ncol+comp+1,projection=ccrs.Robinson(central_longitude = 0))
        im = ax.contourf(lonsGeoid,latsGeoid,lat_lon_pred[:,:,comp], CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        ax.coastlines()
        plt.title('predicted '+v.components[comp],fontsize=fontsize_titles)

    ### predicted sum of components (second row of panels): ###
    ax = fig.add_subplot(3,ncol,2*ncol,projection=ccrs.Robinson(central_longitude = 0))
    im = ax.contourf(lonsGeoid,latsGeoid,np.sum(lat_lon_pred,axis=2), CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    ax.coastlines()
    plt.title('predicted '+'+'.join(v.components),fontsize=fontsize_titles)

    ### colorbar for second row of panels: ###
    cbar = plt.colorbar(im, fraction=0.035, pad=0.04, ax = ax)
    ticks = np.arange(vmin, vmax+deltav, deltav)
    cbar.set_ticks(ticks)
    cbar.set_label(label = v.unit+" (m)",size=fontsize_colorbar)
    cbar.ax.tick_params(labelsize=fontsize_cbar_ticks)

    # colorbar range specifications for error:
    CNT_LINES = np.linspace(vmin_err, vmax_err, 101) # if we want more descrete colors, we need to increase the 101 number. 
    norm = mpl.colors.Normalize(vmax = vmax_err, vmin = vmin_err)


    ### component prediction errors (third row of panels): ###
    for comp in range(v.num_comp):
        ax = fig.add_subplot(3,ncol,2*ncol+comp+1,projection=ccrs.Robinson(central_longitude = 0))
        im = ax.contourf(lonsGeoid,latsGeoid,lat_lon_pred[:,:,comp]-lat_lon_true[:,:,comp], CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        ax.coastlines()
        plt.title('error '+v.components[comp],fontsize=fontsize_titles)

    ### sum prediction errors (third row of panels): ###
    ax = fig.add_subplot(3,ncol,3*ncol,projection=ccrs.Robinson(central_longitude = 0))
    im = ax.contourf(lonsGeoid,latsGeoid,np.sum(lat_lon_pred,axis=2)-np.sum(lat_lon_true,axis=2), CNT_LINES, transform=ccrs.PlateCarree(),cmap=plt.get_cmap('bwr'), extend = "both",norm=norm)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    ax.coastlines()
    plt.title('error '+'+'.join(v.components),fontsize=fontsize_titles)


    ### colorbar for third row of panels: ###
    cbar = plt.colorbar(im, fraction=0.035, pad=0.04, ax = ax)
    ticks = np.arange(vmin_err, vmax_err+deltav_err, deltav_err)
    cbar.set_ticks(ticks)
    cbar.set_label(label = v.unit+" (m)",size=fontsize_colorbar)
    cbar.ax.tick_params(labelsize=fontsize_cbar_ticks)

    plt.suptitle('time step: '+str(time_idx))

    plt.tight_layout()

    plt.savefig(filename)

#####

# function that converts list of lists to a flat list

def flatten_list(list):

    flat_list = []

    for xs in list:
        for x in xs:
            flat_list.append(x)

    return flat_list

#####

# function creating a plot showing the rms-averaged prediction errors of single components or sums of multiple components

def rms_errors_combinations(file_prediction,filename,num_comp_list,subplot_label):

    labels_fontsize = 16
    ticklabels_fontsize = 14

    # construct all combinations of "num_comp" components:
    combination_list = []

    for num_comp in num_comp_list:
        combination_list.append(list(combinations(v.components,num_comp)))

    combination_list = flatten_list(combination_list)

    # compute RMS errors of individual components and of sum of components (all combinations):
    rms_error_absolute = [] # error in m EWH

    for combination in combination_list:
        comp_errors = []

        for comp in combination:
            with h5py.File(file_prediction, "r") as f:
                subgroup_comp = f[v.test_year+'/'+comp]
                pred_signal_lat_lon_t = subgroup_comp['pred_signal_lat_lon_t'][:,:,:]
                true_signal_lat_lon_t = subgroup_comp['true_signal_lat_lon_t'][:,:,:]
            
            comp_errors.append(pred_signal_lat_lon_t-true_signal_lat_lon_t)

        rms_of_error_sum = rms(sum(comp_errors))
        rms_error_absolute.append(rms_of_error_sum)

    components_delta = ['Δ' + comp for comp in v.components]
    combination_list_delta = []
    for num_comp in num_comp_list:
        combination_list_delta.append(list(combinations(components_delta,num_comp)))
    combination_list_delta = flatten_list(combination_list_delta)

    x_axis_entries = []
    for comb in combination_list_delta:
        x_axis_entries.append('+'.join(list(itertools.chain(comb))))


    # Plot absolute RMS errors of individual components and sum of combinations (in m EWH):
    plt.figure(figsize=(len(combination_list)*2,5))
    plt.plot(range(len(combination_list)),rms_error_absolute,'.',markersize=15)
    plt.xticks(range(len(combination_list)),x_axis_entries,fontsize=ticklabels_fontsize)
    plt.ylabel('RMS error in m EWH', fontsize=labels_fontsize)
    plt.title('Absolute RMS errors')
    plt.grid()
    plt.ylim([0,0.01])
    plt.xlim([-0.5,len(combination_list)-0.5])
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.gca().text(-0.3, 0.0093, subplot_label,fontsize=20, va='top', ha='left')

    plt.savefig(filename)


#####

# function similar to rms_errors_combinations, but showing normalized errors

def normalized_rms_errors_combinations(file_prediction,filename,num_comp_list,subplot_label):

    labels_fontsize = 16
    ticklabels_fontsize = 14

    # construct all combinations of "num_comp" components:
    combination_list = []

    for num_comp in num_comp_list:
        combination_list.append(list(combinations(v.components,num_comp)))

    combination_list = flatten_list(combination_list)

    # compute RMS errors of individual components and of sum of components (all combinations):
    rms_error_normalized = []
    # error between 0 (sum of components correctly predicted / errors are assignment errors) 
    # and 1 (sum of components not correctly predicted / absent signal error dominates)

    for combination in combination_list:

        comp_errors = []
        sum_of_rms_errors = 0 

        for comp in combination:
            with h5py.File(file_prediction, "r") as f:
                subgroup_comp = f[v.test_year+'/'+comp]
                pred_signal_lat_lon_t = subgroup_comp['pred_signal_lat_lon_t'][:,:,:]
                true_signal_lat_lon_t = subgroup_comp['true_signal_lat_lon_t'][:,:,:]
            
            comp_errors.append(pred_signal_lat_lon_t-true_signal_lat_lon_t)
            sum_of_rms_errors += rms(pred_signal_lat_lon_t - true_signal_lat_lon_t)

        rms_of_error_sum = rms(sum(comp_errors)) 
        rms_error_normalized.append(rms_of_error_sum/sum_of_rms_errors)

    components_delta = ['Δ' + comp for comp in v.components]
    combination_list_delta = []
    for num_comp in num_comp_list:
        combination_list_delta.append(list(combinations(components_delta,num_comp)))
    combination_list_delta = flatten_list(combination_list_delta)

    x_axis_entries = []
    for comb in combination_list_delta:
        x_axis_entries.append(','.join(list(itertools.chain(comb))))


    # Plot absolute RMS errors of individual components and sum of combinations (in m EWH):
    plt.figure(figsize=(len(combination_list)*2,5))
    plt.plot(range(len(combination_list)),rms_error_normalized,'.',markersize=15)
    plt.xticks(range(len(combination_list)),x_axis_entries,fontsize=ticklabels_fontsize)
    plt.ylabel('RMS errors of sum / sum of RMS errors',fontsize=labels_fontsize)
    plt.title('Normalized RMS errors')
    plt.grid()
    plt.ylim([0, 1])
    plt.xlim([-0.5,len(combination_list)-0.5])
    plt.yticks(fontsize=ticklabels_fontsize)
    plt.gca().text(-0.3, 0.93, subplot_label,fontsize=20, va='top', ha='left')

    plt.savefig(filename)


#####

# function creating a plot showing the prediction errors on the training and test dataset as function of training epoch

def train_test_curve_plot(file_prediction,log_dir,filename,colors_components):

    labels_fontsize = 16
    ticklabels_fontsize = 14

    with h5py.File(log_dir + v.model_id + '_training_logs.hdf5',"r") as f:
        RMS_train_comp = f['RMS_train_comp'][:,:]
        RMS_test_comp = f['RMS_test_comp'][:,:]
        epoch_best_model = f['epoch_best_model'][()]

    # compute rms value for each (test year) signal to have a reference mean amplitude:
    year = v.test_year
    rms_signal_comp = np.zeros([v.num_comp,1])

    for comp in range(v.num_comp):
        # extract RMS vectors for individual components from hdf5 file:
        with h5py.File(file_prediction, "r") as f:
            subgroup_comp = f[year+'/'+v.components[comp]]
            RMS_true = subgroup_comp['RMS_true_signal_t'][:]
            rms_signal_comp[comp] = np.sqrt(np.mean(np.square(RMS_true)))

    plt.figure()

    for comp in range(v.num_comp):
        plt.plot(RMS_train_comp[comp,:],':',color=colors_components[comp],label=v.components[comp] + ' training error')
        plt.plot(RMS_test_comp[comp,:],color=colors_components[comp],label=v.components[comp] + ' test error')
        plt.axhline(y=rms_signal_comp[comp], color=colors_components[comp], linestyle='--',label=v.components[comp] + ' signal (test): ' + '%.3f'%(rms_signal_comp[comp]) + ' m EWH')
        
    plt.xlim([0, epoch_best_model])
    plt.ylim([0, 0.015])
    plt.title('Error on training and test data')
    plt.xlabel('epoch', fontsize=labels_fontsize)
    plt.ylabel('RMS error in m ' + v.unit, fontsize=labels_fontsize)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=ticklabels_fontsize)
    plt.yticks(fontsize=ticklabels_fontsize)


    plt.savefig(filename,bbox_inches='tight')

