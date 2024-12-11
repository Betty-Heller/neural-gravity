#!/usr/bin/env python3


"""j
Code creating a new multi-channel U-Net model for signal separation in spatio-temporal gravity data

the multi-channel U-Net architecture is similar to Kadandale et al. (2020)

parts of the here-implemented U-Net architecture and data augmentation strategies are taken from the tensorflow implementation of pix2pix by Isola et al. (2017) available here: https://www.tensorflow.org/tutorials/generative/pix2pix
j"""

"""j
Packages import
j"""

import tensorflow as tf
from os.path import join
import numpy as np
import os
import h5py
import random
import time


"""j
Import functions and variables
j"""

import variables as v

model_id = v.model_id
model_folder = '.'
SH_folder = '../../Data/SH_coeff/'

inp_tar_folder = model_folder + '/train_test_data_01/' # folder in which the input-target pairs are stored
inp_tar_file = '_'.join(v.components) + '_' + v.unit + '_in_m_' + str(v.deltat) + 'h_' + v.sampling +  '.hdf5'


# extract lons_global, lats_global from group "spatial_grid" of file_prediction
with h5py.File(inp_tar_folder+inp_tar_file,"r") as f:
    group = f['spatial_grid']
    dim = group.attrs['dim']

"""j
Functions building the architecture of the U-Net, and some helper functions:
j"""

# function providing the variables needed to do a triangle plot

# xv = (Nmax+1) x (Nmax+2) matrix containing the x axis values (SH orders) of the triangle plot 
# yv = (Nmax+1) x (Nmax+2) matrix containing the y axis values (SH degrees) of the triangle plot 
# coeff = (Nmax+1) x (Nmax+2) matrix containing the coefficient triangle (rows = degrees, columns = neg.+pos. orders)

def triangle_plot_params(cilm):

    # build coefficient array to be plotted:
    Cnm = cilm[0]
    Snm = cilm[1]
    Snm_triangle = np.fliplr(Snm[:,1:])
    coeff = np.concatenate((Snm_triangle,Cnm),axis=1)

    # build x and y axes for the triangle plot:
    nmax = cilm.shape[1]
    n = np.arange(nmax+1)
    m1 = np.arange(nmax+1)
    m2 = np.flip(m1[1:])*(-1)
    m = np.concatenate((m2,m1),axis=0)
    xv, yv = np.meshgrid(m, n, indexing='xy')

    return xv, yv, coeff


# function that extracts input and target arrays of the test data year from input_target_pairs-hdf5 file 
# and computes predicted and true data, and saves them to hdf5 files in the model folder

# inputs:
# file_datasets: e.g. model_folder + 'train_test_data/AO_H_S_EWH_in_m_168h_do100_lat_t.hdf5'
# year: e.g. '2002'
# model: e.g. generator (trained U-Net model)

# creates:
# hdf5 file in folder data_dir, e.g. Pred_true_signal_EWH_in_m_24h_do100.hdf5 (with internal file structure year -> component -> pred_array, true_array)

def true_pred_arrays(file_datasets,year,model,data_dir):

    # extract dim and numt from file_datasets
    with h5py.File(file_datasets,"r") as f:
        group_year = f[year]
        num_samples = group_year['input'][:,:,:].shape[0] # is equal to dim if v.random_samples = 'no'
        dim = group_year['input'][:,:,:].shape[1] # number of rows and columns of the samples
        numt = group_year.attrs['numt']
        DOY_axis = group_year.attrs['DOY_axis'][:]

    # collect samples of year "year" in samp - row - col - comp arrays:
    pred_array = np.zeros([num_samples,dim,dim,v.num_comp]) 
    true_array = np.zeros([num_samples,dim,dim,v.num_comp])
    
    for idx in range(num_samples): # loop over samples

        # extract sample from hdf5 file:
        input, target = sample_from_h5py(file_datasets,[year,idx]) # input is tensor of shape (1,dim,dim,1), target is tensor of shape (1,dim,dim,numcomp)

        # compute model output for sample_input:
        prediction = model(input, training=True) # prediction is tensor of shape (1,dim,dim,numcomp)
        del input

        ### PREDICTION (pred_array is samp-row-col-comp array):
        pred_array[idx,:,:,:] = prediction[0].numpy() # predicted image (of shape dim x dim x num_comp)
        del prediction

        ### TARGET (true_array is samp-row-col-comp array):
        true_array[idx,:,:,:] = target[0].numpy() # target (true) image (of shape dim x dim x num_comp)
        del target

    # if the training/testing samples are random linear combinations of the actual samples, need to retrieve back the actual samples:
    if v.random_samples == 'yes':
        # read in G_inv:
        with h5py.File(file_datasets,"r") as f:
            group = f['inverse_of_random_weights_matrix']
            G_inv = group['G_inv'][:,:]

        # convert psi_row_col_comp arrays to samp_row_col_comp arrays:
        pred_array = np.tensordot(G_inv,pred_array,1) # dim x dim x dim x v.num_comp
        true_array = np.tensordot(G_inv,true_array,1) # dim x dim x dim x v.num_comp
        del G_inv

    # create hdf5 file where predicted signals and errors can be stored in:
    #file_out = 'Pred_true_signal_' + v.unit + '_in_m_' + str(v.deltat) + 'h.hdf5'
    file_out = v.model_id + '_model_predictions.hdf5'
    with h5py.File(data_dir+file_out, "a") as f: # "a" means: read/write file if exists, create otherwise
        group_year = f.create_group(year) # create a group of name "year"
        group_year.create_dataset('pred_array',dtype=float,data = pred_array)
        group_year.create_dataset('true_array',dtype=float, data = true_array)
        group_year.attrs['dim'] = dim
        group_year.attrs['numt'] = numt # number of original time steps in that year (before interpolation to dim steps)
        group_year.attrs['DOY_axis'] = DOY_axis # time axis in DOY (giving the DOY of the middle/reference points of each time period)

    # write out structure of newly created hdf5 file:
    print('Done with writing model predictions for year ' + year + ' to hdf5 file!')
    #f = nx.nxload(data_dir+file_out)
    #print(f.tree)


# function that builds the RMS of an array along a specific dimension (or, if axis is not specified, of all numbers in the array)

def rms(x, axis=None): # axis specifies which dimension should "disappear" after building the rms
    return np.sqrt(np.mean(x**2, axis=axis))


# function for generating test samples (individual samples, no batches)

# difference to batch_data_from_h5py is that here, only one sample (not a batch of samples) is extracted, and no random jittering is applied!
# sample_idx should be e.g. [1995,40] -> denoting one specific sample (not a list!)

def sample_from_h5py(filename,sample_idx):

    with h5py.File(filename,"r") as f:
        group_name = str(sample_idx[0]) # e.g. '1996' -> the year of the sample 
        inp = f[group_name+'/input'][sample_idx[1],:,:] # sample_idc[1] is the index of the sample within the year
        inp = inp[tf.newaxis,...,tf.newaxis] # add "batch size" and "channels" axes to input image. of shape 1 x dim x dim x 1 
        tar = f[group_name+'/target'][sample_idx[1],:,:,:] # of shape dim x dim x num_comp
        tar = tar[tf.newaxis,...] # add "batch size" axis to target image. of shape 1 x dim x dim x num_comp

    inp32 = np.float32(inp)
    tar32 = np.float32(tar)

    dataset = tf.data.Dataset.from_tensor_slices(inp32)
    batch_dataset = dataset.batch(1)
    # extract (the) one batch from a batch dataset and give back a tensor of size (1,dim,dim,1) (-> necessary as this format is needed by function train_step):
    input_image = next(iter(batch_dataset.take(1)))

    dataset = tf.data.Dataset.from_tensor_slices(tar32)
    batch_dataset = dataset.batch(1)
    target = next(iter(batch_dataset.take(1)))

    return input_image, target


# function that calculates a RMS error of component predictions and sum predictions over 100 randomly chosen samples
# years should be given as list of strings, e.g. years = ['1995','1996'] -> specifies the pool of data from which the 100 samples are chosen

def RMS_error_random100(file_datasets,years,model):
    num_samples_error = 100 # number of samples over which the error should be computed

    # (1) first of all, create an index list to be able to loop over random samples later:
    num_samples = [] # list; i-th entry gives the number of training samples in the i-th year

    with h5py.File(file_datasets,"r") as f:
        for train_year in years:
            num_samples_year = f[train_year+'/input'].shape[0]
            num_samples.append(num_samples_year)

    num_total = sum(num_samples) # number of training samples in total (across all training years)

    if num_total < num_samples_error: # for cases where less than 100 samples are contained in the considered data (e.g. 1 year of 7-day solutions)
        num_samples_error = num_total

    idx_list = random.sample(range(num_total),num_samples_error) # e.g. [486,572,303,248,539,...] -> of length num_samples_error

    year_column = []
    sample_column = []

    for idx in range(len(years)):
        year_column.extend(np.repeat(np.array(years[idx],dtype='int'),num_samples[idx])) # e.g. [1995, 1995, 1995,... , 1996, 1996, 1996,...] -> of length num_total
        sample_column.extend(np.arange(num_samples[idx])) # e.g. [0,1,2,3,4,... , 0,1,2,3,4,...] -> of length num_total

    tmp = np.column_stack((year_column,sample_column)) # e.g. [[1995,0],[1995,1],[1995,2],... , [1996,0],[1996,1],[1996,2],...] -> of shape num_total x 2
    random_sample_list = tmp[idx_list,:] # e.g. [[1996,122],[1996,208],[1995,303],[1995,248],...] -> of shape num_samples_error x 2

    # (2) then, actually read in the random samples, compute predicted data and compute errors:
    RMS_comp_list = np.zeros([num_samples_error,v.num_comp])
    RMS_sum_list = np.zeros([num_samples_error])

    for sample in range(num_samples_error):

        input_image, target = sample_from_h5py(file_datasets,random_sample_list[sample])

        prediction = model(input_image, training=True)

        # RMS error for each component individually (square root of L2 loss), mean is formed across all samples of the batch (here: 1 sample forms the batch), across image height + width
        RMS_comp = tf.math.sqrt(tf.reduce_mean(tf.square(tf.abs(target - prediction)), axis= (0,1,2))) # num_comp values

        # RMS error for sum of components:
        gen_sum = tf.math.reduce_sum(prediction,axis=3,keepdims=True) # sum of predicted components, of shape 1 x dim x dim x 1
        RMS_sum = tf.math.sqrt(tf.reduce_mean(tf.square(tf.abs(gen_sum - input_image)))) # 1 value

        # collect error values of 100 samples in arrays:
        RMS_comp_list[sample,:] = RMS_comp.numpy() # convert to a numpy array (vector of length num_comp)
        RMS_sum_list[sample] = RMS_sum.numpy()

    # build rms values over all 100 samples:
    RMS_comp_random100 = rms(RMS_comp_list,axis=0)
    RMS_sum_random100 = rms(RMS_sum_list)

    return RMS_comp_random100, RMS_sum_random100 # are in the same units as the data


# define downsampler (needed to build the encoder part of the U-Net)

def downsample(filters, size, apply_batchnorm=True): # filters = dimensionality of the output space, size = height and width of the 2D convolution window
    initializer = tf.random_normal_initializer(0., 0.02) # random values with normal distribution (mean=0, stddev. = 0.02)
    
    result = tf.keras.Sequential() # groups a linear stack of layers into a tf.keras.Model.
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False)) # add a 2D convolution layer (e.g. spatial convolution over images)

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# define upsampler (needed to build the decoder part of the U-Net)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# number of filters: num_filters_min -> 2*num_filters_min -> 4*num_filters_min -> ... -> num_filters_max -> num_filters_max -> num_filters_max etc., until the image has dimensions of 1 x 1

num_filters_list = [] # list collecting the number of filters applied in the subsequent layers

num_filters_layer = v.num_filters_min # number of filters in the first layer is num_filters_min
dim_layer = dim # height=width dimension of the image in the first layer is dim

num_filters_list.append(num_filters_layer)
dim_layer = dim_layer/2 # each convolutional layer leads to a halving of the image height and width (because 4x4 filters with stride 2 are applied)

while num_filters_layer < v.num_filters_max: 
    num_filters_layer = num_filters_layer*2 # number of filters is increased by factor of 2 in subsequent layers, until a value of num_filters_max is reached
    num_filters_list.append(num_filters_layer)
    dim_layer = dim_layer/2

while dim_layer > 1:
    num_filters_list.append(num_filters_layer) # convolutional layers with num_filters_max filters are added until the image has height=width=1
    dim_layer = dim_layer/2


# create list down_stack of downsampling operations that are used in the encoder part of the U-Net:

down_stack = []

down_stack.append(downsample(num_filters_list[0], v.filter_size, apply_batchnorm=False))

for num_fil in num_filters_list[1:]:
    down_stack.append(downsample(num_fil, v.filter_size))


# create list up_stack of upsampling operations that are used in the decoder part of the U-Net:

num_filters_list.reverse()
num_filters_list = num_filters_list[1:]

up_stack = []

up_stack.append(upsample(num_filters_list[0], v.filter_size, apply_dropout=True))  
up_stack.append(upsample(num_filters_list[1], v.filter_size, apply_dropout=True))  
up_stack.append(upsample(num_filters_list[2], v.filter_size, apply_dropout=True))  

for num_fil in num_filters_list[3:]:
    up_stack.append(upsample(num_fil, v.filter_size))


# build multi-channel U-Net architecture:

def Generator():

    # 1. Input layer:
    inputs = tf.keras.layers.Input(shape=[dim, dim, 1]) # "shape" gives shape of expected input (not including the batch size!)
    x = inputs

    # 2. Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x) # apply downsampling steps to the inputs one after the other
        skips.append(x) # save the intermediate results to the list "skips"

    skips = reversed(skips[:-1]) # last entry of skips is skipped as this is the "bottom" of the U-Net => No skip connection here
    # order of the intermediate results is reversed as later they are copied in reverse order to the upsampling layers

    # 3. Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x) # apply upsampling steps one after the other
        x = tf.keras.layers.Concatenate()([x, skip]) # up-sampled (new) entries x are combined with intermediate results from the downsampling layers
        
    # 4. Last layer (no skip connection here; number of filters = number of output channels = num_comp)
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(v.num_comp, v.filter_size,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (bs, dim, dim, num_comp)
    x = last(x) # intermediate results x go through the very last convolutional layer

    return tf.keras.Model(inputs=inputs, outputs=x) # for the generator, model input and output are images


generator = Generator()


# mean square value of a quantity (along all dimensions of it)
def MS(y,axis=None):
    return tf.reduce_mean(tf.square(y),axis)

# mean absolute value of a quantity (along all dimensions of it)
def MA(y,axis=None):
    return tf.reduce_mean(tf.abs(y),axis)


# loss function:

def generator_loss(input, gen_output, target):

    ### 1. loss terms constraining gen_output to target (Mean Squared Error component losses):
    MSE_loss_comp_abs = MS(gen_output - target,axis=(0,1,2)) # (v.num_comp,)

    if v.loss_type == 'absolute_MSE':
        normfactor = 1.0
    elif v.loss_type == 'relative_MSE':
        normfactor = MS(target,axis=(0,1,2)) # (v.num_comp,)
    elif v.loss_type == 'signal_weighted_MSE':
        normfactor = MA(target,axis=(0,1,2)) # (v.num_comp,)

    # set a lower limit for the normfactor, to prevent large outliers in the loss function
    normfactor_clip = tf.ones_like(normfactor, dtype=float) * 0.01
    normfactor = tf.maximum(normfactor,normfactor_clip)
    
    L2_loss_comp = MSE_loss_comp_abs / normfactor # (v.num_comp,)

    # now, additionally apply the relative weighting factor between the individual loss terms:
    weights_tensor = tf.constant(v.weights_comp)
    component_loss = tf.tensordot(L2_loss_comp, weights_tensor, 1) # weighted sum of L2 component losses, is a scalar


    ### 2. loss term constraining the sum of the gen_output channels to the input channels (L2 sum loss):
    gen_sum = tf.math.reduce_sum(gen_output,axis=3,keepdims=True) # sum of predicted components, dimension: (bs,dim,dim)

    MSE_loss_sum_abs = MS(gen_sum - input) # scalar

    if v.loss_type == 'absolute_MSE':
        normfactor = 1.
    elif v.loss_type == 'relative_MSE':
        normfactor = MS(input) # scalar
    elif v.loss_type == 'signal_weighted_MSE':
        normfactor = MA(input) # scalar

    # set a lower limit for the normfactor, to prevent large outliers in the loss function
    normfactor_clip = tf.ones_like(normfactor, dtype=float) * 0.01
    normfactor = tf.maximum(normfactor,normfactor_clip)

    sum_loss = v.weight_sum * MSE_loss_sum_abs / normfactor # scalar


    ### 3. loss terms constraining the linearity (linear shape loss) and slope (slope value loss) of selected gen_output channels along the time axis:
    slope_value_loss = 0.
    linear_shape_loss = 0.

    for comp in range(v.num_comp):
        if v.weights_slope_value[comp] != 0.: # only compute this kind of loss for selected components

            # check which of the two sampling axes is time
            if v.sampling[0] == 't': 
                t_axis=1
            elif v.sampling[-1] == 't':
                t_axis=2

            pred = gen_output[:,:,:,comp] # (bs,dim,dim)
            tar = target[:,:,:,comp] # (bs,dim,dim)

            slope_pred_along_t = tf.experimental.numpy.diff(pred,axis=t_axis) # (bs,dim-1,dim) or (bs,dim,dim-1)
            mean_pred_slope = tf.reduce_mean(slope_pred_along_t,axis=t_axis) # (bs,dim)

            slope_tar_along_t = tf.experimental.numpy.diff(tar,axis=t_axis) # (bs,dim-1,dim) or (bs,dim,dim-1)
            mean_tar_slope = tf.reduce_mean(slope_tar_along_t,axis=t_axis) # (bs,dim)

            # the predicted slope should be close to the true slope:
            slope_value_loss += v.weights_slope_value[comp] * MS(mean_pred_slope - mean_tar_slope) # scalar

            # the scatter of the predicted slopes about their mean should be small:
            if t_axis == 1:
                mean_pred_slope_expanded = tf.tile(tf.expand_dims(mean_pred_slope,axis=1),[1,dim-1,1]) # (bs,dim-1,dim)
            elif t_axis == 2:
                mean_pred_slope_expanded = tf.tile(tf.expand_dims(mean_pred_slope,axis=2),[1,1,dim-1]) # (bs,dim,dim-1)

            slope_deviation_along_t = slope_pred_along_t - mean_pred_slope_expanded # (bs,dim-1,dim) or (bs,dim,dim-1)

            linear_shape_loss += v.weights_linear_shape[comp] * MS(slope_deviation_along_t) # scalar
                    
    total_loss = component_loss + sum_loss + slope_value_loss + linear_shape_loss

    return L2_loss_comp, component_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss


# function needed to apply some noise to the training data before training (==> data augmentation)

@tf.function()
def random_jitter(input_image, real_image): # input_image, real_image are single samples (not batches of samples!)
    
    ### Method 1: grow images to larger size and randomly crop back: ###
    if v.data_augmentation[0] == 1:
        print('Data augmentation method 1: grow images to larger size and randomly crop back')
        height = dim + 30
        width = dim + 30
        input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # resizes to a larger image and transform numpy array to tf tensor

        # first argument of tf.image.resize must be: 
        # 4-D Tensor of shape [batch, height, width, channels] 
        # or: 3-D Tensor of shape [height, width, channels]
        real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        stacked_image = tf.concat([input_image, real_image], axis=2) # use concat, as it can deal with tensors having different dimensions along axis=2

        cropped_image = tf.image.random_crop(stacked_image, size=[dim, dim, v.num_comp +1]) # cropping back to get dimensions of dim x dim

        input_image = cropped_image[:,:,0] # subdividing back channels to input and output image
        real_image = cropped_image[:,:,1:]

        input_image = input_image[...,tf.newaxis] # adding "channel" axis to input image
    
    ### Method 2: randomly do a left-right flip: ### 
    if v.data_augmentation[1] == 1:
        if tf.random.uniform(()) > 0.5:
            print('Data augmentation method 2: randomly do a left-right flip')
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

    ### Method 3: randomly do a top-down flip: ###
    if v.data_augmentation[2] == 1:
        if tf.random.uniform(()) > 0.5:
            print('Data augmentation method 3: randomly do a top-down flip')
            input_image = tf.image.flip_up_down(input_image)
            real_image = tf.image.flip_up_down(real_image)

    ### Method 4: adjust brightness (add a value delta to all pixel values): ###
    if v.data_augmentation[3] == 1:
        print('Data augmentation method 4: randomly add value between +- 0.0005 to the pixel values')
        delta = tf.random.uniform(shape=(),minval=-0.005,maxval=0.005) # random number which is to be added to all channels of the input image

        input_image = tf.image.adjust_brightness(input_image,delta*v.num_comp) # add delta*v.num_comp to the input image, such that it still corresponds to the sum of the individual target components
        real_image = tf.image.adjust_brightness(real_image,delta)

    ### Method 5: adjust contrast (change the variance of the signals about their mean value): ###
    # (contrast of the data before the contrast adjustment is "1",
    # afterwards contrast is between 0.5 (=weaker signals) and 1.5 (=stronger signals): 
    # (x - mean) * contrast_factor + mean
    # and the signals' variations about their mean are also randomly multiplied by +- 1
    if v.data_augmentation[4] == 1:
        print('Data augmentation method 5: randomly change variation of signals about their mean')
        random_number = tf.random.uniform(shape=(),minval=-1,maxval=1) # random number between -1 and 1
        random_sign = random_number/tf.abs(random_number) # either -1 or 1 => giving a random sign

        new_contrast = tf.random.uniform(shape=(),minval=0.5,maxval=1.5) * random_sign # scaling factor between -1.5 and -0.5 or between 0.5 and 1.5

        input_image = tf.image.adjust_contrast(input_image,new_contrast)
        real_image = tf.image.adjust_contrast(real_image,new_contrast) # input and target become stretched by same factor (sum of target channels should still correspond to input channel)

    return input_image, real_image


# function specifying the update of parameters in one training step

# input_image and target need to be tf-tensors of shape (bs,dim,dim,1) and (bs,dim,dim,num_comp)

@tf.function
def train_step(input_image, target):
    
    with tf.GradientTape() as gen_tape: # meaning: gen_tape=tf.GradientTape()
        
        # compute output of current version of generator:
        gen_output = generator(input_image, training=True)

        # compute losses (outputs are tf tensors => can convert them to arrays using .numpy())
        L2_loss_comp, L2_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss = generator_loss(input_image, gen_output, target) # loss are averaged values over complete batch

        # compute the gradients 
        generator_gradients = gen_tape.gradient(total_loss, generator.trainable_variables)

        # update generator and discriminator:
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                    generator.trainable_variables))

    return L2_loss_comp, L2_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss


# function computing the loss terms on one sample of the validation dataset
@tf.function
def val_loss(input_image, target):
        
    # compute output of current version of generator:
    gen_output = generator(input_image, training=True)

    # compute losses (outputs are tf tensors => can convert them to arrays using .numpy())
    L2_loss_comp, L2_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss = generator_loss(input_image, gen_output, target) # loss are averaged values over complete batch

    return L2_loss_comp, L2_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss


# function specifying the training loop

# train_years = list of strings, e.g. ['1995', '1996']
# validation_year = string, e.g. '2005'
# test_year = string, e.g. '2006'
# num_epochs = how many iterations are done over the complete training dataset

def fit(file_datasets):

    start_global = time.time()

    val_loss_best_model = 100000 # initialize quality measure defining "best model"
    counter = 0 # initialize counter for number of epochs for which training is continued in case of no improvement

    # loop over epochs (1 epoch means that each of the samples in the training dataset is used once)
    for epoch in np.arange(v.num_epochs):
        
        start = time.time() # start/reset the timer

        print(f"Epoch: {epoch}")

        # RMS errors to measure predictive performance during training
        RMS_train_comp, RMS_train_sum = RMS_error_random100(file_datasets,v.train_years,generator)
        RMS_val_comp, RMS_val_sum = RMS_error_random100(file_datasets,[v.validation_year],generator)
        RMS_test_comp, RMS_test_sum = RMS_error_random100(file_datasets,[v.test_year],generator)

        # write RMS errors on training and validation dataset to hdf5 file:
        with h5py.File(log_dir + model_id + '_training_logs.hdf5', "a") as f:
            f['RMS_train_comp'][:,epoch] = RMS_train_comp
            f['RMS_val_comp'][:,epoch] = RMS_val_comp
            f['RMS_test_comp'][:,epoch] = RMS_test_comp
            f['RMS_train_sum'][epoch] = RMS_train_sum
            f['RMS_val_sum'][epoch] = RMS_val_sum
            f['RMS_test_sum'][epoch] = RMS_test_sum

        # for overfitting monitoring/early stopping: compute loss on one sample of the validation dataset:

        # # -> determine number of samples in validation dataset:
        with h5py.File(file_datasets,"r") as f:
            group_year = f[v.validation_year]
            num_samples = group_year['input'][:,:,:].shape[0]

        # # -> random choice of one of them:
        idx = random.sample(range(num_samples),1)[0]

        # # -> extract validation sample:
        val_input, val_target = sample_from_h5py(file_datasets,[v.validation_year,idx])

        # # -> compute validation loss:
        val_losses = val_loss(val_input, val_target)
        total_loss_val = val_losses[3]

        with h5py.File(log_dir + model_id + '_training_logs.hdf5', "a") as f:
            f['total_loss_val'][epoch] = total_loss_val.numpy()

        if epoch > v.min_number_epochs:
            # save current model to checkpoint in case it performs better than the last checkpointed model for all components:
            if total_loss_val < val_loss_best_model:
                counter = 0 # reset counter
                val_loss_best_model = total_loss_val
                print('currently best model: after ' + str(epoch) + ' epochs')
                epoch_best_model = epoch
                manager.save() # save current model to checkpoint
            else: # don't save current model
                counter += 1
                print('no improvement in epoch ' + str(epoch))
                if counter == 10: # if since 10 epochs there has not been any improvement w.r.t. to the last best model, stop the training
                    break

        ## Shuffle the training data samples and sort them to batches (without actually reading in the data): ##
        num_samples = [] # list; i-th entry gives the number of training samples in the i-th year

        with h5py.File(file_datasets,"r") as f:
            for train_year in v.train_years:
                num_samples_year = f[train_year+'/input'].shape[0]
                num_samples.append(num_samples_year)

        num_total = sum(num_samples) # number of training samples in total (across all training years)

        num_batches = int(num_total / v.bs) # number of batches (across all training years)
        idx_list = random.sample(range(num_total),num_total) # e.g. [486,572,303,248,539,...] -> of length num_total (random indices for full training dataset, can span several years)

        year_column = []
        sample_column = []

        for idx in range(len(v.train_years)):
            year_column.extend(np.repeat(np.array(v.train_years[idx],dtype='int'),num_samples[idx])) # e.g. [1995, 1995, 1995,... , 1996, 1996, 1996,...] -> of length num_total
            sample_column.extend(np.arange(num_samples[idx])) # e.g. [0,1,2,3,4,... , 0,1,2,3,4,...] -> of length num_total

        tmp = np.column_stack((year_column,sample_column)) # e.g. [[1995,0],[1995,1],[1995,2],... , [1996,0],[1996,1],[1996,2],...] -> of shape num_total x 2
        random_sample_list = tmp[idx_list,:] # e.g. [[1996,122],[1996,208],[1995,303],[1995,248],...] -> of shape num_total x 2

        sample_batches = random_sample_list[:num_batches*v.bs,:].reshape((num_batches,v.bs,2)) # e.g. [[[1996,122]],[[1996,208]],[[1995,303]],...] 
        # -> index 0 = batch index, index 1 = sample in batch, index 2 = dataset/year and index within dataset/year
        # -> of shape num_batches x bs x 2

        sample_batches_list = list(sample_batches)

        # if number of training data samples is not divisible by the batch size, append a smaller batch of the remaining data at the end:
        if num_total % v.bs != 0:
            sample_batches_list.append(random_sample_list[num_batches*v.bs:,:])
            num_batches += 1

        ## Learning: ##    
        # loop over batches (span the complete training dataset)
        for i in range(num_batches):

            # create tf-tensors containing the i-th batch of data:
            input_image, target = batch_data_from_h5py(file_datasets,sample_batches_list[i])    

            # give i-th batch of data to the train_step routine => model parameters are updated
            L2_loss_comp, L2_loss, sum_loss, total_loss, slope_value_loss, linear_shape_loss = train_step(input_image, target) # gives back loss values averaged over complete batch
            
            # print a dot (.) every 10-th batch:
            if (i+1) % 10 == 0: 
                print('.', end='', flush=True)

        # loss values are saved after every epoch (i.e. the loss value of the last batch within the epoch is saved)
        with h5py.File(log_dir + model_id + '_training_logs.hdf5', "a") as f:
            f['L2_loss_comp'][:,epoch] = L2_loss_comp.numpy()
            f['L2_loss'][epoch] = L2_loss.numpy()
            f['sum_loss'][epoch] = sum_loss.numpy()
            f['total_loss'][epoch] = total_loss.numpy()
            f['slope_value_loss'][epoch] = slope_value_loss
            f['linear_shape_loss'][epoch] = linear_shape_loss

        # print the time needed for the epoch
        print(f'Time taken for 1 epoch: {time.time()-start:.2f} sec\n')

    # write number of training epochs for "best model":
    with h5py.File(log_dir + model_id + '_training_logs.hdf5', "a") as f:
        f.create_dataset('epoch_best_model',dtype=float, data = epoch_best_model)

    print(f'Total training time: {time.time()-start_global:.2f} sec\n')


# function to generate training data (=> with random jittering applied!) Not for testing!

# function that collects input-target image data of one batch in tf.tensors -> creates 1 batch of training data
# function is given (year,index)-pairs for all samples within the batch: e.g. sample_idc = [[1995,143],[1995,16],[1996,163],[1996,110]], if bs = 4
# input_image and target are tf.tensors of shape (bs,256,256,1) and (bs,256,256,num_comp), data type float32 (contain data of 1 batch)

def batch_data_from_h5py(filename,sample_idc):

    bs_local = sample_idc.shape[0] # local batch size, as the last batch of the epoch could be of smaller size than v.bs (if num_total % v.bs != 0)
    
    with h5py.File(filename,"r") as f:

        inp_samples = np.zeros((bs_local,dim,dim,1))
        tar_samples = np.zeros((bs_local,dim,dim,v.num_comp))

        # collect the samples of the batch from the h5py file:
        for i in range(bs_local): # loop over samples in the batch
            group_name = str(sample_idc[i,0]) # e.g. '1996' -> the year of the i-th sample in the batch 
            inp = f[group_name+'/input'][sample_idc[i,1],:,:] # sample_idc[i,1] is the index of the i-th sample in the batch within the year
            inp = inp[...,tf.newaxis] # add "channels" axis to input image. of shape dim x dim x 1 -> is needed for tf.image.resize in random_jitter
            tar = f[group_name+'/target'][sample_idc[i,1],:,:,:] # of shape dim x dim x num_comp
            
            # special case of data augmentation: data of selected channels (as specified in v.mult_comp) is multiplied by random value between v.mult_factor_min_max[0] and v.mult_factor_min_max[1]  
            for comp in range(v.num_comp): 
                if v.mult_comp[comp] == 1:
                    factor = tf.random.uniform(shape=(),minval=v.mult_factor_min_max[0],maxval=v.mult_factor_min_max[1]) 
                    inp[:,:,0] = inp[:,:,0] - tar[:,:,comp] + tar[:,:,comp] * factor
                    tar[:,:,comp] = tar[:,:,comp] * factor

            inp, tar = random_jitter(inp, tar)

            inp_samples[i,:,:,:] = inp
            tar_samples[i,:,:,:] = tar

    inp32 = np.float32(inp_samples)
    tar32 = np.float32(tar_samples)

    dataset = tf.data.Dataset.from_tensor_slices(inp32)
    batch_dataset = dataset.batch(bs_local)
    # extract (the) one batch from a batch dataset and give back a tensor of size (batchsize,dim,dim,1) (-> necessary as this format is needed by function train_step):
    input_image = next(iter(batch_dataset.take(1))) 
    
    dataset = tf.data.Dataset.from_tensor_slices(tar32)
    batch_dataset = dataset.batch(bs_local)
    target = next(iter(batch_dataset.take(1)))
            
    return input_image, target

"""j
## Train a new model
j"""


"""j
specify folder structure:
j"""

# specify location where checkpoints should be saved:
checkpoint_dir = model_folder + '/training_results_02/checkpoints/'

# specify location where log files should be saved:
log_dir = model_folder + '/training_results_02/logs/'

# specify location where predicted data should be saved:
data_dir_02 = model_folder + '/training_results_02/predicted_true_signals/'

"""j
create new directories:
j"""

os.makedirs(log_dir)

os.makedirs(data_dir_02)

"""j
create datasets in hdf5 file to store performance data during training:
j"""

with h5py.File(log_dir + model_id + '_training_logs.hdf5', "a") as f:

    f.create_dataset('RMS_train_comp', (v.num_comp,v.num_epochs), dtype=float)
    f.create_dataset('RMS_val_comp', (v.num_comp,v.num_epochs), dtype=float)
    f.create_dataset('RMS_test_comp', (v.num_comp,v.num_epochs), dtype=float)

    f.create_dataset('RMS_train_sum', (v.num_epochs,), dtype=float)
    f.create_dataset('RMS_val_sum', (v.num_epochs,), dtype=float)
    f.create_dataset('RMS_test_sum', (v.num_epochs,), dtype=float)

    f.create_dataset('L2_loss_comp', (v.num_comp,v.num_epochs), dtype=float)
    f.create_dataset('L2_loss', (v.num_epochs,), dtype=float)
    f.create_dataset('sum_loss', (v.num_epochs,), dtype=float)
    f.create_dataset('total_loss', (v.num_epochs,), dtype=float)
    f.create_dataset('slope_value_loss', (v.num_epochs,), dtype=float)
    f.create_dataset('linear_shape_loss', (v.num_epochs,), dtype=float)

    f.create_dataset('total_loss_val', (v.num_epochs,), dtype=float)


# check structure of hdf5 file:

#f = nx.nxload(log_dir + model_id + '_training_logs.hdf5')
#print(f.tree)

"""j
Write information on model to info file:
j"""

with open(model_folder + '/info.txt', 'w') as f:
    f.write('Input parameters used for model ' + model_id + ':\n')
    f.write('\n')
    f.write('### Data specifications: ###' + '\n')
    f.write('- signals to be separated: components = ' + ', '.join(v.components) + '\n')
    f.write('- spatial resolution: Nmax_orig = ' + str(v.Nmax_orig) + ', Nmax_flt = ' + str(v.Nmax_flt) + '\n')
    f.write('- temporal resolution: time_resolution_in_hours = ' + str(v.deltat) + '\n')
    f.write('- used data from SH files in folder names with ending: latlont_string = ' + v.latlont_string + '\n')
    f.write('- physical units: unit = ' + v.unit + '\n')
    f.write('- extent of grid: extent = ' + v.extent + '\n')
    if v.extent == 'regional':
        f.write('- regional grid includes latitudes between ' + str(v.lat_interval_regional[0]) + ' and ' + str(v.lat_interval_regional[1]) + '° \n')
        f.write('- regional grid includes longitudes between ' + str(v.lon_interval_regional[0]) + ' and ' + str(v.lon_interval_regional[1]) + '° \n')
    f.write('- dimensions of square samples: ' + str(dim) + ' x ' + str(dim)  + '\n')
    f.write('\n')
    f.write('### Data handling specifications: ###' + '\n')
    f.write('- train_years = ' + ', '.join(v.train_years) + '\n')
    f.write('- validation_year = ' + v.validation_year + '\n')
    f.write('- test_year = ' + v.test_year + '\n')
    f.write('- sampling method: sampling = ' + v.sampling + '\n')
    f.write('- random linear combination of samples: random_samples = ' + v.random_samples + '\n')
    f.write('- switch on (1) or off (0) the 5 data augmentation strategies: data_augmentation = ' + ', '.join(list(map(str,v.data_augmentation))) + '\n')
    f.write('- switch on (1) or off (0) the trend data augmentation for individual components: mult_comp = ' + ', '.join(list(map(str,v.mult_comp))) + '\n')
    f.write('- Interval for random multiplication factor for trend data augmentation: mult_factor_min_max = ' + ', '.join(list(map(str,v.mult_factor_min_max))) + '\n')
    f.write('\n')
    f.write('### Training specifications: ###' + '\n')
    f.write('- dimensions of filters in convolutional layers: filter_size = ' + str(v.filter_size) + '\n')
    f.write('- number of filters in convolutional layers: num_filters_min = ' + str(v.num_filters_min) + ', num_filters_max = ' + str(v.num_filters_max) + '\n')
    f.write('- batch size: batch_size = ' + str(v.bs) + '\n')
    f.write('- learning rate: learning_rate = ' + str(v.lr) + '\n')
    f.write('- parameters of Adam optimizer: beta_1 = ' + str(v.beta_1) + ', beta_2 = ' + str(v.beta_2) + '\n')
    f.write('- minimum number of epochs: min_number_epochs = ' + str(v.min_number_epochs) + '\n')
    f.write('- maximum number of epochs: num_epochs = ' + str(v.num_epochs) + '\n')
    f.write('- normalization type of component and sum loss terms: loss_type = ' + v.loss_type + '\n')
    f.write('- weights of component loss terms: weights_comp = ' + ', '.join(list(map(str,v.weights_comp))) + '\n')
    f.write('- weight of sum loss term: weight_sum = ' + str(v.weight_sum[0]) + '\n')
    f.write('- weights of loss terms constraining the slope of the data along the time direction: weights_slope_value = ' + ', '.join(list(map(str,v.weights_slope_value)))+ '\n')
    f.write('- weights of loss terms constraining the linear shape of the data along the time direction: weights_linear_shape = ' + ', '.join(list(map(str,v.weights_linear_shape)))+ '\n')



# define the optimizers (depend on learning rate)

generator_optimizer = tf.keras.optimizers.Adam(v.lr, beta_1=v.beta_1, beta_2=v.beta_2)
# parameter lr: learning rate
# parameter beta_1: controlling amount of momentum introduced (momentum means smoothing between the parameter updates of subsequent iterations, that's the momentum part of the optimizer)
# parameter beta_2: controlling amount of dampening of the larger gradient components (that's the RMSprop part of the optimizer)


# create checkpoint object
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

# use a checkpoint manager such that old checkpoints are deleted automatically
manager = tf.train.CheckpointManager(checkpoint,directory=checkpoint_dir,max_to_keep=1)


# do the training

fit(inp_tar_folder+inp_tar_file)


"""j
Restore latest checkpoint and save true and predicted signals for test data year to hdf5 file:
j"""

# restore latest checkpoint 
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# compute prediction errors of a before-trained model (generator) and save them to a hdf5 file:

years = v.train_years.copy()
years.append(v.validation_year)
years.append(v.test_year) # list of strings giving the years for which data pairs are needed


for year in years:
    true_pred_arrays(inp_tar_folder+inp_tar_file,year,generator,data_dir_02) # test performance of all training data years


# create success file

f=open('02_success','w')
f.close()

