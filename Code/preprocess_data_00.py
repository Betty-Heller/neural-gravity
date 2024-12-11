#!/usr/bin/env python3


"""j
# Program to create the data files (SH coefficients) needed by the neural network-based framework for signal separation in spatio-temporal gravity data, at the example of the ESA Earth System Model (Dobslaw et al, 2014)
j"""

"""j
Packages import
j"""

from ftplib import FTP
import tarfile
import os
import shutil
import numpy as np
import pyshtools as pysh
import datetime

"""j
change input parameters here:
j"""

components = ['AO','H','I','S']
years_int = np.arange(1995,2007)
latlont_string = '_yearly_mean_subtracted'
deltat = 168

years = []
for year in years_int:
    years.append(str(year))

"""j
### Step 1: download and prepare SH coefficient files (6h time resolution):
j"""

print('\n')
print('Download and prepare SH coefficient files...')
print('\n')

individual_signals = []

for comp in components:
    for ind_comp in comp:
        individual_signals.append(ind_comp)

#####

# connect to FTP server of GFZ to download AOHIS data (Dobslaw et al., 2014):
print('connect to FTP server...')
ftp = FTP('ig2-dmz.gfz-potsdam.de')  # connect to host, default port
ftp.login()  
ftp.cwd('/ESAESM/mtmshc') # change path to folder that contains the .tar.gz files


for ind_comp in individual_signals: # loop over individual signals (e.g. A, O, H, I, S)

    for year in years:

        # check if data is already there:
        folder = '../Data/SH_coeff/' + ind_comp + '/6h/' + year + '/'
        if not os.path.isdir(folder):
            print('create files in folder  ' + folder)
            os.makedirs(folder, exist_ok=True)

            print('download AOHIS datasets...')

            # --> download .tar.gz file from FTP server:
            file = 'mtm_' + year + '_' + ind_comp + '.tar.gz'
            ftp.retrbinary("RETR " + file ,open(folder + file, 'wb').write)
        
            # --> unpack the downloaded .tar.gz file
            tarfile = tarfile.open(folder + file)
            tarfile.extractall(folder)
            tarfile.close()

            # --> remove the .tar.gz file
            os.remove(folder+file)


            # resolve the folder structure with the monthly folders in the year folders:
            print('resolving monthly folder structure...')

            subfolders = os.listdir(folder)
            for subfolder in subfolders:
                files = os.listdir(folder + '/' + subfolder)
                for file in files:
                    shutil.move(folder + '/' + subfolder + '/' + file,folder + '/' + file)
                shutil.rmtree(folder + '/' + subfolder)


            # replace the first line in all icgem files to the one required by the function read_icgem_gfc:
            print('replace first line in all icgem files...')

            folder = '../Data/SH_coeff/' + ind_comp + '/6h/' + year + '/'
            print(folder)

            onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] # list of file names in folder
            onlyfiles = [item for item in onlyfiles if item[0] == 'm'] # just in case there are any hidden files that we are not interested in

            for file in onlyfiles:
                a_file = open(folder+file, "r")
                list_of_lines = a_file.readlines()
                list_of_lines[0] = "product_type            gravity_field\n" # new first line

                a_file = open(folder+file, "w")
                a_file.writelines(list_of_lines)
                a_file.close()
        else:
            print('data folder ' + folder + ' already exists')

#####

# create coefficient files for components that are the sum of several individual signals:
print('create sum of coefficient files where needed...')

for comp in components:
    if len(comp) > 1:
        print(comp)
        for year in years:
            
            folder_new = '../Data/SH_coeff/' + comp + '/6h/' + year + '/'
            
            if not os.path.isdir(folder_new): 
                print('create files in folder  ' + folder_new)
                os.makedirs(folder_new, exist_ok=True)
                filenames = [f for f in os.listdir('../Data/SH_coeff/' + individual_signals[0] + '/6h/' + year + '/')]

                for file in filenames:

                    cilm_sum = np.zeros([2,181,181])

                    for ind_comp in comp:
                        folder_ind_comp = '../Data/SH_coeff/' + ind_comp + '/6h/' + year + '/'
                        file_ind_comp = file[0:7] + ind_comp + file[8:len(file)]

                        cilm, gm, r0 = pysh.shio.read_icgem_gfc(folder_ind_comp + file_ind_comp)
                        cilm_sum = cilm_sum + cilm

                    file_new = file[0:7] + comp + file[8:len(file)]
                    pysh.shio.write_icgem_gfc(folder_new + file_new, cilm_sum,gm = gm, r0 = r0)

            else:
                print('data folder ' + folder_new + ' already exists')

"""j
### Step 2: build SH coefficient files for a lower temporal resolution 
j"""

#####

# create icgem files with SH coefficients of considered temporal resolution:

deltat_orig = 6 # temporal resolution of given data in hours (is 6h for AOHIS files)
deltat_in = datetime.timedelta(hours=deltat_orig) # temporal resolution of input data (6h)
deltat_out = datetime.timedelta(hours=deltat) # temporal resolution for which to compute mean fields
num_6h_interv = deltat/deltat_orig # ratio of time steps (number of 6h intervals in one mean field to compute)



# global start and end point:
start = datetime.datetime(int(years[0]),1,1,0) 
end = datetime.datetime(int(years[-1])+1,1,1,0) 

for comp in components:
    
    if not os.path.isdir('../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h/'): 
        print('create data folder ../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h')

        # start and end of interval of length deltat:
        from_ = start 
        to_ = from_ + deltat_out

        # iterate over numdays intervals over all years:
        while to_ < end:

            interval_new = from_.strftime("%Y%m%d_%H") + '_' + to_.strftime("%Y%m%d_%H")
            fname_out = 'mtmshc_' + comp + '_' + interval_new + '.180'

            year_folder_out = from_.strftime('%Y') # year of numday interval starting point defines output folder
            folder_out =  '../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h/' + year_folder_out + '/' # folder where newly created deltat icgem files are written to
            
            if os.path.isdir(folder_out):
                print('.',end=" ")
            else:
                os.makedirs(folder_out)

            cilm_sum = np.zeros([2,181,181])
            
            # start and end point of very first 6h-interval
            from_6h = from_
            to_6h = from_6h + deltat_in
            
            # iterate over 6h intervals within the numdays interval:
            while from_6h < to_:

                interval_6h = from_6h.strftime("%Y%m%d_%H") + '_' + to_6h.strftime("%Y%m%d_%H")
                fname_in = 'mtmshc_' + comp + '_' + interval_6h[0:11] + '.180'

                year_input = from_6h.strftime('%Y')
                folder_in = '../Data/SH_coeff/' + comp + '/6h/' + year_input + '/' # folder with 6h icgem files
                
                cilm, gm, r0 = pysh.shio.read_icgem_gfc(folder_in + fname_in)
                cilm_sum = cilm_sum + cilm
                
                # go to next 6h-interval
                from_6h = to_6h 
                to_6h = from_6h + deltat_in
            
            cilm_mean = cilm_sum / num_6h_interv

            pysh.shio.write_icgem_gfc(folder_out + fname_out, cilm_mean,gm = gm, r0 = r0)
            
            # go to next numdays-interval
            from_ = to_ 
            to_ = from_ + deltat_out

    else:
        print('data folder ../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h already exists')

"""j
### Step 3: subtract yearly mean
j"""

# function that creates a list of files in folder:
def file_list(folder):
    tmp = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    files = sorted(tmp) 
    return files

#####

if latlont_string == '_yearly_mean_subtracted':

    for comp in components:
        if not os.path.isdir('../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h_yearly_mean_subtracted/'): 
            print('create data folder ../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h_yearly_mean_subtracted/')

            for year in years:
                print(year)

                folder_in = '../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h/' + year + '/'
                folder_out = '../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h_yearly_mean_subtracted/' + year + '/' # folder where newly created deltat icgem files are written to

                if not os.path.isdir(folder_out):
                    os.makedirs(folder_out)

                files = file_list(folder_in)

                cilm_sum = np.zeros([2,181,181])

                for file in files:
                    cilm, gm, r0 = pysh.shio.read_icgem_gfc(folder_in + file)
                    cilm_sum = cilm_sum + cilm
                    
                cilm_mean = cilm_sum / len(files)

                for file in files:
                    cilm, gm, r0 = pysh.shio.read_icgem_gfc(folder_in + file)
                    pysh.shio.write_icgem_gfc(folder_out + file, cilm-cilm_mean,gm = gm, r0 = r0)

        else:
            print('data folder ../Data/SH_coeff/' + comp + '/' + str(deltat) + 'h_yearly_mean_subtracted/ already exists')
#####



