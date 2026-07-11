# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
main_statistics.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Mar 27 08:23:08 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no z dimension)

File to create the statistics of the flow fields: mean, rms, normalization.
"""
# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitions of the parameters
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
st_data_str = "stats_data"

# -----------------------------------------------------------------------------------------------------------------------
# Import Packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_functions.umean import calc_Umean
from py_bin.py_functions.urms import calc_rms
from py_bin.py_class.flow_field import flow_field
import os

# -----------------------------------------------------------------------------------------------------------------------
# Unlock the h5 files for avoiding problems in some clusters
# -----------------------------------------------------------------------------------------------------------------------
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# -----------------------------------------------------------------------------------------------------------------------
# Import information files
# -----------------------------------------------------------------------------------------------------------------------
exec("from "+folder_def+" import "+chd_str+" as chd")
exec("from "+folder_def+" import "+folders_str+" as folders")
exec("from "+folder_def+" import "+st_data_str+" as st_data")

# -----------------------------------------------------------------------------------------------------------------------
# Read parameters (2D PIV: no L_z, no dz)
# -----------------------------------------------------------------------------------------------------------------------
field_ini   = st_data.field_ini
field_fin   = st_data.field_fin
folder      = folders.uvw_folder
file        = folders.uvw_file
save_file   = st_data.save_file
umean_file  = folders.umean_file
data_folder = folders.data_folder
dx          = chd.dx
dy          = chd.dy
L_x         = chd.L_x
L_y         = chd.L_y
unorm_file  = folders.unorm_file
urms_file   = folders.urms_file
rey         = chd.rey
utau        = chd.utau
mean_norm   = bool(st_data.mean_norm)

if mean_norm:
    from py_bin.py_functions.normalization_normaldist import calc_norm
else:
    from py_bin.py_functions.normalization import calc_norm

# -----------------------------------------------------------------------------------------------------------------------
# Create the folder to store the data
# -----------------------------------------------------------------------------------------------------------------------

os.makedirs(data_folder, exist_ok=True)

#try:
    #os.mkdir(data_folder)
    #os.makedirs(data_folder, exist_ok=True)

#except:
    #pass

# -----------------------------------------------------------------------------------------------------------------------
# Obtain the flow characteristics (2D PIV: no down_z, no L_z)
# -----------------------------------------------------------------------------------------------------------------------
Data_flow = {"folder":folder,"file":file,"down_x":dx,"down_y":dy,"L_x":L_x,"L_y":L_y,"rey":rey,"utau":utau}
flow = flow_field(data_in=Data_flow)
flow.shape_tensor()

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the mean values of the velocity (2D PIV: no shpz)
# -----------------------------------------------------------------------------------------------------------------------
Data_umean = {"field_ini":field_ini,"field_fin":field_fin,"folder":folder,"file":file,"save_file":save_file,
              "umean_file":umean_file,"data_folder":data_folder,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":1}
calc_Umean(data_in=Data_umean)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the RMS of the velocity (2D PIV: no shpz, no dz)
# -----------------------------------------------------------------------------------------------------------------------
data_rms = {"field_ini":field_ini,"field_fin":field_fin,"umean_file":umean_file,"data_folder":data_folder,
            "file":file,"folder":folder,"dx":dx,"dy":dy,"dz":1,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":1,
            "save_file":save_file,"urms_file":urms_file}
calc_rms(data_in=data_rms)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the normalization values (2D PIV: no shpz, no dz)
# -----------------------------------------------------------------------------------------------------------------------
data_norm = {"field_ini":field_ini,"field_fin":field_fin,"data_folder":data_folder,"umean_file":umean_file,
             "dx":dx,"dy":dy,"dz":1,"folder":folder,"file":file,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":1,
             "save_file":save_file,"unorm_file":unorm_file}
calc_norm(data_in=data_norm)