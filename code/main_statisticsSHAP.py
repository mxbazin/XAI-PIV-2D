# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
main_statisticsSHAP.py
-------------------------------------------------------------------------------------------------------------------------
Adapted for 2D PIV data (no z dimension)
"""
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
st_data_str = "stats_data_shap"
sh_data_str = "shap_data"

from py_bin.py_functions.shapmean import calc_SHAPmean
from py_bin.py_functions.shaprms import calc_rms, calc_rms_nomean
from py_bin.py_class.flow_field import flow_field
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

exec("from "+folder_def+" import "+chd_str+" as chd")
exec("from "+folder_def+" import "+folders_str+" as folders")
exec("from "+folder_def+" import "+st_data_str+" as st_data")
exec("from "+folder_def+" import "+sh_data_str+" as sh_data")

# 2D PIV: no L_z, no dz
field_ini     = st_data.field_ini
field_fin     = st_data.field_fin
field_delta   = st_data.field_delta
folder        = folders.uvw_folder
file          = folders.uvw_file
folder_shap   = folders.shap_folder
file_shap     = folders.shap_file
nsamples      = sh_data.nsamples
file_shap     = file_shap.replace("$NSAMPLES$",str(nsamples))
save_file     = st_data.save_file
SHAPmean_file = folders.SHAPmean_file
data_folder   = folders.data_folder
dx            = chd.dx
dy            = chd.dy
L_x           = chd.L_x
L_y           = chd.L_y
SHAPrms_file  = folders.SHAPrms_file
rey           = chd.rey
utau          = chd.utau
mean_norm     = bool(st_data.mean_norm)

os.makedirs(data_folder, exist_ok=True)

# 2D PIV: no down_z, no L_z
Data_flow = {"folder":folder,"file":file,"down_x":dx,"down_y":dy,"L_x":L_x,"L_y":L_y,"rey":rey,"utau":utau}
flow = flow_field(data_in=Data_flow)
flow.shape_tensor()

# Calculate mean SHAP (2D: no shpz)
Data_shapmean = {"field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"folder":folder_shap,
                 "file":file_shap,"save_file":save_file,"SHAPmean_file":SHAPmean_file,"data_folder":data_folder,
                 "shpx":flow.shpx,"shpy":flow.shpy,"shpz":1}
calc_SHAPmean(data_in=Data_shapmean)

# Calculate RMS SHAP (2D: no shpz, no dz)
data_rms = {"field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"SHAPmean_file":SHAPmean_file,
            "data_folder":data_folder,"file":file_shap,"folder":folder_shap,"dx":dx,"dy":dy,"dz":1,
            "shpx":flow.shpx,"shpy":flow.shpy,"shpz":1,"save_file":save_file,"SHAPrms_file":SHAPrms_file}
calc_rms_nomean(data_in=data_rms)
calc_rms(data_in=data_rms)