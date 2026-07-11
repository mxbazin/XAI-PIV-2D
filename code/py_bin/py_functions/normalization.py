# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
normalization.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 12:33:24 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (u and v only, no w component)

File to create the normalization values for the velocity fields. The normalization generates
values between 0 and 1 using the minimum and the maximum of the velocity values. The file contains
the following functions:
    Functions:
        - save_norm : function for saving the normalization to a file
        - read_norm : function for reading the normalization file
        - calc_norm : function for calculating the normalization
"""
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------

def save_norm(data_in={"folder":"Data","file":"norm.txt","uumax":0,"vvmax":0,"uumin":0,"vvmin":0,
                       "uvmax":0,"uvmin":0}):
    """
    .....................................................................................................................
    # save_norm: Save the normalization values (max/min of u, v, and uv stress) to a file.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - folder : folder of the generated data
            - file   : file of the normalization data
            - uumax  : maximum u velocity
            - vvmax  : maximum v velocity
            - uumin  : minimum u velocity
            - vvmin  : minimum v velocity
            - uvmax  : maximum uv stress
            - uvmin  : minimum uv stress
    """
    folder = str(data_in["folder"])
    file   = str(data_in["file"])
    uumax  = float(data_in["uumax"])
    vvmax  = float(data_in["vvmax"])
    uumin  = float(data_in["uumin"])
    vvmin  = float(data_in["vvmin"])
    uvmax  = float(data_in["uvmax"])
    uvmin  = float(data_in["uvmin"])

    file_norm = folder+'/'+file
    file_save = open(file_norm, "w+")
    file_save.write(str(uumax)+'\n')
    file_save.write(str(vvmax)+'\n')
    file_save.write(str(uumin)+'\n')
    file_save.write(str(vvmin)+'\n')
    file_save.write(str(uvmax)+'\n')
    file_save.write(str(uvmin)+'\n')
    file_save.close()


def read_norm(data_in={"folder":"Data","file":"norm.txt"}):
    """
    .....................................................................................................................
    # read_norm: Read the normalization file (u, v, uv only).
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - folder : folder to read the data
            - file   : file to read the data

    Returns
    -------
    dict
        Data:
            - uumax : maximum u velocity
            - vvmax : maximum v velocity
            - uumin : minimum u velocity
            - vvmin : minimum v velocity
            - uvmax : maximum uv stress
            - uvmin : minimum uv stress
    """
    folder = str(data_in["folder"])
    file   = str(data_in["file"])

    file_norm = folder+'/'+file
    file_read = open(file_norm, "r")
    uumax = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    vvmax = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    uumin = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    vvmin = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    uvmax = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    uvmin = np.array(file_read.readline().replace('[','').replace(']','').split(','), dtype='float')
    file_read.close()
    data_out = {"uumax":uumax,"vvmax":vvmax,"uumin":uumin,"vvmin":vvmin,"uvmax":uvmax,"uvmin":uvmin}
    return data_out


def calc_norm(data_in={"field_ini":0,"field_fin":9999,"data_folder":"Data","umean_file":"Umean.txt",
                       "dx":1,"dy":1,"folder":"../../piv_data","file":"piv.$INDEX$.h5",
                       "shpx":399,"shpy":199,"save_file":True,"unorm_file":"norm.txt"}):
    """
    .....................................................................................................................
    # calc_norm: Calculate the normalization of the 2D PIV velocity (u and v only).
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - field_ini   : initial snapshot index
            - field_fin   : final snapshot index
            - data_folder : folder for storing generated data
            - umean_file  : file of the mean velocity
            - dx          : downsampling in x
            - dy          : downsampling in y
            - folder      : folder of the velocity data
            - file        : file of the velocity data (with $INDEX$)
            - shpx        : shape in x
            - shpy        : shape in y
            - save_file   : flag to save normalization to file
            - unorm_file  : file to save normalization

    Returns
    -------
    dict (only if save_file=False)
            Data for the normalization. Only returns it in case of not saving a file
        Data:
            - uumax : maximum streamwise velocity
            - vvmax : maximum wall-normal velocity
            - uumin : minimum streamwise velocity
            - vvmin : minimum wall-normal velocity
            - uvmax : maximum uv stress
            - uvmin : minimum uv stress
    """
    from py_bin.py_functions.read_velocity import read_velocity

    field_ini   = int(data_in["field_ini"])
    field_fin   = int(data_in["field_fin"])
    data_folder = str(data_in["data_folder"])
    umean_file  = str(data_in["umean_file"])
    folder      = str(data_in["folder"])
    file        = str(data_in["file"])
    dx          = int(data_in["dx"])
    dy          = int(data_in["dy"])
    shpx        = int(data_in["shpx"])
    shpy        = int(data_in["shpy"])
    save_file   = bool(data_in["save_file"])
    unorm_file  = str(data_in["unorm_file"])

    for ii in range(field_ini, field_fin):
        data_velocity = {"folder":folder,"file":file,"index":ii,"dx":dx,"dy":dy,
                         "shpx":shpx,"shpy":shpy,"padding":0,
                         "data_folder":data_folder,"umean_file":umean_file}
        data_read     = read_velocity(data_velocity)
        uu_i0 = np.array(data_read['uu'], dtype='float')
        vv_i0 = np.array(data_read['vv'], dtype='float')
        uv_i0 = np.multiply(uu_i0, vv_i0)
        if ii == field_ini:
            uumax = np.max(uu_i0);  uumin = np.min(uu_i0)
            vvmax = np.max(vv_i0);  vvmin = np.min(vv_i0)
            uvmax = np.max(uv_i0);  uvmin = np.min(uv_i0)
        else:
            uumax = np.max([uumax, np.max(uu_i0)]); uumin = np.min([uumin, np.min(uu_i0)])
            vvmax = np.max([vvmax, np.max(vv_i0)]); vvmin = np.min([vvmin, np.min(vv_i0)])
            uvmax = np.max([uvmax, np.max(uv_i0)]); uvmin = np.min([uvmin, np.min(uv_i0)])

    if save_file:
        save_norm(data_in={"folder":data_folder,"file":unorm_file,"uumax":uumax,"vvmax":vvmax,
                           "uumin":uumin,"vvmin":vvmin,"uvmax":uvmax,"uvmin":uvmin})
    else:
        return {"uumax":uumax,"vvmax":vvmax,"uumin":uumin,"vvmin":vvmin,"uvmax":uvmax,"uvmin":uvmin}