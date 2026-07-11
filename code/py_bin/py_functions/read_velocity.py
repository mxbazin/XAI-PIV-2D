# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
read_velocity.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 15:18:38 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (399x199, 2 components u/v)

File to read the data of the velocity fields. The file contains the following functions:
    Functions:
        - read_velocity : file to read the velocity (u and v only, 2D PIV)
"""
# -----------------------------------------------------------------------------------------------------------------------
# Read the packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------

def read_velocity(data_in={"folder":"../../piv_data","file":"piv.$INDEX$.h5","index":0,
                           "dx":1,"dy":1,"shpx":399,"shpy":199,"padding":0,
                           "data_folder":"Data","umean_file":"Umean.txt"}):
    """
    .....................................................................................................................
    # read_velocity: Function to read the 2D PIV velocity (u and v components only).
                     Returns fluctuating velocities after subtracting the temporal mean.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - folder      : folder of the velocity data
            - file        : file name with $INDEX$ placeholder
            - index       : snapshot index
            - dx          : downsampling in x
            - dy          : downsampling in y
            - shpx        : number of grid points in x (after downsampling)
            - shpy        : number of grid points in y (after downsampling)
            - padding     : number of padding nodes (0 for non-periodic PIV)
            - data_folder : folder containing the mean velocity file
            - umean_file  : file of the mean velocity

    Returns
    -------
    dict
        Data:
            - uu : u velocity fluctuation, shape (shpy, shpx+2*padding)
            - vv : v velocity fluctuation, shape (shpy, shpx+2*padding)
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import h5py
    from py_bin.py_functions.umean import read_Umean
    from py_bin.py_functions.padding_field import padding_field

    # -------------------------------------------------------------------------------------------------------------------
    # Read the input parameters
    # -------------------------------------------------------------------------------------------------------------------
    folder        = str(data_in["folder"])
    file          = str(data_in["file"])
    index         = str(int(data_in["index"]))
    dx            = int(data_in["dx"])
    dy            = int(data_in["dy"])
    shpx          = int(data_in["shpx"])
    shpy          = int(data_in["shpy"])
    padding       = int(data_in["padding"])
    data_folder   = str(data_in["data_folder"])
    umean_file    = str(data_in["umean_file"])
    file_complete = folder+'/'+file
    file_ii       = file_complete.replace("$INDEX$", index)

    # -------------------------------------------------------------------------------------------------------------------
    # Read mean velocity (u only; v mean is assumed zero or negligible)
    # -------------------------------------------------------------------------------------------------------------------
    try:
        dataUmean  = {"folder":data_folder,"file":umean_file,"dy":dy}
        meanU_data = read_Umean(dataUmean)
    except:
        print("Mean velocity file needs to be provided. Breaking calculation...", flush=True)
        sys.exit()
    UUmean = meanU_data["UUmean"]
    print('Reading field: '+str(file_ii), flush=True)

    # -------------------------------------------------------------------------------------------------------------------
    # Read u and v from HDF5. Expected datasets: 'u' and 'v', shape (my, mx).
    # -------------------------------------------------------------------------------------------------------------------
    file_h5 = h5py.File(file_ii, 'r')
    UU = np.array(file_h5['u'])[::dy, ::dx]
    uu = UU - UUmean.reshape(-1, 1)
    vv = np.array(file_h5['v'])[::dy, ::dx]
    file_h5.close()

    # -------------------------------------------------------------------------------------------------------------------
    # Apply padding if necessary (periodic BC in x)
    # -------------------------------------------------------------------------------------------------------------------
    if padding > 0:
        uu = padding_field(data_in={"field":uu,"shpx":shpx,"shpy":shpy,"padding":padding})["field"]
        vv = padding_field(data_in={"field":vv,"shpx":shpx,"shpy":shpy,"padding":padding})["field"]

    data_output = {"uu":uu, "vv":vv}
    return data_output