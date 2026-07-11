# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
read_norm_velocity.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Apr  5 10:39:47 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (u and v only, no z dimension)

File to read and normalize the 2D PIV flow field.
    Functions:
        - read_norm_velocity : reads the flow and normalizes the values (u, v)
"""
import time

def read_norm_velocity(data_in={"folder":"../../piv_data","file":"piv.$INDEX$.h5",
                                "padding":0,"shpx":399,"shpy":199,"dx":1,"dy":1,
                                "data_folder":"Data","umean_file":"Umean.txt","unorm_file":"norm.txt",
                                "index":0,"data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # read_norm_velocity: Function to read and normalize the 2D PIV velocity (u and v).
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - folder      : folder of the velocity data
            - file        : file name with $INDEX$ placeholder
            - padding     : number of padding nodes (0 for non-periodic PIV)
            - shpx        : number of grid points in x (after downsampling)
            - shpy        : number of grid points in y (after downsampling)
            - dx          : downsampling in x
            - dy          : downsampling in y
            - data_folder : folder containing generated data
            - umean_file  : mean velocity file
            - unorm_file  : normalization file
            - index       : snapshot index
            - data_type   : data type (float32 or float16)
            - mean_norm   : True = normalize with mean/std, False = normalize with min/max

    Returns
    -------
    dict
        Data:
            - norm_velocity : normalized velocity dict (unorm, vnorm)
            - time_read     : time for reading the file
            - time_norm     : time for normalizing
    """
    import sys
    from py_bin.py_functions.read_velocity import read_velocity
    from py_bin.py_functions.norm_velocity import norm_velocity

    folder      = str(data_in["folder"])
    file        = str(data_in["file"])
    padding     = int(data_in["padding"])
    shpx        = int(data_in["shpx"])
    shpy        = int(data_in["shpy"])
    dx          = int(data_in["dx"])
    dy          = int(data_in["dy"])
    data_folder = str(data_in["data_folder"])
    umean_file  = str(data_in["umean_file"])
    unorm_file  = str(data_in["unorm_file"])
    index       = int(data_in["index"])
    mean_norm   = bool(data_in["mean_norm"])
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])
        if not (data_type == "float32" or data_type == "float16"):
            data_type = "float32"
    else:
        print("[read_norm_velocity] Data type needs to be selected.")
        sys.exit()

    tstart             = time.time()
    data_velocity      = {"folder":folder,"file":file,"index":index,"dx":dx,"dy":dy,
                          "shpx":shpx,"shpy":shpy,"padding":padding,
                          "data_folder":data_folder,"umean_file":umean_file}
    data_read_velocity = read_velocity(data_velocity)
    tread              = time.time()
    data_norm          = {"uu":data_read_velocity["uu"],"vv":data_read_velocity["vv"],
                          "folder_data":data_folder,"unorm_file":unorm_file,
                          "data_type":data_type,"mean_norm":mean_norm}
    norm_vel           = norm_velocity(data_norm)
    tnorm              = time.time()

    data_out = {"norm_velocity":norm_vel,"time_read":tread-tstart,"time_norm":tnorm-tread}
    return data_out