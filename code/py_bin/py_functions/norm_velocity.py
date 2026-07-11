# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
norm_velocity.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 11:59:50 2024

@author:  Andres Cremades Botella
Adapted for 2D PIV data (u and v only, no w component)

File to normalize the velocity fields. The normalization generates values between 0 and 1 using the
minimum and the maximum of the velocity values. The file contains the following functions:
    Functions:
        - norm_velocity : function for normalizing the velocity (u, v)
        - dim_velocity  : function for dimensionalizing the velocity (u, v)
"""
import sys
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------

def norm_velocity(data_in={"uu":[],"vv":[],"folder_data":"Data","unorm_file":"norm.txt",
                           "data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # norm_velocity: Normalize u and v velocity fluctuations.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - uu          : u velocity fluctuation
            - vv          : v velocity fluctuation
            - folder_data : folder containing the normalization file
            - unorm_file  : file with the normalization values
            - data_type   : data type (float32 or float16)
            - mean_norm   : True = normalize with mean/std, False = normalize with min/max

    Returns
    -------
    dict
        Data:
            - unorm : normalized u velocity
            - vnorm : normalized v velocity
    """
    uu        = np.array(data_in["uu"], dtype="float")
    vv        = np.array(data_in["vv"], dtype="float")
    mean_norm = bool(data_in["mean_norm"])

    if mean_norm:
        from py_bin.py_functions.normalization_normaldist import read_norm
    else:
        from py_bin.py_functions.normalization import read_norm

    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])
        if not (data_type == "float32" or data_type == "float16"):
            data_type = "float32"
    else:
        print("[norm_velocity] Data type needs to be selected.")
        sys.exit()

    folder_data = str(data_in["folder_data"])
    unorm_file  = str(data_in["unorm_file"])
    unorm_data  = {"folder":folder_data,"file":unorm_file}
    try:
        norm_param = read_norm(unorm_data)
    except:
        print('Normalization file could not be located. Calculation is stopped...', flush=True)
        sys.exit()

    if mean_norm:
        uumean = float(norm_param["uumean"])
        vvmean = float(norm_param["vvmean"])
        uustd  = float(norm_param["uustd"])
        vvstd  = float(norm_param["vvstd"])
        unorm = np.array((uu-uumean)/uustd, dtype=data_type)
        vnorm = np.array((vv-vvmean)/vvstd, dtype=data_type)
    else:
        uumax = float(norm_param["uumax"])
        vvmax = float(norm_param["vvmax"])
        uumin = float(norm_param["uumin"])
        vvmin = float(norm_param["vvmin"])
        print(uumax)
        unorm = np.array((uu-uumin)/(uumax-uumin), dtype=data_type)
        vnorm = np.array((vv-vvmin)/(vvmax-vvmin), dtype=data_type)

    data_out = {"unorm":unorm,"vnorm":vnorm}
    return data_out


def dim_velocity(data_in={"unorm":[],"vnorm":[],"folder_data":"Data","unorm_file":"norm.txt",
                          "data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # dim_velocity: Dimensionalize u and v velocity (inverse of norm_velocity).
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - unorm       : normalized u velocity
            - vnorm       : normalized v velocity
            - folder_data : folder containing the normalization file
            - unorm_file  : file with the normalization values
            - data_type   : data type (float32 or float16)
            - mean_norm   : True = was normalized with mean/std, False = was normalized with min/max

    Returns
    -------
    dict
        Data:
            - uu : dimensional u velocity
            - vv : dimensional v velocity
    """
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])
        if not (data_type == "float32" or data_type == "float16"):
            data_type = "float32"
    else:
        print("[dim_velocity] Data type needs to be selected.")
        sys.exit()

    uu_norm   = np.array(data_in["unorm"], dtype=data_type)
    vv_norm   = np.array(data_in["vnorm"], dtype=data_type)
    mean_norm = bool(data_in["mean_norm"])

    if mean_norm:
        from py_bin.py_functions.normalization_normaldist import read_norm
    else:
        from py_bin.py_functions.normalization import read_norm

    folder_data = str(data_in["folder_data"])
    unorm_file  = str(data_in["unorm_file"])
    unorm_data  = {"folder":folder_data,"file":unorm_file}
    try:
        norm_param = read_norm(unorm_data)
    except:
        print('Normalization file could not be located. Calculation is stopped...', flush=True)
        sys.exit()

    if mean_norm:
        uumean = norm_param["uumean"]
        vvmean = norm_param["vvmean"]
        uustd  = norm_param["uustd"]
        vvstd  = norm_param["vvstd"]
        uu = np.array(uu_norm*uustd+uumean, dtype=data_type)
        vv = np.array(vv_norm*vvstd+vvmean, dtype=data_type)
    else:
        uumax = norm_param["uumax"]
        vvmax = norm_param["vvmax"]
        uumin = norm_param["uumin"]
        vvmin = norm_param["vvmin"]
        uu = np.array(uu_norm*(uumax-uumin)+uumin, dtype=data_type)
        vv = np.array(vv_norm*(vvmax-vvmin)+vvmin, dtype=data_type)

    data_out = {"uu":uu,"vv":vv}
    return data_out