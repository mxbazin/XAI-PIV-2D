# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
shaprms.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Mar 27 08:50:04 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no z dimension, u and v only)

File to create the RMS values for the SHAP fields. Functions contained in the file:
    Functions:
        - read_rms        : function to read the rms of the SHAP
        - save_rms        : function to save the rms of the SHAP
        - calc_rms        : function to calculate the rms of the SHAP
        - calc_rms_nomean : function to calculate the rms minus the mean
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import glob

# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------

def read_rms(data_in={"file":"SHAPrms.txt","folder":"Data"}):
    """
    .....................................................................................................................
    # read_rms: Function for reading the RMS of the SHAP (2D PIV: u and v only)
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data:
            - file   : file of the RMS of the SHAP
            - folder : folder of the generated data

    Returns
    -------
    data_out : dict
        Data:
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the absolute value of the SHAP
    """
    file   = str(data_in["file"])
    folder = str(data_in["folder"])
    file_rms  = folder+'/'+file
    file_read = open(file_rms,"r")
    SHAP_urms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_vrms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_uv   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_mrms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_uv":SHAP_uv,"SHAP_mrms":SHAP_mrms}
    return data_out


def save_rms(data_in={"file":"SHAPrms.txt","folder":"Data","SHAP_urms":[],"SHAP_vrms":[],
                      "SHAP_uv":[],"SHAP_mrms":[]}):
    """
    .....................................................................................................................
    # save_rms: Function for saving the RMS of the SHAP (2D PIV: u and v only)
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data:
            - file      : file of the RMS of the SHAP
            - folder    : folder of the generated data
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the magnitude of the SHAP

    Returns
    -------
    None.
    """
    file      = str(data_in["file"])
    folder    = str(data_in["folder"])
    SHAP_urms = np.array(data_in["SHAP_urms"],dtype='float')
    SHAP_vrms = np.array(data_in["SHAP_vrms"],dtype='float')
    SHAP_uv   = np.array(data_in["SHAP_uv"],dtype='float')
    SHAP_mrms = np.array(data_in["SHAP_mrms"],dtype='float')
    file_rms  = folder+'/'+file
    file_save = open(file_rms, "w+")
    file_save.write(str(SHAP_urms.tolist())+'\n')
    file_save.write(str(SHAP_vrms.tolist())+'\n')
    file_save.write(str(SHAP_uv.tolist())+'\n')
    file_save.write(str(SHAP_mrms.tolist())+'\n')
    file_save.close()


def calc_rms(data_in={"field_ini":0,"field_fin":9999,"field_delta":1,"SHAPmean_file":"SHAPmean.txt",
                      "data_folder":"Data","file":"piv.$INDEX$.h5.shap","folder":"../data/SHAP",
                      "dx":1,"dy":1,"shpx":319,"shpy":199,"save_file":True,
                      "SHAPrms_file":"SHAPrms.txt"}):
    """
    .....................................................................................................................
    # calc_rms: Calculate the RMS of the SHAP along the wall-normal direction (2D PIV: u and v only)
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data:
            - field_ini     : index of the initial field
            - field_fin     : index of the final field
            - field_delta   : separation between fields
            - SHAPmean_file : file of the mean SHAP
            - data_folder   : folder of the generated data
            - file          : file of the SHAP fields (with $INDEX$)
            - folder        : folder of the SHAP fields
            - dx            : downsampling in x
            - dy            : downsampling in y
            - shpx          : shape of the tensor in x (after downsampling)
            - shpy          : shape of the tensor in y (after downsampling)
            - save_file     : flag to save in a file
            - SHAPrms_file  : file to store the RMS

    Returns
    -------
    data_out : dict (only when save_file=False)
        Data:
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the magnitude of the SHAP
    """
    import h5py

    field_ini     = int(data_in["field_ini"])
    field_fin     = int(data_in["field_fin"])
    field_delta   = int(data_in["field_delta"])
    data_folder   = str(data_in["data_folder"])
    file          = str(data_in["file"])
    folder        = str(data_in["folder"])
    dy            = int(data_in["dy"])
    dx            = int(data_in["dx"])
    shpx          = int(data_in["shpx"])
    shpy          = int(data_in["shpy"])
    save_file     = bool(data_in["save_file"])
    SHAPrms_file  = str(data_in["SHAPrms_file"])
    file_comp     = folder+'/'+file

    SHAP_u2_cum = None
    for ii in range(field_ini, field_fin, field_delta):
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('RMS SHAP calculation: '+str(file_ii), flush=True)
        if glob.glob(file_ii):
            fh       = h5py.File(file_ii,'r+')
            SHAP_u   = np.array(fh['SHAP_u'])[::dy, ::dx]   # (shpy, shpx)
            SHAP_v   = np.array(fh['SHAP_v'])[::dy, ::dx]
            fh.close()
            SHAP_m2  = SHAP_u**2 + SHAP_v**2
            SHAP_u2  = SHAP_u**2
            SHAP_v2  = SHAP_v**2
            SHAP_uv  = SHAP_u * SHAP_v
            if SHAP_u2_cum is None:
                SHAP_u2_cum = np.sum(SHAP_u2, axis=1)   # (shpy,)
                SHAP_v2_cum = np.sum(SHAP_v2, axis=1)
                SHAP_uv_cum = np.sum(SHAP_uv, axis=1)
                SHAP_m2_cum = np.sum(SHAP_m2, axis=1)
                nn_cum      = np.ones((shpy,)) * shpx
            else:
                SHAP_u2_cum += np.sum(SHAP_u2, axis=1)
                SHAP_v2_cum += np.sum(SHAP_v2, axis=1)
                SHAP_uv_cum += np.sum(SHAP_uv, axis=1)
                SHAP_m2_cum += np.sum(SHAP_m2, axis=1)
                nn_cum      += np.ones((shpy,)) * shpx
        else:
            print('Skipping field '+str(ii)+' as file was not found', flush=True)

    if SHAP_u2_cum is None:
        print("No SHAP files found. Aborting.", flush=True)
        sys.exit()

    SHAP_urms = np.sqrt(np.divide(SHAP_u2_cum, nn_cum))
    SHAP_vrms = np.sqrt(np.divide(SHAP_v2_cum, nn_cum))
    SHAP_uv   = np.divide(SHAP_uv_cum, nn_cum)
    SHAP_mrms = np.sqrt(np.divide(SHAP_m2_cum, nn_cum))

    if save_file:
        save_rms(data_in={"folder":data_folder,"file":SHAPrms_file,"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,
                          "SHAP_uv":SHAP_uv,"SHAP_mrms":SHAP_mrms})
    else:
        return {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_uv":SHAP_uv,"SHAP_mrms":SHAP_mrms}


def calc_rms_nomean(data_in={"field_ini":0,"field_fin":9999,"field_delta":1,"SHAPmean_file":"SHAPmean.txt",
                             "data_folder":"Data","file":"piv.$INDEX$.h5.shap","folder":"../data/SHAP",
                             "dx":1,"dy":1,"shpx":319,"shpy":199,"save_file":True,
                             "SHAPrms_file":"SHAPrms.txt"}):
    """
    .....................................................................................................................
    # calc_rms_nomean: RMS of SHAP minus the temporal mean (2D PIV: u and v only)
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Same structure as calc_rms.

    Returns
    -------
    data_out : dict (only when save_file=False)
        Same keys as calc_rms.
    """
    from py_bin.py_functions.shapmean import read_SHAPmean
    import h5py

    field_ini     = int(data_in["field_ini"])
    field_fin     = int(data_in["field_fin"])
    field_delta   = int(data_in["field_delta"])
    SHAPmean_file = str(data_in["SHAPmean_file"])
    data_folder   = str(data_in["data_folder"])
    file          = str(data_in["file"])
    folder        = str(data_in["folder"])
    dy            = int(data_in["dy"])
    dx            = int(data_in["dx"])
    shpx          = int(data_in["shpx"])
    shpy          = int(data_in["shpy"])
    save_file     = bool(data_in["save_file"])
    SHAPrms_file  = str(data_in["SHAPrms_file"])
    SHAPrms_file  = SHAPrms_file.replace(".txt","_nomean.txt")
    file_comp     = folder+'/'+file

    try:
        data_SHAPmean = read_SHAPmean(data_in={"folder":data_folder,"file":SHAPmean_file,"dy":dy})
        SHAP_umean    = data_SHAPmean["SHAP_umean"]   # (shpy,)
        SHAP_vmean    = data_SHAPmean["SHAP_vmean"]
    except:
        print("RMS calculations require mean SHAP file. Breaking calculation...", flush=True)
        sys.exit()

    SHAP_u2_cum = None
    for ii in range(field_ini, field_fin, field_delta):
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('RMS SHAP (nomean) calculation: '+str(file_ii), flush=True)
        if glob.glob(file_ii):
            fh     = h5py.File(file_ii,'r+')
            SHAP_u = np.array(fh['SHAP_u'])[::dy, ::dx] - SHAP_umean.reshape(-1, 1)  # (shpy, shpx)
            SHAP_v = np.array(fh['SHAP_v'])[::dy, ::dx] - SHAP_vmean.reshape(-1, 1)
            fh.close()
            SHAP_m2 = SHAP_u**2 + SHAP_v**2
            SHAP_u2 = SHAP_u**2
            SHAP_v2 = SHAP_v**2
            SHAP_uv = SHAP_u * SHAP_v
            if SHAP_u2_cum is None:
                SHAP_u2_cum = np.sum(SHAP_u2, axis=1)
                SHAP_v2_cum = np.sum(SHAP_v2, axis=1)
                SHAP_uv_cum = np.sum(SHAP_uv, axis=1)
                SHAP_m2_cum = np.sum(SHAP_m2, axis=1)
                nn_cum      = np.ones((shpy,)) * shpx
            else:
                SHAP_u2_cum += np.sum(SHAP_u2, axis=1)
                SHAP_v2_cum += np.sum(SHAP_v2, axis=1)
                SHAP_uv_cum += np.sum(SHAP_uv, axis=1)
                SHAP_m2_cum += np.sum(SHAP_m2, axis=1)
                nn_cum      += np.ones((shpy,)) * shpx
        else:
            print('Skipping field '+str(ii)+' as file was not found', flush=True)

    if SHAP_u2_cum is None:
        print("No SHAP files found. Aborting.", flush=True)
        sys.exit()

    SHAP_urms = np.sqrt(np.divide(SHAP_u2_cum, nn_cum))
    SHAP_vrms = np.sqrt(np.divide(SHAP_v2_cum, nn_cum))
    SHAP_uv   = np.divide(SHAP_uv_cum, nn_cum)
    SHAP_mrms = np.sqrt(np.divide(SHAP_m2_cum, nn_cum))

    if save_file:
        save_rms(data_in={"folder":data_folder,"file":SHAPrms_file,"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,
                          "SHAP_uv":SHAP_uv,"SHAP_mrms":SHAP_mrms})
    else:
        return {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_uv":SHAP_uv,"SHAP_mrms":SHAP_mrms}