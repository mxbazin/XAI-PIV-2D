# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
padding_field.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 15:18:38 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no z dimension)

File to create the padding of a 2D field. The file contains the following functions:
    Functions:
        - padding_field : create the padding of the field in x only (periodic in x)
"""
# -----------------------------------------------------------------------------------------------------------------------
# Read the packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------

def padding_field(data_in={"field":[],"shpx":399,"shpy":199,"padding":0}):
    """
    .....................................................................................................................
    # padding_field: Function to apply periodic padding in the x direction for a 2D field.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - field   : 2D field without padding, shape (shpy, shpx)
            - shpx    : number of grid points in x
            - shpy    : number of grid points in y
            - padding : number of padding nodes in x

    Returns
    -------
    dict
        Data:
            - field : field with padding applied, shape (shpy, shpx+2*padding)
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    field   = np.array(data_in["field"])
    shpx    = int(data_in["shpx"])
    shpy    = int(data_in["shpy"])
    padding = int(data_in["padding"])

    # -------------------------------------------------------------------------------------------------------------------
    # Apply periodic padding in x only
    # -------------------------------------------------------------------------------------------------------------------
    field_pad = np.zeros((shpy, shpx+2*padding))
    field_pad[:, padding:-padding]  = field.copy()
    field_pad[:, :padding]          = field[:, -padding:]
    field_pad[:, -padding:]         = field[:, :padding]
    field = field_pad.copy()

    data_output = {"field":field}
    return data_output