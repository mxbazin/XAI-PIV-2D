# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:40:22 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (399x199, 2 components u/v, 10000 snapshots)

File containing the information about the flow:
    - L_x     : Domain size in x (streamwise)
    - L_y     : Domain size in y (wall-normal / cross-stream)
    - rey     : Reynolds number (set to 1 if non-dimensionalisation not needed)
    - utau    : Friction velocity (set to 1 if not applicable)
    - dx      : Downsampling in x
    - dy      : Downsampling in y
    - padding : Number of nodes of the padding (set 0 if no periodic BC)
    - filvol  : Area filter in physical units (used for structure detection)
"""
# ----------------------------------------------------------------------------------------------------------------------
# Import the packages
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Define the dimensions of the 2D PIV domain
#     - L_x : Domain size in x (streamwise)
#     - L_y : Domain size in y (cross-stream / wall-normal)
# Note: set these to your actual physical domain dimensions if known,
#       or use pixel/grid units (L_x=399, L_y=199 for grid units)
# ----------------------------------------------------------------------------------------------------------------------
L_x = 319   # physical or grid units in x (adapt to your PIV calibration)
L_y = 199   # physical or grid units in y (adapt to your PIV calibration)

# ----------------------------------------------------------------------------------------------------------------------
# Physical parameters
#     - rey  : Reynolds number (set to 1 if viscous scaling not used)
#     - utau : Reference velocity (set to 1 if not applicable)
# ----------------------------------------------------------------------------------------------------------------------
rey  = 1.0
utau = 1.0

# ----------------------------------------------------------------------------------------------------------------------
# Downsampling and padding of the fields
#     - dx      : Downsampling in x (1 = no downsampling)
#     - dy      : Downsampling in y (1 = no downsampling)
#     - padding : Number of nodes of the padding (0 = no padding for non-periodic PIV)
# ----------------------------------------------------------------------------------------------------------------------
dx      = 1
dy      = 1
padding = 0

# ----------------------------------------------------------------------------------------------------------------------
# Filter of the area (replaces volume filter for 2D)
#     - filvol : area filter in physical units (used in structure detection)
# ----------------------------------------------------------------------------------------------------------------------
filvol = 0