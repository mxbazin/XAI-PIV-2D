# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
flow_field.py
-------------------------------------------------------------------------------------------------------------------------
Created on Mon Mar 25 11:07:26 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (399x199, 2 components u/v)

File to read the geometric characteristics of the data. The file contains a class:
    Class:
        - flow_field: class containing the information of the flow field. The class has the following functions
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import glob
import h5py
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the class
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class flow_field():
    """
    .....................................................................................................................
    # flow_field: Class to calculate and store the geometrical characteristics of the 2D PIV flow field.
        * Functions:
            - __init__     : initialization function
            - shape_tensor : function to calculate the shape of the tensors of the flow field
            - flow_grid    : calculate the geometry of the mesh
        * Variables:
            - folder   : folder of the flow fields
            - file     : file of the flow fields
            - down_x   : downsampling in x
            - down_y   : downsampling in y
            - L_x      : domain size in x
            - L_y      : domain size in y
            - rey      : Reynolds number (or 1 if not used)
            - utau     : reference velocity (or 1 if not used)
            - mx       : grid points in x
            - my       : grid points in y
            - shpx     : shape of the tensors in x
            - shpy     : shape of the tensors in y
            - delta_x  : grid spacing in x
            - delta_y  : grid spacing in y (scalar for uniform grid)
            - x_h      : grid in x
            - y_h      : grid in y
            - vol_h    : area of the grid cells (2D)
            - voltot   : total area of the domain
    .....................................................................................................................
    """
    def __init__(self, data_in={"folder":"../../piv_data/","file":"piv.$INDEX$.h5",
                                "down_x":1,"down_y":1,"L_x":399,"L_y":199,
                                "rey":1,"utau":1}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Initialize the flow class for 2D PIV data.

        Parameters
        ----------
        data_in : dict
            Data required for obtaining the geometry of the flow.
            Data:
                - folder : folder of the flow fields
                - file   : file of the flow fields (with $INDEX$ placeholder)
                - down_x : downsampling in x
                - down_y : downsampling in y
                - L_x    : domain size in x
                - L_y    : domain size in y
                - rey    : Reynolds number (or 1)
                - utau   : reference velocity (or 1)

        Returns
        -------
        None.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        folder      = str(data_in["folder"])
        file        = str(data_in["file"])
        self.down_x = int(data_in["down_x"])
        self.down_y = int(data_in["down_y"])
        self.L_x    = float(data_in["L_x"])
        self.L_y    = float(data_in["L_y"])
        self.rey    = float(data_in["rey"])
        self.utau   = float(data_in["utau"])

        # --------------------------------------------------------------------------------------------------------------
        # Choose a file from the directory and read the grid dimensions
        # Expects HDF5 files with datasets "mx" and "my" (number of grid points in x and y)
        # --------------------------------------------------------------------------------------------------------------
        file_ref  = file.replace("$INDEX$","*")
        file_com  = folder+'/'+file_ref
        file_base = glob.glob(file_com)[0]
        print("File for measuring the flow: "+file_base, flush=True)
        file_h5   = h5py.File(file_base,'r')
        print("Reading flow field", flush=True)
        self.mx = int(np.array(file_h5["mx"])[0])   # grid points in x
        self.my = int(np.array(file_h5["my"])[0])   # grid points in y
        file_h5.close()
        print("Flow field read", flush=True)

    def shape_tensor(self):
        """
        .................................................................................................................
        # shape_tensor
        .................................................................................................................
        Calculate the shape of the tensors after applying downsampling.

        Returns
        -------
        None.
        """
        self.shpy = int((self.my-1)/self.down_y)+1   # Shape in y
        self.shpx = int((self.mx-1)/self.down_x)+1   # Shape in x

    def flow_grid(self):
        """
        .................................................................................................................
        # flow_grid
        .................................................................................................................
        Calculate the grid geometry for the 2D PIV domain.

        Returns
        -------
        None.
        """
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the grid spacing in x and y (uniform grid assumed)
        # -------------------------------------------------------------------------------------------------------------
        self.delta_x = self.L_x / (self.shpx-1) if self.shpx > 1 else self.L_x
        self.delta_y = self.L_y / (self.shpy-1) if self.shpy > 1 else self.L_y

        # -------------------------------------------------------------------------------------------------------------
        # Build coordinate arrays
        # -------------------------------------------------------------------------------------------------------------
        self.x_h = self.delta_x * np.arange(self.shpx)
        self.y_h = self.delta_y * np.arange(self.shpy)

        # -------------------------------------------------------------------------------------------------------------
        # Viscous-unit coordinates (rey=1 if not used)
        # -------------------------------------------------------------------------------------------------------------
        self.xplus = self.x_h * self.rey
        self.yplus = self.y_h * self.rey

        # -------------------------------------------------------------------------------------------------------------
        # Mid-channel split (kept for compatibility with structure detection code)
        # -------------------------------------------------------------------------------------------------------------
        if np.mod(self.shpy, 2) == 0:
            self.yl_s = int(self.shpy*0.5)
            self.yu_s = int(self.shpy*0.5)
        else:
            self.yl_s = int(self.shpy*0.5)+1
            self.yu_s = int(self.shpy*0.5)

        # -------------------------------------------------------------------------------------------------------------
        # Calculate the area of the grid cells (2D: area instead of volume)
        #     - vol_h    : area per grid cell (shape: (1, shpx))
        #     - voltot   : total domain area
        # -------------------------------------------------------------------------------------------------------------
        self.vol_h    = np.zeros((1, self.shpx)) + self.delta_x * self.delta_y
        self.voltot   = self.L_x * self.L_y