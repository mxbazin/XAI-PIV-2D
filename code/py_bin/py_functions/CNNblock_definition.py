# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
CNNblock_definition.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 12:55:17 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data: Conv3D -> Conv2D, Conv3DTranspose -> Conv2DTranspose

File for defining the blocks of the DL model. The file contains the following functions:
    Functions:
        - block    : function for defining the convolutional block: CNN2D + BN + Activation
        - invblock : function for defining the inverse convolutional block: CNN2DTranspose + BN + Activation
"""

# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------


def block(data_in={"input":[],"nfil":16,"stride":1,"activ":"relu","kernel":3,"dtype":None}):
    """
    .....................................................................................................................
    # block: Conv2D + BatchNormalization + Activation block.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - input  : tensor with the input of the layer
            - nfil   : number of filters
            - stride : stride
            - activ  : activation function
            - kernel : kernel size (scalar, applied as (kernel, kernel))
            - dtype  : output data type (float32, float16, or None)

    Returns
    -------
    dict
        Data:
            - output : tensor with the output of the layer
    """
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

    xx     = data_in["input"]
    nfil   = int(data_in["nfil"])
    stride = int(data_in["stride"])
    activ  = str(data_in["activ"])
    kern   = int(data_in["kernel"])
    kernel = (kern, kern)
    dtype  = data_in["dtype"]
    flagt  = False
    if dtype is not None:
        if dtype != "float32" or dtype != "float16":
            dtype = "float32"
    else:
        flagt = True

    if flagt:
        xx = Conv2D(nfil, kernel_size=kernel, strides=(stride, stride), padding="same")(xx)
        xx = BatchNormalization()(xx)
        xx = Activation(activ)(xx)
    else:
        xx = Conv2D(nfil, kernel_size=kernel, strides=(stride, stride), padding="same", dtype=dtype)(xx)
        xx = BatchNormalization(dtype=dtype)(xx)
        xx = Activation(activ, dtype=dtype)(xx)

    data_out = {"output":xx}
    return data_out


def invblock(data_in={"input":[],"nfil":16,"stride":1,"activ":"relu","kernel":3}):
    """
    .....................................................................................................................
    # invblock: Conv2DTranspose + BatchNormalization + Activation block.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict
        Data:
            - input  : tensor with the input of the layer
            - nfil   : number of filters
            - stride : stride
            - activ  : activation function
            - kernel : kernel size (scalar, applied as (kernel, kernel))

    Returns
    -------
    dict
        Data:
            - output : tensor with the output of the layer
    """
    from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation

    xx     = data_in["input"]
    nfil   = int(data_in["nfil"])
    stride = int(data_in["stride"])
    activ  = str(data_in["activ"])
    kern   = int(data_in["kernel"])
    kernel = (kern, kern)

    xx = Conv2DTranspose(nfil, kernel_size=kernel, strides=(stride, stride), padding="valid")(xx)
    xx = BatchNormalization()(xx)
    xx = Activation(activ)(xx)

    data_out = {"output":xx}
    return data_out