# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotprediction.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no z dimension, u and v only)

Plot the CNN prediction vs simulation for a 2D PIV snapshot. The figure has a
1x3 layout: Simulation | Prediction | Error, all shown as pcolor(shpy, shpx).
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def plotprediction(data_in={"plot_folder":"plots","xlabel":"$x$","ylabel":"$y$",
                            "fontsize":18,"figsize_x":15,"figsize_y":5,"colormap":"viridis","colornum":2,
                            "fig_name":"predictionfield","dpi":200,"index_ii":0,"Unet":None,"flowfield":None,
                            "index_y":0,"xmin":0,"xmax":319,"ymin":0,"ymax":199,"errmax":0.2,"errmin":0,
                            "b_velo_sim":"$u_s$","b_velo_pred":"$u_p$","b_velo_err":"$\\epsilon_u$"}):
    """
    Parameters
    ----------
    data_in : dict
        - plot_folder : folder to store the figure
        - xlabel, ylabel : axis labels
        - fontsize, figsize_x, figsize_y, colormap, colornum, dpi : matplotlib params
        - fig_name : base name of the saved figure
        - index_ii : snapshot index
        - Unet     : trained ann.deep_model instance (provides field_error)
        - flowfield: flow_field instance (provides x_h, y_h, vol_h, shpx, shpy)
        - index_y  : ignored in 2D (kept for signature compatibility)
        - xmin, xmax, ymin, ymax : axis limits
        - errmax, errmin : error colorbar range
        - b_velo_sim / b_velo_pred / b_velo_err : colorbar titles
    """
    plot_folder = str(data_in["plot_folder"])
    xlabel      = str(data_in["xlabel"])
    ylabel      = str(data_in["ylabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap    = str(data_in["colormap"])
    fig_name    = str(data_in["fig_name"])
    dpi         = float(data_in["dpi"])
    index_ii    = int(data_in["index_ii"])
    Unet        = data_in["Unet"]
    flowfield   = data_in["flowfield"]
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    errmax      = float(data_in["errmax"])
    errmin      = float(data_in["errmin"])
    b_velo_sim  = str(data_in["b_velo_sim"])
    b_velo_pred = str(data_in["b_velo_pred"])
    b_velo_err  = str(data_in["b_velo_err"])

    from py_bin.py_functions.normalization import read_norm

    # -------------------------------------------------------------------------
    # Compute prediction / error for this snapshot
    # -------------------------------------------------------------------------
    data_error = Unet.field_error(data_in={"index_ii":index_ii})
    data_norm  = read_norm(data_in={"folder":Unet.data_folder,"file":Unet.umax_file})
    umax       = data_norm["uumax"]
    umin       = data_norm["uumin"]

    # Area-weighted mean errors for the title (2D: vol_h has shape (1, shpx),
    # broadcast over y). err_u/err_v are (shpy, shpx).
    area_map = np.broadcast_to(flowfield.vol_h, (flowfield.shpy, flowfield.shpx))
    erru = np.sum(data_error["err_u"] * area_map) / np.sum(area_map)
    errv = np.sum(data_error["err_v"] * area_map) / np.sum(area_map)

    title_str = ("$\\epsilon_u$: " + "{0:.2f}".format(erru * 100) + "%" +
                 ", $\\epsilon_v$: " + "{0:.2f}".format(errv * 100) + "%")

    # 2D grid for pcolor: x_h (shpx,) and y_h (shpy,)
    xx, yy = np.meshgrid(flowfield.x_h, flowfield.y_h)

    fig_name_full = fig_name + "_field" + str(index_ii)

    # -------------------------------------------------------------------------
    # 1x3 layout: Simulation | Prediction | Error
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(figsize_x, figsize_y))
    fig.suptitle(title_str, fontsize=fontsize)

    sim_u = data_error["sim_u"]
    pre_u = data_error["pre_u"]
    err_u = data_error["err_u"]

    im0 = axs[0].pcolormesh(xx, yy, sim_u, cmap=colormap, vmin=umin, vmax=umax, shading="auto")
    axs[0].set_title("Simulation", fontsize=fontsize)
    axs[0].set_xlabel(xlabel, fontsize=fontsize)
    axs[0].set_ylabel(ylabel, fontsize=fontsize)
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    axs[0].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0], label=b_velo_sim)

    im1 = axs[1].pcolormesh(xx, yy, pre_u, cmap=colormap, vmin=umin, vmax=umax, shading="auto")
    axs[1].set_title("Prediction", fontsize=fontsize)
    axs[1].set_xlabel(xlabel, fontsize=fontsize)
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)
    axs[1].set_aspect("equal")
    fig.colorbar(im1, ax=axs[1], label=b_velo_pred)

    im2 = axs[2].pcolormesh(xx, yy, err_u, cmap=colormap, vmin=errmin, vmax=errmax, shading="auto")
    axs[2].set_title("Error", fontsize=fontsize)
    axs[2].set_xlabel(xlabel, fontsize=fontsize)
    axs[2].set_xlim(xmin, xmax)
    axs[2].set_ylim(ymin, ymax)
    axs[2].set_aspect("equal")
    fig.colorbar(im2, ax=axs[2], label=b_velo_err)

    plt.tight_layout()

    try:
        os.makedirs(plot_folder, exist_ok=True)
    except Exception:
        pass
    fig.savefig(os.path.join(plot_folder, fig_name_full + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(plot_folder, fig_name_full + ".pdf"), bbox_inches="tight")
    plt.close(fig)
