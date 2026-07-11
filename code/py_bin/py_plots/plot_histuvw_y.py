# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_histuvw_y.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no w component)

Plot PDFs of (u, v) inside a given structure type, as a function of the
wall-normal distance y+. In 2D PIV the spanwise velocity component w does not
exist, so only u and v histograms are produced. The function still accepts
w-related keys (``ww_struc``, ``plot_filew``, ``xlabelw``, ``wmin``, ``wmax``)
for signature compatibility with the old 3D callers, but they are ignored.
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_histuvw_y(data_in=None):
    """
    Parameters
    ----------
    data_in : dict
        Required keys:
          - plot_folder : output folder
          - plot_fileu  : file basename for u histogram
          - plot_filev  : file basename for v histogram
          - ylabel      : y-axis label (wall-normal distance label)
          - xlabelu     : x-axis label for u
          - xlabelv     : x-axis label for v
          - fontsize, figsize_x, figsize_y, colormap, dpi : matplotlib params
          - uu_struc    : 1D array of u samples inside the structure
          - vv_struc    : 1D array of v samples inside the structure
          - yplus_struc : wall-normal distance for each sample (same length as uu_struc)
          - yplusmesh   : 1D array of wall-normal mesh positions
          - bins        : number of bins (int) — used for both x and y in the 2D histogram
          - umin, umax, vmin, vmax : histogram ranges (use None to auto-derive)

        Optional / ignored (for backward compat):
          - plot_filew, xlabelw, ww_struc, wmin, wmax, colornum, lev_min, lev_delta, linewidth
    """
    if data_in is None:
        data_in = {}

    plot_folder = str(data_in["plot_folder"])
    plot_fileu  = str(data_in["plot_fileu"])
    plot_filev  = str(data_in["plot_filev"])
    ylabel      = str(data_in["ylabel"])
    xlabelu     = str(data_in["xlabelu"])
    xlabelv     = str(data_in["xlabelv"])
    fontsize    = int(data_in.get("fontsize", 18))
    figsize_x   = int(data_in.get("figsize_x", 10))
    figsize_y   = int(data_in.get("figsize_y", 8))
    colormap    = str(data_in.get("colormap", "viridis"))
    dpi         = float(data_in.get("dpi", 200))

    uu_struc    = np.asarray(data_in["uu_struc"]).ravel()
    vv_struc    = np.asarray(data_in["vv_struc"]).ravel()
    yplus_struc = np.asarray(data_in["yplus_struc"]).ravel()
    yplusmesh   = np.asarray(data_in.get("yplusmesh", []))

    bins        = int(data_in.get("bins", 50))

    def _range(val_min, val_max, data):
        lo = float(val_min) if val_min is not None else float(np.min(data))
        hi = float(val_max) if val_max is not None else float(np.max(data))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    umin, umax = _range(data_in.get("umin"), data_in.get("umax"), uu_struc)
    vmin, vmax = _range(data_in.get("vmin"), data_in.get("vmax"), vv_struc)

    # y range from the mesh if available, otherwise from the samples
    if yplusmesh.size > 0:
        ylo, yhi = float(np.min(yplusmesh)), float(np.max(yplusmesh))
    else:
        ylo, yhi = float(np.min(yplus_struc)), float(np.max(yplus_struc))
    if yhi <= ylo:
        yhi = ylo + 1e-6

    os.makedirs(plot_folder, exist_ok=True)

    def _save_hist(x_samples, xlabel, xmin, xmax, basename):
        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        H, x_edges, y_edges = np.histogram2d(
            x_samples, yplus_struc,
            bins=bins,
            range=[[xmin, xmax], [ylo, yhi]],
            density=True,
        )
        X, Y = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]),
                           0.5 * (y_edges[:-1] + y_edges[1:]), indexing="xy")
        pcm = ax.pcolormesh(X, Y, H.T, cmap=colormap, shading="auto")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_folder, basename + ".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.join(plot_folder, basename + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    _save_hist(uu_struc, xlabelu, umin, umax, plot_fileu)
    _save_hist(vv_struc, xlabelv, vmin, vmax, plot_filev)
