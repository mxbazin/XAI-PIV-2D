"""
Phase-modulation amplitude of the velocity magnitude.

Reuses the `phase_velocity_cache.npz` produced by phase_velocity and
collapses the phase dimension to two 2D maps that together answer
"where does the forcing matter?":

    left  : absolute phase std       σ_φ( |ũ|(x, y, φ) )
    right : relative phase modulation
            σ_φ( |ũ| ) / <|ũ|>_φ
            (dimensionless; masked where <|ũ|> is below 5 % of its peak
             to avoid division-by-zero inflation in the quiet regions).

Run from analysis/:

    python -m figures.phase_modulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import (
    SHPX,
    SHPY,
    apply_style,
    imshow_extent_mm,
    save_figure,
)
from figures._phase import build_phase_velocity_cache


MEAN_FRACTION_THRESHOLD: float = 0.05   # mask < 5 % of peak mean magnitude


def make_figure(save: bool = True) -> plt.Figure:
    apply_style()

    cache = build_phase_velocity_cache()
    u_phase = cache["u_phase"].astype(np.float64)     # (60, SHPY, SHPX)
    v_phase = cache["v_phase"].astype(np.float64)

    mag_phase = np.sqrt(u_phase ** 2 + v_phase ** 2)  # (60, SHPY, SHPX)
    mag_mean = mag_phase.mean(axis=0)                 # (SHPY, SHPX)
    mag_std  = mag_phase.std(axis=0)                  # (SHPY, SHPX)

    mask_quiet = mag_mean < (MEAN_FRACTION_THRESHOLD * mag_mean.max())
    rel_mod = np.where(mask_quiet, np.nan, mag_std / np.maximum(mag_mean, 1e-12))

    extent = imshow_extent_mm(SHPY, SHPX)

    std_vmax = float(np.percentile(mag_std, 99.0))
    rel_vmax = float(np.nanpercentile(rel_mod, 99.0))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.5, 3.6),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    ax_std, ax_rel = axes

    im_std = ax_std.imshow(mag_std, extent=extent,
                           origin="upper", aspect="equal",
                           cmap="inferno", vmin=0.0, vmax=std_vmax,
                           interpolation="nearest")
    ax_std.set_title(r"Absolute phase std  $\sigma_\varphi(|\tilde{\mathbf{u}}|)$",
                     pad=5)
    ax_std.set_xlabel(r"$x$  [mm]")
    ax_std.set_ylabel(r"$y$  [mm]")

    im_rel = ax_rel.imshow(rel_mod, extent=extent,
                           origin="upper", aspect="equal",
                           cmap="inferno", vmin=0.0, vmax=rel_vmax,
                           interpolation="nearest")
    ax_rel.set_title(r"Relative phase modulation  "
                     r"$\sigma_\varphi(|\tilde{\mathbf{u}}|) \,/\, "
                     r"\langle|\tilde{\mathbf{u}}|\rangle_\varphi$",
                     pad=5)
    ax_rel.set_xlabel(r"$x$  [mm]")

    # Two separate colorbars (one per panel, different units)
    fig.subplots_adjust(left=0.075, right=0.90,
                        bottom=0.17, top=0.86,
                        wspace=0.18)

    bbox_std = ax_std.get_position()
    cax_std = fig.add_axes([bbox_std.x1 + 0.005, bbox_std.y0,
                            0.010, bbox_std.height])
    cb_std = fig.colorbar(im_std, cax=cax_std)
    cb_std.set_label(r"$\sigma_\varphi$  [m/s]", rotation=90, labelpad=6)
    cb_std.ax.tick_params(labelsize=8)
    cb_std.outline.set_linewidth(0.6)

    bbox_rel = ax_rel.get_position()
    cax_rel = fig.add_axes([bbox_rel.x1 + 0.005, bbox_rel.y0,
                            0.010, bbox_rel.height])
    cb_rel = fig.colorbar(im_rel, cax=cax_rel)
    cb_rel.set_label(r"$\sigma_\varphi / \langle\cdot\rangle$",
                     rotation=90, labelpad=6)
    cb_rel.ax.tick_params(labelsize=8)
    cb_rel.outline.set_linewidth(0.6)

    fig.suptitle("Phase modulation of the velocity magnitude  "
                 rf"($n_\mathrm{{tot}} = {int(cache['n_total'])}$ snapshots, "
                 rf"period $= {int(cache['period'])}$)",
                 y=0.985, fontsize=10.5)

    if save:
        save_figure(fig, "phase_modulation")
    return fig


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase modulation of the velocity magnitude.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fig = make_figure()
    if not args.no_show:
        plt.show()
