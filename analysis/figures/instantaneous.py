"""
Figure A — Cremades-style instantaneous coincidence between three coherent
structures (Shear / Vorticity / Entrainment), overlaid with SHAP as a black
contour and paired with the velocity-fluctuation magnitude and the SHAP
magnitude from the same snapshot.

Layout: 3 rows x (1 or 2) columns.
    row 1 : 7-colour set-theoretic coincidence map of the three structures,
            SHAP regions drawn on top as black contour lines.
    row 2 : |u'|   = sqrt(u'^2 + v'^2)    (inferno colormap)
    row 3 : ||SHAP|| = sqrt(SHAP_u^2 + SHAP_v^2)   (hot colormap)

The y-axis is inverted (image / PIV convention: row 0 on top).

Run from analysis/:

    python -m figures.instantaneous --index 55
    python -m figures.instantaneous --index 50 --index2 60
    python -m figures.instantaneous --auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Allow running as a plain script (python instantaneous.py) by adding
# the parent "code/" directory to sys.path.
_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import (
    COINCIDENCE_REGION_COLORS,
    MM_PER_PX,
    OUT_DIR,
    SHPX,
    SHPY,
    STRUCTURE_COLORS,
    STRUCTURE_LABELS,
    apply_style,
    imshow_extent_mm,
    save_figure,
)
from figures._structures import (
    all_masks_for_snapshot,
    compute_coincidence_regions,
    compute_rms_cache,
    find_active_index,
)


# Three structures that build the 7-region coincidence display. Order matters:
# it defines which colour corresponds to which "set only / intersection" label.
CLASSIFICATION_TRIO: tuple[str, str, str] = ("shear", "vorticity", "entrainment")

# Short labels used in the legend (matching CLASSIFICATION_TRIO order).
# 7th entry ("All three") is kept for cmap indexing but excluded from the legend.
_REGION_LABELS = [
    r"Shear",
    r"Vort.",
    r"Entr.",
    r"Shear $\cap$ Vort.",
    r"Shear $\cap$ Entr.",
    r"Vort. $\cap$ Entr.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _single_color_cmap(hex_color: str, alpha: float = 0.75) -> ListedColormap:
    """Two-stop colormap: fully transparent -> opaque single colour."""
    from matplotlib.colors import to_rgba
    rgba_on = to_rgba(hex_color, alpha=alpha)
    return ListedColormap([(0, 0, 0, 0), rgba_on])


def _overlay_mask(ax, mask: np.ndarray, color: str, alpha: float = 0.75) -> None:
    cmap = _single_color_cmap(color, alpha=alpha)
    shpy, shpx = mask.shape
    ax.imshow(mask.astype(np.uint8),
              origin="upper",
              aspect="equal",
              cmap=cmap,
              vmin=0, vmax=1,
              interpolation="nearest",
              extent=imshow_extent_mm(shpy, shpx),
              zorder=2)


def _area_fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    denom = valid.sum()
    return float(mask.sum() / denom) if denom else 0.0


def _region_cmap_norm() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(COINCIDENCE_REGION_COLORS)
    norm = BoundaryNorm(boundaries=[0.5 + i for i in range(8)], ncolors=7)
    return cmap, norm


def _draw_structure_panel(ax, data: dict, title: str) -> None:
    ax.set_facecolor("white")
    masks = data["masks"]
    valid_y = data["valid_y"]
    shpy, shpx = data["u"].shape

    a = masks[CLASSIFICATION_TRIO[0]]
    b = masks[CLASSIFICATION_TRIO[1]]
    c = masks[CLASSIFICATION_TRIO[2]]
    region_map = compute_coincidence_regions(a, b, c)

    cmap, norm = _region_cmap_norm()
    extent = imshow_extent_mm(shpy, shpx)
    # Mask both empty (0) and "all three" (7) regions — the latter is typically
    # sparse noise or indicates a corrupted snapshot and clutters the display.
    ax.imshow(
        np.ma.masked_where((region_map == 0) | (region_map == 7), region_map),
        origin="upper",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=extent,
        zorder=2,
    )

    invalid = np.broadcast_to((~valid_y)[:, None], (shpy, shpx))
    _overlay_mask(ax, invalid, "#ededed", alpha=1.0)

    shap_mask = masks["shap"].astype(float)
    ax.contour(
        shap_mask, levels=[0.5],
        colors="black", linewidths=0.7,
        antialiased=True,
        extent=extent,
    )

    ax.set_xlim(0, shpx * MM_PER_PX)
    ax.set_ylim(shpy * MM_PER_PX, 0)
    ax.set_title(title, pad=5)


def _draw_scalar_panel(ax, field: np.ndarray, valid_y: np.ndarray,
                       vmin: float, vmax: float, cmap: str):
    data = field.astype(np.float32).copy()
    data[~valid_y, :] = np.nan
    shpy, shpx = field.shape
    im = ax.imshow(data,
                   origin="upper",
                   aspect="equal",
                   cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest",
                   extent=imshow_extent_mm(shpy, shpx))
    ax.set_xlim(0, shpx * MM_PER_PX)
    ax.set_ylim(shpy * MM_PER_PX, 0)
    return im


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------
def make_figure(index: int | None = None,
                index2: int | None = None,
                indices: list[int] | None = None,
                save: bool = True) -> plt.Figure:
    apply_style()

    rms = compute_rms_cache()
    if indices is None:
        indices = [index] if index2 is None else [index, index2]
    snaps = [all_masks_for_snapshot(i, rms=rms) for i in indices]
    ncols = len(snaps)

    u_fluc_mag = [np.sqrt(s["u_fluc"] ** 2 + s["v_fluc"] ** 2) for s in snaps]
    vel_mag    = [np.sqrt(s["u"] ** 2 + s["v"] ** 2) for s in snaps]
    shap_mag   = [np.sqrt(s["shap_u"] ** 2 + s["shap_v"] ** 2) for s in snaps]

    # Filter: zero out regions where velocity is below noise floor
    VEL_NOISE_FLOOR = 0.05
    for i in range(len(shap_mag)):
        noise_mask = vel_mag[i] < VEL_NOISE_FLOOR
        u_fluc_mag[i] = u_fluc_mag[i].copy()
        u_fluc_mag[i][noise_mask] = 0.0
        shap_mag[i] = shap_mag[i].copy()
        shap_mag[i][noise_mask] = 0.0

    # Fixed colour-scale upper bounds (common across snapshots so columns
    # can be compared directly). These match the |u'| / ||SHAP|| ranges
    # reported in the shap_fields reference plots.
    uvel_vmin, uvel_vmax = 0.0, 0.5
    shap_vmin, shap_vmax = 0.0, 0.013

    # Panel size in inches, data aspect preserved. Absolute margins so the
    # layout stays tight for any ncols.
    PANEL_W = 3.2
    PANEL_H = PANEL_W * (SHPY / SHPX)        # ≈ 2.0" for 199x319 grid
    LEFT_M, RIGHT_M = 0.95, 1.15              # ylabel+row text / colorbar+legend
    TOP_M, BOTTOM_M = 0.30, 0.50
    COL_GAP, ROW_GAP = 0.25, 0.35

    fig_width  = LEFT_M + RIGHT_M + PANEL_W * ncols + COL_GAP * (ncols - 1)
    fig_height = TOP_M + BOTTOM_M + 3 * PANEL_H + 2 * ROW_GAP

    fig, axes = plt.subplots(
        nrows=3, ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex="col", sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    _LEFT_F   = LEFT_M / fig_width
    _RIGHT_F  = 1.0 - RIGHT_M / fig_width
    _BOTTOM_F = BOTTOM_M / fig_height
    _TOP_F    = 1.0 - TOP_M / fig_height

    im_uvel = None
    im_shap = None
    for col, (idx, data, umag, smag) in enumerate(
            zip(indices, snaps, u_fluc_mag, shap_mag)):
        ax_top = axes[0, col]
        ax_mid = axes[1, col]
        ax_bot = axes[2, col]

        _draw_structure_panel(ax_top, data, title=rf"$t = {idx}$")
        im_uvel = _draw_scalar_panel(ax_mid, umag, data["valid_y"],
                                     vmin=uvel_vmin, vmax=uvel_vmax,
                                     cmap="inferno")
        im_shap = _draw_scalar_panel(ax_bot, smag, data["valid_y"],
                                     vmin=shap_vmin, vmax=shap_vmax,
                                     cmap="hot")

        ax_bot.set_xlabel(r"$x$  [mm]")
        if col == 0:
            ax_top.set_ylabel(r"$y$  [mm]")
            ax_mid.set_ylabel(r"$y$  [mm]")
            ax_bot.set_ylabel(r"$y$  [mm]")

    fig.subplots_adjust(left=_LEFT_F, right=_RIGHT_F,
                        bottom=_BOTTOM_F, top=_TOP_F,
                        wspace=COL_GAP / PANEL_W,
                        hspace=ROW_GAP / PANEL_H)

    # Derive the vertical centre / y0 of each row from the layout parameters
    # (in figure-fraction coordinates).
    _panel_h_f = PANEL_H / fig_height
    _row_gap_f = ROW_GAP / fig_height
    _row_y0 = [_TOP_F - (r + 1) * _panel_h_f - r * _row_gap_f for r in range(3)]
    _row_yc = [y0 + _panel_h_f / 2 for y0 in _row_y0]

    # Row labels tucked just inside the left margin.
    _label_x = 0.18 * _LEFT_F
    fig.text(_label_x, _row_yc[0], "Coincident\nstructures",
             ha="left", va="center", rotation=90, fontsize=9.5)
    fig.text(_label_x, _row_yc[1], r"$|u^\prime|$",
             ha="left", va="center", rotation=90, fontsize=10.5)
    fig.text(_label_x, _row_yc[2], r"$\|\mathrm{SHAP}\|$",
             ha="left", va="center", rotation=90, fontsize=10)

    # Legend: 7 colour patches + SHAP (black contour line).
    legend_handles = [
        Patch(facecolor=COINCIDENCE_REGION_COLORS[i], edgecolor="black",
              linewidth=0.4, label=lbl)
        for i, lbl in enumerate(_REGION_LABELS)
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", linewidth=1.0, label="SHAP")
    )
    fig.legend(
        handles=legend_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(_RIGHT_F + 0.01, _TOP_F),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.55,
        fontsize=8,
    )

    # Two stacked colorbars, hugging the right edge of the plot area.
    _cbar_x = _RIGHT_F + 0.012
    _cbar_w = 0.013
    cbar_ax_u = fig.add_axes([_cbar_x, _row_y0[1], _cbar_w, _panel_h_f])
    cb_u = fig.colorbar(im_uvel, cax=cbar_ax_u)
    cb_u.set_label(r"$|u^\prime|$  [m/s]", rotation=90, labelpad=6)
    cb_u.ax.tick_params(labelsize=8)
    cb_u.outline.set_linewidth(0.6)

    cbar_ax_s = fig.add_axes([_cbar_x, _row_y0[2], _cbar_w, _panel_h_f])
    cb_s = fig.colorbar(im_shap, cax=cbar_ax_s)
    cb_s.set_label(r"$\|\mathrm{SHAP}\|$  [m/s]", rotation=90, labelpad=6)
    cb_s.ax.tick_params(labelsize=8)
    cb_s.outline.set_linewidth(0.6)

    suffix = "_".join(f"t{i:05d}" for i in indices)
    if save:
        save_figure(fig, f"instantaneous_{suffix}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_indices(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Figure A (instantaneous structures + PIV).")
    p.add_argument("--index", type=int, default=55,
                   help="First snapshot index (ignored if --auto or --indices is set).")
    p.add_argument("--index2", type=int, default=None,
                   help="Optional second snapshot index for side-by-side comparison.")
    p.add_argument("--indices", type=_parse_indices, default=None,
                   help="Comma-separated list of snapshot indices (e.g. 1000,3000,5000,7000).")
    p.add_argument("--auto", action="store_true",
                   help="Automatically pick the index with the strongest "
                        "activity in the interaction hot spot.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open a matplotlib window.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.indices is not None:
        fig = make_figure(indices=args.indices, save=True)
    else:
        idx = find_active_index() if args.auto else args.index
        fig = make_figure(index=idx, index2=args.index2, save=True)
    if not args.no_show:
        plt.show()
