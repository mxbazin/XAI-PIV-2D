"""
Figure C — Coincidence between SHAP and the jets-specific structure types
as a function of the wall-normal pixel coordinate y (equivalent of Cremades
et al. 2025, Fig. 4).

For each pair (SHAP, X) with X in {Shear, Vorticity, Entrainment, Q_2D}
we plot two curves versus y:
    solid  :  (SHAP ∩ X) / SHAP      -- fraction of the SHAP volume that
                                         also belongs to X.
    dashed :  (SHAP ∩ X) / X         -- fraction of X that also belongs
                                         to SHAP.

A single panel with 8 curves follows Cremades' layout directly; pass
``--split`` to produce a 2x2 panel version instead (one pair per panel).

Run from analysis/:

    python -m figures.coincidence
    python -m figures.coincidence --split
    python -m figures.coincidence --force-recompute
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
    MM_PER_PX,
    STRUCTURE_COLORS,
    apply_style,
    save_figure,
)
from figures._structures import compute_coincidence_cache


# Pair key  -> (short legend label, full math label for captions)
PAIRS: list[tuple[str, str]] = [
    ("shear",       "Shear"),
    ("vorticity",   "Vort."),
    ("entrainment", "Entr."),
    ("q2d",         r"$Q_{2\mathrm{D}}$"),
]

# Hot spot band identified in rapport_complet.txt §5.4 (grid indices → mm)
HOTSPOT_Y = (95 * MM_PER_PX, 150 * MM_PER_PX)


# ---------------------------------------------------------------------------
def _extract_pair(cache: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, p_X_in_SHAP, p_SHAP_in_X) for a given structure key."""
    count_shap = cache["count_shap"].astype(np.float64)
    count_x    = cache[f"count_{key}"].astype(np.float64)
    inter      = cache[f"inter_shap_{key}"].astype(np.float64)
    valid      = cache["valid_y"].astype(bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_x_in_shap = np.where(count_shap > 0, inter / count_shap, np.nan)
        p_shap_in_x = np.where(count_x    > 0, inter / count_x,    np.nan)

    p_x_in_shap[~valid] = np.nan
    p_shap_in_x[~valid] = np.nan

    y = np.arange(len(count_shap), dtype=np.float32) * MM_PER_PX
    return y, p_x_in_shap, p_shap_in_x


def _shade_hotspot(ax) -> None:
    ax.axvspan(HOTSPOT_Y[0], HOTSPOT_Y[1], color="0.90",
               alpha=0.55, zorder=0, linewidth=0)


# ---------------------------------------------------------------------------
def _make_single_panel(cache: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white")
    _shade_hotspot(ax)

    for key, label in PAIRS:
        y, p_x_in_shap, p_shap_in_x = _extract_pair(cache, key)
        color = STRUCTURE_COLORS[key]
        ax.plot(y, p_x_in_shap, "-",  color=color, linewidth=1.5,
                label=rf"$(\mathrm{{SHAP}}\cap$ {label}$)\,/\,\mathrm{{SHAP}}$")
        ax.plot(y, p_shap_in_x, "--", color=color, linewidth=1.5,
                label=rf"$(\mathrm{{SHAP}}\cap$ {label}$)\,/\,${label}")

    ax.set_xlabel(r"$y$  [mm]")
    ax.set_ylabel(r"Coincidence fraction")
    ax.set_xlim(0, (len(cache["count_shap"]) - 1) * MM_PER_PX)
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=2, loc="upper right",
              frameon=True, framealpha=0.96,
              edgecolor="black", fancybox=False,
              handlelength=2.2, fontsize=8.5,
              borderpad=0.55, labelspacing=0.45, columnspacing=1.2)

    n = int(cache["n_snapshots"])
    fig.suptitle(rf"Coincidence of SHAP with coherent structures  ($n = {n}$ snapshots)",
                 y=0.98, fontsize=10.5)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.90)
    return fig


def _make_split_panel(cache: dict) -> plt.Figure:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9.5, 6.3),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    y_max = (len(cache["count_shap"]) - 1) * MM_PER_PX
    for (key, label), ax in zip(PAIRS, axes.flat):
        _shade_hotspot(ax)
        y, p_x_in_shap, p_shap_in_x = _extract_pair(cache, key)
        color = STRUCTURE_COLORS[key]
        ax.plot(y, p_x_in_shap, "-",  color=color, linewidth=1.6,
                label=rf"$(\mathrm{{SHAP}}\cap X)\,/\,\mathrm{{SHAP}}$")
        ax.plot(y, p_shap_in_x, "--", color=color, linewidth=1.6,
                label=rf"$(\mathrm{{SHAP}}\cap X)\,/\,X$")

        ax.set_title(rf"$X = $ {label}", pad=4)
        ax.set_xlim(0, y_max)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                  edgecolor="black", fancybox=False, fontsize=8.5,
                  borderpad=0.5, labelspacing=0.4, handlelength=2.2)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$y$  [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Coincidence fraction")

    n = int(cache["n_snapshots"])
    fig.suptitle(rf"Coincidence of SHAP with coherent structures  ($n = {n}$ snapshots)",
                 y=0.985, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.915,
                        wspace=0.10, hspace=0.22)
    return fig


# ---------------------------------------------------------------------------
def make_figure(split: bool = False, save: bool = True) -> plt.Figure:
    apply_style()
    cache = compute_coincidence_cache()
    fig = _make_split_panel(cache) if split else _make_single_panel(cache)
    if save:
        save_figure(fig, "coincidence_split" if split
                         else "coincidence")
    return fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Figure C (coincidence y profiles).")
    p.add_argument("--split", action="store_true",
                   help="Use a 2x2 panel layout (one pair per panel).")
    p.add_argument("--force-recompute", action="store_true",
                   help="Rebuild the coincidence cache from scratch.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.force_recompute:
        compute_coincidence_cache(force=True)
    fig = make_figure(split=args.split)
    if not args.no_show:
        plt.show()
