"""
Phase-averaged vorticity  ω_z = ∂v/∂x - ∂u/∂y.

Reuses the `phase_velocity_cache.npz` produced by phase_velocity and
computes the out-of-plane vorticity from the phase-averaged (ũ, ṽ) fields.

The goal is to verify the starting-vortex-pair interpretation of the
"parentheses" seen in the SHAP magnitude at the burst phase: a vortex ring
in 2D projection should give two compact lobes of OPPOSITE sign, one on
each side of the jet axis.

Layout: 1 row x N cols, shared symmetric RdBu_r colour scale so opposite
signs are colour-coded directly.

Run from analysis/:

    python -m figures.phase_vorticity
    python -m figures.phase_vorticity --phases 2,7,17,22
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
    PERIOD_SNAP,
    SHPX,
    SHPY,
    apply_style,
    imshow_extent_mm,
    save_figure,
)
from figures._phase import build_phase_velocity_cache


DEFAULT_PHASES: tuple[int, ...] = (2, 7, 17, 22)


def _vorticity_phase(u_phase: np.ndarray, v_phase: np.ndarray) -> np.ndarray:
    """
    Return ω_z(x, y, φ) in physical units [1/s] from (ũ, ṽ)(x, y, φ) in [m/s].

    Grid spacing between vectors is MM_PER_PX millimetres (≈ 0.6315 mm
    with the 16×16 / 50%-overlap final pass), i.e. MM_PER_PX * 1e-3 metres.
    """
    dx = MM_PER_PX * 1e-3
    dy = MM_PER_PX * 1e-3
    dvdx = np.gradient(v_phase, dx, axis=2)   # (φ, y, x) -> ∂v/∂x
    dudy = np.gradient(u_phase, dy, axis=1)   # ∂u/∂y
    return dvdx - dudy


def make_figure(phases: tuple[int, ...] = DEFAULT_PHASES,
                save: bool = True) -> plt.Figure:
    apply_style()

    cache = build_phase_velocity_cache()
    u_phase = cache["u_phase"]
    v_phase = cache["v_phase"]
    counts  = cache["counts"]

    for p in phases:
        if not 0 <= p < PERIOD_SNAP:
            raise ValueError(f"phase {p} out of [0, {PERIOD_SNAP})")

    omega = _vorticity_phase(u_phase, v_phase)   # (60, SHPY, SHPX)

    stack = np.stack([omega[p] for p in phases], axis=0)
    vmax = float(np.percentile(np.abs(stack), 99.0))
    extent = imshow_extent_mm(SHPY, SHPX)

    ncols = len(phases)
    fig, axes = plt.subplots(
        nrows=1, ncols=ncols,
        figsize=(2.7 * ncols + 1.6, 3.1),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    im = None
    for col, phase in enumerate(phases):
        ax = axes[0, col]
        im = ax.imshow(omega[phase], extent=extent,
                       origin="upper", aspect="equal",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(rf"$\varphi = {phase}$  ($n = {counts[phase]}$)", pad=4)
        ax.set_xlabel(r"$x$  [mm]")
        if col == 0:
            ax.set_ylabel(r"$y$  [mm]")

    cbar_ax = fig.add_axes([0.915, 0.17, 0.013, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r"$\tilde{\omega}_z$  [1/s]", rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    n_total = int(cache["n_total"])
    fig.suptitle(rf"Phase-averaged out-of-plane vorticity  "
                 rf"($n_\mathrm{{tot}} = {n_total}$ snapshots, "
                 rf"period $= {PERIOD_SNAP}$)",
                 y=0.98, fontsize=10.5)
    fig.subplots_adjust(left=0.065, right=0.895,
                        bottom=0.17, top=0.85,
                        wspace=0.08)

    if save:
        tag = "_".join(str(p) for p in phases)
        save_figure(fig, f"phase_vorticity_{tag}")
    return fig


# ---------------------------------------------------------------------------
def _parse_phases(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-averaged vorticity figure.")
    p.add_argument("--phases", type=_parse_phases, default=DEFAULT_PHASES,
                   help="Comma-separated list of phases in [0, 30). "
                        "Default: 2,7,17,22")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fig = make_figure(phases=tuple(args.phases))
    if not args.no_show:
        plt.show()
