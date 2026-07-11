"""
Phase-averaged velocity field (Level 0 of the phase analysis).

Displays the magnitude of the phase-averaged velocity
    `|\\tilde{\\mathbf{u}}|(x, y, φ) = sqrt(ũ² + ṽ²)`
at several evenly-spaced phases of the forcing cycle (period = 30 snaps).
A single row of N panels with a shared colour scale so the burst propagating
through the jet interaction region can be read off directly.

Axes in millimetres (see PX_PER_MM in _common.py for calibration details).

Run from analysis/:

    python -m figures.phase_velocity
    python -m figures.phase_velocity --phases 0,15,30,45
    python -m figures.phase_velocity --force-recompute
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
    PERIOD_SNAP,
    SHPX,
    SHPY,
    apply_style,
    imshow_extent_mm,
    save_figure,
)
from figures._phase import build_phase_velocity_cache


DEFAULT_PHASES: tuple[int, ...] = (2, 7, 17, 22)


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

    mag_phase = np.sqrt(u_phase.astype(np.float64) ** 2 +
                        v_phase.astype(np.float64) ** 2)

    stack = np.stack([mag_phase[p] for p in phases], axis=0)
    vmin = 0.0
    vmax = float(np.percentile(stack, 99.0))
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
        im = ax.imshow(mag_phase[phase], extent=extent,
                       origin="upper", aspect="equal",
                       cmap="inferno", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(rf"$\varphi = {phase}$  ($n = {counts[phase]}$)", pad=4)
        ax.set_xlabel(r"$x$  [mm]")
        if col == 0:
            ax.set_ylabel(r"$y$  [mm]")

    cbar_ax = fig.add_axes([0.915, 0.17, 0.013, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r"$|\tilde{\mathbf{u}}|$  [m/s]", rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    n_total = int(cache["n_total"])
    fig.suptitle(rf"Phase-averaged velocity magnitude  "
                 rf"($n_\mathrm{{tot}} = {n_total}$ snapshots, "
                 rf"period $= {PERIOD_SNAP}$)",
                 y=0.98, fontsize=10.5)
    fig.subplots_adjust(left=0.065, right=0.895,
                        bottom=0.17, top=0.85,
                        wspace=0.08)

    if save:
        tag = "_".join(str(p) for p in phases)
        save_figure(fig, f"phase_velocity_{tag}")
    return fig


# ---------------------------------------------------------------------------
def _parse_phases(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-averaged velocity field figure.")
    p.add_argument("--phases", type=_parse_phases, default=DEFAULT_PHASES,
                   help="Comma-separated list of phases in [0, 30). "
                        "Default: 2,7,17,22")
    p.add_argument("--force-recompute", action="store_true",
                   help="Rebuild the phase velocity cache.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.force_recompute:
        build_phase_velocity_cache(force=True)
    fig = make_figure(phases=tuple(args.phases))
    if not args.no_show:
        plt.show()
