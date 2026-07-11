"""
Figure B — Joint probability density function of the streamwise velocity
fluctuation u' and the wall-normal pixel coordinate y, conditioned on
each type of coherent structure (equivalent of Cremades et al. 2025, Fig. 3).

Layout: 2 rows x 3 cols, one panel per conditioning structure plus the
unconditional JPDF for reference.
    row 0 : [full (no cond.)]  [SHAP]        [Shear]
    row 1 : [Vorticity      ]  [Entrainment] [Q_2D]

Each panel is normalised to max = 1 on its own support and displayed on
a logarithmic colour scale spanning 3 decades [1e-3, 1], matching
Cremades' Fig. 3 convention.

Run from analysis/:

    python -m figures.jpdf
    python -m figures.jpdf --force-recompute
    python -m figures.jpdf --max-snapshots 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import (
    CACHE_DIR,
    MM_PER_PX,
    SHPX,
    SHPY,
    apply_style,
    save_figure,
)
from figures._structures import (
    _shap_available_indices,
    compute_gradients,
    compute_rms_cache,
    load_mean_profiles,
    load_shap,
    load_velocity,
    segment_entrainment,
    segment_q2d,
    segment_shap,
    segment_shear,
    segment_vorticity,
)


JPDF_CACHE_PATH = CACHE_DIR / "jpdf_cache.npz"


# JPDF bin definition
U_MIN, U_MAX = -0.6, 0.6
NU = 80
NY = 40
U_BINS = np.linspace(U_MIN, U_MAX, NU + 1)
Y_BINS = np.linspace(0, SHPY, NY + 1)
DU = U_BINS[1] - U_BINS[0]
DY = Y_BINS[1] - Y_BINS[0]


PANEL_ORDER: list[tuple[str, str]] = [
    ("full",        "Full domain"),
    ("shap",        "SHAP"),
    ("shear",       "Shear"),
    ("vorticity",   "Vorticity"),
    ("entrainment", "Entrainment"),
    ("q2d",         r"$Q_{2\mathrm{D}}$"),
]


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------
def _digitize_u(u_field: np.ndarray) -> np.ndarray:
    """Return integer u-bin index with -1 for out-of-range values."""
    idx = np.floor((u_field - U_MIN) / DU).astype(np.int32)
    in_range = (idx >= 0) & (idx < NU)
    idx[~in_range] = -1
    return idx


def build_jpdf_cache(indices: list[int] | None = None,
                     force: bool = False,
                     max_snapshots: int | None = None,
                     progress_every: int = 100) -> dict[str, np.ndarray]:
    """Accumulate a 2D (u', y) histogram for each structure type."""
    if JPDF_CACHE_PATH.exists() and not force:
        npz = np.load(JPDF_CACHE_PATH, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    if indices is None:
        indices = _shap_available_indices()
    if max_snapshots is not None:
        indices = indices[:max_snapshots]

    rms = compute_rms_cache()
    u_mean, v_mean, valid_y = load_mean_profiles()

    y_idx_2d = np.clip(
        np.floor(np.broadcast_to(np.arange(SHPY, dtype=np.float32)[:, None],
                                 (SHPY, SHPX)) / DY).astype(np.int32),
        0, NY - 1,
    )

    hist_keys = [k for k, _ in PANEL_ORDER]
    H = {k: np.zeros((NU, NY), dtype=np.int64) for k in hist_keys}
    n_total = NU * NY

    valid_full_mask = np.broadcast_to(valid_y[:, None], (SHPY, SHPX))

    n_ok = 0
    for i, idx in enumerate(indices):
        try:
            u, v = load_velocity(idx)
            su, sv = load_shap(idx)
        except (OSError, KeyError, FileNotFoundError):
            continue

        u_fluc = u - u_mean[:, None]
        v_fluc = v - v_mean[:, None]
        grads = compute_gradients(u, v)

        masks = {
            "shap":        segment_shap(su, sv, rms),
            "shear":       segment_shear(grads, rms),
            "vorticity":   segment_vorticity(grads, rms),
            "entrainment": segment_entrainment(v_fluc, rms),
            "q2d":         segment_q2d(grads, rms),
            "full":        valid_full_mask.copy(),
        }
        for mk in masks.values():
            mk[~valid_y, :] = False

        u_idx = _digitize_u(u_fluc)
        u_in_range = u_idx >= 0

        combined = u_idx * NY + y_idx_2d
        for key, mk in masks.items():
            sel = combined[mk & u_in_range]
            if sel.size == 0:
                continue
            H[key] += np.bincount(sel, minlength=n_total).reshape(NU, NY)

        n_ok += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  jpdf cache: {i + 1}/{len(indices)} snapshots "
                  f"({n_ok} ok)", flush=True)

    if n_ok == 0:
        raise RuntimeError("build_jpdf_cache: no snapshots could be loaded.")

    cache = {
        **{f"H_{k}": v for k, v in H.items()},
        "u_bins":      U_BINS.astype(np.float32),
        "y_bins":      Y_BINS.astype(np.float32),
        "n_snapshots": np.asarray(n_ok, dtype=np.int32),
        "indices_used": np.asarray(indices, dtype=np.int32),
    }
    np.savez(JPDF_CACHE_PATH, **cache)
    print(f"[cache] JPDF written to {JPDF_CACHE_PATH.name} (n={n_ok})")
    return cache


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def _normalized_pdf(H: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Normalise the histogram and optionally apply a Gaussian smoother."""
    H = H.astype(np.float64)
    if isinstance(sigma, (int, float)) and sigma > 0:
        H = gaussian_filter(H, sigma=float(sigma), mode="nearest")
    elif isinstance(sigma, tuple) and any(s > 0 for s in sigma):
        H = gaussian_filter(H, sigma=sigma, mode="nearest")
    total = H.sum()
    if total == 0:
        return H
    pdf = H / (total * DU * DY)
    peak = pdf.max()
    return pdf / peak if peak > 0 else pdf


def make_figure(save: bool = True,
                sigma: float | tuple[float, float] = (1.4, 0.9)) -> plt.Figure:
    apply_style()
    cache = build_jpdf_cache()

    u_bins = cache["u_bins"]
    y_bins = cache["y_bins"]
    u_centers = 0.5 * (u_bins[:-1] + u_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=(10.5, 6.2),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    norm = LogNorm(vmin=1.0e-3, vmax=1.0)
    levels = np.logspace(-3, 0, 7)
    mesh = None

    for (key, label), ax in zip(PANEL_ORDER, axes.flat):
        H = cache[f"H_{key}"]
        pdf = _normalized_pdf(H, sigma=sigma)

        mesh = ax.contourf(u_centers, y_centers * MM_PER_PX, pdf.T,
                           levels=levels, cmap="viridis",
                           norm=norm, extend="min",
                           antialiased=True)

        ax.set_title(label, pad=4)
        ax.set_xlim(U_MIN, U_MAX)
        ax.set_ylim(SHPY * MM_PER_PX, 0)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$u^\prime$  [m/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$  [mm]")

    cbar_ax = fig.add_axes([0.915, 0.11, 0.012, 0.78])
    cb = fig.colorbar(mesh, cax=cbar_ax)
    cb.set_label(r"$p(u^\prime, y \,|\, X) \,/\, \max$",
                 rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    n = int(cache["n_snapshots"])
    fig.suptitle(rf"JPDF of $u^\prime$ and $y$ for each structure type "
                 rf"($n = {n}$ snapshots)",
                 y=0.985, fontsize=10.5)
    fig.subplots_adjust(left=0.075, right=0.895,
                        bottom=0.095, top=0.925,
                        wspace=0.10, hspace=0.22)

    if save:
        save_figure(fig, "jpdf")
    return fig


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Figure B (JPDF of u' and y).")
    p.add_argument("--force-recompute", action="store_true",
                   help="Rebuild the JPDF cache from scratch.")
    p.add_argument("--max-snapshots", type=int, default=None,
                   help="Limit the cache builder to the first N snapshots (debug).")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.force_recompute:
        build_jpdf_cache(force=True, max_snapshots=args.max_snapshots)
    elif args.max_snapshots is not None and not JPDF_CACHE_PATH.exists():
        build_jpdf_cache(max_snapshots=args.max_snapshots)
    fig = make_figure()
    if not args.no_show:
        plt.show()
