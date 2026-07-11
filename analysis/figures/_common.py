"""
Shared helpers for the figures of the XAI jets study.

Conventions
-----------
- Grid shape is (shpy, shpx) = (199, 319). The y axis is vertical (cross-stream).
- Fluctuations are computed as u' = u - Umean(y), v' = v (Vmean assumed ~0).
- Structure thresholds are read from a single dict here so the whole figure
  series is reproducible from one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths (resolved from this file so scripts work regardless of cwd)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT: Path = _THIS.parents[2]                       # repo root
DATA_DIR:     Path = PROJECT_ROOT / "data"
RESULTS_DIR:  Path = PROJECT_ROOT / "results"
STA_DIR:      Path = RESULTS_DIR / "sta"
PIV_DIR:      Path = DATA_DIR / "piv"
SHAP_DIR:     Path = DATA_DIR / "SHAP"
OUT_DIR:      Path = RESULTS_DIR / "figures"
CACHE_DIR:    Path = RESULTS_DIR / "analysis" / "cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid / dataset constants
# ---------------------------------------------------------------------------
SHPY: int = 199
SHPX: int = 319
NSAMPLES_SHAP: int = 1000
PIV_FILE_TPL:  str = "piv.{idx}.h5.uvw"
SHAP_FILE_TPL: str = f"piv_jets_nsample{NSAMPLES_SHAP}.{{idx}}.h5.shap"
TRANSITION_INDICES = (1483, 2968, 4453, 5938, 7422, 8906)

# ---------------------------------------------------------------------------
# PIV physical calibration
# ---------------------------------------------------------------------------
# Raw camera calibration is 12.669 px/mm (from pivtools config). The velocity
# arrays here live on the PIV vector grid, where the last interrogation pass
# (16x16 window, 50% overlap) produces one vector every 8 raw pixels. So the
# physical spacing of our grid = 8 raw px / 12.669 px/mm ≈ 0.6315 mm.
_RAW_PX_PER_MM:  float = 12.669
_WINDOW_STEP_PX: int   = 8
PX_PER_MM:       float = _RAW_PX_PER_MM / _WINDOW_STEP_PX   # ≈ 1.584 grid pts / mm
MM_PER_PX:       float = 1.0 / PX_PER_MM                    # ≈ 0.6315 mm / grid pt
F_CAM_HZ:        float = 30.0              # camera frame rate
F_VELOCITY_HZ: float = 15.0              # velocity field rate (PIV pairing)
DT_S:          float = 1.0 / F_VELOCITY_HZ  # seconds per velocity field
F_BURST_HZ:    float = 0.5               # burst repetition rate
PERIOD_SNAP:   int   = int(F_VELOCITY_HZ / F_BURST_HZ)   # = 30 velocity fields


def px_to_mm(px) -> float:
    return px * MM_PER_PX


def mm_to_px(mm) -> float:
    return mm * PX_PER_MM


def imshow_extent_mm(shpy: int = SHPY,
                     shpx: int = SHPX) -> tuple[float, float, float, float]:
    """
    `extent` for imshow in millimetres, matching origin='upper':
    (xmin, xmax, ymax, ymin).
    """
    return (0.0, shpx * MM_PER_PX, shpy * MM_PER_PX, 0.0)


# ---------------------------------------------------------------------------
# Structure thresholds (from rapport_complet.txt §3)
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    "shap":       1.25,
    "shear":      1.50,
    "vorticity":  1.50,
    "entrainment": 1.25,
    "q_events":   1.50,
    "q2d":        1.50,
}


# ---------------------------------------------------------------------------
# Colour palette — one colour per structure type, reused across every figure
# ---------------------------------------------------------------------------
STRUCTURE_COLORS: dict[str, str] = {
    "shap":        "#d62728",   # red
    "shear":       "#ff7f0e",   # orange
    "vorticity":   "#9467bd",   # purple
    "entrainment": "#17becf",   # teal / cyan
    "q_events":    "#1f77b4",   # blue
    "q2d":         "#2ca02c",   # green
}

STRUCTURE_LABELS: dict[str, str] = {
    "shap":        r"SHAP",
    "shear":       r"Shear  $|\partial u/\partial y|$",
    "vorticity":   r"Vorticity  $|\omega_z|$",
    "entrainment": r"Entrainment  $|v^\prime|$",
    "q_events":    r"$Q$-events  $|u^\prime v^\prime|$",
    "q2d":         r"$Q_{2\mathrm{D}} = -\det(\nabla u)$",
}


# ---------------------------------------------------------------------------
# Cremades-style coincidence palette for a 3-structure set-theoretic display.
# Seven colours, one per non-empty subset of the three classifying structures.
# Index order must match compute_coincidence_regions() in _structures.py:
#     1 = A only   (first structure of the trio)
#     2 = B only   (second)
#     3 = C only   (third)
#     4 = A ∩ B  not C
#     5 = A ∩ C  not B
#     6 = B ∩ C  not A
#     7 = A ∩ B ∩ C  (all three)
# ---------------------------------------------------------------------------
COINCIDENCE_REGION_COLORS: list[str] = [
    "#d62728",   # 1: A only
    "#2ca02c",   # 2: B only
    "#1f77b4",   # 3: C only
    "#ff7f0e",   # 4: A ∩ B
    "#9467bd",   # 5: A ∩ C
    "#17becf",   # 6: B ∩ C
    "#444444",   # 7: all three
]


# ---------------------------------------------------------------------------
# Matplotlib style — clean, serif, no grid, thin spines
# ---------------------------------------------------------------------------
def apply_style() -> None:
    mpl.rcParams.update({
        "font.family":       "serif",
        "mathtext.fontset":  "stix",
        "font.size":         10.0,
        "axes.titlesize":    10.5,
        "axes.labelsize":    10.0,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   8.5,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "axes.grid":         False,
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


# ---------------------------------------------------------------------------
# Mean-profile I/O (matches py_bin/py_functions/umean.py format)
# ---------------------------------------------------------------------------
def _read_profile_line(line: str) -> np.ndarray:
    return np.asarray(line.strip().replace("[", "").replace("]", "").split(","),
                      dtype=float)


def read_profile_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a 3-line profile file (U, V, W). Returns numpy arrays of length SHPY."""
    with open(path, "r") as fh:
        lines = fh.readlines()
    u = _read_profile_line(lines[0])
    v = _read_profile_line(lines[1])
    w = _read_profile_line(lines[2])
    return u, v, w


def valid_y_mask(umean: np.ndarray, vmean: np.ndarray) -> np.ndarray:
    """Rows where both Umean and Vmean are exactly zero are PIV-masked invalid rows."""
    return ~((umean == 0.0) & (vmean == 0.0))


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def save_figure(fig: plt.Figure, name: str) -> Path:
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out)
    print(f"[saved] {out.relative_to(PROJECT_ROOT)}")
    return out
