"""
Black-box readers for the 2D PIV jets pipeline outputs.

This module is fully self-contained: it never imports from ../code. It only
reads files produced by the pipeline (PIV snapshots, SHAP attributions, mean
profiles, cached rms / coincidence profiles) so that analysis code in this
folder stays decoupled from the pipeline internals.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Paths (resolved from this file — safe regardless of cwd)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT: Path = _THIS.parents[1]
DATA_DIR:     Path = PROJECT_ROOT / "data"
RESULTS_DIR:  Path = PROJECT_ROOT / "results"
PIV_DIR:      Path = DATA_DIR / "piv"
SHAP_DIR:     Path = DATA_DIR / "SHAP"
STA_DIR:      Path = RESULTS_DIR / "sta"
CACHE_DIR:    Path = RESULTS_DIR / "analysis" / "cache"

OUT_DIR:      Path = PROJECT_ROOT / "results" / "postprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants (copied from analysis/figures/_common.py — do not import)
# ---------------------------------------------------------------------------
SHPY: int = 199
SHPX: int = 319
NSAMPLES_SHAP: int = 1000

PIV_FILE_TPL:  str = "piv.{idx}.h5.uvw"
SHAP_FILE_TPL: str = f"piv_jets_nsample{NSAMPLES_SHAP}.{{idx}}.h5.shap"

# Snapshots at run boundaries — avoid when sampling
TRANSITION_INDICES: tuple[int, ...] = (1483, 2968, 4453, 5938, 7422, 8906)

# Raw camera calibration is 12.669 px/mm (pivtools config). Our velocity
# arrays live on the PIV vector grid — last pass has 16x16 windows with 50%
# overlap, producing 1 vector every 8 raw pixels. Effective grid spacing:
# 8 raw px / 12.669 px/mm ≈ 0.6315 mm per grid point.
_RAW_PX_PER_MM:  float = 12.669
_WINDOW_STEP_PX: int   = 8
PX_PER_MM:     float = _RAW_PX_PER_MM / _WINDOW_STEP_PX   # ≈ 1.584 grid pts / mm
MM_PER_PX:     float = 1.0 / PX_PER_MM                    # ≈ 0.6315 mm / grid pt
F_VELOCITY_HZ: float = 15.0
DT_S:          float = 1.0 / F_VELOCITY_HZ
F_BURST_HZ:    float = 0.5
PERIOD_SNAP:   int   = int(F_VELOCITY_HZ / F_BURST_HZ)   # 30

THRESHOLDS: dict[str, float] = {
    "shap":        1.25,
    "shear":       1.50,
    "vorticity":   1.50,
    "entrainment": 1.25,
    "q_events":    1.50,
    "q2d":         1.50,
}


# ---------------------------------------------------------------------------
# Raw field readers
# ---------------------------------------------------------------------------
def load_velocity(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read u, v from PIV snapshot `idx`. Returns arrays of shape (SHPY, SHPX)."""
    path = PIV_DIR / PIV_FILE_TPL.format(idx=idx)
    with h5py.File(path, "r") as hf:
        u = np.asarray(hf["u"], dtype=np.float32)
        v = np.asarray(hf["v"], dtype=np.float32)
    return u, v


def load_shap(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read SHAP_u, SHAP_v from the gradient-SHAP output file for snapshot `idx`."""
    path = SHAP_DIR / SHAP_FILE_TPL.format(idx=idx)
    with h5py.File(path, "r") as hf:
        su = np.asarray(hf["SHAP_u"], dtype=np.float32)
        sv = np.asarray(hf["SHAP_v"], dtype=np.float32)
    return su, sv


def load_mean_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse results/sta/Umean.txt (3 lines: U, V, W) and return (Umean, Vmean, valid_y).
    valid_y is False on rows where the PIV mask zeroed both U and V.
    """
    path = STA_DIR / "Umean.txt"
    with open(path, "r") as fh:
        lines = fh.readlines()

    def _parse(line: str) -> np.ndarray:
        return np.asarray(
            line.strip().replace("[", "").replace("]", "").split(","),
            dtype=float,
        )

    u_mean = _parse(lines[0])
    v_mean = _parse(lines[1])
    valid_y = ~((u_mean == 0.0) & (v_mean == 0.0))
    return u_mean, v_mean, valid_y


def load_fluctuations(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (u', v') = (u - Umean(y), v - Vmean(y)) for snapshot `idx`."""
    u, v = load_velocity(idx)
    u_mean, v_mean, _ = load_mean_profiles()
    return u - u_mean[:, None], v - v_mean[:, None]


# ---------------------------------------------------------------------------
# Pre-computed caches
# ---------------------------------------------------------------------------
def load_rms_cache() -> dict[str, np.ndarray]:
    """
    Load results/analysis/cache/rms_cache.npz.

    Keys: shear_rms, omega_rms, uv_rms, q2d_rms, vfluc_rms, shap_mag_rms,
          indices_used.
    """
    npz = np.load(CACHE_DIR / "rms_cache.npz", allow_pickle=False)
    return {k: npz[k] for k in npz.files}


def load_coincidence_cache() -> dict[str, np.ndarray]:
    """
    Load results/analysis/cache/coinc_profile_cache.npz.

    Keys: count_shap, count_{shear,vorticity,entrainment,q2d},
          inter_shap_{shear,vorticity,entrainment,q2d},
          valid_y, indices_used, n_snapshots.
    """
    npz = np.load(CACHE_DIR / "coinc_profile_cache.npz", allow_pickle=False)
    return {k: npz[k] for k in npz.files}


def load_npz_cache(name: str) -> dict[str, np.ndarray]:
    """
    Generic loader for any .npz under results/analysis/cache/.
    Example: load_npz_cache("jpdf_cache").
    """
    if not name.endswith(".npz"):
        name = name + ".npz"
    npz = np.load(CACHE_DIR / name, allow_pickle=False)
    return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def available_piv_indices() -> list[int]:
    """All idx for which data/piv/piv.{idx}.h5.uvw exists."""
    out = []
    for f in PIV_DIR.iterdir():
        stem = f.name
        if stem.startswith("piv.") and stem.endswith(".h5.uvw"):
            try:
                out.append(int(stem[len("piv."):-len(".h5.uvw")]))
            except ValueError:
                pass
    out.sort()
    return out


def available_shap_indices() -> list[int]:
    """All idx for which data/SHAP/piv_jets_nsample1000.{idx}.h5.shap exists."""
    import re
    pattern = re.compile(rf"piv_jets_nsample{NSAMPLES_SHAP}\.(\d+)\.h5\.shap$")
    out = []
    for f in SHAP_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out
