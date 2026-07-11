"""
Data loading and structure segmentation for the figures.

All segmenters return a boolean mask of shape (SHPY, SHPX). Thresholds are
taken from figures._common.THRESHOLDS by default.

The rms profiles needed for thresholding (shear, vorticity, Reynolds stress,
Q_2D, SHAP magnitude) are computed once over a subset of snapshots and cached
to results/analysis/cache/rms_cache.npz.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from figures._common import (
    CACHE_DIR,
    NSAMPLES_SHAP,
    PIV_DIR,
    PIV_FILE_TPL,
    SHAP_DIR,
    SHAP_FILE_TPL,
    SHPX,
    SHPY,
    STA_DIR,
    THRESHOLDS,
    TRANSITION_INDICES,
    read_profile_file,
    valid_y_mask,
)


RMS_CACHE_PATH   = CACHE_DIR / "rms_cache.npz"
COINC_CACHE_PATH = CACHE_DIR / "coinc_profile_cache.npz"


# Candidate snapshots used by find_active_index (spread across all 7 runs,
# away from transitions, multiples of 5 so the SHAP files also exist).
ACTIVITY_CANDIDATES: tuple[int, ...] = (
    205, 505, 805, 1105, 1405,
    1705, 2005, 2305, 2605, 2905,
    3205, 3505, 3805, 4105, 4405,
    4705, 5005, 5305, 5605, 5905,
    6205, 6505, 6805, 7105, 7405,
    7705, 8005, 8305, 8605, 8885,
    9205, 9505, 9805,
)

# Region of interest for "interaction intensity" (the hot spot identified in
# rapport_complet.txt §5.4: x ~ [200, 275], y ~ [100, 140]).
_HOTSPOT_Y = slice(95, 150)
_HOTSPOT_X = slice(190, 290)


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------
def load_velocity(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read raw u, v from the PIV snapshot (shape (SHPY, SHPX))."""
    path = PIV_DIR / PIV_FILE_TPL.format(idx=idx)
    with h5py.File(path, "r") as hf:
        u = np.asarray(hf["u"], dtype=np.float32)
        v = np.asarray(hf["v"], dtype=np.float32)
    return u, v


def load_shap(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read SHAP_u and SHAP_v from the gradient-SHAP output file."""
    path = SHAP_DIR / SHAP_FILE_TPL.format(idx=idx)
    with h5py.File(path, "r") as hf:
        su = np.asarray(hf["SHAP_u"], dtype=np.float32)
        sv = np.asarray(hf["SHAP_v"], dtype=np.float32)
    return su, sv


def load_mean_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Umean, Vmean, valid_y_mask_bool). Shapes: (SHPY,)."""
    u_mean, v_mean, _ = read_profile_file(STA_DIR / "Umean.txt")
    mask_valid = valid_y_mask(u_mean, v_mean)
    return u_mean, v_mean, mask_valid


def find_active_index(candidates: tuple[int, ...] | list[int] = ACTIVITY_CANDIDATES,
                      verbose: bool = True) -> int:
    """
    Scan `candidates` and return the index with the strongest activity in the
    interaction hot spot, measured as <|v'|^2 + |u'|^2> over the ROI.
    Only requires velocity files (no SHAP), so it is fast (~15 s for 33 idx).
    """
    u_mean, v_mean, _ = load_mean_profiles()
    best_idx, best_score = candidates[0], -np.inf
    for idx in candidates:
        try:
            u, v = load_velocity(idx)
        except (OSError, KeyError, FileNotFoundError):
            continue
        u_fluc = u - u_mean[:, None]
        v_fluc = v - v_mean[:, None]
        score = float(np.mean(u_fluc[_HOTSPOT_Y, _HOTSPOT_X] ** 2 +
                              v_fluc[_HOTSPOT_Y, _HOTSPOT_X] ** 2))
        if verbose:
            print(f"  idx={idx:5d}  activity={score:.4e}")
        if score > best_score:
            best_score, best_idx = score, idx
    if verbose:
        print(f"[auto] selected index {best_idx}  (score={best_score:.4e})")
    return best_idx


def load_fluctuations(idx: int,
                      u_mean: np.ndarray | None = None,
                      v_mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (u', v') = (u - Umean(y), v - Vmean(y))."""
    if u_mean is None or v_mean is None:
        u_mean, v_mean, _ = load_mean_profiles()
    u, v = load_velocity(idx)
    u_fluc = u - u_mean[:, None]
    v_fluc = v - v_mean[:, None]
    return u_fluc, v_fluc


# ---------------------------------------------------------------------------
# Gradient fields and derived quantities
# ---------------------------------------------------------------------------
def compute_gradients(u: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Second-order central differences via np.gradient (unit spacing)."""
    dudy = np.gradient(u, axis=0)
    dudx = np.gradient(u, axis=1)
    dvdy = np.gradient(v, axis=0)
    dvdx = np.gradient(v, axis=1)
    omega_z = dvdx - dudy
    q2d = dudy * dvdx - dudx * dvdy          # -det(grad u)
    return {"dudx": dudx, "dudy": dudy,
            "dvdx": dvdx, "dvdy": dvdy,
            "omega_z": omega_z, "q2d": q2d}


# ---------------------------------------------------------------------------
# RMS cache (y-profiles averaged over x and over a sample of snapshots)
# ---------------------------------------------------------------------------
def _default_sample_indices(n_samples: int = 200, step: int = 5) -> list[int]:
    """
    Pick `n_samples` snapshot indices that are multiples of `step`, evenly
    spread across [0, 10385] and at least 5 frames away from any run transition.
    """
    all_multiples = np.arange(step, 10385 + 1, step)
    safe = [i for i in all_multiples
            if not any(abs(i - t) < 5 for t in TRANSITION_INDICES)]
    if len(safe) <= n_samples:
        return list(safe)
    stride = len(safe) // n_samples
    return [int(safe[k * stride]) for k in range(n_samples)]


def _accumulate_profile(field: np.ndarray, acc_sq: np.ndarray) -> None:
    """In-place accumulate mean-over-x of field**2 into acc_sq (shape (SHPY,))."""
    acc_sq += np.mean(field * field, axis=1)


def compute_rms_cache(indices: list[int] | None = None,
                      force: bool = False) -> dict[str, np.ndarray]:
    """
    Compute (or read from cache) the rms profiles used for thresholding.

    Returned dict keys:
        - 'shear_rms'       : rms(|du/dy|)(y)
        - 'omega_rms'       : rms(|omega_z|)(y)
        - 'uv_rms'          : rms(|u'v'|)(y)    (Reynolds stress magnitude)
        - 'q2d_rms'         : rms(|Q_2D|)(y)
        - 'vfluc_rms'       : rms(|v'|)(y)
        - 'shap_mag_rms'    : rms(||SHAP||)(y)
        - 'indices_used'    : indices that were averaged
    """
    if RMS_CACHE_PATH.exists() and not force:
        npz = np.load(RMS_CACHE_PATH, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    if indices is None:
        indices = _default_sample_indices()

    u_mean, v_mean, _ = load_mean_profiles()

    shear_sq   = np.zeros(SHPY, dtype=np.float64)
    omega_sq   = np.zeros(SHPY, dtype=np.float64)
    uv_sq      = np.zeros(SHPY, dtype=np.float64)
    q2d_sq     = np.zeros(SHPY, dtype=np.float64)
    vfluc_sq   = np.zeros(SHPY, dtype=np.float64)
    shap_sq    = np.zeros(SHPY, dtype=np.float64)

    n_vel, n_shap = 0, 0
    for i, idx in enumerate(indices):
        try:
            u, v = load_velocity(idx)
        except (OSError, KeyError):
            continue
        u_fluc = u - u_mean[:, None]
        v_fluc = v - v_mean[:, None]
        grads = compute_gradients(u, v)

        _accumulate_profile(grads["dudy"],   shear_sq)
        _accumulate_profile(grads["omega_z"], omega_sq)
        _accumulate_profile(u_fluc * v_fluc, uv_sq)
        _accumulate_profile(grads["q2d"],     q2d_sq)
        _accumulate_profile(v_fluc,           vfluc_sq)
        n_vel += 1

        try:
            su, sv = load_shap(idx)
            mag = np.sqrt(su * su + sv * sv)
            _accumulate_profile(mag, shap_sq)
            n_shap += 1
        except (OSError, KeyError, FileNotFoundError):
            pass

        if (i + 1) % 20 == 0:
            print(f"  rms cache: {i + 1}/{len(indices)} snapshots processed "
                  f"({n_vel} velocity, {n_shap} SHAP)", flush=True)

    if n_vel == 0:
        raise RuntimeError("compute_rms_cache: no velocity snapshots could be loaded.")

    shear_rms = np.sqrt(shear_sq / n_vel)
    omega_rms = np.sqrt(omega_sq / n_vel)
    uv_rms    = np.sqrt(uv_sq    / n_vel)
    q2d_rms   = np.sqrt(q2d_sq   / n_vel)
    vfluc_rms = np.sqrt(vfluc_sq / n_vel)
    shap_rms  = np.sqrt(shap_sq  / max(n_shap, 1))

    cache = {
        "shear_rms":    shear_rms.astype(np.float32),
        "omega_rms":    omega_rms.astype(np.float32),
        "uv_rms":       uv_rms.astype(np.float32),
        "q2d_rms":      q2d_rms.astype(np.float32),
        "vfluc_rms":    vfluc_rms.astype(np.float32),
        "shap_mag_rms": shap_rms.astype(np.float32),
        "indices_used": np.asarray(indices, dtype=np.int32),
    }
    np.savez(RMS_CACHE_PATH, **cache)
    print(f"[cache] rms profiles written to {RMS_CACHE_PATH.name} "
          f"(velocity={n_vel}, shap={n_shap})")
    return cache


# ---------------------------------------------------------------------------
# Segmentation (boolean masks, shape (SHPY, SHPX))
# ---------------------------------------------------------------------------
def _threshold_y_profile(field: np.ndarray,
                         rms_profile: np.ndarray,
                         H: float,
                         eps: float = 1e-12) -> np.ndarray:
    denom = rms_profile[:, None]
    denom = np.where(denom < eps, np.inf, denom)
    return np.abs(field) > H * denom


def segment_shear(grads: dict, rms: dict, H: float | None = None) -> np.ndarray:
    H = THRESHOLDS["shear"] if H is None else H
    return _threshold_y_profile(grads["dudy"], rms["shear_rms"], H)


def segment_vorticity(grads: dict, rms: dict, H: float | None = None) -> np.ndarray:
    H = THRESHOLDS["vorticity"] if H is None else H
    return _threshold_y_profile(grads["omega_z"], rms["omega_rms"], H)


def segment_entrainment(v_fluc: np.ndarray, rms: dict, H: float | None = None) -> np.ndarray:
    H = THRESHOLDS["entrainment"] if H is None else H
    return _threshold_y_profile(v_fluc, rms["vfluc_rms"], H)


def segment_q2d(grads: dict, rms: dict, H: float | None = None) -> np.ndarray:
    """Only positive Q_2D (rotation-dominated) above the threshold."""
    H = THRESHOLDS["q2d"] if H is None else H
    denom = rms["q2d_rms"][:, None]
    denom = np.where(denom < 1e-12, np.inf, denom)
    return grads["q2d"] > H * denom


def segment_q_events(u_fluc: np.ndarray, v_fluc: np.ndarray,
                     rms: dict, H: float | None = None) -> dict[str, np.ndarray]:
    """Return {'Q1','Q2','Q3','Q4','any'} masks. Q_i filtered by |u'v'|."""
    H = THRESHOLDS["q_events"] if H is None else H
    uv = u_fluc * v_fluc
    strong = _threshold_y_profile(uv, rms["uv_rms"], H)
    q1 = strong & (u_fluc > 0) & (v_fluc > 0)
    q2 = strong & (u_fluc < 0) & (v_fluc > 0)   # ejection
    q3 = strong & (u_fluc < 0) & (v_fluc < 0)
    q4 = strong & (u_fluc > 0) & (v_fluc < 0)   # sweep
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "any": strong}


def segment_shap(shap_u: np.ndarray, shap_v: np.ndarray,
                 rms: dict, H: float | None = None) -> np.ndarray:
    H = THRESHOLDS["shap"] if H is None else H
    mag = np.sqrt(shap_u * shap_u + shap_v * shap_v)
    return _threshold_y_profile(mag, rms["shap_mag_rms"], H)


def _shap_available_indices() -> list[int]:
    """List all snapshot indices for which a SHAP output file exists."""
    import re
    pattern = re.compile(
        rf"piv_jets_nsample{NSAMPLES_SHAP}\.(\d+)\.h5\.shap$"
    )
    indices: list[int] = []
    for f in SHAP_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            indices.append(int(m.group(1)))
    indices.sort()
    return indices


def compute_coincidence_cache(indices: list[int] | None = None,
                              force: bool = False,
                              progress_every: int = 100) -> dict[str, np.ndarray]:
    """
    Accumulate per-y-row counts of |X|, |SHAP|, and |SHAP ∩ X| over all
    available SHAP snapshots, for X ∈ {shear, vorticity, entrainment, q2d}.

    Returned dict keys:
        count_shap               : (SHPY,)  int64, Σ_{t,x} 1[SHAP]
        count_<X>                : (SHPY,)  int64, Σ_{t,x} 1[X]
        inter_shap_<X>           : (SHPY,)  int64, Σ_{t,x} 1[SHAP ∧ X]
        valid_y                  : (SHPY,)  bool   PIV validity mask
        indices_used             : (N,)     int32  snapshot indices
        n_snapshots              : ()       int32  number actually processed
    """
    if COINC_CACHE_PATH.exists() and not force:
        npz = np.load(COINC_CACHE_PATH, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    if indices is None:
        indices = _shap_available_indices()

    rms = compute_rms_cache()
    u_mean, v_mean, valid_y = load_mean_profiles()

    n_y = SHPY
    counts = {k: np.zeros(n_y, dtype=np.int64)
              for k in ("shap", "shear", "vorticity", "entrainment", "q2d")}
    inters = {k: np.zeros(n_y, dtype=np.int64)
              for k in ("shear", "vorticity", "entrainment", "q2d")}

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

        m = {
            "shap":        segment_shap(su, sv, rms),
            "shear":       segment_shear(grads, rms),
            "vorticity":   segment_vorticity(grads, rms),
            "entrainment": segment_entrainment(v_fluc, rms),
            "q2d":         segment_q2d(grads, rms),
        }
        for mk in m.values():
            mk[~valid_y, :] = False

        for k, mk in m.items():
            counts[k] += mk.sum(axis=1, dtype=np.int64)
        for k in inters:
            inters[k] += (m["shap"] & m[k]).sum(axis=1, dtype=np.int64)

        n_ok += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  coinc cache: {i + 1}/{len(indices)} snapshots "
                  f"({n_ok} ok)", flush=True)

    if n_ok == 0:
        raise RuntimeError("compute_coincidence_cache: no snapshots loaded.")

    cache = {
        "count_shap":              counts["shap"],
        "count_shear":             counts["shear"],
        "count_vorticity":         counts["vorticity"],
        "count_entrainment":       counts["entrainment"],
        "count_q2d":               counts["q2d"],
        "inter_shap_shear":        inters["shear"],
        "inter_shap_vorticity":    inters["vorticity"],
        "inter_shap_entrainment":  inters["entrainment"],
        "inter_shap_q2d":          inters["q2d"],
        "valid_y":                 valid_y.astype(np.int8),
        "indices_used":            np.asarray(indices, dtype=np.int32),
        "n_snapshots":             np.asarray(n_ok, dtype=np.int32),
    }
    np.savez(COINC_CACHE_PATH, **cache)
    print(f"[cache] coincidence profiles written to {COINC_CACHE_PATH.name} "
          f"(n={n_ok})")
    return cache


def compute_coincidence_regions(mask_a: np.ndarray,
                                mask_b: np.ndarray,
                                mask_c: np.ndarray) -> np.ndarray:
    """
    Cremades-style 3-set coincidence region map (integer labels 0..7):

        0 = no structure            (background)
        1 = A \\ (B ∪ C)             (A only)
        2 = B \\ (A ∪ C)             (B only)
        3 = C \\ (A ∪ B)             (C only)
        4 = (A ∩ B) \\ C             (A and B, not C)
        5 = (A ∩ C) \\ B             (A and C, not B)
        6 = (B ∩ C) \\ A             (B and C, not A)
        7 = A ∩ B ∩ C                (all three)
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    c = mask_c.astype(bool)
    out = np.zeros(a.shape, dtype=np.int8)
    out[ a & ~b & ~c] = 1
    out[~a &  b & ~c] = 2
    out[~a & ~b &  c] = 3
    out[ a &  b & ~c] = 4
    out[ a & ~b &  c] = 5
    out[~a &  b &  c] = 6
    out[ a &  b &  c] = 7
    return out


# ---------------------------------------------------------------------------
# Convenience: everything at once for one snapshot
# ---------------------------------------------------------------------------
def all_masks_for_snapshot(idx: int,
                           rms: dict | None = None
                           ) -> dict[str, np.ndarray]:
    """
    One-call helper returning all structure masks plus the raw fields.
    """
    if rms is None:
        rms = compute_rms_cache()

    u_mean, v_mean, mask_y = load_mean_profiles()
    u, v = load_velocity(idx)
    u_fluc = u - u_mean[:, None]
    v_fluc = v - v_mean[:, None]
    grads = compute_gradients(u, v)
    shap_u, shap_v = load_shap(idx)

    masks = {
        "shear":       segment_shear(grads, rms),
        "vorticity":   segment_vorticity(grads, rms),
        "entrainment": segment_entrainment(v_fluc, rms),
        "q_events":    segment_q_events(u_fluc, v_fluc, rms)["any"],
        "q2d":         segment_q2d(grads, rms),
        "shap":        segment_shap(shap_u, shap_v, rms),
    }

    masks_valid = np.broadcast_to(mask_y[:, None], (SHPY, SHPX))
    for k in masks:
        masks[k] = masks[k] & masks_valid

    return {
        "u": u, "v": v,
        "u_fluc": u_fluc, "v_fluc": v_fluc,
        "shap_u": shap_u, "shap_v": shap_v,
        "grads": grads,
        "masks": masks,
        "valid_y": mask_y,
    }
