"""
Phase-conditioned analyses of the 2D PIV jets dataset.

Level 0  -- phase-averaged velocity field     `ũ(x, y, φ)`, `ṽ(x, y, φ)`
Level 1  -- phase-conditioned SHAP statistics `<||SHAP||>(x, y | φ)`

The raw phase mapping global_index -> phase lives in
`data/piv/phase_metadata.npy` (a Python dict, saved with allow_pickle=True).
Phase is an integer in `[0, PERIOD_SNAP)`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from figures._common import (
    CACHE_DIR,
    DATA_DIR,
    NSAMPLES_SHAP,
    PERIOD_SNAP,
    PIV_DIR,
    PIV_FILE_TPL,
    SHAP_DIR,
    SHAP_FILE_TPL,
    SHPX,
    SHPY,
)


PHASE_META_PATH       = DATA_DIR / "piv" / "phase_metadata.npy"
PHASE_VEL_CACHE_PATH  = CACHE_DIR / "phase_velocity_cache.npz"
PHASE_SHAP_CACHE_PATH = CACHE_DIR / "phase_shap_cache.npz"


# ---------------------------------------------------------------------------
def load_phase_metadata() -> dict[int, int]:
    """Return {global_index: phase} loaded from phase_metadata.npy."""
    obj = np.load(PHASE_META_PATH, allow_pickle=True)
    if obj.dtype == object:
        data = obj.item()
    else:
        data = obj.tolist()
    return {int(k): int(v) for k, v in data.items()}


def indices_by_phase(meta: dict[int, int] | None = None,
                     allowed: set[int] | None = None
                     ) -> list[list[int]]:
    """
    Return a list of length PERIOD_SNAP; element φ contains the sorted list
    of global indices that land on phase φ. If `allowed` is given, only
    indices present in that set are kept (used to restrict to SHAP-available
    indices).
    """
    if meta is None:
        meta = load_phase_metadata()
    buckets: list[list[int]] = [[] for _ in range(PERIOD_SNAP)]
    for idx, phase in meta.items():
        if allowed is not None and idx not in allowed:
            continue
        if 0 <= phase < PERIOD_SNAP:
            buckets[phase].append(idx)
    for bucket in buckets:
        bucket.sort()
    return buckets


# ---------------------------------------------------------------------------
# Level 0 — phase-averaged velocity field
# ---------------------------------------------------------------------------
def build_phase_velocity_cache(force: bool = False,
                               progress_every: int = 500
                               ) -> dict[str, np.ndarray]:
    if PHASE_VEL_CACHE_PATH.exists() and not force:
        npz = np.load(PHASE_VEL_CACHE_PATH, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    meta = load_phase_metadata()
    u_sum = np.zeros((PERIOD_SNAP, SHPY, SHPX), dtype=np.float64)
    v_sum = np.zeros((PERIOD_SNAP, SHPY, SHPX), dtype=np.float64)
    counts = np.zeros(PERIOD_SNAP, dtype=np.int64)

    n_ok = 0
    for i, (idx, phase) in enumerate(sorted(meta.items())):
        path = PIV_DIR / PIV_FILE_TPL.format(idx=idx)
        try:
            with h5py.File(path, "r") as hf:
                u = np.asarray(hf["u"], dtype=np.float32)
                v = np.asarray(hf["v"], dtype=np.float32)
        except (OSError, KeyError, FileNotFoundError):
            continue
        u_sum[phase] += u
        v_sum[phase] += v
        counts[phase] += 1
        n_ok += 1

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  phase velocity: {i + 1}/{len(meta)}  ({n_ok} ok)",
                  flush=True)

    nonzero = counts > 0
    u_phase = np.zeros_like(u_sum, dtype=np.float32)
    v_phase = np.zeros_like(v_sum, dtype=np.float32)
    u_phase[nonzero] = (u_sum[nonzero] / counts[nonzero, None, None]).astype(np.float32)
    v_phase[nonzero] = (v_sum[nonzero] / counts[nonzero, None, None]).astype(np.float32)

    cache = {
        "u_phase":       u_phase,
        "v_phase":       v_phase,
        "counts":        counts.astype(np.int32),
        "period":        np.asarray(PERIOD_SNAP, dtype=np.int32),
        "n_total":       np.asarray(n_ok, dtype=np.int32),
    }
    np.savez(PHASE_VEL_CACHE_PATH, **cache)
    print(f"[cache] phase velocity written to {PHASE_VEL_CACHE_PATH.name} "
          f"(n={n_ok}, counts/phase={counts.min()}..{counts.max()})")
    return cache


# ---------------------------------------------------------------------------
# Level 1 — phase-conditioned SHAP magnitude
# ---------------------------------------------------------------------------
def _shap_indices_set() -> set[int]:
    import re
    rx = re.compile(rf"piv_jets_nsample{NSAMPLES_SHAP}\.(\d+)\.h5\.shap$")
    out = set()
    for f in SHAP_DIR.iterdir():
        m = rx.match(f.name)
        if m:
            out.add(int(m.group(1)))
    return out


def build_phase_shap_cache(force: bool = False,
                           progress_every: int = 200
                           ) -> dict[str, np.ndarray]:
    if PHASE_SHAP_CACHE_PATH.exists() and not force:
        npz = np.load(PHASE_SHAP_CACHE_PATH, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    meta = load_phase_metadata()
    shap_set = _shap_indices_set()

    mag_sum = np.zeros((PERIOD_SNAP, SHPY, SHPX), dtype=np.float64)
    counts  = np.zeros(PERIOD_SNAP, dtype=np.int64)

    indices = sorted(i for i in shap_set if i in meta)
    n_ok = 0
    for i, idx in enumerate(indices):
        phase = meta[idx]
        path = SHAP_DIR / SHAP_FILE_TPL.format(idx=idx)
        try:
            with h5py.File(path, "r") as hf:
                su = np.asarray(hf["SHAP_u"], dtype=np.float32)
                sv = np.asarray(hf["SHAP_v"], dtype=np.float32)
        except (OSError, KeyError, FileNotFoundError):
            continue
        mag = np.sqrt(su * su + sv * sv)
        mag_sum[phase] += mag
        counts[phase] += 1
        n_ok += 1

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  phase shap: {i + 1}/{len(indices)}  ({n_ok} ok)",
                  flush=True)

    nonzero = counts > 0
    shap_mag_phase = np.zeros_like(mag_sum, dtype=np.float32)
    shap_mag_phase[nonzero] = (mag_sum[nonzero] / counts[nonzero, None, None]
                               ).astype(np.float32)

    cache = {
        "shap_mag_phase": shap_mag_phase,
        "counts":         counts.astype(np.int32),
        "period":         np.asarray(PERIOD_SNAP, dtype=np.int32),
        "n_total":        np.asarray(n_ok, dtype=np.int32),
    }
    np.savez(PHASE_SHAP_CACHE_PATH, **cache)
    print(f"[cache] phase SHAP written to {PHASE_SHAP_CACHE_PATH.name} "
          f"(n={n_ok}, counts/phase={counts.min()}..{counts.max()})")
    return cache
