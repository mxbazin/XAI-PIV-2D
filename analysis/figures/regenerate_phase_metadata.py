"""
Regenerate phase_metadata.npy with physically correct values.

Two bugs in the original convert_to_h5.py had to be fixed :

1.  `T_period` was computed as `f_cam / f_burst = 30 / 0.5 = 60`, i.e. the
    camera-frame rate divided by the burst rate. But the stored HDF5 files
    are velocity fields (PIV pairs) at 15 Hz, so the true period is 30.

2.  `frame_refs = {1:27, ...}` are the local indices of the first burst
    start in each run, expressed in **camera-frame** units (1-based). They
    were being subtracted from a velocity-field local index directly, which
    mixed two units.  The velocity-field equivalent of a camera frame
    `F_cam` (1-based) is `(F_cam + 1) // 2` (1-based), since PIV pair k
    uses camera frames (2k, 2k+1) (1-based equivalently).

After the fix, the formula becomes
        phase = (local_vel_idx - (frame_ref_velocity - 1)) % 30
and phase 0 corresponds to the velocity field that actually contains the
burst start.

This script :
    1. reads data/piv/run_metadata.npy  (dict {global_index: (run, local_idx)})
    2. recomputes phase with T_period = 30 and frame_refs converted to
       velocity-field units
    3. writes a backup of the old phase_metadata.npy alongside the new one
       (keeps the very first backup, so re-runs never overwrite the pristine
       original)
    4. prints diagnostics so the correction is auditable.

It does NOT touch any .h5 file, any SHAP file, or any stats. Only the
metadata dict is rewritten.

Usage (from code/):

    python -m figures.regenerate_phase_metadata
    python -m figures.regenerate_phase_metadata --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import DATA_DIR


PHASE_META_PATH = DATA_DIR / "piv" / "phase_metadata.npy"
RUN_META_PATH   = DATA_DIR / "piv" / "run_metadata.npy"
BACKUP_SUFFIX   = "_T60_backup"

# Original frame_refs from convert_to_h5.py, in CAMERA-FRAME units (1-based).
FRAME_REFS_CAMERA_1BASED: dict[int, int] = {
    1: 27, 2: 30, 3: 8, 4: 42, 5: 6, 6: 10, 7: 13,
}

# Same quantity expressed in VELOCITY-FIELD units (1-based). Conversion :
#   velocity pair k (1-based) uses camera frames (2k-1, 2k),
#   so camera frame F_cam (1-based) is contained in pair  ceil(F_cam / 2)
#   which equals  (F_cam + 1) // 2  in integer arithmetic.
FRAME_REFS: dict[int, int] = {
    k: (v + 1) // 2 for k, v in FRAME_REFS_CAMERA_1BASED.items()
}

# Corrected period, in units of velocity fields (PIV pairs).
T_PERIOD_NEW: int = 30


def _load_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    return obj.item() if obj.dtype == object else obj.tolist()


def regenerate(dry_run: bool = False) -> None:
    print(f"Reading run metadata       : {RUN_META_PATH}")
    if not RUN_META_PATH.exists():
        raise FileNotFoundError(f"run_metadata.npy not found at {RUN_META_PATH}")
    run_map = _load_dict(RUN_META_PATH)

    print(f"Reading existing phase map : {PHASE_META_PATH}")
    old_phase = _load_dict(PHASE_META_PATH)

    print("")
    print("frame_refs translation (camera 1-based -> velocity 1-based):")
    for r in sorted(FRAME_REFS_CAMERA_1BASED):
        print(f"  run {r}:  camera {FRAME_REFS_CAMERA_1BASED[r]:3d}  "
              f"-> velocity {FRAME_REFS[r]:3d}")

    new_phase: dict[int, int] = {}
    mismatch = 0
    per_run_counts: dict[int, int] = {}
    for global_idx, (run_idx, local_idx) in run_map.items():
        run_idx   = int(run_idx)
        local_idx = int(local_idx)
        if run_idx not in FRAME_REFS:
            raise RuntimeError(f"no frame_ref for run {run_idx}")
        phase_new = (local_idx - (FRAME_REFS[run_idx] - 1)) % T_PERIOD_NEW
        new_phase[int(global_idx)] = int(phase_new)
        per_run_counts[run_idx] = per_run_counts.get(run_idx, 0) + 1
        if int(old_phase.get(int(global_idx), -1)) % T_PERIOD_NEW != phase_new:
            mismatch += 1

    # Diagnostics
    vals = np.asarray(list(new_phase.values()))
    counts = np.bincount(vals, minlength=T_PERIOD_NEW)
    print("")
    print(f"Corrected T_period         : {T_PERIOD_NEW} velocity fields")
    print(f"Total entries              : {len(new_phase)}")
    print(f"Phase range                : [{vals.min()}, {vals.max()}]")
    print(f"Counts per phase           : min={counts.min()}, "
          f"max={counts.max()}, mean={counts.mean():.1f}")
    print(f"Entries per run            : {per_run_counts}")
    print(f"Entries where (old % 30) differs from new : {mismatch}")
    print("")
    print("Sanity samples (global_idx -> old, new) :")
    for sample in (0, 10, 26, 27, 28, 100, 1483, 1484, 1485, 5000):
        if sample in new_phase:
            print(f"  {sample:6d}  :  old={old_phase.get(sample):3d}   "
                  f"new={new_phase[sample]:2d}")

    if dry_run:
        print("\n[dry-run] no files written.")
        return

    backup = PHASE_META_PATH.with_name(PHASE_META_PATH.stem + BACKUP_SUFFIX + ".npy")
    if not backup.exists():
        shutil.copy2(PHASE_META_PATH, backup)
        print(f"\nBacked up old metadata to  : {backup}")
    else:
        print(f"\nBackup already exists      : {backup}  (left untouched)")

    np.save(PHASE_META_PATH, new_phase)
    print(f"Wrote corrected metadata   : {PHASE_META_PATH}")
    print(f"                               (T_period = {T_PERIOD_NEW})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate phase_metadata.npy with T_period = 30 "
                    "(velocity-field rate, PIV pairing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print the new phases but do not save anything.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    regenerate(dry_run=args.dry_run)
