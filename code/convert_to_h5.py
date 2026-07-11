# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:42:48 2026

@author: Maxim
"""

# -*- coding: utf-8 -*-
"""
Convert PIV .mat files to HDF5 format for the SHAP pipeline.

Output files:
    - piv.{global_index}.h5.uvw  : HDF5 with keys u, v, mx, my
    - phase_metadata.npy          : dict mapping global_index -> phase (0..59)
    - run_metadata.npy            : dict mapping global_index -> (run, local_index)
"""
import scipy.io
import numpy as np
import h5py
import os
import glob

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
f_cam    = 30     # Hz acquisition rate
f_burst  = 0.5   # Hz burst frequency
T_period = int(f_cam / f_burst)  # = 60 snapshots per period

base_path  = r"E:\ME695\PIV\2026-03-27"
run_name   = "RUN{:02d}_dt0.5ms_laser30Hz_Apulse_25.5Hz_3V_N7500_T2ms_Bcontinuous_25.5Hz_2.3V"
output_dir = r"C:\Users\Maxim\Downloads\XAI_turbulentchannel_3d_simplified-main\XAI_turbulentchannel_3d_simplified-main\data\piv"
os.makedirs(output_dir, exist_ok=True)

# Frame reference = index of first burst start in each run (1-based from .mat)
# This is the local snapshot index where the burst begins
frame_refs = {1: 27, 2: 30, 3: 8, 4: 42, 5: 6, 6: 10, 7: 13}

# -----------------------------------------------------------------------
# Helper: find snapshot folder (1484 or 1485)
# -----------------------------------------------------------------------
def get_run_path(base_path, run_name_fmt, run_idx):
    for n_snap in ['1484', '1485']:
        path = os.path.join(base_path, run_name_fmt.format(run_idx),
                            f"proc\\calibrated_piv\\{n_snap}\\Cam1\\instantaneous")
        if os.path.exists(path):
            return path
    return None

# -----------------------------------------------------------------------
# Main conversion loop
# -----------------------------------------------------------------------
global_index  = 0
phase_map     = {}   # global_index -> phase (0..59)
run_map       = {}   # global_index -> (run_idx, local_index)
run_lengths   = {}   # run_idx -> number of snapshots

for run_idx in range(1, 8):
    print(f"\n{'='*60}")
    print(f"Processing RUN{run_idx:02d}")
    print(f"{'='*60}")

    run_path = get_run_path(base_path, run_name, run_idx)
    if run_path is None:
        print(f"  WARNING: folder not found for RUN{run_idx:02d}, skipping.")
        continue

    files = sorted([
        f for f in glob.glob(os.path.join(run_path, '*.mat'))
        if os.path.basename(f)[:5].isdigit()
    ])
    n_files = len(files)
    run_lengths[run_idx] = n_files
    print(f"  Found {n_files} snapshots")

    frame_ref = frame_refs[run_idx]  # local index where burst starts (1-based)

    for local_idx, fpath in enumerate(files):
        # ------------------------------------------------------------------
        # Compute phase (0-based local index relative to burst start)
        # ------------------------------------------------------------------
        phase = (local_idx - (frame_ref - 1)) % T_period  # frame_ref is 1-based

        # ------------------------------------------------------------------
        # Read .mat file
        # ------------------------------------------------------------------
        try:
            piv = scipy.io.loadmat(fpath)['piv_result']
            ux  = piv['ux'][0, 1].astype(float)   # shape (199, 319) = (my, mx)
            uy  = piv['uy'][0, 1].astype(float)   # shape (199, 319)
        except Exception as e:
            print(f"  ERROR reading snapshot {local_idx+1}: {e}")
            continue

        my, mx = ux.shape  # 199, 319

        # ------------------------------------------------------------------
        # Write HDF5 — keys expected by read_velocity.py
        # shape stored as (my, mx) = (199, 319), no transpose needed
        # read_velocity does [::dy, ::dx] which works on (my, mx)
        # ------------------------------------------------------------------
        fname = f'piv.{global_index}.h5.uvw'
        fout  = os.path.join(output_dir, fname)

        with h5py.File(fout, 'w') as f:
            f.create_dataset('u',  data=ux)           # (199, 319)
            f.create_dataset('v',  data=uy)           # (199, 319)
            f.create_dataset('mx', data=np.array([mx]))  # [319]
            f.create_dataset('my', data=np.array([my]))  # [199]

        # ------------------------------------------------------------------
        # Store metadata
        # ------------------------------------------------------------------
        phase_map[global_index] = int(phase)
        run_map[global_index]   = (run_idx, local_idx)

        global_index += 1

        if (local_idx + 1) % 300 == 0:
            print(f"  {local_idx+1}/{n_files} converted (global index {global_index-1})")

    print(f"  RUN{run_idx:02d} done — {n_files} snapshots, "
          f"global indices {global_index-n_files} to {global_index-1}")

# -----------------------------------------------------------------------
# Save metadata
# -----------------------------------------------------------------------
np.save(os.path.join(output_dir, 'phase_metadata.npy'), phase_map)
np.save(os.path.join(output_dir, 'run_metadata.npy'),   run_map)

print(f"\n{'='*60}")
print(f"Conversion complete.")
print(f"Total snapshots : {global_index}")
print(f"Output folder   : {output_dir}")
print(f"Phase period    : {T_period} snapshots")
print(f"\nRun lengths:")
for r, n in run_lengths.items():
    print(f"  RUN{r:02d}: {n} snapshots")

# -----------------------------------------------------------------------
# Quick sanity check on first file
# -----------------------------------------------------------------------
print(f"\nSanity check on piv.0.h5.uvw:")
with h5py.File(os.path.join(output_dir, 'piv.0.h5.uvw'), 'r') as f:
    print(f"  Keys : {list(f.keys())}")
    print(f"  u shape : {f['u'].shape}")
    print(f"  mx      : {f['mx'][0]}")
    print(f"  my      : {f['my'][0]}")
    print(f"  u max   : {np.nanmax(f['u'][:]):.4f} m/s")
print(f"  Phase of snapshot 0 : {phase_map[0]}")
print(f"  Run/local of snapshot 0 : {run_map[0]}")