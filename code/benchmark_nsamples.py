# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
benchmark_nsamples.py
-------------------------------------------------------------------------------------------------------------------------
Standalone script to benchmark the SHAP GradientExplainer on a SINGLE snapshot
across a range of `nsamples` values, to decide the production value of nsamples
for the full campaign.

Usage (from the `code/` directory):
    python benchmark_nsamples.py

What it does:
    1. Loads the same configuration as `main_SHAP.py` (channel data, folders,
       training data, shap data).
    2. Instantiates `shap_config` ONCE — the U-Net is loaded onto the GPU once
       and kept in memory for the whole benchmark.
    3. Computes the background tensor once.
    4. Reads ONE (input, output) velocity pair via `read_norm_velocity`.
    5. For each value of nsamples in BENCH_NSAMPLES, runs the SHAP calculation
       on that same snapshot and measures wall-clock time.
    6. Computes the relative L2 error of each estimate against the reference
       (the highest nsamples in the list).
    7. Prints a summary table to stdout and saves it to
       `../results/sta/benchmark_nsamples.txt`.

Output: one row per nsamples with columns
    nsamples | time (s) | rel. L2 err (%) | total 2000 snaps (h)
"""

# ---------------------------------------------------------------------------
# What to benchmark — edit these two constants as needed
# ---------------------------------------------------------------------------
BENCH_INDEX    = 500                                  # snapshot index to use for the study
BENCH_NSAMPLES = [50, 100, 200, 500, 1000, 2000]      # values to test; the LAST one is the reference
N_SNAPS_TOTAL  = 2000                                 # used only for the "total estimated time" column

# ---------------------------------------------------------------------------
# Config file names — same convention as main_SHAP.py
# ---------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
tr_data_str = "training_data"
sh_data_str = "shap_data"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import time
import numpy as np

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import py_bin.py_class.shap_config as sc
from py_bin.py_functions.read_norm_velocity import read_norm_velocity

exec("from " + folder_def + " import " + chd_str + " as chd")
exec("from " + folder_def + " import " + folders_str + " as folders")
exec("from " + folder_def + " import " + tr_data_str + " as tr_data")
exec("from " + folder_def + " import " + sh_data_str + " as sh_data")

# ---------------------------------------------------------------------------
# Gather parameters (copied from main_SHAP.py — keep in sync)
# ---------------------------------------------------------------------------
L_x     = chd.L_x
L_y     = chd.L_y
rey     = chd.rey
utau    = chd.utau
dx      = chd.dx
dy      = chd.dy
padding = chd.padding

shap_folder     = folders.shap_folder
shap_file       = folders.shap_file
uvw_folder      = folders.uvw_folder
uvw_file        = folders.uvw_file
data_folder     = folders.data_folder
umean_file      = folders.umean_file
unorm_file      = folders.unorm_file

ngpu            = tr_data.ngpu
field_ini       = sh_data.field_ini
field_fin       = sh_data.field_fin
field_delta     = sh_data.field_delta
model_folder    = folders.model_folder
model_read      = folders.model_read
nfil            = tr_data.nfil
stride          = tr_data.stride
activation      = tr_data.activation
kernel          = tr_data.kernel
pooling         = tr_data.pooling
delta_pred      = tr_data.delta_pred
data_type       = tr_data.data_type
error_file      = folders.error_file
umax_file       = folders.umax_file
urmspred_file   = folders.urmspred_file
mean_norm       = tr_data.mean_norm
tfrecord_folder = folders.tfrecord_folder
nrep_field      = sh_data.nrep_field
shap_batch      = sh_data.shap_batch
repeat_exist    = sh_data.repeat_exist

# ---------------------------------------------------------------------------
# Instantiate shap_config ONCE — the U-Net is loaded once and stays on GPU
# ---------------------------------------------------------------------------
# nsamples / nsamples_max in the init dict are overridden inside the loop below.
# We set them to the largest benchmark value so the explainer's internal
# buffers are sized appropriately from the start.
nsamples_init     = max(BENCH_NSAMPLES)
nsamples_max_init = nsamples_init

data_shap = {"shap_folder":shap_folder,"shap_file":shap_file,"uvw_folder":uvw_folder,"uvw_file":uvw_file,
             "padding":padding,"dx":dx,"dy":dy,"data_folder":data_folder,"umean_file":umean_file,
             "unorm_file":unorm_file,"L_x":L_x,"L_y":L_y,"rey":rey,"utau":utau,"ngpu":ngpu,
             "field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"model_folder":model_folder,
             "model_read":model_read,"nfil":nfil,"stride":stride,"activation":activation,"kernel":kernel,
             "pooling":pooling,"delta_pred":delta_pred,"nsamples":nsamples_init,"nsamples_max":nsamples_max_init,
             "data_type":data_type,"error_file":error_file,"umax_file":umax_file,"urmspred_file":urmspred_file,
             "mean_norm":mean_norm,"tfrecord_folder":tfrecord_folder,"nrep_field":nrep_field,"shap_batch":shap_batch,
             "repeat_exist":repeat_exist,"flag_model":True}

print("=" * 80)
print(" Loading U-Net and building SHAP model (this happens ONCE)...")
print("=" * 80, flush=True)
t0 = time.time()
shap_model = sc.shap_config(data_in=data_shap)
print(" Model ready in {:.1f} s".format(time.time() - t0), flush=True)

# ---------------------------------------------------------------------------
# Pre-compute the background tensor once (used by all nsamples runs)
# ---------------------------------------------------------------------------
print("\n Building background tensor...", flush=True)
shap_model.background()

# ---------------------------------------------------------------------------
# Read the input velocity and the target (delta_pred ahead) velocity for the
# benchmarking snapshot. Stored in memory, reused for every nsamples value.
# ---------------------------------------------------------------------------
print("\n Reading benchmark snapshot index = {}...".format(BENCH_INDEX), flush=True)

data_norm_in = {"folder":shap_model.uvw_folder, "file":shap_model.uvw_file, "padding":0,
                "shpx":shap_model.shpx, "shpy":shap_model.shpy,
                "dx":shap_model.dx, "dy":shap_model.dy,
                "data_folder":shap_model.data_folder, "umean_file":shap_model.umean_file,
                "unorm_file":shap_model.unorm_file, "index":BENCH_INDEX,
                "data_type":shap_model.data_type, "mean_norm":shap_model.mean_norm}
norm_velocity_in = read_norm_velocity(data_in=data_norm_in)["norm_velocity"]

data_norm_out = dict(data_norm_in)
data_norm_out["index"] = BENCH_INDEX + shap_model.delta_pred
norm_velocity_out = read_norm_velocity(data_in=data_norm_out)["norm_velocity"]

# ---------------------------------------------------------------------------
# Main benchmark loop — hot-swap self.nsamples and time each call
# ---------------------------------------------------------------------------
results       = []      # list of (n, wall_time_s, shap_u, shap_v)
reference_idx = len(BENCH_NSAMPLES) - 1   # the LAST value is the reference

print("\n" + "=" * 80)
print(" Benchmarking SHAP convergence (single snapshot, same input reused)")
print("=" * 80)
print(" Config: shap_batch={}, padding={}, shpy={}, shpx={}".format(
    shap_model.shap_batch, shap_model.padding, shap_model.shpy, shap_model.shpx))
print(" nsamples values: {}".format(BENCH_NSAMPLES))
print()

for n in BENCH_NSAMPLES:
    shap_model.nsamples     = int(n)
    shap_model.nsamples_max = int(n)

    print(" >> nsamples = {:5d} ... ".format(n), end="", flush=True)
    t0 = time.time()
    shap_out = shap_model._calculate_gradientshaps(
        data_in={"norm_velocity_in":  norm_velocity_in,
                 "norm_velocity_out": norm_velocity_out,
                 "x0": 0, "z0": 0}
    )
    t_elapsed = time.time() - t0

    shap_u = np.asarray(shap_out["shap_u"], dtype="float64")
    shap_v = np.asarray(shap_out["shap_v"], dtype="float64")
    results.append((n, t_elapsed, shap_u, shap_v))

    print("done in {:8.2f} s".format(t_elapsed), flush=True)

# ---------------------------------------------------------------------------
# Compute relative L2 errors against the reference (max nsamples)
# ---------------------------------------------------------------------------
ref_n, ref_t, ref_u, ref_v = results[reference_idx]
ref_norm = np.sqrt(np.linalg.norm(ref_u) ** 2 + np.linalg.norm(ref_v) ** 2)
if ref_norm == 0.0:
    ref_norm = 1.0   # avoid division by zero in pathological cases

# ---------------------------------------------------------------------------
# Pretty-print the summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(" Results")
print("=" * 80)
header = "{:>10} | {:>10} | {:>14} | {:>18}".format(
    "nsamples", "time (s)", "rel L2 err (%)", "est 2000 snaps (h)")
print(header)
print("-" * len(header))

summary_lines = [header, "-" * len(header)]

for (n, t_elapsed, shap_u, shap_v) in results:
    diff_norm = np.sqrt(np.linalg.norm(shap_u - ref_u) ** 2 + np.linalg.norm(shap_v - ref_v) ** 2)
    rel_err   = 100.0 * diff_norm / ref_norm
    t_tot_h   = t_elapsed * N_SNAPS_TOTAL / 3600.0

    tag = " (REF)" if n == ref_n else ""
    row = "{:>10d} | {:>10.2f} | {:>13.2f}%{:>6s} | {:>18.2f}".format(
        n, t_elapsed, rel_err, tag, t_tot_h)
    print(row)
    summary_lines.append(row)

print("=" * 80)

# ---------------------------------------------------------------------------
# Persist the table to disk next to the other stats files
# ---------------------------------------------------------------------------
try:
    os.makedirs(data_folder, exist_ok=True)
    out_path = os.path.join(data_folder, "benchmark_nsamples.txt")
    with open(out_path, "w") as fh:
        fh.write("Benchmark for snapshot index {} against reference nsamples = {}\n".format(
            BENCH_INDEX, ref_n))
        fh.write("shap_batch = {}\n".format(shap_model.shap_batch))
        fh.write("shpy = {}, shpx = {}, padding = {}\n\n".format(
            shap_model.shpy, shap_model.shpx, shap_model.padding))
        fh.write("\n".join(summary_lines))
        fh.write("\n")
    print("\n Table saved to {}".format(out_path))
except Exception as exc:
    print("\n Warning: could not write summary file ({})".format(exc))

print("\n Done.")
