#----------------------------------------------------------------------------------------------------------------------
# %% Imports
#----------------------------------------------------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.io import load_velocity, PIV_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

y_min, y_max = 20, 180
transitions_idx = [0, 1483, 2968, 4453, 5938, 7422, 8906]
meta = np.load(PIV_DIR / "phase_metadata.npy", allow_pickle=True).item()

P = max(meta.values()) + 1
SHPY: int = 199
SHPX: int = 319

# --- Calibration
# Raw camera calibration is 12.669 px/mm (pivtools config). Velocity arrays
# live on the PIV vector grid — last pass has 16x16 windows with 50% overlap,
# so 1 vector = 8 raw pixels. Effective grid spacing ≈ 0.6315 mm.
PX_TO_MM = 8.0 / 12.669                   # ≈ 0.6315 mm per grid point
Y_LABEL  = r"$y$ (mm)"

# --- Output folder for all figures (shared with figures)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Exclusion of RUN6
skip_start, skip_stop = transitions_idx[5], transitions_idx[6]
indices = [idx for idx in meta if not (skip_start <= idx < skip_stop)]

#----------------------------------------------------------------------------------------------------------------------
# %% Loading Velocity over all indices
#-----------------------------------------------------------------------------------------------------
cache = Path(__file__).parent / 'phase_mean_cache.npz'

if cache.exists(): 
    data = np.load(cache)
    u_phase = data['u_phase']
    v_phase = data['v_phase']
    counts = data['counts']
else: 
    u_sum = np.zeros((P, SHPY, SHPX), dtype=np.float64)
    v_sum = np.zeros((P, SHPY, SHPX), dtype=np.float64) 
    counts= np.zeros(P, dtype=np.int64)
    for i, idx in enumerate(indices):
        u,v = load_velocity(idx)
        p = meta[idx]
        u_sum[p]+=u
        v_sum[p]+=v
        counts[p]+=1
        if i % 500 == 0: print(f"{i}/{len(indices)}")
    non_zero= counts > 0
    u_phase = np.zeros_like(u_sum)
    v_phase = np.zeros_like(v_sum)
    u_phase[non_zero] = u_sum[non_zero]/counts[non_zero, None, None]
    v_phase[non_zero] = v_sum[non_zero]/counts[non_zero, None, None]
    np.savez(cache, u_phase=u_phase, v_phase=v_phase, counts=counts)

#----------------------------------------------------------------------------------------------------------------------
# %% JPDF P(v', y)
#-----------------------------------------------------------------------------------------------------

# Anchors chosen on well-sampled phases (SHAP campaign sampled unevenly;
# phases {2, 7, 12, 17, 22, 27} each have ~150 SHAP snapshots).
anchors = [2, 7, 17, 27]
phase_to_bin = {a: i for i, a in enumerate(anchors)}
noise_filter = 0.005
N_v, N_y = 60, 40
v_max = 0.3

v_bin = np.linspace(-v_max, v_max, N_v + 1)
y_bin = np.linspace(y_min, y_max, N_y + 1)
H = np.zeros((4, N_v, N_y), dtype=np.int64)

y_grid = np.broadcast_to(np.arange(y_min, y_max)[:, None],
                         (y_max - y_min, SHPX))
for i, idx in enumerate(indices):
    if i % 500 == 0: print(f"{i}/{len(indices)}")
    p = meta[idx]
    if p not in phase_to_bin: continue
    b = phase_to_bin[p]
    u, v = load_velocity(idx)
    v_fluc = v - v_phase[p]
    v_fluc=v_fluc[y_min:y_max] 
    vel_mag = np.sqrt(u**2 + v**2)
    valid = np.ones_like(vel_mag[y_min:y_max], dtype=bool)
    #valid = vel_mag[y_min:y_max] > noise_filter
    v_flat = v_fluc[valid]
    y_flat = y_grid[valid]
    H_idx, _, _ = np.histogram2d(v_flat, y_flat, bins=[v_bin, y_bin])
    H[b] += H_idx.astype(np.int64)

v_centers = 0.5*(v_bin[:-1] + v_bin[1:])
y_centers = 0.5*(y_bin[:-1] + y_bin[1:])
counts_per_bin = [sum(1 for idx in indices if meta.get(idx) == a) for a in anchors]

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True, sharex=True)
dv = v_bin[1] - v_bin[0]
dy = y_bin[1] - y_bin[0]
axes = axes.flatten()   # pour pouvoir faire enumerate(axes) sans souci
titles = ["Rising", "Peak", "Decaying", "Quiescent"]

for b, ax in enumerate(axes):
    JPDF = H[b] / (H[b].sum() * dv * dy)
    JPDF /= JPDF.max()
    cs = ax.contourf(v_centers, y_centers * PX_TO_MM, JPDF.T,
             levels=np.logspace(-3, 0, 7),
             norm=LogNorm(1e-3, 1), cmap='viridis')
    ax.set_title(f"{titles[b]} (φ={anchors[b]}, n={counts_per_bin[b]})")
axes[0].invert_yaxis()
axes[0].set_ylabel(Y_LABEL)
axes[2].set_ylabel(Y_LABEL)
axes[2].set_xlabel(r"$v'$ (m/s)")
axes[3].set_xlabel(r"$v'$ (m/s)")
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
fig.colorbar(cs, cax=cbar_ax, label=r"$p(v', y\,|\,\varphi) / \max$")
fig.subplots_adjust(right=0.9)
fig.suptitle(rf"JPDF of $v'$ and $y$ per phase — {sum(counts_per_bin)} snapshots")
plt.savefig(FIG_DIR / 'JPDF_v_prime_y.png', dpi=300)
plt.show()

# %% JPDF P(u', y)

anchors = [2, 7, 17, 27]   # same well-sampled phases as above
phase_to_bin = {a: i for i, a in enumerate(anchors)}
noise_filter = 0.005
N_v, N_y = 60, 40
N_u = 60
u_max = 0.3
u_bin = np.linspace(-u_max, u_max, N_u + 1)
y_bin = np.linspace(y_min, y_max, N_y + 1)
H = np.zeros((4, N_u, N_y),  dtype=np.int64)

y_grid = np.broadcast_to(np.arange(y_min, y_max)[:, None],
                         (y_max - y_min, SHPX))
for i, idx in enumerate(indices):
    if i % 500 == 0: print(f"{i}/{len(indices)}")
    p = meta[idx]
    if p not in phase_to_bin: continue
    b = phase_to_bin[p]
    u, v = load_velocity(idx)
    v_fluc = v - v_phase[p]
    u_fluc = u - u_phase[p]
    u_fluc = u_fluc[y_min:y_max]
    v_fluc=v_fluc[y_min:y_max] 
    vel_mag = np.sqrt(u**2 + v**2)
    valid = np.ones_like(vel_mag[y_min:y_max], dtype=bool)
    #valid = vel_mag[y_min:y_max] > noise_filter
    v_flat = v_fluc[valid]
    u_flat = u_fluc[valid]
    y_flat = y_grid[valid]
    H_idx, _, _ = np.histogram2d(u_flat, y_flat, bins=[u_bin, y_bin])
    H[b] += H_idx.astype(np.int64)
u_centers = 0.5*(u_bin[:-1] + u_bin[1:])
y_centers = 0.5*(y_bin[:-1] + y_bin[1:])
counts_per_bin = [sum(1 for idx in indices if meta.get(idx) == a) for a in anchors]

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True, sharex=True)
axes = axes.flatten()   # pour pouvoir faire enumerate(axes) sans souci
du = u_bin[1] - u_bin[0]
dy = y_bin[1] - y_bin[0]
titles = ["Rising", "Peak", "Decaying", "Quiescent"]

for b, ax in enumerate(axes):
    JPDF = H[b] / (H[b].sum() * du * dy)
    JPDF /= JPDF.max()
    cs = ax.contourf(u_centers, y_centers * PX_TO_MM, JPDF.T,
             levels=np.logspace(-3, 0, 7),
             norm=LogNorm(1e-3, 1), cmap='viridis')
    ax.set_title(f"{titles[b]} (φ={anchors[b]}, n={counts_per_bin[b]})")
axes[0].invert_yaxis()
axes[0].set_ylabel(Y_LABEL)
axes[2].set_ylabel(Y_LABEL)
axes[2].set_xlabel(r"$u'$ (m/s)")
axes[3].set_xlabel(r"$u'$ (m/s)")
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
fig.colorbar(cs, cax=cbar_ax, label=r"$p(u', y\,|\,\varphi) / \max$")
fig.subplots_adjust(right=0.9)
fig.suptitle(rf"JPDF of $u'$ and $y$ per phase — {sum(counts_per_bin)} snapshots")
plt.savefig(FIG_DIR / 'JPDF_u_prime_y.png', dpi=300)
plt.show()

# %%
# Après avoir calculé la courbe phase-moyennée globale (global_mean sur t_fine)
# Signal phase-moyenné dans ta bande d'intérêt
v_signal = v_phase[:, y_min:y_max, 130:270].mean(axis=(1, 2))
P = len(v_signal)  # 30
t = np.arange(P) / 15.0

plt.figure()
plt.plot(t, v_signal, 'k-', lw=2)
for label, a in zip(['Rising', 'Peak', 'Decaying', 'Quiescent'], anchors):
    plt.axvline(a / 15.0, ls='--', label=f"{label} (φ={a})")
plt.xlabel('t (s)'); plt.ylabel(r'$\langle v \rangle$ (m/s)')
plt.legend()
plt.savefig(FIG_DIR / 'phase_anchors_on_signal.png', dpi=300)
plt.show()

# %% JPDF per phase with structures

from figures._structures import (
    compute_rms_cache,
    compute_gradients,
    segment_shear,
    segment_vorticity,
    segment_entrainment,
    segment_q2d,
    segment_shap,
    load_shap,
    _shap_available_indices,
)

rms = compute_rms_cache()
shap_idx_set = set(_shap_available_indices())
print(f"{len(shap_idx_set)} snapshots avec SHAP")

STRUCTURES = ['shear', 'vorticity', 'entrainment', 'q2d', 'shap']
N_struct = len(STRUCTURES)

# --- Test 1 snapshot (sanity)
idx_test = indices[500]
u, v = load_velocity(idx_test)
grads = compute_gradients(u, v)
v_fluc_test = v - v_phase[meta[idx_test]]

mk_shear = segment_shear(grads, rms)
mk_vort  = segment_vorticity(grads, rms)
mk_ent   = segment_entrainment(v_fluc_test, rms)
mk_q2d   = segment_q2d(grads, rms)
print("shapes:", mk_shear.shape, mk_vort.shape, mk_ent.shape, mk_q2d.shape)
print("fractions:", [m.mean() for m in [mk_shear, mk_vort, mk_ent, mk_q2d]])

# --- Cache
cache = Path(__file__).parent / 'jpdf_phase_struct_shap_cache.npz'

if cache.exists():
    data = np.load(cache)
    H = data['H']
    v_bin = data['v_bin']
    y_bin = data['y_bin']
    n_per_cell = data['n_per_cell']
else:
    N_v, N_y = 60, 40
    v_max = 0.3
    v_bin = np.linspace(-v_max, v_max, N_v + 1)
    y_bin = np.linspace(y_min, y_max, N_y + 1)
    H = np.zeros((4, N_struct, N_v, N_y), dtype=np.int64)
    n_per_cell = np.zeros((4, N_struct), dtype=np.int64)

    y_grid = np.broadcast_to(np.arange(y_min, y_max)[:, None],
                             (y_max - y_min, SHPX))

    for i, idx in enumerate(indices):
        if i % 200 == 0: print(f"{i}/{len(indices)}")
        p = meta[idx]
        if p not in phase_to_bin: continue
        b = phase_to_bin[p]

        u, v = load_velocity(idx)
        v_fluc = v - v_phase[p]
        grads = compute_gradients(u, v)

        masks = {
            'shear':       segment_shear(grads, rms),
            'vorticity':   segment_vorticity(grads, rms),
            'entrainment': segment_entrainment(v_fluc, rms),
            'q2d':         segment_q2d(grads, rms),
        }

        if idx in shap_idx_set:
            try:
                su, sv = load_shap(idx)
                masks['shap'] = segment_shap(su, sv, rms)
            except (OSError, KeyError, FileNotFoundError):
                pass

        v_fluc_band = v_fluc[y_min:y_max]
        for s, name in enumerate(STRUCTURES):
            if name not in masks: continue
            mk = masks[name][y_min:y_max]
            if not mk.any(): continue
            v_flat = v_fluc_band[mk]
            y_flat = y_grid[mk]
            H_idx, _, _ = np.histogram2d(v_flat, y_flat, bins=[v_bin, y_bin])
            H[b, s] += H_idx.astype(np.int64)
            n_per_cell[b, s] += 1

    np.savez(cache, H=H, v_bin=v_bin, y_bin=y_bin, n_per_cell=n_per_cell)

print("H.sum(axis=(2,3)):\n", H.sum(axis=(2, 3)))
print("n_per_cell:\n", n_per_cell)

# --- PLOT 4x5
fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True, sharey=True)
phase_labels = ["Rising", "Peak", "Decaying", "Quiescent"]
struct_labels = ["Shear", "Vorticity", "Entrainment", "Q2D", "SHAP"]

v_centers = 0.5 * (v_bin[:-1] + v_bin[1:])
y_centers = 0.5 * (y_bin[:-1] + y_bin[1:])
dv = v_bin[1] - v_bin[0]
dy = y_bin[1] - y_bin[0]

for b in range(4):
    for s in range(N_struct):
        ax = axes[b, s]
        if H[b, s].sum() == 0: continue
        pdf = H[b, s] / (H[b, s].sum() * dv * dy)
        pdf /= pdf.max()
        cs = ax.contourf(v_centers, y_centers * PX_TO_MM, pdf.T,
                         levels=np.logspace(-3, 0, 7),
                         norm=LogNorm(1e-3, 1), cmap='viridis')
        if b == 0: ax.set_title(f"{struct_labels[s]} (n={n_per_cell[b, s]})")
        if s == 0: ax.set_ylabel(f"{phase_labels[b]}\n{Y_LABEL}")
        if b == 3: ax.set_xlabel(r"$v'$ (m/s)")

axes[0, 0].invert_yaxis()
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
fig.colorbar(cs, cax=cbar_ax, label=r"$p(v', y | \varphi, S)/\max$")
fig.subplots_adjust(right=0.9)
fig.suptitle(r"JPDF of $v'$ and $y$ per phase and structure (incl. SHAP)")
plt.savefig(FIG_DIR / 'JPDF_phase_structure.png', dpi=300)
plt.show()

# %% Coincidence SHAP ∩ structure per phase (phase-resolved fig_c)

STRUCT_4 = ['shear', 'vorticity', 'entrainment', 'q2d']
STRUCT_4_LABELS = ['Shear', 'Vorticity', 'Entrainment', r'$Q_{2D}$']

cache = Path(__file__).parent / 'coinc_phase_cache.npz'

if cache.exists():
    data = np.load(cache)
    count_shap   = data['count_shap']
    count_struct = data['count_struct']
    count_inter  = data['count_inter']
    n_snaps      = data['n_snaps']
else:
    count_shap   = np.zeros((4, SHPY),    dtype=np.int64)
    count_struct = np.zeros((4, 4, SHPY), dtype=np.int64)
    count_inter  = np.zeros((4, 4, SHPY), dtype=np.int64)
    n_snaps      = np.zeros(4,            dtype=np.int64)

    for i, idx in enumerate(indices):
        if i % 200 == 0: print(f"{i}/{len(indices)}")
        p = meta[idx]
        if p not in phase_to_bin: continue
        if idx not in shap_idx_set: continue
        b = phase_to_bin[p]

        try:
            u, v   = load_velocity(idx)
            su, sv = load_shap(idx)
        except (OSError, KeyError, FileNotFoundError):
            continue

        v_fluc  = v - v_phase[p]
        grads   = compute_gradients(u, v)
        mk_shap = segment_shap(su, sv, rms)

        masks_4 = {
            'shear':       segment_shear(grads, rms),
            'vorticity':   segment_vorticity(grads, rms),
            'entrainment': segment_entrainment(v_fluc, rms),
            'q2d':         segment_q2d(grads, rms),
        }

        count_shap[b] += mk_shap.sum(axis=1)
        for s, name in enumerate(STRUCT_4):
            mk = masks_4[name]
            count_struct[b, s] += mk.sum(axis=1)
            count_inter[b, s]  += (mk & mk_shap).sum(axis=1)
        n_snaps[b] += 1

    np.savez(cache,
             count_shap=count_shap,
             count_struct=count_struct,
             count_inter=count_inter,
             n_snaps=n_snaps)

print("n_snaps per phase:", n_snaps)

y_range = slice(y_min, y_max)
y_axis  = np.arange(y_min, y_max)
phase_labels = ["Rising", "Peak", "Decaying", "Quiescent"]
colors = plt.cm.tab10(np.linspace(0, 1, 4))

# %% Structure composition of SHAP per phase (smoothed, NaN-safe)
from scipy.ndimage import uniform_filter1d

MIN_SHAP = 5

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

y_axis_mm = y_axis * PX_TO_MM

for b, ax in enumerate(axes):
    cs = count_shap[b, y_range].astype(np.float64)
    mask_valid = cs > MIN_SHAP

    ax.fill_between(y_axis_mm, 0, 1, where=~mask_valid,
                    color='lightgray', alpha=0.4, step='mid',
                    label='insufficient SHAP' if b == 0 else None)

    for s, name in enumerate(STRUCT_4):
        ci = count_inter[b, s, y_range].astype(np.float64)
        p_raw = np.where(cs > 0, ci / cs, 0)
        p_smooth = uniform_filter1d(p_raw, size=9, mode='nearest')
        p_display = np.where(mask_valid, p_smooth, np.nan)
        ax.plot(y_axis_mm, p_display, color=colors[s], lw=2, label=STRUCT_4_LABELS[s])

    ax.set_title(f"{phase_labels[b]} (n={n_snaps[b]})")
    ax.set_xlim(y_min * PX_TO_MM, y_max * PX_TO_MM)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if b == 0: ax.legend(fontsize=9)

for ax in axes[2:]: ax.set_xlabel(Y_LABEL)
for ax in [axes[0], axes[2]]: ax.set_ylabel(r"$(X \cap \mathrm{SHAP}) / \mathrm{SHAP}$")

fig.suptitle(r"Structure composition of SHAP per phase")
fig.tight_layout()
plt.savefig(FIG_DIR / 'shap_composition_per_phase.png', dpi=300)
plt.show()

# %% Coincidence bar chart per phase (Cremades-style scalar summary)

phase_labels_bar = ["Rising", "Peak", "Decaying", "Quiescent"]
colors_struct = ['#1f77b4', '#d62728', '#e377c2', '#17becf']  # shear, vort, entr, q2d

# Sum counts over the valid y band for each (phase, structure)
shap_tot   = count_shap[:, y_range].sum(axis=1).astype(np.float64)           # (4,)
struct_tot = count_struct[:, :, y_range].sum(axis=2).astype(np.float64)      # (4, 4)
inter_tot  = count_inter[:,  :, y_range].sum(axis=2).astype(np.float64)      # (4, 4)

# Two metrics per (phase, structure)
with np.errstate(divide='ignore', invalid='ignore'):
    frac_in_shap = np.where(shap_tot[:, None] > 0,
                            inter_tot / shap_tot[:, None], 0.0)   # (X ∩ SHAP) / SHAP
    frac_in_X    = np.where(struct_tot > 0,
                            inter_tot / struct_tot, 0.0)          # (X ∩ SHAP) / X

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
x_pos = np.arange(len(STRUCT_4))
bw = 0.35

for b, ax in enumerate(axes):
    bars1 = ax.bar(x_pos - bw/2, 100 * frac_in_shap[b], bw,
                   color=colors_struct, edgecolor='black', linewidth=0.6,
                   label=r'$(X \cap \mathrm{SHAP}) / \mathrm{SHAP}$')
    bars2 = ax.bar(x_pos + bw/2, 100 * frac_in_X[b], bw,
                   color=colors_struct, edgecolor='black', linewidth=0.6,
                   hatch='///', alpha=0.7,
                   label=r'$(X \cap \mathrm{SHAP}) / X$')

    for bar, val in zip(bars1, 100 * frac_in_shap[b]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=7.5)
    for bar, val in zip(bars2, 100 * frac_in_X[b]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(STRUCT_4_LABELS)
    ax.set_title(f"{phase_labels_bar[b]}  (n={n_snaps[b]})")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    if b == 0:
        ax.set_ylabel("Coincidence (%)")
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

fig.suptitle("Coincidence between SHAP and classical structures, per phase")
fig.tight_layout()
plt.savefig(FIG_DIR / 'shap_coincidence_bars_per_phase.png', dpi=300)
plt.show()

# %% Instantaneous coincidence bar chart — one representative snapshot per phase

def _instant_coincidence(idx: int):
    u, v = load_velocity(idx)
    su, sv = load_shap(idx)
    p = meta[idx]
    v_fluc = v - v_phase[p]
    grads = compute_gradients(u, v)

    mk_shap = segment_shap(su, sv, rms)[y_min:y_max]
    masks = {
        'shear':       segment_shear(grads, rms)[y_min:y_max],
        'vorticity':   segment_vorticity(grads, rms)[y_min:y_max],
        'entrainment': segment_entrainment(v_fluc, rms)[y_min:y_max],
        'q2d':         segment_q2d(grads, rms)[y_min:y_max],
    }
    n_shap = int(mk_shap.sum())
    out = {}
    for name, mk in masks.items():
        n_X = int(mk.sum())
        n_inter = int((mk & mk_shap).sum())
        out[name] = {
            'frac_in_shap': n_inter / n_shap if n_shap > 0 else 0.0,
            'frac_in_X':    n_inter / n_X    if n_X    > 0 else 0.0,
            'n_X': n_X, 'n_inter': n_inter,
        }
    return out, n_shap

# Pick the snapshot of each anchor phase with the highest |SHAP| activity
def _pick_representative(anchor_phase: int) -> int:
    candidates = [i for i in indices if meta.get(i) == anchor_phase and i in shap_idx_set]
    best_idx, best_score = None, -1.0
    for i in candidates:
        try:
            su, sv = load_shap(i)
        except (OSError, KeyError, FileNotFoundError):
            continue
        score = float(np.sqrt(su**2 + sv**2).mean())
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

repr_idx = [_pick_representative(a) for a in anchors]
print("Representative snapshots per phase:", dict(zip(phase_labels_bar, repr_idx)))

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)

for b, (ax, idx) in enumerate(zip(axes, repr_idx)):
    if idx is None:
        ax.set_title(f"{phase_labels_bar[b]}  (no SHAP snapshot)")
        continue
    stats, n_shap = _instant_coincidence(idx)
    frac_in_shap_i = np.array([stats[n]['frac_in_shap'] for n in STRUCT_4])
    frac_in_X_i    = np.array([stats[n]['frac_in_X']    for n in STRUCT_4])

    bars1 = ax.bar(x_pos - bw/2, 100 * frac_in_shap_i, bw,
                   color=colors_struct, edgecolor='black', linewidth=0.6,
                   label=r'$(X \cap \mathrm{SHAP}) / \mathrm{SHAP}$')
    bars2 = ax.bar(x_pos + bw/2, 100 * frac_in_X_i, bw,
                   color=colors_struct, edgecolor='black', linewidth=0.6,
                   hatch='///', alpha=0.7,
                   label=r'$(X \cap \mathrm{SHAP}) / X$')

    for bar, val in zip(bars1, 100 * frac_in_shap_i):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=7.5)
    for bar, val in zip(bars2, 100 * frac_in_X_i):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(STRUCT_4_LABELS)
    ax.set_title(f"{phase_labels_bar[b]}  (t={idx}, φ={meta[idx]})")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    if b == 0:
        ax.set_ylabel("Coincidence (%)")
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

fig.suptitle("Coincidence between SHAP and classical structures — "
             "representative instantaneous snapshots")
fig.tight_layout()
plt.savefig(FIG_DIR / 'shap_coincidence_bars_instantaneous.png', dpi=300)
plt.show()

# %%
