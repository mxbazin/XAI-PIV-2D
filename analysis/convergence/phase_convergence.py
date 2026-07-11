#----------------------------------------------------------------------------------------------------------------------
# %% Imports
#----------------------------------------------------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.io import load_velocity
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

y_min, y_max = 20, 180
meta = np.load(r'e:\shap\XAI_PIV_2D_simplified-main\data\piv\phase_metadata.npy', allow_pickle=True).item()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------------------------------------------------------------------------
# %% Velocity over all indices
#----------------------------------------------------------------------------------------------------------------------
cache = Path(__file__).parent / 'scalar_cache_all_band.npy'

if cache.exists():
    all_velocity = np.load(cache)
else:
    all_idx = range(max(meta.keys()) + 1)
    all_velocity = np.zeros(len(all_idx))
    for i, idx in enumerate(all_idx):
        u, v = load_velocity(idx)
        all_velocity[i] = np.mean(v[y_min:y_max, 130:270])
        if i % 1000 == 0: print(f"{i}/{len(all_idx)}")
    np.save(cache, all_velocity)

#----------------------------------------------------------------------------------------------------------------------
# %% Slicing per run 
#----------------------------------------------------------------------------------------------------------------------
transitions_idx = [0, 1483, 2968, 4453, 5938, 7422, 8906]
end = len(all_velocity)
n_per_cycle = 30
t_cycle = np.arange(n_per_cycle) / 15.0

runs = []
for i in range(len(transitions_idx)):
    start = transitions_idx[i]
    stop = transitions_idx[i+1] if i+1 < len(transitions_idx) else end
    first_p0 = next(idx for idx in range(start, stop-1)
                    if meta[idx] == 0 and meta[idx+1] == 1)
    signal = all_velocity[first_p0:stop]
    n_cycles = len(signal) // 30
    cycles = signal[:n_cycles * 30].reshape(n_cycles, 30)
    runs.append(cycles)

target_peak = np.argmax(runs[0].mean(axis=0))
runs_aligned = []
for cycles in runs:
    shift = target_peak - np.argmax(cycles.mean(axis=0))
    runs_aligned.append(np.roll(cycles, shift, axis=1))

#----------------------------------------------------------------------------------------------------------------------
# %% Plot
#----------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
axes = axes.flatten()
t_fine = np.linspace(t_cycle[0], t_cycle[-1], 200)

for r, cycles in enumerate(runs_aligned):
    ax = axes[r]
    mean_i = interp1d(t_cycle, cycles.mean(axis=0), kind='cubic')(t_fine)
    std_i = interp1d(t_cycle, cycles.std(axis=0), kind='cubic')(t_fine)
    ax.fill_between(t_fine, mean_i - std_i, mean_i + std_i,
                    color='steelblue', alpha=0.2, label=r'$\pm 1\sigma$')
    ax.plot(t_fine, mean_i, color='black', lw=2, label='Mean')
    ax.set_title(f'Run {r+1} — {cycles.shape[0]} cycles', fontsize=11)
    if r % 4 == 0:
        ax.set_ylabel(r'$\langle v \rangle$ (m/s)', fontsize=11)
    if r >= 3:
        ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.tick_params(labelsize=10)

axes[0].legend(fontsize=9, loc='best')
axes[7].set_visible(False)
fig.suptitle('Phase-averaged velocity per run', fontsize=13)
fig.tight_layout()
plt.savefig(FIG_DIR / "phase_averaged_velocity.png", dpi=300, bbox_inches='tight')
plt.show()

#----------------------------------------------------------------------------------------------------------------------
# %% Convergence plot 
#----------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8,5))
t_fine = np.linspace(t_cycle[0], t_cycle[-1],200)

colors = plt.cm.tab10(np.linspace(0,1,7))
for r, cycles in enumerate(runs_aligned):
    if r ==5:
        continue 
    mean_i = interp1d(t_cycle, cycles.mean(axis=0), kind='cubic')(t_fine)
    ax.plot(t_fine, mean_i, lw=2, color=colors[r], label=f'Run {r+1}', alpha=0.25)

all_means = []
for r, cycles in enumerate(runs_aligned):
    if r == 5:
        continue
    all_means.append(interp1d(t_cycle, cycles.mean(axis=0), kind='cubic')(t_fine))
global_mean = np.mean(all_means, axis=0)
global_std = np.std(all_means, axis=0)
ax.fill_between(t_fine, global_mean - global_std, global_mean + global_std, 
                color='gray', alpha=0.2, label=r'$\pm 1\sigma$ inter-run')
ax.plot(t_fine, global_mean, color='black', lw=3, ls='--', label='Global mean')

ax.set_xlabel(r'$t$ (s)', fontsize=12)
ax.set_ylabel(r'$\langle v \rangle$ (m/s)', fontsize=12)
ax.set_title('Phase-averaged velocity — 6 runs superimposed')
ax.legend(fontsize=10)
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.savefig(FIG_DIR / "phase_averaged_superimposed_velocity.png", dpi=300, bbox_inches='tight')
plt.show()

#----------------------------------------------------------------------------------------------------------------------
# %% Plot over the good plots
# ----------------------------------------------------------------------------------------------------------------------
all_cycles = np.concatenate([c for r, c in enumerate(runs_aligned) if r != 5])

K_total = all_cycles.shape[0]  # ~294 cycles
error_L2 = np.zeros(K_total)

for k in range(2, K_total):
    mean_k = all_cycles[:k].mean(axis=0)
    mean_km1 = all_cycles[:k-1].mean(axis=0)
    error_L2[k] = np.linalg.norm(mean_k - mean_km1) / np.linalg.norm(mean_k)

plt.figure()
plt.plot(range(2, K_total), error_L2[2:])
plt.axhline(0.01, color='red', ls='--', label='1%')
plt.legend()
plt.xlabel('Number of cycles K')
plt.ylabel(r'$\epsilon(K)$')
plt.title('Convergence of the phase-averaged mean')
plt.savefig(FIG_DIR / "convergence_phase_averaged_ran.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
