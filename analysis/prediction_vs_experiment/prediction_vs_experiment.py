import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% Prediction plot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from analysis.io import load_velocity, load_mean_profiles

# %% Charger le modèle
model = tf.keras.models.load_model(r'e:\shap\XAI_PIV_2D_simplified-main\results\models\trained_model.h5')

# Lire la normalisation min-max
norm = np.loadtxt(r'e:\shap\XAI_PIV_2D_simplified-main\results\sta\norm.txt')
uumax, vvmax, uumin, vvmin = norm[0], norm[1], norm[2], norm[3]

# Charger un snapshot et le suivant (ground truth)
idx = 500
u_t, v_t = load_velocity(idx)
u_t1, v_t1 = load_velocity(idx + 1)

# Fluctuations
u_mean, v_mean, _ = load_mean_profiles()
u_fluc = u_t - u_mean[:, None]
v_fluc = v_t - v_mean[:, None]

# Normaliser l'input (min-max vers [0, 1])
u_norm = (u_fluc - uumin) / (uumax - uumin)
v_norm = (v_fluc - vvmin) / (vvmax - vvmin)
input_field = np.stack([u_norm, v_norm], axis=-1)[np.newaxis, ...]

# Prédiction
pred_norm = model.predict(input_field)[0]

# Dénormaliser la prédiction
u_pred = pred_norm[:,:,0] * (uumax - uumin) + uumin
v_pred = pred_norm[:,:,1] * (vvmax - vvmin) + vvmin

# Ground truth en fluctuations
u_fluc1 = u_t1 - u_mean[:, None]
v_fluc1 = v_t1 - v_mean[:, None]

# Erreur absolue
error_u = np.abs(u_fluc1 - u_pred)

# Plot — composante u
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin = min(u_fluc1.min(), u_pred.min())
vmax = max(u_fluc1.max(), u_pred.max())

im0 = axes[0].imshow(u_fluc1, origin='upper', cmap='inferno', vmin=vmin, vmax=vmax)
axes[0].set_title('Experiment')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_pred, origin='upper', cmap='inferno', vmin=vmin, vmax=vmax)
axes[1].set_title('U-Net prediction')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(error_u, origin='upper', cmap='inferno', vmin=0, vmax=error_u.max())
axes[2].set_title('Error')
plt.colorbar(im2, ax=axes[2])

fig.tight_layout()
plt.savefig(FIG_DIR / 'prediction_vs_experiment.png', dpi=300, bbox_inches='tight')
plt.show()