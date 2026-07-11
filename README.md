# XAI-PIV-2D

Explainable deep learning (SHAP / Expected Gradients) applied to a 2D PIV dataset
of two interacting jets, to reveal the flow regions driving the turbulent field.

Adapted from the 3D turbulent-channel methodology of Cremades et al.
(*Nat. Commun.*, 2024) — original code:
[KTH-FlowAI/XAI_turbulentchannel_3d_simplified](https://github.com/KTH-FlowAI/XAI_turbulentchannel_3d_simplified).
Expected Gradients from Erion et al. (*Nat. Mach. Intell.* 3, 620–631, 2021).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `shap` package is a modified version bundled in `code/py_bin/py_packages/` —
do not install it from PyPI.

## Data

Not included (~13 GB). See [`data/README.md`](data/README.md) for the expected
layout. The trained model (`results/models/trained_model.h5`) is provided so SHAP
can run without retraining.

## Pipeline

Set parameters in [`code/configuration/`](code/configuration/), then run from
`code/`:

```text
main_statistics.py                # flow statistics
prepare_tfrecords.py              # build training data
main_CNN.py                       # train the CNN
main_SHAP.py                      # SHAP values (Expected Gradients)
main_statisticsSHAP.py            # SHAP statistics
main_percolation_shapstruc_range.py   # percolate SHAP structures
```

2D analyses and figures are in [`analysis/`](analysis/).

## License

[MIT](LICENSE).
