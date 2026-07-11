"""
Figure — SHAP convergence study: relative L2 error vs nsamples.

Reads the benchmark table in results/sta/benchmark_nsamples.txt and plots
the relative L2 error of the gradient-SHAP estimate with respect to the
reference computed at nsamples = 2000, as a function of nsamples
(logarithmic x-axis).

The production choice of `nsamples = 1000` used for the 2078-snapshot
campaign is marked with a vertical guide line.

Run from analysis/:

    python -m figures.convergence_nsamples
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import (
    NSAMPLES_SHAP,
    STA_DIR,
    apply_style,
    save_figure,
)


BENCHMARK_FILE = STA_DIR / "benchmark_nsamples.txt"


# ---------------------------------------------------------------------------
def parse_benchmark(path: Path = BENCHMARK_FILE
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nsamples, rel_l2_err_percent, time_s) parsed from the table."""
    row_re = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*(?:\(REF\))?\s*\|\s*([\d.]+)\s*$"
    )
    ns, err, tsec = [], [], []
    for line in path.read_text().splitlines():
        m = row_re.match(line)
        if not m:
            continue
        ns.append(int(m.group(1)))
        tsec.append(float(m.group(2)))
        err.append(float(m.group(3)))
    if not ns:
        raise RuntimeError(f"No benchmark rows parsed from {path}")
    order = np.argsort(ns)
    return (np.asarray(ns)[order],
            np.asarray(err)[order],
            np.asarray(tsec)[order])


# ---------------------------------------------------------------------------
def make_figure(save: bool = True) -> plt.Figure:
    apply_style()
    ns, err, _ = parse_benchmark()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor("white")

    ax.set_xscale("log")

    ax.plot(ns, err,
            color="#1f77b4", linewidth=1.6,
            marker="o", markersize=5.5,
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Benchmark vs reference (2000 samples)",
            zorder=3)

    ax.axvline(NSAMPLES_SHAP, color="#d62728",
               linestyle="--", linewidth=1.2,
               label=rf"production choice  $n_\mathrm{{samples}} = {NSAMPLES_SHAP}$",
               zorder=2)

    for n_i, e_i in zip(ns, err):
        if e_i == 0.0:
            ax.annotate("REF", xy=(n_i, e_i), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8, color="black")
        else:
            ax.annotate(f"{e_i:.1f}%", xy=(n_i, e_i), xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8, color="#1f77b4")

    ax.set_xlabel(r"$n_\mathrm{samples}$")
    ax.set_ylabel(r"Relative $L_2$ error  [%]")
    ax.set_xlim(ns.min() * 0.8, ns.max() * 1.25)
    ax.set_ylim(bottom=-1.0, top=max(err) * 1.15)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", frameon=True,
              edgecolor="black", fancybox=False,
              fontsize=8.5, framealpha=0.95)

    ax.set_title(r"Gradient-SHAP convergence study  —  snapshot $t = 500$",
                 pad=6, fontsize=10.5)

    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.91)

    if save:
        save_figure(fig, "convergence_nsamples")
    return fig


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build convergence-study figure.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fig = make_figure()
    if not args.no_show:
        plt.show()
