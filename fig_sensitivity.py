"""
Figure: local sensitivity of peak pressure and motility index to each model
parameter around the healthy case (response to Reviewer 1, comment 3).

Each parameter theta is perturbed one at a time by +/- h (relative), with all
other parameters at their healthy values, and the normalized central-difference
sensitivity is

    S = [Y(theta (1 + h)) - Y(theta (1 - h))] / (2 h Y(theta)),

i.e. the percent change in the output per percent change in the parameter.
Outputs Y: peak pressure (max over all segments and time, mmHg) and motility
index (area under the pressure-time curve summed over all segments, mmHg s).
The healthy case is deterministic (sigma_phi = 0), so a single run per
perturbation suffices. Tube-law parameters are excluded (see manuscript).

The step H_REPORT is shown in the figure; all steps in H_CHECK are written to
the CSV as a step-size convergence check.

Usage:
    python fig_sensitivity.py
Output:
    figures/fig_sensitivity.pdf, figures/fig_sensitivity.svg,
    outputs/sensitivity.csv
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import plot_style as ps
import pressure_traces as pt

# =============================================================================
# User settings
# =============================================================================
OUT_DIR = "figures"
OUT_NAME = "fig_sensitivity"
CSV_PATH = os.path.join("outputs", "sensitivity.csv")
PHENOTYPE = "healthy"
H_REPORT = 0.01                 # [-] relative step shown in the figure (1%)
H_CHECK = (0.001, 0.01, 0.1)    # [-] relative steps written to the CSV

# (field name in pressure_traces, axis label), in the order of the original figure
PARAMETERS: List[Tuple[str, str]] = [
    ("f_hapc_cpm", r"$f_{\mathrm{HAPC}}$"),
    ("f_cmp_cpm", r"$f_{\mathrm{CMP}}$"),
    ("v_cmp_cmps", r"$v_{\mathrm{CMP}}$"),
    ("phi_duty", r"$\phi_{\mathrm{duty}}$"),
    ("gamma", r"$\gamma$"),
    ("k_w", r"$k_{\mathrm{w}}$"),
    ("a_cmp", r"$a_{\mathrm{CMP}}$"),
    ("alpha_spread_hapc", r"$\alpha_{\mathrm{spread,HAPC}}$"),
    ("alpha_spread_cmp", r"$\alpha_{\mathrm{spread,CMP}}$"),
    ("alpha_asc", r"$\alpha_{\mathrm{asc}}$"),
    ("alpha_desc", r"$\alpha_{\mathrm{desc}}$"),
    ("tau_E", r"$\tau_E$"),
    ("tau_on", r"$\tau_{\mathrm{on}}$"),
    ("tau_off", r"$\tau_{\mathrm{off}}$"),
    ("p_on", r"$p_{\mathrm{on}}$"),
    ("p_off", r"$p_{\mathrm{off}}$"),
    ("tau_P", r"$\tau_P$"),
    ("v_hapc_cmps", r"$v_{\mathrm{HAPC}}$"),
    ("k_hapc", r"$k_{\mathrm{HAPC}}$"),
    ("k_cmp", r"$k_{\mathrm{CMP}}$"),
    ("a_hapc", r"$a_{\mathrm{HAPC}}$"),
]
OUTPUT_LABELS = ("Peak pressure", "Motility index")

# Figure layout (cosmetic)
N_SPLIT = 11                 # [-] parameters in the first heatmap block
CMAP = "RdBu_r"              # diverging: blue (decrease) - white - red (increase)
S_LIM = 1.0                  # [-] colorbar range (-S_LIM to S_LIM)
N_LEVELS = 20                # [-] discrete color levels, 10 on each side of zero (step 0.1)
FIG_ASPECT = 0.76            # [-] height/width
CELL_EDGE = ps.INK            # grid line color between cells (black)
CELL_EDGE_WIDTH = 0.8        # [pt]
H_PAD = 0.5                  # [font sizes] vertical space between the two blocks
LABEL_ANGLE = 45             # [deg] x tick label inclination
N_CBAR_TICKS = 5             # [-] labeled colorbar ticks, including both ends and zero
CBAR_FRACTION = 0.05         # [-] colorbar size as a fraction of the heatmap axes
CBAR_PAD = 0.18              # [-] gap between heatmaps and colorbar, fraction of axes
CBAR_ASPECT = 30             # [-] colorbar length / thickness


# =============================================================================
# Sensitivity computation
# =============================================================================
def baseline_values(phenotype: str) -> Dict[str, float]:
    """Values of all parameters that run_traces accepts as overrides."""
    return {f.name: getattr(g, f.name) for g in pt.parameter_groups(phenotype)
            for f in fields(g)}


def outputs(phenotype: str, overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Peak pressure [mmHg] and motility index [mmHg s] of one simulation."""
    with contextlib.redirect_stdout(io.StringIO()):  # silence the max-pressure print
        res = pt.run_traces(phenotype, write_output=False, make_plot=False,
                            overrides=overrides)
    return np.array([res.P.max(), res.P.sum() * (res.t[1] - res.t[0])])


def central_sensitivity(phenotype: str, h: float) -> np.ndarray:
    """Normalized central-difference sensitivities (n_outputs x n_parameters)."""
    base = baseline_values(phenotype)
    y0 = outputs(phenotype)
    S = np.zeros((len(OUTPUT_LABELS), len(PARAMETERS)))
    for j, (name, _) in enumerate(PARAMETERS):
        y_plus = outputs(phenotype, {name: base[name] * (1.0 + h)})
        y_minus = outputs(phenotype, {name: base[name] * (1.0 - h)})
        S[:, j] = (y_plus - y_minus) / (2.0 * h * y0)
    return S


def write_csv(results: Dict[float, np.ndarray]) -> None:
    """One row per parameter, one column per (output, step)."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    steps = sorted(results)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter"] + [f"S_{o.replace(' ', '_').lower()}_h{h:g}"
                                    for o in OUTPUT_LABELS for h in steps])
        for j, (name, _) in enumerate(PARAMETERS):
            w.writerow([name] + [f"{results[h][i, j]:.4f}"
                                 for i in range(len(OUTPUT_LABELS)) for h in steps])
    print(f"Saved {CSV_PATH}")


# =============================================================================
# Main
# =============================================================================
def make_figure() -> None:
    """Compute the sensitivities, write the CSV, and save the figure as PDF and SVG."""
    results = {h: central_sensitivity(PHENOTYPE, h) for h in H_CHECK}
    write_csv(results)
    S = results[H_REPORT]

    ps.apply_style()
    levels = np.linspace(-S_LIM, S_LIM, N_LEVELS + 1)   # discrete color levels
    if np.max(np.abs(S)) > S_LIM:
        raise ValueError(f"max |S| = {np.max(np.abs(S)):.3f} exceeds S_LIM = {S_LIM}; "
                         "increase S_LIM so no cell is clipped")
    norm = BoundaryNorm(levels, plt.get_cmap(CMAP).N)
    blocks = [slice(0, N_SPLIT), slice(N_SPLIT, len(PARAMETERS))]
    fig, axes = plt.subplots(len(blocks), 1,
                             figsize=(ps.FIG_WIDTH_IN, ps.FIG_WIDTH_IN * FIG_ASPECT))
    for ax, blk in zip(axes, blocks):
        im = ax.imshow(S[:, blk], cmap=CMAP, norm=norm, aspect="equal")
        labels = [lab for _, lab in PARAMETERS[blk]]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=LABEL_ANGLE, ha="right", va="center",
                           rotation_mode="anchor")
        ax.set_yticks(range(len(OUTPUT_LABELS)))
        ax.set_yticklabels(OUTPUT_LABELS)
        ax.tick_params(length=0, top=False, right=False)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(OUTPUT_LABELS)), minor=True)
        ax.grid(which="minor", color=CELL_EDGE, linewidth=CELL_EDGE_WIDTH)
        ax.tick_params(which="minor", length=0)
        ax.set_xlim(-0.5, len(labels) - 0.5)
    fig.tight_layout(h_pad=H_PAD)
    cbar = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal",
                        fraction=CBAR_FRACTION, pad=CBAR_PAD, aspect=CBAR_ASPECT)
    cbar.set_ticks(np.linspace(-S_LIM, S_LIM, N_CBAR_TICKS))  # labeled ticks
    cbar.ax.set_xticks(levels, minor=True)              # one minor tick per level boundary
    cbar.set_label(r"Normalized sensitivity $S$ [-]")
    cbar.ax.tick_params(top=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}"))
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{OUT_NAME}.pdf and .svg")


if __name__ == "__main__":
    make_figure()
