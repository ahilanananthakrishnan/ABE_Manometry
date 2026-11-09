"""
Figure: time courses of the key model variables in the healthy case, for one
proximal (HAPC) and one distal (CMP) segment (response to Reviewer 1, comment 2).

Rows: pacemaker waveform W_s, total excitatory drive E_s (ICC + ENS, after
gating), muscle activation M_s, and pressure p_s. In the distal column, shaded
intervals mark when the rectosigmoid brake is ON (Gamma_CMP = 0).

Usage:
    python fig_state_variables.py
Output:
    figures/fig_state_variables.pdf and figures/fig_state_variables.svg
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
from pressure_traces import SimParams, run_traces

# =============================================================================
# User settings
# =============================================================================
OUT_DIR = "figures"
OUT_NAME = "fig_state_variables"
PHENOTYPE = "healthy"
SEG_PROXIMAL = 25   # [-] HAPC segment shown (1-based)
SEG_DISTAL = 63     # [-] CMP segment shown (1-based)
X_LIM_S = (0.0, SimParams().Tsim)  # [s] time axis (simulated duration)
Y_UNIT = (0.0, 1.0)             # [-] axis range of the normalized variables
Y_PRESSURE_MMHG = (0.0, 150.0)  # [mmHg] pressure axis, same in both columns

# Figure layout (cosmetic)
FIG_ASPECT = 1.05                    # [-] height/width
LINE_WIDTH = 1.8                     # [pt] data lines
BRAKE_SHADE = "0.85"                 # fill for brake-ON intervals
H_PAD = 0.8                          # [font sizes] vertical space between rows


def gate_intervals(t: np.ndarray, gate: np.ndarray) -> List[Tuple[float, float]]:
    """Return (start, end) times of the intervals where the gate is closed (brake ON)."""
    closed = gate < 0.5  # gate is 1 (open) or 0 (brake ON)
    edges = np.diff(closed.astype(int))
    starts = list(t[1:][edges == 1])
    ends = list(t[1:][edges == -1])
    if closed[0]:
        starts.insert(0, t[0])
    if closed[-1]:
        ends.append(t[-1])
    return list(zip(starts, ends))


def make_figure() -> None:
    """Build the figure and save it as PDF and SVG."""
    ps.apply_style()
    gate_out: list = []
    res = run_traces(PHENOTYPE, write_output=False, make_plot=False, gate_out=gate_out)
    t = res.t
    brake_on = gate_intervals(t, gate_out[0])

    rows = [  # (data, label, (ymin, ymax)); min, middle, and max are ticked
        (res.W, r"$W_s$ [-]", Y_UNIT),
        (res.E, r"$E_s$ [-]", Y_UNIT),
        (res.M, r"$M_s$ [-]", Y_UNIT),
        (res.P, r"$p_s$ [mmHg]", Y_PRESSURE_MMHG),
    ]
    columns = [
        (SEG_PROXIMAL, f"Proximal (HAPC), $s={SEG_PROXIMAL}$", False),
        (SEG_DISTAL, f"Distal (CMP), $s={SEG_DISTAL}$", True),
    ]

    fig, axes = plt.subplots(len(rows), len(columns), sharex=True,
                             figsize=(ps.FIG_WIDTH_IN, ps.FIG_WIDTH_IN * FIG_ASPECT))
    for c, (seg, title, shade_brake) in enumerate(columns):
        for r, (data, ylabel, ylim) in enumerate(rows):
            ax = axes[r, c]
            if shade_brake:
                for start, end in brake_on:
                    ax.axvspan(start, end, color=BRAKE_SHADE, linewidth=0)
            ax.plot(t, data[:, seg - 1], color=ps.INK, linewidth=LINE_WIDTH,
                    clip_on=False)  # W_s reaches the upper limit exactly
            ax.set_xlim(*X_LIM_S)
            ax.set_xticks(np.linspace(*X_LIM_S, 3))
            ax.set_ylim(*ylim)
            ax.set_yticks(np.linspace(*ylim, 3))
            if c == 0:
                ax.set_ylabel(ylabel)
        axes[0, c].set_title(title)
        axes[-1, c].set_xlabel("Time [s]")

    fig.tight_layout(h_pad=H_PAD)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}"))
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{OUT_NAME}.pdf and .svg")


if __name__ == "__main__":
    make_figure()
