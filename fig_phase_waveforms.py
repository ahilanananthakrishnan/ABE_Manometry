"""
Figure: phase phi_s(t) (left column) and pacemaker waveform W_s(t) (right
column) of one oscillator in a generic nearest-neighbor chain for
variations of each parameter of the Kuramoto-type phase equation, Eq. (1)
(response to Reviewer 1, comment 1).

Each row shows the table value of one parameter and that value increased by
50% and 100% (half the table value and zero for the coupling strength), with
the others at the values of the healthy HAPC subnetwork (f = 0.4 cpm,
v = 1.2 cm/s, k = 2.0 rad/s, sigma_phi = 0). The healthy phase noise is zero,
so the noise row shows 0, 0.1, and 0.2 rad/sqrt(s):
  A intrinsic frequency f
  B preferred propagation speed v (sets the phase lag delta)
  C coupling strength k
  D phase-noise amplitude sigma_phi

The phase shown is that of the last oscillator in the chain, farthest from the
leading oscillator, so the effects of delta and k that accumulate along the chain
are visible. Only Eq. (1) is integrated; the phase does not depend on the
downstream states (E, M, A, p).

Usage:
    python fig_phase_waveforms.py
Output:
    figures/fig_phase_waveforms.pdf and figures/fig_phase_waveforms.svg
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
from pressure_traces import (
    SEED,
    Geometry,
    OscillatorHealthyBaseline,
    SimParams,
    WaveformParams,
    icc_waveform_skewed,
    phase_chain_increment,
    phenotype_params,
)

# =============================================================================
# User settings (the phase-noise row uses SEED from pressure_traces.py)
# =============================================================================
OUT_DIR = "figures"
OUT_NAME = "fig_phase_waveforms"
T_SHOW_S = SimParams().Tsim  # [s] simulated window (same as the model simulations)
N_CHAIN = 25                 # [-] oscillators in the generic chain

# Sweep values: the table value (baseline, healthy HAPC subnetwork), +50%, and +100%.
# The healthy phase noise is zero, so the noise row uses fixed values instead.
SWEEP_FACTORS = (1.0, 1.5, 2.0)     # [-] multiples of the table value
K_SWEEP_FACTORS = (0.0, 0.5, 1.0)   # [-] coupling strength: uncoupled, half, and table value
SIGMA_SWEEP = (0.0, 0.1, 0.2)       # [rad/sqrt(s)] phase-noise amplitude (0 = healthy)

# Figure layout (cosmetic)
FIG_ASPECT = 1.25                    # [-] height/width
LINE_WIDTH = 2.2                     # [pt] data lines (thicker than the shared default)
LINE_SHADES = ("0.75", "0.45", ps.INK)  # in sweep order, so darker = larger value in every row
WIDTH_RATIOS = (1.0, 1.0, 0.42)     # [-] phi column, W column, legend column
LEGEND_GAP_X = 0.015                 # [figure fraction] gap between W panels and legend boxes
LEGEND_FRAME_WIDTH = 0.8             # [pt] legend box line width
LEGEND_HANDLE_LENGTH = 1.0           # [font sizes] length of the legend line samples
TITLE_GAP = 0.006                    # [figure fraction] gap between row title and panels
ROW_LABEL_X = 0.01                   # [figure fraction] x position of row letters
TIGHT_RECT = (0.03, 0.0, 1.0, 0.97)  # [figure fraction] room for the top title and letters
H_PAD = 2.6                          # [font sizes] vertical space for the row titles
W_PAD = 0.5                          # [font sizes] horizontal space between panels


# =============================================================================
# Phase chain simulation (Eq. 1, Euler-Maruyama)
# =============================================================================
def simulate_phase_chain(n_seg: int, f_cpm: float, k: float, v_cmps: float,
                         sigma_phi: float, retrograde: bool, dx_cm: float,
                         dt: float, t_end: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate Eq. (1) for one chain started from a uniform phase.

    Returns the time vector and the wrapped phase history (n_time x n_seg).
    """
    omega = 2.0 * np.pi * f_cpm / 60.0
    delta = -omega / (v_cmps / dx_cm)
    n_time = int(round(t_end / dt))
    phi = np.zeros(n_seg, dtype=float)
    phi_rec = np.zeros((n_time, n_seg), dtype=float)
    noise_sd = sigma_phi * np.sqrt(dt)
    for i in range(n_time):
        dphi = phase_chain_increment(phi, omega, k, delta, retrograde, dt)
        phi += dphi + noise_sd * rng.standard_normal(n_seg)
        phi_rec[i] = np.mod(phi, 2.0 * np.pi)
    return np.arange(n_time) * dt, phi_rec


def break_wraps(phi: np.ndarray) -> np.ndarray:
    """Insert NaN where the wrapped phase jumps back to 0 so no vertical line is drawn."""
    out = phi.copy()
    out[1:][np.abs(np.diff(phi)) > np.pi] = np.nan
    return out


# =============================================================================
# Main
# =============================================================================
def make_figure() -> None:
    """Build the figure and save it as PDF and SVG."""
    ps.apply_style()
    geom = Geometry()
    base = OscillatorHealthyBaseline()
    healthy = phenotype_params("healthy")
    dt = SimParams().dt

    baseline = dict(f_cpm=base.f_hapc_cpm, k=healthy.k_hapc, v_cmps=healthy.v_hapc_cmps,
                    sigma_phi=healthy.sigma_phi)
    sweep = dict(
        f_cpm=tuple(fac * baseline["f_cpm"] for fac in SWEEP_FACTORS),
        v_cmps=tuple(fac * baseline["v_cmps"] for fac in SWEEP_FACTORS),
        k=tuple(fac * baseline["k"] for fac in K_SWEEP_FACTORS),
        sigma_phi=SIGMA_SWEEP,
    )
    rows: List[Tuple[str, str, Sequence[float]]] = [
        ("f_cpm", r"Intrinsic frequency $f_\kappa$ [cpm]", sweep["f_cpm"]),
        ("v_cmps", r"Propagation speed $v_\kappa$ [cm/s], sets phase lag $\delta_\kappa$",
         sweep["v_cmps"]),
        ("k", r"Coupling strength $k_\kappa$ [rad/s]", sweep["k"]),
        ("sigma_phi", r"Phase noise $\sigma_\phi$ [rad/$\sqrt{\mathrm{s}}$]", sweep["sigma_phi"]),
    ]

    wf = WaveformParams()
    row_handles = []
    fig, axes = plt.subplots(len(rows), 3, sharex="col",
                             figsize=(ps.FIG_WIDTH_IN, ps.FIG_WIDTH_IN * FIG_ASPECT),
                             gridspec_kw=dict(width_ratios=WIDTH_RATIOS))
    for (ax_phi, ax_w, ax_leg), (key, title, values) in zip(axes, rows):
        handles = []
        # same number of decimals for all values in a row (e.g. 0.0, 0.2, 2.0)
        n_dec = max(len(f"{v:g}".partition(".")[2]) for v in values)
        for val, shade in zip(values, LINE_SHADES):
            rng = np.random.default_rng(SEED)
            t, phi = simulate_phase_chain(N_CHAIN, retrograde=False, dx_cm=geom.dx_cm,
                                          dt=dt, t_end=T_SHOW_S, rng=rng,
                                          **dict(baseline, **{key: val}))
            label = f"{val:.{n_dec}f}"
            if np.isclose(val, baseline[key]):
                label += " (baseline)"  # healthy HAPC value from Tables 1 and 2
            line, = ax_phi.plot(t, break_wraps(phi[:, -1]), color=shade,
                                linewidth=LINE_WIDTH, label=label)
            handles.append((val, line))
            ax_w.plot(t, icc_waveform_skewed(phi[:, -1], wf), color=shade, linewidth=LINE_WIDTH,
                      clip_on=False)  # W_s reaches the upper limit exactly
        ax_phi.set_ylim(0.0, 2.0 * np.pi)
        ax_phi.set_yticks([0.0, np.pi, 2.0 * np.pi])
        ax_phi.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        ax_phi.set_ylabel(r"$\phi_s$ [rad]")
        ax_w.set_ylim(0.0, 1.0)
        ax_w.set_yticks([0.0, 0.5, 1.0])
        ax_w.set_ylabel(r"$W_s$ [-]")
        for ax in (ax_phi, ax_w):
            ax.set_xlim(0.0, T_SHOW_S)
            ax.set_xticks([0.0, 0.5 * T_SHOW_S, T_SHOW_S])
        ax_leg.axis("off")  # reserves space; legends are placed after tight_layout
        # legend lists the largest value at the top
        row_handles.append([h for _, h in sorted(handles, key=lambda vh: vh[0], reverse=True)])
    for ax in axes[-1, :2]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout(rect=TIGHT_RECT, h_pad=H_PAD, w_pad=W_PAD)

    # Row titles centered over the two data columns, and row letters at the left
    # One boxed legend per row, entries stacked vertically, just right of the W panel
    extra_artists = []  # kept in the saved bounding box
    for (ax_phi, ax_w, _), (_, title, _), handles, letter in zip(axes, rows, row_handles,
                                                                   "ABCD"):
        left, right = ax_phi.get_position(), ax_w.get_position()
        leg = fig.legend(handles=handles, loc="center left",
                         bbox_to_anchor=(right.x1 + LEGEND_GAP_X, 0.5 * (right.y0 + right.y1)),
                         ncol=1, frameon=True, fancybox=False, edgecolor=ps.INK,
                         handlelength=LEGEND_HANDLE_LENGTH, borderaxespad=0.0)
        leg.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
        extra_artists.append(leg)
        extra_artists.append(fig.text(0.5 * (left.x0 + right.x1), left.y1 + TITLE_GAP, title,
                                      ha="center", va="bottom"))
        extra_artists.append(fig.text(ROW_LABEL_X, left.y1 + TITLE_GAP, letter,
                                      fontsize=ps.PANEL_LABEL_SIZE, va="bottom", ha="left"))

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}"), bbox_extra_artists=extra_artists)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{OUT_NAME}.pdf and .svg")


if __name__ == "__main__":
    make_figure()
