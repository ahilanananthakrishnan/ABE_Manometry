"""
Figure: power spectra of the simulated pressure maps (model curves of Fig. 5
of the paper).

For every segment, the simulated pressure is linearly detrended, multiplied by
a Hann window, and its periodogram is computed. Each periodogram is normalized
to unit total power in the band (0, F_MAX_CPM], and the normalized spectra are
averaged over all segments. The plotted spectra are zero-padded (PLOT_PAD) to
interpolate them on a finer grid. The fraction of power below 2 cpm and between
2 and 6 cpm (the cyclic motor pattern band) is printed and written to the CSV.

Fig. 5 of the paper compares these model spectra with clinical ones computed
by the same method. The clinical manometry recordings are patient data and
cannot be shared, so only the model data are analyzed here.

Author: Ahilan Anantha Krishnan
Units: time [s], frequency [cpm], pressure [mmHg]
Dimension: 1D chain of colon segments

Usage:
    python fig_spectra.py
Output:
    figures/fig_spectra.pdf, figures/fig_spectra.svg, outputs/spectra_bands.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
import pressure_traces as pt

# =============================================================================
# User settings
# =============================================================================
OUT_NAME = "fig_spectra"
CSV_PATH = os.path.join(ps.OUTPUT_DIR, "spectra_bands.csv")
F_MAX_CPM = 12.0              # [cpm] upper limit of the normalization band
PLOT_PAD = 16                 # [-] zero-padding factor for the plotted spectra (smooth curves)
BAND_LOW_CPM = 2.0            # [cpm] lower edge of the cyclic motor pattern band
BAND_HIGH_CPM = 6.0           # [cpm] upper edge of the cyclic motor pattern band

# case -> panel header
CASES: dict[str, str] = {"healthy": "Healthy", "ibsd": "IBS-D", "stc": "STC"}

# Figure layout (cosmetic)
FIG_ASPECT = 0.4              # [-] height/width
X_MAX_CPM = 10.0              # [cpm] x-axis limit
BAND_SHADE = "0.92"           # fill for the 2-6 cpm band
W_PAD = 0.6                   # [font sizes] horizontal space between panels
Y_LIM = (0.0, 0.8)            # [1/cpm] y-axis range, same in all panels
N_TICKS = 3                   # [-] ticks per axis, including both ends


# =============================================================================
# Spectra
# =============================================================================
def normalized_spectrum(x: np.ndarray, fs_hz: float,
                        pad: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean normalized periodogram over the columns of x (time x channels).

    Returns the frequency [cpm] and the spectrum, each channel normalized to
    unit power in (0, F_MAX_CPM] before averaging. ``pad`` > 1 zero-pads the
    FFT to pad times the window length, which interpolates the spectrum on a
    finer frequency grid without changing its resolution (used for plotting).
    """
    n = x.shape[0]
    a = np.column_stack([np.arange(n), np.ones(n)])
    x = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]          # linear detrend
    n_fft = pad * n
    p = np.abs(np.fft.rfft(x * np.hanning(n)[:, None], n=n_fft, axis=0)) ** 2
    f_cpm = np.fft.rfftfreq(n_fft, 1.0 / fs_hz) * 60.0
    band = (f_cpm > 0) & (f_cpm <= F_MAX_CPM)
    p = p / p[band].sum(axis=0, keepdims=True)
    return f_cpm, p.mean(axis=1)


def band_fractions(f_cpm: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Fraction of power below BAND_LOW_CPM and within the 2-6 cpm band."""
    total = s[(f_cpm > 0) & (f_cpm <= F_MAX_CPM)].sum()
    low = s[(f_cpm > 0) & (f_cpm < BAND_LOW_CPM)].sum() / total
    mid = s[(f_cpm >= BAND_LOW_CPM) & (f_cpm <= BAND_HIGH_CPM)].sum() / total
    return low, mid


def model_pressure(case: str) -> tuple[np.ndarray, float]:
    """Simulated pressure (time x segments) and its sampling rate [Hz]."""
    res = pt.run_traces(case, write_output=False, make_plot=False, verbose=False)
    return res.P, 1.0 / (res.t[1] - res.t[0])


# =============================================================================
# Main
# =============================================================================
def make_figure() -> None:
    """Compute the spectra, write the band fractions, and save the figure."""
    ps.apply_style()
    fig, axes = plt.subplots(1, len(CASES),
                             figsize=(ps.FIG_WIDTH_IN, ps.FIG_WIDTH_IN * FIG_ASPECT))
    rows = []
    for ax, (case, title) in zip(axes, CASES.items()):
        p_m, fs_m = model_pressure(case)
        f_p, s_p = normalized_spectrum(p_m, fs_m, PLOT_PAD)   # padded grid: plot
        ax.axvspan(BAND_LOW_CPM, BAND_HIGH_CPM, color=BAND_SHADE, lw=0)
        # power density per cpm, independent of the frequency grid spacing
        ax.plot(f_p, s_p / (f_p[1] - f_p[0]), color=ps.INK, lw=ps.LINE_WIDTH)
        ax.set_xlim(0, X_MAX_CPM)
        ax.set_xticks(np.linspace(0, X_MAX_CPM, N_TICKS))
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(np.linspace(*Y_LIM, N_TICKS))
        ax.set_title(title)  # panel header naming the case, not a plot title
        ax.set_xlabel("Frequency [cpm]")
        if ax is not axes[0]:
            ax.set_yticklabels([])  # shared y range; labels on the first panel only
        ps.style_axes(ax)

        low, mid = band_fractions(*normalized_spectrum(p_m, fs_m))  # native grid
        rows.append([case, f"{low:.2f}", f"{mid:.2f}"])
        print(f"{case:8s} power <{BAND_LOW_CPM:g} cpm {low:.2f} | "
              f"{BAND_LOW_CPM:g}-{BAND_HIGH_CPM:g} cpm {mid:.2f}")
    axes[0].set_ylabel("Normalized power [1/cpm]")
    fig.tight_layout(w_pad=W_PAD)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", f"power_below_{BAND_LOW_CPM:g}cpm",
                         f"power_{BAND_LOW_CPM:g}_{BAND_HIGH_CPM:g}cpm"])
        writer.writerows(rows)
    print(f"Saved {CSV_PATH}")
    ps.save_figure(fig, OUT_NAME)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
