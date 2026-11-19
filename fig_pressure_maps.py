"""
Figure: simulated pressure maps of the three phenotypes (Fig. 4 of the paper,
model panels only).

Each phenotype is simulated with the default settings of pressure_traces.py
and its pressure traces are stacked one per segment, segment 1 (proximal) at
the top, as in manometry. One file is saved per phenotype. The clinical panels
of Fig. 4 are patient data and cannot be shared, so they are not part of this
repository.

Author: Ahilan Anantha Krishnan
Units: time [s], pressure [mmHg]
Dimension: 1D chain of colon segments

Usage:
    python fig_pressure_maps.py
Output:
    figures/fig_pressure_map_<phenotype>.pdf and .svg (healthy, ibsd, stc)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
import pressure_traces as pt

# =============================================================================
# User settings
# =============================================================================
OUT_PREFIX = "fig_pressure_map"
PHENOTYPES = ("healthy", "ibsd", "stc")

# Figure layout (cosmetic)
PANEL_WIDTH_FRACTION = 0.5    # [-] panel width as a fraction of ps.FIG_WIDTH_IN


# =============================================================================
# Main
# =============================================================================
def make_figure() -> None:
    """Simulate each phenotype and save its stacked pressure traces."""
    ps.apply_style()
    pp = pt.PlotParams()
    size_in = ps.FIG_WIDTH_IN * PANEL_WIDTH_FRACTION
    for phenotype in PHENOTYPES:
        res = pt.run_traces(phenotype, write_output=False, make_plot=False, verbose=False)
        fig, ax = plt.subplots(figsize=(size_in, size_in))
        pt.draw_stacked_traces(ax, res.t, res.P, pp)
        fig.tight_layout()
        ps.save_figure(fig, f"{OUT_PREFIX}_{phenotype}")
        plt.close(fig)


if __name__ == "__main__":
    make_figure()
