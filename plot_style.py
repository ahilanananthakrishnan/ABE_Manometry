"""
Shared plotting settings for all manuscript figures (single source of truth for
DPI, fonts, line widths, and colors).

Matches the existing manuscript figures (made with the lab MATLAB template
``set_figure_template`` and LaTeX interpreter): Computer Modern text and math,
19 pt axis text, 1.44 pt axes lines, inward ticks, box on, no grid, square axes,
black line art. Units go in square brackets per the lab
convention (e.g. ``Time [s]``, ``[-]`` for dimensionless). Each axes is sized like the template's
600 x 500 px MATLAB figure, so a multi-panel figure is built large and scaled
down to the text width in LaTeX (font size relative to the panel is preserved).

Import and call ``apply_style()`` before creating a figure, and
``style_axes(ax)`` on every axes after plotting.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Shared settings (measured from the submitted Figs 2-3, i.e. revised Figs 3-4)
# =============================================================================
DPI = 300                  # [dots/in] export resolution for raster output
FIG_WIDTH_IN = 542 / 72    # [in] native width of the submitted pressure-map figure
FONT_SIZE = 17             # [pt] axis labels and tick labels at native size
LEGEND_FONT_SIZE = 17      # [pt] legend text
PANEL_LABEL_SIZE = 17      # [pt] panel letters (A, B, ...)
AXIS_LINE_WIDTH = 1.0      # [pt] axes box and ticks
LINE_WIDTH = 1.5           # [pt] single data curves
TRACE_LINE_WIDTH = 0.6     # [pt] stacked traces, as in the pressure maps
INK = "black"              # text and axis color
INK_MUTED = "0.55"         # grey for reference markers
SERIF_FONTS = ["cmr10", "Computer Modern Roman"]  # LaTeX default font, as in Figs 1-3


def apply_style() -> None:
    """Set matplotlib rcParams to match the lab figure template."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_FONTS,
        "mathtext.fontset": "cm",            # LaTeX (Computer Modern) math
        "axes.formatter.use_mathtext": True,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "xtick.major.width": AXIS_LINE_WIDTH,
        "ytick.major.width": AXIS_LINE_WIDTH,
        "xtick.direction": "in",             # MATLAB default tick direction
        "ytick.direction": "in",
        "xtick.top": True,                   # box on: ticks on all four sides
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,                  # embed TrueType, not Type 3 (journal requirement)
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def style_axes(ax: plt.Axes) -> None:
    """Per-axes part of the template: box on and square axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.set_box_aspect(1.0)
