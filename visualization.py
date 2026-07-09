"""
Visualizes output of manometry model based on the paper's own model code (pressure_traces.py)

Panels (choose any subset/order via PANELS below):
  phi     Pacemaker phase          Eq. (1)
  W       Pacemaker waveform       Eq. (4)
  D       ICC drive                Eq. (5)
  ENS     ENS drive                Eq. (7)
  E       Total excitatory drive   Eq. (8)
  M       Muscle activation        Eq. (10)
  tube    Contraction and pressure Eqs. (12)-(13)
  traces  Pressure trace over time (spatiotemporal waterfall, revealed as it plays)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from pressure_traces import Geometry, SimParams, run_traces

# =============================================================================
# USER SETTINGS
# =============================================================================
PHENOTYPE = "healthy"

R0_CM = 2.5
FRAME_STRIDE = 10
FPS = 20
SEGMENT_VIEW = "hapc"   # "hapc" = segments 1-50 (proximal HAPC region only), "all" = every segment (1-75)
TRACE_GAIN_CM = 20.0   # vertical excursion (cm) of the "traces" panel at peak pressure
OUT_FILE = f"outputs/visualization_components_{PHENOTYPE}_{SEGMENT_VIEW}.mp4"

# Which panels to draw, and in what order. Any subset of PANEL_SPECS' keys below.
PANELS = ["D", "ENS", "M", "tube"]

# --- Overall figure size ---
# Leave FIG_WIDTH_IN as None to auto-size from the number/width of selected panels.
# Set both FIG_WIDTH_IN and FIG_HEIGHT_IN to force an exact size/aspect ratio
# (e.g. FIG_WIDTH_IN=16, FIG_HEIGHT_IN=9 for a 16:9 video).
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 6

PANEL_WIDTH_IN = 3.0   # auto-sizing: inches per unit of panel "width" (used when FIG_WIDTH_IN is None)
WSPACE = 0.45          # horizontal gap between panels (gridspec wspace, in axes-widths)
LEFT_MARGIN_IN = 0.75  # room for the rotated Proximal/Distal labels
RIGHT_MARGIN_IN = 0.25
TOP_MARGIN_IN = 1.7    # room for time text + column titles
BOTTOM_MARGIN_IN = 1.0 # room for x-axis labels/ticks
CBAR_GAP_IN = 0.25
CBAR_WIDTH_IN = 0.15
CBAR_LABEL_IN = 0.55   # extra room for the colorbar's own axis label

NARROW_WIDTH_THRESHOLD = 0.5   # panels with a width_ratio below this get a rotated
                               # title and no x-tick labels, since they're too thin to fit them


# =============================================================================
# Panel definitions: static styling/limits for each selectable panel.
# xlim may be a fixed (lo, hi) tuple or a callable(data) -> (lo, hi).
# =============================================================================
def _xlim_nonneg(data):
    return (-0.02, float(data.max()) * 1.15)


def _xlim_activation(data):
    return (-0.02, max(0.4, float(data.max())) * 1.1)


def _xlim_symmetric(data):
    lo = float(data.min()); hi = float(data.max())
    pad = 0.15 * max(abs(lo), abs(hi), 1e-6)
    return (lo - pad, hi + pad)


PANEL_SPECS = {
    "phi": dict(title="Pacemaker phase", color="#1b2a4a", xlabel=r"$\phi_s$ (rad)",
                width=1.0, xlim=(-0.3, 2 * np.pi + 0.3),
                xticks=([0, np.pi, 2 * np.pi], ["0", "π", "2π"])),
    "W": dict(title="Pacemaker waveform", color="#127a75", xlabel=r"$W_s$",
              width=1.0, xlim=(-0.05, 1.05)),
    "D": dict(title="ICC pacemakers", color="#c98a2c", xlabel=r"$D_s^{\rm ICC}$",
              width=1.0, xlim=_xlim_nonneg),
    "ENS": dict(title="Peristaltic reflex", color="#4c72b0", xlabel=r"$E_s^{\rm ENS}$",
                width=1.0, xlim=_xlim_symmetric, zero_line=True),
    "E": dict(title="Total excitation", color="#a5372a", xlabel=r"$E_s$",
              width=1.15, xlim=_xlim_nonneg),
    "M": dict(title="Muscle activation", color="#6a3d9a", xlabel=r"$M_s$",
              width=1.3, xlim=_xlim_activation, lw=2.2),
    "tube": dict(title="", color="#7a4a1e", width=1.3 / 8),
    "traces": dict(title="Pressure trace over time", color="#7a4a1e", width=2.5),
}


# =============================================================================
# Animation
# =============================================================================
def build_canvas(radius_row, pressure_row, y_grid, N):
    ny = y_grid.size
    canvas = np.full((ny, N), np.nan, dtype=float)
    r_col = radius_row[np.newaxis, :]
    y_col = y_grid[:, np.newaxis]
    inside = np.abs(y_col) <= r_col
    canvas[inside] = np.broadcast_to(pressure_row[np.newaxis, :], (ny, N))[inside]
    return canvas


def make_animation(t_rec, Phi_rec, W_rec, D_ICC_rec, ENS_rec, E_rec, M_rec, A_rec, P_rec,
                    geom: Geometry):
    dx = geom.dx_cm
    if SEGMENT_VIEW == "all":
        seg_lo, seg_hi = 1, geom.N
    elif SEGMENT_VIEW == "hapc":
        seg_lo, seg_hi = 1, geom.N_HAPC
    else:
        raise ValueError("SEGMENT_VIEW must be 'all' or 'hapc'")
    i0, i1 = seg_lo - 1, seg_hi   # 0-based, half-open
    n_show = i1 - i0
    x_seg = (np.arange(i0, i1) + 0.5) * dx
    y_top, y_bot = i0 * dx, i1 * dx

    line_data = {
        "phi": Phi_rec[:, i0:i1],
        "W": W_rec[:, i0:i1],
        "D": D_ICC_rec[:, i0:i1],
        "ENS": ENS_rec[:, i0:i1],
        "E": E_rec[:, i0:i1],
        "M": M_rec[:, i0:i1],
    }
    A_s = A_rec[:, i0:i1]
    P_s = P_rec[:, i0:i1]
    radius = R0_CM * np.sqrt(A_s)

    Tsim = SimParams().Tsim
    frame_idx = np.arange(0, len(t_rec), FRAME_STRIDE)

    ny = 220
    Rmax = R0_CM * 1.02
    y_grid = np.linspace(-Rmax, Rmax, ny)

    p_vmax = float(np.percentile(P_s, 99.5))
    norm = Normalize(vmin=0.0, vmax=p_vmax)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="white")

    # "traces" panel: each segment's pressure offset from its own baseline position,
    # peaks pointing toward proximal (up), revealed over time like a manometry printout.
    trace_y = x_seg[np.newaxis, :] - (P_s / p_vmax) * TRACE_GAIN_CM

    n_panels = len(PANELS)
    show_tube = "tube" in PANELS
    width_ratios = [PANEL_SPECS[key]["width"] for key in PANELS]

    auto_width = PANEL_WIDTH_IN * sum(width_ratios) + \
        (CBAR_GAP_IN + CBAR_WIDTH_IN + CBAR_LABEL_IN if show_tube else 0)
    fig_width = FIG_WIDTH_IN if FIG_WIDTH_IN is not None else auto_width
    fig_height = FIG_HEIGHT_IN

    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_width, fig_height), sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": WSPACE},
    )
    axes = np.atleast_1d(axes)
    k0 = frame_idx[0]

    artists = {}   # key -> Line2D (line panels), list of Line2D (traces), or dict (tube)

    for ax, key in zip(axes, PANELS):
        spec = PANEL_SPECS[key]

        if key == "tube":
            canvas0 = build_canvas(radius[k0], P_s[k0], y_grid, n_show).T
            im = ax.imshow(canvas0, extent=[-Rmax, Rmax, y_bot, y_top], origin="upper",
                            aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
            outline_top, = ax.plot(radius[k0], x_seg, color="#333333", lw=1.2)
            outline_bot, = ax.plot(-radius[k0], x_seg, color="#333333", lw=1.2)
            ax.set_xlim(-Rmax * 1.15, Rmax * 1.15)
            if spec["width"] >= NARROW_WIDTH_THRESHOLD:
                ax.set_xlabel("Radius (cm)")
            else:
                ax.set_xticks([])
            artists["tube"] = {"im": im, "top": outline_top, "bot": outline_bot}
        elif key == "traces":
            trace_lines = [ax.plot([], [], color="black", lw=0.6)[0] for _ in range(n_show)]
            ax.set_xlim(0, Tsim)
            ax.set_xlabel("Time (s)")
            artists["traces"] = trace_lines
        else:
            data = line_data[key]
            line, = ax.plot(data[k0], x_seg, color=spec["color"], lw=spec.get("lw", 2.0))
            xlim = spec["xlim"](data) if callable(spec["xlim"]) else spec["xlim"]
            ax.set_xlim(*xlim)
            ax.set_xlabel(spec["xlabel"])
            if spec.get("zero_line"):
                ax.axvline(0, color="#999999", lw=0.8, ls=":")
            if "xticks" in spec:
                pos, labels = spec["xticks"]
                ax.set_xticks(pos); ax.set_xticklabels(labels)
            artists[key] = line

        if spec["width"] >= NARROW_WIDTH_THRESHOLD:
            ax.set_title(spec["title"], fontsize=11, fontweight="bold", color=spec["color"])
        else:
            ax.set_title(spec["title"], fontsize=9, fontweight="bold", color=spec["color"],
                         rotation=90, rotation_mode="anchor", ha="left", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(y_bot, y_top)
    axes[0].set_yticks([])
    y_label_trans = axes[0].get_yaxis_transform()
    axes[0].text(-0.25, y_top + 0.06 * (y_bot - y_top), "Proximal", transform=y_label_trans,
                 ha="center", va="center", rotation=90, rotation_mode="anchor",
                 fontsize=11, fontweight="bold")
    axes[0].text(-0.25, y_top + 0.94 * (y_bot - y_top), "Distal", transform=y_label_trans,
                 ha="center", va="center", rotation=90, rotation_mode="anchor",
                 fontsize=11, fontweight="bold")

    time_text = fig.text(0.5, 1.0 - 0.5 * TOP_MARGIN_IN / fig_height, "",
                          ha="center", fontsize=11, fontweight="bold")

    left = LEFT_MARGIN_IN / fig_width
    right_base = 1.0 - RIGHT_MARGIN_IN / fig_width
    if show_tube:
        right = right_base - (CBAR_GAP_IN + CBAR_WIDTH_IN + CBAR_LABEL_IN) / fig_width
    else:
        right = right_base
    top = 1.0 - TOP_MARGIN_IN / fig_height
    bottom = BOTTOM_MARGIN_IN / fig_height
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

    if show_tube:
        cbar_ax = fig.add_axes([right + CBAR_GAP_IN / fig_width, bottom,
                                 CBAR_WIDTH_IN / fig_width, top - bottom])
        cbar = fig.colorbar(artists["tube"]["im"], cax=cbar_ax)
        cbar.set_label("Pressure (mmHg)")

    def update(i):
        k = frame_idx[i]
        updated = []
        for key in PANELS:
            if key == "tube":
                canvas = build_canvas(radius[k], P_s[k], y_grid, n_show).T
                tube_artists = artists["tube"]
                tube_artists["im"].set_data(canvas)
                tube_artists["top"].set_xdata(radius[k])
                tube_artists["bot"].set_xdata(-radius[k])
                updated += [tube_artists["im"], tube_artists["top"], tube_artists["bot"]]
            elif key == "traces":
                tt = t_rec[:k + 1]
                for s, line in enumerate(artists["traces"]):
                    line.set_data(tt, trace_y[:k + 1, s])
                updated += artists["traces"]
            else:
                artists[key].set_xdata(line_data[key][k])
                updated.append(artists[key])
        time_text.set_text(f"t = {t_rec[k]:5.1f} s")
        updated.append(time_text)
        return tuple(updated)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                   blit=False, interval=1000/FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    ani.save(OUT_FILE, writer=writer, dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    t_rec, Phi_rec, W_rec, D_ICC_rec, ENS_rec, E_rec, M_rec, A_rec, P_rec, geom = run_traces(
        phenotype=PHENOTYPE, write_output=False, make_plot=False)

    make_animation(t_rec, Phi_rec, W_rec, D_ICC_rec, ENS_rec, E_rec, M_rec, A_rec, P_rec, geom)
    print(f"Animation saved to {OUT_FILE}")
