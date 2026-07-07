"""
Visualizes output of manometry model based on the paper's own model code (pressure_traces.py)
Restricted to the HAPC region

Six panels total:
  1. phi_s        Eq. (1) 
  2. W_s          Eq. (4) 
  3. D_s^ICC      Eq. (5) 
  4. E_s          Eq. (8) 
                          
  5. M_s          Eq. (10)
  6. wall/pressure Eqs.(12)-(13)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# =============================================================================
# USER SETTINGS
# =============================================================================
PHENOTYPE = "healthy"
SEED = 171

R0_CM = 2.5
FRAME_STRIDE = 10
FPS = 20
X_MAX_CM = 100.0
OUT_FILE = f"/mnt/user-data/outputs/colon_pipeline_6panel_{PHENOTYPE}.mp4"

# =============================================================================
# CASE-SPECIFIC PARAMETERS (identical to pressure_traces.py)
# =============================================================================
ph = PHENOTYPE.strip().lower()

if ph == "healthy":
    a_hapc = 0.77; brake_enabled = True; k_hapc = 2.0; k_cmp = 2.0
    v_hapc_cmps = 1.2; sigma_phi = 0.0
elif ph == "ibsd":
    a_hapc = 0.70; brake_enabled = False; k_hapc = 2.0; k_cmp = 2.0
    v_hapc_cmps = 5.0; sigma_phi = 0.0
elif ph == "stc":
    a_hapc = 0.62; brake_enabled = True; k_hapc = 0.2; k_cmp = 0.2
    v_hapc_cmps = 1.1; sigma_phi = 0.5
else:
    raise ValueError("PHENOTYPE must be one of: 'healthy', 'ibsd', 'stc'")


@dataclass(frozen=True)
class Geometry:
    L_cm: float = 150.0
    N: int = 75
    N_HAPC: int = 50

    @property
    def dx_cm(self) -> float:
        return self.L_cm / self.N


@dataclass(frozen=True)
class WaveformParams:
    phi_duty: float = 0.10
    gamma: float = 2.0
    k_w: float = 8.0


@dataclass(frozen=True)
class TubeLawParams:
    k_c: float = 6.0
    M50: float = 0.30
    beta_C: float = 0.80
    p_ref: float = 1.0
    E_w: float = 25.0
    n_p: float = 1.3


@dataclass(frozen=True)
class MuscleParams:
    tau_E: float = 1.0
    tau_on: float = 1.0
    tau_off: float = 1.2


@dataclass(frozen=True)
class ENSParams:
    alpha_asc: float = 0.10
    alpha_desc: float = 0.08


@dataclass(frozen=True)
class OscillatorHealthyBaseline:
    f_hapc_cpm: float = 0.4
    f_cmp_cpm: float = 3.0
    a_cmp: float = 0.75
    alpha_spread_hapc: float = 0.10
    alpha_spread_cmp: float = 0.05
    v_cmp_cmps: float = 5.0


@dataclass(frozen=True)
class BrakeParams:
    F_sense_1based: Tuple[int, int] = (40, 65)
    p_on: float = 50.0
    p_off: float = 20.0
    tau_P: float = 1.2

    @property
    def F_sense_idx(self) -> np.ndarray:
        lo, hi = self.F_sense_1based
        return np.arange(lo - 1, hi)


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.05
    Tsim: float = 150.0

    @property
    def n_run(self) -> int:
        return int(round(self.Tsim / self.dt))


def icc_waveform_skewed(phi: np.ndarray, wf: WaveformParams) -> np.ndarray:
    twopi = 2.0 * np.pi
    phi_wrapped = np.mod(phi, twopi)
    thr = wf.phi_duty * twopi
    W = np.zeros_like(phi_wrapped, dtype=float)
    mask_up = phi_wrapped < thr
    if np.any(mask_up):
        W[mask_up] = (phi_wrapped[mask_up] / thr) ** wf.gamma
    mask_dn = ~mask_up
    if np.any(mask_dn):
        frac = (phi_wrapped[mask_dn] - thr) / (twopi - thr)
        W[mask_dn] = np.exp(-wf.k_w * frac)
    return W


def contraction_from_M(M, tube):
    return 1.0 / (1.0 + np.exp(-tube.k_c * (M - tube.M50)))


def area_from_M(M, tube):
    C = contraction_from_M(M, tube)
    return 1.0 - tube.beta_C * C


def pressure_from_M(M, tube):
    A = area_from_M(M, tube)
    return tube.p_ref + tube.E_w * ((1.0 / A) ** tube.n_p - 1.0)


# =============================================================================
# Step 1: phase + ICC drive (Eq. 5), shared by both branches
# =============================================================================
def run_phase_and_drive():
    geom = Geometry()
    base = OscillatorHealthyBaseline()
    wf = WaveformParams()

    if SEED is not None:
        np.random.seed(SEED)

    f_hapc_hz = base.f_hapc_cpm / 60.0
    f_cmp_hz = base.f_cmp_cpm / 60.0
    omega_hapc = 2.0 * np.pi * f_hapc_hz
    omega_cmp = 2.0 * np.pi * f_cmp_hz

    dx = geom.dx_cm
    delta_hapc = -(omega_hapc / (v_hapc_cmps / dx))
    delta_cmp = -(omega_cmp / (base.v_cmp_cmps / dx))

    N = geom.N
    N_HAPC = geom.N_HAPC

    phi_H = np.zeros(N); phi_C = np.zeros(N)
    phi_H[:N_HAPC] = np.arange(N_HAPC) * delta_hapc
    phi_C[N_HAPC:] = 0.0

    sim = SimParams()
    n_run = sim.n_run

    Phi_rec = np.zeros((n_run, N))
    W_rec = np.zeros((n_run, N))
    D_ICC_rec = np.zeros((n_run, N))    # Eq. (5)
    t_rec = np.zeros(n_run)

    t_all = np.arange(n_run) * sim.dt

    for k, t in enumerate(t_all):
        sd = sigma_phi * np.sqrt(sim.dt)

        dphi_H = np.zeros(N); dphi_C = np.zeros(N)
        wdt_hapc = omega_hapc * sim.dt
        wdt_cmp = omega_cmp * sim.dt
        kdt_hapc = k_hapc * sim.dt
        kdt_cmp = k_cmp * sim.dt

        dphi_H[0] = wdt_hapc
        for s in range(1, N_HAPC):
            s_neigh = s - 1
            dphi_H[s] = wdt_hapc + kdt_hapc * np.sin(phi_H[s_neigh] + delta_hapc - phi_H[s])

        dphi_C[N - 1] = wdt_cmp
        for s in range(N - 2, N_HAPC - 1, -1):
            s_neigh = s + 1
            dphi_C[s] = wdt_cmp + kdt_cmp * np.sin(phi_C[s_neigh] + delta_cmp - phi_C[s])

        if ph in {"healthy", "ibsd", "stc"}:
            phi_H[:N_HAPC] += sd * np.random.randn(N_HAPC)
            phi_C[N_HAPC:] += sd * np.random.randn(N - N_HAPC)

        phi_H += dphi_H
        phi_C += dphi_C

        W_H = icc_waveform_skewed(phi_H, wf)
        W_C = icc_waveform_skewed(phi_C, wf)

        # --- Eq. (5): D_ICC = a_kappa * (W_s + alpha_spread * W_{s-sigma}) ---
        D_ICC = np.zeros(N)
        base_H = a_hapc * W_H[:N_HAPC]
        D_ICC[:N_HAPC] = base_H
        D_ICC[1:N_HAPC] += base.alpha_spread_hapc * base_H[:-1]

        base_C = base.a_cmp * W_C[N_HAPC:]
        D_ICC[N_HAPC:] += base_C
        D_ICC[N_HAPC:N - 1] += base.alpha_spread_cmp * base_C[1:]

        Phi_rec[k, :N_HAPC] = np.mod(phi_H[:N_HAPC], 2*np.pi)
        Phi_rec[k, N_HAPC:] = np.mod(phi_C[N_HAPC:], 2*np.pi)
        W_rec[k, :N_HAPC] = W_H[:N_HAPC]
        W_rec[k, N_HAPC:] = W_C[N_HAPC:]
        D_ICC_rec[k, :] = D_ICC
        t_rec[k] = t

    return t_rec, Phi_rec, W_rec, D_ICC_rec, geom


# =============================================================================
# Step 2: muscle/tube branch, with or without the Eq.9 low-pass filter.
# Also records E_s (Eq. 8, total excitatory drive) at every step.
# =============================================================================
def run_branch(D_ICC_rec, geom: Geometry, use_lowpass: bool):
    ens = ENSParams()
    mus = MuscleParams()
    tube = TubeLawParams()
    brk = BrakeParams()
    sim = SimParams()

    N = geom.N
    N_HAPC = geom.N_HAPC
    n_run = sim.n_run

    E_star = D_ICC_rec[0].copy()   # E*_s(0) = a_kappa*W_s(0), i.e. D_ICC at k=0 (no ENS yet)
    M = E_star.copy()

    p_lp = pressure_from_M(M, tube)
    cmp_enabled = True

    E_tot_rec = np.zeros((n_run, N))   # Eq. (8)
    M_rec = np.zeros((n_run, N))
    A_rec = np.zeros((n_run, N))
    P_rec = np.zeros((n_run, N))

    for k in range(n_run):
        D_ICC = D_ICC_rec[k]

        A_prev = area_from_M(M, tube)
        DeltaA = A_prev[1:] - A_prev[:-1]
        E_ENS = np.zeros(N)
        E_ENS[:-1] += ens.alpha_asc * np.maximum(DeltaA, 0.0)
        E_ENS[1:] -= ens.alpha_desc * np.maximum(DeltaA, 0.0)

        E_total = D_ICC + E_ENS   # Eq. (8), pre-brake

        if brake_enabled:
            p_inst = pressure_from_M(M, tube)
            aP = sim.dt / brk.tau_P
            p_lp = (1.0 - aP) * p_lp + aP * p_inst
            p_gate = float(np.max(p_lp[brk.F_sense_idx]))
            if cmp_enabled:
                if p_gate >= brk.p_on:
                    cmp_enabled = False
            else:
                if p_gate <= brk.p_off:
                    cmp_enabled = True
            if not cmp_enabled:
                E_total[N_HAPC:] = 0.0   # Eq. (8), brake-gated

        if use_lowpass:
            aE = sim.dt / mus.tau_E
            E_star = (1.0 - aE) * E_star + aE * E_total     # Eq. 9
        else:
            E_star = E_total                                 # Eq. 9 bypassed

        tau_s = np.where(E_star >= M, mus.tau_on, mus.tau_off)
        aM = sim.dt / tau_s
        M = (1.0 - aM) * M + aM * E_star                     # Eq. 10 (unchanged)

        E_tot_rec[k, :] = E_total
        M_rec[k, :] = M
        P_rec[k, :] = pressure_from_M(M, tube)
        A_rec[k, :] = area_from_M(M, tube)

    return E_tot_rec, M_rec, A_rec, P_rec


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


def make_animation(t_rec, Phi_rec, W_rec, D_ICC_rec,
                    E_filt, M_filt, A_filt, P_filt,
                    E_nof, M_nof, A_nof, P_nof,
                    geom: Geometry):
    dx = geom.dx_cm
    n_show = int(round(X_MAX_CM / dx))
    x_seg = (np.arange(n_show) + 0.5) * dx

    Phi_s = Phi_rec[:, :n_show]
    W_s = W_rec[:, :n_show]
    D_s = D_ICC_rec[:, :n_show]
    Ef = E_filt[:, :n_show]; Mf = M_filt[:, :n_show]; Af = A_filt[:, :n_show]; Pf = P_filt[:, :n_show]
    En = E_nof[:, :n_show]; Mn = M_nof[:, :n_show]; An = A_nof[:, :n_show]; Pn = P_nof[:, :n_show]

    radius_f = R0_CM * np.sqrt(Af)
    radius_n = R0_CM * np.sqrt(An)

    frame_idx = np.arange(0, len(t_rec), FRAME_STRIDE)

    ny = 220
    Rmax = R0_CM * 1.02
    y_grid = np.linspace(-Rmax, Rmax, ny)

    p_vmax = float(np.percentile(Pn, 99.5))
    norm = Normalize(vmin=0.0, vmax=p_vmax)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="white")

    navy, teal, gold, brick, purple, gray = (
        "#1b2a4a", "#127a75", "#c98a2c", "#a5372a", "#6a3d9a", "#888888"
    )

    fig, axes = plt.subplots(
        6, 1, figsize=(10, 15.5),
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.15, 1.3], "hspace": 0.65},
    )
    ax_phi, ax_W, ax_D, ax_E, ax_M, ax_tube = axes
    k0 = frame_idx[0]

    # --- 1. phase ---
    line_phi, = ax_phi.plot(x_seg, Phi_s[k0], color=navy, lw=2.0)
    ax_phi.set_ylim(-0.3, 2*np.pi + 0.3)
    ax_phi.set_yticks([0, np.pi, 2*np.pi]); ax_phi.set_yticklabels(["0", "π", "2π"])
    ax_phi.set_ylabel(r"$\phi_s$ (rad)")
    ax_phi.set_title("1. Oscillator phase — Eq. (1)", loc="left", fontsize=12,
                      fontweight="bold", color=navy)
    ax_phi.spines[["top", "right"]].set_visible(False)

    # --- 2. waveform ---
    line_W, = ax_W.plot(x_seg, W_s[k0], color=teal, lw=2.0)
    ax_W.set_ylim(-0.05, 1.05)
    ax_W.set_ylabel(r"$W_s$")
    ax_W.set_title("2. Pacemaker waveform — Eq. (4)", loc="left", fontsize=12,
                    fontweight="bold", color=teal)
    ax_W.spines[["top", "right"]].set_visible(False)

    # --- 3. D_ICC (Eq. 5) -- identical between branches ---
    line_D, = ax_D.plot(x_seg, D_s[k0], color=gold, lw=2.0)
    ax_D.set_ylim(-0.02, float(D_s.max())*1.15)
    ax_D.set_ylabel(r"$D_s^{\rm ICC}$")
    ax_D.set_title("3. ICC-derived myogenic drive — Eq. (5)", loc="left", fontsize=12,
                    fontweight="bold", color=gold)
    ax_D.spines[["top", "right"]].set_visible(False)

    # --- 4. E_s total excitatory drive (Eq. 8) -- compare branches ---
    line_Ef, = ax_E.plot(x_seg, Ef[k0], color=gray, lw=2.0, ls="--",
                          label="with low-pass filter branch")
    line_En, = ax_E.plot(x_seg, En[k0], color=brick, lw=2.0,
                          label="without low-pass filter branch")
    ax_E.set_ylim(-0.02, float(max(Ef.max(), En.max()))*1.15)
    ax_E.set_ylabel(r"$E_s$")
    ax_E.set_title("4. Total excitatory drive (ICC + ENS, brake-gated) — Eq. (8)",
                    loc="left", fontsize=12, fontweight="bold", color=brick)
    ax_E.legend(loc="upper right", frameon=False, fontsize=9)
    ax_E.spines[["top", "right"]].set_visible(False)

    # --- 5. muscle activation (Eq. 10) -- compare branches ---
    line_Mf, = ax_M.plot(x_seg, Mf[k0], color=gray, lw=2.0, ls="--",
                          label="with low-pass filter (Eq. 9, baseline)")
    line_Mn, = ax_M.plot(x_seg, Mn[k0], color=purple, lw=2.2,
                          label="without filter (E*ₛ = Eₛ directly)")
    ax_M.set_ylim(-0.02, max(0.4, float(max(Mf.max(), Mn.max())))*1.1)
    ax_M.set_ylabel(r"$M_s$")
    ax_M.set_title("5. Muscle activation — Eq. (10)", loc="left", fontsize=12,
                    fontweight="bold", color=purple)
    ax_M.legend(loc="upper right", frameon=False, fontsize=9)
    ax_M.spines[["top", "right"]].set_visible(False)

    for ax in (ax_phi, ax_W, ax_D, ax_E, ax_M):
        ax.set_xlim(0, X_MAX_CM)
        ax.set_xticklabels([])

    # --- 6. tube / pressure ---
    canvas0 = build_canvas(radius_n[k0], Pn[k0], y_grid, n_show)
    im = ax_tube.imshow(canvas0, extent=[0, X_MAX_CM, -Rmax, Rmax], origin="lower",
                         aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
    outline_top_n, = ax_tube.plot(x_seg, radius_n[k0], color="#333333", lw=1.2)
    outline_bot_n, = ax_tube.plot(x_seg, -radius_n[k0], color="#333333", lw=1.2)
    outline_top_f, = ax_tube.plot(x_seg, radius_f[k0], color="white", lw=1.3, ls="--")
    outline_bot_f, = ax_tube.plot(x_seg, -radius_f[k0], color="white", lw=1.3, ls="--")
    ax_tube.set_xlim(0, X_MAX_CM)
    ax_tube.set_ylim(-Rmax*1.15, Rmax*1.15)
    ax_tube.set_yticks([])
    ax_tube.set_xlabel("Position along colon (cm) — proximal → splenic flexure")
    ax_tube.set_title(r"6. Wall shape ($\propto\sqrt{A_s}$) & pressure — Eqs. (12)-(13)  "
                       "(solid = without filter, dashed white = with filter)",
                       loc="left", fontsize=12, fontweight="bold", color="#7a4a1e")
    cbar = fig.colorbar(im, ax=list(axes), pad=0.015, fraction=0.03, aspect=40)
    cbar.set_label("Pressure (mmHg), without-filter branch")

    fig.suptitle(f"Colon pipeline (phase → waveform → ICC drive → total drive → "
                 f"muscle activation → wall/pressure) — {PHENOTYPE.upper()}",
                 fontsize=13, y=0.997)
    time_text = fig.text(0.5, 0.975, "", ha="center", fontsize=12, fontweight="bold")

    def update(i):
        k = frame_idx[i]
        line_phi.set_ydata(Phi_s[k])
        line_W.set_ydata(W_s[k])
        line_D.set_ydata(D_s[k])
        line_Ef.set_ydata(Ef[k]); line_En.set_ydata(En[k])
        line_Mf.set_ydata(Mf[k]); line_Mn.set_ydata(Mn[k])
        canvas = build_canvas(radius_n[k], Pn[k], y_grid, n_show)
        im.set_data(canvas)
        outline_top_n.set_ydata(radius_n[k]); outline_bot_n.set_ydata(-radius_n[k])
        outline_top_f.set_ydata(radius_f[k]); outline_bot_f.set_ydata(-radius_f[k])
        time_text.set_text(f"t = {t_rec[k]:5.1f} s")
        return (line_phi, line_W, line_D, line_Ef, line_En, line_Mf, line_Mn,
                im, outline_top_n, outline_bot_n, outline_top_f, outline_bot_f, time_text)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                   blit=False, interval=1000/FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000)
    ani.save(OUT_FILE, writer=writer, dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    t_rec, Phi_rec, W_rec, D_ICC_rec, geom = run_phase_and_drive()

    E_filt, M_filt, A_filt, P_filt = run_branch(D_ICC_rec, geom, use_lowpass=True)
    E_nof, M_nof, A_nof, P_nof = run_branch(D_ICC_rec, geom, use_lowpass=False)

    print(f"WITH filter    -> max pressure = {P_filt.max():.1f} mmHg, max E = {E_filt.max():.3f}, max M = {M_filt.max():.3f}")
    print(f"WITHOUT filter -> max pressure = {P_nof.max():.1f} mmHg, max E = {E_nof.max():.3f}, max M = {M_nof.max():.3f}")

    make_animation(t_rec, Phi_rec, W_rec, D_ICC_rec,
                   E_filt, M_filt, A_filt, P_filt,
                   E_nof, M_nof, A_nof, P_nof, geom)
    print(f"Animation saved to {OUT_FILE}")
