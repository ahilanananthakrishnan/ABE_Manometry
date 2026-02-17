"""
stc_traces.py

STC PHENOTYPE — whole-colon pressure traces

STC features:
- Very weak coupling (k_κ small) and zero spread (α_spread,κ = 0)
- Phase drift/noise σ_ϕ > 0 applied directly to phases each step
- Rectosigmoid brake ACTIVE (wide-field, pressure-only, low-pass filtered gate)
- When brake ON (Γ_CMP=0): suspend distal phase evolution AND gate distal total drive

Exports: stc_pressure_traces.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class Geometry:
    L_cm: float = 150.0
    N: int = 75
    N_HAPC: int = 50  # proximal: 1..50 (MATLAB) == 0..49 (Python)

    @property
    def dx_cm(self) -> float:
        return self.L_cm / self.N

    @property
    def s0_cmp(self) -> int:
        return self.N_HAPC

    @property
    def F_sense(self) -> np.ndarray:
        return np.arange(39, 65, dtype=int)


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
class OscillatorParamsSTC:
    # Frequencies (cpm)
    f_hapc_cpm: float = 0.4
    f_cmp_cpm: float = 3.0

    # Coupling strengths (STC weak)
    k_hapc: float = 0.01
    k_cmp: float = 0.01

    # ICC amplitude
    a_hapc: float = 0.60
    a_cmp: float = 0.60

    # Spread (none)
    alpha_spread_hapc: float = 0.00
    alpha_spread_cmp: float = 0.00

    # Preferred speeds (cm/s)
    v_hapc_cmps: float = 1.10
    v_cmp_cmps: float = 5.0

    # Phase drift amplitude σ_ϕ (rad/sqrt(s))
    sigma_phi: float = 0.5


@dataclass(frozen=True)
class BrakeParams:
    p_on: float = 35.0
    p_off: float = 22.0
    t_ref: float = 5.0
    tau_P: float = 1.2


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.05
    Tsim: float = 140.0
    warmup: float = 10.0

    @property
    def n_warm(self) -> int:
        return int(round(self.warmup / self.dt))

    @property
    def n_run(self) -> int:
        return int(round(self.Tsim / self.dt))

    @property
    def n_tot(self) -> int:
        return self.n_warm + self.n_run


@dataclass(frozen=True)
class PlotParams:
    center_traces: bool = True
    trace_gain: float = 1.2
    offset_fixed_mmHg: float = 14.0
    trace_linewidth: float = 0.75
    pad_top_offsets: float = 0.6
    pad_bottom_offsets: float = 0.6
    aspect_traces: Tuple[float, float, float] = (1.0, 1.0, 1.0)


# =============================================================================
# Helpers
# =============================================================================

def icc_waveform_skewed(phi: np.ndarray, wf: WaveformParams) -> np.ndarray:
    twopi = 2.0 * np.pi
    phi_wrapped = np.mod(phi, twopi)
    thr = wf.phi_duty * twopi

    W = np.zeros_like(phi_wrapped, dtype=float)

    mask_up = phi_wrapped < thr
    if np.any(mask_up):
        W[mask_up] = (phi_wrapped[mask_up] / max(thr, np.finfo(float).eps)) ** wf.gamma

    mask_dn = ~mask_up
    if np.any(mask_dn):
        frac = (phi_wrapped[mask_dn] - thr) / max(twopi - thr, np.finfo(float).eps)
        W[mask_dn] = np.exp(-wf.k_w * frac)

    return W


def contraction_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-tube.k_c * (M - tube.M50)))


def area_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    C = contraction_from_M(M, tube)
    return 1.0 - tube.beta_C * C


def pressure_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    A = area_from_M(M, tube)
    A_safe = np.maximum(A, 1e-9)
    return tube.p_ref + tube.E_w * ((1.0 / A_safe) ** tube.n_p - 1.0)


# =============================================================================
# Plotting
# =============================================================================

def plot_traces_equal_spacing(
    t: np.ndarray,
    P: np.ndarray,
    title: str,
    pp: PlotParams,
) -> None:
    """Stacked trace plot."""
    T = P.copy()
    if pp.center_traces:
        T = T - np.nanmedian(T, axis=0, keepdims=True)
    T *= pp.trace_gain

    nsens = T.shape[1]
    offset_dy = max(1.0, pp.offset_fixed_mmHg)
    y_base = (nsens - (np.arange(nsens) + 1)) * offset_dy

    fig = plt.figure(figsize=(12, 4), dpi=120)
    ax = fig.add_subplot(111)

    for s in range(nsens):
        ax.plot(t, T[:, s] + y_base[s], linewidth=pp.trace_linewidth)

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(y_base[-1] - pp.pad_bottom_offsets * offset_dy,
                y_base[0] + pp.pad_top_offsets * offset_dy)

    xticks_req = [50, 100]
    xticks_in = [x for x in xticks_req if t[0] <= x <= t[-1]]
    ax.set_xticks(xticks_in)
    ax.set_xticklabels([str(int(x)) for x in xticks_in])

    step = max(1, int(np.ceil(nsens / 5)))
    idx = np.arange(0, nsens, step)
    pos = y_base[idx]
    order = np.argsort(pos)
    ax.set_yticks(pos[order])
    ax.set_yticklabels([str(int(i + 1)) for i in idx[order]])  # 1-based labels

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Segment #")
    ax.set_title(title)
    ax.grid(False)

    fig.tight_layout()
    plt.show()


# =============================================================================
# Main simulation
# =============================================================================

def stc_traces(
    seed: int = 171,  # MATLAB: rng(0171)
    out_txt: str = "stc_pressure_traces.txt",
    make_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run STC simulation.

    Brake active:
      - pressure LP in sensing field F, gate uses max(F)
      - when brake ON (Γ_CMP=0):
          (i) distal phase evolution suspended (dϕ/dt = 0 for s>=51)
          (ii) distal total drive gated: E_total[s>=51] = 0
    """
    geom = Geometry()
    osc = OscillatorParamsSTC()
    brk = BrakeParams()
    wf = WaveformParams()
    ens = ENSParams()
    mus = MuscleParams()
    tube = TubeLawParams()
    sim = SimParams()
    pp = PlotParams()

    np.random.seed(seed)

    # Frequencies (Hz) and ω (rad/s)
    f_hapc_hz = osc.f_hapc_cpm / 60.0
    f_cmp_hz = osc.f_cmp_cpm / 60.0
    omega_hapc = 2.0 * np.pi * f_hapc_hz
    omega_cmp = 2.0 * np.pi * f_cmp_hz

    # Per-segment phase lags δ_κ — keep exact MATLAB definition
    dx = geom.dx_cm
    delta_hapc = -(omega_hapc / max(osc.v_hapc_cmps / dx, 1e-9))
    delta_cmp = -(omega_cmp / max(osc.v_cmp_cmps / dx, 1e-9))

    N = geom.N
    N_HAPC = geom.N_HAPC
    s0_cmp = geom.s0_cmp
    F = geom.F_sense

    # --- Initial conditions ---
    phi_H = np.zeros(N, dtype=float)
    phi_C = np.zeros(N, dtype=float)

    phi_H[:N_HAPC] = np.arange(N_HAPC) * delta_hapc
    phi_C[s0_cmp:] = 0.0

    E_star = np.zeros(N, dtype=float)
    M = np.zeros(N, dtype=float)

    W_H0 = icc_waveform_skewed(phi_H[:N_HAPC], wf)
    W_C0 = icc_waveform_skewed(phi_C[s0_cmp:], wf)

    E_star[:N_HAPC] = osc.a_hapc * W_H0
    E_star[s0_cmp:] = E_star[s0_cmp:] + osc.a_cmp * W_C0
    M[:] = E_star

    # Brake state Γ_CMP(t)
    cmp_enabled = True          # Γ_CMP = 1 (brake OFF)
    cmp_pause_until = -np.inf

    # Pressure low-pass state p*(t) for all segments (only F used for gate)
    p_lp = pressure_from_M(M, tube)

    # Recording
    P_rec = np.zeros((sim.n_run, N), dtype=float)
    t_rec = np.zeros(sim.n_run, dtype=float)
    rec = 0

    t_all = np.arange(sim.n_tot) * sim.dt

    # --- Main loop ---
    for k, t in enumerate(t_all):
        # 1) Phase noise/drift (applied directly to phases)
        if osc.sigma_phi > 0:
            sd = osc.sigma_phi * np.sqrt(sim.dt)
            phi_H[:N_HAPC] += sd * np.random.randn(N_HAPC)
            phi_C[s0_cmp:] += sd * np.random.randn(N - s0_cmp)

        # 2) Phase evolution (nearest-neighbor coupling)
        dphi_H = np.zeros(N, dtype=float)
        dphi_C = np.zeros(N, dtype=float)

        wdt_hapc = omega_hapc * sim.dt
        wdt_cmp = omega_cmp * sim.dt

        dphi_H[0] = wdt_hapc
        for s in range(1, N_HAPC):
            s_neigh = s - 1
            dphi_H[s] = wdt_hapc + osc.k_hapc * np.sin((phi_H[s_neigh] + delta_hapc) - phi_H[s])

        # 3) Brake sensing (wide-field, pressure-only, low-pass filtered)
        p_inst = pressure_from_M(M, tube)
        aP = sim.dt / max(brk.tau_P, np.finfo(float).eps)
        p_lp = (1.0 - aP) * p_lp + aP * p_inst

        p_gate = float(np.max(p_lp[F]))

        if cmp_enabled and (p_gate >= brk.p_on):
            cmp_enabled = False
            cmp_pause_until = t + brk.t_ref
        if (not cmp_enabled) and (t >= cmp_pause_until) and (p_gate <= brk.p_off):
            cmp_enabled = True

        # Distal CMP: advance only if enabled, else suspend
        if cmp_enabled:
            dphi_C[s0_cmp] = wdt_cmp
            for s in range(s0_cmp + 1, N):
                s_neigh = s - 1
                dphi_C[s] = wdt_cmp + osc.k_cmp * np.sin((phi_C[s_neigh] + (-delta_cmp)) - phi_C[s])
        else:
            dphi_C[s0_cmp:] = 0.0

        phi_H += dphi_H
        phi_C += dphi_C

        # 4) ICC drive D^ICC (STC spread=0; CMP only if enabled)
        D_ICC = np.zeros(N, dtype=float)

        # Proximal ICC
        W_H = icc_waveform_skewed(phi_H[:N_HAPC], wf)
        D_ICC[:N_HAPC] = osc.a_hapc * W_H  # no spread

        # Distal ICC only if CMP enabled
        if cmp_enabled:
            W_C = icc_waveform_skewed(phi_C[s0_cmp:], wf)
            D_ICC[s0_cmp:] += osc.a_cmp * W_C

        # 5) ENS drive E^ENS from ΔA
        A_prev = area_from_M(M, tube)
        DeltaA = A_prev[1:] - A_prev[:-1]

        E_ENS = np.zeros(N, dtype=float)
        E_ENS[:-1] += ens.alpha_asc * np.maximum(DeltaA, 0.0)
        E_ENS[1:] -= ens.alpha_desc * np.maximum(DeltaA, 0.0)

        # 6) Total excitatory drive E_s with distal gating when brake ON
        E_total = D_ICC + E_ENS
        if not cmp_enabled:
            E_total[s0_cmp:] = 0.0  # Γ_CMP=0 gates distal total drive

        # 7) Filtering and muscle activation dynamics
        aE = sim.dt / max(mus.tau_E, np.finfo(float).eps)
        E_star = (1.0 - aE) * E_star + aE * E_total

        tau_s = np.where(E_star >= M, mus.tau_on, mus.tau_off)
        aM = sim.dt / np.maximum(tau_s, np.finfo(float).eps)
        M = (1.0 - aM) * M + aM * E_star

        # 8) Record after warmup
        if k >= sim.n_warm:
            P_rec[rec, :] = pressure_from_M(M, tube)
            t_rec[rec] = t - sim.warmup
            rec += 1
            if rec >= sim.n_run:
                break

    # --- Export ---
    maxP = float(np.max(P_rec))
    print(f"Max pressure across all sensors (STC model): {maxP:.3f} mmHg")

    header = "Time_s" + "".join([f"\tSeg_{i}" for i in range(1, N + 1)])
    data_out = np.column_stack([t_rec, P_rec])

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(header + "\n")
    with open(out_txt, "ab") as f:
        np.savetxt(f, data_out, delimiter="\t", fmt="%.5f")

    print(f"Pressure trace data written to: {out_txt}")

    # --- Plot ---
    if make_plot:
        plot_traces_equal_spacing(
            t_rec,
            P_rec,
            title="STC (model): stacked pressure traces",
            pp=pp,
        )

    return t_rec, P_rec


if __name__ == "__main__":
    stc_traces(seed=171, make_plot=True)
