"""
Whole-colon pressure traces for three phenotypes (Healthy, IBS-D, STC).

State per segment (manuscript):
  phi_s(t)     phase oscillator
  E_s(t)       excitatory drive
  E*_s(t)      filtered excitatory drive
  M_s(t)       muscle activation
  A_s(t)       normalized cross-sectional area
  p_s(t)       pressure

Regions:
  HAPC (proximal): segments 1..N_HAPC
  CMP  (distal):   segments N_HAPC+1..N

Rectosigmoid brake:
  Pressure-sensing field over segments [40,65] (1-based, inclusive) controls a binary
  distal gate that suppresses distal excitation when pressure is high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS (case selection)
# =============================================================================
PHENOTYPE = "healthy"
SEED = 171

OUT_TXT = None
MAKE_PLOT = True


# =============================================================================
# CASE-SPECIFIC PARAMETERS (Table: cases)
# =============================================================================
ph = PHENOTYPE.strip().lower()

if ph == "healthy":
    a_hapc = 0.77          # [-] HAPC pacemaker amplitude
    brake_enabled = True   # [-] rectosigmoid brake active
    k_hapc = 2.0           # [rad/s] coupling strength (HAPC)
    k_cmp = 2.0            # [rad/s] coupling strength (CMP)
    v_hapc_cmps = 1.2      # [cm/s] preferred HAPC propagation speed
    sigma_phi = 0.0        # [rad/sqrt(s)] phase noise amplitude

elif ph == "ibsd":
    a_hapc = 0.70          # [-]
    brake_enabled = False  # [-] brake disabled (Gamma_CMP ≡ 1)
    k_hapc = 2.0           # [rad/s]
    k_cmp = 2.0            # [rad/s]
    v_hapc_cmps = 5.0      # [cm/s]
    sigma_phi = 0.0        # [rad/sqrt(s)]

elif ph == "stc":
    a_hapc = 0.62          # [-]
    brake_enabled = True   # [-]
    k_hapc = 0.2           # [rad/s] weak coupling
    k_cmp = 0.2            # [rad/s] weak coupling
    v_hapc_cmps = 1.1      # [cm/s]
    sigma_phi = 0.5        # [rad/sqrt(s)] added noise

else:
    raise ValueError("PHENOTYPE must be one of: 'healthy', 'ibsd', 'stc'")


# =============================================================================
# MODEL PARAMETERS (Table: waveform)
# =============================================================================
@dataclass(frozen=True)
class Geometry:
    L_cm: float = 150.0   # [cm] total colon length
    N: int = 75           # [-] number of segments
    N_HAPC: int = 50      # [-] proximal boundary (splenic flexure index in chain)

    @property
    def dx_cm(self) -> float:
        return self.L_cm / self.N


@dataclass(frozen=True)
class WaveformParams:
    phi_duty: float = 0.10  # [-] duty fraction of 2π
    gamma: float = 2.0      # [-] upstroke sharpness
    k_w: float = 8.0        # [-] decay constant


@dataclass(frozen=True)
class TubeLawParams:
    k_c: float = 6.0      # [-] contraction curve steepness
    M50: float = 0.30     # [-] contraction curve midpoint
    beta_C: float = 0.80  # [-] max fractional area reduction
    p_ref: float = 1.0    # [mmHg] reference pressure
    E_w: float = 25.0     # [mmHg] wall stiffness
    n_p: float = 1.3      # [-] tube law exponent


@dataclass(frozen=True)
class MuscleParams:
    tau_E: float = 1.0     # [s] excitatory filter constant
    tau_on: float = 1.0    # [s] contraction time constant
    tau_off: float = 1.2   # [s] relaxation time constant


@dataclass(frozen=True)
class ENSParams:
    alpha_asc: float = 0.10   # [-] ascending (oral) reflex gain
    alpha_desc: float = 0.08  # [-] descending (anal) reflex gain


@dataclass(frozen=True)
class OscillatorHealthyBaseline:
    f_hapc_cpm: float = 0.4  # [cpm] HAPC pacemaker frequency
    f_cmp_cpm: float = 3.0   # [cpm] CMP pacemaker frequency
    a_cmp: float = 0.75      # [-] CMP pacemaker amplitude
    alpha_spread_hapc: float = 0.10  # [-] proximal spread coefficient
    alpha_spread_cmp: float = 0.05   # [-] distal spread coefficient
    v_cmp_cmps: float = 5.0  # [cm/s] target CMP propagation speed


@dataclass(frozen=True)
class BrakeParams:
    F_sense_1based: Tuple[int, int] = (40, 65)  # [-] sensing field (1-based inclusive)
    p_on: float = 50.0   # [mmHg] brake ON threshold
    p_off: float = 20.0  # [mmHg] brake OFF threshold
    tau_P: float = 1.2   # [s] pressure low-pass constant

    @property
    def F_sense_idx(self) -> np.ndarray:
        lo, hi = self.F_sense_1based
        return np.arange(lo - 1, hi)


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.05     # [s] time step
    Tsim: float = 150.0  # [s] total simulated time

    @property
    def n_run(self) -> int:
        return int(round(self.Tsim / self.dt))


@dataclass(frozen=True)
class PlotParams:
    center_traces: bool = True         # [-]
    trace_gain: float = 1.2            # [-]
    offset_fixed_mmHg: float = 20.0    # [mmHg]
    trace_linewidth: float = 0.75      # [-]
    pad_top_offsets: float = 0.5       # [-]
    pad_bottom_offsets: float = 0.5    # [-]
    aspect_traces: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # [-]


# =============================================================================
# ICC waveform W_s(phi_s): skewed upstroke + exponential decay
# =============================================================================
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


# =============================================================================
# Smooth muscle + tube law: M -> C(M) -> A(M) -> p(M)
# =============================================================================
def contraction_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-tube.k_c * (M - tube.M50)))


def area_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    C = contraction_from_M(M, tube)
    return 1.0 - tube.beta_C * C


def pressure_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    A = area_from_M(M, tube)
    return tube.p_ref + tube.E_w * ((1.0 / A) ** tube.n_p - 1.0)


# =============================================================================
# Trace plotting (stacked equal spacing)
# =============================================================================
def plot_traces_equal_spacing(t: np.ndarray, P: np.ndarray, title: str, pp: PlotParams) -> None:
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
        ax.plot(t, T[:, s] + y_base[s], linewidth=pp.trace_linewidth, color="black")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(
        y_base[-1] - pp.pad_bottom_offsets * offset_dy,
        y_base[0] + pp.pad_top_offsets * offset_dy,
    )
    xticks_req = [50, 100]
    xticks_in = [x for x in xticks_req if t[0] <= x <= t[-1]]
    ax.set_xticks(xticks_in)
    ax.set_xticklabels([str(int(x)) for x in xticks_in])
    step = max(1, int(np.ceil(nsens / 5)))
    idx = np.arange(0, nsens, step)
    pos = y_base[idx]
    order = np.argsort(pos)
    ax.set_yticks(pos[order])
    ax.set_yticklabels([str(int(i + 1)) for i in idx[order]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Segment #")
    ax.set_title(title)
    try:
        ax.set_box_aspect(pp.aspect_traces[2] / pp.aspect_traces[0])
    except Exception:
        pass
    ax.grid(False)
    fig.tight_layout()
    plt.show()


# =============================================================================
# Main simulation (Euler + Euler-Maruyama for phase noise)
# =============================================================================
def run_traces() -> Tuple[np.ndarray, np.ndarray]:
    geom = Geometry()
    base = OscillatorHealthyBaseline()
    wf = WaveformParams()
    ens = ENSParams()
    mus = MuscleParams()
    tube = TubeLawParams()
    brk = BrakeParams()
    sim = SimParams()
    pp = PlotParams()

    if SEED is not None:
        np.random.seed(SEED)

    # Intrinsic frequencies ω_kappa = 2π f_kappa (f in Hz)
    f_hapc_hz = base.f_hapc_cpm / 60.0
    f_cmp_hz = base.f_cmp_cpm / 60.0
    omega_hapc = 2.0 * np.pi * f_hapc_hz
    omega_cmp = 2.0 * np.pi * f_cmp_hz

    # Per-segment phase lags δ_kappa encoding preferred propagation speeds
    dx = geom.dx_cm
    delta_hapc = -(omega_hapc / (v_hapc_cmps / dx))
    delta_cmp = -(omega_cmp / (base.v_cmp_cmps / dx))

    N = geom.N
    N_HAPC = geom.N_HAPC

    out_txt = OUT_TXT if OUT_TXT is not None else f"{ph}_pressure_traces.txt"

    # Phase states for proximal (HAPC) and distal (CMP) subnetworks
    phi_H = np.zeros(N, dtype=float)
    phi_C = np.zeros(N, dtype=float)

    # Initial conditions: proximal linear phase gradient, distal uniform phase
    phi_H[:N_HAPC] = np.arange(N_HAPC) * delta_hapc
    phi_C[N_HAPC:] = 0.0

    # Initialize filtered excitation and muscle activation consistent with ICC waveform
    E_star = np.zeros(N, dtype=float)
    M = np.zeros(N, dtype=float)

    W_H0 = icc_waveform_skewed(phi_H[:N_HAPC], wf)
    W_C0 = icc_waveform_skewed(phi_C[N_HAPC:], wf)

    E_star[:N_HAPC] = a_hapc * W_H0
    E_star[N_HAPC:] = E_star[N_HAPC:] + base.a_cmp * W_C0
    M[:] = E_star

    # Pressure low-pass (only used for brake sensing)
    p_lp = pressure_from_M(M, tube)
    cmp_enabled = True  # Gamma_CMP(t) == 1 when True, 0 when False

    P_rec = np.zeros((sim.n_run, N), dtype=float)
    t_rec = np.zeros(sim.n_run, dtype=float)

    t_all = np.arange(sim.n_run) * sim.dt

    for k, t in enumerate(t_all):

        # Euler-Maruyama noise scale: sigma_phi * sqrt(dt)
        sd = sigma_phi * np.sqrt(sim.dt)

        dphi_H = np.zeros(N, dtype=float)
        dphi_C = np.zeros(N, dtype=float)

        wdt_hapc = omega_hapc * sim.dt
        wdt_cmp = omega_cmp * sim.dt

        kdt_hapc = k_hapc * sim.dt
        kdt_cmp = k_cmp * sim.dt

        # Nearest-neighbor coupling: HAPC anterograde (s-1), CMP retrograde (s+1)
        dphi_H[0] = wdt_hapc
        for s in range(1, N_HAPC):
            s_neigh = s - 1
            dphi_H[s] = wdt_hapc + kdt_hapc * np.sin(phi_H[s_neigh] + delta_hapc - phi_H[s])

        dphi_C[N - 1] = wdt_cmp
        for s in range(N - 2, N_HAPC - 1, -1):
            s_neigh = s + 1
            dphi_C[s] = wdt_cmp + kdt_cmp * np.sin(phi_C[s_neigh] + delta_cmp - phi_C[s])

        # Phase noise (applied within each subnetwork)
        if ph in {"healthy", "ibsd", "stc"}:
            phi_H[:N_HAPC] += sd * np.random.randn(N_HAPC)
            phi_C[N_HAPC:] += sd * np.random.randn(N - N_HAPC)

        phi_H += dphi_H
        phi_C += dphi_C

        # Rectosigmoid brake: pressure-gated binary distal switch (hysteresis p_on/p_off)
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

        # ICC-derived myogenic drive with local spread
        D_ICC = np.zeros(N, dtype=float)

        W_H = icc_waveform_skewed(phi_H, wf)
        base_H = a_hapc * W_H[:N_HAPC]
        D_ICC[:N_HAPC] = base_H
        D_ICC[1:N_HAPC] += base.alpha_spread_hapc * base_H[:-1]

        W_C = icc_waveform_skewed(phi_C, wf)
        base_C = base.a_cmp * W_C[N_HAPC:]
        D_ICC[N_HAPC:] += base_C
        D_ICC[N_HAPC:N - 1] += base.alpha_spread_cmp * base_C[1:]

        # ENS drive from area gradient: ascending excitation, descending inhibition
        A_prev = area_from_M(M, tube)
        DeltaA = A_prev[1:] - A_prev[:-1]

        E_ENS = np.zeros(N, dtype=float)
        E_ENS[:-1] += ens.alpha_asc * np.maximum(DeltaA, 0.0)
        E_ENS[1:] -= ens.alpha_desc * np.maximum(DeltaA, 0.0)

        # Total excitatory drive; distal gating when brake is ON
        E_total = D_ICC + E_ENS
        if brake_enabled and (not cmp_enabled):
            E_total[N_HAPC:] = 0.0

        # Low-pass filter of excitation: dE*/dt = (E - E*)/tau_E
        aE = sim.dt / mus.tau_E
        E_star[:] = (1.0 - aE) * E_star + aE * E_total

        # Asymmetric muscle activation kinetics (tau_on vs tau_off)
        tau_s = np.where(E_star >= M, mus.tau_on, mus.tau_off)
        aM = sim.dt / tau_s
        M[:] = (1.0 - aM) * M + aM * E_star

        # Record pressure via nonlinear tube law
        P_rec[k, :] = pressure_from_M(M, tube)
        t_rec[k] = t

    maxP = float(np.max(P_rec))
    print(f"Max pressure across all sensors ({ph.upper()} model): {maxP:.3f} mmHg")

    header = "Time_s" + "".join([f"\tSeg_{i}" for i in range(1, N + 1)])
    data_out = np.column_stack([t_rec, P_rec])

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(header + "\n")
    with open(out_txt, "ab") as f:
        np.savetxt(f, data_out, delimiter="\t", fmt="%.5f")

    print(f"Pressure trace data written to: {out_txt}")

    if MAKE_PLOT:
        plot_traces_equal_spacing(
            t_rec,
            P_rec,
            title=f"Pressure traces — {ph.upper()}",
            pp=pp,
        )

    return t_rec, P_rec


if __name__ == "__main__":
    run_traces()