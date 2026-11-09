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

from dataclasses import dataclass, fields, replace
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings (case selection)
# =============================================================================
PHENOTYPE = "healthy"
SEED = 171

OUT_TXT = None
MAKE_PLOT = True


# =============================================================================
# Case-specific parameters (Table: cases)
# =============================================================================
@dataclass(frozen=True)
class PhenotypeParams:
    a_hapc: float        # [-] HAPC pacemaker amplitude
    brake_enabled: bool  # [-] rectosigmoid brake active
    k_hapc: float        # [rad/s] coupling strength (HAPC)
    k_cmp: float         # [rad/s] coupling strength (CMP)
    v_hapc_cmps: float   # [cm/s] preferred HAPC propagation speed
    sigma_phi: float     # [rad/sqrt(s)] phase noise amplitude


def phenotype_params(phenotype: str) -> PhenotypeParams:
    """Return the case-specific parameters for 'healthy', 'ibsd', or 'stc' (case-insensitive)."""
    label = phenotype.strip().lower()
    if label == "healthy":
        return PhenotypeParams(a_hapc=0.77, brake_enabled=True, k_hapc=2.0, k_cmp=2.0,
                                v_hapc_cmps=1.2, sigma_phi=0.0)
    elif label == "ibsd":
        # brake disabled (Gamma_CMP ≡ 1)
        return PhenotypeParams(a_hapc=0.70, brake_enabled=False, k_hapc=2.0, k_cmp=2.0,
                                v_hapc_cmps=5.0, sigma_phi=0.0)
    elif label == "stc":
        # weak coupling, added noise
        return PhenotypeParams(a_hapc=0.62, brake_enabled=True, k_hapc=0.2, k_cmp=0.2,
                                v_hapc_cmps=1.1, sigma_phi=0.5)
    else:
        raise ValueError("PHENOTYPE must be one of: 'healthy', 'ibsd', 'stc'")


# =============================================================================
# Model parameters (Table: waveform)
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
    center_traces: bool = True         # [-] subtract each trace's median before stacking
    trace_gain: float = 1.2            # [-] vertical scaling of each trace
    offset_fixed_mmHg: float = 20.0    # [mmHg] vertical spacing between stacked traces
    trace_linewidth: float = 0.75      # [pt]
    trace_color: str = "black"
    pad_top_offsets: float = 0.5       # [-] headroom above the top trace, in trace spacings
    pad_bottom_offsets: float = 0.5    # [-] footroom below the bottom trace, in trace spacings
    box_aspect: float = 1.0            # [-] axes height / width
    figsize_in: Tuple[float, float] = (12.0, 4.0)  # [in]
    dpi: int = 120
    xticks_s: Tuple[float, ...] = (50.0, 100.0)    # [s] time ticks (dropped if outside the run)
    n_yticks: int = 5                  # [-] approximate number of segment labels


class TraceResults(NamedTuple):
    """Recorded time histories from run_traces; each array is (n_steps, N) except t."""
    t: np.ndarray        # [s] time
    phi: np.ndarray      # [rad] pacemaker phase, wrapped to [0, 2π)
    W: np.ndarray        # [-] ICC waveform
    D_ICC: np.ndarray    # [-] ICC-derived myogenic drive
    E_ENS: np.ndarray    # [-] enteric reflex drive
    E: np.ndarray        # [-] total excitatory drive
    M: np.ndarray        # [-] muscle activation
    A: np.ndarray        # [-] normalized cross-sectional area
    P: np.ndarray        # [mmHg] pressure
    geom: "Geometry"


ParameterGroups = Tuple[PhenotypeParams, OscillatorHealthyBaseline, WaveformParams, ENSParams,
                        MuscleParams, BrakeParams]


def parameter_groups(phenotype: str,
                     overrides: Optional[Dict[str, float]] = None) -> ParameterGroups:
    """Parameter groups that run_traces uses, with ``overrides`` applied by field name.

    Tube-law, geometry, and time-step parameters are not overridable here.
    """
    groups = [phenotype_params(phenotype), OscillatorHealthyBaseline(), WaveformParams(),
              ENSParams(), MuscleParams(), BrakeParams()]
    for name, value in (overrides or {}).items():
        owners = [i for i, g in enumerate(groups) if name in {f.name for f in fields(g)}]
        if len(owners) != 1:
            raise KeyError(f"Unknown parameter for override: {name}")
        groups[owners[0]] = replace(groups[owners[0]], **{name: value})
    return tuple(groups)


# =============================================================================
# Phase oscillator chain (Eq. 1): deterministic Euler increment
# =============================================================================
def phase_chain_increment(phi: np.ndarray, omega: float, k: float, delta: float,
                          retrograde: bool, dt: float) -> np.ndarray:
    """Deterministic Euler increment of Eq. (1) for one nearest-neighbor chain.

    The leading oscillator (first element if anterograde, last if retrograde)
    runs at its intrinsic frequency; every other oscillator is pulled toward
    its upstream neighbor's phase plus the lag ``delta``.
    """
    dphi = np.full_like(phi, omega * dt, dtype=float)
    if retrograde:
        dphi[:-1] += k * dt * np.sin(phi[1:] + delta - phi[:-1])
    else:
        dphi[1:] += k * dt * np.sin(phi[:-1] + delta - phi[1:])
    return dphi


# =============================================================================
# ICC waveform W_s(phi_s): skewed upstroke + exponential decay
# =============================================================================
def icc_waveform_skewed(phi: np.ndarray, wf: WaveformParams) -> np.ndarray:
    """ICC waveform W(phi): power-law upstroke over the duty fraction, then exponential decay."""
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
    """Sigmoidal contraction level C(M) in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-tube.k_c * (M - tube.M50)))


def area_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    """Normalized cross-sectional area A = 1 - beta_C * C(M)."""
    C = contraction_from_M(M, tube)
    return 1.0 - tube.beta_C * C


def pressure_from_A(A: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    """Tube law: pressure [mmHg] from normalized area A."""
    return tube.p_ref + tube.E_w * ((1.0 / A) ** tube.n_p - 1.0)


def pressure_from_M(M: np.ndarray, tube: TubeLawParams) -> np.ndarray:
    """Tube law: pressure [mmHg] from muscle activation M."""
    return pressure_from_A(area_from_M(M, tube), tube)


# =============================================================================
# Trace plotting (stacked equal spacing)
# =============================================================================
def plot_traces_equal_spacing(t: np.ndarray, P: np.ndarray, title: str, pp: PlotParams) -> None:
    """Stack one pressure trace per segment, segment 1 (proximal) at the top, as in manometry."""
    T = P.copy()
    if pp.center_traces:
        T = T - np.nanmedian(T, axis=0, keepdims=True)
    T *= pp.trace_gain
    nsens = T.shape[1]
    offset_dy = max(1.0, pp.offset_fixed_mmHg)  # guard against zero/negative spacing
    y_base = (nsens - (np.arange(nsens) + 1)) * offset_dy
    fig = plt.figure(figsize=pp.figsize_in, dpi=pp.dpi)
    ax = fig.add_subplot(111)
    for s in range(nsens):
        ax.plot(t, T[:, s] + y_base[s], linewidth=pp.trace_linewidth, color=pp.trace_color)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(
        y_base[-1] - pp.pad_bottom_offsets * offset_dy,
        y_base[0] + pp.pad_top_offsets * offset_dy,
    )
    xticks_in = [x for x in pp.xticks_s if t[0] <= x <= t[-1]]
    ax.set_xticks(xticks_in)
    ax.set_xticklabels([str(int(x)) for x in xticks_in])
    step = max(1, int(np.ceil(nsens / pp.n_yticks)))
    idx = np.arange(0, nsens, step)
    pos = y_base[idx]
    order = np.argsort(pos)
    ax.set_yticks(pos[order])
    ax.set_yticklabels([str(int(i + 1)) for i in idx[order]])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Segment [-]")
    ax.set_title(title)
    ax.set_box_aspect(pp.box_aspect)
    ax.grid(False)
    fig.tight_layout()
    plt.show()


# =============================================================================
# Main simulation (Euler + Euler-Maruyama for phase noise)
# =============================================================================
def run_traces(phenotype: str = PHENOTYPE, write_output: bool = True,
               make_plot: bool = MAKE_PLOT, seed: Optional[int] = SEED,
               out_txt: Optional[str] = OUT_TXT, gate_out: Optional[list] = None,
               overrides: Optional[Dict[str, float]] = None,
               random_phase_seed: Optional[int] = None,
               dt: Optional[float] = None) -> TraceResults:
    """Simulate all segments for one phenotype and return the recorded time histories.

    ``seed`` seeds NumPy's global RNG (only affects phenotypes with phase noise).
    ``out_txt`` defaults to '<phenotype>_pressure_traces.txt' in the current directory.
    If ``gate_out`` is a list, the distal brake gate Gamma_CMP(t) (1 = open,
    0 = brake ON) is appended to it as an array, one value per time step.
    ``overrides`` maps parameter field names (e.g. ``"k_w"``, ``"a_hapc"``) to
    new values; it is used by the sensitivity analysis.
    If ``random_phase_seed`` is given, all initial phases are drawn uniformly
    from [0, 2*pi) with that seed instead of the default initial conditions
    (used to check that the phase-locked state does not depend on them).
    ``dt`` overrides the time step [s] (used for the time-step convergence check).
    """
    ph = phenotype.strip().lower()
    case, base, wf, ens, mus, brk = parameter_groups(phenotype, overrides)

    geom = Geometry()
    tube = TubeLawParams()
    sim = SimParams() if dt is None else SimParams(dt=dt)
    pp = PlotParams()

    if seed is not None:
        np.random.seed(seed)

    # Intrinsic frequencies ω_kappa = 2π f_kappa (f in Hz)
    f_hapc_hz = base.f_hapc_cpm / 60.0
    f_cmp_hz = base.f_cmp_cpm / 60.0
    omega_hapc = 2.0 * np.pi * f_hapc_hz
    omega_cmp = 2.0 * np.pi * f_cmp_hz

    # Per-segment phase lags δ_kappa encoding preferred propagation speeds
    dx = geom.dx_cm
    delta_hapc = -(omega_hapc / (case.v_hapc_cmps / dx))
    delta_cmp = -(omega_cmp / (base.v_cmp_cmps / dx))

    N = geom.N
    N_HAPC = geom.N_HAPC

    if out_txt is None:
        out_txt = f"{ph}_pressure_traces.txt"

    # Phase states: only phi_H[:N_HAPC] (proximal HAPC) and phi_C[N_HAPC:] (distal CMP) are used
    phi_H = np.zeros(N, dtype=float)
    phi_C = np.zeros(N, dtype=float)

    # Initial conditions: proximal linear phase gradient, distal uniform phase (zero)
    phi_H[:N_HAPC] = np.arange(N_HAPC) * delta_hapc
    if random_phase_seed is not None:
        rng_ic = np.random.default_rng(random_phase_seed)
        phi_H[:N_HAPC] = rng_ic.uniform(0.0, 2.0 * np.pi, N_HAPC)
        phi_C[N_HAPC:] = rng_ic.uniform(0.0, 2.0 * np.pi, N - N_HAPC)

    # Initialize filtered excitation and muscle activation consistent with ICC waveform
    E_star = np.zeros(N, dtype=float)
    E_star[:N_HAPC] = case.a_hapc * icc_waveform_skewed(phi_H[:N_HAPC], wf)
    E_star[N_HAPC:] = base.a_cmp * icc_waveform_skewed(phi_C[N_HAPC:], wf)
    M = E_star.copy()

    # Pressure low-pass (only used for brake sensing)
    p_lp = pressure_from_M(M, tube)
    cmp_enabled = True  # Gamma_CMP(t) == 1 when True, 0 when False

    Phi_rec = np.zeros((sim.n_run, N), dtype=float)
    W_rec = np.zeros((sim.n_run, N), dtype=float)
    D_ICC_rec = np.zeros((sim.n_run, N), dtype=float)
    ENS_rec = np.zeros((sim.n_run, N), dtype=float)
    E_rec = np.zeros((sim.n_run, N), dtype=float)
    M_rec = np.zeros((sim.n_run, N), dtype=float)
    A_rec = np.zeros((sim.n_run, N), dtype=float)
    P_rec = np.zeros((sim.n_run, N), dtype=float)
    t_rec = np.arange(sim.n_run) * sim.dt
    gate_rec = np.ones(sim.n_run, dtype=float)

    # Per-step constants
    sd = case.sigma_phi * np.sqrt(sim.dt)  # Euler-Maruyama noise scale: sigma_phi * sqrt(dt)
    aP = sim.dt / brk.tau_P
    aE = sim.dt / mus.tau_E

    for k in range(sim.n_run):

        # Nearest-neighbor coupling: HAPC anterograde (s-1), CMP retrograde (s+1)
        dphi_H = np.zeros(N, dtype=float)
        dphi_C = np.zeros(N, dtype=float)
        dphi_H[:N_HAPC] = phase_chain_increment(phi_H[:N_HAPC], omega_hapc, case.k_hapc,
                                                delta_hapc, retrograde=False, dt=sim.dt)
        dphi_C[N_HAPC:] = phase_chain_increment(phi_C[N_HAPC:], omega_cmp, case.k_cmp,
                                                delta_cmp, retrograde=True, dt=sim.dt)

        # Phase noise (applied within each subnetwork)
        if sd > 0.0:
            phi_H[:N_HAPC] += sd * np.random.randn(N_HAPC)
            phi_C[N_HAPC:] += sd * np.random.randn(N - N_HAPC)

        phi_H += dphi_H
        phi_C += dphi_C

        # Area and pressure from the muscle state at the start of this step
        A_prev = area_from_M(M, tube)

        # Rectosigmoid brake: pressure-gated binary distal switch (hysteresis p_on/p_off)
        if case.brake_enabled:
            p_inst = pressure_from_A(A_prev, tube)
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

        W_H = icc_waveform_skewed(phi_H[:N_HAPC], wf)
        base_H = case.a_hapc * W_H
        D_ICC[:N_HAPC] = base_H
        D_ICC[1:N_HAPC] += base.alpha_spread_hapc * base_H[:-1]

        W_C = icc_waveform_skewed(phi_C[N_HAPC:], wf)
        base_C = base.a_cmp * W_C
        D_ICC[N_HAPC:] = base_C
        D_ICC[N_HAPC:N - 1] += base.alpha_spread_cmp * base_C[1:]

        # ENS drive from area gradient: ascending excitation, descending inhibition
        DeltaA_pos = np.maximum(A_prev[1:] - A_prev[:-1], 0.0)

        E_ENS = np.zeros(N, dtype=float)
        E_ENS[:-1] += ens.alpha_asc * DeltaA_pos
        E_ENS[1:] -= ens.alpha_desc * DeltaA_pos

        # Total excitatory drive; distal gating when brake is ON
        E_total = D_ICC + E_ENS
        if not cmp_enabled:
            E_total[N_HAPC:] = 0.0

        # Low-pass filter of excitation: dE*/dt = (E - E*)/tau_E
        E_star[:] = (1.0 - aE) * E_star + aE * E_total

        # Asymmetric muscle activation kinetics (tau_on vs tau_off)
        tau_s = np.where(E_star >= M, mus.tau_on, mus.tau_off)
        aM = sim.dt / tau_s
        M[:] = (1.0 - aM) * M + aM * E_star

        # Record intermediate signals and pressure via nonlinear tube law
        Phi_rec[k, :N_HAPC] = np.mod(phi_H[:N_HAPC], 2 * np.pi)
        Phi_rec[k, N_HAPC:] = np.mod(phi_C[N_HAPC:], 2 * np.pi)
        W_rec[k, :N_HAPC] = W_H
        W_rec[k, N_HAPC:] = W_C
        D_ICC_rec[k, :] = D_ICC
        ENS_rec[k, :] = E_ENS
        E_rec[k, :] = E_total
        M_rec[k, :] = M
        A_rec[k, :] = area_from_M(M, tube)
        P_rec[k, :] = pressure_from_A(A_rec[k], tube)
        gate_rec[k] = 1.0 if cmp_enabled else 0.0

    maxP = float(np.max(P_rec))
    print(f"Max pressure across all sensors ({ph.upper()} model): {maxP:.3f} mmHg")

    if write_output:
        header = "Time_s" + "".join([f"\tSeg_{i}" for i in range(1, N + 1)])
        data_out = np.column_stack([t_rec, P_rec])
        np.savetxt(out_txt, data_out, delimiter="\t", fmt="%.5f", header=header, comments="",
                   encoding="utf-8")
        print(f"Pressure trace data written to: {out_txt}")

    if make_plot:
        plot_traces_equal_spacing(
            t_rec,
            P_rec,
            title=f"Pressure traces — {ph.upper()}",
            pp=pp,
        )

    if gate_out is not None:
        gate_out.append(gate_rec)

    return TraceResults(t_rec, Phi_rec, W_rec, D_ICC_rec, ENS_rec, E_rec, M_rec, A_rec, P_rec,
                        geom)


if __name__ == "__main__":
    run_traces()
