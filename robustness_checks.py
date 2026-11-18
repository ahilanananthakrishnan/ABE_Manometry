"""
Robustness checks quoted in the manuscript text, computed from pressure_traces.py.

  1. Time-step convergence: dt is halved N_HALVINGS times. For the deterministic
     phenotypes (healthy, IBS-D) each run is compared with the finest one by the
     relative L2 difference of the pressure field, the peak pressure, the motility
     index, and the fraction of time the brake is ON. STC draws a different noise
     sequence at every dt, so it is compared through the mean +/- SD of the same
     outputs over REALIZATION_SEEDS.
  2. Initial phases: healthy runs started from random initial phases. Reports the
     peak pressure and the HAPC neighbor phase lag after the transient (the locked state).
  3. Noise realizations: STC, and healthy with sigma_phi = 0.1, over REALIZATION_SEEDS
     (peak pressure, mean +/- SD).
  4. Phase coherence: SD of the neighbor phase difference phi_{s+1} - phi_s (wrapped to
     (-pi, pi]) over all neighbor pairs and all time steps after TRANSIENT_S, per
     subnetwork, averaged over REALIZATION_SEEDS, for healthy coupling with sigma_phi =
     0.1 and 0.5, the STC case, and STC coupling without noise.

Author: Ahilan Anantha Krishnan
Units: length [cm], time [s], pressure [mmHg], phase [rad]
Dimension: 1D chain of colon segments

Usage:
    python robustness_checks.py
Output:
    printed tables and outputs/robustness_checks.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np

from plot_style import OUTPUT_DIR
from pressure_traces import SimParams, TraceResults, motility_index, run_traces

# =============================================================================
# User settings
# =============================================================================
CSV_PATH = os.path.join(OUTPUT_DIR, "robustness_checks.csv")
DETERMINISTIC = ("healthy", "ibsd")  # phenotypes without phase noise
N_HALVINGS = 3                     # [-] dt, dt/2, ..., dt/2**N_HALVINGS (check 1)
REALIZATION_SEEDS = range(10)      # [-] seeds of the noise realizations (checks 1, 3, 4)
INITIAL_PHASE_SEEDS = range(10)    # [-] seeds of the random initial phases (check 2)
TRANSIENT_S = 30.0                 # [s] discarded before measuring phase coherence (2, 4)
SIGMA_HEALTHY_CHECK = 0.1          # [rad/sqrt(s)] noise added to the healthy case (3, 4)
SIGMA_STC = 0.5                    # [rad/sqrt(s)] STC noise, with healthy coupling (check 4)


def run(phenotype: str, **kwargs) -> TraceResults:
    """One silent simulation without output files or plots."""
    return run_traces(phenotype, write_output=False, make_plot=False, verbose=False,
                      **kwargs)


def summary(res: TraceResults) -> np.ndarray:
    """Peak pressure [mmHg], motility index [mmHg s], and fraction of time the brake is ON."""
    return np.array([res.P.max(), motility_index(res), 1.0 - res.gate.mean()])


def relative_l2(coarse: TraceResults, fine: TraceResults, refinement: int) -> float:
    """Relative L2 difference of the pressure field, on the coarse run's time points.

    Row k of a run holds the state after k + 1 steps, so coarse row k matches fine row
    (k + 1) * refinement - 1.
    """
    ref = fine.P[refinement - 1::refinement]
    return float(np.sqrt(((coarse.P - ref) ** 2).sum() / (ref ** 2).sum()))


def neighbor_phase_sd(res: TraceResults, transient_s: float = 0.0) -> tuple[float, float]:
    """SD [rad] of the wrapped neighbor phase difference in the HAPC and CMP subnetworks."""
    n_hapc = res.geom.N_HAPC
    k0 = int(round(transient_s / (res.t[1] - res.t[0])))
    sds = []
    for phi in (res.phi[k0:, :n_hapc], res.phi[k0:, n_hapc:]):
        dphi = np.angle(np.exp(1j * np.diff(phi, axis=1)))  # wrap to (-pi, pi]
        sds.append(float(dphi.std()))
    return sds[0], sds[1]


def main() -> None:
    """Run all checks, print them, and write one CSV row per reported number."""
    rows: list[tuple[str, str, float]] = []   # (check, case, value)
    dts = [SimParams().dt / 2 ** m for m in range(N_HALVINGS + 1)]
    names = ("peak", "motility_index", "brake_on_fraction")

    print("1. Time-step convergence (peak [mmHg], motility index [mmHg s], brake-ON fraction)")
    for ph in DETERMINISTIC:
        runs = [run(ph, dt=dt) for dt in dts]
        print(f"   {ph}: dt [s], rel. L2 vs finest, peak, MI, brake")
        for m, (dt, res) in enumerate(zip(dts, runs)):
            l2 = relative_l2(res, runs[-1], 2 ** (N_HALVINGS - m))
            values = summary(res)
            print(f"     {dt:.5f}  {l2:.4f}  {values[0]:7.2f}  {values[1]:9.0f}  {values[2]:.3f}")
            rows.append(("dt_rel_l2", f"{ph}_dt_{dt:g}", l2))
            rows += [(f"dt_{n}", f"{ph}_dt_{dt:g}", v) for n, v in zip(names, values)]
    print(f"   stc over seeds {list(REALIZATION_SEEDS)}: dt [s], mean +/- SD of peak, MI, brake")
    for dt in dts:
        values = np.array([summary(run("stc", dt=dt, seed=s)) for s in REALIZATION_SEEDS])
        mean, sd = values.mean(axis=0), values.std(axis=0)
        print(f"     {dt:.5f}  {mean[0]:5.1f} +/- {sd[0]:4.1f}  {mean[1]:8.0f} +/- {sd[1]:5.0f}"
              f"  {mean[2]:.2f} +/- {sd[2]:.2f}")
        rows += [(f"dt_{n}_mean", f"stc_dt_{dt:g}", v) for n, v in zip(names, mean)]
        rows += [(f"dt_{n}_sd", f"stc_dt_{dt:g}", v) for n, v in zip(names, sd)]

    print(f"2. Healthy runs from random initial phases (lag after {TRANSIENT_S:g} s)")
    print("   seed, peak [mmHg], HAPC neighbor lag mean [rad], lag SD [rad]")
    for s in INITIAL_PHASE_SEEDS:
        res = run("healthy", random_phase_seed=s)
        k0 = int(round(TRANSIENT_S / (res.t[1] - res.t[0])))
        lag = np.angle(np.exp(1j * np.diff(res.phi[k0:, :res.geom.N_HAPC], axis=1)))
        print(f"     {s}  {res.P.max():6.1f}  {lag.mean():+.4f}  {lag.std():.4f}")
        rows += [("random_phase_peak", f"seed_{s}", res.P.max()),
                 ("random_phase_hapc_lag", f"seed_{s}", float(lag.mean()))]

    print(f"3. Peak pressure [mmHg] over noise seeds {list(REALIZATION_SEEDS)} (mean +/- SD)")
    for label, ph, overrides in [("stc", "stc", None),
                                 (f"healthy_sigma_{SIGMA_HEALTHY_CHECK}", "healthy",
                                  {"sigma_phi": SIGMA_HEALTHY_CHECK})]:
        peaks = [run(ph, seed=s, overrides=overrides).P.max() for s in REALIZATION_SEEDS]
        mean, sd = float(np.mean(peaks)), float(np.std(peaks))
        print(f"   {label:20s} {mean:6.1f} +/- {sd:.1f}")
        rows += [("peak_mean", label, mean), ("peak_sd", label, sd)]

    print(f"4. SD of neighbor phase difference [rad] (HAPC, CMP), after {TRANSIENT_S:g} s, "
          f"mean over seeds {list(REALIZATION_SEEDS)}")
    cases = [
        (f"healthy_sigma_{SIGMA_HEALTHY_CHECK}", "healthy", {"sigma_phi": SIGMA_HEALTHY_CHECK}),
        (f"healthy_sigma_{SIGMA_STC}", "healthy", {"sigma_phi": SIGMA_STC}),
        ("stc", "stc", None),
        ("stc_sigma_0", "stc", {"sigma_phi": 0.0}),
    ]
    for label, ph, overrides in cases:
        sds = np.array([neighbor_phase_sd(run(ph, seed=s, overrides=overrides), TRANSIENT_S)
                        for s in REALIZATION_SEEDS]).mean(axis=0)
        print(f"   {label:20s} {sds[0]:.2f} {sds[1]:.2f}")
        rows += [("phase_diff_sd_hapc", label, sds[0]), ("phase_diff_sd_cmp", label, sds[1])]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "case", "value"])
        writer.writerows((check, case, f"{value:.4f}") for check, case, value in rows)
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
