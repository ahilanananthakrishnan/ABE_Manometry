"""
Robustness checks quoted in the manuscript text, computed from pressure_traces.py.

  1. Time-step convergence: peak pressure with dt and dt/2 for each phenotype.
  2. Initial phases: healthy runs started from random initial phases.
  3. Noise realizations: STC, and healthy with sigma_phi = 0.1, over REALIZATION_SEEDS
     (peak pressure, mean +/- SD).
  4. Phase coherence: SD of the neighbor phase difference phi_{s+1} - phi_s (wrapped to
     (-pi, pi]) over all time steps and neighbor pairs, per subnetwork, for healthy
     coupling with sigma_phi = 0.1 and 0.5, the STC case, and STC coupling without noise.

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
from pressure_traces import SEED, SimParams, TraceResults, run_traces

# =============================================================================
# User settings
# =============================================================================
CSV_PATH = os.path.join(OUTPUT_DIR, "robustness_checks.csv")
PHENOTYPES = ("healthy", "ibsd", "stc")
REALIZATION_SEEDS = range(10)      # [-] seeds of the noise realizations (check 3)
INITIAL_PHASE_SEEDS = range(10)    # [-] seeds of the random initial phases (check 2)
SIGMA_HEALTHY_CHECK = 0.1          # [rad/sqrt(s)] noise added to the healthy case (check 3)
SIGMA_STC = 0.5                    # [rad/sqrt(s)] STC noise, with healthy coupling (check 4)


def run(phenotype: str, **kwargs) -> TraceResults:
    """One silent simulation without output files or plots."""
    return run_traces(phenotype, write_output=False, make_plot=False, verbose=False,
                      **kwargs)


def neighbor_phase_sd(res: TraceResults) -> tuple[float, float]:
    """SD [rad] of the wrapped neighbor phase difference in the HAPC and CMP subnetworks."""
    n_hapc = res.geom.N_HAPC
    sds = []
    for phi in (res.phi[:, :n_hapc], res.phi[:, n_hapc:]):
        dphi = np.angle(np.exp(1j * np.diff(phi, axis=1)))  # wrap to (-pi, pi]
        sds.append(float(dphi.std()))
    return sds[0], sds[1]


def main() -> None:
    """Run all checks, print them, and write one CSV row per reported number."""
    rows: list[tuple[str, str, float]] = []   # (check, case, value)
    dt = SimParams().dt

    print(f"1. Peak pressure [mmHg] with dt = {dt} s and dt/2")
    for ph in PHENOTYPES:
        p_dt = run(ph).P.max()
        p_half = run(ph, dt=0.5 * dt).P.max()
        print(f"   {ph:8s} {p_dt:7.2f} {p_half:7.2f}  ({100 * (p_half - p_dt) / p_dt:+.1f}%)")
        rows += [("peak_dt", ph, p_dt), ("peak_dt_half", ph, p_half)]
    print("   (STC draws a different noise sequence when dt changes)")

    print("2. Healthy peak pressure [mmHg] from random initial phases")
    peaks = [run("healthy", random_phase_seed=s).P.max() for s in INITIAL_PHASE_SEEDS]
    print(f"   seeds {list(INITIAL_PHASE_SEEDS)}: " + " ".join(f"{p:.1f}" for p in peaks))
    rows += [("peak_random_phase", f"seed_{s}", p) for s, p in zip(INITIAL_PHASE_SEEDS, peaks)]

    print(f"3. Peak pressure [mmHg] over noise seeds {list(REALIZATION_SEEDS)} (mean +/- SD)")
    for label, ph, overrides in [("stc", "stc", None),
                                 (f"healthy_sigma_{SIGMA_HEALTHY_CHECK}", "healthy",
                                  {"sigma_phi": SIGMA_HEALTHY_CHECK})]:
        peaks = [run(ph, seed=s, overrides=overrides).P.max() for s in REALIZATION_SEEDS]
        mean, sd = float(np.mean(peaks)), float(np.std(peaks))
        print(f"   {label:20s} {mean:6.1f} +/- {sd:.1f}")
        rows += [("peak_mean", label, mean), ("peak_sd", label, sd)]

    print(f"4. SD of neighbor phase difference [rad] (HAPC, CMP), seed {SEED}")
    cases = [
        (f"healthy_sigma_{SIGMA_HEALTHY_CHECK}", "healthy", {"sigma_phi": SIGMA_HEALTHY_CHECK}),
        (f"healthy_sigma_{SIGMA_STC}", "healthy", {"sigma_phi": SIGMA_STC}),
        ("stc", "stc", None),
        ("stc_sigma_0", "stc", {"sigma_phi": 0.0}),
    ]
    for label, ph, overrides in cases:
        sd_hapc, sd_cmp = neighbor_phase_sd(run(ph, overrides=overrides))
        print(f"   {label:20s} {sd_hapc:.2f} {sd_cmp:.2f}")
        rows += [("phase_diff_sd_hapc", label, sd_hapc), ("phase_diff_sd_cmp", label, sd_cmp)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "case", "value"])
        writer.writerows((check, case, f"{value:.4f}") for check, case, value in rows)
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
