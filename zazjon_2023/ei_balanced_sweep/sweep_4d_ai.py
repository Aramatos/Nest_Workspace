"""
4D parameter sweep for the E-I balanced network:
    N_total  ×  g  ×  epsilon  ×  J_mV

Probes the AI-regime boundary as network size shrinks.

Run from the experiment directory:
    cd zajzon_2023/ei_balanced
    python sweep_4d_ai.py

Re-run after crash / grid extension — same command, skips finished combos.

Place at:  zajzon_2023/ei_balanced/sweep_4d_ai.py
"""

import sys
import os
import time
import itertools

import numpy as np

# ── Path setup (adjust if your layout differs) ──────────────────
#    Expects:  NEST_Workspace/shared/  to be two levels up
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, WORKSPACE_ROOT)

from shared.stats import cv_isi, pairwise_correlation, fano_factor
from shared.sweep_manager import SweepManager
from zazjon_2023.ei_balanced_sweep.network import run_ei_balanced


# ═════════════════════════════════════════════════════════════════
# SWEEP GRID — edit / extend these, re-run, old combos skip
# ═════════════════════════════════════════════════════════════════
N_TOTAL  = [500, 1000, 2500]
G_VALUES = [4.0, 6.0, 8.0, 10.0, 12.0]
EPSILON  = [0.05, 0.1, 0.15, 0.2]
J_MV     = [0.1, 0.2, 0.4]

# ── Fixed parameters (not swept) ────────────────────────────────
SIM_TIME = 1000.0
MU_X     = 2.0             # eta: ratio of ext rate to threshold rate (Brunel)
T_TRANS  = 200.0           # ms of transient to discard
DELAY    = 1.5
TAU_SYN  = 0.5

NEURON_DEFAULTS = {
    "C_m": 250.0, "tau_m": 20.0, "E_L": 0.0,
    "V_reset": 0.0, "V_th": 20.0, "t_ref": 2.0,
}

LOG_PATH = os.path.join(EXPERIMENT_DIR, "results", "sweep_4d_ai.jsonl")


# ═════════════════════════════════════════════════════════════════
# Metric extraction (experiment-specific, uses shared stats)
# ═════════════════════════════════════════════════════════════════
def extract_metrics(output, sim_params):
    evE = output["spike_rec_E"].get("events")
    evI = output["spike_rec_I"].get("events")

    mE = evE["times"] > T_TRANS
    mI = evI["times"] > T_TRANS

    tE, sE = evE["times"][mE], evE["senders"][mE]
    tI, sI = evI["times"][mI], evI["senders"][mI] - sim_params["n_1"]

    dur = (sim_params["sim_time"] - T_TRANS) / 1000.0

    return {
        "fr_E":   float(len(tE) / (sim_params["n_1"] * dur)) if len(tE) else 0.0,
        "fr_I":   float(len(tI) / (sim_params["n_2"] * dur)) if len(tI) else 0.0,
        "cv_E":   float(cv_isi(tE, sE, sim_params["n_1"])),
        "cv_I":   float(cv_isi(tI, sI, sim_params["n_2"])),
        "corr_E": float(pairwise_correlation(tE, sE, sim_params["n_1"])),
        "corr_I": float(pairwise_correlation(tI, sI, sim_params["n_2"])),
        "fano_E": float(fano_factor(tE, sE, sim_params["n_1"],
                                    bin_size=10.0, t_start=T_TRANS,
                                    t_end=sim_params["sim_time"])),
        "fano_I": float(fano_factor(tI, sI, sim_params["n_2"],
                                    bin_size=10.0, t_start=T_TRANS,
                                    t_end=sim_params["sim_time"])),
    }


# ═════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════
def main():
    mgr = SweepManager(LOG_PATH)
    combos = list(itertools.product(N_TOTAL, G_VALUES, EPSILON, J_MV))
    total = len(combos)

    print(f"Sweep:  {len(N_TOTAL)} N × {len(G_VALUES)} g "
          f"× {len(EPSILON)} ε × {len(J_MV)} J  =  {total} combos")
    print(f"Already done: {mgr.n_done}")
    print(f"Log: {LOG_PATH}\n")

    t0 = time.time()
    skipped = 0

    for idx, (N, g, eps, J) in enumerate(combos, 1):
        n1 = int(0.8 * N)
        n2 = int(0.2 * N)

        # Canonical params (deterministic hash key)
        canon = {
            "N_total": N, "g": g, "epsilon": eps, "J_mV": J,
            "n_1": n1, "n_2": n2, "mu_x": MU_X, "sim_time": SIM_TIME,
            "delay": DELAY, "tau_syn": TAU_SYN,
        }

        if mgr.is_done(canon):
            skipped += 1
            continue

        sim_p = {"n_1": n1, "n_2": n2, "epsilon": eps,
                 "mu_x": MU_X, "sim_time": SIM_TIME}
        syn_p = {"J_mV": J, "g": g, "tau_syn": TAU_SYN, "delay": DELAY}

        label = f"N={N:5d}  g={g:4.1f}  ε={eps:.2f}  J={J:.2f}"
        t1 = time.time()

        try:
            out = run_ei_balanced(sim_p, NEURON_DEFAULTS, syn_p)
            metrics = extract_metrics(out, sim_p)
            dt = time.time() - t1
            mgr.record(canon, metrics, dt)
            print(f"[{mgr.n_done:3d}/{total}] ({dt:5.1f}s)  {label}  →  "
                  f"FR_E={metrics['fr_E']:6.2f}  CV_E={metrics['cv_E']:.3f}  "
                  f"Corr_E={metrics['corr_E']:+.5f}  Fano_E={metrics['fano_E']:.3f}")
        except Exception as e:
            dt = time.time() - t1
            print(f"[{'—':>3s}/{total}] ({dt:5.1f}s)  {label}  →  FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\nDone. {mgr.n_done}/{total} complete, "
          f"{skipped} skipped, {elapsed:.0f}s total.")


if __name__ == "__main__":
    main()
