"""
Single balanced E-I random network (one SSN) in NEST.

This is the base network for the Zajzon et al. replication:
iaf_psc_exp neurons, fixed_indegree connectivity, shared Poisson drive.

Returns spike recorders only — caller handles all analysis.

Place at:  zajzon_2023/ei_balanced/network.py
"""

import numpy as np
import nest


def run_ei_balanced(sim_params, neuron_params, syn_params):
    """
    Parameters
    ----------
    sim_params : dict
        n_1, n_2, epsilon, mu_x, sim_time
    neuron_params : dict
        Passed directly to iaf_psc_exp (C_m, tau_m, E_L, V_reset, V_th, t_ref)
    syn_params : dict
        J_mV, g, tau_syn, delay

    Returns
    -------
    dict with keys "spike_rec_E", "spike_rec_I" (NEST NodeCollections)
    """
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.01, "local_num_threads": 5})

    # Derived weights
    w_E = syn_params["J_mV"] * neuron_params["C_m"] / (syn_params["tau_syn"] * np.e)
    w_I = -syn_params["g"] * w_E

    k_e = int(sim_params["epsilon"] * sim_params["n_1"])
    k_i = int(sim_params["epsilon"] * sim_params["n_2"])

    # Populations
    E = nest.Create("iaf_psc_exp", sim_params["n_1"], neuron_params)
    I = nest.Create("iaf_psc_exp", sim_params["n_2"], neuron_params)

    # External drive — mu_x is eta (ratio to threshold rate, Brunel 2000)
    nu_thr = neuron_params["V_th"] * 1000.0 / (
        syn_params["J_mV"] * k_e * neuron_params["tau_m"])
    ext_rate = k_e * sim_params["mu_x"] * nu_thr
    bg = nest.Create("poisson_generator", params={"rate": ext_rate})

    # Recorders
    rec_E = nest.Create("spike_recorder")
    rec_I = nest.Create("spike_recorder")

    # Synaptic specs
    sE = {"synapse_model": "static_synapse", "weight": w_E, "delay": syn_params["delay"]}
    sI = {"synapse_model": "static_synapse", "weight": w_I, "delay": syn_params["delay"]}

    # Feed-forward
    nest.Connect(bg, E, "all_to_all", sE)
    nest.Connect(bg, I, "all_to_all", sE)

    # Recurrent (fixed in-degree)
    nest.Connect(E, E, {"rule": "fixed_indegree", "indegree": k_e}, sE)
    nest.Connect(E, I, {"rule": "fixed_indegree", "indegree": k_e}, sE)
    nest.Connect(I, E, {"rule": "fixed_indegree", "indegree": k_i}, sI)
    nest.Connect(I, I, {"rule": "fixed_indegree", "indegree": k_i}, sI)

    # Record
    nest.Connect(E, rec_E)
    nest.Connect(I, rec_I)

    # Simulate
    nest.Simulate(sim_params["sim_time"])

    return {"spike_rec_E": rec_E, "spike_rec_I": rec_I}
