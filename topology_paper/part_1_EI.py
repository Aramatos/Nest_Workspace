#import nest


import numpy as np
import matplotlib.pyplot as plt
import nest


def sim(sim_params, neuron_params, syn_params):

    # ── Clean slate ──────────────────────────────────────────────────
    print("Simulation Setup Start")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.01,"local_num_threads": 5}) # ms
    
    # ── derived parameters ─────────────────────────────────────────────
    w_E = syn_params["J_mV"] * neuron_params["C_m"] / (syn_params["tau_syn"] * np.e)    # Excitatory weight (pA)
    w_I = -syn_params["g"] * w_E                                           # Inhibitory weight (pA)

    k_e=int(sim_params["epsilon"]*sim_params["n_1"])
    k_i=int(sim_params["epsilon"]*sim_params["n_2"])

    # ── Create neurons ─────────────────────────────────────────────────
    neurons_E = nest.Create("iaf_psc_exp", sim_params["n_1"], neuron_params)
    neurons_I = nest.Create("iaf_psc_exp", sim_params["n_2"], neuron_params)
    
    # ── Create virtual neurons ─────────────────────────────────────────
    poisson_bg = nest.Create("poisson_generator", params={"rate": k_e * sim_params["mu_x"]})

    # ── Create recorders ───────────────────────────────────────────────    spike_rec_E = nest.Create("spike_recorder")
    spike_rec_E = nest.Create("spike_recorder")
    spike_rec_I = nest.Create("spike_recorder")

    # ── Feed Forward Connections ───────────────────────────────────────
    nest.Connect(poisson_bg, neurons_E, "all_to_all",
             {"synapse_model": "static_synapse", "weight": w_E, "delay": syn_params["delay"]})
    nest.Connect(poisson_bg, neurons_I, "all_to_all",
             {"synapse_model": "static_synapse", "weight": w_E, "delay": syn_params["delay"]})

    # ── Recurrent connections (sparse random) ──────────────────────────
    # E → E, E → I, I → E, I → I — all with fixed in-degree
    print("Setting up Connections")

    nest.Connect(neurons_E, neurons_E,
             {"rule": "fixed_indegree", "indegree": k_e},
             {"synapse_model": "static_synapse", "weight": w_E, "delay": syn_params["delay"]})

    nest.Connect(neurons_E, neurons_I,
                {"rule": "fixed_indegree", "indegree": k_e},
                {"synapse_model": "static_synapse", "weight": w_E, "delay": syn_params["delay"]})

    nest.Connect(neurons_I, neurons_E,
                {"rule": "fixed_indegree", "indegree": k_i},
                {"synapse_model": "static_synapse", "weight": w_I, "delay": syn_params["delay"]})

    nest.Connect(neurons_I, neurons_I,
                {"rule": "fixed_indegree", "indegree": k_i},
                {"synapse_model": "static_synapse", "weight": w_I, "delay": syn_params["delay"]})

    # ── Recording ────────────────────────────────────────────────────
    nest.Connect(neurons_E, spike_rec_E)
    nest.Connect(neurons_I, spike_rec_I)

    # ── Simulate ─────────────────────────────────────────────────────
    T_sim = 2000.0  # ms
    print(f"\nSimulating {T_sim} ms...")
    nest.Simulate(T_sim)
    print("Done.")

    output={
        "spike_rec_E":spike_rec_E,
        "spike_rec_I":spike_rec_I,
    }

    return output


# %% Define Params
sim_params = {
    "mu_x":12,         # External input rate (kHz)
    "n_1": 8000,       # Number of excitatory neurons
    "n_2": 2000,    # Number of inhibitory neurons
    "epsilon": 0.1, 
    "sim_time": 1000.0
}

neuron_params = {
    "C_m": 250.0,       # Membrane capacitance (pF)
    "tau_m": 20.0,       # Membrane time constant (ms)
    "E_L": 0.0,          # Leak reversal (mV)
    "V_reset": 0.0,      # Reset potential (mV)
    "V_th": 20.0,        # Threshold (mV)
    "t_ref": 2.0,        # Refractory period (ms)
}

syn_params={
    "J_mV": 0.1,                              # Excitatory PSP amplitude (mV)
    "g": 12.0,                                 # Relative inhibitory strength
    "tau_syn": 0.5,                            # Synaptic time constant (ms)
    "delay": 1.5                               # Synaptic delay (ms)
}

# %%
output = sim(sim_params, neuron_params, syn_params)

#create unique id for this simulation
import uuid
sim_id = str(uuid.uuid4())[:8]
print(f"Simulation ID: {sim_id}")
np.savez(f"Results/output_{sim_id}.npz", **output)

# %%
# Check if output exists, if not load from file
if 'output' not in locals():
    data = np.load("Results/output_43a2a102.npz", allow_pickle=True)
    output = {key: data[key] for key in data.files}

spike_rec_E = output["spike_rec_E"]
spike_rec_I = output["spike_rec_I"]


# ── Quick check: is it in the AI regime? ─────────────────────────
events_E = spike_rec_E.get("events")
events_I = spike_rec_I.get("events")

# Mean firing rate (discard first 500ms transient)
T_trans = 500.0
mask_E = events_E["times"] > T_trans
mask_I = events_I["times"] > T_trans

rate_E = mask_E.sum() / sim_params["n_1"] / ((sim_params["sim_time"] - T_trans) / 1000.0)
rate_I = mask_I.sum() / sim_params["n_2"] / ((sim_params["sim_time"] - T_trans) / 1000.0)

# mask 500 nuerons
senders_E = events_E["senders"]
senders_I = events_I["senders"]
senders_mask_E = senders_E < 500
senders_mask_I = senders_I < 200

print(f"\nMean firing rates:")
print(f"  Excitatory: {rate_E:.1f} spks/s")
print(f"  Inhibitory: {rate_I:.1f} spks/s")

# ── Raster plot (first 500 neurons, 500-1500ms window) ───────────

spike_rec_E = output["spike_rec_E"]
spike_rec_I = output["spike_rec_I"]

# ── Quick check: is it in the AI regime? ─────────────────────────
events_E = spike_rec_E.get("events")
events_I = spike_rec_I.get("events")

# Mean firing rate (discard first 500ms transient)
T_trans = 500.0
mask_E = events_E["times"] > T_trans
mask_I = events_I["times"] > T_trans

rate_E = mask_E.sum() / sim_params["n_1"] / ((sim_params["sim_time"] - T_trans) / 1000.0)
rate_I = mask_I.sum() / sim_params["n_2"] / ((sim_params["sim_time"] - T_trans) / 1000.0)

# mask 500 nuerons
senders_E = events_E["senders"]
senders_I = events_I["senders"]
senders_mask_E = senders_E < 500
senders_mask_I = senders_I < 200

print(f"\nMean firing rates:")
print(f"  Excitatory: {rate_E:.1f} spks/s")
print(f"  Inhibitory: {rate_I:.1f} spks/s")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Combine both conditions into ONE mask, then slice ONCE
both_E = (events_E["times"] > T_trans) & (events_E["senders"] < 500)
ax1.scatter(events_E["times"][both_E], events_E["senders"][both_E],
            s=0.2, c="steelblue", alpha=0.5)
ax1.set_ylabel("Neuron ID")
ax1.set_title(f"Excitatory ({rate_E:.1f} spks/s)")

both_I = (events_I["times"] > T_trans) & (events_I["senders"] < (sim_params["n_1"] + 200))
ax2.scatter(events_I["times"][both_I], events_I["senders"][both_I],
            s=0.2, c="indianred", alpha=0.5)    
ax2.set_ylabel("Neuron ID")
ax2.set_xlabel("Time (ms)")
ax2.set_title(f"Inhibitory ({rate_I:.1f} spks/s)")

fig.suptitle("Step 1: Single SSN — Balanced Random Network")
fig.tight_layout()
plt.savefig("step1_raster.png", dpi=150)
plt.show()
# %%
