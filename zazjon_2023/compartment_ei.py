#import nest
# %%

# %%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import nest
import sys
import os



# ── Path setup (adjust if your layout differs) ──────────────────
#    Expects:  NEST_Workspace/shared/  to be two levels up
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, WORKSPACE_ROOT)


from shared.stats import cv_isi, pairwise_correlation, fano_factor



def plot_raster(spike_times_E, clusters_E, spike_times_I, clusters_I, sim_params,
                stats=None, neuron_params=None, syn_params=None):

    def _fmt(val, spec):
        return "N/A" if (isinstance(val, float) and np.isnan(val)) else format(val, spec)

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        hspace=0.55, wspace=0.10,
        left=0.07, right=0.97, top=0.90, bottom=0.09,
    )
    ax_E    = fig.add_subplot(gs[0, 0])
    ax_I    = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # ── Rasters ─────────────────────────────────────────────────────
    ax_E.scatter(spike_times_E, clusters_E, s=0.4, color="#2166AC", alpha=0.5, rasterized=True)
    ax_I.scatter(spike_times_I, clusters_I, s=0.4, color="#D6604D", alpha=0.5, rasterized=True)

    for ax in (ax_E, ax_I):
        ax.set_xlim(0, sim_params["sim_time"])
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Neuron ID", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    # ── Bold left title + stats subtitle ────────────────────────────
    for ax, pop, keys in [
        (ax_E, "Excitatory", ("fr_E", "cv_E", "corr_E", "fano_E")),
        (ax_I, "Inhibitory", ("fr_I", "cv_I", "corr_I", "fano_I")),
    ]:
        ax.text(0, 1.13, f"{pop} Neurons",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold")
        if stats is not None:
            fr, cv, cc, fano = keys
            subtitle = (
                f"FR = {_fmt(stats[fr], '.2f')} Hz    "
                f"CV = {_fmt(stats[cv], '.3f')}    "
                f"Corr = {_fmt(stats[cc], '.5f')}    "
                f"Fano = {_fmt(stats[fano], '.3f')}"
            )   
            
            ax.text(0, 1.02, subtitle,
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=10, color="#555555")

    # ── Parameters panel ────────────────────────────────────────────
    ax_info.axis("off")
    ax_info.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.03",
        transform=ax_info.transAxes,
        linewidth=1, edgecolor="#CCCCCC",
        facecolor="#F8F9FA", zorder=0,
    ))

    sections = []
    if neuron_params is not None:
        sections.append(("NEURON", neuron_params))
    if syn_params is not None:
        sections.append(("SYNAPSE", syn_params))
    sections.append(("SIMULATION", sim_params))

    y = 0.96
    for section_name, params in sections:
        ax_info.text(0.07, y, section_name,
                     transform=ax_info.transAxes,
                     fontsize=12, fontweight="bold", color="#2166AC", va="top")
        y -= 0.047
        for k, v in params.items():
            ax_info.text(0.07, y, f"{k}", transform=ax_info.transAxes,
                         fontsize=12, color="#444444", va="top")
            ax_info.text(0.93, y, f"{v}", transform=ax_info.transAxes,
                         fontsize=12, color="#111111", va="top", ha="right",
                         fontfamily="monospace")
            y -= 0.042
        y -= 0.022

    fig.suptitle("E–I Recurrent Network", fontsize=15, fontweight="bold", x=0.44, y=0.97)
    plt.show()


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

    # ── Generate stimulus sequence ───────────────────────────────────
    stim_duration =200

    n_stim_steps = int(sim_params["sim_time"] / stim_duration)
    rng = np.random.default_rng(42)
    active_channels = rng.integers(0,sim_params["N_C"],size=n_stim_steps)

    lam = 0.05
    nu_stim = k_e * lam * sim_params["mu_x"]
    
    print(f"Stimulus: {n_stim_steps} steps, rate boost = {nu_stim:.1f} spks/s")
    print(f"  Channel sequence: {active_channels[:10]}...")

    # ── Create neurons ─────────────────────────────────────────────────

    neurons_E = nest.Create("iaf_psc_exp", sim_params["n_1"], neuron_params)
    neurons_I = nest.Create("iaf_psc_exp", sim_params["n_2"], neuron_params)

    c_E = int(sim_params["d"] * sim_params["n_1"])   # 800 exc neurons per sub-population
    c_I = int(sim_params["d"] * sim_params["n_2"])   # 200 inh neurons per sub-population

    sub_pops = []
    for i in range(sim_params["N_C"]):
        sp_E = neurons_E[i * c_E:(i + 1) * c_E]
        sp_I = neurons_I[i * c_I:(i + 1) * c_I]
        sub_pops.append((sp_E, sp_I))

    print(f"Sub-populations: {sim_params['N_C']} channels, {c_E}E + {c_I}I each")
    print(f"  Channel 0 exc IDs: {sub_pops[0][0][0]} to {sub_pops[0][0][-1]}")
    print(f"  Channel 9 exc IDs: {sub_pops[9][0][0]} to {sub_pops[9][0][-1]}")

    
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
    print(f"\nSimulating {sim_params['sim_time']} ms...")
    nest.Simulate(sim_params["sim_time"])
    print("Done.")

    output={
        "spike_rec_E":spike_rec_E,
        "spike_rec_I":spike_rec_I,
        "sub_pops":sub_pops,
    }

    return output


# %% Define Params
sim_params = {
    "N_C": 10,
    "d": 0.1,        
    "mu_x":10,      # External input rate (kHz)
    "n_1": 8000,      # Number of excitatory neurons
    "n_2": 2000,      # Number of inhibitory neurons
    "epsilon": 0.1,  # Connection probability
    "sim_time": 1000.0  # Simulation time (ms)
}

neuron_params = {
    "C_m": 250.0,        # Membrane capacitance (pF)
    "tau_m": 20.0,       # Membrane time constant (ms)
    "E_L": 0.0,          # Leak reversal (mV)
    "V_reset": 0.0,      # Reset potential (mV)
    "V_th": 20.0,        # Threshold (mV)
    "t_ref": 2.0,        # Refractory period (ms)
}

syn_params={
    "J_mV": 0.3,                               # Excitatory PSP amplitude (mV)
    "g": 12.0,                                 # Relative inhibitory strength
    "tau_syn": 0.5,                            # Synaptic time constant (ms)
    "delay": 1.5                               # Synaptic delay (ms)
}


output = sim(sim_params, neuron_params, syn_params)

#create unique id for this simulation
import uuid
sim_id = str(uuid.uuid4())[:8]
print(f"Simulation ID: {sim_id}")
events_E = output["spike_rec_E"].get("events")
events_I = output["spike_rec_I"].get("events")
np.savez(
    f"results/output_{sim_id}.npz",
    spike_times_E=events_E["times"],
    senders_E=events_E["senders"],
    spike_times_I=events_I["times"],
    senders_I=events_I["senders"],
)


T_trans = 0

spike_times_E = events_E["times"]
spike_times_I = events_I["times"]

transient_mask_E = spike_times_E > T_trans
transient_mask_I = spike_times_I > T_trans

spike_times_E = spike_times_E[transient_mask_E]
clusters_E = events_E["senders"][transient_mask_E]
spike_times_I = spike_times_I[transient_mask_I]
clusters_I = events_I["senders"][transient_mask_I] - sim_params["n_1"]  # Adjust inhibitory neuron IDs to start from 1

# compute firing rates
duration_sec = (sim_params["sim_time"] - T_trans) / 1000.0  # Convert ms to seconds
firing_rate_E = len(spike_times_E) / (sim_params["n_1"] * duration_sec)
firing_rate_I = len(spike_times_I) / (sim_params["n_2"] * duration_sec)

cv_ave_E = cv_isi(spike_times_E, clusters_E, sim_params["n_1"])
cv_ave_I = cv_isi(spike_times_I, clusters_I, sim_params["n_2"])

corr_E = pairwise_correlation(spike_times_E, clusters_E, sim_params["n_1"])
corr_I = pairwise_correlation(spike_times_I, clusters_I, sim_params["n_2"])

fano_E = fano_factor(spike_times_E, clusters_E, sim_params["n_1"], bin_size=100)
fano_I = fano_factor(spike_times_I, clusters_I, sim_params["n_2"], bin_size=100)

print(f"Firing Rate  — E: {firing_rate_E:.2f} Hz   I: {firing_rate_I:.2f} Hz")
print(f"Mean CV      — E: {cv_ave_E:.3f}          I: {cv_ave_I:.3f}")
print(f"Mean Corr    — E: {corr_E:.5f}       I: {corr_I:.5f}")

stats = dict(fr_E=firing_rate_E, fr_I=firing_rate_I,
             cv_E=cv_ave_E, cv_I=cv_ave_I,
             corr_E=corr_E, corr_I=corr_I,
             fano_E=fano_E, fano_I=fano_I)

plot_raster(spike_times_E, clusters_E, spike_times_I, clusters_I, sim_params,
            stats=stats, neuron_params=neuron_params, syn_params=syn_params)

# %%
