import numpy as np
import pandas as pd
import nest
import nest.raster_plot
import matplotlib.pyplot as plt
import seaborn as sns

from sim_utils import create_neuron_dicts


# =============================================================================
# SIMULATION FUNCTION
# =============================================================================

def sim(sim_params, neuron_dicts, syn_params, current_params):
    """Run a single TC-RE network simulation and return output data."""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1})

    # Create AdEx neurons
    neuron1 = nest.Create("aeif_cond_exp", n=sim_params["n_ex"], params=neuron_dicts["TC"])
    neuron2 = nest.Create("aeif_cond_exp", n=sim_params["n_inh"], params=neuron_dicts["RE"])

    # Create step current generator
    step_current = nest.Create("step_current_generator", params=current_params)

    # Connect network
    nest.Connect(step_current, neuron1)
    nest.Connect(neuron1, neuron2, "all_to_all", syn_spec={'weight': syn_params["w_e"]})
    nest.Connect(neuron2, neuron1, "all_to_all", syn_spec={'weight': syn_params["w_i"]})

    # Create recorders
    voltmeter1 = nest.Create("voltmeter")
    voltmeter2 = nest.Create("voltmeter")
    spikemeter1 = nest.Create("spike_recorder")
    spikemeter2 = nest.Create("spike_recorder")

    # Connect recorders
    nest.Connect(voltmeter1, neuron1)
    nest.Connect(voltmeter2, neuron2)
    nest.Connect(neuron1, spikemeter1)  # TC
    nest.Connect(neuron2, spikemeter2)  # RE

    # Simulate
    sim_duration = sim_params.get("duration", 1000.0)
    nest.Simulate(sim_duration)

    # Extract data
    plot_data_1 = nest.GetStatus(voltmeter1)[0]
    plot_data_2 = nest.GetStatus(voltmeter2)[0]
    plot_data_3 = nest.GetStatus(spikemeter1)[0]
    plot_data_4 = nest.GetStatus(spikemeter2)[0]

    sim_output = {
        "times_1": plot_data_1["events"]["times"],
        "voltages_1": plot_data_1["events"]["V_m"],
        "times_2": plot_data_2["events"]["times"],
        "voltages_2": plot_data_2["events"]["V_m"],
        "spike_times_1": plot_data_3["events"]["times"],
        "spike_ids_1": plot_data_3["events"]["senders"],
        "spike_times_2": plot_data_4["events"]["times"],
        "spike_ids_2": plot_data_4["events"]["senders"]
    }

    return sim_output


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_isi_with_bursts(spike_times, burst_threshold=50, ignore_start=100.0):
    """Analyze ISI with burst detection."""
    spike_times = np.array(spike_times)
    spike_times = spike_times[spike_times >= ignore_start]
    
    if len(spike_times) < 2:
        return {"intra_burst_isi": np.array([]), "inter_burst_isi": np.array([]), "burst_info": []}
    
    isis = np.diff(spike_times)
    intra_burst_isi = []
    inter_burst_isi = []
    burst_info = []
    current_burst = [spike_times[0]]
    
    for i, isi in enumerate(isis):
        if isi <= burst_threshold:
            current_burst.append(spike_times[i+1])
            intra_burst_isi.append(isi)
        else:
            if len(current_burst) > 1:
                burst_info.append({
                    'start_time': current_burst[0],
                    'end_time': current_burst[-1],
                    'spike_count': len(current_burst),
                    'duration': current_burst[-1] - current_burst[0]
                })
            inter_burst_isi.append(isi)
            current_burst = [spike_times[i+1]]
    
    if len(current_burst) > 1:
        burst_info.append({
            'start_time': current_burst[0],
            'end_time': current_burst[-1],
            'spike_count': len(current_burst),
            'duration': current_burst[-1] - current_burst[0]
        })
    
    return {
        "intra_burst_isi": np.array(intra_burst_isi),
        "inter_burst_isi": np.array(inter_burst_isi),
        "burst_info": burst_info
    }


def compute_metrics(sim_output, sim_params):
    """Compute standard metrics from simulation output."""
    duration_s = sim_params.get("duration", 1000.0) / 1000.0
    
    tc_analysis = analyze_isi_with_bursts(sim_output["spike_times_1"])
    re_analysis = analyze_isi_with_bursts(sim_output["spike_times_2"])
    
    return {
        "tc_rate": len(sim_output["spike_times_1"]) / duration_s,
        "re_rate": len(sim_output["spike_times_2"]) / duration_s,
        "tc_inter_isi_mean": np.mean(tc_analysis["inter_burst_isi"]) if len(tc_analysis["inter_burst_isi"]) > 0 else np.nan,
        "tc_inter_isi_std": np.std(tc_analysis["inter_burst_isi"]) if len(tc_analysis["inter_burst_isi"]) > 0 else np.nan,
        "tc_intra_isi_mean": np.mean(tc_analysis["intra_burst_isi"]) if len(tc_analysis["intra_burst_isi"]) > 0 else np.nan,
        "tc_intra_isi_std": np.std(tc_analysis["intra_burst_isi"]) if len(tc_analysis["intra_burst_isi"]) > 0 else np.nan,
        "re_inter_isi_mean": np.mean(re_analysis["inter_burst_isi"]) if len(re_analysis["inter_burst_isi"]) > 0 else np.nan,
        "re_inter_isi_std": np.std(re_analysis["inter_burst_isi"]) if len(re_analysis["inter_burst_isi"]) > 0 else np.nan,
        "re_intra_isi_mean": np.mean(re_analysis["intra_burst_isi"]) if len(re_analysis["intra_burst_isi"]) > 0 else np.nan,
        "re_intra_isi_std": np.std(re_analysis["intra_burst_isi"]) if len(re_analysis["intra_burst_isi"]) > 0 else np.nan,
        "tc_analysis": tc_analysis,
        "re_analysis": re_analysis,
    }


# =============================================================================
# PARAMETER APPLICATION
# =============================================================================

def apply_param(sim_params, neuron_dicts, syn_params, param_name, value):
    """
    Apply a parameter value to the appropriate dictionary.
    
    Param names can be:
        - "w_e", "w_i" -> syn_params
        - "TC.tau_syn_ex", "RE.a" -> specific neuron
        - "tau_syn_ex", "C_m" -> both neurons
        - "duration", "n_ex" -> sim_params
    """
    if param_name in syn_params:
        syn_params[param_name] = value
    elif param_name in sim_params:
        sim_params[param_name] = value
    elif "." in param_name:
        # Format: "TC.param_name" or "RE.param_name"
        neuron_type, p_name = param_name.split(".", 1)
        neuron_dicts[neuron_type][p_name] = value
    else:
        # Apply to both neuron types
        for neuron_type in neuron_dicts:
            if param_name in neuron_dicts[neuron_type]:
                neuron_dicts[neuron_type][param_name] = value


# =============================================================================
# UNIFIED SWEEP FUNCTION
# =============================================================================

def sweep(sim_params, neuron_dicts, syn_params, current_params, sweep_config):
    """
    Unified sweep function for 1D or 2D parameter sweeps.
    
    Args:
        sim_params, neuron_dicts, syn_params, current_params: Base simulation parameters
        sweep_config: List of dicts, each with:
            - "param": parameter name (e.g., "w_i", "tau_syn_ex", "TC.a")
            - "values": array of values to sweep
            - "label": (optional) display label for plots
    
    Returns:
        results: Dictionary with sweep values and computed metrics
    
    Examples:
        # 1D sweep
        sweep_config = [{"param": "w_i", "values": np.linspace(-1000, 0, 20)}]
        
        # 2D sweep
        sweep_config = [
            {"param": "tau_syn_ex", "values": np.linspace(2, 6, 10)},
            {"param": "tau_syn_in", "values": np.linspace(15, 20, 5)}
        ]
    """
    n_params = len(sweep_config)
    
    if n_params == 1:
        return _sweep_1d(sim_params, neuron_dicts, syn_params, current_params, sweep_config[0])
    elif n_params == 2:
        return _sweep_2d(sim_params, neuron_dicts, syn_params, current_params, sweep_config)
    else:
        raise ValueError(f"Sweep supports 1 or 2 parameters, got {n_params}")


def _sweep_1d(sim_params, neuron_dicts, syn_params, current_params, config):
    """Internal 1D sweep."""
    param_name = config["param"]
    values = config["values"]
    label = config.get("label", param_name)
    
    results = {
        "param": param_name,
        "label": label,
        "values": values,
        "metrics": []
    }
    
    for val in values:
        # Deep copy to avoid mutation
        sp = sim_params.copy()
        nd = {k: v.copy() for k, v in neuron_dicts.items()}
        syp = syn_params.copy()
        
        apply_param(sp, nd, syp, param_name, val)
        sim_output = sim(sp, nd, syp, current_params)
        metrics = compute_metrics(sim_output, sp)
        metrics["sim_output"] = sim_output
        results["metrics"].append(metrics)
    
    return results


def _sweep_2d(sim_params, neuron_dicts, syn_params, current_params, configs):
    """Internal 2D sweep."""
    config1, config2 = configs
    param1, values1 = config1["param"], config1["values"]
    param2, values2 = config2["param"], config2["values"]
    label1 = config1.get("label", param1)
    label2 = config2.get("label", param2)
    
    results = {
        "params": [param1, param2],
        "labels": [label1, label2],
        "values": [values1, values2],
        "metrics_grid": [],  # 2D list
        "rate_matrix_tc": np.zeros((len(values1), len(values2))),
        "rate_matrix_re": np.zeros((len(values1), len(values2)))
    }
    
    for i, val1 in enumerate(values1):
        row = []
        for j, val2 in enumerate(values2):
            sp = sim_params.copy()
            nd = {k: v.copy() for k, v in neuron_dicts.items()}
            syp = syn_params.copy()
            
            apply_param(sp, nd, syp, param1, val1)
            apply_param(sp, nd, syp, param2, val2)
            
            sim_output = sim(sp, nd, syp, current_params)
            metrics = compute_metrics(sim_output, sp)
            metrics["sim_output"] = sim_output
            row.append(metrics)
            
            results["rate_matrix_tc"][i, j] = metrics["tc_rate"]
            results["rate_matrix_re"][i, j] = metrics["re_rate"]
        
        results["metrics_grid"].append(row)
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_simulation(sim_output, title=""):
    """Plot voltage traces and spike rasters for a single simulation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sim_output["times_1"], sim_output["voltages_1"], label="TC", linewidth=1.5)
    ax.plot(sim_output["times_2"], sim_output["voltages_2"], label="RE", linewidth=1.5)
    ax.scatter(sim_output["spike_times_1"], [0]*len(sim_output["spike_times_1"]), 
               color="red", label="TC spikes", s=20)
    ax.scatter(sim_output["spike_times_2"], [-20]*len(sim_output["spike_times_2"]), 
               color="blue", label="RE spikes", s=20)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title if title else "Voltage vs Time")
    ax.set_xlim(0, max(sim_output["times_1"][-1] if len(sim_output["times_1"]) else 800, 800))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_sweep_1d(results, metrics_to_plot=None):
    """
    Plot 1D sweep results.
    
    Args:
        results: Output from sweep() with 1 parameter
        metrics_to_plot: List of metric names to plot, or None for default ISI plots
    """
    if metrics_to_plot is None:
        # Default: ISI plots for TC and RE
        return _plot_isi_sweep(results)
    
    values = results["values"]
    label = results["label"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_name in metrics_to_plot:
        metric_vals = [m[metric_name] for m in results["metrics"]]
        ax.plot(values, metric_vals, marker='o', label=metric_name)
    
    ax.set_xlabel(label)
    ax.set_ylabel("Value")
    ax.set_title(f"Metrics vs {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_isi_sweep(results):
    """Plot ISI metrics for a 1D sweep."""
    values = results["values"]
    label = results["label"]
    metrics = results["metrics"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # TC
    tc_inter = [m["tc_inter_isi_mean"] for m in metrics]
    tc_inter_std = [m["tc_inter_isi_std"] for m in metrics]
    tc_intra = [m["tc_intra_isi_mean"] for m in metrics]
    tc_intra_std = [m["tc_intra_isi_std"] for m in metrics]
    
    axes[0].errorbar(values, tc_inter, yerr=tc_inter_std, label="Inter-burst ISI", marker='o', color='tab:blue')
    axes[0].errorbar(values, tc_intra, yerr=tc_intra_std, label="Intra-burst ISI", marker='s', color='tab:cyan')
    axes[0].set_xlabel(label)
    axes[0].set_ylabel("ISI (ms)")
    axes[0].set_title(f"TC: ISI vs {label}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RE
    re_inter = [m["re_inter_isi_mean"] for m in metrics]
    re_inter_std = [m["re_inter_isi_std"] for m in metrics]
    re_intra = [m["re_intra_isi_mean"] for m in metrics]
    re_intra_std = [m["re_intra_isi_std"] for m in metrics]
    
    axes[1].errorbar(values, re_inter, yerr=re_inter_std, label="Inter-burst ISI", marker='^', color='tab:red')
    axes[1].errorbar(values, re_intra, yerr=re_intra_std, label="Intra-burst ISI", marker='d', color='tab:orange')
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("ISI (ms)")
    axes[1].set_title(f"RE: ISI vs {label}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sweep_2d(results, metric="tc_rate", title=None):
    """
    Plot 2D sweep results as heatmap.
    
    Args:
        results: Output from sweep() with 2 parameters
        metric: Which metric to plot ("tc_rate", "re_rate", or any metric name)
        title: Optional title override
    """
    values1, values2 = results["values"]
    label1, label2 = results["labels"]
    
    if metric == "tc_rate":
        matrix = results["rate_matrix_tc"]
    elif metric == "re_rate":
        matrix = results["rate_matrix_re"]
    else:
        # Build matrix from metrics_grid
        matrix = np.array([[m[metric] for m in row] for row in results["metrics_grid"]])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f",
                xticklabels=np.round(values2, 2), 
                yticklabels=np.round(values1, 2), 
                cmap="viridis", ax=ax)
    ax.set_xlabel(label2)
    ax.set_ylabel(label1)
    ax.set_title(title if title else f"{metric} Heatmap")
    
    return fig


def plot_isi_distributions(tc_analysis, re_analysis, title_prefix=""):
    """Plot ISI distributions for TC and RE populations."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    if len(tc_analysis['inter_burst_isi']) > 0:
        axes[0].hist(tc_analysis['inter_burst_isi'], bins=20, alpha=0.7, color='red')
        axes[0].set_title(f'{title_prefix}TC Inter-burst ISI')
        axes[0].set_xlabel('ISI (ms)')

    if len(re_analysis['intra_burst_isi']) > 0:
        axes[1].hist(re_analysis['intra_burst_isi'], bins=20, alpha=0.7, color='blue')
        axes[1].set_title(f'{title_prefix}RE Intra-burst ISI')
        axes[1].set_xlabel('ISI (ms)')

    if len(re_analysis['inter_burst_isi']) > 0:
        axes[2].hist(re_analysis['inter_burst_isi'], bins=20, alpha=0.7, color='red')
        axes[2].set_title(f'{title_prefix}RE Inter-burst ISI')
        axes[2].set_xlabel('ISI (ms)')

    plt.tight_layout()
    return fig


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

def get_default_params():
    """Return default simulation parameters."""
    data = {
        "neuron_type": ["TC", "RE"],
        "C_m": [1250, 1250],
        "t_ref": [2.5, 2.5],
        "E_L": [-60, -60],
        "Delta_T": [2.5, 2.5],
        "V_th": [-55, -50],
        "V_reset": [-60, -60],
        "g_L": [50, 50],
        "a": [300, 400],
        "b": [0, 20],
        "tau_w": [600, 600],
        "I_e": [0, 0],
        "V_m": [-60, -60],
        "V_peak": [20, 20],
        "tau_syn_ex": [5, 5],
        "tau_syn_in": [20, 20],
    }
    
    neuron_dicts = create_neuron_dicts(data)
    
    sim_params = {
        "resolution": 0.001,
        "simtime": 1000.0,
        "duration": 1000.0,
        "n_inh": 1,
        "n_ex": 1,
    }
    
    syn_params = {
        "w_e": 700.0,
        "w_i": -1000.0,
    }
    
    current_params = {
        "amplitude_times": [50, 100],
        "amplitude_values": [1000, 0.0]
    }
    
    return sim_params, neuron_dicts, syn_params, current_params


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Get base parameters
    sim_params, neuron_dicts, syn_params, current_params = get_default_params()
    
    # =========================================================================
    # ANALYSIS 1: Single simulation
    # =========================================================================
    print("Running single simulation...")
    sim_output = sim(sim_params, neuron_dicts, syn_params, current_params)
    metrics = compute_metrics(sim_output, sim_params)
    
    plot_simulation(sim_output, title="Single Simulation")
    plot_isi_distributions(metrics["tc_analysis"], metrics["re_analysis"])
    
    print(f"TC rate: {metrics['tc_rate']:.2f} Hz")
    print(f"RE rate: {metrics['re_rate']:.2f} Hz")
    
    # =========================================================================
    # ANALYSIS 2: w_i sweep (1D)
    # =========================================================================
    print("\nRunning w_i sweep...")
    sim_params["duration"] = 2000.0
    
    results_wi = sweep(sim_params, neuron_dicts, syn_params, current_params, [
        {"param": "w_i", "values": np.linspace(-1000, 0, 20), "label": "$w_i$ (nS)"}
    ])
    plot_sweep_1d(results_wi)
    
    # =========================================================================
    # ANALYSIS 3: w_e sweep (1D)
    # =========================================================================
    print("Running w_e sweep...")
    
    results_we = sweep(sim_params, neuron_dicts, syn_params, current_params, [
        {"param": "w_e", "values": np.linspace(0, 1000, 20), "label": "$w_e$ (nS)"}
    ])
    plot_sweep_1d(results_we)
    
    # =========================================================================
    # ANALYSIS 4: tau_syn_ex vs tau_syn_in sweep (2D)
    # =========================================================================
    print("Running 2D tau sweep...")
    
    results_tau = sweep(sim_params, neuron_dicts, syn_params, current_params, [
        {"param": "tau_syn_ex", "values": np.round(np.linspace(2, 6, 10), 1), "label": "tau_syn_ex (ms)"},
        {"param": "tau_syn_in", "values": np.round(np.linspace(15, 20, 5), 1), "label": "tau_syn_in (ms)"}
    ])
    plot_sweep_2d(results_tau, metric="tc_rate", title="TC Firing Rate")
    
    # =========================================================================
    # Show all figures
    # =========================================================================
    plt.show()
