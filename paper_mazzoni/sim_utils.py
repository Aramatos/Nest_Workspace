import pandas as pd
import nest
import numpy as np
import os
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

from matplotlib import pyplot as plt

def create_neuron_dicts(data):
    df = pd.DataFrame(data)
    neuron_dicts = {}
    for i, neuron_type in enumerate(df["neuron_type"]):
        neuron_dict = df.loc[i].to_dict()
        neuron_dict.pop("neuron_type")
        neuron_dicts[neuron_type] = neuron_dict
    return neuron_dicts


#replace the dicitonary keys with the corresponding parameter names
def create_parameter_table(adex_params_TC):
    parameter_names = {
        "C_m": "Cm (pF)",  # Capacitance in picoFarads
        "t_ref": "τ_ref (ms)",  # Refractory period in milliseconds
        "E_L": "E_L (mV)",  # Resting potential in millivolts
        "Delta_T": "ΔT (mV)",  # Slope factor in millivolts
        "V_th": "V_th (mV)",  # Threshold potential in millivolts
        "V_reset": "V_reset (mV)",  # Reset potential in millivolts
        "g_L": "g_L (nS)",  # Leak conductance in nanoSiemens
        "a": "a (nS)",  # Subthreshold adaptation in nanoSiemens
        "b": "b (pA)",  # Spike-triggered adaptation in picoAmperes
        "tau_w": "τ_w (ms)",  # Adaptation time constant in milliseconds
        "I_e": "I_e (pA)",  # External current in picoAmperes
        "V_m": "V_m (mV)",  # Membrane potential in millivolts
        "V_peak": "V_peak (mV)",  # Peak potential in millivolts
        "tau_syn_ex": "τ_syn_ex (ms)",  # Excitatory synaptic time constant in milliseconds
        "tau_syn_in": "τ_syn_in (ms)",  # Inhibitory synaptic time constant in milliseconds
    }

    dictionary = {parameter_names[key]: value for key, value in adex_params_TC.items()}

    data = pd.DataFrame.from_dict(dictionary, orient='index', columns=["Value"]).reset_index()
    data.columns = ["Parameter", "Value"]

    # Plot the DataFrame as a table
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')

    # Remove lines of the table
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor('none')  # Set background to transparent

    # Decrease space inside the cells
    table.scale(.8, 2)

    # Control font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Save the table as an image
    plt.savefig("adex_params_table.png", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()



def simulate_single_neuron_mult(neuron_params,syn_params,sim_params,stimulus_params):
    nest.ResetKernel()
    #make simulation with higher resolution

    nest.SetKernelStatus({"resolution": sim_params["resolution"]})

    # Create AdEx neuron
    neuron = nest.Create(sim_params["neuron_model"],params=neuron_params)
    # Set parameters
    nest.SetStatus(neuron, syn_params)
    
    if stimulus_params["type"]=="step_current_generator":
        # Create a step current generator
        step_current = nest.Create("step_current_generator", params=stimulus_params["params"])
        
        # Connect the step current generator to the neuron
        nest.Connect(step_current, neuron)
    elif stimulus_params["type"]=="spike_generator":
        # Create a spike generator
        spike_generator = nest.Create("spike_generator", params= stimulus_params["params"])
        
        # Connect the spike generator to the neuron
        nest.Connect(spike_generator, neuron,syn_spec={"synapse_model": "static_synapse", "receptor_type": 1 , "weight": 1})
    
    # Create a voltmeter to record the membrane potential
    voltmeter = nest.Create("voltmeter")
    multimeter = nest.Create("multimeter")
    spike_recorder = nest.Create("spike_recorder")
    multimeter.set(record_from=["w"])

    # Connect the voltmeter to the neuron
    nest.Connect(multimeter, neuron)
    nest.Connect(voltmeter, neuron)
    nest.Connect(neuron, spike_recorder)

    # Simulate the network
    nest.Simulate(sim_params["sim_time"])
    
    # Retrieve data from the voltmeter
    dmm = nest.GetStatus(voltmeter)[0]
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    dmm = multimeter.get()
    w_2 = dmm["events"]["w"]
    ts_w_2 = dmm["events"]["times"]
    
    events = spike_recorder.get("events")
    senders = events["senders"]
    ts_spikes = events["times"]
    
    return ts, Vms, ts_w_2, w_2, ts_spikes, senders 

def plot_single_neuron_simulation(ts, Vms, ts_w_2, w_2, v_lines, save_path=None, show=True):
    [i1,i2]=v_lines
    # Set font parameters
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 18,
            }

    # Plot the voltage trace and adaptation variable
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the voltage trace
    axs[0].plot(ts, Vms, label="Membrane potential")
    if i1!=0:
        axs[0].axvline(x=i1, color='k', linestyle='--')
    if i2!=0:
        axs[0].axvline(x=i2, color='k', linestyle='--')

    axs[0].set_title("Membrane Potential (mV)", fontdict=font)
    axs[0].set_ylabel("Membrane potential (mV)", fontdict=font)
    axs[0].legend()
    axs[0].grid(True)

    # Plot the adaptation variable
    axs[1].plot(ts_w_2, w_2/1000, label="Adaptation variable W( nA)", color='r')
    if i1!=0:
        axs[1].axvline(x=i1, color='k', linestyle='--')
    if i2!=0:
        axs[1].axvline(x=i2, color='k', linestyle='--')
    axs[1].set_title("Adaptation Current w (nA)", fontdict=font)
    axs[1].set_xlabel("Time (ms)", fontdict=font)
    axs[1].set_ylabel("Adaptation variable (w)", fontdict=font)
    axs[1].legend()
    axs[1].grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# Plot the results
def plot_results(voltmeters, multimeters):
    # Set font parameters
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 18,
            }

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 18), constrained_layout=True)
    titles = [
        "TC Neuron Depolarization", 
        "RE Neuron Depolarization", 
        "TC Neuron Hyperpolarization", 
        "RE Neuron Hyperpolarization", 
        "PYR Neuron Depolarization", 
        "INT Neuron Depolarization"
    ]
    list_1=[0,0,2,2,4,4]
    list_2=[1,1,3,3,5,5]
    list_3=[0,1,0,1,0,1]

    for i in range(6):
        # Extract V_m data
        dmm_vm = nest.GetStatus(voltmeters[i])[0]
        Vms = dmm_vm["events"]["V_m"]
        ts = dmm_vm["events"]["times"]
        
        j=list_1[i]
        k=list_3[i]
        l=list_2[i]

        #V_m sublpot
        axes[j, k].plot(ts, Vms, label=f"{titles[i]} - V_m", color='g')
        axes[j, k].axvline(x=100, color='k', linestyle='--')
        axes[j, k].axvline(x=1000, color='k', linestyle='--')
        axes[j, k].set_title(f"{titles[i]} - V_m", fontdict=font)
        axes[j, k].set_ylabel("V_m [mV]", fontdict=font)
        axes[j, k].set_ylim([-80, 0])
        axes[j, k].grid(True)

        # Extract w data
        dmm_mm = nest.GetStatus(multimeters[i])[0]
        ws = dmm_mm["events"]["w"]
        ts_w = dmm_mm["events"]["times"]

        # W_Subplot
        axes[l, k].plot(ts_w, ws/1000, label=f"{titles[i]} - w", color='g')
        axes[l, k].axvline(x=100, color='k', linestyle='--')
        axes[l, k].axvline(x=1000, color='k', linestyle='--')
        axes[l, k].set_ylabel("w [nA]", fontdict=font)
        axes[l, k].set_xlabel("Time [ms]", fontdict=font)
        axes[l, k].grid(True)

    fig.savefig('4_neuron_simulation.png')
    plt.show()

def FI_curve(neuron_params, input_strength, sim_time=1000.0):
    output_rates = []
    for current in input_strength:
        current_params = {
            "amplitude_times": [1.0, 1000.0],
            "amplitude_values": [current, 0.0]
        }
        ts, Vms, ts_w_2, w_2 = simulate_single_neuron(neuron_params, current_params, sim_time)
        # Detect spikes
        spikes = np.where(np.diff(Vms > neuron_params["V_th"], prepend=False))[0]
        # Calculate the output rate
        output_rate = len(spikes) / (sim_time / 1000.0)  # Convert to Hz
        output_rates.append(output_rate)
        
    return output_rates


def plot_FT_curve(output_rates, input_strength,name):
    fig, ax = plt.subplots()
    ax.plot(input_strength, output_rates)
    ax.set_title(f"F-I curve {name} neuron")
    ax.set_xlabel("Input current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.grid(True)
    plt.savefig('FI_curve.png')

def simulate_two_neruon_pop(neuron_dicts, neuron,current_params, sim_time=2000.0):
    nest.ResetKernel()
    #make simulation with higher resolution

    # Create AdEx neuron
    neuron1 = nest.Create("aeif_cond_exp",params=neuron_dicts["TC"])
    neuron2 = nest.Create("aeif_cond_exp",params=neuron_dicts["RE"])
    # Set parameters
    
    # Create a step current generator
    step_current = nest.Create("step_current_generator", params=current_params)
    
    # Connect the step current generator to the neuron
    nest.Connect(step_current, neuron)

    
    # Create a voltmeter to record the membrane potential
    voltmeter = nest.Create("voltmeter")
    multimeter = nest.Create("multimeter")
    spike_recorder = nest.Create("spike_recorder")
    multimeter.set(record_from=["w"])


    
    # Connect the voltmeter to the neuron
    nest.Connect(multimeter, neuron)
    nest.Connect(voltmeter, neuron)
    nest.Connect(neuron, spike_recorder)

    # Simulate the network
    nest.Simulate(sim_time)
    
    
    # Retrieve data from the voltmeter
    dmm = nest.GetStatus(voltmeter)[0]
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    dmm = multimeter.get()
    w_2 = dmm["events"]["w"]
    ts_w_2 = dmm["events"]["times"]
    
    events = spike_recorder.get("events")
    senders = events["senders"]
    ts_spikes = events["times"]
    
    return ts, Vms, ts_w_2, w_2, ts_spikes, senders 
