import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nest
import nest.raster_plot
import time
import pandas as pd
import numpy as np

from sim_utils import *

def nueuron_six_single(neuron_dicts):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001})

    # Create neurons
    neurons = [nest.Create("aeif_psc_alpha", 1, params=n_dict) for n_dict in [neuron_dicts["TC"], neuron_dicts["RE"], neuron_dicts["TC"], neuron_dicts["RE"], neuron_dicts["PY"], neuron_dicts["INT"]]]

    # Create spike recorders and voltmeters
    spike_recorders = [nest.Create("spike_recorder") for _ in range(6)]
    voltmeters = [nest.Create("voltmeter") for _ in range(6)]
    multimeters = [nest.Create("multimeter") for _ in range(6)]
    for multimeter in multimeters:
        multimeter.set(record_from=["w"])

    # Create step current generators
    step_currents = [
        nest.Create("step_current_generator", params={"amplitude_times": [100.0, 1000.0], "amplitude_values": [1000.0, 0.0]}),
        nest.Create("step_current_generator", params={"amplitude_times": [100.0, 1000.0], "amplitude_values": [-1000.0, 0.0]})
    ]

    # Connect step currents to neurons
    for i in [0, 1, 4, 5]:
        nest.Connect(step_currents[0], neurons[i])
    for i in [2, 3]:
        nest.Connect(step_currents[1], neurons[i])

    # Connect voltmeters, multimeters and spike recorders to neurons
    for vm, sm, neuron, mm in zip(voltmeters, spike_recorders, neurons, multimeters):
        nest.Connect(vm, neuron)
        nest.Connect(neuron, sm)
        nest.Connect(mm, neuron)

    # Simulate the network
    nest.Simulate(2000.0)

    return voltmeters, multimeters 


data = {
    "neuron_type": ["PY", "INT", "TC", "RE"],
    "C_m": [500, 200, 1250, 1250], # pF
    "t_ref": [2, 2, 2.5, 2.5], # ms
    "E_L": [-70, -70, -60, -60], # mV
    "Delta_T": [2.5, 2.5, 2.5, 2.5], # mV
    "V_th": [-52, -52, -50, -50], # mV
    "V_reset": [-59, -59, -60, -60],    
    "g_L": [25, 20, 50, 50],
    "a": [1, 1, 200, 400], # nS
    "b": [600, 0, 0, 20], # pA
    "tau_w": [600, 600, 600, 600],
    "I_e": [0, 0, 0, 0],
    "V_m": [-70, -70, -60, -60],
    "V_peak": [20.0,20,20,20],# mV
    "I_e": [0,0,0,0],               # pA
}

neuron_dicts = create_neuron_dicts(data)


voltmeters, multimeters = nueuron_six_single(neuron_dicts)
plot_results(voltmeters, multimeters)

input_strength = np.arange(0.0, 1500.0, 200.0)
print(input_strength)


def pulse_curves(neuron_params, input_strength, sim_time=2000.0):
    Vms_array = []
    Vms_ts_array = []
    Spikes_ts_array = []
    Senders_array = []
    for current in input_strength:
        current_params = {
            "amplitude_times": [500, 1500.0],
            "amplitude_values": [current, 0.0]
        }
        ts, Vms, ts_w_2, w_2, ts_spikes, senders = simulate_single_neuron(neuron_params, current_params, sim_time)
    
        Vms_array.append(Vms)
        Vms_ts_array.append(ts)
        Spikes_ts_array.append(ts_spikes)
        Senders_array.append(senders)
    return Vms_array, Vms_ts_array, Spikes_ts_array, Senders_array

