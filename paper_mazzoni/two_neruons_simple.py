import nest
import numpy as np

from sim_utils import *

def sim(sim_params, neuron_params,syn_params,current_params):
    nest.ResetKernel()
    # Make simulation with higher resolution
    nest.SetKernelStatus({"resolution": 1})

    # Create AdEx neuron
    for i in neuron_params["neuron_types"]:
        neuron1 = nest.Create("aeif_cond_exp", n=sim_params["n_ex"], params=neuron_params["TC"])
        neuron2 = nest.Create("aeif_cond_exp", n=sim_params["n_inh"], params=neuron_params["RE"])
        

    # Create a poisson generator
    step_current = nest.Create("step_current_generator", params=current_params)

    #sni
    #  Connect the step current generator to the neuron
    #nest.Connect(step_current, neuron1)
    nest.Connect(step_current, neuron1)
    nest.Connect(neuron1, neuron2, "all_to_all",syn_spec={'weight': syn_params["w_e"]})
    nest.Connect(neuron2, neuron1, "all_to_all",syn_spec={'weight': syn_params["w_i"]})

    # Create a voltmeter to record the membrane potential
    voltmeter1 = nest.Create("voltmeter")
    voltmeter2 = nest.Create("voltmeter")
    spikemeter1 = nest.Create("spike_recorder")
    spikemeter2 = nest.Create("spike_recorder")

    # Connect the voltmeters and spike meters to the neurons
    nest.Connect(voltmeter1, neuron1)
    nest.Connect(voltmeter2, neuron2)
    nest.Connect(neuron1, spikemeter1)
    nest.Connect(neuron2, spikemeter2)

    nest.Simulate(1000.0)

    return {
        "voltmeters": [voltmeter1, voltmeter2],
        "spikemeters": [spikemeter1, spikemeter2]
    }

neuron_dicts = {
    "neuron_type": [ "TC", "RE"],
    "C_m": [1250, 1250], # pF
    "t_ref": [ 2.5, 2.5], # ms
    "E_L": [ -60, -60], # mV
    "Delta_T": [ 2.5, 2.5], # mV
    "V_th": [ -55, -50], # mV
    "V_reset": [ -60, -60],    
    "g_L": [ 50, 50],
    "a": [ 300, 400], # nS
    "b": [ 0, 20], # pA
    "tau_w": [ 600, 600],
    "I_e": [ 0, 0],
    "V_m": [ -60, -60],
    "V_peak": [20,20],# mV
    "I_e": [0,0],               # pA
    "V_m": [-60.0,-60.0],
    "tau_syn_ex": [ 2, 2],
    "tau_syn_in": [ 20, 20],
}

current_params = {
    "amplitude_times": [50, 100], 
    "amplitude_values": [1000, 0.0]
}

sim_params = {
    "resolution": 0.001,
    "simtime": 1000.0,
    "n_inh": 1,
    "n_ex": 1,
}
syn_params = {
    "w_e": 700.0,  # weight of excitatory connection in nS
    "w_i": -1000.0,  # weight of inhibitory connection in nS
}



sim(sim_params, neuron_dicts, syn_params, current_params)