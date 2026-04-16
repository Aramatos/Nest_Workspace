
from sim_utils import *

def sim(neuron_params,sim_params,stimulus_params):
    nest.ResetKernel()
    #make simulation with higher resolution

    nest.SetKernelStatus({"resolution": sim_params["resolution"]})

    # Create AdEx neuron
    neuron = nest.Create(sim_params["neuron_model"],params=neuron_params)
    # Set parameters
    
    # Create a step current generator
    step_current = nest.Create(stimulus_params["type"], params=stimulus_params["params"])
    
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

# set inital parameters for the AdEx neuron
adex_params_TC = {
    "C_m": [1250], # pF
    "t_ref": [2.5], # ms
    "E_L": [ -60], # mV
    "Delta_T": [ 2.5], # mV
    "V_th": [ -50], # mV
    "V_reset": [ -60],    
    "g_L": [ 50],
    "a": [ 200], # nS
    "b": [ 0], # pA
    "tau_w": [ 600],
    "I_e": [ 0],
    "V_m": [ -60],
    "V_peak": [20],# mV
    "I_e": [0],# pA
    "V_m": [-60.0],
    "tau_syn_ex": [ .2],
    "tau_syn_in": [ 2],
}

sim_params={
    "resolution":0.001,
    "sim_time": 1000,
    "neuron_model": "aeif_cond_alpha"
}

current_params = {
            "amplitude_times": [100, 1000],
            "amplitude_values": [1000, 0.0]
        }


stimulus_params={
    "type":"step_current_generator",
    "params": current_params
}

# Simulate the neuron
ts, Vms, ts_w_2, w_2, spikes ,senders= sim(adex_params_TC, sim_params,stimulus_params)
plot_single_neuron_simulation(ts, Vms, ts_w_2, w_2,current_params["amplitude_times"])


if __name__ == "__main__":
    SWEEP_KEY=True
    if SWEEP_KEY:
        from sim_utils import parameter_sweep_single_neuron
        parameter_sweep_single_neuron(adex_params_TC, sim_params, stimulus_params, 'b', [0, 20, 40, 60, 80, 100])
