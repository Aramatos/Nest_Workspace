import nest
import numpy as np
import matplotlib.pyplot as plt

# Reset NEST kernel
nest.ResetKernel()

# Set random seed for reproducibility
nest.SetKernelStatus({"rng_seed": 123})

# Define neuron models (LIF neurons)
# ----------------------------------
# Excitatory neurons
exc_neuron_params = {
    "E_L": -70.0,      # Resting membrane potential (mV)
    "V_th": -55.0,     # Spike threshold (mV)
    "V_reset": -70.0,  # Reset potential (mV)
    "t_ref": 2.0,      # Refractory period (ms)
    "C_m": 250.0,      # Membrane capacitance (pF)
    "tau_m": 20.0,     # Membrane time constant (ms)
    "tau_syn_ex": 2.0, # Rise time of excitatory synaptic conductance (ms)
    "tau_syn_in": 2.0  # Rise time of inhibitory synaptic conductance (ms)
}

# Inhibitory neurons (different time constants)
inh_neuron_params = {
    "E_L": -70.0,      # Resting membrane potential (mV)
    "V_th": -55.0,     # Spike threshold (mV)
    "V_reset": -70.0,  # Reset potential (mV)
    "t_ref": 2.0,      # Refractory period (ms)
    "C_m": 250.0,      # Membrane capacitance (pF)
    "tau_m": 10.0,     # Membrane time constant (ms) - faster than excitatory
    "tau_syn_ex": 2.0, # Rise time of excitatory synaptic conductance (ms)
    "tau_syn_in": 2.0  # Rise time of inhibitory synaptic conductance (ms)
}

# Create neuron models
exc_model = "iaf_psc_exp"  # Leaky integrate-and-fire neuron with exponential PSCs
inh_model = "iaf_psc_exp"  # Same model but with different parameters

# Function to create neuron groups
# -------------------------------
def create_neuron_group(name, model, params, count):
    """Create a group of neurons with specified parameters.
    
    Args:
        name: Name for the neuron group
        model: NEST neuron model name
        params: Dictionary of neuron parameters
        count: Number of neurons in the group
        
    Returns:
        ID of the neuron group
    """
    neurons = nest.Create(model, count, params)
    # For easier identification, store the name as a parameter
    # This is purely for bookkeeping and not used by NEST internally
    nest.SetStatus(neurons, {"group_name": name})
    return neurons

# Function to connect neuron groups
# -------------------------------
def connect_groups(source, target, conn_spec, syn_spec):
    """Connect two neuron groups.
    
    Args:
        source: Source neuron group
        target: Target neuron group
        conn_spec: Connection specification dict
        syn_spec: Synapse specification dict
    """
    nest.Connect(source, target, conn_spec, syn_spec)

# Create 6 neuron groups (2 of them single neurons)
# ------------------------------------------------
# Define population sizes (can be easily modified)
population_sizes = {
    "group1": 50,      # Excitatory group
    "group2": 30,      # Inhibitory group
    "group3": 50,      # Excitatory group
    "group4": 30,      # Inhibitory group
    "group5": 1,       # Single excitatory neuron
    "group6": 1        # Single inhibitory neuron
}

# Define which groups are excitatory/inhibitory
neuron_types = {
    "group1": "excitatory",
    "group2": "inhibitory",
    "group3": "excitatory",
    "group4": "inhibitory",
    "group5": "excitatory",  # Single neuron
    "group6": "inhibitory"   # Single neuron
}

# Create all neuron groups
neuron_groups = {}
for name, size in population_sizes.items():
    if neuron_types[name] == "excitatory":
        neuron_groups[name] = create_neuron_group(name, exc_model, exc_neuron_params, size)
    else:
        neuron_groups[name] = create_neuron_group(name, inh_model, inh_neuron_params, size)

# Connection parameters (easily modifiable)
# ----------------------------------------
# Weight of excitatory connections
w_ex = 10.0
# Weight of inhibitory connections (negative for inhibitory effect)
w_in = -30.0
# Connection probability (1.0 = all-to-all)
conn_probability = 1.0
# Synapse specifications
syn_spec_ex = {"weight": w_ex, "delay": 1.0}
syn_spec_in = {"weight": w_in, "delay": 1.0}
# Connection specifications
conn_spec_input = {
    "rule": "pairwise_bernoulli",
    "p": conn_probability
}

# Connect neuron groups
connect_groups(neuron_groups["group1"], neuron_groups["group2"], conn_spec, syn_spec_ex)
connect_groups(neuron_groups["group2"], neuron_groups["group1"], conn_spec, syn_spec_in)
connect_groups(neuron_groups["group3"], neuron_groups["group4"], conn_spec, syn_spec_ex)
connect_groups(neuron_groups["group4"], neuron_groups["group3"], conn_spec, syn_spec_in)
connect_groups(neuron_groups["group5"], neuron_groups["group6"], conn_spec, syn_spec_ex)
connect_groups(neuron_groups["group6"], neuron_groups["group5"], conn_spec, syn_spec_in)
# Connect single neurons to groups (can be modified)

# Add spike recorders to monitor activity
# -------------------------------------
spike_recorders = {}
for name, neurons in neuron_groups.items():
    spike_recorders[name] = nest.Create("spike_recorder")
    nest.Connect(neurons, spike_recorders[name])

# Add a DC input to stimulate the network
# -------------------------------------
dc_generator = nest.Create("dc_generator", 1, {"amplitude": 500.0})

# Connect DC generator to excitatory groups (can be modified)
for name, neurons in neuron_groups.items():
    if neuron_types[name] == "excitatory":
        nest.Connect(dc_generator, neurons)

# Run simulation
# ------------
simulation_time = 500.0  # ms
nest.Simulate(simulation_time)

# Plot results
# ----------
plt.figure(figsize=(12, 8))

# Color mapping for neuron types
colors = {
    "excitatory": "red",
    "inhibitory": "blue"
}

# Plot spike raster for each group
for i, (name, recorder) in enumerate(spike_recorders.items()):
    events = nest.GetStatus(recorder, "events")[0]
    senders = events["senders"]
    times = events["times"]
    
    # If there are spikes, plot them
    if len(times) > 0:
        plt.plot(times, 
                 senders, 
                 ".", 
                 color=colors[neuron_types[name]], 
                 label=f"{name} ({neuron_types[name]})")

plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Spike Raster Plot")
plt.legend()
plt.grid(True)
plt.show()

# Print some statistics
for name, recorder in spike_recorders.items():
    events = nest.GetStatus(recorder, "events")[0]
    n_spikes = len(events["times"])
    n_neurons = population_sizes[name]
    
    if n_neurons > 0:
        rate = (n_spikes / n_neurons) / (simulation_time / 1000.0)
        print(f"{name} ({neuron_types[name]}): {n_spikes} spikes, {rate:.2f} Hz per neuron")