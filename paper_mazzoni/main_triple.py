import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nest
import nest.raster_plot
import time

def log_parameters(N, Ne, Ni1, Ni2, neuron_data, connection_matrix, input_strength, bottom_up_input, top_down_input):
    print("Simulation Parameters:")
    print(f"Total neurons: {N}")
    print(f"Excitatory neurons: {Ne}")
    print(f"Inhibitory group 1 neurons: {Ni1}")
    print(f"Inhibitory group 2 neurons: {Ni2}")
    print("\nNeuron Data:")
    print(neuron_data)
    print("\nConnection Matrix:")
    print(connection_matrix)
    print("\nInput Strength:")
    print(input_strength)
    print("\nBottom-up Input:")
    print(bottom_up_input)
    print("\nTop-down Input:")
    print(top_down_input)

def run_simulation(neuron_data, connection_matrix, input_strength, bottom_up_input, top_down_input, runtime=1000.0, plot=False, num_threads=1):
    N = len(neuron_data)
    Ne = len(neuron_data[neuron_data['type'] == 'excitatory'])
    Ni1 = len(neuron_data[neuron_data['type'] == 'inhibitory_1'])
    Ni2 = len(neuron_data[neuron_data['type'] == 'inhibitory_2'])
    
    log_parameters(N, Ne, Ni1, Ni2, neuron_data, connection_matrix, input_strength, bottom_up_input, top_down_input)
    
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': num_threads})
    
    # Create neuron populations with NEST
    neurons = nest.Create('aeif_cond_exp', N)
    
    for i, neuron in enumerate(neurons):
        nest.SetStatus(int[(neuron)], {
            'tau_m': neuron_data['tau_m'].values[i],
            'C_m': neuron_data['cm'].values[i],
            'E_L': neuron_data['v_rest'].values[i],
            'V_reset': neuron_data['v_reset'].values[i],
            'V_th': neuron_data['v_thresh'].values[i],
            'E_ex': neuron_data['e_rev_E'].values[i],
            'E_in': neuron_data['e_rev_I'].values[i],
            'tau_syn_ex': neuron_data['tau_syn_E'].values[i],
            'tau_syn_in': neuron_data['tau_syn_I'].values[i],
            'a': neuron_data['a'].values[i],
            'b': neuron_data['b'].values[i],
            'tau_w': neuron_data['tau_w'].values[i],
            'I_e': neuron_data['i_offset'].values[i] + bottom_up_input[i] + top_down_input[i],
        })
    
    # Create synapses based on connection matrix
    groups = [
        neuron_data[neuron_data['type'] == 'excitatory'].index,
        neuron_data[neuron_data['type'] == 'inhibitory_1'].index,
        neuron_data[neuron_data['type'] == 'inhibitory_2'].index
    ]
    
    for i, pre_group in enumerate(groups):
        for j, post_group in enumerate(groups):
            if connection_matrix[i, j] > 0:
                conn_dict = {'rule': 'fixed_indegree', 'indegree': int(connection_matrix[i, j] * len(pre_group))}
                syn_dict = {'weight': input_strength[neuron_data.iloc[pre_group[0]]['type']], 'delay': 1.0}
                nest.Connect(neurons[pre_group.to_list()], neurons[post_group.to_list()], conn_dict, syn_dict)
    
    # Set up recording
    spike_recorder = nest.Create('spike_recorder')
    nest.Connect(neurons, spike_recorder)
    
    # Run the simulation
    start_time = time.time()
    nest.Simulate(runtime)
    end_time = time.time()
    
    # Get recorded data
    spikes = nest.GetStatus(spike_recorder, 'events')[0]
    
    # Log simulation time
    print(f"Simulation runtime: {end_time - start_time} seconds")

    if plot:
        nest.raster_plot.from_device(spike_recorder)
        plt.show()

    return spikes

# Define parameters
N = 100  # Total number of neurons
Ne_ratio = 0.8  # Ratio of excitatory neurons
Ni1_ratio = 0.1  # Ratio of first group of inhibitory neurons
Ni2_ratio = 0.1  # Ratio of second group of inhibitory neurons

# Calculate the number of neurons based on the ratios
Ne = int(N * Ne_ratio)
Ni1 = int(N * Ni1_ratio)
Ni2 = int(N * Ni2_ratio)

# Create neuron data DataFrame
neuron_data = pd.DataFrame(index=np.arange(N))
neuron_data['type'] = ['excitatory'] * Ne + ['inhibitory_1'] * Ni1 + ['inhibitory_2'] * Ni2

# Assign different parameters to each neuron type
# Parameters for excitatory neurons
neuron_data.loc[neuron_data['type'] == 'excitatory', 'tau_m'] = 20.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'cm'] = 0.2
neuron_data.loc[neuron_data['type'] == 'excitatory', 'v_rest'] = -65.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'v_reset'] = -65.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'v_thresh'] = -50.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'e_rev_E'] = 0.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'e_rev_I'] = -80.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'tau_syn_E'] = 5.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'tau_syn_I'] = 10.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'a'] = 1
neuron_data.loc[neuron_data['type'] == 'excitatory', 'b'] = 0.02
neuron_data.loc[neuron_data['type'] == 'excitatory', 'tau_w'] = 600.0
neuron_data.loc[neuron_data['type'] == 'excitatory', 'i_offset'] = 4 * np.random.rand(Ne)

# Parameters for first group of inhibitory neurons
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'tau_m'] = 10.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'cm'] = 0.1
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'v_rest'] = -70.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'v_reset'] = -65.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'v_thresh'] = -50.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'e_rev_E'] = 0.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'e_rev_I'] = -75.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'tau_syn_E'] = 3.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'tau_syn_I'] = 8.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'a'] = 1
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'b'] = 0.015
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'tau_w'] = 600.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_1', 'i_offset'] = 2 * np.random.rand(Ni1)
# Parameters for second group of inhibitory neurons
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'tau_m'] = 16.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'cm'] = 0.15
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'v_rest'] = -68.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'v_reset'] = -65.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'v_thresh'] = -52.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'e_rev_E'] = 0.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'e_rev_I'] = -78.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'tau_syn_E'] = 4.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'tau_syn_I'] = 9.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'a'] = 0.003
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'b'] = 0.018
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'tau_w'] = 600.0
neuron_data.loc[neuron_data['type'] == 'inhibitory_2', 'i_offset'] = 2 * np.random.rand(Ni2)

# Define a 3x3 connection matrix indicating connection probabilities between groups
# Rows represent outputs and columns represent inputs: [excitatory, inhibitory_1, inhibitory_2]
connection_matrix = np.array([
    [0.02, 0.01, 0.01],  # Outputs from excitatory neurons
    [0.04, 0.02, 0.01],  # Outputs from inhibitory_1 neurons
    [0.04, 0.01, 0.02]   # Outputs from inhibitory_2 neurons
])
# Define input strength for each neuron group
input_strength = {
    'excitatory': 2.0,
    'inhibitory_1': 4.0,
    'inhibitory_2': 5.0
}
# Define bottom-up and top-down input vectors
bottom_up_input = np.random.rand(N)
top_down_input = np.random.rand(N)

# Example usage
results = run_simulation(neuron_data, connection_matrix, input_strength, bottom_up_input, top_down_input, runtime=1000.0, plot=True)

