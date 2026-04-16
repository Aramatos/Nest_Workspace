import numpy as np
import pandas as pd
import nest

# definte a sim block

def sim_build_1(sim_params,neuron_dicts,syn_params, current_source_params):
    nest.ResetKrenel()
    nest.SetKernelStatus({"Resolution":1})

    #Define Neuron Groups
    neuron_group_1 = nest.Create("aeif_cond_exp", n=sim_params["n_1"],params=neuron_dicts["TC"])
    neuron_group_2 = nest.Create("aeif_cond_exp", n=sim_params["n_1"],params=neuron_dicts["RE"])
    neuron_group_3 = nest.Create("aeif_cond_exp", n=sim_params["n_3"],params=neuron_dicts["PV"])
    neuron_group_4 = nest.Create("aeif_cond_exp", n=sim_params["n_4"],params=neuron_dicts["PC"])
    
    #Define Artificial Sources
    step_current= nest.Create("step_current_generator",params=current_source_params)
    #Definte  connectvitiy matrix, 4 by 4 matrix
    # where each row and column represents a neuron group
    # and the value represents the probability of connection in
    # the range of 0 to 1
    connectivity_matrix = np.array([[0, 1, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0]])
    #Connet Groups
    nest.Connect(step_current,neuron_group_1)
    nest.Connect(neuron_group_1,neuron_group_2, syn_spec=syn_params["TC_RE"])
    nest.Connect(neuron_group_2,neuron_group_1, syn_spec=syn_params["RE_TC"])
    nest.Connect(neuron_group_2,neuron_group_3, syn_spec=syn_params["RE_PV"])
    nest.Connect(neuron_group_3,neuron_group_2, syn_spec=syn_params["PV_RE"])
    nest.Connect(neuron_group_3,neuron_group_4, syn_spec=syn_params["PV_PC"])
    nest.Connect(neuron_group_4,neuron_group_3, syn_spec=syn_params["PC_PV"])
    #Connect Neuron Groups with Connectivity Matrix
    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix[i])):
            if connectivity_matrix[i][j] > 0:
                nest.Connect(neuron_group_1, neuron_group_2, syn_spec=syn_params["TC_RE"])
                nest.Connect(neuron_group_2, neuron_group_1, syn_spec=syn_params["RE_TC"])
                nest.Connect(neuron_group_2, neuron_group_3, syn_spec=syn_params["RE_PV"])
                nest.Connect(neuron_group_3, neuron_group_2, syn_spec=syn_params["PV_RE"])
                nest.Connect(neuron_group_3, neuron_group_4, syn_spec=syn_params["PV_PC"])
                nest.Connect(neuron_group_4, neuron_group_3, syn_spec=syn_params["PC_PV"])
    
    #


    
