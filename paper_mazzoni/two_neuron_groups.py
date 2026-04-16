import nest
import numpy as np


from sim_utils import *

def sim(sim_params, neuron_params,syn_params,current_params):
    nest. ResetKernel()
    # Make simulation with higher resolution
    nest.SetKernelStatus({"resolution": 1})




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
