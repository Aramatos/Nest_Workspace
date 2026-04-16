import numpy as np

def cv_calculation(spike_times):
    if len(spike_times) < 2:
        return float('nan')
    isi = np.diff(spike_times)
    return np.std(isi) / np.mean(isi)

def correlation_coefficient(spike_times1, spike_times2, bin_size=1.0):
    max_time = max(max(spike_times1), max(spike_times2))
    bins = np.arange(0, max_time + bin_size, bin_size)
    
    hist1, _ = np.histogram(spike_times1, bins=bins)
    hist2, _ = np.histogram(spike_times2, bins=bins)
    
    if np.std(hist1) == 0 or np.std(hist2) == 0:
        return float('nan')
    
    return np.corrcoef(hist1, hist2)[0, 1]