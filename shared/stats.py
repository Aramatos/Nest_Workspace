"""
Spike-train statistics — shared across all experiments.

Place at:  NEST_Workspace/shared/stats.py
"""

import numpy as np


def cv_isi(spike_times, senders, n_neurons):
    """Mean coefficient of variation of inter-spike intervals."""
    cv_list = []
    for nid in range(1, n_neurons + 1):
        sp = np.unique(spike_times[senders == nid])
        if len(sp) > 2:
            isi = np.diff(sp)
            mu = np.mean(isi)
            if mu > 0:
                cv_list.append(np.std(isi) / mu)
    return np.nan if len(cv_list) == 0 else np.mean(cv_list)


def pairwise_correlation(spike_times, senders, n_neurons,
                         bin_size=10.0, max_sample=500, rng_seed=42):
    """Mean pairwise Pearson correlation of binned spike counts."""
    if len(spike_times) == 0:
        return np.nan
    bins = np.arange(0, np.max(spike_times) + bin_size, bin_size)
    M = np.zeros((n_neurons, len(bins) - 1))
    for nid in range(1, n_neurons + 1):
        M[nid - 1], _ = np.histogram(spike_times[senders == nid], bins=bins)
    M = M[np.std(M, axis=1) > 0]
    if M.shape[0] < 2:
        return np.nan
    if M.shape[0] > max_sample:
        rng = np.random.default_rng(rng_seed)
        M = M[rng.choice(M.shape[0], max_sample, replace=False)]
    C = np.corrcoef(M)
    iu = np.triu_indices(C.shape[0], k=1)
    return np.nanmean(C[iu])


def fano_factor(spike_times, senders, n_neurons,
                bin_size=10.0, t_start=0.0, t_end=None):
    """Mean Fano factor (Var/Mean of spike counts per bin) across neurons."""
    if len(spike_times) == 0:
        return np.nan
    if t_end is None:
        t_end = np.max(spike_times)
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    ff_list = []
    for nid in range(1, n_neurons + 1):
        counts, _ = np.histogram(spike_times[senders == nid], bins=bins)
        mu = np.mean(counts)
        if mu > 0:
            ff_list.append(np.var(counts) / mu)
    return np.nan if len(ff_list) == 0 else np.mean(ff_list)
