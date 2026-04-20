# %%
"""
Visualize the 4D AI-regime sweep for the E-I balanced network.

Layout per metric figure:
    rows  =  N_total  (ascending ↑)
    cols  =  J_mV     (ascending →)
    cell  =  heatmap of  g (x)  ×  epsilon (y)

Grey cells with × = no spiking activity (fr=0); stats are NaN.

Designed for VS Code Python Interactive (# %% cells).
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(EXPERIMENT_DIR, "results")
LOG_PATH       = os.path.join(RESULTS_DIR, "sweep_4d_ai.jsonl")

records = []
with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from {LOG_PATH}")


# ─────────────────────────────────────────────────────────────────
# Discover grid axes
# ─────────────────────────────────────────────────────────────────
def axis_vals(key):
    return np.array(sorted({r["params"][key] for r in records}))

N_vals   = axis_vals("N_total")
J_vals   = axis_vals("J_mV")
g_vals   = axis_vals("g")
eps_vals = axis_vals("epsilon")

n_N, n_J, n_g, n_eps = len(N_vals), len(J_vals), len(g_vals), len(eps_vals)

lookup = {}
for r in records:
    p = r["params"]
    lookup[(p["N_total"], p["J_mV"], p["g"], p["epsilon"])] = r["metrics"]

n_firing = sum(1 for r in records if r["metrics"]["fr_E"] > 0)
n_silent = len(records) - n_firing
print(f"  Active: {n_firing}/{len(records)}   Silent: {n_silent}/{len(records)}\n")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def slice_2d(metric, N, J):
    Z = np.full((n_eps, n_g), np.nan)
    for ie, e in enumerate(eps_vals):
        for ig, g in enumerate(g_vals):
            m = lookup.get((N, J, g, e))
            if m is not None:
                Z[ie, ig] = m[metric]
    return Z


def silent_mask(N, J):
    fr = slice_2d("fr_E", N, J)
    return (fr == 0.0) | np.isnan(fr)


# ─────────────────────────────────────────────────────────────────
# Core plotting function
# ─────────────────────────────────────────────────────────────────
def plot_metric(metric, title, cmap, norm, fmt=".2f",
                annot_thresh=None, save_name=None):
    """
    Grid-of-heatmaps: rows=N, cols=J, each cell = g(x) × ε(y).

    Parameters
    ----------
    metric       : key into the metrics dict
    title        : figure suptitle
    cmap         : matplotlib colormap name
    norm         : Normalize instance (shared across all panels)
    fmt          : format string for cell annotations
    annot_thresh : if given, only annotate cells where |value| > thresh
    save_name    : filename (no path) for PNG; None = don't save
    """
    cell_w, cell_h = 1.6, 1.3
    fig_w = cell_w * n_J * n_g  + 2.8
    fig_h = cell_h * n_N * n_eps + 2.2

    fig, axes = plt.subplots(
        n_N, n_J,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )

    for iN, N in enumerate(N_vals):
        for iJ, J in enumerate(J_vals):
            ax = axes[iN, iJ]
            Z = slice_2d(metric, N, J)
            mask = silent_mask(N, J)

            # Grey background for silent cells
            grey = np.where(mask, 0.0, np.nan)
            ax.imshow(grey, aspect="auto",
                      cmap=mcolors.ListedColormap(["#E0E0E0"]),
                      vmin=0, vmax=1, origin="lower",
                      extent=[-0.5, n_g - 0.5, -0.5, n_eps - 0.5])

            # Data heatmap (NaN where silent → transparent)
            Z_plot = np.where(mask, np.nan, Z)
            im = ax.imshow(Z_plot, aspect="auto", cmap=cmap, norm=norm,
                           origin="lower",
                           extent=[-0.5, n_g - 0.5, -0.5, n_eps - 0.5])

            # Cell annotations
            outline = [pe.withStroke(linewidth=2, foreground="white")]
            for ie in range(n_eps):
                for ig in range(n_g):
                    if mask[ie, ig]:
                        ax.text(ig, ie, "×", ha="center", va="center",
                                fontsize=8, color="#888888",
                                fontweight="bold")
                    elif not np.isnan(Z[ie, ig]):
                        val = Z[ie, ig]
                        if annot_thresh is not None and abs(val) < annot_thresh:
                            continue
                        ax.text(ig, ie, f"{val:{fmt}}",
                                ha="center", va="center",
                                fontsize=6.5, color="black",
                                path_effects=outline)

            # Ticks
            ax.set_xticks(range(n_g))
            ax.set_xticklabels([f"{g:g}" for g in g_vals], fontsize=7)
            ax.set_yticks(range(n_eps))
            ax.set_yticklabels([f"{e:g}" for e in eps_vals], fontsize=7)

            # Axis labels only on edges
            if iN == n_N - 1:
                ax.set_xlabel("g", fontsize=9)
            else:
                ax.set_xticklabels([])
            if iJ == 0:
                ax.set_ylabel("ε", fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.set_title(f"N={int(N)}  J={J}", fontsize=8, pad=4)

    # Colour bar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 cax=cbar_ax, label=metric)

    fig.suptitle(f"{title}\n",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94,
             f"grey × = no activity  ({n_silent}/{len(records)} combos)",
             ha="center", fontsize=8, color="#777")

    if save_name:
        path = os.path.join(RESULTS_DIR, save_name)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  Saved {save_name}")

    return fig


# ─────────────────────────────────────────────────────────────────
# Safe TwoSlopeNorm
# ─────────────────────────────────────────────────────────────────
def _twoslope(all_Z, center):
    flat = all_Z[~np.isnan(all_Z)]
    if len(flat) == 0:
        return mcolors.Normalize(0, 1)
    vmin = min(float(np.nanmin(flat)), center - 1e-6)
    vmax = max(float(np.nanmax(flat)), center + 1e-6)
    return mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)


def _global_range(metric):
    """All valid (non-silent, non-NaN) values across the full 4D grid."""
    vals = []
    for N in N_vals:
        for J in J_vals:
            Z = slice_2d(metric, N, J)
            mask = silent_mask(N, J)
            v = Z[~mask & ~np.isnan(Z)]
            if len(v):
                vals.append(v)
    return np.concatenate(vals) if vals else np.array([0.0])


# %% ── Firing Rate ──────────────────────────────────────────────
all_fr = _global_range("fr_E")
plot_metric("fr_E", "Firing Rate — E (Hz)", "viridis",
            norm=mcolors.Normalize(vmin=0, vmax=float(np.nanmax(all_fr))),
            fmt=".1f", save_name="firing_rate_E.png")
plt.show()

# %% ── CV(ISI) ──────────────────────────────────────────────────
all_cv = _global_range("cv_E")
plot_metric("cv_E", "CV(ISI) — E", "RdBu_r",
            norm=_twoslope(all_cv, 1.0),
            fmt=".2f", save_name="cv_isi_E.png")
plt.show()

# %% ── Pairwise Correlation ─────────────────────────────────────
all_corr = _global_range("corr_E")
plot_metric("corr_E", "Mean Pairwise Correlation — E", "RdBu_r",
            norm=_twoslope(all_corr, 0.0),
            fmt=".3f", save_name="correlation_E.png")
plt.show()

# %% ── Fano Factor ──────────────────────────────────────────────
all_fano = _global_range("fano_E")
plot_metric("fano_E", "Fano Factor — E (10 ms bins)", "RdBu_r",
            norm=_twoslope(np.clip(all_fano, 0, 3), 1.0),
            fmt=".2f", save_name="fano_factor_E.png")
plt.show()

# %% ── AI-Regime Composite Score ────────────────────────────────
CV_TOL, CORR_TOL, FANO_TOL = 0.5, 0.05, 0.5


def ai_score_2d(N, J):
    cv   = slice_2d("cv_E",   N, J)
    corr = slice_2d("corr_E", N, J)
    fano = slice_2d("fano_E", N, J)
    s_cv   = 1 - np.clip(np.abs(cv   - 1.0) / CV_TOL,   0, 1)
    s_corr = 1 - np.clip(np.abs(corr - 0.0) / CORR_TOL, 0, 1)
    s_fano = 1 - np.clip(np.abs(fano - 1.0) / FANO_TOL, 0, 1)
    return s_cv * s_corr * s_fano


# Inject ai_score into lookup so plot_metric can read it
for N in N_vals:
    for J in J_vals:
        Z = ai_score_2d(N, J)
        mask = silent_mask(N, J)
        for ie, e in enumerate(eps_vals):
            for ig, g in enumerate(g_vals):
                key = (N, J, g, e)
                if key in lookup:
                    lookup[key]["ai_score"] = (
                        float(Z[ie, ig]) if not mask[ie, ig] else float("nan")
                    )

plot_metric(
    "ai_score",
    (f"AI-Regime Score (E)\n"
     f"CV≈1 ± {CV_TOL}   |Corr| ≤ {CORR_TOL}   Fano≈1 ± {FANO_TOL}"),
    "magma",
    norm=mcolors.Normalize(vmin=0, vmax=1),
    fmt=".2f", save_name="ai_score_E.png",
)
plt.show()

# %% ── Summary ──────────────────────────────────────────────────
print(f"\nAll PNGs saved to: {RESULTS_DIR}")
print(f"  {n_silent}/{len(records)} combos had no activity")