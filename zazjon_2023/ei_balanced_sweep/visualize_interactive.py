# %% [markdown]
# # 4D E-I Balanced Sweep — Interactive Visualization
# Metrics: Firing rate, CV(ISI), Pairwise correlation, Fano factor, AI-regime score
#
# **No-activity regions** (fr=0, metrics are NaN) are marked in grey with "×".

# %%
import json
import os
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load data ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results")
LOG_PATH = os.path.join(DATA_DIR, "sweep_4d_ai.jsonl")

records = []
with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from {LOG_PATH}")

# ── Discover grid axes ───────────────────────────────────────────
def axis_vals(key):
    return np.array(sorted({r["params"][key] for r in records}))

N_vals   = axis_vals("N_total")
J_vals   = axis_vals("J_mV")
g_vals   = axis_vals("g")
eps_vals = axis_vals("epsilon")

n_N, n_J, n_g, n_eps = len(N_vals), len(J_vals), len(g_vals), len(eps_vals)

# Lookup: (N, J, g, eps) → metrics dict
lookup = {}
for r in records:
    p = r["params"]
    lookup[(p["N_total"], p["J_mV"], p["g"], p["epsilon"])] = r["metrics"]


def slice_2d(metric, N, J):
    """(n_eps × n_g) array for one (N, J) panel."""
    Z = np.full((n_eps, n_g), np.nan)
    for ie, e in enumerate(eps_vals):
        for ig, g in enumerate(g_vals):
            m = lookup.get((N, J, g, e))
            if m is not None:
                Z[ie, ig] = m[metric]
    return Z


def no_activity_mask(N, J):
    """Boolean mask: True where fr_E == 0 (no spikes)."""
    fr = slice_2d("fr_E", N, J)
    return (fr == 0.0) | np.isnan(fr)

# Summary stats
n_firing = sum(1 for r in records if r["metrics"]["fr_E"] > 0)
n_nan = sum(1 for r in records
            if any(isinstance(r["metrics"].get(k), float)
                   and math.isnan(r["metrics"][k])
                   for k in ["cv_E", "corr_E", "fano_E"]))
print(f"  Firing (fr_E > 0): {n_firing}/{len(records)}")
print(f"  Records with NaN stats: {n_nan}/{len(records)}  "
      f"(mostly due to zero firing rate)")


# %% [markdown]
# ## Helper: build a grid-of-heatmaps figure for one metric
# Each subplot = one (N_total, J_mV) pair. Axes are g × epsilon.
# Grey hatched cells = no spiking activity.

# %%
def make_metric_figure(metric, title, colorscale="Viridis",
                       zmid=None, zmin=None, zmax=None):
    """
    Grid of heatmaps: rows=N_total, cols=J_mV.
    No-activity cells shown as grey with '×' annotation.
    """
    fig = make_subplots(
        rows=n_N, cols=n_J,
        subplot_titles=[f"N={int(N)}  J={J}" for N in N_vals for J in J_vals],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    # Collect all values for consistent color range
    all_vals = []
    for N in N_vals:
        for J in J_vals:
            Z = slice_2d(metric, N, J)
            all_vals.append(Z[~np.isnan(Z)])
    all_vals = np.concatenate(all_vals) if any(len(a) > 0 for a in all_vals) else np.array([0])

    if zmin is None:
        zmin = float(np.nanmin(all_vals)) if len(all_vals) else 0
    if zmax is None:
        zmax = float(np.nanmax(all_vals)) if len(all_vals) else 1

    g_str = [str(g) for g in g_vals]
    eps_str = [str(e) for e in eps_vals]

    for iN, N in enumerate(N_vals):
        for iJ, J in enumerate(J_vals):
            row = iN + 1
            col = iJ + 1
            Z = slice_2d(metric, N, J)
            mask = no_activity_mask(N, J)

            # Replace NaN with None for plotly (shows as blank)
            Z_display = np.where(mask, np.nan, Z)

            # Metric heatmap
            fig.add_trace(
                go.Heatmap(
                    z=Z_display,
                    x=g_str, y=eps_str,
                    colorscale=colorscale,
                    zmid=zmid, zmin=zmin, zmax=zmax,
                    colorbar=dict(title=metric, len=0.3, y=0.5)
                        if (iN == 0 and iJ == n_J - 1) else dict(showticklabels=False, len=0),
                    showscale=(iN == 0 and iJ == n_J - 1),
                    hovertemplate=(
                        f"N={int(N)} J={J}<br>"
                        "g=%{x}<br>ε=%{y}<br>"
                        f"{metric}=%{{z:.4f}}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

            # Grey overlay for no-activity cells
            Z_grey = np.where(mask, 1.0, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=Z_grey,
                    x=g_str, y=eps_str,
                    colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                    zmin=0, zmax=1,
                    showscale=False,
                    hovertemplate=(
                        f"N={int(N)} J={J}<br>"
                        "g=%{x}<br>ε=%{y}<br>"
                        "<b>No activity (fr=0)</b><extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

            # Add "×" text annotations on no-activity cells
            for ie, e in enumerate(eps_vals):
                for ig, g in enumerate(g_vals):
                    if mask[ie, ig]:
                        fig.add_annotation(
                            x=str(g), y=str(e),
                            text="×", showarrow=False,
                            font=dict(size=10, color="dimgrey"),
                            xref=f"x{(iN * n_J + iJ) + 1 if (iN * n_J + iJ) > 0 else ''}",
                            yref=f"y{(iN * n_J + iJ) + 1 if (iN * n_J + iJ) > 0 else ''}",
                        )

            fig.update_xaxes(title_text="g" if iN == n_N - 1 else "",
                             row=row, col=col)
            fig.update_yaxes(title_text="ε" if iJ == 0 else "",
                             row=row, col=col)

    fig.update_layout(
        title=dict(text=f"{title}<br><sub>Grey cells with × = no activity (fr=0, stats are NaN)</sub>",
                   font=dict(size=16)),
        height=280 * n_N + 80,
        width=320 * n_J + 100,
        template="plotly_white",
    )
    return fig


# %% [markdown]
# ## 1. Firing Rate — E (Hz)

# %%
fig_fr = make_metric_figure("fr_E", "Firing Rate — E (Hz)",
                            colorscale="Viridis", zmin=0)
fig_fr.show()
fig_fr.write_image(os.path.join(DATA_DIR, "firing_rate_E.png"), scale=2)
print("Saved firing_rate_E.png")

# %% [markdown]
# ## 2. CV(ISI) — E

# %%
fig_cv = make_metric_figure("cv_E", "CV(ISI) — E",
                            colorscale="RdBu_r", zmid=1.0, zmin=0, zmax=2)
fig_cv.show()
fig_cv.write_image(os.path.join(DATA_DIR, "cv_isi_E.png"), scale=2)
print("Saved cv_isi_E.png")

# %% [markdown]
# ## 3. Pairwise Correlation — E

# %%
fig_corr = make_metric_figure("corr_E", "Mean Pairwise Correlation — E",
                              colorscale="RdBu_r", zmid=0.0)
fig_corr.show()
fig_corr.write_image(os.path.join(DATA_DIR, "correlation_E.png"), scale=2)
print("Saved correlation_E.png")

# %% [markdown]
# ## 4. Fano Factor — E (10 ms bins)

# %%
fig_fano = make_metric_figure("fano_E", "Fano Factor — E (10 ms bins)",
                              colorscale="RdBu_r", zmid=1.0, zmin=0, zmax=3)
fig_fano.show()
fig_fano.write_image(os.path.join(DATA_DIR, "fano_factor_E.png"), scale=2)
print("Saved fano_factor_E.png")

# %% [markdown]
# ## 5. Composite AI-Regime Score
# Score = product of:
# - CV closeness to 1 (tol=0.5)
# - |Correlation| closeness to 0 (tol=0.05)
# - Fano closeness to 1 (tol=0.5)
#
# Score=1 → perfect AI regime. Score=0 → far from AI.

# %%
CV_TOL, CORR_TOL, FANO_TOL = 0.5, 0.05, 0.5

def ai_score_2d(N, J):
    cv   = slice_2d("cv_E",   N, J)
    corr = slice_2d("corr_E", N, J)
    fano = slice_2d("fano_E", N, J)
    s_cv   = 1 - np.clip(np.abs(cv   - 1.0) / CV_TOL,   0, 1)
    s_corr = 1 - np.clip(np.abs(corr - 0.0) / CORR_TOL, 0, 1)
    s_fano = 1 - np.clip(np.abs(fano - 1.0) / FANO_TOL, 0, 1)
    return s_cv * s_corr * s_fano

# Build figure manually for AI score (uses same grid layout)
fig_ai = make_subplots(
    rows=n_N, cols=n_J,
    subplot_titles=[f"N={int(N)}  J={J}" for N in N_vals for J in J_vals],
    horizontal_spacing=0.06,
    vertical_spacing=0.08,
)

g_str = [str(g) for g in g_vals]
eps_str = [str(e) for e in eps_vals]

for iN, N in enumerate(N_vals):
    for iJ, J in enumerate(J_vals):
        row, col = iN + 1, iJ + 1
        Z = ai_score_2d(N, J)
        mask = no_activity_mask(N, J)
        Z_display = np.where(mask, np.nan, Z)

        fig_ai.add_trace(
            go.Heatmap(
                z=Z_display, x=g_str, y=eps_str,
                colorscale="Magma", zmin=0, zmax=1,
                colorbar=dict(title="AI score", len=0.3, y=0.5)
                    if (iN == 0 and iJ == n_J - 1) else dict(showticklabels=False, len=0),
                showscale=(iN == 0 and iJ == n_J - 1),
                hovertemplate=(
                    f"N={int(N)} J={J}<br>"
                    "g=%{x}<br>ε=%{y}<br>"
                    "AI score=%{z:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        # Grey no-activity overlay
        Z_grey = np.where(mask, 1.0, np.nan)
        fig_ai.add_trace(
            go.Heatmap(
                z=Z_grey, x=g_str, y=eps_str,
                colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                zmin=0, zmax=1, showscale=False,
                hovertemplate=(
                    f"N={int(N)} J={J}<br>"
                    "g=%{x}<br>ε=%{y}<br>"
                    "<b>No activity (fr=0)</b><extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        for ie, e in enumerate(eps_vals):
            for ig, g in enumerate(g_vals):
                if mask[ie, ig]:
                    fig_ai.add_annotation(
                        x=str(g), y=str(e),
                        text="×", showarrow=False,
                        font=dict(size=10, color="dimgrey"),
                        xref=f"x{(iN * n_J + iJ) + 1 if (iN * n_J + iJ) > 0 else ''}",
                        yref=f"y{(iN * n_J + iJ) + 1 if (iN * n_J + iJ) > 0 else ''}",
                    )

        fig_ai.update_xaxes(title_text="g" if iN == n_N - 1 else "",
                            row=row, col=col)
        fig_ai.update_yaxes(title_text="ε" if iJ == 0 else "",
                            row=row, col=col)

fig_ai.update_layout(
    title=dict(
        text=(f"AI-Regime Score (E)<br>"
              f"<sub>CV≈1 ± {CV_TOL}  |Corr| ≤ {CORR_TOL}  Fano≈1 ± {FANO_TOL}  |  "
              f"Grey × = no activity (fr=0)</sub>"),
        font=dict(size=16),
    ),
    height=280 * n_N + 80,
    width=320 * n_J + 100,
    template="plotly_white",
)
fig_ai.show()
fig_ai.write_image(os.path.join(DATA_DIR, "ai_score_E.png"), scale=2)
print("Saved ai_score_E.png")

# %% [markdown]
# ## Summary
# - **Grey cells with ×** = no spiking activity (firing rate = 0).
#   All derived stats (CV, correlation, Fano) are NaN for these.
# - This is expected: small J, small ε, or large g push the network
#   below threshold — the external drive is too weak relative to inhibition.
# - Only 55/240 parameter combos produce any spikes;
#   the AI regime (CV≈1, low corr, Fano≈1) is reached in very few.
