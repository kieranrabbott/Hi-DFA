import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ══════════════════════════════════════════════════════════
# Model: killing rate scales with growth rate
#   k(g) = k_max * (g / g_max)^n
# This captures the biological principle that bactericidal
# antibiotics corrupt active targets — faster growth = faster death
# ══════════════════════════════════════════════════════════

g_max  = 1.0     # max growth rate (normalised)
k_max  = 2.5     # killing rate at max growth
n_hill = 1.5     # nonlinearity of growth-kill relationship (>1 = sigmoidal)

def killing_rate(g):
    """Killing rate as a function of growth rate."""
    return k_max * (g / g_max) ** n_hill

def survival_surface(t, g):
    """Survival for cells at constant growth rate g, treated for time t."""
    k = killing_rate(g)
    return np.exp(-k * t)

# ── Grid ──
t_arr = np.linspace(0, 8, 200)
g_arr = np.linspace(0.01, 1.0, 200)
T, G = np.meshgrid(t_arr, g_arr)
S = survival_surface(T, G)

# ── TT trajectory across the surface ──
# TT cells: growth rate increases over time (tolerance lost)
g_0_tt   = 0.08    # initial TT growth rate (very slow)
kappa_tt = 0.35    # rate of growth rate increase
g_tt = g_max - (g_max - g_0_tt) * np.exp(-kappa_tt * t_arr)

# Survival of TT cells requires integrating the time-varying kill rate
dt = t_arr[1] - t_arr[0]
k_tt_t = killing_rate(g_tt)
cum_kill = np.cumsum(k_tt_t) * dt
S_tt = np.exp(-cum_kill)

# ══════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════

# Custom colormap: blue (high survival) → white → warm (low survival)
cmap_surf = LinearSegmentedColormap.from_list('surv', [
    '#B2182B',   # low survival (red)
    '#F4A582',   # 
    '#FDDBC7',   # 
    '#F7F7F7',   # mid (white)
    '#D1E5F0',   # 
    '#92C5DE',   # 
    '#2166AC',   # high survival (blue)
], N=256)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

# ── Surface ──
surf = ax.plot_surface(T, G, S, cmap=cmap_surf, alpha=0.55, 
                        rstride=4, cstride=4, 
                        linewidth=0.1, edgecolor='grey',
                        antialiased=True, zorder=1)

# ── Iso-growth-rate lines (wireframe slices at fixed growth rates) ──
highlight_g = [0.1, 0.3, 0.5, 0.7, 0.9]
for g_val in highlight_g:
    S_line = survival_surface(t_arr, g_val)
    grey = 0.4 + 0.3 * (1 - g_val)  # lighter for slower growers
    ax.plot(t_arr, np.full_like(t_arr, g_val), S_line, 
            color=str(grey), linewidth=0.7, alpha=0.5, zorder=2)

# ── Iso-time lines (slices at fixed treatment times) ──
highlight_t = [1, 2, 4, 6]
for t_val in highlight_t:
    S_line = survival_surface(t_val, g_arr)
    ax.plot(np.full_like(g_arr, t_val), g_arr, S_line,
            color='#666666', linewidth=0.6, alpha=0.3, linestyle='--', zorder=2)

# ══════════════════════════════════════════════════════════
# Trajectories ON the surface
# ══════════════════════════════════════════════════════════

# ── Susceptible: fixed at high growth rate ──
g_susc = 0.95
S_susc = survival_surface(t_arr, g_susc)
COL_SUSC = '#ff8ef4'
COL_SUSC_ALPHA = '#ff8ef475'
COL_SUSC_DARK = '#b564ab'

ax.plot(t_arr, np.full_like(t_arr, g_susc), S_susc, color=COL_SUSC_ALPHA, 
        linewidth=3, zorder=10, solid_capstyle='round')
ax.plot(t_arr, np.full_like(t_arr, g_susc), S_susc, color=COL_SUSC, 
        linewidth=5.5, alpha=0.15, zorder=9)

marker_times_sp = [0, 1, 2, 3]
for tm in marker_times_sp:
    idx = np.argmin(np.abs(t_arr - tm))
    if S_susc[idx] > 5e-3:
        ax.scatter(t_arr[idx], g_susc, S_susc[idx], color=COL_SUSC, s=35,
                  edgecolors='white', linewidths=0.5, zorder=12, depthshade=False)

# Floor projection
ax.plot(t_arr, np.full_like(t_arr, g_susc), np.zeros_like(t_arr), 
        color=COL_SUSC_DARK, linewidth=1, alpha=0.35, linestyle='-', zorder=1)

# Side wall projection (time vs survival at max growth rate wall)
ax.plot(t_arr, np.full_like(t_arr, 1.05), S_susc, 
        color=COL_SUSC_DARK, linewidth=1, alpha=0.35, linestyle='-', zorder=1)

# ── Persister: fixed at very low growth rate ──
g_pers = 0.05
S_pers = survival_surface(t_arr, g_pers)
COL_PERS = '#7cffff'
COL_PERS_ALPHA = '#7cffff75'
COL_PERS_DARK = '#4fb3b3'

ax.plot(t_arr, np.full_like(t_arr, g_pers), S_pers, color=COL_PERS_ALPHA, 
        linewidth=3, zorder=10, solid_capstyle='round')
ax.plot(t_arr, np.full_like(t_arr, g_pers), S_pers, color=COL_PERS, 
        linewidth=5.5, alpha=0.15, zorder=9)

marker_times_p = [0, 2, 4, 6, 8]
for tm in marker_times_p:
    idx = np.argmin(np.abs(t_arr - tm))
    ax.scatter(t_arr[idx], g_pers, S_pers[idx], color=COL_PERS, s=35,
              edgecolors='white', linewidths=0.5, zorder=12, depthshade=False)

# Floor projection
ax.plot(t_arr, np.full_like(t_arr, g_pers), np.zeros_like(t_arr), 
        color=COL_PERS_DARK, linewidth=1, alpha=0.35, linestyle='-', zorder=1)

# Side wall projection (time vs survival)
ax.plot(t_arr, np.full_like(t_arr, 1.05), S_pers, 
        color=COL_PERS_DARK, linewidth=1, alpha=0.35, linestyle='-', zorder=1)

# ── TT trajectory: sweeps across the surface ──
COL_TT = '#ffff7c'
COL_TT_ALPHA = '#ffff7c75'
COL_TT_DARK = '#b3b356'

ax.plot(t_arr, g_tt, S_tt, color=COL_TT_ALPHA, linewidth=3.5, 
        zorder=10, solid_capstyle='round')
ax.plot(t_arr, g_tt, S_tt, color=COL_TT, linewidth=6, 
        alpha=0.15, zorder=9)

marker_times_tt = [0, 1, 2, 3, 5, 7]
for tm in marker_times_tt:
    idx = np.argmin(np.abs(t_arr - tm))
    if S_tt[idx] > 5e-3:
        ax.scatter(t_arr[idx], g_tt[idx], S_tt[idx], 
                  color=COL_TT, s=40, edgecolors='black', linewidths=0.6,
                  zorder=12, depthshade=False)

# Floor projection
ax.plot(t_arr, g_tt, np.zeros_like(t_arr), color=COL_TT_DARK, 
        linewidth=1.2, alpha=0.4, linestyle='-', zorder=1)

# Side wall projection (time vs survival)
ax.plot(t_arr, np.full_like(t_arr, 1.05), S_tt, 
        color=COL_TT_DARK, linewidth=1.2, alpha=0.4, linestyle='-', zorder=1)

# Drop lines from TT markers to floor
for tm in marker_times_tt:
    idx = np.argmin(np.abs(t_arr - tm))
    if S_tt[idx] > 5e-3:
        ax.plot([t_arr[idx], t_arr[idx]], [g_tt[idx], g_tt[idx]], 
                [0, S_tt[idx]], color=COL_TT_DARK, linewidth=0.5, 
                linestyle=':', alpha=0.4, zorder=2)

# ── Annotations ──
# Label trajectories
ax.text(0.5, g_0_tt - 0.06, S_tt[0] + 0.06, 'Transiently\ntolerant', 
        color='#b3b356', fontsize=9.5, fontweight='bold', zorder=15)

ax.text(0.3, g_susc + 0.04, S_susc[0] - 0.12, 'Susceptible', 
        color='#b564ab', fontsize=9.5, fontweight='bold', zorder=15)

ax.text(5.5, g_pers - 0.02, S_pers[np.argmin(np.abs(t_arr - 5.5))] + 0.06, 
        'Persister', color='#4fb3b3', fontsize=9.5, fontweight='bold', zorder=15)

# Conceptual annotations
ax.text(8.3, 0.95, 0.02, 'Fast growers\n(high kill rate)', 
        fontsize=7.5, color='#999999', ha='center', fontstyle='italic', alpha=0.7)
ax.text(8.3, 0.1, 0.02, 'Slow growers\n(low kill rate)', 
        fontsize=7.5, color='#999999', ha='center', fontstyle='italic', alpha=0.7)

# TT direction annotation
ax.text(5.0, 0.72, S_tt[np.argmin(np.abs(t_arr - 5.0))] + 0.08,
        'growth rate\nincreases →', fontsize=7, color='#b3b356', 
        fontstyle='italic', ha='center', zorder=15)

# ── Axes ──
ax.set_xlabel('\nTreatment time (h)', labelpad=10)
ax.set_ylabel('\nGrowth rate (h⁻¹)', labelpad=10)
ax.set_zlabel('\nFraction surviving', labelpad=10)

ax.set_xlim(0, 8)
ax.set_ylim(0, 1.05)
ax.set_zlim(0, 1.05)

ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])

# ── View angle ──
ax.view_init(elev=25, azim=-50)

# ── Clean panes ──
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#DDDDDD')
ax.yaxis.pane.set_edgecolor('#DDDDDD')
ax.zaxis.pane.set_edgecolor('#DDDDDD')
ax.grid(True, alpha=0.12)

plt.tight_layout()

# Save multiple angles
views = {
    'main':      (25, -50),
    'high':      (35, -45),
    'low':       (15, -55),
    'front':     (20, -35),
    'side':      (20, -70),
}

for name, (elev, azim) in views.items():
    ax.view_init(elev=elev, azim=azim)
    plt.show()

# PDF of main view
ax.view_init(elev=25, azim=-50)

print("All surface views saved.")