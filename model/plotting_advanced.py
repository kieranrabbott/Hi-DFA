from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from scipy import special as sps

from .core import STDPModel


DEFAULT_COLORS = {
    "incomplete": "#ff8ef475",
    "induced": "#ffff7c75",
    "preexisting": "#7cffff75",
    "dead": "#ffa09fff",
}


def composition_vs_tau(model, C: float, a: float, tau_grid=None, figsize=(1.3, 1.5)):
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 8.0, 101)
    tau_grid = np.asarray(tau_grid, dtype=float)

    incomplete, induced, preexist, dead = [], [], [], []
    for tau in tau_grid:
        pr = model.predict_condition(C, tau, a)
        s = float(pr.incomplete)
        t = float(pr.induced)
        d = float(pr.preexisting)
        z = max(0.0, 1.0 - (s + t + d))
        incomplete.append(s)
        induced.append(t)
        preexist.append(d)
        dead.append(z)

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        tau_grid,
        np.asarray(incomplete),
        np.asarray(induced),
        np.asarray(preexist),
        np.asarray(dead),
        labels=["Incomplete", "Induced tolerant", "Pre-existing persister", "Dead"],
        colors=[
            DEFAULT_COLORS["incomplete"],
            DEFAULT_COLORS["induced"],
            DEFAULT_COLORS["preexisting"],
            DEFAULT_COLORS["dead"],
        ],
        alpha=0.85,
    )
    ax.set_xlabel("Exposure duration τ (h)")
    ax.set_ylabel("Frac of population at Ab removal")
    ax.set_xlim(tau_grid.min(), tau_grid.max())
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    return fig, ax


def composition_vs_concentration(
    model, tau: float, a: float, C_grid=None, figsize=(1.3, 1.5)
):
    if C_grid is None:
        C_grid = np.geomspace(0.05, 8.0, 181)
    C_grid = np.asarray(C_grid, dtype=float)

    incomplete, induced, preexist, dead = [], [], [], []
    for C in C_grid:
        pr = model.predict_condition(C, tau, a)
        s = float(pr.incomplete)
        t = float(pr.induced)
        d = float(pr.preexisting)
        z = max(0.0, 1.0 - (s + t + d))
        incomplete.append(s)
        induced.append(t)
        preexist.append(d)
        dead.append(z)

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        C_grid,
        np.asarray(incomplete),
        np.asarray(induced),
        np.asarray(preexist),
        np.asarray(dead),
        labels=["Incomplete", "Induced tolerant", "Pre-existing persister", "Dead"],
        colors=[
            DEFAULT_COLORS["incomplete"],
            DEFAULT_COLORS["induced"],
            DEFAULT_COLORS["preexisting"],
            DEFAULT_COLORS["dead"],
        ],
        alpha=0.85,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Antibiotic concentration C (μg/mL)")
    ax.set_ylabel("Frac of population at Ab removal")
    ax.set_xlim(C_grid.min(), C_grid.max())
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    return fig, ax


def composition_vs_age(model, C: float, tau: float, a_grid=None, figsize=(1.3, 1.5)):
    if a_grid is None:
        a_grid = np.linspace(0.0, 96.0, 193)
    a_grid = np.asarray(a_grid, dtype=float)

    incomplete, induced, preexist, dead = [], [], [], []
    for a in a_grid:
        pr = model.predict_condition(C, tau, a)
        s = float(pr.incomplete)
        t = float(pr.induced)
        d = float(pr.preexisting)
        z = max(0.0, 1.0 - (s + t + d))
        incomplete.append(s)
        induced.append(t)
        preexist.append(d)
        dead.append(z)

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        a_grid,
        np.asarray(incomplete),
        np.asarray(induced),
        np.asarray(preexist),
        np.asarray(dead),
        labels=["Incomplete", "Induced tolerant", "Pre-existing persister", "Dead"],
        colors=[
            DEFAULT_COLORS["incomplete"],
            DEFAULT_COLORS["induced"],
            DEFAULT_COLORS["preexisting"],
            DEFAULT_COLORS["dead"],
        ],
        alpha=0.85,
    )
    ax.set_xlabel("Dormancy depth a (h)")
    ax.set_ylabel("Frac of population at Ab removal")
    ax.set_xlim(a_grid.min(), a_grid.max())
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    return fig, ax


def grid_survivor_composition(model, C_grid, tau_grid, a, eps=1e-12):
    C_grid = np.asarray(C_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    n_tau, n_C = tau_grid.size, C_grid.size

    S = np.zeros((n_tau, n_C), dtype=float)
    T = np.zeros((n_tau, n_C), dtype=float)
    D = np.zeros((n_tau, n_C), dtype=float)

    for i, tau in enumerate(tau_grid):
        for j, C in enumerate(C_grid):
            pr = model.predict_condition(C, tau, a)
            S[i, j] = float(pr.incomplete)
            T[i, j] = float(pr.induced)
            D[i, j] = float(pr.preexisting)

    surv = S + T + D
    denom = np.maximum(surv, eps)
    phi_S = S / denom
    phi_T = T / denom
    phi_D = D / denom

    mask0 = surv <= eps
    phi_S[mask0] = 0.0
    phi_T[mask0] = 0.0
    phi_D[mask0] = 0.0

    return phi_S, phi_T, phi_D, surv


def make_cmyk_rgb(
    phi_S,
    phi_T,
    phi_D,
    surv=None,
    mode="survivor_weighted",
    gamma=1.0,
    use_K=True,
    k_gamma=1.0,
    sat=1.0,
):
    if mode not in {"composition", "survivor_weighted", "absolute"}:
        raise ValueError(f"Unknown mode: {mode}")

    S = np.clip(phi_S, 0.0, 1.0)
    T = np.clip(phi_T, 0.0, 1.0)
    D = np.clip(phi_D, 0.0, 1.0)

    if sat != 1.0:
        S_, T_, D_ = S**sat, T**sat, D**sat
        denom = np.maximum(S_ + T_ + D_, 1e-12)
        S, T, D = S_ / denom, T_ / denom, D_ / denom

    Cc = D
    Mm = S
    Yy = T

    if mode == "composition":
        Kk = np.zeros_like(Cc)
    else:
        if surv is None:
            raise ValueError("surv is required for survivor_weighted modes")
        Sv = np.clip(surv, 0.0, 1.0)
        Kk = (1.0 - Sv**k_gamma) if use_K else np.zeros_like(Sv)

    R = (1.0 - Cc) * (1.0 - Kk)
    G = (1.0 - Mm) * (1.0 - Kk)
    B = (1.0 - Yy) * (1.0 - Kk)
    rgb = np.stack([R, G, B], axis=-1)

    if gamma != 1.0:
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)

    return np.clip(rgb, 0.0, 1.0)


def show_rgb_heatmap(
    C_grid,
    tau_grid,
    rgb,
    *,
    ax=None,
    logC=True,
    xlabel="C (µg/mL)",
    ylabel="τ (h)",
    title=None,
):
    C_grid = np.asarray(C_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    created = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        created = True
    else:
        fig = ax.figure

    if logC:
        x = np.log10(C_grid)
        extent = [x.min(), x.max(), tau_grid.min(), tau_grid.max()]
        ax.imshow(rgb, origin="lower", aspect="auto", extent=extent)
        lo, hi = C_grid.min(), C_grid.max()
        e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
        ticks = [10**k for k in range(e0, e1 + 1)]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        extent = [C_grid.min(), C_grid.max(), tau_grid.min(), tau_grid.max()]
        ax.imshow(rgb, origin="lower", aspect="auto", extent=extent)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


def _draw_iso_fraction_grid(ax, vals=(0.2, 0.4, 0.6, 0.8), **kw):
    h = np.sqrt(3.0) / 2.0

    for s in vals:
        u = np.linspace(0, 1, 200)
        T = u * (1.0 - s)
        D = (1.0 - s) - T
        x = T + 0.5 * D
        y = D * h
        ax.plot(x, y, **kw)

    for t in vals:
        u = np.linspace(0, 1, 200)
        S = u * (1.0 - t)
        D = (1.0 - t) - S
        x = t + 0.5 * D
        y = D * h
        ax.plot(x, y, **kw)

    for d in vals:
        y = np.full(2, d * h)
        x = np.array([0.5 * d, 1.0 - 0.5 * d])
        ax.plot(x, y, **kw)


def draw_cmy_ternary_key(ax=None, res=420, sat=1.0, outline=True):
    h = np.sqrt(3.0) / 2.0
    W = int(res)
    H = int(np.round(h * res))

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
    else:
        fig = ax.figure

    xs = np.linspace(0.0, 1.0, W)
    ys = np.linspace(0.0, h, H)
    X, Y = np.meshgrid(xs, ys)

    D = Y / h
    T = X - 0.5 * D
    S = 1.0 - T - D
    mask = (S >= 0) & (T >= 0) & (D >= 0)

    if sat != 1.0:
        Ss, Ts, Ds = S**sat, T**sat, D**sat
        denom = np.maximum(Ss + Ts + Ds, 1e-12)
        S, T, D = Ss / denom, Ts / denom, Ds / denom

    R = 1.0 - D
    G = 1.0 - S
    B = 1.0 - T
    RGB = np.stack([R, G, B], axis=-1)
    RGB[~mask] = 1.0

    ax.imshow(RGB, origin="lower", extent=[0, 1, 0, h], aspect="equal")

    if outline:
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
        ax.plot(verts[:, 0], verts[:, 1], color="black", lw=0.8)
        ax.text(0.0, 0.0, "M  Susceptible (S)", ha="left", va="top", fontsize=6)
        ax.text(1.0, 0.0, "Y  Tolerant (T)", ha="right", va="top", fontsize=6)
        ax.text(0.5, h, "C  Dormant (D)", ha="center", va="bottom", fontsize=6)
        _draw_iso_fraction_grid(ax, vals=(0.2, 0.4, 0.6, 0.8), lw=0.4, alpha=0.5, color="k")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, h)
    ax.set_title("Hue legend: CMY from composition (K = 0)", fontsize=6)
    return fig, ax


def draw_survivor_brightness_bar(
    ax=None,
    k_gamma=1.0,
    cmap="Greys_r",
    label="brightness = survivors$^{k_\\gamma}$",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.3, 1.7))
    else:
        fig = ax.figure

    norm = PowerNorm(gamma=k_gamma, vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax)
    cbar.set_label(label, fontsize=6)
    return fig, ax


def J_discounted(u, tau, k, lam, g):
    alpha = lam + g
    pref = (lam / alpha) ** k * np.exp(g * tau)
    P_b = sps.gammainc(k, alpha * (tau + u))
    P_a = sps.gammainc(k, alpha * tau)
    return float(pref * (P_b - P_a))


def descendant_shares(model: STDPModel, C: float, tau: float, a: float, u: float, g: float):
    pr = model.predict_condition(C, tau, a)
    pi_S = float(pr.incomplete)
    pi_T = float(pr.induced)

    lam = model.lam_from_mean(a)
    k = model.p.k_lag
    J = J_discounted(u, tau, k, lam, g)

    denom = max(pi_S + pi_T + J, 1e-12)
    w_S = pi_S / denom
    w_T = pi_T / denom
    w_D = J / denom
    R_u = np.exp(g * u) * denom

    return w_S, w_T, w_D, R_u


def grid_descendant_composition(model: STDPModel, C_grid, tau_grid, a: float, u: float, g: float):
    C_grid = np.asarray(C_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    n_tau, n_C = tau_grid.size, C_grid.size

    phi_S = np.zeros((n_tau, n_C), float)
    phi_T = np.zeros((n_tau, n_C), float)
    phi_D = np.zeros((n_tau, n_C), float)
    R = np.zeros((n_tau, n_C), float)

    for i, tau in enumerate(tau_grid):
        for j, C in enumerate(C_grid):
            wS, wT, wD, Ru = descendant_shares(model, C, tau, a, u, g)
            phi_S[i, j] = wS
            phi_T[i, j] = wT
            phi_D[i, j] = wD
            R[i, j] = Ru

    return phi_S, phi_T, phi_D, R


def show_heatmap_with_keys_labeled(
    C_grid,
    tau_grid,
    rgb,
    *,
    k_gamma=1.0,
    title_heatmap=None,
    bar_label="brightness",
):
    fig = plt.figure(figsize=(4.0, 1.8))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[4.0, 2.3, 0.15], wspace=0.6)

    ax_hm = fig.add_subplot(gs[0, 0])
    show_rgb_heatmap(C_grid, tau_grid, rgb, ax=ax_hm, logC=True, title=title_heatmap)

    ax_tern = fig.add_subplot(gs[0, 1])
    draw_cmy_ternary_key(ax=ax_tern, sat=1.15, outline=True)

    ax_bar = fig.add_subplot(gs[0, 2])
    draw_survivor_brightness_bar(ax=ax_bar, k_gamma=k_gamma, cmap="Greys_r", label=bar_label)

    return fig, (ax_hm, ax_tern, ax_bar)


def show_contour(
    C_grid,
    tau_grid,
    survival,
    *,
    logC=True,
    xlabel="C (µg/mL)",
    ylabel="τ (h)",
    figsize=(1.5, 1.5),
    levels=None,
    colour_gamma=0.2,
    ax=None,
):
    C_grid = np.asarray(C_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if logC:
        x = np.log10(C_grid)
        extent = [x.min(), x.max(), tau_grid.min(), tau_grid.max()]
    else:
        extent = [C_grid.min(), C_grid.max(), tau_grid.min(), tau_grid.max()]

    norm = PowerNorm(gamma=colour_gamma, vmin=np.min(survival), vmax=np.max(survival))
    ax.contourf(survival, origin="lower", extent=extent, levels=levels, cmap="coolwarm", norm=norm)
    cnt = ax.contour(survival, origin="lower", colors="k", extent=extent, levels=levels, norm=norm)
    ax.clabel(cnt, cnt.levels, inline=True, fontsize=6)

    if logC:
        lo, hi = C_grid.min(), C_grid.max()
        exp_lo, exp_hi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
        ticks = [10**k for k in range(exp_lo, exp_hi + 1)]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(0, 9, 2))

    return fig, ax
