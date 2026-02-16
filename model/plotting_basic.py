from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import STDPModel
from .uncertainty import profile_likelihood


def configure_plot_style(style: str = "nature") -> None:
    """
    Apply notebook-friendly plotting defaults explicitly.
    No style changes occur at import time.
    """
    try:
        import scienceplots  # noqa: F401
    except Exception:
        if style == "nature":
            warnings.warn(
                "scienceplots is not available; falling back to current matplotlib style.",
                RuntimeWarning,
                stacklevel=2,
            )
            style = None

    if style:
        plt.style.use(style)

    matplotlib.rcParams.update(
        {
            "xtick.minor.bottom": False,
            "xtick.top": False,
            "ytick.minor.left": False,
            "ytick.right": False,
            "font.size": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "grid.alpha": 0,
            "savefig.transparent": True,
            "mathtext.default": "regular",
        }
    )


def evaluate_model(model: STDPModel, data, se_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for name, C, tau, a, fr in data:
        pred = model.predict_condition(C, tau, a)
        rows.append(
            {
                "condition": name,
                "C": C,
                "tau": tau,
                "age": a,
                "obs_incomplete": fr["incomplete"],
                "pred_incomplete": pred.incomplete,
                "obs_induced": fr["induced"],
                "pred_induced": pred.induced,
                "obs_preexisting": fr["preexisting"],
                "pred_preexisting": pred.preexisting,
                "pred_dead": pred.dead,
            }
        )

    out = pd.DataFrame(rows)
    out["obs_dead"] = (
        1.0 - out["obs_incomplete"] - out["obs_induced"] - out["obs_preexisting"]
    ).clip(lower=0.0)

    if se_df is not None:
        out = out.merge(se_df, on=["C", "tau", "age"], how="left")
        for col in ["se_incomplete", "se_induced", "se_preexisting"]:
            out[col] = out.get(col, pd.Series(0.0, index=out.index)).fillna(0.0)

    return out


def plot_fraction_comparison(
    frac: str,
    df_fit: pd.DataFrame,
    use_log: bool = False,
    show_legend: bool = False,
    title: str | None = None,
    xlim=None,
    ylim=None,
    n_sigma: float = 1.0,
    ax=None,
    save_path: str | None = None,
):
    if frac not in {"incomplete", "induced", "preexisting", "dead"}:
        raise ValueError("frac must be one of: incomplete, induced, preexisting, dead")

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        created = True
    else:
        fig = ax.figure

    plot_data = df_fit.query("age > 0").reset_index(drop=True)
    obs_col = f"obs_{frac}"
    pred_col = f"pred_{frac}"
    se_col = f"se_{frac}"
    lo_col = f"pred_ci_lo_{frac}"
    hi_col = f"pred_ci_hi_{frac}"

    has_se_x = (se_col in plot_data.columns) and (frac != "dead")
    has_ci_y = (lo_col in plot_data.columns) and (hi_col in plot_data.columns)

    c_vals = sorted(plot_data["C"].unique())
    tau_vals = sorted(plot_data["tau"].unique())
    age_vals = sorted(plot_data["age"].unique())

    cmap = plt.get_cmap("tab10", len(c_vals))
    color_map = {c: cmap(i) for i, c in enumerate(c_vals)}
    marker_pool = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    marker_map = {t: marker_pool[i % len(marker_pool)] for i, t in enumerate(tau_vals)}
    denom = max(len(age_vals) - 1, 1)
    size_map = {a: 2.5 + 3.5 * i / denom for i, a in enumerate(age_vals)}

    for _, row in plot_data.iterrows():
        obs = row[obs_col]
        pred = row[pred_col]
        xerr = n_sigma * row[se_col] if has_se_x else None

        if has_ci_y:
            yerr = np.array([[pred - row[lo_col]], [row[hi_col] - pred]])
            yerr = np.clip(yerr, 0.0, None)
        else:
            yerr = None

        ax.errorbar(
            obs,
            pred,
            xerr=xerr,
            yerr=yerr,
            fmt=marker_map[row["tau"]],
            color=color_map[row["C"]],
            markersize=size_map[row["age"]],
            alpha=0.8,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            zorder=3,
        )

    max_val = max(plot_data[obs_col].max(), plot_data[pred_col].max()) * 1.1
    if use_log:
        pos_obs = plot_data.loc[plot_data[obs_col] > 0, obs_col]
        pos_pred = plot_data.loc[plot_data[pred_col] > 0, pred_col]
        if pos_obs.empty or pos_pred.empty:
            raise ValueError(
                "Cannot use log scale when observed or predicted data are non-positive."
            )
        min_pos = min(float(pos_obs.min()), float(pos_pred.min()))
        max_pos = max(float(pos_obs.max()), float(pos_pred.max()))
        default_lim = (0.8 * min_pos, 1.1 * max_pos)
    else:
        default_lim = (0.0, max_val)
    xlim = xlim if xlim is not None else default_lim
    ylim = ylim if ylim is not None else default_lim

    if use_log and (xlim[0] <= 0 or ylim[0] <= 0):
        raise ValueError(
            "xlim/ylim lower bounds must be > 0 when use_log=True."
        )

    diag_lo = min(xlim[0], ylim[0]) if use_log else 0.0
    diag_hi = max(xlim[1], ylim[1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", lw=1.5, zorder=0)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif frac == "preexisting":
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    if show_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color=color_map[c], lw=0, markersize=5, label=f"C={c}")
            for c in c_vals
        ]
        handles.extend(
            [
                Line2D([0], [0], marker=marker_map[t], color="k", lw=0, markersize=5, label=f"τ={t}h")
                for t in tau_vals
            ]
        )
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="0.25",
                    lw=0,
                    markersize=size_map[a],
                    label=f"age={a:g}h",
                )
                for a in age_vals
            ]
        )
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title if title else f"{frac.capitalize()} Fraction", fontsize=6)

    if save_path:
        fig.savefig(save_path)

    return fig, ax


def plot_all_profiles(
    best_params,
    best_x,
    free_keys,
    data,
    ncols: int = 4,
    **obj_kwargs,
):
    nrows = int(np.ceil(len(free_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, key in zip(axes, free_keys, strict=False):
        grid, prof, _, nll0 = profile_likelihood(
            key,
            best_params,
            best_x,
            free_keys,
            data,
            **obj_kwargs,
        )
        ax.plot(np.exp(grid), prof - nll0, color="steelblue", lw=1.5)
        ax.axhline(1.92, color="firebrick", ls="--", lw=1)
        ax.axvline(np.exp(best_x[free_keys.index(key)]), color="k", ls=":", lw=1)
        ax.set_xlabel(key)
        ax.set_ylabel("ΔNLL")
        ax.set_xscale("log")

    for ax in axes[len(free_keys) :]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes
