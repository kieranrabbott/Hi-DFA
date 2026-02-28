# =============================================================================
# Imports
# =============================================================================
# =============================================================================
# Standard library
# =============================================================================
import math
import random
import warnings
import os
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import copy
import json

import dask
import dask.array as da
from concurrent.futures import ThreadPoolExecutor, as_completed
from qtpy.QtCore import QTimer

# =============================================================================
# Core scientific stack
# =============================================================================
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from tqdm.auto import tqdm

# =============================================================================
# Plotting / stats
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

# =============================================================================
# Optional imaging / IO deps (soft imports)
# =============================================================================
try:
    import tifffile  # noqa: F401
except Exception:  # pragma: no cover
    tifffile = None

try:
    import zarr  # noqa: F401
except Exception:  # pragma: no cover
    zarr = None

try:
    from PIL import Image  # noqa: F401
except Exception:  # pragma: no cover
    Image = None

try:
    from numcodecs import Blosc
    compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)
except Exception:  # pragma: no cover
    compressor = None

try:
    from joblib import Parallel, delayed  # noqa: F401
except Exception:  # pragma: no cover
    Parallel = delayed = None

try:
    from pystackreg import StackReg  # noqa: F401
except Exception:  # pragma: no cover
    StackReg = None

try:
    from skimage import transform  # noqa: F401
    from skimage.transform import resize  # noqa: F401
except Exception:  # pragma: no cover
    transform = resize = None

try:
    from scipy.ndimage import gaussian_filter  # noqa: F401
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover
    gaussian_filter = None

# =============================================================================
# Optional GUI deps (soft imports; required only for interactive functions/classes)
# =============================================================================
_HAS_GUI = True
try:
    import napari  # noqa: F401
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QSizePolicy, QDoubleSpinBox, QSpinBox
    from magicgui.widgets import Container, FloatSlider, Slider, IntSlider, PushButton, Label, CheckBox, ComboBox
except Exception as e:  # pragma: no cover
    # Useful for debugging why it failed:
    print(f"GUI Dependency Import Failed: {e}") 
    _HAS_GUI = False
    napari = None
    FigureCanvasQTAgg = Figure = None
    QSizePolicy = QDoubleSpinBox = QSpinBox = None
    Container = FloatSlider = Slider = PushButton = Label = None
    
# =============================================================================
# Warnings config
# =============================================================================
# General napari warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="napari")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="napari")

# Specific suppression for the qt_viewer deprecation warning
warnings.filterwarnings("ignore", message="Public access to Window.qt_viewer is deprecated")

# =============================================================================
# Shared constants / helpers
# =============================================================================
FLUOR_COLOR_MAP = {
    "GFP": "#2ca02c",
    "mScarlet-I3": "#d62728",
    "mCherry": "#d62728",
    "mVenus": "#FFD700",
    "mVenus-NB": "#FFD700",
}
DEFAULT_FLUOR_COLOR = "teal"


def _channel_color(channel: str) -> str:
    return FLUOR_COLOR_MAP.get(str(channel), DEFAULT_FLUOR_COLOR)


def get_trace_data(df: pd.DataFrame, mother_id: Any) -> pd.DataFrame:
    """
    Returns a single-mother trace sorted by time with duplicate times dropped.
    """
    subset = df[df["mother_id"] == mother_id].sort_values("time")
    subset = subset.drop_duplicates(subset=["time"])
    return subset.reset_index(drop=True)


def _lane_sorting(df: pd.DataFrame) -> Tuple[List[str], Mapping[str, Any], Optional[str]]:
    """
    Returns (sorted_conditions, cond_lane_map, lane_col_found).
    If no lane-ish column exists, conditions are sorted alphabetically and lane is "?".
    """
    lane_cols = ["lane_num", "lane", "lane_id", "lane_ID", "position", "pos"]
    lane_col = next((c for c in lane_cols if c in df.columns), None)

    conditions = sorted(df["condition"].astype(str).unique())
    if lane_col is None:
        return conditions, {c: "?" for c in conditions}, None

    cond_lane = df.groupby("condition")[lane_col].min()
    sorted_conds = cond_lane.sort_values().index.astype(str).tolist()
    cond_lane_map = {c: cond_lane.loc[c] for c in cond_lane.index.astype(str)}
    return sorted_conds, cond_lane_map, lane_col


def _pivot_ratio_df(
    df: pd.DataFrame,
    ratio_cfg: Mapping[str, Any],
    qc_col: str = "fluor_intensity_QC",
) -> pd.DataFrame:
    """
    Filters df to qc_col==True, pivots intensities by channel, and computes ratio with
    per-timepoint denominator thresholding.
    """
    required = ["intensity", "channel", "time", "condition", "mother_id", qc_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    num_chan = ratio_cfg["numerator"]
    den_chan = ratio_cfg["denominator"]
    thresh = ratio_cfg.get("min_denominator_threshold", 0)

    subset = df[df[qc_col] == True].copy()  # noqa: E712
    pivot_df = (
        subset.pivot_table(
            index=["mother_id", "time", "condition"],
            columns="channel",
            values="intensity",
        )
        .reset_index()
    )

    if num_chan not in pivot_df.columns or den_chan not in pivot_df.columns:
        raise ValueError(f"Channels {num_chan} or {den_chan} not found in data.")

    pivot_df["ratio"] = np.nan
    valid = pivot_df[den_chan] >= thresh
    pivot_df.loc[valid, "ratio"] = pivot_df.loc[valid, num_chan] / pivot_df.loc[valid, den_chan]
    pivot_df["ratio"] = pivot_df["ratio"].replace([np.inf, -np.inf], np.nan)

    return pivot_df


# =============================================================================
# 1) Start/End window flags
# =============================================================================
def _allowed_missing(n_required: int, max_missing_percent: float) -> int:
    return math.floor(n_required * max_missing_percent / 100)


def _required_windows(all_tps: Iterable[Any], first_n: int, last_n: int) -> Tuple[List[Any], List[Any]]:
    all_tps = list(all_tps)
    req_start = all_tps[: max(0, int(first_n))]
    req_end = all_tps[-int(last_n) :] if int(last_n) > 0 else []
    return req_start, req_end


def _passes_window(
    df: pd.DataFrame,
    mother_col: str,
    tp_col: str,
    required_tps: Sequence[Any],
    max_missing_percent: float,
) -> pd.Index:
    """
    Return mother_ids that have <= allowed missing timepoints within required_tps.
    """
    n_req = len(required_tps)
    if n_req == 0:
        return pd.Index(df[mother_col].unique())

    allow_missing = _allowed_missing(n_req, max_missing_percent)

    counts = (
        df.loc[df[tp_col].isin(required_tps), [mother_col, tp_col]]
        .groupby(mother_col)[tp_col]
        .nunique()
    )

    all_mothers = pd.Index(df[mother_col].unique())
    counts = counts.reindex(all_mothers, fill_value=0)

    missing = n_req - counts
    return all_mothers[missing <= allow_missing]


def add_start_end_flags(
    df: pd.DataFrame,
    first_x_timepoints: int = 20,
    last_x_timepoints: int = 20,
    max_missing_start_count: int = 2,  # CHANGED: Now Integer Count
    max_missing_end_count: int = 2,    # CHANGED: Now Integer Count
    mother_col: str = "mother_id",
    tp_col: str = "timepoint",
    start_col: str = "at_start",
    end_col: str = "at_end",
) -> pd.DataFrame:
    """
    Adds boolean columns for presence in start/end windows of the experiment.
    Updated to use MAX MISSING COUNT (Integer) instead of percentage.
    """
    all_tps = sorted(df[tp_col].unique())
    req_start, req_end = _required_windows(all_tps, first_x_timepoints, last_x_timepoints)

    # _passes_window helper needs to be updated or handled here. 
    # Since _passes_window is likely internal, we can inline the logic here for clarity & speed.
    
    # 1. Identify Mothers present in Start Window
    # Get all (mother, timepoint) pairs present in data
    present_start = df[df[tp_col].isin(req_start)].groupby(mother_col)[tp_col].nunique()
    # Check: (Required Count - Present Count) <= Max Missing Allowed
    # Note: len(req_start) might be less than first_x_timepoints if experiment is short
    valid_start_moms = present_start[ (len(req_start) - present_start) <= max_missing_start_count ].index

    # 2. Identify Mothers present in End Window
    present_end = df[df[tp_col].isin(req_end)].groupby(mother_col)[tp_col].nunique()
    valid_end_moms = present_end[ (len(req_end) - present_end) <= max_missing_end_count ].index

    out = df.copy()
    out[start_col] = out[mother_col].isin(valid_start_moms)
    out[end_col] = out[mother_col].isin(valid_end_moms)
    return out
    

def print_start_end_summary(
    df: pd.DataFrame,
    mother_col: str = "mother_id",
    start_col: str = "at_start",
    end_col: str = "at_end",
) -> None:
    n_total = df[mother_col].nunique()
    n_start = df.loc[df[start_col], mother_col].nunique()
    n_end = df.loc[df[end_col], mother_col].nunique()
    n_both = df.loc[df[start_col] & df[end_col], mother_col].nunique()

    print(
        "Unique mother_id counts — "
        f"mother_machine_data: {n_total}, "
        f"{start_col}: {n_start}, "
        f"{end_col}: {n_end}, "
        f"{start_col} & {end_col}: {n_both}"
    )


# =============================================================================
# 2) Growth-trace cleaning (point classification)
# =============================================================================
def classify_points(
    df: pd.DataFrame,
    hard_min: float,
    gated_min: float,
    gate_len: int,
    spike_fc_up: float,
    spike_fc_down: float,
    spike_gap: int,
    max_grad: float,
    grad_gap: int,
    use_hard: bool = True,   # NEW
    use_gated: bool = True,  # NEW
    use_spike: bool = True,  # NEW
    use_grad: bool = True,   # NEW
) -> np.ndarray:
    """
    Classifies points into:
        0: Accepted
        1: Rejected (Hard)
        2: Rejected (Gated)
        3: Rejected (Spike)
        4: Rejected (Gradient)
    """
    log_area = df["log_area"].to_numpy(dtype=float)
    log_len = df["log_length"].to_numpy(dtype=float)
    lin_len = np.exp(log_len)
    times = df["time"].to_numpy(dtype=float)

    n = len(df)
    status = np.zeros(n, dtype=int)

    # 1) Hard filter
    if use_hard:
        thresh_hard = np.log(hard_min)
        hard_mask = log_area < thresh_hard
        status[hard_mask] = 1

    # 2) Gated filter
    if use_gated:
        thresh_gated = np.log(gated_min)
        is_above_gated = log_area >= thresh_gated
        gate_start_idx = n
        gate_len = int(gate_len)

        if gate_len <= 1:
            gate_start_idx = 0
        elif n >= gate_len:
            kernel = np.ones(gate_len, dtype=int)
            counts = np.convolve(is_above_gated.astype(int), kernel, mode="valid")
            matches = np.where(counts == gate_len)[0]
            if matches.size:
                gate_start_idx = int(matches[0])

        if gate_start_idx < n:
            gated_fail_mask = ~is_above_gated
            time_mask = np.arange(n) >= gate_start_idx
            # Only mark as Gated (2) if not already Hard (1)
            mask_to_reject = gated_fail_mask & time_mask & (status != 1)
            status[mask_to_reject] = 2

    # 3) Spike filter (asymmetric up/down check)
    if use_spike:
        valid_idx = np.where(status == 0)[0]
        spike_gap = int(spike_gap)

        if valid_idx.size >= 3:
            i = 1
            while i < valid_idx.size - 1:
                curr_idx = valid_idx[i]
                prev_idx = valid_idx[i - 1]

                fold_up = lin_len[curr_idx] / lin_len[prev_idx]
                
                # Check Up Threshold
                if fold_up >= spike_fc_up:
                    search_end = min(valid_idx.size, i + 1 + spike_gap)
                    found_down = False

                    for k in range(i + 1, search_end):
                        future_idx = valid_idx[k]
                        fold_down = lin_len[future_idx] / lin_len[curr_idx]
                        
                        # Check Down Threshold (Reciprocal)
                        if fold_down <= (1.0 / spike_fc_down):
                            status[valid_idx[i:k]] = 3
                            i = k
                            found_down = True
                            break

                    if not found_down:
                        i += 1
                else:
                    i += 1

    # 4) Gradient filter (iterative)
    if use_grad:
        grad_gap = int(grad_gap)
        while True:
            valid_idx = np.where(status == 0)[0]
            if valid_idx.size < 2:
                break

            L = valid_idx[:-1]
            R = valid_idx[1:]
            gap = R - L
            ok_pair = gap <= grad_gap
            if not np.any(ok_pair):
                break

            t_L = times[L[ok_pair]]
            t_R = times[R[ok_pair]]
            y_L = log_len[L[ok_pair]]
            y_R = log_len[R[ok_pair]]

            dt = t_R - t_L
            dy = y_R - y_L

            with np.errstate(divide="ignore", invalid="ignore"):
                slopes = dy / dt

            bad = slopes > max_grad
            if not np.any(bad):
                break

            indices_to_remove = L[ok_pair][bad]
            status[indices_to_remove] = 4

    return status


# =============================================================================
# 2a) Interactive cleaning viewer (napari)
# =============================================================================
if _HAS_GUI:

    # --- Helper for consistent axis styling across all tools ---
    def _format_time_axis(ax, x_max_data):
        """
        Sets X-axis limit to the nearest multiple of 120 above x_max_data.
        Ticks every 120 min, minor ticks every 60 min.
        """
        if x_max_data > 0:
            upper_limit = math.ceil(x_max_data / 120.0) * 120.0
        else:
            upper_limit = 120.0 # Fallback
            
        ax.set_xlim(0, upper_limit)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))

    class TracePlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=8, dpi=100, x_max=1000, y_max=5):
            sns.set_theme(style="ticks", context="talk")
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.ax = self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10)
            super().__init__(self.fig)

            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

            self.x_max, self.y_max = x_max, y_max
            p = sns.color_palette("tab10")
            self.colors = {
                0: p[0],
                1: "#8B0000",
                2: "#FF6347",
                3: "#DAA520",
                4: "#9400D3",
                "line": p[7],
            }

        def update_plot(self, df, hard, gated, glen, s_fc_up, s_fc_down, s_gap, g_max, g_gap, 
                        use_hard, use_gated, use_spike, use_grad):
            self.ax.clear()
            
            # Pass new flags to classify_points
            st = classify_points(
                df, hard, gated, int(glen), s_fc_up, s_fc_down, int(s_gap), g_max, int(g_gap),
                use_hard=use_hard, use_gated=use_gated, use_spike=use_spike, use_grad=use_grad
            )

            df_acc = df.iloc[st == 0]
            if not df_acc.empty:
                self.ax.plot(
                    df_acc["time"],
                    df_acc["log_length"],
                    c=self.colors["line"],
                    lw=2,
                    alpha=0.6,
                    zorder=1,
                )
                self.ax.scatter(
                    df_acc["time"],
                    df_acc["log_length"],
                    c=[self.colors[0]],
                    s=50,
                    edgecolors="w",
                    label="Accepted",
                    zorder=3,
                )

            labels = {
                1: "Hard area min",
                2: "Gated area min",
                3: "Spike filter",
                4: "Gradient filter",
            }
            
            # Only show legends for filters that are active
            active_codes = []
            if use_hard: active_codes.append(1)
            if use_gated: active_codes.append(2)
            if use_spike: active_codes.append(3)
            if use_grad: active_codes.append(4)

            for code in active_codes:
                label = labels[code]
                sub = df.iloc[st == code]
                if not sub.empty:
                    self.ax.scatter(
                        sub["time"],
                        sub["log_length"],
                        c=[self.colors[code]],
                        marker="x",
                        s=80,
                        label=label,
                        zorder=2,
                        alpha=0.9,
                    )

            # Apply consistent axis formatting
            _format_time_axis(self.ax, self.x_max)
            
            self.ax.set_ylim(0, self.y_max)
            self.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            self.ax.set_xlabel("Time (minutes)", fontweight="bold")
            self.ax.set_ylabel("ln[ Cell length (µm) ]", fontweight="bold")
            self.ax.text(
                0.48,
                1.03,
                f"Mother ID: {df['mother_id'].iloc[0]}",
                transform=self.ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=20,
                fontweight="bold",
            )

            h, l = self.ax.get_legend_handles_labels()
            by_l = dict(zip(l, h))
            order = ["Accepted", "Hard area min", "Gated area min", "Spike filter", "Gradient filter"]
            self.ax.legend(
                [by_l[o] for o in order if o in by_l],
                [o for o in order if o in by_l],
                loc="lower left",
                bbox_to_anchor=(0.50, 1.01),
                ncol=5,
                frameon=False,
                fontsize=12,
            )
            sns.despine(ax=self.ax, trim=True)
            self.ax.grid(True, which="major", linestyle="--", alpha=0.5)
            self.draw()

    class DivisionPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=8, dpi=100, x_max=1000, y_max=5):
            sns.set_theme(style="ticks", context="talk")

            self.fig = Figure(figsize=(width, height), dpi=dpi)
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_deriv = self.fig.add_subplot(gs[1], sharex=self.ax_main)
            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)

            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

            self.x_max = x_max
            self.y_max = y_max

            self.COLOR_TRACE = sns.color_palette("tab10")[0]
            self.COLOR_DIV = "#2ca02c"
            self.COLOR_REJ = "#d62728"
            self.COLOR_DERIV = "gray"

        def update_plot(self, df, prominence, min_dist, drop_min, drop_max):
            self.ax_main.clear()
            self.ax_deriv.clear()

            valid_idxs, rej_idxs, xmid, neg_deriv = detect_divisions(df, prominence, min_dist, drop_min, drop_max)

            y_interp = df["log_length_cleaned"].interpolate(method="linear")
            self.ax_main.plot(df["time"], y_interp, c=self.COLOR_TRACE, lw=1.5, alpha=0.4, linestyle="--", label="Interpolated Gap")
            self.ax_main.plot(df["time"], df["log_length_cleaned"], c=self.COLOR_TRACE, lw=2.0, alpha=0.9, label="Cleaned Trace")

            if valid_idxs.size:
                div_times = df["time"].iloc[valid_idxs]
                div_vals = df["log_length_cleaned"].iloc[valid_idxs]
                self.ax_main.scatter(div_times, div_vals, c=self.COLOR_DIV, s=100, marker="v", zorder=5, label="Accepted Division")

            if rej_idxs.size:
                rej_times = df["time"].iloc[rej_idxs]
                rej_vals = df["log_length_cleaned"].iloc[rej_idxs]
                self.ax_main.scatter(rej_times, rej_vals, c=self.COLOR_REJ, s=80, marker="x", zorder=5, label="Rejected Candidate")

            self.ax_main.set_ylabel("ln(Length)", fontweight="bold")
            self.ax_main.set_ylim(0, self.y_max)
            self.ax_main.grid(True, linestyle="--", alpha=0.5)
            plt.setp(self.ax_main.get_xticklabels(), visible=False)
            self.ax_main.legend(loc="upper left", frameon=False, fontsize=10)

            mid = df["mother_id"].iloc[0] if not df.empty else "N/A"
            self.ax_main.text(
                0.5,
                1.02,
                f"Mother ID: {mid} | Detected: {len(valid_idxs)}",
                transform=self.ax_main.transAxes,
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

            if xmid.size:
                self.ax_deriv.plot(xmid, neg_deriv, c=self.COLOR_DERIV, lw=1.5, label="-d(ln L)/dt")
                self.ax_deriv.axhline(prominence, color="k", linestyle=":", alpha=0.7, label=f"Prominence ({prominence})")

                if valid_idxs.size:
                    t_valid = df["time"].iloc[valid_idxs].to_numpy()
                    for t_v in t_valid:
                        idx_closest = np.abs(xmid - t_v).argmin()
                        if np.abs(xmid[idx_closest] - t_v) < 10:
                            self.ax_deriv.scatter(xmid[idx_closest], neg_deriv[idx_closest], c=self.COLOR_DIV, s=30)

            # Apply consistent axis formatting
            _format_time_axis(self.ax_deriv, self.x_max)
            
            self.ax_deriv.set_ylabel("-Slope", fontweight="bold")
            self.ax_deriv.set_xlabel("Time (minutes)", fontweight="bold")
            self.ax_deriv.set_ylim(-0.05, 0.4)
            self.ax_deriv.grid(True, linestyle=":", alpha=0.5)

            sns.despine(ax=self.ax_main, trim=True)
            sns.despine(ax=self.ax_deriv, trim=True)
            self.draw()

    class GrowthPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=6, dpi=100, x_max=1000, y_max=5):
            sns.set_theme(style="ticks", context="talk")
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.ax = self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15)
            super().__init__(self.fig)

            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

            self.x_max = x_max
            self.y_max = y_max

            self.COLOR_TRACE = sns.color_palette("tab10")[0]
            self.COLOR_DIV = "#2ca02c"

        def update_plot(self, df, t1, t2, min_divs, max_nan_pct):
            self.ax.clear()

            is_growing, n_divs, nan_pct = evaluate_growth(df, t1, t2, min_divs, max_nan_pct)

            self.ax.axvspan(t1, t2, color="gold", alpha=0.15, zorder=0, label="Check Window")
            self.ax.plot(df["time"], df["log_length_cleaned"], c=self.COLOR_TRACE, lw=2, alpha=0.8)

            divs = df[df["division_event"] == True]  # noqa: E712
            if not divs.empty:
                in_window = divs[(divs["time"] >= t1) & (divs["time"] <= t2)]
                out_window = divs[(divs["time"] < t1) | (divs["time"] > t2)]

                if not in_window.empty:
                    self.ax.scatter(
                        in_window["time"],
                        in_window["log_length_cleaned"],
                        c=self.COLOR_DIV,
                        s=120,
                        marker="v",
                        zorder=5,
                        edgecolors="k",
                        label="Div (In Window)",
                    )
                if not out_window.empty:
                    self.ax.scatter(
                        out_window["time"],
                        out_window["log_length_cleaned"],
                        c="gray",
                        s=60,
                        marker="v",
                        zorder=4,
                        alpha=0.5,
                        label="Div (Outside)",
                    )

            status_text = "GROWING" if is_growing else "NOT GROWING"
            status_color = "#2ca02c" if is_growing else "#d62728"

            title_str = (
                f"Mother ID: {df['mother_id'].iloc[0]}\n"
                f"Status: {status_text}  |  Divs in Window: {n_divs} (Req: {min_divs})  |  "
                f"NaNs: {nan_pct:.1f}% (Max: {max_nan_pct}%)"
            )

            # Apply consistent axis formatting
            _format_time_axis(self.ax, self.x_max)

            self.ax.set_title(title_str, color=status_color, fontweight="bold", fontsize=14, pad=10)
            self.ax.set_ylabel("ln(Length)", fontweight="bold")
            self.ax.set_xlabel("Time (minutes)", fontweight="bold")
            self.ax.set_ylim(0, self.y_max)
            self.ax.grid(True, linestyle="--", alpha=0.5)
            self.ax.legend(loc="lower left", frameon=True, fontsize=10)

            sns.despine(ax=self.ax)
            self.draw()
            
        
def run_interactive_filtering(df: pd.DataFrame, config: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Napari-based interactive tuning for cleaning parameters.
    Returns the final dictionary of parameters selected by the user.
    """
    if not _HAS_GUI:
        raise ImportError("Interactive filtering requires napari + qtpy + magicgui installed.")

    all_moms = sorted(df["mother_id"].unique())
    n_moms = len(all_moms)

    y_vals = df["log_length"].dropna()
    global_y_max = np.ceil(np.percentile(y_vals, [99.5])[0] * 2) / 2
    global_x_max = df["time"].max()

    print(f"Loaded {n_moms} traces.")

    viewer = napari.Viewer(title="Growth trace cleaning testing")
    viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
    try:
        viewer.window.qt_viewer.dockLayerList.setVisible(False)
        viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    except Exception:
        pass

    plot_canvas = TracePlotter(x_max=global_x_max, y_max=global_y_max)
    dock_plot = viewer.window.add_dock_widget(plot_canvas, area="top", name="Trace Analysis")
    dock_plot.widget().setMinimumHeight(600)

    # Final params dict, default booleans to True
    final_params = {k: v["default"] for k, v in config.items()}
    final_params.update({"use_hard": True, "use_gated": True, "use_spike": True, "use_grad": True})

    # --- GUI FIX HELPER ---
    def fix_widget_focus(mg_widget, step=None, decimals=2):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox, QDoubleSpinBox, QSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
                if step is not None:
                    spinner.setSingleStep(step)
                if isinstance(spinner, QDoubleSpinBox):
                    spinner.setDecimals(decimals)
        except Exception as e:
            print(f"Note: GUI fix failed for {mg_widget.label}: {e}")

    def make_float(key, lbl, dec=2):
        w = FloatSlider(
            min=config[key]["min"],
            max=config[key]["max"],
            step=config[key]["step"],
            value=config[key]["default"],
            label=lbl,
        )
        fix_widget_focus(w, step=config[key]["step"], decimals=dec)
        return w

    def make_int(key, lbl):
        w = Slider(
            min=config[key]["min"],
            max=config[key]["max"],
            step=config[key]["step"],
            value=config[key]["default"],
            label=lbl,
        )
        fix_widget_focus(w, step=int(config[key]["step"]))
        return w

    # --- 1. Create Widgets ---
    
    # Checkboxes
    chk_hard = CheckBox(value=True, label="") # Row 1
    chk_gated = CheckBox(value=True, label="") # Row 2
    chk_spike = CheckBox(value=True, label="") # Row 3
    # Row 4 is skipped (spike_down)
    chk_grad = CheckBox(value=True, label="") # Row 5

    # Sliders
    w_hard = make_float("hard_min", "Hard Min Area", dec=2)
    w_gated = make_float("gated_min", "Gated Min Area", dec=2)
    w_gate_len = make_int("gate_len", "Gate Length")
    w_spike_up = make_float("spike_fc_up", "Spike FC (Up)", dec=1)
    w_spike_down = make_float("spike_fc_down", "Spike FC (Down)", dec=1)
    w_spike_gap = make_int("spike_gap", "Spike Time Gap")
    w_grad_max = make_float("grad_max", "Max Gradient", dec=3)
    w_grad_gap = make_int("grad_gap", "Grad Check Gap")

    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(100)

    # --- 2. LAYOUT ---
    
    # Column 0: Checkboxes (Far Left)
    c0 = Container(
        widgets=[
            chk_hard,       # Aligns with w_hard
            chk_gated,      # Aligns with w_gated
            chk_spike,      # Aligns with w_spike_up
            Label(value=""),# Spacer for w_spike_down row
            chk_grad        # Aligns with w_grad_max
        ],
        layout="vertical"
    )

    # Column 1: Main Parameters (Original c1)
    c1 = Container(
        widgets=[w_hard, w_gated, w_spike_up, w_spike_down, w_grad_max], 
        layout="vertical"
    )
    
    # Column 2: Secondary Parameters (Original c2)
    c2 = Container(
        widgets=[
            Label(value=""), # Spacer for w_hard
            w_gate_len,           
            w_spike_gap,          
            Label(value=""), # Spacer for w_spike_down     
            w_grad_gap            
        ], 
        layout="vertical"
    )
    
    # Column 3: Save Button
    c3 = Container(widgets=[Label(value=""), w_save], layout="vertical")
    
    # Combine Columns
    main_c = Container(widgets=[c0, c1, c2, c3], layout="horizontal")

    all_widgets = [
        chk_hard, chk_gated, chk_spike, chk_grad,
        w_hard, w_gated, w_gate_len, w_spike_up, w_spike_down, w_spike_gap, w_grad_max, w_grad_gap
    ]

    def update():
        idx = viewer.dims.current_step[0]
        if idx < n_moms:
            mid = all_moms[int(idx)]
            data = get_trace_data(df, mid)
            if not data.empty:
                # Optional: Disable sliders if checkbox is off
                w_hard.enabled = chk_hard.value
                w_gated.enabled = chk_gated.value
                w_gate_len.enabled = chk_gated.value
                w_spike_up.enabled = chk_spike.value
                w_spike_down.enabled = chk_spike.value
                w_spike_gap.enabled = chk_spike.value
                w_grad_max.enabled = chk_grad.value
                w_grad_gap.enabled = chk_grad.value

                plot_canvas.update_plot(
                    data,
                    w_hard.value,
                    w_gated.value,
                    w_gate_len.value,
                    w_spike_up.value,
                    w_spike_down.value,
                    w_spike_gap.value,
                    w_grad_max.value,
                    w_grad_gap.value,
                    # Pass booleans
                    chk_hard.value,
                    chk_gated.value,
                    chk_spike.value,
                    chk_grad.value
                )
            viewer.status = f"Mother ID: {mid} | Index: {int(idx)+1}/{n_moms}"

    for w in all_widgets:
        w.changed.connect(update)
    viewer.dims.events.current_step.connect(lambda e: update())

    @w_save.clicked.connect
    def save():
        print("\n" + "=" * 40 + "\n PARAMETERS SAVED\n" + "=" * 40)
        
        # Save Sliders
        keys = ["hard_min", "gated_min", "gate_len", "spike_fc_up", "spike_fc_down", "spike_gap", "grad_max", "grad_gap"]
        slider_widgets = [w_hard, w_gated, w_gate_len, w_spike_up, w_spike_down, w_spike_gap, w_grad_max, w_grad_gap]
        for k, w in zip(keys, slider_widgets):
            val = w.value
            print(f"{k.upper():<15} = {val:.3f}" if isinstance(val, float) else f"{k.upper():<15} = {val}")
            final_params[k] = val
        
        # Save Booleans
        bool_keys = ["use_hard", "use_gated", "use_spike", "use_grad"]
        bool_widgets = [chk_hard, chk_gated, chk_spike, chk_grad]
        print("-" * 20)
        for k, w in zip(bool_keys, bool_widgets):
            val = w.value
            print(f"{k.upper():<15} = {val}")
            final_params[k] = val

        print("=" * 40 + "\n")
        viewer.close()

    viewer.window.add_dock_widget(main_c, area="bottom", name="Filter Controls")
    viewer.dims.set_current_step(0, 0)
    update()
    viewer.window.qt_viewer.window().showMaximized()
    
    napari.run()
    
    return final_params


def run_batch_cleaning(df_main: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """
    Applies cleaning classification to all mothers; sets log_length_cleaned to NaN for rejected points.
    Updated for separate spike up/down thresholds and enable/disable flags.
    """
    print("Step 1: Preparing Data and Applying Filters...")
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)

    df_main["log_length_cleaned"] = df_main["log_length"]
    status = np.zeros(len(df_main), dtype=int)

    # Extract flags, defaulting to True if missing
    use_hard = params.get("use_hard", True)
    use_gated = params.get("use_gated", True)
    use_spike = params.get("use_spike", True)
    use_grad = params.get("use_grad", True)
    
    print(f"Active Filters: Hard={use_hard}, Gated={use_gated}, Spike={use_spike}, Grad={use_grad}")

    for _, group in tqdm(df_main.groupby("mother_id"), desc="Processing Growth Trace Cleaning"):
        st = classify_points(
            group,
            params["hard_min"],
            params["gated_min"],
            params["gate_len"],
            params["spike_fc_up"],   
            params["spike_fc_down"], 
            params["spike_gap"],
            params["grad_max"],
            params["grad_gap"],
            # Pass new flags
            use_hard=use_hard,
            use_gated=use_gated,
            use_spike=use_spike,
            use_grad=use_grad
        )
        status[group.index.to_numpy()] = st

    df_main["cleaning_status"] = status
    rejected = df_main["cleaning_status"] != 0
    df_main.loc[rejected, "log_length_cleaned"] = np.nan

    print(f"Cleaning Complete. {int(rejected.sum())} points removed.")
    return df_main

def generate_pdf_report(df_main: pd.DataFrame, filename: str) -> None:
    """
    Generates a PDF with 8x2 grid plots.
    Splits the report into sections (separate pages) for each Lane/Condition.
    Layout optimized: Increased Title-Legend gap, Reduced Legend-Plot gap.
    """
    print("Step 2: Generating PDF Report...")

    colors = {
        0: sns.color_palette("tab10")[0],
        1: "#8B0000",
        2: "#FF6347",
        3: "#DAA520",
        4: "#9400D3",
        "line": sns.color_palette("tab10")[7],
    }
    labels = {1: "Hard area min", 2: "Gated area min", 3: "Spike filter", 4: "Gradient filter"}

    # Get global limits for consistent plotting
    y_vals = df_main["log_length"].dropna()
    global_y_max = np.ceil(np.percentile(y_vals, [99.5])[0] * 2) / 2
    global_x_max = df_main["time"].max()

    # Sort conditions to create sections
    sorted_conds, cond_lane_map, _ = _lane_sorting(df_main)

    rows, cols = 8, 2
    plots_per_page = rows * cols

    with PdfPages(filename) as pdf:
        # Loop through each Condition/Lane "Section"
        for cond in sorted_conds:
            lane_num = cond_lane_map.get(str(cond), "?")
            
            # Filter data for this section
            section_df = df_main[df_main["condition"].astype(str) == str(cond)]
            unique_moms = section_df["mother_id"].unique()
            
            if len(unique_moms) == 0:
                continue

            # Create chunks (pages) for this section
            mom_chunks = [unique_moms[i : i + plots_per_page] for i in range(0, len(unique_moms), plots_per_page)]

            for page_idx, mom_batch in enumerate(mom_chunks):
                fig, axes = plt.subplots(rows, cols, figsize=(8.27 * 1.5, 11.69 * 2.0))
                axes = axes.flatten()

                # 1. Title stays at 0.99
                header_text = (
                    f"Lane {lane_num}: {cond}\n"
                    f"Page {page_idx + 1} of {len(mom_chunks)} for this lane"
                )
                
                plt.suptitle(
                    header_text,
                    fontsize=16,
                    fontweight="bold",
                    y=0.99,
                )

                # 2. Legend moved DOWN (from 0.975 to 0.960) to increase gap from Title
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0], label="Accepted"),
                    plt.Line2D([0], [0], marker="x", color=colors[1], lw=0, label=labels[1]),
                    plt.Line2D([0], [0], marker="x", color=colors[2], lw=0, label=labels[2]),
                    plt.Line2D([0], [0], marker="x", color=colors[3], lw=0, label=labels[3]),
                    plt.Line2D([0], [0], marker="x", color=colors[4], lw=0, label=labels[4]),
                ]
                fig.legend(
                    handles=handles, 
                    loc="upper center", 
                    ncol=5, 
                    fontsize=10, 
                    frameon=False, 
                    bbox_to_anchor=(0.5, 0.96) # Lowered
                )

                for i, ax in enumerate(axes):
                    if i >= len(mom_batch):
                        ax.axis("off")
                        continue

                    mid = mom_batch[i]
                    trace = section_df[section_df["mother_id"] == mid]
                    st = trace["cleaning_status"].to_numpy()

                    trace_acc = trace[st == 0]
                    if not trace_acc.empty:
                        ax.plot(trace_acc["time"], trace_acc["log_length"], c=colors["line"], lw=1.5, alpha=0.6, zorder=1)
                        ax.scatter(
                            trace_acc["time"],
                            trace_acc["log_length"],
                            c=[colors[0]],
                            s=25,
                            edgecolors="w",
                            lw=0.5,
                            zorder=3,
                        )

                    for code in [1, 2, 3, 4]:
                        trace_rej = trace[st == code]
                        if not trace_rej.empty:
                            ax.scatter(
                                trace_rej["time"],
                                trace_rej["log_length"],
                                c=[colors[code]],
                                marker="x",
                                s=40,
                                zorder=2,
                                alpha=0.9,
                            )

                    ax.set_title(f"Mother ID: {mid}", fontsize=10, fontweight="bold", pad=3)
                    ax.set_xlim(0, global_x_max)
                    ax.set_ylim(0, global_y_max)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.tick_params(labelsize=8)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    sns.despine(ax=ax)

                    if i % cols == 0:
                        ax.set_ylabel("ln[ Cell length (µm) ]", fontsize=9)
                    if i >= (len(mom_batch) - cols):
                        ax.set_xlabel("Time (min)", fontsize=9)

                # 3. Plots Top adjusted to 0.955 (was 0.96)
                # Since legend moved down by 0.015 and plots moved down by only 0.005,
                # the visual gap between them is reduced.
                plt.tight_layout(rect=[0, 0.02, 1, 0.955]) 
                pdf.savefig(fig)
                plt.close(fig)

    print(f"PDF Report saved to: {filename}")

# =============================================================================
# 3) Division detection (bridge & skip)
# =============================================================================
def forward_pairs(y: Sequence[float], t: Sequence[float], max_gap_points: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds pairs of valid indices (i, j) bridging over NaNs/gaps.
    Supports adjacent pairs and one-point bridges when max_gap_points>=1.
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    n = y.size
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    fin = np.isfinite(y)

    i1 = np.where(fin[:-1] & fin[1:])[0]
    j1 = i1 + 1

    if max_gap_points >= 1 and n >= 3:
        i2 = np.where(fin[:-2] & (~fin[1:-1]) & fin[2:])[0]
        j2 = i2 + 2
        I = np.concatenate([i1, i2])
        J = np.concatenate([j1, j2])
    else:
        I, J = i1, j1

    if I.size == 0:
        return I, J

    order = np.argsort(0.5 * (t[I] + t[J]))
    return I[order], J[order]


def forward_slopes_loglen(
    y_logL: Sequence[float],
    t: Sequence[float],
    max_gap_points: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates derivative d(logL)/dt using valid points only.
    Returns xmid, dlog, L_idx, R_idx.
    """
    L, R = forward_pairs(y_logL, t, max_gap_points=max_gap_points)
    if L.size == 0:
        return np.array([]), np.array([]), L, R

    t = np.asarray(t, float)
    y_logL = np.asarray(y_logL, float)

    dt = t[R] - t[L]
    good = (dt != 0) & np.isfinite(dt)
    L, R = L[good], R[good]
    if L.size == 0:
        return np.array([]), np.array([]), L, R

    dlog = (y_logL[R] - y_logL[L]) / (t[R] - t[L])
    xmid = 0.5 * (t[L] + t[R])
    return xmid, dlog, L, R


def detect_divisions_core(
    df: pd.DataFrame,
    peak_prominence: float,
    min_dist: float,
    drop_min: float,
    drop_max: float,
) -> np.ndarray:
    """
    Returns accepted division indices (0..N-1 relative to df) corresponding to the
    RIGHT index after the drop (post-division point).
    """
    t = df["time"].to_numpy()
    y = df["log_length_cleaned"].to_numpy()  # contains NaNs

    xmid, dlog, L_all, R_all = forward_slopes_loglen(y, t, max_gap_points=1)
    if dlog.size == 0:
        return np.array([], dtype=int)

    neg_deriv = -dlog

    if xmid.size > 1:
        dt_med = np.nanmedian(np.diff(xmid))
        if not np.isfinite(dt_med) or dt_med <= 0:
            dt_med = 3.0
    else:
        dt_med = 3.0

    dist_pts = max(1, int(round(min_dist / dt_med)))
    peaks, _ = find_peaks(neg_deriv, prominence=peak_prominence, distance=dist_pts)
    if peaks.size == 0:
        return np.array([], dtype=int)

    eL = L_all[peaks]
    eR = R_all[peaks]

    candidates: List[int] = []
    for li, ri in zip(eL, eR):
        val_pre = y[li]
        val_post = y[ri]
        if np.isfinite(val_pre) and np.isfinite(val_post):
            log_drop = val_pre - val_post
            if drop_min <= log_drop <= drop_max:
                candidates.append(int(ri))

    if not candidates:
        return np.array([], dtype=int)

    candidates = np.array(candidates, dtype=int)
    cand_t = t[candidates]
    order = np.argsort(cand_t)
    candidates = candidates[order]
    cand_t = cand_t[order]

    kept = [int(candidates[0])]
    last_t = float(cand_t[0])
    for idx, tt in zip(candidates[1:], cand_t[1:]):
        if (tt - last_t) >= min_dist:
            kept.append(int(idx))
            last_t = float(tt)

    return np.array(kept, dtype=int)


def detect_divisions(
    df: pd.DataFrame,
    peak_prominence: float,
    min_dist: float,
    drop_min: float,
    drop_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - valid_divisions (post-division indices, df-relative)
      - rejected_candidates (df-relative)
      - xmid (derivative x positions)
      - neg_deriv (-dlog/dt values)
    """
    t = df["time"].to_numpy()
    y = df["log_length_cleaned"].to_numpy()

    xmid, dlog, L_all, R_all = forward_slopes_loglen(y, t, max_gap_points=1)
    if dlog.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    neg_deriv = -dlog

    if xmid.size > 1:
        dt_med = np.nanmedian(np.diff(xmid))
        if not np.isfinite(dt_med) or dt_med <= 0:
            dt_med = 3.0
    else:
        dt_med = 3.0

    dist_pts = max(1, int(round(min_dist / dt_med)))
    peaks, _ = find_peaks(neg_deriv, prominence=peak_prominence, distance=dist_pts)

    valid: List[int] = []
    rejected: List[int] = []

    if peaks.size:
        eL = L_all[peaks]
        eR = R_all[peaks]

        candidates: List[int] = []
        for li, ri in zip(eL, eR):
            val_pre = y[li]
            val_post = y[ri]
            if np.isfinite(val_pre) and np.isfinite(val_post):
                log_drop = val_pre - val_post
                if drop_min <= log_drop <= drop_max:
                    candidates.append(int(ri))
                else:
                    rejected.append(int(ri))

        if candidates:
            candidates = np.array(candidates, dtype=int)
            cand_t = t[candidates]
            order = np.argsort(cand_t)
            candidates = candidates[order]
            cand_t = cand_t[order]

            kept = [int(candidates[0])]
            last_t = float(cand_t[0])

            for idx, tt in zip(candidates[1:], cand_t[1:]):
                if (tt - last_t) >= min_dist:
                    kept.append(int(idx))
                    last_t = float(tt)
                else:
                    rejected.append(int(idx))

            valid = kept

    return np.array(valid), np.array(rejected), xmid, neg_deriv


# =============================================================================
# 3a) Interactive division testing viewer (napari)
# =============================================================================
if _HAS_GUI:

    class DivisionPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=8, dpi=100, x_max=1000, y_max=5):
            sns.set_theme(style="ticks", context="talk")

            self.fig = Figure(figsize=(width, height), dpi=dpi)
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_deriv = self.fig.add_subplot(gs[1], sharex=self.ax_main)
            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)

            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

            self.x_max = x_max
            self.y_max = y_max

            self.COLOR_TRACE = sns.color_palette("tab10")[0]
            self.COLOR_DIV = "#2ca02c"
            self.COLOR_REJ = "#d62728"
            self.COLOR_DERIV = "gray"

        def update_plot(self, df, prominence, min_dist, drop_min, drop_max):
            self.ax_main.clear()
            self.ax_deriv.clear()

            valid_idxs, rej_idxs, xmid, neg_deriv = detect_divisions(df, prominence, min_dist, drop_min, drop_max)

            y_interp = df["log_length_cleaned"].interpolate(method="linear")
            self.ax_main.plot(df["time"], y_interp, c=self.COLOR_TRACE, lw=1.5, alpha=0.4, linestyle="--", label="Interpolated Gap")
            self.ax_main.plot(df["time"], df["log_length_cleaned"], c=self.COLOR_TRACE, lw=2.0, alpha=0.9, label="Cleaned Trace")

            if valid_idxs.size:
                div_times = df["time"].iloc[valid_idxs]
                div_vals = df["log_length_cleaned"].iloc[valid_idxs]
                self.ax_main.scatter(div_times, div_vals, c=self.COLOR_DIV, s=100, marker="v", zorder=5, label="Accepted Division")

            if rej_idxs.size:
                rej_times = df["time"].iloc[rej_idxs]
                rej_vals = df["log_length_cleaned"].iloc[rej_idxs]
                self.ax_main.scatter(rej_times, rej_vals, c=self.COLOR_REJ, s=80, marker="x", zorder=5, label="Rejected Candidate")

            self.ax_main.set_ylabel("ln(Length)", fontweight="bold")
            self.ax_main.set_xlim(0, self.x_max)
            self.ax_main.set_ylim(0, self.y_max)
            self.ax_main.grid(True, linestyle="--", alpha=0.5)
            plt.setp(self.ax_main.get_xticklabels(), visible=False)
            self.ax_main.legend(loc="upper left", frameon=False, fontsize=10)

            mid = df["mother_id"].iloc[0] if not df.empty else "N/A"
            self.ax_main.text(
                0.5,
                1.02,
                f"Mother ID: {mid} | Detected: {len(valid_idxs)}",
                transform=self.ax_main.transAxes,
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

            if xmid.size:
                self.ax_deriv.plot(xmid, neg_deriv, c=self.COLOR_DERIV, lw=1.5, label="-d(ln L)/dt")
                self.ax_deriv.axhline(prominence, color="k", linestyle=":", alpha=0.7, label=f"Prominence ({prominence})")

                if valid_idxs.size:
                    t_valid = df["time"].iloc[valid_idxs].to_numpy()
                    for t_v in t_valid:
                        idx_closest = np.abs(xmid - t_v).argmin()
                        if np.abs(xmid[idx_closest] - t_v) < 10:
                            self.ax_deriv.scatter(xmid[idx_closest], neg_deriv[idx_closest], c=self.COLOR_DIV, s=30)

            self.ax_deriv.set_ylabel("-Slope", fontweight="bold")
            self.ax_deriv.set_xlabel("Time (minutes)", fontweight="bold")
            self.ax_deriv.set_ylim(-0.05, 0.4)
            self.ax_deriv.grid(True, linestyle=":", alpha=0.5)

            sns.despine(ax=self.ax_main, trim=True)
            sns.despine(ax=self.ax_deriv, trim=True)
            self.draw()


def run_division_testing(df: pd.DataFrame, config: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Napari-based interactive tuning for division detection parameters.
    """
    if not _HAS_GUI:
        raise ImportError("Division testing requires napari + qtpy + magicgui installed.")

    all_moms = sorted(df["mother_id"].unique())
    n_moms = len(all_moms)

    y_vals = df["log_length_cleaned"].dropna()
    global_y_max = np.ceil(np.percentile(y_vals, [99.5])[0] * 2) / 2
    global_x_max = df["time"].max()

    print(f"Loaded {n_moms} traces.")

    viewer = napari.Viewer(title="Division Detection Testing")
    viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
    try:
        viewer.window.qt_viewer.dockLayerList.setVisible(False)
        viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    except Exception:
        pass

    plot_canvas = DivisionPlotter(x_max=global_x_max, y_max=global_y_max)
    dock_plot = viewer.window.add_dock_widget(plot_canvas, area="top", name="Division Analysis")
    dock_plot.widget().setMinimumHeight(600)

    final_div_params = {k: v["default"] for k, v in config.items()}

    def fix_widget_focus(mg_widget, step=None, decimals=2):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox, QDoubleSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
                if step is not None:
                    spinner.setSingleStep(step)
                if isinstance(spinner, QDoubleSpinBox):
                    spinner.setDecimals(decimals)
        except Exception as e:
            print(f"Note: GUI fix failed for {mg_widget.label}: {e}")

    def make_slider(key, label, is_float=True, decimals=2):
        params = config[key]
        if is_float:
            w = FloatSlider(min=params['min'], max=params['max'], step=params['step'], value=params['default'], label=label)
            fix_widget_focus(w, step=params['step'], decimals=decimals)
        else:
            w = Slider(min=params['min'], max=params['max'], step=params['step'], value=params['default'], label=label)
            fix_widget_focus(w, step=int(params['step']))
        return w

    w_prom = make_slider('prominence', "Peak Prominence", is_float=True, decimals=3)
    w_dist = make_slider('min_dist', "Min Div Interval", is_float=False)
    w_drop_min = make_slider('drop_min', "Min Drop (Log)", is_float=True, decimals=2)
    w_drop_max = make_slider('drop_max', "Max Drop (Log)", is_float=True, decimals=1)

    # FIXED TEXT HERE
    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(100)

    col1 = Container(widgets=[w_prom, w_drop_min], layout="vertical")
    col2 = Container(widgets=[w_dist, w_drop_max], layout="vertical")
    col3 = Container(widgets=[Label(value=""), w_save], layout="vertical")
    main_c = Container(widgets=[col1, col2, col3], layout="horizontal")

    def update():
        idx = viewer.dims.current_step[0]
        if idx < n_moms:
            mid = all_moms[int(idx)]
            data = get_trace_data(df, mid)
            if not data.empty:
                plot_canvas.update_plot(data, w_prom.value, w_dist.value, w_drop_min.value, w_drop_max.value)
            viewer.status = f"Mother ID: {mid} | Index: {int(idx)+1}/{n_moms}"

    for w in [w_prom, w_dist, w_drop_min, w_drop_max]:
        w.changed.connect(update)
    viewer.dims.events.current_step.connect(lambda e: update())

    @w_save.clicked.connect
    def save():
        print("\n" + "=" * 40 + "\n DIVISION PARAMETERS SAVED\n" + "=" * 40)
        params = {
            "peak_prominence": w_prom.value,
            "min_division_interval": w_dist.value,
            "drop_min": w_drop_min.value,
            "drop_max": w_drop_max.value,
        }
        for k, v in params.items():
            print(f"{k:<25}: {v}")
            final_div_params[k] = v
        print("=" * 40 + "\n")
        viewer.close()

    viewer.window.add_dock_widget(main_c, area="bottom", name="Detection Controls")
    viewer.dims.set_current_step(0, 0)
    update()
    viewer.window.qt_viewer.window().showMaximized()
    
    napari.run()
    return final_div_params


def run_batch_division_detection(df_main: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """
    Runs division detection per mother on a single (time-unique) trace, and flags ALL rows
    (e.g. multiple channels) that match detected division times.

    Adds boolean column:
      - division_event
    """
    print("Step 1: Detecting Division Events...")

    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    df_main["division_event"] = False

    all_indices_to_flag: List[int] = []

    for _, group in tqdm(df_main.groupby("mother_id"), desc="Processing Division Event Identification"):
        calc_trace = group.drop_duplicates(subset=["time"]).copy()

        local_idxs = detect_divisions_core(
            calc_trace,
            params["prominence"],
            params["min_division_interval"],
            params["drop_min"],
            params["drop_max"],
        )

        if local_idxs.size:
            division_times = calc_trace.iloc[local_idxs]["time"].to_numpy()
            rows_to_flag = group[group["time"].isin(division_times)].index.to_numpy()
            all_indices_to_flag.extend(rows_to_flag.tolist())

    if all_indices_to_flag:
        df_main.loc[all_indices_to_flag, "division_event"] = True

    n_unique_events = df_main[df_main["division_event"]].drop_duplicates(subset=["mother_id", "time"]).shape[0]
    print(f"Detection Complete. Flagged {n_unique_events} unique division events (across all channels).")

    return df_main


# =============================================================================
# 4) Growth classification (Growing Before)
# =============================================================================

def evaluate_growth(
    df: pd.DataFrame,
    t1: float,
    t2: float,
    min_divs: int,
    max_nan_pct: float,
    min_mean_gr_window: float,
    min_mean_gr_total: float,
    global_timepoints: np.ndarray = None, 
) -> Tuple[bool, bool, bool, int, float, float, float, float]:
    """
    Returns:
      (is_growing_overall, window_pass, total_pass, n_divs, win_nan_pct, tot_nan_pct, win_mean_gr, tot_mean_gr)
    """
    # 1. Setup Global Expectations
    if global_timepoints is None:
        global_timepoints = np.sort(df["time"].unique())
        
    # --- WINDOW CALCULATIONS ---
    # Expected in Window
    mask_win_exp = (global_timepoints >= t1) & (global_timepoints <= t2)
    exp_times_win = global_timepoints[mask_win_exp]
    n_exp_win = len(exp_times_win)
    
    # Actual in Window
    df_win = df[(df["time"] >= t1) & (df["time"] <= t2)]
    
    if n_exp_win > 0:
        # Missing
        n_pres_win = len(df_win)
        n_miss_win = n_exp_win - n_pres_win
        
        # Bad Data (NaNs present)
        c_len = 'log_length_cleaned'
        is_bad_win = df_win[c_len].isna() if c_len in df_win.columns else pd.Series(True, index=df_win.index)
        if 'growth_rate' in df_win.columns: is_bad_win = is_bad_win | df_win['growth_rate'].isna()
        
        n_bad_pres_win = is_bad_win.sum()
        win_nan_pct = ((n_miss_win + n_bad_pres_win) / n_exp_win) * 100.0
    else:
        win_nan_pct = 100.0

    # Divisions (Window)
    n_divs = int(df_win["division_event"].sum()) if 'division_event' in df_win.columns else 0

    # Mean GR (Window)
    win_mean_gr = 0.0
    if 'growth_rate_smoothed' in df_win.columns:
        w_gr = df_win['growth_rate_smoothed'].dropna()
        if not w_gr.empty:
            win_mean_gr = float(w_gr.mean())

    # --- TOTAL CALCULATIONS ---
    # Expected Total
    n_exp_tot = len(global_timepoints)
    
    if n_exp_tot > 0:
        # Missing
        n_pres_tot = len(df)
        n_miss_tot = n_exp_tot - n_pres_tot
        
        # Bad Data (NaNs present)
        is_bad_tot = df[c_len].isna() if c_len in df.columns else pd.Series(True, index=df.index)
        if 'growth_rate' in df.columns: is_bad_tot = is_bad_tot | df['growth_rate'].isna()
        
        n_bad_pres_tot = is_bad_tot.sum()
        tot_nan_pct = ((n_miss_tot + n_bad_pres_tot) / n_exp_tot) * 100.0
    else:
        tot_nan_pct = 100.0

    # Mean GR (Total)
    tot_mean_gr = 0.0
    if 'growth_rate_smoothed' in df.columns:
        t_gr = df['growth_rate_smoothed'].dropna()
        if not t_gr.empty:
            tot_mean_gr = float(t_gr.mean())
    
    # --- EVALUATE ---
    
    # Path A: Window Check
    window_pass = (
        (n_divs >= int(min_divs)) and 
        (win_nan_pct <= float(max_nan_pct)) and 
        (win_mean_gr >= float(min_mean_gr_window))
    )
    
    # Path B: Total Check
    total_pass = (
        (tot_nan_pct <= float(max_nan_pct)) and 
        (tot_mean_gr >= float(min_mean_gr_total))
    )
    
    is_growing_overall = window_pass or total_pass
    
    return is_growing_overall, window_pass, total_pass, n_divs, win_nan_pct, tot_nan_pct, win_mean_gr, tot_mean_gr


def launch_interactive_growth_testing(
    df: pd.DataFrame, 
    t1: float, 
    t2: float, 
    config: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """
    Napari-based interactive tuning for 'growing_before' classification.
    """
    try:
        req_min_divs = config["min_divs"]
        req_max_nan = config["max_nan_pct"]
        req_min_gr_win = config["min_mean_growth_rate_window"]
        req_min_gr_tot = config["min_mean_growth_rate_total"]
    except KeyError as e:
        raise KeyError(f"Missing required key in GROWTH_BEFORE_CONFIG: {e}")

    if not _HAS_GUI: 
        print("Napari not installed/available. Returning defaults.")
        return {
            "t1": t1, "t2": t2,
            "min_division_events": req_min_divs["default"],
            "max_missing_timepoints_fraction": req_max_nan["default"],
            "min_mean_growth_rate_window": req_min_gr_win["default"] / 100.0,
            "min_mean_growth_rate_total": req_min_gr_tot["default"] / 100.0,
        }

    if 'division_event' not in df.columns: raise ValueError("Error: 'division_event' column missing.")
    
    if "at_start" in df.columns:
        candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
    else:
        candidates = sorted(df["mother_id"].unique())
        
    n_moms = len(candidates)
    if n_moms == 0: return {}

    y_vals = df["log_length_cleaned"].dropna()
    global_y_max = np.ceil(np.percentile(y_vals, [99.5])[0] * 2) / 2 if not y_vals.empty else 5
    global_tps = np.sort(df["time"].unique())
    global_x_max = global_tps.max() if len(global_tps) > 0 else 600

    print(f"Checking Pre-Treatment Window: {t1} min -> {t2} min")

    viewer = napari.Viewer(title="Growth Classification Testing")
    viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
    try:
        viewer.window.qt_viewer.dockLayerList.setVisible(False)
        viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    except Exception: pass

    # --- Plotter ---
    class GrowthPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=6, dpi=100, x_max=1000, y_max=5):
            sns.set_theme(style="ticks", context="talk")
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.ax = self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.15)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.x_max, self.y_max = x_max, y_max
            self.COLOR_TRACE = sns.color_palette("tab10")[0]

        def update_plot(self, df, t1, t2, min_divs, max_nan_pct, min_gr_win, min_gr_tot, x_max_limit, global_tps):
            self.ax.clear()

            is_growing, win_pass, tot_pass, n_divs, win_nan_pct, tot_nan_pct, win_mean_gr, tot_mean_gr = evaluate_growth(
                df, t1, t2, min_divs, max_nan_pct, min_gr_win, min_gr_tot,
                global_timepoints=global_tps
            )

            # Draw Trace & Window
            self.ax.axvspan(t1, t2, color="gold", alpha=0.15, zorder=0)
            self.ax.plot(df["time"], df["log_length_cleaned"], c=self.COLOR_TRACE, lw=2, alpha=0.8)

            # Draw Divisions
            divs = df[df["division_event"] == True]
            if not divs.empty:
                in_win = divs[(divs["time"] >= t1) & (divs["time"] <= t2)]
                if not in_win.empty:
                    self.ax.scatter(in_win["time"], in_win["log_length_cleaned"], c="#2ca02c", s=120, marker="v", zorder=5, edgecolors="k")

            # --- 4-ROW TITLE ---
            C_PASS, C_FAIL, C_GREY, C_TEXT = "#2ca02c", "#d62728", "gray", "black"
            c_main = C_PASS if is_growing else C_FAIL
            c_win  = C_PASS if win_pass else C_GREY
            c_tot  = C_PASS if tot_pass else C_GREY

            # Values (x100)
            win_disp = win_mean_gr * 100
            tot_disp = tot_mean_gr * 100
            
            # Text Lines
            txt_r1 = f"Mother ID: {df['mother_id'].iloc[0]}  |  Status: {'GROWING' if is_growing else 'NOT GROWING'}"
            txt_r2 = f"Divisions: {n_divs}  |  Window NaN: {win_nan_pct:.1f}%  |  Total Check NaN: {tot_nan_pct:.1f}%  (Max {max_nan_pct}%)"
            
            w_status = "PASS" if win_pass else "FAIL"
            # Format: GR: val (req >= req, x10^-2)
            txt_r3 = f"Window Check: {w_status}  |  Window Mean GR: {win_disp:.2f} (req >= {min_gr_win*100:.2f}, x10^-2)"
            
            t_status = "PASS" if tot_pass else "FAIL"
            txt_r4 = f"Total Check: {t_status}     |  Total Mean GR: {tot_disp:.2f} (req >= {min_gr_tot*100:.2f}, x10^-2)"

            # Y-Coords (approx 0.06 spacing)
            self.ax.text(0.5, 1.22, txt_r1, transform=self.ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold', color=c_main)
            self.ax.text(0.5, 1.16, txt_r2, transform=self.ax.transAxes, ha='center', va='bottom', fontsize=9, color=C_TEXT)
            self.ax.text(0.5, 1.10, txt_r3, transform=self.ax.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_win)
            self.ax.text(0.5, 1.04, txt_r4, transform=self.ax.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_tot)

            self.ax.set_xlim(0, x_max_limit)
            self.ax.set_ylim(0, self.y_max)
            self.ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
            self.ax.grid(True, linestyle="--", alpha=0.5)
            self.draw()

    plotter = GrowthPlotter(x_max=global_x_max, y_max=global_y_max)
    viewer.window.add_dock_widget(plotter, area="top", name="Growth Analysis")

    # Controls
    w_min_divs = Slider(min=req_min_divs["min"], max=req_min_divs["max"], step=req_min_divs["step"], value=req_min_divs["default"], label="Min Divs (Window Only)")
    w_max_nan = FloatSlider(min=req_max_nan["min"], max=req_max_nan["max"], step=req_max_nan["step"], value=req_max_nan["default"], label="Max Bad %")
    w_min_gr_win = FloatSlider(min=req_min_gr_win["min"], max=req_min_gr_win["max"], step=req_min_gr_win["step"], value=req_min_gr_win["default"], label="Window Min GR (x10^-2)")
    w_min_gr_tot = FloatSlider(min=req_min_gr_tot["min"], max=req_min_gr_tot["max"], step=req_min_gr_tot["step"], value=req_min_gr_tot["default"], label="Total Min GR (x10^-2)")

    def fix_focus(w):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QDoubleSpinBox
            for s in w.native.findChildren(QAbstractSlider): s.setFocusPolicy(Qt.StrongFocus)
            for s in w.native.findChildren(QDoubleSpinBox): s.setDecimals(2)
        except: pass
    
    for w in [w_min_divs, w_max_nan, w_min_gr_win, w_min_gr_tot]: fix_focus(w)

    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(60)

    c1 = Container(widgets=[w_min_divs, w_max_nan], layout="vertical")
    c2 = Container(widgets=[w_min_gr_win, w_min_gr_tot], layout="vertical")
    c3 = Container(widgets=[Label(value=""), w_save], layout="vertical")
    main_c = Container(widgets=[c1, c2, c3], layout="horizontal")
    
    final_params = {
        "t1": t1, "t2": t2,
        "min_division_events": req_min_divs["default"],
        "max_missing_timepoints_fraction": req_max_nan["default"],
        "min_mean_growth_rate_window": req_min_gr_win["default"] / 100.0,
        "min_mean_growth_rate_total": req_min_gr_tot["default"] / 100.0
    }

    x_max_limit = np.ceil(global_x_max / 120) * 120

    def update():
        idx = viewer.dims.current_step[0]
        if idx < n_moms:
            mid = candidates[int(idx)]
            trace = df[df["mother_id"] == mid].sort_values("time")
            plotter.update_plot(
                trace, t1, t2, 
                w_min_divs.value, w_max_nan.value, 
                w_min_gr_win.value / 100.0, w_min_gr_tot.value / 100.0,
                x_max_limit, global_tps
            )
            viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

    for w in [w_min_divs, w_max_nan, w_min_gr_win, w_min_gr_tot]: w.changed.connect(update)
    viewer.dims.events.current_step.connect(lambda e: update())

    @w_save.clicked.connect
    def save():
        final_params.update({
            "min_division_events": w_min_divs.value,
            "max_missing_timepoints_fraction": w_max_nan.value,
            "min_mean_growth_rate_window": w_min_gr_win.value / 100.0,
            "min_mean_growth_rate_total": w_min_gr_tot.value / 100.0
        })
        print(f"Growing Before Params Saved: {final_params}")
        viewer.close()

    viewer.window.add_dock_widget(main_c, area="bottom", name="Criteria Controls")
    viewer.dims.set_current_step(0, 0)
    update()
    viewer.window.qt_viewer.window().showMaximized()
    napari.run()
    return final_params

def _process_growth_batch(mid, group, t1, t2, min_divs, max_nan_pct, min_gr_win, min_gr_tot, global_timepoints):
    try:
        calc_trace = group.drop_duplicates(subset=["time"])
        # Only unpack the first return value (is_growing)
        is_growing = evaluate_growth(
            calc_trace, t1, t2, min_divs, max_nan_pct, min_gr_win, min_gr_tot,
            global_timepoints=global_timepoints
        )[0]
        return mid, is_growing
    except Exception:
        return mid, False

def execute_batch_growth_classification(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df_main is None: return None
    if not params: raise ValueError("Error: growth_params missing.")

    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n STARTING GROWTH CLASSIFICATION (Parallel) \n{'='*40}")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    df_main["growing_before"] = False

    t1, t2 = params["t1"], params["t2"]
    min_divs, max_nan = params["min_division_events"], params["max_missing_timepoints_fraction"]
    min_gr_win = params.get("min_mean_growth_rate_window", -100.0)
    min_gr_tot = params.get("min_mean_growth_rate_total", -100.0)
    
    print(f"Criteria Window: Divs>={min_divs}, NaN<{max_nan}%, GR>={min_gr_win}")
    print(f"Criteria Total:  NaN<{max_nan}%, GR>={min_gr_tot} (Divs ignored)")
    
    global_tps = np.sort(df_main["time"].unique())
    candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique() if "at_start" in df_main.columns else df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids in window [{t1}, {t2}]...")
    
    grouped = df_main.loc[df_main["mother_id"].isin(candidates)].groupby("mother_id")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_growth_batch)(mid, group, t1, t2, min_divs, max_nan, min_gr_win, min_gr_tot, global_tps)
        for mid, group in tqdm(grouped, desc="Classifying", total=len(candidates))
    )
    
    growing_moms = [mid for mid, is_growing in results if is_growing]
    df_main.loc[df_main["mother_id"].isin(growing_moms), "growing_before"] = True

    n_growing = len(growing_moms)
    print(f"Classification Complete. {n_growing}/{len(candidates)} classified as Growing.")
    
    return df_main

# =============================================================================
# 5) Background intensity histograms + filtering impact
# =============================================================================
def plot_fluorescence_bg_histograms_custom(df: pd.DataFrame, line_percentiles: Sequence[float]) -> None:
    """
    Plots wide histograms of background intensity for growing cells (non-PC).
    Adds vertical dashed lines at specified percentiles (computed per channel).
    X-axis clipped to 0.1% - 99%.
    """
    required_cols = ["growing_before", "channel", "intensity_bg"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    subset = df[(df["growing_before"] == True) & (df["channel"] != "PC")].copy()  # noqa: E712
    if subset.empty:
        print("No data found for growing cells in fluorescence channels (non-PC).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    unique_channels = sorted(subset["channel"].unique())
    print(f"Plotting histograms for channels: {unique_channels}")

    for channel in unique_channels:
        chan_data = subset.loc[subset["channel"] == channel, "intensity_bg"].dropna()
        if chan_data.empty:
            continue

        xlims = np.percentile(chan_data, [0.1, 99])
        lines_vals = np.percentile(chan_data, line_percentiles)

        plt.figure(figsize=(15, 4))
        sns.histplot(chan_data, kde=True, color=_channel_color(channel), edgecolor="white", line_kws={"linewidth": 2})

        for p, val in zip(line_percentiles, lines_vals):
            plt.axvline(val, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label=f"{p}%: {val:.1f}")

        plt.xlim(xlims[0], xlims[1])
        plt.title(f"Channel: {channel}", fontweight="bold", loc="left")
        plt.xlabel("Background Intensity")
        plt.ylabel("Count")
        plt.legend(title="Percentiles", loc="upper left", bbox_to_anchor=(1, 1), frameon=True, fontsize=10)

        sns.despine()
        plt.tight_layout()
        plt.show()


def plot_filtered_bg_histograms(df: pd.DataFrame, percentile_limits: Mapping[str, Tuple[float, float]]) -> None:
    """
    Filters data based on percentile limits for background intensity (per channel) and plots
    histograms of the remaining data.
    If 'growing_before' exists, percentile cutoffs are computed using growing cells only.
    """
    if "intensity_bg" not in df.columns or "channel" not in df.columns:
        print("Error: Missing 'intensity_bg' or 'channel' columns.")
        return

    filtered_dfs = []
    print("Filtering Criteria and Cutoffs (Growing Cells Only if available):")

    for channel, (low_p, high_p) in percentile_limits.items():
        if "growing_before" in df.columns:
            subset = df[(df["growing_before"] == True) & (df["channel"] == channel)]  # noqa: E712
        else:
            subset = df[df["channel"] == channel]

        if subset.empty:
            print(f"  {channel}: No data found.")
            continue

        vals = subset["intensity_bg"].dropna()
        if vals.empty:
            continue

        low_val, high_val = np.percentile(vals, [low_p, high_p])
        print(f"  {channel}: Keeping range {low_p}% ({low_val:.2f}) - {high_p}% ({high_val:.2f})")

        valid_rows = subset[(subset["intensity_bg"] >= low_val) & (subset["intensity_bg"] <= high_val)].copy()
        filtered_dfs.append(valid_rows)

    if not filtered_dfs:
        print("No data remaining after filtering.")
        return

    filtered_data = pd.concat(filtered_dfs, ignore_index=True)

    sns.set_theme(style="whitegrid", context="talk")
    for channel in sorted(filtered_data["channel"].unique()):
        chan_data = filtered_data.loc[filtered_data["channel"] == channel, "intensity_bg"].dropna()
        if chan_data.empty:
            continue

        plt.figure(figsize=(15, 4))
        sns.histplot(chan_data, kde=True, color=_channel_color(channel), edgecolor="white", line_kws={"linewidth": 2})

        dmin, dmax = float(chan_data.min()), float(chan_data.max())
        span = dmax - dmin
        if span > 0:
            plt.xlim(dmin - 0.02 * span, dmax + 0.02 * span)

        plt.title(f"Filtered Channel: {channel}", fontweight="bold", loc="left")
        plt.xlabel("Background Intensity")
        plt.ylabel("Count")

        sns.despine()
        plt.tight_layout()
        plt.show()


def visualize_filtering_impact(df: pd.DataFrame, percentile_limits: Mapping[str, Tuple[float, float]]) -> None:
    """
    Calculates and visualizes the % of timepoints filtered out per mother_id
    based on percentile limits for background intensity. Requires growing_before.
    """
    required = ["intensity_bg", "channel", "growing_before", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return

    subset = df[df["growing_before"] == True].copy()  # noqa: E712
    if subset.empty:
        print("No growing mothers found.")
        return

    print("Calculating Cutoffs and Impact per Mother...")
    results = []

    for channel, (low_p, high_p) in percentile_limits.items():
        chan_data = subset[subset["channel"] == channel]
        if chan_data.empty:
            print(f"  {channel}: No data found.")
            continue

        vals = chan_data["intensity_bg"].dropna()
        if vals.empty:
            continue

        low_val, high_val = np.percentile(vals, [low_p, high_p])
        print(f"  {channel} Limits: {low_p}% ({low_val:.2f}) - {high_p}% ({high_val:.2f})")

        chan_df = chan_data.copy()
        chan_df["is_filtered"] = (chan_df["intensity_bg"] < low_val) | (chan_df["intensity_bg"] > high_val)

        grouped = chan_df.groupby("mother_id")["is_filtered"].agg(["mean"]).reset_index()
        grouped["pct_filtered"] = grouped["mean"] * 100.0
        grouped["channel"] = channel
        results.append(grouped[["mother_id", "channel", "pct_filtered"]])

    if not results:
        print("No results calculated.")
        return

    impact_df = pd.concat(results, ignore_index=True)
    sns.set_theme(style="whitegrid", context="talk")

    for channel in sorted(impact_df["channel"].unique()):
        chan_impact = impact_df[impact_df["channel"] == channel]
        plt.figure(figsize=(12, 5))

        sns.histplot(data=chan_impact, x="pct_filtered", bins=20, color=_channel_color(channel), edgecolor="white", kde=False)
        plt.title(f"Data Loss Distribution: {channel}", fontweight="bold", loc="left")
        plt.xlabel("% of Timepoints Filtered per Mother")
        plt.ylabel("Number of Mothers")

        avg_loss = chan_impact["pct_filtered"].mean()
        med_loss = chan_impact["pct_filtered"].median()
        plt.text(
            0.95,
            0.9,
            f"Mean Loss: {avg_loss:.1f}%\nMedian Loss: {med_loss:.1f}%",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            fontsize=10,
        )

        sns.despine()
        plt.tight_layout()
        plt.show()


def troubleshoot_high_loss_dual_plot(
    df: pd.DataFrame,
    percentile_limits: Mapping[str, Tuple[float, float]],
    loss_threshold: float = 10.0,
    sample_n: int = 5,
    seed: int = 0,
) -> None:
    """
    Finds mother/channel traces with >loss_threshold% filtered timepoints and plots:
      1) intensity_bg (left axis) with kept/filtered points
      2) intensity (right axis)
    """
    required = ["intensity_bg", "intensity", "channel", "growing_before", "mother_id", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return

    subset = df[df["growing_before"] == True].copy()  # noqa: E712
    if subset.empty:
        print("No growing mothers found.")
        return

    print(f"Identifying mothers with >{loss_threshold}% data loss...")
    candidates = []

    for channel, (low_p, high_p) in percentile_limits.items():
        chan_data = subset[subset["channel"] == channel]
        if chan_data.empty:
            continue

        vals = chan_data["intensity_bg"].dropna()
        if vals.empty:
            continue

        low_val, high_val = np.percentile(vals, [low_p, high_p])

        chan_df = chan_data.copy()
        chan_df["is_filtered"] = (chan_df["intensity_bg"] < low_val) | (chan_df["intensity_bg"] > high_val)

        stats = chan_df.groupby("mother_id")["is_filtered"].agg(["count", "sum"])
        stats["pct"] = (stats["sum"] / stats["count"]) * 100.0

        bad_moms = stats[stats["pct"] > loss_threshold].index.tolist()
        for mid in bad_moms:
            candidates.append(
                {
                    "mother_id": mid,
                    "channel": channel,
                    "cutoffs": (low_val, high_val),
                    "loss_pct": float(stats.loc[mid, "pct"]),
                }
            )

    if not candidates:
        print("No mothers found above loss threshold.")
        return

    print(f"Found {len(candidates)} channel-traces with high loss.")
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(sample_n, len(candidates)))

    sns.set_theme(style="ticks", context="talk")

    for item in selected:
        mid = item["mother_id"]
        chan = item["channel"]
        low_cut, high_cut = item["cutoffs"]
        loss = item["loss_pct"]

        trace = subset[(subset["mother_id"] == mid) & (subset["channel"] == chan)].sort_values("time")
        if trace.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Background Intensity", fontweight="bold")

        ax1.plot(trace["time"], trace["intensity_bg"], alpha=0.4, lw=1.5, zorder=1)

        valid = trace[(trace["intensity_bg"] >= low_cut) & (trace["intensity_bg"] <= high_cut)]
        ax1.scatter(valid["time"], valid["intensity_bg"], color="teal", s=30, label="Background (Kept)", zorder=2)

        invalid = trace[(trace["intensity_bg"] < low_cut) | (trace["intensity_bg"] > high_cut)]
        ax1.scatter(invalid["time"], invalid["intensity_bg"], color="red", marker="x", s=50, label="Background (Filtered)", zorder=3)

        ax1.axhline(low_cut, color="red", linestyle="--", alpha=0.3, label="BG Cutoff")
        ax1.axhline(high_cut, color="red", linestyle="--", alpha=0.3)
        ax1.grid(True, linestyle=":", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Cell Intensity", fontweight="bold")
        ax2.plot(trace["time"], trace["intensity"], lw=2.5, alpha=0.9, label="Cell Intensity")

        plt.title(f"Mother {mid} | Channel: {chan} | Data Loss: {loss:.1f}%", fontweight="bold", pad=15)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True, fontsize=10)

        plt.tight_layout()
        plt.show()


# =============================================================================
# 6) Fluorescence QC (background stability + intensity thresholds)
# =============================================================================
def apply_fluorescence_qc(
    df: pd.DataFrame,
    percentile_limits: Mapping[str, Tuple[float, float]],
    loss_threshold: float = 80.0,
) -> pd.DataFrame:
    """
    Background-stability QC based on intensity_bg percentile cutoffs computed on growing population.
    Mothers with >= loss_threshold% timepoints outside cutoffs (in any channel) fail.

    Creates/updates:
      - fluor_bg_QC (bool)
    """
    required = ["intensity_bg", "channel", "growing_before", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["fluor_bg_QC"] = False

    subset = df[df["growing_before"] == True].copy()  # noqa: E712
    if subset.empty:
        print("No growing cells to QC.")
        return df

    print(f"Applying Background Intensity QC (Threshold: >= {loss_threshold}% filtered)...")
    discarded_mids: set = set()

    for channel, (low_p, high_p) in percentile_limits.items():
        chan_data = subset[subset["channel"] == channel]
        if chan_data.empty:
            continue

        vals = chan_data["intensity_bg"].dropna()
        if vals.empty:
            continue

        low_val, high_val = np.percentile(vals, [low_p, high_p])
        print(f"  Channel {channel}: Cutoffs {low_p}% ({low_val:.2f}) - {high_p}% ({high_val:.2f})")

        chan_df = chan_data.copy()
        chan_df["is_filtered"] = (chan_df["intensity_bg"] < low_val) | (chan_df["intensity_bg"] > high_val)

        stats = chan_df.groupby("mother_id")["is_filtered"].agg(["count", "sum"])
        stats["pct_loss"] = (stats["sum"] / stats["count"]) * 100.0

        bad = stats[stats["pct_loss"] >= loss_threshold].index.tolist()
        discarded_mids.update(bad)
        print(f"    -> {len(bad)} mothers failed in {channel}")

    all_growing_ids = subset["mother_id"].unique()
    good_mids = [mid for mid in all_growing_ids if mid not in discarded_mids]
    df.loc[df["mother_id"].isin(good_mids), "fluor_bg_QC"] = True

    print("\nQC Complete.")
    print(f"Total Growing Mothers: {len(all_growing_ids)}")
    print(f"Passed QC: {len(good_mids)}")
    print(f"Discarded (High BG Noise): {len(discarded_mids)}")

    if discarded_mids:
        print("\nDiscarded Mothers by Condition:")
        if "condition" in df.columns:
            discarded_rows = df[df["mother_id"].isin(discarded_mids)].drop_duplicates("mother_id")
            print(discarded_rows["condition"].value_counts())
        else:
            print("(Column 'condition' not found in dataframe)")

    return df


def plot_mean_intensity_histograms(df: pd.DataFrame, t1: float, t2: float) -> None:
    """
    Mean intensity per mother per channel within [t1, t2] for fluor_bg_QC==True (non-PC),
    then histograms per channel.
    """
    required = ["fluor_bg_QC", "intensity", "channel", "time", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    subset = df[
        (df["fluor_bg_QC"] == True)  # noqa: E712
        & (df["time"] >= t1)
        & (df["time"] <= t2)
        & (df["channel"] != "PC")
    ].copy()

    if subset.empty:
        print("No data found matching criteria (fluor_bg_QC=True in time window).")
        return

    mean_intensities = subset.groupby(["mother_id", "channel"])["intensity"].mean().reset_index()
    unique_channels = sorted(mean_intensities["channel"].unique())
    print(f"Calculating mean intensity between {t1}-{t2} min.")
    print(f"Plotting distributions for: {unique_channels}")

    sns.set_theme(style="whitegrid", context="talk")
    for channel in unique_channels:
        chan_data = mean_intensities.loc[mean_intensities["channel"] == channel, "intensity"].dropna()
        if chan_data.empty:
            continue

        plt.figure(figsize=(15, 4))
        sns.histplot(chan_data, kde=True, color=_channel_color(channel), edgecolor="white", line_kws={"linewidth": 2})

        plt.title(f"Mean Intensity Distribution: {channel} (t={t1}-{t2} min)", fontweight="bold", loc="left")
        plt.xlabel("Mean Cell Intensity (a.u.)")
        plt.ylabel("Count of Mothers")

        mean_val = chan_data.mean()
        median_val = chan_data.median()
        plt.text(
            0.98,
            0.9,
            f"Mean: {mean_val:.1f}\nMedian: {median_val:.1f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            fontsize=10,
        )

        sns.despine()
        plt.tight_layout()
        plt.show()


def plot_mean_intensity_histograms_zoomed_bins(
    df: pd.DataFrame,
    t1: float,
    t2: float,
    x_limits: Mapping[str, Tuple[float, float]],
    n_bins: int = 50,
) -> None:
    """
    Like plot_mean_intensity_histograms, but with explicit bin count and optional xlim per channel.
    """
    required = ["fluor_bg_QC", "intensity", "channel", "time", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    subset = df[
        (df["fluor_bg_QC"] == True)  # noqa: E712
        & (df["time"] >= t1)
        & (df["time"] <= t2)
        & (df["channel"] != "PC")
    ].copy()

    if subset.empty:
        print("No data found matching criteria.")
        return

    mean_intensities = subset.groupby(["mother_id", "channel"])["intensity"].mean().reset_index()
    unique_channels = sorted(mean_intensities["channel"].unique())
    print(f"Plotting mean intensity ({t1}-{t2} min) with {n_bins} bins.")

    sns.set_theme(style="whitegrid", context="talk")
    for channel in unique_channels:
        chan_data = mean_intensities.loc[mean_intensities["channel"] == channel, "intensity"].dropna()
        if chan_data.empty:
            continue

        plt.figure(figsize=(15, 4))
        sns.histplot(
            chan_data,
            bins=int(n_bins),
            kde=True,
            color=_channel_color(channel),
            edgecolor="white",
            line_kws={"linewidth": 2},
        )

        title_suffix = ""
        if channel in x_limits:
            plt.xlim(x_limits[channel])
            title_suffix = f" (Zoom: {x_limits[channel]})"

        plt.title(f"Mean Intensity: {channel}{title_suffix}", fontweight="bold", loc="left")
        plt.xlabel("Mean Cell Intensity (a.u.)")
        plt.ylabel("Count")

        mean_val = chan_data.mean()
        median_val = chan_data.median()
        plt.text(
            0.98,
            0.9,
            f"Total Mean: {mean_val:.1f}\nTotal Median: {median_val:.1f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            fontsize=10,
        )

        sns.despine()
        plt.tight_layout()
        plt.show()


def apply_intensity_qc(
    df: pd.DataFrame,
    t1: float,
    t2: float,
    min_thresholds: Mapping[str, Optional[float]],
) -> pd.DataFrame:
    """
    Mean-intensity QC within [t1, t2] for mothers that passed fluor_bg_QC.
    Mothers failing any channel threshold are discarded.

    Creates/updates:
      - fluor_intensity_QC (bool)
    """
    required = ["fluor_bg_QC", "intensity", "channel", "time", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    print(f"Applying Intensity QC (Time: {t1}-{t2} min)...")

    df["fluor_intensity_QC"] = False

    subset = df[(df["fluor_bg_QC"] == True) & (df["time"] >= t1) & (df["time"] <= t2)].copy()  # noqa: E712
    if subset.empty:
        print("No mothers passed background QC to check.")
        return df

    means = subset.groupby(["mother_id", "channel"])["intensity"].mean().reset_index()

    discarded_mids: set = set()
    for channel, threshold in min_thresholds.items():
        if threshold is None:
            continue

        print(f"  Checking {channel} >= {threshold}...")
        chan_means = means[means["channel"] == channel]
        failures = chan_means[chan_means["intensity"] < threshold]["mother_id"].unique()
        discarded_mids.update(failures.tolist())
        print(f"    -> {len(failures)} mothers failed (mean < {threshold})")

    all_bg_passed = df[df["fluor_bg_QC"] == True]["mother_id"].unique()  # noqa: E712
    good_mids = [mid for mid in all_bg_passed if mid not in discarded_mids]
    df.loc[df["mother_id"].isin(good_mids), "fluor_intensity_QC"] = True

    print("\nQC Complete.")
    print(f"Candidates (Passed BG QC): {len(all_bg_passed)}")
    print(f"Passed Intensity QC: {len(good_mids)}")
    print(f"Discarded (Low Intensity): {len(discarded_mids)}")

    if discarded_mids:
        print("\nDiscarded Mothers by Condition:")
        if "condition" in df.columns:
            discarded_rows = df[df["mother_id"].isin(discarded_mids)].drop_duplicates("mother_id")
            print(discarded_rows["condition"].value_counts())
        else:
            print("(Column 'condition' not found)")

    return df


# =============================================================================
# 7) Population mean plots
# =============================================================================
def plot_population_mean_fluorescence(df: pd.DataFrame) -> None:
    """
    Plots population mean intensity (+/- SD) vs time for all mothers passing fluor_intensity_QC.
    Dynamically creates stacked subplots for every fluorescence channel found in the data.
    """
    required = ["fluor_intensity_QC", "intensity", "channel", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    # Filter for QC-passed and non-PhaseContrast data
    subset = df[(df["fluor_intensity_QC"] == True) & (df["channel"] != "PC")].copy()  # noqa: E712
    if subset.empty:
        print("No mothers passed Intensity QC to plot.")
        return

    # Calculate stats per channel/time
    stats = subset.groupby(["channel", "time"])["intensity"].agg(["mean", "std", "count"]).reset_index()
    unique_channels = sorted(stats["channel"].unique())
    n_channels = len(unique_channels)
    
    if n_channels == 0:
        print("No fluorescence channels found to plot.")
        return

    print(f"Plotting population mean for {n_channels} channels: {unique_channels}")

    sns.set_theme(style="ticks", context="talk")
    
    # Create a vertical stack of subplots, sharing the X-axis (Time)
    fig, axes = plt.subplots(
        nrows=n_channels, 
        ncols=1, 
        figsize=(12, 4 * n_channels), 
        sharex=True, 
        squeeze=False # Ensures axes is always a 2D array even if n=1
    )
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i, channel in enumerate(unique_channels):
        ax = axes[i]
        data = stats[stats["channel"] == channel].sort_values("time")
        color = _channel_color(channel)
        
        # Plot Mean Line
        ax.plot(
            data["time"], 
            data["mean"], 
            color=color, 
            lw=2.5, 
            label=f"{channel} Mean"
        )
        
        # Plot Standard Deviation Shading
        ax.fill_between(
            data["time"], 
            data["mean"] - data["std"], 
            data["mean"] + data["std"], 
            color=color, 
            alpha=0.2, 
            label=f"{channel} SD"
        )

        # Styling
        ax.set_ylabel(f"{channel}\nIntensity (a.u.)", fontweight="bold", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        
        # Only set X label on the bottom-most plot
        if i == n_channels - 1:
            ax.set_xlabel("Time (min)", fontweight="bold")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
        
        sns.despine(ax=ax)

    plt.suptitle("Population Mean Intensity (QC Passed)", fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()


def plot_population_mean_fluorescence_ordered_by_lane(
    df: pd.DataFrame,
    shading_config: Sequence[Mapping[str, Any]],
) -> None:
    """
    Plots mean intensity (+/- SD) by condition, ordered by lane number.
    
    Supports Multi-Axis scaling:
      - Left Axis 1 (Main): First non-red channel (e.g. GFP)
      - Left Axis 2 (Offset): Second non-red channel (e.g. mVenus)
      - Right Axis: Red channel (e.g. mScarlet)
    """
    # 1. Validation
    required = ['fluor_intensity_QC', 'intensity', 'channel', 'time', 'condition']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    # 2. Filter Data (Global QC check)
    subset_all = df[
        (df['fluor_intensity_QC'] == True) & 
        (df['channel'] != 'PC')
    ].copy()
    
    if subset_all.empty:
        print("No mothers passed Intensity QC to plot.")
        return

    # 3. Lane Sorting
    sorted_conditions, cond_lane_map, _ = _lane_sorting(subset_all)

    # 4. Plotting Loop
    sns.set_theme(style="ticks", context="talk")

    # Keywords to identify "Red" channels for the Right Axis
    RIGHT_AXIS_KEYWORDS = ["scarlet", "cherry", "red", "rfp"]

    for cond in sorted_conditions:
        cond_data = subset_all[subset_all['condition'].astype(str) == str(cond)]
        if cond_data.empty: continue
            
        stats = cond_data.groupby(['channel', 'time'])['intensity'].agg(['mean', 'std', 'count']).reset_index()
        unique_channels = sorted(stats['channel'].unique())
        max_time_in_data = cond_data['time'].max()
        
        # Sort channels into groups
        left_channels = []
        right_channels = []

        for ch in unique_channels:
            if any(k in ch.lower() for k in RIGHT_AXIS_KEYWORDS):
                right_channels.append(ch)
            else:
                left_channels.append(ch)

        # Handle edge case: if only Reds exist, put one on left
        if not left_channels and right_channels:
            left_channels = [right_channels.pop(0)]

        # --- Create Figure ---
        fig, ax_main = plt.subplots(figsize=(12, 6))
        
        # --- Add Shading (on main axis) ---
        for region in shading_config:
            r_start = region.get('start', 0)
            r_end = region.get('end')
            if r_end is None: r_end = max_time_in_data
            ax_main.axvspan(r_start, r_end, color=region['color'], alpha=region['alpha'], zorder=0, label=region.get('label', '_nolegend_'))

        ax_main.xaxis.set_major_locator(ticker.MultipleLocator(120))
        
        # --- Helper to plot single channel ---
        def plot_single_channel(ax, channel_name):
            data = stats[stats['channel'] == channel_name].sort_values('time')
            color = _channel_color(channel_name)
            
            ax.plot(data['time'], data['mean'], color=color, lw=2.5, label=f"{channel_name} Mean")
            ax.fill_between(data['time'], data['mean'] - data['std'], data['mean'] + data['std'], color=color, alpha=0.2)
            
            ax.set_ylabel(f"{channel_name} Intensity", color=color, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=color)
            return color

        # ---------------------------------------------------------
        # 1. Main Left Axis (e.g. GFP)
        # ---------------------------------------------------------
        if len(left_channels) > 0:
            ch1 = left_channels[0]
            plot_single_channel(ax_main, ch1)
            ax_main.set_xlabel('Time (min)', fontweight='bold')
            ax_main.grid(True, linestyle=':', alpha=0.3)

        # ---------------------------------------------------------
        # 2. Offset Left Axis (e.g. mVenus) - THE SPLIT
        # ---------------------------------------------------------
        ax_left_2 = None
        if len(left_channels) > 1:
            ch2 = left_channels[1]
            ax_left_2 = ax_main.twinx()
            
            # Move this axis to the left
            ax_left_2.yaxis.set_label_position('left')
            ax_left_2.yaxis.set_ticks_position('left')
            
            # Offset the spine outward so it doesn't overlap with the first axis
            ax_left_2.spines["left"].set_position(("outward", 70))
            
            plot_single_channel(ax_left_2, ch2)
            
            # Ensure the new spine is visible
            ax_left_2.set_frame_on(True)
            ax_left_2.patch.set_visible(False)
            
            # Only show the relevant spine
            for sp in ax_left_2.spines.values():
                sp.set_visible(False)
            ax_left_2.spines["left"].set_visible(True)

        # ---------------------------------------------------------
        # 3. Right Axis (e.g. mScarlet)
        # ---------------------------------------------------------
        ax_right = None
        if len(right_channels) > 0:
            ch_right = right_channels[0]
            ax_right = ax_main.twinx()
            plot_single_channel(ax_right, ch_right)

        # --- Title & Legend ---
        n_moms = cond_data['mother_id'].nunique()
        lane_num = cond_lane_map.get(str(cond), "?")
        plt.title(f"Lane {lane_num}: {cond} (n={n_moms})", fontweight='bold', pad=15)
        
        # Gather legends from all active axes
        axes_list = [ax_main, ax_left_2, ax_right]
        lines, labels = [], []
        for ax in axes_list:
            if ax is not None:
                l, lb = ax.get_legend_handles_labels()
                lines.extend(l)
                labels.extend(lb)
        
        # Remove duplicates (mainly for shading if plotted multiple times) and sort
        by_label = dict(zip(labels, lines))
        # Filter out _nolegend_
        final_dict = {k: v for k, v in by_label.items() if not k.startswith("_")}
        
        ax_main.legend(final_dict.values(), final_dict.keys(), loc='upper left', frameon=True, fontsize=10)

        # Clean up spines
        sns.despine(ax=ax_main, right=(ax_right is not None), left=False)
        if ax_right:
            sns.despine(ax=ax_right, left=False, right=False)

        # Tight layout is crucial here to make space for the offset axis
        plt.tight_layout()
        plt.show()

# =============================================================================
# 8) Ratio mean plot
# =============================================================================
def plot_population_mean_ratio_ordered_by_lane(
    df: pd.DataFrame,
    shading_config: Sequence[Mapping[str, Any]],
    ratio_cfg: Mapping[str, Any],
    y_percentiles: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plots population MEAN ratio (+/- SD) vs time per condition, ordered by lane.
    Excludes ratio timepoints where denominator intensity < threshold.

    This is the ratio version (kept under your original function name).
    """
    required = ["fluor_intensity_QC", "intensity", "channel", "time", "condition", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    pivot_df = _pivot_ratio_df(df, ratio_cfg, qc_col="fluor_intensity_QC")

    sorted_conditions, cond_lane_map, _ = _lane_sorting(df[df["fluor_intensity_QC"] == True].copy())  # noqa: E712
    sns.set_theme(style="ticks", context="talk")

    for cond in sorted_conditions:
        cond_data = pivot_df[pivot_df["condition"].astype(str) == str(cond)]
        if cond_data.empty:
            continue

        stats = cond_data.groupby("time")["ratio"].agg(["mean", "std"]).reset_index()
        max_time = float(cond_data["time"].max())

        fig, ax = plt.subplots(figsize=(12, 6))

        for region in shading_config:
            r_start = region.get("start", 0)
            r_end = region.get("end") or max_time
            ax.axvspan(r_start, r_end, color=region["color"], alpha=region["alpha"], zorder=0, label=region.get("label", "_nolegend_"))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
        ax.plot(stats["time"], stats["mean"], color="purple", lw=2.5, label="Mean Ratio")
        ax.fill_between(stats["time"], stats["mean"] - stats["std"], stats["mean"] + stats["std"], color="purple", alpha=0.2, label="SD")

        if y_percentiles:
            raw = cond_data["ratio"].dropna()
            if not raw.empty:
                ymin, ymax = np.percentile(raw, y_percentiles)
                buf = (ymax - ymin) * 0.05
                ax.set_ylim(ymin - buf, ymax + buf)

        ax.set_xlabel("Time (min)", fontweight="bold")
        ax.set_ylabel(ratio_cfg.get("label", "Ratio"), fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.3)

        n_moms = cond_data["mother_id"].nunique()
        lane_num = cond_lane_map.get(str(cond), "?")
        plt.title(f"Lane {lane_num}: {cond} (n={n_moms})", fontweight="bold", pad=15)

        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, fontsize=10)

        sns.despine()
        plt.tight_layout()
        plt.show()


# =============================================================================
# 9) Region summaries + copy/paste helpers
# =============================================================================
def summarize_regions_with_singlecell_sd_and_gfp(
    df: pd.DataFrame,
    ratio_cfg: Mapping[str, Any],
    region1: Tuple[float, float] = (1170, 1200),
    region2: Tuple[float, float] = (1410, 1440),
    min_points_per_region: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["fluor_intensity_QC", "intensity", "channel", "time", "condition", "mother_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    num_chan = ratio_cfg["numerator"]
    den_chan = ratio_cfg["denominator"]
    thresh = ratio_cfg.get("min_denominator_threshold", 0)

    subset = df[df["fluor_intensity_QC"] == True].copy()  # noqa: E712
    pivot_df = (
        subset.pivot_table(index=["mother_id", "time", "condition"], columns="channel", values="intensity")
        .reset_index()
    )

    if num_chan not in pivot_df.columns or den_chan not in pivot_df.columns:
        raise ValueError(f"Channels {num_chan} or {den_chan} not found in data.")

    pivot_df["ratio"] = np.nan
    valid_mask = pivot_df[den_chan] >= thresh
    pivot_df.loc[valid_mask, "ratio"] = pivot_df.loc[valid_mask, num_chan] / pivot_df.loc[valid_mask, den_chan]
    pivot_df["ratio"] = pivot_df["ratio"].replace([np.inf, -np.inf], np.nan)

    s1, e1 = region1
    s2, e2 = region2
    pivot_df["in_r1"] = (pivot_df["time"] >= s1) & (pivot_df["time"] <= e1)
    pivot_df["in_r2"] = (pivot_df["time"] >= s2) & (pivot_df["time"] <= e2)

    def per_mother_region_stats(g: pd.DataFrame) -> pd.Series:
        r1_rat = g.loc[g["in_r1"], "ratio"].dropna()
        r2_rat = g.loc[g["in_r2"], "ratio"].dropna()
        r1_gfp = g.loc[g["in_r1"], num_chan].dropna()

        return pd.Series(
            {
                "r1_npts_ratio": int(r1_rat.size),
                "r2_npts_ratio": int(r2_rat.size),
                "r1_mean_ratio": r1_rat.mean() if r1_rat.size else np.nan,
                "r2_mean_ratio": r2_rat.mean() if r2_rat.size else np.nan,
                "r1_npts_gfp": int(r1_gfp.size),
                "r1_mean_gfp": r1_gfp.mean() if r1_gfp.size else np.nan,
            }
        )

    per_mother = (
        pivot_df.groupby(["condition", "mother_id"], sort=False)
        .apply(per_mother_region_stats)
        .reset_index()
    )

    ok = (per_mother["r1_npts_ratio"] >= min_points_per_region) & (per_mother["r2_npts_ratio"] >= min_points_per_region)
    per_mother_ok = per_mother.loc[ok].copy()

    per_mother_ok["fold_decrease_%"] = np.where(
        (per_mother_ok["r1_mean_ratio"].notna())
        & (per_mother_ok["r1_mean_ratio"] != 0)
        & (per_mother_ok["r2_mean_ratio"].notna()),
        (1.0 - (per_mother_ok["r2_mean_ratio"] / per_mother_ok["r1_mean_ratio"])) * 100.0,
        np.nan,
    )

    sorted_conditions, cond_lane_map, lane_col = _lane_sorting(df)

    rows = []
    for cond, g in per_mother_ok.groupby("condition", sort=False):
        lane_val = cond_lane_map.get(str(cond), None)
        rows.append(
            {
                "condition": cond,
                "lane": lane_val,
                "n_mothers_used": int(g["mother_id"].nunique()),
                f"r1_mean_ratio_{s1}-{e1}": g["r1_mean_ratio"].mean(),
                f"r1_sd_ratio_{s1}-{e1}": g["r1_mean_ratio"].std(ddof=1),
                f"r2_mean_ratio_{s2}-{e2}": g["r2_mean_ratio"].mean(),
                f"r2_sd_ratio_{s2}-{e2}": g["r2_mean_ratio"].std(ddof=1),
                "fold_decrease_mean_%": g["fold_decrease_%"].mean(),
                "fold_decrease_sd_%": g["fold_decrease_%"].std(ddof=1),
                "exp_phase_mean_GFP": g["r1_mean_gfp"].mean(),
                f"median_r1_npts_ratio_{s1}-{e1}": float(g["r1_npts_ratio"].median()),
                f"median_r2_npts_ratio_{s2}-{e2}": float(g["r2_npts_ratio"].median()),
                f"median_r1_npts_gfp_{s1}-{e1}": float(g["r1_npts_gfp"].median()),
            }
        )

    summary = pd.DataFrame(rows)
    if "lane" in summary.columns and lane_col is not None:
        summary = summary.sort_values(["lane", "condition"], na_position="last").reset_index(drop=True)
    else:
        summary = summary.sort_values(["condition"]).reset_index(drop=True)

    return summary, per_mother_ok


def print_copy_pastable_summary_piped_aligned(summary_df: pd.DataFrame) -> None:
    def fmt_1dp(x):
        return "NA" if pd.isna(x) else f"{x:.1f}"

    chunks = []
    for _, row in summary_df.iterrows():
        lane = row.get("lane", np.nan)
        lane_str = "?" if pd.isna(lane) else str(int(lane) if float(lane).is_integer() else lane)
        c1 = f"Lane {lane_str}"
        c2 = str(row["condition"])
        dr_mean = fmt_1dp(row.get("fold_decrease_mean_%", np.nan))
        dr_sd = fmt_1dp(row.get("fold_decrease_sd_%", np.nan))
        c3 = f"Vm-sensitive dynamic range: {dr_mean} ± {dr_sd}%"
        gfp = fmt_1dp(row.get("exp_phase_mean_GFP", np.nan))
        c4 = f"Exponential phase mean GFP: {gfp}"
        chunks.append((c1, c2, c3, c4))

    if not chunks:
        print("(no rows)")
        return

    w1 = max(len(c[0]) for c in chunks)
    w2 = max(len(c[1]) for c in chunks)
    w3 = max(len(c[2]) for c in chunks)
    w4 = max(len(c[3]) for c in chunks)

    for c1, c2, c3, c4 in chunks:
        print(f"{c1:<{w1}} | {c2:<{w2}} | {c3:<{w3}} | {c4:<{w4}}")


def make_copy_pastable_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    def fmt_1dp(x):
        return "NA" if pd.isna(x) else f"{x:.1f}"

    out = summary_df.copy()

    if "lane" in out.columns:
        out["Lane"] = out["lane"].apply(lambda x: "?" if pd.isna(x) else str(int(x) if float(x).is_integer() else x))
    else:
        out["Lane"] = "?"

    out["Condition"] = out["condition"].astype(str)
    out["Vm-sensitive dynamic range (%)"] = out["fold_decrease_mean_%"].apply(fmt_1dp) + " ± " + out["fold_decrease_sd_%"].apply(fmt_1dp)
    out["Exp phase mean GFP"] = out["exp_phase_mean_GFP"].apply(fmt_1dp)

    if "n_mothers_used" in out.columns:
        out["n"] = out["n_mothers_used"].astype(int)

    cols = ["Lane", "Condition", "Vm-sensitive dynamic range (%)", "Exp phase mean GFP"]
    if "n" in out.columns:
        cols.append("n")

    return out[cols].sort_values(["Lane", "Condition"]).reset_index(drop=True)


# =============================================================================
# 10) Individual trace plots + PDF report
# =============================================================================
def plot_individual_traces_by_lane(df, shading_config, ratio_cfg, sample_size=10):
    """
    Creates individual trace plots for GFP, mScarlet, and Ratio for each condition.
    Fixes the 'mean' KeyError and the palette UserWarning.
    """
    # 1. Prepare Data
    subset = df[df['fluor_intensity_QC'] == True].copy()
    num_chan = ratio_cfg['numerator']
    den_chan = ratio_cfg['denominator']
    thresh = ratio_cfg['min_denominator_threshold']

    pivot_df = subset.pivot_table(
        index=['mother_id', 'time', 'condition'], 
        columns='channel', 
        values='intensity'
    ).reset_index()

    # Calculate Filtered Ratio
    pivot_df['ratio'] = np.nan
    valid_mask = pivot_df[den_chan] >= thresh
    pivot_df.loc[valid_mask, 'ratio'] = pivot_df.loc[valid_mask, num_chan] / pivot_df.loc[valid_mask, den_chan]

    # 2. Sorting Logic
    lane_cols = ['lane_num', 'lane', 'lane_id', 'position']
    lane_col_found = next((col for col in lane_cols if col in df.columns), None)
    
    if lane_col_found:
        cond_lane_map = df.groupby('condition')[lane_col_found].min()
        sorted_conditions = cond_lane_map.sort_values().index.tolist()
    else:
        sorted_conditions = sorted(pivot_df['condition'].unique())
        cond_lane_map = {c: "?" for c in sorted_conditions}

    # 3. Plotting Loop
    sns.set_theme(style="ticks", context="notebook")
    
    plot_types = [
        {'col': num_chan, 'label': f'{num_chan} Intensity', 'color': 'tab:green'},
        {'col': den_chan, 'label': f'{den_chan} Intensity', 'color': 'tab:red'},
        {'col': 'ratio', 'label': 'Fluorescence Ratio', 'color': 'purple'}
    ]

    for cond in sorted_conditions:
        cond_data = pivot_df[pivot_df['condition'] == cond]
        all_moms = cond_data['mother_id'].unique()
        
        # Randomly sample mothers
        selected_moms = random.sample(list(all_moms), min(sample_size, len(all_moms)))
        plot_data = cond_data[cond_data['mother_id'].isin(selected_moms)].sort_values('time')
        
        lane_num = cond_lane_map[cond]

        for p_type in plot_types:
            col_name = p_type['col']
            if col_name not in cond_data.columns: continue

            fig, ax = plt.subplots(figsize=(10, 5))
            
            # --- Add Shading ---
            max_t = cond_data['time'].max()
            for region in shading_config:
                r_end = region.get('end') or max_t
                ax.axvspan(region['start'], r_end, color=region['color'], alpha=region['alpha'], zorder=0)

            # --- Plot Individual Traces ---
            # Using units=mother_id and estimator=None plots individual lines 
            # without the palette warning.
            sns.lineplot(
                data=plot_data, 
                x='time', 
                y=col_name, 
                units='mother_id', 
                estimator=None,
                color=p_type['color'], 
                alpha=0.3, 
                lw=1,
                ax=ax,
                zorder=1
            )
            
            # --- Plot Population Mean (Calculated correctly) ---
            pop_stats = cond_data.groupby('time')[col_name].mean().reset_index()
            ax.plot(
                pop_stats['time'], 
                pop_stats[col_name], # Use the column name directly, not 'mean'
                color='black', 
                lw=3, 
                label='Population Mean',
                zorder=2
            )

            # Formatting
            ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
            ax.set_title(f"Lane {lane_num}: {cond} | {p_type['label']} (Sample n={len(selected_moms)})", fontweight='bold')
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(p_type['label'])
            
            # Zoom logic for Ratio
            if col_name == 'ratio':
                vals = plot_data['ratio'].dropna()
                if not vals.empty:
                    ax.set_ylim(0, np.percentile(vals, 99) * 1.5)

            ax.legend(loc='upper left', frameon=True, fontsize=10)
            sns.despine()
            plt.tight_layout()
            plt.show()

def plot_individual_traces_by_lane_smoothed(df, shading_config, ratio_cfg, sample_size=10, lowess_frac=0.1):
    """
    Creates individual trace plots with LOWESS smoothing for GFP, mScarlet, and Ratio.
    """
    # 1. Prepare Data
    subset = df[df['fluor_intensity_QC'] == True].copy()
    num_chan = ratio_cfg['numerator']
    den_chan = ratio_cfg['denominator']
    thresh = ratio_cfg['min_denominator_threshold']

    pivot_df = subset.pivot_table(
        index=['mother_id', 'time', 'condition'], 
        columns='channel', 
        values='intensity'
    ).reset_index()

    # Calculate Filtered Ratio
    pivot_df['ratio'] = np.nan
    valid_mask = pivot_df[den_chan] >= thresh
    pivot_df.loc[valid_mask, 'ratio'] = pivot_df.loc[valid_mask, num_chan] / pivot_df.loc[valid_mask, den_chan]

    # 2. Sorting Logic
    lane_cols = ['lane_num', 'lane', 'lane_id', 'position']
    lane_col_found = next((col for col in lane_cols if col in df.columns), None)
    
    if lane_col_found:
        cond_lane_map = df.groupby('condition')[lane_col_found].min()
        sorted_conditions = cond_lane_map.sort_values().index.tolist()
    else:
        sorted_conditions = sorted(pivot_df['condition'].unique())
        cond_lane_map = {c: "?" for c in sorted_conditions}

    # 3. Plotting Loop
    sns.set_theme(style="ticks", context="notebook")
    
    plot_types = [
        {'col': num_chan, 'label': f'{num_chan} Intensity', 'color': 'tab:green'},
        {'col': den_chan, 'label': f'{den_chan} Intensity', 'color': 'tab:red'},
        {'col': 'ratio', 'label': 'Fluorescence Ratio', 'color': 'purple'}
    ]

    for cond in sorted_conditions:
        cond_data = pivot_df[pivot_df['condition'] == cond]
        all_moms = cond_data['mother_id'].unique()
        
        selected_moms = random.sample(list(all_moms), min(sample_size, len(all_moms)))
        plot_data = cond_data[cond_data['mother_id'].isin(selected_moms)].sort_values('time')
        
        lane_num = cond_lane_map[cond]

        for p_type in plot_types:
            col_name = p_type['col']
            if col_name not in cond_data.columns: continue

            fig, ax = plt.subplots(figsize=(10, 5))
            
            # --- Add Shading ---
            max_t = cond_data['time'].max()
            for region in shading_config:
                r_end = region.get('end') or max_t
                ax.axvspan(region['start'], r_end, color=region['color'], alpha=region['alpha'], zorder=0)

            # --- Plot Individual Traces (Raw + Smoothed) ---
            for mid in selected_moms:
                mom_trace = plot_data[plot_data['mother_id'] == mid].dropna(subset=[col_name])
                if len(mom_trace) < 5: continue # Need enough points to smooth
                
                # Raw (faint)
                ax.plot(mom_trace['time'], mom_trace[col_name], color=p_type['color'], alpha=0.1, lw=0.5, zorder=1)
                
                # LOWESS Smoothing
                smoothed = sm.nonparametric.lowess(mom_trace[col_name], mom_trace['time'], frac=lowess_frac)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color=p_type['color'], alpha=0.4, lw=1.5, zorder=2)

            # --- Plot Smoothed Population Mean ---
            pop_stats = cond_data.groupby('time')[col_name].mean().reset_index().dropna()
            if len(pop_stats) > 10:
                pop_smoothed = sm.nonparametric.lowess(pop_stats[col_name], pop_stats['time'], frac=0.05) # Light smoothing on mean
                ax.plot(pop_smoothed[:, 0], pop_smoothed[:, 1], color='black', lw=3, label='Population Mean (Smoothed)', zorder=3)
            else:
                ax.plot(pop_stats['time'], pop_stats[col_name], color='black', lw=3, label='Population Mean', zorder=3)

            # Formatting
            ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
            ax.set_title(f"Lane {lane_num}: {cond} | {p_type['label']} (Smoothed, n={len(selected_moms)})", fontweight='bold')
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(p_type['label'])
            
            if col_name == 'ratio':
                vals = plot_data['ratio'].dropna()
                if not vals.empty:
                    ax.set_ylim(0, np.percentile(vals, 99) * 1.5)

            ax.legend(loc='upper left', frameon=True, fontsize=10)
            sns.despine()
            plt.tight_layout()
            plt.show()

def generate_single_cell_report(
    df: pd.DataFrame,
    filename: str,
    ratio_config: Mapping[str, Any],  # NEW ARGUMENT
    sample_size: int = 10,
    lowess_frac: float = 0.05,
    seed: int = 42,
    y_limits: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> None:
    """
    Generates a PDF with consistent mother selection and optional Y-axis limits.
    Now uses ratio_config to determine channels and thresholds dynamically.
    """
    if sm is None:
        raise ImportError("generate_single_cell_report requires statsmodels installed (statsmodels.api as sm).")

    random.seed(seed)
    np.random.seed(seed)

    # 1. Extract Config
    num_col = ratio_config.get('numerator', 'GFP')
    den_col = ratio_config.get('denominator', 'mScarlet-I3')
    thresh = ratio_config.get('min_denominator_threshold', 15)
    
    # 2. Prepare Data
    subset = df[df["fluor_intensity_QC"] == True].copy()
    pivot_df = (
        subset.pivot_table(index=["mother_id", "time", "condition"], columns="channel", values="intensity")
        .reset_index()
    )

    # 3. Dynamic Ratio Calculation
    pivot_df["ratio"] = np.nan
    if den_col in pivot_df.columns and num_col in pivot_df.columns:
        mask = pivot_df[den_col] >= thresh
        pivot_df.loc[mask, "ratio"] = pivot_df.loc[mask, num_col] / pivot_df.loc[mask, den_col]

    sorted_conditions, cond_lane_map, _ = _lane_sorting(df)

    # 4. Define Colors Dynamically
    c_num = _channel_color(num_col)
    c_den = _channel_color(den_col)
    c_ratio = "purple"

    with PdfPages(filename) as pdf:
        sns.set_theme(style="ticks", context="notebook")

        for cond in sorted_conditions:
            cond_data = pivot_df[pivot_df["condition"].astype(str) == str(cond)]
            if cond_data.empty:
                continue

            all_moms = sorted(cond_data["mother_id"].unique())
            selected_moms = random.sample(all_moms, min(sample_size, len(all_moms)))
            lane_num = cond_lane_map.get(str(cond), "?")

            print(f"Generating plots for Lane {lane_num} ({cond})...")

            for mid in selected_moms:
                mom = cond_data[cond_data["mother_id"] == mid].sort_values("time")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                def plot_channels(ax, data, smooth: bool):
                    ax_den = ax.twinx()
                    ax_ratio = ax.twinx()
                    # Offset the right spine for ratio to prevent overlap
                    ax_ratio.spines["right"].set_position(("outward", 60))

                    def _maybe_lowess(series_name: str, target_ax, color):
                        v = data.dropna(subset=[series_name])
                        if len(v) > 5:
                            sm_data = sm.nonparametric.lowess(v[series_name], v["time"], frac=lowess_frac)
                            target_ax.plot(sm_data[:, 0], sm_data[:, 1], lw=2, color=color)
                        else:
                            target_ax.plot(v["time"], v[series_name], lw=1, color=color)

                    if not smooth:
                        if num_col in data.columns:
                            ax.plot(data["time"], data[num_col], lw=1.2, label=num_col, color=c_num, alpha=0.6)
                        if den_col in data.columns:
                            ax_den.plot(data["time"], data[den_col], lw=1.2, label=den_col, color=c_den, alpha=0.6)
                        ax_ratio.plot(data["time"], data["ratio"], lw=1.5, alpha=0.8, label="Ratio", color=c_ratio)
                    else:
                        if num_col in data.columns:
                            _maybe_lowess(num_col, ax, c_num)
                        if den_col in data.columns:
                            _maybe_lowess(den_col, ax_den, c_den)
                        _maybe_lowess("ratio", ax_ratio, c_ratio)

                    # Coloring Axis Labels and Ticks
                    ax.set_ylabel(num_col, fontweight="bold", color=c_num)
                    ax.tick_params(axis='y', labelcolor=c_num)
                    
                    ax_den.set_ylabel(den_col, fontweight="bold", color=c_den)
                    ax_den.tick_params(axis='y', labelcolor=c_den)
                    
                    ax_ratio.set_ylabel("Ratio", fontweight="bold", color=c_ratio)
                    ax_ratio.tick_params(axis='y', labelcolor=c_ratio)

                    # Dynamic Y-Limits
                    if y_limits:
                        if y_limits.get(num_col) and num_col in data.columns:
                            ax.set_ylim(y_limits[num_col])
                        if y_limits.get(den_col) and den_col in data.columns:
                            ax_den.set_ylim(y_limits[den_col])
                        if y_limits.get("ratio"):
                            ax_ratio.set_ylim(y_limits["ratio"])

                    ax.set_title("Raw" if not smooth else f"Smoothed (frac={lowess_frac})", fontsize=12, pad=5, loc="left")

                plot_channels(ax1, mom, smooth=False)
                plot_channels(ax2, mom, smooth=True)

                plt.suptitle(f"Lane {lane_num}: {cond} | Mother: {mid}", fontweight="bold", y=0.97)
                ax2.set_xlabel("Time (min)")
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(120))

                plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Report saved: {filename}")
    

def plot_gfp_bg_histograms_by_condition(df: pd.DataFrame, line_percentiles: Sequence[float]) -> None:
    """
    Plots stacked histograms of GFP background intensity, one per condition.
    - Filters for Channel = 'GFP' and Growing Cells (growing_before=True).
    - All plots share the same X-axis limits (0.1% to 99% of total data).
    - Adds vertical dashed lines at specified percentiles for each condition.
    """
    # 1. Select Data (GFP & Growing only)
    target_channel = "GFP"
    required = ["growing_before", "channel", "intensity_bg", "condition"]
    
    if not all(c in df.columns for c in required):
        print(f"Error: Missing columns. Required: {required}")
        return

    subset = df[
        (df["growing_before"] == True) & 
        (df["channel"] == target_channel)
    ].copy()

    if subset.empty:
        print(f"No growing cell data found for channel: {target_channel}")
        return

    # 2. Setup Plotting
    # Sort conditions naturally or by lane if available
    sorted_conds, _, _ = _lane_sorting(subset)
    n_conds = len(sorted_conds)
    
    # Calculate global X-limits to apply to all plots (0.1% - 99%)
    all_vals = subset["intensity_bg"].dropna()
    if all_vals.empty: return
    
    global_xlim = np.percentile(all_vals, [0.1, 99])
    
    # Create Stacked Subplots
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=n_conds, 
        ncols=1, 
        figsize=(12, 3 * n_conds), 
        sharex=True, 
        constrained_layout=True
    )
    
    # Handle single condition case (axes is not a list)
    if n_conds == 1: axes = [axes]

    print(f"Plotting {target_channel} background histograms for {n_conds} conditions...")

    # 3. Iterate and Plot
    for i, cond in enumerate(sorted_conds):
        ax = axes[i]
        cond_data = subset.loc[subset["condition"].astype(str) == str(cond), "intensity_bg"].dropna()
        
        color = _channel_color(target_channel)
        
        if cond_data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        else:
            # Histogram
            sns.histplot(
                cond_data, 
                kde=True, 
                color=color, 
                edgecolor="white", 
                line_kws={"linewidth": 2}, 
                ax=ax
            )
            
            # Percentile Lines (Calculated per condition)
            lines_vals = np.percentile(cond_data, line_percentiles)
            for p, val in zip(line_percentiles, lines_vals):
                ax.axvline(val, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
                # Label top percentiles only to avoid clutter? Or just legend.
                # Here we just keep lines as requested.

        # Formatting
        ax.set_xlim(global_xlim)
        ax.set_title(f"Condition: {cond} (n={len(cond_data)})", fontweight="bold", loc="left", fontsize=12)
        ax.set_ylabel("Count")
        
        # Only label X-axis on bottom plot
        if i == n_conds - 1:
            ax.set_xlabel(f"{target_channel} Background Intensity")

    plt.suptitle(f"{target_channel} Background Intensity by Condition", fontweight="bold", y=1.02)
    plt.show()

# =============================================================================
# Safety Wrappers & Pipeline Executors
# =============================================================================

def make_safe(func):
    """Decorator: Returns None if the first argument (df) is None."""
    def wrapper(df, *args, **kwargs):
        if df is None:
            print(f"Skipping {func.__name__}: Data not loaded.")
            return None
        return func(df, *args, **kwargs)
    return wrapper

def make_safe_tuple(func):
    """Decorator: Returns (None, None) if df is None (for unpacking)."""
    def wrapper(df, *args, **kwargs):
        if df is None:
            print(f"Skipping {func.__name__}: Data not loaded.")
            return None, None
        return func(df, *args, **kwargs)
    return wrapper

# --- 1. Apply Safety Wrappers to Existing Functions ---
# This "Monkey-patches" the functions so they handle None automatically.
plot_fluorescence_bg_histograms_custom = make_safe(plot_fluorescence_bg_histograms_custom)
plot_filtered_bg_histograms = make_safe(plot_filtered_bg_histograms)
visualize_filtering_impact = make_safe(visualize_filtering_impact)
troubleshoot_high_loss_dual_plot = make_safe(troubleshoot_high_loss_dual_plot)
apply_fluorescence_qc = make_safe(apply_fluorescence_qc)
plot_mean_intensity_histograms = make_safe(plot_mean_intensity_histograms)
plot_mean_intensity_histograms_zoomed_bins = make_safe(plot_mean_intensity_histograms_zoomed_bins)
apply_intensity_qc = make_safe(apply_intensity_qc)
plot_population_mean_fluorescence = make_safe(plot_population_mean_fluorescence)
plot_population_mean_fluorescence_ordered_by_lane = make_safe(plot_population_mean_fluorescence_ordered_by_lane)
plot_population_mean_ratio_ordered_by_lane = make_safe(plot_population_mean_ratio_ordered_by_lane)
plot_individual_traces_by_lane = make_safe(plot_individual_traces_by_lane)
plot_individual_traces_by_lane_smoothed = make_safe(plot_individual_traces_by_lane_smoothed)
generate_single_cell_report = make_safe(generate_single_cell_report)

# Special handling for functions that return multiple values (to avoid unpacking errors)
summarize_regions_with_singlecell_sd_and_gfp = make_safe_tuple(summarize_regions_with_singlecell_sd_and_gfp)


# --- 2. Define High-Level Workflow Executors ---
# These replace the complex execution blocks with clean function calls.
def execute_batch_start_end_flags(
    df: Optional[pd.DataFrame], 
    params: Optional[Dict[str, Any]]
) -> Optional[pd.DataFrame]:
    """
    Executes batch application of start/end flags.
    Wrapper for clean notebook execution.
    """
    if df is None:
        print("Skipping Start/End Check: Data not loaded.")
        return None
    
    if not params:
        raise ValueError("CRITICAL ERROR: 'start_end_params' missing. Please run the interactive Start/End tool first.")

    return run_batch_start_end_flags(df, params)

def launch_interactive_cleaning(df: Optional[pd.DataFrame], config: Dict) -> Optional[Dict]:
    """Safely launches the interactive cleaning tool."""
    if df is not None:
        return run_interactive_filtering(df, config)
    print("Skipping Cleaning Tool: Data not loaded.")
    return None

def execute_batch_cleaning(
    df: Optional[pd.DataFrame], 
    params: Optional[Dict[str, Any]],
    num_IDs_to_plot_per_lane: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Executes batch cleaning. 
    If num_IDs_to_plot_per_lane is set, the PDF report will only include a 
    random sample of that many mothers per lane, but the returned 
    DataFrame will still contain ALL data.
    """
    if df is None:
        print("Skipping Batch Cleaning: Data not loaded.")
        return None
    if not params:
        raise ValueError("CRITICAL ERROR: 'cleaning_params' missing.")

    # 1. Generate Filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_folder = os.path.basename(os.getcwd())
    pdf_filename = f"growth_traces_cleaned_{current_folder}_{timestamp}.pdf"

    print(f"{'='*40}\n STARTING BATCH CLEANING \n{'='*40}")
    print(f"Report: {pdf_filename}")
    
    # 2. Run Cleaning on FULL Dataset
    df_clean = run_batch_cleaning(df.copy(), params)
    
    # 3. Prepare Data for PDF Report
    df_for_report = df_clean
    
    if num_IDs_to_plot_per_lane:
        print(f"Sampling ~{num_IDs_to_plot_per_lane} mothers per lane for the PDF report...")
        sorted_conds, _, _ = _lane_sorting(df_clean)
        sampled_mids = []
        
        # Sample IDs from each condition/lane group
        for cond in sorted_conds:
            cond_moms = df_clean[df_clean['condition'].astype(str) == str(cond)]['mother_id'].unique()
            if len(cond_moms) > num_IDs_to_plot_per_lane:
                sampled_mids.extend(random.sample(list(cond_moms), num_IDs_to_plot_per_lane))
            else:
                sampled_mids.extend(cond_moms)
                
        df_for_report = df_clean[df_clean['mother_id'].isin(sampled_mids)]

    # 4. Generate Report
    generate_pdf_report(df_for_report, pdf_filename)
    
    # 5. Cleanup
    if 'cleaning_status' in df_clean.columns:
        df_clean.drop(columns=['cleaning_status'], inplace=True)
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_clean
        
def launch_interactive_division_testing(df: Optional[pd.DataFrame], config: Dict) -> Optional[Dict]:
    if df is not None and 'log_length_cleaned' in df.columns:
        return run_division_testing(df, config)
    print("Skipping: Load data with 'log_length_cleaned' first.")
    return None

def execute_batch_division_detection(df: Optional[pd.DataFrame], params: Dict) -> Optional[pd.DataFrame]:
    if df is None: return None
    if not params: raise ValueError("Error: div_params missing.")
    if 'log_length_cleaned' not in df.columns: raise ValueError("Error: Cleaning not run.")

    print(f"{'='*40}\n STARTING DIVISION DETECTION \n{'='*40}")
    df_out = run_batch_division_detection(df.copy(), params)
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_out

# =============================================================================
# New: Image-based Interactive Tools
# =============================================================================

def _load_zarr_safe(path: str) -> Any:
    """Helper to load Zarr array safely."""
    if zarr is None:
        raise ImportError("Zarr is not installed. Please install it to use image features.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zarr file not found: {path}")
    return zarr.open(path, mode='r')

# --- HELPER CLASS FOR FILTERED VIEWING ---
class LazyFilteredArray:
    """
    Virtual Array that wraps a Zarr/Numpy array to expose only specific indices (rows).
    Used to show only filtered 'Mother IDs' in Napari without loading everything to RAM.
    """
    def __init__(self, source_array, valid_indices):
        self.source = source_array
        self.indices = list(valid_indices) # List of valid real_ids
        # Shape: (N_Filtered, Time, Y, X)
        self.shape = (len(valid_indices), *source_array.shape[1:])
        self.dtype = source_array.dtype
        self.ndim = source_array.ndim

    def __getitem__(self, key):
        # Napari requests slices (tuple of ints or slices)
        try:
            if isinstance(key, tuple):
                idx = key[0]
                rest = key[1:]
                
                if isinstance(idx, int):
                    # Map virtual index -> real mother_id
                    real_id = self.indices[idx]
                    return self.source[real_id].__getitem__(rest)
                elif isinstance(idx, slice):
                    # Slicing range of mothers
                    real_ids = self.indices[idx]
                    # Stack results (heavy, but rarely requested by Napari for 4D)
                    arrays = [self.source[i].__getitem__(rest) for i in real_ids]
                    return np.stack(arrays) if arrays else np.array([], dtype=self.dtype)
            
            elif isinstance(idx, int):
                return self.source[self.indices[key]]
                
        except Exception:
            pass # Fallback for edge cases
        return np.zeros(self.shape[1:], dtype=self.dtype)

# =============================================================================
# Image-based Start/End Checks
# =============================================================================

class VirtualMapArray:
    """
    A lightweight wrapper that maps a contiguous virtual index (0..N) 
    to a non-contiguous set of Real Mother IDs in a Zarr array.
    
    Optimized to return concrete Numpy arrays directly to Napari, 
    bypassing Dask's slow graph generation for non-contiguous slicing.
    """
    def __init__(self, zarr_array, valid_indices):
        self.source = zarr_array
        self.indices = valid_indices # List of real integers
        # Shape: (N_Filtered, Time, Y, X)
        self.shape = (len(valid_indices), *zarr_array.shape[1:])
        self.ndim = zarr_array.ndim
        self.dtype = zarr_array.dtype

    def __getitem__(self, key):
        # 1. Normalize key to separate the Mother Dimension (dim 0) from the rest
        if isinstance(key, tuple):
            idx = key[0]
            rest = key[1:]
        else:
            idx = key
            rest = ()

        # 2. Handle Integer Indexing (Standard Scrolling)
        # Check for both int and numpy integer types
        if isinstance(idx, (int, np.integer)):
            idx = int(idx) # Cast to Python int
            
            # Normalize negative indices
            if idx < 0: 
                idx += len(self.indices)
            
            if idx < 0 or idx >= len(self.indices):
                # Robust fallback for out-of-bounds requests
                try:
                    dummy = self.source[0].__getitem__(rest)
                    return np.zeros_like(dummy)
                except Exception:
                    return np.zeros((1,)*len(rest), dtype=self.dtype)
            
            # Map Virtual -> Real
            real_idx = self.indices[idx]
            
            # Fetch directly from Zarr -> Returns Numpy Array
            return self.source[real_idx].__getitem__(rest)
            
        # 3. Handle Slicing (Batch/Thumbnail operations)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            virtual_range = range(start, stop, step)
            real_indices = [self.indices[i] for i in virtual_range]
            
            if len(real_indices) == 0:
                # Handle empty result
                try:
                    dummy = self.source[0].__getitem__(rest)
                    return np.zeros((0, *dummy.shape), dtype=self.dtype)
                except:
                    return np.array([], dtype=self.dtype)

            # Fetch items one by one (Fast random access in Zarr)
            items = [self.source[i].__getitem__(rest) for i in real_indices]
            
            # Stack into a single numpy array
            return np.stack(items, axis=0)
            
        else:
            # Fallback for unexpected types
            return np.zeros(self.shape, dtype=self.dtype)

def run_interactive_start_end_image_check(
    df: pd.DataFrame, 
    trenches_path: str,
    masks_path: str,
    config: Mapping[str, Mapping[str, Any]],
    upscaled: bool = True,
    jump_to_end: bool = False,
    strict_index_match: bool = True,
    params_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Napari-based interactive tuning for Start/End checks using IMAGES.
    """
    if not _HAS_GUI:
        raise ImportError("Interactive testing requires napari + qtpy + magicgui installed.")

    print(f"Loading images from:\n - {trenches_path}\n - {masks_path}")
    trenches_zarr = _load_zarr_safe(trenches_path)
    masks_zarr = _load_zarr_safe(masks_path)
    
    valid_mids = sorted(df["mother_id"].unique().astype(int))
    max_zarr_idx = trenches_zarr.shape[0] - 1
    safe_mids = [m for m in valid_mids if 0 <= m <= max_zarr_idx]
    n_moms = len(safe_mids)
    
    if n_moms == 0:
        print("Warning: No valid traces to show.")
        return {}
    
    if strict_index_match:
        import dask.array as da
        limit = min(max(safe_mids) + 1, trenches_zarr.shape[0])
        trenches_view = da.from_array(trenches_zarr)[:limit]
        masks_view = da.from_array(masks_zarr)[:limit]
        def get_real_mid(idx): return idx
        viewer_title = f"Check (Index = Mother ID)"
    else:
        trenches_view = VirtualMapArray(trenches_zarr, safe_mids)
        masks_view = VirtualMapArray(masks_zarr, safe_mids)
        def get_real_mid(idx): return safe_mids[idx]
        viewer_title = f"Check (Showing {n_moms} filtered traces)"

    trace_lookup = df.groupby("mother_id")["timepoint"].apply(set).to_dict()
    global_tps = sorted(df["timepoint"].unique())

    viewer = napari.Viewer(title=viewer_title)
    trench_scale = [1, 1, 2, 2] if upscaled else [1, 1, 1, 1]
    viewer.add_image(trenches_view, name="Trenches (PC)", colormap="gray", blending="additive", scale=trench_scale)
    viewer.add_labels(masks_view, name="Masks (Filtered)", opacity=0.6)

    final_params = {k: v["default"] for k, v in config.items()}

    def fix_widget_focus(mg_widget, step=None):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
                if step is not None:
                    spinner.setSingleStep(step)
        except Exception:
            pass

    def make_int_slider(key, label):
        params = config[key]
        w = Slider(min=params['min'], max=params['max'], step=params['step'], value=params['default'], label=label)
        fix_widget_focus(w, step=int(params['step']))
        return w

    w_first_n = make_int_slider('n_first', "Num First Timepoints")
    w_miss_first = make_int_slider('miss_first', "Max Miss First (Count)")
    w_last_n = make_int_slider('n_last', "Num Last Timepoints")
    w_miss_last = make_int_slider('miss_last', "Max Miss Last (Count)")
    
    # FIXED TEXT HERE
    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(60)

    from qtpy.QtWidgets import QLabel, QWidget, QVBoxLayout
    from qtpy.QtCore import Qt
    class StatusOverlay(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.lbl = QLabel()
            self.lbl.setAlignment(Qt.AlignCenter)
            self.lbl.setStyleSheet("font-weight: bold; font-size: 16px; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 5px;")
            layout = QVBoxLayout()
            layout.addWidget(self.lbl)
            layout.setContentsMargins(0,0,0,0)
            self.setLayout(layout)
        def update_text(self, mid, is_start, miss_start, is_end, miss_end):
            s_color = "#39FF14" if is_start else "#FF3131"
            e_color = "#39FF14" if is_end else "#FF3131"
            s_text = "PASS" if is_start else "FAIL"
            e_text = "PASS" if is_end else "FAIL"
            self.lbl.setText(f"<span style='color:white'>Mother {mid} | </span><span style='color:white'>Start: </span><span style='color:{s_color}'>{s_text} (Miss: {miss_start})</span><span style='color:white'>   End: </span><span style='color:{e_color}'>{e_text} (Miss: {miss_end})</span>")

    status_widget = StatusOverlay()
    viewer.window.add_dock_widget(status_widget, area='top', name="Status")

    c1 = Container(widgets=[w_first_n, w_miss_first], layout="vertical")
    c2 = Container(widgets=[w_last_n, w_miss_last], layout="vertical")
    c3 = Container(widgets=[w_save], layout="vertical")
    main_c = Container(widgets=[c1, c2, c3], layout="horizontal")

    def update_status():
        current_step = viewer.dims.current_step
        if not current_step: return
        idx = int(current_step[0])
        if not strict_index_match and idx >= len(safe_mids): return
        
        mid = get_real_mid(idx)
        present_tps = trace_lookup.get(mid, [])

        if present_tps:
            req_start = global_tps[:w_first_n.value]
            miss_s = len([t for t in req_start if t not in present_tps])
            
            n_end = w_last_n.value
            req_end = global_tps[-n_end:] if n_end > 0 else []
            miss_e = len([t for t in req_end if t not in present_tps])
            
            status_widget.update_text(mid, miss_s <= w_miss_first.value, miss_s, miss_e <= w_miss_last.value, miss_e)
        else:
             status_widget.lbl.setText(f"<span style='color:white'>Mother {mid}: NO DATA</span>")

    for w in [w_first_n, w_miss_first, w_last_n, w_miss_last]:
        w.changed.connect(update_status)
    viewer.dims.events.current_step.connect(lambda e: update_status())

    @w_save.clicked.connect
    def save():
        print("\n" + "=" * 40 + "\n START/END PARAMS SAVED\n" + "=" * 40)
        current_vals = {
            "n_first": w_first_n.value,
            "miss_first": w_miss_first.value,
            "n_last": w_last_n.value,
            "miss_last": w_miss_last.value,
        }
        
        for k, v in current_vals.items():
            print(f"{k:<20}: {v}")
            final_params[k] = v
            
        if params_dict is not None:
            print(">> Updating 'params' variable in-place.")
            params_dict.update(current_vals)
            
        print("=" * 40 + "\n")
        viewer.close()

    viewer.window.add_dock_widget(main_c, area="bottom", name="Check Controls")
    
    init_mom_idx = min(safe_mids) if strict_index_match else 0
    viewer.dims.set_current_step(0, init_mom_idx)
    
    tp_idx = len(global_tps) - 1 if jump_to_end else max(0, min(w_first_n.value - 1, len(global_tps)-1))
    viewer.dims.set_current_step(1, tp_idx)
    
    update_status()
    viewer.window.qt_viewer.window().showMaximized()
    
    napari.run()
    return final_params
    

# --- INSPECTION WRAPPERS ---
def _update_config_defaults(base_config: Mapping[str, Mapping[str, Any]], current_params: Mapping[str, Any]) -> Dict:
    """
    Creates a deep copy of the base_config and updates the 'default' value 
    for each key to match the current_params.
    """
    new_config = copy.deepcopy(base_config)
    for key, value in current_params.items():
        if key in new_config:
            new_config[key]['default'] = value
    return new_config

def inspect_start_fails(df, params, trenches_path, masks_path, config):
    print("Identifying Start Failures...")
    temp_df = run_batch_start_end_flags(df, params)
    fails_df = temp_df[temp_df["at_start"] == False].copy()
    
    if fails_df.empty:
        print("No Start Failures found!")
        return params

    print(f"Found {fails_df['mother_id'].nunique()} Start Failures. Launching viewer...")
    
    # Sync sliders to current params
    updated_config = _update_config_defaults(config, params)
    
    # Pass 'params' explicitly for in-place update
    run_interactive_start_end_image_check(
        fails_df, trenches_path, masks_path, updated_config, 
        upscaled=True, jump_to_end=False, strict_index_match=False,
        params_dict=params  # <--- Linked here
    )
    return params

    
def inspect_end_fails(df, params, trenches_path, masks_path, config):
    print("Identifying End Failures...")
    temp_df = run_batch_start_end_flags(df, params)
    fails_df = temp_df[temp_df["at_end"] == False].copy()
    
    if fails_df.empty:
        print("No End Failures found!")
        return params

    print(f"Found {fails_df['mother_id'].nunique()} End Failures. Launching viewer...")
    
    # Sync sliders to current params
    updated_config = _update_config_defaults(config, params)
    
    # Pass 'params' explicitly for in-place update
    run_interactive_start_end_image_check(
        fails_df, trenches_path, masks_path, updated_config, 
        upscaled=True, jump_to_end=True, strict_index_match=False,
        params_dict=params  # <--- Linked here
    )
    return params

def run_batch_start_end_flags(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """
    Applies start/end flags using the dictionary returned by the interactive tool.
    Updated for COUNT-based missing frames.
    """
    print(f"{'='*40}\n APPLYING START/END FLAGS \n{'='*40}")
    
    # 1. Resolve Parameters
    n_first = params.get('n_first', 20)
    n_last = params.get('n_last', 20)
    
    # Defaults to integer counts now
    miss_start = int(params.get('miss_first', 2))
    miss_end = int(params.get('miss_last', 2))
    
    print(f"Start Window: First {n_first} pts (Max {miss_start} missing)")
    print(f"End Window:   Last  {n_last} pts (Max {miss_end} missing)")
    
    # 2. Apply
    df_out = add_start_end_flags(
        df,
        first_x_timepoints=n_first,
        last_x_timepoints=n_last,
        max_missing_start_count=miss_start, # Updated arg name
        max_missing_end_count=miss_end      # Updated arg name
    )
    
    # 3. Summarize
    print_start_end_summary(df_out)
    print("-" * 40)
    
    return df_out    


# Growth Curve Smoothing
# =============================================================================
# Core Smoothing Logic
# =============================================================================

def _calculate_robust_gradient(t_window, y_window):
    """Calculates robust slope using linear regression."""
    valid = np.isfinite(y_window)
    if valid.sum() < 2:
        return None
    slope, _ = np.polyfit(t_window[valid], y_window[valid], 1)
    return slope

def flatten_trace_by_divisions(time, log_len, div_mask, grad_win_pre=5, grad_win_post=5):
    """Flattens sawtooth trace using gradient projection."""
    y = log_len.copy()
    n = len(y)
    div_indices = np.where(div_mask)[0]
    shifts = []
    
    for i in sorted(div_indices):
        if i == 0 or i >= n: continue
            
        pre_idx = i - 1
        search_limit = max(0, i - 3)
        while pre_idx >= search_limit and np.isnan(y[pre_idx]):
            pre_idx -= 1
            
        if pre_idx < search_limit or np.isnan(y[pre_idx]) or np.isnan(y[i]):
            continue
            
        # Define windows
        w_start_pre = max(0, pre_idx - grad_win_pre + 1)
        t_pre = time[w_start_pre : pre_idx + 1]
        y_pre_win = y[w_start_pre : pre_idx + 1]
        
        w_end_post = min(n, i + grad_win_post)
        t_post = time[i : w_end_post]
        y_post_win = y[i : w_end_post]
        
        # Calculate gradients
        grad_pre = _calculate_robust_gradient(t_pre, y_pre_win)
        grad_post = _calculate_robust_gradient(t_post, y_post_win)
        
        if grad_pre is not None and grad_post is not None:
            avg_grad = (grad_pre + grad_post) / 2
        elif grad_pre is not None:
            avg_grad = grad_pre
        elif grad_post is not None:
            avg_grad = grad_post
        else:
            avg_grad = 0.0
            
        dt = time[i] - time[pre_idx]
        val_pre = y[pre_idx]
        val_post = y[i]
        
        expected_val = val_pre + (avg_grad * dt)
        step_delta = expected_val - val_post
        
        if step_delta < -1.0: continue
            
        y[i:] += step_delta
        shifts.append({'index': i, 'delta': step_delta, 'time': time[i]})
        
    return y, shifts

def unflatten_trace(flat_y, shifts):
    """Restores sawtooth pattern."""
    y = flat_y.copy()
    for shift in shifts:
        idx = shift['index']
        delta = shift['delta']
        if idx < len(y):
            y[idx:] -= delta
    return y

def _run_lowess(t, y, frac):
    """Helper to run LOWESS and interpolate back to t."""
    valid_mask = np.isfinite(y)
    if valid_mask.sum() > 5:
        # OPTIMIZATION: is_sorted=True significantly speeds up statsmodels lowess
        smoothed = sm.nonparametric.lowess(y[valid_mask], t[valid_mask], frac=frac, is_sorted=True)
        return np.interp(t, smoothed[:, 0], smoothed[:, 1])
    else:
        return y

def apply_smoothing_pipeline(df_trace, window_size=20, window_early=20, early_limit=0.0, start_flat=0.0):
    """
    Pipeline: Flatten -> Pre-process Flat Start -> Dual LOWESS -> Enforce Flat Start -> Unflatten.
    
    New Argument:
      start_flat (float): Time (min) from start where growth is forced to be flat (median of region).
    """
    t = df_trace["time"].values
    y_raw = df_trace["log_length_cleaned"].values
    divs = df_trace["division_event"].fillna(False).values
    n_points = len(t)
    
    # 1. Flatten
    y_flat_raw, shifts = flatten_trace_by_divisions(t, y_raw, divs)
    
    # --- STEP 1.5: Handle Flat Start (Input Modification) ---
    # We modify the input to LOWESS so the smooth curve naturally approaches the flat value
    y_input_for_lowess = y_flat_raw.copy()
    flat_mask = (t <= start_flat)
    flat_median = None
    
    if start_flat > 0 and flat_mask.any():
        # Calculate median of the FLATTENED data in this region
        valid_pts = y_flat_raw[flat_mask]
        valid_pts = valid_pts[np.isfinite(valid_pts)]
        
        if len(valid_pts) > 0:
            flat_median = np.median(valid_pts)
            # Overwrite input data for LOWESS with this median to guide the smoother
            y_input_for_lowess[flat_mask] = flat_median

    # 2. Dual LOWESS Strategy (Using modified input)
    frac_main = min(1.0, max(0.01, window_size / n_points)) if n_points > 0 else 0.1
    y_smooth_main = _run_lowess(t, y_input_for_lowess, frac_main)

    if early_limit > 0 and window_early is not None and n_points > 0:
        frac_early = min(1.0, max(0.01, window_early / n_points))
        y_smooth_early = _run_lowess(t, y_input_for_lowess, frac_early)
        
        # 3. Blend Curves
        blend_width = 25.0 
        k = 4.0 / (blend_width / 2.0)
        weights = 1.0 / (1.0 + np.exp(k * (t - early_limit)))
        
        y_smooth_cont = (weights * y_smooth_early) + ((1.0 - weights) * y_smooth_main)
        used_frac = (frac_main, frac_early)
    else:
        y_smooth_cont = y_smooth_main
        used_frac = (frac_main, None)
        
    # --- STEP 3.5: Enforce Flat Start (Output Constraint) ---
    # Strictly force the output to be the median value in the flat region
    # (Fixes any minor wiggles LOWESS might have left)
    if flat_median is not None:
        y_smooth_cont[flat_mask] = flat_median
        
    # 4. Unflatten
    y_fit = unflatten_trace(y_smooth_cont, shifts)
    
    return t, y_raw, y_fit, y_flat_raw, y_smooth_cont, divs, used_frac
    
# =============================================================================
# Helper for Parallel Execution
# =============================================================================

def _process_mother_group(mid, group, win_main, win_early, t_lim):
    """Worker function for parallel processing."""
    try:
        # group is passed as a DataFrame
        _, _, y_fit, _, _, _, _ = apply_smoothing_pipeline(
            group, window_size=win_main, window_early=win_early, early_limit=t_lim
        )
        return pd.Series(y_fit, index=group.index)
    except Exception as e:
        # Fail gracefully
        print(f"Error smoothing mother {mid}: {e}")
        return pd.Series(np.nan, index=group.index)

# =============================================================================
# Interactive Viewer (Same as before)
# =============================================================================

if _HAS_GUI:
    class SmoothingPlotter(FigureCanvasQTAgg):
        def __init__(self, width=14, height=8, dpi=100, zoom_xlims=(100, 400)):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.zoom_xlims = zoom_xlims
            
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
            
            self.ax_flat = self.fig.add_subplot(gs[:, 0])     
            self.ax_fit  = self.fig.add_subplot(gs[0, 1])     
            self.ax_orig = self.fig.add_subplot(gs[1, 1], sharex=self.ax_fit) 
            
            self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, y_raw, y_fit, y_flat_raw, y_smooth_cont, divs, fracs, w_main, w_early, t_lim):
            self.ax_flat.clear()
            self.ax_fit.clear()
            self.ax_orig.clear()
            
            C_RAW = 'gray'
            C_DIV = 'cyan'    
            C_FIT = '#d62728' 
            C_FLAT_TREND = 'purple'
            C_LIM = 'orange'
            
            LW_FIT = 2.0      
            LW_CONNECT = 2.5 
            
            point_colors = np.where(divs, C_DIV, C_RAW)
            
            # --- 1. LEFT PLOT: Flattened Raw ---
            self.ax_flat.scatter(t, y_flat_raw, c=point_colors, s=15, alpha=0.6, label='Flattened Raw', zorder=5)
            if t_lim > 0:
                self.ax_flat.axvline(t_lim, color=C_LIM, linestyle='--', alpha=0.6, label='Early Limit')

            self.ax_flat.set_title(f"Cumulative Growth\n(Flattened Raw)", fontweight="bold", fontsize=10)
            self.ax_flat.set_xlabel("Time (min)")
            self.ax_flat.set_ylabel("Cumulative log(L)")
            
            self.ax_flat.set_xlim(self.zoom_xlims)
            x_min, x_max = self.zoom_xlims
            mask_view = (t >= x_min) & (t <= x_max)
            if mask_view.any():
                y_view = y_flat_raw[mask_view]
                y_view = y_view[np.isfinite(y_view)]
                if len(y_view) > 0:
                    y_span = y_view.max() - y_view.min()
                    pad = y_span * 0.05 if y_span > 0 else 0.1
                    self.ax_flat.set_ylim(y_view.min() - pad, y_view.max() + pad)
            
            # --- 2. TOP RIGHT: Final Fitted Trace ---
            valid = np.isfinite(y_raw)
            if valid.any():
                self.ax_fit.plot(t[valid], y_raw[valid], c=C_RAW, lw=LW_CONNECT, alpha=0.4, zorder=1)
                
            self.ax_fit.scatter(t, y_raw, c=point_colors, s=20, alpha=0.6, label='Raw Data', zorder=2)
            self.ax_fit.plot(t, y_fit, c=C_FIT, lw=LW_FIT, label=f'Final Fit', zorder=3)
            
            if divs.any():
                div_t = t[divs]
                div_y = y_raw[divs]
                self.ax_fit.scatter(div_t, div_y, c=C_DIV, s=40, zorder=10)

            title_str = f"Mother ID: {mid} | Final Fit | Main: {w_main} pts"
            if t_lim > 0:
                title_str += f" | Early: {w_early} pts (<{t_lim} min)"
            
            self.ax_fit.set_title(title_str, fontweight="bold")
            self.ax_fit.set_ylabel("log(Length)")
            self.ax_fit.grid(True, ls=':', alpha=0.3)
            
            # --- 3. BOTTOM RIGHT: Original Cleaned Trace ---
            if valid.any():
                self.ax_orig.plot(t[valid], y_raw[valid], c=C_RAW, lw=LW_CONNECT, alpha=0.4, zorder=1)
            
            self.ax_orig.scatter(t, y_raw, c=point_colors, s=20, alpha=0.6, label='Original Cleaned', zorder=2)
            
            if divs.any():
                div_t = t[divs]
                div_y = y_raw[divs]
                self.ax_orig.scatter(div_t, div_y, c=C_DIV, s=40, zorder=10)

            self.ax_orig.set_title(f"Original Cleaned Trace", fontweight="bold")
            self.ax_orig.set_ylabel("log(Length)")
            self.ax_orig.set_xlabel("Time (min)")
            self.ax_orig.grid(True, ls=':', alpha=0.3)
            
            self.draw()

    def fix_widget_focus(mg_widget):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
        except Exception:
            pass

    def run_interactive_smoothing(df: pd.DataFrame, config: dict, zoom_xlims=(100, 400)):
        if not _HAS_GUI:
            print("Napari/GUI not available.")
            return None

        if "at_start" in df.columns:
            candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
        else:
            candidates = sorted(df["mother_id"].unique())
            
        n_moms = len(candidates)
        if n_moms == 0:
            print("No candidates found (at_start=True).")
            return None

        print(f"Loaded {n_moms} candidates for smoothing check.")

        viewer = napari.Viewer(title="Growth Trace Smoothing Testing")
        viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
        
        try:
            viewer.window.qt_viewer.dockLayerList.setVisible(False)
            viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        except Exception:
            pass
        
        plotter = SmoothingPlotter(zoom_xlims=zoom_xlims)
        viewer.window.add_dock_widget(plotter, area="top", name="Smoothing Plots")
        
        # Config Defaults
        cfg_win = config.get('lowess_window', {'min': 5, 'max': 200, 'step': 1, 'default': 20})
        cfg_early = config.get('lowess_window_early', {'min': 5, 'max': 200, 'step': 1, 'default': 20})
        cfg_lim = config.get('early_time_lim', {'min': 0, 'max': 1000, 'step': 10, 'default': 0})
        cfg_flat = config.get('start_flat', {'min': 0, 'max': 300, 'step': 5, 'default': 0})
        
        # Widgets
        w_window = IntSlider(
            min=cfg_win['min'], max=cfg_win['max'], step=cfg_win['step'], 
            value=cfg_win['default'], label="LOWESS Window (points)"
        )
        w_win_early = IntSlider(
            min=cfg_early['min'], max=cfg_early['max'], step=cfg_early['step'], 
            value=cfg_early['default'], label="LOWESS Window EARLY"
        )
        w_lim_early = FloatSlider(
            min=cfg_lim['min'], max=cfg_lim['max'], step=cfg_lim['step'], 
            value=cfg_lim['default'], label="Early Time Limit (min)"
        )
        w_start_flat = FloatSlider(
            min=cfg_flat['min'], max=cfg_flat['max'], step=cfg_flat['step'], 
            value=cfg_flat['default'], label="Force Flat Start (min)"
        )
        
        # Fix Focus
        for w in [w_window, w_win_early, w_lim_early, w_start_flat]:
            fix_widget_focus(w)
        
        w_save = PushButton(text="Finish and Save Parameters")
        w_save.native.setMinimumHeight(60)
        
        # Layout
        row1 = Container(widgets=[w_window, w_start_flat], layout="horizontal", labels=True)
        row2 = Container(widgets=[w_win_early, w_lim_early], layout="horizontal", labels=True)
        main_container = Container(widgets=[row1, row2, Label(value=""), w_save])
        viewer.window.add_dock_widget(main_container, area="bottom", name="Controls")
        
        final_params = {
            'lowess_window': cfg_win['default'],
            'lowess_window_early': cfg_early['default'],
            'early_time_lim': cfg_lim['default'],
            'start_flat': cfg_flat['default']
        }

        def update():
            idx = viewer.dims.current_step[0]
            if idx < n_moms:
                mid = candidates[int(idx)]
                trace = df[df["mother_id"] == mid].sort_values("time")
                
                win_main = w_window.value
                win_early = w_win_early.value
                t_lim = w_lim_early.value
                t_flat = w_start_flat.value
                
                t, y_raw, y_fit, y_flat_raw, y_smooth_cont, divs, fracs = apply_smoothing_pipeline(
                    trace, window_size=win_main, window_early=win_early, early_limit=t_lim, start_flat=t_flat
                )
                
                plotter.update_plot(
                    mid, t, y_raw, y_fit, y_flat_raw, y_smooth_cont, divs, fracs, 
                    w_main=win_main, w_early=win_early, t_lim=t_lim
                )
                
                # Visualize the Flat Cutoff in the plotter
                if t_flat > 0:
                    plotter.ax_flat.axvline(t_flat, color='green', linestyle='--', alpha=0.5, label='Flat Cutoff')
                    plotter.ax_fit.axvline(t_flat, color='green', linestyle='--', alpha=0.5)
                    plotter.draw()
                    
                viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

        for w in [w_window, w_win_early, w_lim_early, w_start_flat]:
            w.changed.connect(update)
            
        viewer.dims.events.current_step.connect(lambda e: update())
        
        @w_save.clicked.connect
        def on_save():
            final_params['lowess_window'] = w_window.value
            final_params['lowess_window_early'] = w_win_early.value
            final_params['early_time_lim'] = w_lim_early.value
            final_params['start_flat'] = w_start_flat.value
            print(f"Saved Params:\n  Main Window: {final_params['lowess_window']}\n  Early Window: {final_params['lowess_window_early']}\n  Early Limit: {final_params['early_time_lim']}\n  Start Flat: {final_params['start_flat']}")
            viewer.close()

        viewer.dims.set_current_step(0, 0)
        update()
        viewer.window.qt_viewer.window().showMaximized()
        napari.run()
        
        return final_params

# =============================================================================
# Batch Smoothing Execution
# =============================================================================

import multiprocessing

def _process_single_trace(group, win_main, win_early, t_lim, start_flat):
    """
    Worker function for parallel processing.
    Runs the pipeline on a single group and returns the result Series.
    """
    try:
        _, _, y_fit, _, _, _, _ = apply_smoothing_pipeline(
            group, window_size=win_main, window_early=win_early, early_limit=t_lim, start_flat=start_flat
        )
        return pd.Series(y_fit, index=group.index, dtype="float64")
    except Exception:
        return None

def execute_batch_smoothing(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies the Flatten -> Dual LOWESS -> Unflatten pipeline to all mothers.
    """
    if df_main is None: return None
    
    # Extract params
    win_main = params.get('lowess_window', 20)
    win_early = params.get('lowess_window_early', 20)
    t_lim = params.get('early_time_lim', 0)
    start_flat = params.get('start_flat', 0.0)
    
    # Determine CPU cores
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"{'='*40}\n STARTING BATCH SMOOTHING (Parallel n_jobs={n_jobs}) \n{'='*40}")
    print(f"  Main Window:  {win_main} pts")
    print(f"  Early Window: {win_early} pts")
    print(f"  Early Limit:  {t_lim} mins")
    print(f"  Start Flat:   {start_flat} mins")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    
    if 'log_length_smoothed' not in df_main.columns:
        df_main['log_length_smoothed'] = np.nan
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    # Execute Parallel Loop
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_trace)(group, win_main, win_early, t_lim, start_flat)
        for _, group in tqdm(grouped, desc="Smoothing growth traces", total=len(candidates))
    )
    
    # Collect results
    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        print("Concatenating results...")
        combined_updates = pd.concat(valid_results)
        df_main.loc[combined_updates.index, 'log_length_smoothed'] = combined_updates
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main

# =============================================================================
# Missing Division Detection Logic
# =============================================================================

def _calc_derivative_reverse_lookback(y, lookback=2):
    """
    Calculates a modified reverse derivative (negative drop) at index i by comparing 
    y[i] against the MAXIMUM value in the window y[i-lookback : i].
    
    The denominator is assumed to be 1 (unit time step) regardless of where the 
    max was found, to emphasize the total magnitude of the drop even if spread.
    
    Returns: Array of 'drops' (positive values indicate a drop).
    """
    n = len(y)
    deriv = np.zeros(n)
    
    # We can only look back starting from index 1
    # For indices < lookback, we look back as far as possible (down to index 0)
    
    for i in range(1, n):
        # Define window start
        start_idx = max(0, i - int(lookback))
        
        # Get window of previous values
        prev_window = y[start_idx : i]
        
        # Find max in that window
        if len(prev_window) > 0:
            max_prev = np.max(prev_window)
            
            # Calculate drop
            # Denominator is effectively 1.0 per instructions
            drop = max_prev - y[i]
            
            # We are interested in positive drops (reverse derivative)
            deriv[i] = drop
            
    return deriv

def detect_missing_divisions(t, y_smooth, existing_divs, 
                             filter_prominence, min_dist, 
                             drop_min, drop_max,
                             division_peak_lookback=2,
                             snap_enabled=True, div_timepoint_win=12.0,
                             search_prominence=None):
    """
    Detects division events based on sharp drops in y_smooth.
    Logic: Find Peaks -> Filter Overlap -> Filter Drop Size -> Snap -> Filter Dist (Internal & External).
    """
    if search_prominence is None:
        search_prominence = filter_prominence

    # 1. Calculate Drop Metric (Lookback Derivative)
    neg_deriv = _calc_derivative_reverse_lookback(y_smooth, lookback=division_peak_lookback)
    
    # Estimate sampling rate for distance conversion
    if len(t) > 1:
        dt_avg = np.median(np.diff(t))
        if dt_avg <= 0: dt_avg = 3.0
    else:
        dt_avg = 3.0
        
    dist_points = max(1, int(min_dist / dt_avg))

    # 2. Find Peaks (Candidates)
    # Note: distance=dist_points enforces min_dist for the *initial* candidates (New <-> New)
    peaks, properties = find_peaks(neg_deriv, prominence=search_prominence, distance=dist_points)
    
    # --- Overlap QC (Before Snapping) ---
    # Ignore peaks that are on top of or within 1 index of an existing division
    if existing_divs.any():
        existing_indices = np.where(existing_divs)[0]
        keep_mask_overlap = np.ones(len(peaks), dtype=bool)
        for i, p in enumerate(peaks):
            if np.any(np.abs(existing_indices - p) <= 1):
                keep_mask_overlap[i] = False
        
        peaks = peaks[keep_mask_overlap]
        if 'prominences' in properties:
            properties['prominences'] = properties['prominences'][keep_mask_overlap]

    # --- Drop Magnitude QC ---
    keep_mask_drop = np.zeros(len(peaks), dtype=bool)
    for i, p in enumerate(peaks):
        drop_val = neg_deriv[p]
        if drop_min <= drop_val <= drop_max:
            keep_mask_drop[i] = True
            
    peaks = peaks[keep_mask_drop]
    if 'prominences' in properties:
        properties['prominences'] = properties['prominences'][keep_mask_drop]

    # 3. Initial Classification (Prominence)
    peak_status = np.zeros(len(peaks), dtype=int) 
    
    if 'prominences' in properties:
        peak_proms = properties['prominences']
        peak_status[peak_proms < filter_prominence] = 0 
        peak_status[peak_proms >= filter_prominence] = 2 
    else:
        peak_proms = np.zeros(len(peaks))

    # 4. Snap to Local Minima (Forward Scan)
    if snap_enabled and div_timepoint_win > 0:
        valid_indices = np.where(peak_status == 2)[0]
        for i in valid_indices:
            p_idx = peaks[i]
            start_t = t[p_idx]
            max_t = start_t + div_timepoint_win
            
            # Search forward from peak
            forward_search = np.where((t >= start_t) & (t <= max_t))[0]
            
            if len(forward_search) > 0:
                local_y = y_smooth[forward_search]
                min_local_idx = np.argmin(local_y)
                peaks[i] = forward_search[min_local_idx]

    # 5. Post-Snap Internal Distance Check (New vs New)
    # Snapping might have moved peaks closer than min_dist. We re-check strict spacing.
    # Logic: Greedily keep peaks with higher prominence.
    valid_indices = np.where(peak_status == 2)[0]
    
    if len(valid_indices) > 1:
        # Get data for valid peaks
        current_peaks = peaks[valid_indices]
        current_proms = peak_proms[valid_indices]
        
        # Sort by prominence descending
        sort_order = np.argsort(current_proms)[::-1]
        
        accepted_temp = []
        
        for idx in sort_order:
            # Indices relative to 'valid_indices' array
            p_idx_in_peaks = valid_indices[idx] 
            p_val = peaks[p_idx_in_peaks]
            p_t = t[p_val]
            
            is_too_close = False
            for acc_t in accepted_temp:
                if abs(acc_t - p_t) < min_dist:
                    is_too_close = True
                    break
            
            if is_too_close:
                peak_status[p_idx_in_peaks] = 1 # Mark as Fail Distance
            else:
                accepted_temp.append(p_t)

    # 6. Check Distance to EXISTING divisions (New vs Existing)
    if existing_divs.any():
        existing_indices = np.where(existing_divs)[0]
        valid_indices = np.where(peak_status == 2)[0]
        
        for i in valid_indices:
            p_idx = peaks[i]
            t_p = t[p_idx]
            dists = np.abs(t[existing_indices] - t_p)
            
            if np.any(dists < min_dist):
                peak_status[i] = 1 # Mark as Fail Distance

    # 7. Create Mask
    final_valid_indices = peaks[peak_status == 2]
    final_valid_indices = np.unique(final_valid_indices)
    
    new_div_mask = np.zeros(len(t), dtype=bool)
    if len(final_valid_indices) > 0:
        new_div_mask[final_valid_indices] = True
        
    return {
        'new_div_mask': new_div_mask,
        'neg_deriv': neg_deriv,
        'peaks': peaks,
        'peak_prominences': peak_proms,
        'peak_status': peak_status
    }

# =============================================================================
# Interactive Viewer (Missing Divisions)
# =============================================================================

if _HAS_GUI:
    class MissingDivPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=8, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            
            # Layout: Top (Length), Bottom (Derivative/Prominence)
            gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
            self.ax_len = self.fig.add_subplot(gs[0])
            self.ax_peak = self.fig.add_subplot(gs[1], sharex=self.ax_len)
            
            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, y_smooth, existing_divs, det_data, filter_prom, global_max_time):
            self.ax_len.clear()
            self.ax_peak.clear()
            
            new_div_mask = det_data['new_div_mask']
            neg_deriv = det_data['neg_deriv']
            peaks = det_data['peaks']
            proms = det_data['peak_prominences']
            status = det_data['peak_status']
            
            C_TRACE = 'purple'
            C_EXISTING = 'cyan'
            C_NEW = 'magenta' 
            C_DERIV = 'gray'
            C_STEM = 'blue'
            C_FAIL_DIST = 'orange'
            
            # --- 1. Top Plot: Length ---
            self.ax_len.plot(t, y_smooth, c=C_TRACE, lw=2, label='Smoothed Length')
            
            if existing_divs.any():
                self.ax_len.scatter(t[existing_divs], y_smooth[existing_divs], 
                                  c=C_EXISTING, s=60, marker='v', zorder=5, label='Current Divs')
            
            if new_div_mask.any():
                self.ax_len.scatter(t[new_div_mask], y_smooth[new_div_mask], 
                                  c=C_NEW, s=100, marker='*', zorder=6, label='New Divisions')
            
            self.ax_len.set_title(f"Mother {mid}: Length Trace", fontweight="bold")
            self.ax_len.set_ylabel("log(Length)")
            self.ax_len.legend(loc="upper left", fontsize=9)
            self.ax_len.grid(True, ls=':', alpha=0.3)
            
            # --- 2. Bottom Plot: Prominence ---
            self.ax_peak.plot(t, neg_deriv, c=C_DERIV, lw=1.0, alpha=0.4, label='Lookback Drop Metric')
            
            # Draw Stems
            if len(peaks) > 0:
                markerline, stemlines, baseline = self.ax_peak.stem(
                    t[peaks], proms, linefmt=C_STEM, markerfmt='none', basefmt='none'
                )
                plt.setp(stemlines, 'linewidth', 1.5, 'alpha', 0.6)
                
                # Valid
                valid = (status == 2)
                if valid.any():
                    self.ax_peak.scatter(t[peaks[valid]], proms[valid], c=C_NEW, s=80, marker='*', zorder=5, label='New Division')
                
                # Fail Distance (Plot snapped locations that failed)
                dist_fail = (status == 1)
                if dist_fail.any():
                    # Only plot if NOT overlapping existing div
                    fail_indices = peaks[dist_fail]
                    fail_proms = proms[dist_fail]
                    
                    if existing_divs.any():
                        existing_t = t[existing_divs]
                        to_plot_t = []
                        to_plot_p = []
                        for ft, fp in zip(t[fail_indices], fail_proms):
                            # If not exactly on top of existing
                            if not np.any(np.abs(existing_t - ft) < 0.1):
                                to_plot_t.append(ft)
                                to_plot_p.append(fp)
                        if to_plot_t:
                            self.ax_peak.scatter(to_plot_t, to_plot_p, c=C_FAIL_DIST, s=40, marker='x', zorder=4, label='Too Close')
                    else:
                        self.ax_peak.scatter(t[fail_indices], fail_proms, c=C_FAIL_DIST, s=40, marker='x', zorder=4, label='Too Close')
                
                # Fail Prominence
                low_prom = (status == 0)
                if low_prom.any():
                    self.ax_peak.scatter(t[peaks[low_prom]], proms[low_prom], c='gray', s=10, marker='.', zorder=3)

            # Threshold Line
            self.ax_peak.axhline(filter_prom, color='red', linestyle='--', alpha=0.8, label=f'Threshold: {filter_prom:.3f}')
            
            if existing_divs.any():
                self.ax_peak.scatter(t[existing_divs], np.zeros(existing_divs.sum()), 
                                   c=C_EXISTING, s=40, marker='v', zorder=5, label='Current Divs')

            self.ax_peak.set_title("Peak Prominence (Lookback)", fontweight="bold")
            self.ax_peak.set_ylabel("Prominence")
            self.ax_peak.set_xlabel("Time (min)")
            self.ax_peak.legend(loc="upper right", fontsize=9)
            self.ax_peak.grid(True, ls=':', alpha=0.3)
            
            x_max_limit = np.ceil(global_max_time / 120) * 120
            self.ax_len.set_xlim(0, x_max_limit)
            
            max_y = max(filter_prom * 1.5, proms.max() * 1.1 if len(peaks) > 0 else 0.1)
            self.ax_peak.set_ylim(0, max_y)
            
            self.draw()

    def _fix_focus_unique(mg_widget, decimals=None):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox, QDoubleSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
                if decimals is not None and isinstance(spinner, QDoubleSpinBox):
                    spinner.setDecimals(decimals)
        except Exception:
            pass

    def run_interactive_missing_divisions(df: pd.DataFrame, config: dict):
        if not _HAS_GUI: return None

        if "at_start" in df.columns:
            candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
        else:
            candidates = sorted(df["mother_id"].unique())
        
        n_moms = len(candidates)
        if n_moms == 0: return None
        
        if "log_length_smoothed" not in df.columns:
            return None
            
        global_max_time = df["time"].max() if "time" in df.columns else 600

        viewer = napari.Viewer(title="Missing Division Finder")
        viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
        try:
            viewer.window.qt_viewer.dockLayerList.setVisible(False)
            viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        except: pass
            
        plotter = MissingDivPlotter()
        viewer.window.add_dock_widget(plotter, area="top", name="Detection Plots")
        
        # --- Config Controls ---
        cfg_prom = config.get('prominence', {'min': 0.0, 'max': 0.2, 'step': 0.005, 'default': 0.035})
        cfg_dist = config.get('min_dist',   {'min': 0,   'max': 60,  'step': 1,     'default': 15})
        cfg_win = config.get('division_timepoint_window',{'min': 0, 'max': 30, 'step': 3, 'default': 12})
        cfg_look = config.get('division_peak_lookback', {'min': 1, 'max': 10, 'step': 1, 'default': 2})
        
        cfg_drop_min = config.get('drop_min', {'min': 0.05, 'max': 2.0, 'step': 0.05, 'default': 0.1})
        cfg_drop_max = config.get('drop_max', {'min': 0.5, 'max': 5.0, 'step': 0.1, 'default': 3.0})

        # Row 1
        w_prom = FloatSlider(min=cfg_prom['min'], max=cfg_prom['max'], step=cfg_prom['step'], value=cfg_prom['default'], label="Prominence Threshold")
        w_dist = IntSlider(min=cfg_dist['min'], max=cfg_dist['max'], step=cfg_dist['step'], value=cfg_dist['default'], label="Min Dist from Existing (min)")
        
        # Row 2 (Drop Size)
        w_drop_min = FloatSlider(min=cfg_drop_min['min'], max=cfg_drop_min['max'], step=cfg_drop_min['step'], value=cfg_drop_min['default'], label="Min Drop Size")
        w_drop_max = FloatSlider(min=cfg_drop_max['min'], max=cfg_drop_max['max'], step=cfg_drop_max['step'], value=cfg_drop_max['default'], label="Max Drop Size")

        # Row 3 (Snap / Window / Lookback)
        w_snap_on = CheckBox(value=True, label="Enable Snap")
        w_div_win = IntSlider(min=cfg_win['min'], max=cfg_win['max'], step=cfg_win['step'], value=cfg_win['default'], label="Division Timepoint Window (min)")
        w_look = IntSlider(min=cfg_look['min'], max=cfg_look['max'], step=cfg_look['step'], value=cfg_look['default'], label="Peak Lookback (pts)")
        
        _fix_focus_unique(w_prom, decimals=3)
        _fix_focus_unique(w_dist)
        _fix_focus_unique(w_drop_min, decimals=2)
        _fix_focus_unique(w_drop_max, decimals=1)
        _fix_focus_unique(w_div_win)
        _fix_focus_unique(w_look)
            
        w_save = PushButton(text="Finish and Save Parameters")
        w_save.native.setMinimumHeight(60)
        
        c1 = Container(widgets=[w_prom, w_dist], layout="horizontal", labels=True)
        c2 = Container(widgets=[w_drop_min, w_drop_max], layout="horizontal", labels=True)
        c3 = Container(widgets=[w_snap_on, w_div_win, w_look], layout="horizontal", labels=True)
        main_c = Container(widgets=[c1, c2, c3, Label(value=""), w_save])
        viewer.window.add_dock_widget(main_c, area="bottom", name="Controls")
        
        final_params = {
            'prominence': cfg_prom['default'],
            'min_dist': cfg_dist['default'],
            'drop_min': cfg_drop_min['default'],
            'drop_max': cfg_drop_max['default'],
            'snap_enabled': True,
            'division_timepoint_window': cfg_win['default'],
            'division_peak_lookback': cfg_look['default']
        }

        def update():
            idx = viewer.dims.current_step[0]
            if idx < n_moms:
                mid = candidates[int(idx)]
                trace = df[df["mother_id"] == mid].sort_values("time")
                
                t = trace["time"].values
                y_smooth = trace["log_length_smoothed"].values
                existing = trace["division_event"].fillna(False).values
                
                det_data = detect_missing_divisions(
                    t, y_smooth, existing,
                    filter_prominence=w_prom.value,
                    min_dist=w_dist.value,
                    drop_min=w_drop_min.value,
                    drop_max=w_drop_max.value,
                    division_peak_lookback=w_look.value,
                    snap_enabled=w_snap_on.value,
                    div_timepoint_win=w_div_win.value,
                    search_prominence=0.001 
                )
                
                plotter.update_plot(
                    mid, t, y_smooth, existing, det_data, 
                    w_prom.value, global_max_time
                )
                viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

        for w in [w_prom, w_dist, w_drop_min, w_drop_max, w_snap_on, w_div_win, w_look]:
            w.changed.connect(update)
        viewer.dims.events.current_step.connect(lambda e: update())
        
        @w_save.clicked.connect
        def on_save():
            final_params.update({
                'prominence': w_prom.value,
                'min_dist': w_dist.value,
                'drop_min': w_drop_min.value,
                'drop_max': w_drop_max.value,
                'snap_enabled': w_snap_on.value,
                'division_timepoint_window': w_div_win.value,
                'division_peak_lookback': w_look.value
            })
            print(f"Saved Missing Div Params: {final_params}")
            viewer.close()
            
        viewer.dims.set_current_step(0, 0)
        update()
        viewer.window.qt_viewer.window().showMaximized()
        napari.run()
        
        return final_params

def _process_missing_divs(group, params):
    """Worker for finding missing divs."""
    try:
        t = group["time"].values
        y_smooth = group["log_length_smoothed"].values
        existing = group["division_event"].fillna(False).values
        
        # Handle renaming compatibility
        snap_en = params.get('snap_enabled', True)
        win = params.get('division_timepoint_window', params.get('snap_window', 12.0))
        
        det_data = detect_missing_divisions(
            t, y_smooth, existing,
            filter_prominence=params['prominence'],
            min_dist=params['min_dist'],
            drop_min=params.get('drop_min', 0.1),
            drop_max=params.get('drop_max', 3.0),
            division_peak_lookback=params.get('division_peak_lookback', 2),
            snap_enabled=snap_en,
            div_timepoint_win=win,
            search_prominence=params['prominence']
        )
        
        return pd.Series(det_data['new_div_mask'], index=group.index)
    except Exception:
        return None

def execute_batch_missing_divisions(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Detects missing division events and updates the 'division_event' column.
    """
    if df_main is None: return None
    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n FINDING MISSING DIVISIONS (Parallel n_jobs={n_jobs}) \n{'='*40}")
    print(f"  Params: {params}")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    
    if "division_event" not in df_main.columns:
        df_main["division_event"] = False
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_missing_divs)(group, params)
        for _, group in tqdm(grouped, desc="Identifying missing division events", total=len(candidates))
    )
    
    total_added = 0
    for res in results:
        if res is not None and res.any():
            df_main.loc[res.index[res], 'division_event'] = True
            total_added += res.sum()
            
    print(f"Added {total_added} new division events.")
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main
    

# =============================================================================
# Derivative Calculation Logic
# =============================================================================

def _calc_deriv_and_smooth(t, y, window_size):
    """
    Calculates central derivative of y w.r.t t, then smooths with LOWESS.
    """
    # 1. Central Derivative
    deriv_raw = np.gradient(y, t)
    
    # 2. Smooth
    n_points = len(t)
    if n_points > 0:
        frac = min(1.0, max(0.01, window_size / n_points))
    else:
        frac = 0.1
        
    valid = np.isfinite(deriv_raw)
    if valid.sum() > 5:
        smoothed = sm.nonparametric.lowess(deriv_raw[valid], t[valid], frac=frac)
        deriv_smooth = np.interp(t, smoothed[:, 0], smoothed[:, 1])
    else:
        deriv_smooth = deriv_raw
        
    return deriv_raw, deriv_smooth, frac

def apply_derivative_pipeline(df_trace, win_d1=20, win_d2=20, start_flat=0.0):
    """
    Pipeline: 
    1. Flatten log_length_smoothed (to remove division drops).
    2. Calc D1 (Growth Rate) -> Smooth -> Enforce 0 if t <= start_flat.
    3. Calc D2 -> Smooth -> Enforce 0 if t <= start_flat.
    """
    # Sort to ensure time consistency
    df_trace = df_trace.sort_values("time").copy()
    
    # Capture FULL time range (including NaNs)
    t = df_trace["time"].values
    
    if "log_length_smoothed" not in df_trace.columns:
        col = "log_length_fit" if "log_length_fit" in df_trace.columns else "log_length_cleaned"
        y_input = df_trace[col].values
    else:
        y_input = df_trace["log_length_smoothed"].values
    
    y_raw_cleaned = df_trace["log_length_cleaned"].values if "log_length_cleaned" in df_trace.columns else y_input

    divs = df_trace["division_event"].fillna(False).values
    
    # 1. Flatten trace
    try:
        y_flat, _ = flatten_trace_by_divisions(t, y_input, divs)
    except NameError:
        y_flat = y_input 
    
    # 2. First Derivative
    d1_raw, d1_smooth, frac1 = _calc_deriv_and_smooth(t, y_flat, win_d1)
    
    # --- Enforce Flat Start on D1 ---
    if start_flat > 0:
        d1_smooth[t <= start_flat] = 0.0

    # 3. Second Derivative
    # Note: We calculate D2 from the *already enforced* D1, but smoothing might smear the step.
    d2_raw, d2_smooth, frac2 = _calc_deriv_and_smooth(t, d1_smooth, win_d2)
    
    # --- Enforce Flat Start on D2 ---
    if start_flat > 0:
        d2_smooth[t <= start_flat] = 0.0
    
    # Return full 't' array alongside data
    return t, y_input, y_raw_cleaned, d1_raw, d1_smooth, d2_raw, d2_smooth, divs, frac1, frac2
    
# =============================================================================
# Interactive Viewer (Derivatives)
# =============================================================================

if _HAS_GUI:
    class DerivativePlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=10, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            
            # Layout: Three rows (Length, D1, D2)
            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.15)
            self.ax_len = self.fig.add_subplot(gs[0])
            self.ax_d1  = self.fig.add_subplot(gs[1], sharex=self.ax_len)
            self.ax_d2  = self.fig.add_subplot(gs[2], sharex=self.ax_len)
            
            self.fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, y_smooth, y_raw_cleaned, d1_raw, d1_smooth, d2_raw, d2_smooth, divs, win1, win2, global_max_time):
            self.ax_len.clear()
            self.ax_d1.clear()
            self.ax_d2.clear()

            # --- 1. CALCULATE FIXED GLOBAL X-LIMITS ---
            x_max_limit = np.ceil(global_max_time / 120) * 120
            
            # Define ticks
            major_ticks = np.arange(0, x_max_limit + 1, 120)
            minor_ticks = np.arange(0, x_max_limit + 1, 60)

            # --- PREPARE STYLING CONSTANTS ---
            C_RAW = 'gray'
            C_DIV = 'cyan'
            C_FIT_RED = '#d62728' 
            C_SMOOTH_1 = '#1f77b4' 
            C_SMOOTH_2 = '#ff7f0e' 
            
            # Masks for separate plotting
            # We want divs to be ON TOP (zorder 3) and Opaque (alpha 1.0)
            is_div = divs.astype(bool)
            not_div = ~is_div
            
            # --- PLOT 1: Length ---
            # 1. Raw points (non-divs) behind line
            self.ax_len.scatter(t[not_div], y_raw_cleaned[not_div], c=C_RAW, s=10, alpha=0.5, label='Raw Cleaned', zorder=1)
            
            # 2. Smoothed Line (Middle)
            self.ax_len.plot(t, y_smooth, c=C_FIT_RED, lw=2, label='Smoothed Input', zorder=2)
            
            # 3. Div points (Top, Opaque)
            if is_div.any():
                self.ax_len.scatter(t[is_div], y_raw_cleaned[is_div], c=C_DIV, s=20, alpha=1.0, zorder=3, label='Division')
            
            self.ax_len.set_title(f"Mother {mid}: Smoothed Length Trace", fontweight="bold")
            self.ax_len.set_ylabel("log(Length)")
            self.ax_len.legend(loc="upper left", fontsize=8)
            
            # Grid & Ticks
            self.ax_len.set_xticks(major_ticks)
            self.ax_len.set_xticks(minor_ticks, minor=True)
            self.ax_len.grid(True, which='major', ls=':', alpha=0.6)
            self.ax_len.grid(True, which='minor', ls=':', alpha=0.3)
            self.ax_len.tick_params(labelbottom=False)

            # --- PLOT 2: First Derivative ---
            self.ax_d1.scatter(t[not_div], d1_raw[not_div], c=C_RAW, s=10, alpha=0.5, label='Raw d1', zorder=1)
            self.ax_d1.plot(t, d1_smooth, c=C_SMOOTH_1, lw=2, label='Smoothed d1', zorder=2)
            
            if is_div.any():
                self.ax_d1.scatter(t[is_div], d1_raw[is_div], c=C_DIV, s=20, alpha=1.0, zorder=3)
            
            self.ax_d1.set_title(f"1st Derivative (Growth Rate) | Window: {win1} pts", fontweight="bold")
            self.ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            self.ax_d1.set_ylabel("d(log L) / dt\n(x10^-2 1/min)")
            self.ax_d1.set_ylim(0, 0.025) 
            self.ax_d1.legend(loc="upper left", fontsize=8)
            
            # Grid & Ticks
            self.ax_d1.set_xticks(major_ticks)
            self.ax_d1.set_xticks(minor_ticks, minor=True)
            self.ax_d1.grid(True, which='major', ls=':', alpha=0.6)
            self.ax_d1.grid(True, which='minor', ls=':', alpha=0.3)
            self.ax_d1.tick_params(labelbottom=False)
            
            # --- PLOT 3: Second Derivative ---
            self.ax_d2.scatter(t[not_div], d2_raw[not_div], c=C_RAW, s=10, alpha=0.5, label='Raw d2', zorder=1)
            self.ax_d2.plot(t, d2_smooth, c=C_SMOOTH_2, lw=2, label='Smoothed d2', zorder=2)
            
            if is_div.any():
                self.ax_d2.scatter(t[is_div], d2_raw[is_div], c=C_DIV, s=20, alpha=1.0, zorder=3)
            
            self.ax_d2.set_title(f"second Derivative | Window: {win2} pts", fontweight="bold")
            self.ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
            self.ax_d2.set_ylabel("d2(log L) / dt2\n(x10^-4 1/min2)")
            self.ax_d2.set_ylim(-0.00025, 0.00025) 
            self.ax_d2.set_xlabel("Time (min)")
            self.ax_d2.legend(loc="upper left", fontsize=8)
            
            # Grid & Ticks (Labels visible here)
            self.ax_d2.set_xticks(major_ticks)
            self.ax_d2.set_xticks(minor_ticks, minor=True)
            self.ax_d2.grid(True, which='major', ls=':', alpha=0.6)
            self.ax_d2.grid(True, which='minor', ls=':', alpha=0.3)
            
            # Apply X-Limit LAST to ensure it holds
            self.ax_len.set_xlim(0, x_max_limit)
            
            self.draw()

    # --- FOCUS FIX HELPER ---
    def fix_widget_focus(mg_widget):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox
            native = mg_widget.native
            # Force sliders to be focusable via keyboard
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            # Downgrade focus priority of spinboxes/text fields
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
        except Exception:
            pass

    def run_interactive_derivative_smoothing(df: pd.DataFrame, config: dict):
        if not _HAS_GUI:
            print("Napari/GUI not available.")
            return None

        if "at_start" in df.columns:
            candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
        else:
            candidates = sorted(df["mother_id"].unique())
        
        n_moms = len(candidates)
        if n_moms == 0:
            print("No candidates found.")
            return None
            
        if "log_length_smoothed" not in df.columns:
            if "log_length_fit" in df.columns: pass
            else:
                 print("Error: No smoothed length column found.")
                 return None

        # --- CALCULATE GLOBAL MAX TIME ---
        global_max_time = df["time"].max() if "time" in df.columns else 600

        # --- GET START_FLAT FROM CONFIG ---
        # Assuming config (e.g. DERIVATIVE_CONFIG) contains 'start_flat' key 
        # or the user manually merged it.
        sf_param = config.get('start_flat', 0.0)
        if isinstance(sf_param, dict):
             start_flat_val = sf_param.get('default', 0.0)
        else:
             start_flat_val = float(sf_param)

        print(f"Loaded {n_moms} traces. Start Flat enforced at t <= {start_flat_val} min.")

        viewer = napari.Viewer(title="Derivative Smoothing Testing")
        viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
        
        try:
            viewer.window.qt_viewer.dockLayerList.setVisible(False)
            viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        except Exception:
            pass
            
        plotter = DerivativePlotter()
        viewer.window.add_dock_widget(plotter, area="top", name="Deriv Plots")
        
        # --- Controls ---
        p_d1 = config.get('lowess_window_first_deriv', {'min': 5, 'max': 100, 'step': 1, 'default': 20})
        p_d2 = config.get('lowess_window_second_deriv', {'min': 5, 'max': 100, 'step': 1, 'default': 20})
        
        w_win_d1 = IntSlider(
            min=p_d1['min'], max=p_d1['max'], step=p_d1['step'], 
            value=p_d1['default'], label="LOWESS Window 1st Deriv (pts)"
        )
        w_win_d2 = IntSlider(
            min=p_d2['min'], max=p_d2['max'], step=p_d2['step'], 
            value=p_d2['default'], label="LOWESS Window second Deriv (pts)"
        )
        
        fix_widget_focus(w_win_d1)
        fix_widget_focus(w_win_d2)
        
        w_save = PushButton(text="Finish and Save Parameters")
        w_save.native.setMinimumHeight(60)
        
        c1 = Container(widgets=[w_win_d1], layout="horizontal", labels=True)
        c2 = Container(widgets=[w_win_d2], layout="horizontal", labels=True)
        main_c = Container(widgets=[c1, c2, Label(value=""), w_save])
        
        viewer.window.add_dock_widget(main_c, area="bottom", name="Controls")
        
        final_params = {
            'lowess_window_first_deriv': p_d1['default'],
            'lowess_window_second_deriv': p_d2['default'],
            'start_flat': start_flat_val # Pass this through
        }

        def update():
            idx = viewer.dims.current_step[0]
            if idx < n_moms:
                mid = candidates[int(idx)]
                trace = df[df["mother_id"] == mid]
                
                win1 = w_win_d1.value
                win2 = w_win_d2.value
                
                t, y_input, y_raw_c, d1_raw, d1_smooth, d2_raw, d2_smooth, divs, _, _ = apply_derivative_pipeline(
                    trace, win_d1=win1, win_d2=win2, start_flat=start_flat_val
                )
                
                plotter.update_plot(mid, t, y_input, y_raw_c, d1_raw, d1_smooth, d2_raw, d2_smooth, divs, win1, win2, global_max_time)
                
                # Visualise start_flat cutoff
                if start_flat_val > 0:
                    for ax in [plotter.ax_d1, plotter.ax_d2]:
                        ax.axvline(start_flat_val, color='green', linestyle='--', alpha=0.5)
                        # Re-draw happens inside update_plot usually, but calling draw() again is safe
                    plotter.draw()
                    
                viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

        w_win_d1.changed.connect(update)
        w_win_d2.changed.connect(update)
        viewer.dims.events.current_step.connect(lambda e: update())
        
        @w_save.clicked.connect
        def on_save():
            final_params['lowess_window_first_deriv'] = w_win_d1.value
            final_params['lowess_window_second_deriv'] = w_win_d2.value
            print(f"Saved Deriv Params: {final_params}")
            viewer.close()
            
        viewer.dims.set_current_step(0, 0)
        update()
        viewer.window.qt_viewer.window().showMaximized()
        napari.run()
        
        return final_params
        

def _process_deriv_trace(group, win_d1, win_d2, start_flat):
    """Worker for derivative calculation."""
    try:
        _, _, _, _, d1_smooth, _, d2_smooth, _, _, _ = apply_derivative_pipeline(
            group, win_d1=win_d1, win_d2=win_d2, start_flat=start_flat
        )
        return pd.DataFrame({
            'growth_rate': d1_smooth,
            'second_deriv': d2_smooth
        }, index=group.index)
    except Exception:
        return None
        
def execute_batch_derivative_smoothing(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculates smoothed 1st and second derivatives for all 'at_start' mothers.
    Adds columns: 'growth_rate', 'second_deriv'.
    """
    if df_main is None: return None
    
    win_d1 = params.get('lowess_window_first_deriv', 20)
    win_d2 = params.get('lowess_window_second_deriv', 20)
    start_flat = params.get('start_flat', 0.0) # Extract param
    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"{'='*40}\n STARTING BATCH DERIVATIVES (Parallel n_jobs={n_jobs}) \n{'='*40}")
    print(f"  Window 1st Deriv: {win_d1} pts")
    print(f"  Window second Deriv: {win_d2} pts")
    print(f"  Start Flat: {start_flat} mins")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    df_main['growth_rate'] = np.nan
    df_main['second_deriv'] = np.nan
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_deriv_trace)(group, win_d1, win_d2, start_flat) # Pass param
        for _, group in tqdm(grouped, desc="Calculating derivatives", total=len(candidates))
    )
    
    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        print("Concatenating results...")
        combined = pd.concat(valid_results)
        df_main.loc[combined.index, ['growth_rate', 'second_deriv']] = combined
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main


# =============================================================================
# Derivative Cleanup Logic
# =============================================================================

def _clean_and_smooth(t, raw_data, min_thresh, max_thresh, adj_points, window_size):
    """
    1. Identify points < min_thresh OR > max_thresh.
    2. Dilate mask by adj_points.
    3. Set to NaN.
    4. Smooth remaining data with LOWESS.
    """
    data_clean = raw_data.copy()
    bad_mask = np.zeros(len(data_clean), dtype=bool)
    
    # 1. Identify bad points
    if min_thresh is not None:
        bad_mask |= (data_clean < min_thresh)
        
    if max_thresh is not None:
        bad_mask |= (data_clean > max_thresh)
        
    # 2. Dilate (add adjacent points)
    if adj_points > 0 and bad_mask.any():
        structure = np.ones(2 * adj_points + 1, dtype=bool) 
        bad_mask = binary_dilation(bad_mask, structure=structure)
            
    # 3. Apply NaN
    data_clean[bad_mask] = np.nan
        
    # 4. Smooth
    n_points = len(t)
    frac = min(1.0, max(0.01, window_size / n_points)) if n_points > 0 else 0.1
    
    valid = np.isfinite(data_clean)
    if valid.sum() > 5:
        smoothed = sm.nonparametric.lowess(data_clean[valid], t[valid], frac=frac)
        data_smooth = np.interp(t, smoothed[:, 0], smoothed[:, 1])
    else:
        data_smooth = data_clean 
        
    return data_clean, data_smooth

def apply_derivative_cleanup_pipeline(df_trace, 
                                      d1_min=0.0, d1_max=None, d1_adj=1, 
                                      d2_min=-0.00025, d2_max=None, d2_adj=1,
                                      win_d1=20, win_d2=20, start_flat=0.0):
    """
    Pipeline: Flatten -> Calc D1 -> Clean D1 (Min/Max) -> Smooth D1 -> Calc D2 -> Clean D2 (Min/Max) -> Smooth D2.
    Added: start_flat (enforces 0 in [0, start_flat] for smooth derivatives).
    """
    df_trace = df_trace.sort_values("time").copy()
    t = df_trace["time"].values
    
    # 1. Get Inputs
    if "log_length_smoothed" not in df_trace.columns:
        col = "log_length_fit" if "log_length_fit" in df_trace.columns else "log_length_cleaned"
        y_input = df_trace[col].values
    else:
        y_input = df_trace["log_length_smoothed"].values
        
    y_raw_cleaned = df_trace["log_length_cleaned"].values if "log_length_cleaned" in df_trace.columns else y_input
    divs = df_trace["division_event"].fillna(False).values
    
    # 2. Flatten for Derivative Calc
    try:
        y_flat, shifts = flatten_trace_by_divisions(t, y_input, divs)
    except NameError:
        y_flat = y_input; shifts = []

    # 3. D1: Calc -> Clean -> Smooth
    d1_raw_initial = np.gradient(y_flat, t)
    d1_clean, d1_smooth = _clean_and_smooth(t, d1_raw_initial, d1_min, d1_max, d1_adj, win_d1)
    
    # --- Enforce Flat Start D1 ---
    if start_flat > 0:
        d1_smooth[t <= start_flat] = 0.0
    
    # 4. D2: Calc (from D1 Smooth) -> Clean -> Smooth
    d2_raw_initial = np.gradient(d1_smooth, t)
    d2_clean, d2_smooth = _clean_and_smooth(t, d2_raw_initial, d2_min, d2_max, d2_adj, win_d2)
    
    # --- Enforce Flat Start D2 ---
    if start_flat > 0:
        d2_smooth[t <= start_flat] = 0.0
    
    # 5. Generate Combined Outlier Mask
    mask = np.isnan(d1_clean) | np.isnan(d2_clean)
    
    return (t, y_input, y_raw_cleaned,             # 0, 1, 2
            d1_raw_initial, d1_clean, d1_smooth,   # 3, 4, 5
            d2_raw_initial, d2_clean, d2_smooth,   # 6, 7, 8
            divs, mask)                            # 9, 10

# =============================================================================
# Interactive Viewer (Cleanup)
# =============================================================================

if _HAS_GUI:
    class CleanupPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=10, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.15)
            self.ax_len = self.fig.add_subplot(gs[0])
            self.ax_d1  = self.fig.add_subplot(gs[1], sharex=self.ax_len)
            self.ax_d2  = self.fig.add_subplot(gs[2], sharex=self.ax_len)
            self.fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, y_smooth, y_raw, d1_raw, d1_clean, d1_smooth, d2_raw, d2_clean, d2_smooth, divs, global_max_time, start_flat):
            self.ax_len.clear(); self.ax_d1.clear(); self.ax_d2.clear()

            x_max_limit = np.ceil(global_max_time / 120) * 120
            major_ticks = np.arange(0, x_max_limit + 1, 120)
            minor_ticks = np.arange(0, x_max_limit + 1, 60)

            C_RAW = 'gray'
            C_EXCLUDED = 'red'
            C_DIV = 'cyan'
            C_FIT_RED = '#d62728' 
            C_SMOOTH_1 = '#1f77b4' 
            C_SMOOTH_2 = '#ff7f0e' 
            
            # --- Plot 1: Length ---
            point_colors = np.where(divs, C_DIV, C_RAW)
            point_sizes  = np.where(divs, 10, 10)
            self.ax_len.scatter(t, y_raw, c=point_colors, s=point_sizes, alpha=0.5, label='Raw')
            self.ax_len.plot(t, y_smooth, c=C_FIT_RED, lw=2, label='Smoothed Input')
            
            if divs.any():
                self.ax_len.scatter(t[divs], y_raw[divs], c=C_DIV, s=20, alpha=1.0, zorder=10)

            self.ax_len.set_title(f"Mother {mid}: Smoothed Length", fontweight="bold")
            self.ax_len.set_ylabel("log(Length)")
            self.ax_len.legend(loc="upper left", fontsize=8)
            
            # --- Plot 2: First Derivative ---
            excluded_mask = np.isnan(d1_clean) & np.isfinite(d1_raw)
            kept_mask = np.isfinite(d1_clean)
            
            self.ax_d1.scatter(t[kept_mask], d1_raw[kept_mask], c=C_RAW, s=10, alpha=0.4, label='Kept')
            self.ax_d1.scatter(t[excluded_mask], d1_raw[excluded_mask], c=C_EXCLUDED, marker='x', s=20, alpha=0.8, label='Removed')
            self.ax_d1.plot(t, d1_smooth, c=C_SMOOTH_1, lw=2, label='Re-Smoothed')
            
            if divs.any():
                self.ax_d1.scatter(t[divs], d1_raw[divs], c=C_DIV, s=20, alpha=1.0, zorder=10)

            self.ax_d1.set_title("1st Derivative Cleanup", fontweight="bold")
            self.ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            self.ax_d1.set_ylabel("d(log L)/dt\n(x10^-2 1/min)")
            self.ax_d1.legend(loc="upper left", fontsize=8)

            # --- Plot 3: Second Derivative ---
            excluded_mask_d2 = np.isnan(d2_clean) & np.isfinite(d2_raw)
            kept_mask_d2 = np.isfinite(d2_clean)
            
            self.ax_d2.scatter(t[kept_mask_d2], d2_raw[kept_mask_d2], c=C_RAW, s=10, alpha=0.4, label='Kept')
            self.ax_d2.scatter(t[excluded_mask_d2], d2_raw[excluded_mask_d2], c=C_EXCLUDED, marker='x', s=20, alpha=0.8, label='Removed')
            self.ax_d2.plot(t, d2_smooth, c=C_SMOOTH_2, lw=2, label='Re-Smoothed')
            
            if divs.any():
                self.ax_d2.scatter(t[divs], d2_raw[divs], c=C_DIV, s=20, alpha=1.0, zorder=10)

            self.ax_d2.set_title("Second Derivative Cleanup", fontweight="bold")
            self.ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
            self.ax_d2.set_ylabel("d2/dt2\n(x10^-4 1/min2)")
            self.ax_d2.set_xlabel("Time (min)")
            self.ax_d2.legend(loc="upper left", fontsize=8)

            # --- Visualise Start Flat ---
            if start_flat > 0:
                for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                    ax.axvline(start_flat, color='green', linestyle='--', alpha=0.5, label='Start Flat')

            for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.grid(True, which='major', ls=':', alpha=0.6)
                ax.grid(True, which='minor', ls=':', alpha=0.3)
                if ax != self.ax_d2: ax.tick_params(labelbottom=False)
            
            self.ax_len.set_xlim(0, x_max_limit)
            self.draw()

    def run_interactive_derivative_cleanup(df: pd.DataFrame, config: dict, prev_deriv_params: dict):
        if not _HAS_GUI: return None
    
        if "at_start" in df.columns:
            candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
        else:
            candidates = sorted(df["mother_id"].unique())
        
        n_moms = len(candidates)
        if n_moms == 0: return None
        
        win_d1 = prev_deriv_params.get('lowess_window_first_deriv', 20)
        win_d2 = prev_deriv_params.get('lowess_window_second_deriv', 20)
        
        # --- EXTRACT START_FLAT ---
        sf_param = prev_deriv_params.get('start_flat', 0.0)
        if isinstance(sf_param, dict):
             start_flat_val = sf_param.get('default', 0.0)
        else:
             start_flat_val = float(sf_param)
        
        global_max_time = df["time"].max() if "time" in df.columns else 600
    
        viewer = napari.Viewer(title="Derivative Cleanup Tuner")
        viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
        try:
            viewer.window.qt_viewer.dockLayerList.setVisible(False)
            viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        except: pass
            
        plotter = CleanupPlotter()
        viewer.window.add_dock_widget(plotter, area="top", name="Cleanup Plots")
        
        # --- Configure Controls ---
        cfg_d1_min = config.get('min_d1', {'min': -5.0, 'max': 5.0, 'step': 0.1, 'default': 0.0})
        cfg_d2_min = config.get('min_d2', {'min': -5.0, 'max': 5.0, 'step': 0.1, 'default': -2.5})
        cfg_d1_max = config.get('max_d1', {'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 0.4})
        cfg_d2_max = config.get('max_d2', {'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 0.4})
        cfg_adj1 = config.get('adj_d1', {'min': 0, 'max': 5, 'step': 1, 'default': 1})
        cfg_adj2 = config.get('adj_d2', {'min': 0, 'max': 5, 'step': 1, 'default': 1})
    
        # Helpers
        def fix_widget_focus(mg_widget):
            try:
                from qtpy.QtCore import Qt
                from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox
                native = mg_widget.native
                for slider in native.findChildren(QAbstractSlider): slider.setFocusPolicy(Qt.StrongFocus)
                for spinner in native.findChildren(QAbstractSpinBox): spinner.setFocusPolicy(Qt.ClickFocus)
            except Exception: pass

        # Widgets
        w_use_d1 = CheckBox(value=True, label="Enable")
        w_min_d1 = FloatSlider(min=cfg_d1_min['min'], max=cfg_d1_min['max'], step=cfg_d1_min['step'], value=cfg_d1_min['default'], label="Min 1st Deriv (x10^-2)")
        w_max_d1 = FloatSlider(min=cfg_d1_max['min'], max=cfg_d1_max['max'], step=cfg_d1_max['step'], value=cfg_d1_max['default'], label="Max 1st Deriv (x10^-2)")
        w_adj_d1 = IntSlider(min=cfg_adj1['min'], max=cfg_adj1['max'], step=cfg_adj1['step'], value=cfg_adj1['default'], label="Del Adjacent (pts)")
        
        w_use_d2 = CheckBox(value=True, label="Enable")
        w_min_d2 = FloatSlider(min=cfg_d2_min['min'], max=cfg_d2_min['max'], step=cfg_d2_min['step'], value=cfg_d2_min['default'], label="Min 2nd Deriv (x10^-4)")
        w_max_d2 = FloatSlider(min=cfg_d2_max['min'], max=cfg_d2_max['max'], step=cfg_d2_max['step'], value=cfg_d2_max['default'], label="Max 2nd Deriv (x10^-4)")
        w_adj_d2 = IntSlider(min=cfg_adj2['min'], max=cfg_adj2['max'], step=cfg_adj2['step'], value=cfg_adj2['default'], label="Del Adjacent (pts)")
        
        for w in [w_min_d1, w_max_d1, w_adj_d1, w_min_d2, w_max_d2, w_adj_d2]: fix_widget_focus(w)
            
        w_save = PushButton(text="Finish and Save Parameters")
        w_save.native.setMinimumHeight(60)
        
        c1_a = Container(widgets=[w_use_d1, w_min_d1, w_adj_d1], layout="horizontal", labels=True)
        c1_b = Container(widgets=[Label(value=""), w_max_d1, Label(value="")], layout="horizontal", labels=True)
        c2_a = Container(widgets=[w_use_d2, w_min_d2, w_adj_d2], layout="horizontal", labels=True)
        c2_b = Container(widgets=[Label(value=""), w_max_d2, Label(value="")], layout="horizontal", labels=True)
        
        main_c = Container(widgets=[c1_a, c1_b, c2_a, c2_b, Label(value=""), w_save])
        viewer.window.add_dock_widget(main_c, area="bottom", name="Controls")
        
        final_params = {
            'use_d1': True, 
            'min_d1': cfg_d1_min['default'] * 1e-2, 
            'max_d1': cfg_d1_max['default'] * 1e-2,
            'adj_d1': cfg_adj1['default'],
            'use_d2': True, 
            'min_d2': cfg_d2_min['default'] * 1e-4, 
            'max_d2': cfg_d2_max['default'] * 1e-4,
            'adj_d2': cfg_adj2['default'],
            'start_flat': start_flat_val # Pass through
        }
    
        def update():
            idx = viewer.dims.current_step[0]
            if idx < n_moms:
                mid = candidates[int(idx)]
                trace = df[df["mother_id"] == mid]
                
                d1_min_val = (w_min_d1.value * 1e-2) if w_use_d1.value else None
                d1_max_val = (w_max_d1.value * 1e-2) if w_use_d1.value else None
                d2_min_val = (w_min_d2.value * 1e-4) if w_use_d2.value else None
                d2_max_val = (w_max_d2.value * 1e-4) if w_use_d2.value else None
                
                res = apply_derivative_cleanup_pipeline(
                    trace, 
                    d1_min=d1_min_val, d1_max=d1_max_val, d1_adj=w_adj_d1.value,
                    d2_min=d2_min_val, d2_max=d2_max_val, d2_adj=w_adj_d2.value,
                    win_d1=win_d1, win_d2=win_d2,
                    start_flat=start_flat_val # Pass updated param
                )
                
                t, y_in, y_raw, d1_r, d1_c, d1_s, d2_r, d2_c, d2_s, divs, _ = res
                plotter.update_plot(mid, t, y_in, y_raw, d1_r, d1_c, d1_s, d2_r, d2_c, d2_s, divs, global_max_time, start_flat_val)
                viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"
    
        for w in [w_use_d1, w_min_d1, w_max_d1, w_adj_d1, w_use_d2, w_min_d2, w_max_d2, w_adj_d2]:
            w.changed.connect(update)
        viewer.dims.events.current_step.connect(lambda e: update())
        
        @w_save.clicked.connect
        def on_save():
            final_params.update({
                'use_d1': w_use_d1.value,
                'min_d1': w_min_d1.value * 1e-2, 
                'max_d1': w_max_d1.value * 1e-2,
                'adj_d1': w_adj_d1.value,
                'use_d2': w_use_d2.value,
                'min_d2': w_min_d2.value * 1e-4, 
                'max_d2': w_max_d2.value * 1e-4,
                'adj_d2': w_adj_d2.value
            })
            print(f"Saved Cleanup Params: {final_params}")
            viewer.close()
            
        viewer.dims.set_current_step(0, 0)
        update()
        viewer.window.qt_viewer.window().showMaximized()
        napari.run()
        
        return final_params
        

# =============================================================================
# Batch Cleanup Execution (Parallel)
# =============================================================================

def _process_cleanup_trace(group, params, prev_deriv_params):
    """Worker for cleanup."""
    try:
        win_d1 = prev_deriv_params.get('lowess_window_first_deriv', 20)
        win_d2 = prev_deriv_params.get('lowess_window_second_deriv', 20)
        start_flat = prev_deriv_params.get('start_flat', 0.0)
        if isinstance(start_flat, dict): start_flat = start_flat.get('default', 0.0)
        
        # Extract Min/Max params
        d1_min = params['min_d1'] if params.get('use_d1', True) else None
        d1_max = params['max_d1'] if params.get('use_d1', True) else None
        d2_min = params['min_d2'] if params.get('use_d2', True) else None
        d2_max = params['max_d2'] if params.get('use_d2', True) else None
        
        # This pipeline ALREADY re-calculates the smoothing on valid points only
        res = apply_derivative_cleanup_pipeline(
            group,
            d1_min=d1_min, d1_max=d1_max, d1_adj=params['adj_d1'],
            d2_min=d2_min, d2_max=d2_max, d2_adj=params['adj_d2'],
            win_d1=win_d1, win_d2=win_d2,
            start_flat=start_flat
        )
        
        d1_raw, d1_s = res[3], res[5]
        d2_raw, d2_s = res[6], res[8]
        mask = res[10]
        
        d1_raw_filtered = d1_raw.copy()
        d2_raw_filtered = d2_raw.copy()
        
        # --- MODIFIED: Apply mask ONLY to Raw traces ---
        if mask.any():
            d1_raw_filtered[mask] = np.nan
            d2_raw_filtered[mask] = np.nan
                    
        return pd.DataFrame({
            'growth_rate': d1_raw_filtered,
            'second_deriv': d2_raw_filtered,
            'growth_rate_smoothed': d1_s,  
            'second_deriv_smoothed': d2_s
        }, index=group.index)
    except Exception:
        return None

def execute_batch_derivative_cleanup(df_main: pd.DataFrame, cleanup_params: dict, prev_deriv_params: dict) -> pd.DataFrame:
    """
    Applies derivative cleanup & re-smoothing.
    """
    if df_main is None: return None
    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n STARTING BATCH CLEANUP (Parallel n_jobs={n_jobs}) \n{'='*40}")
    print(f"  Cleanup Params: {cleanup_params}")
    
    # Ensure start_flat is recognized for logging
    sf = prev_deriv_params.get('start_flat', 0.0)
    if isinstance(sf, dict): sf = sf.get('default', 0.0)
    print(f"  Start Flat: {sf} mins")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    
    cols_to_update = ['growth_rate', 'second_deriv', 'growth_rate_smoothed', 'second_deriv_smoothed']
    for col in cols_to_update:
        if col not in df_main.columns:
            df_main[col] = np.nan
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_cleanup_trace)(group, cleanup_params, prev_deriv_params)
        for _, group in tqdm(grouped, desc="Cleaning derivatives", total=len(candidates))
    )
    
    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        print("Concatenating results...")
        combined = pd.concat(valid_results)
        df_main.loc[combined.index, cols_to_update] = combined[cols_to_update]
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main
    
    
# =============================================================================
# Final Timepoint Check Viewer
# =============================================================================

def _calculate_iterative_nan_cutoff(t, is_nan_mask, max_nan_pct, start_win_len, fold_decrease):
    """
    Iteratively finds the earliest failure point.
    1. Scans with 'start_win_len' to find first window failing max_nan_pct.
    2. Zooms into that window, shrinks window length by 'fold_decrease', and repeats.
    3. Stops when window length < avg time interval (single point).
    
    Constraint: Windows must strictly fit within the experiment duration (t_max).
    """
    n = len(t)
    if n == 0: return None, []
    
    # Global max time (to prevent checking truncated windows at the end)
    t_max = t[-1]
    
    # Calculate approx time resolution
    if n > 1:
        min_dt = np.min(np.diff(t))
        if min_dt <= 0: min_dt = 3.0
    else:
        min_dt = 3.0
        
    threshold = max_nan_pct / 100.0
    
    current_win_len = start_win_len
    
    # Initial Search Range: Whole Trace
    search_start_idx = 0
    search_end_idx = n
    
    final_cutoff = None
    identified_windows = []
    
    while current_win_len >= min_dt:
        found_idx = None
        
        # Scan within the current restricted search range
        for i in range(search_start_idx, search_end_idx):
            t_start = t[i]
            t_end_win = t_start + current_win_len
            
            # --- CRITICAL CHECK: Window must fit in remaining time ---
            if t_end_win > t_max:
                # If the sliding window extends past the end of the experiment, 
                # we cannot evaluate the full window, so we skip it.
                # Since t is sorted, subsequent 'i' will also fail, so we can break loop for speed.
                break 

            # Find end index of this rolling window
            j_end = np.searchsorted(t, t_end_win, side='right')
            
            # Slice Mask
            window_mask = is_nan_mask[i:j_end]
            if len(window_mask) == 0: continue
            
            if np.mean(window_mask) >= threshold:
                found_idx = i
                break # Stop at the FIRST failure in this range
        
        if found_idx is not None:
            # We found a failure window!
            t_found = t[found_idx]
            final_cutoff = t_found
            
            # Record the window for plotting
            identified_windows.append((t_found, t_found + current_win_len))
            
            # UPDATE PARAMETERS FOR NEXT ITERATION
            
            # 1. Restrict Search Range to INSIDE the window we just found.
            search_start_idx = found_idx
            search_end_idx = np.searchsorted(t, t_found + current_win_len, side='right')
            
            # 2. Shrink Window
            current_win_len /= fold_decrease
            
        else:
            # If we didn't find a denser sub-region (or ran out of valid windows), stop.
            break
            
    return final_cutoff, identified_windows

if _HAS_GUI:
    class FinalTimepointCheckPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=10, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            
            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            self.ax_len = self.fig.add_subplot(gs[0])
            self.ax_d1  = self.fig.add_subplot(gs[1], sharex=self.ax_len)
            self.ax_d2  = self.fig.add_subplot(gs[2], sharex=self.ax_len)
            
            self.fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08)
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, 
                        y_raw, y_smooth,
                        d1_raw, d1_smooth,
                        d2_raw, d2_smooth,
                        divs, cutoff_time, windows, global_max_time):
            
            self.ax_len.clear()
            self.ax_d1.clear()
            self.ax_d2.clear()

            # --- Limits & Ticks ---
            x_max_limit = np.ceil(global_max_time / 120) * 120
            major_ticks = np.arange(0, x_max_limit + 1, 120)
            minor_ticks = np.arange(0, x_max_limit + 1, 60)

            # --- Styles ---
            C_RAW = 'gray'
            C_DIV = 'cyan'
            C_LEN_LINE = '#d62728' 
            C_D1_LINE  = '#1f77b4' 
            C_D2_LINE  = '#ff7f0e' 
            C_CUTOFF   = 'black'
            C_SHADE    = 'gray'
            
            # Masks
            is_div = divs.astype(bool)
            
            # === Plot 1: Length ===
            if y_raw is not None:
                self.ax_len.scatter(t, y_raw, c=C_RAW, s=10, alpha=0.5, label='Cleaned Raw', zorder=1)
            if y_smooth is not None:
                self.ax_len.plot(t, y_smooth, c=C_LEN_LINE, lw=2, label='Smoothed', zorder=2)
            if is_div.any() and y_raw is not None:
                valid_divs = is_div & np.isfinite(y_raw)
                if valid_divs.any():
                    self.ax_len.scatter(t[valid_divs], y_raw[valid_divs], c=C_DIV, s=25, alpha=1.0, zorder=3, label='Division')
            
            self.ax_len.set_title(f"Mother {mid}: log(Length)", fontweight="bold")
            self.ax_len.set_ylabel(r"ln[Cell length ($\mu$m)]")
            self.ax_len.legend(loc="upper left", fontsize=8)

            # === Plot 2: Growth Rate ===
            if d1_raw is not None:
                self.ax_d1.scatter(t, d1_raw, c=C_RAW, s=10, alpha=0.5, label='Raw D1', zorder=1)
            if d1_smooth is not None:
                self.ax_d1.plot(t, d1_smooth, c=C_D1_LINE, lw=2, label='Smoothed D1', zorder=2)
            if is_div.any() and d1_raw is not None:
                valid_divs = is_div & np.isfinite(d1_raw)
                if valid_divs.any():
                    self.ax_d1.scatter(t[valid_divs], d1_raw[valid_divs], c=C_DIV, s=25, alpha=1.0, zorder=3)

            self.ax_d1.set_title("Growth Rate (1st Deriv)", fontweight="bold")
            self.ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            self.ax_d1.set_ylabel(r"d(ln L)/dt ($\times 10^{-2}$ min$^{-1}$)")
            self.ax_d1.legend(loc="upper left", fontsize=8)

            # === Plot 3: Acceleration ===
            if d2_raw is not None:
                self.ax_d2.scatter(t, d2_raw, c=C_RAW, s=10, alpha=0.5, label='Raw D2', zorder=1)
            if d2_smooth is not None:
                self.ax_d2.plot(t, d2_smooth, c=C_D2_LINE, lw=2, label='Smoothed D2', zorder=2)
            if is_div.any() and d2_raw is not None:
                valid_divs = is_div & np.isfinite(d2_raw)
                if valid_divs.any():
                    self.ax_d2.scatter(t[valid_divs], d2_raw[valid_divs], c=C_DIV, s=25, alpha=1.0, zorder=3)

            self.ax_d2.set_title("Acceleration (2nd Deriv)", fontweight="bold")
            self.ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
            self.ax_d2.set_ylabel(r"d$^2$(ln L)/dt$^2$ ($\times 10^{-4}$ min$^{-2}$)")
            self.ax_d2.set_xlabel("Time (min)")
            self.ax_d2.legend(loc="upper left", fontsize=8)

            # --- Plot Iterative Windows ---
            # Shading logic: darkest where all overlap
            for (w_start, w_end) in windows:
                for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                    ax.axvspan(w_start, w_end, color=C_SHADE, alpha=0.15)

            # --- Final Cutoff Line ---
            if cutoff_time is not None:
                for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                    ax.axvline(cutoff_time, color=C_CUTOFF, linestyle='--', linewidth=2, alpha=0.8, label='Final Timepoint')

            # --- Common Formatting ---
            for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.grid(True, which='major', ls=':', alpha=0.6)
                ax.grid(True, which='minor', ls=':', alpha=0.3)
                ax.set_xlim(0, x_max_limit)
                if ax != self.ax_d2:
                    ax.tick_params(labelbottom=False)
            
            self.draw()

    def _fix_focus_check(mg_widget, decimals=None):
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractSlider, QAbstractSpinBox, QDoubleSpinBox
            native = mg_widget.native
            for slider in native.findChildren(QAbstractSlider):
                slider.setFocusPolicy(Qt.StrongFocus)
            for spinner in native.findChildren(QAbstractSpinBox):
                spinner.setFocusPolicy(Qt.ClickFocus)
                spinner.setKeyboardTracking(False)
                if hasattr(spinner, 'lineEdit') and spinner.lineEdit():
                    spinner.lineEdit().setFocusPolicy(Qt.ClickFocus)
                if decimals is not None and isinstance(spinner, QDoubleSpinBox):
                    spinner.setDecimals(decimals)
        except Exception:
            pass

    def run_final_timepoint_check_viewer(df: pd.DataFrame, config: dict):
            if not _HAS_GUI: return None
    
            if "at_start" in df.columns:
                candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
            else:
                candidates = sorted(df["mother_id"].unique())
            
            n_moms = len(candidates)
            if n_moms == 0: return None
            
            if "growth_rate" not in df.columns:
                print("Error: 'growth_rate' column missing (needed for NaN check).")
                return None
            
            global_tps = np.sort(df["time"].unique())
            global_max_time = global_tps.max() if len(global_tps) > 0 else 600
    
            viewer = napari.Viewer(title="Final Timepoint Check")
            viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
            try:
                viewer.window.qt_viewer.dockLayerList.setVisible(False)
                viewer.window.qt_viewer.dockLayerControls.setVisible(False)
            except: pass
                
            plotter = FinalTimepointCheckPlotter()
            viewer.window.add_dock_widget(plotter, area="top", name="Final Plots")
            
            # --- Config Controls ---
            p_nan = config.get('max_window_nan_pct', {'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 50.0})
            p_win = config.get('window_length', {'min': 0, 'max': 600, 'step': 3, 'default': 240})
            p_fold = config.get('window_fold_decrease', {'min': 1.1, 'max': 10.0, 'step': 0.1, 'default': 2.0})
    
            # Row 1
            w_nan = FloatSlider(min=p_nan['min'], max=p_nan['max'], step=p_nan['step'], value=p_nan['default'], label="Max NaN %")
            w_win = IntSlider(min=p_win['min'], max=p_win['max'], step=p_win['step'], value=p_win['default'], label="Window (min)")
            
            # Row 2
            w_fold = FloatSlider(min=p_fold['min'], max=p_fold['max'], step=p_fold['step'], value=p_fold['default'], label="Win Fold Decr")
            
            for w in [w_nan, w_win, w_fold]:
                _fix_focus_check(w)
                if isinstance(w, FloatSlider): _fix_focus_check(w, decimals=1)
            
            w_save = PushButton(text="Finish and Save Parameters")
            w_save.native.setMinimumHeight(60)
            
            # Layout
            c1 = Container(widgets=[w_nan, w_win], layout="horizontal", labels=True)
            c2 = Container(widgets=[w_fold, Label(value="")], layout="horizontal", labels=True)
            main_c = Container(widgets=[c1, c2, Label(value=""), w_save])
            
            viewer.window.add_dock_widget(main_c, area="bottom", name="Controls")
            
            final_params = {
                'max_window_nan_pct': p_nan['default'],
                'window_length': p_win['default'],
                'window_fold_decrease': p_fold['default']
            }
    
            def update():
                idx = viewer.dims.current_step[0]
                if idx < n_moms:
                    mid = candidates[int(idx)]
                    
                    # Filter & Reindex
                    trace_orig = df[df["mother_id"] == mid]
                    trace = (trace_orig.drop_duplicates("time")
                                       .set_index("time")
                                       .reindex(global_tps))
                    
                    t = global_tps
                    
                    # --- FIX: Avoid FutureWarning for Downcasting ---
                    # Convert to float (handles NaNs safely), fill with 0, then cast to bool
                    divs = trace["division_event"].astype(float).fillna(0).astype(bool).values
                    
                    y_raw = trace["log_length_cleaned"].values if "log_length_cleaned" in df.columns else None
                    d1_raw = trace["growth_rate"].values
                    col_d2_raw = "second_deriv"
                    d2_raw = trace[col_d2_raw].values if col_d2_raw in df.columns else None
                    
                    y_sm  = trace["log_length_smoothed"].values if "log_length_smoothed" in df.columns else None
                    d1_sm = trace["growth_rate_smoothed"].values if "growth_rate_smoothed" in df.columns else None
                    col_d2_sm = "second_deriv_smoothed"
                    d2_sm = trace[col_d2_sm].values if col_d2_sm in df.columns else None
                    
                    # Combined NaN Logic
                    mask_len = np.isnan(y_raw) if y_raw is not None else np.zeros_like(t, dtype=bool)
                    mask_d1  = np.isnan(d1_raw) if d1_raw is not None else np.zeros_like(t, dtype=bool)
                    mask_d2  = np.isnan(d2_raw) if d2_raw is not None else np.zeros_like(t, dtype=bool)
                    
                    combined_nan = mask_len | mask_d1 | mask_d2
                    
                    # Iterative Calculation
                    cutoff, windows = _calculate_iterative_nan_cutoff(
                        t, combined_nan, 
                        max_nan_pct=w_nan.value,
                        start_win_len=w_win.value,
                        fold_decrease=w_fold.value
                    )
                    
                    plotter.update_plot(
                        mid, t,
                        y_raw, y_sm,
                        d1_raw, d1_sm,
                        d2_raw, d2_sm,
                        divs, cutoff, windows, global_max_time
                    )
                    viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"
    
            for w in [w_nan, w_win, w_fold]:
                w.changed.connect(update)
            viewer.dims.events.current_step.connect(lambda e: update())
            
            @w_save.clicked.connect
            def on_save():
                final_params.update({
                    'max_window_nan_pct': w_nan.value,
                    'window_length': w_win.value,
                    'window_fold_decrease': w_fold.value
                })
                print(f"Saved Final Check Params: {final_params}")
                viewer.close()
            
            viewer.dims.set_current_step(0, 0)
            update()
            viewer.window.qt_viewer.window().showMaximized()
            napari.run()
            
            return final_params

# =============================================================================
# Batch Final Timepoint Check Execution
# =============================================================================

def _process_final_cutoff(mid, group, global_tps, params):
    """
    Worker function to calculate the final cutoff time for a single mother.
    Now explicitly accepts 'mid' (mother_id) as an argument.
    """
    try:
        # 1. Reindex to Global Timepoints (Insert NaNs for gaps)
        trace = (group.drop_duplicates("time")
                      .set_index("time")
                      .reindex(global_tps))
        
        t = global_tps
        
        # 2. Extract Columns for NaN Checking
        y_raw = trace["log_length_cleaned"].values if "log_length_cleaned" in trace.columns else None
        d1_raw = trace["growth_rate"].values
        
        col_d2 = "second_deriv"
        d2_raw = trace[col_d2].values if col_d2 in trace.columns else None
        
        # 3. Create Combined NaN Mask
        mask_len = np.isnan(y_raw) if y_raw is not None else np.zeros_like(t, dtype=bool)
        mask_d1  = np.isnan(d1_raw) if d1_raw is not None else np.zeros_like(t, dtype=bool)
        mask_d2  = np.isnan(d2_raw) if d2_raw is not None else np.zeros_like(t, dtype=bool)
        
        combined_nan = mask_len | mask_d1 | mask_d2
        
        # 4. Calculate Cutoff
        cutoff, _ = _calculate_iterative_nan_cutoff(
            t, combined_nan, 
            max_nan_pct=params['max_window_nan_pct'],
            start_win_len=params['window_length'],
            fold_decrease=params['window_fold_decrease']
        )
        
        return mid, cutoff
        
    except Exception:
        return mid, None

def execute_batch_final_timepoint_check(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies the Final Timepoint Check logic to all mothers.
    """
    if df_main is None: return None
    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n EXECUTING FINAL TIMEPOINT CHECK (Parallel n_jobs={n_jobs}) \n{'='*40}")
    print(f"  Params: {params}")
    
    df_main = df_main.sort_values(["mother_id", "time"]).reset_index(drop=True)
    
    # 1. Establish Global Timepoints
    global_tps = np.sort(df_main["time"].unique())
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    # 2. Calculate Cutoffs in Parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_final_cutoff)(mid, group, global_tps, params)
        for mid, group in tqdm(grouped, desc="Identifying final timepoints", total=len(candidates))
    )
    
    # Convert results to a dictionary {mother_id: cutoff_time}
    cutoff_map = {mid: cutoff for mid, cutoff in results}
    
    # 3. Apply Updates to DataFrame
    print("Applying cutoffs and masking data...")
    
    # Initialize final_time column
    if "final_time" not in df_main.columns:
        df_main["final_time"] = np.nan
        
    # Columns to mask
    cols_to_mask = [
        "log_length_cleaned", "growth_rate", "second_deriv", 
        "log_length_smoothed", "growth_rate_smoothed", "second_deriv_smoothed"
    ]
    # Check actual column existence
    cols_present = [c for c in cols_to_mask if c in df_main.columns]
    cols_present = list(set(cols_present))

    # Map the cutoff times
    # FIX: Convert to numeric immediately to turn Python None into numpy NaN
    # This prevents the "'>=' not supported between 'int' and 'NoneType'" error
    cutoff_series = df_main["mother_id"].map(cutoff_map)
    df_main["final_time"] = pd.to_numeric(cutoff_series, errors='coerce')
    
    # Mask data where time >= final_time
    # Comparison with NaN returns False, so rows with no cutoff are correctly skipped
    mask_condition = (df_main["time"] >= df_main["final_time"]) & (df_main["final_time"].notna())
    
    if mask_condition.any():
        n_masked = mask_condition.sum()
        print(f"Masking {n_masked} timepoints across {len(candidates)} traces.")
        df_main.loc[mask_condition, cols_present] = np.nan
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main


# =============================================================================
# Growth Slowdown Logic
# =============================================================================

def _detect_and_measure_slowdowns(t, d1_smooth, d2_smooth, params):
    """
    1. Detects raw events using AA0/AU0 logic.
    2. Calculates initial baseline.
    3. Snaps timings (if enabled).
    4. QC: "Running Reset" - Trims start iteratively until cumulative lost time
       never dips below zero.
    """
    raw_events = []
    n = len(t)
    if n < 2: return []
    
    dt = np.median(np.diff(t))
    if dt <= 0: dt = 3.0
    
    # --- 1. Parameters ---
    min_dip_area = params['min_dip_area'] * 1e-4
    recovery_factor = params.get('recovery_factor', 1.0)
    
    base_win_idx = int(np.ceil(params.get('baseline_window_min', 20.0) / dt))
    base_metric = params.get('baseline_metric', 'Median')
    max_baseline = params.get('max_baseline', 2.0) * 1e-2 
    
    snap_enabled = params.get('snap_timings_to_curve', True)
    snap_win_idx = int(np.ceil(params.get('snap_window_min', 15.0) / dt))
    
    # --- 2. Raw Detection ---
    is_neg = (d2_smooth < 0) & np.isfinite(d2_smooth)
    i = 0
    while i < n:
        if is_neg[i]:
            start_idx = i
            j = i + 1
            while j < n and is_neg[j]: j += 1
            
            neg_indices = np.arange(start_idx, j)
            if len(neg_indices) > 1:
                d2_slice = d2_smooth[neg_indices]
                t_slice = t[neg_indices]
                dip_area = -np.trapz(d2_slice, t_slice)
                
                if dip_area >= min_dip_area:
                    recovery_start_idx = j
                    end_idx = recovery_start_idx
                    total_dip = dip_area
                    total_rec = 0.0
                    k = recovery_start_idx
                    
                    while k < n - 1:
                        if np.isnan(d2_smooth[k]) or np.isnan(d2_smooth[k+1]):
                            end_idx = k; break
                        
                        dt_step = t[k+1] - t[k]
                        val_avg = (d2_smooth[k] + d2_smooth[k+1]) / 2.0
                        area_step = val_avg * dt_step
                        
                        if area_step < 0: total_dip += abs(area_step)
                        else: total_rec += area_step
                            
                        if total_rec >= (total_dip * recovery_factor):
                            end_idx = k + 1; break
                        k += 1; end_idx = k
                    
                    raw_events.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'rec_start_idx': recovery_start_idx
                    })
            i = j
        else:
            i += 1

    if not raw_events: return []

    # --- 3. Baseline & Snapping ---
    refined_events = []
    for ev in raw_events:
        s, e = ev['start_idx'], ev['end_idx']
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        
        baseline = _calc_baseline(d1_smooth, s, e, n, base_win_idx, base_metric)
        if np.isfinite(baseline) and baseline > max_baseline:
            baseline = max_baseline
        
        if snap_enabled and np.isfinite(baseline):
            # Snap Start
            if d1_smooth[s] > baseline:
                search_limit = min(n, s + snap_win_idx)
                for k in range(s, search_limit):
                    if d1_smooth[k] <= baseline:
                        s = k; break
            # Snap End
            if d1_smooth[e] > baseline:
                search_limit = max(s, e - snap_win_idx)
                for k in range(e, search_limit - 1, -1):
                    if d1_smooth[k] <= baseline:
                        e = k; break
            elif d1_smooth[e] < baseline:
                search_limit = min(n - 1, e + snap_win_idx)
                for k in range(e, search_limit):
                    if d1_smooth[k] >= baseline:
                        e = k; break
        
        ev['start_idx'] = s
        ev['end_idx'] = e
        ev['baseline'] = baseline
        refined_events.append(ev)

    # --- 4. Merge Overlaps ---
    refined_events.sort(key=lambda x: x['start_idx'])
    merged_events = []
    if refined_events:
        curr = refined_events[0]
        for next_ev in refined_events[1:]:
            if next_ev['start_idx'] <= curr['end_idx']:
                curr['end_idx'] = max(curr['end_idx'], next_ev['end_idx'])
            else:
                merged_events.append(curr)
                curr = next_ev
        merged_events.append(curr)
        
    # --- 5. Metrics & Strict QC (Running Reset) ---
    final_output = []
    for ev in merged_events:
        s, e = ev['start_idx'], ev['end_idx']
        
        # 1. Recalculate baseline for merged window
        baseline = _calc_baseline(d1_smooth, s, e, n, base_win_idx, base_metric)
        if np.isfinite(baseline) and baseline > max_baseline:
            baseline = max_baseline
        
        if not np.isfinite(baseline): continue

        # 2. Extract Data
        t_slice = t[s : e + 1]
        gr_slice = d1_smooth[s : e + 1]
        
        if len(t_slice) < 2: continue

        # 3. "Running Reset" Algorithm to fix Negative Lost Time
        rate_diff = baseline - gr_slice
        rate_diff = np.nan_to_num(rate_diff, nan=0.0)
        
        dt_vec = np.diff(t_slice)
        avg_height = (rate_diff[:-1] + rate_diff[1:]) / 2.0
        step_areas = avg_height * dt_vec
        
        running_sum = 0.0
        trim_steps = 0
        
        # Iterate through the area steps.
        # If running sum drops below zero, the event implies "negative loss" (speedup).
        # We assume the event hasn't truly started yet. Reset and move start forward.
        for area in step_areas:
            running_sum += area
            if running_sum < -1e-9: # tolerance for float noise
                running_sum = 0.0
                trim_steps += 1
            else:
                # Valid accumulation, continue
                pass
        
        # Apply the trim
        if trim_steps > 0:
            s += trim_steps
            # Verify we didn't trim the whole event
            if s >= e: continue
            
            # Update slices for final calc
            t_slice = t[s : e + 1]
            gr_slice = d1_smooth[s : e + 1]
            rate_diff = baseline - gr_slice
            rate_diff = np.nan_to_num(rate_diff, nan=0.0)

        # 4. Final Metrics
        t_start, t_end = t[s], t[e]
        duration = t_end - t_start
        
        growth_deficit = np.trapz(rate_diff, x=t_slice)
        
        time_lost = 0.0
        if baseline > 0.0001:
            time_lost = growth_deficit / baseline
            
        if growth_deficit <= 1e-9: continue
        
        rec_idx = max(0, min(ev['rec_start_idx'], n - 1))
        
        final_output.append({
            'start_t': t_start,
            'end_t': t_end,
            'recovery_start_t': t[rec_idx],
            'duration': duration,
            'baseline_gr': baseline,
            'growth_deficit': growth_deficit,
            'time_lost': time_lost,
            'is_valid': True
        })
        
    return final_output

def _calc_baseline(d1, s, e, n, win_idx, metric):
    s_start = max(0, s - win_idx)
    pre_vals = d1[s_start : s]
    e_end = min(n, e + win_idx)
    post_vals = d1[min(n, e) : e_end]
    ref = np.concatenate([pre_vals, post_vals])
    ref = ref[np.isfinite(ref)]
    
    if len(ref) == 0: 
        valid_mask = np.isfinite(d1)
        if not valid_mask.any(): return np.nan
        closest_idx = np.abs(np.where(valid_mask)[0] - s).argmin()
        return d1[np.where(valid_mask)[0][closest_idx]]
        
    if metric == 'Mean': return np.mean(ref)
    else: return np.median(ref)
        
# =============================================================================
# Interactive Plotter
# =============================================================================

if _HAS_GUI:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QAbstractSpinBox, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox

    class SlowdownTestPlotter(FigureCanvasQTAgg):
        def __init__(self, width=12, height=10, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.10)
            
            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
            self.ax_len = self.fig.add_subplot(gs[0])
            self.ax_d1  = self.fig.add_subplot(gs[1], sharex=self.ax_len)
            self.ax_d2  = self.fig.add_subplot(gs[2], sharex=self.ax_len)
            
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        def update_plot(self, mid, t, y_raw, y_sm, d1_raw, d1_sm, d2_raw, d2_sm, 
                        divs, events, global_max_time, params, show_raw):
            
            self.ax_len.clear(); self.ax_d1.clear(); self.ax_d2.clear()
            x_max_limit = np.ceil(global_max_time / 120) * 120
            
            # 1. Length
            if show_raw and y_raw is not None:
                self.ax_len.scatter(t, y_raw, c='gray', s=10, alpha=0.5, zorder=1)
            if y_sm is not None:
                self.ax_len.plot(t, y_sm, c='#d62728', lw=2, zorder=2)
            if divs.any() and y_raw is not None and show_raw:
                valid_divs = divs.astype(bool) & np.isfinite(y_raw)
                if valid_divs.any():
                    self.ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=25, alpha=1.0, zorder=3)
            self.ax_len.set_title(f"Mother {mid}: log(Length)", fontweight="bold")
            self.ax_len.set_ylabel(r"ln[Length ($\mu$m)]")

            # 2. Growth Rate
            if show_raw and d1_raw is not None:
                self.ax_d1.scatter(t, d1_raw, c='gray', s=10, alpha=0.5, zorder=1)
            if d1_sm is not None:
                self.ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2, zorder=2)
            
            self.ax_d1.set_title("Growth Rate (Green=Deficit, Gold=Overshoot)", fontweight="bold")
            self.ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            self.ax_d1.set_ylabel(r"d(ln L)/dt" + "\n" + r"($\times 10^{-2}$ min$^{-1}$)")

            # 3. Acceleration
            if show_raw and d2_raw is not None:
                self.ax_d2.scatter(t, d2_raw, c='gray', s=10, alpha=0.5, zorder=1)
            if d2_sm is not None:
                self.ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2, zorder=2)
            
            self.ax_d2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
            self.ax_d2.set_title("Second deriv (Red=Slowdown;Area under 0, Blue= Speed up;Area above 0)", fontweight="bold")
            self.ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
            self.ax_d2.set_ylabel(r"d$^2$(ln L)/dt$^2$" + "\n" + r"($\times 10^{-4}$ min$^{-2}$)")
            self.ax_d2.set_xlabel("Time (min)")

            # --- Events ---
            for ev in events:
                # 1. Pale Highlight
                c_fill = 'green'
                a_fill = 0.15 
                for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                    ax.axvspan(ev['start_t'], ev['end_t'], color=c_fill, alpha=a_fill, zorder=0)
                
                # 2. Plot Baseline & Deficit
                if d1_sm is not None and np.isfinite(ev['baseline_gr']):
                    mask = (t >= ev['start_t']) & (t <= ev['end_t'])
                    if mask.any():
                        self.ax_d1.plot([ev['start_t'], ev['end_t']], 
                                        [ev['baseline_gr'], ev['baseline_gr']], 
                                        color='black', linestyle='--', linewidth=1.5)
                        
                        self.ax_d1.fill_between(t, ev['baseline_gr'], d1_sm, 
                                                where=(mask & (d1_sm < ev['baseline_gr'])), 
                                                color='green', alpha=0.3, interpolate=True)
                        
                        self.ax_d1.fill_between(t, ev['baseline_gr'], d1_sm, 
                                                where=(mask & (d1_sm > ev['baseline_gr'])), 
                                                color='gold', alpha=0.3, interpolate=True)

                # 3. Acceleration Shading
                if d2_sm is not None:
                    mask = (t >= ev['start_t']) & (t <= ev['end_t'])
                    if mask.any():
                        self.ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm < 0)), 
                                                interpolate=True, color='red', alpha=0.3)
                        self.ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm > 0)), 
                                                interpolate=True, color='blue', alpha=0.3)

                # 4. Text Labels
                center_t = (ev['start_t'] + ev['end_t']) / 2
                text_y = ev['baseline_gr'] if np.isfinite(ev['baseline_gr']) else 0.02
                
                line1 = f"duration: {ev['duration']:.0f} mins"
                line2 = f"area: {ev['growth_deficit']:.3f}"
                line3 = f"lost_time: {ev['time_lost']:.1f} mins"
                label_txt = f"{line1}\n{line2}\n{line3}"
                
                y_pos = text_y - 0.001
                
                self.ax_d1.text(center_t, y_pos, label_txt, 
                                color='black', fontsize=9, 
                                ha='center', va='top', 
                                fontweight='bold')

            for ax in [self.ax_len, self.ax_d1, self.ax_d2]:
                ax.grid(True, which='major', ls=':', alpha=0.6)
                ax.set_xlim(0, x_max_limit)
                if ax != self.ax_d2: ax.tick_params(labelbottom=False)
            self.draw()

    def run_interactive_slowdown_tester(df: pd.DataFrame, config: dict):
        if not _HAS_GUI: return None
        
        if "at_start" in df.columns:
            candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
        else:
            candidates = sorted(df["mother_id"].unique())
            
        n_moms = len(candidates)
        if n_moms == 0: return None
        
        global_tps = np.sort(df["time"].unique())
        global_max_time = global_tps.max() if len(global_tps) > 0 else 600

        viewer = napari.Viewer(title="Growth Slowdown Tester")
        viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
        try:
            viewer.window.qt_viewer.dockLayerList.setVisible(False)
            viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        except: pass
            
        plotter = SlowdownTestPlotter()
        viewer.window.add_dock_widget(plotter, area="top", name="Slowdown Plots")
        
        # --- Config Controls ---
        # 1. Trigger
        p_dip = config.get('min_dip_area', {'min': 0.0, 'max': 20.0, 'step': 0.1, 'default': 2.0})
        w_dip = FloatSlider(min=p_dip['min'], max=p_dip['max'], step=p_dip['step'], value=p_dip['default'], label="Min Dip Area (Under 0) x10^-4")
        
        # 2. Recovery
        p_fac = config.get('recovery_factor', {'min': 0.1, 'max': 3.0, 'step': 0.1, 'default': 1.0})
        w_fac = FloatSlider(min=p_fac['min'], max=p_fac['max'], step=p_fac['step'], value=p_fac['default'], label="Recovery Factor (x Dip Area)")
        
        # 3. Baseline
        p_win = config.get('baseline_window_min', {'min': 5, 'max': 120, 'step': 5, 'default': 20})
        w_base_win = IntSlider(min=p_win['min'], max=p_win['max'], step=p_win['step'], value=p_win['default'], label="Baseline Window (mins)")
        
        p_max_base = config.get('max_baseline', {'min': 0.5, 'max': 5.0, 'step': 0.1, 'default': 2.0})
        w_max_base = FloatSlider(min=p_max_base['min'], max=p_max_base['max'], step=p_max_base['step'], value=p_max_base['default'], label="Max Fast Growth Rate (x10^-2)")

        w_base_met = ComboBox(choices=['Median', 'Mean'], value='Median', label="Baseline Metric")
        
        # 4. Snapping
        w_snap = CheckBox(value=True, label="Snap Timings to Curve")
        p_snap = config.get('snap_window_min', {'min': 0, 'max': 60, 'step': 3, 'default': 15})
        w_snap_win = IntSlider(min=p_snap['min'], max=p_snap['max'], step=p_snap['step'], value=p_snap['default'], label="Snap Search Window (mins)")
        
        # 5. Display
        w_raw = CheckBox(value=False, label="Show Raw Points")
        
        w_save = PushButton(text="Finish and Save Parameters")
        w_save.native.setMinimumHeight(60)

        # --- FIX: Robust Focus Setting + Event Hook ---
        def fix_focus(widget):
            try:
                slider = widget.native.findChild(QSlider)
                if slider: slider.setFocusPolicy(Qt.StrongFocus)
                
                spins = widget.native.findChildren((QDoubleSpinBox, QSpinBox))
                for s in spins:
                    s.setFocusPolicy(Qt.ClickFocus)
                    s.lineEdit().setFocusPolicy(Qt.ClickFocus)
            except: pass

        widgets_to_fix = [w_dip, w_fac, w_base_win, w_max_base, w_snap_win]
        for w in widgets_to_fix:
            fix_focus(w)
            
        def regain_focus(w):
            try:
                slider = w.native.findChild(QSlider)
                if slider: slider.setFocus()
            except: pass

        # --- Layout ---
        c1 = Container(widgets=[w_dip, w_fac], layout="horizontal", labels=True)
        c2 = Container(widgets=[w_base_win, w_max_base], layout="horizontal", labels=True)
        c3 = Container(widgets=[w_base_met, w_snap, w_snap_win, w_raw], layout="horizontal", labels=True)
        
        main_c = Container(widgets=[c1, c2, c3, Label(value=""), w_save])
        viewer.window.add_dock_widget(main_c, area="bottom", name="Controls")
        
        final_params = {}

        # --- VISIBILITY FIX: Enable/Disable only, Force Visible=True ---
        def update_visibility(event=None):
            w_snap_win.enabled = w_snap.value 
            w_snap_win.visible = True # Force visible to prevent layout shift

        w_snap.changed.connect(update_visibility)
        update_visibility()

        def update(source_widget=None):
            if source_widget in widgets_to_fix:
                regain_focus(source_widget)

            idx = viewer.dims.current_step[0]
            if idx < n_moms:
                mid = candidates[int(idx)]
                trace = df[df["mother_id"] == mid].sort_values("time")
                t = trace["time"].values
                divs = trace["division_event"].astype(float).fillna(0).astype(bool).values
                
                y_raw = trace["log_length_cleaned"].values if "log_length_cleaned" in df.columns else None
                y_sm  = trace["log_length_smoothed"].values if "log_length_smoothed" in df.columns else None
                d1_raw = trace["growth_rate"].values if "growth_rate" in df.columns else None
                d1_sm  = trace["growth_rate_smoothed"].values
                
                d2_col = "second_deriv"
                d2_sm_col = "second_deriv_smoothed"
                d2_raw = trace[d2_col].values if d2_col in trace.columns else None
                d2_sm  = trace[d2_sm_col].values
                
                if d1_sm is None or d2_sm is None: return

                current_params = {
                    'min_dip_area': w_dip.value,
                    'recovery_factor': w_fac.value,
                    'baseline_window_min': w_base_win.value,
                    'baseline_metric': w_base_met.value,
                    'max_baseline': w_max_base.value,
                    'snap_timings_to_curve': w_snap.value,
                    'snap_window_min': w_snap_win.value
                }
                
                events = _detect_and_measure_slowdowns(t, d1_sm, d2_sm, params=current_params)
                
                plotter.update_plot(
                    mid, t, y_raw, y_sm, d1_raw, d1_sm, d2_raw, d2_sm,
                    divs, events, global_max_time, current_params, show_raw=w_raw.value
                )
                viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

        w_dip.changed.connect(lambda: update(w_dip))
        w_fac.changed.connect(lambda: update(w_fac))
        w_base_win.changed.connect(lambda: update(w_base_win))
        w_max_base.changed.connect(lambda: update(w_max_base))
        w_snap_win.changed.connect(lambda: update(w_snap_win))
        
        w_base_met.changed.connect(lambda: update(None))
        w_snap.changed.connect(lambda: update(None))
        w_raw.changed.connect(lambda: update(None))
        
        viewer.dims.events.current_step.connect(lambda e: update(None))
        
        @w_save.clicked.connect
        def on_save():
            final_params.update({
                'min_dip_area': w_dip.value,
                'recovery_factor': w_fac.value,
                'baseline_window_min': w_base_win.value,
                'baseline_metric': w_base_met.value,
                'max_baseline': w_max_base.value,
                'snap_timings_to_curve': w_snap.value,
                'snap_window_min': w_snap_win.value
            })
            print(f"Saved Params: {final_params}")
            viewer.close()
        
        viewer.dims.set_current_step(0, 0)
        update(None)
        viewer.window.qt_viewer.window().showMaximized()
        napari.run()
        
        return final_params

# =============================================================================
# Batch Execution
# =============================================================================

def _process_slowdown_batch(group, params):
    """
    Worker function for batch processing.
    Calculates detailed slowdown metrics including cumulative lost time.
    """
    try:
        trace = group.sort_values("time")
        t = trace["time"].values
        
        if "second_deriv_smoothed" in trace.columns: d2_col = "second_deriv_smoothed"
        else: return None
        if "growth_rate_smoothed" not in trace.columns: return None

        d1_sm = trace["growth_rate_smoothed"].values
        d2_sm = trace[d2_col].values
        
        # Run Detection
        events = _detect_and_measure_slowdowns(t, d1_sm, d2_sm, params=params)
        valid_events = [ev for ev in events if ev.get('is_valid', True)]
        if not valid_events: return None
        
        out_df = pd.DataFrame(index=trace.index)
        
        out_df['slowdown'] = False
        # CHANGED: Separate columns instead of interval string
        out_df['slowdown_start'] = np.nan
        out_df['slowdown_end'] = np.nan
        
        out_df['current_slowdown_duration'] = np.nan
        out_df['total_slowdown_duration'] = np.nan
        out_df['current_lost_time'] = np.nan
        out_df['total_lost_time'] = np.nan
        
        for ev in valid_events:
            s_idx = np.searchsorted(t, ev['start_t'])
            e_idx = np.searchsorted(t, ev['end_t'])
            
            # --- Calculation for Dataframe Columns ---
            baseline = ev['baseline_gr']
            if not np.isfinite(baseline): continue
            
            t_slice = t[s_idx : e_idx + 1]
            gr_slice = d1_sm[s_idx : e_idx + 1]
            
            if len(t_slice) < 2: continue
            
            rate_diff = baseline - gr_slice
            rate_diff = np.nan_to_num(rate_diff, nan=0.0)
            
            dt = np.diff(t_slice)
            avg_height = (rate_diff[:-1] + rate_diff[1:]) / 2.0
            step_areas = avg_height * dt
            
            cum_lost_profile = np.concatenate([[0.0], np.cumsum(step_areas)]) / baseline
            cum_lost_profile = np.maximum(cum_lost_profile, 0.0)
            
            # --- Populate Output ---
            subset_indices = trace.index[s_idx : e_idx + 1]
            if len(subset_indices) != len(t_slice): continue
            
            out_df.loc[subset_indices, 'slowdown'] = True
            # CHANGED: Assign numeric start/end
            out_df.loc[subset_indices, 'slowdown_start'] = ev['start_t']
            out_df.loc[subset_indices, 'slowdown_end'] = ev['end_t']
            
            out_df.loc[subset_indices, 'total_slowdown_duration'] = ev['duration']
            out_df.loc[subset_indices, 'total_lost_time'] = ev['time_lost']
            
            out_df.loc[subset_indices, 'current_slowdown_duration'] = t_slice - t_slice[0]
            out_df.loc[subset_indices, 'current_lost_time'] = cum_lost_profile
            
        return out_df[out_df['slowdown'] == True]
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None
        

def execute_batch_slowdown_detection(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Batch execution wrapper."""
    if df_main is None: return None
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n EXECUTING BATCH SLOWDOWN DETECTION \n{'='*40}")
    
    # 1. Initialize Columns
    # CHANGED: Replaced 'slowdown_interval' with 'slowdown_start', 'slowdown_end'
    new_cols = [
        'slowdown', 
        'slowdown_start',
        'slowdown_end',
        'current_slowdown_duration', 
        'total_slowdown_duration', 
        'current_lost_time', 
        'total_lost_time'
    ]
    
    df_main['slowdown'] = False
    
    # Initialize all numeric columns with NaN
    for c in new_cols[1:]:
        df_main[c] = np.nan
        
    # 2. Select Candidates
    if "at_start" in df_main.columns: 
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else: 
        candidates = df_main["mother_id"].unique()
    
    print(f"Processing {len(candidates)} mothers...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    # 3. Parallel Processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_slowdown_batch)(group, params)
        for _, group in tqdm(grouped, desc="Detecting slowdowns", total=len(candidates))
    )
    
    # 4. Apply Updates
    valid_updates = [res for res in results if res is not None and not res.empty]
    
    if valid_updates:
        print("Applying annotations...")
        updates_df = pd.concat(valid_updates)
        
        idx = updates_df.index
        for col in new_cols:
            df_main.loc[idx, col] = updates_df[col]
        
        n_points = len(updates_df)
        print(f"Annotated {n_points} timepoints in slowdown regions.")
    else:
        print("No events detected.")
        
    print(f"{'='*40}\n DONE \n{'='*40}")
    return df_main
    
def generate_survivor_slowdown_pdf(
    df: pd.DataFrame, 
    slowdown_params: dict,
    treatment_start: float, 
    treatment_end: float,
    lanes_to_check: list = None, 
    max_IDs_per_lane: int = 50,
    filename: str = None
):
    """
    Generates a PDF report for 'Survivors' (growing_before=True AND growing_after=True).
    Matches the 3-panel layout, but ALSO recalculates and plots the rich visual 
    slowdown annotations (baselines, shading, and duration/lost_time labels) 
    used in the interactive testers.
    """
    if df is None: return
    
    print(f"{'='*50}\n GENERATING SURVIVOR SLOWDOWN PDF \n{'='*50}")

    # 1. Validation
    req_cols = ['mother_id', 'growing_before', 'growing_after', 'time', 'log_length_smoothed', 'growth_rate_smoothed']
    for c in req_cols:
        if c not in df.columns:
            print(f"Error: Required column '{c}' missing from DataFrame.")
            return

    # Determine second derivative column name
    d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in df.columns else "2nd_deriv_smoothed"
    if d2_col not in df.columns:
        print(f"Error: Second derivative column missing.")
        return

    # 2. Identify Survivors
    survivor_mask = (df['growing_before'] == True) & (df['growing_after'] == True)
    survivors = df[survivor_mask]['mother_id'].unique()
    
    if len(survivors) == 0:
        print("No survivors found.")
        return

    # 3. Filter by Lane (Optional)
    if lanes_to_check is not None:
        if 'lane' not in df.columns:
            print("Error: 'lane' column missing, cannot filter by lane.")
            return
        print(f"Filtering for lanes: {lanes_to_check}")
        lane_moms = df[df['lane'].isin(lanes_to_check)]['mother_id'].unique()
        survivors = np.intersect1d(survivors, lane_moms)
        
    if len(survivors) == 0:
        print(f"No survivors found in the specified lanes.")
        return

    # 4. Selection Logic (Max N per lane)
    final_list = []
    if 'lane' in df.columns:
        meta = df[['mother_id', 'lane']].drop_duplicates().set_index('mother_id')
        present_lanes = meta.loc[survivors, 'lane'].unique()
        for lane in sorted(present_lanes):
            moms_in_lane = sorted(meta[meta['lane'] == lane].index.intersection(survivors))
            if len(moms_in_lane) > max_IDs_per_lane:
                final_list.extend(moms_in_lane[:max_IDs_per_lane])
            else:
                final_list.extend(moms_in_lane)
    else:
        # Fallback if no lane column
        if len(survivors) > max_IDs_per_lane:
            final_list = sorted(survivors)[:max_IDs_per_lane]
        else:
            final_list = sorted(survivors)

    if not final_list:
        print("No IDs selected.")
        return

    # 5. Setup Output
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"survivor_slowdowns_{timestamp}.pdf"

    # Global X Limit
    x_max = df['time'].max()
    x_limit = np.ceil(x_max / 120) * 120
    
    print(f"Generating PDF: {filename} ({len(final_list)} plots)...")

    # 6. Plotting Loop
    with PdfPages(filename) as pdf:
        for mid in tqdm(final_list, desc="Plotting Survivors"):
            trace = df[df['mother_id'] == mid].sort_values('time')
            t = trace['time'].values
            
            # Extract Data
            y_raw = trace['log_length_cleaned'].values if 'log_length_cleaned' in trace.columns else None
            y_sm  = trace['log_length_smoothed'].values
            d1_sm = trace['growth_rate_smoothed'].values
            d2_sm = trace[d2_col].values
            divs = trace["division_event"].astype(float).fillna(0).astype(bool).values if "division_event" in trace.columns else np.zeros(len(t), dtype=bool)

            # Recalculate exact slowdown events for rich visualization
            events = _detect_and_measure_slowdowns(t, d1_sm, d2_sm, params=slowdown_params)
            valid_events = [ev for ev in events if ev.get('is_valid', True)]

            # --- Layout: 3 Vertical Panels ---
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
            plt.subplots_adjust(top=0.92, bottom=0.10, left=0.12, right=0.95, hspace=0.3)
            
            ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
            
            # Title Construction
            title_parts = [f"Survivor Slowdowns: Mother {mid}"]
            if 'lane' in trace.columns:
                title_parts.append(f"Lane {trace['lane'].iloc[0]}")
            if 'condition' in trace.columns:
                title_parts.append(f"{trace['condition'].iloc[0]}")
            
            fig.suptitle(" | ".join(title_parts), fontweight='bold', fontsize=12)
            
            # --- Panel 1: Log Length ---
            if y_raw is not None:
                ax_len.scatter(t, y_raw, c='gray', s=10, alpha=0.3, zorder=1)
            ax_len.plot(t, y_sm, c='#d62728', lw=2, label='Smoothed', zorder=2)
            
            if divs.any() and y_raw is not None:
                valid_divs = divs & np.isfinite(y_raw)
                ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=30, marker='o', edgecolors='k', zorder=3, label='Division')
                
            ax_len.set_ylabel(r"ln[Length ($\mu$m)]", fontweight='bold')
            ax_len.set_ylim(0.25, 3.25)

            # --- Panel 2: Growth Rate ---
            ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2)
            ax_d1.set_ylabel(r"Rate ($\times 10^{-2} min^{-1}$)", fontweight='bold')
            ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            ax_d1.set_ylim(-0.001, 0.025)

            # --- Panel 3: Acceleration ---
            ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
            ax_d2.axhline(0, c='k', ls='-', lw=0.8, alpha=0.5)
            ax_d2.set_ylabel(r"Accel ($\times 10^{-4} min^{-2}$)", fontweight='bold')
            ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
            ax_d2.set_ylim(-0.00025, 0.00025)
            ax_d2.set_xlabel("Time (min)", fontweight='bold')

            # --- Apply Slowdown Highlights & Labels ---
            for ev in valid_events:
                mask = (t >= ev['start_t']) & (t <= ev['end_t'])
                if not mask.any(): continue
                
                # 1. Background Highlight (All Panels)
                for ax in axes:
                    ax.axvspan(ev['start_t'], ev['end_t'], color='green', alpha=0.15, zorder=0)

                # 2. Rate Panel Annotations
                baseline = ev['baseline_gr']
                if np.isfinite(baseline):
                    ax_d1.plot([ev['start_t'], ev['end_t']], [baseline, baseline], color='black', linestyle='--', linewidth=1.5)
                    
                    # Fill Deficit (Green) and Overshoot (Gold)
                    ax_d1.fill_between(t, baseline, d1_sm, where=(mask & (d1_sm < baseline)), color='green', alpha=0.3, interpolate=True)
                    ax_d1.fill_between(t, baseline, d1_sm, where=(mask & (d1_sm > baseline)), color='gold', alpha=0.3, interpolate=True)

                    # Text Labels (Duration, Area, Lost Time)
                    center_t = (ev['start_t'] + ev['end_t']) / 2
                    line1 = f"duration: {ev['duration']:.0f} m"
                    line2 = f"area: {ev['growth_deficit']:.3f}"
                    line3 = f"lost: {ev['time_lost']:.1f} m"
                    label_txt = f"{line1}\n{line2}\n{line3}"
                    
                    ax_d1.text(center_t, baseline - 0.001, label_txt, color='black', fontsize=8, ha='center', va='top', fontweight='bold')

                # 3. Acceleration Panel Shading
                ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm < 0)), color='red', alpha=0.3, interpolate=True)
                ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm > 0)), color='blue', alpha=0.3, interpolate=True)

            # --- Common Formatting ---
            for ax in axes:
                # Treatment Shading
                if treatment_start is not None and treatment_end is not None:
                    ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.10, zorder=0)
                    
                    if ax == ax_len:
                        treat_center = (treatment_start + treatment_end) / 2
                        y_min, y_max = ax.get_ylim()
                        text_y = y_max - (0.05 * (y_max - y_min))
                        ax.text(treat_center, text_y, "Treatment", color='darkred', ha='center', va='top', fontweight='bold', fontsize=8, alpha=0.8)

                ax.set_xlim(0, x_limit)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
                ax.grid(True, which='major', ls='-', alpha=0.4)
                ax.grid(True, which='minor', ls=':', alpha=0.2)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Done. Report saved to {os.path.abspath(filename)}")

def plot_example_survivor_shifted_pub(df, slowdown_params, treatment_start, treatment_end, mother_id=33354, save_name="example_survivor_plot"):
    """
    Plots cell length and growth rate for a specific mother_id, 
    shifting the x-axis to 'Time since treatment start' (-420 to 660).
    Formatted for publication (7 x 5.25 inches, 8pt fonts, specific z-orders).
    Saves the result as SVG and high-res PNG.
    """
    # 1. Extract and sort trace for the target mother
    trace = df[df['mother_id'] == mother_id].sort_values('time')
    if trace.empty:
        print(f"Error: No data found for mother_id {mother_id}")
        return

    # Extract raw time and shift it relative to treatment start
    t_raw = trace['time'].values
    t_shifted = t_raw - treatment_start
    
    # 2. Extract Data
    y_sm  = trace['log_length_smoothed'].values
    d1_sm = trace['growth_rate_smoothed'].values
    
    d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
    d2_sm = trace[d2_col].values
    
    divs = trace["division_event"].astype(float).fillna(0).astype(bool).values if "division_event" in trace.columns else np.zeros(len(t_raw), dtype=bool)

    # 3. Detect slowdowns using original (un-shifted) time to maintain baseline accuracy
    events = _detect_and_measure_slowdowns(t_raw, d1_sm, d2_sm, params=slowdown_params)
    valid_events = [ev for ev in events if ev.get('is_valid', True)]
    
    # Keep ONLY the first slowdown event
    if len(valid_events) > 0:
        valid_events = [valid_events[0]]

    # 4. Setup Figure Layout
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4))
    
    # Apply subplots_adjust parameters
    fig.subplots_adjust(top=0.95, bottom=0.09, left=0.09, right=0.98, hspace=0.45)
    
    ax_len, ax_d1 = axes[0], axes[1]
    
    # ==========================================
    # Panel 1: Log Length
    # ==========================================
    ax_len.set_title(r"Cell length trace of an example transiently tolerant survivor treated during exponential growth", fontsize=8)
    
    # Plot smoothed line and divisions
    ax_len.plot(t_shifted, y_sm, c='#d62728', lw=1, label='Smoothed', zorder=2)
    
    if divs.any():
        ax_len.scatter(t_shifted[divs], y_sm[divs], c='cyan', s=10, marker='o', edgecolors='k', linewidths=0.5, zorder=3, label='Division')
        
    ax_len.set_ylabel(r"ln[Cell length ($\mu$m)]", fontsize=8)
    ax_len.set_ylim(0.5, 2.5)
    ax_len.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # ==========================================
    # Panel 2: Growth Rate
    # ==========================================
    ax_d1.set_title(r"Growth rate of the example transiently tolerant survivor", fontsize=8)
    ax_d1.plot(t_shifted, d1_sm, c='#1f77b4', lw=1, zorder=2)
    
    ax_d1.set_ylabel(r"Growth rate (1/l dl/dt)", fontsize=8)
    ax_d1.set_ylim(0, 0.021)
    ax_d1.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    
    # 5. Apply Highlights, Baselines, and Shading (Shifted)
    for ev in valid_events:
        # Shift event timings
        ev_start_shift = ev['start_t'] - treatment_start
        ev_end_shift = ev['end_t'] - treatment_start
        
        mask = (t_shifted >= ev_start_shift) & (t_shifted <= ev_end_shift)
        if not mask.any(): continue
        
        # Background Highlight for Slowdown (Transparent Blue)
        for ax in axes:
            ax.axvspan(ev_start_shift, ev_end_shift, color='#1f77b4', alpha=0.15, zorder=0, linewidth=0)

        # Rate Panel Annotations
        baseline = ev['baseline_gr']
        if np.isfinite(baseline):
            ax_d1.plot([ev_start_shift, ev_end_shift], [baseline, baseline], color='black', linestyle='--', linewidth=1, zorder=3)
            
            # Fill Deficit (Now matching the transparent blue color) and Overshoot (Gold)
            ax_d1.fill_between(t_shifted, baseline, d1_sm, where=(mask & (d1_sm < baseline)), color='#1f77b4', alpha=0.3, interpolate=True, linewidth=0, zorder=1)
            ax_d1.fill_between(t_shifted, baseline, d1_sm, where=(mask & (d1_sm > baseline)), color='gold', alpha=0.3, interpolate=True, linewidth=0, zorder=1)

            # Updated Text Labels (Duration to whole minute, Lost Time to whole minute, increased linespacing)
            center_t = (ev_start_shift + ev_end_shift) / 2
            label_txt = f"Slowdown duration: {ev['duration']:.0f} minutes\nSlowdown amount: {ev['time_lost']:.0f} minutes"
            
            # Moved further down (-0.0015 instead of -0.0005) and brought to front (zorder=10)
            ax_d1.text(center_t, baseline - 0.0015, label_txt, color='black', ha='center', va='top', 
                       fontsize=8, linespacing=1.5, zorder=10)

    # ==========================================
    # Common Formatting & Treatment Shading
    # ==========================================
    treat_end_shifted = treatment_end - treatment_start
    
    for ax in axes:
        # Treatment Window
        if treatment_start is not None and treatment_end is not None:
            ax.axvspan(0, treat_end_shifted, color='darkred', alpha=0.10, zorder=0, linewidth=0)
            
            if ax == ax_len:
                treat_center = treat_end_shifted / 2
                y_min, y_max = ax.get_ylim()
                text_y = y_max - (0.05 * (y_max - y_min))
                # Brought to front with zorder=10
                ax.text(treat_center, text_y, r"Antibiotic treatment", color='darkred', ha='center', va='top', 
                        fontsize=8, zorder=10)

        ax.set_xlim(-420, 660)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
        
        # Apply the x-axis label to both subplots individually
        ax.set_xlabel(r"Time since treatment start (minutes)", fontsize=8)
        
        # Ensure tick labels use 8pt font
        ax.tick_params(axis='both', labelsize=8)

    # ==========================================
    # Save Outputs
    # ==========================================
    if save_name:
        # Save as SVG (vector graphic format natively supported by your Nature style preferences)
        plt.savefig(f"{save_name}.svg", format='svg')
        # Save as PNG with 300 DPI (high resolution)
        plt.savefig(f"{save_name}.png", format='png', dpi=300)
        print(f"Plot successfully saved as '{save_name}.svg' and '{save_name}.png'")

    plt.show()

# =============================================================================
# Post-Growth Classification (Growing After)
# =============================================================================

def _check_growing_status(trace, t3, t4, min_divs, max_nan_pct, min_mean_gr, global_timepoints=None):
    """
    Helper to determine if a cell is 'growing_after' based on:
      1. Division count
      2. Data quality (NaN %)
      3. Mean Growth Rate (NEW)
    """
    # 1. Determine Expected Timepoints
    if global_timepoints is None:
        expected_times = trace["time"].unique()
    else:
        mask_exp = (global_timepoints >= t3) & (global_timepoints <= t4)
        expected_times = global_timepoints[mask_exp]
        
    n_expected = len(expected_times)
    
    # Defaults if window is empty
    if n_expected == 0:
        return False, 100.0, 0, 0.0
        
    # 2. Get Actual Data
    mask = (trace['time'] >= t3) & (trace['time'] <= t4)
    sub = trace[mask]
    
    # 3. Count Missing Rows
    n_present = len(sub)
    n_missing = n_expected - n_present
    
    # 4. Count Explicit NaNs in Present Rows
    c_len = 'log_length_cleaned'
    is_bad = sub[c_len].isna() if c_len in sub.columns else pd.Series(True, index=sub.index)
    
    c_gr = 'growth_rate'
    if c_gr in sub.columns:
        is_bad = is_bad | sub[c_gr].isna()
        
    c_d2 = 'second_deriv'
    if c_d2 not in sub.columns:
        if '2nd_deriv' in sub.columns: c_d2 = '2nd_deriv'
    
    if c_d2 in sub.columns:
        is_bad = is_bad | sub[c_d2].isna()
        
    n_bad_present = is_bad.sum()
    
    # 5. Total Bad & Percentage
    total_bad = n_missing + n_bad_present
    nan_pct = (total_bad / n_expected) * 100.0
    
    # 6. Divisions
    div_count = 0
    if 'division_event' in sub.columns:
        div_count = sub['division_event'].fillna(0).astype(bool).sum()

    # 7. Mean Growth Rate (NEW)
    mean_gr = 0.0
    if 'growth_rate_smoothed' in sub.columns:
        gr_vals = sub['growth_rate_smoothed'].dropna()
        if not gr_vals.empty:
            mean_gr = float(gr_vals.mean())
        
    # Decision
    is_growing = (
        (div_count >= min_divs) and 
        (nan_pct <= max_nan_pct) and
        (mean_gr >= min_mean_gr)
    )
    
    return is_growing, nan_pct, div_count, mean_gr

def launch_interactive_post_growth_testing(
    df: pd.DataFrame, 
    t3: float, 
    t4: float, 
    config: dict
) -> Dict[str, Any]:
    """
    Napari-based interactive tuning for 'growing_after' classification (T3-T4).
    Now includes min_mean_growth_rate_window.
    """
    # 1. Validation & Non-GUI Fallback
    try:
        req_min_divs = config["min_divs"]
        req_max_nan = config["max_nan_pct"]
        # Handle new key gracefully if older config passed
        req_min_gr = config.get("min_mean_growth_rate_window", {'min': -5.0, 'max': 5.0, 'step': 0.1, 'default': 0.0})
    except KeyError as e:
        raise KeyError(f"Missing required key in GROWTH_AFTER_CONFIG: {e}")

    if not _HAS_GUI: 
        print("Napari not installed/available. Returning defaults from config.")
        return {
            'min_divs': req_min_divs['default'],
            'max_nan_pct': req_max_nan['default'],
            'min_mean_growth_rate_window': req_min_gr['default'] / 100.0,
            't3': t3, 't4': t4
        }
    
    # 2. Data Setup
    if "at_start" in df.columns:
        candidates = sorted(df[df["at_start"] == True]["mother_id"].unique())
    else:
        candidates = sorted(df["mother_id"].unique())
        
    n_moms = len(candidates)
    if n_moms == 0: 
        print("No candidates found.")
        return {}
    
    global_tps = np.sort(df["time"].unique())
    global_max_time = global_tps.max() if len(global_tps) > 0 else 600
    
    print(f"Loaded {n_moms} traces.")
    print(f"Checking Post-Treatment Window: {t3} min -> {t4} min")

    # 3. Viewer Setup
    viewer = napari.Viewer(title="Post-Growth Tester (Growing After)")
    viewer.add_image(np.zeros((n_moms, 1, 1)), name="Navigator", opacity=0.0)
    try:
        viewer.window.qt_viewer.dockLayerList.setVisible(False)
        viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    except: pass

    # 4. Plotter Class
    class PostGrowthPlotter(FigureCanvasQTAgg):
        def __init__(self, width=10, height=6, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.ax = self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15) 
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def update_plot(self, mid, t, y, divs, t3, t4, is_growing, nan_pct, div_count, mean_gr, 
                        min_divs, max_nan, min_gr, x_max_limit, growing_before_status):
            self.ax.clear()
            
            # Plot Trace
            if y is not None:
                self.ax.plot(t, y, 'b.-', alpha=0.5, label="Log Length")
                if divs.any():
                    valid_divs = divs & np.isfinite(y)
                    self.ax.scatter(t[valid_divs], y[valid_divs], c='cyan', s=60, zorder=3, edgecolors='k', label="Divisions")
            
            # Highlight Window
            self.ax.axvspan(t3, t4, color='gray', alpha=0.1, label="Check Window (Post)")
            
            # Status Logic
            status_color = "#2ca02c" if is_growing else "#d62728"
            status_text = "PASS (Growing)" if is_growing else "FAIL"
            
            # Growing Before Info
            if growing_before_status is True:
                gb_text = "YES"
            elif growing_before_status is False:
                gb_text = "NO"
            else:
                gb_text = "N/A"

            # Display values (scaled for display)
            gr_disp = mean_gr * 100.0
            min_gr_disp = min_gr * 100.0

            # Title Construction (Multi-line)
            title = (
                f"Mother {mid}  |  Growing Before: {gb_text}\n"
                f"Post-Growth Status: {status_text}\n"
                f"Divisions: {div_count} (Req >= {min_divs}) | "
                f"Bad Pts: {nan_pct:.1f}% (Req <= {max_nan:.1f}%)\n"
                f"Mean GR: {gr_disp:.2f} (Req >= {min_gr_disp:.2f}, x10^-2)"
            )
            
            self.ax.set_title(title, color=status_color, fontweight='bold', fontsize=11)
            
            self.ax.set_xlabel("Time (min)")
            self.ax.set_ylabel("Log Length")
            self.ax.legend(loc='upper left', frameon=True, fontsize=9)
            
            self.ax.set_xlim(0, x_max_limit)
            self.ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
            self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
            self.ax.grid(True, which='major', ls='-', alpha=0.4)
            self.draw()

    plotter = PostGrowthPlotter()
    viewer.window.add_dock_widget(plotter, area="top", name="Trace View")

    # 5. Controls
    w_divs = IntSlider(
        min=req_min_divs["min"], 
        max=req_min_divs["max"], 
        step=req_min_divs["step"], 
        value=req_min_divs["default"], 
        label="Min Divisions"
    )
    
    w_nan = FloatSlider(
        min=req_max_nan["min"], 
        max=req_max_nan["max"], 
        step=req_max_nan["step"], 
        value=req_max_nan["default"], 
        label="Max % Bad (NaN+Miss)"
    )

    w_min_gr = FloatSlider(
        min=req_min_gr["min"],
        max=req_min_gr["max"],
        step=req_min_gr["step"],
        value=req_min_gr["default"],
        label="Min Mean GR (x10^-2)"
    )
    
    # Fix GUI focus for new slider
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QAbstractSlider, QDoubleSpinBox
        for s in w_min_gr.native.findChildren(QAbstractSlider): s.setFocusPolicy(Qt.StrongFocus)
        for s in w_min_gr.native.findChildren(QDoubleSpinBox): s.setDecimals(2)
    except: pass
    
    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(60)
    
    # Layout: Split into columns
    c1 = Container(widgets=[w_divs, w_nan], layout="vertical")
    c2 = Container(widgets=[w_min_gr, Label(value="")], layout="vertical")
    c_ctrl = Container(widgets=[c1, c2, w_save], layout="vertical")
    
    viewer.window.add_dock_widget(c_ctrl, area="bottom", name="Parameters")

    # 6. Update Logic
    final_params = {
        'min_divs': req_min_divs['default'],
        'max_nan_pct': req_max_nan['default'],
        'min_mean_growth_rate_window': req_min_gr['default'] / 100.0,
        't3': t3, 't4': t4
    }
    
    x_max_limit = np.ceil(global_max_time / 120) * 120

    def update():
        idx = viewer.dims.current_step[0]
        if idx < n_moms:
            mid = candidates[int(idx)]
            trace = df[df["mother_id"] == mid].sort_values("time")
            t = trace["time"].values
            y = trace["log_length_cleaned"].values if "log_length_cleaned" in trace.columns else None
            divs = trace["division_event"].astype(bool).values if "division_event" in trace.columns else np.zeros(len(t), dtype=bool)
            
            # Retrieve 'growing_before' status
            if "growing_before" in df.columns:
                gb_val = trace["growing_before"].iloc[0]
                gb_status = bool(gb_val) if pd.notna(gb_val) else False
            else:
                gb_status = None

            # Calc Status
            # IMPORTANT: w_min_gr.value is in x10^-2 units (e.g. 1.0), logic needs raw (0.01)
            raw_min_gr = w_min_gr.value / 100.0

            is_growing, nan_pct, div_count, mean_gr = _check_growing_status(
                trace, t3, t4, w_divs.value, w_nan.value, raw_min_gr,
                global_timepoints=global_tps
            )
            
            plotter.update_plot(
                mid, t, y, divs, t3, t4, 
                is_growing, nan_pct, div_count, mean_gr,
                w_divs.value, w_nan.value, raw_min_gr,
                x_max_limit, 
                gb_status
            )
            viewer.status = f"Mother {mid} ({int(idx)+1}/{n_moms})"

    w_divs.changed.connect(update)
    w_nan.changed.connect(update)
    w_min_gr.changed.connect(update)
    viewer.dims.events.current_step.connect(lambda e: update())
    
    @w_save.clicked.connect
    def on_save():
        final_params.update({
            'min_divs': w_divs.value,
            'max_nan_pct': w_nan.value,
            'min_mean_growth_rate_window': w_min_gr.value / 100.0,
            't3': t3,
            't4': t4
        })
        print(f"Growing After Params Saved: {final_params}")
        viewer.close()

    viewer.dims.set_current_step(0, 0)
    update()
    viewer.window.qt_viewer.window().showMaximized()
    napari.run()
    
    return final_params


def _process_post_growth_batch(group, t3, t4, min_divs, max_nan_pct, min_mean_gr, global_timepoints):
    """
    Worker function for parallel post-growth classification.
    Accepts min_mean_gr.
    """
    try:
        is_growing, nan_pct, div_count, mean_gr = _check_growing_status(
            group, t3, t4, min_divs, max_nan_pct, min_mean_gr,
            global_timepoints=global_timepoints
        )
        
        # Create a DataFrame with the same index as the group
        res = pd.DataFrame(index=group.index)
        res['growing_after'] = is_growing
        
        return res
    except Exception:
        return None


def execute_batch_post_growth_classification(df_main: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies 'growing_after' classification to the dataset using parallel processing.
    Includes mean growth rate check.
    """
    if df_main is None: return None
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"{'='*40}\n CLASSIFYING GROWING_AFTER (Parallel) \n{'='*40}")
    
    # Extract params
    min_divs = params.get('min_divs', 2)
    max_nan_pct = params.get('max_nan_pct', 25.0)
    min_mean_gr = params.get('min_mean_growth_rate_window', -100.0) # Default lax if missing
    t3 = params.get('t3', df_main['time'].max() - 360) 
    t4 = params.get('t4', df_main['time'].max())
    
    print(f"Window: [{t3:.1f}, {t4:.1f}]")
    print(f"Criteria: Divs >= {min_divs}, Bad Data <= {max_nan_pct}%, Mean GR >= {min_mean_gr}")
    
    # 1. Establish Global Timepoints
    global_tps = np.sort(df_main["time"].unique())
    
    # Initialize columns
    cols = ['growing_after']
    df_main['growing_after'] = False
    
    if "at_start" in df_main.columns:
        candidates = df_main.loc[df_main["at_start"] == True, "mother_id"].unique()
    else:
        candidates = df_main["mother_id"].unique()
        
    print(f"Processing {len(candidates)} mother_ids...")
    
    candidate_mask = df_main["mother_id"].isin(candidates)
    grouped = df_main.loc[candidate_mask].groupby("mother_id")
    
    # Parallel Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_post_growth_batch)(group, t3, t4, min_divs, max_nan_pct, min_mean_gr, global_tps)
        for _, group in tqdm(grouped, desc="Classifying growing_after", total=len(candidates))
    )
    
    # Merge results
    valid_updates = [res for res in results if res is not None and not res.empty]
    
    if valid_updates:
        print("Applying classifications...")
        updates_df = pd.concat(valid_updates)
        
        idx = updates_df.index
        df_main.loc[idx, cols] = updates_df[cols]
        
        n_growing_moms = updates_df[updates_df['growing_after']].index.nunique()
        print(f"Done. Approximately {n_growing_moms} mothers classified as 'growing_after'.")
    else:
        print("No mothers classified.")

    return df_main

# Survivors PDF code
def generate_survivor_pdf(
    df: pd.DataFrame, 
    treatment_start: float, 
    treatment_end: float,
    lanes_to_check: list = None, 
    max_IDs_per_lane: int = 50,
    filename: str = None
):
    """
    Generates a PDF report for 'Survivors' (growing_before=True AND growing_after=True).
    Matches the 3-panel layout (Length, Rate, Accel) of the slowdown reports.
    """
    if df is None: return
    
    print(f"{'='*40}\n GENERATING SURVIVOR PDF \n{'='*40}")

    # 1. Validation
    req_cols = ['mother_id', 'growing_before', 'growing_after', 'time', 'log_length_smoothed']
    for c in req_cols:
        if c not in df.columns:
            print(f"Error: Required column '{c}' missing from DataFrame.")
            return

    # 2. Identify Survivors
    survivor_mask = (df['growing_before'] == True) & (df['growing_after'] == True)
    survivors = df[survivor_mask]['mother_id'].unique()
    
    if len(survivors) == 0:
        print("No survivors found.")
        return

    # 3. Filter by Lane
    if lanes_to_check is not None:
        if 'lane' not in df.columns:
            print("Error: 'lane' column missing, cannot filter by lane.")
            return
        print(f"Filtering for lanes: {lanes_to_check}")
        lane_moms = df[df['lane'].isin(lanes_to_check)]['mother_id'].unique()
        survivors = np.intersect1d(survivors, lane_moms)
        
    if len(survivors) == 0:
        print(f"No survivors found in the specified lanes.")
        return

    print(f"Found {len(survivors)} survivors to plot.")

    # 4. Setup Output
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"survivor_report_{timestamp}.pdf"

    # 5. Selection Logic (Max N per lane)
    final_list = []
    if 'lane' in df.columns:
        meta = df[['mother_id', 'lane']].drop_duplicates().set_index('mother_id')
        present_lanes = meta.loc[survivors, 'lane'].unique()
        for lane in sorted(present_lanes):
            moms_in_lane = sorted(meta[meta['lane'] == lane].index.intersection(survivors))
            if len(moms_in_lane) > max_IDs_per_lane:
                final_list.extend(moms_in_lane[:max_IDs_per_lane])
            else:
                final_list.extend(moms_in_lane)
    else:
        # Fallback if no lane column
        if len(survivors) > max_IDs_per_lane:
            final_list = sorted(survivors)[:max_IDs_per_lane]
        else:
            final_list = sorted(survivors)

    if not final_list:
        print("No IDs selected.")
        return

    # Global X Limit
    x_max = df['time'].max()
    x_limit = np.ceil(x_max / 120) * 120
    
    print(f"Generating PDF: {filename} ({len(final_list)} plots)...")

    # 6. Plotting Loop
    with PdfPages(filename) as pdf:
        for mid in tqdm(final_list, desc="Plotting Survivors"):
            trace = df[df['mother_id'] == mid].sort_values('time')
            t = trace['time'].values
            
            # Data Extraction
            y_raw = trace['log_length_cleaned'].values if 'log_length_cleaned' in trace.columns else None
            y_sm  = trace['log_length_smoothed'].values
            
            d1_sm = trace['growth_rate_smoothed'].values if 'growth_rate_smoothed' in trace.columns else None
            
            d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
            d2_sm = trace[d2_col].values if d2_col in trace.columns else None
            
            divs = trace["division_event"].astype(float).fillna(0).astype(bool).values if "division_event" in trace.columns else np.zeros(len(t), dtype=bool)

            # --- Layout: 3 Vertical Panels ---
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
            plt.subplots_adjust(top=0.92, bottom=0.10, left=0.12, right=0.95, hspace=0.3)
            
            ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
            
            # Title Construction
            title_parts = [f"Survivor: Mother {mid}"]
            if 'lane' in trace.columns:
                title_parts.append(f"Lane {trace['lane'].iloc[0]}")
            if 'condition' in trace.columns:
                title_parts.append(f"{trace['condition'].iloc[0]}")
            
            fig.suptitle(" | ".join(title_parts), fontweight='bold', fontsize=12)
            
            # --- Panel 1: Log Length ---
            if y_raw is not None:
                # Plot raw points faintly behind
                ax_len.scatter(t, y_raw, c='gray', s=10, alpha=0.3, zorder=1)
                
            ax_len.plot(t, y_sm, c='#d62728', lw=2, label='Smoothed', zorder=2)
            
            if divs.any() and y_raw is not None:
                valid_divs = divs & np.isfinite(y_raw)
                ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=30, marker='o', edgecolors='k', zorder=3, label='Division')
                
            ax_len.set_ylabel(r"ln[Length ($\mu$m)]", fontweight='bold')
            ax_len.set_ylim(0.25, 3.25) # Standardize limits if appropriate, or remove

            # --- Panel 2: Growth Rate ---
            if d1_sm is not None:
                ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2)
                ax_d1.set_ylabel(r"Growth Rate ($\times 10^{-2} min^{-1}$)", fontweight='bold')
                ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
                ax_d1.set_ylim(-0.001, 0.025)

            # --- Panel 3: Acceleration ---
            if d2_sm is not None:
                ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                ax_d2.axhline(0, c='k', ls='-', lw=0.8, alpha=0.5)
                ax_d2.set_ylabel(r"Accel ($\times 10^{-4} min^{-2}$)", fontweight='bold')
                ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
                ax_d2.set_ylim(-0.00025, 0.00025)
            
            ax_d2.set_xlabel("Time (min)", fontweight='bold')

            # --- Common Formatting ---
            for ax in axes:
                # Treatment Shading
                if treatment_start is not None and treatment_end is not None:
                    ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.15, zorder=0)
                    
                    # Label treatment on top graph
                    if ax == ax_len:
                        treat_center = (treatment_start + treatment_end) / 2
                        y_min, y_max = ax.get_ylim()
                        text_y = y_max - (0.05 * (y_max - y_min))
                        ax.text(treat_center, text_y, "Treatment", color='darkred', 
                                ha='center', va='top', fontweight='bold', fontsize=8, alpha=0.8)

                ax.set_xlim(0, x_limit)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
                ax.grid(True, which='major', ls='-', alpha=0.4)
                ax.grid(True, which='minor', ls=':', alpha=0.2)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Done. Report saved to {os.path.abspath(filename)}")

def generate_non_survivor_pdf(
    df: pd.DataFrame, 
    treatment_start: float, 
    treatment_end: float,
    lanes_to_check: list = None, 
    max_IDs_per_lane: int = 50,
    filename: str = None
):
    """
    Generates a PDF report for 'Non-Survivors' (growing_before=True AND growing_after=False).
    Matches the 3-panel layout (Length, Rate, Accel) of the slowdown reports.
    """
    if df is None: return
    
    print(f"{'='*40}\n GENERATING NON-SURVIVOR PDF \n{'='*40}")

    # 1. Validation
    req_cols = ['mother_id', 'growing_before', 'growing_after', 'time', 'log_length_smoothed']
    for c in req_cols:
        if c not in df.columns:
            print(f"Error: Required column '{c}' missing from DataFrame.")
            return

    # 2. Identify Non-Survivors
    non_survivor_mask = (df['growing_before'] == True) & (df['growing_after'] == False)
    non_survivors = df[non_survivor_mask]['mother_id'].unique()
    
    if len(non_survivors) == 0:
        print("No non-survivors found.")
        return

    # 3. Filter by Lane
    if lanes_to_check is not None:
        if 'lane' not in df.columns:
            print("Error: 'lane' column missing, cannot filter by lane.")
            return
        print(f"Filtering for lanes: {lanes_to_check}")
        lane_moms = df[df['lane'].isin(lanes_to_check)]['mother_id'].unique()
        non_survivors = np.intersect1d(non_survivors, lane_moms)
        
    if len(non_survivors) == 0:
        print(f"No non-survivors found in the specified lanes.")
        return

    print(f"Found {len(non_survivors)} non-survivors to plot.")

    # 4. Setup Output
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"non_survivor_report_{timestamp}.pdf"

    # 5. Selection Logic (Max N per lane)
    final_list = []
    if 'lane' in df.columns:
        meta = df[['mother_id', 'lane']].drop_duplicates().set_index('mother_id')
        present_lanes = meta.loc[non_survivors, 'lane'].unique()
        for lane in sorted(present_lanes):
            moms_in_lane = sorted(meta[meta['lane'] == lane].index.intersection(non_survivors))
            if len(moms_in_lane) > max_IDs_per_lane:
                final_list.extend(moms_in_lane[:max_IDs_per_lane])
            else:
                final_list.extend(moms_in_lane)
    else:
        # Fallback if no lane column
        if len(non_survivors) > max_IDs_per_lane:
            final_list = sorted(non_survivors)[:max_IDs_per_lane]
        else:
            final_list = sorted(non_survivors)

    if not final_list:
        print("No IDs selected.")
        return

    # Global X Limit
    x_max = df['time'].max()
    x_limit = np.ceil(x_max / 120) * 120
    
    print(f"Generating PDF: {filename} ({len(final_list)} plots)...")

    # 6. Plotting Loop
    with PdfPages(filename) as pdf:
        for mid in tqdm(final_list, desc="Plotting Non-Survivors"):
            trace = df[df['mother_id'] == mid].sort_values('time')
            t = trace['time'].values
            
            # Data Extraction
            y_raw = trace['log_length_cleaned'].values if 'log_length_cleaned' in trace.columns else None
            y_sm  = trace['log_length_smoothed'].values
            
            d1_sm = trace['growth_rate_smoothed'].values if 'growth_rate_smoothed' in trace.columns else None
            
            d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
            d2_sm = trace[d2_col].values if d2_col in trace.columns else None
            
            divs = trace["division_event"].astype(float).fillna(0).astype(bool).values if "division_event" in trace.columns else np.zeros(len(t), dtype=bool)

            # --- Layout: 3 Vertical Panels ---
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
            plt.subplots_adjust(top=0.92, bottom=0.10, left=0.12, right=0.95, hspace=0.3)
            
            ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
            
            # Title Construction
            title_parts = [f"Non-Survivor: Mother {mid}"]
            if 'lane' in trace.columns:
                title_parts.append(f"Lane {trace['lane'].iloc[0]}")
            if 'condition' in trace.columns:
                title_parts.append(f"{trace['condition'].iloc[0]}")
            
            fig.suptitle(" | ".join(title_parts), fontweight='bold', fontsize=12)
            
            # --- Panel 1: Log Length ---
            if y_raw is not None:
                # Plot raw points faintly behind
                ax_len.scatter(t, y_raw, c='gray', s=10, alpha=0.3, zorder=1)
                
            ax_len.plot(t, y_sm, c='#d62728', lw=2, label='Smoothed', zorder=2)
            
            if divs.any() and y_raw is not None:
                valid_divs = divs & np.isfinite(y_raw)
                ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=30, marker='o', edgecolors='k', zorder=3, label='Division')
                
            ax_len.set_ylabel(r"ln[Length ($\mu$m)]", fontweight='bold')
            ax_len.set_ylim(0.25, 3.25) # Standardize limits if appropriate, or remove

            # --- Panel 2: Growth Rate ---
            if d1_sm is not None:
                ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2)
                ax_d1.set_ylabel(r"Growth Rate ($\times 10^{-2} min^{-1}$)", fontweight='bold')
                ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
                ax_d1.set_ylim(-0.001, 0.025)

            # --- Panel 3: Acceleration ---
            if d2_sm is not None:
                ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                ax_d2.axhline(0, c='k', ls='-', lw=0.8, alpha=0.5)
                ax_d2.set_ylabel(r"Accel ($\times 10^{-4} min^{-2}$)", fontweight='bold')
                ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
                ax_d2.set_ylim(-0.00025, 0.00025)
            
            ax_d2.set_xlabel("Time (min)", fontweight='bold')

            # --- Common Formatting ---
            for ax in axes:
                # Treatment Shading
                if treatment_start is not None and treatment_end is not None:
                    ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.15, zorder=0)
                    
                    # Label treatment on top graph
                    if ax == ax_len:
                        treat_center = (treatment_start + treatment_end) / 2
                        y_min, y_max = ax.get_ylim()
                        text_y = y_max - (0.05 * (y_max - y_min))
                        ax.text(treat_center, text_y, "Treatment", color='darkred', 
                                ha='center', va='top', fontweight='bold', fontsize=8, alpha=0.8)

                ax.set_xlim(0, x_limit)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
                ax.grid(True, which='major', ls='-', alpha=0.4)
                ax.grid(True, which='minor', ls=':', alpha=0.2)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Done. Report saved to {os.path.abspath(filename)}")


# Lag time calculation code
def analyze_global_growth_thresholds(df: pd.DataFrame, 
                                     percentages: list = [5, 10, 15, 20], 
                                     n_plot: int = 50, 
                                     treatment_start: float = None, 
                                     treatment_end: float = None,
                                     filename: str = None,
                                     require_growing_before: bool = True,
                                     require_growing_after: bool = False,
                                     start_time: float = 0.0):
    """
    Calculates global growth rate percentiles (top X%) and plots them 
    on random traces for visual inspection.

    Args:
        df: DataFrame containing 'growth_rate_smoothed'.
        percentages: List of 'top X%' values to test (e.g., [5, 10] means 95th and 90th percentiles).
        n_plot: Number of example mothers to plot PER LANE (ordered by mother_id).
        treatment_start: Start of treatment (min) for shading.
        treatment_end: End of treatment (min) for shading.
        filename: Output filename.
        require_growing_before: If True, only use mothers with growing_before == True for threshold calc.
        require_growing_after: If True, only use mothers with growing_after == True for threshold calc.
        start_time: Time (min) to start collecting data for the global max growth rate calculation.
    """
    if df is None: return
    if 'growth_rate_smoothed' not in df.columns:
        print("Error: 'growth_rate_smoothed' column not found.")
        return

    print(f"{'='*40}\n ANALYZING GLOBAL GROWTH THRESHOLDS \n{'='*40}")

    # --- 1. Filter Population for Threshold Calculation ---
    eligible_df = df.copy()
    if require_growing_before and 'growing_before' in eligible_df.columns:
        eligible_df = eligible_df[eligible_df['growing_before'] == True]
    if require_growing_after and 'growing_after' in eligible_df.columns:
        eligible_df = eligible_df[eligible_df['growing_after'] == True]

    valid_mids = eligible_df['mother_id'].unique()
    
    if len(valid_mids) == 0:
        print("Error: No mothers found matching the growth criteria.")
        return

    # Apply start_time filter for the calculation
    df_calc = eligible_df[eligible_df['time'] >= start_time]

    # Calculate Global Percentiles
    all_rates = df_calc['growth_rate_smoothed'].dropna().values
    
    if len(all_rates) == 0:
        print(f"Error: No valid growth rates found after {start_time} min.")
        return
    
    thresholds = {}
    print(f"Calculated Global Thresholds (Top X%) using eligible mothers after {start_time} min:")
    for p in percentages:
        # Top 5% = 95th percentile
        perc_val = 100 - p
        thresh = np.percentile(all_rates, perc_val)
        thresholds[f"Top {p}%"] = thresh
        print(f"  Top {p}% (>{perc_val}th perc): {thresh:.5f} [min^-1] ({thresh*100:.2f} x10^-2)")

    # --- 2. Select Random Mothers (Sampling PER LANE from eligible pool) ---
    selected_moms = []
    if n_plot > 0:
        lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df.columns), None)
        
        if lane_col:
            for lane, group in eligible_df.groupby(lane_col):
                lane_moms = sorted(group['mother_id'].unique())
                if len(lane_moms) > n_plot:
                    selected_moms.extend(sorted(random.sample(lane_moms, n_plot)))
                else:
                    selected_moms.extend(lane_moms)
        else:
            # Fallback if no lane column is found
            all_moms = sorted(valid_mids)
            if len(all_moms) > n_plot:
                selected_moms = sorted(random.sample(all_moms, n_plot))
            else:
                selected_moms = all_moms
        
    # --- 3. Setup PDF ---
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"growth_threshold_test_{timestamp}.pdf"
        
    # Colormap for thresholds
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(percentages)))
    
    global_max_time = df["time"].max() if "time" in df.columns else 600
    x_limit = np.ceil(global_max_time / 120) * 120

    print(f"\nGenerating PDF report for {len(selected_moms)} total traces: {filename}...")
    
    with PdfPages(filename) as pdf:
        for mid in tqdm(selected_moms, desc="Plotting"):
            
            trace = df[df['mother_id'] == mid].sort_values('time')
            t = trace['time'].values
            
            # Data
            y_sm = trace["log_length_smoothed"].values if "log_length_smoothed" in trace.columns else None
            d1_sm = trace["growth_rate_smoothed"].values if "growth_rate_smoothed" in trace.columns else None
            
            d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
            d2_sm = trace[d2_col].values if d2_col in trace.columns else None

            # Setup Plot
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
            plt.subplots_adjust(top=0.92, bottom=0.10, left=0.1, right=0.95, hspace=0.3)
            ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
            
            # Title
            lane_info = f"Lane: {trace['lane'].iloc[0]}" if 'lane' in trace.columns else ""
            fig.suptitle(f"Mother ID: {mid} | {lane_info}", fontweight='bold')
            
            # Plot 1: Length
            if y_sm is not None: 
                ax_len.plot(t, y_sm, c='#d62728', lw=2)
            ax_len.set_ylabel("ln(Length)")
            
            # Plot 2: Growth Rate (WITH THRESHOLDS)
            if d1_sm is not None:
                ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2, label='Rate')
                
                # Add Threshold Lines
                for i, (label, val) in enumerate(thresholds.items()):
                    ax_d1.axhline(val, color=colors[i], linestyle='--', alpha=0.8, lw=1.5, 
                                  label=f"{label}: {val*100:.1f}")
                
                ax_d1.set_ylabel(r"Growth Rate ($\times 10^{-2}$)")
                ax_d1.legend(loc='upper right', fontsize=8, framealpha=0.9)
                ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))

            # Plot 3: Acceleration
            if d2_sm is not None:
                ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                ax_d2.axhline(0, c='k', ls='-', lw=0.5)
                ax_d2.set_ylabel(r"Accel ($\times 10^{-4}$)")
                ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))

            # Common Formatting
            for ax in axes:
                if treatment_start is not None and treatment_end is not None:
                    ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1)
                
                ax.set_xlim(0, x_limit)
                ax.grid(True, ls=':', alpha=0.4)
                
            ax_d2.set_xlabel("Time (min)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved to: {os.path.abspath(filename)}")
    return thresholds

def calculate_accumulated_lag(df: pd.DataFrame, 
                              top_percentile: float = 10.0, 
                              allow_recovery: bool = True,
                              n_plot: int = 50,
                              treatment_start: float = None, 
                              treatment_end: float = None,
                              filename: str = None,
                              accumulation_start_time: float = 0.0,
                              require_growing_before: bool = True,
                              require_growing_after: bool = False,
                              start_time: float = 0.0):
    """
    Calculates 'Accumulated Lag' based on the area between the global max growth rate 
    and the actual growth rate curve.

    Args:
        top_percentile: Defines Global Max GR (e.g., 10.0 = Top 10% / 90th percentile).
        allow_recovery: If True, rates > Global Max subtract from accumulated lag.
                        If False, rates > Global Max are treated as 0 deficit.
        n_plot: Number of example mothers to plot PER LANE (ordered by mother_id).
        treatment_start: Used for plot shading.
        treatment_end: Used for plot shading.
        accumulation_start_time: Time (min) to begin accumulating lag. Deficits before 
                                 this time are ignored (set to 0).
        require_growing_before: If True, only use mothers with growing_before == True for global max rate.
        require_growing_after: If True, only use mothers with growing_after == True for global max rate.
        start_time: Time (min) to start collecting data for the global max growth rate calculation.
    
    Adds columns: 'accum_lag_current', 'accum_lag_total'.
    """
    if df is None: return None
    if 'growth_rate_smoothed' not in df.columns:
        print("Error: 'growth_rate_smoothed' column not found.")
        return None

    print(f"{'='*40}\n CALCULATING ACCUMULATED LAG \n{'='*40}")

    # --- 1. Filter Population for Global Max Rate Calculation ---
    eligible_df = df.copy()
    if require_growing_before and 'growing_before' in eligible_df.columns:
        eligible_df = eligible_df[eligible_df['growing_before'] == True]
    if require_growing_after and 'growing_after' in eligible_df.columns:
        eligible_df = eligible_df[eligible_df['growing_after'] == True]

    valid_mids = eligible_df['mother_id'].unique()
    if len(valid_mids) == 0:
        print("Error: No mothers found matching the growth criteria.")
        return df

    # Apply start_time filter for the calculation window
    df_calc = eligible_df[eligible_df['time'] >= start_time]

    # Calculate Global Max Growth Rate
    all_rates = df_calc['growth_rate_smoothed'].dropna().values
    if len(all_rates) == 0:
        print(f"Error: No valid growth rates found after {start_time} min for threshold calculation.")
        return df

    perc_threshold = 100 - top_percentile
    global_max_gr = np.percentile(all_rates, perc_threshold)
    
    print(f"Global Max Growth Rate (Top {top_percentile}% after {start_time}m): {global_max_gr:.6f} min^-1")
    print(f"Lag Recovery (subtract when rate > max): {allow_recovery}")
    print(f"Accumulation Start Time: {accumulation_start_time} min")

    # --- 2. Process Each Mother (calculates for full DataFrame) ---
    def _calc_lag_for_group(group):
        group = group.sort_values('time')
        t = group['time'].values
        gr = group['growth_rate_smoothed'].values
        
        # Handle NaNs: fill with global_max_gr so they contribute 0 deviation
        gr_filled = np.nan_to_num(gr, nan=global_max_gr)
        
        # Calculate Deficit
        if allow_recovery:
            # Recovery Mode: Simple difference. 
            # If GR > Max, deficit is negative (reduces lag).
            deficit = global_max_gr - gr_filled
        else:
            # Strict Mode: Only accumulate positive lag. 
            # If GR > Max, deficit is 0.
            deficit = np.maximum(0, global_max_gr - gr_filled)
        
        # Zero out deficit before accumulation start time
        if accumulation_start_time is not None:
            deficit[t < accumulation_start_time] = 0.0

        # Integrate Deficit over Time
        if len(t) > 1:
            cum_area = cumulative_trapezoid(deficit, t, initial=0)
        else:
            cum_area = np.zeros_like(deficit)
            
        # Calculate Lag (Minutes)
        cum_lag = cum_area / global_max_gr
        
        return pd.DataFrame({
            'accum_lag_current': cum_lag,
            'accum_lag_total': cum_lag[-1]
        }, index=group.index)

    print("Computing lag for all mothers...")
    lag_cols = df.groupby('mother_id', group_keys=False).apply(_calc_lag_for_group)
    
    # Assign back
    df['accum_lag_current'] = lag_cols['accum_lag_current']
    df['accum_lag_total'] = lag_cols['accum_lag_total']
    
    # --- 3. Visualization ---
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_str = "with_recovery" if allow_recovery else "no_recovery"
        filename = f"accumulated_lag_{rec_str}_top{int(top_percentile)}pct_start{int(accumulation_start_time)}_{timestamp}.pdf"

    # Select and Sort mothers (Sampling PER LANE from eligible mothers)
    selected_moms = []
    if n_plot > 0:
        lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df.columns), None)
        
        if lane_col:
            for lane, group in eligible_df.groupby(lane_col):
                lane_moms = sorted(group['mother_id'].unique())
                if len(lane_moms) > n_plot:
                    selected_moms.extend(sorted(random.sample(lane_moms, n_plot)))
                else:
                    selected_moms.extend(lane_moms)
        else:
            all_moms = sorted(valid_mids)
            if len(all_moms) > n_plot:
                selected_moms = sorted(random.sample(all_moms, n_plot))
            else:
                selected_moms = all_moms
        
    global_max_time = df["time"].max() if "time" in df.columns else 600
    x_limit = np.ceil(global_max_time / 120) * 120

    if selected_moms:
        print(f"Generating PDF report for {len(selected_moms)} total traces: {filename}")
        
        with PdfPages(filename) as pdf:
            for mid in tqdm(selected_moms, desc="Plotting Lag"):
                
                trace = df[df['mother_id'] == mid].sort_values('time')
                t = trace['time'].values
                
                y_sm = trace["log_length_smoothed"].values if "log_length_smoothed" in trace.columns else None
                gr = trace["growth_rate_smoothed"].values if "growth_rate_smoothed" in trace.columns else None
                
                d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
                d2_sm = trace[d2_col].values if d2_col in trace.columns else None
                
                lag_total = trace['accum_lag_total'].iloc[0] if 'accum_lag_total' in trace.columns else 0.0
    
                # Setup 3-Panel Plot
                fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
                plt.subplots_adjust(top=0.92, bottom=0.10, left=0.1, right=0.95, hspace=0.3)
                ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
                
                # Title
                lane_info = f"Lane: {trace['lane'].iloc[0]}" if 'lane' in trace.columns else ""
                fig.suptitle(f"Mother ID: {mid} | {lane_info} | Accum Lag: {lag_total:.1f} mins", fontweight='bold')
                
                # Panel 1: Length
                if y_sm is not None: 
                    ax_len.plot(t, y_sm, c='#d62728', lw=2)
                ax_len.set_ylabel("ln(Length)")
                ax_len.grid(True, ls=':', alpha=0.4)
                
                # Panel 2: Growth Rate
                if gr is not None:
                    ax_d1.plot(t, gr, c='#1f77b4', lw=2, label='Growth Rate')
                    ax_d1.axhline(global_max_gr, color='black', linestyle='--', lw=1.5, label=f'Global Max ({global_max_gr*100:.2f})')
                    
                    gr_filled = np.nan_to_num(gr, nan=global_max_gr)
                    
                    # Create mask for "After Start Time" shading
                    mask_active = (t >= accumulation_start_time)
    
                    # Shade Lag (Red) - Only after accumulation starts
                    ax_d1.fill_between(t, global_max_gr, gr_filled, 
                                       where=(mask_active & (gr_filled < global_max_gr)), 
                                       color='red', alpha=0.3, interpolate=True, 
                                       label='Lag (+)')
                    
                    # Shade Recovery (Green) - Only after accumulation starts, if enabled
                    if allow_recovery:
                        ax_d1.fill_between(t, global_max_gr, gr_filled,
                                           where=(mask_active & (gr_filled > global_max_gr)),
                                           color='green', alpha=0.3, interpolate=True,
                                           label='Recovery (-)')
                    
                    # Visual Indicator for Start Time
                    if accumulation_start_time > 0:
                        ax_d1.axvline(accumulation_start_time, color='gray', linestyle=':', lw=2, label="Accum Start")
    
                    ax_d1.set_ylabel(r"Growth Rate ($\times 10^{-2}$)")
                    ax_d1.legend(loc='upper right', fontsize=8, framealpha=0.9)
                    ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
                    
                    # Limits
                    valid_gr = gr[np.isfinite(gr)]
                    y_max_data = np.max(valid_gr) if len(valid_gr) > 0 else 0.03
                    upper_lim = max(y_max_data, global_max_gr) * 1.2
                    if not np.isfinite(upper_lim): upper_lim = 0.05
                    ax_d1.set_ylim(-0.005, upper_lim)
                    ax_d1.grid(True, ls=':', alpha=0.4)
    
                # Panel 3: Acceleration
                if d2_sm is not None:
                    ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                    ax_d2.axhline(0, c='k', ls='-', lw=0.5)
                    ax_d2.set_ylabel(r"Accel ($\times 10^{-4}$)")
                    ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
                    ax_d2.grid(True, ls=':', alpha=0.4)
                    
                ax_d2.set_xlabel("Time (min)")
    
                # Common Formatting
                for ax in axes:
                    if treatment_start is not None and treatment_end is not None:
                        ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1)
                    ax.set_xlim(0, x_limit)
    
                pdf.savefig(fig)
                plt.close(fig)
    
        print(f"Done. Columns 'accum_lag_current'/'accum_lag_total' added. PDF saved to {os.path.abspath(filename)}")
    else:
        print("Done. Columns 'accum_lag_current'/'accum_lag_total' added. (No plots generated).")

    return df


# temp data analysis functions
def plot_slowdown_survivor_growth_rates(df: pd.DataFrame, 
                                        check_time: float = 120.0, 
                                        treatment_start: float = None, 
                                        treatment_end: float = None,
                                        require_growing_before: bool = True,
                                        require_growing_after: bool = True,
                                        mode_round_interval: float = 0.0005):
    """
    Plots Mean, Median, and Mode of growth_rate_smoothed vs time for cells 
    that exhibit a slowdown at 'check_time'.
    
    Layout: 2 Columns of plots for individual conditions.
    """
    if df is None:
        print("Error: DataFrame is None.")
        return

    print(f"{'='*40}\n PLOTTING SLOWDOWN SURVIVOR TRACES \n{'='*40}")

    # 1. Identify Columns
    lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df.columns), None)
    cond_col = next((c for c in ['condition', 'Condition'] if c in df.columns), None)

    if not lane_col:
        print("Error: Could not find a 'lane' column.")
        return
    
    # 2. Filter Base Population
    candidates = df.copy()
    if require_growing_before and 'growing_before' in candidates.columns:
        candidates = candidates[candidates['growing_before'] == True]
    if require_growing_after and 'growing_after' in candidates.columns:
        candidates = candidates[candidates['growing_after'] == True]

    if candidates.empty:
        print("No cells matched the growth criteria.")
        return

    # 3. Filter for Slowdown Survivors
    target_rows = candidates[
        (candidates['time'] >= check_time - 2.0) & 
        (candidates['time'] <= check_time + 2.0) &
        (candidates['slowdown'] == True)
    ]
    
    slowdown_mids = target_rows['mother_id'].unique()
    if len(slowdown_mids) == 0:
        print(f"No mothers found with an active slowdown at t={check_time}.")
        return

    print(f"Found {len(slowdown_mids)} Slowdown Cells out of {candidates['mother_id'].nunique()} Base Candidates.")
    
    # 4. Extract Traces
    slowdown_df = df[df['mother_id'].isin(slowdown_mids)].copy()
    
    # 5. Grouping
    if cond_col:
        groups = slowdown_df[[lane_col, cond_col]].drop_duplicates().sort_values(lane_col)
    else:
        groups = slowdown_df[[lane_col]].drop_duplicates().sort_values(lane_col)
        groups['condition'] = "Unknown"

    # Helper for mode
    def calculate_custom_mode(series):
        rounded = (series / mode_round_interval).round() * mode_round_interval
        m = rounded.mode()
        return m.iloc[0] if not m.empty else np.nan

    # --- SETUP SUBPLOTS ---
    n_groups = len(groups)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)
    
    # Create the figure for the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = np.array(axes).flatten() # Flatten to 1D array for easy indexing

    sns.set_theme(style="ticks", context="talk")
    summary_data = []

    # 6. Iterate and Plot
    for i, (_, row) in enumerate(groups.iterrows()):
        ax = axes[i]
        lane_val = row[lane_col]
        cond_val = row[cond_col] if cond_col else "Unknown"
        
        # Filter Data
        if cond_col:
            group_total = candidates[(candidates[lane_col] == lane_val) & (candidates[cond_col] == cond_val)]
            group_slow = slowdown_df[(slowdown_df[lane_col] == lane_val) & (slowdown_df[cond_col] == cond_val)]
        else:
            group_total = candidates[candidates[lane_col] == lane_val]
            group_slow = slowdown_df[slowdown_df[lane_col] == lane_val]
            
        if group_slow.empty: 
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
        
        # Stats
        n_total = group_total['mother_id'].nunique()
        n_slow = group_slow['mother_id'].nunique()
        pct = (n_slow / n_total * 100.0) if n_total > 0 else 0.0
        
        # Calculate Aggregates
        grouped_time = group_slow.groupby('time')['growth_rate_smoothed']
        mean_trace = grouped_time.mean()
        median_trace = grouped_time.median()
        mode_trace = grouped_time.apply(calculate_custom_mode)

        summary_data.append({
            'label': f"Lane {lane_val}: {cond_val}",
            'n_str': f"(n={n_slow})",
            'mean': mean_trace,
            'median': median_trace,
            'mode': mode_trace
        })

        # --- Plot on Subplot (ax) ---
        # Individual Traces
        sns.lineplot(
            data=group_slow,
            x="time",
            y="growth_rate_smoothed",
            units="mother_id",
            estimator=None,
            color="gray",
            alpha=0.3,
            lw=1,
            ax=ax
        )
        
        # Mean Trace Overlay
        ax.plot(mean_trace.index, mean_trace.values, color="#1f77b4", lw=3, label="Mean Growth Rate")
        
        # Shading
        if treatment_start is not None and treatment_end is not None:
            ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1)
        
        ax.axvline(check_time, color='green', linestyle='--', alpha=0.8)

        # Labels & Title
        criteria_str = []
        if require_growing_before: criteria_str.append("GrowBefore")
        if require_growing_after: criteria_str.append("GrowAfter")
        filter_lbl = "+".join(criteria_str) if criteria_str else "All"

        title_str = (f"Lane {lane_val}: {cond_val}\n"
                     f"Pop: {filter_lbl} | n={n_slow}/{n_total} ({pct:.1f}%)")
        ax.set_title(title_str, fontweight="bold", fontsize=12)
        
        ax.set_ylabel(r"Growth Rate ($\times 10^{-2}$)")
        ax.set_xlabel("Time (min)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
        
        # Only add legend to the first plot to save space, or all if preferred
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=10)
        
        ax.grid(True, linestyle=":", alpha=0.4)
        sns.despine(ax=ax)

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

    # 7. Generate Summary Plots (Mean, Median, Mode)
    # These remain full-width for detailed comparison
    if summary_data:
        metrics = [
            ('mean', 'Mean Growth Rate'),
            ('median', 'Median Growth Rate'),
            ('mode', f'Mode Growth Rate (Rounded to {mode_round_interval})')
        ]

        palette = sns.color_palette("tab10", len(summary_data))
        
        for metric_key, metric_title in metrics:
            plt.figure(figsize=(12, 7))
            
            for i, item in enumerate(summary_data):
                series = item[metric_key]
                label = f"{item['label']} {item['n_str']}"
                plt.plot(series.index, series.values, lw=2.5, color=palette[i], label=label)
            
            if treatment_start is not None and treatment_end is not None:
                plt.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1, label='Treatment')
                
            plt.axvline(check_time, color='green', linestyle='--', alpha=0.8, label=f'Selection (t={check_time})')
            
            full_title = f"Summary: {metric_title} of Slowdown Cells"
            if require_growing_after: full_title += " (Survivors)"
            
            plt.title(full_title, fontweight="bold")
            plt.ylabel(r"Growth Rate ($\times 10^{-2} min^{-1}$)")
            plt.xlabel("Time (min)")
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            
            plt.legend(loc="upper right", frameon=True, fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.4)
            sns.despine()
            plt.tight_layout()
            plt.show()

def generate_slowdown_survivor_pdf(df: pd.DataFrame, 
                                   slowdown_params: dict, 
                                   treatment_start: float, 
                                   treatment_end: float, 
                                   check_time: float = 120.0,
                                   filename: str = None,
                                   require_growing_before: bool = True,
                                   require_growing_after: bool = True):
    """
    Generates a PDF report for cells that exhibit a slowdown at 'check_time',
    filtered by growth status.
    
    Args:
        df: Input DataFrame.
        slowdown_params: Params for visualizing detection logic.
        require_growing_before: Filter for growing_before == True.
        require_growing_after: Filter for growing_after == True.
    """
    if df is None: return

    print(f"{'='*40}\n GENERATING SLOWDOWN CELL PDF \n{'='*40}")
    
    # 1. Generate Filename
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_folder = os.path.basename(os.getcwd())
        # Append filter status to filename for clarity
        tag = "survivors" if require_growing_after else "all_slowdowns"
        filename = f"{tag}_slowdown_traces_t{int(check_time)}_{current_folder}_{timestamp}.pdf"

    # 2. Identify Lane/Condition Columns
    lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df.columns), None)
    cond_col = next((c for c in ['condition', 'Condition'] if c in df.columns), None)

    # 3. Filter Candidates
    candidates = df.copy()

    if require_growing_before:
        if 'growing_before' in candidates.columns:
            candidates = candidates[candidates['growing_before'] == True]
            
    if require_growing_after:
        if 'growing_after' in candidates.columns:
            candidates = candidates[candidates['growing_after'] == True]

    # 4. Filter for Slowdown at Check Time
    target_rows = candidates[
        (candidates['time'] >= check_time - 2.0) & 
        (candidates['time'] <= check_time + 2.0) &
        (candidates['slowdown'] == True)
    ]
    
    final_mids = target_rows['mother_id'].unique()
    n_moms = len(final_mids)
    
    if n_moms == 0:
        print(f"No cells found matching criteria at t={check_time}.")
        return

    print(f"Found {n_moms} matching cells. Generating PDF...")

    # 5. Setup Global Limits
    global_tps = df["time"].unique()
    global_max_time = global_tps.max() if len(global_tps) > 0 else 600
    x_limit = np.ceil(global_max_time / 120) * 120

    # 6. Generate PDF
    with PdfPages(filename) as pdf:
        for mid in tqdm(sorted(final_mids), desc="Generating Pages"):
            
            trace = df[df['mother_id'] == mid].sort_values('time')
            t = trace['time'].values
            
            y_raw = trace["log_length_cleaned"].values if "log_length_cleaned" in trace.columns else None
            y_sm = trace["log_length_smoothed"].values if "log_length_smoothed" in trace.columns else None
            d1_sm = trace["growth_rate_smoothed"].values if "growth_rate_smoothed" in trace.columns else None
            
            d2_col = "second_deriv_smoothed" if "second_deriv_smoothed" in trace.columns else "2nd_deriv_smoothed"
            d2_sm = trace[d2_col].values if d2_col in trace.columns else None
            
            divs = trace["division_event"].astype(float).fillna(0).astype(bool).values if "division_event" in trace.columns else np.zeros(len(t), dtype=bool)

            events = []
            if d1_sm is not None and d2_sm is not None:
                events = _detect_and_measure_slowdowns(t, d1_sm, d2_sm, params=slowdown_params)
                events = [ev for ev in events if ev.get('is_valid', True)]

            # --- PLOTTING ---
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
            plt.subplots_adjust(top=0.95, bottom=0.10, left=0.1, right=0.95, hspace=0.3)
            
            ax_len, ax_d1, ax_d2 = axes[0], axes[1], axes[2]
            
            # Construct Title
            title_parts = [f"Mother ID: {mid}"]
            if lane_col:
                lid = trace[lane_col].iloc[0]
                title_parts.append(f"Lane: {lid}")
            if cond_col:
                cval = trace[cond_col].iloc[0]
                title_parts.append(f"{cval}")
            
            title_parts.append(f"[Slowdown @ {check_time}min]")
            title_str = " | ".join(title_parts)
            
            # Plot 1: Log Length
            if y_sm is not None: ax_len.plot(t, y_sm, c='#d62728', lw=2)
            if divs.any() and y_raw is not None:
                valid_divs = divs & np.isfinite(y_raw)
                ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=30, zorder=3, edgecolors='k')
            
            ax_len.set_ylabel(r"ln[Length ($\mu$m)]", fontweight='bold')
            ax_len.set_title(title_str, fontweight='bold', fontsize=11)
            ax_len.set_ylim(0.25, 3.25)

            # Plot 2: Growth Rate
            if d1_sm is not None:
                ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2)
                ax_d1.set_ylabel(r"Growth Rate ($\times 10^{-2} min^{-1}$)", fontweight='bold')
                ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
                
                for ev in events:
                    mask = (t >= ev['start_t']) & (t <= ev['end_t'])
                    if not mask.any(): continue
                    
                    baseline = ev['baseline_gr']
                    if np.isfinite(baseline):
                        ax_d1.plot([ev['start_t'], ev['end_t']], [baseline, baseline], 'k--', lw=1)
                        ax_d1.fill_between(t, baseline, d1_sm, where=(mask & (d1_sm < baseline)), color='green', alpha=0.3, interpolate=True)
                        ax_d1.fill_between(t, baseline, d1_sm, where=(mask & (d1_sm > baseline)), color='gold', alpha=0.3, interpolate=True)
                        
                        # Add Check Time Marker
                        if ev['start_t'] <= check_time <= ev['end_t']:
                             ax_d1.axvline(check_time, color='purple', linestyle=':', lw=2, alpha=0.7)
            
            ax_d1.set_ylim(-0.001, 0.025)

            # Plot 3: Acceleration
            if d2_sm is not None:
                ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                ax_d2.axhline(0, c='k', ls='-', lw=0.8, alpha=0.5)
                ax_d2.set_ylabel(r"Accel ($\times 10^{-4} min^{-2}$)", fontweight='bold')
                ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*10000:.1f}'))
                
                for ev in events:
                    mask = (t >= ev['start_t']) & (t <= ev['end_t'])
                    if not mask.any(): continue
                    ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm < 0)), color='red', alpha=0.3, interpolate=True)
                    ax_d2.fill_between(t, 0, d2_sm, where=(mask & (d2_sm > 0)), color='blue', alpha=0.3, interpolate=True)
            
            ax_d2.set_ylim(-0.00025, 0.00025)

            # Common Formatting
            for ax in axes:
                ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.15, zorder=0)
                
                # Treatment Label
                y_min, y_max = ax.get_ylim()
                text_y = y_max - (0.01 * (y_max - y_min))
                treat_center = (treatment_start + treatment_end) / 2
                ax.text(treat_center, text_y, "Treatment", color='darkred', alpha=1.0, 
                        ha='center', va='top', fontweight='bold', fontsize=8, zorder=10)
                
                for ev in events:
                    ax.axvspan(ev['start_t'], ev['end_t'], color='green', alpha=0.1, zorder=0)
                
                ax.set_xlim(0, x_limit)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
                ax.grid(True, which='major', ls='-', alpha=0.4)
                ax.grid(True, which='minor', ls=':', alpha=0.2)
                ax.set_xlabel("Time (min)", fontweight='bold')

            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"PDF saved to: {os.path.abspath(filename)}")

def calculate_lane_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the survival percentage for each lane based on unique mothers.
    Definition: % of (growing_after == True) given (growing_before == True).
    N values represent unique mother_ids, not timepoints.
    """
    if df is None: return None
    
    # 1. Identify Lane and Condition columns
    lane_col = next((c for c in ['lane', 'lane_num', 'lane_id', 'position'] if c in df.columns), None)
    cond_col = next((c for c in ['condition', 'Condition'] if c in df.columns), None)
    
    if not lane_col:
        print("Error: Could not find a 'lane' column.")
        return None

    # 2. Filter: We only care about cells that were healthy BEFORE treatment
    if 'growing_before' not in df.columns:
        print("Error: 'growing_before' column missing.")
        return None
    
    # Filter for growing_before population
    base_pop = df[df['growing_before'] == True].copy()
    
    if base_pop.empty:
        print("No cells found with growing_before == True.")
        return None

    # 3. Deduplicate to get one row per mother_id
    # Since growing_before/growing_after are constant per mother, any row works.
    unique_moms = base_pop.drop_duplicates(subset=['mother_id'])

    # Ensure growing_after exists and is boolean (handle NaNs as False)
    if 'growing_after' not in unique_moms.columns:
        print("Warning: 'growing_after' missing. Assuming 0 survivors.")
        unique_moms['growing_after'] = False
    else:
        unique_moms['growing_after'] = unique_moms['growing_after'].fillna(False).astype(bool)

    # 4. Group by Lane (and Condition if available)
    group_cols = [lane_col]
    if cond_col: group_cols.append(cond_col)
    
    # Aggregation:
    # n_before = Count of rows (since we already deduplicated to unique mothers)
    # n_survivors = Sum of growing_after (True=1, False=0)
    stats = unique_moms.groupby(group_cols)['growing_after'].agg(
        n_before='count',
        n_survivors='sum'
    ).reset_index()
    
    # 5. Calculate Percentage
    stats['survival_%'] = (stats['n_survivors'] / stats['n_before']) * 100.0
    
    # 6. Formatting & Printing
    try:
        stats = stats.sort_values(lane_col)
    except: 
        pass
        
    # --- NEW: Dynamic Column Width Calculation ---
    cond_header = "Condition"
    if cond_col and not stats.empty:
        # Find the longest condition string, or default to header length if it's longer
        max_cond_len = stats[cond_col].astype(str).map(len).max()
        cond_width = max(len(cond_header), max_cond_len) + 2 # +2 for a little padding
    else:
        cond_width = len(cond_header) + 2

    # Calculate total table width for the horizontal lines
    # Lane(10) + |(3) + Cond(cond_width) + |(3) + N_b(10) + |(3) + N_a(10) + |(3) + Surv(10) = 42 + cond_width
    total_width = 42 + cond_width
    
    print(f"{'='*total_width}\n SURVIVAL SUMMARY (Unique Mothers) \n{'='*total_width}")
    print(f"{'Lane':<10} | {cond_header:<{cond_width}} | {'N(Before)':<10} | {'N(After)':<10} | {'Survival %'}")
    print("-" * total_width)
    
    for _, row in stats.iterrows():
        l_val = str(row[lane_col])
        c_val = str(row[cond_col]) if cond_col else "N/A"
        n_b = int(row['n_before'])
        n_a = int(row['n_survivors'])
        pct = row['survival_%']
        
        # Inject the dynamic cond_width into the f-string formatting
        print(f"{l_val:<10} | {c_val:<{cond_width}} | {n_b:<10} | {n_a:<10} | {pct:.2f}%")
        
    print("-" * total_width)
    
    return stats

def plot_slowdown_survivor_growth_rates_by_window(
    df: pd.DataFrame, 
    window_start: float,
    window_end: float,
    treatment_start: float = None, 
    treatment_end: float = None,
    require_growing_before: bool = True,
    require_growing_after: bool = True,
    mode_round_interval: float = 0.0005  # NEW ARGUMENT
):
    """
    Plots growth_rate_smoothed vs time for cells that START a slowdown 
    within the window [window_start, window_end].
    
    Layout: 2 Columns of plots for individual conditions.
    """
    if df is None:
        print("Error: DataFrame is None.")
        return

    print(f"{'='*40}\n PLOTTING TRACES: SLOWDOWN STARTING {window_start}-{window_end} min \n{'='*40}")

    # 1. Check Columns
    lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df.columns), None)
    cond_col = next((c for c in ['condition', 'Condition'] if c in df.columns), None)
    
    if 'slowdown_start' not in df.columns:
        print("Error: 'slowdown_start' column missing. Please run the updated batch slowdown detection.")
        return

    if not lane_col:
        print("Error: Could not find a 'lane' column.")
        return
    
    # 2. Filter Base Population (Growth Criteria)
    candidates = df.copy()

    if require_growing_before:
        if 'growing_before' in candidates.columns:
            candidates = candidates[candidates['growing_before'] == True]
        else:
            print("Warning: 'growing_before' column missing. Skipping check.")
            
    if require_growing_after:
        if 'growing_after' in candidates.columns:
            candidates = candidates[candidates['growing_after'] == True]
        else:
            print("Warning: 'growing_after' column missing. Skipping check.")

    if candidates.empty:
        print("No cells matched the growth criteria.")
        return

    # 3. Filter for Slowdowns Starting in Window
    # We look for rows where a slowdown is active AND the recorded start time is in range.
    # slowdown_start is populated on rows where slowdown==True.
    mask_window = (
        (candidates['slowdown'] == True) & 
        (candidates['slowdown_start'] >= window_start) & 
        (candidates['slowdown_start'] <= window_end)
    )
    
    slowdown_mids = candidates.loc[mask_window, 'mother_id'].unique()
    
    if len(slowdown_mids) == 0:
        print(f"No mothers found with a slowdown starting between {window_start} and {window_end} min.")
        return

    print(f"Found {len(slowdown_mids)} Target Cells out of {candidates['mother_id'].nunique()} Base Candidates.")
    
    # 4. Extract Traces
    slowdown_df = df[df['mother_id'].isin(slowdown_mids)].copy()
    
    # 5. Grouping
    if cond_col:
        groups = slowdown_df[[lane_col, cond_col]].drop_duplicates().sort_values(lane_col)
    else:
        groups = slowdown_df[[lane_col]].drop_duplicates().sort_values(lane_col)
        groups['condition'] = "Unknown"

    sns.set_theme(style="ticks", context="talk")
    summary_data = []

    # Helper for mode calculation
    def calculate_custom_mode(series):
        # Round to nearest interval
        rounded = (series / mode_round_interval).round() * mode_round_interval
        m = rounded.mode()
        return m.iloc[0] if not m.empty else np.nan

    # --- SETUP SUBPLOTS ---
    n_groups = len(groups)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)
    
    # Create the figure for the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = np.array(axes).flatten() # Flatten to 1D array for easy indexing

    # 6. Generate Individual Plots
    for i, (_, row) in enumerate(groups.iterrows()):
        ax = axes[i]
        lane_val = row[lane_col]
        cond_val = row[cond_col] if cond_col else "Unknown"
        
        # A. Filter Data
        if cond_col:
            group_total = candidates[(candidates[lane_col] == lane_val) & (candidates[cond_col] == cond_val)]
            group_slow = slowdown_df[(slowdown_df[lane_col] == lane_val) & (slowdown_df[cond_col] == cond_val)]
        else:
            group_total = candidates[candidates[lane_col] == lane_val]
            group_slow = slowdown_df[slowdown_df[lane_col] == lane_val]
            
        if group_slow.empty: 
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
        
        # B. Stats
        n_total = group_total['mother_id'].nunique()
        n_slow = group_slow['mother_id'].nunique()
        pct = (n_slow / n_total * 100.0) if n_total > 0 else 0.0
        
        # C. Calculate Aggregates
        grouped_time = group_slow.groupby('time')['growth_rate_smoothed']
        mean_trace = grouped_time.mean()
        median_trace = grouped_time.median()
        mode_trace = grouped_time.apply(calculate_custom_mode)

        summary_data.append({
            'label': f"Lane {lane_val}: {cond_val}",
            'n_str': f"(n={n_slow})",
            'mean': mean_trace,
            'median': median_trace,
            'mode': mode_trace
        })

        # D. Plotting
        # Individual Traces
        sns.lineplot(
            data=group_slow,
            x="time",
            y="growth_rate_smoothed",
            units="mother_id",
            estimator=None,
            color="gray",
            alpha=0.3,
            lw=1,
            ax=ax
        )
        
        # Mean Trace Overlay
        ax.plot(mean_trace.index, mean_trace.values, color="#1f77b4", lw=3, label="Mean Growth Rate")
        
        # Shading: Treatment
        if treatment_start is not None and treatment_end is not None:
            ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1, zorder=0)
        
        # Shading: Selection Window
        ax.axvspan(window_start, window_end, color='green', alpha=0.2, zorder=0, label="Selection Window")

        # Title
        criteria_str = []
        if require_growing_before: criteria_str.append("GrowBefore")
        if require_growing_after: criteria_str.append("GrowAfter")
        filter_lbl = "+".join(criteria_str) if criteria_str else "All"

        title_str = (f"Lane {lane_val}: {cond_val}\n"
                     f"Pop: {filter_lbl} | n={n_slow}/{n_total} ({pct:.1f}%) | Start: {window_start}-{window_end}m")
        ax.set_title(title_str, fontweight="bold", fontsize=12)
        
        ax.set_ylabel(r"Growth Rate ($\times 10^{-2}$)")
        ax.set_xlabel("Time (min)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=10)
        
        ax.grid(True, linestyle=":", alpha=0.4)
        sns.despine(ax=ax)

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

    # 7. Generate Summary Plots (Mean, Median, Mode)
    if summary_data:
        metrics = [
            ('mean', 'Mean Growth Rate'),
            ('median', 'Median Growth Rate'),
            ('mode', f'Mode Growth Rate (Rounded to {mode_round_interval})')
        ]

        palette = sns.color_palette("tab10", len(summary_data))
        
        for metric_key, metric_title in metrics:
            plt.figure(figsize=(12, 7))
            
            for i, item in enumerate(summary_data):
                # Retrieve the specific series (mean, median, or mode)
                series = item[metric_key]
                label = f"{item['label']} {item['n_str']}"
                
                plt.plot(series.index, series.values, 
                         lw=2.5, color=palette[i], label=label)
            
            if treatment_start is not None and treatment_end is not None:
                plt.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1, label='Treatment')
                
            plt.axvspan(window_start, window_end, color='green', alpha=0.2, label='Selection Window')
            
            title = f"Summary: {metric_title} (Slowdown Start: {window_start}-{window_end} min)"
            plt.title(title, fontweight="bold")
            plt.ylabel(r"Growth Rate ($\times 10^{-2} min^{-1}$)")
            plt.xlabel("Time (min)")
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            
            plt.legend(loc="upper right", frameon=True, fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.4)
            sns.despine()
            plt.tight_layout()
            plt.show()

def plot_survivor_growth_rate_distribution_overlaid(
    df: pd.DataFrame,
    treatment_start: float,
    win_start: float = None,  
    win_end: float = None,    
    require_growing_before: bool = True,
    require_growing_after: bool = True,
    bins: int = 20,
    alpha: float = 0.4
):
    """
    Calculates the mean growth rate for each survivor mother_id within the window 
    [win_start, win_end] and plots overlaid probability histograms.
    
    Defaults:
      win_start = treatment_start + 50
      win_end   = treatment_start + 60
    """
    if df is None:
        print("Error: DataFrame is None.")
        return

    # 1. Define Window (Handle Defaults)
    if win_start is None:
        win_start = treatment_start + 50.0
    if win_end is None:
        win_end = treatment_start + 60.0
        
    t1 = win_start
    t2 = win_end
    
    print(f"{'='*40}\n GROWTH RATE DISTRIBUTION (OVERLAID) \n{'='*40}")
    print(f"Window: {t1:.1f} min to {t2:.1f} min")

    # 2. Filter Population
    candidates = df.copy()
    
    if require_growing_before:
        if 'growing_before' in candidates.columns:
            candidates = candidates[candidates['growing_before'] == True]
            
    if require_growing_after:
        if 'growing_after' in candidates.columns:
            candidates = candidates[candidates['growing_after'] == True]

    if candidates.empty:
        print("No cells matched the growth criteria.")
        return

    # 3. Extract Data in Window
    window_data = candidates[(candidates['time'] >= t1) & (candidates['time'] <= t2)]
    
    if window_data.empty:
        print(f"No data points found in the window {t1}-{t2} min.")
        return

    # 4. Calculate Mean Growth Rate per Mother
    cond_col = 'condition' if 'condition' in window_data.columns else 'Condition'
    cols_to_group = ['mother_id', cond_col]
        
    mother_stats = window_data.groupby(cols_to_group)['growth_rate_smoothed'].mean().reset_index()
    mother_stats.rename(columns={'growth_rate_smoothed': 'mean_growth_rate'}, inplace=True)
    
    # 5. Sorting (Sort legend by Lane if possible)
    try:
        sorted_conds, _, _ = _lane_sorting(candidates)
        mother_stats[cond_col] = pd.Categorical(
            mother_stats[cond_col].astype(str), 
            categories=[str(c) for c in sorted_conds], 
            ordered=True
        )
    except NameError:
        pass 

    # 6. Plotting
    sns.set_theme(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.histplot(
        data=mother_stats,
        x='mean_growth_rate',
        hue=cond_col,
        stat='probability',   
        common_norm=False,    
        bins=bins,
        kde=True,             
        element="step",       
        fill=True,            
        alpha=alpha,          
        palette='tab10',      
        linewidth=2,
        ax=ax
    )

    # 7. Formatting
    try:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title="Condition")
    except Exception:
        print("Warning: Could not move legend (requires seaborn >= 0.11.2).")

    # Dynamic Title
    pop_type = "Survivor" if require_growing_after else "Cell"
    
    # Calculate offset for title clarity if relevant
    offset_s = t1 - treatment_start
    offset_e = t2 - treatment_start
    time_str = f"{t1:.0f}-{t2:.0f} min (T+{offset_s:.0f} to T+{offset_e:.0f})"
    
    ax.set_title(f"{pop_type} Growth Rates | Window: {time_str}", fontweight='bold', pad=15)
    
    ax.set_xlabel(r"Mean Growth Rate (min$^{-1}$)", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')

    sns.despine()
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_growth_rate_stats_by_condition(
    df: pd.DataFrame,
    require_growing_before: bool = True,
    require_growing_after: bool = False,
    treatment_start: float = None,
    treatment_end: float = None,
    mode_round_interval: float = 0.001,  # Updated default
    plot_xmin: float = 0.0,              # New Range Control
    plot_xmax: float = 240.0             # New Range Control
):
    """
    Plots Mean, Median, and Mode growth rates vs Time for each condition.
    Restricts the plot to the range [plot_xmin, plot_xmax].
    
    Mode Calculation: Data is first rounded to the nearest 'mode_round_interval' 
    (default 0.001) before finding the most frequent value.
    """
    if df is None:
        print("Error: DataFrame is None.")
        return

    print(f"{'='*40}\n PLOTTING GROWTH RATE STATISTICS (Mean/Median/Mode) \n{'='*40}")
    print(f"Plot Range: {plot_xmin} to {plot_xmax} min")
    
    # 1. Filter Data (Population)
    candidates = df.copy()
    
    if require_growing_before:
        if 'growing_before' in candidates.columns:
            candidates = candidates[candidates['growing_before'] == True]
    
    if require_growing_after:
        if 'growing_after' in candidates.columns:
            candidates = candidates[candidates['growing_after'] == True]
            
    if candidates.empty:
        print("No cells matched the growth criteria.")
        return
        
    if 'growth_rate_smoothed' not in candidates.columns:
        print("Error: 'growth_rate_smoothed' column missing.")
        return

    # 2. Filter Data (Time Range)
    # We filter data BEFORE calculating stats to speed it up and ensure the plot limits are respected
    candidates = candidates[(candidates['time'] >= plot_xmin) & (candidates['time'] <= plot_xmax)]
    
    if candidates.empty:
        print(f"No data points found in the range {plot_xmin}-{plot_xmax} min.")
        return

    # 3. Sort Conditions
    try:
        sorted_conds, _, _ = _lane_sorting(candidates)
    except NameError:
        sorted_conds = sorted(candidates['condition'].unique())

    # 4. Calculate Statistics
    print("Calculating statistics...")
    
    def calc_mode_rounded(series):
        # Round to nearest interval
        rounded = (series / mode_round_interval).round() * mode_round_interval
        m = rounded.mode()
        return m.iloc[0] if not m.empty else np.nan

    cond_col = 'condition' if 'condition' in candidates.columns else 'Condition'
    
    stats = candidates.groupby(['time', cond_col])['growth_rate_smoothed'].agg(
        Mean='mean',
        Median='median',
        Mode=calc_mode_rounded
    ).reset_index()

    # 5. Plotting
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    metrics = ['Mean', 'Median', 'Mode']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        sns.lineplot(
            data=stats,
            x='time',
            y=metric,
            hue=cond_col,
            hue_order=sorted_conds,
            palette='tab10',
            lw=2.5,
            ax=ax,
            legend=(i == 0) # Only show legend on top plot
        )
        
        # Treatment Shading
        if treatment_start is not None and treatment_end is not None:
            # Only shade if within plot range
            start = max(plot_xmin, treatment_start)
            end = min(plot_xmax, treatment_end)
            if end > start:
                ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.1, zorder=0)
                
                # Label only on top plot
                if i == 0:
                    y_min, y_max = ax.get_ylim()
                    # Center text within the visible treatment window
                    treat_center = (treatment_start + treatment_end) / 2
                    if plot_xmin <= treat_center <= plot_xmax:
                        ax.text(treat_center, y_max, "Treatment", 
                                color='darkred', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_ylabel(f"{metric} Growth Rate\n($\\min^{{-1}}$)", fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.axhline(0, color='black', lw=1, alpha=0.3)
        
        # Explicitly set X limits
        ax.set_xlim(plot_xmin, plot_xmax)
        
        if metric == 'Mode':
            ax.set_title(f"Mode (Rounded to nearest {mode_round_interval})", fontsize=11, loc='right')

    # Legend
    if axes[0].get_legend():
        sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1.02, 1), title="Condition", frameon=True)

    axes[-1].set_xlabel("Time (min)", fontweight='bold')
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(60)) # Ticks every 60m for this range
    
    # Dynamic Title
    pop_type = []
    if require_growing_before: pop_type.append("Growing Before")
    if require_growing_after: pop_type.append("Growing After")
    if not pop_type: pop_type.append("All Cells")
    
    plt.suptitle(f"Growth Rate Statistics ({' + '.join(pop_type)})", fontweight='bold', y=0.95)
    plt.subplots_adjust(right=0.85)
    plt.show()

# Slowdown analysis functions
def get_candidate_mothers(df_data, df_lost, check_idxs):
    """
    Optimized filtering to find mother_ids where the slowdown interval 
    ends at the experiment's final time, using the numeric slowdown_end column.
    """
    # 1. Filter candidates
    candidates = df_lost.query("growing_after == True").loc[check_idxs].copy()
    
    # 2. Find the final time of the experiment
    final_time = df_data["time"].max()
    
    # Extract only the necessary columns and drop NaNs to speed up grouping
    meta = df_data.dropna(subset=["slowdown_end"])[["mother_id", "slowdown_end"]]
    
    # 3. Find eligible IDs (max slowdown_end matches final_time)
    max_end = meta.groupby("mother_id")["slowdown_end"].max()
    eligible_ids = max_end[np.isclose(max_end, final_time)].index
    
    # 4. Return sorted unique list
    return (
        candidates.loc[candidates["mother_id"].isin(eligible_ids), "mother_id"]
        .dropna().unique().tolist()
    )

def plot_single_mother(df, mother_id, current_idx, total_count):
    """
    Fast plotting using matplotlib primitives (avoids Seaborn overhead).
    """
    # Fast filtering
    subset = df.loc[df["mother_id"] == mother_id]
    t = subset["time"].values
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Left Axis
    ax1.plot(t, subset["log_length_smoothed"].values, color="red", label="log_length")
    ax1.set_ylabel("log_length_smoothed", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("time")
    
    # Right Axis
    ax2 = ax1.twinx()
    ax2.plot(t, subset["growth_rate_smoothed"].values, color="blue", label="growth_rate")
    ax2.set_ylabel("growth_rate_smoothed", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    
    ax1.set_title(f"Mother ID: {mother_id} (Idx {current_idx}/{total_count})")
    plt.tight_layout()
    plt.show()

def filter_fast_survivors(max_lost_time, mother_machine_data, fresh_media_start, treatment_start, lanes=None, top_percent=10, gr_threshold=1.5):
    """
    Filters max_lost_time to keep only survivors whose top X% growth rate 
    (within the specified time window) is >= a given threshold.
    """
    # 1. Base filter for survivors (and optionally specific lanes)
    query_str = "growing_before == True and growing_after == True"
    if lanes is not None:
        query_str += f" and lane in {lanes}"
    
    df_survivors = max_lost_time.query(query_str)
    
    # 2. Extract growth rates for these specific mothers WITHIN the specific time window
    mm_subset = mother_machine_data[
        (mother_machine_data['mother_id'].isin(df_survivors['mother_id'])) &
        (mother_machine_data['time'] >= fresh_media_start) &
        (mother_machine_data['time'] <= treatment_start)
    ]
    
    # 3. Calculate the quantile (e.g., top 10% = 0.90 quantile) for each mother
    q_val = 1.0 - (top_percent / 100.0)
    gr_quantiles = mm_subset.groupby('mother_id')['growth_rate_smoothed'].quantile(q_val)
    
    # 4. Scale threshold to absolute value (e.g. 1.5 -> 0.015) and identify passes
    actual_threshold = gr_threshold / 100.0
    passed_ids = gr_quantiles[gr_quantiles >= actual_threshold].index
    
    # 5. Filter the final dataframe
    filtered_df = df_survivors[df_survivors['mother_id'].isin(passed_ids)].copy()
    
    # Print the summary stats directly from the function
    print(f"Original survivors in target lanes: {len(df_survivors)}")
    print(f"Filtered survivors (Top {top_percent}% GR >= {gr_threshold} x 10^-2 between {fresh_media_start} and {treatment_start} min): {len(filtered_df)}")
    
    return filtered_df

# Lag time analysis functions
def plot_lag_histograms(df: pd.DataFrame, 
                        treatment_start: float,
                        times_relative_to_treatment_start: list = [60, 120, 180, 240], 
                        bin_width: float = 5.0,
                        require_growing_before: bool = True,
                        require_growing_after: bool = False,
                        ylim: tuple = None): 
    """
    Plots probability density histograms of 'accum_lag_current' at specific timepoints relative to treatment start.
    Filters based on growing_before and growing_after statuses.
    
    CRITICAL: The probability density is normalized to the INITIAL filtered population 
    (i.e., cells that satisfy require_growing_before, regardless of growing_after). 
    As cells die, the total area under the histogram will shrink to reflect the loss 
    of the original population.
    
    Args:
        ylim: Optional tuple to fix the y-axis limits (e.g., (0, 0.005)).
    """
    if df is None: return
    if 'accum_lag_current' not in df.columns:
        print("Error: 'accum_lag_current' column not found. Run calculate_accumulated_lag first.")
        return

    print(f"{'='*60}\n PLOTTING LAG HISTOGRAMS (NORMALIZED TO INITIAL POPULATION) \n{'='*60}")

    # 1. Determine Initial Population (Before growing_after filter)
    df_initial = df.copy()
    if require_growing_before:
        if 'growing_before' in df_initial.columns:
            df_initial = df_initial[df_initial['growing_before'] == True]

    if df_initial.empty:
        print("Error: No data remaining after filtering for initial population (growing_before).")
        return

    # 2. Determine Grouping for Initial Counts
    hue_col = None
    hue_order = None
    if 'condition' in df_initial.columns and df_initial['condition'].nunique() > 1:
        hue_col = 'condition'
    elif 'lane' in df_initial.columns:
        hue_col = 'lane'
        
    if hue_col:
        try:
            if hue_col == 'condition':
                # Attempt to use a local _lane_sorting function if it exists
                hue_order, _, _ = _lane_sorting(df_initial)
            else:
                hue_order = sorted(df_initial[hue_col].unique())
        except (NameError, Exception):
            hue_order = sorted(df_initial[hue_col].unique())
            
    # Calculate the INITIAL population size for each group for true normalization
    initial_cells = df_initial.drop_duplicates('mother_id')
    if hue_col:
        initial_counts = initial_cells[hue_col].value_counts().to_dict()
        print("Initial population counts per condition (before 'growing_after' filter):")
        for cond, count in initial_counts.items():
            print(f"  - {cond}: {count} cells")
    else:
        initial_counts = {None: len(initial_cells)}
        print(f"Initial population count (before 'growing_after' filter): {initial_counts[None]} cells")

    # 3. Apply growing_after filter for the actual plotting data
    df_filtered = df_initial.copy()
    if require_growing_after:
        if 'growing_after' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['growing_after'] == True]

    if df_filtered.empty:
        print("Error: No data remaining after filtering for survivors (growing_after).")
        return

    # 4. Setup Figure
    n_t = len(times_relative_to_treatment_start)
    cols = 2
    rows = int(np.ceil(n_t / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), constrained_layout=True)
    axes = np.array(axes).flatten() if hasattr(axes, 'flatten') else np.array([axes])
    
    # 5. Generate Plots
    for i, rel_tp in enumerate(times_relative_to_treatment_start):
        if i >= len(axes): break
        ax = axes[i]
        
        # Calculate absolute timepoint
        tp = treatment_start + rel_tp
        
        mask = (df_filtered['time'] >= tp - 2.0) & (df_filtered['time'] <= tp + 2.0)
        subset = df_filtered[mask].copy()
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            ax.set_title(f"Accumulated lag {rel_tp} mins after treatment start\n(n=0)", fontweight='bold')
            ax.set_xlim(0, tp)
            if ylim is not None:
                ax.set_ylim(ylim)
            continue
            
        subset['dist'] = (subset['time'] - tp).abs()
        subset = subset.sort_values('dist').drop_duplicates('mother_id')
        
        # Calculate custom weights to normalize by INITIAL population density
        if hue_col:
            subset['weight'] = subset[hue_col].map(lambda x: 1.0 / (initial_counts.get(x, 1) * bin_width))
        else:
            subset['weight'] = 1.0 / (initial_counts[None] * bin_width)
        
        sns.histplot(
            data=subset,
            x='accum_lag_current',
            weights='weight',     # Use our custom initial-population weights
            hue=hue_col,
            hue_order=hue_order,  
            stat='count',         # 'count' with weights produces our custom density
            common_norm=False,    
            multiple="layer", 
            element="step",
            binwidth=bin_width,
            kde=True,
            ax=ax,
            palette="tab10",
            alpha=0.3
        )
        
        ax.set_title(f"Accumulated lag {rel_tp} mins after treatment start\n(surviving n={len(subset)})", fontweight='bold')
        ax.set_xlabel("Accumulated Lag (min)")
        ax.set_ylabel("Density (Norm. to Initial Pop)")
        
        # Handle axis limits
        ax.set_xlim(0, max(tp, 10))  # Prevent issues if tp is exactly 0
        if ylim is not None:
            ax.set_ylim(ylim)
            
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, linestyle=':', alpha=0.4)

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.show()

def plot_interval_lag_histograms(df: pd.DataFrame, 
                                 treatment_start: float,
                                 intervals_relative_to_treatment_start: list = [(60, 180), (60, 240)], 
                                 bin_width: float = 5.0,
                                 require_growing_before: bool = True,
                                 require_growing_after: bool = False): 
    """
    Plots probability density histograms of the CHANGE in 'accum_lag_current' over specific time intervals relative to treatment start.
    Filters based on growing_before and growing_after statuses.
    
    CRITICAL: The probability density is normalized to the INITIAL filtered population 
    (e.g., all cells that were growing_before).
    """
    if df is None: return
    if 'accum_lag_current' not in df.columns:
        print("Error: 'accum_lag_current' column not found.")
        return

    print(f"{'='*60}\n PLOTTING INTERVAL LAG HISTOGRAMS (NORMALIZED TO INITIAL POPULATION) \n{'='*60}")

    # 1. Filter Population
    df_filtered = df.copy()
    if require_growing_before:
        if 'growing_before' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['growing_before'] == True]
            
    if require_growing_after:
        if 'growing_after' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['growing_after'] == True]

    if df_filtered.empty:
        print("Error: No data remaining after filtering.")
        return

    # 2. Grouping & Enforcing Order/Color Consistency
    hue_col = None
    hue_order = None
    if 'condition' in df_filtered.columns and df_filtered['condition'].nunique() > 1:
        hue_col = 'condition'
    elif 'lane' in df_filtered.columns:
        hue_col = 'lane'
        
    if hue_col:
        try:
            if hue_col == 'condition':
                hue_order, _, _ = _lane_sorting(df_filtered)
            else:
                hue_order = sorted(df_filtered[hue_col].unique())
        except NameError:
            hue_order = sorted(df_filtered[hue_col].unique())
            
    # Calculate the INITIAL population size for each group for true normalization
    initial_cells = df_filtered.drop_duplicates('mother_id')
    if hue_col:
        initial_counts = initial_cells[hue_col].value_counts().to_dict()
    else:
        initial_counts = {None: len(initial_cells)}
        
    n_i = len(intervals_relative_to_treatment_start)
    cols = 2
    rows = int(np.ceil(n_i / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), constrained_layout=True)
    axes = np.array(axes).flatten() if hasattr(axes, 'flatten') else np.array([axes])
    
    for i, (rel_start, rel_end) in enumerate(intervals_relative_to_treatment_start):
        if i >= len(axes): break
        ax = axes[i]
        
        # Calculate absolute timepoints
        t_start = treatment_start + rel_start
        t_end = treatment_start + rel_end
        
        # Get Data Start
        mask_start = (df_filtered['time'] >= t_start - 2.0) & (df_filtered['time'] <= t_start + 2.0)
        df_start = df_filtered[mask_start].copy()
        df_start['dist'] = (df_start['time'] - t_start).abs()
        df_start = df_start.sort_values('dist').drop_duplicates('mother_id')
        df_start = df_start.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_start'})
        
        # Get Data End
        mask_end = (df_filtered['time'] >= t_end - 2.0) & (df_filtered['time'] <= t_end + 2.0)
        df_end = df_filtered[mask_end].copy()
        df_end['dist'] = (df_end['time'] - t_end).abs()
        df_end = df_end.sort_values('dist').drop_duplicates('mother_id')
        df_end = df_end.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_end'})
        
        # Merge
        merged = df_start.join(df_end, how='inner')
        merged['lag_delta'] = merged['lag_end'] - merged['lag_start']
        
        # Add Meta
        if hue_col:
            mapping = df_filtered[['mother_id', hue_col]].drop_duplicates('mother_id').set_index('mother_id')[hue_col]
            merged[hue_col] = merged.index.map(mapping)
        
        if merged.empty:
            ax.text(0.5, 0.5, "No Overlapping Data", ha='center', transform=ax.transAxes)
            ax.set_title(f"Accumulated lag between {rel_start}-{rel_end} mins after treatment start\n(n=0)", fontweight='bold')
            continue
            
        # Calculate custom weights to normalize by INITIAL population density
        if hue_col:
            merged['weight'] = merged[hue_col].map(lambda x: 1.0 / (initial_counts[x] * bin_width))
        else:
            merged['weight'] = 1.0 / (initial_counts[None] * bin_width)
            
        sns.histplot(
            data=merged,
            x='lag_delta',
            weights='weight',     # Use our custom initial-population weights
            hue=hue_col,
            hue_order=hue_order,  
            stat='count',         # 'count' with weights produces our custom density
            common_norm=False,    
            multiple="layer", 
            element="step",
            binwidth=bin_width,
            kde=True,
            ax=ax,
            palette="tab10",
            alpha=0.3
        )
        
        duration = t_end - t_start
        ax.set_title(f"Accumulated lag between {rel_start}-{rel_end} mins after treatment start\n(surviving n={len(merged)})", fontweight='bold')
        ax.set_xlabel(f"Accumulated Lag in Window (min)")
        ax.set_ylabel("Density (Norm. to Initial Pop)")
        ax.set_xlim(0, duration)
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, linestyle=':', alpha=0.4)

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.show()

def generate_interval_lag_pdf(
    df: pd.DataFrame, 
    treatment_start: float,
    treatment_end: float,
    interval_relative_to_treatment_start: tuple = (60, 120),
    min_lag_amount: float = 10.0,
    max_lag_amount: float = 16.0,
    require_growing_before: bool = True,
    require_growing_after: bool = False,
    max_IDs_per_condition: int = 50,
    filename: str = None
):
    """
    Identifies cells with a specific amount of 'accum_lag_current' over a given time interval,
    and generates a 3-panel PDF report (Length, Growth Rate, Acceleration) for those individual cells.
    The PDF is split into sections based on 'condition' (or 'lane').
    """
    if df is None: return
    if 'accum_lag_current' not in df.columns:
        print("Error: 'accum_lag_current' column not found.")
        return

    rel_start, rel_end = interval_relative_to_treatment_start
    print(f"{'='*60}")
    print(f" GENERATING INTERVAL LAG PDF")
    print(f" Interval: T+{rel_start} to T+{rel_end} min")
    print(f" Target Lag Range: {min_lag_amount} to {max_lag_amount} min (inclusive)")
    print(f"{'='*60}")

    # 1. Filter Population
    df_filtered = df.copy()
    if require_growing_before:
        if 'growing_before' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['growing_before'] == True]
            
    if require_growing_after:
        if 'growing_after' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['growing_after'] == True]

    if df_filtered.empty:
        print("Error: No data remaining after filtering.")
        return

    # 2. Calculate absolute timepoints
    t_start = treatment_start + rel_start
    t_end = treatment_start + rel_end
    
    # Get Data Start
    mask_start = (df_filtered['time'] >= t_start - 2.0) & (df_filtered['time'] <= t_start + 2.0)
    df_start = df_filtered[mask_start].copy()
    df_start['dist'] = (df_start['time'] - t_start).abs()
    df_start = df_start.sort_values('dist').drop_duplicates('mother_id')
    df_start = df_start.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_start'})
    
    # Get Data End
    mask_end = (df_filtered['time'] >= t_end - 2.0) & (df_filtered['time'] <= t_end + 2.0)
    df_end = df_filtered[mask_end].copy()
    df_end['dist'] = (df_end['time'] - t_end).abs()
    df_end = df_end.sort_values('dist').drop_duplicates('mother_id')
    df_end = df_end.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_end'})
    
    # Merge and calculate lag_delta
    merged = df_start.join(df_end, how='inner')
    merged['lag_delta'] = merged['lag_end'] - merged['lag_start']
    
    if merged.empty:
        print("No cells found that overlap the entire interval.")
        return

    # Filter to targeted lag amounts
    target_moms_df = merged[(merged['lag_delta'] >= min_lag_amount) & (merged['lag_delta'] <= max_lag_amount)].copy()
    if target_moms_df.empty:
        print(f"No cells found with a lag delta between {min_lag_amount} and {max_lag_amount} min.")
        return
        
    # 3. Grouping & Sorting by condition
    group_col = None
    if 'condition' in df_filtered.columns and df_filtered['condition'].nunique() > 1:
        group_col = 'condition'
    elif 'lane' in df_filtered.columns:
        group_col = 'lane'
        
    if group_col:
        mapping = df_filtered[['mother_id', group_col]].drop_duplicates('mother_id').set_index('mother_id')[group_col]
        target_moms_df[group_col] = target_moms_df.index.map(mapping)
        
        try:
            if group_col == 'condition':
                hue_order, _, _ = _lane_sorting(df_filtered)
            else:
                hue_order = sorted(target_moms_df[group_col].dropna().unique())
        except NameError:
            hue_order = sorted(target_moms_df[group_col].dropna().unique())
            
        groups_to_plot = [g for g in hue_order if g in target_moms_df[group_col].values]
    else:
        target_moms_df['group'] = 'All'
        group_col = 'group'
        groups_to_plot = ['All']

    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'interval_lag_{min_lag_amount}to{max_lag_amount}m_report_{timestamp}.pdf'

    x_max = df_filtered['time'].max()
    x_limit = np.ceil(x_max / 120) * 120

    print(f"Generating PDF: {filename}...")
    
    with PdfPages(filename) as pdf:
        for group in groups_to_plot:
            group_moms = target_moms_df[target_moms_df[group_col] == group].index.tolist()
            if len(group_moms) == 0:
                continue
                
            # Create a section title page for the condition
            fig_title, ax_title = plt.subplots(figsize=(8.0, 7.0))
            ax_title.axis('off')
            title_text = f"Condition: {group}\n\n"
            title_text += f"{len(group_moms)} cells found\n"
            title_text += f"Target Lag Range: {min_lag_amount} - {max_lag_amount} min\n"
            title_text += f"Interval: T+{rel_start} to T+{rel_end} min"
            ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16, fontweight='bold')
            pdf.savefig(fig_title)
            plt.close(fig_title)
            
            # Subsample to prevent massive PDF if desired
            if max_IDs_per_condition is not None and len(group_moms) > max_IDs_per_condition:
                print(f"  - {group}: Plotting {max_IDs_per_condition} of {len(group_moms)} cells")
                group_moms = group_moms[:max_IDs_per_condition]
            else:
                print(f"  - {group}: Plotting all {len(group_moms)} cells")
                
            for mid in tqdm(group_moms, desc=f"Plotting {group}"):
                trace = df_filtered[df_filtered['mother_id'] == mid].sort_values('time')
                t = trace['time'].values
                
                y_raw = trace['log_length_cleaned'].values if 'log_length_cleaned' in trace.columns else None
                y_sm = trace['log_length_smoothed'].values
                d1_sm = trace['growth_rate_smoothed'].values if 'growth_rate_smoothed' in trace.columns else None
                d2_col = 'second_deriv_smoothed' if 'second_deriv_smoothed' in trace.columns else '2nd_deriv_smoothed'
                d2_sm = trace[d2_col].values if d2_col in trace.columns else None
                
                divs = trace['division_event'].astype(float).fillna(0).astype(bool).values if 'division_event' in trace.columns else np.zeros(len(t), dtype=bool)
                
                fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
                plt.subplots_adjust(top=0.92, bottom=0.1, left=0.12, right=0.95, hspace=0.3)
                (ax_len, ax_d1, ax_d2) = (axes[0], axes[1], axes[2])
                
                # Fetch exact lag amount to put in the title
                cell_lag = target_moms_df.loc[mid, 'lag_delta']
                
                title_parts = [f"Mother {mid}"]
                if group_col != 'group':
                    title_parts.append(f"{group}")
                title_parts.append(f"Interval Lag: {cell_lag:.1f}m")
                fig.suptitle(' | '.join(title_parts), fontweight='bold', fontsize=12)
                
                # Top panel: Length
                if y_raw is not None:
                    ax_len.scatter(t, y_raw, c='gray', s=10, alpha=0.3, zorder=1)
                ax_len.plot(t, y_sm, c='#d62728', lw=2, label='Smoothed', zorder=2)
                if divs.any() and y_raw is not None:
                    valid_divs = divs & np.isfinite(y_raw)
                    ax_len.scatter(t[valid_divs], y_raw[valid_divs], c='cyan', s=30, marker='o', edgecolors='k', zorder=3, label='Division')
                ax_len.set_ylabel('ln[Length ($\\mu$m)]', fontweight='bold')
                ax_len.set_ylim(0.25, 3.25)
                
                # Middle panel: Growth Rate
                if d1_sm is not None:
                    ax_d1.plot(t, d1_sm, c='#1f77b4', lw=2)
                    ax_d1.set_ylabel('Rate ($\\times 10^{-2} min^{-1}$)', fontweight='bold')
                    ax_d1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.1f}'))
                    ax_d1.set_ylim(-0.001, 0.025)
                
                # Bottom panel: Acceleration
                if d2_sm is not None:
                    ax_d2.plot(t, d2_sm, c='#ff7f0e', lw=2)
                    ax_d2.axhline(0, c='k', ls='-', lw=0.8, alpha=0.5)
                    ax_d2.set_ylabel('Accel ($\\times 10^{-4} min^{-2}$)', fontweight='bold')
                    ax_d2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 10000:.1f}'))
                    ax_d2.set_ylim(-0.00025, 0.00025)
                    
                ax_d2.set_xlabel('Time (min)', fontweight='bold')
                
                # Format all axes
                for ax in axes:
                    if treatment_start is not None and treatment_end is not None:
                        ax.axvspan(treatment_start, treatment_end, color='darkred', alpha=0.15, zorder=0)
                        if ax == ax_len:
                            treat_center = (treatment_start + treatment_end) / 2
                            y_min, y_max = ax.get_ylim()
                            text_y = y_max - 0.05 * (y_max - y_min)
                            ax.text(treat_center, text_y, 'Treatment', color='darkred', ha='center', va='top', fontweight='bold', fontsize=8, alpha=0.8)
                    
                    # Highlight the interval being inspected in yellow
                    ax.axvspan(t_start, t_end, color='yellow', alpha=0.15, zorder=0)
                    if ax == ax_len:
                        interval_center = (t_start + t_end) / 2
                        ax.text(interval_center, text_y, 'Window', color='olive', ha='center', va='top', fontweight='bold', fontsize=8, alpha=0.8)
                            
                    ax.set_xlim(0, x_limit)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(60))
                    ax.grid(True, which='major', ls='-', alpha=0.4)
                    ax.grid(True, which='minor', ls=':', alpha=0.2)
                    
                pdf.savefig(fig)
                plt.close(fig)
                
    print(f"Done. Report saved to {os.path.abspath(filename)}")

def compare_interval_lag_between_lanes(
    df: pd.DataFrame,
    treatment_start: float,
    lane1: Any = 0,
    lane2: Any = 1,
    interval_start_relative: float = 0,
    times_relative_to_treatment_start: List[float] = [30, 60, 90, 120, 180, 240],
    require_growing_before: bool = True,
    require_growing_after: bool = True,
    bin_width: float = 3.0
) -> pd.DataFrame:
    """
    Calculates the differences in the mean, median, and mode of the interval lag 
    (change in accumulated lag) between two defined lanes.
    
    The intervals are calculated from `interval_start_relative` to each time 
    in `times_relative_to_treatment_start`.
    """
    if df is None: return None
    if 'accum_lag_current' not in df.columns:
        print("Error: 'accum_lag_current' column not found.")
        return None

    print(f"\n{'='*65}")
    print(f" COMPARING INTERVAL LAG: LANE {lane1} vs LANE {lane2} ")
    print(f" Interval Start: T+{interval_start_relative} min")
    print(f"{'='*65}")

    # 1. Filter Population
    df_filtered = df.copy()
    if require_growing_before and 'growing_before' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['growing_before'] == True]
    if require_growing_after and 'growing_after' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['growing_after'] == True]

    if df_filtered.empty:
        print("Error: No data remaining after filtering.")
        return None

    # Identify the appropriate lane column
    lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df_filtered.columns), None)
    if not lane_col:
        print("Error: Could not find a 'lane' column in the DataFrame.")
        return None

    def calculate_custom_mode(series, interval):
        """Finds the mode by rounding values to the specified bin_width."""
        if series.empty: return np.nan
        # Round to nearest bin
        rounded = (series / interval).round() * interval
        m = rounded.mode()
        return m.iloc[0] if not m.empty else np.nan

    # 2. Extract Base Time Data (The start of the interval)
    t_start = treatment_start + interval_start_relative
    
    mask_start = (df_filtered['time'] >= t_start - 2.0) & (df_filtered['time'] <= t_start + 2.0)
    df_start = df_filtered[mask_start].copy()
    if df_start.empty:
        print(f"Error: No data found around start time {t_start} min.")
        return None
        
    df_start['dist'] = (df_start['time'] - t_start).abs()
    df_start = df_start.sort_values('dist').drop_duplicates('mother_id')
    df_start = df_start.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_start'})
    
    lane_mapping = df_filtered[['mother_id', lane_col]].drop_duplicates('mother_id').set_index('mother_id')[lane_col]

    results = []

    # 3. Iterate over the requested times
    for rel_end in times_relative_to_treatment_start:
        t_end = treatment_start + rel_end
        
        mask_end = (df_filtered['time'] >= t_end - 2.0) & (df_filtered['time'] <= t_end + 2.0)
        df_end = df_filtered[mask_end].copy()
        
        if df_end.empty:
            continue
            
        df_end['dist'] = (df_end['time'] - t_end).abs()
        df_end = df_end.sort_values('dist').drop_duplicates('mother_id')
        df_end = df_end.set_index('mother_id')[['accum_lag_current']].rename(columns={'accum_lag_current': 'lag_end'})
        
        # Merge and calculate delta (lag_end - lag_start)
        merged = df_start.join(df_end, how='inner')
        merged['lag_delta'] = merged['lag_end'] - merged['lag_start']
        merged[lane_col] = merged.index.map(lane_mapping)
        
        # Split into lanes
        l1_data = merged[merged[lane_col] == lane1]['lag_delta'].dropna()
        l2_data = merged[merged[lane_col] == lane2]['lag_delta'].dropna()
        
        # Calculate Stats
        l1_m = l1_data.mean() if not l1_data.empty else np.nan
        l1_med = l1_data.median() if not l1_data.empty else np.nan
        l1_mo = calculate_custom_mode(l1_data, bin_width) if not l1_data.empty else np.nan
        
        l2_m = l2_data.mean() if not l2_data.empty else np.nan
        l2_med = l2_data.median() if not l2_data.empty else np.nan
        l2_mo = calculate_custom_mode(l2_data, bin_width) if not l2_data.empty else np.nan
        
        # Calculate Differences (Lane 1 - Lane 2)
        diff_m = l1_m - l2_m if (pd.notna(l1_m) and pd.notna(l2_m)) else np.nan
        diff_med = l1_med - l2_med if (pd.notna(l1_med) and pd.notna(l2_med)) else np.nan
        diff_mo = l1_mo - l2_mo if (pd.notna(l1_mo) and pd.notna(l2_mo)) else np.nan
        
        results.append({
            'Time_Relative': rel_end,
            f'Lane_{lane1}_n': len(l1_data),
            f'Lane_{lane2}_n': len(l2_data),
            f'Lane_{lane1}_Mean': l1_m,
            f'Lane_{lane2}_Mean': l2_m,
            'Diff_Mean': diff_m,
            f'Lane_{lane1}_Median': l1_med,
            f'Lane_{lane2}_Median': l2_med,
            'Diff_Median': diff_med,
            f'Lane_{lane1}_Mode': l1_mo,
            f'Lane_{lane2}_Mode': l2_mo,
            'Diff_Mode': diff_mo
        })

    results_df = pd.DataFrame(results)
    
    # 4. Print Summary Table
    if not results_df.empty:
        print(f"Differences (Lane {lane1} - Lane {lane2})")
        print(f"{'Time':<6} | {'N1':<4} | {'N2':<4} | {'Mean Diff':<10} | {'Median Diff':<11} | {'Mode Diff':<9}")
        print("-" * 65)
        for _, row in results_df.iterrows():
            t_val = int(row['Time_Relative'])
            n1 = int(row[f'Lane_{lane1}_n'])
            n2 = int(row[f'Lane_{lane2}_n'])
            
            m_str = f"{row['Diff_Mean']:.2f}" if pd.notna(row['Diff_Mean']) else "NaN"
            md_str = f"{row['Diff_Median']:.2f}" if pd.notna(row['Diff_Median']) else "NaN"
            mo_str = f"{row['Diff_Mode']:.2f}" if pd.notna(row['Diff_Mode']) else "NaN"
            
            print(f"T+{t_val:<4} | {n1:<4} | {n2:<4} | {m_str:<10} | {md_str:<11} | {mo_str:<9}")
            
    return results_df

def calculate_survival_per_lag_bin(
    df: pd.DataFrame,
    treatment_start: float,
    times_relative_to_treatment_start: List[float] = [30, 60, 90, 120, 180, 240],
    bin_width: float = 3.0,
    require_growing_before: bool = True
) -> pd.DataFrame:
    """
    Calculates the survival percentage for cells grouped by their accumulated lag bin.
    
    'All cells' defined by require_growing_before=True.
    'Survivors' defined by those cells that also have growing_after=True.
    """
    if df is None or df.empty: return None
    if 'accum_lag_current' not in df.columns:
        print("Error: 'accum_lag_current' column not found.")
        return None
        
    print(f"\n{'='*65}")
    print(f" CALCULATING SURVIVAL PERCENTAGES PER LAG BIN (Width: {bin_width}m) ")
    print(f"{'='*65}")

    # 1. Base filter for "All Cells"
    df_filtered = df.copy()
    if require_growing_before and 'growing_before' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['growing_before'] == True]
        
    if df_filtered.empty:
        print("Error: No data remaining after filtering for 'growing_before'.")
        return None
        
    lane_col = next((c for c in ['lane', 'lane_id', 'lane_num', 'position'] if c in df_filtered.columns), None)
    if not lane_col:
        print("Error: Could not find a 'lane' column.")
        return None
        
    # Get standard mapping for lane and growing_after per mother
    mother_info = df_filtered[['mother_id', lane_col, 'growing_after']].drop_duplicates('mother_id').set_index('mother_id')
    
    all_results = []
    
    # 2. Iterate over the requested times
    for rel_time in times_relative_to_treatment_start:
        t_eval = treatment_start + rel_time
        
        # Extract data close to t_eval (+/- 2 mins) to get the lag at that time
        mask = (df_filtered['time'] >= t_eval - 2.0) & (df_filtered['time'] <= t_eval + 2.0)
        df_time = df_filtered[mask].copy()
        
        if df_time.empty:
            continue
            
        # Keep the closest time point per mother
        df_time['dist'] = (df_time['time'] - t_eval).abs()
        df_time = df_time.sort_values('dist').drop_duplicates('mother_id')
        
        # Join with mother info (lane & survival status)
        df_time = df_time.set_index('mother_id').join(mother_info, rsuffix='_info', how='inner')
        
        # Handle potential column overlaps from the join
        lane_c = lane_col + '_info' if lane_col + '_info' in df_time.columns else lane_col
        ga_c = 'growing_after_info' if 'growing_after_info' in df_time.columns else 'growing_after'
        
        df_time['lag'] = df_time['accum_lag_current']
        df_time = df_time.dropna(subset=['lag'])
        
        if df_time.empty:
            continue
            
        # 3. Assign to bins based on bin_width
        df_time['bin_start'] = np.floor(df_time['lag'] / bin_width) * bin_width
        
        # 4. Group by Lane and Bin
        grouped = df_time.groupby([lane_c, 'bin_start'])
        
        for (lane_val, bin_val), group in grouped:
            total_cells = len(group)
            
            # Count True values in growing_after
            survivors = group[ga_c].fillna(False).astype(bool).sum() 
            surv_pct = (survivors / total_cells) * 100.0 if total_cells > 0 else 0.0
            
            bin_end = bin_val + bin_width
            bin_label = f"[{bin_val:g}, {bin_end:g})"
            
            # Use lowercase keys for the dataframe columns
            all_results.append({
                'time_relative': rel_time,
                'lane': lane_val,
                'bin_start': bin_val,
                'bin_label': bin_label,
                'total_cells': total_cells,
                'survivors': int(survivors),
                'survival_percentage': surv_pct
            })
            
    if not all_results:
        print("No data extracted for the requested times.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(all_results)
    
    # Sort sequentially for clean output
    results_df = results_df.sort_values(by=['time_relative', 'lane', 'bin_start']).reset_index(drop=True)
    
    return results_df


def plot_survival_vs_lag_bin(
    survival_df: pd.DataFrame, 
    lane: int, 
    time_relative: float
):
    """
    Plots survival_percentage against bin_start for a specified lane and time_relative.
    Displays the plot inline in the notebook.
    """
    if survival_df is None or survival_df.empty:
        print("Error: Provided DataFrame is empty or None.")
        return
        
    # Filter the dataframe for the specific lane and relative time
    plot_df = survival_df[(survival_df['lane'] == lane) & (survival_df['time_relative'] == time_relative)].copy()
    
    if plot_df.empty:
        print(f"No data found for lane {lane} at relative time {time_relative}.")
        return
        
    # Sort just in case to ensure a clean line plot
    plot_df = plot_df.sort_values('bin_start')

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    
    # Plotting line with circular markers
    ax.plot(plot_df['bin_start'], plot_df['survival_percentage'], marker='o', 
            linestyle='-', color='#1f77b4', lw=2, markersize=6)
    
    # Formatting
    ax.set_title(f"Survival Percentage vs Accumulated Lag\nLane: {lane} | Time post-treatment: {time_relative} min", fontweight='bold')
    ax.set_xlabel("Accumulated Lag (Bin Start, minutes)")
    ax.set_ylabel("Survival Percentage (%)")
    ax.set_ylim(-5, 105) # Add a little padding below 0 and above 100
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def simulate_population_shift(
    survival_df: pd.DataFrame,
    lane: int,
    time_relative: float,
    shift_amount: float = 12.0,
    safeguard: bool = True
) -> pd.DataFrame:
    """
    Simulates shifting the entire population by a specified time amount (modifying their lag)
    and calculates the expected new number of survivors based on the survival percentage 
    of the destination bins.
    
    Args:
        survival_df: DataFrame output from `calculate_survival_per_lag_bin`.
        lane: The specific lane to simulate.
        time_relative: The relative time post-treatment to simulate.
        shift_amount: Minutes to shift the population (e.g., +12.0).
        safeguard: If True, a cohort's expected survival percentage cannot drop below 
                   what it originally was in its source bin.
    """
    if survival_df is None or survival_df.empty:
        print("Error: Provided DataFrame is empty.")
        return None

    # Filter for the specific lane and time
    df_subset = survival_df[(survival_df['lane'] == lane) & 
                            (survival_df['time_relative'] == time_relative)].copy()
    
    if df_subset.empty:
        print(f"No data found for lane {lane} at time {time_relative}.")
        return None
        
    # Create a lookup dictionary for destination survival rates
    # Using bin_start as the key
    survival_lookup = dict(zip(df_subset['bin_start'], df_subset['survival_percentage']))
    
    results = []
    original_total_survivors = 0
    expected_total_survivors = 0
    total_population = df_subset['total_cells'].sum()
    
    for _, row in df_subset.iterrows():
        source_bin = row['bin_start']
        source_pct = row['survival_percentage']
        cells_moving = row['total_cells']
        
        # Where do these cells land?
        dest_bin = source_bin + shift_amount
        
        # What is the historic survival rate of the destination bin?
        # If the destination bin is beyond our data, assume 0% survival
        dest_pct = survival_lookup.get(dest_bin, 0.0)
        
        # Apply safeguard if requested
        if safeguard:
            applied_pct = max(source_pct, dest_pct)
        else:
            applied_pct = dest_pct
            
        # Calculate expected survivors for this cohort moving to the new bin
        expected_surv = cells_moving * (applied_pct / 100.0)
        
        original_total_survivors += row['survivors']
        expected_total_survivors += expected_surv
        
        results.append({
            'source_bin': source_bin,
            'dest_bin': dest_bin,
            'total_cells_moved': cells_moving,
            'source_survival_pct': source_pct,
            'dest_survival_pct': dest_pct,
            'applied_survival_pct': applied_pct,
            'expected_survivors': expected_surv
        })
        
    results_df = pd.DataFrame(results)
    
    # Calculate overall statistics
    orig_overall_pct = (original_total_survivors / total_population * 100) if total_population > 0 else 0
    expected_overall_pct = (expected_total_survivors / total_population * 100) if total_population > 0 else 0
    
    print(f"\n{'='*65}")
    print(f" SIMULATED POPULATION SHIFT: {shift_amount:+} mins")
    print(f" Lane: {lane} | Time Relative: {time_relative} min")
    print(f" Safeguard Enabled: {safeguard}")
    print(f"{'='*65}")
    print(f"Total Eligible Population (growing_before=True) : {total_population}")
    print(f"Original Survivors before shift                 : {original_total_survivors} ({orig_overall_pct:.2f}%)")
    print(f"Expected Survivors after shift                  : {expected_total_survivors:.1f} ({expected_overall_pct:.2f}%)")
    
    diff = expected_total_survivors - original_total_survivors
    print(f"Difference in Survivors                         : {diff:+.1f}")
    print(f"{'='*65}\n")
    
    return results_df

# Napari cell classification

def _maybe_downcast_raw(raw_np):
    """Downcasts float images to uint16 to save memory and Napari rendering lag."""
    if raw_np.dtype in (np.float32, np.float64):
        lo, hi = np.percentile(raw_np, (0.1, 99.9))
        if hi > lo: 
            return (np.clip((raw_np - lo) / (hi - lo), 0, 1) * 65535.0).astype(np.uint16)
    return raw_np

def _load_cell_data(mid, trenches_da, masks_da, frames_to_load, pc_channel, mask_to_uint8):
    """Worker function to pull raw arrays into memory."""
    raw_da = trenches_da[mid, :frames_to_load, pc_channel, :, :]
    mask_da = masks_da[mid, :frames_to_load, 0, :, :]
    
    raw_np, mask_np = da.compute(raw_da, mask_da)
    raw_np = np.ascontiguousarray(_maybe_downcast_raw(raw_np))
    mask_np = np.ascontiguousarray(mask_np)
    
    if mask_to_uint8 and mask_np.dtype != np.uint8: 
        mask_np = (mask_np != 0).astype(np.uint8, copy=False)
        
    return raw_np, mask_np

def _generate_growth_plot(mid, df, treatment_start, treatment_end):
    """Worker function to pre-render the matplotlib growth plot to an RGBA array."""
    cell_df = df[df['mother_id'] == mid]
    fig = plt.Figure(figsize=(8, 2), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Support for new data format (time / log_length_smoothed) and old format
    if not cell_df.empty:
        if 'time' in cell_df.columns and 'log_length_smoothed' in cell_df.columns:
            ax.plot(cell_df['time'].values, cell_df['log_length_smoothed'].values, linewidth=1)
        elif 'time_average' in cell_df.columns and 'log_area' in cell_df.columns:
            ax.plot(cell_df['time_average'].values, cell_df['log_area'].values, linewidth=1)
            
    if treatment_start is not None and treatment_end is not None:
        ax.axvspan(treatment_start, treatment_end, alpha=0.2, color='red')
        
    ax.set(xlabel='Time (min)', ylabel='log(Size)', title=f"ID {mid}")
    fig.tight_layout(pad=0.3)
    canvas.draw()
    
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    h, w = canvas.get_width_height()[::-1]
    rgb = buf.reshape((h, w, 4))[..., :3]
    plt.close(fig)
    
    return rgb

def preload_classifier_data(all_ids, processed_ids, trenches_da, masks_da, growing_df, 
                            frames_to_load, treatment_start, treatment_end, pc_channel=0, mask_to_uint8=True):
    """Handles threadpool loading of required data arrays."""
    max_workers = max(4, (os.cpu_count() or 8) - 2)
    ids_to_load = [mid for mid in all_ids if mid not in processed_ids]
    
    if not ids_to_load:
        return {}

    data_cache = {}
    print(f"Pre-loading {len(ids_to_load)} new survivor cells into RAM...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_mid = {
            executor.submit(_load_cell_data, mid, trenches_da, masks_da, frames_to_load, pc_channel, mask_to_uint8): mid 
            for mid in ids_to_load
        }
        growth_futures = {
            executor.submit(_generate_growth_plot, mid, growing_df, treatment_start, treatment_end): mid 
            for mid in ids_to_load
        }
        
        temp_results = {mid: {} for mid in ids_to_load}
        all_futures = list(future_to_mid.keys()) + list(growth_futures.keys())
        
        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Loading data & plots"):
            if future in future_to_mid:
                temp_results[future_to_mid[future]]['data'] = future.result()
            else:
                temp_results[growth_futures[future]]['plot'] = future.result()
                    
    for mid, res in temp_results.items():
        if 'data' in res and 'plot' in res:
            data_cache[mid] = (*res['data'], res['plot'])
            
    return data_cache

def run_cell_classifier(data_cache, all_ids, discard_mother_ids, persister_mother_ids, survivor_mother_ids, 
                        save_filepath=None, play_interval_ms=10, upscale_xy=(1, 2, 2), 
                        growth_scale=(0.375, 0.375), bottom_margin_wu=6.0):
    """Launches the Napari viewer with the preloaded cache."""
    processed = set(discard_mother_ids) | set(persister_mother_ids) | set(survivor_mother_ids)
    remaining_ids = [m for m in all_ids if m not in processed]
    
    # Helper to print stats when done or closed
    def print_summary(event=None):
        total = len(all_ids)
        n_discard = len(discard_mother_ids)
        n_persister = len(persister_mother_ids)
        # Survivor list includes persisters, so we separate them for the printout
        n_survivor_only = len([x for x in survivor_mother_ids if x not in persister_mother_ids])
        
        classified_set = set(discard_mother_ids) | set(survivor_mother_ids) | set(persister_mother_ids)
        classified = len(classified_set)
        
        print("\n" + "-"*30)
        print("   CLASSIFICATION SUMMARY   ")
        print("-"*30)
        print(f"Total Initial Cells    : {total}")
        print(f"Classified so far      : {classified} / {total}")
        print(f"  - Discarded          : {n_discard}")
        print(f"  - Standard Survivors : {n_survivor_only}")
        print(f"  - Persisters         : {n_persister}")
        if classified < total:
            print(f"Remaining to classify  : {total - classified}")
        print("-"*30 + "\n")

    if not remaining_ids:
        print("All selected survivors have been classified!")
        print_summary()
        return

    viewer = napari.Viewer()
    
    # Connect the summary function to the window closing event
    viewer.events.closing.connect(print_summary)
    
    state = {
        'index': 0, 'image_layer': None, 'mask_layer': None, 'growth_layer': None,
        'raw_buf': None, 'mask_buf': None, 'growth_buf': None, 'contrast_locked': False
    }
    
    timer = QTimer()
    timer.setInterval(play_interval_ms)
    
    def play_time():
        try:
            if state['image_layer'] is not None and state['image_layer'] in viewer.layers:
                viewer.dims.set_current_step(0, (viewer.dims.current_step[0] + 1) % state['image_layer'].data.shape[0])
            else:
                timer.stop()
        except:
            timer.stop()
            
    timer.timeout.connect(play_time)
    
    def _ensure_buffers(raw_data, mask_data, growth_img):
        if state['raw_buf'] is None:
            state['raw_buf'] = np.empty_like(raw_data)
            state['mask_buf'] = np.empty_like(mask_data)
            state['growth_buf'] = np.empty_like(growth_img)
            np.copyto(state['raw_buf'], raw_data)
            np.copyto(state['mask_buf'], mask_data)
            np.copyto(state['growth_buf'], growth_img)
            
            state['image_layer'] = viewer.add_image(state['raw_buf'], name="FOV", interpolation='nearest', scale=upscale_xy)
            
            if not state['contrast_locked']:
                clo, chi = np.percentile(raw_data, (0.5, 99.5))
                state['image_layer'].contrast_limits = (float(clo), float(chi))
                state['contrast_locked'] = True
                
            state['mask_layer'] = viewer.add_labels(state['mask_buf'], name="Mask", visible=False, opacity=0.5)
            
            fy_img, fx_img = raw_data.shape[-2], raw_data.shape[-1]
            sy_img, sx_img = upscale_xy[1], upscale_xy[2]
            sy_msk = (fy_img * sy_img) / max(mask_data.shape[-2], 1)
            sx_msk = (fx_img * sx_img) / max(mask_data.shape[-1], 1)
            state['mask_layer'].scale = (1, sy_msk, sx_msk)
            
            state['growth_layer'] = viewer.add_image(state['growth_buf'], name="Growth", rgb=True, blending='translucent')
            
            gh_world, gw_world = growth_img.shape[0] * growth_scale[0], growth_img.shape[1] * growth_scale[1]
            state['growth_layer'].scale = growth_scale
            state['growth_layer'].translate = ((fy_img * sy_img) - gh_world - bottom_margin_wu, ((fx_img * sx_img) - gw_world) * 0.5)
        else:
            np.copyto(state['raw_buf'], raw_data)
            np.copyto(state['mask_buf'], mask_data)
            np.copyto(state['growth_buf'], growth_img)
            state['image_layer'].refresh(); state['mask_layer'].refresh(); state['growth_layer'].refresh()
            
    def load_current():
        if state['index'] >= len(remaining_ids): return
        was_active = timer.isActive()
        if was_active: timer.stop()
        
        mid = remaining_ids[state['index']]
        if mid not in data_cache:
            print(f"\nWarning: ID {mid} not in cache. Skipping.")
            advance()
            return
            
        print(f"\rDisplaying mother_id {mid} ({state['index'] + 1}/{len(remaining_ids)})", end="")
        raw_data, mask_data, growth_img = data_cache[mid]
        
        with viewer.layers.events.blocker(), viewer.dims.events.current_step.blocker():
            _ensure_buffers(raw_data, mask_data, growth_img)
            viewer.layers.selection.active = state['image_layer']
            viewer.dims.set_current_step(0, 0)
            viewer.title = f"({state['index'] + 1}/{len(remaining_ids)}) ID = {mid} | 'd'=Discard | 's'=Survivor | 'p'=Persister | Space=Pause/Resume"
            
        if was_active: timer.start()
        
    def advance():
        if save_filepath:
            save_classification_lists(save_filepath, discard_mother_ids, persister_mother_ids, survivor_mother_ids, verbose=False)
            
        state['index'] += 1
        if state['index'] < len(remaining_ids): 
            load_current()
        else: 
            print("\nAll selected survivors have been classified!")
            if save_filepath:
                save_classification_lists(save_filepath, discard_mother_ids, persister_mother_ids, survivor_mother_ids, verbose=True)
            viewer.close() # This will automatically trigger print_summary() via the event hook

    @viewer.bind_key('Space')
    def toggle_play(viewer):
        if timer.isActive(): timer.stop(); print("\nPlayback paused")
        else: timer.start(); print("\nPlayback resumed")
            
    @viewer.bind_key('d')
    def mark_discard(viewer):
        mid = remaining_ids[state['index']]
        if mid not in discard_mother_ids: discard_mother_ids.append(mid)
        print(f"\nDiscarded: {mid}"); advance()
        
    @viewer.bind_key('p')
    def mark_persister(viewer):
        mid = remaining_ids[state['index']]
        if mid not in persister_mother_ids: persister_mother_ids.append(mid)
        if mid not in survivor_mother_ids: survivor_mother_ids.append(mid) 
        print(f"\nPersister (& Survivor): {mid}"); advance()
        
    @viewer.bind_key('s')
    def mark_survivor(viewer):
        mid = remaining_ids[state['index']]
        if mid not in survivor_mother_ids: survivor_mother_ids.append(mid)
        print(f"\nSurvivor: {mid}"); advance()
        
    load_current()
    timer.start()
    
    print(f"\nNapari window loaded. Progress will auto-save.")
    napari.run()

def start_classification_session(
    mother_machine_data: pd.DataFrame, 
    treatment_start: float, 
    treatment_end: float,
    config: dict,
    discard_list: list, 
    persister_list: list, 
    survivor_list: list, 
    cache_dict: dict,
    save_filepath: str = "classification_results.json",
    trenches_file: str = "trenches.zarr",
    masks_file: str = "masks_upscaled_filtered.zarr",
    include_lanes: list = [0, 1, 2, 3],
    max_t_for_review: int = None,
    auto_load: bool = True
):
    """Master function to prepare data and launch Napari."""
    print("Initializing classification session...")

    if auto_load and os.path.exists(save_filepath):
        total_in_memory = len(discard_list) + len(persister_list) + len(survivor_list)
        if total_in_memory == 0:
            print(f"Loading previous session data from {save_filepath}...")
            with open(save_filepath, 'r') as f:
                data = json.load(f)
                discard_list.extend(data.get("discard_mother_ids", []))
                persister_list.extend(data.get("persister_mother_ids", []))
                survivor_list.extend(data.get("survivor_mother_ids", []))
        else:
            print(f"Resuming active session. Found {total_in_memory} classified cells in memory.")
            
    dask.config.set({
        "array.slicing.split_large_chunks": True, 
        "optimization.fuse.active": True, 
        "scheduler": "threads"
    })

    # Retrieve PC channel dynamically from the config dictionary
    pc_channel = config.get("PC_channel", config.get("pc_channel", 0))

    trenches_da = da.from_zarr(zarr.open_array(trenches_file, mode="r"))
    masks_da = da.from_zarr(zarr.open_array(masks_file, mode="r"))

    try: 
        frames_to_load = int(mother_machine_data["timepoint"].max())
    except Exception: 
        frames_to_load = int(trenches_da.shape[1])
        
    frames_to_load = int(min(frames_to_load, trenches_da.shape[1]))
    if max_t_for_review is not None: 
        frames_to_load = min(frames_to_load, int(max_t_for_review))

    # Strict check to ensure both growth flags exist and are True
    if "growing_before" not in mother_machine_data.columns or "growing_after" not in mother_machine_data.columns:
        raise ValueError("mother_machine_data is missing 'growing_before' or 'growing_after' columns.")

    df_filtered = mother_machine_data[
        (mother_machine_data['lane'].isin(include_lanes)) &
        (mother_machine_data['growing_before'] == True) &
        (mother_machine_data['growing_after'] == True)
    ].copy()

    all_ids = sorted(df_filtered['mother_id'].unique())

    if not all_ids:
        print("Error: No surviving cells found matching the criteria in the specified lanes.")
        return

    processed_ids = set(discard_list) | set(persister_list) | set(survivor_list)
    ids_to_skip_loading = processed_ids.union(cache_dict.keys())

    new_data = preload_classifier_data(
        all_ids=all_ids,
        processed_ids=ids_to_skip_loading,
        trenches_da=trenches_da,
        masks_da=masks_da,
        growing_df=df_filtered,
        frames_to_load=frames_to_load,
        treatment_start=treatment_start,
        treatment_end=treatment_end,
        pc_channel=pc_channel
    )
    
    cache_dict.update(new_data)
    for pid in processed_ids:
        cache_dict.pop(pid, None)

    run_cell_classifier(
        data_cache=cache_dict,
        all_ids=all_ids,
        discard_mother_ids=discard_list,
        persister_mother_ids=persister_list,
        survivor_mother_ids=survivor_list,
        save_filepath=save_filepath
    )

def save_classification_lists(filepath: str, discard_list: list, persister_list: list, survivor_list: list, verbose: bool = True):
    """Saves the current state of classification lists to a JSON file."""
    data = {
        "discard_mother_ids": [int(x) for x in discard_list],
        "persister_mother_ids": [int(x) for x in persister_list],
        "survivor_mother_ids": [int(x) for x in survivor_list]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    if verbose:
        print(f"\nProgress saved to: {filepath}")