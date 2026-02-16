from __future__ import annotations

"""
Compatibility shim for legacy imports from model.model.

This module intentionally has no import-time side effects (no data loading,
no fitting, no plotting). Use package modules directly for new code.
"""

import functools
import warnings

from .core import (
    USE_LOG_HAZARDS as use_log_hazards,
    ConditionPrediction,
    STDPModel,
    STDPParams,
    _I_polyexp,
    _J_and_dJ_trunc,
    _J_trunc,
    _erlang_sf,
    hS,
    hT,
    m,
    rST,
)
from .data import load_data, load_standard_errors
from .fit import FitConfig, FitResult
from .fit import compute_class_weights as _compute_class_weights_new
from .fit import fit_over_k_lag as _fit_over_k_lag_new
from .fit import fit_stdp as _fit_stdp_new
from .objective import (
    DEFAULT_FREE_KEYS,
    NUM_BOUNDS,
    _log_bounds,
    _nll_dirichlet,
    _observed_matrix,
    _pack_params,
    _pred_matrix,
    _tolerance_penalty,
    _unpack_params,
    make_objective as _make_objective_new,
)
from .plotting_advanced import (
    J_discounted as _J_discounted_new,
    composition_vs_age as _composition_vs_age_new,
    composition_vs_concentration as _composition_vs_concentration_new,
    composition_vs_tau as _composition_vs_tau_new,
    descendant_shares as _descendant_shares_new,
    draw_cmy_ternary_key as _draw_cmy_ternary_key_new,
    draw_survivor_brightness_bar as _draw_survivor_brightness_bar_new,
    grid_descendant_composition as _grid_descendant_composition_new,
    grid_survivor_composition as _grid_survivor_composition_new,
    make_cmyk_rgb as _make_cmyk_rgb_new,
    show_contour as _show_contour_new,
    show_heatmap_with_keys_labeled as _show_heatmap_with_keys_labeled_new,
    show_rgb_heatmap as _show_rgb_heatmap_new,
)
from .plotting_basic import (
    configure_plot_style as _configure_plot_style_new,
    evaluate_model as _evaluate_model_new,
    plot_all_profiles as _plot_all_profiles_new,
    plot_fraction_comparison as _plot_fraction_comparison_new,
)
from .uncertainty import UncertaintyConfig
from .uncertainty import build_objectives as _build_objectives_new
from .uncertainty import compute_prediction_ci as _compute_prediction_ci_new
from .uncertainty import hessian_central as _hessian_central_new
from .uncertainty import profile_likelihood as _profile_likelihood_new
from .uncertainty import robust_inverse as _robust_inverse_new

_NUM_BOUNDS = NUM_BOUNDS
_COMPAT_DATA = None
_COMPAT_DF_FIT = None


def _warn(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`model.model.{old_name}` is deprecated; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _deprecated_passthrough(old_name: str, new_name: str, fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        _warn(old_name, new_name)
        return fn(*args, **kwargs)

    return wrapped


def load_data_from_csv(path):
    global _COMPAT_DATA
    _warn("load_data_from_csv", "model.data.load_data")
    _COMPAT_DATA = load_data(path)
    return _COMPAT_DATA


def load_se_from_csv(path):
    _warn("load_se_from_csv", "model.data.load_standard_errors")
    return load_standard_errors(path)


def fit_stdp(*args, **kwargs):
    """
    Legacy return shape:
      (best_params, best_value, optimizer_result)

    New API returns FitResult in model.fit.fit_stdp.
    """
    _warn("fit_stdp", "model.fit.fit_stdp")
    result = _fit_stdp_new(*args, **kwargs)
    return result.best_params, result.best_value, result.optimizer_result


def fit_over_k_lag(*args, **kwargs):
    _warn("fit_over_k_lag", "model.fit.fit_over_k_lag")
    return _fit_over_k_lag_new(*args, **kwargs)


def evaluate_model(model: STDPModel, data=None, se_df=None):
    global _COMPAT_DF_FIT
    _warn("evaluate_model", "model.plotting_basic.evaluate_model")

    dataset = data if data is not None else _COMPAT_DATA
    if dataset is None:
        raise ValueError(
            "No DATA provided. Pass `data=...` or call load_data_from_csv(...) first."
        )

    _COMPAT_DF_FIT = _evaluate_model_new(model, dataset, se_df=se_df)
    return _COMPAT_DF_FIT


def plot_fraction_comparison(frac: str, df_fit=None, **kwargs):
    _warn("plot_fraction_comparison", "model.plotting_basic.plot_fraction_comparison")

    frame = df_fit if df_fit is not None else _COMPAT_DF_FIT
    if frame is None:
        raise ValueError(
            "No df_fit provided. Pass `df_fit=...` or call evaluate_model(...) first."
        )

    return _plot_fraction_comparison_new(frac=frac, df_fit=frame, **kwargs)


# Deprecated passthroughs for relocated entry points.
make_objective = _deprecated_passthrough(
    "make_objective", "model.objective.make_objective", _make_objective_new
)
compute_class_weights = _deprecated_passthrough(
    "compute_class_weights", "model.fit.compute_class_weights", _compute_class_weights_new
)
build_objectives = _deprecated_passthrough(
    "build_objectives", "model.uncertainty.build_objectives", _build_objectives_new
)
profile_likelihood = _deprecated_passthrough(
    "profile_likelihood", "model.uncertainty.profile_likelihood", _profile_likelihood_new
)
compute_prediction_ci = _deprecated_passthrough(
    "compute_prediction_ci", "model.uncertainty.compute_prediction_ci", _compute_prediction_ci_new
)
hessian_central = _deprecated_passthrough(
    "hessian_central", "model.uncertainty.hessian_central", _hessian_central_new
)
robust_inverse = _deprecated_passthrough(
    "robust_inverse", "model.uncertainty.robust_inverse", _robust_inverse_new
)
plot_all_profiles = _deprecated_passthrough(
    "plot_all_profiles", "model.plotting_basic.plot_all_profiles", _plot_all_profiles_new
)
configure_plot_style = _deprecated_passthrough(
    "configure_plot_style", "model.plotting_basic.configure_plot_style", _configure_plot_style_new
)

composition_vs_tau = _deprecated_passthrough(
    "composition_vs_tau", "model.plotting_advanced.composition_vs_tau", _composition_vs_tau_new
)
composition_vs_concentration = _deprecated_passthrough(
    "composition_vs_concentration",
    "model.plotting_advanced.composition_vs_concentration",
    _composition_vs_concentration_new,
)
composition_vs_age = _deprecated_passthrough(
    "composition_vs_age", "model.plotting_advanced.composition_vs_age", _composition_vs_age_new
)
grid_survivor_composition = _deprecated_passthrough(
    "grid_survivor_composition",
    "model.plotting_advanced.grid_survivor_composition",
    _grid_survivor_composition_new,
)
make_cmyk_rgb = _deprecated_passthrough(
    "make_cmyk_rgb", "model.plotting_advanced.make_cmyk_rgb", _make_cmyk_rgb_new
)
show_rgb_heatmap = _deprecated_passthrough(
    "show_rgb_heatmap", "model.plotting_advanced.show_rgb_heatmap", _show_rgb_heatmap_new
)
draw_cmy_ternary_key = _deprecated_passthrough(
    "draw_cmy_ternary_key",
    "model.plotting_advanced.draw_cmy_ternary_key",
    _draw_cmy_ternary_key_new,
)
draw_survivor_brightness_bar = _deprecated_passthrough(
    "draw_survivor_brightness_bar",
    "model.plotting_advanced.draw_survivor_brightness_bar",
    _draw_survivor_brightness_bar_new,
)
J_discounted = _deprecated_passthrough(
    "J_discounted", "model.plotting_advanced.J_discounted", _J_discounted_new
)
descendant_shares = _deprecated_passthrough(
    "descendant_shares", "model.plotting_advanced.descendant_shares", _descendant_shares_new
)
grid_descendant_composition = _deprecated_passthrough(
    "grid_descendant_composition",
    "model.plotting_advanced.grid_descendant_composition",
    _grid_descendant_composition_new,
)
show_heatmap_with_keys_labeled = _deprecated_passthrough(
    "show_heatmap_with_keys_labeled",
    "model.plotting_advanced.show_heatmap_with_keys_labeled",
    _show_heatmap_with_keys_labeled_new,
)
show_contour = _deprecated_passthrough(
    "show_contour", "model.plotting_advanced.show_contour", _show_contour_new
)


def __getattr__(name):
    if name == "DATA":
        warnings.warn(
            "`model.model.DATA` is no longer auto-loaded. Use load_data(...) explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _COMPAT_DATA
    raise AttributeError(name)


def _demo_main() -> None:
    print("This module is a compatibility shim.")
    print("Use model.notebook.ModelWorkflow for notebook-first usage.")


if __name__ == "__main__":
    _demo_main()
