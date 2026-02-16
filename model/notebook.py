from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .core import STDPModel, STDPParams
from .data import load_data, load_standard_errors
from .fit import FitConfig, FitResult, fit_stdp
from .objective import DEFAULT_FREE_KEYS
from .plotting_basic import configure_plot_style, evaluate_model, plot_fraction_comparison
from .uncertainty import (
    UncertaintyConfig,
    compute_prediction_ci as compute_prediction_ci_fn,
    draw_laplace_samples,
)


@dataclass
class WorkflowConfig:
    fit_config: FitConfig = field(default_factory=FitConfig)
    kappa: float = 5000.0
    lam_pen: float = 1e-2
    lam_hill_constraint: float = 1e6
    rho: float = 0.8
    ages_for_pen: tuple[float, ...] = (24.0, 48.0, 72.0)
    class_weights: tuple[float, float, float, float] | None = None
    auto_style: bool = True
    plot_style: str = "nature"


class ModelWorkflow:
    def __init__(
        self,
        params: STDPParams | None = None,
        config: WorkflowConfig | None = None,
    ):
        self.params = params or STDPParams()
        self.config = config or WorkflowConfig()

        self.model = STDPModel(self.params)
        self.data = None
        self.fit_result: FitResult | None = None
        self.df_eval: pd.DataFrame | None = None
        self.free_keys = list(DEFAULT_FREE_KEYS)
        self._style_applied = False

    def _ensure_data(self, data=None):
        if data is not None:
            return data
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_data(...) first.")
        return self.data

    def _maybe_style(self):
        if self.config.auto_style and not self._style_applied:
            configure_plot_style(self.config.plot_style)
            self._style_applied = True

    def load_data(self, source):
        self.data = load_data(source)
        return self.data

    def fit(
        self,
        profile: str | None = None,
        free_keys: Sequence[str] | None = None,
        config: FitConfig | None = None,
    ) -> FitResult:
        data = self._ensure_data()

        fit_cfg = config or self.config.fit_config
        if profile is not None:
            fit_cfg = replace(fit_cfg, profile=profile)

        keys = list(DEFAULT_FREE_KEYS if free_keys is None else free_keys)

        result = fit_stdp(
            data=data,
            start=self.params,
            free_keys=keys,
            config=fit_cfg,
            kappa=self.config.kappa,
            lam_pen=self.config.lam_pen,
            lam_hill_constraint=self.config.lam_hill_constraint,
            rho=self.config.rho,
            ages_for_pen=self.config.ages_for_pen,
            class_weights=self.config.class_weights,
        )

        self.fit_result = result
        self.free_keys = result.free_keys
        self.params = result.best_params
        self.model = STDPModel(self.params)
        return result

    def evaluate(self, se=None, data=None) -> pd.DataFrame:
        ds = self._ensure_data(data)
        se_df = None
        if se is not None:
            se_df = load_standard_errors(se)

        self.df_eval = evaluate_model(self.model, ds, se_df=se_df)
        return self.df_eval

    def predict(self, C: float, tau: float, age: float):
        return self.model.predict_condition(C=C, tau=tau, a=age)

    def predict_grid(self, C_grid, tau_grid, age: float) -> pd.DataFrame:
        C_grid = np.asarray(C_grid, dtype=float)
        tau_grid = np.asarray(tau_grid, dtype=float)

        rows = []
        for tau in tau_grid:
            for C in C_grid:
                pred = self.model.predict_condition(C=float(C), tau=float(tau), a=float(age))
                rows.append(
                    {
                        "C": float(C),
                        "tau": float(tau),
                        "age": float(age),
                        "pred_incomplete": pred.incomplete,
                        "pred_induced": pred.induced,
                        "pred_preexisting": pred.preexisting,
                        "pred_dead": pred.dead,
                    }
                )

        return pd.DataFrame(rows)

    def plot_fraction(self, frac: str, df: pd.DataFrame | None = None, **kwargs):
        self._maybe_style()
        frame = df if df is not None else self.df_eval
        if frame is None:
            frame = self.evaluate()
        return plot_fraction_comparison(frac=frac, df_fit=frame, **kwargs)

    def compute_prediction_ci(
        self,
        draws_log: np.ndarray | None = None,
        uncertainty: UncertaintyConfig | None = None,
    ) -> pd.DataFrame:
        data = self._ensure_data()

        if self.fit_result is None:
            raise ValueError("Fit must be run before CI computation.")

        cfg = uncertainty or UncertaintyConfig()

        if draws_log is None:
            draws_log, _, _ = draw_laplace_samples(
                best_params=self.fit_result.best_params,
                free_keys=self.fit_result.free_keys,
                data=data,
                config=cfg,
                kappa=self.config.kappa,
                lam_pen=self.config.lam_pen,
                lam_hill_constraint=self.config.lam_hill_constraint,
                rho=self.config.rho,
                ages_for_pen=self.config.ages_for_pen,
                class_weights=self.config.class_weights,
            )

        ci_df = compute_prediction_ci_fn(
            draws_log=draws_log,
            template=self.fit_result.best_params,
            free_keys=self.fit_result.free_keys,
            data=data,
            q_lo=cfg.q_lo,
            q_hi=cfg.q_hi,
        )

        if self.df_eval is not None:
            self.df_eval = self.df_eval.merge(ci_df, on=["condition", "C", "tau", "age"], how="left")

        return ci_df
