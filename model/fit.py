from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .core import STDPParams
from .objective import (
    DEFAULT_FREE_KEYS,
    _log_bounds,
    _observed_matrix,
    _pack_params,
    _unpack_params,
    make_objective,
)


@dataclass
class FitConfig:
    profile: str = "fast"
    n_starts: Optional[int] = None
    seed: int = 2025
    maxiter: Optional[int] = None
    ftol: float = 1e-9
    gtol: float = 1e-7
    jitter_sigma: float = 3.5
    method: str = "L-BFGS-B"

    def resolved(self) -> "FitConfig":
        cfg = FitConfig(**self.__dict__)
        if cfg.profile not in {"fast", "thorough", "custom"}:
            raise ValueError("profile must be one of: fast, thorough, custom")

        if cfg.profile == "fast":
            if cfg.n_starts is None:
                cfg.n_starts = 12
            if cfg.maxiter is None:
                cfg.maxiter = 400
        elif cfg.profile == "thorough":
            if cfg.n_starts is None:
                cfg.n_starts = 240
            if cfg.maxiter is None:
                cfg.maxiter = 1500
        else:
            if cfg.n_starts is None or cfg.maxiter is None:
                raise ValueError(
                    "Custom fit profile requires explicit n_starts and maxiter."
                )

        return cfg


@dataclass
class FitResult:
    best_params: STDPParams
    best_value: float
    best_x: np.ndarray
    free_keys: List[str]
    optimizer_result: object
    config: FitConfig


@dataclass
class KLagSearchResult:
    best_k_lag: int
    best_fit: FitResult
    all_fits: Dict[int, FitResult]


def fit_stdp(
    data,
    start: STDPParams,
    free_keys: List[str] | None = None,
    config: FitConfig | None = None,
    kappa: float = 5000.0,
    lam_pen: float = 1e-2,
    rho: float = 0.8,
    ages_for_pen=(24.0, 48.0, 72.0),
    class_weights=None,
    lam_hill_constraint: float = 1e6,
) -> FitResult:
    cfg = (config or FitConfig()).resolved()
    rng = np.random.default_rng(cfg.seed)

    objective, keys = make_objective(
        data,
        free_keys=free_keys,
        kappa=kappa,
        lam_pen=lam_pen,
        lam_hill_constraint=lam_hill_constraint,
        rho=rho,
        ages_for_pen=ages_for_pen,
        class_weights=class_weights,
    )

    x0 = _pack_params(start, keys)
    bounds = _log_bounds(keys)

    best_val = np.inf
    best_x = x0.copy()
    best_res = None

    for _ in range(cfg.n_starts):
        x_init = x0 + rng.normal(0.0, cfg.jitter_sigma, size=x0.shape)
        res = minimize(
            lambda z: objective(z, start),
            x_init,
            bounds=bounds,
            method=cfg.method,
            options={"maxiter": cfg.maxiter, "ftol": cfg.ftol, "gtol": cfg.gtol},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = res.x
            best_res = res

    best_params = _unpack_params(best_x, start, keys)
    return FitResult(
        best_params=best_params,
        best_value=best_val,
        best_x=best_x,
        free_keys=keys,
        optimizer_result=best_res,
        config=cfg,
    )


def compute_class_weights(
    P_obs: np.ndarray, beta: float = 0.75, min_w: float = 0.2, max_w: float = 20.0
) -> np.ndarray:
    prev = np.clip(P_obs.mean(axis=0), 1e-6, 1.0)
    w = (prev.mean() / prev) ** beta
    return np.clip(w, min_w, max_w)


def _fit_for_k(
    k: int,
    data,
    start: STDPParams,
    free_keys,
    config: FitConfig,
    kappa: float,
    lam_pen: float,
    lam_hill_constraint: float,
    rho: float,
    ages_for_pen,
    class_weights,
):
    seed_k = int(config.seed + k)
    config_k = replace(config, seed=seed_k)
    start_k = replace(start, k_lag=int(k))

    fit = fit_stdp(
        data=data,
        start=start_k,
        free_keys=free_keys,
        config=config_k,
        kappa=kappa,
        lam_pen=lam_pen,
        lam_hill_constraint=lam_hill_constraint,
        rho=rho,
        ages_for_pen=ages_for_pen,
        class_weights=class_weights,
    )
    return k, fit


def fit_over_k_lag(
    k_values: Sequence[int],
    data,
    start: STDPParams,
    free_keys: List[str] | None = None,
    config: FitConfig | None = None,
    kappa: float = 5000.0,
    lam_pen: float = 1e-2,
    rho: float = 0.8,
    ages_for_pen=(24.0, 48.0, 72.0),
    class_weights=None,
    lam_hill_constraint: float = 1e6,
    n_jobs: int = -1,
) -> KLagSearchResult:
    cfg = (config or FitConfig()).resolved()
    keys = DEFAULT_FREE_KEYS if free_keys is None else list(free_keys)

    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_for_k)(
                k,
                data,
                start,
                keys,
                cfg,
                kappa,
                lam_pen,
                lam_hill_constraint,
                rho,
                ages_for_pen,
                class_weights,
            )
            for k in k_values
        )
    except Exception:
        results = [
            _fit_for_k(
                k,
                data,
                start,
                keys,
                cfg,
                kappa,
                lam_pen,
                lam_hill_constraint,
                rho,
                ages_for_pen,
                class_weights,
            )
            for k in k_values
        ]

    fit_map = {k: fit for k, fit in results}
    best_k, best_fit = min(results, key=lambda item: item[1].best_value)
    return KLagSearchResult(best_k_lag=best_k, best_fit=best_fit, all_fits=fit_map)
