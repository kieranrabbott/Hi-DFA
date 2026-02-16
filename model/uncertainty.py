from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .core import STDPModel, STDPParams
from .objective import (
    _log_bounds,
    _nll_dirichlet,
    _observed_matrix,
    _pack_params,
    _pred_matrix,
    _tolerance_penalty,
    _unpack_params,
    make_objective,
)


@dataclass
class UncertaintyConfig:
    n_draws: int = 5000
    q_lo: float = 2.5
    q_hi: float = 97.5
    ridge: float = 1e-9
    random_seed: int = 1


def build_objectives(
    template: STDPParams,
    data,
    free_keys,
    kappa=5000.0,
    lam_pen=1e-2,
    rho=0.8,
    ages_for_pen=(24.0, 48.0, 72.0),
    class_weights=None,
):
    P_obs = _observed_matrix(data)

    def unpack(z):
        return _unpack_params(z, template, free_keys)

    def nll_only(z):
        p = unpack(z)
        model = STDPModel(p)
        P_pred = _pred_matrix(model, data)
        return _nll_dirichlet(P_obs, P_pred, kappa=kappa, class_weights=class_weights)

    def pen_only(z):
        p = unpack(z)
        return _tolerance_penalty(p, rho=rho, ages=ages_for_pen)

    def total(z):
        return nll_only(z) + lam_pen * pen_only(z)

    return total, nll_only, pen_only, unpack


def hessian_central(f, x, h=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    if h is None:
        h = 1e-3 * np.maximum(1.0, np.abs(x))
    h = np.asarray(h, dtype=float)

    f0 = f(x)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        f_plus = f(x + ei)
        f_minus = f(x - ei)
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h[i] ** 2)

    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            f_pp = f(x + ei + ej)
            f_pm = f(x + ei - ej)
            f_mp = f(x - ei + ej)
            f_mm = f(x - ei - ej)
            hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = hij
            H[j, i] = hij

    return H


def robust_inverse(H, ridge=1e-9):
    Hs = 0.5 * (H + H.T)
    try:
        return np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Hs)
        w_clipped = np.clip(w, ridge, None)
        return (V * (1.0 / w_clipped)) @ V.T


def compute_prediction_ci(
    draws_log: np.ndarray,
    template: STDPParams,
    free_keys: list,
    data,
    q_lo: float = 2.5,
    q_hi: float = 97.5,
    fracs: tuple = ("incomplete", "induced", "preexisting"),
) -> pd.DataFrame:
    n_draws = draws_log.shape[0]
    n_cond = len(data)

    store = {f: np.empty((n_draws, n_cond), dtype=float) for f in fracs}

    for i, log_x in enumerate(draws_log):
        p = _unpack_params(log_x, template, free_keys)
        model = STDPModel(p)
        for j, (_, C, tau, a, _) in enumerate(data):
            pred = model.predict_condition(C, tau, a)
            for frac in fracs:
                store[frac][i, j] = float(getattr(pred, frac))

    rows = []
    for j, (name, C, tau, a, _) in enumerate(data):
        row = {"condition": name, "C": C, "tau": tau, "age": a}
        for frac in fracs:
            v = store[frac][:, j]
            row[f"pred_ci_lo_{frac}"] = float(np.percentile(v, q_lo))
            row[f"pred_ci_hi_{frac}"] = float(np.percentile(v, q_hi))
        rows.append(row)

    return pd.DataFrame(rows)


def draw_laplace_samples(
    best_params: STDPParams,
    free_keys: Sequence[str],
    data,
    config: UncertaintyConfig | None = None,
    kappa: float = 5000.0,
    lam_pen: float = 1e-2,
    rho: float = 0.8,
    ages_for_pen=(24.0, 48.0, 72.0),
    class_weights=None,
):
    cfg = config or UncertaintyConfig()

    objective_total, nll_only, _, _ = build_objectives(
        template=best_params,
        data=data,
        free_keys=list(free_keys),
        kappa=kappa,
        lam_pen=lam_pen,
        rho=rho,
        ages_for_pen=ages_for_pen,
        class_weights=class_weights,
    )

    x_hat = _pack_params(best_params, list(free_keys))
    H_data = hessian_central(nll_only, x_hat)
    Sigma_data = robust_inverse(H_data, ridge=cfg.ridge)

    rng = np.random.default_rng(cfg.random_seed)
    draws_log = rng.multivariate_normal(mean=x_hat, cov=Sigma_data, size=cfg.n_draws)
    return draws_log, x_hat, Sigma_data


def profile_likelihood(
    param_key: str,
    best_params: STDPParams,
    best_x: np.ndarray,
    free_keys,
    data,
    n_points: int = 40,
    delta_log: float = 1.5,
    **obj_kwargs,
):
    objective, _ = make_objective(data, free_keys=free_keys, **obj_kwargs)

    idx = free_keys.index(param_key)
    x_opt_val = best_x[idx]
    nll_opt = objective(best_x, best_params)

    grid = np.linspace(x_opt_val - delta_log, x_opt_val + delta_log, n_points)
    profile_nll = []

    inner_keys = [k for k in free_keys if k != param_key]
    inner_obj, _ = make_objective(data, free_keys=inner_keys, **obj_kwargs)
    inner_bounds = _log_bounds(inner_keys)
    inner_x0 = np.array([best_x[free_keys.index(k)] for k in inner_keys])

    for log_val in grid:
        fixed = STDPParams(**best_params.__dict__)
        setattr(fixed, param_key, float(np.exp(log_val)))

        res = minimize(
            lambda z: inner_obj(z, fixed),
            inner_x0,
            bounds=inner_bounds,
            method="L-BFGS-B",
            options=dict(maxiter=500, ftol=1e-9),
        )
        profile_nll.append(res.fun)

    profile_nll = np.array(profile_nll)
    threshold_95 = nll_opt + 1.92
    return grid, profile_nll, threshold_95, nll_opt
