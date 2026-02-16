from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .core import STDPModel, STDPParams, hS, hT


EPS_LOG_LOWER = 1e-8

NUM_BOUNDS = {
    "mu0": (0.03, 40.0),
    "mu24p": (0.0, 40.0),
    "kS_kT_ratio": (1.0, 1000.0),
    "kT": (1e-4, 100.0),
    "kST": (1e-3, 100.0),
    "n": (0.0, 40.5),
    "nST": (0.0, 40.5),
    "a50": (10.0, 300000.0),
    "r0": (0.0, 1.0),
}

DEFAULT_FREE_KEYS = [
    "mu0",
    "mu24p",
    "kS_kT_ratio",
    "kT",
    "kST",
    "n",
    "nST",
    "a50",
    "r0",
]


def _observed_matrix(data) -> np.ndarray:
    rows = []
    for _, _, _, _, fr in data:
        s = float(fr["incomplete"])
        t = float(fr["induced"])
        d = float(fr["preexisting"])
        dead = max(0.0, 1.0 - (s + t + d))
        rows.append([s, t, d, dead])

    P = np.asarray(rows, dtype=float)
    P = np.clip(P, 0.0, 1.0)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P


def _pred_matrix(model: STDPModel, data) -> np.ndarray:
    rows = []
    for _, C, tau, a, _ in data:
        pr = model.predict_condition(C, tau, a)
        row = np.array(
            [pr.incomplete, pr.induced, pr.preexisting, pr.dead],
            dtype=float,
        )
        row = np.clip(row, 0.0, 1.0)
        row = row / max(row.sum(), 1e-12)
        rows.append(row)

    return np.vstack(rows)


def _nll_dirichlet(
    p_hat: np.ndarray,
    p_pred: np.ndarray,
    kappa: float = 5000.0,
    class_weights=None,
    eps: float = 1e-12,
) -> float:
    Pp = np.clip(p_pred, eps, 1.0 - eps)
    Pp = Pp / Pp.sum(axis=1, keepdims=True)

    Ph = np.clip(p_hat, eps, 1.0 - eps)
    Ph = Ph / Ph.sum(axis=1, keepdims=True)

    alpha = kappa * Ph

    if class_weights is None:
        w = np.ones(Pp.shape[1], dtype=float)
    else:
        w = np.asarray(class_weights, dtype=float)
        if w.ndim != 1 or w.size != Pp.shape[1]:
            raise ValueError("class_weights must be length 4")
        if np.any(w <= 0):
            raise ValueError("class_weights must be strictly positive")

    ll = np.sum(w * (alpha - 1.0) * np.log(Pp))
    return -float(ll)


def _tolerance_penalty(
    p: STDPParams,
    rho: float = 0.8,
    C_min: float = 0.05,
    C_max: float = 300.0,
    nC: int = 61,
    ages: Tuple[float, ...] = (24.0, 48.0, 72.0),
    k_soft: float = 40.0,
) -> float:
    Cs = np.geomspace(C_min, C_max, nC)
    total = 0.0

    for a in ages:
        for C in Cs:
            hs = hS(C, p.kT, p.kS_kT_ratio, p.n)
            ht = hT(C, p.kT, p.n)

            if hs <= 1e-12:
                continue
            ratio = ht / max(hs, 1e-12)
            total += np.log1p(np.exp(k_soft * (ratio - rho))) / k_soft

    denom = nC * max(len(ages), 1)
    return float(total / max(denom, 1))


def _pack_params(p: STDPParams, free_keys: List[str]) -> np.ndarray:
    vals = []
    for key in free_keys:
        val = float(getattr(p, key))
        if val <= 0.0:
            raise ValueError(
                f"Parameter '{key}' must be > 0 for log-parameterization; got {val}."
            )
        vals.append(np.log(val))
    return np.asarray(vals, dtype=float)


def _unpack_params(x: np.ndarray, template: STDPParams, free_keys: List[str]) -> STDPParams:
    q = STDPParams(**asdict(template))
    for val, key in zip(x, free_keys, strict=False):
        setattr(q, key, float(np.exp(val)))
    return q


def _log_bounds(free_keys: List[str]) -> List[Tuple[float, float]]:
    bounds = []
    for key in free_keys:
        if key not in NUM_BOUNDS:
            raise KeyError(f"No numeric bounds configured for key '{key}'.")
        lo, hi = NUM_BOUNDS[key]
        lo_eff = lo if lo > 0 else EPS_LOG_LOWER
        if hi <= lo_eff:
            raise ValueError(
                f"Invalid bounds for '{key}': lower={lo_eff} upper={hi}."
            )
        bounds.append((np.log(lo_eff), np.log(hi)))
    return bounds


def make_objective(
    data,
    free_keys: List[str] | None = None,
    kappa: float = 5000.0,
    lam_pen: float = 1e-2,
    rho: float = 0.8,
    ages_for_pen: Tuple[float, ...] = (24.0, 48.0, 72.0),
    class_weights=None,
):
    free_keys = DEFAULT_FREE_KEYS if free_keys is None else list(free_keys)
    P_obs = _observed_matrix(data)

    def objective(x: np.ndarray, template: STDPParams) -> float:
        cand = _unpack_params(x, template, free_keys)
        model = STDPModel(cand)
        P_pred = _pred_matrix(model, data)
        nll = _nll_dirichlet(P_obs, P_pred, kappa=kappa, class_weights=class_weights)
        pen = _tolerance_penalty(cand, rho=rho, ages=ages_for_pen)
        return float(nll + lam_pen * pen)

    return objective, free_keys
