from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .core import STDPModel, STDPParams, get_hill_hazard_flags, hS, hT, rST


EPS_LOG_LOWER = 1e-8


def effective_lag_rate(p: STDPParams, a: float) -> float:
    """Effective lag rate accounting for the hybrid Erlang + Exp mixture.

    The mean lag time is  E[L] = w · μ_Erlang(a) + (1−w) / λ_slow ,
    and the effective rate is 1 / E[L].

    When ``w_lag ≈ 0`` the Erlang component is negligible and the effective
    rate collapses to ``lam_slow``, *not* the Erlang rate ``k_lag / μ(a)``.
    This distinction matters for the regime-dominance and Hill-lambda
    constraint penalties.
    """
    model = STDPModel(p)
    mu_erlang = model.mean_lag(a)
    lam_s = model.slow_rate(a)
    mean_lag = p.w_lag * mu_erlang + (1.0 - p.w_lag) / max(lam_s, 1e-9)
    return 1.0 / max(mean_lag, 1e-9)

NUM_BOUNDS = {
    "w_lag": (0, 1),
    "lam_slow": (1e-6, 10.0),
    "mu0": (0.03, 400.0),
    "mu24p": (0.0, 400.0),
    "kS_kT_ratio": (1.0, 1000.0),
    "kT": (1e-4, 100.0),
    "kST": (1e-3, 1000.0),
    "K": (1e-6, 1e3),
    "KST": (1e-6, 1e6),
    "n": (0.0, 4.5),
    "nST": (0.0, 4.5),
    "a50": (0.0, 300000.0),
    "r0": (0.0, 1.0),
    "nS": (0.0, 4.5),
    "nT": (0.0, 4.5),
    "KS": (1e-6, 1e3),
    "KT": (1e-6, 1e3),
    "mu_s0": (0.1, 1e6),
    "mu_s24p": (1e-6, 50000.0),
}

DEFAULT_FREE_KEYS = [
    "w_lag",
    "lam_slow",
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


def _survivor_composition_nll(
    model: STDPModel,
    data,
    kappa_surv: float = 500.0,
    eps: float = 1e-12,
) -> float:
    """Dirichlet NLL on the renormalised survivor composition (S, T, D).

    The standard four-class Dirichlet NLL is dominated by the *dead* class
    at high antibiotic concentrations (often >99 % dead), so the model
    receives almost no gradient signal about *which* state the survivors
    belong to.  This auxiliary NLL renormalises observed and predicted
    fractions to (S, T, D) only, weighting the composition equally across
    all concentrations regardless of total survival rate.
    """
    total_nll = 0.0

    for _, C, tau, a, fr in data:
        obs_s = float(fr["incomplete"])
        obs_t = float(fr["induced"])
        obs_d = float(fr["preexisting"])
        obs_total = obs_s + obs_t + obs_d

        if obs_total < eps:
            continue

        phi_obs = np.array([obs_s, obs_t, obs_d]) / obs_total
        phi_obs = np.clip(phi_obs, eps, 1.0 - eps)
        phi_obs = phi_obs / phi_obs.sum()

        pr = model.predict_condition(C, tau, a)
        pred_s = max(float(pr.incomplete), 0.0)
        pred_t = max(float(pr.induced), 0.0)
        pred_d = max(float(pr.preexisting), 0.0)
        pred_total = pred_s + pred_t + pred_d

        if pred_total < eps:
            total_nll += kappa_surv * 10.0
            continue

        phi_pred = np.array([pred_s, pred_t, pred_d]) / pred_total
        phi_pred = np.clip(phi_pred, eps, 1.0 - eps)
        phi_pred = phi_pred / phi_pred.sum()

        alpha = kappa_surv * phi_obs
        total_nll -= float(np.sum((alpha - 1.0) * np.log(phi_pred)))

    return float(total_nll)


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
            nS_eff = p.nS if p.nS is not None else p.n
            KS_eff = p.KS if p.KS is not None else p.K
            nT_eff = p.nT if p.nT is not None else p.n
            KT_eff = p.KT if p.KT is not None else p.K
            hs = hS(C, p.kT, p.kS_kT_ratio, nS_eff, KS_eff)
            ht = hT(C, p.kT, nT_eff, KT_eff)

            if hs <= 1e-12:
                continue
            ratio = ht / max(hs, 1e-12)
            total += np.log1p(np.exp(k_soft * (ratio - rho))) / k_soft

    denom = nC * max(len(ages), 1)
    return float(total / max(denom, 1))


def _hill_lambda_constraint_penalty(
    p: STDPParams,
    ages: Tuple[float, ...],
    weight: float = 1e6,
) -> float:
    """
    Enforce effective_lag_rate(age) <= hT_inf under Hill hT kinetics.

    For hT(C) = kT * C^n / (K^n + C^n), hT_inf = kT.
    Uses :func:`effective_lag_rate` (accounts for the Erlang/Exponential
    mixture weight ``w_lag``) instead of the bare Erlang rate.
    """
    if weight <= 0:
        return 0.0

    hill_flags = get_hill_hazard_flags()
    if not hill_flags["hT_effective"]:
        return 0.0

    ht_inf = float(p.kT)
    max_violation = 0.0

    for age in ages:
        lam = effective_lag_rate(p, float(age))
        max_violation = max(max_violation, lam - ht_inf)

    if max_violation <= 0.0:
        return 0.0

    return float(weight * (max_violation**2))


def _regime_dominance_penalty(
    p: STDPParams,
    ages: Tuple[float, ...],
    min_kT_over_lam: float = 2.0,
    k_soft: float = 20.0,
    weight: float = 1e4,
    **_kwargs,
) -> float:
    """
    Penalise kT / λ_eff(a) < min_kT_over_lam.

    At saturating C, hT → kT. The Regime IV boundary is hT ≥ 2λ + g,
    above which permanent T-over-D dominance is impossible (SI §S6.3).
    With min_kT_over_lam = 2.0 (default), this targets the Regime IV
    boundary (ignoring g, which is conservative).

    Uses :func:`effective_lag_rate` (accounts for the Erlang/Exponential
    mixture weight ``w_lag``) instead of the bare Erlang rate.

    Uses softplus(x) = log(1 + exp(k_soft * x)) / k_soft for smooth
    L-BFGS-B gradients.
    """
    if weight <= 0:
        return 0.0

    hill_flags = get_hill_hazard_flags()
    if not hill_flags["hT_effective"]:
        return 0.0

    ht_inf = float(p.kT)  # Hill saturation limit

    total = 0.0
    for a in ages:
        lam = effective_lag_rate(p, float(a))
        violation = min_kT_over_lam - (ht_inf / max(lam, 1e-9))
        total += np.log1p(np.exp(k_soft * violation)) / k_soft

    return float(weight * total / max(len(ages), 1))


def _param_regularization_penalty(
    p: STDPParams,
    n_max: float = 10.0,
    nST_max: float = 10.0,
    r0_max: float = 0.5,
    ratio_min: float = 1.5,
    k_soft: float = 5.0,
) -> float:
    """
    Penalise biologically implausible parameter values.

    Uses softplus(x) = log(1 + exp(k*x)) / k for smooth L-BFGS-B gradients.
    """
    total = 0.0
    # Extreme Hill exponents → near-step-function dose-response
    total += np.log1p(np.exp(k_soft * (p.n - n_max))) / k_soft
    total += np.log1p(np.exp(k_soft * (p.nST - nST_max))) / k_soft
    # Baseline switching should be small (spontaneous, no drug stress)
    total += np.log1p(np.exp(k_soft * (p.r0 - r0_max))) / k_soft
    # S death rate should exceed T death rate meaningfully
    total += np.log1p(np.exp(k_soft * (ratio_min - p.kS_kT_ratio))) / k_soft
    return float(total)


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
    lam_hill_constraint: float = 1e6,
    lam_regime: float = 0.0,
    lam_reg: float = 1e3,
    kappa_surv: float = 500.0,
    lam_surv: float = 1.0,
):
    free_keys = DEFAULT_FREE_KEYS if free_keys is None else list(free_keys)
    P_obs = _observed_matrix(data)
    ages_for_constraint = tuple(sorted({float(row[3]) for row in data}))
    if not ages_for_constraint:
        ages_for_constraint = ages_for_pen

    def objective(x: np.ndarray, template: STDPParams) -> float:
        cand = _unpack_params(x, template, free_keys)
        model = STDPModel(cand)
        P_pred = _pred_matrix(model, data)
        nll = _nll_dirichlet(P_obs, P_pred, kappa=kappa, class_weights=class_weights)
        surv_nll = _survivor_composition_nll(model, data, kappa_surv=kappa_surv)
        pen = _tolerance_penalty(cand, rho=rho, ages=ages_for_pen)
        hill_pen = _hill_lambda_constraint_penalty(
            cand,
            ages=ages_for_constraint,
            weight=lam_hill_constraint,
        )
        regime_pen = _regime_dominance_penalty(
            cand,
            ages=ages_for_constraint,
            weight=lam_regime,
        )
        reg_pen = _param_regularization_penalty(cand)
        return float(
            nll + lam_surv * surv_nll + lam_pen * pen
            + hill_pen + regime_pen + lam_reg * reg_pen
        )

    return objective, free_keys
