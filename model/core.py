from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict

import numpy as np


USE_LOG_HAZARDS = False
USE_LOG_HAZARDS_HS: bool | None = None
USE_LOG_HAZARDS_HT: bool | None = None
USE_LOG_HAZARDS_RST: bool | None = None
USE_HILL_HAZARD = False
USE_HILL_HAZARD_HS: bool | None = None
USE_HILL_HAZARD_HT: bool | None = None
USE_HILL_HAZARD_RST: bool | None = None
_UNCHANGED = object()


def _resolve_log_hazard_flag(override: bool | None) -> bool:
    if override is None:
        return bool(USE_LOG_HAZARDS)
    return bool(override)


def _resolve_hill_hazard_flag(override: bool | None) -> bool:
    if override is None:
        return bool(USE_HILL_HAZARD)
    return bool(override)


def set_log_hazard_flags(
    *,
    global_default=_UNCHANGED,
    hS=_UNCHANGED,
    hT=_UNCHANGED,
    rST=_UNCHANGED,
) -> dict[str, bool | None]:
    """
    Set hazard log-transform flags.

    `USE_LOG_HAZARDS` remains the global fallback. Per-hazard overrides
    (`hS`, `hT`, `rST`) take precedence when set. Pass `None` for a per-hazard
    override to clear it and fall back to `global_default`.
    """
    global USE_LOG_HAZARDS
    global USE_LOG_HAZARDS_HS, USE_LOG_HAZARDS_HT, USE_LOG_HAZARDS_RST

    if global_default is not _UNCHANGED:
        if global_default is None:
            raise ValueError("global_default must be True or False when provided.")
        USE_LOG_HAZARDS = bool(global_default)
    if hS is not _UNCHANGED:
        USE_LOG_HAZARDS_HS = None if hS is None else bool(hS)
    if hT is not _UNCHANGED:
        USE_LOG_HAZARDS_HT = None if hT is None else bool(hT)
    if rST is not _UNCHANGED:
        USE_LOG_HAZARDS_RST = None if rST is None else bool(rST)

    return get_log_hazard_flags()


def get_log_hazard_flags() -> dict[str, bool | None]:
    return {
        "global_default": bool(USE_LOG_HAZARDS),
        "hS_override": USE_LOG_HAZARDS_HS,
        "hT_override": USE_LOG_HAZARDS_HT,
        "rST_override": USE_LOG_HAZARDS_RST,
        "hS_effective": _resolve_log_hazard_flag(USE_LOG_HAZARDS_HS),
        "hT_effective": _resolve_log_hazard_flag(USE_LOG_HAZARDS_HT),
        "rST_effective": _resolve_log_hazard_flag(USE_LOG_HAZARDS_RST),
    }


def set_hill_hazard_flags(
    *,
    global_default=_UNCHANGED,
    hS=_UNCHANGED,
    hT=_UNCHANGED,
    rST=_UNCHANGED,
) -> dict[str, bool | None]:
    """
    Set Hill-form hazard flags.

    `USE_HILL_HAZARD` is the global fallback. Per-hazard overrides
    (`hS`, `hT`, `rST`) take precedence when set. Pass `None` for a
    per-hazard override to clear it and fall back to `global_default`.
    """
    global USE_HILL_HAZARD
    global USE_HILL_HAZARD_HS, USE_HILL_HAZARD_HT, USE_HILL_HAZARD_RST

    if global_default is not _UNCHANGED:
        if global_default is None:
            raise ValueError("global_default must be True or False when provided.")
        USE_HILL_HAZARD = bool(global_default)
    if hS is not _UNCHANGED:
        USE_HILL_HAZARD_HS = None if hS is None else bool(hS)
    if hT is not _UNCHANGED:
        USE_HILL_HAZARD_HT = None if hT is None else bool(hT)
    if rST is not _UNCHANGED:
        USE_HILL_HAZARD_RST = None if rST is None else bool(rST)

    return get_hill_hazard_flags()


def get_hill_hazard_flags() -> dict[str, bool | None]:
    return {
        "global_default": bool(USE_HILL_HAZARD),
        "hS_override": USE_HILL_HAZARD_HS,
        "hT_override": USE_HILL_HAZARD_HT,
        "rST_override": USE_HILL_HAZARD_RST,
        "hS_effective": _resolve_hill_hazard_flag(USE_HILL_HAZARD_HS),
        "hT_effective": _resolve_hill_hazard_flag(USE_HILL_HAZARD_HT),
        "rST_effective": _resolve_hill_hazard_flag(USE_HILL_HAZARD_RST),
    }


@dataclass(frozen=True)
class ConditionPrediction:
    incomplete: float
    induced: float
    preexisting: float
    dead: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "incomplete": float(self.incomplete),
            "induced": float(self.induced),
            "preexisting": float(self.preexisting),
            "dead": float(self.dead),
        }


@dataclass
class STDPParams:
    """
    Parameters for the analytical S/T/D model.

    - hS and hT share exponent n.
    - rST uses a distinct exponent nST.
    - Lag distribution is a hybrid Erlang + Exponential mixture:
        f_L(t) = w_lag * Erlang(k_lag, lam) + (1-w_lag) * Exp(lam_slow)
      where lam = k_lag / mu_lag(a).
    """

    w_lag: float = 0.95
    lam_slow: float = 0.1
    k_lag: int = 6
    mu0: float = 0.23
    mu24p: float = 0.27

    kT: float = 0.5
    kS_kT_ratio: float = 5.0
    kST: float = 1.35
    K: float = 1.0
    KST: float = 1.0

    n: float = 2.0
    nST: float = 1.5

    a50: float = 20.0
    r0: float = 0.01

    # Age-dependent slow-component mean lag (None → use fixed lam_slow)
    mu_s0: float | None = None    # baseline slow-component mean lag (h)
    mu_s24p: float | None = None  # age slope for slow component (h per 24h)

    # Independent Hill exponents/K (None → fall back to shared n / K)
    nS: float | None = None
    nT: float | None = None
    KS: float | None = None
    KT: float | None = None


def m(a: float, a50: float) -> float:
    """Stress-memory function of age."""
    return float(a) / (float(a) + float(a50))


def _safe_pow(base: float, exp: float) -> float:
    """Compute base**exp, returning inf on overflow instead of raising."""
    try:
        return float(base ** exp)
    except OverflowError:
        return float("inf")


def _erlang_sf(k: int, lam: float, tau: float) -> float:
    """
    Erlang survival function: P(L > tau) = exp(-lam*tau) * sum_{m=0}^{k-1} (lam*tau)^m / m!
    """
    if tau <= 0.0:
        return 1.0
    x = lam * tau
    ex = float(np.exp(-x))
    if ex == 0.0:
        return 0.0
    s = 0.0
    term = 1.0  # (lam*tau)^0 / 0!
    for mm in range(k):
        s += term
        term *= x / (mm + 1)
    return float(ex * s)


def _phi_erlang(k: int, lam: float, alpha: float, tau: float) -> float:
    """
    Erlang phi integral:
      phi(alpha, tau) = integral_0^tau exp(-alpha*(tau-t)) * f_Erlang(t; k, lam) dt

    Uses the J_m recurrence for numerical stability:
      J_1 = (exp(-lam*tau) - exp(-alpha*tau)) / (alpha - lam)
      J_m = (tau^{m-1} * exp(-lam*tau) - (m-1) * J_{m-1}) / (alpha - lam)
      phi = lam^k / (k-1)! * J_k
    """
    if tau <= 0.0:
        return 0.0

    diff = alpha - lam
    lam_k = lam ** k
    coeff = lam_k / factorial(k - 1)

    # When alpha ≈ lam, use Taylor expansion
    if abs(diff) * tau < 1e-8:
        # phi_Erlang ≈ lam * tau^k * exp(-lam*tau) / (k-1)! * [1 - diff*tau/(k+1) + ...]
        # which is lam * (lam*tau)^{k-1} * tau * exp(-lam*tau) / (k-1)!
        x = diff * tau
        series = 1.0 - x * tau / (2.0 * k)
        return float(coeff * tau ** (k - 1) * np.exp(-lam * tau) * tau / k * series)

    # J_1 recurrence base
    e_lam = float(np.exp(-lam * tau))
    e_alpha = float(np.exp(-alpha * tau))
    J = (e_lam - e_alpha) / diff

    # J_m for m = 2..k
    for mm in range(2, k + 1):
        J = (tau ** (mm - 1) * e_lam - (mm - 1) * J) / diff

    return float(coeff * J)


def _hybrid_sf(w: float, k: int, lam: float, lam_s: float, tau: float) -> float:
    """Survival function for hybrid Erlang + Exponential mixture."""
    if tau <= 0.0:
        return 1.0
    return float(
        w * _erlang_sf(k, lam, tau)
        + (1.0 - w) * np.exp(-lam_s * tau)
    )


def _phi_hybrid(
    w: float, k: int, lam: float, lam_s: float, alpha: float, tau: float
) -> float:
    """
    phi(alpha, tau) for hybrid Erlang + Exponential mixture:
      w * phi_Erlang(k, lam, alpha, tau) + (1-w) * phi_component(lam_s, alpha, tau)
    """
    return float(
        w * _phi_erlang(k, lam, alpha, tau)
        + (1.0 - w) * _phi_component(lam_s, alpha, tau)
    )


def hS(C: float, kT: float, kS_kT_ratio: float, nS: float, K: float = 1.0) -> float:
    """Susceptible-state death hazard."""
    C = float(C)
    if C <= 0.0:
        return 0.0

    if _resolve_hill_hazard_flag(USE_HILL_HAZARD_HS):
        c_pow = _safe_pow(C, nS)
        denom = _safe_pow(float(K), nS) + c_pow
        base_hazard = float(kT * kS_kT_ratio * c_pow / max(denom, 1e-300))
    else:
        base_hazard = float(kT * kS_kT_ratio * _safe_pow(C, nS))

    if _resolve_log_hazard_flag(USE_LOG_HAZARDS_HS):
        return float(np.log1p(base_hazard))
    return base_hazard


def hT(C: float, kT: float, nT: float, K: float = 1.0) -> float:
    """Tolerant-state death hazard."""
    C = float(C)
    if C <= 0.0:
        return 0.0

    if _resolve_hill_hazard_flag(USE_HILL_HAZARD_HT):
        c_pow = _safe_pow(C, nT)
        denom = _safe_pow(float(K), nT) + c_pow
        base_hazard = float(kT * c_pow / max(denom, 1e-300))
    else:
        base_hazard = float(kT * _safe_pow(C, nT))

    if _resolve_log_hazard_flag(USE_LOG_HAZARDS_HT):
        return float(np.log1p(base_hazard))
    return base_hazard


def rST(
    C: float,
    a: float,
    kST: float,
    nST: float,
    a50: float,
    r0: float,
    KST: float = 1.0,
) -> float:
    """S -> T switching hazard."""
    C = float(C)
    baseline = float(r0)
    if C <= 0.0:
        hazard_C = 0.0
    else:
        if _resolve_hill_hazard_flag(USE_HILL_HAZARD_RST):
            c_num = _safe_pow(C, nST)
            denom = _safe_pow(float(KST), nST) + c_num
            hazard_C = float(kST * c_num / max(denom, 1e-300))
        else:
            hazard_C = float(kST * _safe_pow(C, nST))
        if _resolve_log_hazard_flag(USE_LOG_HAZARDS_RST):
            hazard_C = float(np.log1p(hazard_C))
    return float(m(a, a50) * (baseline + hazard_C))


def _phi_component(lam_i: float, alpha: float, tau: float) -> float:
    """
    Single-exponential contribution to phi(alpha, tau):
      phi_i = lam_i * [exp(-lam_i*tau) - exp(-alpha*tau)] / (alpha - lam_i)

    Uses Taylor expansion when |alpha - lam_i|*tau is small.
    """
    if tau <= 0.0:
        return 0.0

    diff = alpha - lam_i
    if abs(diff) * tau < 1e-8:
        x = diff * tau
        series = 1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0
        return float(lam_i * tau * np.exp(-lam_i * tau) * series)

    return float(lam_i * (np.exp(-lam_i * tau) - np.exp(-alpha * tau)) / diff)


def _biexp_sf(w: float, lam1: float, lam2: float, tau: float) -> float:
    """Survival S(L>tau) for mixture of two exponentials."""
    if tau <= 0.0:
        return 1.0
    return float(w * np.exp(-lam1 * tau) + (1.0 - w) * np.exp(-lam2 * tau))


def _phi_biexp(w: float, lam1: float, lam2: float, alpha: float, tau: float) -> float:
    """
    phi(alpha, tau) = integral_0^tau exp(-alpha*(tau-ell)) f_biexp(ell) d_ell
    = w * phi_component(lam1, alpha, tau) + (1-w) * phi_component(lam2, alpha, tau)
    """
    return float(
        w * _phi_component(lam1, alpha, tau)
        + (1.0 - w) * _phi_component(lam2, alpha, tau)
    )


class STDPModel:
    """Analytical S/T/D model with hybrid Erlang + Exponential lag."""

    def __init__(self, params: STDPParams):
        self.p = params

    def mean_lag(self, a: float) -> float:
        mu = self.p.mu0 + (float(a) / 24.0) * self.p.mu24p
        return float(max(mu, 1e-9))

    def erlang_rate(self, a: float) -> float:
        """Return the Erlang rate lambda = k_lag / mu_lag(a)."""
        mu = self.mean_lag(a)
        return float(self.p.k_lag / mu)

    def slow_rate(self, a: float) -> float:
        """Return the exponential-component rate λ_s, optionally age-dependent.

        When ``mu_s0`` is set, ``λ_s(a) = 1 / (mu_s0 + (a/24)·mu_s24p)``.
        Otherwise falls back to the fixed ``lam_slow`` parameter.
        """
        if self.p.mu_s0 is not None:
            mu_s24p = self.p.mu_s24p if self.p.mu_s24p is not None else 0.0
            mu = self.p.mu_s0 + (float(a) / 24.0) * mu_s24p
            return float(1.0 / max(mu, 1e-9))
        return float(self.p.lam_slow)

    def lam_from_mean(self, a: float) -> float:
        """Return the Erlang rate. Kept for constraint compatibility."""
        return self.erlang_rate(a)

    def pi_preexisting(self, a: float, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        lam = self.erlang_rate(a)
        return float(np.clip(
            _hybrid_sf(self.p.w_lag, self.p.k_lag, lam, self.slow_rate(a), tau),
            0.0, 1.0,
        ))

    def pi_incomplete(self, C: float, tau: float, a: float) -> float:
        nS_eff = self.p.nS if self.p.nS is not None else self.p.n
        KS_eff = self.p.KS if self.p.KS is not None else self.p.K
        h_s = hS(C, self.p.kT, self.p.kS_kT_ratio, nS_eff, KS_eff)
        r = rST(C, a, self.p.kST, self.p.nST, self.p.a50, self.p.r0, self.p.KST)
        H = h_s + r
        lam = self.erlang_rate(a)
        val = _phi_hybrid(self.p.w_lag, self.p.k_lag, lam, self.slow_rate(a), H, tau)
        return float(np.clip(val, 0.0, 1.0))

    def pi_induced(self, C: float, tau: float, a: float) -> float:
        nS_eff = self.p.nS if self.p.nS is not None else self.p.n
        KS_eff = self.p.KS if self.p.KS is not None else self.p.K
        nT_eff = self.p.nT if self.p.nT is not None else self.p.n
        KT_eff = self.p.KT if self.p.KT is not None else self.p.K
        r = rST(C, a, self.p.kST, self.p.nST, self.p.a50, self.p.r0, self.p.KST)
        ht = hT(C, self.p.kT, nT_eff, KT_eff)
        h_s = hS(C, self.p.kT, self.p.kS_kT_ratio, nS_eff, KS_eff)

        H = h_s + r
        w = self.p.w_lag
        k = self.p.k_lag
        lam = self.erlang_rate(a)
        lam_s = self.slow_rate(a)

        phi_ht = _phi_hybrid(w, k, lam, lam_s, ht, tau)
        phi_H = _phi_hybrid(w, k, lam, lam_s, H, tau)

        if abs(H - ht) > 1e-10 * max(1.0, abs(H), abs(ht)):
            val = (r / (H - ht)) * (phi_ht - phi_H)
        else:
            eps = 1e-6 * max(1.0, abs(ht))
            phi_ht_plus = _phi_hybrid(w, k, lam, lam_s, ht + eps, tau)
            d_phi = (phi_ht_plus - phi_ht) / eps
            val = -r * d_phi

        return float(np.clip(val, 0.0, 1.0))

    def predict_condition(self, C: float, tau: float, a: float) -> ConditionPrediction:
        pi_pre = self.pi_preexisting(a, tau)
        pi_inc = self.pi_incomplete(C, tau, a)
        pi_ind = self.pi_induced(C, tau, a)
        pi_dead = max(0.0, 1.0 - pi_pre - pi_inc - pi_ind)

        return ConditionPrediction(
            incomplete=float(pi_inc),
            induced=float(pi_ind),
            preexisting=float(pi_pre),
            dead=float(pi_dead),
        )
