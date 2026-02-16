from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict

import numpy as np


USE_LOG_HAZARDS = True


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
    """

    k_lag: int = 3
    mu0: float = 0.23
    mu24p: float = 0.27

    kT: float = 0.5
    kS_kT_ratio: float = 5.0
    kST: float = 1.35

    n: float = 2.0
    nST: float = 1.5

    a50: float = 20.0
    r0: float = 0.01


def m(a: float, a50: float) -> float:
    """Stress-memory function of age."""
    return float(a) / (float(a) + float(a50))


def hS(C: float, kT: float, kS_kT_ratio: float, nS: float) -> float:
    """Susceptible-state death hazard."""
    C = float(C)
    if C <= 0.0:
        return 0.0

    base_hazard = float(kT * kS_kT_ratio * (C**nS))
    if USE_LOG_HAZARDS:
        return float(np.log1p(base_hazard))
    return base_hazard


def hT(C: float, kT: float, nT: float) -> float:
    """Tolerant-state death hazard."""
    C = float(C)
    if C <= 0.0:
        return 0.0

    base_hazard = float(kT * (C**nT))
    if USE_LOG_HAZARDS:
        return float(np.log1p(base_hazard))
    return base_hazard


def rST(C: float, a: float, kST: float, nST: float, a50: float, r0: float) -> float:
    """S -> T switching hazard."""
    C = float(C)
    baseline = float(r0)
    if C <= 0.0:
        hazard_C = 0.0
    else:
        hazard_C = float(kST * (C**nST))
        if USE_LOG_HAZARDS:
            hazard_C = float(np.log1p(hazard_C))
    return float(m(a, a50) * (baseline + hazard_C))


def _I_polyexp(b: float, tau: float, k: int) -> float:
    """
    I_k(tau; b) = ∫_0^tau t^{k-1} e^{b t} dt, integer k ≥ 1.
    """
    t = float(tau)
    if abs(b * t) < 1e-6:
        s = t**k / k
        term = s
        mm = 1
        while True:
            term *= (b * t) / (k + mm)
            term /= mm
            s += term
            if abs(term) < 1e-15 * max(abs(s), 1.0) or mm > 200:
                break
            mm += 1
        return float(s)

    I = (np.exp(b * t) - 1.0) / b
    for kk in range(2, k + 1):
        I = np.exp(b * t) * t ** (kk - 1) / b - (kk - 1) / b * I
    return float(I)


def _J_trunc(k: int, lam: float, alpha: float, tau: float) -> float:
    """J(alpha) = ∫_0^tau e^{alpha l} f_Erlang(l; k, lam) dl."""
    b = alpha - lam
    I = _I_polyexp(b, tau, k)
    return float((lam**k) * I / factorial(k - 1))


def _J_and_dJ_trunc(k: int, lam: float, alpha: float, tau: float) -> tuple[float, float]:
    """Return J(alpha) and dJ/dalpha."""
    b = alpha - lam
    Ik = _I_polyexp(b, tau, k)
    Ik1 = _I_polyexp(b, tau, k + 1)
    J = (lam**k) * Ik / factorial(k - 1)
    dJ = (lam**k) * Ik1 / factorial(k - 1)
    return float(J), float(dJ)


def _erlang_sf(k: int, lam: float, tau: float) -> float:
    """Survival S(L>tau) for Erlang(k, lam), closed form."""
    x = lam * tau
    s = 0.0
    term = 1.0
    for mm in range(k):
        if mm > 0:
            term *= x / mm
        s += term
    return float(np.exp(-x) * s)


class STDPModel:
    """Analytical S/T/D model."""

    def __init__(self, params: STDPParams):
        self.p = params

    def mean_lag(self, a: float) -> float:
        mu = self.p.mu0 + (float(a) / 24.0) * self.p.mu24p
        return float(max(mu, 1e-9))

    def lam_from_mean(self, a: float) -> float:
        return float(self.p.k_lag) / self.mean_lag(a)

    def pi_preexisting(self, a: float, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        lam = self.lam_from_mean(a)
        return float(np.clip(_erlang_sf(self.p.k_lag, lam, tau), 0.0, 1.0))

    def pi_incomplete(self, C: float, tau: float, a: float) -> float:
        h_s = hS(C, self.p.kT, self.p.kS_kT_ratio, self.p.n)
        r = rST(C, a, self.p.kST, self.p.nST, self.p.a50, self.p.r0)
        H = h_s + r
        lam = self.lam_from_mean(a)
        JH = _J_trunc(self.p.k_lag, lam, H, tau)
        return float(np.clip(np.exp(-H * tau) * JH, 0.0, 1.0))

    def pi_induced(self, C: float, tau: float, a: float) -> float:
        r = rST(C, a, self.p.kST, self.p.nST, self.p.a50, self.p.r0)
        ht = hT(C, self.p.kT, self.p.n)
        h_s = hS(C, self.p.kT, self.p.kS_kT_ratio, self.p.n)

        H = h_s + r
        lam = self.lam_from_mean(a)

        if abs(H - ht) > 1e-10:
            Jh = _J_trunc(self.p.k_lag, lam, ht, tau)
            JH = _J_trunc(self.p.k_lag, lam, H, tau)
            val = (r / (H - ht)) * (np.exp(-ht * tau) * Jh - np.exp(-H * tau) * JH)
        else:
            Jh, dJh = _J_and_dJ_trunc(self.p.k_lag, lam, ht, tau)
            val = r * np.exp(-ht * tau) * (tau * Jh - dJh)

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
