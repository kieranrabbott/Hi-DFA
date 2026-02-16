# STDP Model Writeup (Implementation-Matched)

This document describes the mathematics implemented in the current Python model in `/Users/georgeos/GitHub/Hi-DFA/model`.

## 1) Model states and observables

We model three survivor-linked states during antibiotic exposure:

- `D`: dormant (lag-phase) cells
- `S`: resuscitated susceptible cells
- `T`: resuscitated tolerant cells

For each condition `(C, \tau, a)` (concentration, treatment duration, culture age), the model predicts:

- incomplete-treatment survivor fraction: `\pi_S`
- antibiotic-induced tolerant survivor fraction: `\pi_T`
- pre-existing persister (still dormant) fraction: `\pi_D`
- dead fraction: `\pi_{dead} = 1 - \pi_S - \pi_T - \pi_D`

The data pipeline uses incomplete/induced/preexisting and infers `dead` as the complement.

## 2) Lag-time model

### 2.1 Age-dependent mean lag

The mean lag is

$$
\mu_{lag}(a) = \mu_0 + \mu_{24p}\left(\frac{a}{24}\right),
$$

with implementation safeguard `\mu_{lag}(a) \ge 10^{-9}`.

### 2.2 Erlang lag distribution

Lag time `L` is Erlang with integer shape `k_{lag}` and rate

$$
\lambda(a) = \frac{k_{lag}}{\mu_{lag}(a)}.
$$

So

$$
f_L(\ell\mid a)=\frac{\lambda(a)^{k_{lag}}\,\ell^{k_{lag}-1}e^{-\lambda(a)\ell}}{(k_{lag}-1)!},
$$

and the survival function is

$$
S_L(\tau\mid a)=\Pr[L>\tau]=e^{-\lambda(a)\tau}\sum_{m=0}^{k_{lag}-1}\frac{(\lambda(a)\tau)^m}{m!}.
$$

The pre-existing persister fraction is exactly

$$
\pi_D(\tau,a)=S_L(\tau\mid a).
$$

## 3) Antibiotic-phase hazards

The implementation supports per-hazard switches for:

- linear power-law form vs Hill form
- raw hazard vs `\log(1+x)` transformed hazard

Both are configurable globally and per hazard (`h_S`, `h_T`, `r_{S\to T}`).

### 3.1 Stress-memory factor

$$
m(a)=\frac{a}{a+a_{50}}.
$$

### 3.2 Susceptible death hazard `h_S(C)`

Base (non-Hill):

$$
g_S(C)=k_T\,k_{S/T}\,C^n.
$$

Hill:

$$
g_S(C)=k_T\,k_{S/T}\,\frac{C^n}{K^n+C^n}.
$$

Final applied hazard:

$$
h_S(C)=\begin{cases}
\log(1+g_S(C)), & \text{if log flag for }h_S\text{ is ON}\\
g_S(C), & \text{otherwise}
\end{cases}
$$

with `h_S(C)=0` for `C\le 0`.

### 3.3 Tolerant death hazard `h_T(C)`

Base (non-Hill):

$$
g_T(C)=k_T\,C^n.
$$

Hill:

$$
g_T(C)=k_T\,\frac{C^n}{K^n+C^n}.
$$

Final applied hazard:

$$
h_T(C)=\begin{cases}
\log(1+g_T(C)), & \text{if log flag for }h_T\text{ is ON}\\
g_T(C), & \text{otherwise}
\end{cases}
$$

with `h_T(C)=0` for `C\le 0`.

### 3.4 Switching hazard `r_{S\to T}(C,a)`

Concentration-dependent switching term:

- non-Hill: $g_{ST}(C)=k_{ST}C^{n_{ST}}$
- Hill: $g_{ST}(C)=k_{ST}\dfrac{C^{n_{ST}}}{K_{ST}^{n_{ST}}+C^{n_{ST}}}$

Then the implemented switching hazard is

$$
r_{S\to T}(C,a)=m(a)\left[r_0+\tilde g_{ST}(C)\right],
$$

where

$$
\tilde g_{ST}(C)=\begin{cases}
\log(1+g_{ST}(C)), & \text{if log flag for }r_{S\to T}\text{ is ON}\\
g_{ST}(C), & \text{otherwise.}
\end{cases}
$$

For `C\le 0`, the implementation sets `g_{ST}(C)=0`, hence `r_{S\to T}(C,a)=m(a)r_0`.

## 4) Antibiotic-phase closed-form survivor fractions

Define

$$
H(C,a)=h_S(C)+r_{S\to T}(C,a).
$$

Also define

$$
J(\alpha)=\int_0^\tau e^{\alpha \ell}f_L(\ell\mid a)\,d\ell.
$$

Then:

### 4.1 Incomplete-treatment survivors

$$
\pi_S(C,\tau,a)=e^{-H\tau}J(H).
$$

### 4.2 Induced-tolerant survivors

For $H\neq h_T$,

$$
\pi_T(C,\tau,a)=\frac{r_{S\to T}}{H-h_T}\left[e^{-h_T\tau}J(h_T)-e^{-H\tau}J(H)\right].
$$

For $H\approx h_T$, the limit form is used; numerically the implementation uses a stable finite-difference evaluation around this branch.

### 4.3 Dead fraction

$$
\pi_{dead}=\max\{0,1-\pi_D-\pi_S-\pi_T\}.
$$

## 5) Post-treatment descendant composition (used in plots)

At post-treatment time `u` with growth rate `g`, the descendant decomposition uses

- `\pi_S`, `\pi_T` from treatment-end survivor fractions
- a discounted dormant contribution

$$
J_{disc}(u,\tau;k,\lambda,g)
=\left(\frac{\lambda}{\lambda+g}\right)^k e^{g\tau}
\left[P\big(k,(\lambda+g)(\tau+u)\big)-P\big(k,(\lambda+g)\tau\big)\right],
$$

where `P(k,\cdot)` is the regularized lower incomplete gamma CDF.

Then

$$
\text{denom}=\pi_S+\pi_T+J_{disc},
$$

$$
w_S=\frac{\pi_S}{\text{denom}},\quad
w_T=\frac{\pi_T}{\text{denom}},\quad
w_D=\frac{J_{disc}}{\text{denom}},
$$

and relative regrown mass is

$$
R(u)=e^{gu}\,\text{denom}.
$$

## 6) Fitting objective

Let observed class composition per condition be `\mathbf y_i=(y_{i1},\dots,y_{i4})` and prediction be `\mathbf p_i`.

### 6.1 Dirichlet pseudo-likelihood term

With concentration parameter `\kappa` and optional class weights `w_j>0`, the minimized negative pseudo-log-likelihood is

$$
\mathcal L_{NLL}
=-\sum_i\sum_{j=1}^4 w_j\,(\kappa y_{ij}-1)\log p_{ij},
$$

with clipping/renormalization for numerical stability.

### 6.2 Tolerance-ratio soft penalty

The penalty used is an average softplus on

$$
\frac{h_T(C)}{h_S(C)}-\rho
$$

over a concentration grid and representative ages:

$$
\mathcal P_{tol}
=\frac{1}{N_C N_A}\sum_{a\in A}\sum_{C\in\mathcal C}
\frac{1}{k_{soft}}\log\left(1+e^{k_{soft}(h_T/h_S-\rho)}\right).
$$

### 6.3 Hill-only lag-vs-kill constraint

Under active Hill `h_T`, the implementation adds

$$
\mathcal P_{\lambda}
=\lambda_{hill}\left[\max_{a\in A_{data}}\left(\lambda(a)-h_{T,\infty}\right)_+\right]^2,
$$

with

$$
h_{T,\infty}=k_T
$$

for Hill `h_T(C)=k_T\,C^n/(K^n+C^n)`.

If Hill `h_T` is not active, `\mathcal P_{\lambda}=0`.

### 6.4 Total objective

$$
\mathcal J(\theta)=\mathcal L_{NLL}+\lambda_{pen}\mathcal P_{tol}+\mathcal P_{\lambda}.
$$

## 7) Optimization parameterization and bounds

Optimization is performed in log-parameter space (`\theta=\log` parameters), with box bounds.

Current bound set includes:

- lag parameters: `mu0`, `mu24p`
- hazards/switching: `kT`, `kS_kT_ratio`, `kST`
- Hill parameters: `K`, `KST`
- exponents: `n`, `nST`
- memory/baseline: `a50`, `r0`

Multi-start `L-BFGS-B` is used; the best objective value is retained.

## 8) Numerical stability notes

The Erlang-convolution integrals are evaluated with scaled forms (not naive `e^{bt}` expansions) in the main prediction path to avoid overflow/non-finite Hessians during uncertainty analyses.
