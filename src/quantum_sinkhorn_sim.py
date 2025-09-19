#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quantum_sinkhorn_sim.py
Hybrid quantum–classical Sinkhorn (simulated):
- QAE-like marginal estimation via multinomial sampling (shots ~ shots_factor / ae_eps),
- Log-domain updates to avoid overflow/underflow,
- Per-iteration gauge-fixing,
- Optional warm-start using a few classical Sinkhorn sweeps,
- Optional damping of log-updates and periodic exact refresh,
- Hadamard-test-like noisy cost estimator for illustration.

This file is self-contained and exposes:
    - quantum_sinkhorn_sim(...)
    - hadamard_cost_sim(...)
    - compute_cost(...)
    - generate_cost_matrix_rect(...) and CLI for smoke tests
"""

import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Numerical stability constants
# ---------------------------------------------------------------------
EPS_MIN = 1e-300          # strictly positive lower bound
CAP_MAX = 1e300           # soft upper cap to avoid inf
LOG_CAP = 700.0           # clamp for exp(logx) to avoid overflow (exp(709) ~ 8.2e307)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def gibbs_kernel(C, tau):
    """K = exp(-tau * C) with underflow protection."""
    K = np.exp(-tau * C)
    return np.clip(K, EPS_MIN, None)

def _safe_prob_from_scores(scores):
    """
    Convert a vector of positive (unnormalized) scores into a valid probability.
    Cleans NaN/Inf, imposes [EPS_MIN, CAP_MAX], normalizes to sum to 1.
    Returns (p, Z) where Z is the sanitized sum of scores.
    """
    s = np.array(scores, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=CAP_MAX, neginf=0.0)
    s = np.clip(s, EPS_MIN, CAP_MAX)
    Z = float(np.sum(s))
    if not np.isfinite(Z) or Z <= 0:
        p = np.ones_like(s) / len(s)
        return p, float(len(s))
    p = s / Z
    p = np.clip(p, 0.0, 1.0)
    ssum = float(p.sum())
    if not np.isfinite(ssum) or ssum <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / ssum
    return p, Z

def qae_multinomial_estimate(prob_vec, shots, rng):
    """
    Multinomial frequency estimator ~ QAE (unbiased; var ~ p(1-p)/shots).
    Ensures prob_vec is valid.
    """
    shots = int(max(1, shots))
    p = np.array(prob_vec, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0 or not np.isfinite(s):
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    counts = rng.multinomial(shots, p)
    return counts / shots

def materialize_from_logs(logu, logv):
    """Return u=exp(logu), v=exp(logv) with clipping to avoid overflow."""
    u = np.exp(np.clip(logu, -LOG_CAP, LOG_CAP))
    v = np.exp(np.clip(logv, -LOG_CAP, LOG_CAP))
    # Extra cleanup
    u = np.nan_to_num(u, nan=1.0, posinf=CAP_MAX, neginf=EPS_MIN)
    v = np.nan_to_num(v, nan=1.0, posinf=CAP_MAX, neginf=EPS_MIN)
    u = np.clip(u, EPS_MIN, 1.0 / EPS_MIN)
    v = np.clip(v, EPS_MIN, 1.0 / EPS_MIN)
    return u, v

# ---------------------------------------------------------------------
# Simulated QAE marginal subroutines
# ---------------------------------------------------------------------
def simulate_row_marginals(K, logu, logv, rng, ae_eps=0.02, shots_factor=100, use_exact_norm=True):
    """
    Estimate M_i = u_i (K v)_i via QAE-like sampling:
      scores_i = u_i * (K v)_i
      p_row = scores / sum(scores)
      p_hat ~ Multinomial(shots, p_row) / shots
      M_hat = p_hat * Z  (Z exact or noisy)
    """
    u, v = materialize_from_logs(logu, logv)
    Kv = K @ v
    scores = u * Kv
    p_row, Z = _safe_prob_from_scores(scores)
    shots = max(int(np.ceil(shots_factor / max(ae_eps, 1e-6))), 1)
    p_hat = qae_multinomial_estimate(p_row, shots, rng)
    if use_exact_norm:
        M_hat = p_hat * Z
    else:
        Z_hat = Z * (1.0 + rng.normal(0.0, ae_eps))
        if not np.isfinite(Z_hat) or Z_hat <= 0:
            Z_hat = Z
        M_hat = p_hat * max(Z_hat, EPS_MIN)
    M_hat = np.nan_to_num(M_hat, nan=EPS_MIN, posinf=CAP_MAX, neginf=EPS_MIN)
    M_hat = np.clip(M_hat, EPS_MIN, CAP_MAX)
    return M_hat, Z

def simulate_col_marginals(K, logu, logv, rng, ae_eps=0.02, shots_factor=100, use_exact_norm=True):
    """Estimate N_j = v_j (K^T u)_j with the same scheme as rows."""
    u, v = materialize_from_logs(logu, logv)
    KTu = K.T @ u
    scores = v * KTu
    p_col, Z = _safe_prob_from_scores(scores)
    shots = max(int(np.ceil(shots_factor / max(ae_eps, 1e-6))), 1)
    p_hat = qae_multinomial_estimate(p_col, shots, rng)
    if use_exact_norm:
        N_hat = p_hat * Z
    else:
        Z_hat = Z * (1.0 + rng.normal(0.0, ae_eps))
        if not np.isfinite(Z_hat) or Z_hat <= 0:
            Z_hat = Z
        N_hat = p_hat * max(Z_hat, EPS_MIN)
    N_hat = np.nan_to_num(N_hat, nan=EPS_MIN, posinf=CAP_MAX, neginf=EPS_MIN)
    N_hat = np.clip(N_hat, EPS_MIN, CAP_MAX)
    return N_hat, Z

# ---------------------------------------------------------------------
# Cost estimation (Hadamard-test-like, simulated)
# ---------------------------------------------------------------------
def hadamard_cost_sim(C, u, v, K, rng, ae_eps=0.02):
    """Return (exact_cost, noisy_cost). Noise ~ N(0, ae_eps * mean(|C|))."""
    gamma = (u[:, None]) * K * (v[None, :])
    exact = float(np.sum(C * gamma))
    scale = max(1e-12, float(np.mean(np.abs(C))))
    noisy = exact + rng.normal(0.0, ae_eps * scale)
    return exact, noisy

# ---------------------------------------------------------------------
# Core algorithm (log-domain + gauge fixing + optional damping/refresh/warm-start)
# ---------------------------------------------------------------------
def quantum_sinkhorn_sim(
    C, mu, nu,
    tau=1.0,
    tol=1e-4,
    ae_eps=0.02,
    max_iter=2000,
    shots_factor=100,
    use_exact_norm=True,
    seed=42,
    return_history=False,
    # New controls:
    damping_eta=1.0,            # in (0,1]; fraction of each log-update to apply
    exact_check_period=0,       # if >0, every s iters compute exact marginals and apply a mild refresh
    warmstart_steps=0,          # number of classical Sinkhorn sweeps before the quantum loop
):
    """
    Simulated hybrid quantum–classical Sinkhorn with:
      - QAE-like marginal estimation (multinomial frequencies),
      - Log-domain updates, gauge-fixing per iteration,
      - Optional warm-start (few classical sweeps),
      - Optional damping and periodic exact refresh.

    Args:
        C (ndarray): cost matrix (n x m).
        mu (ndarray): source marginal (n,).
        nu (ndarray): target marginal (m,).
        tau (float): entropic regularization parameter.
        tol (float): stopping tolerance on L1 marginal error (based on noisy estimates).
        ae_eps (float): target additive precision for QAE simulation.
        max_iter (int): maximum quantum iterations.
        shots_factor (int): shots ~ shots_factor / ae_eps.
        use_exact_norm (bool): if True, reconstruct marginals with exact Z.
        seed (int): RNG seed.
        return_history (bool): if True, return (u, v, gamma, iters, info).
        damping_eta (float): in (0,1]; smaller => more damping => more stable but slower.
        exact_check_period (int): if >0, every s iters do exact refresh (stabilizes noisy updates).
        warmstart_steps (int): if >0, perform this many classical Sinkhorn sweeps before quantum loop.

    Returns:
        u, v, gamma, iters, info (if return_history=True; info contains err_hist, Z_hist, K)
    """
    rng = np.random.default_rng(seed)
    n, m = C.shape
    K = gibbs_kernel(C, tau)

    # log-domain init
    logu = np.zeros(n, dtype=float)
    logv = np.zeros(m, dtype=float)

    # Warm-start (optional): a few classical sweeps to start near a good region
    if warmstart_steps and warmstart_steps > 0:
        try:
            # local import to avoid hard dependency when used standalone
            from classical_sinkhorn import sinkhorn as _classical_sinkhorn
            u0, v0, _, _, _ = _classical_sinkhorn(
                C, mu, nu, tau=tau, eps=1e-2, max_iter=int(warmstart_steps), return_history=True
            )
            logu = np.log(np.clip(u0, EPS_MIN, None))
            logv = np.log(np.clip(v0, EPS_MIN, None))
        except Exception:
            # if classical module not available, silently skip warm-start
            pass

    err_hist = []
    z_hist = []

    log_mu = np.log(np.clip(mu, EPS_MIN, 1.0))
    log_nu = np.log(np.clip(nu, EPS_MIN, 1.0))

    eta = float(max(0.0, min(1.0, damping_eta)))  # clamp to [0,1]

    for t in range(1, max_iter + 1):
        # --- QAE-like row marginals ---
        M_hat, Z_row = simulate_row_marginals(
            K, logu, logv, rng,
            ae_eps=ae_eps,
            shots_factor=shots_factor,
            use_exact_norm=use_exact_norm
        )
        M_hat = np.clip(np.nan_to_num(M_hat, nan=EPS_MIN, posinf=CAP_MAX, neginf=EPS_MIN), EPS_MIN, CAP_MAX)

        # damped log-update: logu += eta * (log(mu) - log(M_hat))
        logu += eta * (log_mu - np.log(np.clip(M_hat, EPS_MIN, None)))

        # --- QAE-like column marginals ---
        N_hat, Z_col = simulate_col_marginals(
            K, logu, logv, rng,
            ae_eps=ae_eps,
            shots_factor=shots_factor,
            use_exact_norm=use_exact_norm
        )
        N_hat = np.clip(np.nan_to_num(N_hat, nan=EPS_MIN, posinf=CAP_MAX, neginf=EPS_MIN), EPS_MIN, CAP_MAX)

        # damped log-update: logv += eta * (log(nu) - log(N_hat))
        logv += eta * (log_nu - np.log(np.clip(N_hat, EPS_MIN, None)))

        # --- gauge-fixing in log-domain: center logu and compensate in logv ---
        delta = float(np.mean(logu))
        if np.isfinite(delta):
            logu -= delta
            logv += delta

        # --- noisy error for stopping ---
        err_rows = float(np.sum(np.abs(M_hat - mu)))
        err_cols = float(np.sum(np.abs(N_hat - nu)))
        err = max(err_rows, err_cols)
        err_hist.append(err)
        z_hist.append((Z_row, Z_col))

        # --- periodic exact refresh (optional) ---
        if exact_check_period and exact_check_period > 0 and (t % int(exact_check_period) == 0):
            u_tmp, v_tmp = materialize_from_logs(logu, logv)
            M_exact = (u_tmp[:, None] * K * v_tmp[None, :]).sum(axis=1)
            N_exact = (u_tmp[:, None] * K * v_tmp[None, :]).sum(axis=0)
            # mild refresh to re-center (0.5 is a robust default)
            logu += 0.5 * (log_mu - np.log(np.clip(M_exact, EPS_MIN, None)))
            logv += 0.5 * (log_nu - np.log(np.clip(N_exact, EPS_MIN, None)))
            # gauge-fixing again
            d2 = float(np.mean(logu))
            if np.isfinite(d2):
                logu -= d2
                logv += d2
            # optional early stop on exact error
            err_rows_exact = float(np.sum(np.abs(M_exact - mu)))
            err_cols_exact = float(np.sum(np.abs(N_exact - nu)))
            err_exact = max(err_rows_exact, err_cols_exact)
            if err_exact <= tol:
                err_hist.append(err_exact)  # record exact error too
                break

        if err <= tol:
            break

    # Materialize u, v and build gamma
    u, v = materialize_from_logs(logu, logv)
    gamma = (u[:, None]) * K * (v[None, :])
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=1.0 / EPS_MIN, neginf=0.0)
    gamma = np.clip(gamma, 0.0, 1.0 / EPS_MIN)

    info = {
        "err_hist": np.array(err_hist, dtype=float),
        "Z_hist": np.array(z_hist, dtype=float),
        "K": K
    }

    if return_history:
        return u, v, gamma, t, info
    return u, v, gamma, t

# ---------------------------------------------------------------------
# Exact cost <C, gamma>
# ---------------------------------------------------------------------
def compute_cost(C, gamma):
    return float(np.sum(C * gamma))

# ---------------------------------------------------------------------
# Simple rectangular cost generator for standalone smoke tests
# ---------------------------------------------------------------------
def generate_cost_matrix_rect(n, m, seed=None):
    """Squared distances on [0,1] + small jitter."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    X = x[:, None]
    Y = y[None, :]
    C = (X - Y) ** 2
    C += 0.01 * rng.random((n, m))
    return C

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulated hybrid quantum-classical Sinkhorn (log-domain, QAE-based)")
    parser.add_argument("--n", type=int, default=8, help="Rows (n)")
    parser.add_argument("--m", type=int, default=0, help="Cols (m=0 => m=n)")
    parser.add_argument("--tau", type=float, default=1.0, help="Entropic parameter tau (>0)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Stopping tolerance on L1 marginal error")
    parser.add_argument("--ae_eps", type=float, default=0.02, help="Target additive precision for QAE simulation")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum iterations")
    parser.add_argument("--shots_factor", type=int, default=100, help="Shots multiplier: shots ~ shots_factor / ae_eps")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--use_exact_norm", action="store_true", help="Use exact normalization Z for marginal reconstruction")
    parser.add_argument("--save", type=str, default="", help="Optional path to save results (npz)")
    # New knobs
    parser.add_argument("--damping_eta", type=float, default=1.0, help="Damping factor for log-updates in (0,1]; e.g., 0.5")
    parser.add_argument("--exact_check_period", type=int, default=0, help="If >0, do exact refresh every s iterations (e.g., 10)")
    parser.add_argument("--warmstart_steps", type=int, default=0, help="If >0, do that many classical Sinkhorn sweeps before quantum loop")
    args = parser.parse_args()

    n = args.n
    m = args.m if args.m > 0 else n

    # Try to use utils if available; fallback to rectangular cost otherwise
    try:
        from src import utils  # when running from repo root
        C = utils.generate_cost_matrix(n, seed=args.seed) if n == m else generate_cost_matrix_rect(n, m, seed=args.seed)
        mu = utils.normalize_distribution(np.ones(n))
        nu = utils.normalize_distribution(np.ones(m))
    except Exception:
        C = generate_cost_matrix_rect(n, m, seed=args.seed)
        mu = np.ones(n) / n
        nu = np.ones(m) / m

    u, v, gamma, iters, info = quantum_sinkhorn_sim(
        C, mu, nu,
        tau=args.tau,
        tol=args.tol,
        ae_eps=args.ae_eps,
        max_iter=args.max_iter,
        shots_factor=args.shots_factor,
        use_exact_norm=args.use_exact_norm,
        seed=args.seed,
        return_history=True,
        damping_eta=args.damping_eta,
        exact_check_period=args.exact_check_period,
        warmstart_steps=args.warmstart_steps,
    )

    exact_cost = compute_cost(C, gamma)
    _, noisy_cost = hadamard_cost_sim(C, u, v, info["K"], np.random.default_rng(args.seed + 1), ae_eps=args.ae_eps)

    print(f"[Quantum-Sim] n={n}, m={m}, tau={args.tau}, tol={args.tol}, ae_eps={args.ae_eps}, iters={iters}")
    print(f"  Exact transport cost <C, gamma>   = {exact_cost:.6f}")
    print(f"  Noisy (Hadamard-like) cost est.  = {noisy_cost:.6f}")
    if len(info["err_hist"]) > 0:
        print(f"  Final marginal L1 error (noisy)  = {info['err_hist'][-1]:.3e}")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            C=C, mu=mu, nu=nu,
            u=u, v=v, gamma=gamma,
            iters=iters, err_hist=info["err_hist"], Z_hist=info["Z_hist"],
            tau=args.tau, tol=args.tol, ae_eps=args.ae_eps, shots_factor=args.shots_factor,
            use_exact_norm=args.use_exact_norm,
            damping_eta=args.damping_eta, exact_check_period=args.exact_check_period, warmstart_steps=args.warmstart_steps,
        )
        print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()