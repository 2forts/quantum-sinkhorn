#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
quantum_sinkhorn_sim.py
Simulation of a hybrid quantum–classical Sinkhorn algorithm using a probabilistic
model for quantum amplitude estimation (QAE) and a Hadamard-test-like cost readout.
\"\"\"

import argparse
import numpy as np
from pathlib import Path

try:
    import utils
except ImportError:
    utils = None

EPS_MIN = 1e-300

def gibbs_kernel(C, tau):
    \"\"\"K = exp(-tau * C) with underflow protection.\"\"\"
    K = np.exp(-tau * C)
    return np.maximum(K, EPS_MIN)

def normalize(p):
    s = np.sum(p)
    if s <= 0:
        raise ValueError(\"Cannot normalize a nonpositive vector.\")
    return p / s

def l1_error(p, q):
    return float(np.sum(np.abs(p - q)))

def generate_cost_matrix_rect(n, m, seed=None):
    \"\"\"Generate a rectangular cost matrix using squared Euclidean distances on [0,1].\"\"\"
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    X = x[:, None]
    Y = y[None, :]
    C = (X - Y) ** 2
    # small random jitter to avoid perfect structure
    C += 0.01 * rng.random((n, m))
    return C

def qae_multinomial_estimate(prob_vec, shots, rng):
    \"\"\"Estimate a probability vector via multinomial sampling (proxy for QAE outcomes).
    Returns an unbiased frequency estimator with variance ~ prob*(1-prob)/shots.
    \"\"\"
    if shots <= 0:
        raise ValueError(\"shots must be positive\")
    counts = rng.multinomial(shots, prob_vec)
    return counts / shots

def simulate_row_marginals(K, u, v, rng, ae_eps=0.02, shots_factor=100, use_exact_norm=True):
    \"\"\"Simulate QAE estimates for row marginals M_i = u_i (K v)_i.
    If use_exact_norm, we compute Z = u^T K v exactly and recover M by M = p_row * Z.
    \"\"\"
    Kv = K @ v
    Z = float(np.dot(u, Kv))
    if Z <= 0:
        # degenerate, fallback to exact (should not happen with positive K,u,v)
        return u * Kv, Z
    p_row = (u * Kv) / Z
    shots = max(int(np.ceil(shots_factor / max(ae_eps, 1e-6))), 1)
    p_hat = qae_multinomial_estimate(p_row, shots, rng)
    if use_exact_norm:
        M_hat = p_hat * Z
    else:
        # if not using exact norm, allow a small random scaling to mimic separate normalization
        Z_hat = Z * (1.0 + rng.normal(0.0, ae_eps))
        M_hat = p_hat * max(Z_hat, EPS_MIN)
    return M_hat, Z

def simulate_col_marginals(K, u, v, rng, ae_eps=0.02, shots_factor=100, use_exact_norm=True):
    \"\"\"Simulate QAE estimates for column marginals N_j = v_j (K^T u)_j.\"\"\"
    KTu = K.T @ u
    Z = float(np.dot(v, KTu))
    if Z <= 0:
        return v * KTu, Z
    p_col = (v * KTu) / Z
    shots = max(int(np.ceil(shots_factor / max(ae_eps, 1e-6))), 1)
    p_hat = qae_multinomial_estimate(p_col, shots, rng)
    if use_exact_norm:
        N_hat = p_hat * Z
    else:
        Z_hat = Z * (1.0 + rng.normal(0.0, ae_eps))
        N_hat = p_hat * max(Z_hat, EPS_MIN)
    return N_hat, Z

def hadamard_cost_sim(C, u, v, K, rng, ae_eps=0.02):
    \"\"\"Simulate a Hadamard-test-like estimate of <C, gamma> with additive noise ~ ae_eps.
    Exact cost plus Gaussian noise with std proportional to ae_eps * scale.
    \"\"\"
    gamma = (u[:, None]) * K * (v[None, :])
    exact = float(np.sum(C * gamma))
    # scale noise by average magnitude of C for a reasonable visual effect
    scale = max(1e-12, float(np.mean(np.abs(C))))
    noisy = exact + rng.normal(0.0, ae_eps * scale)
    return exact, noisy

def quantum_sinkhorn_sim(C, mu, nu, tau=1.0, tol=1e-4, ae_eps=0.02, max_iter=2000,
                         shots_factor=100, use_exact_norm=True, seed=42, return_history=False):
    \"\"\"Hybrid quantum-classical Sinkhorn simulation.
    
    Args:
        C (ndarray): Cost matrix (n x m).
        mu (ndarray): Source marginal (n,).
        nu (ndarray): Target marginal (m,).
        tau (float): Entropic regularization parameter.
        tol (float): Stopping tolerance on L1 marginal error (using QAE estimates).
        ae_eps (float): Target additive precision for QAE simulation.
        max_iter (int): Maximum number of iterations.
        shots_factor (int): Controls shot count via shots ~ shots_factor / ae_eps.
        use_exact_norm (bool): If True, use exact Z = u^T K v when reconstructing marginals.
        seed (int): RNG seed.
        return_history (bool): If True, return dictionaries with history.
        
    Returns:
        u, v, gamma, iters, info (optional)
    \"\"\"
    rng = np.random.default_rng(seed)
    n, m = C.shape
    K = gibbs_kernel(C, tau)
    
    u = np.ones(n)
    v = np.ones(m)
    err_hist = []
    z_hist = []
    
    for t in range(1, max_iter + 1):
        # Simulated QAE for row marginals
        M_hat, Z_row = simulate_row_marginals(K, u, v, rng, ae_eps=ae_eps,
                                              shots_factor=shots_factor, use_exact_norm=use_exact_norm)
        # Update u
        M_hat = np.where(M_hat <= 0, EPS_MIN, M_hat)
        u = mu / M_hat
        
        # Simulated QAE for column marginals
        N_hat, Z_col = simulate_col_marginals(K, u, v, rng, ae_eps=ae_eps,
                                              shots_factor=shots_factor, use_exact_norm=use_exact_norm)
        N_hat = np.where(N_hat <= 0, EPS_MIN, N_hat)
        v = nu / N_hat
        
        # Compute L1 marginal error w.r.t targets using the latest estimates
        err = max(l1_error(M_hat, mu), l1_error(N_hat, nu))
        err_hist.append(err)
        z_hist.append((Z_row, Z_col))
        
        if err <= tol:
            break
    
    gamma = (u[:, None]) * K * (v[None, :])
    
    info = {
        \"err_hist\": np.array(err_hist),
        \"Z_hist\": np.array(z_hist, dtype=float),
        \"K\": K
    }
    if return_history:
        return u, v, gamma, t, info
    return u, v, gamma, t

def compute_cost(C, gamma):
    return float(np.sum(C * gamma))

def main():
    parser = argparse.ArgumentParser(description=\"Simulated hybrid quantum-classical Sinkhorn (QAE-based)\")
    parser.add_argument(\"--n\", type=int, default=8, help=\"Number of rows n\")
    parser.add_argument(\"--m\", type=int, default=0, help=\"Number of cols m (0 => m=n)\")
    parser.add_argument(\"--tau\", type=float, default=1.0, help=\"Entropic parameter tau (>0)\")
    parser.add_argument(\"--tol\", type=float, default=1e-4, help=\"Stopping tolerance on L1 marginal error\")
    parser.add_argument(\"--ae_eps\", type=float, default=0.02, help=\"Target additive precision for QAE simulation\")
    parser.add_argument(\"--max_iter\", type=int, default=2000, help=\"Maximum iterations\")
    parser.add_argument(\"--shots_factor\", type=int, default=100, help=\"Shots multiplier: shots ~ shots_factor / ae_eps\")
    parser.add_argument(\"--seed\", type=int, default=123, help=\"Random seed\")
    parser.add_argument(\"--use_exact_norm\", action=\"store_true\", help=\"Use exact normalization Z when reconstructing marginals\")
    parser.add_argument(\"--save\", type=str, default=\"\", help=\"Optional path to save results (npz)\")
    args = parser.parse_args()
    
    n = args.n
    m = args.m if args.m > 0 else n
    
    # Generate cost matrix and marginals
    if utils is not None:
        if n == m:
            C = utils.generate_cost_matrix(n, seed=args.seed)
        else:
            C = generate_cost_matrix_rect(n, m, seed=args.seed)
        mu = utils.normalize_distribution(np.ones(n))
        nu = utils.normalize_distribution(np.ones(m))
    else:
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
        return_history=True
    )
    
    exact_cost = compute_cost(C, gamma)
    # Simulate Hadamard-test-like cost readout
    _, noisy_cost = hadamard_cost_sim(C, u, v, info[\"K\"], np.random.default_rng(args.seed+1), ae_eps=args.ae_eps)
    
    print(f\"[Quantum-Sim] n={n}, m={m}, tau={args.tau}, tol={args.tol}, ae_eps={args.ae_eps}\")
    print(f\"  Converged in {iters} iterations\")
    print(f\"  Exact transport cost <C, gamma>   = {exact_cost:.6f}\")
    print(f\"  Noisy (Hadamard-like) cost est.  = {noisy_cost:.6f}\")
    if len(info[\"err_hist\"])>0:
        print(f\"  Final marginal L1 error (est.)  = {info['err_hist'][-1]:.3e}\")
    
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, C=C, mu=mu, nu=nu, u=u, v=v, gamma=gamma,
                 iters=iters, err_hist=info[\"err_hist\"], Z_hist=info[\"Z_hist\"],
                 tau=args.tau, tol=args.tol, ae_eps=args.ae_eps, shots_factor=args.shots_factor,
                 use_exact_norm=args.use_exact_norm)
        print(f\"Saved results to {out_path}\")
    
if __name__ == \"__main__\":
    main()
