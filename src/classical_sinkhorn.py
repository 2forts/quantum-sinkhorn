#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classical_sinkhorn.py
Basic implementation of entropic optimal transport via classical Sinkhorn scaling.
"""

import argparse
import numpy as np
from pathlib import Path

try:
    # Expect utils.py to be available in PYTHONPATH or same dir
    import utils
except ImportError:
    utils = None

def sinkhorn(C, mu, nu, tau=1.0, eps=1e-4, max_iter=5000, return_history=False):
    """
    Classical Sinkhorn scaling for entropic OT.
    
    Args:
        C (ndarray): Cost matrix (n x m).
        mu (ndarray): Source marginal (n,).
        nu (ndarray): Target marginal (m,).
        tau (float): Entropic regularization parameter (>0).
        eps (float): Stopping tolerance on L1 marginal error.
        max_iter (int): Maximum number of iterations.
        return_history (bool): If True, also return error history.
        
    Returns:
        u, v, gamma, iters, err (if return_history) 
    """
    n, m = C.shape
    # Gibbs kernel
    K = np.exp(-tau * C)
    # Avoid zeros
    K = np.maximum(K, 1e-300)
    # Initialize scaling vectors
    u = np.ones(n)
    v = np.ones(m)
    err_hist = []
    
    for t in range(1, max_iter + 1):
        # Row scaling to match mu
        Ku = K @ v
        # Avoid division by zero
        Ku = np.where(Ku == 0, 1e-300, Ku)
        u = mu / Ku
        
        # Column scaling to match nu
        KT_u = K.T @ u
        KT_u = np.where(KT_u == 0, 1e-300, KT_u)
        v = nu / KT_u
        
        # Current marginals
        M = u * (K @ v)      # row sums
        N = v * (K.T @ u)    # column sums
        
        # L1 marginal errors
        err = max(np.sum(np.abs(M - mu)), np.sum(np.abs(N - nu)))
        if return_history:
            err_hist.append(err)
        
        if err <= eps:
            break
    
    # Transport plan
    gamma = (u[:, None]) * K * (v[None, :])
    
    if return_history:
        return u, v, gamma, t, np.array(err_hist)
    return u, v, gamma, t

def compute_cost(C, gamma):
    """Transport cost <C, gamma>."""
    return float(np.sum(C * gamma))

def main():
    parser = argparse.ArgumentParser(description="Classical Sinkhorn for entropic OT")
    parser.add_argument("--n", type=int, default=8, help="Problem size n (square n x n)")
    parser.add_argument("--tau", type=float, default=1.0, help="Entropic parameter tau (>0)")
    parser.add_argument("--eps", type=float, default=1e-4, help="Stopping tolerance on L1 marginal error")
    parser.add_argument("--max_iter", type=int, default=5000, help="Maximum iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="", help="Optional path to save results (npz)")
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # Generate a symmetric nonnegative cost matrix if utils available, else random
    if utils is not None:
        C = utils.generate_cost_matrix(args.n, seed=args.seed)
        mu = utils.normalize_distribution(np.ones(args.n))
        nu = utils.normalize_distribution(np.ones(args.n))
    else:
        A = rng.random((args.n, args.n))
        C = (A + A.T) / 2.0
        mu = np.ones(args.n) / args.n
        nu = np.ones(args.n) / args.n
    
    u, v, gamma, iters, err_hist = sinkhorn(C, mu, nu, tau=args.tau, eps=args.eps, max_iter=args.max_iter, return_history=True)
    cost = compute_cost(C, gamma)
    
    print(f"[Sinkhorn] n={args.n}, tau={args.tau}, eps={args.eps}")
    print(f"  Converged in {iters} iterations")
    print(f"  Transport cost <C, gamma> = {cost:.6f}")
    print(f"  Final marginal L1 error   = {err_hist[-1] if len(err_hist)>0 else np.nan:.3e}")
    
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, C=C, mu=mu, nu=nu, u=u, v=v, gamma=gamma, iters=iters, err_hist=err_hist, tau=args.tau, eps=args.eps)
        print(f"Saved results to {out_path}")
    
if __name__ == "__main__":
    main()
