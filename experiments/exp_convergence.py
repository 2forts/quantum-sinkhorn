#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp_convergence.py
# Reproduce convergence trajectories (L1 marginal error vs iterations)
# for classical and quantum-simulated Sinkhorn.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Add repo root to path
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.append(repo_root)

from classical_sinkhorn import sinkhorn as classical_sinkhorn
from quantum_sinkhorn_sim import quantum_sinkhorn_sim
import utils

def run(n=16, tau=1.0, eps=1e-4, tol=1e-4, ae_eps=0.02, seed=42, outdir="../figures"):
    # Problem
    C = utils.generate_cost_matrix(n, seed=seed)
    mu = utils.normalize_distribution(np.ones(n))
    nu = utils.normalize_distribution(np.ones(n))
    
    # Classical with history
    u_c, v_c, gamma_c, iters_c, err_hist_c = classical_sinkhorn(
        C, mu, nu, tau=tau, eps=eps, max_iter=5000, return_history=True
    )
    
    # Quantum-simulated with history
    u_q, v_q, gamma_q, iters_q, info = quantum_sinkhorn_sim(
        C, mu, nu, tau=tau, tol=tol, ae_eps=ae_eps, max_iter=5000,
        shots_factor=100, use_exact_norm=True, seed=seed, return_history=True
    )
    err_hist_q = info["err_hist"]
    
    # Plot
    plt.figure()
    x_c = np.arange(1, len(err_hist_c)+1)
    x_q = np.arange(1, len(err_hist_q)+1)
    plt.loglog(x_c, err_hist_c, label="Classical (L1 marginal error)")
    plt.loglog(x_q, err_hist_q, label="Quantum-sim (L1 marginal error)")
    plt.xlabel("Iterations")
    plt.ylabel("L1 marginal error")
    plt.legend()
    plt.title(f"Convergence trajectories (n={n}, tau={tau})")
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"convergence_n{n}_tau{tau}.png")
    plt.savefig(fpath, bbox_inches="tight", dpi=180)
    print("Saved:", fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence experiment")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-4, help="Classical tolerance")
    parser.add_argument("--tol", type=float, default=1e-4, help="Quantum-sim tolerance")
    parser.add_argument("--ae_eps", type=float, default=0.02, help="QAE target precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="../figures")
    args = parser.parse_args()
    run(n=args.n, tau=args.tau, eps=args.eps, tol=args.tol, ae_eps=args.ae_eps, seed=args.seed, outdir=args.outdir)
