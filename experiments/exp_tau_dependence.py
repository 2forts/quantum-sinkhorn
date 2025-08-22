#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp_tau_dependence.py
# Explore dependence on regularization parameter tau.

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

def run(n=16, taus=(0.2, 0.5, 1.0, 2.0, 5.0), tol=1e-4, ae_eps=0.02, seed=7, outdir="../figures"):
    C = utils.generate_cost_matrix(n, seed=seed)
    mu = utils.normalize_distribution(np.ones(n))
    nu = utils.normalize_distribution(np.ones(n))
    
    iters_classical = []
    iters_quantum = []
    for tau in taus:
        _, _, _, it_c, _ = classical_sinkhorn(C, mu, nu, tau=tau, eps=1e-4, max_iter=10000, return_history=True)
        _, _, _, it_q, _ = quantum_sinkhorn_sim(
            C, mu, nu, tau=tau, tol=tol, ae_eps=ae_eps, max_iter=10000,
            shots_factor=100, use_exact_norm=True, seed=seed, return_history=True
        )
        iters_classical.append(it_c)
        iters_quantum.append(it_q)
    
    taus_arr = np.array(taus, dtype=float)
    plt.figure()
    plt.plot(taus_arr, iters_classical, marker="o", linestyle="-", label="Classical iterations")
    plt.plot(taus_arr, iters_quantum, marker="s", linestyle="-", label="Quantum-sim iterations")
    plt.xlabel("Regularization parameter tau")
    plt.ylabel("Iterations to tolerance")
    plt.title(f"Convergence iterations vs tau (n={n})")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"tau_dependence_n{n}.png")
    plt.savefig(fpath, bbox_inches="tight", dpi=180)
    print("Saved:", fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tau dependence experiment")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--ae_eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="../figures")
    args = parser.parse_args()
    run(n=args.n, tol=args.tol, ae_eps=args.ae_eps, seed=args.seed, outdir=args.outdir)
