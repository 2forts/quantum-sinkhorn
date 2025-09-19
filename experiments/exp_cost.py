#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp_cost.py
# Estimate transport cost: classical exact vs quantum-sim Hadamard-like estimate.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Add repo root to path
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.append(repo_root)

from src.classical_sinkhorn import sinkhorn as classical_sinkhorn, compute_cost as classical_cost
from src.quantum_sinkhorn_sim import quantum_sinkhorn_sim, hadamard_cost_sim
from src import utils


def run(n=16, tau=1.0, tol=1e-4, ae_eps_list=(0.05, 0.03, 0.02, 0.01),
        seed=123, outdir="../figures", shots_factor=100):
    C = utils.generate_cost_matrix(n, seed=seed)
    mu = utils.normalize_distribution(np.ones(n))
    nu = utils.normalize_distribution(np.ones(n))
    
    # Classical exact plan and cost
    u_c, v_c, gamma_c, iters_c, _ = classical_sinkhorn(
        C, mu, nu, tau=tau, eps=1e-8, max_iter=8000, return_history=True
    )
    exact = classical_cost(C, gamma_c)
    
    # Quantum-sim noisy cost for different ae_eps
    noisy_vals = []
    for ae in ae_eps_list:
        u_q, v_q, gamma_q, iters_q, info = quantum_sinkhorn_sim(
            C, mu, nu, tau=tau, tol=tol, ae_eps=ae, max_iter=5000,
            shots_factor=shots_factor, use_exact_norm=True, seed=seed, return_history=True
        )
        _, noisy = hadamard_cost_sim(C, u_q, v_q, info["K"],
                                     np.random.default_rng(seed+1), ae_eps=ae)
        noisy_vals.append(noisy)
    
    # Plot absolute error vs ae_eps
    plt.figure()
    x = np.array(ae_eps_list, dtype=float)
    plt.plot(x, np.abs(np.array(noisy_vals) - exact), marker="o", linestyle="-",
             label="|Noisy cost - Exact|")
    plt.gca().invert_xaxis()  # smaller epsilon to the right
    plt.xlabel("Target QAE precision (ae_eps)")
    plt.ylabel("Absolute cost error")
    plt.title(f"Cost estimation vs QAE precision (n={n}, tau={tau})")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"cost_estimation_n{n}_tau{tau}.png")
    plt.savefig(fpath, bbox_inches="tight", dpi=180)
    print("Saved:", fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost estimation experiment")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="../figures")
    parser.add_argument("--ae_eps_list", type=float, nargs="+",
                        default=[0.05, 0.03, 0.02, 0.01],
                        help="List of amplitude estimation epsilons to test")
    parser.add_argument("--shots_factor", type=int, default=100,
                        help="Factor controlling number of multinomial samples in AE simulation")
    args = parser.parse_args()
    run(n=args.n, tau=args.tau, tol=args.tol,
        ae_eps_list=args.ae_eps_list, seed=args.seed,
        outdir=args.outdir, shots_factor=args.shots_factor)