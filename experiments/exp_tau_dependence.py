#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp_tau_dependence.py
# Explore dependence on regularization parameter tau (classical vs quantum-sim).

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --------------------------- Robust path setup ---------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]           # repo root = parent of 'experiments'
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classical_sinkhorn import sinkhorn as classical_sinkhorn
from quantum_sinkhorn_sim import quantum_sinkhorn_sim
import utils


def run(
    n=16,
    tau_list=(0.2, 0.3, 0.5, 0.8, 1.0),
    tol=1e-4,
    ae_eps=0.02,
    seed=7,
    outdir=str(REPO_ROOT / "figures"),
    # extra knobs for fair/robust comparisons:
    shots_factor=100,
    max_iter=10000,
    use_exact_norm=True,
    damping_eta=1.0,
    exact_check_period=0,
    warmstart_steps=0,
    # auto tolerance guard:
    auto_tol_factor=4.0,
):
    """
    Sweep over tau values and compare iterations-to-tolerance for:
      - Classical Sinkhorn (exact marginals per iteration)
      - Quantum-simulated Sinkhorn (QAE-like noisy marginals)

    Args:
        n: problem size (n x n).
        tau_list: iterable of tau values to test.
        tol: requested stopping tolerance for quantum-sim (L1 of marginals).
        ae_eps: target additive precision for QAE simulation.
        seed: RNG seed for reproducibility.
        outdir: directory to save the figure.
        shots_factor: shots ~ shots_factor / ae_eps for multinomial simulator.
        max_iter: maximum iterations for both algorithms.
        use_exact_norm: reconstruct marginals with exact Z in QAE simulation.
        damping_eta: damping factor (0,1] for log updates in quantum-sim.
        exact_check_period: if >0, periodic exact refresh every s iterations.
        warmstart_steps: if >0, do a few classical sweeps before quantum loop.
        auto_tol_factor: enforce effective_tol = max(tol, auto_tol_factor * ae_eps).
    """
    rng = np.random.default_rng(seed)
    C = utils.generate_cost_matrix(n, seed=seed)
    mu = utils.normalize_distribution(np.ones(n))
    nu = utils.normalize_distribution(np.ones(n))

    # Auto-adjust tolerance to avoid asking below the AE noise floor
    base_tol = float(tol)
    effective_tol = max(base_tol, float(auto_tol_factor) * float(ae_eps))
    if effective_tol > base_tol:
        print(f"[tau-sweep] Requested tol={base_tol:g} too tight for ae_eps={ae_eps:g}. "
              f"Using effective tol={effective_tol:g} (auto_tol_factor={auto_tol_factor:g}).")

    taus = list(map(float, tau_list))
    iters_classical = []
    iters_quantum = []

    print(f"[tau-sweep] n={n}, ae_eps={ae_eps}, shots_factor={shots_factor}, "
          f"max_iter={max_iter}, use_exact_norm={use_exact_norm}, "
          f"eta={damping_eta}, s={exact_check_period}, warm={warmstart_steps}")

    for tau in taus:
        # Classical iterations to (very tight) tolerance for baseline
        _, _, _, it_c, _ = classical_sinkhorn(
            C, mu, nu, tau=tau, eps=min(1e-8, effective_tol * 1e-3), max_iter=max_iter, return_history=True
        )

        # Quantum-sim iterations to (noisy) effective tolerance with given knobs
        _, _, _, it_q, _ = quantum_sinkhorn_sim(
            C, mu, nu,
            tau=tau,
            tol=effective_tol,
            ae_eps=ae_eps,
            max_iter=max_iter,
            shots_factor=shots_factor,
            use_exact_norm=use_exact_norm,
            seed=seed,
            return_history=True,
            damping_eta=damping_eta,
            exact_check_period=exact_check_period,
            warmstart_steps=warmstart_steps,
        )

        iters_classical.append(it_c)
        iters_quantum.append(it_q)
        print(f"  tau={tau:.3f} -> classical iters: {it_c:4d} | quantum-sim iters: {it_q:4d}")

    taus_arr = np.array(taus, dtype=float)
    iters_classical = np.array(iters_classical, dtype=int)
    iters_quantum = np.array(iters_quantum, dtype=int)

    # Plot iterations vs tau
    plt.figure()
    plt.plot(taus_arr, iters_classical, marker="o", linestyle="-", label="Classical iterations")
    plt.plot(taus_arr, iters_quantum, marker="s", linestyle="-", label="Quantum-sim iterations")
    plt.xlabel("Regularization parameter $\\tau$")
    plt.ylabel("Iterations to tolerance")
    plt.title(
        f"Iterations vs $\\tau$ (n={n}, ae_eps={ae_eps}, tol_eff={effective_tol}, "
        f"eta={damping_eta}, s={exact_check_period}, warm={warmstart_steps})"
    )
    plt.legend()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / (
        f"tau_dependence_n{n}_ae{ae_eps}_toleff{effective_tol}_"
        f"eta{damping_eta}_s{exact_check_period}_warm{warmstart_steps}.png"
    )
    plt.savefig(fpath, bbox_inches="tight", dpi=180)
    print("Saved:", fpath)

    return {
        "taus": taus_arr,
        "iters_classical": iters_classical,
        "iters_quantum": iters_quantum,
        "effective_tol": effective_tol,
        "figure_path": str(fpath),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tau dependence experiment (classical vs quantum-sim)")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tau_list", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--ae_eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default=str(REPO_ROOT / "figures"))
    parser.add_argument("--shots_factor", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--use_exact_norm", action="store_true")
    parser.add_argument("--damping_eta", type=float, default=1.0)
    parser.add_argument("--exact_check_period", type=int, default=0)
    parser.add_argument("--warmstart_steps", type=int, default=0)
    parser.add_argument("--auto_tol_factor", type=float, default=4.0,
                        help="Use effective_tol = max(tol, auto_tol_factor * ae_eps) to avoid unreachable tol.")
    args = parser.parse_args()

    run(
        n=args.n,
        tau_list=args.tau_list,
        tol=args.tol,
        ae_eps=args.ae_eps,
        seed=args.seed,
        outdir=args.outdir,
        shots_factor=args.shots_factor,
        max_iter=args.max_iter,
        use_exact_norm=args.use_exact_norm,
        damping_eta=args.damping_eta,
        exact_check_period=args.exact_check_period,
        warmstart_steps=args.warmstart_steps,
        auto_tol_factor=args.auto_tol_factor,
    )