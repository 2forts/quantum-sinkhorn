#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exp_convergence.py
Plot L1 marginal error vs iterations for:
- Classical Sinkhorn (exact marginals per iteration)
- Quantum-simulated Sinkhorn (QAE-like noisy marginals)
Also prints a post-hoc exact L1 error for the quantum-sim plan.

This script is robust to being executed from anywhere because it
injects the repo's `src/` directory into sys.path explicitly.
"""

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

import utils
from classical_sinkhorn import sinkhorn as classical_sinkhorn
from classical_sinkhorn import compute_cost as classical_cost
from quantum_sinkhorn_sim import quantum_sinkhorn_sim, gibbs_kernel

# --------------------------- Stability knobs ----------------------------
EPS_MIN = 1e-300
CAP_MAX = 1e300

def run(
    n=16,
    tau=0.5,
    eps=1e-8,
    tol=3e-3,
    ae_eps=0.01,
    shots_factor=800,
    max_iter=4000,
    seed=42,
    outdir=str(REPO_ROOT / "figures"),
    # quantum-sim stabilization (forwarded to core):
    use_exact_norm=False,
    damping_eta=1.0,
    exact_check_period=0,
    warmstart_steps=0,
):
    """
    Run classical and quantum-sim convergence experiment and save a PNG.

    Args:
        n: problem size (n x n).
        tau: entropic regularization parameter.
        eps: classical tolerance (used inside classical Sinkhorn).
        tol: quantum-sim stopping tolerance on L1 marginal error (noisy).
        ae_eps: target additive precision for QAE simulation.
        shots_factor: shots ~ shots_factor / ae_eps.
        max_iter: maximum iterations for quantum-sim.
        seed: RNG seed.
        outdir: directory to save the figure.
        use_exact_norm: use exact Z to reconstruct marginals in QAE simulation.
        damping_eta: in (0,1]; fraction of each log-update to apply (helps stability).
        exact_check_period: every s iterations, compute exact marginals and apply a mild refresh.
        warmstart_steps: number of classical Sinkhorn sweeps before quantum loop.
    """
    rng = np.random.default_rng(seed)

    # --- Problem ---
    C = utils.generate_cost_matrix(n, seed=seed)
    mu = utils.normalize_distribution(np.ones(n))
    nu = utils.normalize_distribution(np.ones(n))

    # ------------------------- Classical (baseline) -------------------------
    u_c, v_c, gamma_c, iters_c, err_hist_c = classical_sinkhorn(
        C, mu, nu, tau=tau, eps=eps, max_iter=10000, return_history=True
    )
    cost_c = classical_cost(C, gamma_c)

    # ------------------------- Quantum-sim (noisy) --------------------------
    u_q, v_q, gamma_q, iters_q, info_q = quantum_sinkhorn_sim(
        C, mu, nu,
        tau=tau,
        tol=tol,
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

    # --------------------- Post-hoc exact error for quantum -----------------
    row_exact = gamma_q.sum(axis=1)
    col_exact = gamma_q.sum(axis=0)
    err_q_posthoc = max(
        float(np.sum(np.abs(row_exact - mu))),
        float(np.sum(np.abs(col_exact - nu)))
    )

    # ------------------------------ Plotting --------------------------------
    # Clean any non-positive or non-finite values before log-plotting
    err_c = np.array(err_hist_c, dtype=float)
    err_q = np.array(info_q["err_hist"], dtype=float)

    def _clean(y):
        y = np.array(y, dtype=float)
        y = np.nan_to_num(y, nan=np.inf, posinf=np.inf, neginf=np.inf)
        mask = np.isfinite(y) & (y > 0)
        x = np.arange(1, mask.sum() + 1)
        return x, y[mask]

    x_c, y_c = _clean(err_c)
    x_q, y_q = _clean(err_q)

    plt.figure()
    plt.loglog(x_c, y_c, label="Classical (L1 marginal error)")
    plt.loglog(x_q, y_q, label="Quantum-sim (L1 marginal error)")
    plt.xlabel("Iterations")
    plt.ylabel("L1 marginal error")
    plt.title(
        f"Convergence (n={n}, tau={tau}, ae_eps={ae_eps}, "
        f"eta={damping_eta}, s={exact_check_period}, warm={warmstart_steps})"
    )
    plt.legend()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"convergence_n{n}_tau{tau}_ae{ae_eps}_eta{damping_eta}_s{exact_check_period}_warm{warmstart_steps}.png"
    plt.savefig(fpath, bbox_inches="tight", dpi=180)
    print("Saved:", fpath)

    # ------------------------------ Reporting -------------------------------
    print("\n=== Summary ===")
    print(f"Classical:  iters={iters_c}, final L1={err_c[-1]:.3e}, cost={cost_c:.6f}")
    print(f"Quantum-sim: iters={iters_q}, final noisy L1={err_q[-1]:.3e}")
    print(f"Quantum post-hoc exact L1 marginal error: {err_q_posthoc:.6f}")

    return {
        "iters_classical": iters_c,
        "iters_quantum": iters_q,
        "final_classical_L1": float(err_c[-1]),
        "final_quantum_noisy_L1": float(err_q[-1]) if len(err_q) else np.nan,
        "quantum_posthoc_exact_L1": err_q_posthoc,
        "figure_path": str(fpath),
    }

# ----------------------------------- CLI ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence experiment (classical vs quantum-sim)")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-8, help="Classical Sinkhorn tolerance")
    parser.add_argument("--tol", type=float, default=3e-3, help="Quantum-sim stopping tolerance (noisy)")
    parser.add_argument("--ae_eps", type=float, default=0.01, help="Target QAE precision (additive)")
    parser.add_argument("--shots_factor", type=int, default=800, help="Shots ~ shots_factor / ae_eps")
    parser.add_argument("--max_iter", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=str(REPO_ROOT / "figures"))
    parser.add_argument("--use_exact_norm", action="store_true", help="Use exact Z in QAE simulation")
    parser.add_argument("--damping_eta", type=float, default=1.0, help="Damping factor for log-updates in (0,1]; e.g., 0.5")
    parser.add_argument("--exact_check_period", type=int, default=0, help="If >0, do exact refresh every s iterations")
    parser.add_argument("--warmstart_steps", type=int, default=0, help="If >0, classical Sinkhorn sweeps before quantum loop")
    args = parser.parse_args()

    run(
        n=args.n,
        tau=args.tau,
        eps=args.eps,
        tol=args.tol,
        ae_eps=args.ae_eps,
        shots_factor=args.shots_factor,
        max_iter=args.max_iter,
        seed=args.seed,
        outdir=args.outdir,
        use_exact_norm=args.use_exact_norm,
        damping_eta=args.damping_eta,
        exact_check_period=args.exact_check_period,
        warmstart_steps=args.warmstart_steps,
    )