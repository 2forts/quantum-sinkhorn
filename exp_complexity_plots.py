#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp_complexity_plots.py
# Generate illustrative asymptotic curves used in the complexity analysis figure.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def run(outdir="../figures"):
    # Panel (a): accuracy scaling
    eps = np.logspace(-3, 0, 200)
    y_q = 1/eps
    y_c = 1/(eps**2)
    plt.figure()
    plt.loglog(eps, y_q, label="Quantum  ~  O~(1/ε)")
    plt.loglog(eps, y_c, linestyle="--", label="Classical  ~  Θ~(1/ε²)")
    plt.xlabel("Error tolerance ε")
    plt.ylabel("Query complexity (arb. units)")
    plt.title("(a) Accuracy scaling")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    fpath_a = os.path.join(outdir, "complexity_accuracy.png")
    plt.savefig(fpath_a, bbox_inches="tight", dpi=180)
    print("Saved:", fpath_a)
    
    # Panel (b): dimensional scaling
    n = np.logspace(2, 7, 200)
    y_c2 = n
    y_q2 = (np.log(n))**3  # illustrative polylog
    plt.figure()
    plt.loglog(n, y_c2, label="Classical  ~  O~(n)")
    plt.loglog(n, y_q2, linestyle="--", label="Quantum  ~  O~((log n)^3)")
    plt.xlabel("Dimension n")
    plt.ylabel("Query complexity (arb. units)")
    plt.title("(b) Dimensional scaling")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    fpath_b = os.path.join(outdir, "complexity_dimension.png")
    plt.savefig(fpath_b, bbox_inches="tight", dpi=180)
    print("Saved:", fpath_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate complexity analysis illustrative curves")
    parser.add_argument("--outdir", type=str, default="../figures")
    args = parser.parse_args()
    run(outdir=args.outdir)
