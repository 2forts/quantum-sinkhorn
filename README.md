# Quantum Sinkhorn: Hybrid Quantum–Classical Algorithm for Entropic Optimal Transport

This repository contains the simulation code, numerical experiments, and figure-generation scripts accompanying the paper:

> **Entropic optimal transport with quantum amplitude estimation**  
> Francisco Orts, 2025  

The project implements both **classical Sinkhorn scaling** and its **quantum-inspired analogue**, where quantum subroutines such as amplitude estimation are simulated to assess convergence, accuracy, and complexity. All experiments in the manuscript can be reproduced using the code in this repository.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/quantum-sinkhorn.git
cd quantum-sinkhorn
pip install -r requirements.txt
```

Dependencies include:
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn` (optional, for nicer plots)
- `jupyter` (optional, for interactive notebooks)

---

## 🧮 Repository structure

```
quantum-sinkhorn/
│
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── src/                    # Core implementations
│   ├── classical_sinkhorn.py
│   ├── quantum_sinkhorn_sim.py
│   ├── utils.py
│   └── ...
├── experiments/            # Scripts to reproduce paper figures
│   ├── exp_convergence.py
│   ├── exp_cost.py
│   ├── exp_tau_dependence.py
│   └── exp_complexity_plots.py
├── figures/                # Auto-generated plots
├── data/                   # Example datasets / cost matrices
└── notebooks/              # Jupyter notebooks with demos
```

---

## ▶️ Usage

### Run a simple example

Compute entropic OT between two small distributions:

```bash
python src/classical_sinkhorn.py --n 4 --tau 1.0
python src/quantum_sinkhorn_sim.py --n 4 --tau 1.0 --epsilon 0.05
```

### Reproduce experiments

Each paper figure has a corresponding script in the `experiments/` folder.  
For example, to reproduce convergence plots:

```bash
python experiments/exp_convergence.py
```

Generated figures will be saved in the `figures/` folder.

---

## 📊 Numerical experiments

The experiments include:
1. **Convergence trajectories** of scaling vectors ($u,v$).  
2. **Cost functional estimation** via Hadamard test simulation.  
3. **Dependence on regularization parameter $\tau$**.  
4. **Complexity scaling** in error tolerance $\varepsilon$ and dimension $n$.  

Exact details are described in the *Methods* section of the manuscript.

---

## 🔁 Reproducibility

- All simulations use fixed random seeds.  
- Classical stochastic baselines use Monte Carlo with $\mathcal{O}(1/\varepsilon^2)$ samples.  
- The full workflow has been tested with Python 3.10+.

---


## 📖 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ✨ Acknowledgements

This work was supported by project PID2021-123278OB-I00 (funded by MCIN/AEI/10.13039/501100011033/FEDER "A way to make Europe"), and from the assistance with reference POST_2024_00998 funded by the Regional Government of Andalusia/CUII and the ESF+. We thank the open-source community for providing the tools that made this project possible.
