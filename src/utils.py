"""
utils.py - Helper functions for Quantum Sinkhorn simulations
"""

import numpy as np

def generate_cost_matrix(n, seed=None):
    """
    Generate a random symmetric cost matrix with nonnegative entries.

    Args:
        n (int): Dimension of the cost matrix (n x n).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Symmetric cost matrix of shape (n, n).
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.rand(n, n)
    C = (A + A.T) / 2.0  # symmetrize
    return C


def gibbs_kernel(C, tau):
    """
    Construct Gibbs kernel K = exp(-tau * C).

    Args:
        C (np.ndarray): Cost matrix (n x n).
        tau (float): Regularization parameter.

    Returns:
        np.ndarray: Gibbs kernel of shape (n, n).
    """
    return np.exp(-tau * C)


def normalize_distribution(p):
    """
    Normalize a vector to sum to 1.

    Args:
        p (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized probability distribution.
    """
    p = np.maximum(p, 0)
    if np.sum(p) == 0:
        raise ValueError("Input vector cannot be all zeros.")
    return p / np.sum(p)


def l1_error(p, q):
    """
    Compute L1 error between two distributions.

    Args:
        p (np.ndarray): First distribution.
        q (np.ndarray): Second distribution.

    Returns:
        float: L1 distance.
    """
    return np.sum(np.abs(p - q))


def kl_divergence(p, q):
    """
    Compute KL divergence KL(p||q).

    Args:
        p (np.ndarray): First distribution.
        q (np.ndarray): Second distribution.

    Returns:
        float: KL divergence value.
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))
