"""
Kernel (covariance) functions for Gaussian processes.
"""

import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Radial Basis Function (squared exponential) kernel.

    k(x, x') = σ² exp(-||x-x'||² / (2ℓ²))

    Parameters
    ----------
    X1 : array (N1, d)
    X2 : array (N2, d)
    lengthscale : float (ℓ)
    variance : float (σ²)

    Returns
    -------
    K : array (N1, N2)
    """
    D = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-0.5 * D / lengthscale**2)


def matern32_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Matérn 3/2 kernel.

    k(x, x') = σ² (1 + √3 r/ℓ) exp(-√3 r/ℓ), where r = ||x-x'||
    """
    D = cdist(X1, X2, metric='euclidean')
    scaled = np.sqrt(3) * D / lengthscale
    return variance * (1 + scaled) * np.exp(-scaled)


def matern52_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Matérn 5/2 kernel.

    k(x, x') = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)
    """
    D = cdist(X1, X2, metric='euclidean')
    scaled = np.sqrt(5) * D / lengthscale
    return variance * (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
