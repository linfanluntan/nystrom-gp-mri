"""
Exact Gaussian Process regression (reference implementation).

Used as the gold-standard baseline for evaluating Nyström accuracy.
Computational cost: O(N³) time, O(N²) memory.
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve
from .kernels import rbf_kernel


class FullGP:
    """Exact GP regression via Cholesky factorization."""

    def __init__(self, lengthscale=0.15, variance=1.0, noise_var=0.1):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var

    def fit(self, X_train, y_train):
        K = rbf_kernel(X_train, X_train, self.lengthscale, self.variance)
        K += self.noise_var * np.eye(len(X_train))
        self.L_ = cholesky(K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), y_train)
        self._X_train = X_train
        return self

    def predict(self, X_test, return_var=True):
        K_s = rbf_kernel(self._X_train, X_test, self.lengthscale, self.variance)
        mu = K_s.T @ self.alpha_

        if return_var:
            K_ss = rbf_kernel(X_test, X_test, self.lengthscale, self.variance)
            V = cho_solve((self.L_, True), K_s)
            var = np.diag(K_ss) - np.sum(K_s * V, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, var
        return mu
