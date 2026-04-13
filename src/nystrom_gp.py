"""
Nyström-Accelerated Gaussian Process Regression.

Implements the low-rank GP posterior via Woodbury identity:
    K ≈ Φ Φᵀ,  where Φ = C W^{-1/2}
    (Φ Φᵀ + σ²I)⁻¹ y = σ⁻²(y - Φ(Φᵀ Φ + σ²I_M)⁻¹ Φᵀ y)

References:
    [1] Williams & Seeger (2001). Using the Nyström method to speed up kernel machines. NeurIPS.
    [2] Rasmussen & Williams (2006). Gaussian Processes for Machine Learning. MIT Press, Ch. 8.
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve
from .kernels import rbf_kernel
from .landmarks import select_landmarks


class NystromGP:
    """
    Nyström-approximated Gaussian Process regression.

    Parameters
    ----------
    M : int
        Number of landmark (inducing) points.
    landmark_method : str
        Landmark selection strategy: 'random', 'kmeans', or 'grid'.
    lengthscale : float
        RBF kernel lengthscale.
    variance : float
        RBF kernel signal variance.
    noise_var : float
        Observation noise variance.
    jitter : float
        Diagonal jitter for numerical stability.
    """

    def __init__(self, M=100, landmark_method='kmeans',
                 lengthscale=0.15, variance=1.0, noise_var=0.1, jitter=1e-6):
        self.M = M
        self.landmark_method = landmark_method
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.jitter = jitter

        # Fitted quantities
        self.Z_ = None          # landmark points (M, d)
        self.Phi_train_ = None  # low-rank factor (N_train, M)
        self.L_A_ = None        # Cholesky of (Φᵀ Φ + σ² I_M)
        self.v_ = None          # solve vector for posterior mean
        self.W_sqrt_inv_ = None # W^{-1/2} for cross-covariance projection

    def fit(self, X_train, y_train):
        """
        Fit the Nyström GP.

        Parameters
        ----------
        X_train : array (N, d) — training coordinates
        y_train : array (N,) — training observations

        Returns
        -------
        self
        """
        N = len(X_train)
        M = min(self.M, N - 1)

        # Select landmarks
        self.Z_ = select_landmarks(X_train, M, self.landmark_method)

        # Compute kernel blocks
        W = rbf_kernel(self.Z_, self.Z_, self.lengthscale, self.variance)
        W += self.jitter * np.eye(M)
        C_train = rbf_kernel(X_train, self.Z_, self.lengthscale, self.variance)

        # Low-rank factor: Φ = C W^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(W)
        eigvals = np.maximum(eigvals, self.jitter)
        self.W_sqrt_inv_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.Phi_train_ = C_train @ self.W_sqrt_inv_  # (N, M)

        # Woodbury: (Φ Φᵀ + σ²I)⁻¹ y = σ⁻²(y - Φ(Φᵀ Φ + σ²I)⁻¹ Φᵀ y)
        A = self.Phi_train_.T @ self.Phi_train_ + self.noise_var * np.eye(M)
        self.L_A_ = cholesky(A, lower=True)
        self.v_ = cho_solve((self.L_A_, True), self.Phi_train_.T @ y_train)

        self._X_train = X_train
        self._y_train = y_train

        return self

    def predict(self, X_test, return_var=True):
        """
        Predict at test points.

        Parameters
        ----------
        X_test : array (N_test, d)
        return_var : bool — whether to return predictive variance

        Returns
        -------
        mu : array (N_test,) — posterior mean
        var : array (N_test,) — posterior variance (if return_var=True)
        """
        C_test = rbf_kernel(X_test, self.Z_, self.lengthscale, self.variance)
        Phi_test = C_test @ self.W_sqrt_inv_

        mu = Phi_test @ self.v_

        if return_var:
            # var = k(x*,x*) - Φ_test (Φᵀ Φ + σ²I)⁻¹ Φ_testᵀ  [diagonal]
            B = cho_solve((self.L_A_, True), Phi_test.T)  # (M, N_test)
            var = self.variance - np.sum(Phi_test.T * B, axis=0) + self.noise_var
            var = np.maximum(var, self.jitter)
            return mu, var
        return mu

    def get_landmarks(self):
        """Return the fitted landmark points."""
        return self.Z_

    def get_shrinkage_info(self):
        """Return the effective rank and condition number."""
        if self.L_A_ is None:
            return {}
        A_diag = np.diag(self.L_A_)**2
        return {
            'M': len(self.Z_),
            'condition_number': A_diag.max() / A_diag.min(),
            'effective_rank': np.sum(A_diag > 0.01 * A_diag.max()),
        }
