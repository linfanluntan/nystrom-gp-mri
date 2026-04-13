"""
Hierarchical Multi-Subject GP with Nyström Acceleration.

Implements population-level GP estimation + per-subject shrinkage:
    μⱼ_hier = λⱼ · μⱼ_indep + (1 - λⱼ) · μ_pop
    λⱼ = nⱼ / (nⱼ + σ² / τ²)

This mirrors partial pooling in classical HBMs, operating at the
spatial-field level with Nyström-accelerated GP.
"""

import numpy as np
from .nystrom_gp import NystromGP
from .landmarks import select_landmarks
from .kernels import rbf_kernel


class HierarchicalNystromGP:
    """
    Multi-subject hierarchical GP with Nyström acceleration.

    Parameters
    ----------
    M : int — landmarks for Nyström
    lengthscale, variance, noise_var : float — GP hyperparameters
    """

    def __init__(self, M=80, lengthscale=0.15, variance=1.0, noise_var=0.3):
        self.M = M
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.pop_gp_ = None
        self.subject_gps_ = None

    def fit_predict(self, coords, subjects):
        """
        Fit hierarchical model and predict for all subjects.

        Parameters
        ----------
        coords : array (N, d) — full voxel grid
        subjects : list of dicts with keys:
            'obs_idx': array of observed voxel indices
            'y': array of observations

        Returns
        -------
        results : list of dicts per subject with keys:
            'mu_naive', 'mu_indep', 'mu_hier', 'var_indep', 'var_hier', 'lambda_j'
        """
        N = len(coords)
        J = len(subjects)
        Z = select_landmarks(coords, self.M, 'kmeans')

        # Step 1: Estimate population field from all data
        voxel_counts = np.zeros(N)
        voxel_sums = np.zeros(N)
        for s in subjects:
            for k, idx in enumerate(s['obs_idx']):
                voxel_counts[idx] += 1
                voxel_sums[idx] += s['y'][k]

        observed = voxel_counts > 0
        obs_coords = coords[observed]
        obs_means = voxel_sums[observed] / np.maximum(voxel_counts[observed], 1)
        obs_noise = self.noise_var / np.maximum(voxel_counts[observed], 1)

        pop_gp = NystromGP(M=self.M, lengthscale=self.lengthscale,
                           variance=self.variance, noise_var=np.mean(obs_noise))
        pop_gp.fit(obs_coords, obs_means)
        mu_pop, var_pop = pop_gp.predict(coords)
        self.pop_gp_ = pop_gp

        # Estimate between-subject variance (τ²)
        subject_means = [np.mean(s['y']) for s in subjects]
        tau2 = max(np.var(subject_means), 0.01)

        # Step 2: Per-subject predictions
        results = []
        self.subject_gps_ = []

        for j in range(J):
            X_j = coords[subjects[j]['obs_idx']]
            y_j = subjects[j]['y']
            n_j = len(y_j)

            # Naive: subject mean everywhere
            mu_naive = np.full(N, np.mean(y_j))

            # Independent GP
            gp_j = NystromGP(M=self.M, lengthscale=self.lengthscale,
                             variance=self.variance, noise_var=self.noise_var)
            gp_j.fit(X_j, y_j)
            mu_indep, var_indep = gp_j.predict(coords)
            self.subject_gps_.append(gp_j)

            # Shrinkage factor
            lambda_j = n_j / (n_j + self.noise_var / tau2)
            lambda_j = np.clip(lambda_j, 0.05, 0.99)

            # Hierarchical: blend
            mu_hier = lambda_j * mu_indep + (1 - lambda_j) * mu_pop
            var_hier = lambda_j**2 * var_indep + (1 - lambda_j)**2 * var_pop

            results.append({
                'mu_naive': mu_naive,
                'mu_indep': mu_indep,
                'mu_hier': mu_hier,
                'var_indep': var_indep,
                'var_hier': var_hier,
                'lambda_j': lambda_j,
                'n_j': n_j,
            })

        return results
