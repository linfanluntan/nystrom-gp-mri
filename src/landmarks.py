"""
Landmark (inducing point) selection strategies for Nyström GP.
"""

import numpy as np


def select_landmarks(X, M, method='kmeans', seed=42):
    """
    Select M landmark points from data X.

    Parameters
    ----------
    X : array (N, d)
    M : int — number of landmarks
    method : str — 'random', 'kmeans', or 'grid'
    seed : int

    Returns
    -------
    Z : array (M, d) — landmark coordinates
    """
    N = len(X)
    M = min(M, N - 1)

    if method == 'random':
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, M, replace=False)
        return X[idx].copy()

    elif method == 'kmeans':
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=M, random_state=seed, n_init=3)
        km.fit(X)
        return km.cluster_centers_

    elif method == 'grid':
        d = X.shape[1]
        m_per_dim = max(2, int(round(M ** (1.0 / d))))
        grids = [np.linspace(X[:, i].min(), X[:, i].max(), m_per_dim) for i in range(d)]
        mesh = np.meshgrid(*grids)
        landmarks = np.column_stack([m.ravel() for m in mesh])
        if len(landmarks) > M:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(landmarks), M, replace=False)
            landmarks = landmarks[idx]
        return landmarks

    else:
        raise ValueError(f"Unknown landmark method: {method}")
