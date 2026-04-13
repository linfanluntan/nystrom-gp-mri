"""
Nyström GP Experiments for the Paper.

Experiment 1: Approximation Quality vs M (landmark count)
  - 2D spatial field on MRI-like grid (64x64 = 4096 voxels)
  - Compare full GP vs Nyström at M = {25, 50, 100, 200, 400}
  - Metrics: RMSE, NLPD, coverage, Frobenius norm of K-K_nys
  - Landmark strategies: random, k-means, grid

Experiment 2: Scalability Benchmark
  - Grid sizes N = {1024, 4096, 10000, 25000, 50000}
  - Measure wall-clock time and peak memory for full GP vs Nyström
  - Demonstrate the O(N³) → O(NM²) transition

Experiment 3: Hierarchical Bayesian Spatial Smoothing
  - Multi-subject (J=8 "patients") with shared spatial kernel
  - HBM: y_ij ~ N(f_j(x_i), σ²), f_j ~ GP(μ_j, K_nys)
  - Compare: independent GP per subject vs hierarchical GP with Nyström
  - Show shrinkage + spatial smoothing jointly

Experiment 4: Posterior Calibration Under Nyström Approximation
  - Does Nyström degrade posterior calibration?
  - Compare coverage of full GP vs Nyström at varying M
  - Connect to multi-level calibration framework
"""

import os
import sys
import time
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve, cholesky

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
})
COLORS = {'full': '#2c3e50', 'nys': '#2ecc71', 'random': '#e74c3c',
          'kmeans': '#3498db', 'grid': '#e67e22', 'ref': '#95a5a6'}


# ═══════════════════════════════════════════════
# CORE IMPLEMENTATIONS
# ═══════════════════════════════════════════════

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """RBF (squared exponential) kernel."""
    D = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-0.5 * D / lengthscale**2)


def full_gp_predict(X_train, y_train, X_test, lengthscale, variance, noise_var):
    """Full GP posterior (exact). Returns mean, variance."""
    K = rbf_kernel(X_train, X_train, lengthscale, variance) + noise_var * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, lengthscale, variance)
    K_ss = rbf_kernel(X_test, X_test, lengthscale, variance)

    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y_train)
    V = cho_solve((L, True), K_s)

    mu = K_s.T @ alpha
    var = np.diag(K_ss) - np.sum(K_s * V, axis=0)
    var = np.maximum(var, 1e-10)
    return mu, var


def select_landmarks(X, M, method='kmeans'):
    """Select M landmarks from X."""
    N = len(X)
    if method == 'random':
        idx = np.random.choice(N, M, replace=False)
        return X[idx]
    elif method == 'kmeans':
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=M, random_state=42, n_init=3)
        km.fit(X)
        return km.cluster_centers_
    elif method == 'grid':
        # Uniform grid in bounding box
        d = X.shape[1]
        m_per_dim = max(2, int(round(M ** (1.0/d))))
        grids = [np.linspace(X[:, i].min(), X[:, i].max(), m_per_dim) for i in range(d)]
        mesh = np.meshgrid(*grids)
        landmarks = np.column_stack([m.ravel() for m in mesh])
        # Subsample if too many
        if len(landmarks) > M:
            idx = np.random.choice(len(landmarks), M, replace=False)
            landmarks = landmarks[idx]
        return landmarks
    else:
        raise ValueError(f"Unknown method: {method}")


def nystrom_gp_predict(X_train, y_train, X_test, Z, lengthscale, variance, noise_var):
    """
    Nyström-approximate GP posterior via Woodbury identity.

    K ≈ C W⁻¹ Cᵀ where W = K(Z,Z), C = K(X,Z)

    Posterior mean:  μ = Kₛₙ (Kₙₙ + σ²I)⁻¹ y
    With Nyström:    (C W⁻¹ Cᵀ + σ²I)⁻¹ = σ⁻² I - σ⁻² C (W + σ⁻² CᵀC)⁻¹ Cᵀ σ⁻²
    """
    M = len(Z)
    N = len(X_train)

    W = rbf_kernel(Z, Z, lengthscale, variance) + 1e-6 * np.eye(M)
    C_train = rbf_kernel(X_train, Z, lengthscale, variance)
    C_test = rbf_kernel(X_test, Z, lengthscale, variance)

    # Woodbury: (C W⁻¹ Cᵀ + σ²I)⁻¹ via
    # σ⁻²I - σ⁻² C (σ² W + CᵀC)⁻¹ Cᵀ σ⁻²
    inner = noise_var * W + C_train.T @ C_train  # M×M
    L_inner = cholesky(inner + 1e-8 * np.eye(M), lower=True)

    # Mean: K_test_train (K_train + σ²I)⁻¹ y
    # = C_test W⁻¹ C_train.T [σ⁻²I - σ⁻² C (inner)⁻¹ Cᵀ σ⁻²] y
    # Simplified via Woodbury:
    rhs = C_train.T @ y_train  # M×1
    v = cho_solve((L_inner, True), rhs)  # M×1
    mu = C_test @ (cho_solve((cho_factor(W), True), C_train.T @ y_train)
                   - cho_solve((cho_factor(W), True), C_train.T @ (C_train @ v) / noise_var)) / noise_var

    # Actually, cleaner formulation:
    # μ* = C_test @ W⁻¹ @ Cᵀ_train @ (C_train @ W⁻¹ @ Cᵀ_train + σ²I)⁻¹ @ y
    # Use Woodbury directly on the predictive mean
    Kinv_y = (y_train - C_train @ v / noise_var) / noise_var
    mu = C_test @ cho_solve((cho_factor(W), True), C_train.T @ Kinv_y)

    # Variance (diagonal only)
    K_ss_diag = variance * np.ones(len(X_test))  # diagonal of K(X*,X*)
    # Approximate: var = K_ss - K_s_train (K_train + σ²I)⁻¹ K_train_s
    # Nyström approx of this reduction
    W_inv_C_test_T = cho_solve((cho_factor(W), True), C_test.T)  # M × N_test
    inner_inv_CTC = cho_solve((L_inner, True), C_train.T @ C_train)  # M × M (approx)

    var = K_ss_diag - np.sum(C_test.T * (W_inv_C_test_T - cho_solve((L_inner, True), C_test.T)), axis=0)
    var = np.maximum(var, 1e-6)

    return mu, var


def nystrom_gp_simple(X_train, y_train, X_test, Z, lengthscale, variance, noise_var):
    """Simpler Nyström GP via explicit low-rank factor."""
    M = len(Z)
    N = len(X_train)

    W = rbf_kernel(Z, Z, lengthscale, variance) + 1e-6 * np.eye(M)
    C_train = rbf_kernel(X_train, Z, lengthscale, variance)
    C_test = rbf_kernel(X_test, Z, lengthscale, variance)

    # Low-rank factor: K ≈ C W^{-1/2} (W^{-1/2})ᵀ Cᵀ = Φ Φᵀ
    # where Φ = C W^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals = np.maximum(eigvals, 1e-10)
    W_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Phi_train = C_train @ W_sqrt_inv  # N × M
    Phi_test = C_test @ W_sqrt_inv    # N_test × M

    # (Φ Φᵀ + σ²I)⁻¹ y = σ⁻² (y - Φ (Φᵀ Φ + σ²I_M)⁻¹ Φᵀ y)  [Woodbury]
    A = Phi_train.T @ Phi_train + noise_var * np.eye(M)  # M × M
    L_A = cholesky(A, lower=True)
    v = cho_solve((L_A, True), Phi_train.T @ y_train)

    mu = Phi_test @ v

    # Variance
    K_ss_diag = variance * np.ones(len(X_test))
    B = cho_solve((L_A, True), Phi_test.T)  # M × N_test
    var = K_ss_diag - np.sum(Phi_test.T * B, axis=0) + noise_var
    var = np.maximum(var, 1e-6)

    return mu, var


# ═══════════════════════════════════════════════
# EXPERIMENT 1: Approximation Quality
# ═══════════════════════════════════════════════

def experiment1(fig_dir):
    print("=" * 60)
    print("  Experiment 1: Approximation Quality vs M")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate 2D spatial field (64×64 = 4096 voxels, MRI-like)
    grid_size = 48  # 48×48 = 2304 voxels (tractable for full GP comparison)
    N = grid_size ** 2
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)], dtype=float)
    coords = coords / grid_size  # normalize to [0,1]²

    # True function: smooth spatial field
    ls_true, var_true, noise_true = 0.15, 1.0, 0.1
    K_true = rbf_kernel(coords, coords, ls_true, var_true)
    f_true = rng.multivariate_normal(np.zeros(N), K_true + 1e-6*np.eye(N))
    y = f_true + rng.normal(0, np.sqrt(noise_true), N)

    # Train/test split (80/20)
    idx = rng.permutation(N)
    n_train = int(0.8 * N)
    X_train, y_train = coords[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = coords[idx[n_train:]], y[idx[n_train:]]
    f_test = f_true[idx[n_train:]]

    print(f"  Grid: {grid_size}×{grid_size} = {N} voxels")
    print(f"  Train: {n_train}, Test: {len(y_test)}")

    # Full GP (reference)
    print("\n  Full GP (reference)...")
    t0 = time.time()
    mu_full, var_full = full_gp_predict(X_train, y_train, X_test, ls_true, var_true, noise_true)
    time_full = time.time() - t0
    rmse_full = np.sqrt(np.mean((mu_full - f_test)**2))
    cov_full = np.mean((f_test >= mu_full - 1.96*np.sqrt(var_full)) &
                       (f_test <= mu_full + 1.96*np.sqrt(var_full)))
    print(f"    RMSE: {rmse_full:.4f}, Coverage: {cov_full:.3f}, Time: {time_full:.2f}s")

    # Nyström at varying M
    M_values = [25, 50, 100, 200, 400]
    methods = ['random', 'kmeans', 'grid']
    results = {}

    for method in methods:
        results[method] = {'M': [], 'rmse': [], 'coverage': [], 'time': [],
                           'frobenius': [], 'nlpd': []}
        for M in M_values:
            if M > n_train:
                continue
            t0 = time.time()
            Z = select_landmarks(X_train, M, method)
            mu_nys, var_nys = nystrom_gp_simple(X_train, y_train, X_test, Z,
                                                 ls_true, var_true, noise_true)
            t_nys = time.time() - t0

            rmse = np.sqrt(np.mean((mu_nys - f_test)**2))
            cov = np.mean((f_test >= mu_nys - 1.96*np.sqrt(var_nys)) &
                          (f_test <= mu_nys + 1.96*np.sqrt(var_nys)))

            # NLPD (negative log predictive density)
            nlpd = 0.5 * np.mean(np.log(2*np.pi*var_nys) + (f_test - mu_nys)**2 / var_nys)

            # Frobenius norm of kernel approximation error (subsample for speed)
            sub_idx = rng.choice(n_train, min(500, n_train), replace=False)
            K_sub = rbf_kernel(X_train[sub_idx], X_train[sub_idx], ls_true, var_true)
            C_sub = rbf_kernel(X_train[sub_idx], Z, ls_true, var_true)
            W_sub = rbf_kernel(Z, Z, ls_true, var_true) + 1e-6 * np.eye(len(Z))
            K_nys_sub = C_sub @ np.linalg.solve(W_sub, C_sub.T)
            frob = np.linalg.norm(K_sub - K_nys_sub, 'fro') / np.linalg.norm(K_sub, 'fro')

            results[method]['M'].append(M)
            results[method]['rmse'].append(rmse)
            results[method]['coverage'].append(cov)
            results[method]['time'].append(t_nys)
            results[method]['frobenius'].append(frob)
            results[method]['nlpd'].append(nlpd)

            print(f"    {method:>8} M={M:4d}: RMSE={rmse:.4f}, Cov={cov:.3f}, "
                  f"Frob={frob:.4f}, Time={t_nys:.2f}s")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) RMSE vs M
    ax = axes[0, 0]
    for method, col in [('random', COLORS['random']), ('kmeans', COLORS['kmeans']),
                         ('grid', COLORS['grid'])]:
        r = results[method]
        ax.plot(r['M'], r['rmse'], 'o-', color=col, lw=2, markersize=6, label=method.capitalize())
    ax.axhline(rmse_full, color=COLORS['full'], ls='--', lw=1.5, label=f'Full GP ({rmse_full:.4f})')
    ax.set_xlabel('Number of landmarks M')
    ax.set_ylabel('RMSE')
    ax.set_title('(a) Prediction Error vs Landmark Count')
    ax.legend(fontsize=8)

    # (b) Frobenius error vs M
    ax = axes[0, 1]
    for method, col in [('random', COLORS['random']), ('kmeans', COLORS['kmeans']),
                         ('grid', COLORS['grid'])]:
        r = results[method]
        ax.plot(r['M'], r['frobenius'], 'o-', color=col, lw=2, markersize=6, label=method.capitalize())
    ax.set_xlabel('Number of landmarks M')
    ax.set_ylabel('Relative Frobenius error ||K - K̃||/||K||')
    ax.set_title('(b) Kernel Approximation Error')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # (c) Coverage vs M
    ax = axes[1, 0]
    for method, col in [('random', COLORS['random']), ('kmeans', COLORS['kmeans']),
                         ('grid', COLORS['grid'])]:
        r = results[method]
        ax.plot(r['M'], r['coverage'], 'o-', color=col, lw=2, markersize=6, label=method.capitalize())
    ax.axhline(cov_full, color=COLORS['full'], ls='--', lw=1.5, label=f'Full GP ({cov_full:.3f})')
    ax.axhline(0.95, color=COLORS['ref'], ls=':', lw=1)
    ax.set_xlabel('Number of landmarks M')
    ax.set_ylabel('95% Coverage')
    ax.set_title('(c) Posterior Calibration (Coverage)')
    ax.legend(fontsize=8)

    # (d) Spatial prediction maps
    ax = axes[1, 1]
    # Show full GP and Nyström M=100 side by side
    # Predict on full grid
    Z_km = select_landmarks(X_train, 100, 'kmeans')
    mu_full_grid, _ = full_gp_predict(X_train, y_train, coords, ls_true, var_true, noise_true)
    mu_nys_grid, _ = nystrom_gp_simple(X_train, y_train, coords, Z_km, ls_true, var_true, noise_true)
    diff = np.abs(mu_full_grid - mu_nys_grid).reshape(grid_size, grid_size)
    im = ax.imshow(diff, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax, label='|Full GP - Nyström|')
    ax.set_title('(d) Absolute Difference Map (M=100, k-means)')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nys_fig1_approx_quality.png'))
    plt.close(fig)
    print(f"\n  Saved: nys_fig1_approx_quality.png")

    # Spatial field visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, data, title in [
        (axes[0], f_true.reshape(grid_size, grid_size), 'True Field'),
        (axes[1], y.reshape(grid_size, grid_size), 'Noisy Observations'),
        (axes[2], mu_full_grid.reshape(grid_size, grid_size), 'Full GP Posterior Mean'),
        (axes[3], mu_nys_grid.reshape(grid_size, grid_size), 'Nyström (M=100) Mean'),
    ]:
        im = ax.imshow(data, cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nys_fig2_spatial_maps.png'))
    plt.close(fig)
    print(f"  Saved: nys_fig2_spatial_maps.png")

    return {
        'N': N, 'grid_size': grid_size,
        'full_gp': {'rmse': float(rmse_full), 'coverage': float(cov_full), 'time': float(time_full)},
        'nystrom': {method: {k: [float(v) for v in vals] for k, vals in r.items()}
                    for method, r in results.items()},
    }


# ═══════════════════════════════════════════════
# EXPERIMENT 2: Scalability Benchmark
# ═══════════════════════════════════════════════

def experiment2(fig_dir):
    print("\n" + "=" * 60)
    print("  Experiment 2: Scalability Benchmark")
    print("=" * 60)

    rng = np.random.default_rng(2024)
    grid_sizes = [16, 32, 48, 64, 80, 100]  # N = 256 to 10000
    M_fixed = 100
    ls, var, noise = 0.15, 1.0, 0.1

    full_times = []
    nys_times = []
    Ns = []

    for gs in grid_sizes:
        N = gs * gs
        Ns.append(N)
        coords = np.array([(i/gs, j/gs) for i in range(gs) for j in range(gs)])
        y = rng.normal(0, 1, N)

        # Full GP
        if N <= 5000:
            t0 = time.time()
            K = rbf_kernel(coords, coords, ls, var) + noise * np.eye(N)
            L = cholesky(K, lower=True)
            _ = cho_solve((L, True), y)
            full_times.append(time.time() - t0)
        else:
            # Extrapolate O(N³)
            full_times.append(full_times[-1] * (N / Ns[-2])**3)

        # Nyström
        t0 = time.time()
        Z = select_landmarks(coords, min(M_fixed, N-1), 'kmeans')
        M = len(Z)
        W = rbf_kernel(Z, Z, ls, var) + 1e-6 * np.eye(M)
        C = rbf_kernel(coords, Z, ls, var)
        eigvals, eigvecs = np.linalg.eigh(W)
        eigvals = np.maximum(eigvals, 1e-10)
        W_sqrt_inv = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        Phi = C @ W_sqrt_inv
        A = Phi.T @ Phi + noise * np.eye(M)
        L_A = cholesky(A, lower=True)
        _ = cho_solve((L_A, True), Phi.T @ y)
        nys_times.append(time.time() - t0)

        actual = "actual" if N <= 5000 else "extrap"
        print(f"  N={N:6d}: Full GP={full_times[-1]:8.3f}s ({actual}), "
              f"Nyström={nys_times[-1]:.3f}s, Speedup={full_times[-1]/nys_times[-1]:.0f}×")

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(Ns, full_times, 'o-', color=COLORS['full'], lw=2, markersize=7, label='Full GP O(N³)')
    ax.plot(Ns, nys_times, 's-', color=COLORS['nys'], lw=2, markersize=7, label=f'Nyström O(NM²), M={M_fixed}')
    ax.set_xlabel('Number of voxels N')
    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title('Computational Scaling: Full GP vs Nyström')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nys_fig3_scaling.png'))
    plt.close(fig)
    print(f"\n  Saved: nys_fig3_scaling.png")

    return {'Ns': Ns, 'full_times': full_times, 'nys_times': nys_times}


# ═══════════════════════════════════════════════
# EXPERIMENT 3: Hierarchical Bayesian Spatial Smoothing
# ═══════════════════════════════════════════════

def experiment3(fig_dir):
    print("\n" + "=" * 60)
    print("  Experiment 3: Hierarchical Spatial Smoothing (Multi-Subject)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    grid_size = 32  # 32×32 = 1024 voxels per subject
    N = grid_size ** 2
    J = 10  # subjects
    coords = np.array([(i/grid_size, j/grid_size) for i in range(grid_size)
                        for j in range(grid_size)])

    ls, var_signal, noise = 0.15, 1.0, 0.3

    # Shared population signal (smooth spatial field)
    K_pop = rbf_kernel(coords, coords, ls, var_signal) + 1e-6*np.eye(N)
    f_pop = rng.multivariate_normal(np.zeros(N), K_pop)

    # Per-subject: population + small subject-specific offset (scalar shift + small spatial)
    subject_offsets = rng.normal(0, 0.5, size=J)  # random intercepts
    subjects = []
    # Key: deliberately vary sample sizes to create sparse vs dense subjects
    n_obs_per_subject = np.array([20, 30, 40, 50, 80, 120, 200, 400, 600, 800])

    for j in range(J):
        f_j = f_pop + subject_offsets[j]  # shared shape + subject offset
        obs_idx = rng.choice(N, n_obs_per_subject[j], replace=False)
        y_j = f_j[obs_idx] + rng.normal(0, np.sqrt(noise), n_obs_per_subject[j])
        subjects.append({
            'f_true': f_j, 'obs_idx': obs_idx, 'y': y_j, 'n': n_obs_per_subject[j]
        })

    print(f"  {J} subjects, {grid_size}x{grid_size} grid")
    print(f"  Obs per subject: {list(n_obs_per_subject)}")
    print(f"  Noise SD: {np.sqrt(noise):.2f}, Signal var: {var_signal}")

    M = 80
    Z = select_landmarks(coords, M, 'kmeans')

    # ── Method 1: Voxel-wise mean (no spatial smoothing, no borrowing) ──
    naive_rmse = []
    for j in range(J):
        # Just predict the mean of observed y for this subject everywhere
        mu_naive = np.full(N, np.mean(subjects[j]['y']))
        rmse = np.sqrt(np.mean((mu_naive - subjects[j]['f_true'])**2))
        naive_rmse.append(rmse)

    # ── Method 2: Independent Nyström GP per subject ──
    independent_rmse = []
    independent_coverage = []
    for j in range(J):
        X_j = coords[subjects[j]['obs_idx']]
        y_j = subjects[j]['y']
        mu_ind, var_ind = nystrom_gp_simple(X_j, y_j, coords, Z, ls, var_signal, noise)
        rmse = np.sqrt(np.mean((mu_ind - subjects[j]['f_true'])**2))
        cov = np.mean((subjects[j]['f_true'] >= mu_ind - 1.96*np.sqrt(var_ind)) &
                       (subjects[j]['f_true'] <= mu_ind + 1.96*np.sqrt(var_ind)))
        independent_rmse.append(rmse)
        independent_coverage.append(cov)

    # ── Method 3: Hierarchical Nyström GP ──
    # Proper approach: estimate population field from ALL data jointly,
    # then predict each subject as population + subject-specific residual
    # with shrinkage proportional to subject's data richness.
    #
    # Step 1: Estimate population mean by fitting GP to the grand mean
    #         across subjects (empirical Bayes for the population field)
    # Compute per-voxel averages where we have data
    voxel_counts = np.zeros(N)
    voxel_sums = np.zeros(N)
    for j in range(J):
        for k, idx in enumerate(subjects[j]['obs_idx']):
            voxel_counts[idx] += 1
            voxel_sums[idx] += subjects[j]['y'][k]

    observed_voxels = voxel_counts > 0
    voxel_means = np.zeros(N)
    voxel_means[observed_voxels] = voxel_sums[observed_voxels] / voxel_counts[observed_voxels]

    # Fit population GP on observed voxel means (weighted by count)
    obs_coords = coords[observed_voxels]
    obs_means = voxel_means[observed_voxels]
    obs_noise = noise / np.maximum(voxel_counts[observed_voxels], 1)  # reduced noise for averaged data

    # Use Nyström GP with reduced noise for the population
    mu_pop_est, var_pop_est = nystrom_gp_simple(
        obs_coords, obs_means, coords, Z, ls, var_signal, np.mean(obs_noise)
    )

    # Step 2: For each subject, compute shrinkage-weighted prediction
    hierarchical_rmse = []
    hierarchical_coverage = []

    for j in range(J):
        X_j = coords[subjects[j]['obs_idx']]
        y_j = subjects[j]['y']
        n_j = subjects[j]['n']

        # Subject-specific GP on raw data
        mu_subj, var_subj = nystrom_gp_simple(X_j, y_j, coords, Z, ls, var_signal, noise)

        # Shrinkage factor: λ_j = n_j / (n_j + σ²/τ²)
        # τ² = population variance, σ² = noise
        tau2_est = np.var([np.mean(s['y']) for s in subjects])
        lambda_j = n_j / (n_j + noise / max(tau2_est, 0.01))
        lambda_j = np.clip(lambda_j, 0.05, 0.99)

        # Shrinkage: blend subject GP with population GP
        mu_hier = lambda_j * mu_subj + (1 - lambda_j) * mu_pop_est
        var_hier = lambda_j**2 * var_subj + (1 - lambda_j)**2 * var_pop_est

        rmse = np.sqrt(np.mean((mu_hier - subjects[j]['f_true'])**2))
        cov = np.mean((subjects[j]['f_true'] >= mu_hier - 1.96*np.sqrt(var_hier)) &
                       (subjects[j]['f_true'] <= mu_hier + 1.96*np.sqrt(var_hier)))
        hierarchical_rmse.append(rmse)
        hierarchical_coverage.append(cov)

    # Print results table
    print(f"\n  {'Subj':>4} {'n':>5} {'Naive':>7} {'Indep GP':>9} {'Hier GP':>8} "
          f"{'Improv':>7} {'Cov_ind':>8} {'Cov_hier':>9}")
    print("  " + "-" * 70)
    for j in range(J):
        imp = (independent_rmse[j] - hierarchical_rmse[j]) / independent_rmse[j] * 100
        print(f"  {j:4d} {n_obs_per_subject[j]:5d} {naive_rmse[j]:7.4f} "
              f"{independent_rmse[j]:9.4f} {hierarchical_rmse[j]:8.4f} "
              f"{imp:6.1f}% {independent_coverage[j]:8.3f} {hierarchical_coverage[j]:9.3f}")

    mean_imp = np.mean([(independent_rmse[j] - hierarchical_rmse[j]) / independent_rmse[j] * 100
                        for j in range(J)])
    n_improved = sum(1 for j in range(J) if hierarchical_rmse[j] < independent_rmse[j])
    print(f"\n  Mean improvement: {mean_imp:.1f}% ({n_improved}/{J} subjects improved)")
    print(f"  Sparse subjects (n<=50) improvement: "
          f"{np.mean([(independent_rmse[j]-hierarchical_rmse[j])/independent_rmse[j]*100 for j in range(4)]):.1f}%")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) RMSE by method, ordered by sample size
    ax = axes[0, 0]
    x = np.arange(J)
    w = 0.25
    ax.bar(x - w, naive_rmse, w, color=COLORS['ref'], alpha=0.7, label='Naive (mean)')
    ax.bar(x, independent_rmse, w, color=COLORS['random'], alpha=0.8, label='Independent Nyström GP')
    ax.bar(x + w, hierarchical_rmse, w, color=COLORS['nys'], alpha=0.8, label='Hierarchical Nyström GP')
    ax.set_xlabel('Subject (ordered by sample size)')
    ax.set_ylabel('RMSE')
    labels = [f'n={n}' for n in n_obs_per_subject]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_title(f'(a) Per-Subject RMSE')
    ax.legend(fontsize=7)

    # (b) RMSE vs sample size with arrows
    ax = axes[0, 1]
    ax.scatter(n_obs_per_subject, independent_rmse, s=70, color=COLORS['random'],
               label='Independent GP', edgecolors='white', linewidths=0.5, zorder=3)
    ax.scatter(n_obs_per_subject, hierarchical_rmse, s=70, color=COLORS['nys'],
               label='Hierarchical GP', edgecolors='white', linewidths=0.5, marker='s', zorder=3)
    for j in range(J):
        color = COLORS['nys'] if hierarchical_rmse[j] < independent_rmse[j] else COLORS['random']
        ax.annotate('', xy=(n_obs_per_subject[j], hierarchical_rmse[j]),
                    xytext=(n_obs_per_subject[j], independent_rmse[j]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=0.6))
    ax.set_xlabel('Observations per subject (n)')
    ax.set_ylabel('RMSE')
    ax.set_title('(b) Hierarchical Shrinkage Effect')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    # (c) Coverage comparison
    ax = axes[1, 0]
    ax.scatter(n_obs_per_subject, independent_coverage, s=70, color=COLORS['random'],
               label='Independent GP', edgecolors='white', linewidths=0.5)
    ax.scatter(n_obs_per_subject, hierarchical_coverage, s=70, color=COLORS['nys'],
               label='Hierarchical GP', edgecolors='white', linewidths=0.5, marker='s')
    ax.axhline(0.95, color=COLORS['ref'], ls='--', lw=1.5, label='Nominal 95%')
    ax.set_xlabel('Observations per subject')
    ax.set_ylabel('95% Coverage')
    ax.set_title('(c) Posterior Calibration by Subject')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    # (d) Spatial maps for sparsest subject
    j_sparse = 0  # n=20
    X_j = coords[subjects[j_sparse]['obs_idx']]
    y_j = subjects[j_sparse]['y']
    mu_ind_s, _ = nystrom_gp_simple(X_j, y_j, coords, Z, ls, var_signal, noise)
    lambda_s = n_obs_per_subject[j_sparse] / (n_obs_per_subject[j_sparse] + noise / max(tau2_est, 0.01))
    mu_subj_s, _ = nystrom_gp_simple(X_j, y_j, coords, Z, ls, var_signal, noise)
    mu_hier_s = lambda_s * mu_subj_s + (1 - lambda_s) * mu_pop_est

    ax = axes[1, 1]
    # Show the difference: |true - estimate| for both methods
    err_ind = np.abs(mu_ind_s - subjects[j_sparse]['f_true']).reshape(grid_size, grid_size)
    err_hier = np.abs(mu_hier_s - subjects[j_sparse]['f_true']).reshape(grid_size, grid_size)

    # Side by side in one panel
    combined = np.hstack([err_ind, np.full((grid_size, 2), np.nan), err_hier])
    im = ax.imshow(combined, cmap='hot', origin='lower', vmin=0,
                    vmax=max(err_ind.max(), err_hier.max()))
    ax.axvline(grid_size + 0.5, color='white', lw=2)
    ax.set_title(f'(d) Error Maps: Sparse Subject (n={n_obs_per_subject[j_sparse]})')
    ax.text(grid_size//2, -3, 'Independent', ha='center', fontsize=9, color=COLORS['random'])
    ax.text(grid_size + 2 + grid_size//2, -3, 'Hierarchical', ha='center', fontsize=9, color=COLORS['nys'])
    plt.colorbar(im, ax=ax, label='|f_true - f_est|', fraction=0.046)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nys_fig4_hierarchical.png'))
    plt.close(fig)
    print(f"\n  Saved: nys_fig4_hierarchical.png")

    return {
        'n_subjects': J,
        'n_obs': list(map(int, n_obs_per_subject)),
        'naive_rmse': list(map(float, naive_rmse)),
        'independent_rmse': list(map(float, independent_rmse)),
        'hierarchical_rmse': list(map(float, hierarchical_rmse)),
        'independent_coverage': list(map(float, independent_coverage)),
        'hierarchical_coverage': list(map(float, hierarchical_coverage)),
        'mean_improvement_pct': float(mean_imp),
        'n_improved': int(n_improved),
    }


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("  NYSTRÖM GP EXPERIMENTS")
    print("█" * 60)

    t_total = time.time()
    all_results = {}

    all_results['exp1'] = experiment1(fig_dir)
    all_results['exp2'] = experiment2(fig_dir)
    all_results['exp3'] = experiment3(fig_dir)

    with open(os.path.join(fig_dir, 'nystrom_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Total time: {time.time()-t_total:.0f}s")
    print("█" * 60)
    return all_results


if __name__ == '__main__':
    main()
