"""
Generate publication-quality replacement figures for the Nyström paper.
Replaces ugly Figures 1 and 7, and adds an experiment design overview.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.spatial.distance import cdist
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FIG_DIR = '/home/claude/hbm-calibration/figures'


def fig_nystrom_workflow():
    """Figure 1 replacement: Clean Nyström workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Color scheme
    c_input = '#3498db'
    c_nys = '#2ecc71'
    c_output = '#e67e22'
    c_hier = '#9b59b6'
    c_arrow = '#2c3e50'

    def box(x, y, w, h, text, color, fontsize=10, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor='white',
                               linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color='white', weight=weight,
                multialignment='center')

    def arrow(x1, y1, x2, y2, text='', color=c_arrow):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.2, text, ha='center', va='bottom',
                    fontsize=8, color=color, style='italic')

    # Title
    ax.text(6, 5.7, 'Nyström-Accelerated Hierarchical Bayesian MRI Workflow',
            ha='center', va='center', fontsize=14, weight='bold', color=c_arrow)

    # Row 1: Input
    box(0.2, 4.2, 2.2, 1.0, 'MRI Voxel Data\nN = 10⁴–10⁶ voxels', c_input, 10, True)

    # Row 1: Landmark selection
    box(3.2, 4.2, 2.2, 1.0, 'Landmark Selection\nM ≪ N points\n(k-means / grid)', c_nys, 9)

    # Row 1: Nyström blocks
    box(6.2, 4.2, 2.8, 1.0, 'Compute Nyström Blocks\nW = K(Z,Z)  [M×M]\nC = K(X,Z)  [N×M]', c_nys, 9)

    # Row 1: Low-rank factor
    box(9.7, 4.2, 2.1, 1.0, 'Low-Rank Factor\nΦ = C W⁻¹ᐟ²\nK ≈ ΦΦᵀ', c_nys, 9)

    arrow(2.4, 4.7, 3.2, 4.7)
    arrow(5.4, 4.7, 6.2, 4.7)
    arrow(9.0, 4.7, 9.7, 4.7)

    # Row 2: Three axes
    box(0.2, 2.4, 3.0, 1.2, 'Spatial Covariance\n• Voxel smoothing\n• CAR/SPDE replacement\n• O(NM²) vs O(N³)', '#2980b9', 9)
    box(3.8, 2.4, 3.2, 1.2, 'Temporal Covariance\n• DCE/DWI timecourses\n• Functional dim. reduction\n• Curve-shape uncertainty', '#27ae60', 9)
    box(7.6, 2.4, 3.6, 1.2, 'Multimodal Covariance\n• Joint ADC–DCE modeling\n• Cross-modal correlations\n• Block kernel approximation', '#d35400', 9)

    arrow(1.7, 4.2, 1.7, 3.6, 'spatial')
    arrow(5.4, 4.2, 5.4, 3.6, 'temporal')
    arrow(9.4, 4.2, 9.4, 3.6, 'multimodal')

    # Row 3: Hierarchical inference
    box(1.5, 0.5, 3.5, 1.4, 'Hierarchical Bayesian Inference\n• Population GP (shared structure)\n• Subject-specific deviations\n• Shrinkage ∝ 1/data richness', c_hier, 9)

    box(6.5, 0.5, 4.5, 1.4, 'Calibrated Outputs\n• Posterior mean maps (per voxel)\n• Uncertainty maps (per voxel)\n• Calibrated credible intervals\n• Multi-level coverage verification', c_output, 9)

    arrow(3.0, 2.4, 3.0, 1.9)
    arrow(5.0, 2.4, 5.0, 1.5, '', c_hier)
    arrow(5.0, 1.2, 6.5, 1.2)

    # Memory/compute callout
    ax.text(11.5, 3.5, 'Memory:\n80 GB → 240 MB\n(330× reduction)\n\nCompute:\n10¹⁵ → 10¹⁰ FLOP\n(100,000× reduction)',
            ha='center', va='center', fontsize=8, color=c_nys,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=c_nys, linewidth=1.5),
            multialignment='center')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'nys_fig_workflow.png'), facecolor='white')
    plt.close(fig)
    print("Saved: nys_fig_workflow.png")


def fig_kernel_approximation():
    """Figure replacement for kernel visualization: actual computed matrices."""
    rng = np.random.default_rng(42)

    # Small grid for visualization
    N = 200
    coords = rng.uniform(0, 1, (N, 2))
    coords = coords[np.argsort(coords[:, 0])]  # sort for visual structure

    ls, var = 0.15, 1.0
    D = cdist(coords, coords, 'sqeuclidean')
    K_full = var * np.exp(-0.5 * D / ls**2)

    # Landmarks
    M_values = [10, 30, 80]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5),
                              gridspec_kw={'height_ratios': [1, 0.4]})

    # Top row: kernel matrices
    vmin, vmax = K_full.min(), K_full.max()

    im = axes[0, 0].imshow(K_full, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'Full Kernel K\n(N={N}, {N}×{N})', fontsize=11, weight='bold')
    axes[0, 0].set_xlabel('Voxel index')
    axes[0, 0].set_ylabel('Voxel index')

    for i, M in enumerate(M_values):
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=M, random_state=42, n_init=3)
        km.fit(coords)
        Z = km.cluster_centers_

        W = var * np.exp(-0.5 * cdist(Z, Z, 'sqeuclidean') / ls**2) + 1e-6 * np.eye(M)
        C = var * np.exp(-0.5 * cdist(coords, Z, 'sqeuclidean') / ls**2)
        K_nys = C @ np.linalg.solve(W, C.T)

        axes[0, i+1].imshow(K_nys, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        frob = np.linalg.norm(K_full - K_nys, 'fro') / np.linalg.norm(K_full, 'fro')
        axes[0, i+1].set_title(f'Nyström (M={M})\nFrob. error: {frob:.1%}',
                                fontsize=11, weight='bold')
        axes[0, i+1].set_xlabel('Voxel index')

    # Bottom row: error matrices (difference)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Reference\n(exact)', ha='center', va='center',
                     fontsize=12, style='italic', transform=axes[1, 0].transAxes)

    for i, M in enumerate(M_values):
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=M, random_state=42, n_init=3)
        km.fit(coords)
        Z = km.cluster_centers_
        W = var * np.exp(-0.5 * cdist(Z, Z, 'sqeuclidean') / ls**2) + 1e-6 * np.eye(M)
        C = var * np.exp(-0.5 * cdist(coords, Z, 'sqeuclidean') / ls**2)
        K_nys = C @ np.linalg.solve(W, C.T)
        err = np.abs(K_full - K_nys)

        im_err = axes[1, i+1].imshow(err, cmap='hot', aspect='auto', vmin=0, vmax=0.3)
        axes[1, i+1].set_title(f'|K − K̃| (M={M})', fontsize=10)
        axes[1, i+1].set_xlabel('Voxel index')

    plt.colorbar(im_err, ax=axes[1, -1], label='Absolute error', fraction=0.05)

    fig.suptitle('Nyström Kernel Approximation: Convergence with Increasing M',
                 fontsize=14, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'nys_fig_kernel_approx.png'), facecolor='white')
    plt.close(fig)
    print("Saved: nys_fig_kernel_approx.png")


def fig_experiment_design():
    """Experiment design overview figure showing inputs, methods, metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    c1 = '#3498db'
    c2 = '#2ecc71'
    c3 = '#9b59b6'
    cb = '#ecf0f1'

    for ax, title, color, content in [
        (axes[0], 'Experiment 1\nApproximation Quality', c1,
         ['INPUT', '48×48 voxel grid (N=2,304)', 'RBF kernel (ℓ=0.15, σ²=1.0)',
          'Noise σ²ₙ=0.1, 80/20 split', '',
          'METHODS', 'Full GP (exact reference)',
          'Nyström: M ∈ {25,50,100,200,400}',
          'Landmarks: random / k-means / grid', '',
          'METRICS', '• RMSE vs true latent field',
          '• ||K−K̃||_F / ||K||_F  (kernel error)',
          '• 95% posterior coverage',
          '• Wall-clock time']),
        (axes[1], 'Experiment 2\nScalability Benchmark', c2,
         ['INPUT', 'Grid sizes: N ∈ {256..10,000}',
          'Fixed M=100 (k-means)', 'Same kernel parameters', '',
          'METHODS', 'Full GP: Cholesky solve',
          'Nyström: Woodbury solve',
          'N>4096: Full GP extrapolated O(N³)', '',
          'METRICS', '• Wall-clock time (seconds)',
          '• Speedup ratio (Full/Nyström)',
          '• Crossover point identification',
          '• Log-log scaling verification']),
        (axes[2], 'Experiment 3\nHierarchical Multi-Subject', c3,
         ['INPUT', 'J=10 subjects, 32×32 grid (N=1,024)',
          'Shared population field f_pop',
          'Subject offsets δⱼ ~ N(0, 0.25)',
          'n ∈ {20,30,...,800} obs/subject', '',
          'METHODS', '(a) Naive: subject mean everywhere',
          '(b) Independent Nyström GP / subject',
          '(c) Hierarchical: population GP +',
          '    shrinkage-weighted subject GP', '',
          'METRICS', '• Per-subject RMSE',
          '• Improvement vs independent (%)',
          '• 95% coverage per subject',
          '• RMSE vs sample size (shrinkage)']),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title box
        rect = FancyBboxPatch((0.02, 0.88), 0.96, 0.1, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.5, 0.93, title, ha='center', va='center', fontsize=12,
                color='white', weight='bold', multialignment='center')

        # Content
        y = 0.85
        for line in content:
            if line == '':
                y -= 0.015
                continue
            if line in ('INPUT', 'METHODS', 'METRICS'):
                ax.text(0.05, y, line, fontsize=9, weight='bold', color=color)
                y -= 0.005
                ax.plot([0.05, 0.95], [y, y], '-', color=color, lw=0.8, alpha=0.5)
            else:
                ax.text(0.08, y, line, fontsize=8.5, color='#2c3e50')
            y -= 0.05

    fig.suptitle('Experimental Design: Three Complementary Validation Experiments',
                 fontsize=14, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'nys_fig_experiment_design.png'), facecolor='white')
    plt.close(fig)
    print("Saved: nys_fig_experiment_design.png")


def fig_gp_demo():
    """Figure 7 replacement: Proper GP posterior visualization on 2D field."""
    rng = np.random.default_rng(42)

    grid_size = 40
    N = grid_size ** 2
    coords = np.array([(i/grid_size, j/grid_size) for i in range(grid_size)
                        for j in range(grid_size)])

    ls, var, noise = 0.12, 1.0, 0.15
    D = cdist(coords, coords, 'sqeuclidean')
    K = var * np.exp(-0.5 * D / ls**2) + 1e-6 * np.eye(N)
    f_true = rng.multivariate_normal(np.zeros(N), K)

    # Sparse observations (only 15% observed)
    n_obs = int(0.15 * N)
    obs_idx = rng.choice(N, n_obs, replace=False)
    y_obs = f_true[obs_idx] + rng.normal(0, np.sqrt(noise), n_obs)

    # Nyström GP prediction
    from sklearn.cluster import MiniBatchKMeans
    M = 80
    km = MiniBatchKMeans(n_clusters=M, random_state=42, n_init=3)
    km.fit(coords[obs_idx])
    Z = km.cluster_centers_

    W = var * np.exp(-0.5 * cdist(Z, Z, 'sqeuclidean') / ls**2) + 1e-6 * np.eye(M)
    C_train = var * np.exp(-0.5 * cdist(coords[obs_idx], Z, 'sqeuclidean') / ls**2)
    C_all = var * np.exp(-0.5 * cdist(coords, Z, 'sqeuclidean') / ls**2)

    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals = np.maximum(eigvals, 1e-10)
    W_sqrt_inv = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T

    Phi_train = C_train @ W_sqrt_inv
    Phi_all = C_all @ W_sqrt_inv

    from scipy.linalg import cholesky, cho_solve
    A = Phi_train.T @ Phi_train + noise * np.eye(M)
    L_A = cholesky(A, lower=True)
    v = cho_solve((L_A, True), Phi_train.T @ y_obs)
    mu = Phi_all @ v

    # Variance
    B = cho_solve((L_A, True), Phi_all.T)
    var_pred = var - np.sum(Phi_all.T * B, axis=0) + noise
    var_pred = np.maximum(var_pred, 1e-6)
    std_pred = np.sqrt(var_pred)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # True field
    im0 = axes[0].imshow(f_true.reshape(grid_size, grid_size), cmap='RdBu_r',
                          origin='lower', vmin=-2.5, vmax=2.5)
    axes[0].set_title('(a) True Latent Field', fontsize=11, weight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Observations (sparse)
    obs_map = np.full(N, np.nan)
    obs_map[obs_idx] = y_obs
    im1 = axes[1].imshow(obs_map.reshape(grid_size, grid_size), cmap='RdBu_r',
                          origin='lower', vmin=-2.5, vmax=2.5)
    axes[1].set_title(f'(b) Sparse Observations\n(n={n_obs}, {n_obs/N:.0%} coverage)',
                       fontsize=11, weight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Posterior mean
    im2 = axes[2].imshow(mu.reshape(grid_size, grid_size), cmap='RdBu_r',
                          origin='lower', vmin=-2.5, vmax=2.5)
    rmse = np.sqrt(np.mean((mu - f_true)**2))
    axes[2].set_title(f'(c) Nyström GP Mean\n(M={M}, RMSE={rmse:.3f})',
                       fontsize=11, weight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Posterior uncertainty
    im3 = axes[3].imshow(std_pred.reshape(grid_size, grid_size), cmap='YlOrRd',
                          origin='lower')
    axes[3].set_title('(d) Posterior Uncertainty (SD)', fontsize=11, weight='bold')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('Nyström GP Spatial Prediction from Sparse Observations',
                 fontsize=14, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'nys_fig_gp_demo.png'), facecolor='white')
    plt.close(fig)
    print("Saved: nys_fig_gp_demo.png")


if __name__ == '__main__':
    fig_nystrom_workflow()
    fig_kernel_approximation()
    fig_experiment_design()
    fig_gp_demo()
    print("\nAll replacement figures generated.")
