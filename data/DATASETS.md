# Datasets Documentation

## Overview

All three experiments use **synthetic spatial data** generated with NumPy random generators and RBF (squared exponential) kernels. No external data downloads are needed. All random seeds are fixed for exact reproducibility.

Synthetic data is used because:
1. We can compare Nyström predictions against the **exact full GP** (known optimal solution)
2. We can control the spatial correlation structure (lengthscale, variance)
3. We can set exact observation counts per subject to test sparse-data scenarios
4. No network access is required

For real MRI data validation (future work), we reference: OpenNeuro, OASIS (Marcus et al., 2007), and fastMRI (Zbontar et al., 2018).

---

## Experiment 1: Approximation Quality

**Script:** `experiments/run_all.py :: experiment1()`

**Dataset: 2D spatial field on a 48×48 grid**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Grid size | 48×48 = 2,304 voxels | Simulates a single MRI slice |
| Kernel | RBF (squared exponential) | Smooth spatial correlations |
| Lengthscale (ℓ) | 0.15 | Correlation range (~7 voxels) |
| Signal variance (σ²) | 1.0 | Amplitude of spatial field |
| Noise variance (σ²_n) | 0.1 | Observation noise (SNR ≈ 10) |
| Train/test split | 80/20 (1,843 / 461) | Random permutation |
| Random seed | 42 | |

**Generative process:**
```python
# Coordinates normalized to [0,1]²
coords = [(i/48, j/48) for i in range(48) for j in range(48)]

# True spatial field drawn from GP prior
K = RBF(coords, coords, ℓ=0.15, σ²=1.0) + 1e-6 × I
f_true ~ MultivariateNormal(0, K)

# Noisy observations
y = f_true + ε,  where ε ~ N(0, 0.1)
```

**Accessing the data:**
```python
from src.kernels import rbf_kernel
import numpy as np

rng = np.random.default_rng(42)
gs = 48; N = gs**2
coords = np.array([(i/gs, j/gs) for i in range(gs) for j in range(gs)])

K = rbf_kernel(coords, coords, 0.15, 1.0) + 1e-6 * np.eye(N)
f_true = rng.multivariate_normal(np.zeros(N), K)
y = f_true + rng.normal(0, np.sqrt(0.1), N)

# f_true is the ground truth; y is what we observe
# The goal is to recover f_true from y using GP regression
```

---

## Experiment 2: Scalability Benchmark

**Script:** `experiments/run_all.py :: experiment2()`

**Dataset: Multiple grid sizes**

| Grid | N (voxels) | Purpose |
|------|-----------|---------|
| 16×16 | 256 | Baseline (Nyström overhead dominates) |
| 32×32 | 1,024 | Near crossover point |
| 48×48 | 2,304 | Nyström starts winning |
| 64×64 | 4,096 | Clear Nyström advantage |
| 80×80 | 6,400 | Full GP extrapolated |
| 100×100 | 10,000 | Full GP extrapolated |

**Fixed parameters:** ℓ=0.15, σ²=1.0, σ²_n=0.1, M=100 (k-means landmarks).

For N > 4,096, the full GP time is extrapolated using the O(N³) scaling law fitted to measured times at smaller N. Nyström times are always measured directly.

---

## Experiment 3: Hierarchical Multi-Subject

**Script:** `experiments/run_all.py :: experiment3()`

**Dataset: 10 subjects sharing a common spatial field**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Grid | 32×32 = 1,024 voxels per subject | |
| J (subjects) | 10 | Simulates 10 patients |
| Population field | GP draw with ℓ=0.15, σ²=1.0 | Shared spatial structure |
| Subject offsets | δ_j ~ N(0, 0.25) | Each patient has a scalar shift |
| Noise | σ²_n = 0.3 | Higher noise than Exp 1 |
| Obs per subject | [20, 30, 40, 50, 80, 120, 200, 400, 600, 800] | Deliberately unequal |
| Landmark count | M = 80 (k-means) | |
| Random seed | 42 | |

**Generative process:**
```python
# Population field (shared across all subjects)
K_pop = RBF(coords, coords, ℓ=0.15, σ²=1.0)
f_pop ~ MultivariateNormal(0, K_pop)

# Per-subject fields
for j in range(10):
    f_j = f_pop + δ_j           # population + subject offset
    obs_idx = random_subset(N, n_j)   # sparse observations
    y_j = f_j[obs_idx] + ε_j    # noisy measurements
```

**Key design choice:** The observation counts span 20 to 800. Subjects with n=20 have only 2% of voxels observed — these are the cases where hierarchical borrowing from the population estimate matters most.

**Accessing the data:**
```python
from src.kernels import rbf_kernel
import numpy as np

rng = np.random.default_rng(42)
gs = 32; N = gs**2
coords = np.array([(i/gs, j/gs) for i in range(gs) for j in range(gs)])

K = rbf_kernel(coords, coords, 0.15, 1.0) + 1e-6 * np.eye(N)
f_pop = rng.multivariate_normal(np.zeros(N), K)

offsets = rng.normal(0, 0.5, 10)
n_obs = [20, 30, 40, 50, 80, 120, 200, 400, 600, 800]

for j in range(10):
    f_j = f_pop + offsets[j]
    obs_idx = rng.choice(N, n_obs[j], replace=False)
    y_j = f_j[obs_idx] + rng.normal(0, np.sqrt(0.3), n_obs[j])
    # f_j is ground truth; obs_idx + y_j is what we observe
```

---

## Kernel Function

All experiments use the RBF (squared exponential) kernel:

```
k(x, x') = σ² × exp(-||x - x'||² / (2ℓ²))
```

**Implementation:** `src/kernels.py :: rbf_kernel()`

Also available: `matern32_kernel()` and `matern52_kernel()` for future experiments with rougher spatial fields.

---

## Reproducibility

```bash
cd nystrom-gp-mri

# Run all experiments (generates figures/ and prints results)
python experiments/run_all.py

# Quick demo using src/ library
python examples/quickstart.py

# Generate paper figures only (requires experiments to have run)
python experiments/make_figures.py
```

Expected runtime: ~30 seconds for all 3 experiments.

---

## Future: Real MRI Datasets

When network access is available, validation on real data should use:

| Dataset | URL | Description |
|---------|-----|-------------|
| OASIS-1 | https://sites.wustl.edu/oasisbrains/ | 416 subjects, T1-weighted brain MRI |
| fastMRI | https://fastmri.med.nyu.edu/ | 6,970 brain MRIs (T1, T2, FLAIR) |
| OpenNeuro | https://openneuro.org/ | 1000+ neuroimaging datasets |

These would replace the synthetic spatial fields with real voxel-wise intensity values, testing whether the Nyström approximation quality holds on real MRI correlation structures.
