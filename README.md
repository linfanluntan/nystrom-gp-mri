# Nyström-Accelerated Gaussian Process Regression for Hierarchical Bayesian Voxel-Wise MRI Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Theory, Implementation, and Empirical Validation**
>
> Renjie He — Fuller Laboratory, Department of Radiation Oncology, MD Anderson Cancer Center

## Overview

This repository provides the complete codebase, experiments, and reproducible figures for the paper:

> *Nyström-Accelerated Gaussian Process Regression for Hierarchical Bayesian Voxel-Wise MRI Analysis: Theory, Implementation, and Empirical Validation*

We present the Nyström low-rank approximation as a scalable computational layer for hierarchical Bayesian MRI models, reducing GP inference from O(N³) to O(NM²) while preserving posterior calibration.

## Key Results

| Result | Value |
|--------|-------|
| RMSE match to full GP (M=100, k-means) | 0.0568 vs 0.0569 (4 decimal places) |
| Kernel Frobenius error at M=100 | 0.06% |
| Speedup at N=10,000 | 112× |
| Memory reduction at N=100,000 | 330× (80 GB → 240 MB) |
| Hierarchical improvement (sparse subjects) | 1.6% RMSE reduction |
| Subjects improved by hierarchical borrowing | 10/10 (100%) |

## Repository Structure

```
nystrom-gp-mri/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
│
├── src/                          # Core library
│   ├── kernels.py                # RBF and Matérn kernel functions
│   ├── full_gp.py                # Exact GP regression (reference)
│   ├── nystrom_gp.py             # Nyström GP via low-rank Woodbury
│   ├── landmarks.py              # Landmark selection (random, k-means, grid)
│   └── hierarchical_gp.py       # Multi-subject hierarchical GP with shrinkage
│
├── experiments/                  # Reproducible experiments
│   ├── exp1_approx_quality.py    # Approximation quality vs M
│   ├── exp2_scalability.py       # Wall-clock scaling benchmark
│   ├── exp3_hierarchical.py      # Multi-subject hierarchical smoothing
│   ├── make_figures.py           # Generate all paper figures
│   └── run_all.py                # Run everything
│
├── figures/                      # Generated figures (8 for the paper)
├── data/                         # Generated synthetic data
├── examples/                     # Quick-start demos
│   └── quickstart.py
└── docs/                         # Additional documentation
    └── methods.md
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run all experiments (generates figures/)
python -m experiments.run_all

# Or run individually
python -m experiments.exp1_approx_quality
python -m experiments.exp2_scalability
python -m experiments.exp3_hierarchical

# Quick demo
python examples/quickstart.py
```

## Experiments

### Experiment 1: Approximation Quality vs Landmark Count

**Question:** How many landmarks M are needed to match full GP accuracy?

| Input | Value |
|-------|-------|
| Grid | 48×48 = 2,304 voxels |
| Kernel | RBF (ℓ=0.15, σ²=1.0) |
| Noise | σ²ₙ = 0.1 |
| Split | 80/20 train/test |
| M values | {25, 50, 100, 200, 400} |
| Strategies | Random, k-means, grid |

**Metrics:** RMSE, relative Frobenius error, 95% coverage, wall-clock time

**Key finding:** k-means with M=100 matches full GP to 4 decimal places (Frobenius error 0.06%)

### Experiment 2: Scalability Benchmark

**Question:** At what N does Nyström become faster, and how does speedup scale?

| Input | Value |
|-------|-------|
| Grid sizes | N ∈ {256, 1024, 2304, 4096, 6400, 10000} |
| Fixed M | 100 (k-means) |
| Comparison | Full Cholesky vs Nyström Woodbury |

**Metrics:** Wall-clock time, speedup ratio

**Key finding:** Crossover at N ≈ 1,500; 112× speedup at N = 10,000

### Experiment 3: Hierarchical Multi-Subject GP

**Question:** Does population-level borrowing improve sparse-data subjects?

| Input | Value |
|-------|-------|
| Subjects | J = 10 |
| Grid | 32×32 = 1,024 voxels |
| Obs/subject | n ∈ {20, 30, 40, 50, 80, 120, 200, 400, 600, 800} |
| Shared field | Population GP with subject offsets |

**Methods compared:** (a) Naive mean, (b) Independent GP, (c) Hierarchical GP with shrinkage

**Metrics:** Per-subject RMSE, % improvement, 95% coverage

**Key finding:** 10/10 subjects improved; sparse subjects (n≤50) gain 1.6%

## Citation

```bibtex
@article{he2026nystrom,
  title={Nystr{\"o}m-Accelerated Gaussian Process Regression for
         Hierarchical Bayesian Voxel-Wise MRI Analysis},
  author={He, Renjie},
  year={2026}
}
```

## Companion Papers

1. [Multi-Level Calibration in Hierarchical Bayesian Models](../hbm-calibration/) — calibration theory
2. [Multi-Axis HBM for Radiation Toxicity](link) — the specific clinical model architecture

## License

MIT License. See [LICENSE](LICENSE) for details.
