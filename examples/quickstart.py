"""Quick-start: Nystrom GP on a 2D spatial field."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src import NystromGP, FullGP
from src.kernels import rbf_kernel
rng = np.random.default_rng(42)
N = 900
coords = np.array([(i/30, j/30) for i in range(30) for j in range(30)])
K = rbf_kernel(coords, coords, 0.15, 1.0) + 1e-6*np.eye(N)
f = rng.multivariate_normal(np.zeros(N), K)
y = f + rng.normal(0, 0.3, N)
idx = rng.permutation(N)
X_tr, y_tr = coords[idx[:700]], y[idx[:700]]
X_te, f_te = coords[idx[700:]], f[idx[700:]]
full = FullGP(lengthscale=0.15, variance=1.0, noise_var=0.1)
full.fit(X_tr, y_tr)
mu_full, _ = full.predict(X_te)
print(f"Full GP RMSE: {np.sqrt(np.mean((mu_full - f_te)**2)):.4f}")
for M in [25, 50, 100]:
    nys = NystromGP(M=M, lengthscale=0.15, variance=1.0, noise_var=0.1)
    nys.fit(X_tr, y_tr)
    mu_nys, _ = nys.predict(X_te)
    print(f"Nystrom (M={M:3d}) RMSE: {np.sqrt(np.mean((mu_nys - f_te)**2)):.4f}")
