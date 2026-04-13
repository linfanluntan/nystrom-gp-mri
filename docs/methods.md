# Nystrom GP: Core Algorithm
1. Kernel blocks: W = k(Z,Z), C = k(X,Z)
2. Low-rank factor: Phi = C W^{-1/2}
3. Woodbury solve: v = (Phi^T Phi + sigma^2 I_M)^{-1} Phi^T y
4. Posterior mean: mu* = Phi_test v
5. Posterior variance: sigma^2* = k(x*,x*) - diag(Phi_test A^{-1} Phi_test^T)
