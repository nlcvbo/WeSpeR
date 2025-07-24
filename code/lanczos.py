
import numpy as np
from scipy.linalg import eigh_tridiagonal 
from scipy.stats import norm
from numba import jit

@jit(nopython=True)
def lanczos(A, v, K, ortho = False):
    n = len(v) 
    alpha, beta = np.zeros(K, dtype=A.dtype), np.zeros(K-1, dtype=A.dtype)
    V = np.zeros((K,n), dtype=A.dtype) # Orthonormal basis matrix
    V[0] = v

    w = A @ v
    alpha_k = np.conj(w).T @ v
    w = w - alpha_k * v
    alpha[0] = alpha_k

    for k in range(1,K):
        beta_k = np.linalg.norm(w)
        beta[k-1] = beta_k
        if beta_k < 1e-10:
            print(f"Terminating early at iteration {k} due to small beta_k: {beta_k}")
            break
        V[k] = w / beta_k
        w = A @ V[k]
        alpha_k = np.conj(w).T @ V[k]
        alpha[k] = alpha_k
        w = w - alpha_k*V[k] - beta_k*V[k - 1]
        # Reorthogonalization step
        if ortho:
            for j in range(k + 1):
                w = w - np.dot(np.conj(V[j]), w) * V[j]
    return alpha, beta, V.T

def lanczos_f(A, v, K, f, ortho  =False):
    n = len(v)
    v_norm = np.linalg.norm(v)
    v = v / v_norm 

    alpha, beta, V = lanczos(A, v, K, ortho = ortho)

    eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
    fT = eigvecs @ (f(eigvals)[:,None] * np.conj(eigvecs).T)

    e1 = np.zeros((fT.shape[1],))
    e1[0] = v_norm

    return V @ (fT @ e1)

def lanczos_fA(A, K, f, num_probes = 1, v0=None, ortho=False):
    assert num_probes >= 1
    n = A.shape[0]

    if v0 is None:
        v0 = np.random.randn(n)
    v0 = v0 / np.linalg.norm(v0)

    fA_approx = np.zeros((n,n))
    for k in range(num_probes):
        alpha, beta, V = lanczos(A, v0, K, ortho=ortho)

        # Compute f(T)
        eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
        f_T = eigvecs @ np.diag(f(eigvals)) @ eigvecs.T

        # Approximate f(A) ≈ V f(T) V^T
        fA_approx += V @ f_T @ V.T
        
        v0 = np.random.randn(n)
        v0 = v0 / np.linalg.norm(v0)
    return fA_approx/num_probes

def lanczos_quadrature(A, num_probes, K, ortho = False):
    n = A.shape[0]
    all_eigs = []
    all_weights = []

    for probe_idx in range(num_probes):
        v = np.random.normal(size=n)
        v = v/np.linalg.norm(v)
        alpha, beta, _ = lanczos(A, v, K, ortho = ortho)
        eigvals, eigvecs = eigh_tridiagonal(alpha, beta)

        weights = eigvecs[0, :]**2
        all_eigs.append(eigvals)
        all_weights.append(weights)

    grid_points = np.sort(np.concatenate(all_eigs))[::num_probes]
    csd_estimates = np.zeros(grid_points.shape[0], dtype=np.float64)
    for probe_idx in range(num_probes):
        for i, grid_point in enumerate(grid_points):
            indicator = all_eigs[probe_idx] <= grid_point
            csd_estimates[i] += (all_weights[probe_idx][indicator]).sum()
    csd_estimates /= num_probes

    return grid_points, csd_estimates
