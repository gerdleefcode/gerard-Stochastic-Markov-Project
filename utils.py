import math
import os
import random
from typing import Optional, Tuple

import numpy as np


def set_seed(seed: Optional[int] = 42) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_stochastic_matrix(P: np.ndarray, atol: float = 1e-8) -> bool:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if np.any(P < -atol):
        return False
    row_sums = P.sum(axis=1)
    return np.allclose(row_sums, 1.0, atol=atol)


def normalize_rows(M: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M[M < 0] = 0.0
    row_sums = M.sum(axis=1, keepdims=True)
    out = np.divide(M, np.maximum(row_sums, eps), where=row_sums>0)
    # Only fix true zero rows to uniform
    zero_rows = (row_sums.squeeze() == 0)
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / M.shape[1]
    return out

def normalize_vector(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v[v < 0] = 0.0
    s = v.sum()
    if s == 0:
        return np.full_like(v, 1.0 / len(v), dtype=float)
    return v / max(s, eps)


def random_stochastic_matrix(n: int, alpha: float = 1.0) -> np.ndarray:
    # Each row ~ Dirichlet(alpha, ..., alpha)
    return np.vstack([np.random.dirichlet(alpha * np.ones(n)) for _ in range(n)])


def random_categorical(p: np.ndarray) -> int:
    return int(np.random.choice(len(p), p=p))


def stationary_distribution(P: np.ndarray, method: str = "eigen") -> np.ndarray:
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if method == "eigen":
        w, v = np.linalg.eig(P.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(w - 1.0))
        vec = np.real(v[:, idx])
        vec = np.maximum(vec, 0.0)
        if vec.sum() == 0:
            vec = np.abs(vec)
        return normalize_vector(vec)
    elif method == "solve":
        A = (P.T - np.eye(n))
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.maximum(pi, 0.0)
        return normalize_vector(pi)
    else:
        raise ValueError("method must be 'eigen' or 'solve'")


def slem(P: np.ndarray, tol: float = 1e-10) -> float:
    # second largest eigenvalue modulus (SLEM)
    eigvals = np.linalg.eigvals(P)
    eigvals = np.real_if_close(eigvals, tol=1000)
    mods = np.abs(eigvals)
    # Remove the one closest to 1 (the stationary eigenvalue)
    idx_one = np.argmin(np.abs(mods - 1.0))
    mods_wo_one = np.delete(mods, idx_one)
    return float(np.max(mods_wo_one)) if mods_wo_one.size > 0 else 0.0


def mixing_time_upper_bound(P: np.ndarray, eps: float = 1e-3) -> Tuple[int, float]:
    """
    Returns (t_mix_upper_bound, spectral_gap) using a standard bound:
      t_mix(eps) <= log(1/(eps*pi_min)) / (1 - lambda_2)
    where lambda_2 is SLEM.
    """
    pi = stationary_distribution(P)
    pi_min = float(np.maximum(np.min(pi), 1e-15))
    lam2 = slem(P)
    gap = max(1e-12, 1.0 - lam2)
    t_mix = int(math.ceil(np.log(1.0 / (eps * pi_min)) / gap))
    return t_mix, gap


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[idx] = 1.0
    return v


def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
    out = m + np.log(s)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out