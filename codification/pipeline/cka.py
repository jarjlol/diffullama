"""Linear CKA (Kornblith et al., 2019) -- similarity of two representations of the same inputs.

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)

X: n x d1, Y: n x d2, rows = the same n inputs, columns centred. 1.0 = identical up to
an orthogonal transform and isotropic scaling; ~0 = unrelated. Two implementations:
`cka_torch` for real activations, `cka_reference` (pure Python, slow) to test it.
"""
from __future__ import annotations


def _center(M):
    n, d = len(M), len(M[0])
    mu = [sum(M[i][j] for i in range(n)) / n for j in range(d)]
    return [[M[i][j] - mu[j] for j in range(d)] for i in range(n)]


def _gram_frob2(A, B):
    """||A^T B||_F^2 for n x d matrices A, B."""
    da, db, n = len(A[0]), len(B[0]), len(A)
    return sum(sum(A[k][i] * B[k][j] for k in range(n)) ** 2 for i in range(da) for j in range(db))


def cka_reference(X, Y) -> float:
    Xc, Yc = _center(X), _center(Y)
    return _gram_frob2(Yc, Xc) / ((_gram_frob2(Xc, Xc) * _gram_frob2(Yc, Yc)) ** 0.5)


def cka_torch(X, Y) -> float:
    import torch
    X = X.double(); Y = Y.double()
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    num = (Y.T @ X).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return float(num / den)
