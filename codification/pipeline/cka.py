"""Linear CKA (Kornblith et al., 2019) -- similarity of two representations of the same inputs.

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)

X: n x d1, Y: n x d2, rows = the same n inputs, columns centred. 1.0 = identical up to
an orthogonal transform and isotropic scaling; ~0 = unrelated. Two implementations:
`cka_torch` for real activations, `cka_reference` (pure Python, slow) to test it.

Debiased CKA replaces each HSIC with its unbiased estimator (Song et al., 2012), computed on
Gram matrices with zeroed diagonals. The plain estimator is inflated when n is not much larger
than d, and by different amounts for different widths (768 to 4096 here, n = 2000), so models of
different width are compared on the debiased value as well.
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


def _hsic_u_reference(K, L):
    """Unbiased HSIC of two n x n Gram matrices whose diagonals are zero."""
    n = len(K)
    tr = sum(K[i][j] * L[j][i] for i in range(n) for j in range(n))
    rk, rl = [sum(r) for r in K], [sum(r) for r in L]
    return (tr + sum(rk) * sum(rl) / ((n - 1) * (n - 2)) - 2 * sum(a * b for a, b in zip(rk, rl)) / (n - 2)) / (n * (n - 3))


def _gram0(M):
    n = len(M)
    return [[0.0 if i == j else sum(a * b for a, b in zip(M[i], M[j])) for j in range(n)] for i in range(n)]


def cka_debiased_reference(X, Y) -> float:
    K, L = _gram0(X), _gram0(Y)
    return _hsic_u_reference(K, L) / ((_hsic_u_reference(K, K) * _hsic_u_reference(L, L)) ** 0.5)


def cka_debiased_torch(X, Y) -> float:
    X = X.double(); Y = Y.double()
    K = X @ X.T; L = Y @ Y.T
    K.fill_diagonal_(0); L.fill_diagonal_(0)

    def hsic(A, B):
        n = A.shape[0]
        return ((A * B.T).sum() + A.sum() * B.sum() / ((n - 1) * (n - 2))
                - 2 * (A.sum(0) @ B.sum(1)) / (n - 2)) / (n * (n - 3))
    return float(hsic(K, L) / (hsic(K, K) * hsic(L, L)).sqrt())
