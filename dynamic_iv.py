"""Identification checks for Gaussian VAR(1), state order (I, X, H, Y).

These check a specified structural model, not validity of observational IVs.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def instrument_indices(selected, d):
    a = np.asarray(selected)
    if a.ndim != 1:
        raise ValueError('Selection must be a one-dimensional mask or index list')
    if a.dtype.kind == 'b':
        if len(a) != d:
            raise ValueError('Boolean mask has wrong length')
        a = np.flatnonzero(a)
    elif a.size and a.dtype.kind not in 'iu':
        raise ValueError('Instrument indices must be integers')
    a = a.astype(int)
    if np.any(a < 0) or np.any(a >= d) or len(np.unique(a)) != len(a):
        raise ValueError('Instrument indices must be unique and in range')
    return a


def innovation_covariance(Sigma, dim):
    q = np.asarray(Sigma, dtype=float)
    if q.ndim == 0:
        q = float(q) * np.eye(dim)
    if q.shape != (dim, dim) or not np.all(np.isfinite(q)):
        raise ValueError('Invalid innovation covariance shape or values')
    if not np.allclose(q, q.T) or np.linalg.eigvalsh(q).min() < -1e-12:
        raise ValueError('Innovation covariance must be symmetric positive semidefinite')
    return q


def validate_selected_instrument_history(A, d, p, selected, Sigma=0.1, atol=1e-12):
    """Sufficient conditions for B=I_selected[t-3], Z=I_selected[t-2].

    Requires a stable structural VAR(1) with autonomous selected IV blocks,
    no direct I->H/Y effects, and iid Gaussian innovations independent between
    IV and non-IV components. Stricter than necessary, intentionally.
    """
    if not isinstance(A, dict) or set(A) != {1}:
        raise ValueError('Local history validation supports VAR(1) only')
    a = np.asarray(A[1], dtype=float)
    dim = d + 2*p + 1
    if a.shape != (dim, dim) or not np.all(np.isfinite(a)):
        raise ValueError('Expected (I,X,H,Y) with dim(H)=p and scalar Y')
    if np.max(np.abs(np.linalg.eigvals(a))) >= 1:
        raise ValueError('The VAR must be stable')
    idx = instrument_indices(selected, d)
    outside = np.setdiff1d(np.arange(dim), idx)
    if np.any(np.abs(a[np.ix_(idx, outside)]) > atol):
        raise ValueError('Selected IV history omits parents of the selected block')
    if np.any(np.abs(a[d+p:, :d]) > atol):
        raise ValueError('Instruments must not directly affect H or Y')
    q = innovation_covariance(Sigma, dim)
    if np.any(np.abs(q[:d, d:]) > atol):
        raise ValueError('IV and non-IV innovations must be independent')
    return {'conditioning': 'selected_instrument_history', 'selected': idx.tolist()}


def population_moments(A, d, p, selected, Sigma=0.1, conditioning='instrument_history'):
    """Exact stationary moments. x_history is an invalid negative control."""
    a = np.asarray(A[1], dtype=float)
    idx = instrument_indices(selected, d)
    if conditioning == 'instrument_history':
        validate_selected_instrument_history(A, d, p, idx, Sigma)
    elif conditioning not in {'x_history', 'none'}:
        raise ValueError('Unknown diagnostic conditioning set')
    if np.max(np.abs(np.linalg.eigvals(a))) >= 1:
        raise ValueError('The VAR must be stable')
    stationary = solve_discrete_lyapunov(a, innovation_covariance(Sigma, len(a)))
    xidx = np.arange(d, d+p)
    yidx = np.array([d+2*p])

    def cov(left, right):
        ii, ti = left
        jj, tj = right
        c = np.linalg.matrix_power(a, ti-tj) @ stationary if ti >= tj else stationary @ np.linalg.matrix_power(a, tj-ti).T
        return c[np.ix_(ii, jj)]

    z, x, y = (idx, -2), (xidx, -1), (yidx, 0)
    G, h = cov(z, x), cov(z, y)
    if conditioning != 'none' and (len(idx) or conditioning == 'x_history'):
        b = (idx, -3) if conditioning == 'instrument_history' else (xidx, -2)
        G -= cov(z, b) @ np.linalg.solve(cov(b, b), cov(b, x))
        h -= cov(z, b) @ np.linalg.solve(cov(b, b), cov(b, y))
    beta = a[yidx[0], xidx]
    h = h.ravel()
    return {'G': G, 'h': h, 'beta': beta, 'moment': h-G@beta,
            'rank': int(np.linalg.matrix_rank(G)), 'stationary_covariance': stationary}
