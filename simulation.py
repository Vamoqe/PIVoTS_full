"""Stationary Gaussian VAR simulation, including correct multi-lag initialization."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from dynamic_iv import innovation_covariance


def sim_data(A, n, Sigma=0.1, plot=False, seed=None, initialization="stationary"):
    return sim_data_inner(A, n, Sigma, plot, seed, initialization)


def sim_data_inner(A, n, Sigma=0.1, plot=False, seed=None, initialization="stationary"):
    if not isinstance(A, dict) or not A or any(not isinstance(k, int) or k < 1 for k in A):
        raise ValueError("A must map positive integer lags to square matrices")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    d = np.asarray(next(iter(A.values()))).shape[0]
    if any(np.asarray(a).shape != (d,d) or not np.all(np.isfinite(a)) for a in A.values()):
        raise ValueError("All lag matrices must be finite and have the same square shape")
    lags = max(A)
    F = np.zeros((d*lags,d*lags))
    F[:d] = np.concatenate([np.asarray(A.get(k,np.zeros((d,d)))) for k in range(1,lags+1)],axis=1)
    F[d:,:-d] = np.eye(d*(lags-1))
    if np.max(np.abs(np.linalg.eigvals(F))) >= 1:
        raise ValueError("Stationary simulation requires a stable VAR")
    Q = innovation_covariance(Sigma,d)
    rng = np.random.default_rng(seed)
    if initialization == "stationary":
        Qstate = np.zeros_like(F); Qstate[:d,:d] = Q
        S = solve_discrete_lyapunov(F,Qstate)
        state = rng.multivariate_normal(np.zeros(len(F)), (S+S.T)/2, check_valid="raise")
    elif initialization == "zero":
        state = np.zeros(len(F))
    else:
        raise ValueError("initialization must be stationary or zero")
    data = np.empty((n,d)); data[0] = state[:d]
    noise = rng.multivariate_normal(np.zeros(d),Q,size=n-1,check_valid="raise")
    for t in range(1,n):
        state = F @ state
        state[:d] += noise[t-1]
        data[t] = state[:d]
    if plot:
        import matplotlib.pyplot as plt
        import networkx as nx
        nx.draw_circular(nx.from_numpy_array(F.T,create_using=nx.DiGraph),with_labels=True)
        plt.show()
    return data


def get_skeleton(id_I, id_X, id_H, id_Y):
    d = len(id_I)+len(id_X)+len(id_H)+len(id_Y)
    A_empty = np.zeros(shape=(d,d))

    for id in [id_I, id_X, id_H, id_Y]:
        for i in id:
            for j in id:
                A_empty[i,j] = 1

    for id_1, id_2 in [(id_I, id_X), (id_X, id_Y), (id_H, id_X), (id_H, id_Y)]:
        for i in id_1:
            for j in id_2:
                A_empty[j, i] = 1
    return A_empty
