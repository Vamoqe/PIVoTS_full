
"""
utility functions
"""
import numpy as np
import logging
logging.basicConfig(level=20)
from jax import numpy as jnp
import jax
from jax import random
import sys
import os

from scipy.spatial.distance import cdist
sys.path.insert(0, "../src")

import simulation

from itertools import product
import statsmodels.api as sm
def projection(P1):
    """
    computation of projection matrix

    Parameters
    ----------
    P1: ndarray

    Returns
    -------
    ndarray
    """
    return P1 @ np.linalg.solve(P1.T @ P1, P1.T)


def cosine_distance(mat1, mat2):
    """
    computation of cosine distance

    Parameters
    ----------
    mat1: nd.array
    mat2: nd.array

    Returns
    -------
    float

    """

    mat_comp = np.vstack([mat1, mat2])

    cdist_mat = cdist(mat_comp, mat2, 'cosine')
    cdist_mat[cdist_mat > 1] = 2 - cdist_mat[cdist_mat > 1]

    return np.sum(cdist_mat)


def generate_sparse_vector(key, n, p, perc=0.8):
    """
    generate sparse vector with *perc* zero values and *(1-perc)* uniform distributed values on the interval of [-5.0;
    5.0]

    Parameters
    ----------
    key: jax key
    n: int
        number of samples
    p: int
        number pf treatment variables
    perc: float
        percentage of zero values

    Returns
    -------
    nd.array with perc of zero entries
    """

    p0 = int(jnp.floor(p * perc))
    p1 = p - p0

    key, key_sub = jax.random.split(key)
    vec_non_zero = jax.random.uniform(key_sub, (p1, n), minval=-5.0, maxval=5.0).reshape(p1, n)
    vec_zeros = jnp.zeros((p0, n))
    vec = jnp.concatenate([vec_non_zero, vec_zeros])
    key, key_sub = jax.random.split(key)
    vec = jax.random.permutation(key_sub, vec, independent=True)
    return vec.reshape(p, n)

def generate_scenario(key, d, p, d_identify=None, alpha_sparse=0.3, do_permute=False, do_cluster=True):
    """
    generate a scenario of $\alpha$, $\beta$ and mixing matrix $M$ and $e$ for estimation

    Parameters
    ----------
    key: jax key
    d: int
    p: int
    d_identify: int
    conf_zero_perc: float

    Returns
    -------
    alpha_star: nd.array
    beta_star: ndarray
    e_confounder: ndarray
    M: ndarray
        mixing matrix for confounder

    """
    l = p
    if d_identify is None:
        d_identify = int(np.ceil(d / 2))

    key, key_sub = jax.random.split(key)
    beta_prelim = generate_sparse_vector(key_sub, 1, d_identify, 0).squeeze()
    beta_prelim = np.hstack([beta_prelim, np.zeros(p - d_identify)])

    key, key_sub = jax.random.split(key)
    alpha_prelim = generate_sparse_vector(key_sub, d, d_identify, alpha_sparse).T
    alpha_prelim = np.hstack([alpha_prelim, np.zeros((d, p - d_identify))])

    if do_cluster:
        eig_vals = np.zeros((d_identify,))
        while (eig_vals).any() < 0.2:
            key, key_sub = jax.random.split(key)
            lam_pick = jax.random.choice(key_sub, np.arange(d), (d_identify,), replace=False)
            alpha_cap = alpha_prelim[lam_pick, :]
            eig_vals, _ = np.linalg.eig(projection(alpha_cap[:, :d_identify]))
            print(lam_pick)
            print(eig_vals)
        dim_c1 = np.int32((d - d_identify) / 2)
        dim_c2 = d - d_identify - dim_c1

        key, key_sub = jax.random.split(key)
        alpha_cluster1 = np.repeat(alpha_cap[d_identify-1, :], dim_c1).reshape((p, dim_c1)).T + (
            jax.random.normal(key_sub, (dim_c1, p))) / 100
        key, key_sub = jax.random.split(key)
        alpha_cluster2 = np.repeat(alpha_cap[d_identify-2, :], dim_c2).reshape((p, dim_c2)).T + (
            jax.random.normal(key_sub, (dim_c2, p))) / 100
        alpha_prelim = np.vstack([alpha_cap, alpha_cluster1, alpha_cluster2])


    if do_permute:
        key, key_sub = jax.random.split(key)
        vec_permute = jax.random.permutation(key_sub, np.vstack([beta_prelim.T, alpha_prelim]).T, independent=False).T

        alpha_star = vec_permute[1:, :]
        beta_star = vec_permute[0, :]
    else:
        alpha_star = alpha_prelim
        beta_star = beta_prelim

    key, key_sub = jax.random.split(key)
    breakpoint()
    M = jax.random.normal(key_sub, (l, p)).reshape(l, p)
    breakpoint()
    key, key_sub = jax.random.split(key)
    e_confounder = jax.random.uniform(key_sub, (l, )) / jax.random.uniform(key_sub, (l, )).sum()

    return alpha_star, beta_star, e_confounder, M




#def generate_time_scenario(key, d, p, d_identify=None, alpha_sparse=0.3, do_permute=False, do_cluster=True,tol = 0.1):
def generate_time_scenario_prev(
    key,
    d,
    p,
    d_identify=None,
    alpha_sparse=0.3,
    do_permute=False,
    do_cluster=True,
    tol=0.1,
    target_rho=0.8,
    confound_x_strength=1.0,
    confound_y_strength=1.0,
    iv_x_strength=1.0,
    beta_strength=1.0,
):   
    """
    generate a scenario of $\alpha$, $\beta$ and mixing matrix $M$ and $e$ for estimation

    Parameters
    ----------
    key: jax key
    d: int (dI)
    p: int (dx)
    d_identify: int (dx-Null(X))
    conf_zero_perc: float
    tol: tolerance to avoid single root

    Returns
    -------
    alpha_star: nd.array
    beta_star: ndarray
    e_confounder: ndarray
    M: ndarray
        mixing matrix for confounder

    """
    id_I, id_X, id_H, id_Y = np.split(np.arange(d+p+p+1), np.cumsum([d, p, p, 1]))[:4] 
    #print(id_I,id_X)  
    def sample_alphas(skel, instrument_strength=1):
        out = skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
            np.random.choice([-1, 1], size=skel.shape)
        for row, col in product(id_X, id_I):
            out[row, col] *= instrument_strength
        return out

    skel = simulation.get_skeleton(id_I,id_X,id_H,id_Y)
    
    A = sample_alphas(skel,instrument_strength=1)
    #print(A)
    while np.max(abs(np.linalg.eigvals(A)))>1-tol :
        r = np.max(abs(np.linalg.eigvals(A)))
        #print(r)
        A = A/(2)
    A = rescale_transition_matrix(A, target_rho=target_rho)
    #print(np.max(abs(np.linalg.eigvals(A))))
    #--------------------------------------time transition matrix construct
    l = p
    if d_identify is None:
        d_identify = int(np.ceil(d / 2))

    key, key_sub = jax.random.split(key)
    beta_prelim = generate_sparse_vector(key_sub, 1, d_identify, 0).squeeze()
    beta_prelim = np.hstack([beta_prelim, np.zeros(p - d_identify)])

    key, key_sub = jax.random.split(key)
    alpha_prelim = generate_sparse_vector(key_sub, d, d_identify, alpha_sparse).T
    alpha_prelim = np.hstack([alpha_prelim, np.zeros((d, p - d_identify))])

    if do_cluster:
        eig_vals = np.zeros((d_identify,))
        while (eig_vals).any() < 0.2:
            #print("hello")
            key, key_sub = jax.random.split(key)
            lam_pick = jax.random.choice(key_sub, np.arange(d), (d_identify,), replace=False)
            alpha_cap = alpha_prelim[lam_pick, :]
            eig_vals, _ = np.linalg.eig(projection(alpha_cap[:, :d_identify]))
            #print(lam_pick)
            #print(f"hello{eig_vals}")
        dim_c1 = np.int32((d - d_identify) / 2)
        dim_c2 = d - d_identify - dim_c1

        key, key_sub = jax.random.split(key)
        alpha_cluster1 = np.repeat(alpha_cap[d_identify-1, :], dim_c1).reshape((p, dim_c1)).T + (
            jax.random.normal(key_sub, (dim_c1, p))) / 100
        key, key_sub = jax.random.split(key)
        alpha_cluster2 = np.repeat(alpha_cap[d_identify-2, :], dim_c2).reshape((p, dim_c2)).T + (
            jax.random.normal(key_sub, (dim_c2, p))) / 100
        alpha_prelim = np.vstack([alpha_cap, alpha_cluster1, alpha_cluster2])


    if do_permute:
        key, key_sub = jax.random.split(key)
        vec_permute = jax.random.permutation(key_sub, np.vstack([beta_prelim.T, alpha_prelim]).T, independent=False).T

        alpha_star = vec_permute[1:, :]
        beta_star = vec_permute[0, :]
    else:
        alpha_star = alpha_prelim
        beta_star = beta_prelim

    print("A generated")
    key, key_sub = jax.random.split(key)
    M = jax.random.normal(key_sub, (l, p)).reshape(l, p)

    key, key_sub = jax.random.split(key)
    e_confounder = jax.random.uniform(key_sub, (l, )) / jax.random.uniform(key_sub, (l, )).sum()
    # ------- conbined into time transition matrix
    #breakpoint()
    A[d:d+p,:d] = np.array(alpha_star).T.tolist() # transpose alpha star and beta star to fit time IV data generating process
    A[d+p+l:,d:d+p] = np.array(beta_star).T.tolist()
    A = {1:A}
    #return alpha_star, beta_star, e_confounder, M
    #breakpoint()
    return A


def generate_time_scenario(
    key,
    d,
    p,
    d_identify=None,
    target_rho=0.8,
    iv_x_strength=1.0,
    confound_x_strength=1.2,
    confound_y_strength=1.5,
    beta_strength=1.0,
    n_confounders=3,
    seed=253,
    persistence_mode="diagonal",
):
    """
    Generate a stable VAR(1) transition matrix with controlled IV and confounding structure.

    W_t = A W_{t-1} + eps_t
    W_t = (I_t, X_t, H_t, Y_t)

    Main causal blocks:
        I -> X
        H -> X
        X -> Y
        H -> Y
    """

    if not 0 <= target_rho < 1 or d < 1 or p < 1:
        raise ValueError("Require positive dimensions and 0 <= target_rho < 1")
    if persistence_mode not in {"diagonal", "legacy_rescale"}:
        raise ValueError("persistence_mode must be diagonal or legacy_rescale")
    if n_confounders < 0 or (d_identify is not None and d_identify < 0):
        raise ValueError("Active dimension counts cannot be negative")
    rng = np.random.default_rng(seed)

    l = p
    dim = d + p + l + 1

    id_I, id_X, id_H, id_Y = np.split(
        np.arange(dim),
        np.cumsum([d, p, l, 1])
    )[:4]

    A = np.zeros((dim, dim))

    # Mild own persistence for instruments, X, and H
    A[np.ix_(id_I, id_I)] = 0.2 * np.eye(d)
    A[np.ix_(id_X, id_X)] = 0.2 * np.eye(p)
    A[np.ix_(id_H, id_H)] = 0.3 * np.eye(l)

    # Select active confounders
    n_confounders = min(n_confounders, l)
    active_H = id_H[:n_confounders]

    # I -> X
    A[np.ix_(id_X, id_I)] = (
        iv_x_strength
        * rng.uniform(0.5, 1.0, size=(len(id_X), len(id_I)))
        * rng.choice([-1, 1], size=(len(id_X), len(id_I)))
    )

    # H -> X
    A[np.ix_(id_X, active_H)] = (
        confound_x_strength
        * rng.uniform(0.6, 1.0, size=(len(id_X), len(active_H)))
        * rng.choice([-1, 1], size=(len(id_X), len(active_H)))
    )

    # Sparse beta: X -> Y
    if d_identify is None:
        d_identify = p

    active_X = id_X[:d_identify]

    A[np.ix_(id_Y, active_X)] = (
        beta_strength
        * rng.uniform(0.5, 1.0, size=(len(id_Y), len(active_X)))
        * rng.choice([-1, 1], size=(len(id_Y), len(active_X)))
    )

    # H -> Y
    A[np.ix_(id_Y, active_H)] = (
        confound_y_strength
        * rng.uniform(0.8, 1.2, size=(len(id_Y), len(active_H)))
        * rng.choice([-1, 1], size=(len(id_Y), len(active_H)))
    )

    # Change persistence without scaling causal/first-stage structural effects.
    # Stationary variances and effective precision can still change with rho.
    if persistence_mode == "legacy_rescale":
        A = rescale_transition_matrix(A, target_rho=target_rho)
    else:
        A[np.ix_(id_I, id_I)] = (2 * target_rho / 3) * np.eye(d)
        A[np.ix_(id_X, id_X)] = (2 * target_rho / 3) * np.eye(p)
        A[np.ix_(id_H, id_H)] = target_rho * np.eye(l)

    return {1: A}

# key = jax.random.PRNGKey(11)
# a = generate_time_scenario(key,3,5)
# print(a)


def generate_rv(key, d, p, n):
    """
    generate random variables for the experiment

    Parameters
    ----------
    key: jax key
    d: int 
        number of instruments
    p: int 
        number of treatment variables
    n: int 
        number of samples

    Returns
    -------
    z: nd array
        instrument matrix with n x d
    u: nd.array
        confounder matrix with n x l resp. nxp

    Notes
    --------
    l=p as the dimension of the confounder -> to match Janzing's paper, the confounder has the same dimensionality as
    the treatment variable
    """

    l = p

    # instruments
    key, key_sub = jax.random.split(key)
    #z = random.bernoulli(key_sub, 0.5, (n, d)).astype(float)
    z = random.rademacher(key_sub, (n, d)).astype(float)

    # confounder
    key, key_sub = jax.random.split(key)
    u = random.normal(key_sub, (n, l))

    return z, u


def generate_model(alpha, beta, z, u, e, M, lam, error_model="simple"):
    """
    generate samples from the instrumental variable model specified with the parameters $\alpha, \beta, e,
    M$ and the list $lambda$ that specifies which instruments are currently switched on

    Parameters
    ----------
    alpha: ndarray
    beta: ndarray
    z: ndarray
        instrument matrix
    u: ndarray
        confounder matrix
    e: ndarray
        confounder vector
    M: ndarray
        mixing matrix
    lam: list of bools
        indication which instruments are currently "switched on"
    error_model: string
        error model, currently only supports "simple"

    Returns
    -------
    x: ndarray
    y: ndarray
    z: ndarray

    """
    alpha = alpha[lam, :]
    z = z[:, lam]

    if error_model == "simple":
        # simple noise model
        c_x = u@M
        c_y = u@e

    x = z @ alpha + c_x
    y = x @ beta + c_y

    return x, y, z

def generate_time_model(A, d, p, n, seed=None, initialization="stationary"):
    """
    A: ndarray  parameter of causal and time series
    """
    predict = lambda x,b: sm.OLS(x, b).fit().predict()
    def get_2d(a):
        if len(a.shape) <= 1:
            a = a.reshape(-1, 1)
        return a
    l = p # confounder dim
    id_I, id_X, id_H, id_Y = np.split(np.arange(d+p+l+1), np.cumsum([d, p, l, 1]))[:4]
    #id_I = id_I[lam] # turn on/off some instrumental variable
    data = simulation.sim_data(A,n+3,seed=seed,initialization=initialization)
    I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

    #I_j = np.asarray(I[1:n+1]) # First data of I_j can't b e conditionedS
    #X_j = np.asarray(X[2:n+2]) # X in the time domain of I_j
    #Y_j = np.asarray(Y[3:n+3]) # Y in the time domain of I_j
    #breakpoint()
    
    return X, Y, I
"""
    # print("shape")
    # print(np.asarray(I[0:n]).shape)
    # print(np.asarray(X[1:n+1]).shape)
    # print(np.asarray(Y[2:n+2]).shape)

    B = np.hstack((
        get_2d(np.asarray(I[0:n])),                  #PA(I_j)
        get_2d(np.asarray(X[1:n+1])),                #PA(X_j)
        get_2d(np.asarray(Y[2:n+2]))                 #PA(Y_j)
                  ))
    B = sm.add_constant(B)

    r_I = get_2d(I_j) - get_2d(predict(get_2d(I_j), get_2d(B)))       
    r_X = get_2d(X_j) - get_2d(predict(get_2d(X_j), get_2d(B)))
    r_Y = get_2d(Y_j) - get_2d(predict(get_2d(Y_j), get_2d(B)))
    return r_X, r_Y, r_I
"""
def truncated_residual(X, Y, I_raw, n, lam):
    """Align Y_t, X_{t-1}, I_selected,t-2 and condition on I_selected,t-3.

    Validity requires autonomous selected IV blocks, exclusion and innovation
    exogeneity; see dynamic_iv.validate_selected_instrument_history. This
    numerical residualizer alone does not establish IV validity. An empty
    selection centers X/Y for the OLS comparison.
    """
    from dynamic_iv import instrument_indices
    X, Y, I_raw = (np.asarray(a, dtype=float) for a in (X, Y, I_raw))
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2 or Y.ndim != 2 or I_raw.ndim != 2 or Y.shape[1] != 1:
        raise ValueError("Expected 2D X/I and a scalar outcome")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if any(len(a) < n+3 or not np.all(np.isfinite(a)) for a in (X, Y, I_raw)):
        raise ValueError("Need at least n+3 finite observations per series")
    idx = instrument_indices(lam, I_raw.shape[1])
    I = I_raw[:, idx]
    B = np.column_stack((np.ones(n), I[:n]))
    if n <= B.shape[1]:
        raise ValueError("More observations than historical controls are required")
    aligned = np.column_stack((X[2:n+2], Y[3:n+3], I[1:n+1]))
    residual = aligned - B @ np.linalg.lstsq(B, aligned, rcond=None)[0]
    p = X.shape[1]
    return residual[:, :p], residual[:, p:p+1], residual[:, p+1:]


def save_scenario(key, alpha, beta, e, m, save_path, name_id="scen",A = None,X = None,Y = None,I = None):
    """
    Save generated scenario for reproducibility

    Parameters
    ----------
    key
    alpha
    beta
    e
    save_path

    A,X,Y,I : time setting pregenerated Data and utils
    Returns
    -------
    dict: to be saved
    """

    d, p = alpha.shape
    save_dict = {
        "key": key,
        "alpha": alpha,
        "d": d,
        "p": p,
        "beta": beta,
        "conf_e": e,
        "m": m,
        "A":A,
        "X":X,
        "Y":Y,
        "I":I
    }

    name = str(name_id)+ ".npy"
    np.save(os.path.join(save_path, name), save_dict)
    return save_dict



def rescale_transition_matrix(A, target_rho=0.8):
    """
    Rescale transition matrix so that its spectral radius equals target_rho.
    """
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho == 0:
        return A
    return A * target_rho / rho

def generate_clean_time_iv_model(
    n,
    d,
    p,
    beta_true,
    seed=253,
    target_rho=0.8,
    n_confounders=3,
    iv_x_strength=1.0,
    confound_x_strength=1.5,
    confound_y_strength=2.0,
    noise_x=1.0,
    noise_y=1.0,
):
    rng = np.random.default_rng(seed)

    h_dim = n_confounders

    # Generate stable AR(1) instruments and confounders
    I = np.zeros((n, d))
    H = np.zeros((n, h_dim))

    for t in range(1, n):
        I[t] = target_rho * I[t - 1] + rng.normal(0, 1, size=d)
        H[t] = target_rho * H[t - 1] + rng.normal(0, 1, size=h_dim)

    # First-stage coefficients: I -> X
    Pi = rng.normal(0, iv_x_strength / np.sqrt(d), size=(d, p))

    # Confounding coefficients: H -> X
    Lambda = rng.normal(0, confound_x_strength / np.sqrt(h_dim), size=(h_dim, p))

    # H -> Y
    gamma = rng.normal(0, confound_y_strength / np.sqrt(h_dim), size=h_dim)

    # Endogenous regressors
    X = I @ Pi + H @ Lambda + rng.normal(0, noise_x, size=(n, p))

    # Outcome
    Y = X @ beta_true + H @ gamma + rng.normal(0, noise_y, size=n)

    return X, Y.reshape(-1, 1), I, H

