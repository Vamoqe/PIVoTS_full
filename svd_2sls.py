
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import logging
import cvxpy as cp
logging.basicConfig(level=20)
sys.path.insert(0, "../src")


def svd_2sls(x, y, z, max_iv=2, eps=0.01):
    """
    $\beta$-estimation estimation of image and nullspace for the underspecified setting

    Parameters
    ----------
    x: ndarray
    y: ndarray
    z: ndarray
    max_iv: int
        maximum dimension of the image space
    eps: float
        cut-off of eigenvalues

    Returns
    -------
    beta: ndarray
    PV: ndarray
        projection matrix on image space
    PV_comp: ndarray
        projection matrix on orthogonal part of image space
    """

    ## First stage regression
    xx = np.hstack([np.ones(x.shape[0])[..., np.newaxis], x])
    reg1 = LinearRegression(fit_intercept=True).fit(z, xx)
    xhat = reg1.predict(z)

    ## Singular value decomposition of the first step
    U, D, VT = np.linalg.svd(xhat)

    ## Preparation of variables, Truncation
    rank_D = np.min([(D > eps).sum(), max_iv + 1])
    U1, D1, VT1 = U[:, :rank_D], D[:rank_D], VT[:rank_D, :]
    VT1_comp = VT[rank_D:, :]
    VT1 = VT[:rank_D, :]

    ## Computation of beta estimate
    beta_est = VT1.T @ np.diag(1 / D1) @ U1.T @ y
    V1_comp = VT1_comp.T
    V1 = VT1.T

    ## Computation of projection matrices for further computation purposes
    def projection(P1):
        return P1@np.linalg.solve(P1.T@P1, P1.T)

    P1_comp = projection(V1_comp)
    P1 = projection(V1)

    return beta_est, P1, P1_comp


def combine_svd_estimation(beta_list, P_list, ridge=1e-8):
    """
    Least-squares aggregation of multiple projected beta estimates.

    Solve:
        min_b sum_r ||P_r b - beta_r||^2 + ridge * ||b||^2

    This is more stable than exact equality constraints in finite samples.
    """
    dim_b = len(beta_list[0])

    lhs = ridge * np.eye(dim_b)
    rhs = np.zeros(dim_b)

    for beta_r, P_r in zip(beta_list, P_list):
        beta_r = np.asarray(beta_r).reshape(-1)
        lhs += P_r.T @ P_r
        rhs += P_r.T @ beta_r

    beta_combined = np.linalg.solve(lhs, rhs)
    return beta_combined

def check_component_coverage(P_list, eps=0.1):
    """
    returns a vector which can be used to compute the coverage of the standard basis

    Parameters
    ----------
    P_list: list of ndarrays

    Returns
    -------
    vec
    """

    P_union = np.hstack([proj for proj in P_list])
    U, D, _ = np.linalg.svd(P_union)
    rank_D = (D > eps).sum()
    # determine a basis for the column space of P_union
    U_col = U[:, :rank_D]
    P_col = U_col@U_col.T
    p, _ = P_col.shape

    # compute the angle between the two vectors
    cover = []
    for jter in np.arange(p):
        p_vec = P_col[:, jter]
        cover.append(p_vec[jter] / np.sqrt(p_vec@p_vec))

    return np.arccos(cover)
