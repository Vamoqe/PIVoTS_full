"""Local identified projections and least-squares reconstruction."""
import numpy as np


def svd_2sls(x, y, z, max_iv=2, eps=1e-8):
    """Return (projected coefficient including intercept, P, I-P).

    eps is an absolute cutoff on singular values divided by sqrt(n). It is
    a numerical rank tolerance, not a weak-instrument validity test. Inputs
    should use a documented common scale. No n-by-n matrix is allocated.
    """
    x, z = np.asarray(x,dtype=float), np.asarray(z,dtype=float)
    y = np.asarray(y,dtype=float).reshape(-1)
    if x.ndim != 2 or z.ndim != 2 or len(x) != len(z) or len(y) != len(x) or not len(x):
        raise ValueError('X, Y and Z must have matching nonempty sample dimensions')
    if not all(np.all(np.isfinite(v)) for v in (x,y,z)) or eps < 0 or max_iv < 0:
        raise ValueError('Inputs and tolerances must be finite and valid')
    n, p = x.shape
    xx = np.column_stack((np.ones(n),x))
    zz = np.column_stack((np.ones(n),z))
    xhat = zz @ np.linalg.lstsq(zz,xx,rcond=None)[0]
    U,D,VT = np.linalg.svd(xhat,full_matrices=False)
    rank = min(int(np.sum(D/np.sqrt(n) > eps)),int(max_iv)+1)
    V = VT[:rank].T
    beta = V @ ((U[:,:rank].T @ y)/D[:rank])
    P = V @ V.T
    return beta, P, np.eye(p+1)-P


def combine_svd_estimation(beta_list, P_list, ridge=0.0):
    """Least-squares reconstruction; rank deficiency returns a minimum-norm
    representative, not an identified full parameter. Fixed ridge is optional
    shrinkage and does not recover unidentified directions.
    """
    if not beta_list or len(beta_list) != len(P_list) or ridge < 0:
        raise ValueError('Need matching nonempty projections and estimates, ridge >= 0')
    p = np.asarray(beta_list[0]).size
    Q = np.vstack(P_list)
    theta = np.concatenate([np.asarray(b).reshape(-1) for b in beta_list])
    if Q.shape != (len(P_list)*p,p) or not np.all(np.isfinite(Q)) or not np.all(np.isfinite(theta)):
        raise ValueError('Invalid projection/estimate shapes or values')
    if ridge:
        Q = np.vstack((Q,np.sqrt(ridge)*np.eye(p)))
        theta = np.concatenate((theta,np.zeros(p)))
    return np.linalg.lstsq(Q,theta,rcond=None)[0]


def check_component_coverage(P_list, eps=1e-8):
    """Angles to the union span; zero projected directions have angle pi/2.

    This is geometric coverage, not confidence interval coverage or IV strength.
    """
    if not P_list or eps < 0:
        raise ValueError('Need projections and a nonnegative rank tolerance')
    P = np.hstack(P_list)
    if P.ndim != 2 or not np.all(np.isfinite(P)):
        raise ValueError('Projection matrices must be finite')
    U,D,_ = np.linalg.svd(P,full_matrices=False)
    basis = U[:,D > eps]
    projection = basis @ basis.T
    # For an orthogonal projector ||P e_j||^2 = P_jj.
    return np.arccos(np.sqrt(np.clip(np.diag(projection),0.0,1.0)))
