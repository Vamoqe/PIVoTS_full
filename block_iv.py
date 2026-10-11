"""Finite-candidate block IV moments, HAC weighting and pilot-only selection.

All blocks target the same coefficient vector on the same time scale. Block
validity is an input assumption. Numerical rank is not a weak-IV test. No
confidence intervals or optimal-greedy guarantees are provided here.
"""
import numpy as np


def stack_moments(blocks):
    if not blocks:
        raise ValueError('At least one block is required')
    G,h,slices = [],[],[]
    n,p = np.asarray(blocks[0][0]).shape
    offset=0
    for x,y,z in blocks:
        x,y,z=np.asarray(x,float),np.asarray(y,float).reshape(-1),np.asarray(z,float)
        if x.shape != (n,p) or y.shape != (n,) or z.ndim != 2 or len(z) != n or z.shape[1] < 1:
            raise ValueError('Blocks need matching times, parameter dimensions and nonempty instruments')
        if not all(np.all(np.isfinite(a)) for a in (x,y,z)):
            raise ValueError('Block data must be finite')
        G.append(z.T@x/n); h.append(z.T@y/n)
        slices.append(np.arange(offset,offset+z.shape[1])); offset+=z.shape[1]
    return np.vstack(G),np.concatenate(h),slices


def moment_series(blocks,beta):
    return np.column_stack([np.asarray(z)*(np.asarray(y).reshape(-1)-np.asarray(x)@beta)[:,None]
                            for x,y,z in blocks])


def hac_covariance(values,bandwidth=None):
    """Joint Bartlett HAC, including cross-block and serial covariance."""
    f=np.asarray(values,float)
    if f.ndim != 2 or len(f)<2 or not np.all(np.isfinite(f)):
        raise ValueError('HAC needs a finite time-by-moment matrix with >=2 rows')
    n=len(f)
    if bandwidth is None:
        bandwidth=min(n-1,int(np.floor(4*(n/100)**(2/9))))
    if not isinstance(bandwidth,(int,np.integer)) or not 0 <= bandwidth < n:
        raise ValueError('HAC bandwidth must be an integer between 0 and n-1')
    f=f-f.mean(axis=0)
    omega=f.T@f/n
    for lag in range(1,bandwidth+1):
        gamma=f[lag:].T@f[:-lag]/n
        omega+=(1-lag/(bandwidth+1))*(gamma+gamma.T)
    return (omega+omega.T)/2


def covariance_inverse(omega,relative_floor=1e-10):
    eigen,vectors=np.linalg.eigh((omega+omega.T)/2)
    floor=max(float(eigen.max())*relative_floor,1e-12)
    regularized=np.maximum(eigen,floor)
    return (vectors/regularized)@vectors.T


def fit_gmm(blocks,bandwidth=None,rank_tol=1e-9):
    """Two-step GMM point estimate. Underrank returns a projection representative.

    Weights use estimated joint HAC. The returned sample_rank is numerical and
    cannot certify population identification. Constant moments are omitted:
    inputs are centered historical residuals and the target consists of slopes.
    """
    G,h,_=stack_moments(blocks)
    initial=np.linalg.lstsq(G,h,rcond=rank_tol)[0]
    omega=hac_covariance(moment_series(blocks,initial),bandwidth)
    W=covariance_inverse(omega)
    eigen,vectors=np.linalg.eigh(W)
    root=(vectors*np.sqrt(np.maximum(eigen,0)))@vectors.T
    weighted=root@G
    estimate,_,rank,singular=np.linalg.lstsq(weighted,root@h,rcond=rank_tol)
    identified=int(rank)==G.shape[1]
    minimum=float(singular[-1]) if identified else 0.0
    return {'estimate':estimate,'sample_rank':int(rank),'full_sample_rank':identified,
            'estimand':'full_parameter_candidate' if identified else 'identified_projection_representative',
            'min_weighted_singular_value':minimum,
            'condition_number':float(singular[0]/minimum) if minimum>0 else None,
            'omega_min_eigenvalue':float(np.linalg.eigvalsh(omega).min()),
            'omega_eigenvalue_floor':max(float(np.linalg.eigvalsh(omega).max())*1e-10,1e-12),
            'omega_regularized':bool(np.linalg.eigvalsh(omega).min()<max(float(np.linalg.eigvalsh(omega).max())*1e-10,1e-12)),
            'G':G,'h':h,'omega':omega,'weight':W}


def information_matrix(G,omega,rows):
    if not len(rows):
        return np.zeros((G.shape[1],G.shape[1]))
    D=G[rows]
    return D.T@covariance_inverse(omega[np.ix_(rows,rows)])@D


def select_blocks(G,omega,block_rows,costs,budget,method='information',seed=0,ridge=1e-6):
    """Greedy acquisition schedule based entirely on pilot summaries.

    Information uses logdet increments/cost. It is not claimed submodular or
    globally optimal. Strength uses each block's standalone trace/cost.
    """
    costs=np.asarray(costs,float)
    if len(costs)!=len(block_rows) or not np.all(np.isfinite(costs)) or np.any(costs<=0):
        raise ValueError('Each block must have a finite positive incremental cost')
    if not np.isfinite(budget) or budget<0 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError('Need finite nonnegative budget and positive score ridge')
    if method not in {'information','random','strength'}:
        raise ValueError('Unknown selection method')
    selected,trajectory=[],[]
    spent=0.; rng=np.random.default_rng(seed); eye=np.eye(G.shape[1])
    while True:
        affordable=[m for m in range(len(costs)) if m not in selected and spent+costs[m]<=budget+1e-10]
        if not affordable:
            break
        rows=np.concatenate([block_rows[m] for m in selected]).astype(int) if selected else np.array([],dtype=int)
        old=information_matrix(G,omega,rows)
        old_log=np.linalg.slogdet(old+ridge*eye)[1]
        scores=[]
        for m in affordable:
            if method=='random':
                score=rng.uniform()
            elif method=='strength':
                score=np.trace(information_matrix(G,omega,block_rows[m]))/costs[m]
            else:
                new=information_matrix(G,omega,np.r_[rows,block_rows[m]])
                score=(np.linalg.slogdet(new+ridge*eye)[1]-old_log)/costs[m]
            scores.append(float(score))
        best=int(np.argmax(scores)); m=affordable[best]
        selected.append(m); spent+=costs[m]
        trajectory.append({'block':m,'cost':float(spent),'score':scores[best]})
    return selected,trajectory


def fit_projected_blocks(blocks,rank_tol=1e-9):
    """Equal-weight projection integration for same-data estimator comparisons."""
    projections,parameters=[],[]
    for block in blocks:
        G,h,_=stack_moments([block])
        inverse=np.linalg.pinv(G,rcond=rank_tol)
        projections.append(inverse@G); parameters.append(inverse@h)
    Q=np.vstack(projections); theta=np.concatenate(parameters)
    beta,_,rank,_=np.linalg.lstsq(Q,theta,rcond=rank_tol)
    return {'estimate':beta,'sample_rank':int(rank),'full_sample_rank':int(rank)==Q.shape[1]}
