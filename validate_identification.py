"""Exact population checks and a reproducible, bounded convergence experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from dynamic_iv import population_moments
from utils import generate_time_scenario, generate_time_model, truncated_residual
from svd_2sls import svd_2sls, combine_svd_estimation


def validate(p=3, d=6, seed=253, sample_sizes=(300,1500,6000), replications=12):
    if replications < 2 or len(sample_sizes) < 2 or min(sample_sizes) <= d+1:
        raise ValueError('Need >=2 replications and >=2 sample sizes larger than d+1')
    A = generate_time_scenario(None,d,p,d_identify=p,seed=seed)
    good = population_moments(A,d,p,list(range(d)))
    bad = population_moments(A,d,p,list(range(d)),conditioning='x_history')
    local = population_moments(A,d,p,[0])
    G = local['G']; projection = np.linalg.pinv(G) @ G
    local_error = np.linalg.norm(np.linalg.pinv(G) @ local['h'] - projection @ local['beta'])
    errors = {int(n): [] for n in sample_sizes}
    projected_errors = {int(n): [] for n in sample_sizes}
    block_size=max(1,p-1)
    blocks=[list(range(start,min(start+block_size,d))) for start in range(0,d,block_size)]
    seeds = [seed+10000+i for i in range(replications)]
    for trial_seed in seeds:
        X,Y,I = generate_time_model(A,d,p,max(sample_sizes),seed=trial_seed)
        for n in sample_sizes:
            x,y,z = truncated_residual(X,Y,I,int(n),list(range(d)))
            bhat,_,_ = svd_2sls(x,y,z,max_iv=d)
            errors[int(n)].append(float(np.linalg.norm(bhat-np.r_[0,good['beta']])))
            local_estimates=[]; projections=[]
            for indices in blocks:
                bx,by,bz=truncated_residual(X,Y,I,int(n),indices)
                local_beta,P,_=svd_2sls(bx,by,bz,max_iv=len(indices))
                local_estimates.append(local_beta); projections.append(P)
            combined=combine_svd_estimation(local_estimates,projections)
            projected_errors[int(n)].append(float(np.linalg.norm(combined-np.r_[0,good['beta']])) )
    curve = [{'n': n, 'rmse': float(np.sqrt(np.mean(np.square(values)))),
              'mean_l2_error': float(np.mean(values)), 'errors': values,
              'projected_rmse':float(np.sqrt(np.mean(np.square(projected_errors[n])))),
              'projected_errors':projected_errors[n]}
             for n,values in sorted(errors.items())]
    checks = {'population_moment_zero': bool(np.linalg.norm(good['moment']) < 1e-10),
              'wrong_control_detected': bool(np.linalg.norm(bad['moment']) > 1e-5),
              'population_full_rank': good['rank'] == p,
              'local_projection_identity': bool(local_error < 1e-10),
              'large_sample_rmse_lower': curve[-1]['rmse'] < curve[0]['rmse'],
              'large_sample_projected_rmse_lower':curve[-1]['projected_rmse']<curve[0]['projected_rmse']}
    return {'schema_version': 1, 'settings': {'p':p,'d':d,'seed':seed,'replications':replications},
            'conditioning': 'selected_I[t-3]', 'initialization': 'stationary_lyapunov',
            'valid_moment_norm': float(np.linalg.norm(good['moment'])),
            'invalid_x_history_moment_norm': float(np.linalg.norm(bad['moment'])),
            'population_rank': good['rank'], 'local_projection_error': float(local_error),
            'trial_seeds': seeds, 'convergence': curve, 'checks': checks,
            'passed': all(checks.values()),
            'scope': 'Diagnostic for this specified DGP; not a general consistency theorem.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replications',type=int,default=12)
    parser.add_argument('--sample_sizes',type=int,nargs='+',default=[300,1500,6000])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    result = validate(replications=args.replications,sample_sizes=args.sample_sizes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['valid_moment_norm','invalid_x_history_moment_norm','population_rank','checks','passed']},indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
