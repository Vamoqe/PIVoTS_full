"""Pilot-guided block acquisition with explicitly charged observation costs.

Experimental research runner: freezes the acquisition schedule using a paid
pilot, then estimates on a disjoint later time segment. No true coefficients
enter selection. Finite-buffer dependence is not assumed to be independence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from block_iv import fit_gmm, fit_projected_blocks, stack_moments, select_blocks
from dynamic_iv import validate_selected_instrument_history
from run_optimization import _atomic_json, _version_info
from utils import generate_time_scenario, generate_time_model, truncated_residual


def _record(fit,true_beta,scale):
    estimate=fit['estimate']/scale
    full=fit['full_sample_rank']
    return {'estimate':estimate.tolist(),'sample_rank':fit['sample_rank'],
            'full_sample_rank':full,
            'l2_error':float(np.linalg.norm(estimate-true_beta)) if full else None,
            'min_weighted_singular_value':fit.get('min_weighted_singular_value'),
            'condition_number':fit.get('condition_number'),
            'omega_min_eigenvalue':fit.get('omega_min_eigenvalue'),
            'omega_eigenvalue_floor':fit.get('omega_eigenvalue_floor'),
            'omega_regularized':fit.get('omega_regularized'),
            'interpretation':'full_parameter_candidate' if full else 'projection_representative_only'}


def run_pilot_selection(output_dir,n_runs=10,n_pilot=300,n_main=1000,gap=100,
                        p=10,d=15,block_size=3,instrument_budget=12,seed=253,
                        target_rho=.8,hac_bandwidth=None):
    integers=dict(n_runs=n_runs,n_pilot=n_pilot,n_main=n_main,p=p,d=d,
                  block_size=block_size,instrument_budget=instrument_budget)
    if any(not isinstance(v,(int,np.integer)) or v<1 for v in integers.values()):
        raise ValueError('Dimensions, lengths, replications and budget must be positive integers')
    if not isinstance(gap,(int,np.integer)) or gap<0:
        raise ValueError('gap must be a nonnegative integer')
    if d%block_size or not block_size<p or instrument_budget%block_size or instrument_budget>d:
        raise ValueError('Use equal blocks smaller than p, d divisible by block_size, and a divisible budget <= d')
    if n_pilot<=max(p,d)+2 or n_main<=block_size+2:
        raise ValueError('Pilot/main segment too short for this finite-dimensional design')
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    result_path=output/'pilot_results.json'
    if result_path.exists():
        raise FileExistsError('Pilot output exists; choose a new directory')
    blocks=[list(range(i,i+block_size)) for i in range(0,d,block_size)]
    A=generate_time_scenario(None,d,p,d_identify=p,seed=seed,target_rho=target_rho)
    for indices in blocks:
        validate_selected_instrument_history(A,d,p,indices)
    true_beta=A[1][-1,d:d+p]
    pilot_cost=(n_pilot+1)*d
    unit_cost=(n_main+1)*block_size
    acquisition_budget=(n_main+1)*instrument_budget
    config=dict(integers,gap=gap,seed=seed,target_rho=target_rho,hac_bandwidth=hac_bandwidth,
                conditioning='own_block_I[t-3]',persistence_mode='diagonal',initialization='stationary',
                cost_unit='one instrument variable at one observed time',
                cost_scope='IV observations including one required history row; common X/Y costs excluded',
                comparison_scope='all selectors receive and pay for the same broad pilot',
                pilot_cost=pilot_cost,main_budget=acquisition_budget,total_budget=pilot_cost+acquisition_budget,
                standardization='pilot regressor standard deviations, shared across blocks and methods',
                schedule='pilot_frozen_before_main_segment',score_ridge=1e-6,
                uncertainty='point_estimation_only; no post-selection confidence intervals')
    version=_version_info()
    for name in ('block_iv.py','run_pilot_selection.py'):
        version['source_sha256'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    result={'schema_version':1,'status':'running','config':config,'version':version,
            'beta':true_beta.tolist(),'A':A[1].tolist(),'blocks':blocks,'trials':[]}
    _atomic_json(result_path,result)
    try:
        for trial in range(n_runs):
            data_seed=seed+10000+trial
            # Required n+3 alignment rows are not shared between the two segments.
            main_start=n_pilot+3+gap
            X,Y,I=generate_time_model(A,d,p,main_start+n_main,seed=data_seed)
            scale=np.std(X[2:n_pilot+2],axis=0,ddof=1)
            if np.any(scale<=1e-12):
                raise ValueError('Pilot regressor scale is zero')
            pilot=[]
            for indices in blocks:
                x,y,z=truncated_residual(X,Y,I,n_pilot,indices)
                pilot.append((x/scale,y.ravel(),z))
            initial=fit_gmm(pilot,hac_bandwidth)
            if not initial['full_sample_rank']:
                raise ValueError('Pilot moments do not support a full-rank initial estimate')
            G,h,rows=stack_moments(pilot)
            schedules={}
            for method in ('information','random','strength'):
                schedules[method]=select_blocks(G,initial['omega'],rows,
                    [unit_cost]*len(blocks),acquisition_budget,method=method,seed=data_seed+500000)
            # Only now access main data. Full data exists solely inside the simulator.
            main_cache={}
            def main_block(m):
                if m not in main_cache:
                    x,y,z=truncated_residual(X[main_start:],Y[main_start:],I[main_start:],n_main,blocks[m])
                    main_cache[m]=(x/scale,y.ravel(),z)
                return main_cache[m]
            trial_result={'trial_id':trial,'data_seed':data_seed,'regressor_scale':scale.tolist(),'methods':{}}
            for method,(chosen,trajectory) in schedules.items():
                selected_data=[main_block(m) for m in chosen]
                fit=fit_gmm(selected_data,hac_bandwidth)
                projected=fit_projected_blocks(selected_data)
                spent=unit_cost*len(chosen)
                trial_result['methods'][method]={**_record(fit,true_beta,scale),
                    'selected_blocks':chosen,'selection_trajectory':trajectory,
                    'pilot_cost':pilot_cost,'main_cost':spent,'total_cost':pilot_cost+spent,
                    'within_budget':spent<=acquisition_budget,
                    'same_data_projection':_record(projected,true_beta,scale)}
            full=fit_gmm([main_block(m) for m in range(len(blocks))],hac_bandwidth)
            trial_result['methods']['full_gmm_reference']={**_record(full,true_beta,scale),
                'selected_blocks':list(range(len(blocks))),'pilot_cost':pilot_cost,
                'main_cost':unit_cost*len(blocks),'total_cost':pilot_cost+unit_cost*len(blocks),
                'within_budget':unit_cost*len(blocks)<=acquisition_budget,
                'role':'unrestricted_reference_not_an_error_bound'}
            result['trials'].append(trial_result)
            _atomic_json(result_path,result)
        result['status']='complete'
        result['summary']={}
        for method in result['trials'][0]['methods']:
            records=[trial['methods'][method] for trial in result['trials']]
            errors=np.array([r['l2_error'] for r in records if r['l2_error'] is not None],float)
            result['summary'][method]={'attempted':n_runs,'full_sample_rank_count':len(errors),
                'under_rank_count':n_runs-len(errors),
                'error_summary_population':'conditional_on_full_sample_rank',
                'mean_l2_error':float(errors.mean()) if len(errors) else None,
                'rmse':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                'p95_l2_error':float(np.quantile(errors,.95)) if len(errors) else None,
                'mean_total_cost':float(np.mean([r['total_cost'] for r in records]))}
        result['paired_comparisons']={}
        for comparator in ('random','strength'):
            pairs=[(t['methods']['information']['l2_error'],t['methods'][comparator]['l2_error']) for t in result['trials']]
            differences=np.array([a-b for a,b in pairs if a is not None and b is not None])
            result['paired_comparisons'][comparator]={'attempted':n_runs,'common_full_sample_rank':len(differences),
                'excluded_pairs':n_runs-len(differences),
                'mean_l2_difference_information_minus_comparator':float(differences.mean()) if len(differences) else None,
                'mcse':float(differences.std(ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None}
        _atomic_json(result_path,result)
    except Exception as error:
        result['status']='failed'; result['failure']=f'{type(error).__name__}: {error}'
        _atomic_json(result_path,result)
        raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    for name,default in [('n_runs',10),('n_pilot',300),('n_main',1000),('gap',100),('p',10),('d',15),('block_size',3),('instrument_budget',12),('seed',253)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--target_rho',type=float,default=.8)
    parser.add_argument('--hac_bandwidth',type=int)
    args=vars(parser.parse_args()); args['output_dir']=args.pop('output')
    result=run_pilot_selection(**args)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__': main()
