"""Simulation summaries with explicit failure counts and paired comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_result(path):
    directory=Path(path)
    result=np.load(directory/'results.npy',allow_pickle=True).item()
    if result.get('schema_version')!=2:
        raise ValueError('Use schema-v2 results; older results use different identification/settings and must be rerun')
    manifest=json.loads((directory/'manifest.json').read_text())
    if result.get('status')!='complete' or manifest.get('status')!='complete':
        raise ValueError('Refusing to summarize an incomplete/failed run')
    if result.get('config_sha256')!=manifest.get('config_sha256'):
        raise ValueError('Result/manifest configuration mismatch')
    return result


def norm_error(diff_array):
    arr=np.asarray(diff_array,dtype=float)
    if arr.ndim not in (1,2):
        raise ValueError('Errors must be trial-by-coefficient arrays')
    return np.abs(arr) if arr.ndim==1 else np.linalg.norm(arr,axis=1)


def error_statistics(diff_array):
    all_errors=norm_error(diff_array)
    errors=all_errors[np.isfinite(all_errors)]
    n=len(errors); attempted=len(all_errors)
    sd=float(np.std(errors,ddof=1)) if n>1 else np.nan
    return {'Mean':float(np.mean(errors)) if n else np.nan,
            'Median':float(np.median(errors)) if n else np.nan,'SD':sd,
            'MCSE':sd/np.sqrt(n) if n else np.nan,
            'Q25':float(np.quantile(errors,.25)) if n else np.nan,
            'Q75':float(np.quantile(errors,.75)) if n else np.nan,
            'P95':float(np.quantile(errors,.95)) if n else np.nan,
            'P99':float(np.quantile(errors,.99)) if n else np.nan,
            'N':n,'Attempted':attempted,'Failed':attempted-n,
            'FailureRate':(attempted-n)/attempted if attempted else np.nan}


def summarize_errors(res):
    methods=[('Proposed (noisy oracle)','diff_beta'),('Random','diff_benchmark_beta'),
             ('Single subset','diff_first_subset_beta'),('Selected-union IV','diff_singlex_beta'),
             ('Full IV (unrestricted)','diff_full_iv_beta'),('OLS','diff_ols_beta')]
    return pd.DataFrame([dict(Method=name,**error_statistics(res[key])) for name,key in methods if key in res])


def paired_error_comparison(res,left='diff_beta',right='diff_benchmark_beta'):
    a,b=norm_error(res[left]),norm_error(res[right])
    if a.shape!=b.shape:
        raise ValueError('Paired comparisons require one result per shared trial')
    valid=np.isfinite(a)&np.isfinite(b)
    difference=a[valid]-b[valid]
    n=len(difference)
    return {'left':left,'right':right,'attempted':len(a),'paired_finite':n,
            'excluded_pairs':len(a)-n,'mean_difference':float(difference.mean()) if n else np.nan,
            'mcse':float(difference.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan}


def save_latex_table(df,save_path,filename):
    directory=Path(save_path); directory.mkdir(parents=True,exist_ok=True)
    # Avoid pandas' optional Jinja2 dependency for a plain numeric table.
    def cell(value):
        if isinstance(value, (float,np.floating)):
            return f"{value:.4f}" if np.isfinite(value) else "--"
        text=str(value)
        return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    rows=[r"\begin{tabular}{l"+"r"*(len(df.columns)-1)+"}",r"\hline",
          " & ".join(cell(c) for c in df.columns)+r" \\",r"\hline"]
    rows.extend(" & ".join(cell(v) for v in row)+r" \\" for row in df.itertuples(index=False,name=None))
    rows.extend([r"\hline",r"\end{tabular}"])
    (directory/filename).write_text("\n".join(rows)+"\n")


def summarize_underid(base_path):
    rows=[]
    for size,rounds in [(1,15),(3,5),(5,3)]:
        res=load_result(Path(base_path)/f'p10_d15_d_id15_d_max{size}_n_rounds{rounds}')
        rows.append(dict(Setting=f'dmax={size}, R={rounds}',**error_statistics(res['diff_beta'])))
    return pd.DataFrame(rows)


def summarize_rho(base_path):
    rows=[]
    for rho in [.2,.5,.8,.95]:
        tag=str(rho).replace('.','p')
        res=load_result(Path(base_path)/f'rho{tag}'/'p10_d15_d_id15_d_max3_n_rounds5')
        rows.append(dict(Setting=f'rho={rho}',**error_statistics(res['diff_beta'])))
    return pd.DataFrame(rows)


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_path',type=Path,default=root/'Simulation_Results')
    parser.add_argument('--save_path',type=Path,default=root/'Simulation_Tables')
    parser.add_argument('--result_path',type=Path,help='Summarize a single run instead of the full grid')
    args=parser.parse_args(); args.save_path.mkdir(parents=True,exist_ok=True)
    result=load_result(args.result_path or args.base_path/'main'/'p10_d15_d_id15_d_max3_n_rounds5')
    tables={'main_error_summary':summarize_errors(result)}
    if args.result_path is None:
        tables.update(underid_summary=summarize_underid(args.base_path/'underid'),rho_summary=summarize_rho(args.base_path/'rho'))
    for name,table in tables.items():
        table.to_csv(args.save_path/f'table_{name}.csv',index=False)
        save_latex_table(table,args.save_path,f'table_{name}.tex')
        print(table.to_string(index=False))
    pd.DataFrame([paired_error_comparison(result)]).to_csv(args.save_path/'paired_proposed_random.csv',index=False)


if __name__=='__main__': main()
