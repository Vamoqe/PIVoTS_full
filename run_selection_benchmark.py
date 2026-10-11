"""Compare paid pilot acquisition with genuinely pilot-free random acquisition.

All arms have existing X/Y records for the same calendar horizon. Only IV
observations are charged. True population moments are used for evaluation,
never for choosing a schedule. This is a finite-model benchmark, not inference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_reporting import summarize_trials, write_benchmark_tables
from block_iv import fit_gmm, select_blocks, stack_moments
from dynamic_iv import population_moments, validate_selected_instrument_history
from run_optimization import _atomic_json, _version_info
from utils import generate_time_model, generate_time_scenario, truncated_residual

SCENARIOS = ('dense', 'heterogeneous', 'redundant')
METHODS = ('no_pilot_random_same_n', 'no_pilot_random_equal_budget',
           'pilot_random', 'pilot_information', 'pilot_strength')
PRIMARY = 'no_pilot_random_equal_budget'
RANK_TOL = 1e-9


def _positive_integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def observation_plan(n_pilot, n_main, gap, d, instrument_budget):
    for name, value in [('n_pilot', n_pilot), ('n_main', n_main), ('d', d),
                        ('instrument_budget', instrument_budget)]:
        _positive_integer(name, value)
    _positive_integer('gap', gap, 0)
    if instrument_budget > d:
        raise ValueError('Instrument capacity cannot exceed available instruments')
    pilot_cost = (n_pilot + 1) * d
    total = pilot_cost + (n_main + 1) * instrument_budget
    horizon = n_pilot + n_main + gap + 6
    affordable_n = total // instrument_budget - 1
    return {'pilot_cost': pilot_cost, 'total_budget': total,
            'calendar_rows': horizon, 'pilot_main_start': n_pilot + 3 + gap,
            'no_pilot_equal_budget_n': min(affordable_n, horizon - 3),
            'no_pilot_budget_only_n': affordable_n,
            'no_pilot_calendar_capped': affordable_n > horizon - 3}


def build_scenario(scenario, p, d, block_size, seed, target_rho=.8):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    for name, value in [('p', p), ('d', d), ('block_size', block_size)]:
        _positive_integer(name, value)
    _positive_integer('seed', seed, 0)
    if d % block_size or block_size >= p or d < p:
        raise ValueError('Use complete equal blocks smaller than p and d >= p')
    A = generate_time_scenario(None, d, p, d_identify=p, seed=seed,
                               target_rho=target_rho)
    alpha = A[1][d:d+p, :d].copy()
    original_norm = float(np.linalg.norm(alpha))
    metadata = {'definition': scenario, 'duplicated_blocks': [],
                'block_strength_factors': [1.] * (d // block_size)}
    if scenario == 'heterogeneous':
        rng = np.random.default_rng(np.random.SeedSequence([seed, 911]))
        factors = rng.permutation(np.geomspace(.15, 1., d // block_size))
        alpha *= np.repeat(factors, block_size)[None, :]
        metadata['block_strength_factors'] = factors.tolist()
    elif scenario == 'redundant':
        if d < 2 * block_size or d - block_size < p:
            raise ValueError('Redundant scenario needs d - block_size >= p')
        alpha[:, block_size:2*block_size] = alpha[:, :block_size]
        metadata['duplicated_blocks'] = [0, 1]
    # Hold total first-stage squared coefficient magnitude fixed across scenarios.
    alpha *= original_norm / np.linalg.norm(alpha)
    A[1][d:d+p, :d] = alpha
    metadata.update(first_stage_frobenius_norm=float(np.linalg.norm(alpha)),
                    normalization='same Frobenius norm as dense with this DGP seed')
    return A, metadata


def population_identification(G_blocks, selected, p):
    if not selected:
        return {'population_rank': 0, 'population_full_rank': False,
                'population_min_singular_value': 0., 'population_condition_number': None}
    G = np.vstack([G_blocks[m] for m in selected])
    singular = np.linalg.svd(G, compute_uv=False)
    rank = int(np.count_nonzero(singular > RANK_TOL * singular[0])) if singular[0] > 0 else 0
    full = rank == p
    minimum = float(singular[-1]) if full else 0.
    return {'population_rank': rank, 'population_full_rank': full,
            'population_min_singular_value': minimum,
            'population_condition_number': float(singular[0] / minimum) if minimum > 0 else None}


def random_schedule(n_blocks, n_selected, selection_seed):
    """Choose blocks before observing either pilot or main data."""
    return np.random.default_rng(selection_seed).permutation(n_blocks)[:n_selected].tolist()


def _scale(X, start, n):
    scale = np.std(X[start+2:start+n+2], axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError('Regressor scale must be finite and positive')
    return scale


def _blocks(X, Y, I, indices, chosen, start, n, scale):
    segment = slice(start, start+n+3)
    out = []
    for m in chosen:
        # The estimator receives only the instruments actually purchased.
        local_I = I[segment, :][:, indices[m]]
        x, y, z = truncated_residual(X[segment], Y[segment], local_I,
                                     n, list(range(len(indices[m]))))
        out.append((x / scale, y.ravel(), z))
    return out


def _cost_record(chosen, n, start, paid_pilot, plan, block_size, scale_source):
    main_cost = (n+1) * len(chosen) * block_size if chosen else 0
    total = paid_pilot + main_cost
    return {'selected_blocks': chosen, 'n_main': n, 'pilot_cost': paid_pilot,
            'main_cost': main_cost, 'total_cost': total,
            'unused_budget': plan['total_budget'] - total,
            'within_budget': total <= plan['total_budget'],
            'calendar_start': start, 'calendar_stop': start+n+3,
            'within_calendar': start+n+3 <= plan['calendar_rows'],
            'scale_source': scale_source}


def _failed_record(base, error):
    return {**base, 'status': 'failed', 'failure': f'{type(error).__name__}: {error}',
            'estimate': None, 'l2_error': None, 'sample_full_rank': None,
            'population_full_rank': base.get('population_full_rank')}


def _estimate_arm(X, Y, I, indices, chosen, start, n, scale, true_beta,
                  G_blocks, paid_pilot, plan, scale_source, block_size):
    base = _cost_record(chosen, n, start, paid_pilot, plan, block_size, scale_source)
    base.update(population_identification(G_blocks, chosen, len(true_beta)))
    try:
        if scale is None:
            scale = _scale(X, start, n)
        fit = fit_gmm(_blocks(X, Y, I, indices, chosen, start, n, scale),
                      bandwidth=plan['hac_bandwidth'])
        estimate = fit['estimate'] / scale
        if not np.all(np.isfinite(estimate)):
            raise ValueError('Nonfinite GMM estimate')
        identified = base['population_full_rank'] and fit['full_sample_rank']
        return {**base, 'status': 'ok' if identified else 'underidentified',
                'failure': None, 'estimate': estimate.tolist(),
                'sample_full_rank': fit['full_sample_rank'], 'sample_rank': fit['sample_rank'],
                'regressor_scale': scale.tolist(),
                'l2_error': float(np.linalg.norm(estimate - true_beta)) if identified else None,
                'estimand': 'full_parameter_candidate' if identified else 'representative_only',
                'sample_min_weighted_singular_value': fit['min_weighted_singular_value'],
                'omega_regularized': fit['omega_regularized']}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _failed_record(base, error)


def run_selection_benchmark(output_dir, n_runs=30, n_pilot=300, n_main=1000, gap=100,
                            p=10, d=15, block_size=3, instrument_budget=12,
                            dgp_seeds=(253, 517, 809), scenarios=SCENARIOS,
                            target_rho=.8, hac_bandwidth=None):
    for name, value in [('n_runs', n_runs), ('p', p), ('d', d), ('block_size', block_size)]:
        _positive_integer(name, value)
    plan = observation_plan(n_pilot, n_main, gap, d, instrument_budget)
    if d % block_size or instrument_budget % block_size or block_size >= p or d < p:
        raise ValueError('Require equal blocks < p, d >= p, and divisible d/instrument_budget')
    if n_pilot <= max(p, d)+2 or n_main <= max(p, block_size)+2:
        raise ValueError('Pilot/main segments too short for the configured dimensions')
    seeds, scenarios = list(dgp_seeds), list(scenarios)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide nonempty distinct DGP seeds')
    for seed in seeds:
        _positive_integer('dgp_seed', seed, 0)
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError('Provide nonempty distinct supported scenarios')
    if 'redundant' in scenarios and d-block_size < p:
        raise ValueError('Redundant scenario requires d - block_size >= p')
    if hac_bandwidth is not None:
        _positive_integer('hac_bandwidth', hac_bandwidth, 0)
        if hac_bandwidth >= min(n_pilot, n_main):
            raise ValueError('HAC bandwidth must be shorter than every segment')
    plan['hac_bandwidth'] = hac_bandwidth
    config = dict(n_runs=n_runs, n_pilot=n_pilot, n_main=n_main, gap=gap, p=p, d=d,
                  block_size=block_size, instrument_budget=instrument_budget,
                  dgp_seeds=seeds, scenarios=scenarios, target_rho=target_rho,
                  hac_bandwidth=hac_bandwidth, observation_plan=plan, primary=PRIMARY,
                  cost_unit='one IV variable at one time, including required history',
                  xy_assumption='full-horizon X/Y are existing records available to all arms; zero incremental X/Y cost',
                  capacity_scope='instrument_budget caps formal IV count; broad pilot temporarily observes all d',
                  calendar='common total horizon; no-pilot starts immediately, pilot main starts after pilot and gap',
                  randomization='shared Random schedule drawn independently before observing data',
                  standardization='pilot arms use pilot X; no-pilot arms use their own main X',
                  selection_score='pilot joint-HAC regularized logdet increment per cost; ridge=1e-6',
                  scenario_strength='first-stage Frobenius norm fixed across scenarios for each DGP seed',
                  rank_tolerance=RANK_TOL,
                  uncertainty='conditional Monte Carlo errors within fixed DGP; no post-selection confidence intervals',
                  checkpoint_interval=10)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path, protocol_path = root/'benchmark_results.json', root/'protocol.json'
    if any((root/name).exists() for name in ('benchmark_results.json', 'protocol.json',
                                            'method_summary.csv', 'paired_comparisons.csv')):
        raise FileExistsError('Benchmark output exists; choose a new directory')
    version = _version_info()
    for name in ('block_iv.py', 'run_selection_benchmark.py', 'benchmark_reporting.py'):
        version['source_sha256'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    result = {'schema_version': 1, 'kind': 'no_pilot_cost_benchmark', 'status': 'running',
              'config': config, 'config_sha256': config_hash, 'version': version, 'scenarios': []}
    # Freeze the protocol before examining any simulated outcomes.
    _atomic_json(protocol_path, {'config': config, 'config_sha256': config_hash, 'version': version})
    _atomic_json(result_path, result)
    indices = [list(range(i, i+block_size)) for i in range(0, d, block_size)]
    n_selected = instrument_budget // block_size
    try:
        for scenario in scenarios:
            for dgp_seed in seeds:
                A, metadata = build_scenario(scenario, p, d, block_size, dgp_seed, target_rho)
                G_blocks, moment_norms = [], []
                for block in indices:
                    validate_selected_instrument_history(A, d, p, block)
                    pop = population_moments(A, d, p, block)
                    G_blocks.append(pop['G'])
                    moment_norms.append(float(np.linalg.norm(pop['moment'])))
                all_identification = population_identification(G_blocks, list(range(len(indices))), p)
                if not all_identification['population_full_rank']:
                    raise ValueError('Candidate union must identify the whole parameter')
                true_beta = A[1][-1, d:d+p]
                entry = {'scenario': scenario, 'dgp_seed': dgp_seed, 'A': A[1].tolist(),
                         'beta': true_beta.tolist(), 'blocks': indices, 'metadata': metadata,
                         'population_union': all_identification,
                         'population_moment_norm_by_block': moment_norms,
                         'population_G_by_block': [G.tolist() for G in G_blocks], 'trials': []}
                result['scenarios'].append(entry)
                for trial in range(n_runs):
                    data_seed = int(np.random.SeedSequence([dgp_seed, trial, 10000]).generate_state(1)[0])
                    selection_seed = int(np.random.SeedSequence([dgp_seed, trial, 500000]).generate_state(1)[0])
                    chosen_random = random_schedule(len(indices), n_selected, selection_seed)
                    X, Y, I = generate_time_model(A, d, p, plan['calendar_rows']-3, seed=data_seed)
                    records = {}
                    for method, length in [('no_pilot_random_same_n', n_main),
                                           (PRIMARY, plan['no_pilot_equal_budget_n'])]:
                        records[method] = _estimate_arm(X, Y, I, indices, chosen_random, 0, length,
                            None, true_beta, G_blocks, 0, plan, 'own_main', block_size)
                    pilot_scale = None
                    pilot_scale_error = None
                    try:
                        pilot_scale = _scale(X, 0, n_pilot)
                    except (ValueError, FloatingPointError) as error:
                        pilot_scale_error = error
                    if pilot_scale_error is None:
                        records['pilot_random'] = _estimate_arm(X, Y, I, indices, chosen_random,
                            plan['pilot_main_start'], n_main, pilot_scale, true_beta, G_blocks,
                            plan['pilot_cost'], plan, 'pilot', block_size)
                    else:
                        base = _cost_record([], 0, plan['pilot_main_start'], plan['pilot_cost'],
                                            plan, block_size, 'pilot')
                        records['pilot_random'] = _failed_record(base, pilot_scale_error)
                    schedules, selector_error = {}, pilot_scale_error
                    if selector_error is None:
                        try:
                            pilot = _blocks(X, Y, I, indices, list(range(len(indices))),
                                            0, n_pilot, pilot_scale)
                            initial = fit_gmm(pilot, bandwidth=hac_bandwidth)
                            if not initial['full_sample_rank']:
                                raise ValueError('Pilot has insufficient sample rank for HAC initial estimate')
                            G, _, rows = stack_moments(pilot)
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                            selector_error = error
                    selection_errors = {}
                    if selector_error is None:
                        unit_cost = (n_main+1)*block_size
                        for rule in ('information', 'strength'):
                            try:
                                schedules[rule] = select_blocks(G, initial['omega'], rows,
                                    [unit_cost]*len(indices), (n_main+1)*instrument_budget,
                                    method=rule, seed=selection_seed)
                            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                                selection_errors[rule] = error
                    for rule in ('information', 'strength'):
                        method = 'pilot_'+rule
                        if rule not in schedules:
                            base = _cost_record([], 0, plan['pilot_main_start'], plan['pilot_cost'],
                                                plan, block_size, 'pilot')
                            records[method] = _failed_record(base, selection_errors.get(rule, selector_error))
                        else:
                            chosen, trajectory = schedules[rule]
                            records[method] = _estimate_arm(X, Y, I, indices, chosen,
                                plan['pilot_main_start'], n_main, pilot_scale, true_beta, G_blocks,
                                plan['pilot_cost'], plan, 'pilot', block_size)
                            records[method]['selection_trajectory'] = trajectory
                    entry['trials'].append({'trial_id': trial, 'data_seed': data_seed,
                                             'selection_seed': selection_seed, 'methods': records})
                    if (trial+1) % config['checkpoint_interval'] == 0:
                        _atomic_json(result_path, result)
                entry['summary'] = summarize_trials(entry['trials'], primary=PRIMARY)
                # A secondary comparison isolates selection rules within the paid pilot regime.
                entry['pilot_rule_comparisons'] = summarize_trials(entry['trials'], primary='pilot_random')['paired_comparisons']
                _atomic_json(result_path, result)
                print(f'{scenario}, DGP seed {dgp_seed}: {n_runs} trials complete', flush=True)
        result['status'] = 'complete'
        _atomic_json(result_path, result)
        write_benchmark_tables(result, root)
    except Exception as error:
        result['status'] = 'failed'
        result['failure'] = f'{type(error).__name__}: {error}'
        _atomic_json(result_path, result)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, default in [('n_runs', 30), ('n_pilot', 300), ('n_main', 1000), ('gap', 100),
                          ('p', 10), ('d', 15), ('block_size', 3), ('instrument_budget', 12)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--dgp_seeds', type=int, nargs='+', default=[253, 517, 809])
    parser.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=list(SCENARIOS))
    parser.add_argument('--target_rho', type=float, default=.8)
    parser.add_argument('--hac_bandwidth', type=int)
    args = vars(parser.parse_args())
    args['output_dir'] = args.pop('output')
    result = run_selection_benchmark(**args)
    print(f"Saved {len(result['scenarios'])} scenario/seed groups to {args['output_dir']}")


if __name__ == '__main__':
    main()
