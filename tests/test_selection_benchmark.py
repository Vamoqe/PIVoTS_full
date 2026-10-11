"""Regression checks for cost-fair, calendar-limited acquisition comparisons."""
import itertools
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dynamic_iv import population_moments
import run_selection_benchmark as benchmark
from run_selection_benchmark import (
    build_scenario,
    observation_plan,
    run_selection_benchmark,
)


METHODS = (
    'no_pilot_random_same_n',
    'no_pilot_random_equal_budget',
    'pilot_random',
    'pilot_information',
    'pilot_strength',
)


class ObservationPlanTests(unittest.TestCase):
    def test_same_budget_buys_more_main_observations_without_pilot(self):
        plan = observation_plan(300, 1000, 100, 15, 12)
        self.assertEqual(plan['pilot_cost'], 4515)
        self.assertEqual(plan['total_budget'], 16527)
        self.assertEqual(plan['no_pilot_equal_budget_n'], 1376)
        self.assertEqual(plan['pilot_main_start'], 403)
        self.assertEqual(plan['calendar_rows'], 1406)
        self.assertLessEqual((plan['no_pilot_equal_budget_n'] + 1) * 12,
                             plan['total_budget'])

    def test_saved_pilot_cost_cannot_extend_the_common_calendar(self):
        plan = observation_plan(60, 80, 0, 6, 2)
        self.assertEqual(plan['calendar_rows'], 146)
        self.assertEqual(plan['no_pilot_equal_budget_n'], 143)
        self.assertLess((143 + 1) * 2, plan['total_budget'])


class ScenarioTests(unittest.TestCase):
    def test_scenarios_change_geometry_without_changing_total_strength(self):
        p, d, block_size = 3, 6, 2
        scenarios = {name: build_scenario(name, p, d, block_size, seed=81)[0]
                     for name in ('dense', 'heterogeneous', 'redundant')}
        dense = scenarios['dense'][1]
        mask = np.ones(dense.shape, dtype=bool)
        mask[d:d + p, :d] = False
        total_strength = np.linalg.norm(dense[d:d + p, :d], ord='fro')
        for name, A in scenarios.items():
            with self.subTest(scenario=name):
                np.testing.assert_array_equal(A[1][mask], dense[mask])
                self.assertAlmostEqual(np.linalg.norm(A[1][d:d + p, :d], ord='fro'),
                                       total_strength, places=11)
                moments = population_moments(A, d, p, list(range(d)))
                self.assertEqual(moments['rank'], p)
                self.assertLess(np.linalg.norm(moments['moment']), 1e-10)
                self.assertLess(max(abs(np.linalg.eigvals(A[1]))), 1)

    def test_redundant_blocks_can_fail_identification_despite_enough_instruments(self):
        p, d, block_size = 3, 6, 2
        A, metadata = build_scenario('redundant', p, d, block_size, seed=81)
        blocks = [list(range(start, start + block_size))
                  for start in range(0, d, block_size)]
        ranks = [population_moments(A, d, p, sum((blocks[i] for i in chosen), []))['rank']
                 for chosen in itertools.combinations(range(len(blocks)), 2)]
        self.assertTrue(any(rank < p for rank in ranks))
        self.assertTrue(any(rank == p for rank in ranks))


class SelectionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.kwargs = dict(n_runs=2, n_pilot=60, n_main=80, gap=30,
                          p=3, d=6, block_size=2, instrument_budget=4,
                          dgp_seeds=(81,), scenarios=('dense', 'redundant'))
        cls.result = run_selection_benchmark(cls.directory.name + '/first', **cls.kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_paired_runs_are_reproducible(self):
        repeated = run_selection_benchmark(self.directory.name + '/repeat', **self.kwargs)
        self.assertEqual(self.result['status'], 'complete')
        self.assertEqual(self.result['scenarios'], repeated['scenarios'])

    def test_random_arms_share_selection_and_pay_only_for_observed_data(self):
        plan = observation_plan(60, 80, 30, 6, 4)
        self.assertEqual(len(self.result['scenarios']), 2)
        for scenario in self.result['scenarios']:
            self.assertEqual(len(scenario['trials']), 2)
            for trial in scenario['trials']:
                methods = trial['methods']
                self.assertEqual(set(methods), set(METHODS))
                same_n = methods['no_pilot_random_same_n']
                same_budget = methods['no_pilot_random_equal_budget']
                pilot_random = methods['pilot_random']
                self.assertEqual(same_n['selected_blocks'], same_budget['selected_blocks'])
                self.assertEqual(same_n['selected_blocks'], pilot_random['selected_blocks'])
                for name, record in methods.items():
                    with self.subTest(scenario=scenario['scenario'], method=name):
                        self.assertEqual(len(record['selected_blocks']), 2)
                        self.assertEqual(len(set(record['selected_blocks'])), 2)
                        self.assertEqual(record['main_cost'], (record['n_main'] + 1) * 4)
                        self.assertEqual(record['total_cost'], record['pilot_cost'] + record['main_cost'])
                        self.assertEqual(record['unused_budget'], plan['total_budget'] - record['total_cost'])
                        self.assertLessEqual(record['total_cost'], plan['total_budget'])
                        self.assertEqual(record['calendar_stop'], record['calendar_start'] + record['n_main'] + 3)
                        self.assertLessEqual(record['calendar_stop'], plan['calendar_rows'])
                        if name.startswith('no_pilot'):
                            self.assertEqual(record['pilot_cost'], 0)
                            self.assertEqual(record['calendar_start'], 0)
                            self.assertEqual(record['scale_source'], 'own_main')
                        else:
                            self.assertEqual(record['pilot_cost'], 366)
                            self.assertEqual(record['calendar_start'], 93)
                            self.assertEqual(record['n_main'], 80)
                        if not record['population_full_rank']:
                            self.assertIsNone(record['l2_error'])
                self.assertEqual(same_n['n_main'], 80)
                self.assertEqual(same_budget['n_main'], 171)
                self.assertEqual(same_budget['total_cost'], 688)

    def test_completed_results_cannot_be_overwritten_silently(self):
        with self.assertRaises(FileExistsError):
            run_selection_benchmark(self.directory.name + '/first', **self.kwargs)


class InformationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.p, self.d, self.q = 3, 6, 2
        self.indices = [[0, 1], [2, 3], [4, 5]]
        self.plan = observation_plan(60, 80, 30, self.d, 4)
        self.plan['hac_bandwidth'] = None

    def test_no_pilot_estimate_ignores_unpurchased_iv_and_later_data(self):
        A, _ = build_scenario('dense', self.p, self.d, self.q, seed=81)
        X, Y, I = benchmark.generate_time_model(A, self.d, self.p, 173, seed=94)
        G = [population_moments(A, self.d, self.p, block)['G'] for block in self.indices]
        beta = A[1][-1, self.d:self.d + self.p]
        args = (self.indices, [0, 2], 0, 80, None, beta, G, 0,
                self.plan, 'own_main', self.q)
        expected = benchmark._estimate_arm(X, Y, I, *args)
        self.assertEqual(expected['status'], 'ok')
        altered_X, altered_Y, altered_I = X.copy(), Y.copy(), I.copy()
        altered_I[:, [2, 3]] = np.nan
        altered_X[83:] = np.nan
        altered_Y[83:] = np.nan
        altered_I[83:] = np.nan
        actual = benchmark._estimate_arm(altered_X, altered_Y, altered_I, *args)
        for key, value in expected.items():
            if isinstance(value, float) or key in ('estimate', 'regressor_scale'):
                np.testing.assert_allclose(value, actual[key], rtol=1e-12, atol=1e-12)
            else:
                self.assertEqual(value, actual[key])

    def test_population_rank_overrules_accidental_sample_full_rank(self):
        A, _ = build_scenario('redundant', self.p, self.d, self.q, seed=81)
        X, Y, I = benchmark.generate_time_model(A, self.d, self.p, 173, seed=94)
        G = [population_moments(A, self.d, self.p, block)['G'] for block in self.indices]
        beta = A[1][-1, self.d:self.d + self.p]
        record = benchmark._estimate_arm(X, Y, I, self.indices, [0, 1], 0, 80,
            None, beta, G, 0, self.plan, 'own_main', self.q)
        self.assertTrue(record['sample_full_rank'])
        self.assertFalse(record['population_full_rank'])
        self.assertEqual(record['population_rank'], 2)
        self.assertEqual(record['status'], 'underidentified')
        self.assertIsNone(record['l2_error'])

    def test_selector_failure_does_not_remove_random_arms_or_hide_pilot_cost(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(benchmark, 'select_blocks', side_effect=ValueError('forced pilot failure')):
                result = run_selection_benchmark(root, n_runs=1, n_pilot=60, n_main=80,
                    gap=30, p=3, d=6, block_size=2, instrument_budget=4,
                    dgp_seeds=(81,), scenarios=('dense',))
        self.assertEqual(result['status'], 'complete')
        methods = result['scenarios'][0]['trials'][0]['methods']
        for name in METHODS[:3]:
            self.assertEqual(methods[name]['status'], 'ok')
        for name in METHODS[3:]:
            self.assertEqual(methods[name]['status'], 'failed')
            self.assertIn('forced pilot failure', methods[name]['failure'])
            self.assertIsNone(methods[name]['l2_error'])
            self.assertEqual(methods[name]['pilot_cost'], 366)
            self.assertEqual(methods[name]['total_cost'], 366)

    def test_information_selector_failure_preserves_strength_arm(self):
        choose = benchmark.select_blocks

        def fail_information_only(*args, **kwargs):
            if kwargs.get('method') == 'information':
                raise ValueError('information rule failed')
            return choose(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            with patch.object(benchmark, 'select_blocks', side_effect=fail_information_only):
                result = run_selection_benchmark(root, n_runs=1, n_pilot=60, n_main=80,
                    gap=30, p=3, d=6, block_size=2, instrument_budget=4,
                    dgp_seeds=(81,), scenarios=('dense',))
        methods = result['scenarios'][0]['trials'][0]['methods']
        self.assertEqual(methods['pilot_information']['status'], 'failed')
        self.assertIn('information rule failed', methods['pilot_information']['failure'])
        strength = methods['pilot_strength']
        self.assertEqual(strength['status'], 'ok')
        self.assertEqual(len(strength['selected_blocks']), 2)
        self.assertEqual(strength['pilot_cost'], 366)
        self.assertEqual(strength['main_cost'], 324)
        self.assertEqual(strength['total_cost'], 690)

    def test_pilot_schedule_does_not_use_later_main_outcomes(self):
        kwargs = dict(n_runs=1, n_pilot=60, n_main=80, gap=30, p=3, d=6,
                      block_size=2, instrument_budget=4,
                      dgp_seeds=(81,), scenarios=('dense',))
        generate = benchmark.generate_time_model
        selector_inputs = []
        choose = benchmark.select_blocks

        def capture_selection(G, omega, *args, **kw):
            selector_inputs.append((G.copy(), omega.copy()))
            return choose(G, omega, *args, **kw)

        def altered_later_data(*args, **kw):
            X, Y, I = generate(*args, **kw)
            X[93:] *= 7
            Y[93:] += 150
            I[93:] *= 3
            return X, Y, I

        with tempfile.TemporaryDirectory() as root:
            with patch.object(benchmark, 'select_blocks', side_effect=capture_selection):
                original = run_selection_benchmark(root + '/original', **kwargs)
                with patch.object(benchmark, 'generate_time_model', side_effect=altered_later_data):
                    altered = run_selection_benchmark(root + '/altered', **kwargs)
        self.assertEqual(len(selector_inputs), 4)
        for before, after in zip(selector_inputs[:2], selector_inputs[2:]):
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
        before = original['scenarios'][0]['trials'][0]['methods']
        after = altered['scenarios'][0]['trials'][0]['methods']
        for method in ('pilot_information', 'pilot_strength'):
            self.assertEqual(before[method]['selected_blocks'], after[method]['selected_blocks'])


if __name__ == '__main__':
    unittest.main()
