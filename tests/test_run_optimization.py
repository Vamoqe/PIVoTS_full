"""Small end-to-end checks of replication, paired budgets and saved provenance."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from run_optimization import build_parser, make_round_schedule, run_optimization


class RunOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pivots-run-tests-")
        cls.parameters = dict(
            seed=47, n_runs=2, n=100, p=3, d=4, d_id=3, d_max=2,
            n_rounds=3, name_id="tiny", target_rho=0.4, budget=3,
        )
        cls.first_path = Path(cls.temp.name) / "first"
        cls.second_path = Path(cls.temp.name) / "second"
        cls.first = run_optimization(fig_save_path=cls.first_path, **cls.parameters)
        cls.second = run_optimization(fig_save_path=cls.second_path, **cls.parameters)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_repeated_seed_reproduces_every_estimator(self):
        for field in ("diff_beta", "diff_benchmark_beta", "diff_first_subset_beta",
                      "diff_singlex_beta", "diff_full_iv_beta", "diff_ols_beta",
                      "diff_beta_trajectory", "sel_inst", "benchmark_sel_inst"):
            np.testing.assert_allclose(self.first[field], self.second[field], equal_nan=True,
                                       err_msg=field)
        np.testing.assert_array_equal(self.first["trial_data_seeds"], [10047, 10048])
        self.assertFalse(np.array_equal(self.first["diff_ols_beta"][0], self.first["diff_ols_beta"][1]))

    def test_each_trial_has_exactly_one_result_and_common_budget(self):
        result = self.first
        for field in ("diff_beta", "diff_benchmark_beta", "diff_first_subset_beta",
                      "diff_singlex_beta", "diff_full_iv_beta", "diff_ols_beta"):
            self.assertEqual(result[field].shape, (2, 4), field)
        for field in ("err_beta", "err_benchmark_beta", "err_first_subset_beta",
                      "err_singlex_beta", "err_full_iv_beta", "err_ols_beta"):
            self.assertEqual(result[field].shape, (2,), field)
        self.assertEqual(result["diff_beta_trajectory"].shape, (2, 2, 4))
        self.assertEqual(result["diff_benchmark_beta_trajectory"].shape, (2, 2, 4))
        self.assertEqual(result["singlex_cover"].shape, (2, 1, 4))
        np.testing.assert_array_equal(result["used_budget"], [3, 3])
        np.testing.assert_array_equal(result["used_budget"], result["benchmark_used_budget"])
        np.testing.assert_array_equal(result["cumulative_cost_trajectory"], [[2, 3], [2, 3]])
        for key in ("selection_groups", "benchmark_selection_groups"):
            for trial_groups in result[key]:
                self.assertEqual([len(group) for group in trial_groups], [2, 1])
                self.assertEqual(len(set(index for group in trial_groups for index in group)), 3)

    def test_manifest_and_reference_dataset_are_explicit(self):
        manifest = json.loads((self.first_path / "manifest.json").read_text())
        saved = np.load(self.first_path / "results.npy", allow_pickle=True).item()
        scenario = np.load(self.first_path / "tiny.npy", allow_pickle=True).item()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["completed_trials"], 2)
        self.assertEqual(saved["config"], manifest["config"])
        self.assertEqual(saved["config_sha256"], manifest["config_sha256"])
        self.assertEqual(saved["config"]["selection"], "noisy_oracle")
        self.assertEqual(saved["config"]["initialization"], "stationary_lyapunov")
        self.assertEqual(saved["config"]["conditioning"], "instrument_history")
        self.assertEqual(scenario["dataset_role"], "reference_only_not_a_trial")
        self.assertAlmostEqual(saved["actual_rho"], 0.4, places=10)
        self.assertIn("run_optimization.py", manifest["version"]["source_sha256"])

    def test_single_trial_round_and_coefficient_preserve_axes(self):
        result = run_optimization(
            seed=47, n_runs=1, n=60, p=1, d=2, d_id=1, d_max=1,
            n_rounds=1, name_id="one", fig_save_path=Path(self.temp.name) / "one",
        )
        self.assertEqual(result["diff_beta"].shape, (1, 2))
        self.assertEqual(result["diff_beta_trajectory"].shape, (1, 1, 2))
        self.assertEqual(result["cover_trajectory"].shape, (1, 1, 2))
        self.assertTrue(np.isfinite(result["err_beta"]).all())
        self.assertTrue(np.isnan(result["conf_strength_hat"]).all())
        self.assertIn("legacy_estimator_requires_at_least_two_regressors",
                      result["legacy_confounding_diagnostic_notes"][0])

    def test_existing_results_cannot_be_silently_replaced(self):
        before = (self.first_path / "manifest.json").read_bytes()
        parameters = dict(self.parameters, target_rho=0.6)
        with self.assertRaises(FileExistsError):
            run_optimization(fig_save_path=self.first_path, **parameters)
        self.assertEqual(before, (self.first_path / "manifest.json").read_bytes())

    def test_failed_explicit_overwrite_does_not_leave_old_complete_results(self):
        failure_path = Path(self.temp.name) / "failed"
        failure_path.mkdir()
        np.save(failure_path / "results.npy", {"schema_version": 1, "status": "complete"})
        with patch("run_optimization.generate_time_scenario", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run_optimization(fig_save_path=failure_path, overwrite=True, **self.parameters)
        manifest = json.loads((failure_path / "manifest.json").read_text())
        results = np.load(failure_path / "results.npy", allow_pickle=True).item()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_trials"], 0)
        self.assertEqual(results["status"], "failed")
        self.assertEqual(results["schema_version"], 2)


class RunnerConfigurationTests(unittest.TestCase):
    def test_budget_schedule_stops_at_resource_limit(self):
        self.assertEqual(make_round_schedule(4, 3, 4), ([3, 1], 4))
        self.assertEqual(make_round_schedule(10, 3, 4, budget=5), ([3, 2], 5))
        with self.assertRaises(ValueError):
            make_round_schedule(4, 2, 3, budget=0)

    def test_defaults_and_slurm_entrypoint_are_shared(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.confound_x_strength, 1.2)
        self.assertEqual(args.confound_y_strength, 1.5)
        slurm = importlib.import_module("run_optimization_slurm")
        main_module = importlib.import_module("run_optimization")
        self.assertIs(slurm.main, main_module.main)

    def test_grid_import_is_inert_and_rho_uses_same_settings(self):
        with patch("subprocess.run") as spawn:
            grid = importlib.import_module("run_simulation_grid")
            importlib.import_module("run_rho_only")
            spawn.assert_not_called()
        args = grid.build_sweep_parser().parse_args(["--n_runs", "2", "--n", "100"])
        with patch.object(grid, "run_one") as run:
            grid.run_rho_sweep(args)
        self.assertEqual(run.call_count, 4)
        for call in run.call_args_list:
            for name in ("p", "d", "d_id", "d_max", "n_rounds"):
                self.assertEqual(call.kwargs[name], grid.BASELINE_SETTINGS[name])
            self.assertEqual(call.kwargs["n_runs"], 2)
            self.assertEqual(call.kwargs["n"], 100)


if __name__ == "__main__":
    unittest.main()
