"""Regression checks for exclusion accounting and paired Monte Carlo errors."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_reporting import summarize_trials, write_benchmark_tables


PRIMARY = "no_pilot_random_equal_budget"


def record(error, status="ok", population=True, sample=True, cost=100, n_main=20):
    return {"status": status, "population_full_rank": population,
            "sample_full_rank": sample, "l2_error": error,
            "total_cost": cost, "n_main": n_main}


class BenchmarkReportingTests(unittest.TestCase):
    def test_pairing_uses_only_common_usable_trials_and_preserves_sign(self):
        primary = [record(1), record(2), record(np.nan), record(4),
                   record(None, "underidentified", population=False, sample=False)]
        candidate = [record(3), record(1), record(2),
                     record(None, "failed", sample=False), record(7)]
        for index, item in enumerate(candidate):
            item["total_cost"] = 100 + index
            item["n_main"] = [10, 20, None, 40, 50][index]
        trials = [{"trial_id": index, "methods": {PRIMARY: left, "candidate": right}}
                  for index, (left, right) in enumerate(zip(primary, candidate))]
        summary = summarize_trials(trials)
        baseline = summary["methods"][PRIMARY]
        method = summary["methods"]["candidate"]
        paired = summary["paired_comparisons"]["candidate"]
        self.assertEqual((baseline["attempted"], baseline["valid_count"], baseline["excluded_count"]), (5, 3, 2))
        self.assertEqual(baseline["failure_count"], 1)
        self.assertEqual(baseline["invalid_ok_error_count"], 1)
        self.assertEqual(baseline["population_underidentified_count"], 1)
        self.assertEqual(baseline["sample_under_rank_count"], 1)
        self.assertEqual(method["valid_count"], 4)
        self.assertEqual(method["failure_count"], 1)
        self.assertEqual(method["sample_under_rank_count"], 1)
        self.assertAlmostEqual(method["mean_l2"], 13 / 4)
        self.assertAlmostEqual(method["rmse"], np.sqrt(63 / 4))
        # Cost/sample-length accounting includes failed attempts; missing values
        # have their own denominator and are not silently counted as zero.
        self.assertAlmostEqual(method["mean_cost"], 102)
        self.assertEqual(method["cost_recorded_count"], 5)
        self.assertAlmostEqual(method["mean_n_main"], 30)
        self.assertEqual(method["n_main_recorded_count"], 4)
        self.assertEqual(method["n_main_missing_count"], 1)
        self.assertEqual((paired["paired_count"], paired["excluded_count"]), (2, 3))
        self.assertEqual(paired["method_only_valid_count"], 2)
        self.assertEqual(paired["primary_only_valid_count"], 1)
        self.assertEqual(paired["neither_valid_count"], 0)
        # Shared L2 differences: [2, -1]; squared-error differences: [8, -3].
        self.assertAlmostEqual(paired["mean_l2_difference"], .5)
        self.assertAlmostEqual(paired["l2_difference_mcse"], 1.5)
        self.assertAlmostEqual(paired["mean_squared_error_difference"], 2.5)
        self.assertAlmostEqual(paired["squared_error_difference_mcse"], 5.5)
        json.dumps(summary, allow_nan=False)

    def test_zero_usable_errors_and_nonfinite_successes_are_explicit(self):
        bad = [record(None), record(np.nan), record(np.inf), record(-1),
               record(99, "underidentified", population=False), record(17, "failed")]
        trials = [{"trial_id": index, "methods": {PRIMARY: item, "candidate": record(1)}}
                  for index, item in enumerate(bad)]
        result = summarize_trials(trials)
        summary = result["methods"][PRIMARY]
        self.assertEqual(summary["valid_count"], 0)
        self.assertEqual(summary["excluded_count"], 6)
        self.assertEqual(summary["failure_count"], 5)
        self.assertEqual(summary["invalid_ok_error_count"], 4)
        self.assertEqual(summary["population_underidentified_count"], 1)
        for key in ("mean_l2", "rmse", "median", "p95", "p99", "max", "mean_l2_mcse"):
            self.assertIsNone(summary[key])
        pair = result["paired_comparisons"]["candidate"]
        self.assertEqual(pair["paired_count"], 0)
        self.assertIsNone(pair["mean_squared_error_difference"])
        self.assertIsNone(pair["squared_error_difference_mcse"])
        json.dumps(result, allow_nan=False)

    def test_single_pair_has_no_estimable_mcse_and_missing_record_is_counted(self):
        trials = [
            {"trial_id": 0, "methods": {PRIMARY: record(2), "candidate": record(1)}},
            {"trial_id": 1, "methods": {PRIMARY: record(2)}},
        ]
        result = summarize_trials(trials)
        method = result["methods"]["candidate"]
        pair = result["paired_comparisons"]["candidate"]
        self.assertEqual(method["missing_record_count"], 1)
        self.assertEqual(method["failure_count"], 1)
        self.assertEqual(method["excluded_count"], 1)
        self.assertIsNone(method["mean_l2_mcse"])
        self.assertEqual(pair["paired_count"], 1)
        self.assertEqual(pair["mean_squared_error_difference"], -3)
        self.assertEqual(pair["mean_l2_difference"], -1)
        self.assertIsNone(pair["squared_error_difference_mcse"])
        self.assertIsNone(pair["l2_difference_mcse"])
        self.assertFalse(pair["squared_mcse_unrepresentable"])

    def test_extreme_finite_errors_do_not_create_hidden_infinite_statistics(self):
        trials = [{"trial_id": index, "methods": {PRIMARY: record(0), "candidate": record(1e200)}}
                  for index in range(2)]
        result = summarize_trials(trials)
        method = result["methods"]["candidate"]
        pair = result["paired_comparisons"]["candidate"]
        self.assertEqual(method["mean_l2"], 1e200)
        self.assertEqual(method["rmse"], 1e200)
        self.assertIsNone(pair["mean_squared_error_difference"])
        self.assertTrue(pair["squared_mean_unrepresentable"])
        self.assertEqual(pair["paired_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_duplicate_trials_are_rejected(self):
        trial = {"trial_id": 0, "methods": {PRIMARY: record(1)}}
        with self.assertRaisesRegex(ValueError, "Duplicate trial IDs"):
            summarize_trials([trial, trial])
        self.assertEqual(summarize_trials([]), {"methods": {}, "paired_comparisons": {}})

    def test_csv_keeps_dgp_runs_separate_and_all_exclusion_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            scenarios = []
            for seed, error in ((3, 1), (9, 4)):
                summary = summarize_trials([
                    {"trial_id": 0, "methods": {PRIMARY: record(error), "candidate": record(2)}},
                ])
                scenarios.append({"scenario": "weak", "dgp_seed": seed, "summary": summary})
            paths = write_benchmark_tables({"scenarios": scenarios}, directory)
            with Path(paths["method_summary_csv"]).open(newline="") as stream:
                methods = list(csv.DictReader(stream))
            with Path(paths["paired_comparisons_csv"]).open(newline="") as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(methods), 4)
            self.assertEqual(len(pairs), 2)
            self.assertEqual({row["dgp_seed"] for row in pairs}, {"3", "9"})
            self.assertIn("failure_count", methods[0])
            self.assertIn("excluded_count", pairs[0])
            self.assertEqual({row["squared_error_difference_mcse"] for row in pairs}, {""})
            self.assertEqual({row["mean_squared_error_difference"] for row in pairs}, {"3.0", "-12.0"})


if __name__ == "__main__":
    unittest.main()
