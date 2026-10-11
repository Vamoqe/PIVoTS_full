import unittest
import numpy as np
from make_simulation_tables import error_statistics, paired_error_comparison

class ReportingTests(unittest.TestCase):
    def test_nonfinite_failures_are_counted(self):
        result=error_statistics([[3,4],[np.nan,1],[np.inf,0]])
        self.assertEqual((result['Attempted'],result['N'],result['Failed']),(3,1,2))
        self.assertEqual(result['Mean'],5)
    def test_all_failed_is_defined(self):
        result=error_statistics([[np.nan,0]])
        self.assertEqual(result['N'],0); self.assertTrue(np.isnan(result['Mean']))
    def test_paired_comparison_uses_common_finite_trials(self):
        res={'diff_beta':[[3,4],[np.nan,1]],'diff_benchmark_beta':[[0,2],[1,1]]}
        result=paired_error_comparison(res)
        self.assertEqual(result['paired_finite'],1); self.assertEqual(result['mean_difference'],3)
        with self.assertRaises(ValueError):
            paired_error_comparison(dict(res,diff_beta=[[1,1]]))
