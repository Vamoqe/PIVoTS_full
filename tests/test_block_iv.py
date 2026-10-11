import tempfile
import unittest
import numpy as np
from block_iv import hac_covariance,fit_gmm,stack_moments,information_matrix,select_blocks
from run_pilot_selection import run_pilot_selection


class BlockIVTests(unittest.TestCase):
    def test_hac_uses_cross_block_dependence(self):
        rng=np.random.default_rng(2); a=rng.normal(size=200)
        f=np.column_stack((a,2*a+rng.normal(size=200)*.1))
        omega=hac_covariance(f,3)
        self.assertGreater(abs(omega[0,1]),1)
        self.assertGreaterEqual(np.linalg.eigvalsh(omega).min(),-1e-10)

    def test_information_increment_accounts_for_correlated_noise(self):
        G=np.array([[1.,0.],[0.,1.],[1.,1.]])
        omega=np.array([[2.,.5,.3],[.5,1.,.2],[.3,.2,1.5]])
        gain=information_matrix(G,omega,np.array([0,1]))-information_matrix(G,omega,np.array([0]))
        self.assertGreaterEqual(np.linalg.eigvalsh(gain).min(),-1e-10)

    def test_information_prefers_complementary_strong_direction(self):
        G=np.array([[2.,0.],[1.9,0.],[0.,1.]])
        chosen,trajectory=select_blocks(G,np.eye(3),[np.array([i]) for i in range(3)],[1,1,1],2)
        self.assertEqual(chosen,[0,2]); self.assertEqual(trajectory[-1]['cost'],2)

    def test_underidentification_is_not_labeled_full_recovery(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(100,3)); y=x@np.ones(3)
        fit=fit_gmm([(x,y,x[:,:1])])
        self.assertFalse(fit['full_sample_rank']); self.assertEqual(fit['sample_rank'],1)

    def test_end_to_end_pilot_is_reproducible_and_charged(self):
        with tempfile.TemporaryDirectory() as root:
            kwargs=dict(n_runs=2,n_pilot=100,n_main=120,gap=30,p=3,d=6,block_size=2,instrument_budget=4,seed=81)
            a=run_pilot_selection(root+'/a',**kwargs); b=run_pilot_selection(root+'/b',**kwargs)
            self.assertEqual(a['trials'],b['trials'])
            self.assertEqual(a['status'],'complete')
            for trial in a['trials']:
                for method in ['information','random','strength']:
                    result=trial['methods'][method]
                    self.assertEqual(result['pilot_cost'],606)
                    self.assertEqual(result['main_cost'],484)
                    self.assertEqual(result['total_cost'],1090)
                    self.assertEqual(len(result['selected_blocks']),2)
                self.assertFalse(trial['methods']['full_gmm_reference']['within_budget'])
            with self.assertRaises(FileExistsError):
                run_pilot_selection(root+'/a',**kwargs)


if __name__=='__main__': unittest.main()
