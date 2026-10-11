import unittest
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from dynamic_iv import population_moments, validate_selected_instrument_history
from simulation import sim_data
from utils import generate_time_scenario, truncated_residual


class DynamicIVTests(unittest.TestCase):
    def setUp(self):
        self.d,self.p = 6,3
        self.A = generate_time_scenario(None,self.d,self.p,seed=253)

    def test_population_moment_and_local_projection(self):
        for idx in ([0],[1,2],list(range(self.d))):
            result = population_moments(self.A,self.d,self.p,idx)
            np.testing.assert_allclose(result['moment'],0,atol=1e-12)
            P = np.linalg.pinv(result['G']) @ result['G']
            np.testing.assert_allclose(np.linalg.pinv(result['G'])@result['h'],P@result['beta'],atol=1e-12)
        self.assertEqual(result['rank'],self.p)

    def test_wrong_conditioning_is_detectable_at_population_level(self):
        result = population_moments(self.A,self.d,self.p,list(range(self.d)),conditioning='x_history')
        self.assertGreater(np.linalg.norm(result['moment']),1e-5)

    def test_rho_changes_persistence_not_structural_coefficients(self):
        low = generate_time_scenario(None,self.d,self.p,seed=253,target_rho=0.2)[1]
        high = generate_time_scenario(None,self.d,self.p,seed=253,target_rho=0.95)[1]
        np.testing.assert_array_equal(low-np.diag(np.diag(low)),high-np.diag(np.diag(high)))
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(low))),0.2)
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(high))),0.95)

    def test_unsupported_dynamic_paths_fail_explicitly(self):
        a = self.A[1].copy(); a[0,1] = .05
        with self.assertRaisesRegex(ValueError,'omits parents'):
            validate_selected_instrument_history({1:a},self.d,self.p,[0])
        a = self.A[1].copy(); a[-1,0] = .05
        with self.assertRaisesRegex(ValueError,'directly affect'):
            validate_selected_instrument_history({1:a},self.d,self.p,[0])
        q = .1*np.eye(len(a)); q[0,-1]=q[-1,0]=.01
        with self.assertRaisesRegex(ValueError,'independent'):
            validate_selected_instrument_history(self.A,self.d,self.p,[0],q)

    def test_alignment_and_history_residualization(self):
        rng = np.random.default_rng(7); n=20
        X,Y,I = rng.normal(size=(n+3,3)),rng.normal(size=(n+3,1)),rng.normal(size=(n+3,4))
        rx,ry,rz = truncated_residual(X,Y,I,n,[0,2])
        B = np.column_stack((np.ones(n),I[:n,[0,2]]))
        np.testing.assert_allclose(B.T@np.column_stack((rx,ry,rz)),0,atol=1e-12)
        aligned=np.column_stack((X[2:n+2],Y[3:n+3],I[1:n+1][:,[0,2]]))
        expected=aligned-B@np.linalg.lstsq(B,aligned,rcond=None)[0]
        np.testing.assert_allclose(np.column_stack((rx,ry,rz)),expected)
        self.assertEqual(truncated_residual(X,Y,I,n,[False]*4)[2].shape,(n,0))
        with self.assertRaises(ValueError):
            truncated_residual(X,Y,I,n,[0,0])

    def test_stationary_start_and_reproducibility(self):
        A={1:np.array([[.9]])}
        first=np.array([sim_data(A,1,seed=i)[0,0] for i in range(1000)])
        self.assertAlmostEqual(np.var(first),.1/(1-.9**2),delta=.06)
        np.testing.assert_array_equal(sim_data(A,20,seed=4),sim_data(A,20,seed=4))
        self.assertEqual(sim_data({1:np.zeros((2,2))},4,seed=3).shape,(4,2))
        with self.assertRaises(ValueError):
            sim_data({1:np.eye(2)},4)

    def test_var2_companion_stationary_covariance(self):
        A={1:np.array([[.3]]),2:np.array([[.2]])}
        F=np.array([[.3,.2],[1,0]])
        S=solve_discrete_lyapunov(F,np.diag([.1,0]))
        data=sim_data(A,20000,seed=51)[:,0]
        self.assertAlmostEqual(np.var(data),S[0,0],delta=.008)


if __name__=='__main__': unittest.main()
