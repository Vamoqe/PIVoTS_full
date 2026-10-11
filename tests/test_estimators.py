import unittest
import numpy as np
import jax
from svd_2sls import svd_2sls, combine_svd_estimation, check_component_coverage
from optimization_submodular import SetProposal, BaseSet, LinearExperiment


class EstimatorTests(unittest.TestCase):
    def test_noiseless_identified_projection_and_nullspace(self):
        rng=np.random.default_rng(19); x=rng.normal(size=(200,3)); z=rng.normal(size=(200,1))
        beta=np.array([.3,1,2,-1]); y=np.column_stack((np.ones(200),x))@beta
        b,P,Pc=svd_2sls(x,y,z,max_iv=1)
        np.testing.assert_allclose(b,P@beta,atol=1e-11)
        np.testing.assert_allclose(P+Pc,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(P@P,P,atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(P,tol=1e-8),2)

    def test_more_parameters_than_samples_preserves_nullspace(self):
        x=np.arange(21,dtype=float).reshape(3,7); y=np.arange(3.); z=y[:,None]
        _,P,Pc=svd_2sls(x,y,z,max_iv=1)
        self.assertEqual(Pc.shape,(8,8))
        np.testing.assert_allclose(P+Pc,np.eye(8))

    def test_coverage_zero_direction_and_exact_integration(self):
        P=np.diag([1.,0.]); Q=np.eye(2)-P; beta=np.array([1.,2.])
        np.testing.assert_allclose(check_component_coverage([P]),[0,np.pi/2])
        np.testing.assert_allclose(combine_svd_estimation([P@beta,Q@beta],[P,Q]),beta)

    def test_iid_branch_returns_generated_data(self):
        exp=LinearExperiment(jax.random.PRNGKey(2),20,np.eye(3),np.ones(3),np.ones(3),np.eye(3))
        x,y,z=exp.run([0])
        self.assertEqual(x.shape,(20,3)); self.assertEqual(y.shape,(20,)); self.assertEqual(z.shape,(20,1))

    def test_budget_and_seed_set(self):
        class Dummy:
            def run(self,idx):
                rng=np.random.default_rng(22); x=rng.normal(size=(30,3))
                return x,x.sum(axis=1),x[:,idx]
        model=SetProposal(3,1,Dummy(),d_max=2,similarity_matrix=np.eye(3))
        model.fit()
        self.assertEqual(model.cumulative_cost,1)
        self.assertEqual(len(model.current_idxs),1)
        seed=BaseSet(np.array([0.,1.,0.,0.]),np.diag([1.,1.,0.,0.]),np.diag([0.,0.,1.,1.]),[0])
        model=SetProposal(1,2,Dummy(),d_max=2,similarity_matrix=np.eye(3),BaseSet=seed,round_sizes=[2])
        model.fit()
        self.assertEqual(sorted(model.current_idxs),[0,1,2])
        self.assertEqual(model.cumulative_cost,2)


if __name__=='__main__': unittest.main()
