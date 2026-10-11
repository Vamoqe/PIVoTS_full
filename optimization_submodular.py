import numpy as np
import jax
import plotly.graph_objects as go
from itertools import combinations
from utils import cosine_distance

# own imports
from svd_2sls import svd_2sls, combine_svd_estimation, check_component_coverage
from utils import generate_rv, generate_model, generate_time_model, generate_time_scenario, truncated_residual



class Experiment:
    """ general Experiment Class """

    def __init__(self, key, n):
        self._key = key
        self.n = n

    def run(self, idxs):
        x = None
        y = None
        z = None
        return x, y, z


class LinearExperiment(Experiment):
    """ representation of our artificial experiment """
    def __init__(self, key, n, alpha, beta, e, M, time_setting = False, A=None, X=None, Y=None, I=None):
        super().__init__(key, n)
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.M = M
        d, p = self.alpha.shape
        self.time_setting = time_setting                                # If true, A, X, Y, I must filled in
        self.A = A
        self.X, self.Y, self.I = X, Y, I
        #self._key, _ = jax.random.split(self._key, 2)                  We don't generate time data everytime, instedad, we pregenerate it 
        #self.A = generate_time_scenario(self._key,d,p)                     in run_optimization step 1. And get it when init the class LinearExperient
        #self.X, self.Y, self.I = generate_time_model(self.A,d,p,self.n)    as parameter fill in.    
        # test

    def run(self, lam):
        """
        Run Experiment
        Parameters
        ----------
        lam: list of boolean
            indicate which instruments are used for the particular experiment run

        Returns
        -------

        """
        d, p = self.alpha.shape
        if self.time_setting:
            from dynamic_iv import instrument_indices, validate_selected_instrument_history
            idx = instrument_indices(lam, d)
            cache = getattr(self, "_validated_blocks", set())
            block = tuple(idx)
            if block not in cache:
                validate_selected_instrument_history(self.A, d, p, idx)
                cache.add(block)
                self._validated_blocks = cache
            x, y, z = truncated_residual(self.X, self.Y, self.I, self.n, idx)
            return x, y.ravel(), z
        self._key, subkey = jax.random.split(self._key)
        z, u = generate_rv(subkey, d, p, self.n)
        return generate_model(self.alpha, self.beta, z, u, self.e, self.M, lam)


class BaseSet:
    def __init__(self, beta, PV, PV_comp, idxs):
        self.beta = beta
        self.PV_comp = PV_comp
        self.PV = PV
        self.idxs = idxs  # indicate the indices of the alpha that has been used


class SetProposal:
    def __init__(self,
                 n_rounds,
                 budget,
                 experiment,
                 d_max=1,
                 similarity_matrix=None,
                 BaseSet=None,
                 selection_method="similarity",
                 key=jax.random.PRNGKey(2710),
                 round_sizes=None
                 ):

        if not isinstance(n_rounds, (int, np.integer)) or n_rounds < 1 or d_max < 1 or budget < 0 or np.isnan(budget):
            raise ValueError("Require positive rounds/d_max and nonnegative budget")
        if selection_method not in {"similarity", "random"}:
            raise ValueError("Unknown selection method")
        similarity_matrix = np.asarray(similarity_matrix, dtype=float)
        if similarity_matrix.ndim != 2 or not np.all(np.isfinite(similarity_matrix)):
            raise ValueError("Need finite two-dimensional selection features")
        if round_sizes is not None:
            round_sizes = list(round_sizes)
            if len(round_sizes) != n_rounds or any(not isinstance(v, (int,np.integer)) or not 1 <= v <= d_max for v in round_sizes):
                raise ValueError("round_sizes must specify a positive size <= d_max for each round")
        self._round_sizes = round_sizes
        self._d_max, self._budget = int(d_max), budget
        self._experiment, self._selection_method = experiment, selection_method
        self._selection_trajectory, self._estimation_trajectory = [], []
        self._n_rounds, self._iter_round = n_rounds, 0
        self._similarity_alpha, self._key = similarity_matrix, key
        self._estimation_function = svd_2sls
        self._coverage_trajectory, self._beta_trajectory, self._P_trajectory = [], [], []
        self.cumulative_cost, self._cost_trajectory = 0, []
        self.current_idxs, self.current_beta = [], None
        if BaseSet is not None:
            from dynamic_iv import instrument_indices
            self.current_idxs = instrument_indices(BaseSet.idxs, len(similarity_matrix)).tolist()
            self.current_beta = np.asarray(BaseSet.beta).copy()
            self._beta_trajectory.append(self.current_beta)
            self._P_trajectory.append(BaseSet.PV)
        # A BaseSet is an already-acquired seed; only additional acquisition is charged.
        self._possible_idx_sets = [i for i in range(len(similarity_matrix)) if i not in self.current_idxs]

        # parameters for evaluation
        self.figure = go.Figure()
        self._error_to_beta = []
        self._error_to_nonzero_beta = []


    def _distance(self, idx_new, idx_old, type="cosine"):
        """
        compute distance based on similarity matrix
        Parameters
        ----------
        idx_new: list
            list of indices for newly selected instruments
        idx_old: list
            list of indices for already used instruments
        type: str

        Returns
        -------
        float

        """
        if type == "cosine":
            mat1 = self._similarity_alpha[idx_new, :]
            mat2 = self._similarity_alpha[idx_old, :]

        return cosine_distance(mat2, mat1)


    def _select(self, idxs_sets):
        """
        selection function for most promising idx set
        **Note**: Implementation so far are *random* and *similarity* (matrix with pairwise similarities),
        customize if needed

        Parameters
        ----------
        idxs_sets: list

        Returns
        -------
        int: returns best index
        """

        gains = np.zeros(len(idxs_sets), dtype='float64')
        jter = 0

        for idx_set in idxs_sets:
            if self._selection_method == "similarity":
                if self.current_idxs == []:
                    gain_idx = self._distance(idx_set, idx_set) / (len(idx_set) + len(idx_set) - 1) - np.log(len(
                        idx_set))
                else:
                    gain_idx = self._distance(idx_set, self.current_idxs) / (len(idx_set) + len(self.current_idxs) -
                                                                             1) - np.log(len(idx_set))

            elif self._selection_method == "random":
                self._key, subkey = jax.random.split(self._key)
                gain_idx = jax.random.uniform(subkey)

            # Update gain and sample cost
            gains[jter] = gain_idx

            # Update count
            jter += 1

        best_idx = gains.argmax()

        return idxs_sets[best_idx]


    def _cost_budget(self, idx_list, budget_method="idx_length"):
        """
        budget computation for idx set
        **Note**: Implementation so far are *idx_length* which means direct proportional to the number of
        instruments, customize here if needed

        Parameters
        ----------
        idx_list
        budget_method

        Returns
        -------

        """
        if budget_method == "idx_length":
            return len(idx_list)
        else:
            raise NotImplementedError


    def fit(self, budget_method="idx_length"):
        """
        Fit functionality, fit the model with the parameters defined in the instantiation of the class
        Parameters
        ----------
        budget_method: str
            default *idx_length*; cost of the experiment is the number of instruments

        Returns
        -------

        """
        budget = self.cumulative_cost
        while budget < self._budget and self._possible_idx_sets and self._iter_round < self._n_rounds:
            remaining = self._budget-budget
            if self._round_sizes is None:
                sizes = range(1, min(self._d_max, len(self._possible_idx_sets))+1)
            else:
                sizes = [self._round_sizes[self._iter_round]]
            idxs_sets = [list(c) for size in sizes if size <= remaining
                         for c in combinations(self._possible_idx_sets, size)]
            if not idxs_sets:
                break

            # select most promising idx set
            best_idx = self._select(idxs_sets)

            # run experiment with updated indices
            x, y, z = self._experiment.run(best_idx)

            # estimate parameters from new experiment data
            beta_est, PV, PV_comp = self._estimation_function(x, y, z, max_iv=len(best_idx))

            # update the parameters
            self.current_idxs += best_idx

            self._selection_trajectory.append(best_idx)
            self._beta_trajectory.append(beta_est)
            self._P_trajectory.append(PV)

            # update estimator
            if self.current_beta is None:
                self.current_beta = beta_est
            else:
                beta_combined = combine_svd_estimation(self._beta_trajectory, self._P_trajectory)
                self.current_beta = beta_combined

            self._estimation_trajectory.append(self.current_beta)

            # check coverage of standard basis
            cover_vec = check_component_coverage(self._P_trajectory)
            self._coverage_trajectory.append(cover_vec)

            budget += self._cost_budget(best_idx, budget_method)
            self.cumulative_cost = budget
            self._cost_trajectory.append(budget)

            self._possible_idx_sets = [idx for idx in self._possible_idx_sets if idx not in self.current_idxs]
            self._iter_round += 1





