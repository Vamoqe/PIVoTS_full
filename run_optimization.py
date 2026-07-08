import logging
import numpy as np
import jax
import argparse
from sklearn.linear_model import LinearRegression

import os
from utils import generate_scenario, save_scenario, generate_time_model, generate_time_scenario
from optimization_submodular import LinearExperiment, SetProposal
from svd_2sls import svd_2sls, check_component_coverage
from confounder_strength import estimate_conf_strength, compute_conf_strength
from visualization import visualize_results



def run_optimization(
    seed,
    n_runs,
    n,
    p,
    d,
    d_id,
    d_max,
    n_rounds,
    name_id,
    fig_save_path,
    target_rho=0.8,
    confound_x_strength=1.5,
    confound_y_strength=2.0,
    iv_x_strength=1.0,
    beta_strength=1.0
    ):
    """
    run optimization function n_runs times

    Parameters
    ----------
    seed: int
    n_runs: int
    n: int
    p: int
    d: int
    d_id: int
    d_max: int
    n_rounds: int
    fig_save_path: str
        path to directory in which the results should be saved to

    Returns
    -------

    """
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    # ----------------------------------------------------------------------------------------------------------------
    # 1. Generate scenario
    # ----------------------------------------------------------------------------------------------------------------
    key, key_sub = jax.random.split(key)
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Scenario Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #alph1, beta1, e1, m1 = generate_scenario(key_sub, d, p, d_id)
    #breakpoint()
    # save generated alpha, beta etc.
    #alpha, beta, e, m = 1,1,1,1 # Don't use them, dummy variable to keep program runing
    key, key_sub = jax.random.split(key)
    #A = generate_time_scenario(key_sub,d,p,d_id)

    A = generate_time_scenario(
    key_sub,
    d,
    p,
    d_id,
    target_rho=target_rho,
    iv_x_strength=iv_x_strength,
    confound_x_strength=confound_x_strength,
    confound_y_strength=confound_y_strength,
    beta_strength=beta_strength,
    seed=seed
    )
    #breakpoint()
    alpha, beta, e, m = A[1][d:d+p,:d].T, A[1][d+p+p:,d:d+p].flatten(), A[1][d+p+p:,d+p:d+p+p].flatten(), A[1][d:d+p,d+p:d+p+p].T # under VAR(1) setting
    #breakpoint()
    print(f"beta:{beta}")
    #X, Y, I = generate_time_model(A,d,p,n)
    #save_scenario(key_sub, alpha, beta, e, m, fig_save_path, name_id=name_id,A=A,X=X,Y=Y,I=I)
    X0, Y0, I0 = generate_time_model(A, d, p, n, seed=seed)

    save_scenario(
        key_sub,
        alpha,
        beta,
        e,
        m,
        fig_save_path,
        name_id=name_id,
        A=A,
        X=X0,
        Y=Y0,
        I=I0
    )

    #breakpoint()
    # ----------------------------------------------------------------------------------------------------------------
    # 2. Initialize models
    # ----------------------------------------------------------------------------------------------------------------
    # Initialize experiment setting
    key, key_sub = jax.random.split(key)
    key_chain = jax.random.split(key_sub, n_runs)

    iter = 0
    diff_beta = []
    diff_beta_trajectory = []
    cover_trajectory = []
    diff_benchmark_beta = []
    diff_benchmark_beta_trajectory = []
    diff_singlex_beta = []

    diff_first_subset_beta = []
    diff_ols_beta = []

    singlex_cover = []
    benchmark_cover_trajectory = []
    cs = []
    cs_hat = []
    mnorm = []
    mnorm_hat = []
    sel_inst = []
    benchmark_sel_inst = []

    for key in key_chain:
        print(
            "****************************** " + str(iter) + "/" + str(n_runs) + " ************************************")

        # ----------------------------------------------------------------------------------------------------------------
        # 2. Initialize models
        # ----------------------------------------------------------------------------------------------------------------
        key, key_sub = jax.random.split(key)


        #linear_experiment = LinearExperiment(key_sub, n, alpha, beta, e, m, time_setting=True, A=A, X=X, Y=Y, I=I)

        mc_seed = seed + iter + 10000

        X_run, Y_run, I_run = generate_time_model(
            A,
            d,
            p,
            n,
            seed=mc_seed
        )

        linear_experiment = LinearExperiment(
            key_sub,
            n,
            alpha,
            beta,
            e,
            m,
            time_setting=True,
            A=A,
            X=X_run,
            Y=Y_run,
            I=I_run
        )

        beta_0 = np.hstack([0.0, beta]).squeeze()


        ## Computation of confounding strength according to Dominik Janzig
        # generate the fully observed setting (switch off all instruments)
        x, y, z = linear_experiment.run([False] * d)
        reg = LinearRegression(fit_intercept=True).fit(x, y)
        beta_obs = reg.coef_
        #breakpoint()
        beta_obs_intercept = np.hstack([reg.intercept_, reg.coef_])

        diff_ols_beta.append(beta_obs_intercept - beta_0)

        conf_strength = compute_conf_strength(beta, m, e)
        conf_strength_hat = estimate_conf_strength(0, x, y)
        max_norm_hat = (1 - conf_strength_hat) * beta_obs @ beta_obs
        max_norm = (1 - conf_strength) * beta_obs @ beta_obs

        key, key_sub = jax.random.split(key)
        similarity_matrix = alpha + jax.random.normal(key_sub, alpha.shape)

        ## Initialize model "Similarity"
        key, key_sub = jax.random.split(key)
        model = SetProposal(n_rounds,
                            budget=np.inf,
                            experiment=linear_experiment,
                            d_max=d_max,
                            similarity_matrix=similarity_matrix,
                            selection_method="similarity",
                            key=key_sub
                            )

        # ----------------------------------------------------------------------------------------------------------------
        # 3a. Train model
        # ----------------------------------------------------------------------------------------------------------------
        model.fit()
        print(np.round(model.current_beta, 1))
        print(np.round(beta_0, 1))
        #breakpoint()
        # ----------------------------------------------------------------------------------------------------------------
        # 3b. Train benchmark models
        # ----------------------------------------------------------------------------------------------------------------
        ## Initialize model "Random"
        key, key_sub = jax.random.split(key)
        model_benchmark = SetProposal(n_rounds,
                                      budget=np.inf,
                                      experiment=linear_experiment,
                                      d_max=d_max,
                                      similarity_matrix=similarity_matrix,
                                      selection_method="random",
                                      key=key_sub
                                      )
        model_benchmark.fit()

        ## Initialize model "Single Experiment"
        inst_singlex = model.current_idxs

        inst_first_subset = model._selection_trajectory[0]

        x, y, z = model._experiment.run(inst_first_subset)
        beta_first_subset, P_first_subset, _ = svd_2sls(
            x,
            y,
            z,
            max_iv=len(inst_first_subset)
        )

        inst_union = model.current_idxs

        x, y, z = model._experiment.run(inst_union)
        beta_union, P_union, _ = svd_2sls(
            x,
            y,
            z,
            max_iv=len(inst_union)
        )
        cover_union = check_component_coverage([P_union])

        diff_first_subset_beta.append(beta_first_subset.squeeze() - beta_0)
        diff_singlex_beta.append(beta_union.squeeze() - beta_0)
        singlex_cover.append(np.array(cover_union)[np.newaxis, ...])

        # run experiment of model with the final indices of the model
        x, y, z = model._experiment.run(inst_singlex)
        # estimate beta from single experiment
        beta_singlex, P_singlex, _ = svd_2sls(x, y, z, max_iv=len(inst_singlex))
        cover_singlex = check_component_coverage([P_singlex])

        # ----------------------------------------------------------------------------------------------------------------
        # 4. Compute results
        # ----------------------------------------------------------------------------------------------------------------
        ## Append model performance
        diff_beta.append((model.current_beta.squeeze() - beta_0))
        diff_beta_trajectory.append((np.array(model._estimation_trajectory).squeeze() - beta_0))
        cover_trajectory.append(np.array(model._coverage_trajectory).squeeze())

        ## Append benchmark performance "Random" and "Singlex"
        diff_benchmark_beta.append((model_benchmark.current_beta.squeeze() - beta_0))
        diff_benchmark_beta_trajectory.append((np.array(model_benchmark._estimation_trajectory).squeeze() - beta_0))
        benchmark_cover_trajectory.append(np.array(model_benchmark._coverage_trajectory).squeeze())
        #breakpoint()
        diff_singlex_beta.append(beta_singlex.squeeze() - beta_0)
        singlex_cover.append(np.array(cover_singlex)[np.newaxis, ...])

        ## Append confounder strength and its estimation
        cs.append(conf_strength)
        cs_hat.append(conf_strength_hat)
        mnorm.append(max_norm)
        mnorm_hat.append(max_norm_hat)
        sel_inst.append(model.current_idxs)
        benchmark_sel_inst.append(model_benchmark.current_idxs)

        iter += 1

        # ----------------------------------------------------------------------------------------------------------------
        # 5. Store results as npy-file
        # ----------------------------------------------------------------------------------------------------------------
        save_dict = {
            "diff_beta": np.array(diff_beta),
            "diff_beta_trajectory": np.array(diff_beta_trajectory),
            "diff_benchmark_beta": np.array(diff_benchmark_beta),
            "diff_benchmark_beta_trajectory": np.array(diff_benchmark_beta_trajectory,dtype = object), ## Error occur that different run with different shape
            "cover_trajectory": np.array(cover_trajectory),
            "benchmark_cover_trajectory": np.array(benchmark_cover_trajectory,dtype=object),
            "beta": np.array(beta),
            "beta_obs": np.array(beta_obs),
            "beta_obs_intercept": np.array(beta_obs_intercept),
            "beta_hat": np.array(diff_beta) + np.array(beta_0),
            "conf_strength": np.array(cs),
            "conf_strength_hat": np.array(cs_hat),
            "max_norm": np.array(mnorm),
            "max_norm_hat": np.array(mnorm_hat),
            "sel_inst": np.array(sel_inst,dtype=object),
            "benchmark_sel_inst": np.array(benchmark_sel_inst,dtype=object),
            "singlex_cover": np.array(singlex_cover),
            "diff_singlex_beta": np.array(diff_singlex_beta),
            "diff_first_subset_beta": np.array(diff_first_subset_beta),
            "diff_ols_beta": np.array(diff_ols_beta),
            "err_beta": np.linalg.norm(np.array(diff_beta), axis=1),
            "err_benchmark_beta": np.linalg.norm(np.array(diff_benchmark_beta), axis=1),
            "err_first_subset_beta": np.linalg.norm(np.array(diff_first_subset_beta), axis=1),
            "err_singlex_beta": np.linalg.norm(np.array(diff_singlex_beta), axis=1),
            "err_ols_beta": np.linalg.norm(np.array(diff_ols_beta), axis=1)
        }
        # save results to file called "results.npy"
        np.save(os.path.join(fig_save_path, "results.npy"), save_dict)


if __name__ == '__main__':

    # COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()

    # set up command line arguments
    parser.add_argument("--fig_path", type=str,
                        default="/Users/TSEYU/OneDrive/Desktop/underspecified-iv-master/FiniteSample_Results")
    parser.add_argument("--n_runs", help="Number of runs for finite sample properties.", type=int, default=500)

    # Arguments for Algorithm
    parser.add_argument("--p", help="Number of treatment variables", type=int, default=10)
    parser.add_argument("--d", help="Number of instruments", type=int, default=15)
    parser.add_argument("--d_id", help="Number of instruments that are needed for identification", type=int, default=15)
    parser.add_argument("--n", help="Number of samples per experiment", type=int, default=1000)
    parser.add_argument("--seed", help="Seed.", type=int, default=253)
    parser.add_argument("--n_rounds", help="Optimization, number of rounds.", type=int, default=5)
    parser.add_argument("--d_max", help="Optimization, magnitude of selected set for each round.",
                        type=int, default=3)
    parser.add_argument("--target_rho", type=float, default=0.8)
    parser.add_argument("--confound_x_strength", type=float, default=1.2)
    parser.add_argument("--confound_y_strength", type=float, default=1.5)
    parser.add_argument("--iv_x_strength", type=float, default=1.0)
    parser.add_argument("--beta_strength", type=float, default=1.0)

    args = parser.parse_args()

    d, p, n, d_max, n_rounds, d_id, seed = args.d, args.p, args.n, args.d_max, args.n_rounds, args.d_id, args.seed

    d_id = args.d_id
    fig_path = args.fig_path  # where to save the figures
    n_runs = args.n_runs

    target_rho = args.target_rho
    confound_x_strength = args.confound_x_strength
    confound_y_strength = args.confound_y_strength
    iv_x_strength = args.iv_x_strength
    beta_strength = args.beta_strength

    ## Create directory with name_id
    name_id = "p" + str(p) + "_d" + str(d) + "_d_id" + str(d_id) + "_d_max" + str(d_max) + "_n_rounds" + str(
        n_rounds)

    fig_save_path = os.path.join(fig_path, str(name_id))
    try:
        os.makedirs(fig_save_path)
    except FileExistsError:
        pass

    # run optimization
    run_optimization(
        seed,
        n_runs,
        n,
        p,
        d,
        d_id,
        d_max,
        n_rounds,
        name_id,
        fig_save_path,
        target_rho=target_rho,
        iv_x_strength=iv_x_strength,
        confound_x_strength=confound_x_strength,
        confound_y_strength=confound_y_strength,
        beta_strength=beta_strength)


    # visualize results
    #visualize_results(name_id, fig_save_path)



