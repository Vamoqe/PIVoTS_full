"""Reproducible finite-budget VAR(1) simulation with a noisy-oracle selector."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess

import jax
import numpy as np
from sklearn.linear_model import LinearRegression

from confounder_strength import estimate_conf_strength, compute_conf_strength
from optimization_submodular import LinearExperiment, SetProposal
from svd_2sls import svd_2sls, check_component_coverage
from utils import generate_time_model, generate_time_scenario

RESULT_SCHEMA_VERSION = 2
CONDITIONING = "instrument_history"
DEFAULT_STRENGTHS = {
    "confound_x_strength": 1.2,
    "confound_y_strength": 1.5,
    "iv_x_strength": 1.0,
    "beta_strength": 1.0,
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npy(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    temporary.replace(path)


def _version_info():
    source_dir = Path(__file__).resolve().parent
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "jax", "jaxlib", "scikit-learn", "statsmodels"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    versions["jax_enable_x64"] = bool(jax.config.x64_enabled)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # Hash working files as well: a commit alone does not identify uncommitted code.
    hashes = {}
    for name in ("run_optimization.py", "optimization_submodular.py", "svd_2sls.py",
                 "utils.py", "simulation.py", "dynamic_iv.py", "confounder_strength.py"):
        path = source_dir / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"git_commit": commit, "source_sha256": hashes, "runtime": versions}


def make_round_schedule(d, d_max, n_rounds, budget=None):
    """Return common exact block sizes, stopping at the actual finite IV budget."""
    for name, value in (("d", d), ("d_max", d_max), ("n_rounds", n_rounds)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1
    ):
        raise ValueError("budget must be a positive integer")
    capacity = int(min(d, d_max * n_rounds))
    effective_budget = capacity if budget is None else min(int(budget), capacity)
    remaining = effective_budget
    schedule = []
    while remaining:
        size = min(d_max, remaining)
        schedule.append(int(size))
        remaining -= size
    return schedule, effective_budget


def _validate_run(seed, n_runs, n, p, d_id, target_rho, strengths):
    for name, value in (("n_runs", n_runs), ("n", n), ("p", p), ("d_id", d_id)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n < 5:
        raise ValueError("n must be at least 5")
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed + n_runs + 10000 >= 2**32:
        raise ValueError("seed and trial seeds must fit unsigned 32-bit integers")
    if not np.isfinite(target_rho) or not 0 <= target_rho < 1:
        raise ValueError("target_rho must be in [0, 1)")
    if any(not np.isfinite(v) or v < 0 for v in strengths.values()):
        raise ValueError("strength parameters must be finite and nonnegative")


def _legacy_confounding_diagnostics(beta, m, e, x, y):
    """Retain optional IID-era fields without blocking valid low-rank runs."""
    notes = []
    product = np.linalg.pinv(m.T).T @ e
    if beta @ beta + product @ product == 0:
        theoretical = np.nan
        notes.append("theoretical_ratio_undefined_zero_signal")
    else:
        theoretical = compute_conf_strength(beta, m, e)
    if x.shape[1] < 2:
        estimated = np.nan
        notes.append("legacy_estimator_requires_at_least_two_regressors")
    elif np.linalg.matrix_rank(np.cov(x, rowvar=False)) < x.shape[1]:
        estimated = np.nan
        notes.append("legacy_estimator_requires_invertible_sample_covariance")
    else:
        estimated = estimate_conf_strength(0, x, y)
        if not np.isfinite(estimated):
            notes.append("legacy_estimator_returned_nonfinite_value")
    return theoretical, estimated, notes


def run_optimization(
    seed, n_runs, n, p, d, d_id, d_max, n_rounds, name_id, fig_save_path,
    target_rho=0.8, confound_x_strength=1.2, confound_y_strength=1.5,
    iv_x_strength=1.0, beta_strength=1.0, *, budget=None,
    persistence_mode="diagonal", overwrite=False,
):
    """Run independent samples of a fixed DGP and return schema-v2 results.

    Proposed and random selection use the same exact block-size schedule and
    finite total budget. The proposed selector intentionally uses true alpha plus
    standard-normal noise: this is a noisy-oracle benchmark, not a feasible
    data-estimated acquisition procedure. Full-IV is an unrestricted reference.
    Existing outputs require explicit overwrite; a manifest records completion or
    failure and the number of fully saved trials.
    """
    strengths = dict(confound_x_strength=confound_x_strength,
                     confound_y_strength=confound_y_strength,
                     iv_x_strength=iv_x_strength, beta_strength=beta_strength)
    _validate_run(seed, n_runs, n, p, d_id, target_rho, strengths)
    schedule, effective_budget = make_round_schedule(d, d_max, n_rounds, budget)
    if persistence_mode not in ("diagonal", "legacy_rescale"):
        raise ValueError("persistence_mode must be diagonal or legacy_rescale")
    if not name_id or Path(name_id).name != name_id or name_id in (".", ".."):
        raise ValueError("name_id must be a single nonempty filename component")
    config = {
        "seed": int(seed), "n_runs": int(n_runs), "n": int(n), "p": int(p),
        "d": int(d), "d_id": int(d_id), "d_max": int(d_max),
        "n_rounds_requested": int(n_rounds), "n_rounds_effective": len(schedule),
        "round_sizes": schedule, "budget_requested": None if budget is None else int(budget),
        "budget_effective": effective_budget, "target_rho": float(target_rho),
        **{key: float(value) for key, value in strengths.items()},
        "persistence_mode": persistence_mode,
        "conditioning": CONDITIONING,
        "conditioning_description": "constant and selected I[t-3]; no X-history controls",
        "alignment": {"instrument": "I[t-2]", "regressor": "X[t-1]", "target": "Y[t]"},
        "initialization": "stationary_lyapunov", "burnin": 0,
        "innovation_covariance": "0.1 * identity", "n_confounders": min(3, int(p)),
        "selection": "noisy_oracle", "selection_noise_sd": 1.0,
        "local_estimator": "svd_2sls", "global_estimator": "projection_least_squares",
        "full_iv_role": "unrestricted_reference", "trial_data_seed_rule": "seed + 10000 + trial_id",
        "trial_selection_seed_rule": "jax.random.fold_in(PRNGKey(seed), trial_id)",
        "legacy_confounding_diagnostics": "IID-era compatibility diagnostics; not validated dynamic confounding measures",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_dir = Path(fig_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.npy"
    scenario_path = output_dir / (name_id + ".npy")
    prior_files = [path.name for path in (manifest_path, results_path, scenario_path) if path.exists()]
    if prior_files and not overwrite:
        raise FileExistsError(
            f"Existing output in {output_dir}: {', '.join(prior_files)}. "
            "Use a new directory or explicitly pass overwrite=True / --overwrite."
        )
    previous_manifest = None
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            previous_manifest = {"unreadable": True}
    if prior_files:
        logging.warning("Explicitly overwriting %s; previous and new configs are recorded in manifest.json", output_dir)
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION, "status": "running",
        "started_at": _utc_now(), "completed_trials": 0,
        "config": config, "config_sha256": config_hash, "version": _version_info(),
        "reference_dataset": {"file": scenario_path.name, "role": "reference_only_not_a_trial", "seed": int(seed)},
        "overwritten_files": prior_files,
        "previous_run": None if previous_manifest is None else {
            key: previous_manifest.get(key) for key in ("schema_version", "config_sha256", "config", "status")
        },
    }
    _atomic_json(manifest_path, manifest)
    save_dict = {
        "schema_version": RESULT_SCHEMA_VERSION, "status": "running", "completed_trials": 0,
        "config": config, "config_sha256": config_hash, "version": manifest["version"],
    }
    # In an explicit overwrite, never leave an old completed result beside the
    # new manifest while this run is in progress or if it fails before trial 1.
    _atomic_npy(results_path, save_dict)
    try:
        key = jax.random.PRNGKey(seed)
        A = generate_time_scenario(
            key, d, p, d_id, target_rho=target_rho, seed=seed,
            persistence_mode=persistence_mode, **strengths,
        )
        alpha = A[1][d:d+p, :d].T
        beta = A[1][d+2*p:, d:d+p].flatten()
        e = A[1][d+2*p:, d+p:d+2*p].flatten()
        m = A[1][d:d+p, d+p:d+2*p].T
        beta_0 = np.r_[0.0, beta]
        actual_rho = float(np.max(np.abs(np.linalg.eigvals(A[1]))))
        manifest["dgp"] = {"A": {str(lag): matrix.tolist() for lag, matrix in A.items()},
                           "actual_rho": actual_rho, "beta": beta.tolist()}
        _atomic_json(manifest_path, manifest)
        X0, Y0, I0 = generate_time_model(A, d, p, n, seed=seed, initialization="stationary")
        _atomic_npy(scenario_path, {
            "schema_version": RESULT_SCHEMA_VERSION, "dataset_role": "reference_only_not_a_trial",
            "seed": int(seed), "key": np.asarray(key), "alpha": alpha, "d": d, "p": p,
            "beta": beta, "conf_e": e, "m": m, "A": A, "X": X0, "Y": Y0, "I": I0,
            "config": config, "actual_rho": actual_rho,
        })
        errors = {method: [] for method in ("proposed", "random", "first_subset", "union", "full_iv", "ols")}
        trajectories = {method: [] for method in ("proposed", "random")}
        coverages = {method: [] for method in ("proposed", "random", "union", "full_iv")}
        groups = {method: [] for method in ("proposed", "random")}
        costs = {method: [] for method in ("proposed", "random")}
        selected = {method: [] for method in ("proposed", "random")}
        data_seeds, selection_keys, cs, cs_hat, norms, norm_hats = [], [], [], [], [], []
        ols_estimates, diagnostic_notes = [], []
        for trial_id in range(n_runs):
            logging.info("Trial %s/%s", trial_id + 1, n_runs)
            data_seed = seed + trial_id + 10000
            trial_key = jax.random.fold_in(key, trial_id)
            experiment_key, feature_key, proposed_key, random_key = jax.random.split(trial_key, 4)
            X, Y, I = generate_time_model(A, d, p, n, seed=data_seed, initialization="stationary")
            experiment = LinearExperiment(experiment_key, n, alpha, beta, e, m,
                                          time_setting=True, A=A, X=X, Y=Y, I=I)
            x, y, _ = experiment.run([False] * d)
            reg = LinearRegression(fit_intercept=True).fit(x, y)
            ols_beta = np.r_[reg.intercept_, reg.coef_]
            conf_strength, conf_hat, notes = _legacy_confounding_diagnostics(beta, m, e, x, y)
            features = alpha + jax.random.normal(feature_key, alpha.shape)
            models = {}
            for method, selection_method, selection_key in (
                ("proposed", "similarity", proposed_key), ("random", "random", random_key)
            ):
                model = SetProposal(
                    len(schedule), budget=effective_budget, experiment=experiment,
                    d_max=d_max, similarity_matrix=features, selection_method=selection_method,
                    key=selection_key, round_sizes=schedule,
                )
                model.fit()
                actual_sizes = [len(group) for group in model._selection_trajectory]
                if actual_sizes != schedule or model.cumulative_cost != effective_budget:
                    raise RuntimeError("Selector did not follow the shared exact budget schedule")
                models[method] = model
                errors[method].append(np.asarray(model.current_beta).reshape(-1) - beta_0)
                trajectories[method].append(np.stack(model._estimation_trajectory, axis=0).reshape(len(schedule), p + 1) - beta_0)
                coverages[method].append(np.stack(model._coverage_trajectory, axis=0))
                groups[method].append([list(map(int, group)) for group in model._selection_trajectory])
                costs[method].append(list(map(int, model._cost_trajectory)))
                selected[method].append(list(map(int, model.current_idxs)))
            # Distinct baselines: one selected block, selected union, all instruments.
            selections = {
                "first_subset": models["proposed"]._selection_trajectory[0],
                "union": models["proposed"].current_idxs,
                "full_iv": list(range(d)),
            }
            for method, indices in selections.items():
                x, y, z = experiment.run(indices)
                estimated, projection, _ = svd_2sls(x, y, z, max_iv=len(indices))
                errors[method].append(np.asarray(estimated).reshape(-1) - beta_0)
                if method in coverages:
                    coverages[method].append(check_component_coverage([projection]))
            errors["ols"].append(ols_beta - beta_0)
            ols_estimates.append(ols_beta)
            data_seeds.append(data_seed)
            selection_keys.append(np.asarray(trial_key))
            cs.append(conf_strength)
            cs_hat.append(conf_hat)
            norms.append((1 - conf_strength) * reg.coef_ @ reg.coef_)
            norm_hats.append((1 - conf_hat) * reg.coef_ @ reg.coef_)
            diagnostic_notes.append(notes)
            stacked_errors = {method: np.stack(values, axis=0) for method, values in errors.items()}
            save_dict = {
                "schema_version": RESULT_SCHEMA_VERSION, "status": "running", "completed_trials": trial_id + 1,
                "config": config, "config_sha256": config_hash, "version": manifest["version"],
                "A": A, "actual_rho": actual_rho, "alpha": alpha,
                "trial_ids": np.arange(trial_id + 1), "trial_data_seeds": np.array(data_seeds, dtype=np.uint32),
                "trial_selection_keys": np.stack(selection_keys),
                "beta": beta, "beta_obs": reg.coef_, "beta_obs_intercept": ols_beta,
                "beta_obs_trajectory": np.stack(ols_estimates),
                "beta_hat": stacked_errors["proposed"] + beta_0,
                "diff_beta": stacked_errors["proposed"], "diff_benchmark_beta": stacked_errors["random"],
                "diff_first_subset_beta": stacked_errors["first_subset"], "diff_singlex_beta": stacked_errors["union"],
                "diff_full_iv_beta": stacked_errors["full_iv"], "diff_ols_beta": stacked_errors["ols"],
                "diff_beta_trajectory": np.stack(trajectories["proposed"]),
                "diff_benchmark_beta_trajectory": np.stack(trajectories["random"]),
                "cover_trajectory": np.stack(coverages["proposed"]),
                "benchmark_cover_trajectory": np.stack(coverages["random"]),
                "singlex_cover": np.stack(coverages["union"])[:, None, :],
                "full_iv_cover": np.stack(coverages["full_iv"])[:, None, :],
                "sel_inst": np.asarray(selected["proposed"], dtype=int),
                "benchmark_sel_inst": np.asarray(selected["random"], dtype=int),
                "selection_groups": groups["proposed"], "benchmark_selection_groups": groups["random"],
                "selected_counts": np.array([[len(group) for group in trial] for trial in groups["proposed"]]),
                "benchmark_selected_counts": np.array([[len(group) for group in trial] for trial in groups["random"]]),
                "cumulative_cost_trajectory": np.asarray(costs["proposed"], dtype=int),
                "benchmark_cumulative_cost_trajectory": np.asarray(costs["random"], dtype=int),
                "used_budget": np.array(costs["proposed"], dtype=int)[:, -1],
                "benchmark_used_budget": np.array(costs["random"], dtype=int)[:, -1],
                "conf_strength": np.array(cs), "conf_strength_hat": np.array(cs_hat),
                "max_norm": np.array(norms), "max_norm_hat": np.array(norm_hats),
                "legacy_confounding_diagnostic_notes": diagnostic_notes,
                "baseline_instrument_counts": {"first_subset": schedule[0], "union": effective_budget,
                                               "full_iv": int(d), "ols": 0},
            }
            for method, field in (("proposed", "err_beta"), ("random", "err_benchmark_beta"),
                                  ("first_subset", "err_first_subset_beta"), ("union", "err_singlex_beta"),
                                  ("full_iv", "err_full_iv_beta"), ("ols", "err_ols_beta")):
                save_dict[field] = np.linalg.norm(stacked_errors[method], axis=1)
            _atomic_npy(results_path, save_dict)
            manifest["completed_trials"] = trial_id + 1
            _atomic_json(manifest_path, manifest)
        save_dict["status"] = "complete"
        _atomic_npy(results_path, save_dict)
        manifest.update(status="complete", finished_at=_utc_now())
        _atomic_json(manifest_path, manifest)
        return save_dict
    except Exception as error:
        manifest.update(status="failed", finished_at=_utc_now(),
                        error={"type": type(error).__name__, "message": str(error)})
        _atomic_json(manifest_path, manifest)
        if save_dict is not None:
            save_dict["status"] = "failed"
            _atomic_npy(results_path, save_dict)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig_path", default=str(Path(__file__).resolve().parent / "Simulation_Results"))
    for name, default in (("n_runs", 500), ("p", 10), ("d", 15), ("d_id", 15),
                          ("n", 1000), ("seed", 253), ("n_rounds", 5), ("d_max", 3)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--target_rho", type=float, default=0.8)
    for name, default in DEFAULT_STRENGTHS.items():
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--budget", type=int, default=None, help="Total selected-IV budget; capped by d and round capacity.")
    parser.add_argument("--persistence_mode", choices=("diagonal", "legacy_rescale"), default="diagonal")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing run and record its previous config.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    name_id = f"p{args.p}_d{args.d}_d_id{args.d_id}_d_max{args.d_max}_n_rounds{args.n_rounds}"
    output_dir = Path(args.fig_path) / name_id
    parameters = vars(args).copy()
    parameters.pop("fig_path")
    return run_optimization(name_id=name_id, fig_save_path=output_dir, **parameters)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
