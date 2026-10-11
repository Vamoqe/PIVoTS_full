import os
import argparse
from pathlib import Path
from make_simulation_tables import load_result, summarize_errors
import numpy as np
import matplotlib.pyplot as plt


def norm_error(diff_array):
    arr = np.asarray(diff_array, dtype=float)
    error = np.linalg.norm(arr, axis=1)
    return error[np.isfinite(error)]  # Counts of omitted failures are saved in the summary CSV.


def trajectory_error(diff_traj):
    """
    diff_traj shape:
        n_runs x n_rounds x p
    """
    arr = np.asarray(diff_traj, dtype=float)
    return np.linalg.norm(arr, axis=2)


def coverage_score(cover_traj, threshold=0.3):
    """
    Smaller angle means better coverage.

    score = fraction of coordinates covered.
    """
    arr = np.asarray(cover_traj, dtype=float)
    return (arr < threshold).mean(axis=2)


def mean_se(x, axis=0):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis)
    se = x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis]) if x.shape[axis]>1 else np.zeros_like(mean)
    return mean, se

def print_error_summary(res):
    print(summarize_errors(res).to_string(index=False))

def plot_main_error(result_path, save_path):
    res = load_result(result_path)

    print("\n=== Main experiment ===")
    print_error_summary(res)

    methods = []
    errors = []

    if "diff_beta" in res:
        methods.append("Proposed (noisy oracle)")
        errors.append(norm_error(res["diff_beta"]))

    if "diff_benchmark_beta" in res:
        methods.append("Random")
        errors.append(norm_error(res["diff_benchmark_beta"]))

    if "diff_first_subset_beta" in res:
        methods.append("Single subset")
        errors.append(norm_error(res["diff_first_subset_beta"]))

    for field,label in [("diff_singlex_beta","Selected-union IV"),("diff_full_iv_beta","Full IV (unrestricted)")]:
        if field in res:
            methods.append(label)
            errors.append(norm_error(res[field]))

    if "diff_ols_beta" in res:
        methods.append("OLS")
        errors.append(norm_error(res["diff_ols_beta"]))

    # Combine all errors to choose a focused y-axis
    all_err = np.concatenate(errors)
    if not len(all_err):
        raise ValueError("No finite errors to plot; inspect failure counts")
    upper = max(np.quantile(all_err, 0.95), 1e-12)

    # Add small margin
    upper = upper * 1.15

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        errors,
        labels=methods,
        showmeans=True,
        showfliers=False,   # hide extreme outliers
        whis=(5, 95)        # whiskers show 5% to 95%
    )

    plt.ylim(0, upper)
    plt.ylabel(r"$\|\hat{\beta}-\beta_0\|_2$")
    plt.title("Global estimation error (focused axis; finite trials)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig1_main_error_focused.png"), dpi=300)
    plt.savefig(os.path.join(save_path, "fig1_main_error_focused.pdf"))
    plt.close()
    summarize_errors(res).to_csv(Path(save_path)/"error_summary_with_failures.csv",index=False)
    plt.figure(figsize=(9,5))
    plt.boxplot(errors,labels=methods,showfliers=True)
    plt.ylabel(r"$\|\hat{\beta}-\beta_0\|_2$")
    plt.title("Global estimation error — full range")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(Path(save_path)/"fig1_main_error_full.png",dpi=200)
    plt.savefig(Path(save_path)/"fig1_main_error_full.pdf")
    plt.close()

def plot_error_trajectory(result_path, save_path):
    res = load_result(result_path)

    proposed = trajectory_error(res["diff_beta_trajectory"])
    random = trajectory_error(res["diff_benchmark_beta_trajectory"])

    proposed_mean, proposed_se = mean_se(proposed, axis=0)
    random_mean, random_se = mean_se(random, axis=0)

    rounds = np.arange(1, len(proposed_mean) + 1)

    plt.figure(figsize=(7, 5))

    plt.plot(rounds, proposed_mean, marker="o", label="Proposed (noisy oracle)")
    plt.fill_between(
        rounds,
        proposed_mean - 1.96 * proposed_se,
        proposed_mean + 1.96 * proposed_se,
        alpha=0.2
    )

    plt.plot(rounds, random_mean, marker="o", label="Random")
    plt.fill_between(
        rounds,
        random_mean - 1.96 * random_se,
        random_mean + 1.96 * random_se,
        alpha=0.2
    )

    plt.xlabel("Round")
    plt.ylabel(r"$\|\hat{\beta}^{(r)}-\beta_0\|_2$")
    plt.title("Sequential reconstruction error")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig2_error_trajectory.pdf"))
    plt.savefig(os.path.join(save_path, "fig2_error_trajectory.png"), dpi=300)
    plt.close()


def plot_coverage_trajectory(result_path, save_path, threshold=0.3):
    res = load_result(result_path)

    proposed = coverage_score(res["cover_trajectory"], threshold=threshold)
    random = coverage_score(res["benchmark_cover_trajectory"], threshold=threshold)

    proposed_mean, proposed_se = mean_se(proposed, axis=0)
    random_mean, random_se = mean_se(random, axis=0)

    rounds = np.arange(1, len(proposed_mean) + 1)

    plt.figure(figsize=(7, 5))

    plt.plot(rounds, proposed_mean, marker="o", label="Proposed (noisy oracle)")
    plt.fill_between(
        rounds,
        proposed_mean - 1.96 * proposed_se,
        proposed_mean + 1.96 * proposed_se,
        alpha=0.2
    )

    plt.plot(rounds, random_mean, marker="o", label="Random")
    plt.fill_between(
        rounds,
        random_mean - 1.96 * random_se,
        random_mean + 1.96 * random_se,
        alpha=0.2
    )

    plt.xlabel("Round")
    plt.ylabel("Projection coverage score")
    plt.title("Accumulated projection-space coverage")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig3_projection_coverage.pdf"))
    plt.savefig(os.path.join(save_path, "fig3_projection_coverage.png"), dpi=300)
    plt.close()


def plot_underid_comparison(base_path, save_path):
    settings = [
        ("dmax=1, rounds=15", "p10_d15_d_id15_d_max1_n_rounds15"),
        ("dmax=3, rounds=5", "p10_d15_d_id15_d_max3_n_rounds5"),
        ("dmax=5, rounds=3", "p10_d15_d_id15_d_max5_n_rounds3"),
    ]

    labels = []
    errors = []

    for label, folder in settings:
        path = os.path.join(base_path, folder)
        res = load_result(path)

        labels.append(label)
        errors.append(norm_error(res["diff_beta"]))

    # 對敏感性分析不要用太高的 quantile
    # 0.85 或 0.90 都可以，建議先用 0.90
    all_err = np.concatenate(errors)
    upper = np.nanquantile(all_err, 0.90) * 1.10

    # 也可以設定一個合理下限，避免圖太扁
    upper = max(upper, 0.4)

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        errors,
        labels=labels,
        showmeans=True,
        showfliers=False,
        whis=(10, 90)
    )

    plt.ylim(0, 2.0)
    plt.ylabel(r"$\|\hat{\beta}-\beta_0\|_2$")
    plt.title("Sensitivity to underidentification severity")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig4_underid_severity_focused.png"), dpi=300)
    plt.savefig(os.path.join(save_path, "fig4_underid_severity_focused.pdf"))
    plt.close()



def plot_rho_comparison(base_path, save_path):
    settings = [
        ("rho=0.2", os.path.join("rho0p2", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("rho=0.5", os.path.join("rho0p5", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("rho=0.8", os.path.join("rho0p8", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("rho=0.95", os.path.join("rho0p95", "p10_d15_d_id15_d_max3_n_rounds5")),
    ]

    labels = []
    errors = []

    for label, folder in settings:
        path = os.path.join(base_path, folder)
        res = load_result(path)

        labels.append(label)
        errors.append(norm_error(res["diff_beta"]))

    # 對敏感性分析使用較保守的 y-axis
    all_err = np.concatenate(errors)
    upper = np.nanquantile(all_err, 0.90) * 1.10

    upper = max(upper, 0.4)

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        errors,
        labels=labels,
        showmeans=True,
        showfliers=False,
        whis=(10, 90)
    )

    plt.ylim(0, 2.0)
    plt.ylabel(r"$\|\hat{\beta}-\beta_0\|_2$")
    plt.title("Sensitivity to time-series dependence")
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig5_time_dependence_focused.png"), dpi=300)
    plt.savefig(os.path.join(save_path, "fig5_time_dependence_focused.pdf"))
    plt.close()
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=root/"Simulation_Results")
    parser.add_argument("--save_path", type=Path, default=root/"Simulation_Figures")
    parser.add_argument("--result_path", type=Path, help="Plot one run instead of the full grid")
    args = parser.parse_args()
    main_path = args.result_path or args.base_path/"main"/"p10_d15_d_id15_d_max3_n_rounds5"
    plot_main_error(main_path, args.save_path)
    plot_error_trajectory(main_path, args.save_path)
    plot_coverage_trajectory(main_path, args.save_path, threshold=0.3)
    if args.result_path is None:
        plot_underid_comparison(args.base_path/"underid", args.save_path)
        plot_rho_comparison(args.base_path/"rho", args.save_path)
