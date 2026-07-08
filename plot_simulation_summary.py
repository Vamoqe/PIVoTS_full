import os
import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    return np.load(os.path.join(path, "results.npy"), allow_pickle=True).item()


def norm_error(diff_array):
    arr = np.asarray(diff_array, dtype=float)
    return np.linalg.norm(arr, axis=1)


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
    se = x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
    return mean, se

def print_error_summary(res):
    methods = {}

    if "diff_beta" in res:
        methods["Proposed"] = res["diff_beta"]

    if "diff_benchmark_beta" in res:
        methods["Random"] = res["diff_benchmark_beta"]

    if "diff_first_subset_beta" in res:
        methods["Single subset"] = res["diff_first_subset_beta"]

    if "diff_singlex_beta" in res:
        methods["Union IV"] = res["diff_singlex_beta"]

    if "diff_ols_beta" in res:
        methods["OLS"] = res["diff_ols_beta"]

    print("\nError summary:")
    print("-" * 70)

    for name, diff in methods.items():
        err = norm_error(diff)

        if len(err) == 0:
            print(f"{name:15s} no valid result")
            continue

        print(
            f"{name:15s} "
            f"mean={err.mean():.4f}, "
            f"median={np.median(err):.4f}, "
            f"sd={err.std(ddof=1):.4f}, "
            f"n={len(err)}"
        )

def plot_main_error(result_path, save_path):
    res = load_result(result_path)

    print("\n=== Main experiment ===")
    print_error_summary(res)

    methods = []
    errors = []

    if "diff_beta" in res:
        methods.append("Proposed")
        errors.append(norm_error(res["diff_beta"]))

    if "diff_benchmark_beta" in res:
        methods.append("Random")
        errors.append(norm_error(res["diff_benchmark_beta"]))

    if "diff_first_subset_beta" in res:
        methods.append("Single subset")
        errors.append(norm_error(res["diff_first_subset_beta"]))

    # Omit Union IV if you do not want it
    # if "diff_singlex_beta" in res:
    #     methods.append("Union IV")
    #     errors.append(norm_error(res["diff_singlex_beta"]))

    if "diff_ols_beta" in res:
        methods.append("OLS")
        errors.append(norm_error(res["diff_ols_beta"]))

    # Combine all errors to choose a focused y-axis
    all_err = np.concatenate(errors)
    upper = np.nanquantile(all_err, 0.95)

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
    plt.title("Global estimation error")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fig1_main_error_focused.png"), dpi=300)
    plt.savefig(os.path.join(save_path, "fig1_main_error_focused.pdf"))
    plt.close()

def plot_error_trajectory(result_path, save_path):
    res = load_result(result_path)

    proposed = trajectory_error(res["diff_beta_trajectory"])
    random = trajectory_error(res["diff_benchmark_beta_trajectory"])

    proposed_mean, proposed_se = mean_se(proposed, axis=0)
    random_mean, random_se = mean_se(random, axis=0)

    rounds = np.arange(1, len(proposed_mean) + 1)

    plt.figure(figsize=(7, 5))

    plt.plot(rounds, proposed_mean, marker="o", label="Proposed")
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

    plt.plot(rounds, proposed_mean, marker="o", label="Proposed")
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

    main_result_path = "../Simulation_Results/main/p10_d15_d_id15_d_max3_n_rounds5"
    underid_base_path = "../Simulation_Results/underid"
    rho_base_path = "../Simulation_Results/rho"
    save_path = "../Simulation_Figures"

    plot_main_error(main_result_path, save_path)
    plot_error_trajectory(main_result_path, save_path)
    plot_coverage_trajectory(main_result_path, save_path, threshold=0.3)

    plot_underid_comparison(underid_base_path, save_path)
    plot_rho_comparison(rho_base_path, save_path)