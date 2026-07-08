import os
import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    return np.load(path, allow_pickle=True).item()


def norm_error(diff_array):
    arr = np.asarray(diff_array, dtype=float)
    return np.linalg.norm(arr, axis=1)


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
    print("-" * 60)

    for name, diff in methods.items():
        err = norm_error(diff)
        print(
            f"{name:15s} "
            f"mean={err.mean():.4f}, "
            f"median={np.median(err):.4f}, "
            f"sd={err.std(ddof=1):.4f}"
        )


def plot_fig1(npy_path, save_path, fig_name="fig1_test_conf"):
    res = load_result(npy_path)

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

    if "diff_singlex_beta" in res:
        methods.append("Union IV")
        errors.append(norm_error(res["diff_singlex_beta"]))

    if "diff_ols_beta" in res:
        methods.append("OLS")
        errors.append(norm_error(res["diff_ols_beta"]))

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.boxplot(errors, labels=methods, showmeans=True)
    plt.ylabel(r"$\|\hat{\beta}-\beta_0\|_2$")
    plt.title("Global estimation error under stronger confounding")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, fig_name + ".png"), dpi=300)
    plt.savefig(os.path.join(save_path, fig_name + ".pdf"))
    plt.close()


if __name__ == "__main__":

    npy_path = "../Simulation_Results/underid/p10_d15_d_id15_d_max5_n_rounds3/results.npy"

    # 如果你的資料夾有 rho tag，改用這行：
    # npy_path = "../Simulation_Results/test_conf/p10_d10_d_id10_d_max2_n_rounds5_rho0p8/results.npy"

    save_path = "../Simulation_Figures"

    plot_fig1(
        npy_path=npy_path,
        save_path=save_path,
        fig_name="fig1_test_conf"
    )