import os
import numpy as np
import pandas as pd


def load_result(path):
    result_path = os.path.join(path, "results.npy")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Cannot find results.npy at: {result_path}")
    return np.load(result_path, allow_pickle=True).item()


def norm_error(diff_array):
    arr = np.asarray(diff_array, dtype=float)

    if arr.ndim == 1:
        err = np.abs(arr)
    else:
        err = np.linalg.norm(arr, axis=1)

    err = err[~np.isnan(err)]
    return err


def summarize_errors(res):
    methods = {}

    if "diff_beta" in res:
        methods["Proposed"] = res["diff_beta"]

    if "diff_benchmark_beta" in res:
        methods["Random"] = res["diff_benchmark_beta"]

    if "diff_first_subset_beta" in res:
        methods["Single subset"] = res["diff_first_subset_beta"]

    if "diff_ols_beta" in res:
        methods["OLS"] = res["diff_ols_beta"]

    rows = []

    for name, diff in methods.items():
        err = norm_error(diff)

        rows.append({
            "Method": name,
            "Mean": np.mean(err),
            "Median": np.median(err),
            "SD": np.std(err, ddof=1),
            "Q25": np.quantile(err, 0.25),
            "Q75": np.quantile(err, 0.75),
            "N": len(err),
        })

    return pd.DataFrame(rows)


def save_latex_table(df, save_path, filename):
    os.makedirs(save_path, exist_ok=True)

    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        escape=False,
        column_format="lrrrrrr"
    )

    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
        f.write(latex)


def summarize_underid(base_path):
    settings = [
        ("$d_{\\max}=1, R=15$", "p10_d15_d_id15_d_max1_n_rounds15"),
        ("$d_{\\max}=3, R=5$", "p10_d15_d_id15_d_max3_n_rounds5"),
        ("$d_{\\max}=5, R=3$", "p10_d15_d_id15_d_max5_n_rounds3"),
    ]

    rows = []

    for label, folder in settings:
        res = load_result(os.path.join(base_path, folder))
        err = norm_error(res["diff_beta"])

        rows.append({
            "Setting": label,
            "Mean": np.mean(err),
            "Median": np.median(err),
            "SD": np.std(err, ddof=1),
            "Q25": np.quantile(err, 0.25),
            "Q75": np.quantile(err, 0.75),
            "N": len(err),
        })

    return pd.DataFrame(rows)


def summarize_rho(base_path):
    settings = [
        ("$\\rho=0.2$", os.path.join("rho0p2", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("$\\rho=0.5$", os.path.join("rho0p5", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("$\\rho=0.8$", os.path.join("rho0p8", "p10_d15_d_id15_d_max3_n_rounds5")),
        ("$\\rho=0.95$", os.path.join("rho0p95", "p10_d15_d_id15_d_max3_n_rounds5")),
    ]

    rows = []

    for label, folder in settings:
        res = load_result(os.path.join(base_path, folder))
        err = norm_error(res["diff_beta"])

        rows.append({
            "Setting": label,
            "Mean": np.mean(err),
            "Median": np.median(err),
            "SD": np.std(err, ddof=1),
            "Q25": np.quantile(err, 0.25),
            "Q75": np.quantile(err, 0.75),
            "N": len(err),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    base_path = "../Simulation_Results"
    save_path = "../Simulation_Tables"

    main_path = os.path.join(
        base_path,
        "main",
        "p10_d15_d_id15_d_max3_n_rounds5"
    )

    underid_path = os.path.join(base_path, "underid")
    rho_path = os.path.join(base_path, "rho")

    # Table 3: main experiment
    res_main = load_result(main_path)
    df_main = summarize_errors(res_main)

    print("\nMain experiment summary")
    print(df_main)

    os.makedirs(save_path, exist_ok=True)
    df_main.to_csv(os.path.join(save_path, "table_main_error_summary.csv"), index=False)
    save_latex_table(df_main, save_path, "table_main_error_summary.tex")

    # Table 4: underidentification sensitivity
    df_underid = summarize_underid(underid_path)

    print("\nUnderidentification sensitivity summary")
    print(df_underid)

    df_underid.to_csv(os.path.join(save_path, "table_underid_summary.csv"), index=False)
    save_latex_table(df_underid, save_path, "table_underid_summary.tex")

    # Table 5: rho sensitivity
    df_rho = summarize_rho(rho_path)

    print("\nTime dependence sensitivity summary")
    print(df_rho)

    df_rho.to_csv(os.path.join(save_path, "table_rho_summary.csv"), index=False)
    save_latex_table(df_rho, save_path, "table_rho_summary.tex")

    print("\nTables saved to:")
    print(save_path)