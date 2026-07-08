import os
import subprocess


BASE_PATH = "../Simulation_Results_overlap"


def run_one(
    exp_name,
    p,
    d,
    d_id,
    n,
    d_max,
    n_rounds,
    target_rho,
    n_runs,
    seed,
    use_overlap_union=False,
    union_overlap_start_ratio=0.4,
    union_overlap_end_ratio=0.5,
):
    """
    Run one simulation setting.

    Parameters
    ----------
    exp_name : str
        Parent experiment folder name.
    p, d, d_id, n, d_max, n_rounds : int
        Simulation parameters.
    target_rho : float
        VAR spectral radius.
    n_runs : int
        Monte Carlo repetitions.
    seed : int
        Random seed.
    use_overlap_union : bool
        Whether to use reduced-overlap union IV benchmark.
        This requires run_optimization.py to support --use_overlap_union.
    union_overlap_start_ratio, union_overlap_end_ratio : float
        The fraction of sample used as the common-overlap window for Union IV.
        For example, 0.4 to 0.5 means Union IV only uses 40%-50% of the sample.
    """

    fig_path = os.path.join(BASE_PATH, exp_name)

    cmd = [
        "python", "run_optimization.py",
        "--fig_path", fig_path,
        "--p", str(p),
        "--d", str(d),
        "--d_id", str(d_id),
        "--n", str(n),
        "--d_max", str(d_max),
        "--n_rounds", str(n_rounds),
        "--target_rho", str(target_rho),
        "--n_runs", str(n_runs),
        "--seed", str(seed),
    ]

    # Use this only if you added these arguments in run_optimization.py.
    if use_overlap_union:
        cmd.extend([
            "--use_overlap_union",
            "--union_overlap_start_ratio", str(union_overlap_start_ratio),
            "--union_overlap_end_ratio", str(union_overlap_end_ratio),
        ])

    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":

    # ============================================================
    # Global settings
    # ============================================================
    seed = 253
    n_runs = 50

    # Main baseline setting
    p = 10
    d = 10
    d_id = 10
    n = 1000

    # If your overlap union code is not yet implemented in run_optimization.py,
    # set this to False.
    use_overlap_union = True

    # Union IV only uses this overlap window:
    # [0.4n, 0.5n), i.e. 100 observations if n=1000.
    union_overlap_start_ratio = 0.4
    union_overlap_end_ratio = 0.5

    # ============================================================
    # Experiment 1: Main estimation experiment
    # ============================================================
    run_one(
        exp_name="main",
        p=p,
        d=d,
        d_id=d_id,
        n=n,
        d_max=2,
        n_rounds=5,
        target_rho=0.8,
        n_runs=n_runs,
        seed=seed,
        use_overlap_union=use_overlap_union,
        union_overlap_start_ratio=union_overlap_start_ratio,
        union_overlap_end_ratio=union_overlap_end_ratio,
    )

    # ============================================================
    # Experiment 2: Underidentification severity
    # Fixed total IV budget roughly equal to 10.
    # ============================================================
    for d_max, n_rounds in [
        (1, 10),
        (2, 5),
        (5, 2),
    ]:
        run_one(
            exp_name="underid",
            p=p,
            d=d,
            d_id=d_id,
            n=n,
            d_max=d_max,
            n_rounds=n_rounds,
            target_rho=0.8,
            n_runs=n_runs,
            seed=seed,
            use_overlap_union=use_overlap_union,
            union_overlap_start_ratio=union_overlap_start_ratio,
            union_overlap_end_ratio=union_overlap_end_ratio,
        )

    # ============================================================
    # Experiment 3: Time-series dependence
    # ============================================================
    for rho in [0.2, 0.5, 0.8, 0.95]:
        run_one(
            exp_name="rho",
            p=p,
            d=d,
            d_id=d_id,
            n=n,
            d_max=2,
            n_rounds=5,
            target_rho=rho,
            n_runs=n_runs,
            seed=seed,
            use_overlap_union=use_overlap_union,
            union_overlap_start_ratio=union_overlap_start_ratio,
            union_overlap_end_ratio=union_overlap_end_ratio,
        )