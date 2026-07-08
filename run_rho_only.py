import os
import subprocess

BASE_PATH = "../Simulation_Results"

def run_one(exp_name, p, d, d_id, n, d_max, n_rounds, target_rho, n_runs, seed):
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

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":

    for rho in [0.2, 0.5, 0.8, 0.95]:
        rho_tag = str(rho).replace(".", "p")
        run_one(
            exp_name= f"rho/rho{rho_tag}",
            p=10,
            d=10,
            d_id=10,
            n=1000,
            d_max=2,
            n_rounds=5,
            target_rho=rho,
            n_runs=500,
            seed=253
        )