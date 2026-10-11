"""Run the named baseline, allocation and persistence experiments explicitly."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

BASE_PATH = Path(__file__).resolve().parent / "Simulation_Results"
BASELINE_SETTINGS = {
    "p": 10, "d": 15, "d_id": 15, "n": 1000, "d_max": 3,
    "n_rounds": 5, "target_rho": 0.8, "n_runs": 500, "seed": 253,
    "confound_x_strength": 1.2, "confound_y_strength": 1.5,
    "iv_x_strength": 1.0, "beta_strength": 1.0,
    "persistence_mode": "diagonal",
}


def run_one(exp_name, p, d, d_id, n, d_max, n_rounds, target_rho, n_runs, seed,
            *, base_path=BASE_PATH, overwrite=False, **options):
    """Launch one run using this interpreter, independent of the caller's cwd."""
    runner = Path(__file__).resolve().with_name("run_optimization.py")
    parameters = dict(p=p, d=d, d_id=d_id, n=n, d_max=d_max, n_rounds=n_rounds,
                      target_rho=target_rho, n_runs=n_runs, seed=seed, **options)
    command = [sys.executable, "-B", str(runner), "--fig_path", str(Path(base_path).resolve() / exp_name)]
    for name, value in parameters.items():
        if value is not None:
            command.extend(["--" + name, str(value)])
    if overwrite:
        command.append("--overwrite")
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True, cwd=runner.parent)


def build_sweep_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base_path", type=Path, default=BASE_PATH)
    parser.add_argument("--n_runs", type=int, default=BASELINE_SETTINGS["n_runs"])
    parser.add_argument("--n", type=int, default=BASELINE_SETTINGS["n"])
    parser.add_argument("--seed", type=int, default=BASELINE_SETTINGS["seed"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--persistence_mode", choices=("diagonal", "legacy_rescale"), default="diagonal")
    return parser


def sweep_settings(args):
    return dict(BASELINE_SETTINGS, n_runs=args.n_runs, n=args.n, seed=args.seed,
                persistence_mode=args.persistence_mode)


def run_rho_sweep(args):
    settings = sweep_settings(args)
    for rho in (0.2, 0.5, 0.8, 0.95):
        rho_tag = str(rho).replace(".", "p")
        run_one(f"rho/rho{rho_tag}", **dict(settings, target_rho=rho),
                base_path=args.base_path, overwrite=args.overwrite)


def main(argv=None):
    args = build_sweep_parser().parse_args(argv)
    settings = sweep_settings(args)
    run_one("main", **settings, base_path=args.base_path, overwrite=args.overwrite)
    for size, rounds in ((1, 15), (3, 5), (5, 3)):
        run_one("underid", **dict(settings, d_max=size, n_rounds=rounds),
                base_path=args.base_path, overwrite=args.overwrite)
    run_rho_sweep(args)


if __name__ == "__main__":
    main()
