"""SLURM-compatible entry point sharing the local runner's complete CLI.

Run this file through sbatch with the same arguments as run_optimization.py.
Configuration, seeds, runtime versions and completion status are written to
manifest.json by the shared runner. Plotting is a separate reporting step.
"""

from run_optimization import main


if __name__ == "__main__":
    main()
