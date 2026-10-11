"""Reserved entry point: asynchronous/missing-block experiments are not implemented.

The former script passed nonexistent flags and compared unequal observation
windows. Use run_simulation_grid.py for synchronized baseline checks, or
run_pilot_selection.py for explicitly costed acquisition. A future asynchronous
runner must include a missing-aware stacked-moment GMM comparator.
"""

def main():
    raise SystemExit(__doc__)


if __name__ == '__main__': main()
