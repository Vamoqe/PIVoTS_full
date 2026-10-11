"""Run only the persistence sweep using exactly the full-grid baseline settings."""

from run_simulation_grid import build_sweep_parser, run_rho_sweep


def main(argv=None):
    args = build_sweep_parser(__doc__).parse_args(argv)
    run_rho_sweep(args)


if __name__ == "__main__":
    main()
