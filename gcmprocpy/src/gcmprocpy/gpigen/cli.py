"""Command-line interface: ``gpigen``."""

import argparse
import sys

from . import __version__
from .core import generate_gpi
from .dataset import save_gpi


def build_parser():
    p = argparse.ArgumentParser(
        prog="gpigen",
        description="Build TIEGCM GPI NetCDF files from GFZ Potsdam data.",
    )
    p.add_argument("--version", action="version", version=f"gpigen {__version__}")
    p.add_argument(
        "--start",
        default="1960-01-01",
        help="Start date (YYYY-MM-DD, YYYYDDD, or ISO). Default: 1960-01-01.",
    )
    p.add_argument(
        "--end",
        default=None,
        help="End date (inclusive). Default: yesterday.",
    )
    p.add_argument(
        "--source",
        choices=["json", "textfile"],
        default="json",
        help="Data source. Default: json (GFZ JSON API).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=81,
        help="Averaging window in days for f107a. Default: 81.",
    )
    p.add_argument(
        "--trailing",
        action="store_true",
        help="Use a trailing average instead of centered.",
    )
    p.add_argument(
        "--status",
        default="def",
        help="GFZ 'status' query param (json source). Default: def.",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the output .nc file. Default: cwd.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Explicit output path (overrides --output-dir/--prefix).",
    )
    p.add_argument(
        "--prefix",
        default="gpi",
        help="Filename prefix: <prefix>_<beg>-<end>.nc. Default: gpi.",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Build the dataset but do not write a file.",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Also write per-year f107d/f107a PNGs (needs the 'plot' extra).",
    )
    p.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory for plots. Default: ./plots.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Where to store the downloaded text file (textfile source).",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    try:
        ds = generate_gpi(
            start=args.start,
            end=args.end,
            source=args.source,
            window=args.window,
            centered=not args.trailing,
            status=args.status,
            cache_dir=args.cache_dir,
            verbose=verbose,
        )

        if not args.no_write:
            path = save_gpi(
                ds, output_dir=args.output_dir, prefix=args.prefix, path=args.output
            )
            print(f"NetCDF file written: {path}")

        if args.plots:
            from .plotting import make_plots

            written = make_plots(ds, output_dir=args.plots_dir)
            print(f"Wrote {len(written)} plot(s) to {args.plots_dir}/")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # Turn expected pipeline errors into a clean CLI message, not a traceback.
        raise SystemExit(f"gpigen: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
