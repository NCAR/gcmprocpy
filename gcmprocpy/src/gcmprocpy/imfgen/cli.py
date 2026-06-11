"""Command-line interface: ``imfgen``."""

import argparse
import sys

from . import __version__
from .core import generate_imf, generate_imf_years
from .dataset import save_imf


def build_parser():
    p = argparse.ArgumentParser(
        prog="imfgen",
        description="Build TIEGCM IMF NetCDF files from OMNI or BCWIND data.",
    )
    p.add_argument("--version", action="version", version=f"imfgen {__version__}")
    p.add_argument(
        "--source", choices=["omni", "bcwind"], default="omni",
        help="Data source. Default: omni (OMNI 1-minute ASCII).",
    )
    p.add_argument(
        "--start", default=None,
        help="Start date (YYYY-MM-DD, YYYYDDD, or ISO). omni default: 1982-01-01.",
    )
    p.add_argument(
        "--end", default=None,
        help="End date (inclusive). omni default: yesterday.",
    )
    p.add_argument(
        "--window", type=int, default=10,
        help="Trailing-average window in minutes (omni). Default: 10.",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="Directory for omni_min<year>.asc files (omni). Default: cwd.",
    )
    p.add_argument(
        "--bcwind-path", default=None,
        help="Path to the BCWIND HDF5 file (required for --source bcwind).",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Do not fetch missing OMNI files over FTP; use local files only.",
    )
    p.add_argument(
        "--output-dir", default=".",
        help="Directory for the output .nc file(s). Default: cwd.",
    )
    p.add_argument(
        "--output", default=None,
        help="Explicit output path (single-file mode; overrides --output-dir/--prefix).",
    )
    p.add_argument(
        "--prefix", default=None,
        help="Filename prefix: <prefix>_<beg>-<end>.nc. Default: imf_OMNI / imf_bcwind.",
    )
    p.add_argument(
        "--split-years", action="store_true",
        help="Write one file per calendar year instead of a single range file.",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Build the dataset but do not write a file.",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    try:
        if args.split_years:
            if args.source != "omni":
                raise SystemExit("--split-years is only supported for --source omni.")
            if args.output:
                raise SystemExit(
                    "--output cannot be combined with --split-years; "
                    "use --output-dir / --prefix."
                )
            written = []
            for ds in generate_imf_years(
                start=args.start, end=args.end, window=args.window,
                cache_dir=args.cache_dir, download=not args.no_download,
                verbose=verbose,
            ):
                if not args.no_write:
                    written.append(save_imf(ds, output_dir=args.output_dir,
                                            prefix=args.prefix))
            if written:
                print(f"Wrote {len(written)} NetCDF file(s):")
                for p in written:
                    print(f"  {p}")
            return 0

        ds = generate_imf(
            start=args.start,
            end=args.end,
            source=args.source,
            window=args.window,
            cache_dir=args.cache_dir,
            bcwind_path=args.bcwind_path,
            download=not args.no_download,
            verbose=verbose,
        )

        if not args.no_write:
            path = save_imf(ds, output_dir=args.output_dir, prefix=args.prefix,
                            path=args.output)
            print(f"NetCDF file written: {path}")
    except (ValueError, FileNotFoundError) as exc:
        # Turn expected pipeline errors into a clean CLI message, not a traceback.
        raise SystemExit(f"imfgen: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
