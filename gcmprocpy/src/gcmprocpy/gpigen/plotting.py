"""Optional per-year f107d / f107a plots (requires the ``plot`` extra)."""

import os


def make_plots(ds, output_dir="plots"):
    """Write ``f107d_<year>.png`` and ``f107a_<year>.png`` for each year.

    Returns the list of files written. Importing matplotlib lazily keeps it an
    optional dependency for the core pipeline.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install gpigen[plot]"
        ) from exc

    year_day = ds["year_day"].values
    f107d = ds["f107d"].values
    f107a = ds["f107a"].values
    years = [int(str(d)[:4]) for d in year_day]

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for year in sorted(set(years)):
        idx = [i for i, y in enumerate(years) if y == year]
        for var, values, label in (
            ("f107d", f107d[idx], "f107d"),
            ("f107a", f107a[idx], "f107a"),
        ):
            plt.figure(figsize=(10, 6))
            plt.plot(year_day[idx], values, label=f"Year {year}", color="b")
            plt.title(f"{label} vs Year Day ({year})")
            plt.xlabel("Year Day")
            plt.ylabel(label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, f"{var}_{year}.png")
            plt.savefig(path)
            plt.close()
            written.append(path)
    return written
