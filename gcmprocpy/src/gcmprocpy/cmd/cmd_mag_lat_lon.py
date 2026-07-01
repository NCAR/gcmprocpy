#!/usr/bin/env python3
from ..plot_gen import plt_mag_lat_lon
from ..io import load_datasets, save_output
import argparse
import os


def cmd_parser():
    parser = argparse.ArgumentParser(description="Parser for loading, plotting, and saving")

    # Loading datasets
    parser.add_argument('-dir', '--directory', type=str, help='Path to the directory containing the datasets')
    parser.add_argument('-dsf', '--dataset_filter', type=str, help='Filter for the dataset file names', default=None)

    # Saving output
    parser.add_argument('-o_dir', '--output_directory', type=str, help='Directory where the plot will be saved.', default=os.getcwd())
    parser.add_argument('-o_file', '--filename', type=str, required=True, help='Filename for the saved plot.')
    parser.add_argument('-o_format', '--output_format', type=str, required=True, help='Format of the output plot, e.g., "png", "pdf".', default='jpg')

    # Plotting parameters
    parser.add_argument('-var', '--variable_name', type=str, help='The name of the variable to plot')
    parser.add_argument('-t', '--time', type=str, help='The selected time, e.g., "2022-01-01T12:00:00"', default=None)
    parser.add_argument('-mt', '--mtime', nargs=3, type=int, help='The selected time as a list, e.g., [1, 12, 0]', default=None)
    parser.add_argument('-lev', '--level', type=float, default=None,
                        help="Vertical level. Required for geographic variables (sets the geographic->magnetic conversion height); for native-magnetic variables it selects the magnetic vertical level ('mean' if omitted).")
    parser.add_argument('-lt', '--level_type', type=str, default='pressure', choices=['pressure', 'height'], help='Level type: pressure or height (km). Defaults to pressure.')
    parser.add_argument('-unit', '--variable_unit', type=str, help='The desired unit of the variable', default=None)
    parser.add_argument('-ci', '--contour_intervals', type=int, help='Number of contour intervals', default=20)
    parser.add_argument('-cv', '--contour_value', type=float, help='Value between each contour interval', default=None)
    parser.add_argument('-si', '--symmetric_interval', action='store_true', help='Center the color scale on zero')
    parser.add_argument('-cmap', '--cmap_color', type=str, help='Colormap', default=None)
    parser.add_argument('-cmin', '--cmap_lim_min', type=float, help='Minimum color-scale value', default=None)
    parser.add_argument('-cmax', '--cmap_lim_max', type=float, help='Maximum color-scale value', default=None)
    parser.add_argument('-lc', '--line_color', type=str, help='Contour line color', default='white')
    parser.add_argument('-mlat_min', '--mlat_minimum', type=float, help='Minimum magnetic latitude', default=None)
    parser.add_argument('-mlat_max', '--mlat_maximum', type=float, help='Maximum magnetic latitude', default=None)
    parser.add_argument('-mlon_min', '--mlon_minimum', type=float, help='Minimum magnetic longitude', default=None)
    parser.add_argument('-mlon_max', '--mlon_maximum', type=float, help='Maximum magnetic longitude', default=None)
    parser.add_argument('-grid', '--grid', action='store_true', help='Overlay coordinate grid lines on the plot.')
    parser.add_argument('-clean', '--clean_plot', action='store_true', help='Hide the subtext.')
    return parser


def cmd_plt_mag_lat_lon():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory, args.dataset_filter)
    plot = plt_mag_lat_lon(
        datasets, args.variable_name, time=args.time, mtime=args.mtime, level=args.level,
        level_type=args.level_type, variable_unit=args.variable_unit,
        contour_intervals=args.contour_intervals, contour_value=args.contour_value,
        symmetric_interval=args.symmetric_interval, cmap_color=args.cmap_color,
        cmap_lim_min=args.cmap_lim_min, cmap_lim_max=args.cmap_lim_max, line_color=args.line_color,
        mlat_minimum=args.mlat_minimum, mlat_maximum=args.mlat_maximum,
        mlon_minimum=args.mlon_minimum, mlon_maximum=args.mlon_maximum,
        grid=args.grid, clean_plot=args.clean_plot,
    )
    save_output(args.output_directory, args.filename, args.output_format, plot)
