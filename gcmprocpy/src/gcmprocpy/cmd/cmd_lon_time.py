#!/usr/bin/env python3
from ..plot_gen import plt_lon_time
from ..io import load_datasets, save_output
import argparse
import os

def cmd_parser():
    parser = argparse.ArgumentParser(description="Parser for loading, plotting, and saving")

    # Loading datasets
    parser.add_argument('-dir','--directory', type=str, help='Path to the directory containing the datasets')
    parser.add_argument('-dsf','--dataset_filter', type=str, help='Filter for the dataset file names', default=None)

    # Saving output
    parser.add_argument('-o_dir','--output_directory', type=str, help='Directory where the plot will be saved.', default=os.getcwd())
    parser.add_argument('-o_file','--filename', type=str, required=True, help='Filename for the saved plot.')
    parser.add_argument('-o_format','--output_format', type=str, help='Format of the output plot, e.g., "png", "pdf".', default='jpg')

    # Plotting parameters
    parser.add_argument('-var','--variable_name', type=str, required=True, help='The name of the variable to plot.')
    parser.add_argument('-lat','--latitude', type=float, required=True, help='The specific latitude value for the plot.')
    parser.add_argument('-lvl','--level', type=float, help='The specific level value for the plot.', default=None)
    parser.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable.', default=None)
    parser.add_argument('-ci','--contour_intervals', type=int, help='The number of contour intervals. Defaults to 10.', default=10)
    parser.add_argument('-cv','--contour_value', type=int, help='The value between each contour interval.', default=None)
    parser.add_argument('-si','--symmetric_interval', action='store_true', help='If True, contour intervals symmetric around zero. Defaults to False.')
    parser.add_argument('-cmc','--cmap_color', type=str, help='The color map of the contour.', default=None)
    parser.add_argument('-lc','--line_color', type=str, help='The color for all lines in the plot. Defaults to white.', default='white')
    parser.add_argument('-lon_min','--longitude_minimum', type=float, help='Minimum longitude value for the plot.', default=None)
    parser.add_argument('-lon_max','--longitude_maximum', type=float, help='Maximum longitude value for the plot.', default=None)
    parser.add_argument('--mtime_minimum', type=float, help='Minimum time value for the plot.', default=None)
    parser.add_argument('--mtime_maximum', type=float, help='Maximum time value for the plot.', default=None)
    parser.add_argument('-clean','--clean_plot', action='store_true', help='Generate a clean plot without title/colorbar. Defaults to False.')
    return (parser)


def cmd_plt_lon_time():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory, args.dataset_filter)
    plot = plt_lon_time(datasets, variable_name=args.variable_name, latitude=args.latitude, level=args.level, variable_unit=args.variable_unit, contour_intervals=args.contour_intervals, contour_value=args.contour_value, symmetric_interval=args.symmetric_interval, cmap_color=args.cmap_color, line_color=args.line_color, longitude_minimum=args.longitude_minimum, longitude_maximum=args.longitude_maximum, mtime_minimum=args.mtime_minimum, mtime_maximum=args.mtime_maximum, clean_plot=args.clean_plot)
    save_output(args.output_directory, args.filename, args.output_format, plot)
