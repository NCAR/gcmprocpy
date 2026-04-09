#!/usr/bin/env python3
from ..plot_gen import plt_var_time
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
    parser.add_argument('-lon','--longitude', type=float, required=True, help='The specific longitude value for the plot.')
    parser.add_argument('-lvl','--level', type=float, help='The specific level value for the plot.', default=None)
    parser.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable.', default=None)
    parser.add_argument('--mtime_minimum', type=float, help='Minimum time value for the plot.', default=None)
    parser.add_argument('--mtime_maximum', type=float, help='Maximum time value for the plot.', default=None)
    parser.add_argument('-clean','--clean_plot', action='store_true', help='Generate a clean plot without title/colorbar. Defaults to False.')
    return (parser)


def cmd_plt_var_time():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory, args.dataset_filter)
    plot = plt_var_time(datasets, variable_name=args.variable_name, latitude=args.latitude, longitude=args.longitude, level=args.level, variable_unit=args.variable_unit, mtime_minimum=args.mtime_minimum, mtime_maximum=args.mtime_maximum, clean_plot=args.clean_plot)
    save_output(args.output_directory, args.filename, args.output_format, plot)
