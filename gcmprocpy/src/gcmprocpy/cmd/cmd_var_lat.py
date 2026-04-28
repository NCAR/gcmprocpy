#!/usr/bin/env python3
from ..plot_gen import plt_var_lat
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
    parser.add_argument('-o_format','--output_format', type=str, required=True, help='Format of the output plot, e.g., "png", "pdf".', default='jpg')

    # Plotting parameters
    parser.add_argument('-var','--variable_name', type=str, help='The name of the variable with latitude, longitude, and lev/ilev dimensions')
    parser.add_argument('-lvl','--level', type=float, help='The selected lev/ilev value (or omit to use a height in km via -lt height)')
    parser.add_argument('-t','--time', type=str, help='The selected time, e.g., "2022-01-01T12:00:00"', default=None)
    parser.add_argument('-mt','--mtime', nargs=3, type=int, help='The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins', default=None)
    parser.add_argument('-lon','--longitude', type=str, help='The specific longitude value, or "mean" for zonal mean (default).', default='mean')
    parser.add_argument('-lt','--level_type', type=str, default='pressure', choices=['pressure', 'height'], help='Whether level is specified as pressure or height (km). Defaults to pressure.')
    parser.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable', default=None)
    parser.add_argument('-lat_min','--latitude_minimum', type=float, help='Minimum latitude on the x-axis', default=None)
    parser.add_argument('-lat_max','--latitude_maximum', type=float, help='Maximum latitude on the x-axis', default=None)
    return (parser)




def cmd_plt_var_lat():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory, args.dataset_filter)
    longitude = args.longitude
    if longitude not in (None, 'mean'):
        try:
            longitude = float(longitude)
        except ValueError:
            pass
    plot = plt_var_lat(datasets, args.variable_name, args.level, args.time, args.mtime,
                       longitude=longitude, level_type=args.level_type,
                       variable_unit=args.variable_unit,
                       latitude_minimum=args.latitude_minimum,
                       latitude_maximum=args.latitude_maximum)
    save_output(args.output_directory, args.filename, args.output_format, plot)
