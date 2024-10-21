#!/usr/bin/env python3
from ..plot_gen import plt_lev_lon
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
    parser.add_argument('-var','--variable_name', type=str, help="The name of the variable with latitude, longitude, and lev/ilev dimensions.")
    parser.add_argument('-lat','--latitude', type=float, help="The specific latitude value for the plot.")
    parser.add_argument('-t','--time', type=str, help="The selected time, e.g., '2022-01-01T12:00:00'.", default=None)
    parser.add_argument('-mt','--mtime', type=int, nargs=3, help="The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.", default=None)
    parser.add_argument('-unit','--variable_unit', type=str, help="The desired unit of the variable.", default=None)
    parser.add_argument('-ci','--contour_intervals', type=int, help="The number of contour intervals. Defaults to 20.", default=20)
    parser.add_argument('-cv','--contour_value', type=int, help="The value between each contour interval.", default=None)
    parser.add_argument('-si','--symmetric_interval', action='store_true', help="If True, the contour intervals will be symmetric around zero. Defaults to False.")
    parser.add_argument('-cmc','--cmap_color', type=str, help="The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.", default=None)
    parser.add_argument('-lc','--line_color', type=str, help="The color for all lines in the plot. Defaults to 'white'.", default='white')
    parser.add_argument('-lvl_min','--level_minimum', type=float, help="Minimum level value for the plot. Defaults to None.", default=None)
    parser.add_argument('-lvl_max','--level_maximum', type=float, help="Maximum level value for the plot. Defaults to None.", default=None)
    parser.add_argument('-lon_min','--longitude_minimum', type=float, help="Minimum longitude value for the plot. Defaults to -180.", default=-180)
    parser.add_argument('-lon_max','--longitude_maximum', type=float, help="Maximum longitude value for the plot. Defaults to 175.", default=175)
    return (parser)




def cmd_plt_lev_lon():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory,args.dataset_filter)
    plot = plt_lev_lon(datasets, variable_name=args.variable_name, latitude=args.latitude, time=args.time, mtime=args.mtime, variable_unit=args.variable_unit, contour_intervals=args.contour_intervals, contour_value=args.contour_value, symmetric_interval=args.symmetric_interval, cmap_color=args.cmap_color, line_color=args.line_color, level_minimum=args.level_minimum, level_maximum=args.level_maximum, longitude_minimum=args.longitude_minimum)
    save_output(args.output_directory,args.filename,args.output_format,plot)

