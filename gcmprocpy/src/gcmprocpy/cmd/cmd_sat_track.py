#!/usr/bin/env python3
from ..plot_gen import plt_sat_track
from ..io import load_datasets, save_output
import argparse
import numpy as np
import os

def cmd_parser():
    parser = argparse.ArgumentParser(description="Parser for loading, plotting, and saving satellite track interpolation plots")

    # Loading datasets
    parser.add_argument('-dir','--directory', type=str, help='Path to the directory containing the datasets')
    parser.add_argument('-dsf','--dataset_filter', type=str, help='Filter for the dataset file names', default=None)

    # Saving output
    parser.add_argument('-o_dir','--output_directory', type=str, help='Directory where the plot will be saved.', default=os.getcwd())
    parser.add_argument('-o_file','--filename', type=str, required=True, help='Filename for the saved plot.')
    parser.add_argument('-o_format','--output_format', type=str, help='Format of the output plot, e.g., "png", "pdf".', default='jpg')

    # Plotting parameters
    parser.add_argument('-var','--variable_name', type=str, required=True, help='The name of the variable to plot.')
    parser.add_argument('-sat_file','--satellite_file', type=str, required=True, help='Path to CSV file with satellite track data (columns: time, lat, lon).')
    parser.add_argument('-lvl','--level', type=float, help='The specific level value. If omitted, plots all levels as contour.', default=None)
    parser.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable.', default=None)
    parser.add_argument('-ci','--contour_intervals', type=int, help='The number of contour intervals. Defaults to 10.', default=10)
    parser.add_argument('-cv','--contour_value', type=int, help='The value between each contour interval.', default=None)
    parser.add_argument('-si','--symmetric_interval', action='store_true', help='If True, contour intervals symmetric around zero. Defaults to False.')
    parser.add_argument('-cmc','--cmap_color', type=str, help='The color map of the contour.', default=None)
    parser.add_argument('-lc','--line_color', type=str, help='The color for all lines in the plot. Defaults to white.', default='white')
    parser.add_argument('-clean','--clean_plot', action='store_true', help='Generate a clean plot without title/colorbar. Defaults to False.')
    return (parser)


def cmd_plt_sat_track():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory, args.dataset_filter)

    # Load satellite track from CSV (columns: time, lat, lon)
    import csv
    sat_time = []
    sat_lat = []
    sat_lon = []
    with open(args.satellite_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            sat_time.append(np.datetime64(row[0]))
            sat_lat.append(float(row[1]))
            sat_lon.append(float(row[2]))
    sat_time = np.array(sat_time)
    sat_lat = np.array(sat_lat)
    sat_lon = np.array(sat_lon)

    plot = plt_sat_track(datasets, variable_name=args.variable_name, sat_time=sat_time, sat_lat=sat_lat, sat_lon=sat_lon, level=args.level, variable_unit=args.variable_unit, contour_intervals=args.contour_intervals, contour_value=args.contour_value, symmetric_interval=args.symmetric_interval, cmap_color=args.cmap_color, line_color=args.line_color, clean_plot=args.clean_plot)
    save_output(args.output_directory, args.filename, args.output_format, plot)
