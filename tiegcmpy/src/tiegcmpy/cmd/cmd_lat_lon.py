#!/usr/bin/env python3
from ..plot_gen import plt_lat_lon
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
    parser.add_argument('-var','--variable_name', type=str, required=True, help='The name of the variable with latitude, longitude, and lev/ilev dimensions.')
    parser.add_argument('-t','--time', type=str, help='The selected time, e.g., "2022-01-01T12:00:00".')
    parser.add_argument('-mt','--mtime', nargs='+', type=int, help='The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.')
    parser.add_argument('-lvl','--level', type=float, help='The selected lev/ilev value.')
    parser.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable.')
    parser.add_argument('-ci','--contour_intervals', type=int, help='The number of contour intervals. Defaults to 20.')
    parser.add_argument('-cv','--contour_value', type=int, help='The value between each contour interval.')
    parser.add_argument('-si','--symmetric_interval', action='store_true', help='If True, the contour intervals will be symmetric around zero. Defaults to False.')
    parser.add_argument('-cmc','--cmap_color', type=str, help='The color map of the contour. Defaults to "viridis" for Density, "inferno" for Temp, "bwr" for Wind, "viridis" for undefined.')
    parser.add_argument('-lc','--line_color', type=str, default='white', help='The color for all lines in the plot. Defaults to "white".')
    parser.add_argument('-cst','--coastlines', action='store_true', help='Shows coastlines on the plot. Defaults to False.')
    parser.add_argument('-nsh','--nightshade', action='store_true', help='Shows nightshade on the plot. Defaults to False.')
    parser.add_argument('-gm','--gm_equator', action='store_true', help='Shows geomagnetic equator on the plot. Defaults to False.')
    parser.add_argument('-lat_min','--latitude_minimum', type=float, help='Minimum latitude to slice plots. Defaults to -87.5.')
    parser.add_argument('-lat_max','--latitude_maximum', type=float, help='Maximum latitude to slice plots. Defaults to 87.5.')
    parser.add_argument('-lon_min','--longitude_minimum', type=float, help='Minimum longitude to slice plots. Defaults to -180.')
    parser.add_argument('-lon_max','--longitude_maximum', type=float, help='Maximum longitude to slice plots. Defaults to 175.')
    return (parser)




def cmd_plt_lat_lon():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory,args.dataset_filter)
    plot =  plt_lat_lon(datasets, variable_name=args.variable_name, time=args.time, mtime=args.mtime, level=args.level, variable_unit=args.variable_unit, contour_intervals=args.contour_intervals, contour_value=args.contour_value, symmetric_interval=args.symmetric_interval, cmap_color=args.cmap_color, line_color=args.line_color, coastlines=args.coastlines, nightshade=args.nightshade, gm_equator=args.gm_equator, latitude_minimum=args.latitude_minimum, latitude_maximum=args.latitude_maximum, longitude_minimum=args.longitude_minimum, longitude_maximum=args.longitude_maximum)
    save_output(args.output_directory,args.filename,args.output_format,plot)



