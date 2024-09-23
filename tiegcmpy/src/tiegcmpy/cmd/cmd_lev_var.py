#!/usr/bin/env python3
from ..plot_gen import plt_lev_var
from ..io import load_datasets, save_output
import argparse

def cmd_parser():
    parser = argparse.ArgumentParser(description="Main parser")
    subparsers = parser.add_subparsers()

    args_load_datasets = subparsers.add_parser('load', help='Load datasets from a directory')
    args_load_datasets.add_argument('-dir','--directory', type=str, help='Path to the directory containing the datasets')
    args_load_datasets.add_argument('-dsf','--dataset_filter', type=str, help='Filter for the dataset file names', default=None)
    
    args_save_output= subparsers.add_parser('save', help='Save plot to a directroy')
    args_save_output.add_argument('-o_dir','--output_directory', type=str, required=True, help='Directory where the plot will be saved.')
    args_save_output.add_argument('-o_file','--filename', type=str, required=True, help='Filename for the saved plot.')
    args_save_output.add_argument('-o_format','--output_format', type=str, required=True, help='Format of the output plot, e.g., "png", "pdf".')

    args_plot= subparsers.add_parser('plot', help='Plotting parameters')
    args_plot.add_argument('-var','--variable_name', type=str, help='The name of the variable with latitude, longitude, and lev/ilev dimensions')
    args_plot.add_argument('-lat','--latitude', type=float, help='The specific latitude value for the plot')
    args_plot.add_argument('-t','--time', type=str, help='The selected time, e.g., "2022-01-01T12:00:00"', default=None)
    args_plot.add_argument('-mt','--mtime', nargs=3, type=int, help='The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins', default=None)
    args_plot.add_argument('-lon','--longitude', type=float, help='The specific longitude value for the plot', default=None)
    args_plot.add_argument('-unit','--variable_unit', type=str, help='The desired unit of the variable', default=None)
    args_plot.add_argument('-lvl_min','--level_minimum', type=float, help='Minimum level value for the plot', default=None)
    args_plot.add_argument('-lvl_max','--level_maximum', type=float, help='Maximum level value for the plot', default=None)
    return (parser)




def cmd_plt_lev_var():
    parser = cmd_parser()
    args = parser.parse_args()
    datasets = load_datasets(args.directory,args.dataset_filter)
    plot = plt_lev_var(datasets,args.variable_name,args.latitude,args.time,args.mtime,args.longitude,args.variable_unit,args.level_minimum,args.level_maximum)
    save_output(args.output_directory,args.filename,args.output_format,plot)

