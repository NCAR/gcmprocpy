#!/usr/bin/env python3
import os
import sys
import inspect
import logging
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from .getoptions import get_options
from .plot_gen import plt_lat_lon, plt_lev_var, plt_lev_lon, plt_lev_lat, plt_lev_time, plt_lat_time
from .containers import ModelDataset
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_options()

    if args.recursive:  # If recursive flag is provided, enter into recursive mode immediately.
        cached_datasets = []
        if args.directory:
            files = sorted(os.listdir(args.directory))
            logger.info("Loading datasets globally.")
            for file in files:
                if file.endswith('.nc') and (args.dataset_filter is None or args.dataset_filter in file):
                    file_path = os.path.join(args.directory, file)
                    ds = xr.open_dataset(file_path)
                    model = 'WACCM-X' if ds.lev.units == 'hPa' else 'TIE-GCM'
                    cached_datasets.append(ModelDataset(ds=ds, filename=file, model=model))
        if args.multiple_output:
            filename = args.multiple_output + '.pdf'
            output_directory = os.path.join(args.output_directory, 'proc')
            os.makedirs(output_directory, exist_ok=True)
            output = os.path.join(output_directory, filename)

            with PdfPages(output) as pdf:
                while True:  # Keep running until the user inputs "exit"
                    print("Enter command or 'exit' to terminate: ")
                    next_command = input()
                    if next_command.strip().lower() == 'exit':
                        break
                    else:
                        sys.argv = [sys.argv[0]] + next_command.split()
                        args = get_options()
                        plot = plot_routine(args, cached_datasets=cached_datasets, multiple_output=True)
                        pdf.savefig(plot, bbox_inches='tight', pad_inches=0.5)
                        plt.close(plot)
        else:
            while True:  # Keep running until the user inputs "exit"
                print("Enter command or 'exit' to terminate: ")
                next_command = input()
                if next_command.strip().lower() == 'exit':
                    break
                else:
                    sys.argv = [sys.argv[0]] + next_command.split()
                    args = get_options()
                    plot_routine(args, cached_datasets=cached_datasets)  # Extract the common execution logic to a new function
    else:
        plot_routine(args)  # Execute the command once if recursive flag is not provided.



def plot_routine(args, cached_datasets=None, multiple_output=False):
    #
    # Map plot types to their respective functions
    #
    plot_functions = {
        'lat_lon': plt_lat_lon,
        'lev_var': plt_lev_var,
        'lev_lon': plt_lev_lon,
        'lev_lat': plt_lev_lat,
        'lev_time': plt_lev_time,
        'lat_time': plt_lat_time,
    }
    #
    # Get the plotting function based on the user input plot type
    #
    plot_function = plot_functions.get(args.plot_type)

    if plot_function:
        #
        # Checking if a directory is provided in the arguments
        #
        datasets = []
        if cached_datasets:
            datasets = cached_datasets
        else:
            if args.directory:
                files = sorted(os.listdir(args.directory))
                logger.info("Loading datasets.")
                for file in files:
                    if file.endswith('.nc') and (args.dataset_filter is None or args.dataset_filter in file):
                        file_path = os.path.join(args.directory, file)
                        ds = xr.open_dataset(file_path)
                        model = 'WACCM-X' if ds.lev.units == 'hPa' else 'TIE-GCM'
                        datasets.append(ModelDataset(ds=ds, filename=file, model=model))
            elif args.dataset:
                file = args.dataset
                if file.endswith('.nc'):
                    ds = xr.open_dataset(file)
                    model = 'WACCM-X' if ds.lev.units == 'hPa' else 'TIE-GCM'
                    datasets.append(ModelDataset(ds=ds, filename=file, model=model))

        #
        # Check and validate the specified time argument
        #
        if args.time: 
            available_times = set()  
            args.time = np.datetime64(args.time, 'ns')
            for mds in datasets:
                times = mds.ds['time'].values
                available_times.update(times)
            if np.datetime64(args.time) not in available_times:
                raise ValueError(f"The specified time {args.time} is not available in the datasets.")
        #
        # Check and validate the specified time argument
        #
        if args.mtime:
            available_mtimes = set()
            for mds in datasets:
                mtimes = [tuple(m) for m in mds.ds['mtime'].values]
                available_mtimes.update(mtimes)
            input_mtime = tuple(args.mtime)  
            if input_mtime not in available_mtimes:
                sorted_mtimes = sorted(list(available_mtimes))  
                raise ValueError(f"The specified mtime {args.mtime} is not available in the datasets.") #Available mtimes are {sorted_mtimes}")
            args.mtime = input_mtime
        #
        # Build the arguments dictionary for the plotting function based on its signature
        #
        function_args = {}
        for param in inspect.signature(plot_function).parameters.keys():
            if hasattr(args, param):
                function_args[param] = getattr(args, param)
            elif param == 'datasets':
                function_args['datasets'] = datasets
        

        

        #
        # Call the plotting function with the constructed arguments
        #
        plot_object = plot_function(**function_args)
        #
        # Save the plot if an output format is specified
        #
        if multiple_output:
            #
            # Return plot_object to build muilti plot pdfs
            #
            return(plot_object)
        else:
            #
            # Set name to file if provided
            #                     
            if args.standard_output:
                filename = f"{args.standard_output}.{args.output_format}"
            else:
                #
                # Extract non-None arguments values and convert them to strings
                #
                    #arg_strings = [f"{key}={value}" for key, value in function_args.items() if value is not None and key != 'datasets']
                keyexections = ['datasets', 'output_directory']
                arg_strings = [f"{value}" for key, value in function_args.items() if value is not None and key not in keyexections]
                filename_prefix = f"plt[{args.plot_type}]_" +'_'.join(arg_strings)
                filename = f"{filename_prefix}.{args.output_format}"
            #
            # Check if output_directory is provided, else use the default current working directory
            #
            output_directory = os.path.join(args.output_directory, 'proc')
            os.makedirs(output_directory, exist_ok=True)
            output = os.path.join(output_directory, filename)
            #
            # Save plot as the given format type
            #
            plot_object.savefig(output, format=args.output_format, bbox_inches='tight', pad_inches=0.5)
            logger.info(f"Plot saved as {filename}")
    else:
        logger.error(f"Invalid plot_type: {args.plot_type}")


if __name__ == "__main__":
    main()


