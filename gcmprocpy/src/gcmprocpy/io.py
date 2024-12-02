import os
import sys  
import inspect
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np



def load_datasets(directory,dataset_filter = None):
    """
    Loads netCDF datasets for the plotting routines.

    Args:
        directory (str): The location of the directory where the files are stored or the path to a single file.
        dataset_filter (str, optional): The string to filter the NetCDF files to select from (e.g., 'prim', 'sech'). Defaults to None.

    Returns:
        list[tuple]: A list containing tuples, each with an xarray.Dataset object and the corresponding filename and model in string.
    """

    datasets=[]
    if os.path.isdir(directory):
        files = sorted(os.listdir(directory)) 
        print("Loading datasets globally.") 
        for file in files:
            if file.endswith('.nc') and (dataset_filter is None or dataset_filter in file):
                file_path = os.path.join(directory, file)
                dataset = xr.open_dataset(file_path)
                file_name = file
                if dataset.lev.units == 'hPa':
                    model = 'WACCM-X'
                else:
                    model = 'TIE-GCM'
                datasets.append([dataset, file_name, model])
    else:
        file_name = os.path.basename(directory)
        dataset = xr.open_dataset(directory)
        if dataset.lev.units == 'hPa':
            model = 'WACCM-X'
        else:
            model = 'TIE-GCM'
        datasets.append([dataset, file_name, model])
    return(datasets)

def close_datasets(datasets):
    """
    Closes the xarray datasets.

    Args:
        datasets (list[tuple]): A list containing tuples, each with an xarray.Dataset object and the corresponding filename and model in string.

    Returns:
        None
    """
    for dataset in datasets:
        dataset[0].close()
    return

def save_output(output_directory,filename,output_format,plot_object):
    output_directory = os.path.join(output_directory, 'proc')
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, f'{filename}.{output_format}')
    plot_object.savefig(output, format=output_format, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as {filename}")


def print_handler(string, verbose):
    """
    Prints a string if verbose is set to True.
    
    Args:
        string (str): The string to print.
        verbose (bool): A boolean to determine if the string should be printed.
    
    Returns:
        None
    """
    if verbose:
        print(string)
    return