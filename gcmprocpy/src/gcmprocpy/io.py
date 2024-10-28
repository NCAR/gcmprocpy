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
        list[tuple]: A list containing tuples, each with an xarray.Dataset object and the corresponding filename in string.
    """

    datasets=[]
    if os.path.isdir(directory):
        files = sorted(os.listdir(directory)) 
        print("Loading datasets globally.") 
        for file in files:
            if file.endswith('.nc') and (dataset_filter is None or dataset_filter in file):
                file_path = os.path.join(directory, file)
                datasets.append([xr.open_dataset(file_path), file])
    else:
        file = os.path.basename(directory)
        datasets.append([xr.open_dataset(directory), file])
    return(datasets)


def save_output(output_directory,filename,output_format,plot_object):
    output_directory = os.path.join(output_directory, 'proc')
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, f'{filename}.{output_format}')
    plot_object.savefig(output, format=output_format, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as {filename}")