import os
import sys
import inspect
import logging
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import netCDF4
from .containers import ModelDataset

logger = logging.getLogger(__name__)



def load_datasets(directory,dataset_filter = None):
    """
    Loads netCDF datasets for the plotting routines.

    Args:
        directory (str): The location of the directory where the files are stored or the path to a single file.
        dataset_filter (str, optional): The string to filter the NetCDF files to select from (e.g., 'prim', 'sech'). Defaults to None.

    Returns:
        list[ModelDataset]: A list of ModelDataset objects, each containing an xarray.Dataset, filename, and model type.
    """

    datasets=[]
    if os.path.isdir(directory):
        files = sorted(os.listdir(directory))
        logger.info("Loading datasets globally.")
        for file in files:
            if file.endswith('.nc') and (dataset_filter is None or dataset_filter in file):
                file_path = os.path.join(directory, file)
                ds = xr.open_dataset(file_path, chunks='auto', decode_timedelta=False)
                model = 'WACCM-X' if ds.lev.units == 'hPa' else 'TIE-GCM'
                datasets.append(ModelDataset(ds=ds, filename=file, model=model))
    else:
        file_name = os.path.basename(directory)
        ds = xr.open_dataset(directory, chunks='auto', decode_timedelta=False)
        model = 'WACCM-X' if ds.lev.units == 'hPa' else 'TIE-GCM'
        datasets.append(ModelDataset(ds=ds, filename=file_name, model=model))
    return(datasets)

def close_datasets(datasets):
    """
    Closes the xarray datasets.

    Args:
        datasets (list[ModelDataset]): A list of ModelDataset objects.

    Returns:
        None
    """
    for dataset in datasets:
        dataset.ds.close()
    return


def save_derived(datasets, variable_names, overwrite=False, verbose=True):
    """Compute derivable field(s) on the full native grid and append them
    in place to each dataset's source NetCDF file, so subsequent loads read
    them directly instead of recomputing.

    Only *derivable* intermediate fields are persisted this way — quantities
    computed on the full grid from other fields (e.g. ``'N2'`` and the
    composition ratios). Slice-based derived *outputs* (emissions, OH bands,
    EP flux) are not handled here.

    The dataset is closed, the variable is appended via netCDF4, and the file
    is reopened, so the in-memory ``datasets`` keep working and the data cache
    is cleared.

    Args:
        datasets (list[ModelDataset]): Loaded datasets (modified in place).
        variable_names (str | list[str]): Field name(s) to compute and write.
        overwrite (bool): NetCDF cannot delete a variable in place, so a field
            already present on disk is skipped with a warning regardless;
            regenerate into a fresh copy to replace it.
        verbose (bool): Log progress.

    Returns:
        list[str]: ``"<path>:<var>"`` entries actually written.

    Raises:
        ValueError: If a name is not present and not derivable, or a dataset
            has no on-disk source path.
        PermissionError: If a source file is read-only or locked.
    """
    from .data_derived import ensure_field
    from .containers import clear_data_cache

    if isinstance(variable_names, str):
        variable_names = [variable_names]

    written = []
    for mds in datasets:
        path = mds.ds.encoding.get('source')
        if not path:
            raise ValueError(
                f"Dataset '{mds.filename}' has no on-disk source path; cannot persist."
            )

        # 1. Compute + realize the full-grid arrays (dataset must be open).
        pending = {}
        for name in variable_names:
            if not ensure_field(mds, name):
                raise ValueError(
                    f"'{name}' is not present in '{mds.filename}' and is not a "
                    f"derivable field for model {mds.model}. Derivable: requires "
                    f"its inputs to be in the file."
                )
            da = mds.ds[name]
            pending[name] = (tuple(da.dims), np.asarray(da.values, dtype='f8'),
                             dict(da.attrs))

        # 2. Close (HDF5 disallows concurrent read+write opens of the same file).
        mds.ds.close()

        # 3. Append the new variables in place.
        try:
            nc = netCDF4.Dataset(path, 'a')
        except (PermissionError, OSError) as exc:
            mds.ds = xr.open_dataset(path, chunks='auto', decode_timedelta=False)
            raise PermissionError(
                f"Cannot append to '{path}' (read-only or locked): {exc}. "
                f"Persist into a writable copy of the file instead."
            )
        try:
            for name, (dims, arr, attrs) in pending.items():
                if name in nc.variables:
                    if verbose:
                        logger.warning(
                            "'%s' already present in %s; skipping (in-place "
                            "overwrite is unsupported).", name, os.path.basename(path))
                    continue
                missing_dims = [d for d in dims if d not in nc.dimensions]
                if missing_dims:
                    logger.warning("Skipping '%s': dimensions %s not in %s.",
                                   name, missing_dims, os.path.basename(path))
                    continue
                v = nc.createVariable(name, 'f8', dims)
                v[:] = arr
                for key, val in attrs.items():
                    try:
                        v.setncattr(key, val)
                    except Exception:  # pragma: no cover - attribute type quirks
                        pass
                written.append(f"{path}:{name}")
                if verbose:
                    logger.info("Wrote derived '%s' into %s.", name,
                                os.path.basename(path))
        finally:
            nc.close()

        # 4. Reopen the augmented file and rebind the in-memory dataset.
        mds.ds = xr.open_dataset(path, chunks='auto', decode_timedelta=False)
        mds._time_values = mds.ds['time'].values
        mds._time_set = set(mds._time_values)

    clear_data_cache()
    return written


def save_output(output_directory,filename,output_format,plot_object):
    output_directory = os.path.join(output_directory, 'proc')
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, f'{filename}.{output_format}')
    plot_object.savefig(output, format=output_format, bbox_inches='tight', pad_inches=0.5)
    logger.info(f"Plot saved as {filename}")


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
        logger.debug(string)
    return