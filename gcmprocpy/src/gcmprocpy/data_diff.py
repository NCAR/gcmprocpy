"""Difference field computation for perturbation vs control comparisons."""
import numpy as np
from .containers import PlotData


def compute_diff(pert_values, cntr_values, diff_type='RAW'):
    """Compute difference between perturbation and control arrays.

    Args:
        pert_values (numpy.ndarray): Perturbation field values.
        cntr_values (numpy.ndarray): Control field values.
        diff_type (str): 'RAW' for simple subtraction, 'PERCENT' for
            percentage difference ``(pert - cntr) / cntr * 100``.

    Returns:
        numpy.ndarray: Difference values.  For PERCENT mode, grid points
        where ``|cntr| <= 1e-20`` are set to zero to avoid division by zero.

    Raises:
        ValueError: If *pert_values* and *cntr_values* have different shapes
            or *diff_type* is not 'RAW' or 'PERCENT'.
    """
    pert_values = np.asarray(pert_values, dtype=float)
    cntr_values = np.asarray(cntr_values, dtype=float)

    if pert_values.shape != cntr_values.shape:
        raise ValueError(
            f"Shape mismatch: pert {pert_values.shape} vs cntr {cntr_values.shape}"
        )

    diff_type = diff_type.upper()

    if diff_type == 'RAW':
        return pert_values - cntr_values
    elif diff_type == 'PERCENT':
        diff = np.zeros_like(pert_values)
        safe = np.abs(cntr_values) > 1e-20
        diff[safe] = (pert_values[safe] - cntr_values[safe]) / cntr_values[safe] * 100.0
        return diff
    else:
        raise ValueError(f"diff_type must be 'RAW' or 'PERCENT', got '{diff_type}'")


def diff_plotdata(pert_result, cntr_result, diff_type='RAW'):
    """Compute difference between two PlotData objects.

    Args:
        pert_result (PlotData): Perturbation result from an ``arr_*`` function.
        cntr_result (PlotData): Control result from the same ``arr_*`` function.
        diff_type (str): 'RAW' or 'PERCENT'.

    Returns:
        PlotData: New PlotData with difference values and an updated
        ``variable_long_name`` that includes the diff type label.

    Raises:
        ValueError: If the two PlotData value arrays have different shapes.
    """
    diff_values = compute_diff(pert_result.values, cntr_result.values, diff_type)

    label = "RAW diff" if diff_type.upper() == 'RAW' else "PERCENT diff"
    long_name = f"{pert_result.variable_long_name} ({label})"
    unit = pert_result.variable_unit if diff_type.upper() == 'RAW' else '%'

    return PlotData(
        values=diff_values,
        variable_unit=unit,
        variable_long_name=long_name,
        model=pert_result.model,
        filename=pert_result.filename,
        levs=pert_result.levs,
        lats=pert_result.lats,
        lons=pert_result.lons,
        mtime=pert_result.mtime,
        mtime_values=pert_result.mtime_values,
        selected_lat=pert_result.selected_lat,
        selected_lon=pert_result.selected_lon,
        selected_lev=pert_result.selected_lev,
    )
