import numpy as np
from .containers import PlotData
from .data_parse import arr_lat_lon, batch_arr_lat_lon, arr_lev_var, arr_lev_lon, arr_lev_lat, arr_lev_time, arr_lat_time, calc_avg_ht, min_max, get_time


def mkeno53(arr_temp, arr_o, arr_no):
    """
    Calucates 5.3 micron NO emission (from John Wise).
    The formula used is:

    N(5.3 mic) = (2.63E-22) * exp[-2715 / Tk] * [O] * [NO]
                 -----------------------------------------
                   (4 * Pi) * (10.78 + 6.5E-11 * [O])

    Where:
    
    - [O] is the oxygen concentration.
    - [NO] is the nitric oxide concentration.
    - Tk is the temperature in Kelvin.

    .. math::

        N(5.3 \, \mu m) = \\frac{2.63 \\times 10^{-22} \\cdot \\exp\\left(-\\frac{2715}{T_k}\\right) \\cdot [O] \\cdot [NO]}{4 \\pi \\cdot \\left(10.78 + 6.5 \\times 10^{-11} \\cdot [O]\\right)}
    
    Args:
        arr_temp (numpy.ndarray): Array of temperatures in Kelvin.
        arr_o (numpy.ndarray): Array of oxygen concentrations.
        arr_no (numpy.ndarray): Array of nitric oxide concentrations.
    Returns:
        numpy.ndarray: Calculated NO emission at 5.3 microns.
    """
    pi = 3.14156
    NO_emission = (2.63e-22 * np.exp(-2715 / arr_temp) * arr_o * arr_no) / (4 * pi * (10.78 + 6.5e-11 * arr_o))
    return NO_emission



# Function for mkeco215

def mkeco215(arr_temp, arr_o, arr_co2):
    """
    Calucates 15 micron CO2 emission (from John Wise).

    The formula used is:

    N(15 mic) = (5.94E-26) * sqrt(Tk) * exp[-960 / Tk] * [O] * [CO2]
                 ---------------------------------------------------
                     (4 * Pi) * (1.28 + 3.5E-13 * sqrt(Tk) * [O])

    Where:

    - [O] is the oxygen concentration.
    - [CO2] is the carbon dioxide concentration.
    - Tk is the temperature in Kelvin.

    The 15 micron term is only the O-CO2 collisional component,
    but it accounts for at least 80% of the radiance above 110 km.

    .. math::

        N(15 \, \mu m) = \\frac{5.94 \\times 10^{-26} \\cdot \\sqrt{T_k} \\cdot \\exp\\left(-\\frac{960}{T_k}\\right) \\cdot [O] \\cdot [CO_2]}{4 \\pi \\cdot \\left(1.28 + 3.5 \\times 10^{-13} \\cdot \\sqrt{T_k} \\cdot [O]\\right)}
    
    Args:
        arr_temp (numpy.ndarray): Array of temperatures (Tk).
        arr_o (numpy.ndarray): Array of oxygen concentrations [O].
        arr_co2 (numpy.ndarray): Array of CO2 concentrations [CO2].

    Returns:
        numpy.ndarray: Calculated 15 micron CO2 emission.
    """
    pi = 3.14156
    CO2_emission = (5.94e-26 * np.sqrt(arr_temp) * np.exp(-960 / arr_temp) * arr_o * arr_co2) / (4 * pi * (1.28 + 3.5e-13 * np.sqrt(arr_temp) * arr_o))
    return CO2_emission



# Function for mkeoh83
def mkeoh83(arr_temp, arr_o, arr_o2, arr_n2):
    """
    Calculate OH emission for the v(8,3) band.
    
    Args:
        arr_temp (numpy.ndarray): Array of temperatures (K).
        arr_o (numpy.ndarray): Array of atomic oxygen densities (cm^-3).
        arr_o2 (numpy.ndarray): Array of molecular oxygen densities (cm^-3).
        arr_n2 (numpy.ndarray): Array of molecular nitrogen densities (cm^-3).
    Returns:
        numpy.ndarray: Calculated OH emission for the v(8,3) band.
    """
    f8 = 0.29
    pk6n2 = 5.70e-34 * (300 / arr_temp) ** 2.62
    pk6o2 = 5.96e-34 * (300 / arr_temp) ** 2.37
    OH_emission = f8 * arr_o * arr_o2 * (pk6n2 * arr_n2 + pk6o2 * arr_o2) / (260 + 2e-11 * arr_o2)
    return OH_emission

def arr_mkeno53(datasets, variable_name, time, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Generate 5.3-micron NO emission array from datasets.
    This function processes the given datasets to generate an array of 
    5.3-micron NO emissions based on temperature, O1, and NO data.
    
    Requires the following variables:
    - TN or T: Temperature
    - O1 or O: Oxygen
    - NO: Nitric oxide concentration

    Args:
        datasets (list): List of datasets to process.
        variable_name (str): Name of the variable to process.
        time (datetime): Specific time for which data is to be processed.
        selected_lev_ilev (int, optional): Selected level or ilev. Defaults to None.
        selected_unit (str, optional): Unit of the variable. Defaults to None.
        plot_mode (bool, optional): Flag to indicate if plot mode is enabled. Defaults to False.
    Returns:
        Union[xarray.DataArray, tuple]:
            If plot_mode is False, returns an numpy array containing 5.3-micron NO emissions for the specified time and level.
            If plot_mode is True, returns a tuple containing:
            - NO_emission (ndarray): Array of 5.3-micron NO emissions.
            - level (ndarray): Array of levels.
            - unique_lats (ndarray): Array of unique latitudes.
            - unique_lons (ndarray): Array of unique longitudes.
            - str: Empty string placeholder.
            - str: Description of the emission ("5.3-micron NO").
            - selected_mtime (datetime): Selected time.
            - filename (str): Name of the file processed.
    """
    TIEGCM = False
    WACCMX = False

    for mds in datasets:
        if 'TN' in mds.ds.variables:
            TIEGCM = True
            break
        elif 'T' in mds.ds.variables:
            WACCMX = True
            break

    if TIEGCM == True:
        temp_name = 'TN'
        o_name = 'O1'
        no_name = 'NO'
    elif WACCMX == True:
        temp_name = 'T'
        o_name = 'O'
        no_name = 'NO'
    results = batch_arr_lat_lon(datasets, [temp_name, o_name, no_name], time, selected_lev_ilev, selected_unit, plot_mode)
    result_temp, result_o, result_no = results[temp_name], results[o_name], results[no_name]

    if plot_mode:
        NO_emission = mkeno53(result_temp.values, result_o.values, result_no.values)
        return PlotData(
            values=NO_emission, variable_unit="photons cm-3 sec-1",
            variable_long_name="5.3-micron NO", model=result_temp.model,
            filename=result_temp.filename, lats=result_temp.lats,
            lons=result_temp.lons, selected_lev=result_temp.selected_lev,
            mtime=result_temp.mtime,
        )
    else:
        return mkeno53(result_temp, result_o, result_no)

def arr_mkeco215(datasets, variable_name, time, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Generate CO2 emissions using the mkeco215 model based on temperature, oxygen, and CO2 data.
    
    Requires the following variables:
    - TN or T: Temperature
    - O1 or O: Oxygen
    - CO2: CO2 concentration

    Args:
        datasets (list): List of datasets to be used for extracting variables.
        variable_name (str): Name of the variable to be processed.
        time (datetime): Specific time for which the data is to be processed.
        selected_lev_ilev (int, optional): Specific level or ilevel to be selected. Defaults to None.
        selected_unit (str, optional): Unit to which the data should be converted. Defaults to None.
        plot_mode (bool, optional): If True, additional plotting information is returned. Defaults to False.
    Returns:
        Union[xarray.DataArray, tuple]:
            If plot_mode is False, returns an numpy array containing CO2 emissions for the specified time and level.
            If plot_mode is True, returns a tuple containing:
            - CO2_emission (numpy.ndarray): CO2 emissions calculated by the mkeco215 model.
            - level (int): Selected level or ilevel.
            - unique_lats (numpy.ndarray): Unique latitudes.
            - unique_lons (numpy.ndarray): Unique longitudes.
            - str: Placeholder string.
            - str: Long name for the 15-micron CO2 emission.
            - datetime: Selected time.
            - str: Filename of the dataset.
    """
    TIEGCM = False
    WACCMX = False

    for mds in datasets:
        if 'TN' in mds.ds.variables:
            TIEGCM = True
            break
        elif 'T' in mds.ds.variables:
            WACCMX = True
            break

    if TIEGCM == True:
        temp_name = 'TN'
        o_name = 'O1'
        co2_name = 'CO2'
    elif WACCMX == True:
        temp_name = 'T'
        o_name = 'O'
        co2_name = 'CO2'

    results = batch_arr_lat_lon(datasets, [temp_name, o_name, co2_name], time, selected_lev_ilev, selected_unit, plot_mode)
    result_temp, result_o, result_co2 = results[temp_name], results[o_name], results[co2_name]

    if plot_mode:
        CO2_emission = mkeco215(result_temp.values, result_o.values, result_co2.values)
        return PlotData(
            values=CO2_emission, variable_unit="photons cm-3 sec-1",
            variable_long_name="15-micron CO2", model=result_temp.model,
            filename=result_temp.filename, lats=result_temp.lats,
            lons=result_temp.lons, selected_lev=result_temp.selected_lev,
            mtime=result_temp.mtime,
        )
    else:
        return mkeco215(result_temp, result_o, result_co2)

def arr_mkeoh83(datasets, variable_name, time, selected_lev_ilev = None, selected_unit = None, plot_mode = False):
    """
    Generate OH emissions using the mkeoh83 model based on temperature, oxygen, and nitrogen data.
    
    Requires the following variables:
    - TN or T: Temperature
    - O1 or O: Oxygen
    - O2: Molecular oxygen
    - N2: Molecular nitrogen

    Args:
        datasets (list): List of datasets to be used for extracting variables.
        variable_name (str): Name of the variable to be processed.
        time (datetime): Specific time for which the data is to be processed.
        selected_lev_ilev (int, optional): Specific level or ilevel to be selected. Defaults to None.
        selected_unit (str, optional): Unit to which the data should be converted. Defaults to None.
        plot_mode (bool, optional): If True, additional plotting information is returned. Defaults to False.
    Returns:
        Union[numpy.ndarray, tuple]:
            If plot_mode is False, returns a numpy array containing OH emissions for the specified time and level.
            If plot_mode is True, returns a tuple containing:
            - OH_emission (numpy.ndarray): OH emissions calculated by the mkeoh83 model.
            - level (int): Selected level or ilevel.
            - unique_lats (numpy.ndarray): Unique latitudes.
            - unique_lons (numpy.ndarray): Unique longitudes.
            - str: Placeholder string.
            - str: Long name for the OH v(8,3) emission.
            - datetime: Selected time.
            - str: Filename of the dataset.
    """
    TIEGCM = False
    WACCMX = False

    for mds in datasets:
        if 'TN' in mds.ds.variables:
            TIEGCM = True
            break
        elif 'T' in mds.ds.variables:
            WACCMX = True
            break

    if TIEGCM == True:
        temp_name = 'TN'
        o_name = 'O1'
        o2_name = 'O2'
        n2_name = 'N2'
    elif WACCMX == True:
        temp_name = 'T'
        o_name = 'O'
        o2_name = 'O2'
        n2_name = 'N2'

    results = batch_arr_lat_lon(datasets, [temp_name, o_name, o2_name, n2_name], time, selected_lev_ilev, selected_unit, plot_mode)
    result_temp, result_o = results[temp_name], results[o_name]
    result_o2, result_n2 = results[o2_name], results[n2_name]

    if plot_mode:
        OH_emission = mkeoh83(result_temp.values, result_o.values, result_o2.values, result_n2.values)
        return PlotData(
            values=OH_emission, variable_unit="photons cm-3 sec-1",
            variable_long_name="OH v(8,3)", model=result_temp.model,
            filename=result_temp.filename, lats=result_temp.lats,
            lons=result_temp.lons, selected_lev=result_temp.selected_lev,
            mtime=result_temp.mtime,
        )
    else:
        return mkeoh83(result_temp, result_o, result_o2, result_n2)


    

    # Function for OH v(8,3) emission