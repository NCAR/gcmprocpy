import logging
import numpy as np
import matplotlib.pyplot as plt
from .data_parse import arr_lat_lon, batch_arr_lat_lon, arr_lev_var,arr_lev_lon, arr_lev_lat,arr_lev_time,arr_lat_time, arr_lon_time, arr_var_time, arr_sat_track, arr_var_lat, arr_var_lon, calc_avg_ht, min_max, get_time, height_to_pres_level, interpolate_to_height

logger = logging.getLogger(__name__)
from .containers import resolve_derived
import gcmprocpy.data_emissions  
import gcmprocpy.data_oh         
import gcmprocpy.data_epflux     
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from cartopy.util import add_cyclic_point
from datetime import datetime, timezone
import matplotlib.ticker as mticker
import matplotlib.path as mpath
from matplotlib import get_backend
import math
import geomag
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import mplcursors

def is_notebook():
    """
    Detects if the code is running inside a Jupyter Notebook.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def longitude_to_local_time(longitude):
    """
    Convert longitude to local time.

    Args:
        longitude (float): Longitude value.

    Returns:
        float: Local time corresponding to the given longitude.
    """

    local_time = (longitude / 15) % 24
    return local_time

def local_time_to_longitude(local_time):
    """
    Convert local time to longitude.

    Args:
        local_time (float): Local time value.

    Returns:
        float: Longitude corresponding to the given local time.
    """
    if local_time == 'mean':
        longitude = 'mean'
    else:
        #
        # Each hour of local time corresponds to 15 degrees of longitude
        #
        longitude = (local_time * 15) % 360
        #
        # Adjusting the longitude to be between -180 and 180 degrees
        #
        if longitude > 180:
            longitude = longitude - 360

    return longitude

def _level_label(level, avg_ht=None, original_height=None):
    """Return a subtitle string for the level, respecting height mode."""
    if level == 'mean':
        return 'ZP= Mean'
    if original_height is not None:
        return 'HT=' + str(original_height) + 'KM'
    if avg_ht is not None:
        return 'ZP=' + str(level) + ' AVG HT=' + str(avg_ht) + 'KM'
    return 'ZP=' + str(level)


def color_scheme(variable_name, model=None):
    """
    Sets color scheme for plots based on variable name and model type.

    Args:
        variable_name (str): The name of the variable.
        model (str, optional): The model type ('TIE-GCM' or 'WACCM-X'). If provided, uses model-specific mappings.

    Returns:
        tuple:
            str: Color scheme of the contour map.
            str: Color scheme of contour lines.
    """
    from .containers import MODEL_DEFAULTS

    if model and model in MODEL_DEFAULTS:
        defaults = MODEL_DEFAULTS[model]
        for category in ('density', 'temperature_type', 'wind', 'electric', 'radiation'):
            if category in defaults and variable_name in defaults[category]['vars']:
                return defaults[category]['cmap'], defaults[category]['line_color']
    else:
        # Check all models for a match
        for m in MODEL_DEFAULTS.values():
            for category in ('density', 'temperature_type', 'wind', 'electric', 'radiation'):
                if category in m and variable_name in m[category]['vars']:
                    return m[category]['cmap'], m[category]['line_color']

    return 'viridis', 'white'



def _polar_boundary():
    """Returns a circular boundary path for polar stereographic plots."""
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * radius + center)


def _compute_gm_equator_lats(unique_lons):
    """Compute geomagnetic equator latitudes for an array of longitudes.

    Hoisted out of plt_lat_lon's three projection branches so the geomag
    solver (slow) runs once per plot instead of once per branch.
    """
    gm = geomag.geomag.GeoMag()
    return [gm.GeoMag(0, lon).dec for lon in unique_lons]


def _quiver_overlay(ax, x_coords, y_coords, u_values, v_values, *,
                    density=15, scale=None, color='black', transform=None):
    """Draw a thinned quiver overlay on the given axes.

    No-op if either component is None. Keeps stride/scale/color logic
    in one place so callers (lat-lon projections, lev-lat/lev-lon
    slices) don't duplicate the call.
    """
    if u_values is None or v_values is None:
        return
    d = density
    kwargs = {'color': color, 'scale': scale, 'zorder': 5}
    if transform is not None:
        kwargs['transform'] = transform
    ax.quiver(x_coords[::d], y_coords[::d],
              u_values[::d, ::d], v_values[::d, ::d], **kwargs)


def _polar_ring_labels(ax, hemisphere, time, label_type='lt', fontsize=9):
    """Draw 12 local-time or longitude labels around a polar stereographic ring.

    Mirrors the tgcmproc IDL ``labpol.pro`` look — labels at every 30° of
    azimuth, just outside the map boundary, with a footer caption.

    label_type: 'lt' for solar local time (HH), 'lon' for geographic longitude (deg).
    """
    if label_type not in ('lt', 'lon'):
        return

    ut_hours = (time.astype('datetime64[s]').astype('int64') / 3600.0) % 24
    boundary_lat = 40 if hemisphere == 'north' else -40
    proj = ccrs.PlateCarree()

    for lon in range(-180, 180, 30):
        x_data, y_data = ax.projection.transform_point(lon, boundary_lat, proj)
        x_disp, y_disp = ax.transData.transform((x_data, y_data))
        x_axes, y_axes = ax.transAxes.inverted().transform((x_disp, y_disp))
        dx = x_axes - 0.5
        dy = y_axes - 0.5
        r = math.sqrt(dx * dx + dy * dy)
        if r == 0:
            continue
        offset = 0.05
        x_lab = 0.5 + (dx / r) * (r + offset)
        y_lab = 0.5 + (dy / r) * (r + offset)

        if label_type == 'lt':
            lt = (ut_hours + lon / 15.0) % 24
            label = f"{int(round(lt)) % 24:02d}"
        else:
            label = f"{int(lon)}"

        ax.text(x_lab, y_lab, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=fontsize)

    caption = 'SOLAR LOCAL TIME (HRS)' if label_type == 'lt' else 'GEOGRAPHIC LONGITUDE (DEG)'
    ax.text(0.5, -0.07, caption, transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize)


def _polar_panel(ax, unique_lons, unique_lats, variable_values, contour_levels,
                 cmap_color, cmap_lim_min, cmap_lim_max, line_color, coastlines,
                 nightshade, time, gm_equator_lats, hemisphere,
                 wind_u_values=None, wind_v_values=None, wind_density=15,
                 wind_scale=None, wind_color='black', polar_label='lt'):
    """Draws a single polar contour panel on the given axes."""
    ax.set_boundary(_polar_boundary(), transform=ax.transAxes)
    if coastlines:
        ax.add_feature(cfeature.COASTLINE, edgecolor=line_color, linewidth=1.5)
    if nightshade:
        ax.add_feature(Nightshade(datetime.fromtimestamp(time.astype('O') / 1e9, tz=timezone.utc), alpha=0.4))
    if gm_equator_lats is not None:
        ax.plot(unique_lons, gm_equator_lats, color=line_color, linestyle='--',
                transform=ccrs.Geodetic(), label='Geomagnetic Equator')

    if hemisphere == 'north':
        ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

    cf = ax.contourf(unique_lons, unique_lats, variable_values, cmap=cmap_color,
                     levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max,
                     transform=ccrs.PlateCarree())
    cl = ax.contour(unique_lons, unique_lats, variable_values, colors=line_color,
                    linewidths=0.5, levels=contour_levels, transform=ccrs.PlateCarree())
    ax.clabel(cl, inline=True, fontsize=8, colors=line_color)

    _quiver_overlay(ax, unique_lons, unique_lats, wind_u_values, wind_v_values,
                    density=wind_density, scale=wind_scale, color=wind_color,
                    transform=ccrs.PlateCarree())

    show_inline_labels = polar_label not in ('lt', 'lon')
    gl = ax.gridlines(draw_labels=show_inline_labels, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    if hemisphere == 'north':
        gl.ylocator = mticker.FixedLocator(np.arange(40, 91, 10))
    else:
        gl.ylocator = mticker.FixedLocator(np.arange(-90, -39, 10))

    _polar_ring_labels(ax, hemisphere, time, label_type=polar_label)

    title = 'North Pole' if hemisphere == 'north' else 'South Pole'
    ax.set_title(title, fontsize=14)
    return cf


def plt_lat_lon(datasets, variable_name, time= None, mtime=None, level = None, level_type = 'pressure', variable_unit = None, center_longitude = 0, central_latitude = 0, projection = 'mercator', contour_intervals = None, contour_value = None,symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white', coastlines=False, nightshade=False, gm_equator=False, latitude_minimum = None, latitude_maximum = None, longitude_minimum = None, longitude_maximum = None, wind = False, wind_density = 15, wind_scale = None, wind_color = 'black', polar_label = 'lt', clean_plot = False, verbose = False ):

    """
    Generates a Latitude vs Longitude contour plot for a variable.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
        level (float, optional): The selected lev/ilev value.
        variable_unit (str, optional): The desired unit of the variable.
        center_longitude (float, optional): The central longitude for the plot. Defaults to 0.
        central_latitude (float, optional): The central latitude for the plot. Used by the orthographic projection to set the viewing angle. Defaults to 0.
        projection (str, optional): Map projection type. Options: 'mercator' (default), 'north_polar', 'south_polar', 'polar' (both hemispheres side by side), 'orthographic' (satellite view), 'mollweide' (equal-area).
        contour_intervals (int, optional): The number of contour intervals. Defaults to 20. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
        nightshade (bool, optional): Shows nightshade on the plot. Defaults to False.
        gm_equator (bool, optional): Shows geomagnetic equator on the plot. Defaults to False.
        latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to -87.5.
        latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to 87.5.
        longitude_minimum (float, optional): Minimum longitude to slice plots. Defaults to -180.
        longitude_maximum (float, optional): Maximum longitude to slice plots. Defaults to 175.
        wind (bool, optional): Overlay wind vectors on the plot. Uses model-specific defaults (TIE-GCM: UN/VN, WACCM-X: U/V). Defaults to False.
        wind_density (int, optional): Stride for thinning wind vectors (every Nth point). Defaults to 15.
        wind_scale (float, optional): Scale factor for quiver arrows. Larger values make arrows shorter. Defaults to None (auto-scaled).
        wind_color (str, optional): Color of the wind vectors. Defaults to 'black'.
        polar_label (str, optional): Perimeter labels for polar projections. 'lt' (default) draws solar local-time labels at every 30° azimuth (the tgcmproc look), 'lon' draws geographic longitude labels, None disables ring labels and falls back to inline gridline labels. Ignored for non-polar projections.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    # Printing Execution data
    if time is None:
        time = get_time(datasets, mtime)
    if contour_intervals is None:
        contour_intervals = 20
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(level)+"]---------------")
    # Generate 2D arrays, extract variable_unit
    '''
    if level is not None:
        try:
            data, level,  unique_lats, unique_lons, variable_unit, variable_long_name, selected_mtime, filename =lat_lon_lev(datasets, variable_name, time, level, variable_unit)
        except ValueError:
            data, level,  unique_lats, unique_lons, variable_unit, variable_long_name, selected_mtime, filename =lat_lon_ilev(datasets, variable_name, time, level, variable_unit)
        if level != 'mean':
            avg_ht=calc_avg_ht(datasets, time,level)
    else:
        data, unique_lats, unique_lons, variable_unit, variable_long_name, selected_mtime, filename =lat_lon(datasets, variable_name, time)
    '''
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    # Convert height to pressure level if needed
    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        level = height_to_pres_level(datasets, time, _original_height)

    handler, is_derived = resolve_derived(variable_name)
    if is_derived:
        result = handler(datasets, variable_name, time, selected_lev_ilev=level, selected_unit=variable_unit, plot_mode=True)
    else:
        result = arr_lat_lon(datasets, variable_name, time, selected_lev_ilev=level, selected_unit=variable_unit, plot_mode=True)

    variable_values = result.values
    level = result.selected_lev
    unique_lats = result.lats
    unique_lons = result.lons
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    model = result.model
    filename = result.filename

    # WACCM-X uses 0 to 360 longitude range, convert to -180 to 180
    unique_lons = np.where(unique_lons > 180, unique_lons - 360, unique_lons)
    sorted_indices = np.argsort(unique_lons)
    unique_lons = unique_lons[sorted_indices]
    variable_values = variable_values[:, sorted_indices]

    # Extract wind vector data if wind overlay is enabled
    wind_u_values = None
    wind_v_values = None
    if wind:
        from .containers import MODEL_DEFAULTS
        wind_u = MODEL_DEFAULTS[model]['wind_u']
        wind_v = MODEL_DEFAULTS[model]['wind_v']
        wind_results = batch_arr_lat_lon(datasets, [wind_u, wind_v], time,
                                         selected_lev_ilev=level, plot_mode=True)
        if wind_results is not None:
            wind_u_values = wind_results[wind_u].values[:, sorted_indices]
            wind_v_values = wind_results[wind_v].values[:, sorted_indices]

    # Adjust cyclic point handling for central_longitude=180
    if wind_u_values is not None:
        wind_u_values, _ = add_cyclic_point(wind_u_values, coord=unique_lons, axis=1)
        wind_v_values, _ = add_cyclic_point(wind_v_values, coord=unique_lons, axis=1)
    variable_values, unique_lons = add_cyclic_point(variable_values, coord=unique_lons, axis=1)

    if level != 'mean' and level is not None:
            avg_ht=calc_avg_ht(datasets, time,level)
    if latitude_minimum is None:
        latitude_minimum = np.nanmin(unique_lats)
    if latitude_maximum is None:
        latitude_maximum = np.nanmax(unique_lats)
    if longitude_minimum is None:
        longitude_minimum = -180
    if longitude_maximum is None:
        longitude_maximum = 180
    min_val, max_val = min_max(variable_values)
    selected_day=selected_mtime[0]
    selected_hour=selected_mtime[1]
    selected_min=selected_mtime[2]
    selected_sec=selected_mtime[3]

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max
        
    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)
    # Extract values, latitudes, and longitudes from the array
    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)  # Divide by 2 to get intervals for one side
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)  # Generate negative levels symmetrically
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))


    # Generate contour plot

    interval_value = contour_value if contour_value else (max_val - min_val) / (contour_intervals - 1)

    # Compute geomagnetic equator once (shared across all projection branches)
    gm_equator_lats = _compute_gm_equator_lats(unique_lons) if gm_equator else None

    # ---- Polar projection branch ----
    if projection in ('north_polar', 'south_polar', 'polar'):
        polar_args = dict(unique_lons=unique_lons, unique_lats=unique_lats,
                          variable_values=variable_values, contour_levels=contour_levels,
                          cmap_color=cmap_color, cmap_lim_min=cmap_lim_min,
                          cmap_lim_max=cmap_lim_max, line_color=line_color,
                          coastlines=coastlines, nightshade=nightshade, time=time,
                          gm_equator_lats=gm_equator_lats,
                          wind_u_values=wind_u_values, wind_v_values=wind_v_values,
                          wind_density=wind_density, wind_scale=wind_scale,
                          wind_color=wind_color, polar_label=polar_label)

        if projection == 'polar':
            plot = plt.figure(figsize=(18, 10))
            ax_n = plot.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
            cf = _polar_panel(ax_n, hemisphere='north', **polar_args)
            ax_s = plot.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo())
            _polar_panel(ax_s, hemisphere='south', **polar_args)
            plot.subplots_adjust(left=0.05, right=0.88, wspace=0.15, top=0.82, bottom=0.12)
            cbar_ax = plot.add_axes([0.91, 0.15, 0.02, 0.55])
        elif projection == 'north_polar':
            plot = plt.figure(figsize=(9, 10))
            ax_n = plot.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
            cf = _polar_panel(ax_n, hemisphere='north', **polar_args)
            plot.subplots_adjust(left=0.05, right=0.82, top=0.82, bottom=0.12)
            cbar_ax = plot.add_axes([0.86, 0.15, 0.03, 0.55])
        else:  # south_polar
            plot = plt.figure(figsize=(9, 10))
            ax_s = plot.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
            cf = _polar_panel(ax_s, hemisphere='south', **polar_args)
            plot.subplots_adjust(left=0.05, right=0.82, top=0.82, bottom=0.12)
            cbar_ax = plot.add_axes([0.86, 0.15, 0.03, 0.55])

        cbar = plot.colorbar(cf, cax=cbar_ax, orientation='vertical')
        cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
        cbar.ax.tick_params(labelsize=9)

        if not clean_plot:
            title_str = variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')'
            if level is not None:
                title_str += '\n' + _level_label(level, avg_ht, _original_height)
            plot.suptitle(title_str, fontsize=16, y=0.94)
            minmax_str = "Min, Max = " + str("{:.2e}".format(min_val)) + ", " + str("{:.2e}".format(max_val))
            ci_str = "Contour Interval = " + str("{:.2e}".format(interval_value))
            time_str = "Time=" + str(time.astype('M8[s]').astype(datetime))
            mtime_str = "Day,Hr,Min,Sec=" + str(selected_day) + "," + str(selected_hour) + "," + str(selected_min) + "," + str(selected_sec)
            plot.text(0.5, 0.06, minmax_str + "   " + ci_str + "\n" + time_str + "   " + mtime_str + "   " + str(filename),
                      ha='center', va='center', fontsize=10, transform=plot.transFigure)

        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, center_longitude, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum, cf, unique_lons, unique_lats, variable_values
        return plot

    # ---- Orthographic (satellite view) projection branch ----
    if projection == 'orthographic':
        plot = plt.figure(figsize=(10, 10))
        ax = plot.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
            central_longitude=center_longitude, central_latitude=central_latitude))

        if coastlines:
            ax.add_feature(cfeature.COASTLINE, edgecolor=line_color, linewidth=1.5)
        if nightshade:
            ax.add_feature(Nightshade(datetime.fromtimestamp(time.astype('O')/1e9, tz=timezone.utc), alpha=0.4))
        if gm_equator_lats is not None:
            ax.plot(unique_lons, gm_equator_lats, color=line_color, linestyle='--',
                    transform=ccrs.Geodetic(), label='Geomagnetic Equator')

        ax.set_global()
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))

        cf = ax.contourf(unique_lons, unique_lats, variable_values, cmap=cmap_color,
                         levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max,
                         transform=ccrs.PlateCarree())
        cl = ax.contour(unique_lons, unique_lats, variable_values, colors=line_color,
                        linewidths=0.5, levels=contour_levels, transform=ccrs.PlateCarree())
        ax.clabel(cl, inline=True, fontsize=8, colors=line_color)

        _quiver_overlay(ax, unique_lons, unique_lats, wind_u_values, wind_v_values,
                        density=wind_density, scale=wind_scale, color=wind_color,
                        transform=ccrs.PlateCarree())

        cbar = plot.colorbar(cf, label=variable_name + " [" + variable_unit + "]",
                             fraction=0.046, pad=0.04, shrink=0.65)
        cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
        cbar.ax.tick_params(labelsize=9)

        if not clean_plot:
            title_str = variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')'
            if level is not None:
                title_str += '\n' + _level_label(level, avg_ht, _original_height)
            ax.set_title(title_str, fontsize=16, pad=20)
            minmax_str = "Min, Max = " + str("{:.2e}".format(min_val)) + ", " + str("{:.2e}".format(max_val))
            ci_str = "Contour Interval = " + str("{:.2e}".format(interval_value))
            time_str = "Time=" + str(time.astype('M8[s]').astype(datetime))
            mtime_str = "Day,Hr,Min,Sec=" + str(selected_day) + "," + str(selected_hour) + "," + str(selected_min) + "," + str(selected_sec)
            plot.text(0.5, 0.06, minmax_str + "   " + ci_str + "\n" + time_str + "   " + mtime_str + "   " + str(filename),
                      ha='center', va='center', fontsize=10, transform=plot.transFigure)

        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, center_longitude, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum, cf, unique_lons, unique_lats, variable_values
        return plot

    # ---- Mollweide projection branch ----
    if projection == 'mollweide':
        plot = plt.figure(figsize=(14, 7))
        ax = plot.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=center_longitude))

        if coastlines:
            ax.add_feature(cfeature.COASTLINE, edgecolor=line_color, linewidth=1.5)
        if nightshade:
            ax.add_feature(Nightshade(datetime.fromtimestamp(time.astype('O')/1e9, tz=timezone.utc), alpha=0.4))
        if gm_equator_lats is not None:
            ax.plot(unique_lons, gm_equator_lats, color=line_color, linestyle='--',
                    transform=ccrs.Geodetic(), label='Geomagnetic Equator')

        ax.set_global()
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))

        cf = ax.contourf(unique_lons, unique_lats, variable_values, cmap=cmap_color,
                         levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max,
                         transform=ccrs.PlateCarree())
        cl = ax.contour(unique_lons, unique_lats, variable_values, colors=line_color,
                        linewidths=0.5, levels=contour_levels, transform=ccrs.PlateCarree())
        ax.clabel(cl, inline=True, fontsize=8, colors=line_color)

        _quiver_overlay(ax, unique_lons, unique_lats, wind_u_values, wind_v_values,
                        density=wind_density, scale=wind_scale, color=wind_color,
                        transform=ccrs.PlateCarree())

        cbar = plot.colorbar(cf, label=variable_name + " [" + variable_unit + "]",
                             fraction=0.046, pad=0.04, shrink=0.65)
        cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
        cbar.ax.tick_params(labelsize=9)

        if not clean_plot:
            title_str = variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')'
            if level is not None:
                title_str += '\n' + _level_label(level, avg_ht, _original_height)
            ax.set_title(title_str, fontsize=16, pad=20)
            minmax_str = "Min, Max = " + str("{:.2e}".format(min_val)) + ", " + str("{:.2e}".format(max_val))
            ci_str = "Contour Interval = " + str("{:.2e}".format(interval_value))
            time_str = "Time=" + str(time.astype('M8[s]').astype(datetime))
            mtime_str = "Day,Hr,Min,Sec=" + str(selected_day) + "," + str(selected_hour) + "," + str(selected_min) + "," + str(selected_sec)
            plot.text(0.5, 0.06, minmax_str + "   " + ci_str + "\n" + time_str + "   " + mtime_str + "   " + str(filename),
                      ha='center', va='center', fontsize=10, transform=plot.transFigure)

        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, center_longitude, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum, cf, unique_lons, unique_lats, variable_values
        return plot

    # ---- Mercator (default) projection branch ----

    # Clean plot
    if not clean_plot:
        figure_height = 6
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))

    subtitle_ht= 100
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center_longitude))



    # Check if add_coastlines parameter is True
    if coastlines:
        ax.add_feature(cfeature.COASTLINE, edgecolor=line_color, linewidth=1.5)
    if nightshade:
        ax.add_feature(Nightshade(datetime.fromtimestamp(time.astype('O')/1e9, tz=timezone.utc), alpha=0.4))
    if gm_equator_lats is not None:
        ax.plot(unique_lons, gm_equator_lats, color=line_color, linestyle='--',
                transform=ccrs.Geodetic(), label='Geomagnetic Equator')

    contour_filled = plt.contourf(unique_lons, unique_lats, variable_values, cmap=cmap_color, levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(unique_lons, unique_lats, variable_values, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)

    _quiver_overlay(ax, unique_lons, unique_lats, wind_u_values, wind_v_values,
                    density=wind_density, scale=wind_scale, color=wind_color,
                    transform=ccrs.PlateCarree())

    cbar = plt.colorbar(contour_filled, label=variable_name + " [" + variable_unit + "]",fraction=0.046, pad=0.04, shrink=0.65)
    cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)


    plt.xlabel('Longitude (Deg)', fontsize=14)
    #ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],fontsize=9)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    plt.xticks(fontsize=9)

    plt.ylabel('Latitude (Deg)', fontsize=14)
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    plt.yticks(fontsize=9)

    plt.xlim(longitude_minimum,longitude_maximum)
    plt.ylim(latitude_minimum,latitude_maximum)

    plt.tight_layout()



    if not clean_plot:
        # Add plot title
        plt.title(variable_long_name + ' ' + variable_name + ' (' + variable_unit + ') ' + '\n\n', fontsize=18)
        # Add plot subtitle
        if level is not None:
            plt.text(0, subtitle_ht, _level_label(level, avg_ht, _original_height), ha='center', va='center', fontsize=14)
        else:
            plt.text(0, subtitle_ht, '', ha='center', va='center', fontsize=14)


        # Add subtext to the plot
        plt.text(-90, -115, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center',fontsize=14)
        plt.text(90, -115, "Contour Interval = "+str("{:.2e}".format(interval_value)), ha='center', va='center',fontsize=14)
        plt.text(-90, -125, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center',fontsize=14)
        plt.text(90, -125, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center',fontsize=14)
        plt.text(0, -135, str(filename), ha='center', va='center',fontsize=14)


    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            # Integrate mplcursors by attaching to each PolyCollection
            cursor = mplcursors.cursor(contour_filled, hover=True)
            @cursor.connect("add")


            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target
                # Find the nearest longitude index
                if (x + center_longitude) > 180:
                    adjusted_lon =  - (360 -x -center_longitude)
                elif (x + center_longitude) < -180:
                    adjusted_lon = x + 360 + center_longitude #180 + (x + center_longitude)
                else:
                    adjusted_lon = x + center_longitude

                lon_idx = (np.abs(unique_lons - adjusted_lon)).argmin()

                # Find the nearest latitude index
                lat_idx = (np.abs(unique_lats - y)).argmin()

                # Retrieve the corresponding value
                value = variable_values[lat_idx, lon_idx]

                # Set annotation text
                sel.annotation.set(
                    text=f"Lon: {unique_lons[lon_idx]:.2f}°\nLat: {unique_lats[lat_idx]:.2f}°\n{variable_name}: {value:.2e} {variable_unit}"
                )

                # Customize annotation appearance
                sel.annotation.get_bbox_patch().set(alpha=0.9)
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, center_longitude, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum, contour_filled, unique_lons, unique_lats, variable_values
        elif plot is not None:
            plt.close(plot)
        return plot



def plt_lev_var(datasets, variable_name, latitude, time= None, mtime=None, longitude = None, log_level=True, variable_unit = None, level_minimum = None, level_maximum = None, y_axis = 'pressure', clean_plot = False, verbose = False):
    """
    Generates a Level vs Variable line plot for a given latitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        latitude (float): The specific latitude value for the plot.
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
        longitude (float, optional): The specific longitude value for the plot.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        variable_unit (str, optional): The desired unit of the variable.
        level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
        level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Line plot.
    """

    # Printing Execution data
    if time is None:
        time = get_time(datasets, mtime)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(latitude)+"]---["+str(longitude)+"]---------------")
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    result = arr_lev_var(datasets, variable_name, time, latitude, longitude, variable_unit, plot_mode=True)
    variable_values = result.values
    levs_ilevs = result.levs
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    model = result.model
    filename = result.filename

    # Convert levels to heights if requested
    if y_axis == 'height':
        height_levs = np.array([calc_avg_ht(datasets, time, lev) for lev in levs_ilevs])
        levs_ilevs = height_levs

    if level_minimum is None:
        level_minimum = np.nanmin(levs_ilevs)
    if level_maximum is None:
        level_maximum = np.nanmax(levs_ilevs)

    min_val, max_val = min_max(variable_values)
    selected_day=selected_mtime[0]
    selected_hour=selected_mtime[1]
    selected_min=selected_mtime[2]
    selected_sec=selected_mtime[3]
    
    # Clean plot
    if not clean_plot:
        figure_height = 6 
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(variable_values, levs_ilevs)
    plt.xlabel(variable_long_name, fontsize=14, labelpad=15)
    if y_axis == 'height':
        plt.ylabel('Height (km)', fontsize=14)
    else:
        plt.ylabel('LN(P0/P) (INTERFACES)', fontsize=14)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.ylim(level_minimum, level_maximum)

    if model == 'WACCM-X' and y_axis != 'height':
        plt.gca().invert_yaxis()
    
    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n',fontsize=18 )   
        if longitude == 'mean' and latitude == 'mean':
            plt.text(0.5, 1.08,"LAT= Mean LON= Mean", ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        elif longitude == 'mean':
            plt.text(0.5, 1.08,'LAT='+str(latitude)+" LON= Mean", ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        elif latitude == 'mean':
            plt.text(0.5, 1.08,'LAT= Mean'+" LON="+str(longitude), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        else:
            plt.text(0.5, 1.08,'LAT='+str(latitude)+" LON="+str(longitude), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        
        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        plt.text(0.5, -0.25, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        plt.text(0.5, -0.3, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.35, str(filename), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)

    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            # Integrate mplcursors by attaching to each PolyCollection
            cursor = mplcursors.cursor(plot, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                # Get the x (variable value) and y (level) from the cursor's target
                x, y = sel.target

                # Set annotation text to show level and variable value
                sel.annotation.set(
                    text=f"Level: {y:.2f} ln(P0/P)\n{variable_name}: {x:.2e} {variable_unit}")

                # Customize the appearance of the annotation box
                sel.annotation.get_bbox_patch().set(alpha=0.9)

            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, level_minimum, level_maximum
        elif plot is not None:
            plt.close(plot)
        return plot


def plt_var_lat(datasets, variable_name, level, time=None, mtime=None, longitude=None,
                level_type='pressure', variable_unit=None,
                latitude_minimum=None, latitude_maximum=None,
                clean_plot=False, verbose=False):
    """
    Generates a meridional 1D line plot (variable vs latitude) at a fixed longitude and level.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        level (float): The selected lev/ilev value (or 'mean').
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0].
        longitude (Union[float, str], optional): The specific longitude, or 'mean' for zonal mean.
        level_type (str, optional): 'pressure' (default) or 'height'.
        variable_unit (str, optional): The desired unit of the variable.
        latitude_minimum (float, optional): Minimum latitude on the x-axis.
        latitude_maximum (float, optional): Maximum latitude on the x-axis.
        clean_plot (bool, optional): If True, hide subtext. Defaults to False.
        verbose (bool, optional): If True, log execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Line plot.
    """
    if time is None:
        time = get_time(datasets, mtime)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(level)+"]---["+str(longitude)+"]---------------")
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        level = height_to_pres_level(datasets, time, _original_height)

    result = arr_var_lat(datasets, variable_name, time, level, longitude,
                         selected_unit=variable_unit, plot_mode=True)
    variable_values = result.values
    lats = result.lats
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    filename = result.filename
    selected_lon = result.selected_lon
    selected_lev = result.selected_lev

    if latitude_minimum is None:
        latitude_minimum = np.nanmin(lats)
    if latitude_maximum is None:
        latitude_maximum = np.nanmax(lats)

    min_val, max_val = min_max(variable_values)
    selected_day = selected_mtime[0]
    selected_hour = selected_mtime[1]
    selected_min = selected_mtime[2]
    selected_sec = selected_mtime[3]

    if not clean_plot:
        figure_height = 6
        figure_width = 10
    else:
        figure_height = 5
        figure_width = 10

    plot = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(lats, variable_values)
    plt.xlabel('Latitude (Deg)', fontsize=14)
    plt.ylabel(variable_long_name + ' (' + variable_unit + ')', fontsize=14, labelpad=15)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(latitude_minimum, latitude_maximum)
    plt.grid(True, alpha=0.3)

    if not clean_plot:
        plt.title(variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')\n\n', fontsize=18)
        if selected_lon == 'mean':
            loc_str = 'LON= Mean'
        else:
            loc_str = 'LON=' + str(selected_lon)
        if selected_lev is not None:
            loc_str += '  ' + _level_label(selected_lev, original_height=_original_height)
        plt.text(0.5, 1.08, loc_str, ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.25, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.3, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.35, str(filename), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            cursor = mplcursors.cursor(plot, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                sel.annotation.set(text=f"Lat: {x:.2f}\u00b0\n{variable_name}: {y:.2e} {variable_unit}")
                sel.annotation.get_bbox_patch().set(alpha=0.9)
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, latitude_minimum, latitude_maximum
        elif plot is not None:
            plt.close(plot)
        return plot


def plt_var_lon(datasets, variable_name, level, time=None, mtime=None, latitude=None,
                level_type='pressure', variable_unit=None,
                longitude_minimum=None, longitude_maximum=None,
                clean_plot=False, verbose=False):
    """
    Generates a zonal 1D line plot (variable vs longitude) at a fixed latitude and level.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        level (float): The selected lev/ilev value (or 'mean').
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0].
        latitude (Union[float, str], optional): The specific latitude, or 'mean' for meridional mean.
        level_type (str, optional): 'pressure' (default) or 'height'.
        variable_unit (str, optional): The desired unit of the variable.
        longitude_minimum (float, optional): Minimum longitude on the x-axis.
        longitude_maximum (float, optional): Maximum longitude on the x-axis.
        clean_plot (bool, optional): If True, hide subtext. Defaults to False.
        verbose (bool, optional): If True, log execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Line plot.
    """
    if time is None:
        time = get_time(datasets, mtime)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(level)+"]---["+str(latitude)+"]---------------")
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')

    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        level = height_to_pres_level(datasets, time, _original_height)

    result = arr_var_lon(datasets, variable_name, time, level, latitude,
                         selected_unit=variable_unit, plot_mode=True)
    variable_values = result.values
    lons = result.lons
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    filename = result.filename
    selected_lat = result.selected_lat
    selected_lev = result.selected_lev

    if longitude_minimum is None:
        longitude_minimum = np.nanmin(lons)
    if longitude_maximum is None:
        longitude_maximum = np.nanmax(lons)

    min_val, max_val = min_max(variable_values)
    selected_day = selected_mtime[0]
    selected_hour = selected_mtime[1]
    selected_min = selected_mtime[2]
    selected_sec = selected_mtime[3]

    if not clean_plot:
        figure_height = 6
        figure_width = 10
    else:
        figure_height = 5
        figure_width = 10

    plot = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(lons, variable_values)
    plt.xlabel('Longitude (Deg)', fontsize=14)
    plt.ylabel(variable_long_name + ' (' + variable_unit + ')', fontsize=14, labelpad=15)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(longitude_minimum, longitude_maximum)
    plt.grid(True, alpha=0.3)

    if not clean_plot:
        plt.title(variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')\n\n', fontsize=18)
        if selected_lat == 'mean':
            loc_str = 'LAT= Mean'
        else:
            loc_str = 'LAT=' + str(selected_lat)
        if selected_lev is not None:
            loc_str += '  ' + _level_label(selected_lev, original_height=_original_height)
        plt.text(0.5, 1.08, loc_str, ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.25, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.3, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.35, str(filename), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            cursor = mplcursors.cursor(plot, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                sel.annotation.set(text=f"Lon: {x:.2f}\u00b0\n{variable_name}: {y:.2e} {variable_unit}")
                sel.annotation.get_bbox_patch().set(alpha=0.9)
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, longitude_minimum, longitude_maximum
        elif plot is not None:
            plt.close(plot)
        return plot


def plt_lev_lon(datasets, variable_name, latitude, time= None, mtime=None, log_level=True, variable_unit = None, contour_intervals = 20, contour_value = None,symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white',  level_minimum = None, level_maximum = None, longitude_minimum = None, longitude_maximum = None, y_axis = 'pressure', wind = False, wind_density = 5, wind_scale = None, wind_color = 'black', clean_plot = False, verbose = False):
    """
    Generates a Level vs Longitude contour plot for a given latitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        latitude (float): The specific latitude value for the plot.
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 20. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
        level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
        longitude_minimum (float, optional): Minimum longitude value for the plot. Defaults to -180.
        longitude_maximum (float, optional): Maximum longitude value for the plot. Defaults to 175.
        wind (bool, optional): Overlay (U, W) wind vectors on the cross-section. Uses model-specific defaults (TIE-GCM: UN/WN, WACCM-X: U/OMEGA). Defaults to False.
        wind_density (int, optional): Stride for thinning wind vectors (every Nth point). Defaults to 5.
        wind_scale (float, optional): Scale factor for quiver arrows. Larger values make arrows shorter. Defaults to None (auto-scaled).
        wind_color (str, optional): Color of the wind vectors. Defaults to 'black'.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    # Printing Execution data
    if time is None:
        time = get_time(datasets, mtime)
    if contour_intervals is None:
        contour_intervals = 20
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(latitude)+"]---------------")
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    # Generate 2D arrays, extract variable_unit
    result = arr_lev_lon(datasets, variable_name, time, latitude, variable_unit, log_level, plot_mode=True)
    variable_values = result.values
    unique_lons = result.lons
    unique_levs = result.levs
    latitude = result.selected_lat
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    model = result.model
    filename = result.filename

    # Interpolate to height surfaces if requested
    if y_axis == 'height':
        variable_values, unique_levs = interpolate_to_height(
            datasets, variable_values, unique_levs, time)

    if level_minimum is None:
        level_minimum = np.nanmin(unique_levs)
    if level_maximum is None:
        level_maximum = np.nanmax(unique_levs)
    if longitude_minimum is None:
        longitude_minimum = np.nanmin(unique_lons)
    if longitude_maximum is None:   
        longitude_maximum = np.nanmax(unique_lons)

    min_val, max_val = min_max(variable_values)
    selected_day=selected_mtime[0]
    selected_hour=selected_mtime[1]
    selected_min=selected_mtime[2]
    selected_sec=selected_mtime[3]

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)  # Divide by 2 to get intervals for one side
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)  # Generate negative levels symmetrically
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))
    if -180 in unique_lons:
        lon_idx = np.where(unique_lons == -180)[0][-1]  # Get the index of the last occurrence of -180
        unique_lons = np.append(unique_lons, 180)
        variable_values = np.insert(variable_values, -1, variable_values[:, lon_idx], axis=1)

    # Clean plot
    if not clean_plot:
        figure_height = 6 
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))
    contour_filled = plt.contourf(unique_lons, unique_levs, variable_values, cmap= cmap_color, levels=contour_levels,vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(unique_lons, unique_levs, variable_values, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)
    cbar = plt.colorbar(contour_filled, label=variable_name+" ["+variable_unit+"]")
    cbar.set_label(variable_name+" ["+variable_unit+"]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n',fontsize=18 )   
        if latitude == 'mean':
            plt.text(0.5, 1.10,'ZONAL MEANS', ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        else:
            plt.text(0.5, 1.10,'LAT='+str(latitude), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        
    if y_axis == 'height':
        plt.ylabel('Height (km)',fontsize=14)
    elif log_level:
        plt.ylabel('LN(P0/P) (INTERFACES)',fontsize=14)
    else:
        plt.ylabel('PRESSURE (HPA)',fontsize=14)
    plt.xlabel('Longitude (Deg)',fontsize=14)
    plt.xticks([value for value in unique_lons if value % 30 == 0],fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(longitude_minimum,longitude_maximum)
    plt.ylim(level_minimum, level_maximum)

    if model == 'WACCM-X' and y_axis != 'height':
        plt.gca().invert_yaxis()

    if wind:
        from .containers import MODEL_DEFAULTS
        u_name = MODEL_DEFAULTS[model]['wind_u']
        u_res = arr_lev_lon(datasets, u_name, time, latitude,
                            log_level=log_level, plot_mode=True)
        w_res = None
        w_candidates = [MODEL_DEFAULTS[model]['wind_w'], 'WN', 'W', 'OMEGA']
        for w_name in dict.fromkeys(w_candidates):
            try:
                w_res = arr_lev_lon(datasets, w_name, time, latitude,
                                    log_level=log_level, plot_mode=True)
            except (KeyError, AttributeError):
                continue
            if w_res is not None:
                break
        if u_res is not None and w_res is not None:
            u_vals = u_res.values
            w_vals = w_res.values
            w_levs = w_res.levs
            if w_vals.shape[0] == u_vals.shape[0] + 1:
                w_vals = 0.5 * (w_vals[:-1, :] + w_vals[1:, :])
                w_levs = 0.5 * (w_levs[:-1] + w_levs[1:])
            if w_vals.shape == u_vals.shape:
                _quiver_overlay(plt.gca(), u_res.lons, u_res.levs,
                                u_vals, w_vals,
                                density=wind_density, scale=wind_scale,
                                color=wind_color)

    if not clean_plot:
        # Add subtext to the plot
        plt.text(0.25, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.75, -0.2, "Contour Interval = "+str("{:.2e}".format(interval_value)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.25, -0.25, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.75, -0.25, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.3, str(filename), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)

    center_longitude = 0
    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            # Integrate mplcursors by attaching to each PolyCollection
            cursor = mplcursors.cursor(contour_filled, hover=True)
            @cursor.connect("add")


            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target
                # Find the nearest longitude index
                if (x + center_longitude) > 180:
                    adjusted_lon =  - (360 -x -center_longitude)
                elif (x + center_longitude) < -180:
                    adjusted_lon = x + 360 + center_longitude #180 + (x + center_longitude) 
                else:
                    adjusted_lon = x + center_longitude
                
                lon_idx = (np.abs(unique_lons - adjusted_lon)).argmin() 
                
                # Find the nearest latitude index
                level_idx = (np.abs(unique_levs - y)).argmin()
                
                # Retrieve the corresponding value
                value = variable_values[level_idx, lon_idx]
                
                # Set annotation text
                sel.annotation.set(
                    text=f"Lon: {unique_lons[lon_idx]:.2f}°\nLev: {unique_levs[level_idx]:.2f}°\n{variable_name}: {value:.2e} {variable_unit}"
                )
                
                # Customize annotation appearance
                sel.annotation.get_bbox_patch().set(alpha=0.9)
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, latitude, time, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, level_minimum, level_maximum, longitude_minimum, longitude_maximum, contour_filled, unique_lons, unique_levs, variable_values
        elif plot is not None:
            plt.close(plot)
        return plot


def plt_lev_lat(datasets, variable_name, time= None, mtime=None, longitude = None, log_level = True, variable_unit = None, contour_intervals = 20, contour_value = None,symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white', level_minimum = None, level_maximum = None, latitude_minimum = None,latitude_maximum = None, y_axis = 'pressure', wind = False, epflux = False, wind_density = 5, wind_scale = None, wind_color = 'black', clean_plot = False, verbose = False):
    """
    Generates a Level vs Latitude contour plot for a specified time and/or longitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (list[int], optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
        longitude (float, optional): The specific longitude value for the plot.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 20. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
        level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
        latitude_minimum (float, optional): Minimum latitude value for the plot. Defaults to -87.5.
        latitude_maximum (float, optional): Maximum latitude value for the plot. Defaults to 87.5.
        wind (bool, optional): Overlay (V, W) wind vectors on the cross-section. Uses model-specific defaults (TIE-GCM: VN/WN, WACCM-X: V/OMEGA). Defaults to False.
        epflux (bool, optional): Overlay (EPVY, EPVZ) Eliassen-Palm flux vectors instead of winds. Mutually exclusive with ``wind``. Defaults to False.
        wind_density (int, optional): Stride for thinning overlay vectors (every Nth point). Defaults to 5.
        wind_scale (float, optional): Scale factor for quiver arrows. Larger values make arrows shorter. Defaults to None (auto-scaled).
        wind_color (str, optional): Color of the overlay vectors. Defaults to 'black'.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    # Printing Execution data
    if time is None:
        time = get_time(datasets, mtime)
    if contour_intervals is None:
        contour_intervals = 20
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(time)+"]---["+str(longitude)+"]---------------")
    # Generate 2D arrays, extract variable_unit
    if isinstance(time, str):
        time = np.datetime64(time, 'ns')
    handler, is_derived = resolve_derived(variable_name)
    if is_derived:
        result = handler(datasets, variable_name, time, log_level=log_level)
    else:
        result = arr_lev_lat(datasets, variable_name, time, longitude, variable_unit, plot_mode=True)
    variable_values = result.values
    unique_lats = result.lats
    unique_levs = result.levs
    longitude = result.selected_lon
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    selected_mtime = result.mtime
    model = result.model
    filename = result.filename

    # Interpolate to height surfaces if requested
    if y_axis == 'height':
        variable_values, unique_levs = interpolate_to_height(
            datasets, variable_values, unique_levs, time)

    if level_minimum is None:
        level_minimum = np.nanmin(unique_levs)
    if level_maximum is None:
        level_maximum = np.nanmax(unique_levs)
    if latitude_minimum is None:
        latitude_minimum = np.nanmin(unique_lats)
    if latitude_maximum is None:
        latitude_maximum = np.nanmax(unique_lats)

    min_val, max_val = min_max(variable_values)
    selected_day=selected_mtime[0]
    selected_hour=selected_mtime[1]
    selected_min=selected_mtime[2]
    selected_sec=selected_mtime[3]

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)  # Divide by 2 to get intervals for one side
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)  # Generate negative levels symmetrically
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))

    
    
    interval_value = contour_value if contour_value else (max_val - min_val) / (contour_intervals - 1)
    
        # Clean plot
    if not clean_plot:
        figure_height = 6 
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))
    contour_filled = plt.contourf(unique_lats, unique_levs, variable_values, cmap= cmap_color, levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(unique_lats, unique_levs, variable_values, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)
    cbar = plt.colorbar(contour_filled, label=variable_name+" ["+variable_unit+"]")
    cbar.set_label(variable_name+" ["+variable_unit+"]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n',fontsize=18 )   
    
        if longitude == 'mean' or longitude is None:
            plt.text(0.5, 1.08,'ZONAL MEANS', ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 1.08,'LON='+str(longitude)+" SLT="+str(longitude_to_local_time(longitude))+"Hrs", ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
    if y_axis == 'height':
        plt.ylabel('Height (km)',fontsize=14)
    elif log_level:
        plt.ylabel('LN(P0/P) (INTERFACES)',fontsize=14)
    else:
        plt.ylabel('PRESSURE (HPA)',fontsize=14)
    plt.xlabel('Latitude (Deg)',fontsize=14)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(latitude_minimum,latitude_maximum)
    plt.ylim(level_minimum,level_maximum)

    if model == 'WACCM-X' and y_axis != 'height':
        plt.gca().invert_yaxis()

    if wind or epflux:
        from .data_epflux import arr_epflux
        overlay_u = overlay_v = None
        overlay_x = unique_lats
        overlay_y = unique_levs
        if epflux:
            epvy = arr_epflux(datasets, 'EPVY', time, log_level=log_level)
            epvz = arr_epflux(datasets, 'EPVZ', time, log_level=log_level)
            if epvy is not None and epvz is not None:
                overlay_u = epvy.values
                overlay_v = epvz.values
                overlay_x = epvy.lats
                overlay_y = epvy.levs
        else:
            from .containers import MODEL_DEFAULTS
            v_name = MODEL_DEFAULTS[model]['wind_v']
            v_res = arr_lev_lat(datasets, v_name, time, longitude,
                                log_level=log_level, plot_mode=True)
            w_res = None
            w_candidates = [MODEL_DEFAULTS[model]['wind_w'], 'WN', 'W', 'OMEGA']
            for w_name in dict.fromkeys(w_candidates):
                try:
                    w_res = arr_lev_lat(datasets, w_name, time, longitude,
                                        log_level=log_level, plot_mode=True)
                except (KeyError, AttributeError):
                    continue
                if w_res is not None:
                    break
            if v_res is not None and w_res is not None:
                u_vals = v_res.values
                w_vals = w_res.values
                w_levs = w_res.levs
                # Interpolate W from ilev to lev when sizes differ by 1
                if w_vals.shape[0] == u_vals.shape[0] + 1:
                    w_vals = 0.5 * (w_vals[:-1, :] + w_vals[1:, :])
                    w_levs = 0.5 * (w_levs[:-1] + w_levs[1:])
                if w_vals.shape == u_vals.shape:
                    overlay_u = u_vals
                    overlay_v = w_vals
                    overlay_x = v_res.lats
                    overlay_y = v_res.levs
        _quiver_overlay(plt.gca(), overlay_x, overlay_y, overlay_u, overlay_v,
                        density=wind_density, scale=wind_scale, color=wind_color)

    if not clean_plot:
        # Add subtext to the plot
        plt.text(0.25, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.75, -0.2, "Contour Interval = "+str("{:.2e}".format(interval_value)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.25, -0.25, "Time = "+str(time.astype('M8[s]').astype(datetime)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.75, -0.25, "Day, Hour, Min, Sec = "+str(selected_day)+","+str(selected_hour)+","+str(selected_min)+","+str(selected_sec), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.50, -0.3, str(filename), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)


    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            # Integrate mplcursors by attaching to each PolyCollection
            cursor = mplcursors.cursor(contour_filled, hover=True)
            @cursor.connect("add")


            def on_add(sel):
                # sel.target gives the coordinates where the cursor is
                x, y = sel.target

                lat_idx = (np.abs(unique_lats - x)).argmin()

                # Find the nearest latitude index
                level_idx = (np.abs(unique_levs - y)).argmin()
                
                # Retrieve the corresponding value
                value = variable_values[level_idx, lat_idx]
                
                # Set annotation text
                sel.annotation.set(
                    text=f"Lat: {unique_lats[lat_idx]:.2f}°\nLev: {unique_levs[level_idx]:.2f}°\n{variable_name}: {value:.2e} {variable_unit}"
                )
                
                # Customize annotation appearance
                sel.annotation.get_bbox_patch().set(alpha=0.9)
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, time, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, level_minimum, level_maximum, latitude_minimum, latitude_maximum, contour_filled, unique_lats, unique_levs, variable_values
        elif plot is not None:
            plt.close(plot)
        return plot




def plt_lev_time(datasets, variable_name, latitude, longitude = None, log_level = True, variable_unit = None, contour_intervals = 10, contour_value = None,symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white',  level_minimum = None, level_maximum = None, mtime_minimum=None, mtime_maximum=None, y_axis = 'pressure', clean_plot = False, verbose = False):
    """
    Generates a Level vs Time contour plot for a specified latitude and/or longitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, time, and ilev dimensions.
        latitude (float): The specific latitude value for the plot.
        longitude (float, optional): The specific longitude value for the plot.
        log_level (bool): A flag indicating whether to display level in log values. Default is True.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 10. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        level_minimum (float, optional): Minimum level value for the plot. Defaults to None.
        level_maximum (float, optional): Maximum level value for the plot. Defaults to None.
        mtime_minimum (float, optional): Minimum time value for the plot. Defaults to None.
        mtime_maximum (float, optional): Maximum time value for the plot. Defaults to None.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    if contour_intervals is None:
        contour_intervals = 20
    #print(datasets)
    result = arr_lev_time(datasets, variable_name, latitude, longitude, variable_unit, plot_mode=True)
    variable_values_all = result.values
    levs_ilevs = result.levs
    mtime_values = result.mtime_values
    longitude = result.selected_lon
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    model = result.model

    # Interpolate to height surfaces if requested
    if y_axis == 'height':
        first_time = datasets[0]._time_values[0]
        variable_values_all, levs_ilevs = interpolate_to_height(
            datasets, variable_values_all, levs_ilevs, first_time)

    if level_minimum is None:
        level_minimum = np.nanmin(levs_ilevs)
    if level_maximum is None:
        level_maximum = np.nanmax(levs_ilevs)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(latitude)+"]---["+str(longitude)+"]---------------")
    


    num_deleted_before = 0
    num_deleted_after = 0

    if mtime_minimum is not None and mtime_maximum is not None:
        new_mtime_values = []
        for t_mtime in mtime_values:
            mtime_total_minutes = t_mtime[0] * 24 * 60 *60 + t_mtime[1] * 60 *60+ t_mtime[2] *60+ t_mtime[3]
            mtime_min_total = mtime_minimum[0] * 24 * 60*60 + mtime_minimum[1] * 60 *60+ mtime_minimum[2]*60 + mtime_minimum[3]
            mtime_max_total = mtime_maximum[0] * 24 * 60*60 + mtime_maximum[1] * 60 *60+ mtime_maximum[2]*60 + mtime_minimum[3]
            if mtime_total_minutes >= mtime_min_total and mtime_total_minutes <= mtime_max_total:                
                new_mtime_values.append(t_mtime)  
            else:
                if mtime_total_minutes < mtime_min_total:
                    num_deleted_before += 1
                elif mtime_total_minutes > mtime_max_total:
                    num_deleted_after += 1
        mtime_values = new_mtime_values
        mtime_values_sorted = sorted(mtime_values, key=lambda x: (x[0], x[1], x[2], x[3]))
        variable_values_all = variable_values_all[:, num_deleted_before:-num_deleted_after]

    min_val, max_val = np.nanmin(variable_values_all), np.nanmax(variable_values_all)

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)  # Divide by 2 to get intervals for one side
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)  # Generate negative levels symmetrically
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))

    
    
    interval_value = contour_value if contour_value else (max_val - min_val) / (contour_intervals - 1)
    mtime_tuples = [tuple(entry) for entry in mtime_values]
    try:    # Modify this part to show both day and hour
        unique_times = sorted(list(set([(day, hour) for day, hour, _, _ in mtime_values])))
        time_indices = [i for i, (day, hour, _, _) in enumerate(mtime_tuples) if i == 0 or mtime_tuples[i-1][:2] != (day, hour)]
        if len(time_indices) >24:
            unique_times = sorted(list(set([day for day, _, _, _ in mtime_values])))
            time_indices = [i for i, (day, _, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]
    except (ValueError, TypeError):
        unique_times = sorted(list(set([day for day, _, _ in mtime_values])))
        time_indices = [i for i, (day, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]

        # Clean plot
    if not clean_plot:
        figure_height = 6 
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))
    X, Y = np.meshgrid(range(len(mtime_values)), levs_ilevs)
    contour_filled = plt.contourf(X, Y, variable_values_all, cmap=cmap_color, levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(X, Y, variable_values_all, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)
    cbar = plt.colorbar(contour_filled, label=variable_name+" ["+variable_unit+"]")
    cbar.set_label(variable_name+" ["+variable_unit+"]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    try:
        plt.xticks(time_indices, ["{}-{:02d}h".format(day, hour) for day, hour in unique_times], rotation=45)
        plt.xlabel("Model Time (Day,Hour) from "+str(unique_times[0])+" to "+str(unique_times[-1]), fontsize=14) 
    except (ValueError, TypeError):
        plt.xticks(time_indices, unique_times, rotation=45)
        plt.xlabel("Model Time (Day) from "+str(np.nanmin(unique_times))+" to "+str(np.nanmax(unique_times)) ,fontsize=14)
    if y_axis == 'height':
        plt.ylabel('Height (km)',fontsize=14)
    else:
        plt.ylabel('LN(P0/P) (INTERFACES)',fontsize=14)

    plt.tight_layout()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(level_minimum,level_maximum)

    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n',fontsize=18 )   
        # Add subtext to the plot
        if longitude == 'mean' and latitude == 'mean':
            plt.text(0.5, 1.08,'  LAT= Mean LON= Mean', ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        elif longitude == 'mean':
            plt.text(0.5, 1.08,'  LAT='+str(latitude)+" LON= Mean", ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        elif latitude == 'mean':
            plt.text(0.5, 1.08,'  LAT= Mean'+" LON="+str(longitude), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        else:
            plt.text(0.5, 1.08,'  LAT='+str(latitude)+" LON="+str(longitude), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes) 
        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.25, "Contour Interval = "+str("{:.2e}".format(interval_value)), ha='center', va='center',fontsize=14, transform=plt.gca().transAxes)
    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, level_minimum, level_maximum, contour_filled, levs_ilevs, variable_values_all
        if plot is not None:
            plt.close(plot)
        return plot


def plt_lon_time(datasets, variable_name, latitude, level = None, level_type = 'pressure', variable_unit = None, contour_intervals = 10, contour_value = None, symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white', longitude_minimum = None, longitude_maximum = None, mtime_minimum=None, mtime_maximum=None, clean_plot = False, verbose = False):
    """
    Generates a Longitude vs Time contour plot for a specified latitude and/or level.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, time, and lev/ilev dimensions.
        latitude (float): The specific latitude value for the plot.
        level (float, optional): The specific level value for the plot.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 10. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour.
        cmap_lim_min (float, optional): Minimum limit for the color map.
        cmap_lim_max (float, optional): Maximum limit for the color map.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        longitude_minimum (float, optional): Minimum longitude value for the plot.
        longitude_maximum (float, optional): Maximum longitude value for the plot.
        mtime_minimum (list, optional): Minimum time value as [day, hour, min, sec].
        mtime_maximum (list, optional): Maximum time value as [day, hour, min, sec].
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    if contour_intervals is None:
        contour_intervals = 20
    # Convert height to pressure level if needed
    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        first_time = datasets[0]._time_values[0]
        level = height_to_pres_level(datasets, first_time, _original_height)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(latitude)+"]---["+str(level)+"]---------------")

    result = arr_lon_time(datasets, variable_name, latitude, level, variable_unit, plot_mode=True)
    variable_values_all = result.values
    unique_lons = result.lons
    mtime_values = result.mtime_values
    latitude = result.selected_lat
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    model = result.model
    filename = result.filename

    if longitude_minimum is None:
        longitude_minimum = np.nanmin(unique_lons)
    if longitude_maximum is None:
        longitude_maximum = np.nanmax(unique_lons)

    num_deleted_before = 0
    num_deleted_after = 0

    if mtime_minimum is not None and mtime_maximum is not None:
        new_mtime_values = []
        for t_mtime in mtime_values:
            mtime_total_minutes = t_mtime[0] * 24 * 60 *60 + t_mtime[1] * 60 *60+ t_mtime[2] *60+ t_mtime[3]
            mtime_min_total = mtime_minimum[0] * 24 * 60*60 + mtime_minimum[1] * 60 *60+ mtime_minimum[2]*60 + mtime_minimum[3]
            mtime_max_total = mtime_maximum[0] * 24 * 60*60 + mtime_maximum[1] * 60 *60+ mtime_maximum[2]*60 + mtime_maximum[3]
            if mtime_total_minutes >= mtime_min_total and mtime_total_minutes <= mtime_max_total:
                new_mtime_values.append(t_mtime)
            else:
                if mtime_total_minutes < mtime_min_total:
                    num_deleted_before += 1
                elif mtime_total_minutes > mtime_max_total:
                    num_deleted_after += 1
        mtime_values = new_mtime_values
        if num_deleted_after > 0:
            variable_values_all = variable_values_all[:, num_deleted_before:-num_deleted_after]
        else:
            variable_values_all = variable_values_all[:, num_deleted_before:]

    min_val, max_val = np.nanmin(variable_values_all), np.nanmax(variable_values_all)

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))

    interval_value = contour_value if contour_value else (max_val - min_val) / (contour_intervals - 1)

    mtime_tuples = [tuple(entry) for entry in mtime_values]
    try:
        unique_times = sorted(list(set([(day, hour) for day, hour, _, _ in mtime_values])))
        time_indices = [i for i, (day, hour, _, _) in enumerate(mtime_tuples) if i == 0 or mtime_tuples[i-1][:2] != (day, hour)]
        if len(time_indices) >24:
            unique_times = sorted(list(set([day for day, _, _, _ in mtime_values])))
            time_indices = [i for i, (day, _, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]
    except (ValueError, TypeError):
        unique_times = sorted(list(set([day for day, _, _ in mtime_values])))
        time_indices = [i for i, (day, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]

    if not clean_plot:
        figure_height = 6
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10

    plot = plt.figure(figsize=(figure_width, figure_height))
    X, Y = np.meshgrid(range(len(mtime_values)), unique_lons)
    contour_filled = plt.contourf(X, Y, variable_values_all, cmap=cmap_color, levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(X, Y, variable_values_all, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)
    cbar = plt.colorbar(contour_filled, label=variable_name + " [" + variable_unit + "]")
    cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    try:
        plt.xticks(time_indices, ["{}-{:02d}h".format(day, hour) for day, hour in unique_times], rotation=45)
        plt.xlabel("Model Time (Day,Hour) from "+str(unique_times[0])+" to "+str(unique_times[-1]), fontsize=14)
    except (ValueError, TypeError):
        plt.xticks(time_indices, unique_times, rotation=45)
        plt.xlabel("Model Time (Day) from "+str(np.nanmin(unique_times))+" to "+str(np.nanmax(unique_times)), fontsize=14)

    plt.ylabel('Longitude (Deg)', fontsize=14)

    plt.tight_layout()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(longitude_minimum, longitude_maximum)

    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n', fontsize=18)
        subtitle_parts = []
        if level is not None:
            subtitle_parts.append(_level_label(level, original_height=_original_height))
        if latitude is not None:
            subtitle_parts.append('LAT= Mean' if latitude == 'mean' else 'LAT=' + str(latitude))
        if subtitle_parts:
            plt.text(0.5, 1.08, '  ' + ' '.join(subtitle_parts), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.25, "Contour Interval = "+str("{:.2e}".format(interval_value)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, longitude_minimum, longitude_maximum, contour_filled, unique_lons, variable_values_all
        if plot is not None:
            plt.close(plot)
        return plot


def plt_var_time(datasets, variable_name, latitude, longitude, level = None, level_type = 'pressure', variable_unit = None, mtime_minimum=None, mtime_maximum=None, clean_plot = False, verbose = False):
    """
    Generates a Variable vs Time line plot at a specific lat/lon/level location.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, time, and lev/ilev dimensions.
        latitude (float): The specific latitude value for the plot.
        longitude (float): The specific longitude value for the plot.
        level (float, optional): The specific level value for the plot.
        variable_unit (str, optional): The desired unit of the variable.
        mtime_minimum (list, optional): Minimum time value as [day, hour, min, sec].
        mtime_maximum (list, optional): Maximum time value as [day, hour, min, sec].
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Line plot.
    """

    # Convert height to pressure level if needed
    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        first_time = datasets[0]._time_values[0]
        level = height_to_pres_level(datasets, first_time, _original_height)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(latitude)+"]---["+str(longitude)+"]---["+str(level)+"]---------------")

    result = arr_var_time(datasets, variable_name, latitude, longitude, level, variable_unit, plot_mode=True)
    variable_values = result.values
    mtime_values = result.mtime_values
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    model = result.model
    filename = result.filename

    num_deleted_before = 0
    num_deleted_after = 0

    if mtime_minimum is not None and mtime_maximum is not None:
        new_mtime_values = []
        for t_mtime in mtime_values:
            mtime_total_minutes = t_mtime[0] * 24 * 60 *60 + t_mtime[1] * 60 *60+ t_mtime[2] *60+ t_mtime[3]
            mtime_min_total = mtime_minimum[0] * 24 * 60*60 + mtime_minimum[1] * 60 *60+ mtime_minimum[2]*60 + mtime_minimum[3]
            mtime_max_total = mtime_maximum[0] * 24 * 60*60 + mtime_maximum[1] * 60 *60+ mtime_maximum[2]*60 + mtime_maximum[3]
            if mtime_total_minutes >= mtime_min_total and mtime_total_minutes <= mtime_max_total:
                new_mtime_values.append(t_mtime)
            else:
                if mtime_total_minutes < mtime_min_total:
                    num_deleted_before += 1
                elif mtime_total_minutes > mtime_max_total:
                    num_deleted_after += 1
        mtime_values = new_mtime_values
        if num_deleted_after > 0:
            variable_values = variable_values[num_deleted_before:-num_deleted_after]
        else:
            variable_values = variable_values[num_deleted_before:]

    min_val, max_val = np.nanmin(variable_values), np.nanmax(variable_values)

    mtime_tuples = [tuple(entry) for entry in mtime_values]
    try:
        unique_times = sorted(list(set([(day, hour) for day, hour, _, _ in mtime_values])))
        time_indices = [i for i, (day, hour, _, _) in enumerate(mtime_tuples) if i == 0 or mtime_tuples[i-1][:2] != (day, hour)]
        if len(time_indices) > 24:
            unique_times = sorted(list(set([day for day, _, _, _ in mtime_values])))
            time_indices = [i for i, (day, _, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]
    except (ValueError, TypeError):
        unique_times = sorted(list(set([day for day, _, _ in mtime_values])))
        time_indices = [i for i, (day, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]

    if not clean_plot:
        figure_height = 6
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10

    plot = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(range(len(mtime_values)), variable_values, linewidth=1.5)

    try:
        plt.xticks(time_indices, ["{}-{:02d}h".format(day, hour) for day, hour in unique_times], rotation=45)
        plt.xlabel("Model Time (Day,Hour) from "+str(unique_times[0])+" to "+str(unique_times[-1]), fontsize=14)
    except (ValueError, TypeError):
        plt.xticks(time_indices, unique_times, rotation=45)
        plt.xlabel("Model Time (Day) from "+str(np.nanmin(unique_times))+" to "+str(np.nanmax(unique_times)), fontsize=14)

    plt.ylabel(variable_name + ' (' + variable_unit + ')', fontsize=14)

    plt.tight_layout()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True, alpha=0.3)

    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n', fontsize=18)
        subtitle_parts = []
        if latitude is not None:
            subtitle_parts.append('LAT=' + str(latitude))
        if longitude is not None:
            subtitle_parts.append('LON=' + str(longitude))
        if level is not None:
            subtitle_parts.append(_level_label(level, original_height=_original_height))
        plt.text(0.5, 1.08, '  ' + ' '.join(subtitle_parts), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.2, "Min, Max = "+str("{:.2e}".format(min_val))+", "+str("{:.2e}".format(max_val)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit
        if plot is not None:
            plt.close(plot)
        return plot



def plt_lat_time(datasets, variable_name, level = None, level_type = 'pressure', longitude = None,  variable_unit = None, contour_intervals = 10, contour_value = None, symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white', latitude_minimum = None,latitude_maximum = None, mtime_minimum=None, mtime_maximum=None, clean_plot = False, verbose = False):
    """
    Generates a Latitude vs Time contour plot for a specified level and/or longitude.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, time, and ilev dimensions.
        level (float): The specific level value for the plot.
        longitude (float, optional): The specific longitude value for the plot.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 10. Ignored if contour_value is provided.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        latitude_minimum (float, optional): Minimum latitude value for the plot. Defaults to -87.5.
        latitude_maximum (float, optional): Maximum latitude value for the plot. Defaults to 87.5.
        mtime_minimum (float, optional): Minimum time value for the plot. Defaults to None.
        mtime_maximum (float, optional): Maximum time value for the plot. Defaults to None.
        clean_plot (bool, optional): A flag indicating whether to display the subtext. Defaults to False.
        verbose (bool, optional): A flag indicating whether to print execution data. Defaults to False.
        
    Returns:
        matplotlib.figure.Figure: Contour plot.
    """

    if contour_intervals is None:
        contour_intervals = 20
    # Convert height to pressure level if needed
    _original_height = None
    if level_type == 'height' and level is not None and level != 'mean':
        _original_height = float(level)
        first_time = datasets[0]._time_values[0]
        level = height_to_pres_level(datasets, first_time, _original_height)
    if verbose:
        logger.debug("---------------["+variable_name+"]---["+str(level)+"]---["+str(longitude)+"]---------------")
    '''
    if level is not None:
        try:
            variable_values_all, unique_lats, mtime_values, longitude, variable_unit, variable_long_name, filename = lat_time_lev(datasets, variable_name, level, longitude, variable_unit)
        except:
            variable_values_all, unique_lats, mtime_values, longitude, variable_unit, variable_long_name, filename = lat_time_ilev(datasets, variable_name, level, longitude, variable_unit)
    else:
        variable_values_all, unique_lats, mtime_values, longitude, variable_unit, variable_long_name, filename = lat_time(datasets, variable_name, longitude, variable_unit)
    '''
    result = arr_lat_time(datasets, variable_name, longitude, level, variable_unit, plot_mode=True)
    variable_values_all = result.values
    unique_lats = result.lats
    mtime_values = result.mtime_values
    longitude = result.selected_lon
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    model = result.model
    filename = result.filename
    # Assuming the levels are consistent across datasets, but using the minimum size for safety
    if latitude_minimum is None:
        latitude_minimum = np.nanmin(unique_lats)
    if latitude_maximum is None:
        latitude_maximum = np.nanmax(unique_lats)
    num_deleted_before = 0
    num_deleted_after = 0

    if mtime_minimum is not None and mtime_maximum is not None:
        new_mtime_values = []
        for t_mtime in mtime_values:
            mtime_total_minutes = t_mtime[0] * 24 * 60 *60 + t_mtime[1] * 60 *60+ t_mtime[2] *60+ t_mtime[3]
            mtime_min_total = mtime_minimum[0] * 24 * 60*60 + mtime_minimum[1] * 60 *60+ mtime_minimum[2]*60 + mtime_minimum[3]
            mtime_max_total = mtime_maximum[0] * 24 * 60*60 + mtime_maximum[1] * 60 *60+ mtime_maximum[2]*60 + mtime_minimum[3]
            if mtime_total_minutes >= mtime_min_total and mtime_total_minutes <= mtime_max_total:
                new_mtime_values.append(t_mtime)   
            else:
                if mtime_total_minutes < mtime_min_total:
                    num_deleted_before += 1
                elif mtime_total_minutes > mtime_max_total:
                    num_deleted_after += 1
        mtime_values = new_mtime_values
        mtime_values_sorted = sorted(mtime_values, key=lambda x: (x[0], x[1], x[2], x[3]))
        variable_values_all = variable_values_all[:, num_deleted_before:-num_deleted_after]
    min_val, max_val = np.nanmin(variable_values_all), np.nanmax(variable_values_all)
    
    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif not symmetric_interval:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val))/10)*10
        interval_value = range_half / (contour_intervals // 2)  # Divide by 2 to get intervals for one side
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)  # Generate negative levels symmetrically
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))

    
    interval_value = contour_value if contour_value else (max_val - min_val) / (contour_intervals - 1)

    mtime_tuples = [tuple(entry) for entry in mtime_values]
    try:    # Modify this part to show both day and hour
        unique_times = sorted(list(set([(day, hour) for day, hour, _, _ in mtime_values])))
        time_indices = [i for i, (day, hour, _, _) in enumerate(mtime_tuples) if i == 0 or mtime_tuples[i-1][:2] != (day, hour)]
        if len(time_indices) >24:
            unique_times = sorted(list(set([day for day, _, _, _ in mtime_values])))
            time_indices = [i for i, (day, _, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]
    except (ValueError, TypeError):
        unique_times = sorted(list(set([day for day, _, _ in mtime_values])))
        time_indices = [i for i, (day, _, _) in enumerate(mtime_values) if i == 0 or mtime_values[i-1][0] != day]

        # Clean plot
    if not clean_plot:
        figure_height = 6 
        figure_width = 10
    elif clean_plot:
        figure_height = 5
        figure_width = 10
    # Generate contour plot
    plot = plt.figure(figsize=(figure_width, figure_height))
    X, Y = np.meshgrid(range(len(mtime_values)), unique_lats)
    contour_filled = plt.contourf(X, Y, variable_values_all, cmap=cmap_color, levels=contour_levels, vmin=cmap_lim_min, vmax=cmap_lim_max)
    contour_lines = plt.contour(X, Y, variable_values_all, colors=line_color, linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)
    cbar = plt.colorbar(contour_filled, label=variable_name + " [" + variable_unit + "]")
    cbar.set_label(variable_name + " [" + variable_unit + "]", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    try:
        plt.xticks(time_indices, ["{}-{:02d}h".format(day, hour) for day, hour in unique_times], rotation=45)
        plt.xlabel("Model Time (Day,Hour) from "+str(unique_times[0])+" to "+str(unique_times[-1]), fontsize=14) 
    except (ValueError, TypeError):
        plt.xticks(time_indices, unique_times, rotation=45)
        plt.xlabel("Model Time (Day) from "+str(np.nanmin(unique_times))+" to "+str(np.nanmax(unique_times)) ,fontsize=14)
    
    plt.ylabel('Latitude (Deg)',fontsize=14)
        
    plt.tight_layout()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(latitude_minimum, latitude_maximum)

    if not clean_plot:
        plt.title(variable_long_name+' '+variable_name+' ('+variable_unit+') '+'\n\n',fontsize=18 )   
        # Add subtext to the plot
        subtitle_parts = []
        if level is not None:
            subtitle_parts.append(_level_label(level, original_height=_original_height))
        if longitude is not None:
            subtitle_parts.append('LON= Mean' if longitude == 'mean' else 'LON=' + str(longitude))
        if subtitle_parts:
            plt.text(0.5, 1.08, '  ' + ' '.join(subtitle_parts), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.2, "Min, Max = " + str("{:.2e}".format(min_val)) + ", " + str("{:.2e}".format(max_val)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, -0.25, "Contour Interval = " + str("{:.2e}".format(interval_value)), ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            plt.show(block=False)
        return plot
    else:
        backend = get_backend()
        if "Qt" in backend:
            return plot, variable_unit, contour_intervals, contour_value, symmetric_interval, cmap_color, cmap_lim_min, cmap_lim_max, line_color, latitude_minimum, latitude_maximum, contour_filled, unique_lats, variable_values_all
        if plot is not None:
            plt.close(plot)
        return plot


def plt_sat_track(datasets, variable_name, sat_time, sat_lat, sat_lon,
                  level=None, variable_unit=None, contour_intervals=10,
                  contour_value=None, symmetric_interval=False,
                  cmap_color=None, cmap_lim_min=None, cmap_lim_max=None,
                  line_color='white', clean_plot=False, verbose=False):
    """
    Plots model data interpolated along a satellite trajectory.

    If a level is specified, produces a 1D line plot of the variable vs time
    along the track. If no level is given, produces a 2D contour plot of
    the variable (levels vs along-track time).

    Args:
        datasets (list[ModelDataset]): Loaded model datasets.
        variable_name (str): The variable to plot.
        sat_time (array-like): Satellite timestamps (numpy datetime64).
        sat_lat (array-like): Satellite latitudes (degrees).
        sat_lon (array-like): Satellite longitudes (degrees).
        level (float, optional): Level/ilev to extract at. If None, plots all levels vs time.
        variable_unit (str, optional): Desired unit of the variable.
        contour_intervals (int, optional): Number of contour intervals (2D mode). Defaults to 10.
        contour_value (float, optional): Value between each contour interval.
        symmetric_interval (bool, optional): Symmetric contour intervals around zero. Defaults to False.
        cmap_color (str, optional): Colormap name.
        cmap_lim_min (float, optional): Minimum colormap limit.
        cmap_lim_max (float, optional): Maximum colormap limit.
        line_color (str, optional): Contour line color. Defaults to 'white'.
        clean_plot (bool, optional): If True, hides subtext. Defaults to False.
        verbose (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    result = arr_sat_track(datasets, variable_name, sat_time, sat_lat, sat_lon,
                           selected_lev_ilev=level, selected_unit=variable_unit,
                           plot_mode=True)

    variable_values = result.values
    variable_unit = result.variable_unit
    variable_long_name = result.variable_long_name
    model = result.model

    sat_time = np.asarray(sat_time, dtype='datetime64[ns]')
    n_points = len(sat_time)

    if cmap_color is None:
        cmap_color, line_color = color_scheme(variable_name, model)

    # 1D line plot when a specific level is selected
    if level is not None or variable_values.ndim == 1:
        if clean_plot:
            plot = plt.figure(figsize=(10, 4))
        else:
            plot = plt.figure(figsize=(10, 5))

        x_vals = np.arange(n_points)
        plt.plot(x_vals, variable_values)
        plt.xlabel('Along-Track Point', fontsize=14)
        plt.ylabel(variable_name + ' [' + variable_unit + ']', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if not clean_plot:
            title = variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')'
            if level is not None:
                title += '\nZP=' + str(level)
            title += ' — Satellite Track'
            plt.title(title, fontsize=16)

        if is_notebook():
            backend = get_backend()
            if "inline" in backend or "nbagg" in backend:
                plt.show(block=False)
            return plot
        else:
            if plot is not None:
                plt.close(plot)
            return plot

    # 2D contour plot: levels vs along-track points
    levs = result.levs
    if clean_plot:
        plot = plt.figure(figsize=(10, 5))
    else:
        plot = plt.figure(figsize=(10, 6))

    min_val, max_val = min_max(variable_values)

    if cmap_lim_min is None:
        cmap_lim_min = min_val
    else:
        min_val = cmap_lim_min
    if cmap_lim_max is None:
        cmap_lim_max = max_val
    else:
        max_val = cmap_lim_max

    if contour_value is not None:
        contour_levels = np.arange(min_val, max_val + contour_value, contour_value)
        interval_value = contour_value
    elif symmetric_interval:
        range_half = math.ceil(max(abs(min_val), abs(max_val)) / 10) * 10
        interval_value = range_half / (contour_intervals // 2)
        positive_levels = np.arange(interval_value, range_half + interval_value, interval_value)
        negative_levels = -np.flip(positive_levels)
        contour_levels = np.concatenate((negative_levels, [0], positive_levels))
    else:
        contour_levels = np.linspace(min_val, max_val, contour_intervals)
        interval_value = (max_val - min_val) / (contour_intervals - 1)

    X = np.arange(n_points)
    Y = levs

    contour_filled = plt.contourf(X, Y, variable_values, cmap=cmap_color,
                                  levels=contour_levels, vmin=cmap_lim_min,
                                  vmax=cmap_lim_max)
    contour_lines = plt.contour(X, Y, variable_values, colors=line_color,
                                linewidths=0.5, levels=contour_levels)
    plt.clabel(contour_lines, inline=True, fontsize=8, colors=line_color)

    cbar = plt.colorbar(contour_filled, fraction=0.046, pad=0.04, shrink=0.65)
    cbar.set_label(variable_name + ' [' + variable_unit + ']', size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=9)

    plt.xlabel('Along-Track Point', fontsize=14)
    plt.ylabel('Level', fontsize=14)
    plt.tight_layout()

    if not clean_plot:
        title = variable_long_name + ' ' + variable_name + ' (' + variable_unit + ')'
        title += ' — Satellite Track'
        plt.title(title + '\n\n', fontsize=16)
        plt.text(0.5, -0.15, "Min, Max = " + str("{:.2e}".format(min_val)) + ", " +
                 str("{:.2e}".format(max_val)) + "   Contour Interval = " +
                 str("{:.2e}".format(interval_value)),
                 ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)

    if is_notebook():
        backend = get_backend()
        if "inline" in backend or "nbagg" in backend:
            plt.show(block=False)
        return plot
    else:
        if plot is not None:
            plt.close(plot)
        return plot
