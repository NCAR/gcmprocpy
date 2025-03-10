from .data_parse import arr_lat_lon,arr_lev_var,arr_lev_lon, arr_lev_lat,arr_lev_time,arr_lat_time, calc_avg_ht, min_max, get_time, time_list
from .plot_gen import plt_lat_lon, plt_lev_var, plt_lev_lon, plt_lev_lat, plt_lev_time, plt_lat_time
import matplotlib.pyplot as plt
import os
from IPython.display import Video, display
import numpy as np
import shutil
from matplotlib.animation import FuncAnimation

def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])

def mov_lat_lon(datasets, variable_name, level = None,  variable_unit = None, center_longitude = 0, contour_intervals = None, contour_value = None,symmetric_interval= False, cmap_color = None, cmap_lim_min = None, cmap_lim_max = None, line_color = 'white', coastlines=False, nightshade=False, gm_equator=False, latitude_minimum = None, latitude_maximum = None, longitude_minimum = None, longitude_maximum = None, time_minimum=None, time_maximum=None, fps=None, clean_plot= False):

    """
    Generates a Latitude vs Longitude contour plot for a variable and creates a video of the plot over time.

    Args:
        datasets (xarray.Dataset): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        level (float, optional): The selected lev/ilev value. Defaults to None.
        variable_unit (str, optional): The desired unit of the variable. Defaults to None.
        center_longitude (float, optional): The central longitude for the plot. Defaults to 0.        
        contour_intervals (int, optional): The number of contour intervals. Defaults to None.
        contour_value (int, optional): The value between each contour interval. Defaults to None.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to None.
        cmap_lim_min (float, optional): Minimum limit for the color map. Defaults to the minimum value of the variable.
        cmap_lim_max (float, optional): Maximum limit for the color map. Defaults to the maximum value of the variable.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
        nightshade (bool, optional): Shows nightshade on the plot. Defaults to False.
        gm_equator (bool, optional): Shows geomagnetic equator on the plot. Defaults to False.
        latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to None.
        latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to None.
        longitude_minimum (float, optional): Minimum longitude to slice plots. Defaults to None.
        longitude_maximum (float, optional): Maximum longitude to slice plots. Defaults to None.
        time_minimum (Union[np.datetime64, str], optional): Minimum time for the plot. Defaults to None.
        time_maximum (Union[np.datetime64, str], optional): Maximum time for the plot. Defaults to None.
        fps (int, optional): Frames per second for the video. Defaults to None.

    Returns:
        Video file: A video file of the contour plot over the specified time range.
    """

    if isinstance(time_minimum, str):
        time_minimum = np.datetime64(time_minimum, 'ns')
    if isinstance(time_maximum, str):
        time_maximum = np.datetime64(time_maximum, 'ns')

    timestamps = np.array(time_list(datasets))
    
    try:
        filtered_timestamps = timestamps[(timestamps >= time_minimum) & (timestamps <= time_maximum)]
    except:
        filtered_timestamps = timestamps
    count = 0
    
    output_dir = os.path.join(os.getcwd(),f"mov_lat_lon_{variable_name}_{level}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for timestamp in filtered_timestamps:
        plot = plt_lat_lon(datasets, variable_name, time= timestamp, level = level,  variable_unit = variable_unit,  center_longitude = center_longitude, contour_intervals = contour_intervals, contour_value = contour_value,symmetric_interval= symmetric_interval, cmap_color = cmap_color,  cmap_lim_min = cmap_lim_min, cmap_lim_max = cmap_lim_max,line_color = 'white', coastlines=coastlines, nightshade=nightshade, gm_equator=gm_equator, latitude_minimum = latitude_minimum, latitude_maximum = latitude_maximum, longitude_minimum = longitude_minimum, longitude_maximum = longitude_maximum)
        plot_filename = f"plt_lat_lon_{count}.png"

    
        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        plot.savefig(os.path.join(output_dir,plot_filename), bbox_inches='tight', pad_inches=0.5)  # Use savefig to save the plot
        plt.close(plot)  # Close the figure to free up memory
        count += 1
    
    output_dir = os.path.join(os.getcwd(),f"mov_lat_lon_{variable_name}_{level}")
    
    images = [img for img in os.listdir(output_dir) if img.endswith(".png")]
    images.sort(key=extract_number) 
    
    output_file = f'mov_lat_lon_{variable_name}_{level}.mp4'  # Update as needed
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update(frame):
        img = plt.imread(frame)
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off() 
    if fps is None:
        fps = 5
    filepaths = [os.path.join(output_dir, img) for img in images]    
    ani = FuncAnimation(fig, update, frames=filepaths, repeat_delay=1000, interval=1000/fps)

    ani.save(output_file)
    plt.close(fig)
    return (Video(output_file, embed=True))