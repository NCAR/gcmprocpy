from .data_parse import arr_lat_lon,arr_lev_var,arr_lev_lon, arr_lev_lat,arr_lev_time,arr_lat_time, calc_avg_ht, min_max, get_time, time_list
from .plot_gen import plt_lat_lon, plt_lev_var, plt_lev_lon, plt_lev_lat, plt_lev_time, plt_lat_time
import matplotlib.pyplot as plt
import os
import cv2
from IPython.display import Video


def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])

def mov_lat_lon(datasets, variable_name, level = None,  variable_unit = None, contour_intervals = None, contour_value = None,symmetric_interval= False, cmap_color = None, line_color = 'white', coastlines=False, nightshade=False, gm_equator=False, latitude_minimum = None, latitude_maximum = None, longitude_minimum = None, longitude_maximum = None, localtime_minimum = None, localtime_maximum = None ):

    """
    Generates a Latitude vs Longitude contour plot for a variable.
    
    Parameters:
        datasets (xarray): The loaded dataset/s using xarray.
        variable_name (str): The name of the variable with latitude, longitude, and lev/ilev dimensions.
        time (np.datetime64, optional): The selected time, e.g., '2022-01-01T12:00:00'.
        mtime (array, optional): The selected time as a list, e.g., [1, 12, 0] for 1st day, 12 hours, 0 mins.
        level (float, optional): The selected lev/ilev value.
        variable_unit (str, optional): The desired unit of the variable.
        contour_intervals (int, optional): The number of contour intervals. Defaults to 20.
        contour_value (int, optional): The value between each contour interval.
        symmetric_interval (bool, optional): If True, the contour intervals will be symmetric around zero. Defaults to False.
        cmap_color (str, optional): The color map of the contour. Defaults to 'viridis' for Density, 'inferno' for Temp, 'bwr' for Wind, 'viridis' for undefined.
        line_color (str, optional): The color for all lines in the plot. Defaults to 'white'.
        coastlines (bool, optional): Shows coastlines on the plot. Defaults to False.
        nightshade (bool, optional): Shows nightshade on the plot. Defaults to False.
        gm_equator (bool, optional): Shows geomagnetic equator on the plot. Defaults to False.
        latitude_minimum (float, optional): Minimum latitude to slice plots. Defaults to -87.5.
        latitude_maximum (float, optional): Maximum latitude to slice plots. Defaults to 87.5.
        longitude_minimum (float, optional): Minimum longitude to slice plots. Defaults to -180.
        longitude_maximum (float, optional): Maximum longitude to slice plots. Defaults to 175.
        localtime_minimum (float, optional): Minimum local time to slice plots. Defaults to None.
        localtime_maximum (float, optional): Maximum local time to slice plots. Defaults to None.
    
    Returns:
        Contour plot.
    """

    timestamps = time_list(datasets)
    count = 0

    for timestamp in timestamps:
        plot = plt_lat_lon(datasets, variable_name, time= timestamp, level = level,  variable_unit = variable_unit, contour_intervals = contour_intervals, contour_value = contour_value,symmetric_interval= symmetric_interval, cmap_color = cmap_color, line_color = 'white', coastlines=coastlines, nightshade=nightshade, gm_equator=gm_equator, latitude_minimum = latitude_minimum, latitude_maximum = latitude_maximum, longitude_minimum = longitude_minimum, longitude_maximum = longitude_maximum, localtime_minimum = localtime_minimum, localtime_maximum = localtime_maximum)
        plot_filename = f"plt_lat_lon_{count}.png"

        output_dir = os.path.join(os.getcwd(),f"mov_lat_lon_{variable_name}_{level}")
    
        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        plot.savefig(os.path.join(output_dir,plot_filename), bbox_inches='tight', pad_inches=0.5)  # Use savefig to save the plot
        plt.close(plot)  # Close the figure to free up memory
        count += 1
    
    output_dir = os.path.join(os.getcwd(),"mov_lat_lon")
    
    images = [img for img in os.listdir(output_dir) if img.endswith(".png")]
    images.sort(key=extract_number) 
    
    # Read the first image to get the frame size
    frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = frame.shape

    output_file = f'mov_lat_lon_{variable_name}_{level}.mp4'  # Update as needed

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_file, fourcc, 5, (width, height))  # 1 is the fps, adjust as needed

    for image in images:
        video.write(cv2.imread(os.path.join(output_dir, image)))

    cv2.destroyAllWindows()
    video.release()

    # Step 3: Display the Video in Jupyter Notebook
    Video(output_file, embed=True)