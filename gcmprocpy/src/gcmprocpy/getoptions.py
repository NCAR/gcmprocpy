import argparse
import os
def get_options():
    parser = argparse.ArgumentParser(description='Generate different types of plots based on user input.')
    #
    # Input arugmets
    #
    parser.add_argument('-ds','--dataset', type=str, help='Path to the sigular dataset') #dependency check
    parser.add_argument('-dir','--directory', type=str, help='Directory path containing the datasets.')
    parser.add_argument('-dsf','--dataset_filter', type=str,choices=['prim', 'sech'], help='Filter to load datasets.')
    #
    # Output arugmets
    #
    parser.add_argument('-outdir','--output_directory', type=str, default=str(os.getcwd()), help='Directory to save the plots. Default: Current working directory')
    parser.add_argument('-fout','--output_format', type=str, choices=['jpeg', 'pdf'], default='jpeg', help='Format to save the plots. Default: jpeg')
    parser.add_argument('-stdout','--standard_output', type=str, help='Custom file Name without extension')
    #
    # Interactive mode arugmets
    #
    parser.add_argument('-rec','--recursive', action='store_true', help='Enable interactive mode until the user inputs "exit".')
    parser.add_argument('-multiout','--multiple_output', type=str, help='Custom file name without extension and Enables multiple output in a single pdf')
    #
    # Plot generation primary arguments
    #
    parser.add_argument('-plt','--plot_type', type=str, choices=['lat_lon', 'lev_var', 'lev_lon', 'lev_lat', 'lev_time', 'lat_time'], help='Type of the plot to be generated.')
    parser.add_argument('-var','--variable_name', type=str, help='Name of the variable to be plotted.')
    parser.add_argument('-time','--time', type=str, help='Selected time for the plot in YYYY-MM-DDTHH:MM:SS format.')
    parser.add_argument('-mtime','--mtime',  nargs='*', type=int, help='Selected time for the plot in [Day, Hour, Min, Sec] for 3.0 or [Day, Hour, Min] for 2.0 format.')
    parser.add_argument('-zp','--level', type=str, help='Selected lev/ilev for the plot.')
    parser.add_argument('-lat', '--latitude', type=str, help='Selected latitude for the plot.')
    parser.add_argument('-lon','--longitude', type=str, help='Selected longitude for the plot.')
    #
    # Plot generation secondary arguments
    #
    parser.add_argument('-unit','--variable_unit', type=str, help='Selected unit of a given variable for the plot')
    parser.add_argument('-cint','--contour_intervals', type=float, help='Selected number interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time]')
    parser.add_argument('-cval','--contour_value', type=float, help='Selected value of interval of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time]')
    parser.add_argument('-ccolor','--cmap_color', type=float, help='Selected color of cmap of contour for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time]')
    parser.add_argument('-lcolor','--line_color', type=float, help='Selected color of contour lines for the plots [lat_lon, lev_lon, lev_lat, lev_time, lat_time]')
    parser.add_argument('-cstl','--coastlines', action='store_true', help='Add coast lines to the lat_lon plots')
    parser.add_argument('-ntsh','--nightshade', action='store_true', help='Add nightshade to the lat_lon plots')
    parser.add_argument('-gmeq','--gm_equator', action='store_true', help='Add geomagnetic equator to the lat_lon plots')

    #
    # Plot slicing arguments 
    #
    parser.add_argument('-zpmin','--level_minimum', type=float, help='Minimum level to slice plots [lev_var, lev_lon, lev_lat, lev_time]')
    parser.add_argument('-zpmax','--level_maximum', type=float, help='Maximum level to slice plots [lev_var, lev_lon, lev_lat, lev_time]')
    parser.add_argument('-latmin','--latitude_minimum', type=float, help='Minimum latitude to slice plots [lat_lon, lev_lat, lat_time]')
    parser.add_argument('-latmax','--latitude_maximum', type=float, help='Maximum latitude to slice plots [lat_lon, lev_lat, lat_time]')
    parser.add_argument('-lonmin','--longitude_minimum', type=float, help='Minimum longitude to slice plots [lat_lon, lev_lon]')
    parser.add_argument('-lonmax','--longitude_maximum', type=float, help='Maximum longitude to slice plots [lat_lon, lev_lon]')

    
    args = parser.parse_args()
    
    plot_requirements = {
        'lat_lon': {'required': ['variable_name', 'time', 'mtime'],
                    'optional': ['level','variable_unit','coastlines','nightshade','gm_equator','contour_intervals','contour_value','cmap_color','line_color','latitude_minimum','latitude_maximum','longitude_minimum','longitude_maximum'],
                    },
        
        'lev_var': {'required': ['variable_name', 'time', 'mtime', 'latitude', 'longitude'],
                    'optional': ['cmap_color','line_color','variable_unit','level_minimum','level_maximum']
                    },
        'lev_lon': {'required': ['variable_name', 'time', 'mtime', 'latitude'],
                    'optional': ['cmap_color','line_color','variable_unit','contour_intervals','contour_value','level_minimum','level_maximum','longitude_minimum','longitude_maximum']
                    },
        'lev_lat': {'required': ['variable_name', 'time', 'mtime', 'longitude'],
                    'optional': ['cmap_color','line_color','variable_unit','contour_intervals','contour_value','level_minimum','level_maximum','latitude_minimum','latitude_maximum']
                    },
        'lev_time': {'required': ['variable_name', 'latitude', 'longitude'],
                    'optional': ['cmap_color','line_color','variable_unit','contour_intervals','contour_value','level_minimum','level_maximum'] 
                    },
        'lat_time': {'required': ['variable_name', 'longitude'],
                    'optional': ['cmap_color','line_color','level','variable_unit','contour_intervals','contour_value','latitude_minimum','latitude_maximum']
                    },
    }
    '''
    valid_values = {
        'valid_variables': ['TN', 'UN', 'VN', 'WN', 'O2', 'O1', 'N2', 'NO', 'N4S', 'HE', 'NE', 'TE', 'TI', 'TEC', 'O2P', 'OP', 'POTEN', 'UI_ExB', 'VI_ExB', 'WI_ExB', 'DEN', 'QJOULE', 'HMF2', 'NMF2', 'Z', 'ZG', 'ZMAG', 'TLBC', 'ULBC', 'VLBC', 'TLBC_NM', 'ULBC_NM', 'VLBC_NM', 'LBC'],
        'valid_longitudes': [-180., -175., -170., -165., -160., -155., -150., -145., -140., -135., -130., -125., -120., -115., -110., -105., -100.,  -95., -90.,  -85.,  -80.,  -75.,  -70.,  -65.,  -60.,  -55.,  -50., -45.,  -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,  0.,    5.,   10.,   15.,   20.,   25.,   30.,   35.,   40., 45.,   50.,   55.,   60.,   65.,   70.,   75.,   80.,   85., 90.,   95.,  100.,  105.,  110.,  115.,  120.,  125.,  130., 135.,  140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.],
        'valid_latitudes': [-87.5, -82.5, -77.5, -72.5, -67.5, -62.5, -57.5, -52.5, -47.5, -42.5, -37.5, -32.5, -27.5, -22.5, -17.5, -12.5,  -7.5,  -2.5, 2.5, 7.5,  12.5,  17.5,  22.5,  27.5,  32.5,  37.5,  42.5, 47.5,  52.5,  57.5,  62.5,  67.5,  72.5,  77.5,  82.5,  87.5],
        'valid_levels': [-7.  , -6.75, -6.5 , -6.25, -6.  , -5.75, -5.5 , -5.25, -5.  , -4.75, -4.5 , -4.25, -4.  , -3.75, -3.5 , -3.25, -3.  , -2.75, -2.5 , -2.25, -2.  , -1.75, -1.5 , -1.25, -1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75, 2.  ,  2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  , 4.25,  4.5 ,  4.75,  5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25, 6.5 ,  6.75,  7.  ,  7.25],
    }
    '''
    if args.recursive:
        print('Entering Interactive Mode')

    if args.multiple_output:
         if getattr(args, 'recursive') is None:
             parser.error(f"--multiple_output requires tiepy to be in interactive mode to pass multiple outputs using --recursive")
    #
    # Check and verify required arguments and values for the plots.
    #
    if args.plot_type in plot_requirements:
        requirements = plot_requirements[args.plot_type]
        #
        # Check for required arguments except time, mtime, longitude.
        #
        for arg in requirements['required']:
            if arg in ['time', 'mtime', 'longitude']:
                continue
            if getattr(args, arg) is None:
                parser.error(f"{args.plot_type} requires the argument --{arg}")
        #
        # Enforce that either 'time' or 'mtime' must be provided
        #
        if 'time' in requirements['required'] or 'mtime' in requirements['required']:
            if getattr(args, 'time') is None and getattr(args, 'mtime') is None:
                parser.error(f"{args.plot_type} requires either the argument --time or --mtime")
        #
        # Check if 'longitude' is in requirements and at least one of them is provided
        #
        if 'longitude' in requirements['required']:
            if getattr(args, 'longitude') is None:
                parser.error(f"{args.plot_type} requires either the argument --longitude or --localtime")
        '''
        #
        # Check if valid values of 'longitude' is provided.
        #
        if 'longitude' in requirements['required']:
            if args.longitude is not None and args.longitude not in valid_values['valid_longitudes']:
                parser.error(f"Invalid longitude for {args.plot_type}. \n              Valid longitudes are {str(valid_values['valid_longitudes'])}")
        #
        # Check if valid values of 'level' is provided.
        #
        if 'level' in requirements['required']:
            if float(args.level) not in valid_values['valid_levels']:
                parser.error(f"Invalid level for {args.plot_type}. \n              Valid levels are {str(valid_values['valid_levels'])}")
        #
        # Check if valid values of 'latitude' is provided.
        #
        if 'latitude' in requirements['required']:
            if args.latitude not in valid_values['valid_latitudes']:
                parser.error(f"Invalid latitude for {args.plot_type}. \n              Valid latitudes are {str(valid_values['valid_latitudes'])}")
        #
        # Check if valid values of 'variable_name' is provided.
        #
        if 'variable_name' in requirements['required']:     
            if args.variable_name not in valid_values['valid_variables']:
                parser.error(f"Invalid variable_name for {args.plot_type}. \n              Valid variables are {' '.join(valid_values['valid_variables'])}")
        '''        
        #
        # Check and verify optional arguments for the plots.
        #
        
        #
        # Set valid optional argument flags
        #
        coastline_plot = True if 'coastlines' in requirements.get('optional', []) else False
        level_bound_plot = True if any(key in requirements.get('optional', []) for key in ['level_minimum', 'level_maximum']) else False
        longitude_bound_plot = True if any(key in requirements.get('optional', []) for key in ['latitude_minimum', 'latitude_maximum', 'longitude_minimum', 'longitude_maximum']) else False
        latitude_bound_plot = True if any(key in requirements.get('optional', []) for key in ['latitude_minimum', 'latitude_maximum']) else False
        #
        # Check if coastlines is used and applies to the provided plot.
        #
        if coastline_plot is not True:
            if args.coastlines is not None:
                parser.error(f"--coastlines can only be used with plot types: lat_lon") 
        #
        # Check if level_minimum and/or level_maximum is used and applies to the provided plot.
        #
        if level_bound_plot is not True:
            if args.level_minimum is not None or args.level_maximum is not None:
                parser.error(f"--level_minimum and --level_maximum are not valid for {args.plot_type} \n              Valid optoins for {args.plot_type}: {', '.join(plot_requirements[args.plot_type]['optional'])} ")     
        #
        # Check if longitude_minimum and/or longitude_maximum is used and applies to the provided plot.
        #
        if longitude_bound_plot is not True:
            if args.longitude_minimum is not None or args.longitude_maximum is not None:
                parser.error(f"--longitude_minimum and --longitude_maximum are not valid for {args.plot_type} \n              Valid optoins for {args.plot_type}: {', '.join(plot_requirements[args.plot_type]['optional'])} ")  
        #
        # Check if latitude_minimum and/or latitude_maximum is used and applies to the provided plot.
        #
        if latitude_bound_plot is not True:
            if args.latitude_minimum is not None or args.latitude_maximum is not None:
                parser.error(f"--latitude_minimum and --latitude_maximum are not valid for {args.plot_type} \n              Valid optoins for {args.plot_type}: {', '.join(plot_requirements[args.plot_type]['optional'])} ") 


        if args.dataset:
            if args.dataset.endswith('.nc') is not True:
                parser.error(f"{args.plot_type} expects '.nc' files")
    #
    # Setting variable type float if level value is numerical 
    #
    if args.level != 'mean' and args.level != None:
        try:
            args.level = float(args.level)
        except:
            parser.error(f"{args.plot_type} expects --level values to be numerical float values or 'mean'")
    #
    # Setting variable type float if latitude value is numerical 
    #
    if args.latitude != 'mean' and args.latitude != None:
        try:
            args.latitude = float(args.latitude)
        except:
            parser.error(f"{args.plot_type} expects --latitude values to be numerical float values or 'mean'")
    #
    # Setting variable type float if longitude value is numerical 
    #
    if args.longitude != 'mean'and args.longitude != None:
        try:
            args.longitude = float(args.longitude)
        except:
            parser.error(f"{args.plot_type} expects --longitude values to be numerical float values or 'mean'")

    return args
