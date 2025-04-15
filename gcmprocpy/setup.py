from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gcmprocpy',
    version='1.2.1',
    author = "Nikhil Rao",
    author_email = "nikhilr@ucar.edu",
    description='A Python3 post processing tool for TIE-GCM and WACCM-X',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NCAR/gcmprocpy', 
    python_requires='>=3.8',
    install_requires=[
        'cartopy',
        'matplotlib',
        'numpy',
        'xarray',
        'ipython',
        'geomag',
        'netcdf4',
        'ipympl',
        'mplcursors',
        'PyQt5',
    ],
    package_dir={'': 'src'},  
    packages=find_packages(where='src'), 
    entry_points={
        'console_scripts': [
            'lat_lon= gcmprocpy.cmd.cmd_lat_lon:cmd_plt_lat_lon',
            'lev_var= gcmprocpy.cmd.cmd_lev_var:cmd_plt_lev_var',
            'lev_lat= gcmprocpy.cmd.cmd_lev_lat:cmd_plt_lev_lat',
            'lev_lon= gcmprocpy.cmd.cmd_lev_lon:cmd_plt_lev_lon',
            'lev_time= gcmprocpy.cmd.cmd_lev_time:cmd_plt_lev_time',
            'lat_time= gcmprocpy.cmd.cmd_lat_time:cmd_plt_lat_time',
            'gcmprocpy= gcmprocpy.gui.gcmprocpy:main',     
        ]
    }
)
