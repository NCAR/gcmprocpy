from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tiegcmpy',
    version='1.5.6',
    author = "Nikhil Rao",
    author_email = "nikhilr@ucar.edu",
    description='A Python3 post processing tool for TIE-GCM',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NCAR/tiegcmpy', 
    python_requires='>=3.8',
    install_requires=[
        'cartopy',
        'matplotlib',
        'numpy',
        'xarray',
        'ipython',
        'geomag',
    ],
    package_dir={'': 'src'},  
    packages=find_packages(where='src'), 
    entry_points={
        'console_scripts': [
            'tiegcmpy= tiegcmpy.main:main',
            'getoptions= tiegcmpy.getoptions:main'
        ]
    }
)
