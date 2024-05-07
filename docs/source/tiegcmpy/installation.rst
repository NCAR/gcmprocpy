
Installation
============

Custom
------

Conda Installation
~~~~~~~~~~~~~~~~~~~

1. Download Miniconda Installer

   - Visit the `Miniconda Downloads <https://docs.conda.io/en/latest/miniconda.html>`_ page.
   - Select the Linux installer for Python 2.x or 3.x as per your requirements.

2. Open a Terminal

   - Open a terminal window on Linux by pressing ``Ctrl + Alt + T``, or search for "Terminal" in the applications menu.

3. Navigate to Download Directory

   .. code-block:: bash

       cd ~/Downloads

   Change to the directory where the installer was downloaded, usually the ``Downloads`` directory.

4. Make the Installer Executable

   - Make the downloaded script executable. Replace ``Miniconda3-latest-Linux-x86_64.sh`` with the actual downloaded file name.

     .. code-block:: bash

         chmod +x Miniconda3-latest-Linux-x86_64.sh

5. Run the Installer

   - Execute the installer script and follow the on-screen instructions.

     .. code-block:: bash

         ./Miniconda3-latest-Linux-x86_64.sh

   - You'll need to approve the license agreement and choose the installation location.

6. Initialize Conda

   - After installation, initialize Miniconda to add Conda to your PATH.

7. Close and Reopen Your Terminal

   - To apply the changes, close and reopen your terminal window.

Creating Environment
~~~~~~~~~~~~~~~

Create python 3.8 environment

.. note::

   The name of the conda environment in this example is ``tiegcm``.

.. code-block:: bash

    conda create --name tiegcm python=3.8

Installing tiegcmpy
~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install tiegcmpy, run the following command:

.. code-block:: bash

    pip install tiegcmpy

NCAR Derecho
------------
Creating Environment
~~~~~~~~~~~~~~~

Load Conda module

.. code-block:: bash

    module load conda

Create python 3.8 environment

.. note::

   The name of the conda environment in this example is ``tiegcm``.

.. code-block:: bash

    conda create --name tiegcm python=3.8

Activate Environment
~~~~~~~~~~~~~~~

.. note::

   Make sure the conda module is loaded.

.. code-block:: bash

    conda activate tiegcm

Installing tiegcmpy
~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install tiegcmpy, run the following command:

.. code-block:: bash

    pip install tiegcmpy

NASA Pleiades
-------------
Creating Environment
~~~~~~~~~~~~~~~

Load Conda module

.. code-block:: bash

    module use -a /swbuild/analytix/tools/modulefiles
    module load miniconda3/v4

.. note::

   Replace ``$USER`` with your username on Pleiades.

.. code-block:: bash

    export CONDA_PKGS_DIRS=/nobackup/$USER/.conda/pkgs

Create python 3.8 environment

.. code-block:: bash

    conda create -n tiegcm python=3.8

Activate Environment
~~~~~~~~~~~~~~~

.. note::

   The name of your environment will be set to ``my_{environment_name}`` due to Pleiades deployment.
   Make sure the conda module is loaded.

.. code-block:: bash

    conda activate my_tiegcm

Installing tiegcmpy
~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install tiegcmpy, run the following command:

.. code-block:: bash

    pip install tiegcmpy
