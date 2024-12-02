
Installation
====================================================================================================================================================================================================================================

Custom
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Conda Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create python 3.8 environment

.. note::

   The name of the conda environment in this example is ``tiegcm``.

.. code-block:: bash

    conda create --name tiegcm python=3.8

Installing gcmprocpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install gcmprocpy, run the following command:

.. code-block:: bash

    pip install gcmprocpy

NCAR Derecho
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Creating Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load Conda module

.. code-block:: bash

    module load conda

Create python 3.8 environment

.. note::

   The name of the conda environment in this example is ``tiegcm``.

.. code-block:: bash

    conda create --name tiegcm python=3.8

Activate Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Make sure the conda module is loaded.

.. code-block:: bash

    conda activate tiegcm

Installing gcmprocpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install gcmprocpy, run the following command:

.. code-block:: bash

    pip install gcmprocpy

Installing gcmprocpy for Jupyter Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Make sure the conda module is loaded.

.. code-block:: bash

    conda activate tiegcm

Install ipykernal to use the conda environment for Jupyter notebooks.

.. code-block:: bash

    pip install ipykernel

VS Code Jupyter Notebooks (Casper Nodes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

### Step 1: Request an Interactive Session

To begin, you need to request an interactive session on a **compute node** using `qsub`. This will allocate resources for your job, allowing you to use multiple processors instead of running on the login node.

1. Open your terminal on the **Casper login node** and enter the following command to request an interactive session:

   ```bash
   qsub -I -A P28100045 -q casper -l select=1:ncpus=4:mpiprocs=4 -l walltime=01:00:00
   ```

   - `-I` specifies an interactive session.
   - `-A P28100045` is your project/account code (replace with your own).
   - `-q casper` requests the **Casper** queue.
   - `-l select=1:ncpus=4:mpiprocs=4` requests 1 node with 4 CPUs and 4 MPI processes.
   - `-l walltime=01:00:00` specifies the walltime limit (1 hour in this case).

2. Once the job is submitted, you will see the following output indicating the job's status:

   ```bash
   qsub: waiting for job 2884283.casper-pbs to start
   qsub: job 2884283.casper-pbs ready
   ```

### Step 2: Check the Hostname of the Compute Node

After the job is ready, you need to check the hostname of the compute node that has been allocated to you.

1. Run the following command to display the hostname of your current session:

   ```bash
   echo $HOSTNAME
   ```

   This will output something like:

   ```bash
   crhtc62
   ```

   This `hostname` is used to connect to the compute node via **SSH**.

### Step 3: Connect to the Compute Node from VSC

1. Open **Visual Studio Code** on your local machine.
2. Use the **Remote-SSH** extension in VSC to connect to the compute node.
3. In VSC, configure the SSH connection to the node using the following format:

   ```
   $HOSTNAME.hpc.ucar.edu
   ```

   Replace `$HOSTNAME` with the actual hostname from the previous step (e.g., `crhtc62`).

4. Once connected, you can start editing and running code on the compute node as needed.

.. note::

    This process **only works on Casper** and will not work on **Derecho** due to firewall rules on Derecho.

NCAR JupyterHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   NCAR JupyterHub only workes when matplotlib is in inline backend. 
   Use the following at the start of your Jupyter notebook to enable inline backend.

   .. code-block:: python

       %matplotlib inline

Open JupyterHub by visiting the `NCAR JupyterHub <https://jupyterhub.hpc.ucar.edu/stable/hub/home>`_.

Create a new Jupyter notebook by clicking on the ``New`` button or select an existing notebook.

Change the kernel to the conda environment by clicking on the ``Kernel`` tab on the top left and selecting the conda environment. The conda environment will be listed as ``Python [conda env:tiegcm]``, where ``tiegcm`` is the name of the conda environment.

Follow API documentation to use gcmprocpy in the Jupyter notebook.

NASA Pleiades
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Creating Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   The name of your environment will be set to ``my_{environment_name}`` due to Pleiades deployment.
   Make sure the conda module is loaded.

.. code-block:: bash

    conda activate my_tiegcm

Installing gcmprocpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   cartopy requires geos which doesn't install properly via the pip install. Use the command below if you face the issue.

    .. code-block:: bash

        conda install -c conda-forge cartopy

To install gcmprocpy, run the following command:

.. code-block:: bash

    pip install gcmprocpy
