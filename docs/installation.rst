Installation
============

If the automatic (pip based) install doesn't work below, you may need to follow some of the manual steps.


Python Environment
------------------

If you don't yet have a working python installation/environment, then e.g. on the MPCDF machines (vera, freya, raven, virgo, and so on) you can:

1. Set up a clean anaconda environment:

.. code-block:: bash

    module load anaconda/3/2023.03
    mkdir -p ~/.local/envs
    conda create --prefix=~/.local/envs/myenv python=3.12
    source activate ~/.local/envs/myenv

2. Add the following lines to your ``~/.bashrc`` file for permanence

.. code-block:: bash

    module load gcc/12
    module load openmpi/4.1
    module load fftw-serial/3.3.10
    module load hdf5-serial/1.12.2
    module load gsl/2.4

    module load anaconda/3/2023.03
    source activate ~/.local/envs/myenv


Automatic Installation
----------------------

If you do not plan on editing this package and hope to simply use it as is (not a common use case), you can simply install it

.. code-block:: bash

    pip install temet 
    
Or, if you would like to install the most up-to-date version directly from the repository, 

.. code-block:: bash

    pip install git+ssh://git@github.com/dnelson/temet.git

If you plan on editing, making changes, and adding functionality to this package (**recommended choice**), then first 
clone the repository, then install the package in 'editable' mode. This means that the files in this directory are not copied 
anywhere, but are used as is. Any changes you make are reflected (i.e. immediately usable) in your python environment.

.. code-block:: bash

    git clone git@github.com:dnelson/temet.git
    pip install -e temet


Download Data Files
-------------------

Several large tabulated data files are used to compute e.g. stellar luminosities (from FSPS), ion abundances and emissivities (from CLOUDY), and X-ray emission (from APEC/XPSEC). For convenience these can be downloaded as

.. code-block:: bash

    cd temet/temet/tables/
    wget -r -nH --cut-dirs=1 --no-parent --reject="index.html*" -e robots=off temet.tng-project.org/tables/

The default plotting style uses the Roboto font. To install this font locally, do

.. code-block:: bash

    mkdir -p ~/.fonts/Roboto
    cd ~/.fonts/Roboto/
    wget https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Light.ttf
    wget https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-LightItalic.ttf

Note: on MPCDF machines, suggested organization of simulation directories is as follows

.. code-block:: bash

    mkdir ~/sims.TNG
    mkdir ~/sims.TNG/L75n1820TNG
    mkdir ~/sims.TNG/L75n1820TNG/data.files
    cd ~/sims.TNG/L75n1820TNG/
    ln -s /virgotng/universe/IllustrisTNG/L75n1820TNG/output .
    ln -s /virgotng/universe/IllustrisTNG/L75n1820TNG/postprocessing .

the last two lines create symlinks to the actual output directory where the simulation data files 
(``groupcat_*`` and ``snapdir_*``) reside, as well as to the postprocessing directory (containing ``trees``, etc).
Replace as needed with the actual path on your machine.


External Package Installation
-----------------------------

Several external tools and post-processing codes are used, for specific analysis routines. 
The following additional installation steps are therefore optional, depending on application.

1. Although most stellar light/magnitude tables of relevance are pre-computed and have been downloaded in the previous steps, the `FSPS <https://github.com/cconroy20/fsps>`_ stellar population synthesis package is required to generate new SPS tables. To install:

.. code-block:: bash

    mkdir ~/code
    cd ~/code/
    git clone https://github.com/cconroy20/fsps

edit the ``src/sps_vars.f90`` file and switch the defaults spectral and isochrone libraries to

.. code-block:: c

    MILES 1
    PADOVA 1 (and so MIST 0)

edit ``src/Makefile`` and make sure the F90FLAGS line contains ``-fPIC``, then compile FSPS

.. code-block:: bash

    make

add the following line to your ``~/.bashrc`` file (make sure the path is correct)

.. code-block:: bash

    export SPS_HOME=$HOME/code/fsps/

2. Although the X-ray emission tables have been pre-computed, the creation of new tables requires the `AtomDB APEC <http://www.atomdb.org/>`_ files.

.. code-block:: bash

    mkdir ~/code/atomdb/
    cd ~/code/atomdb/
    wget --content-disposition http://www.atomdb.org/download_process.php?fname=apec_v3_0_9
    wget --content-disposition http://www.atomdb.org/download_process.php?fname=apec_v3_0_9_nei
    tar -xvf *.bz2 --strip-components 1
    rm *.bz2

3. The `SKIRT <https://skirt.ugent.be/>`_ dust radiative transfer code can be used to compute dust-attenuated stellar light images and spectra, dust infrared emission, and many further sophisticated observables.

.. code-block:: bash

    mkdir ~/code/SKIRT9/
    cd ~/code/SKIRT9/
    git clone https://github.com/SKIRT/SKIRT9.git git
    cd git
    chmod +rx *.sh
    ./makeSKIRT.sh
    ./downloadResources.sh

link the executable into your local bin directory

.. code-block:: bash

    mkdir ~/.local/bin
    cd ~/.local/bin
    ln -s ~/code/SKIRT9/release/SKIRT/main/skirt .

add the following lines to your ``~/.bashrc`` file for permanence

.. code-block:: bash

    export PATH=$HOME/.local/bin:$PATH

Now you're all set!


Updating
--------

The instructions above install a local copy into your home directory, which you are free to edit and modify as required. 
At any time you can then update your copy to the newest version, pulling in any changes, bugfixes, and so on, with::

    git pull

However if you have made changes in the meantime, you may see a message similar to "error: Your local changes to the 
following files would be overwritten by merge. Please commit your changes or stash them before you merge." 
In this case, you want to keep your local work, but also make the update. 
Please read this `quick git tutorial <https://happygitwithr.com/pull-tricky.html>`_ on the topic.
