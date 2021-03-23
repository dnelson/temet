Installation
============

In the future a ``pip install`` option will exist, but for now, manual installation is required.

Manual Installation
-------------------

The following steps will install this package and get the environment set up on a typical cluster environment, for instance the MPCDF machines (freya, isaac, draco, raven, virgo, and so on). 

1. Clone the repository into your home directory, here into a ``python`` directory

.. code-block:: bash

    cd ~
    git clone git@github.com:dnelson86/python.git

2. Clone the public Illustris data analysis python scripts

.. code-block:: bash

    mkdir ~/illustris_release
    cd ~/illustris_release
    git clone git@github.com:illustristng/illustris_python.git

3. Make sure both are set in the ``$PYTHONPATH`` environment variable, and set the ``$PYTHONSTARTUP``.
For example, add the following lines to your ``~/.bashrc`` file

.. code-block:: bash

    export PYTHONPATH+=:$HOME/python/:$HOME/illustris_release/
    export PYTHONSTARTUP=$HOME/python/.startup.py

4. Load or install python (3.6+, 3.9.x recommended). For example, on the MPCDF machines, using a clean anaconda

.. code-block:: bash

    module load anaconda/3/2019.03
    mkdir -p ~/.local/envs
    conda create --prefix=~/.local/envs/myenv python=3.9
    source activate ~/.local/envs/myenv

and add the following lines to your ``~/.bashrc`` file for permanence

.. code-block:: bash

    module load intel/19.0.5
    module load impi/2019.5
    module load fftw/3.3.8
    module load hdf5-serial/intel-18.0/1.8.21
    module load gsl/2.4

    module load anaconda/3/2019.03
    source activate ~/.local/envs/myenv
    export PATH=$HOME/.local/envs/myenv/bin/:$PATH

5. Install all python dependencies as required

.. code-block:: bash

    pip install --user -r ~/python/requirements.txt

6. Point ``matplotlib`` to the default settings file

.. code-block:: bash

    mkdir -p ~/.config/matplotlib
    ln -s ~/python/matplotlibrc ~/.config/matplotlib/

and install the Roboto font used by default

.. code-block:: bash

    mkdir -p ~/.fonts/Roboto
    cd ~/.fonts/Roboto/
    wget https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Light.ttf
    wget https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-LightItalic.ttf

7. Several large tabulated data files are used to compute e.g. stellar luminosities (from FSPS), ion abundances and emissivities (from CLOUDY), and x-ray emission (from XPSEC). For convenience these can be downloaded as

.. code-block:: bash

    cd ~/python/tables/
    wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" -e robots=off www.tng-project.org/files/dnelson_tables/

8. Organize simulation directories as follows

.. code-block:: bash

    mkdir ~/sims.TNG
    mkdir ~/sims.TNG/L75n1820TNG
    mkdir ~/sims.TNG/L75n1820TNG/data.files
    cd ~/sims.TNG/L75n1820TNG/
    ln -s /virgo/simulations/IllustrisTNG/L75n1820TNG/output .
    ln -s /virgo/simulations/IllustrisTNG/L75n1820TNG/postprocessing .

note that the last two lines create symlinks to the actual output directory where the simulation data files 
(``groupcat_*`` and ``snapdir_*``) reside, as well as to the postprocessing directory (containing ``trees``, etc).
Replace as needed with the actual path on your machine.


Installation (continued)
------------------------

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

add the following line to your ``~/.bashrc`` file

.. code-block:: bash

    export SPS_HOME=$HOME/code/fsps/

2. Although the x-ray emission tables have been pre-computed, the creation of new tables requires the `AtomDB APEC <http://www.atomdb.org/>`_ files.

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

The instructions above install a local copy into your home directory, which you are free to edit and modify as required. At any time you can then update your copy to the newest version, pulling in any changes, bugfixes, and so on, with::

    git pull

However if you have made changes in the meantime, you may see a message similar to "error: Your local changes to the following files would be overwritten by merge. Please commit your changes or stash them before you merge." In this case, you want to keep your local work, but also make the update. Please read this `quick git tutorial <https://happygitwithr.com/pull-tricky.html>`_ on the topic.
