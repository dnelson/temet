
python
======

Code repository for Dylan Nelson.

* ArepoVTK/ - helper utilities for ArepoVTK vis
* cosmo/ - analysis specific for cosmological/comoving boxes
* data/ - external text data files from the literature
* ICs/ - misc idealized initial condition generation
* obs/ - analysis and data reduction for observations
* plot/ - plotting
* projects/ - routines, both analysis and plotting, specific to papers
* tracer/ - Monte Carlo tracer particles
* util/ - misc helper utilities
* vis/ - visualization


installation
============

1. Clone the repository into your home directory, here into a `python` directory

        cd ~
        hg clone ssh://hg@bitbucket.org/dnelson86/python

2. Clone the public Illustris data analysis python scripts

        mkdir ~/illustris_release
        cd ~/illustris_release
        hg clone ssh://hg@bitbucket.org/illustris/illustris_python

3. Make sure both are set in the `$PYTHONPATH` environment variable, and set the `$PYTHONSTARTUP`.
For example, add the following lines to your `.bashrc` file

        export PYTHONPATH+=:$HOME/python/:$HOME/illustris_release/
        export PYTHONSTARTUP=$HOME/python/.startup.py

4. Load or install python (3.7.x currently tested). For example, on the MPCDF machines, using a clean anaconda

        module load anaconda/3_5.3.0
        cd ~
        mkdir -p .local/envs
        conda create --prefix=~/.local/envs/myenv python=3.7
        source activate ~/.local/envs/myenv

    and add the following lines to your `.bashrc` file for permanence

        module load intel/18.0
        module load impi/2018.4
        module load fftw/3.3.6
        module load hdf5-serial/intel-18.0/1.8.18
        module load gsl/2.2

        module load anaconda/3_5.3.0
        source activate ~/.local/envs/myenv
        export PATH=~/.local/envs/myenv/bin/:$PATH

5. Install all python dependencies as required

        pip install -r ~/python/requirements.txt

6. Point `matplotlib` to the default settings file

        mkdir -p ~/.config/matplotlib
        ln -s ~/python/matplotlibrc ~/.config/matplotlib/

    and install the Roboto font used by default

        mkdir -p ~/.fonts/Roboto
        cd ~/.fonts/Roboto/
        wget https://github.com/google/fonts/raw/master/apache/roboto/Roboto-Light.ttf
        wget https://github.com/google/fonts/raw/master/apache/roboto/Roboto-LightItalic.ttf

7. Organize simulation directories as follows

        mkdir ~/sims.TNG
        mkdir ~/sims.TNG/L75n1820TNG
        mkdir ~/sims.TNG/L75n1820TNG/data.files
        cd ~/sims.TNG/L75n1820TNG/
        ln -s /virgo/simulations/IllustrisTNG/L75n1820TNG/output .
        ln -s /virgo/simulations/IllustrisTNG/L75n1820TNG/postprocessing .

    note that the last two lines create symlinks to the actual output directory where the simulation data files 
    (`groupcat_*` and `snapdir_*`) reside, as well as to the postprocessing directory (containing `trees`, etc).
    Replace as needed with the actual path on your machine.


getting started
===============

Most analysis is based around a "simulation parameters" object (typically called `sP`), which specifies the 
simulation and snapshot of interest, among other details.

For example, to load some data from the group catalog and snapshot of TNG100-2 at z=2

```python
ipython

>>> sP = simParams(res=910, run='tng', redshift=2.0)

>>> subs = sP.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos'])
>>> sub_masses_logmsun = sP.units.codeMassToLogMsun( subs['SubhaloMass'] )

>>> gas_pos = sP.snapshotSubset('gas', 'pos')
>>> dm_vel_sub10 = sP.snapshotSubset('dm', 'vel', subhaloID=10)
```
