
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

4. Load or install python (3.6.x and 3.7.x currently tested). For example, on the MPCDF machines, using a clean anaconda

        module load anaconda/3_5.3.0
        mkdir -p ~/.local/envs
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
        export PATH=$HOME/.local/envs/myenv/bin/:$PATH

5. The FSPS stellar population synthesis package is required

        cd ~/
        git clone https://github.com/cconroy20/fsps

    edit the `src/sps_vars.f90` file and switch the defaults spectral and isochrone libraries to

        MILES 1
        PADOVA 1

    compile FSPS

        make

    add the following line to your `.bashrc` file

        export SPS_HOME=$HOME/fsps/

6. Install all python dependencies as required

        pip install -r ~/python/requirements.txt

7. Point `matplotlib` to the default settings file

        mkdir -p ~/.config/matplotlib
        ln -s ~/python/matplotlibrc ~/.config/matplotlib/

    and install the Roboto font used by default

        mkdir -p ~/.fonts/Roboto
        cd ~/.fonts/Roboto/
        wget https://github.com/google/fonts/raw/master/apache/roboto/Roboto-Light.ttf
        wget https://github.com/google/fonts/raw/master/apache/roboto/Roboto-LightItalic.ttf

8. Organize simulation directories as follows

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


generic exploratory plots
=========================

Several types of basic plots can be made, particularly for cosmological simulations. These routines are quite 
general and can plot any quantity at the particle or catalog level which is known.

For example, for general particle-level plots:

```python
>>> sP = simParams(res=1820, run='tng', redshift=0.0)
>>> sP2 = simParams(res=910, run='tng', redshift=0.0)

>>> plot.general.plotHistogram1D([sP], 'gas', 'temp')
>>> plot.general.plotHistogram1D([sP], 'gas', 'temp', subhaloIDs=[0,1,2,3,4])
>>> plot.general.plotHistogram1D([sP, sP2], 'gas', 'dens', qRestrictions=[('temp',5.0,np.inf)])

>>> plot.general.plotPhaseSpace2D(sP, 'gas', xQuant='numdens', yQuant='temp')
>>> plot.general.plotPhaseSpace2D(sP, 'gas', xQuant='numdens', yQuant='temp', meancolors=['dens','O VI frac'])

>>> plot.general.plotParticleMedianVsSecondQuant(sP, 'dm', xQuant='velmag', yQuant='veldisp')

>>> plot.general.plotStackedRadialProfiles1D([sP], subhalo=[0], 'gas', ptProperty='entropy', op='median')
>>> plot.general.plotStackedRadialProfiles1D([sP,sP2], subhalo=[[0,1,2],[1,4,5]], 'gas', ptProperty='bmag')
```

And for general group catalog-level plots:

```python
>>> sP = simParams(run='tng300-1', redshift=0.0)

>>> plot.cosmoGeneral.quantHisto2D(sP, pdf=None, yQuant='sfr2_surfdens', xQuant='mstar2_log', cenSatSelect='cen')
>>> plot.cosmoGeneral.quantHisto2D(sP, pdf=None, yQuant='stellarage', xQuant='mstar_30pkpc', cQuant='Krot_stars2')

>>> plot.cosmoGeneral.quantSlice1D([sP], pdf=None, xQuant='sfr2', yQuants=['BH_BolLum','Z_gas'], sQuant='mstar_30pkpc_log', sRange=[10.0,10.2])

>>> plot.cosmoGeneral.quantMedianVsSecondQuant([sP], pdf=None, yQuants=['Z_stars','ssfr'], xQuant='mhalo_200')
```

Note that at the bottom of `plot/general.py` and `plot/cosmoGeneral.py` there are several "driver" functions which 
show more complex examples of making these types of plots, including advanced functionality, and automatic 
generation of large sets of plots exploring all possible relationships and quantities. The configuration and 
metadata of known simulation properties can be found in `plot/quantities.py`.


visualization
=============

There are broadly two types of visualizations: box-based and halo-based, spanning all particle types and fields.

* `vis/boxDrivers.py` contains numerous driver functions which create different types of full box images.
* `vis/boxMovieDrivers.py` create frames for movies, including many of the available TNG movies.
* `vis/haloDrivers.py` as above, except targeted for images of individual galaxies and/or halos.
* `vis/haloMovieDrivers.py` generate frames for halo/galaxy-centric movies, including time/merger tree tracking.

All such driver functions generally set a large number of configurable options, which are then passed into the 
extremely general `vis.box.renderBox()` or `vis.halo.renderSingleHalo()` functions, which accept a large number of 
arguments.


data catalogs
=============

All "supplementary data catalog" types products are produced in `cosmo/auxcatalog.py`. At the bottom of this file all 
known data catalog names are listed, together with their functional definition (i.e., how to compute them, and with 
what parameters). For example, 

```python
'Subhalo_StellarZ_SDSSFiber_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),
```

gives the generating function of this catalog. A large number of catalogs are produced with the `subhaloRadialReduction()`, 
`subhaloStellarPhot()`, and `subhaloRadialProfile()` functions, which are fully generalized to operate on any particle type 
and field with different statistical reductions, aperture definitions, weighting, particle restrictions, and so on.

These functions implement the `pSplit` parallelization scheme, meaning that all such computations can be chunked and partial 
subsets of the full group catalog can be operated on at once, to reduce memory usage and distribute computational cost.


reproducing published papers
============================

The complete analysis and plot set of published papers can (theoretically) be reproduced with the following entry points 
inside the `projects/` directory. Note that in practice some analyses are costly and would better be done (and were actually 
done) by splitting into many parallel jobs on a cluster. Also note that exact reproduction may require use of an (older) code 
version, tagged on the date near the finalization of the paper.

[Nelson et al. (2018a) - TNG colors](http://arxiv.org/abs/1707.03395) - `projects.color.paperPlots()`

[Nelson et al. (2018b) - TNG oxygen](http://arxiv.org/abs/1712.00016) - `projects.oxygen.paperPlots()`

[Nelson et al. (2019b) - TNG50 outflows](http://arxiv.org/abs/1902.05554) - `projects.outflows.paperPlots()`
