
python
======

Code repository for Dylan Nelson.

* cosmo/ - analysis specific for cosmological/comoving boxes
* data/ - external text data files from the literature
* ICs/ - misc idealized initial condition generation
* load/ - loading of group catalogs, snapshots, auxcats
* obs/ - analysis and data reduction for observations
* plot/ - generalized plotting
* projects/ - routines, both analysis and plotting, specific to papers
* tables/ - large pre-computed datafiles (empty until downloaded)
* tracer/ - Monte Carlo tracer particles
* util/ - misc helper utilities
* vis/ - visualization


installation
============

1. Clone the repository into your home directory, here into a `python` directory

        cd ~
        git clone git@github.com:dnelson86/python.git

2. Clone the public Illustris data analysis python scripts

        mkdir ~/illustris_release
        cd ~/illustris_release
        git clone git@github.com:illustristng/illustris_python.git

3. Make sure both are set in the `$PYTHONPATH` environment variable, and set the `$PYTHONSTARTUP`.
For example, add the following lines to your `.bashrc` file

        export PYTHONPATH+=:$HOME/python/:$HOME/illustris_release/
        export PYTHONSTARTUP=$HOME/python/.startup.py

4. Load or install python (3.6+, 3.8.x recommended). For example, on the MPCDF machines, using a clean anaconda

        module load anaconda/3/2019.03
        mkdir -p ~/.local/envs
        conda create --prefix=~/.local/envs/myenv python=3.8
        source activate ~/.local/envs/myenv

    and add the following lines to your `.bashrc` file for permanence

        module load intel/19.0.5
        module load impi/2019.5
        module load fftw/3.3.8
        module load hdf5-serial/intel-18.0/1.8.21
        module load gsl/2.4

        module load anaconda/3/2019.03
        source activate ~/.local/envs/myenv
        export PATH=$HOME/.local/envs/myenv/bin/:$PATH

5. The FSPS stellar population synthesis package is required to generate new SPS tables

        mkdir ~/code
        cd ~/code/
        git clone https://github.com/cconroy20/fsps

    edit the `src/sps_vars.f90` file and switch the defaults spectral and isochrone libraries to

        MILES 1
        PADOVA 1 (and so MIST 0)

    edit `src/Makefile` and make sure the F90FLAGS line contains `-fPIC`, then compile FSPS

        make

    add the following line to your `.bashrc` file

        export SPS_HOME=$HOME/code/fsps/

6. Install all python dependencies as required

        pip install --user -r ~/python/requirements.txt

7. Point `matplotlib` to the default settings file

        mkdir -p ~/.config/matplotlib
        ln -s ~/python/matplotlibrc ~/.config/matplotlib/

    and install the Roboto font used by default

        mkdir -p ~/.fonts/Roboto
        cd ~/.fonts/Roboto/
        wget https://github.com/google/fonts/raw/master/apache/roboto/static/Roboto-Light.ttf
        wget https://github.com/google/fonts/raw/master/apache/roboto/static/Roboto-LightItalic.ttf

8. Several large tabulated data files are used to compute e.g. stellar luminosities (from FSPS), ion abundances and emissivities (from CLOUDY), and x-ray emission (from XPSEC). For convenience these can be downloaded as

        cd ~/python/tables/
        wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" -e robots=off www.tng-project.org/files/dnelson_tables/

9. Organize simulation directories as follows

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

In addition to shorthand names for fields such as "pos" (mapping to "Coordinates"), many custom fields at 
both the particle and group catalog level are defined. Loading data can also be done with shorthands, for example

```python
>>> sP = simParams(run='tng50-1', redshift=0.0)

>>> subs = sP.subhalos('mstar_30pkpc')
>>> x = sP.gas('cellsize_kpc')

>>> fof10 = sP.halo(10) # all fields
>>> sub_sat1 = sP.subhalo( fof10['GroupFirstSub']+1 ) # all fields
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

>>> plot.general.plotStackedRadialProfiles1D([sP], subhaloIDs=[0], 'gas', ptProperty='entropy', op='median')
>>> plot.general.plotStackedRadialProfiles1D([sP,sP2], subhaloIDs=[[0,1,2],[1,4,5]], 'gas', ptProperty='bmag')
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
arguments (see code for details).

One could in theory call these functions directly:

```python
>>> panels = []
>>> panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2'} )
>>> panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2'} )
>>>
>>> commonVars = {'run':'tng', 'res':270, 'redshift':0.0}
>>>
>>> class plotConfig:
>>>     pass
>>>
>>> vis.box.renderBox(panels, plotConfig, commonVars)
```

each entry in `panels` will add one image/view/panel to the figure, and these can differ in any way possible by 
setting different parameters. The `plotConfig` class holds settings which affect the overall figure as a whole, 
and `commonVars` (which is usually all local variables in the driver functions) are shared between all panels.

For a halo-based render, the process is the same, and `hInd` specifies the subhalo ID:

```python
>>> run = 'tng50-3'
>>> redshift = 0.0
>>>
>>> sP = simParams(run=run, redshift=redshift)
>>> fof10 = sP.halo(10)
>>> fof11 = sP.halo(11)
>>>
>>> panels = []
>>> panels.append( {'hInd':fof10['GroupFirstSub'], 'partType':'gas', 'partField':'coldens_msunkpc2'} )
>>> panels.append( {'hInd':fof10['GroupFirstSub'], 'partType':'gas', 'partField':'temp'} )
>>> panels.append( {'hInd':fof11['GroupFirstSub'], 'partType':'gas', 'partField':'coldens_msunkpc2'} )
>>> panels.append( {'hInd':fof11['GroupFirstSub'], 'partType':'gas', 'partField':'temp'} )
>>>
>>> class plotConfig:
>>>     plotStyle = 'edged'
>>>
>>> vis.halo.renderSingleHalo(panels, plotConfig, locals())
```


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
For example:

```python
>>> sP = simParams(run='tng100-1', redshift=2.0)
>>> for i in range(8):
>>>     x = sP.auxCat('Subhalo_Mass_30pkpc_Stars', pSplit=[i,8])
```

In general, loading an auxCat which has already been created:

```python
>>> x = sP.auxCat('Subhalo_Mass_30pkpc_Stars')
```


reproducing published papers
============================

The complete analysis and plot set of published papers can (theoretically) be reproduced with the following entry points 
inside the `projects/` directory. Note that in practice some analyses are costly and would better be done (and were actually 
done) by splitting into many parallel jobs on a cluster. Also note that exact reproduction may require use of an (older) code 
version, tagged on the date near the finalization of the paper.

[Nelson et al. (2018a) - TNG colors](http://arxiv.org/abs/1707.03395) - `projects.color.paperPlots()`

[Nelson et al. (2018b) - TNG oxygen](http://arxiv.org/abs/1712.00016) - `projects.oxygen.paperPlots()`

[Nelson et al. (2019b) - TNG50 outflows](http://arxiv.org/abs/1902.05554) - `projects.outflows.paperPlots()`

[Nelson et al. (2020) - TNG50 small-scale CGM](http://arxiv.org/abs/2005.09654) - `projects.lrg.paperPlots()`

[Nelson et al. (in prep) - TNG50 MgII emission](#) - `projects.mg2emission.paperPlots()`
