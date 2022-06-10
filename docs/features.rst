Feature Overview
================

Here we describe the broad features to give a sense of the types of analyses which can be performed and the general functionality of the code.


Generalized Plots
-----------------

Several types of basic plots can be made, particularly for cosmological simulations. These routines are quite 
general and can plot any quantity at the particle or catalog level which is known.

For example, for general particle-level plots:

.. code-block:: python

    sP = simParams(res=1820, run='tng', redshift=0.0)
    sP2 = simParams(res=910, run='tng', redshift=0.0)

    plot.general.plotHistogram1D([sP], 'gas', 'temp')
    plot.general.plotHistogram1D([sP], 'gas', 'temp', subhaloIDs=[0,1,2,3,4])
    plot.general.plotHistogram1D([sP, sP2], 'gas', 'dens', qRestrictions=[('temp',5.0,np.inf)])

    plot.general.plotPhaseSpace2D(sP, 'gas', xQuant='numdens', yQuant='temp')
    plot.general.plotPhaseSpace2D(sP, 'gas', xQuant='numdens', yQuant='temp', meancolors=['dens','O VI frac'])

    plot.general.plotParticleMedianVsSecondQuant(sP, 'dm', xQuant='velmag', yQuant='veldisp')

    plot.general.plotStackedRadialProfiles1D([sP], subhaloIDs=[0], 'gas', ptProperty='entropy', op='median')
    plot.general.plotStackedRadialProfiles1D([sP,sP2], subhaloIDs=[[0,1,2],[1,4,5]], 'gas', ptProperty='bmag')

And for general group catalog-level plots:

.. code-block:: python

    sP = simParams(run='tng300-1', redshift=0.0)

    plot.cosmoGeneral.quantHisto2D(sP, pdf=None, yQuant='sfr2_surfdens', xQuant='mstar2_log', cenSatSelect='cen')
    plot.cosmoGeneral.quantHisto2D(sP, pdf=None, yQuant='stellarage', xQuant='mstar_30pkpc', cQuant='Krot_stars2')

    plot.cosmoGeneral.quantSlice1D([sP], pdf=None, xQuant='sfr2', yQuants=['BH_BolLum','Z_gas'], sQuant='mstar_30pkpc_log', sRange=[10.0,10.2])

    plot.cosmoGeneral.quantMedianVsSecondQuant([sP], pdf=None, yQuants=['Z_stars','ssfr'], xQuant='mhalo_200')

Note that at the bottom of `plot/general.py` and `plot/cosmoGeneral.py` there are several "driver" functions which 
show more complex examples of making these types of plots, including advanced functionality, and automatic 
generation of large sets of plots exploring all possible relationships and quantities. The configuration and 
metadata of known simulation properties can be found in `plot/quantities.py`.


Visualization
-------------

There are broadly two types of visualizations: box-based and halo-based, spanning all particle types and fields.

* :py:mod:`vis.boxDrivers` contains numerous driver functions which create different types of full box images.
* :py:mod:`vis.boxMovieDrivers` create frames for movies, including many of the available TNG movies.
* :py:mod:`vis.haloDrivers` as above, except targeted for images of individual galaxies and/or halos.
* :py:mod:`vis.haloMovieDrivers` generate frames for halo/galaxy-centric movies, including time/merger tree tracking.

All such driver functions generally set a large number of configurable options, which are then passed into the 
extremely general :py:func:`vis.box.renderBox` or :py:func:`vis.halo.renderSingleHalo` functions, which accept a large number of 
arguments (see code for details).

One could in theory call these functions directly:

.. code-block:: python

    panels = []
    panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2'} )
    panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2'} )

    commonVars = {'run':'tng', 'res':270, 'redshift':0.0}

    class plotConfig:
        pass

    vis.box.renderBox(panels, plotConfig, commonVars)

each entry in `panels` will add one image/view/panel to the figure, and these can differ in any way possible by 
setting different parameters. The `plotConfig` class holds settings which affect the overall figure as a whole, 
and `commonVars` (which is usually all local variables in the driver functions) are shared between all panels.

For a halo-based render, the process is the same, and `subhaloInd` specifies the subhalo ID:

.. code-block:: python

    run = 'tng50-3'
    redshift = 0.0

    sP = simParams(run=run, redshift=redshift)
    fof10 = sP.halo(10)
    fof11 = sP.halo(11)

    panels = []
    panels.append( {'subhaloInd':fof10['GroupFirstSub'], 'partType':'gas', 'partField':'coldens_msunkpc2'} )
    panels.append( {'subhaloInd':fof10['GroupFirstSub'], 'partType':'gas', 'partField':'temp'} )
    panels.append( {'subhaloInd':fof11['GroupFirstSub'], 'partType':'gas', 'partField':'coldens_msunkpc2'} )
    panels.append( {'subhaloInd':fof11['GroupFirstSub'], 'partType':'gas', 'partField':'temp'} )

    class plotConfig:
        plotStyle = 'edged'

    vis.halo.renderSingleHalo(panels, plotConfig, locals())


Data Catalogs
-------------

All "supplementary data catalog" types products are produced in `cosmo/auxcatalog.py`. At the bottom of this file all 
known data catalog names are listed, together with their functional definition (i.e., how to compute them, and with 
what parameters). For example, 

.. code-block:: python

    'Subhalo_StellarZ_SDSSFiber_rBandLumWt'    : \
         partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),

gives the generating function of this catalog. A large number of catalogs are produced with the `subhaloRadialReduction()`, 
`subhaloStellarPhot()`, and `subhaloRadialProfile()` functions, which are fully generalized to operate on any particle type 
and field with different statistical reductions, aperture definitions, weighting, particle restrictions, and so on.

These functions implement the `pSplit` parallelization scheme, meaning that all such computations can be chunked and partial 
subsets of the full group catalog can be operated on at once, to reduce memory usage and distribute computational cost. 
For example:

.. code-block:: python

    sP = simParams(run='tng100-1', redshift=2.0)
    for i in range(8):
        x = sP.auxCat('Subhalo_Mass_30pkpc_Stars', pSplit=[i,8])

In general, loading an auxCat which has already been created:

.. code-block:: python

    x = sP.auxCat('Subhalo_Mass_30pkpc_Stars')


Synthetic Observations
----------------------

Functionality for creating "mock" or "synthetic" observations of certain types is included for:

* Gas absorption (i.e. column densities) for hydrogen and metal ions (using CLOUDY postprocessing).
* Gas absorption spectra (and equivalent widths) for hydrogen and metal ions, using the Voronoi mesh.
* Gas emission (i.e. line fluxes) using CLOUDY postprocessing.
* X-ray emission using APEC/AtomDB models.
* Stellar light emission, optionally including dust attenuation models and/or emission lines, using FSPS.
* Stellar light and dust emission, using SKIRT radiative transfer postprocessing.


Known Simulation Families
-------------------------

A few suites of simulations (those available on the MPCDF storage systems) are 'known' and can be selected 
by name. In this case metadata and other important attributes are hardcoded in 
:mod:`simParams <tenet.util.simParams>`. Currently, the simulation families specified in this way are:

* Illustris
    * Illustris-1, Illustris-2, Illustris-3, Illustris-1-Dark, Illustris-2-Dark, Illustris-3-Dark, Illustris-2-NR, Illustris-3-NR

* TNG100
    * TNG100-1, TNG100-2, TNG100-3, TNG100-1-Dark, TNG100-2-Dark, TNG100-3-Dark

* TNG300
    * TNG300-1, TNG300-2, TNG300-3, TNG300-1-Dark, TNG300-2-Dark, TNG300-3-Dark

* TNG50
    * TNG50-1, TNG50-2, TNG50-3, TNG50-4, TNG50-1-Dark, TNG50-2-Dark, TNG50-3-Dark, TNG50-4-Dark

* TNG variations
    * L25n512_{xxxx}, L25n256_{xxxx}, L25n128_{xxxx} where `variant={xxxx}` gives the 4-digit variation number.

* EAGLE
    * Eagle100, Eagle100-Dark (rewritten versions)

* SIMBA
    * Simba100, Simba50 (rewritten versions)

* Millennium
    * Millennium-1, Millennium-2 (rewritten versions)

* TNG-Cluster
    * TNG-Cluster, TNG-Cluster-Dark

* Auriga
    * Au{h}_L{r}, Au{h}_L{r}_DM where `hInd={h}` gives the halo ID, and `res={r}` gives the resolution level.

* TNG zooms
    * TNG50_zoom, TNG100_zoom, TNG_zoom where `hInd` gives the haloID, and `res` gives the resolution.
