Catalogs
========

.. contents::
    :local:

The creation of "catalogs" of values for halos and subhalos (i.e. galaxies) is a common analysis task.
As these can be extensive calculations, and the results can be generally of interest, these are stored as on-disk 
catalog files. We typically refer to such results as "post-processing catalogs", "supplementary catalogs", or 
"auxiliary catalogs".

Several general-purpose functions are provided to facilitate the calculation of such catalogs, primarily based on 
statistical reduction operations applied to the member particles/cells of subhalos and/or their parent halos. 
Other more specialized functions e.g. compute radial profiles, assess spatial distributions for environmental 
metrics, measure statistics over full boxes, derive properties based on time evolution, and so on.

These catalogs can then be combined with other previously computed 
values to understand correlations and relationships between galaxy and halo properties in general.

.. warning::

    In general, a catalog value of ``np.nan`` is specifically used to indicate that no value was computed 
    for a given subhalo.


Known Catalogs
--------------

A large number of default catalog names are listed in :py:mod:`temet.load.auxcat_fields`, together with their 
definition (i.e., the function that computes each, and with what parameters). For example::

    'Subhalo_StellarZ_SDSSFiber_rBandLumWt' : \
         partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),

specifies that this quantity -- mean stellar metallicity computed from a SDSS fiber-like aperture, weighting stars by 
their r-band luminosity -- defines the catalog.

Many catalogs are (can be) produced by generalized analysis routines, including ``subhaloRadialReduction()``, 
``subhaloRadialProfile()``, ``subhaloCatNeighborQuant()``, and so on. 
See :ref:`Analysis Capabilities <catalogs_analysis>` for details.
These functions are highly versatile, and can compute a wide variety of catalogs by performing calculations on 
different particle types and fields, subject to arbitrary additional restrictions, weighting, and so on.


Defining New Catalogs
---------------------

In order to create a new catalog, you simply need to give it a name, and attach to that name the exact 
definition of the value to compute. This definition is the function which generates the values, 
together with any required arguments for that function. For example::

   'Subhalo_Mass_30pkpc_Stars' = 
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0),

is the definition of the ``Subhalo_Mass_30pkpc_Stars`` catalog, a commonly used measurement for the 
stellar mass of a galaxy. To create this catalog, the :py:func:`~temet.catalog.subhalo.subhaloRadialReduction` 
function is called. This is a versatile analysis function which applies a particular 'reduction operation' 
(i.e. mathematical/statistical method) to a particular field of the particles/cells which 
belong to a subhalo. Its behavior is controlled by setting the following options:

* ``ptType = 'stars'`` specifies that we will process stellar particles.
* ``ptProperty = 'Masses'`` specifies that we will operate on the ``Masses`` snapshot field. Any known 
  :ref:`physical quantity <quantities>` of a particle/cell can be used.
* ``op = 'sum'`` specifies that we will take the sum (total) of this field.
* ``rad = 30.0`` requests an aperture restriction, such that only particles within a 3D radial distance of 
  30 pkpc are included.

Let's take a second, more complex, example::

   'Subhalo_CoolingTime_OVI_HaloGas' =
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',weighting='O VI mass',
             rad='r015_1rvir_halo',ptRestrictions={'StarFormationRate':['eq',0.0]}),

Here we use the same function to compute the mean cooling time for every gas cell, weighted by the mass of 
the OVI ion in that cell, restricted to a radial shell of :math:`0.15 < r/r_{\rm vir} < 1.0` (representing 
the "halo", i.e. excluding the central galaxy), and further restricted to cells with absolutely zero 
``StarFormationRate``.

Let's look at a third, final example, based on a different generating function::

   'Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVel_sfrWt' =
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel_abs',op='mean',weighting='sfr',
             scope='fof',proj2D=['edge-on',None]),

In this case we compute, for every subhalo, a radial profile of the mean line-of-sight velocity (absolute 
value) of gas, weighted by the cell SFR. In this calculation all cells inside the parent FoF halo 
are included. Further, it is carried out in 2D projection, with the specific viewing direction defined 
by rotating the galaxy into its edge-on orientation.

.. note::

  These three examples are all entries in a python dict. The catalog name, as a string, is the dictionary 
  key, and a ``partial(func, args)`` is the value. 

  If you aren't familiar with :py:func:`functools.partial`, this creates a new function which "wraps" the 
  original ``func`` with one or more arguments specified. In this case, we always specify all analysis arguments, 
  leaving only the :py:class:`~temet.util.simParams.simParams` instance to be passed in when the catalog is 
  requested/generated.

To **define a new catalog**, you add an entry to the :py:data:`temet.load.auxcat_fields.def_fields` 
dictionary::

  from temet.load.auxcat_fields import def_fields

  def_fields['my_catalog_name'] = partial(my_catalog_function, arg1=val1, arg2=val2, ...)

The easiest way is to follow existing examples.


Generating and Loading Catalogs
-------------------------------

Once you have added a new entry to the auxCat dictionary, you can request this catalog for a given run::

    sim = simParams(run='tng100-2', redshift=0.0)
    data, meta = sim.auxCat('MyCatalogName')

The first time we request this catalog, it will be computed and automatically saved to disk::

    Compute and save: [TNG100-2] [MyCatalogName]
    Quantity [X] enclosed within a radius of [Y] for [Z]. (selection function details).
    Total # Subhalos: 118820, [118820] in selection, processing [118820] subhalos now...
        0.0%
        10.0%
        20.0%
        30.0%
        40.0%
        50.0%
        60.0%
        70.0%
        80.0%
        90.0%
    Saved new [/home/user/sims.TNG/L75n910TNG/data.files/auxCat/MyCatalogName.hdf5].

All methods to generate catalogs should support ``pSplit``-based :ref:`parallelism`. For example, you could launch 
eight independet jobs, each executing::

    sim = simParams(run='tng100-2', redshift=0.0)
    x = sim.auxCat('MyCatalogName', pSplit=[i,8])

After the final job completes, the eight intermediate output files can be combined into the final catalog::

    data, meta = sim.auxCat('MyCatalogName', pSplit=[0,8])

In some cases you may wish to compute a catalog across several chunks in this way, usually in order to
reduce the peak memory usage by avoiding loads of entire snapshots at once. In this case, if it still makes sense 
to run these split calcluations one after another on a single machine, the following shorthand can be used::

    sim = simParams(run='tng100-2', redshift=0.0)
    data, meta = sim.auxCatSplit('MyCatalogName')


Using Catalog Values in General Plots
-------------------------------------

After generating a catalog, you can directly load these values and use them for analysis, as above. On the 
other hand, you can also integrate them in the generalized plotting framework, to more easily explore 
correlations with other existing quantities.

To do so, you should add a new auxCat entry (see :ref:`custom_group_quantities`, following existing examples.
You can also provide metadata: a label, units, and default bounds, to be used in plots.


.. note::

  You can also add the name to :py:func:`temet.plot.quantities.quantList` (optional), which returns a list of 
  all "known" subhalo propertie.

  The return of this function with ``alwaysAvail == True`` defines the list of properties available for 
  interactive exploration on the 
  `IllustrisTNG Data Release Group Catalog Plot Utility <https://www.tng-project.org/data/groupcat/>`_.


After you have made this addition, you can then load the corresponding data as::

    x = sim.subhalos('my_new_field')

And request it on any of the general plot routines, for instance::

    temet.plot.subhalos.median(sim, 'my_new_field', 'mstar_30pkpc')


.. _catalogs_analysis:

Analysis Capabilities
---------------------

The following functions currently exist to compute catalogs of these types. First, to compute one or more 
values per halo or subhalo, based on the particles/cells which belong to each halo or subhalo:

* :py:func:`~temet.catalog.group.fofRadialSumType` - sum a property of a given particle type, within a 
  particular aperture, for all FoF halos.
* :py:func:`~temet.catalog.subhalo.subhaloRadialReduction` - compute an arbitrary summary statistic on any 
  particle/cell quantity, for all subhalos.
* :py:func:`~temet.catalog.subhalo.subhaloStellarPhot` - compute photometry, half-light radii, and similar 
  properties based on stellar light mocks, for all subhalos.
* :py:func:`~temet.catalog.profile.subhaloRadialProfile` - compute radial profiles, for any particle/cell 
  quantity, for all subhalos.

To compute a value per subhalo, based on past history:

* :py:func:`~temet.catalog.temporal.mergerTreeQuant` - compute a property for each subhalo based on its merger 
  tree, for instance halo formation time, number of mergers, and so on.
* :py:func:`~temet.catalog.temporal.tracerTracksQuant` - compute a property for each subhalo based on the 
  Lagrangian histories of its member tracer particles, for instance the average entropy of smoothly accreted 
  gas at the time of its accretion.

To compute a value per subhalo, based on other group catalog objects:

* :py:func:`~temet.catalog.subhalo.subhaloCatNeighborQuant` - compute a property for each subhalo, based on the 
  other subhalos in the catalog, for instance the number of neighbors, and other environmental densities.

To compute a statistic or value for an entire simulation as a whole:

* :py:func:`~temet.catalog.box.wholeBoxCDDF` - compute the column density distribution function for a given 
  gas species (e.g. HI, H2, OVI, and so on) by projecting the entire simulation onto a very large uniform 
  2D grid.

To compute quantities based on images/maps/2D projections of physical quantities:

* :py:func:`~temet.catalog.maps.projections`
* :py:func:`~temet.catalog.maps.summarize_projection_2d`

More specialized routines include:

* :py:func:`~temet.catalog.gasflows.instantaneousMassFluxes` - compute the radial mass, energy, or 
  momentum flux rates (outflowing/inflowing).
* :py:func:`~temet.catalog.gasflows.radialMassFluxes` - compute high dimensional histograms of these fluxes as a 
  function of (rad, vrad, dens, temp, metallicity), together with marginalized 1D and 2D histograms over 
  these properties.
* :py:func:`~temet.catalog.gasflows.massLoadingsSN` - compute the mass loading factor, energy loading 
  factor, or momentum loading factors.
* :py:func:`~temet.catalog.gasflows.outflowVelocities` - compute an 'outflow velocity', which is a 
  summary statistic derived from the outwards radial mass fluxes.

* :py:func:`~temet.projects.rshock.healpixThresholdedRadius` - compute characteristic radii based on spherical 
  healpix sampling of particles/cells around halos, for instance the virial shock radius or splashback radius.


Common Options
--------------

Many of these have a number of common options which are useful to describe in some detail, in case they 
are relevant for a specific catalog:

:ptType: the particle type, should be one of ``gas``, ``bh``, ``stars``, or ``bhs``.

:ptProperty: the particle quantity/field, can be any known :ref:`physical property <quantities>`.

:op: a mathematical operation, typically can be one of e.g. ``min``, ``max``, ``sum``, ``mean``, 
  ``median``, and so on. Often can be any python function, e.g. :py:func:`numpy.std` or even a function 
  you define yourself. Can also frequently be a "custom user-defined function" defined by a string. 
  These are mostly handled inside :py:func:`temet.catalog.subhalo._process_custom_func`, some examples being 
  ``Krot`` (the :math:`\kappa_{\rm \star,rot}` measurement of stellar rotational support) or ``halfrad`` 
  (the radius enclosing half of the total of the given quantity). Additional custom operations can be 
  added to this function.

:rad: a radial restriction. If not specified, all particles/cells in the given scope are included. If a 
  number, should be input in **pkpc**. Otherwise can be a string understood by 
  :py:func:`temet.catalog.common._radialRestriction`, for instance ``r015_1rvir_halo``, ``rvir``, 
  ``2rhalfstars``, ``sdss_fiber``, ``legac_slit``, and so on. Additional specialized radial restrictions 
  can be added to this function.

:ptRestrictions: one or more particle quantity restrictions to apply. If specified, should be a 
  dictionary, where each key is a quantity name, and each value is a 2-tuple specifying an 
  ``[inequality,value]`` pair to apply. The ``inequality`` is a string, one of ``'gt','lt','eq'``, and 
  the ``value`` is the numeric bound. For example, ``{'StarFormationRate':['gt':0.0]}`` includes 
  **only** gas cells with positive star formation rates.

:weighting: any particle quantity/field, by which to weight the particles/cells, for the given operation.

:scope: defines **which** particles/cells should be included. The typical options are ``subhalo`` 
  (default, include all gravitationally bound members of the subhalo, e.g. in the case of centrals this 
  excludes both satellites and inner FoF fuzz), ``fof`` (instead include all members of the parent fof 
  halo, e.g. this means that centrals and satellites of the same halo compute over the same particles, 
  modulo any radial restriction), or ``global`` (include all particles/cells in the entire snapshot, 
  important e.g. for radial profiles extending to large distance). Note that the last option may come 
  with significant performance penalties.

The following options restrict the computation to a subset of all objects. For instance, if you only care 
about central galaxies, you can skip satellites, or if you only care about relatively massive halos, you 
can skip low-mass halos. The main purpose here is to save (a significant amount of) computation time when 
creating the catalog.

:minStellarMass: minimum stellar mass of the galaxy [:math:`\log M_{\rm sun}`]. A nonzero value therefore 
  excludes all dark subhalos by definition.

:minHaloMass: minimum stellar mass of the parent halo [:math:`\log M_{\rm sun}`]. This compares against 
  the ``mhalo_200_log`` quantity for subhalos, which is only defined for centrals. A nonzero value 
  therefore excludes all satellites by definition.

:cenSatSelect: one of ``cen``, ``sat``, or ``all``.


Current Catalog Definitions
---------------------------

The code base currently has the following catalogs definitions, as listed in the 
:py:mod:`temet.load.auxcat_fields` dictionary.

.. exec::

    from temet.load.auxcat_fields import def_fields

    print('.. csv-table::')
    print('    :header: "Catalog Name", "Generator Function", "Arguments"')
    print('    :widths: 10, 30, 60')
    print('')

    for key in def_fields.keys():
        func = def_fields[key] # partial
        func_name = ":py:func:`~%s.%s`" % (func.func.__module__, func.func.__name__)
        func_args = ', '.join(['``%s`` = %s' % (k,v) for k,v in func.keywords.items()])
        print('    "%s", "%s", "%s"' % (key,func_name,func_args))

    print('\n')
