Advanced Topics
===============


.. _parallelism:

Parallelism
-----------

This library is primarily serial, but supports parallelism in three key ways:

1. Parallel loading of data. Snapshots are handled via :py:func:`~temet.load.snapshot.snapshotSubsetParallel`, 
   that is used by default when requesting data via the shorthand form of e.g. ``sim.gas('pos')``. To handle GIL and 
   HDF5 library issues, the approach is multiple reader (sub)processes that each read hyperslabs of HDF5 datasets 
   into a single, shared memory array. Group catalogs are handled via 
   ``illustris_python.groupcat`` (see `<http://www.github.com/illustristng/illustris_python>`__), 
   where the approach is multi-process reading and pickled aggregation.

2. Parallel processing of analysis tasks, primarily in the catalog creation routines. The approach is a custom 
   decomposition of both work and data, into a number of sub-chunks. The configuration is always controlled 
   with a parameter named ``pSplit``, that is a 2-tuple defining the current job number, and total number of jobs.
   Each "split" job can be run in series, or independently i.e. distributed across multiple nodes in an HPC environment.

3. Parallel algorithms for CPU-intensive tasks, such as spatial search acceleration structures (e.g. tree search), 
   computing distances between large sets of points (e.g. two-point correlation functions), and so on.
   The methods accelerated in this way are: :py:mod:`~temet.util.match`, :py:mod:`~temet.util.sphMap`,
   :py:mod:`~temet.util.tpcf`, :py:mod:`~temet.util.treeSearch`, and :py:mod:`~temet.util.voronoiRay`.
   in all cases, the common approach is GIL-releasing numba-JIT compiled functions and multithreading, i.e. 
   the computation is parallelized across the CPU cores of a single node.

There is no MPI-based parallelism.


Caching
-------

An instantiated :py:class:`~temet.util.simParams.simParams` object has a built-in data cache (memory backed).

Fields loaded from the group catalog are automatically cached, and subsequent "loads" will obtain data from the 
cache rather than reloading it from disk. This makes repeated plots of similar quantities, i.e. when tweaking 
plotting parameters, much faster.

Fields loaded from snapshots are not automatically cached, due to their size. However, you can manually cache them 
into the `sim.data` dictionary, by following the key name convention. Snapshot loading functions always query this 
cache before accessing data on disk. This can be an effective way to speed up repeated access/analysis, i.e. in a 
loop over (sub)halos of interest, without loading global-scope snapshot data from disk each time.

If the snapshot is changed, via ``sim.setSnap()`` or ``sim.setRedshift()``, the cache is cleared.


Units
-----

There is no automatic unit handling or conversion when loading data.

When explicitly loading datasets that exist in the actual snapshot or group catalog files, their units will be 
as in the files (typically, in the code unit system of the simulation).

There is a rather ad-hoc collection of unit conversion functions in :py:mod:`~temet.util.units` that perform 
common unit conversion tasks, e.g. converting code lengths to physical kpc, or code masses to solar masses.

When loading fields via aliases, and when loading custom fields, the resulting arrays are typically in useful, 
physical units, such as :math:`\rm{[kpc]}`, :math:`\rm{[M_\odot]}`, or :math:`\rm{[K]}`, as described by the field metadata.

Simulation instances have a :py:attr:`~temet.util.simParams.simParams.units` attribute that is an instance of
:py:class:`~temet.util.units.units`, and provides convenient access to these unit conversion functions. Simulations 
can be unit aware, such that e.g. the :py:func:`~temet.util.units.codeLengthToKpc` function returns :math:`\rm{pkpc}` 
lengths for both :math:`\rm{ckpc/h}` (e.g. TNG) and :math:`\rm{cMpc/h}` (e.g. Auriga) unit system simulations. For calculations that 
depend on cosmological scale factor/redshift, a simulation instance set to a specific snapshot/redshift will also 
have its units attribute aware of the current scale factor.

There is no symbolic unit handling, e.g. via :py:mod:`pint` or :py:mod:`unyt`. As a result, 
there is no general purpose unit conversion framework. There is also no automatic unit discovery via simulation 
metadata. The reasons are mostly legacy, as this library predates the era when simulations commonly included metadata 
and aims to support such legacy datasets. Future improvements can include more generic unit handling.


Synthetic Observations
----------------------

Functionality for creating "mock" or "synthetic" observations of certain types is included for:

* Gas absorption (i.e. column densities) for hydrogen and metal ions (using CLOUDY postprocessing).
* Gas absorption spectra (and equivalent widths) for hydrogen and metal ions, using the Voronoi mesh.
* Gas emission (i.e. line fluxes) using CLOUDY postprocessing.
* X-ray emission using APEC/AtomDB models.
* Stellar light emission, optionally including dust attenuation models and/or emission lines, using FSPS.
* Stellar light and dust emission, using SKIRT radiative transfer postprocessing.

In all cases except SKIRT, the philosophy is to use these external tools to precompute grids of relevant quantities
(e.g. ionization fractions, line emissivities, stellar SEDs) as a function of physical parameters (e.g. density, 
temperature, metallicity, age, etc.), and then interpolate these grids on-the-fly during analysis to produce the 
desired synthetic observations.
