Loading Data
============

.. contents::


After specifying a simulation and redshift (i.e. snapshot) of interest::

    import temet

    sim = temet.sim('tng50-1', redshift=1.0)

We can then load the corresponding data. 

Note that the snapshot/redshift of a simulation instance can be set, or changed, at any time via ``sim.setSnap()`` 
or ``sim.setRedshift()``.


Metadata
--------

The simulation instance ``sim`` has many attributes that provide metadata about the simulation, such as cosmological 
parameters, the unit system and unit conversion functions, the list of available snapshots, and so on. For example::

    sim.numPart

    >>> array([9103872834, 10077696000, 0, 10077696000, 736380469, 3797])

    sim.redshift

    >>> 0.99729425

    sim.snap

    >>> 99

    sim.numHalos

    >>> 10544757

    sim.numSubhalos

    >>> 6780233
    
Several helper functions compute cosmological quantities, such as the age of the Universe at the current redshift::

    sim.scalefac

    >>> 0.50067735

    sim.tage

    >>> 13.80273

    sim.boxSizeCubicPhysicalMpc

    >>> 17311.775

Dictionaries of the original configuration and run-time parameters used to run the simulation can be accessed as::

    sim.config

    >>> {'ADAPTIVE_HYDRO_SOFTENING': np.bytes_(b''),
         'ALLOW_DIRECT_SUMMATION': np.bytes_(b''),
         ...

    sim.params

    >>> {'AGB_MassTransferOn': np.int32(1),
         'ActivePartFracForNewDomainDecomp': np.float64(0.0025),
         'AdaptiveHydroSofteningSpacing': np.float64(1.2),
         ...

For complete documentation of all simulation attributes and methods, see :class:`~temet.util.simParams.simParams`.


Snapshots
---------

To load one or more fields of particle/cell-level data from the entire snapshot::

    gas_pos = sim.snapshotSubset('gas', 'Coordinates')
    star_masses = sim.snapshotSubset('stars', 'mass')

    dm = sim.snapshotSubset('dm', ['pos', 'vel'])

You can either specify the exact dataset names as available in a snapshot, or you can use alias/generic/custom names.
The latter are simulation independent. For example, "pos" simply loads "Coordinates" in GADGET/AREPO-type snapshots, 
while it can load a different field name in other simulation types.

You can load the particles/cells that belong only to a specific subhalo or halo by specifying the optional 
``subhaloID`` or ``haloID`` arguments, respectively::

    gas_pos_sub10 = sim.snapshotSubset('gas', 'pos', subhaloID=10)
    star_masses_halo5 = sim.snapshotSubset('stars', 'mass', haloID=5)

In addition to shorthand/generic aliases, many custom fields are defined. These calculate properties on-the-fly that 
are not directly available in the snapshot files. See :doc:`quantities` for the list of custom fields (you can 
also add your own). For example::

    x = sim.snapshotSubset('gas', 'cellsize_kpc')

The :py:func:`~temet.load.snapshot.snapshotSubset` function is serial, and uses one reader to access the filesystem. 
When data is stored on high-performance parallel filesystems, reading data is significantly faster with multiple 
readers/workers. The :py:func:`~temet.load.snapshot.snapshotSubsetParallel` function is a parallel (multi-threaded) 
version that can be much faster for large data loads::

    pos = sim.snapshotSubsetP('dm', 'pos')

Finally, you can also load data using shorthand names for particle types, for example::

    temp = sim.gas('temp')
    z = sim.stars('z_form')

These always use the parallel loader by default.


Group Catalogs
--------------

The analogous function for loading group catalog data (halo and subhalo properties) is :py:func:`~temet.load.groupcat.groupCat`. 
This accepts the arguments ``fieldsSubhalos`` and/or ``fieldsHalos``, each a list of strings of requested field names.
The ``.halos()`` and ``.subhalos()`` shortcuts of a simulation instance can always be used instead in practice::

    Group_M200c = sim.halos('Group_M_Crit200')

    subhalos = sim.subhalos(['SubhaloSFR','SubhaloMassType']) # code mass units

    mstar_logmsun = sim.units.codeMassToLogMsun(subhalos['SubhaloMassType'][:,4])

Custom fields are also defined at the group catalog level; see :doc:`quantities` for details. For example::

    mstar_30pkpc = sim.subhalos('mstar_30pkpc') # msun

.. note::

  When loading snapshot or group catalog fields by their actual names, data is loaded and returned as is.
  In particular, the units of the resulting arrays are as in the files
  (typically, in the code unit system of the simulation).

  In contrast, custom fields typically return arrays in useful, physical units, 
  such as :math:`\rm{kpc}`, :math:`\rm{M_\odot}`, or :math:`\rm{K}`. 
  
  All returns are linear by default, but can be automatically converted to log10 by appending  ``_log`` to the 
  field name, e.g. ``'mstar_30pkpc_log'``.

In addition, you can load all of the available fields for a particular halo or subhalo by its ID::

    fof = sim.halo(10)

    central_subhalo = sim.subhalo(fof['GroupFirstSub']+1)
    first_satellite_subhalo = sim.subhalo(fof['GroupFirstSub']+1)


Merger Trees
------------

The `merger tree <https://www.tng-project.org/data/docs/specifications/#sec4>`__ data structure can be accessed 
via the :py:func:`~temet.load.mergertree.mergerTree` function. 
For example, to load the full main progenitor branch (MPB) of a given subhalo::

    subID = 100
    mpb = sim.loadMPB(subID)

    print(mpb['SnapNum'], mpb['SubfindID'], mpb['SubhaloMass'])

This loads all available fields. You can also load only a subset of fields by specifying the optional ``fields`` 
argument, which accepts a list of strings of requested field names::

    mpb = sim.loadMPB(subID, fields=['SnapNum','SubfindID','SubhaloMass','SubhaloSFR'])

Analogously, the main descendant branch (MDB) that goes forward in time towards :math:`z=0` can be loaded with::

    mdb = sim.loadMDB(subID)

For loading a large number of merger trees at once, the :py:func:`~temet.load.mergertree.loadMPBs` function 
accepts a list of subhalo IDs and returns a dictionary of arrays. It is significantly faster than calling
:py:func:`~temet.load.mergertree.loadMPB` multiple times.

.. note::

  In all cases, the ``treeName`` parameter can be used to request a specific merger tree (it must have already been 
  computed for the simulation).
  
  By default, the `SubLink <https://github.com/nelson-group/sublink>`__ trees are used.

To access the complete trees, including all progenitors and descendants, use the 
:py:func:`illustris_python.sublink.loadTree` function directly::

    tree = sim.loadTree(subID)

Finally, a helper function to return specific (possibly custom) quantities along the MPB of a subhalo, optionally 
(i) interpolating over snapshots where the progenitor is missing from the tree, and (ii) smoothing the result in time, 
is provided with :py:func:`~temet.cosmo.mergertree.quantMPB`. It can be accessed as::

    mhalo = sim.quantMPB(subID, 'mhalo_200', add_ghosts=True, smooth=True)
    pos = sim.quantMPB(subID, 'SubhaloPos', add_ghosts=True, smooth=False)


Postprocessing and Auxiliary Catalogs
-------------------------------------

Pre-existing postprocessing catalogs, or user-created aux catalogs, can both be loaded manually e.g. with :py:mod:`h5py`.

However, they are typically accessed by creating (new) custom fields, at either the particle/cell or group catalog level, 
(see :doc:`quantities` for details). The defining function handles loading and processing. The resulting 
fields can then be loaded, plotted, or visualized.


Known Simulation Families
-------------------------

A few suites of simulations (those available on the MPCDF storage systems) are 'known' and can be selected 
by name. In this case metadata and other important attributes are hardcoded in 
:mod:`simParams <temet.util.simParams>`. Currently, the simulation families specified in this way are:

* Illustris
    * Illustris-1, Illustris-2, Illustris-3, Illustris-1-Dark, Illustris-2-Dark, Illustris-3-Dark, Illustris-2-NR, Illustris-3-NR

* TNG100
    * TNG100-1, TNG100-2, TNG100-3, TNG100-1-Dark, TNG100-2-Dark, TNG100-3-Dark

* TNG300
    * TNG300-1, TNG300-2, TNG300-3, TNG300-1-Dark, TNG300-2-Dark, TNG300-3-Dark

* TNG50
    * TNG50-1, TNG50-2, TNG50-3, TNG50-4, TNG50-1-Dark, TNG50-2-Dark, TNG50-3-Dark, TNG50-4-Dark

* TNG variations
    * L25n512_{xxxx}, L25n256_{xxxx}, L25n128_{xxxx} where ``variant={xxxx}`` gives the 4-digit variation number.

* EAGLE
    * Eagle100, Eagle100-Dark (rewritten versions)

* SIMBA
    * Simba100, Simba50 (rewritten versions)

* Millennium
    * Millennium-1, Millennium-2 (rewritten versions)

* TNG-Cluster
    * TNG-Cluster, TNG-Cluster-Dark

* Auriga
    * Au{h}_L{r}, Au{h}_L{r}_DM where ``hInd={h}`` gives the halo ID, and ``res={r}`` gives the resolution level.

* TNG zooms
    * TNG50_zoom, TNG100_zoom, TNG_zoom where ``hInd`` gives the haloID, and ``res`` gives the resolution.

* cosmosTNG
    * TNG model and DMO versions of the eight ``variant={A,B,C,D,E,F,G,H}`` at ``res={1,2,3}``.

* GIBLE
    * RF8, RF64, RF512, RF4096 where ``hInd`` gives the haloID, and ``res`` gives the resolution.
