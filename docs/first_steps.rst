First Steps Walkthrough
=======================

First, start a command-line ``ipython`` session, or a Jupyter notebook. Then import the library

.. code-block:: python

    import temet

Simulation Selection
--------------------

Most analysis is based around a "simulation parameters" object (typically called ``sim`` below), 
which specifies the simulation and snapshot of interest, among other details.  You can then select a simulation and snapshot from a known list, e.g. the TNG100-2 simulation of IllustrisTNG at redshift two:

.. code-block:: python

    sim = temet.sim(res=910, run='tng', redshift=2.0)

Or the TNG50-1 simulation at redshift zero, by its short-hand name

.. code-block:: python

    sim = temet.sim(run='tng50-1', redshift=0.0)

Note that if you would like to select a simulation which is not in the pre-defined list of known simulations, 
i.e. a simulation you have run yourself, then you can specify it simply by path

.. code-block:: python

    sim = temet.sim('/home/user/sims/sim_run/', redshift=0.0)

In all cases, the redshift or snapshot number is optionally used to pre-select the particular snapshot of 
interest. You can also specify the overall simulation without this, for example

.. code-block:: python

    sim = temet.sim('tng300-1')


Loading Data
------------

Once a simulation and snapshot is selected you can load the corresponding data. For example, to load one or more 
particular fields from the group catalogs

.. code-block:: python

    subs = sim.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos'])
    sub_masses_logmsun = sim.units.codeMassToLogMsun( subs['SubhaloMass'] )

To load particle-level data from the snapshot itself

.. code-block:: python

    gas_pos = sim.snapshotSubset('gas', 'pos')
    star_masses = sim.snapshotSubsetP('stars', 'mass')
    dm_vel_sub10 = sim.snapshotSubset('dm', 'vel', subhaloID=10)

In addition to shorthand names for fields such as "pos" (mapping to "Coordinates"), many custom fields 
at both the particle and group catalog level are defined. Note that ``snapshotSubsetP()`` is the 
parallel (multi-threaded) version, and will be significantly faster. Loading data can also be done with 
shorthands, for example

.. code-block:: python

    subs = sim.subhalos('mstar_30pkpc')
    x = sim.gas('cellsize_kpc')

    fof10 = sim.halo(10) # all fields
    sub_sat1 = sim.subhalo( fof10['GroupFirstSub']+1 ) # all fields


Exploratory Plots for Galaxies
------------------------------

The various plotting functions in :py:mod:`plot.cosmoGeneral <temet.plot.cosmoGeneral>` are designed to 
be as general and automatic as possible. They are idea for a quick look or for exploring trends in the 
objects of the group catalogs, i.e. galaxies (subhalos).

Let's examine a classic observed galaxy scaling relation: the correlation between gas-phase metallicity, 
and stellar mass, the "mass-metallicity relation" (MZR).

.. code-block:: python

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.median(sim, 'Z_gas', 'mstar_30pkpc')

This produces a PDF figure named ``medianQuants_TNG100-1_Z_gas_mstar_30pkpc_cen.pdf`` in the current working 
directory. It shows the mass-metallicity relation of TNG100 galaxies at :math:`z=0`, and looks like this:

.. image:: _static/first_steps_medianQuants_1.png

We can enrich the plot in a number of ways, both by tweaking minor aesthetic options, and by including 
additional information from the simulation. For example, we will shift the x-axis bounds, and also 
include individual subhalos as colored points, coloring based on gas fraction::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.median(sim, 'Z_gas', 'mstar_30pkpc', 
      xlim=[8.0, 11.5], scatterColor='fgas2')

This produces the following figure, which highlights how lower mass galaxies have high gas fractions of 
nearly unity, i.e. :math:`M_{\rm gas} \gg M_\star`, and that gas fraction slowly decreasing with stellar 
mass until :math:`M_\star \sim 10^{10.5} M_\odot`. At this point, the overall gas metallicity turns over 
and starts to decrease, as indicated by the black median line. Gas fractions also drop rapidly, reaching 
:math:`f_{\rm gas} \sim 10^{-4}` before starting to slowly rise again. This feature marks the onset of 
galaxy quenching due to supermassive black hole feedback.

.. image:: _static/first_steps_medianQuants_2.png

Once you add a custom calculation for a new property of subhalos, i.e. compute a value which isn't available 
by default, you can use the same plotting routines to understand how it varies across the galaxy population, 
and correlates with other galaxy properties.


Exploratory Plots for Snapshots
-------------------------------

Similarly, :py:mod:`plot.general <temet.plot.general>` provides general plotting routines focused on snapshots, 
i.e. particle-level data. These are also then suitable for non-cosmological simulations.

Functionality includes 1D histograms, 2D distributions, median relations, and radial profiles.

For example, we could plot the traditional 2D "phase diagram" of density versus temperature. However, we can 
also use any (known) quantity on either axis. Furthermore, while color can represent the distribution of 
mass, it can also be used to show the value of a third particle/cell property, in each pixel. Let's look at 
the relationship between gas pressure and magnetic field strength at :math:`z=0`::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='pres', yQuant='bmag')

.. image:: _static/first_steps_phase2D_1.png

For cosmological simulations, we can also look at particle/cell properties for one or more (stacked) halos. 
For example, the relationship between (halocentric) radial velocity and (halocentric) distance, for all dark 
matter particles within the tenth most massive halo of TNG50-1 at :math:`z=2`::

    sim = temet.sim(run='tng50-1', redshift=2.0)
    haloIDs = [9]

    opts = {'xlim':[-0.6,0.3], 'ylim':[-800,600], 'clim':[-4.7,-2.3], 'ctName':'inferno'}
    temet.plot.snapshot.phaseSpace2d(sim, 'dm', xQuant='rad_rvir', yQuant='vrad', haloIDs=haloIDs, **opts)

.. image:: _static/first_steps_phase2D_2.png

Here we see individual gravitationally bound substructures (subhalos) within the halo as bright vertical 
features.


Visualizing a Halo and its Galaxy
---------------------------------

TODO.


Computing a Custom Post-processing Catalog
------------------------------------------

So far we have been exclusively exploring and visualizing existing data -- either properties which are directly 
available in the catalogs or snapshots (e.g. galaxy stellar mass, gas cell magnetic field strength), or which 
can be easily derived from them (e.g. dark matter particle radial velocity with respect to its parent halo).

Instead, we may be interested to compute a new physical quantity of interest for each object in the catalog.

We typically refer to such results as "post-processing catalogs", "supplementary catalogs", or "auxiliary catalogs".

TODO.
