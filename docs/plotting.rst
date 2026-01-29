Plotting
========

.. contents::
    :local:

Several types of generic plots can be made, particularly for cosmological simulations. These routines are quite 
general and can plot any quantity at the particle or catalog level which is known.

There are two general types of plots which are broadly useful for exploration:

* those based on (group) catalog values, i.e. for relations in the halo/galaxy population
* those based on (particle/cell) snapshot values, i.e. for relations between gas/stellar/DM properties

Custom (i.e. user defined) fields can be used within these plot routines.


Overview
---------

Examples of snapshot-level plots::

    sim = temet.sim(res=1820, run='tng', redshift=0.0)
    sim2 = temet.sim(res=910, run='tng', redshift=0.0)

    plot.snapshot.histogram1d([sim], 'gas', 'temp')
    plot.snapshot.histogram1d([sim], 'gas', 'temp', subhaloIDs=[0,1,2,3,4])
    plot.snapshot.histogram1d([sim, sim2], 'gas', 'dens', qRestrictions=[('temp',5.0,np.inf)])

    plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='numdens', yQuant='temp')
    plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='numdens', yQuant='temp', meancolors=['dens','O VI frac'])

    plot.snapshot.median(sim, 'dm', xQuant='velmag', yQuant='veldisp')

    plot.snapshot.profilesStacked1d([sim], subhaloIDs=[0], 'gas', ptProperty='entropy', op='median')
    plot.snapshot.profilesStacked1d([sim,sim2], subhaloIDs=[[0,1,2],[1,4,5]], 'gas', ptProperty='bmag')

Examples of group catalog-level plots::

    sim = temet.sim(run='tng300-1', redshift=0.0)

    plot.subhalos.histogram2d(sim, pdf=None, yQuant='sfr2_surfdens', xQuant='mstar2_log', cenSatSelect='cen')
    plot.subhalos.histogram2d(sim, pdf=None, yQuant='stellarage', xQuant='mstar_30pkpc', cQuant='Krot_stars2')

    plot.subhalos.slice1d([sim], pdf=None, xQuant='sfr2', yQuants=['BH_BolLum','Z_gas'], sQuant='mstar_30pkpc_log', sRange=[10.0,10.2])

    plot.subhalos.median([sim], pdf=None, yQuants=['Z_stars','ssfr'], xQuant='mhalo_200')

Note that :py:mod:`temet.plot.drivers` contains several "driver" functions which 
show more complex examples of making these types of plots, including advanced functionality, and automatic 
generation of large sets of plots exploring all possible relationships and quantities.


Exploratory Plots for Galaxies (Catalog Values)
-----------------------------------------------

We continue from the :doc:`first_steps` here. The plotting functions in 
:py:mod:`plot.subhalos <temet.plot.subhalos>` can be useful for exploring trends in the 
objects of the group catalogs, i.e. galaxies (subhalos).

Let's examine a classic observed galaxy scaling relation: the correlation between gas-phase metallicity, 
and stellar mass, the "mass-metallicity relation" (MZR)::

    sim = temet.sim(run='tng300-1', redshift=0.0)

    temet.plot.subhalos.median(sim, 'Z_gas', 'mstar_30pkpc')

This produces a PDF figure named ``median_TNG300-1_Z_gas-mstar_30pkpc_cen.pdf`` in the current working 
directory. It shows the mass-metallicity relation of TNG300 galaxies at :math:`z=0`, and looks like this:

.. image:: _static/first_steps_median_1.png

We can enrich the plot in a number of ways, both by tweaking minor aesthetic options, and by including 
additional information from the simulation. For example, we will shift the x-axis bounds, and also 
include individual subhalos as colored points, coloring based on gas fraction::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.median(sim, 'Z_gas', 'mstar2', xlim=[8.0, 11.5], ylim=[-0.7, 0.5], scatterColor='fgas2')

This produces the following figure, which highlights how lower mass galaxies have high gas fractions of 
nearly unity, i.e. :math:`M_{\rm gas} \gg M_\star`, and that gas fraction slowly decreasing with stellar 
mass until :math:`M_\star \sim 10^{10.5} M_\odot`. At this point, the overall gas metallicity turns over 
and starts to decrease, as indicated by the black median line. Gas fractions also drop rapidly, reaching 
:math:`f_{\rm gas} \sim 10^{-4}` before starting to slowly rise again. This feature marks the onset of 
galaxy quenching due to supermassive black hole feedback.

.. image:: _static/first_steps_median_2.png

To more clearly see, and better understand, subtle secondary correlations, we can change the coloring. 
Instead of using color to represent the actual gas fraction values, we can instead represent the 
**relative** gas fractions, with respect to their median value at each stellar mass::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.median(sim, 'Z_gas', 'mstar_30pkpc', 
                               xlim=[8.0, 11.5], ylim=[-0.7, 0.5], 
                               scatterColor='fgas2', cRel=[0.5, 2.0, False])

More generally, the color quantity is made relative to its running median value as a function of the 
x-axis value (indicated in the colorbar label with the prefix :math:`\Delta`). The ``cRel`` 3-tuple 
controls the behavior. The first two values give the lower and upper limits of the relative color scale, 
while the third entry specifies whether we should log the values. For instance, you could also try 
``cRel=[-0.3,0.3,True]`` to obtain a similar plot. The resulting figure:

.. image:: _static/plotting_median_3.png

The figure clearly reveals that intermediate mass galaxies, e.g. with 
:math:`10^8 < M_\star / \rm{M}_\odot < 10^9`, have lower gas fractions if they have higher gas 
metallicities, and vice versa, at **fixed** stellar mass. This is the 
`fundamental metallicity relation <https://arxiv.org/abs/1005.0006>`_.

Instead of plotting individual colored markers, which can be misleading due to overcrowding and overlap, 
we can also use the technique of two-dimensional histograms, where the color can either indicate the 
number of objects in each pixel::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.histogram2d(sim, 'Z_gas', 'mstar_30pkpc', xlim=[8.0, 11.5], ylim=[-0.9, 0.5])

.. image:: _static/plotting_histo2D_1.png

.. note:: This is the same plot made by the following API endpoint of the TNG public data release:
    https://www.tng-project.org/api/TNG100-1/snapshots/99/subhalos/plot.png?xQuant=mstar2&yQuant=Z_gas
    and this API request is handled using the same plotting function we just called.

Alternatively, the color can indicate the median value of a third property for all the objects in each pixel::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.histogram2d(sim, 'Z_gas', 'mstar_30pkpc', cQuant='size_stars', xlim=[8.0, 11.5], ylim=[-0.9, 0.7])

.. image:: _static/plotting_histo2D_2.png

Finally, we can provide a more quantitative look at secondary correlations with a vertical 'slice'. 
For example, given the plot above, we can look at the correlation between gas metallicity and stellar size 
(half mass radius) for galaxies in particular, narrow stellar mass bins, contrasting 
:math:`10^{9.5} < M_\star / \rm{M}_\odot < 10^{9.6}` against :math:`10^{10.5} < M_\star / \rm{M}_\odot < 10^{10.6}`::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.subhalos.slice([sim], xQuant='size_stars', yQuants=['Z_gas'], sQuant='mstar_30pkpc_log',
      sRange=[[9.4, 9.6],[10.4, 10.7]], ylim=[-0.5, 0.5], xlim=[0.0, 1.2])

.. image:: _static/plotting_histo2D_3.png

This confirms the visual impression above, that the direction of the correlation between size and metallicity 
is different at these two mass scales. Note that automatic limits will be chosen for quantities, but these 
can be overriden by e.g. the ``xlim`` and ``ylim`` arguments when needed.

These methods are the bread and butter of exploring trends. Once you add a custom calculation for a new 
property of subhalos, i.e. compute a value which isn't available by default, you can understand how it varies 
across the galaxy population, and correlates with other galaxy properties, with these same methods.


Exploratory Plots for Snapshots (Particle/Cell Values)
------------------------------------------------------

Here we also continue from the corresponding section of the :doc:`first_steps`.

Similar to above, :py:mod:`plot.snapshot <temet.plot.snapshot>` provides general plotting routines focused 
on snapshots, i.e. particle-level data. These are also then suitable for non-cosmological simulations.

We can examine the simple one-dimensional distributions (i.e. histograms) of any known quantity::

    sims = [temet.sim(run='tng100-3', redshift=0.0),
            temet.sim(run='illustris-3', redshift=0.0)]

    temet.plot.snapshot.histogram1d(sims, 'gas', 'temp')

.. image:: _static/plotting_histo_1.png

We can also restrict to one or more halos/subhalos, or apply cuts on other quantities. For example, the temperature 
distribution of gas within several different bins of halo mass::

    sim = temet.sim(run='tng100-3', redshift=0.0)

    mass_bins = [[14.0,14.2], [13.0,13.05], [12.0,12.01]]

    m200 = sim.units.codeMassToLogMsun(sim.halos('Group_M_Crit200'))
    haloIDs = []

    for mass_bin in mass_bins:
        ids = np.where((m200 >= mass_bin[0]) & (m200 < mass_bin[1]))[0]
        print(mass_bin, ids.size)
        haloIDs.append(ids)

    temet.plot.snapshot.histogram1d([sim], 'gas', 'temp', haloIDs=haloIDs)

.. image:: _static/plotting_histo_2.png

We can also plot the "phase space" of any two quantities, showing how they relate to one another. 
Furthermore, while color can represent the distribution of mass, it can also be used to show the value of a 
third particle/cell property, in each pixel (as we show below). Let's look at 
the relationship between gas pressure and magnetic field strength at :math:`z=0`::

    sim = temet.sim(run='tng100-1', redshift=0.0)

    temet.plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='pres', yQuant='bmag')

.. image:: _static/first_steps_phase2D_1.png

For cosmological simulations, we can also look at particle/cell properties for one or more (stacked) halos. 
For example, the radial profiles of gas temperature around Milky Way-mass halos, comparing TNG50, EAGLE, 
and SIMBA at :math:`z=0`::

    sims = [temet.sim('tng50-1', redshift=0.0),
            temet.sim('eagle', redshift=0.0),
            temet.sim('simba', redshift=0.0)]

    subIDs = []
    for sim in sims:
        m200 = sim.subhalos('m200c_log')
        ids = np.where((m200 > 12.0) & (m200 < 12.1))[0]
        subIDs.append(ids)

    temet.plot.snapshot.profilesStacked1d(sims, subhaloIDs=subIDs, ptType='gas', ptProperty='temp', xlim=[0.9,2.7], ylim=[4, 6.5])

.. image:: _static/first_steps_profile.png

We can also look at particle/cell properties for one or more (stacked) halos. 
For example, the relationship between (halocentric) radial velocity and (halocentric) distance, for all dark 
matter particles within the tenth most massive halo of TNG50-1 at :math:`z=2`::

    sim = temet.sim(run='tng50-1', redshift=2.0)
    haloIDs = [9]

    opts = {'xlim':[-0.6,0.3], 'ylim':[-800,600], 'clim':[-4.7,-2.3], 'ctName':'inferno'}
    temet.plot.snapshot.phaseSpace2d(sim, 'dm', xQuant='rad_rvir', yQuant='vrad', haloIDs=haloIDs, **opts)

.. image:: _static/first_steps_phase2D_2.png

Here we see individual gravitationally bound substructures (subhalos) within the halo as bright vertical 
features.

As with the group catalog plotting routines above, color can represent either the distribution of mass in 
this plane, or it can be used to visualize the value of a third particle/cell property, in each pixel. 
For example, we can plot the usual density-temperature phase diagram (of SIMBA at :math:`z=0`) twice: 
once where color shows the distribution of gas mass, and once where color indicates the mean metallicity of gas 
in each pixel::

    sim = temet.sim(run='simba', redshift=0.0)

    temet.plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='numdens', yQuant='temp', ylim=[1,9])

    temet.plot.snapshot.phaseSpace2d(sim, 'gas', xQuant='numdens', yQuant='temp', ylim=[1,9], meancolors=['Z_solar'])

.. image:: _static/plotting_phase_1.png
    :scale: 48%
.. image:: _static/plotting_phase_2.png
    :scale: 48%

.. tip::

    In the case of a cosmological simulation with group catalogs, any of these plots can be restricted to one or 
    more [sub]halos, for instance to create stacks in different bins of galaxy properties such as mass.
