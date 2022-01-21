First Steps Walkthrough
=======================

First, start a command-line ``ipython`` session, or a Jupyter notebook. Then import the library

.. code-block:: python

    import tenet

Simulation Selection
--------------------

Most analysis is based around a "simulation parameters" object (typically called ``sP`` or ``sim`` below), 
which specifies the simulation and snapshot of interest, among other details.  You can then select a simulation and snapshot from a known list, e.g. the TNG100-2 simulation of IllustrisTNG at redshift two:

.. code-block:: python

    sP = tenet.sim(res=910, run='tng', redshift=2.0)

Or the TNG50-1 simulation at redshift zero, by its short-hand name

.. code-block:: python

    sP = tenet.sim(run='tng50-1', redshift=0.0)

Note that if you would like to select a simulation which is not in the pre-defined list of known simulations, 
i.e. a simulation you have run yourself, then you can specify it simply by path

.. code-block:: python

    sP = tenet.sim('/home/user/sims/sim_run/', redshift=0.0)

In all cases, the redshift or snapshot number is optionally used to pre-select the particular snapshot of 
interest. You can also specify the overall simulation without this, for example

.. code-block:: python

    sim = tenet.sim('tng300-1')


Loading Data
------------

Once a simulation and snapshot is selected you can load the corresponding data. For example, to load one or more 
particular fields from the group catalogs

.. code-block:: python

    subs = sP.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos'])
    sub_masses_logmsun = sP.units.codeMassToLogMsun( subs['SubhaloMass'] )

To load particle-level data from the snapshot itself

.. code-block:: python

    gas_pos = sP.snapshotSubset('gas', 'pos')
    dm_vel_sub10 = sP.snapshotSubset('dm', 'vel', subhaloID=10)

In addition to shorthand names for fields such as "pos" (mapping to "Coordinates"), many custom fields 
at both the particle and group catalog level are defined. Loading data can also be done with shorthands, 
for example

.. code-block:: python

    subs = sP.subhalos('mstar_30pkpc')
    x = sP.gas('cellsize_kpc')

    fof10 = sP.halo(10) # all fields
    sub_sat1 = sP.subhalo( fof10['GroupFirstSub']+1 ) # all fields


Exploratory Plots
-----------------

The various plotting functions in :py:mod:`plot.general` and :py:mod:`plot.cosmoGeneral` are designed to 
be as general and automatic as possible. They are idea for a quick look or for exploring trends in the 
data.

Let's examine a classic observed galaxy scaling relation: the correlation between gas-phase metallicity, 
and stellar mass, the "mass-metallicity relation" (MZR).

.. code-block:: python

    sP = tenet.sim(run='tng100-1', redshift=0.0)

    plot.cosmoGeneral.quantMedianVsSecondQuant(sP, 'Z_gas', 'mstar_30pkpc')

Produces a PDF figure named ``medianQuants_TNG100-1_Z_gas_mstar_30pkpc_cen.pdf`` in the current working 
directory. It shows the mass-metallicity relation of TNG100 galaxies at :math:`z=0`, and looks like this:

.. image:: _static/first_steps_medianQuants_1.png

We can enrich the plot in a number of ways, both by tweaking minor aesthetic options, and by including 
additional information from the simulation. For example, we will shift the x-axis bounds, and also 
include individual subhalos as colored points, coloring based on gas fraction::

    sP = tenet.sim(run='tng100-1', redshift=0.0)

    plot.cosmoGeneral.quantMedianVsSecondQuant(sP, 'Z_gas', 'mstar_30pkpc', 
      xlim=[8.0, 11.5], scatterColor='fgas2')

This produces the following figure, which highlights how lower mass galaxies have high gas fractions of 
nearly unity, i.e. :math:`M_{\rm gas} \gg M_\star`, and that gas fraction slowly decreasing with stellar 
mass until :math:`M_\star \sim 10^{10.5} M_\odot`. At this point, the overall gas metallicity turns over 
and starts to decrease, as indicated by the black median line. Gas fractions also drop rapidly, reaching 
:math:`f_{\rm gas} \sim 10^{-4}` before starting to slowly rise again. This feature marks the onset of 
galaxy quenching due to supermassive black hole feedback.

.. image:: _static/first_steps_medianQuants_2.png

Look at crel.

Instead of individual colored markers, switch to quantHisto2D.

.. note:: This is the exact plot made by the following API endpoint of the TNG public data release

    https://www.tng-project.org/api/TNG100-1/snapshots/99/subhalos/plot.png?xQuant=mstar2&yQuant=Z_gas

    and this API request is handled using the exact plotting function we just called.


Picking an Interesting Object
-----------------------------

Let's pick one. Fields of catalog loading commands are the same. "Object" could be either halo or galaxy.


Visualizing a Halo or Galaxy
----------------------------

TODO.
