First Steps Walkthrough
=======================

Most analysis is based around a "simulation parameters" object (typically called ``sP``), which specifies the 
simulation and snapshot of interest, among other details.

For example, to load some data from the group catalog and snapshot of TNG100-2 at z=2, start a command-line ``ipython`` session, or a Jupyter notebook, and

.. code-block:: python

    sP = simParams(res=910, run='tng', redshift=2.0)

    subs = sP.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos'])
    sub_masses_logmsun = sP.units.codeMassToLogMsun( subs['SubhaloMass'] )

    gas_pos = sP.snapshotSubset('gas', 'pos')
    dm_vel_sub10 = sP.snapshotSubset('dm', 'vel', subhaloID=10)

In addition to shorthand names for fields such as "pos" (mapping to "Coordinates"), many custom fields at 
both the particle and group catalog level are defined. Loading data can also be done with shorthands, for example

.. code-block:: python

    sP = simParams(run='tng50-1', redshift=0.0)

    subs = sP.subhalos('mstar_30pkpc')
    x = sP.gas('cellsize_kpc')

    fof10 = sP.halo(10) # all fields
    sub_sat1 = sP.subhalo( fof10['GroupFirstSub']+1 ) # all fields

