.. _quantities:

Physical Quantities
===================

TODO.


Default Snapshot Quantities
---------------------------

These are the fields which are directly available from the on-disk snapshot files themselves.
For example, ``PartType0/StarFormationRate`` (for gas) or ``PartType4/Masses`` (for stars). 
For complete documentation of available fields, their definitions, and units, see e.g. 

* the IllustrisTNG Public Data Release `Snapshot Documentation <https://www.tng-project.org/data/docs/specifications/#sec1>`_.


Default Group Catalog Quantities
--------------------------------

These are the fields which are directly available from the on-disk group catalog files themselves.
For example, ``SubhaloMass`` or ``Group_M_Crit200``. For complete documentation of available fields, their 
definitions, and units, see e.g. 

* the IllustrisTNG Public Data Release `Group Catalog Documentation <https://www.tng-project.org/data/docs/specifications/#sec2>`_.


Custom Snapshot Quantities
--------------------------

The code base currently has the following particle/cell-level custom quantities defined, as
described in the :py:func:`plot.quantities.simParticleQuantity` function, along with associated 
metadata including a description, units, reasonable bounds, and so on.

.. exec::
    # code executes in python/ directory, load the source of a file
    import re

    with open('plot/quantities.py','r') as f:
        lines = f.readlines()

    # start output (is then run through the rest/html build parser)
    print('.. csv-table::')
    print('    :header: "Quantity Name", "Aliases", "Units", "Description"')
    print('    :widths: 10, 30, 20, 40')
    print('')

    start = False

    for line in lines:
        # skip to our function of interest
        if "def simParticleQuantity(" in line:
            start = True
        if not start:
            continue

        # find conditionals
        if "if prop in [" in line:
            m = re.findall("'(.+?)'", line)
            #m = re.search("'(.+?)'", line)
            if m:
                name = m[0]
                aliases = ', '.join(m[1:])
                print('    "%s", "%s", "", ""' % (name,aliases))
                #print("    " + str(m.group(1)))

        # TODO: handle lines of type "if 'something' in ptProperty"

    #print('\n')


.. _custom_group_quantities:

Custom Group Catalog Quantities
-------------------------------

TODO.
