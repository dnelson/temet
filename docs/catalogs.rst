Catalogs
========

TODO.


Defining a New Catalog
----------------------

TODO.


Current Catalog Definitions
---------------------------

The code base currently has the following catalog definitions, as listed in the 
``fieldComputeFunctionMapping`` dictionary of :py:mod:`cosmo.auxcatalog`.

.. exec::

    from cosmo.auxcatalog import fieldComputeFunctionMapping

    print('.. csv-table::')
    print('    :header: "Catalog Name", "Generator Function", "Arguments"')
    print('    :widths: 10, 30, 60')
    print('')

    for key in fieldComputeFunctionMapping.keys():
        func = fieldComputeFunctionMapping[key] # partial
        func_name = ":py:func:`~%s.%s`" % (func.func.__module__, func.func.__name__)
        func_args = ', '.join(['``%s`` = %s' % (k,v) for k,v in func.keywords.items()])
        print('    "%s", "%s", "%s"' % (key,func_name,func_args))

    print('\n')
