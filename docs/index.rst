
temet toolkit: documentation
============================

Welcome to the documentation for **temet**, a python toolkit for the execution, analysis, and 
visualization of numerical simulations. Specifically, hydrodynamical simulations run with the 
`AREPO <https://wwwmpa.mpa-garching.mpg.de/~volker/arepo/>`_ moving mesh code, as well as codes 
producing similarly structured outputs including 
`GADGET-4 <https://wwwmpa.mpa-garching.mpg.de/gadget4/>`_, 
`GIZMO <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html>`_ and 
`SWIFT <http://swift.dur.ac.uk/>`_.

In addition, this codebase is focused on cosmological simulations for large-scale structure and 
galaxy formation, particularly those processed with a substructure identification algorithm 
(halo finder) such as ``subfind``, including `Illustris <https://www.illustris-project.org/>`_, 
`IllustrisTNG <https://www.tng-project.org/>`_, EAGLE, and so on. Generally speaking, any 
simulation data available from the TNG public data release platform can be directly analyzed 
with this toolkit.

.. image:: _static/logo_lg.png

.. toctree::
    :titlesonly:
    :caption: Getting Started

    installation
    first_steps
    features

.. toctree::
    :titlesonly:
    :caption: Usage

    quantities
    catalogs
    visualization
    plotting
    cookbook
    advanced

.. toctree::
    :maxdepth: 1
    :caption: Reference

    source/temet.ICs
    source/temet.cosmo
    source/temet.load
    source/temet.obs
    source/temet.plot
    source/temet.projects
    source/temet.tracer
    source/temet.util
    source/temet.vis


Citation and Acknowledgment
===========================

This code was originally written by Dylan Nelson (dnelson@uni-heidelberg.de).

If you find it useful and/or make use of it for a scientific publication, please cite `Nelson, D. (in prep)`.


Index
=====

* :ref:`genindex`
* :ref:`modindex`

To contribute to this documentation, update either in-code comments or supplementary files in
the ``docs/`` directory, then commit. 
