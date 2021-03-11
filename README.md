python
======

A python toolkit for the execution, analysis, and visualization of numerical simulations. In particular, with a particular emphasis on hydrodynamical simulations run with the [AREPO](https://wwwmpa.mpa-garching.mpg.de/~volker/arepo/) moving mesh code, as well as codes producing similarly structured outputs including [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) and [SWIFT](http://swift.dur.ac.uk/).

In addition, this codebase is focused on cosmological simulations for large-scale structure and galaxy formation, particularly those processed with the ``subfind`` substructure identification algorithm, such as [Illustris](https://www.illustris-project.org), [IllustrisTNG](https://www.tng-project.org/), and EAGLE.


Documentation
-------------

Please see [online documentation](https://www.tng-project.org/files/dnelson_python_docs/).

Installation and usage instructions are available there.


Acknowledgment & Citation
-------------------------

TODO.


Organization
------------

* cosmo/ - analysis specific for cosmological/comoving boxes
* data/ - external text data files from the literature
* ICs/ - misc idealized initial condition generation
* load/ - loading of group catalogs, snapshots, auxcats
* obs/ - analysis and data reduction for observations
* plot/ - generalized plotting
* projects/ - routines, both analysis and plotting, specific to papers
* tables/ - large pre-computed datafiles (empty until downloaded)
* tracer/ - Monte Carlo tracer particles
* util/ - misc helper utilities
* vis/ - visualization


Reproducing Published Papers
----------------------------

The complete analysis and plot set of published papers can (theoretically) be reproduced with the following entry points 
inside the `projects/` directory. Note that in practice some analyses are costly and would better be done (and were actually 
done) by splitting into many parallel jobs on a cluster. Also note that exact reproduction may require use of an (older) code 
version, tagged on the date near the finalization of the paper.

[Nelson et al. (2018a) - TNG colors](http://arxiv.org/abs/1707.03395) - `projects.color.paperPlots()`

[Nelson et al. (2018b) - TNG oxygen](http://arxiv.org/abs/1712.00016) - `projects.oxygen.paperPlots()`

[Nelson et al. (2019b) - TNG50 outflows](http://arxiv.org/abs/1902.05554) - `projects.outflows.paperPlots()`

[Nelson et al. (2020) - TNG50 small-scale CGM](http://arxiv.org/abs/2005.09654) - `projects.lrg.paperPlots()`

[Nelson et al. (in prep) - TNG50 MgII emission](#) - `projects.mg2emission.paperPlots()`


Contributing
------------

TODO.
