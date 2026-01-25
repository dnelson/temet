"""Visualization routines, which are broadly split into two types: 'box' and 'halo'.

In each case, an assortment of drivers and examples are available in :py:mod:`vis.boxDrivers`
and :py:mod:`vis.haloDrivers`, respectively. Light option parsing and pre-processing is
handled in :py:mod:`vis.box` and :py:mod:`vis.halo`, respectively. In both cases, most of
the work is actually accomplished in :py:mod:`vis.common`.
"""

from . import box, boxDrivers, boxMovieDrivers, common, halo, haloDrivers, haloMovieDrivers, lic, quantities, render
