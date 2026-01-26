"""Catalog-based analysis and post-processing (auxcat creation) for cosmological simulations.

The functions here are rarely called directly. Instead they are typically invoked from
within a particular auxCat request.
"""

from . import box, common, gasflows, group, maps, profile, subhalo, temporal
