"""Utilities, units, numba-accelerated algorithms, and other algorithmic implementations."""

from . import (
    boxRemap,
    dataConvert,
    dataConvertSim,
    extern,
    helper,
    match,
    rotation,
    sphMap,
    subfind,
    tpcf,
    treeSearch,
    virtualSimFile,
    voronoi,
    voronoiRay,
)
from .simParams import simParams
from .units import units


# avoid GPU errors on github CI runners
try:
    from cuda.pathfinder import DynamicLibNotFoundError
    from numba import cuda

    from . import delaunay
except (ImportError, DynamicLibNotFoundError):
    print("Warning: Numba CUDA not available. Tetrahedral rendering requires CUDA.")
