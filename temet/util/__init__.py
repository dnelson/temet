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
    from numba import cuda
    try:
        from cuda.pathfinder import DynamicLibNotFoundError
        _cuda_errs = (ImportError, DynamicLibNotFoundError)
    except ImportError:
        _cuda_errs = (ImportError,)

    from . import delaunay
except _cuda_errs:
    print("Warning: Numba CUDA not available. Tetrahedral rendering requires CUDA.")
