#!python
#cython: wraparound = False
#cython: boundscheck = False
#cython: language_level=3
import numpy as np
cimport numpy as np
import cython
cimport cython 

#from libcpp cimport bool

ctypedef fused real:
    cython.char
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double

cdef extern from "<parallel/algorithm>" namespace "__gnu_parallel":
    cdef void sort[T](T first, T last) nogil
    #cdef void sort[T](T first, T last, bool cmp) nogil 

def mysort(real[:] a):
    "In-place parallel sort for numpy types"
    sort(&a[0], &a[a.shape[0]])

#cdef bool mycmp(long* a, long* b):
#    return a < b

#def myargsort(real[:] a):
#    "Parallel argsort (return indices) for numpy types"
#    assert a.ndims == 1
#    inds = np.arange(a.size)
#
#    # sort indices by comparing values in a (lambda function comparator)
#    sort(&inds[0], &inds[inds.shape[0]], [&a](size_t i1, size_t i2){ return a[i1] < a[i2]; });
#    #sort(&inds[0], &inds[inds.shape[0]], mycmp);
#    return inds

