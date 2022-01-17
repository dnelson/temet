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
    #cdef void sort[T](T first, T last, int cmp(void*,void*)) nogil 

def mysort(real[:] a):
    "In-place parallel sort for numpy types"
    sort(&a[0], &a[a.shape[0]])

#----------

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

#-----------

#cdef long[:] values # define a global (cannot use real here)

#cdef int cmp_func(const void* a, const void* b):
#    cdef int a_ind = (<int *>a)[0]
#    cdef int b_ind = (<int *>b)[0]
#    return (values[a_ind] < values[b_ind])

#def myargsort(long[:] input_values):
#    "Parallel sort try."""
#    global values
#    values = input_values
#
#    assert input_values.ndim == 1
#    cdef np.ndarray[long, ndim=1] inds
#    inds = np.arange(input_values.size)
#
#    # sort indices by comparing values in a (lambda function comparator)
#    #sort(&inds[0], &inds[inds.shape[0]], [&a](size_t i1, size_t i2){ return a[i1] < a[i2]; });
#    sort(&inds[0], &inds[inds.shape[0]], &cmp_func);
#    return inds

