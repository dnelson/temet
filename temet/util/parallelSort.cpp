// c++ -O3 -Wall -shared -std=c++11 -fopenmp -D_GLIBCXX_PARALLEL -fPIC $(python3 -m pybind11 --includes) parallelSort.cpp -o parallelSort.so
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <parallel/algorithm>
//#include <iostream>

namespace py = pybind11;

template<typename T> void sort(py::array_t<T> array) {
    // get array data buffer
    py::buffer_info buf = array.request();

    //std::cout << "Size: " << buf.size << std::endl;

    // sort input array directly
    __gnu_parallel::sort((T*)buf.ptr, (T*)buf.ptr + buf.size);
}

template<typename T> py::array_t<long int> argsort(py::array_t<T> array) {
    // get array data buffer
    py::buffer_info buf = array.request();
    T* data = (T*)buf.ptr;

    //std::cout << "Size: " << buf.size << std::endl;
    //std::cout << "OMP threads: " << omp_get_num_threads() << " (max = "
    //          << omp_get_max_threads() << ")" << std::endl;

    // allocate output buffer
    py::array_t<long int> inds = py::array_t<long int>(buf.size);
    py::buffer_info output_buf = inds.request();
    long int* inds_ptr = (long int*) output_buf.ptr;

    // fill output buffer with ascending indices to start
    for(auto i = 0; i < buf.size; i++)
        inds_ptr[i] = i;

    // sort index array, using a lambda comparator which compares elements of array[]
    __gnu_parallel::stable_sort(inds_ptr, 
                                inds_ptr + output_buf.size, 
                                [&data](size_t i1, size_t i2){ return data[i1] < data[i2]; });

    return inds;
}

PYBIND11_MODULE(parallelSort, m) {
    // in-place
    m.def("sort", &sort<float>, "A function that sorts in-place (float).");
    m.def("sort", &sort<double>, "A function that sorts in-place (double).");
    m.def("sort", &sort<short int>, "A function that sorts in-place (int16).");
    m.def("sort", &sort<long int>, "A function that sorts in-place (int32).");
    m.def("sort", &sort<long long int>, "A function that sorts in-place (int64).");
    m.def("sort", &sort<unsigned short int>, "A function that sorts in-place (uint16).");
    m.def("sort", &sort<unsigned long int>, "A function that sorts in-place (uint32).");
    m.def("sort", &sort<unsigned long long int>, "A function that sorts in-place (uint64).");

    // argsort
    m.def("argsort", &argsort<long int>, "An argsort-like function (int32 input).");
    m.def("argsort", &argsort<long long int>, "An argsort-like function (int64 input).");
    m.def("argsort", &argsort<unsigned long int>, "An argsort-like function (uint32 input).");
    m.def("argsort", &argsort<unsigned long long int>, "An argsort-like function (uint64 input).");
}

/*
 *
 * original cython-based code follows:
 *

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
*/

