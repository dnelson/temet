#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "proto.h"
#include "allvars.h"
#include "/usr/lib64/python2.4/site-packages/numpy/core/include/numpy/arrayobject.h"
//#include "/usr/lib64/python2.4/site-packages/numpy/core/include/numpy/arrayobject.h"



PyObject * _GetPHKeys (PyObject * self, PyObject * args)
{
  PyArrayObject *pos,  *pyReturn_val;
  npy_intp dims[1];
  int* data_pos;
  int i;
  long long *return_val;

  if (!PyArg_ParseTuple
      (args,
       "O!i::PHorder( pos, BitsPerDimension)",
       &PyArray_Type, &pos, &BitsPerDimension))
    {
      return 0;
    }


  if (pos->nd != 2 || pos->dimensions[1] != 3
      || pos->descr->type_num != PyArray_INT)
    {
      PyErr_SetString (PyExc_ValueError, "pos has to be of dimensions [n,3] and type int32");
      return 0;
    }

  NumPart = pos->dimensions[0];

  dims[0] = NumPart;

  pyReturn_val = (PyArrayObject *) PyArray_SimpleNew (1, dims, NPY_UINT64);
  return_val = (long long *) pyReturn_val->data;
  memset (return_val, 0, NumPart * sizeof (long long));

  P = (struct particle_data *) malloc (NumPart * sizeof (struct particle_data));

  data_pos = (int *) pos->data;


  for (i = 0; i < NumPart; i++)
    {
      P[i].Pos[0] = (int) *data_pos;
      data_pos = (int *) ((char *) data_pos + pos->strides[1]);
      P[i].Pos[1] = (float) *data_pos;
      data_pos = (int *) ((char *) data_pos + pos->strides[1]);
      P[i].Pos[2] = (float) *data_pos;
      data_pos =(int *) ((char *) data_pos - 2 * pos->strides[1] +  pos->strides[0]);
  }

  get_keys();

  for (i = 0; i < NumPart; i++)
   {
    return_val[i]=P[i].Id;
   } 
  free (P);
 
  return PyArray_Return (pyReturn_val);
}




PyObject * _GetInversePHKeys (PyObject * self, PyObject * args)
{
  PyArrayObject *pos,  *pyReturn_val;
  npy_intp dims[2];
  long long* data_key;
  int i;
  int *return_val;

  if (!PyArg_ParseTuple
      (args,
       "O!i::PHorder( pos, BitsPerDimension)",
       &PyArray_Type, &pos, &BitsPerDimension))
    {
      return 0;
    }


  if (pos->nd != 1  || pos->descr->type_num != PyArray_UINT64)
    {
      PyErr_SetString (PyExc_ValueError, "pos has to be of dimensions [n] and type uint64");
      printf("%d %d\n", pos->descr->type_num, PyArray_UINT64);
      return 0;
    }

  NumPart = pos->dimensions[0];

  dims[0] = NumPart;
  dims[1] = 3;

  pyReturn_val = (PyArrayObject *) PyArray_SimpleNew (2, dims, NPY_INT);
  return_val = (int *) pyReturn_val->data;
  memset (return_val, 0, NumPart * sizeof (int));

  P = (struct particle_data *) malloc (NumPart * sizeof (struct particle_data));

  data_key = (long long *) pos->data;

  for (i = 0; i < NumPart; i++)
    {
      P[i].Id = (long long) *data_key;
      data_key = (long long *) ((char *) data_key + pos->strides[0]);
  }

  get_pos();
 
  for (i = 0; i < NumPart; i++)
   {
    return_val[3*i+0]=P[i].Pos[0];
    return_val[3*i+1]=P[i].Pos[1];
    return_val[3*i+2]=P[i].Pos[2];
   } 
  free (P);


  return PyArray_Return (pyReturn_val);
}





static PyMethodDef PeanoHilbertKeyMethods[] = {
  {"GetPHKeys", _GetPHKeys, METH_VARARGS,
   "Get Peano-Hilbert keys of particles."},
  {"GetInversePHKeys", _GetInversePHKeys, METH_VARARGS,
   "Get position of particles."},
  {NULL, NULL, 0, NULL},
};

void
initPeanoHilbertKey (void)
{
  Py_InitModule ("PeanoHilbertKey", PeanoHilbertKeyMethods);
  import_array ();
}
