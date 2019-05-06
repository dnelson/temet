# compile the parallelSort.pyx wrapper of the GCC parallel sort

NPDIR=/u/dnelson/.local/lib/python3.7/site-packages/numpy/core/include
PYLIBDIR=/u/dnelson/.local/envs/dylan3/lib/

all:
	cython --cplus -3 parallelSort.pyx  
	icc -g -march=native -std=c++11 -O3 -fpic   -I${NPDIR} -c parallelSort.cpp -o parallelSort.o -fopenmp `python3-config --includes`
	icc -g -march=native -std=c++11 -O3 -shared -I${NPDIR} -L${PYLIBDIR} -o parallelSort.so parallelSort.o `python3-config --libs` -lgomp 
	rm -f parallelSort.o parallelSort.cpp

clean:
	rm -f parallelSort.cpp parallelSort.o parallelSort.so

