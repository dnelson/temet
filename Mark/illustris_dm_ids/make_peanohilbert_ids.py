import readsnap as rs
import PeanoHilbertKey as ph
import numpy as np
import sys

fname_in="ICs"               #N-GenIC output
fname_out="ICs_ph"           #output
Num=4                        #number of IC files
BITS_PER_DIMENSION=21        #requires 3*BITS_PER_DIMENSION IDs





def read_dummy(f):
	return np.fromfile(f, dtype="uint32", count=1)

def write_dummy(f, value):
	dummy=np.array([value], dtype="uint32")
	dummy.tofile(f)








for n in range(0, Num):
	print "filenum=", n

	#read header
	f_in=open(fname_in+"."+str(n), "rb")
	f_out=open(fname_out+"."+str(n), "wb")

	head=rs.snapshot_header(fname_in+"."+str(n))
	BoxSize=head.boxsize
	N=head.npart.sum()
	fac=1.0 / BoxSize * ( 1 << (BITS_PER_DIMENSION))
	print "N       = ", N
	print "BoxSize = ", BoxSize
	print "ph fac  = ", fac

	print "HEADER reading"
	dummy1=read_dummy(f_in)
	head=np.fromfile(f_in, dtype="byte", count=256)
	dummy2=read_dummy(f_in)
	if (dummy1!=dummy2):
		print "ERROR dummy:", dummy1, dummy2
		sys.exit()
	

	print "HEADER writing"
	write_dummy(f_out, dummy1)
	head.tofile(f_out)
	write_dummy(f_out, dummy2)


	print "POS reading"
	dummy1=read_dummy(f_in)
	pos=np.fromfile(f_in, dtype="float32", count=N*3)
	dummy2=read_dummy(f_in)
        if (dummy1!=dummy2):
                print "ERROR dummy:", dummy1, dummy2
                sys.exit()


	print "POS writing"
	write_dummy(f_out, dummy1)
	pos.tofile(f_out)
	write_dummy(f_out, dummy2)


	print "VEL reading"
	dummy1=read_dummy(f_in)
	vel=np.fromfile(f_in, dtype="float32", count=N*3)
	dummy2=read_dummy(f_in)
        if (dummy1!=dummy2):
                print "ERROR dummy:", dummy1, dummy2
                sys.exit()


	print "VEL writing"
	write_dummy(f_out, dummy1)
	vel.tofile(f_out)
	write_dummy(f_out, dummy2)



	print "ID reading"
	dummy1=read_dummy(f_in)
	id=np.fromfile(f_in, dtype="uint64", count=N)
	dummy2=read_dummy(f_in)
        if (dummy1!=dummy2):
                print "ERROR dummy:", dummy1, dummy2
                sys.exit()

	keys=ph.GetPHKeys((fac*pos.reshape(N,3)).astype("int32"), BITS_PER_DIMENSION).astype("uint64")


	print "ID writing"
	write_dummy(f_out, dummy1)
	id.tofile(f_out)
	write_dummy(f_out, dummy2)


	f_in.close()
	f_out.close()

