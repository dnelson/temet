from mpi4py import MPI
import numpy as np
import tables
import sys
import readhaloHDF5
import readsnapHDF5 as rs
import conversions as co
from scipy import interpolate
from scipy.ndimage import zoom
import readsubfHDF5
import time

hubble=0.704
Omega0=0.2726
OmegaBaryon=0.0456
OmegaLambda=0.7274



f=tables.openFile("stellar_photometrics_extended.hdf5")
Z_array	     = f.root.LogMetallicity_bins[:]
ages_array   = f.root.LogAgeInGyr_bins[:]

Mag_U_array  = f.root.Magnitude_U[:]
Mag_B_array  = f.root.Magnitude_B[:]
Mag_V_array  = f.root.Magnitude_V[:]
Mag_K_array  = f.root.Magnitude_K[:]
Mag_J_array  = f.root.Magnitude_J[:]
Mag_u_array  = f.root.Magnitude_u[:]
Mag_g_array  = f.root.Magnitude_g[:]
Mag_r_array  = f.root.Magnitude_r[:]
Mag_i_array  = f.root.Magnitude_i[:]
Mag_z_array  = f.root.Magnitude_z[:]
f.close()


ZZ, AA = np.meshgrid(Z_array, ages_array)

fUint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_U_array, kx=1, ky=1)
fBint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_B_array, kx=1, ky=1)
fVint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_V_array, kx=1, ky=1)
fKint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_K_array, kx=1, ky=1)
fJint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_J_array, kx=1, ky=1)
fuint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_u_array, kx=1, ky=1)
fgint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_g_array, kx=1, ky=1)
frint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_r_array, kx=1, ky=1)
fiint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_i_array, kx=1, ky=1)
fzint = interpolate.RectBivariateSpline(Z_array, ages_array, Mag_z_array, kx=1, ky=1)

def fU(Z, A, M):
        return fUint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fB(Z, A, M):
        return fBint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fV(Z, A, M):
        return fVint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fK(Z, A, M):
        return fKint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fJ(Z, A, M):
        return fJint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fu(Z, A, M):
	return fuint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fg(Z, A, M):
        return fgint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fr(Z, A, M):
        return frint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fi(Z, A, M):
        return fiint(Z, A) - 2.5 * np.log10(1e10*M/hubble)
def fz(Z, A, M):
        return fzint(Z, A) - 2.5 * np.log10(1e10*M/hubble)


vfU=np.vectorize(fU)
vfB=np.vectorize(fB)
vfV=np.vectorize(fV)
vfK=np.vectorize(fK)
vfJ=np.vectorize(fJ)
vfu=np.vectorize(fu)
vfg=np.vectorize(fg)
vfr=np.vectorize(fr)
vfi=np.vectorize(fi)
vfz=np.vectorize(fz)





base="/n/hernquistfs1/Illustris/Runs/Illustris-3/output/"
subnum=10
num=135
cat=readsubfHDF5.subfind_catalog(base, num, long_ids=True, keysel=["SubhaloStellarPhotometrics","SubhaloLenType"])
head   = rs.snapshot_header(base+"/snapdir_"+str(num).zfill(3)+"/"+"snap"+"_"+str(num).zfill(3)+".0")
ascale = head.time


magsub_U              = np.zeros(cat.nsubs)
magsub_U_global       = np.zeros(cat.nsubs)
magsub_B              = np.zeros(cat.nsubs)
magsub_B_global       = np.zeros(cat.nsubs)
magsub_V              = np.zeros(cat.nsubs)
magsub_V_global       = np.zeros(cat.nsubs)
magsub_K              = np.zeros(cat.nsubs)
magsub_K_global       = np.zeros(cat.nsubs)
magsub_J              = np.zeros(cat.nsubs)
magsub_J_global       = np.zeros(cat.nsubs)
magsub_u              = np.zeros(cat.nsubs)
magsub_u_global       = np.zeros(cat.nsubs)
magsub_g              = np.zeros(cat.nsubs)
magsub_g_global       = np.zeros(cat.nsubs)
magsub_r              = np.zeros(cat.nsubs)
magsub_r_global       = np.zeros(cat.nsubs)
magsub_i              = np.zeros(cat.nsubs)
magsub_i_global       = np.zeros(cat.nsubs)
magsub_z              = np.zeros(cat.nsubs)
magsub_z_global       = np.zeros(cat.nsubs)







comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


index_selection = cat.SubhaloLenType[:,4] > 0
totsubs         = np.arange(0, cat.nsubs, dtype='uint32')[index_selection]
Ntotsubs        = totsubs.shape[0]
index           = (totsubs % size) == rank
dosubs          = totsubs[index]
Ndosubs         = dosubs.shape[0]

#Ndosubs=100

t0 = time.time()
for i in range(0, Ndosubs):
        print rank, i, Ndosubs
        sys.stdout.flush()

        subnum=dosubs[i]

#	readhaloHDF5.reset()

	gz   = readhaloHDF5.readhalo(base, "snap", num, "GZ  ", 4, -1, subnum, long_ids=True, double_output=False)
	age  = readhaloHDF5.readhalo(base, "snap", num, "GAGE", 4, -1, subnum, long_ids=True, double_output=False)
	mass = readhaloHDF5.readhalo(base, "snap", num, "GIMA", 4, -1, subnum, long_ids=True, double_output=False)
	mag  = readhaloHDF5.readhalo(base, "snap", num, "GSPH", 4, -1, subnum, long_ids=True, double_output=False)
	#                "GSPH":["GFM_StellarPhotometrics",8], #band luminosities: U, B, V, K, g, r, i, z


	#filter wind
	idx = age>=0
	if (idx.any()==False):
		continue
	gz   = gz[idx]
	age  = age[idx]
	mass = mass[idx]
	mag  = mag[idx]

	age  = co.GetTime(ascale, OmegaM=Omega0, OmegaL=OmegaLambda, h=hubble) - co.GetTime(age, OmegaM=Omega0, OmegaL=OmegaLambda, h=hubble)
	
	idx     = gz < 10.0**Z_array.min() 
	gz[idx] = 10.0**Z_array.min()*1.001
	idx     = gz > 10.0**Z_array.max()
	gz[idx] = 10.0**Z_array.max()*0.999

	idx      = age < 10.0**ages_array.min()
	age[idx] = 10.0**ages_array.min()*1.001
	idx      = age > 10.0**ages_array.max()
	age[idx] = 10.0**ages_array.max()*0.999
	
	log_Z = np.log10(gz)
	log_A = np.log10(age)

	#print "CHECK:"
	#print log_Z.min(), log_Z.max()
	#print Z_array.min(), Z_array.max()
	#print np.isnan(log_Z).any()
	#print log_A.min(), log_A.max()
	#print ages_array.min(), ages_array.max()
	#print np.isnan(log_A).any()

	magsub_U[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfU(log_Z,log_A,mass))).sum())
	magsub_B[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfB(log_Z,log_A,mass))).sum())
	magsub_V[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfV(log_Z,log_A,mass))).sum())
	magsub_K[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfK(log_Z,log_A,mass))).sum())
	magsub_J[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfJ(log_Z,log_A,mass))).sum())
	magsub_u[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfu(log_Z,log_A,mass))).sum())
	magsub_g[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfg(log_Z,log_A,mass))).sum())
	magsub_r[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfr(log_Z,log_A,mass))).sum())
	magsub_i[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfi(log_Z,log_A,mass))).sum())
	magsub_z[subnum]  = -2.5 * np.log10((10.0**(-0.4 * vfz(log_Z,log_A,mass))).sum())

	#print cat.SubhaloStellarPhotometrics[subnum,4] - magsub_g[subnum], cat.SubhaloStellarPhotometrics[subnum,5] - magsub_r[subnum], cat.SubhaloStellarPhotometrics[subnum,6] - magsub_i[subnum],cat.SubhaloStellarPhotometrics[subnum,7] - magsub_z[subnum] 


comm.Barrier()

comm.Allreduce(magsub_U,        magsub_U_global,        op=MPI.SUM)
comm.Allreduce(magsub_B,        magsub_B_global,        op=MPI.SUM)
comm.Allreduce(magsub_V,        magsub_V_global,        op=MPI.SUM)
comm.Allreduce(magsub_K,        magsub_K_global,        op=MPI.SUM)
comm.Allreduce(magsub_J,        magsub_J_global,        op=MPI.SUM)
comm.Allreduce(magsub_u,	magsub_u_global,	op=MPI.SUM)
comm.Allreduce(magsub_g,        magsub_g_global,        op=MPI.SUM)
comm.Allreduce(magsub_r,        magsub_r_global,        op=MPI.SUM)
comm.Allreduce(magsub_i,        magsub_i_global,        op=MPI.SUM)
comm.Allreduce(magsub_z,        magsub_z_global,        op=MPI.SUM)

t1 = time.time()


if (rank==0):
        print "time =", t1-t0
        fname = "test3.hdf5"           #postbase + "/galprop/galprop_" + str(num).zfill(3) + ".hdf5"
        f=tables.openFile(fname, mode = "w")
	f.createArray(f.root,   "magsub_U",     magsub_U_global)
	f.createArray(f.root,   "magsub_B",     magsub_B_global)
	f.createArray(f.root,   "magsub_V",     magsub_V_global)
	f.createArray(f.root,   "magsub_K",     magsub_K_global)
	f.createArray(f.root,   "magsub_J",     magsub_J_global)
        f.createArray(f.root,	"magsub_u",	magsub_u_global)
	f.createArray(f.root,   "magsub_g",     magsub_g_global)
	f.createArray(f.root,   "magsub_r",     magsub_r_global)
	f.createArray(f.root,   "magsub_i",     magsub_i_global)
	f.createArray(f.root,   "magsub_z",     magsub_z_global)
        f.close()


