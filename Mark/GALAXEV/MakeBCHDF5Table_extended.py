import numpy as np
import tables
import sys



filenames=["bc2003_hr_m22_chab_ssp.1color",
           "bc2003_hr_m32_chab_ssp.1color",
           "bc2003_hr_m42_chab_ssp.1color",
           "bc2003_hr_m52_chab_ssp.1color",
           "bc2003_hr_m62_chab_ssp.1color",
           "bc2003_hr_m72_chab_ssp.1color"]

filenames2=["bc2003_hr_m22_chab_ssp.1ABmag",
            "bc2003_hr_m32_chab_ssp.1ABmag",
            "bc2003_hr_m42_chab_ssp.1ABmag",
            "bc2003_hr_m52_chab_ssp.1ABmag",
            "bc2003_hr_m62_chab_ssp.1ABmag",
            "bc2003_hr_m72_chab_ssp.1ABmag"]


Zvals=len(filenames)
Agevals=220

ages_array=np.zeros(Agevals)                                           #Log[Gyr]
Mag_U_array=np.zeros([Zvals,Agevals])                                  #absolute U band magnitude (Vega)
Mag_B_array=np.zeros([Zvals,Agevals])                                  #absolute B band magnitude (Vega)
Mag_V_array=np.zeros([Zvals,Agevals])                                  #absolute V band magnitude (Vega)
Mag_K_array=np.zeros([Zvals,Agevals])                                  #absolute K band magnitude (Vega)
Mag_J_array=np.zeros([Zvals,Agevals])                                  #absolute J band magnitude (Vega)    
Mag_u_array=np.zeros([Zvals,Agevals])                                  #absolute u band magnitude (AB)
Mag_g_array=np.zeros([Zvals,Agevals])                                  #absolute g band magnitude (AB)
Mag_r_array=np.zeros([Zvals,Agevals])                                  #absolute r band magnitude (AB)
Mag_i_array=np.zeros([Zvals,Agevals])                                  #absolute i band magnitude (AB)
Mag_z_array=np.zeros([Zvals,Agevals])                                  #absolute z band magnitude (AB)
Z_array=np.log10(np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])) #metallicity (!NOT in Z_solar!)

#fill array
for Zval in range(0,Zvals):

	filename=filenames[Zval]
	print filename
	data=np.loadtxt(filename)
	ages_array=data[:,0]-9.0
        Mag_U_array[Zval,:]=data[:,2]
        Mag_B_array[Zval,:]=data[:,3]
        Mag_V_array[Zval,:]=data[:,4]	
	Mag_K_array[Zval,:]=data[:,5]
	Mag_J_array[Zval,:]=data[:,2]-data[:,10]

	filename=filenames2[Zval]
	print filename
	data=np.loadtxt(filename)
	Mag_u_array[Zval,:]=data[:,3]+data[:,2]
	Mag_g_array[Zval,:]=data[:,2]
	Mag_r_array[Zval,:]=data[:,2]-data[:,4]
	Mag_i_array[Zval,:]=data[:,2]-data[:,5]
	Mag_z_array[Zval,:]=data[:,2]-data[:,6]


hdf5_filename="stellar_photometrics_extended.hdf5"
f=tables.openFile(hdf5_filename, mode = "w")
f.createArray(f.root, "N_LogMetallicity", np.array([Zvals], dtype="int32"))
f.createArray(f.root, "N_LogAgeInGyr", np.array([Agevals], dtype="int32"))
f.createArray(f.root, "LogMetallicity_bins", Z_array)
f.createArray(f.root, "LogAgeInGyr_bins", ages_array)
f.createArray(f.root, "Magnitude_U", Mag_U_array)
f.createArray(f.root, "Magnitude_B", Mag_B_array)
f.createArray(f.root, "Magnitude_V", Mag_V_array)
f.createArray(f.root, "Magnitude_K", Mag_K_array)
f.createArray(f.root, "Magnitude_J", Mag_J_array)
f.createArray(f.root, "Magnitude_u", Mag_u_array)
f.createArray(f.root, "Magnitude_g", Mag_g_array)
f.createArray(f.root, "Magnitude_r", Mag_r_array)
f.createArray(f.root, "Magnitude_i", Mag_i_array)
f.createArray(f.root, "Magnitude_z", Mag_z_array)
f.close()
