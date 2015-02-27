from math import *
import numpy as np


#constants
Gravity            = 6.6738e-8
SolarMass          = 1.989e33
SolarLum           = 3.826e33
ProtonMass         = 1.6725e-24
Boltzmann          = 1.38066e-16
Hydrogen_MassFrac  = 0.76
SolarMetallicity   = 0.02
CLight             = 2.99792458e10
Thompson           = 6.65245873e-25

#constants
SOLAR_MASS       = 1.989e33
SEC_PER_YEAR     = 3.155e7
SEC_PER_MEGAYEAR = 3.155e13

#set unit system
UnitMass_in_g            = 1.989e43        # 1.e10 solar masses
UnitVelocity_in_cm_per_s = 1e5             # 1 km/s
UnitLength_in_cm         = 3.085678e21     # 1 kpc
Hubble_in_cgs            = 3.2407789e-18   # h/sec
#derived
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitTime_in_Gyr          = UnitTime_in_s / SEC_PER_MEGAYEAR / 1000.0 
UnitDensity_in_cgs       = UnitMass_in_g / UnitLength_in_cm**3.
UnitEnergy_in_cgs        = UnitMass_in_g * UnitLength_in_cm**2. / UnitTime_in_s**2.
G                        = 6.672e-8 / UnitLength_in_cm**3. * UnitMass_in_g * UnitTime_in_s**2.
Hubble                   = Hubble_in_cgs  * UnitTime_in_s    

#mean atomic weights
H_maw  = 1.00794
He_maw = 4.002602
Li_maw = 6.941 
Be_maw = 9.012182
B_maw  = 10.811
C_maw  = 12.0107
N_maw  = 14.0067
O_maw  = 15.9994
F_maw  = 18.9984032
Ne_maw = 20.1791
Na_maw = 22.989770
Mg_maw = 24.3050
Al_maw = 26.981538
Si_maw = 28.0855
P_maw  = 30.973761
S_maw  = 32.065
Cl_maw = 35.453 
Ar_maw = 39.948
K_maw  = 39.0983
Ca_maw = 40.078
Sc_maw = 44.955910
Ti_maw = 47.867
V_maw  = 50.9415
Cr_maw = 51.9961
Mn_maw = 54.938049
Fe_maw = 55.845
Co_maw = 58.933200
Ni_maw = 58.6934
Cu_maw = 63.546
Zn_maw = 65.409

#solar n_i_over_n_H (Wiersma et al 2009)
n_H_over_n_H  = 1.0
n_He_over_n_H = 0.1
n_C_over_n_H  = 2.64e-4
n_N_over_n_H  = 8.51e-5
n_O_over_n_H  = 4.90e-4
n_Ne_over_n_H = 1.00e-4
n_Mg_over_n_H = 3.47e-5
n_Si_over_n_H = 3.47e-5
n_S_over_n_H  = 1.86e-5
n_Ca_over_n_H = 2.29e-6
n_Fe_over_n_H = 2.82e-5

#solar luminsoty in different bands (http://mips.as.arizona.edu/~cnaw/sun.html, http://www.astro.umd.edu/~ssm/ASTR620/mags.html)
mag_sun_U	=	 5.61	 #B&M
mag_sun_B	=	 5.48	 #B&M
mag_sun_V	=	 4.83	 #B&M
mag_sun_R	=	 4.42	 #B&M
mag_sun_I	=	 4.08	 #B&M
mag_sun_J	=	 3.64	 #B&M
mag_sun_H	=	 3.32	 #B&M
mag_sun_K	=	 3.28	 #B&M
mag_sun_Kprime	=	 3.27	 #*
#Spitzer		
mag_sun_36mu	=	 3.24	 #Oh
mag_sun_45mu	=	 3.27	 #Oh
#SDSS		
mag_sun_u	=	 6.55	 #S&G
mag_sun_g	=	 5.12	 #S&G
mag_sun_r	=	 4.68	 #S&G
mag_sun_i	=	 4.57	 #S&G
mag_sun_z	=	 4.60	 #S&G


def SetUnits(mass, velocity, length):
        """
        RETURNS: nothing, only resets internal unit system
        INPUT: mass     : mass unit 
               velocity : velocity unit
	       length   : length unit 
        """
	global UnitMass_in_g, UnitVelocity_in_cm_per_s, UnitLength_in_cm
	global UnitTime_in_s, UnitTime_in_Gyr, UnitDensity_in_cgs, UnitEnergy_in_cgs, G, Hubble
        #set unit system
	UnitMass_in_g            = mass 
	UnitVelocity_in_cm_per_s = velocity
	UnitLength_in_cm         = length 
	#derived
	UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
	UnitTime_in_Gyr          = UnitTime_in_s / (3600.*24.*365.*10.**9.)
	UnitDensity_in_cgs       = UnitMass_in_g / UnitLength_in_cm**3.
	UnitEnergy_in_cgs        = UnitMass_in_g * UnitLength_in_cm**2. / UnitTime_in_s**2.
	G                        = 6.672e-8 / UnitLength_in_cm**3. * UnitMass_in_g * UnitTime_in_s**2.
	Hubble                   = Hubble_in_cgs  * UnitTime_in_s

def GetUFullyIonized(T, gamma):
        """
        RETURNS: u in simulation units for fully ionized primordial gas 
        INPUT: T     : temperature in Kelvin 
               gamma : adiabatic index
        """
        Nelec = 1. + 2. * (1. - Hydrogen_MassFrac) / (4. * Hydrogen_MassFrac)
        MeanWeight = 4./(1.+3.*Hydrogen_MassFrac+4.*Hydrogen_MassFrac*Nelec) * ProtonMass
	return T/(gamma-1.)*Boltzmann/UnitEnergy_in_cgs*UnitMass_in_g / MeanWeight

def GetTempFullyIonized(u, gamma):
        """
        RETURNS: temperature in Kelvin for fully ionized primordial gas 
        INPUT: u     : thermal energy
               gamma : adiabatic index
        """
	Nelec = 1. + 2. * (1. - Hydrogen_MassFrac) / (4. * Hydrogen_MassFrac)
        MeanWeight = 4./(1.+3.*Hydrogen_MassFrac+4.*Hydrogen_MassFrac*Nelec) * ProtonMass
	return (gamma-1.)*u/Boltzmann*UnitEnergy_in_cgs/UnitMass_in_g * MeanWeight

def GetTempNeutral(u, gamma):
        """
        RETURNS: temperature in Kelvin for neutral primordial gas 
        INPUT: u     : thermal energy
               gamma : adiabatic index
        """
        MeanWeight = 4./(1.+3.*Hydrogen_MassFrac) * ProtonMass 
        return (gamma-1.)*u/Boltzmann*UnitEnergy_in_cgs/UnitMass_in_g * MeanWeight

def GetTemp(u, Nelec, gamma):
	"""
	RETURNS: temperature in Kelvin
	INPUT: u     : thermal energy
	       Nelec : electron abundance
	       gamma : adiabatic index	
    	"""
	MeanWeight = 4./(1.+3.*Hydrogen_MassFrac+4.*Hydrogen_MassFrac*Nelec) * ProtonMass
	return (gamma-1.)*u/Boltzmann*UnitEnergy_in_cgs/UnitMass_in_g * MeanWeight

def GetPressure(u, rho, gamma):
	"""
	RETURNS: pressure in simulation units
	INPUT: u     : thermal energy
	       rho   : density
	       gamma : adiabatic index	
    	"""
	return (gamma-1.)*rho*u

def GetEntropyFromTemperature(u, rho, Nelec, gamma):
	"""
	RETURNS: entropic function in simulation units using T/rho^(2/3)
        INPUT: u     : thermal energy
               rho   : density
	       Nelec : electron abundance
               gamma : adiabatic index
	"""
	return GetTemp(u, Nelec, gamma) / rho**(2.0/3.0)


def GetEntropyFromPressure(u, rho, gamma):
	"""
	RETURNS: entropic function in simulation units using P/rho^gamma
	INPUT: u     : thermal energy
	       rho   : density
	       gamma : adiabatic index	
    	"""
	return GetPressure(u, rho, gamma) / rho**gamma

def GetEntropyFromTemperatureFullyIonized(u, rho):
        """
        RETURNS: entropic function in simulation units using T/rho^(2/3) [NB: assume a gamma=5/3] 
        INPUT: u     : thermal energy
               rho   : density
        """
        return GetTempFullyIonized(u,5./3.) / rho**(2.0/3.0)

def GetEntropyFromTemperatureNeutral(u, rho):
        """
        RETURNS: entropic function in simulation units using T/rho^(2/3) [NB: assume a gamma=5/3] 
        INPUT: u     : thermal energy
               rho   : density
        """
        return GetTempNeutral(u,5./3.) / rho**(2.0/3.0)

def GetSoundSpeed(u, rho, gamma):
        """
        RETURNS: returns sound speed in simulation units 
        INPUT: u     : thermal energy
               rho   : density
               gamma : adiabatic index  
        """
	return np.sqrt(gamma * GetPressure(u, rho, gamma) / rho)
	
def GetRhoCrit(z=0, OmegaM=0.27,OmegaL=0.73):
	"""
	RETURNS: critical density
	INPUT: z     : redshift
    	"""
	ascale = 1.0/(1.0 + z)
	Efunc = (OmegaM / (ascale**3) + (1 - OmegaM - OmegaL) / (ascale**2) + OmegaL)
	return 3.*(Hubble*Efunc)**2./(8.*pi*G)

def GetnH(rho, ascale, h=0.7):
	"""
	RETURNS: physical hydrogen number density in cm^-3
	INPUT: rho    : density
	       ascale : scale factor	
	       h      : Hubble constant
    	"""
	return (rho*UnitDensity_in_cgs*h**2.) * (Hydrogen_MassFrac/ProtonMass) / ascale**3.

def GetTime(ascale, OmegaM=0.27,OmegaL=0.73,h=0.7):
	"""
	RETURNS: time for given cosmology and scale factor in simulation units
	INPUT: ascale : scale factor
	       OmegaM : Omega Matter
	       OmegaL : Omega Lambda
	       h      : Hubble constant
    	"""
	aml=(OmegaM/OmegaL)**(1./3.)
	return 1./(h*Hubble) * 2./(3.* (1.-OmegaM)**0.5) * np.log((ascale/aml)**1.5 + (1.+(ascale/aml)**3.)**0.5) * UnitTime_in_Gyr

def GetLookBackTime(ascale, OmegaM=0.27,OmegaL=0.73,h=0.7):
	"""
	RETURNS: lookback time in simulation units
	INPUT: ascale : scale factor
	       OmegaM : Omega Matter
	       OmegaL : Omega Lambda
	       h      : Hubble constant
    	"""

	return GetTime(1.,OmegaM, OmegaL, h) - GetTime(ascale,OmegaM, OmegaL, h)
	
def CoolingRate_cgs(coor, h=0.7):
        """
        RETURNS: cooling rate in erg s^-1 g^-1 
        INPUT: coor   : cooling rate in code units (Gadget and Arepo both save du/dt, where u is energy per mass)
	       h      : Hubble constant
        """

	return coor * UnitEnergy_in_cgs *  UnitTime_in_s**(-1.0) * UnitMass_in_g**(-1.0) * h

def GetMachNumPressure(press_1, press_2, gamma):
        """
        RETURNS: Upstream Mach number from Rankine-Hugoniot condtions as a function of pressure 
        INPUT: press_1 : upstream pressure (in front of shock) 
               press_2 : downstream pressure (wake of shock)
               gamma : adiabatic index  
        """
	
	return np.sqrt(((gamma+1.)*press_2/press_1+(gamma-1.))/(2.*gamma))

def PhiNFW(r, M200, R200, c):
        """
        RETURNS: Potential of NFW profile 
        INPUT: r    : radius to evaluate 
               M200 : virial mass 
               R200 : virial radius
               c    : concentration 
        """

	rs=R200/c
	return -G*M200/(rs*(np.log(1+c)-c/(1+c)))*np.log(1+r/rs)/(r/rs)


def hubble_function(ascale, OmegaM=0.27,OmegaL=0.73):
        """
        RETURNS: hubble constant in simulation units
        INPUT: ascale : scale factor
               OmegaM : Omega Matter
               OmegaL : Omega Lambda
        """
	return Hubble * (OmegaM / (ascale**3) + (1 - OmegaM - OmegaL) / (ascale**2) + OmegaL);


def getAbundanceMatchingStellarMass_Moster_2013(Mhalo, redshift):
	"""
	RETURNS: expected stellar mass in M_sun 
	INPUT: halo mass in M_sun
	"""
	M10 = 11.590
	M11 = 1.195
	N10 = 0.0351
	N11 = -0.0247
	beta10 = 1.376
	beta11 = -0.826
	gamma10 = 0.608
	gamma11 = 0.329

	M10_sigma = 0.236
	M11_sigma = 0.353
	N10_sigma = 0.0058
	N11_sigma = 0.0069
	beta10_sigma = 0.153
	beta11_sigma = 0.225
	gamma10_sigma = 0.059
	gamma11_sigma = 0.173

	a = 1.0/(1.0+redshift)

	#old code
	#M1 = 10.0**(M10 + M11*(1.0-a))
	#N = N10 + N11*(1.0-a)
	#beta = beta10 + beta11*(1.0-a)
	#gamma = gamma10 + gamma11*(1.0-a)
	#Abest = Mhalo  *  2 * N * ((Mhalo/M1)**(-beta) + (Mhalo/M1)**gamma)**(-1)
	#return [Abest]

	#new code
	Mhalo = np.log10(Mhalo)
	M1 = M10+M11*(1.0-a)
	N  = N10+N11*(1.0-a)
	beta  = beta10+beta11*(1.0-a)
	gamma  = gamma10+gamma11*(1.0-a)
	smp= Mhalo+np.log10(2.0*N)-np.log10((10.0**(Mhalo-M1))**(-beta)+(10.0**(Mhalo-M1))**(gamma))
	eta = np.exp(np.log(10.)*(Mhalo-M1))
	alpha = eta**(-beta)+eta**gamma
	dmdM10 = (gamma*eta**gamma-beta*eta**(-beta))/alpha
	dmdM11 = (gamma*eta**gamma-beta*eta**(-beta))/alpha*(1.0-a)
	dmdN10 = np.log10(np.exp(1.0))/N
	dmdN11 = np.log10(np.exp(1.0))/N*(1.0-a)
	dmdbeta10 = np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**(-beta)
	dmdbeta11 = np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**(-beta)*(1.0-a)
	dmdgamma10 = -np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**gamma
	dmdgamma11 = -np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**gamma*(1.0-a)
	sigma = np.sqrt(dmdM10*dmdM10*M10_sigma*M10_sigma + dmdM11*dmdM11*M11_sigma*M11_sigma + dmdN10*dmdN10*N10_sigma*N10_sigma + dmdN11*dmdN11*N11_sigma*N11_sigma + dmdbeta10*dmdbeta10*beta10_sigma*beta10_sigma + dmdbeta11*dmdbeta11*beta11_sigma*beta11_sigma + dmdgamma10*dmdgamma10*gamma10_sigma*gamma10_sigma + dmdgamma11*dmdgamma11*gamma11_sigma*gamma11_sigma)
	sml = smp-sigma
	smu = smp+sigma
	smll = smp - 2*sigma
	smuu = smp + 2*sigma
	smlll = smp - 3*sigma
	smuuu = smp + 3*sigma
	return [10.0**smp, 10.0**sml, 10.0**smu, 10.0**smll, 10.0**smuu, 10.0**smlll, 10.0**smuuu]


def getAbundanceMatchingStellarMass_Kravtsov_2014(Mhalo):
        """
        RETURNS: expected stellar mass in M_sun 
        INPUT: halo mass in M_sun
        """


	M1      = 10**11.35
	epsilon = 10.0**(-1.642)
	alpha   = -1.779
	delta   = 4.394
	gamma   = 0.547
	def f(x):
		return -np.log10(10.0**(alpha*x)+1.0) + delta * (np.log10(1+np.exp(x)))**gamma/(1+np.exp(10**(-x)))

	return 10.0**(np.log10(epsilon*M1) + f(np.log10(Mhalo/M1)) - f(0))


def getAbundanceMatchingStellarMass_Guo_2010(Mhalo):
        """
        RETURNS: expected stellar mass in M_sun 
        INPUT: halo mass in M_sun
        """
	Abest = Mhalo  *  0.129 * ((Mhalo/10**11.4)**(-0.926) + (Mhalo/10.0**11.4)**(0.261))**(-2.440)
	return Abest


def getAbundanceMatchingStellarMass_Moster_2010(Mhalo):
        """
        RETURNS: expected stellar mass in M_sun 
        INPUT: halo mass in M_sun
        """
	Abest = Mhalo *  2*0.02820*((Mhalo/10.0**11.884)**(-(1.057)) + (Mhalo/10.0**11.884)**0.556)**(-1)
	return Abest

def getEddingtonBHAR(Mbh, radeff, h=0.7):
	return 4.0 * np.pi *Gravity * CLight * ProtonMass / (radeff * CLight**2.0 * Thompson) * Mbh * UnitTime_in_s / h

def getLuminsoties(mag, band):
	if (band=="V"):
		f_mag = 10.0**(mag/(-2.5))
		f_sun = 10.0**(mag_sun_V/(-2.5))
		return f_mag/f_sun 
