"""
util/self.py
  Physical units, conversions.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

class units(object):
    """ Contains static methods which perform various unit conversions.
        Can also be instantiated with a redshift/sP, in which case contains the relevant unit 
        system and redshift-dependent constants.
    """
    # units (from parameter file, currently same for all runs)
    UnitLength_in_cm         = 3.085678**21       # 1.0 kpc
    UnitMass_in_g            = 1.989 * 10.0**43   # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0**5             # 1 km/sec

    # derived units
    UnitTime_in_s       = None
    UnitDensity_in_cgs  = None
    UnitPressure_in_cgs = None
    UnitEnergy_in_cgs   = None
    UnitTemp_in_cgs     = None

    # non-cgs units
    UnitMass_in_Msun  = None
    UnitTime_in_yr    = None

    # constants
    boltzmann         = 1.380650e-16    # cgs (erg/K)
    mass_proton       = 1.672622e-24    # cgs
    gamma             = 1.666666667     # 5/3
    hydrogen_massfrac = 0.76            # XH (solar)
    helium_massfrac   = 0.25            # Y (solar)
    mu                = 0.6             # for ionized primordial (e.g. hot halo gas)
    HubbleParam       = 0.7             # little h (All.HubbleParam), e.g. H0 in 100 km/s/Mpc
    Gravity           = 6.673e-8        # G in cgs, cm**3/g/s**2
    H0_kmsMpc         = 70.0            # km/s/Mpc

    # derived constants (code units)
    H0          = None    # km/s/kpc (hubble constant at z=0)
    G           = None    # kpc (km/s)**2 / 1e10 msun
    rhoCrit     = None    # 1e10 msun / kpc**3 (critical density, z=0)

    # derived cosmology parameters
    f_b         = None     # baryon fraction

    # redshift dependent values (code units)
    H2_z_fact   = None     # H^2(z)
    H_z         = None     # hubble constant at redshift z
    rhoCrit_z   = None     # critical density at redshift z
    scalefac    = None     # a=1/(1+z)

    # unit conversions
    s_in_yr     = 3.155693e7
    s_in_Myr    = 3.155693e13
    s_in_Gyr    = 3.155693e16
    Msun_in_g   = 1.98892 * 10.0**33
    pc_in_cm    = 3.085680e18
    Mpc_in_cm   = 3.085680e24
    kpc_in_km   = 3.085680e16

    # derived unit conversions
    kmS_in_kpcYr  = None
    kmS_in_kpcGyr = None
    
    def __init__(self, sP=None):
        """ Compute derived and redshift dependent units and values. """
        # derived units
        self.UnitTime_in_s       = self.UnitLength_in_cm / self.UnitVelocity_in_cm_per_s
        self.UnitDensity_in_cgs  = self.UnitMass_in_g / self.UnitLength_in_cm**3.0
        self.UnitPressure_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm / self.UnitTime_in_s**2.0
        self.UnitEnergy_in_cgs   = self.UnitMass_in_g * self.UnitLength_in_cm**2.0 / self.UnitTime_in_s**2.0
        self.UnitTemp_in_cgs     = self.UnitEnergy_in_cgs / self.UnitMass_in_g

        # non-cgs units
        self.UnitMass_in_Msun    = self.UnitMass_in_g / self.Msun_in_g
        self.UnitTime_in_yr      = self.UnitTime_in_s / self.s_in_yr

        # derived constants (in code units)
        self.H0 = self.HubbleParam * 100 * 1e5 / (self.Mpc_in_cm) / \
                   self.UnitVelocity_in_cm_per_s * self.UnitLength_in_cm
        self.G  = self.Gravity / self.UnitLength_in_cm**3.0 * self.UnitMass_in_g * self.UnitTime_in_s**2.0

        self.rhoCrit = 3.0 * self.H0**2.0 / (8.0*np.pi*self.G) #code, z=0

        # derived cosmology parameters
        self.f_b = sP.omega_b / sP.omega_m

        # redshift dependent values (code units)
        self._sP = sP

        if sP.redshift is not None:
            self.H2_z_fact = sP.omega_m*(1+sP.redshift)**3.0 + \
                              sP.omega_L + \
                              sP.omega_k*(1+sP.redshift)**2.0
            self.H_z       = self.H0 * np.sqrt(self.H2_z_fact)
            self.rhoCrit_z = self.rhoCrit * self.H2_z_fact
            self.scalefac  = 1.0 / (1+sP.redshift)

        # derived unit conversions
        self.kmS_in_kpcYr  = self.s_in_Myr / self.kpc_in_km / 1e6 # Myr->yr
        self.kmS_in_kpcGyr = self.s_in_Myr / self.kpc_in_km * 1e3 # Myr->Gyr

    # --- unit conversions to/from code units ---

    @staticmethod
    def codeMassToLogMsun(mass):
        """ Convert mass from code units (10**10 msun/h) to (log msun) self. """
        mass_msun = np.array(mass, dtype='float32')
        mass_msun *= np.float32(self.UnitMass_in_g / self.Msun_in_g)
        mass_msun /= self._sP.HubbleParam
        
        return self.logZeroSafe(mass_msun)

    @staticmethod
    def codeMassToVirTemp(mass, meanmolwt=None, log=False):
        """ Convert from halo mass in code units to virial temperature in Kelvin, 
            at the specified redshift (Barkana & Loeb (2001) eqn.26). """
        if sP.redshift is None:
            raise Exception("Need redshift.")
        if not meanmolwt:
            meanmolwt = units.meanmolwt(Y=0.25, Z=0.0) # default is primordial

        # mass to msun
        mass_msun = mass * sP.units.UnitMass_in_g / sP.units.Msun_in_g

        little_h = 1.0 # do not multiply by h since mass_msun is already over h

        omega_m_z = sP.omega_m * (1+sP.redshift)**3.0 / \
                    ( sP.omega_m*(1+sP.redshift)**3.0 + sP.omega_L + sP.omega_k*(1+sP.redshift)**2.0 )

        Delta_c = 18*np.pi**2 + 82*(omega_m_z-1.0) - 39*(omega_m_z-1.0)**2.0

        Tvir = 1.98e4 * (meanmolwt/0.6) * (mass_msun/1e8*little_h)**(2.0/3.0) * \
                        (sP.omega_m/omega_m_z * Delta_c / 18.0 / np.pi**2.0)**(1.0/3.0) * \
                        (1.0 + sP.redshift)/10.0 # K
             
        if log: Tvir = logZeroSafe(Tvir)
        return Tvir

    @staticmethod
    def logMsunToVirTemp(mass, sP, meanmolwt=None, log=False):
        """ Convert halo mass (in log msun, no little h) to virial temperature at specified redshift. """
        if sP.redshift is None:
            raise Exception("Need redshift.")
        if not meanmolwt:
            meanmolwt = units.meanmolwt(Y=0.25, Z=0.0) # default is primordial

        # mass to msun
        mass_msun = 10.0**mass

        omega_m_z = sP.omega_m * (1+sP.redshift)**3.0 / \
                    ( sP.omega_m*(1+sP.redshift)**3.0 + sP.omega_L + sP.omega_k*(1+sP.redshift)**2.0 )

        Delta_c = 18*np.pi**2 + 82*(omega_m_z-1.0) - 39*(omega_m_z-1.0)**2.0

        Tvir = 1.98e4 * (meanmolwt/0.6) * (mass_msun/1e8*sP.units.HubbleParam)**(2.0/3.0) * \
                        (sP.omega_m/omega_m_z * Delta_c / 18.0 / np.pi^2.0)**(1.0/3.0) * \
                        (1.0 + redshift)/10.0 # K
             
        if log: Tvir = logZeroSafe(Tvir)
        return Tvir

    @staticmethod
    def codeTempToLogK(temp, sP):
        """ Convert temperature in code units (e.g. tracer temp output) to log Kelvin. """
        temp_k = temp * sP.units.UnitTemp_in_cgs
        return logZeroSafe(temp_k)

    @staticmethod
    def codeDensToPhys(dens, sP, cgs=False):
        """ Convert density comoving->physical and add little_h factors. """
        if sP.redshift is None:
            raise Exception("Need redshift.")

        dens_phys = dens * sP.units.HubbleParam**2 / sP.units.scalefac**3

        if cgs:
            dens_phys *= sP.units.UnitDensity_in_cgs

        return dens_phys

    @staticmethod
    def nH0ToPhys(nh0, dens, sP, cgs):
        """ Convert (NeutralHydrogenAbundance,Density) pair from code units to mass density of 
            neutral hydrogen, optionally in cgs units. """
        dens_phys = units.codeDensToPhys(dens, sP, cgs=cgs)
        dens_phys *= sP.units.hydrogen_massfrac # hydrogen mass density (code or cgs units)
        # note: hydrogen number density = dens_phys / sP.units.proton_mass

        dens_phys *= nh0 # neutral hydrogen mass density (code or cgs units)
        # note: H+ number density = (H number density) - (H0 number density)

        return dens_phys

    @staticmethod
    def UToTemp(u, nelec, sP, log=False):
        """ Convert (U,Ne) pair in code units to temperature in Kelvin. """
        # hydrogen mass fraction default
        hmassfrac = sP.units.hydrogen_massfrac

        # calculate mean molecular weight
        meanmolwt = 4.0/(1.0 + 3.0 * hmassfrac + 4.0* hmassfrac * nelec) * sP.units.mass_proton

        # calculate temperature (K)
        temp = (sP.units.gamma-1.0) * u / sP.units.boltzmann * sP.units.UnitEnergy_in_cgs / \
               sP.units.UnitMass_in_g * meanmolwt

        if log: temp = logZeroSafe(temp)
        return temp

    @staticmethod
    def TempToU(temp, sP, log=False):
        """ Convert temperature in Kelvin to InternalEnergy (u) in code units. """
        if np.max(temp) <= 10.0:
            raise Exception("Error: input temp probably in log, check.")

        meanmolwt = 0.6 * sP.units.mass_proton # ionized, T > 10^4 K

        # temp = (gamma-1.0) * u / units.boltzmann * units.UnitEnergy_in_cgs / units.UnitMass_in_g * meanmolwt
        u = temp * sP.units.boltzmann * sP.units.UnitMass_in_g / \
            (sP.units.UnitEnergy_in_cgs * meanmolwt * (sP.units.gamma-1.0))

        if log: u = logZeroSafe(u)
        return u

    @staticmethod
    def codeMassToVirEnt(mass, sP, log=False):
        """ Given a total halo mass, return a S200 (e.g. Pvir/rho_200crit^gamma). """
        virTemp = codeMassToVirTemp(mass, sP=sP, log=False)
        virU = convertTempToU(virTemp)
        r200crit = critBaryonRatioToCode(200.0, sP=sP)

        s200 = calcEntropyCGS(virU, r200crit, sP=sP, log=log)

        return s200

    @staticmethod
    def coolingRateToCGS(coolrate, sP):
        """ Convert code units (du/dt) to erg/s/g (cgs). """
        coolrate_cgs = coolrate * sP.units.UnitEnergy_in_cgs * sP.units.UnitTime_in_s**(-1.0) * \
                       sP.units.UnitMass_in_g**(-1.0) * sP.units.HubbleParam

        return coolrate_cgs

    @staticmethod
    def tracerEntToCGS(ent, sP, log=False):
        """ Fix cosmological/unit system in TRACER_MC[MaxEnt], output in cgs [K cm^2]. """
        if sP.redshift is None:
            raise Exception("Need redshift.")

        a3inv = 1.0 / sP.units.scalefac**3.0

        # Note: dens=dens*a3inv but in the tracers only converted in dens^gamma not in the pressure
        # have to make this adjustment in loading tracers
        # for SFR, for gas and tracers, Pressure = GAMMA_MINUS1 * localSphP[i].Density * localSphP[i].Utherm;
        # for TRACER_MC, EntMax = SphP.Pressure / pow(SphP.Density * All.cf_a3inv, GAMMA);

        # fix Pressure
        ent *= a3inv * sP.units.UnitPressure_in_cgs / sP.units.boltzmann

        # fix Density
        ent /= (sP.units.UnitDensity_in_cgs / sP.units.mass_proton)**sP.units.gamma

        if log: ent = logZeroSafe(ent)
        return ent

    @staticmethod
    def calcEntropyCGS(u, dens, sP, log=False):
        """ Calculate entropy as P/rho^gamma, converting rho from comoving to physical. """
        if sP.redshift is None:
            raise Exception("Need redshift.")

        a3inv = 1.0 / sP.units.scalefac**3.0

        # cosmological and unit system conversions
        dens *= sP.units.HubbleParam**2.0 # remove all little h factors

        # pressure in [K/cm^3]
        pressure = (sP.units.gamma-1.0) * u * dens * a3inv * sP.units.UnitPressure_in_cgs / sP.units.boltzmann

        # entropy in [K cm^2]
        entropy  = pressure / (dens * sP.units.UnitDensity_in_cgs/sP.units.mass_proton*a3inv)**sP.units.gamma
      
        if log: entropy = logZeroSafe(entropy)
        return entropy

    @staticmethod
    def calcPressureCGS(u, dens, sP, log=False):
        """ Calculate pressure as (gamma-1)*u*rho in cgs units, converting rho from comoving to physical. """
        if sP.redshift is None:
            raise Exception("Need redshift.")

        a3inv = 1.0 / sP.units.scalefac**3.0

        pressure = (sP.units.gamma-1.0) * u * (dens*a3inv)

        # convert to CGS = 1 barye (ba) = 1 dyn/cm^2 = 0.1 Pa = 0.1 N/m^2 = 0.1 kg/m/s^2
        # and divide by boltzmann's constant -> [K/cm^3]
        pressure *= sP.units.UnitPressure_in_cgs / sP.units.boltzmann

        if log: pressure = logZeroSafe(pressure)
        return pressure

    @staticmethod
    def codeDensToCritRatio(rho, sP, baryon=None, log=False):
        """ Normalize code density by the critical (total/baryonic) density at some redshift. """
        if sP.redshift is None:
            raise Exception("Need redshift.")
        if baryon is None:
            raise Exception("Specify baryon True or False, note... change of behavior.")

        rho_crit = sP.units.rhoCrit_z
        if baryon:
            rho_crit *= sP.units.omega_b

        if log: return logZeroSafe(rho/rho_crit)
        return rho/rho_crit

    @staticmethod
    def critRatioToCodeDens(ratioToCrit, sP, baryon=None):
        """ Convert a ratio of the critical density at some redshift to a code density. """
        if sP.redshift is None:
            raise Exception("Need redshift.")
        if baryon is None:
            raise Exception("Specify baryon True or False, note... change of behavior.")

        if baryon:
            return ratioToCritB * sP.units.omega_b * sP.units.rhoCrit_z

        return ratioToCrit * sP.units.rhoCrit_z

    @staticmethod
    def redshiftToAge(z, sP):
        """ Calculate age of the universe [Gyr] at the given redshift (numerically, arbitrary cosmology). """
        raise Exception("TODO")

    @staticmethod
    def redshiftToAgeFlat(z, sP):
        """ Calculate age of the universe [Gyr] at the given redshift (assuming flat cosmology).
            Analytical formula from Peebles, p.317, eq 13.2. """
        w = np.where(z >= 0.0)

        age = np.zeros(len(z)) - 100.0 # negative indicates not set

        age[w] = 2*np.asinh(np.sqrt( (1-sP.units.omega_m)/sP.units.omega_m ) * (1+z[w])**(-3.0/2.0)) / \
                   (sP.units.H0_kmsMpc * 3 * np.sqrt(1-sP.units.omega_m))
        age[w] *= 3.085678e+19 / 3.15567e+7 / 1e9 # Gyr

        if len(age) == 1:
            return age[0]
        return age

    # --- other ---

    def meanmolwt(self, Y, Z):
        """ Mean molecular weight, from Monaco+ (2007) eqn 14, for hot halo gas.
            Y = helium fraction (0.25)
            Z = metallicity (non-log metal mass/total mass)
        """
        mu = 4.0 / (8 - 5*Y - 6*Z)
        return mu

    def logZeroSafe(self, x):
        """ Take log of input variable or array, keeping zeros at zero. """
        if x.ndim:
            # array
            w = np.where(x == 0.0)
            x[w] = 1.0
        else:
            # single scalar
            if x == 0.0:
                x = 1.0

        return np.log10(x)