"""
util/self.py
  Physical units, conversions.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from util.helper import logZeroSafe, logZeroNaN, logZeroMin
from cosmo.util import correctPeriodicDistVecs

class units(object):
    """ Contains static methods which perform various unit conversions.
        Can also be instantiated with a redshift/sP, in which case contains the relevant unit 
        system and redshift-dependent constants.
    """
    # units (from parameter file, currently same for all runs)
    UnitLength_in_cm         = 3.085678e21   # 1.0 kpc
    UnitMass_in_g            = 1.989e43      # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0e5         # 1 km/sec

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
    boltzmann_keV     = 11604505.0      # Kelvin/KeV
    mass_proton       = 1.672622e-24    # cgs
    gamma             = 1.666666667     # 5/3
    hydrogen_massfrac = 0.76            # XH (solar)
    helium_massfrac   = 0.24            # Y (solar)
    mu                = 0.6             # for ionized primordial (e.g. hot halo gas)
    HubbleParam       = 0.7             # little h (All.HubbleParam), e.g. H0 in 100 km/s/Mpc
    Gravity           = 6.673e-8        # G in cgs, cm**3/g/s**2
    H0_kmsMpc         = 70.0            # km/s/Mpc
    H0_h1_s           = 3.24078e-18     # H0 (with h=1) in [1/s] (=H0_kmsMpc/HubbleParam/kpc_in_km)
    Z_solar           = 0.0127          # solar metallicity = (massZ/massTot) in the sun
    L_sun             = 3.839e33        # solar luminosity [erg/s]
    Msun_in_g         = 1.98892e33      # solar mass [g]
    c_cgs             = 2.9979e10       # speed of light in [cm/s]

    # code parameters
    CourantFac  = 0.3     # typical (used only in load:dt_courant)

    # derived constants (code units)
    H0          = None    # km/s/kpc (hubble constant at z=0)
    G           = None    # kpc (km/s)**2 / 1e10 msun
    rhoCrit     = None    # 1e10 msun / kpc**3 (critical density, z=0)

    # derived cosmology parameters
    f_b         = None     # baryon fraction

    # redshift dependent values (code units)
    H2_z_fact   = None     # H^2(z)
    H_z         = None     # hubble constant at redshift z [km/s/kpc]
    rhoCrit_z   = None     # critical density at redshift z
    scalefac    = None     # a=1/(1+z)

    # unit conversions
    s_in_yr     = 3.155693e7
    s_in_Myr    = 3.155693e13
    s_in_Gyr    = 3.155693e16
    pc_in_cm    = 3.085680e18
    Mpc_in_cm   = 3.085680e24
    kpc_in_km   = 3.085680e16

    # derived unit conversions
    kmS_in_kpcYr  = None
    kmS_in_kpcGyr = None
    
    def __init__(self, sP=None):
        """ Compute derived and redshift dependent units and values. """
        self._sP = sP

        # Mpc for lengths instead of the usual kpc?
        if self._sP.mpcUnits:
            self.UnitLength_in_cm = 3.085678e24 # 1000.0 kpc

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
        self.f_b = self._sP.omega_b / self._sP.omega_m

        # redshift dependent values (code units)
        if self._sP.redshift is not None:
            self.H2_z_fact = self._sP.omega_m*(1+self._sP.redshift)**3.0 + \
                              self._sP.omega_L + \
                              self._sP.omega_k*(1+self._sP.redshift)**2.0
            self.H_z       = self.H0 * np.sqrt(self.H2_z_fact)
            self.rhoCrit_z = self.rhoCrit * self.H2_z_fact
            self.scalefac  = 1.0 / (1+self._sP.redshift)

        # derived unit conversions
        self.kmS_in_kpcYr  = self.s_in_Myr / self.kpc_in_km / 1e6 # Myr->yr
        self.kmS_in_kpcGyr = self.s_in_Myr / self.kpc_in_km * 1e3 # Myr->Gyr

    # --- unit conversions to/from code units ---

    def codeMassToMsun(self, mass):
        """ Convert mass from code units (10**10 msun/h) to (msun). """
        mass_msun = mass.astype('float32') * (self.UnitMass_in_Msun) / self._sP.HubbleParam
        
        return mass_msun

    def codeMassToLogMsun(self, mass):
        """ Convert mass from code units (10**10 msun/h) to (log msun). """
        return logZeroSafe( self.codeMassToMsun(mass) )

    def codeMassToVirTemp(self, mass, meanmolwt=None, log=False):
        """ Convert from halo mass in code units to virial temperature in Kelvin, 
            at the specified redshift (Barkana & Loeb (2001) eqn.26). """
        assert self._sP.redshift is not None
        if not meanmolwt:
            meanmolwt = self.meanmolwt(Y=0.25, Z=0.0) # default is primordial

        # mass to msun
        mass_msun = mass.astype('float32') * self.UnitMass_in_g / self.Msun_in_g

        little_h = 1.0 # do not multiply by h since mass_msun is already over h

        omega_m_z = self._sP.omega_m * (1+self._sP.redshift)**3.0 / \
                    ( self._sP.omega_m*(1+self._sP.redshift)**3.0 + \
                      self._sP.omega_L + \
                      self._sP.omega_k*(1+self._sP.redshift)**2.0 )

        Delta_c = 18*np.pi**2 + 82*(omega_m_z-1.0) - 39*(omega_m_z-1.0)**2.0

        Tvir = 1.98e4 * (meanmolwt/0.6) * (mass_msun/1e8*little_h)**(2.0/3.0) * \
                        (self._sP.omega_m/omega_m_z * Delta_c / 18.0 / np.pi**2.0)**(1.0/3.0) * \
                        (1.0 + self._sP.redshift)/10.0 # K
             
        if log:
            Tvir = logZeroSafe(Tvir)
        return Tvir

    def logMsunToVirTemp(self, mass, meanmolwt=None, log=False):
        """ Convert halo mass (in log msun, no little h) to virial temperature at specified redshift. """
        assert self._sP.redshift is not None
        if not meanmolwt:
            meanmolwt = self.meanmolwt(Y=0.25, Z=0.0) # default is primordial

        # mass to msun
        mass_msun = 10.0**mass.astype('float32')

        omega_m_z = self._sP.omega_m * (1+self._sP.redshift)**3.0 / \
                    ( self._sP.omega_m*(1+self._sP.redshift)**3.0 + \
                      self._sP.omega_L + \
                      self._sP.omega_k*(1+self._sP.redshift)**2.0 )

        Delta_c = 18*np.pi**2 + 82*(omega_m_z-1.0) - 39*(omega_m_z-1.0)**2.0

        Tvir = 1.98e4 * (meanmolwt/0.6) * (mass_msun/1e8*self.HubbleParam)**(2.0/3.0) * \
                        (self._sP.omega_m/omega_m_z * Delta_c / 18.0 / np.pi**2.0)**(1.0/3.0) * \
                        (1.0 + self._sP.redshift)/10.0 # K
             
        if log:
            Tvir = logZeroSafe(Tvir)
        return Tvir

    def codeTempToLogK(self, temp):
        """ Convert temperature in code units (e.g. tracer temp output) to log Kelvin. """
        temp_k = temp.astype('float32') * self.UnitTemp_in_cgs

        return logZeroSafe(temp_k)

    def codeLengthToComovingKpc(self, x):
        """ Convert length/distance in code units to comoving kpc. """
        x_phys = np.array(x, dtype='float32') / self._sP.HubbleParam # remove little h factor
        x_phys *= (3.085678e21/self.UnitLength_in_cm) # account for non-kpc code lengths

        return x_phys

    def codeLengthToKpc(self, x):
        """ Convert length/distance in code units to physical kpc. """
        assert self._sP.redshift is not None

        return self.codeLengthToComovingKpc(x) * self.scalefac # comoving -> physical

    def physicalKpcToCodeLength(self, x):
        """ Convert a length in [pkpc] to code units [ckpc/h]. """
        assert self._sP.redshift is not None

        x_comoving = np.array(x, dtype='float32') / self.scalefac
        x_comoving /= (3.085678e21/self.UnitLength_in_cm) # account for non-kpc code lengths
        x_comoving *= self._sP.HubbleParam # add little h factor

        return x_comoving

    def particleCodeVelocityToKms(self, x):
        """ Convert velocity field (for cells/particles, not group properties) into km/s. """
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype='float32') * np.sqrt(self.scalefac)
        x_phys *= (1.0e5/self.UnitVelocity_in_cm_per_s) # account for non-km/s code units

        return x_phys

    def groupCodeVelocityToKms(self, x):
        """ Convert velocity vector (for groups, not subhalos nor particles) into km/s. """
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype='float32') / self.scalefac
        x_phys *= (1.0e5/self.UnitVelocity_in_cm_per_s) # account for non-km/s code units

        return x_phys

    def subhaloCodeVelocityToKms(self, x):
        """ Convert velocity vector (for subhalos, not groups nor particles) into km/s. """
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype='float32')
        x_phys *= (1.0e5/self.UnitVelocity_in_cm_per_s) # account for non-km/s code units

        return x_phys

    def particleCodeBFieldToGauss(self, b):
        """ Convert magnetic field 3-vector (for cells) into Gauss, input b is PartType0/MagneticField. """
        UnitMagneticField_in_cgs = np.float32( np.sqrt(self.UnitPressure_in_cgs) )

        b_gauss = b * self.HubbleParam # remove little h factor
        b_gauss /= self.scalefac**2.0 # convert 'comoving' into physical

        b_gauss *= UnitMagneticField_in_cgs # [Gauss] = [g^(1/2) * cm^(-1/2) * s^(-1)]
        return b_gauss

    def particleAngMomVecInKpcKmS(self, pos, vel, mass, haloPos, haloVel):
        """ Calculate particle angular momentum 3-vector in [kpc km/s] given input arrays of pos,vel,mass 
        and the halo CM position and velocity to compute relative to. Includes Hubble correction. """
        # make copies of input arrays
        gas_mass = self.codeMassToMsun( mass.astype('float32') )
        gas_pos  = pos.astype('float32')
        gas_vel  = vel.astype('float32')

        # calculate position, relative to subhalo center (pkpc)
        for i in range(3):
            if haloPos.ndim == 1: # scalar
                gas_pos[:,i] -= haloPos[i]
            else:
                gas_pos[:,i] -= haloPos[:,i]

        correctPeriodicDistVecs( gas_pos, self._sP )
        xyz = self.codeLengthToKpc( gas_pos )

        rad = np.sqrt( xyz[:,0]**2.0 + xyz[:,1]**2.0 + xyz[:,2]**2.0 ) # equals np.linalg.norm(xyz,2,axis=1)
        rad[rad == 0.0] = 1e-5

        # calculate momentum, correcting velocities for subhalo CM motion and hubble flow (Msun km/s)
        gas_vel = self.particleCodeVelocityToKms( gas_vel )

        for i in range(3):
            # SubhaloVel already peculiar, no scalefactor needed
            if haloVel.ndim == 1: # scalar
                gas_vel[:,i] -= haloVel[i]
            else:
                gas_vel[:,i] -= haloVel[:,i]

        v_H = self.H_z * rad # Hubble expansion velocity magnitude (km/s) at each position

        # add Hubble expansion velocity 3-vector at each position (km/s)
        for i in range(3):
            gas_vel[:,i] += (xyz[:,i] / rad * v_H)

        mom = np.zeros( (gas_mass.size,3), dtype='float32' )

        for i in range(3):
            mom[:,i] = gas_mass * gas_vel[:,i]

        # calculate angular momentum of each particle, rr x pp
        ang_mom = np.cross(xyz,mom)

        return ang_mom

    def particleSpecAngMomMagInKpcKmS(self, pos, vel, mass, haloPos, haloVel, log=False):
        """ Wrap particleAngMomVecInKpcKmS() to calculate particle *specific* angular momentum 
        *magnitude* in [kpc km/s] given input arrays. """
        ang_mom = self.particleAngMomVecInKpcKmS(pos, vel, mass, haloPos, haloVel)

        # magnitude
        ang_mom_mag = np.linalg.norm(ang_mom,2,axis=1)

        # specific
        gas_mass = self.codeMassToMsun( mass.astype('float32') )
        ang_mom_mag /= gas_mass

        if log:
            ang_mom_mag = logZeroSafe(ang_mom_mag)

        return ang_mom_mag

    def particleRadialVelInKmS(self, pos, vel, haloPos, haloVel):
        """ Calculate particle radial velocity in [km/s] (negative=inwards) given input arrays of pos,vel 
        and the halo CM position and velocity to compute relative to. Includes Hubble correction. """
        # make copies of input arrays
        gas_pos = pos.astype('float32')
        gas_vel = vel.astype('float32')

        # calculate position, relative to subhalo center (pkpc)
        for i in range(3):
            if haloPos.ndim == 1: # scalar
                gas_pos[:,i] -= haloPos[i]
            else:
                gas_pos[:,i] -= haloPos[:,i]

        correctPeriodicDistVecs( gas_pos, self._sP )

        xyz = self.codeLengthToKpc( gas_pos )
        rad = np.linalg.norm(xyz, 2, axis=1)

        # correct velocities for subhalo CM motion
        gas_vel = self.particleCodeVelocityToKms( gas_vel )

        for i in range(3):
            # SubhaloVel already peculiar, no scalefactor needed
            if haloVel.ndim == 1: # scalar
                gas_vel[:,i] -= haloVel[i]
            else:
                gas_vel[:,i] -= haloVel[:,i]

        # correct velocities for hubble flow (neglect mass growth term)
        vrad_noH = ( gas_vel[:,0] * xyz[:,0] + \
                     gas_vel[:,1] * xyz[:,1] + \
                     gas_vel[:,2] * xyz[:,2] ) / rad # radial velocity (km/s), negative=inwards

        v_H = self.H_z * rad # Hubble expansion velocity magnitude (km/s) at each position
        vrad = vrad_noH + v_H # radial velocity (km/s) with hubble expansion subtracted

        return vrad

    def codeDensToPhys(self, dens, cgs=False, numDens=False):
        """ Convert mass density comoving->physical and add little_h factors. 
            Input: dens in code units should have [10^10 Msun/h / (ckpc/h)^3] = [10^10 Msun h^2 / ckpc^3].
            Return: [10^10 Msun/kpc^3] or [g/cm^3 if cgs=True] or [1/cm^3 if cgs=True and numDens=True].
        """
        assert self._sP.redshift is not None
        if numDens and not cgs:
            raise Exception('Odd choice.')

        dens_phys = dens.astype('float32') * self.HubbleParam**2 / self.scalefac**3

        if cgs:
            dens_phys *= self.UnitDensity_in_cgs
        if numDens:
            dens_phys /= self.mass_proton
        return dens_phys

    def codeColDensToPhys(self, colDens, cgs=False, numDens=False, msunKpc2=False, totKpc2=False):
        """ Convert a mass column density [mass/area] from comoving -> physical and remove little_h factors.
            Input: colDens in code units should have [10^10 Msun/h / (ckpc/h)^2] = [10^10 Msun * h / ckpc^2].
            Return: [10^10 Msun/kpc^2] or 
                    [g/cm^2 if cgs=True] or 
                    [1/cm^2] if cgs=True and numDens=True, which is in fact [H atoms/cm^2]] or 
                    [Msun/kpc^2 if msunKpc2=True] or 
                    [[orig units]/kpc^2 if totKpc2=True].
        """
        assert self._sP.redshift is not None
        if numDens and not cgs:
            raise Exception('Odd choice.')
        if (msunKpc2 or totKpc2) and (numDens or cgs):
            raise Exception("Invalid combination.")

        # convert to 'physical code units' of 10^10 Msun/kpc^2
        colDensPhys = colDens.astype('float32') * self.HubbleParam / self.scalefac**2.0

        if cgs:
            UnitColumnDensity_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm**2.0
            colDensPhys *= UnitColumnDensity_in_cgs # g/cm^2
        if numDens:
            colDensPhys /= self.mass_proton # 1/cm^2
        if msunKpc2:
            colDensPhys *= (self.UnitMass_in_g/self.Msun_in_g) # remove 10^10 factor
            colDensPhys *= (3.085678e21/self.UnitLength_in_cm)**2.0 # account for non-kpc units
        if totKpc2:
            # non-mass quantity input as numerator, assume it did not have an h factor
            colDensPhys *= self.HubbleParam
            colDensPhys *= (3.085678e21/self.UnitLength_in_cm)**2.0 # account for non-kpc units

        return colDensPhys

    def UToTemp(self, u, nelec, log=False):
        """ Convert (U,Ne) pair in code units to temperature in Kelvin. """
        # hydrogen mass fraction default
        hmassfrac = self.hydrogen_massfrac

        # calculate mean molecular weight
        meanmolwt = 4.0/(1.0 + 3.0 * hmassfrac + 4.0* hmassfrac * nelec.astype('float32')) 
        meanmolwt *= self.mass_proton

        # calculate temperature (K)
        temp = u.astype('float32')
        temp *= (self.gamma-1.0) / self.boltzmann * \
                self.UnitEnergy_in_cgs / self.UnitMass_in_g * meanmolwt

        if log:
            temp = logZeroSafe(temp)
        return temp

    def TempToU(self, temp, log=False):
        """ Convert temperature in Kelvin to InternalEnergy (u) in code units. """
        if np.max(temp) <= 10.0:
            raise Exception("Error: input temp probably in log, check.")

        meanmolwt = 0.6 * self.mass_proton # ionized, T > 10^4 K

        # temp = (gamma-1.0) * u / units.boltzmann * units.UnitEnergy_in_cgs / units.UnitMass_in_g * meanmolwt
        u = temp.astype('float32')
        u *= self.boltzmann * self.UnitMass_in_g / \
            (self.UnitEnergy_in_cgs * meanmolwt * (self.gamma-1.0))

        if log:
            u = logZeroSafe(u)
        return u

    def coolingRateToCGS(self, coolrate):
        """ Convert code units (du/dt) to erg/s/g (cgs). """
        coolrate_cgs = coolrate.astype('float32')
        coolrate_cgs *= self.UnitEnergy_in_cgs * self.UnitTime_in_s**(-1.0) * \
                       self.UnitMass_in_g**(-1.0) * self.HubbleParam

        return coolrate_cgs

    def tracerEntToCGS(self, ent, log=False):
        """ Fix cosmological/unit system in TRACER_MC[MaxEnt], output in cgs [K cm^2]. """
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        # Note: dens=dens*a3inv but in the tracers only converted in dens^gamma not in the pressure
        # have to make this adjustment in loading tracers
        # for SFR, for gas and tracers, Pressure = GAMMA_MINUS1 * localSphP[i].Density * localSphP[i].Utherm;
        # for TRACER_MC, EntMax = SphP.Pressure / pow(SphP.Density * All.cf_a3inv, GAMMA);

        # fix Pressure
        ent_cgs = ent.astype('float32')
        ent_cgs *= a3inv * self.UnitPressure_in_cgs / self.boltzmann

        # fix Density
        ent_cgs /= (self.UnitDensity_in_cgs / self.mass_proton)**self.gamma

        if log:
            ent_cgs = logZeroSafe(ent_cgs)
        return ent_cgs

    def calcXrayLumBolometric(self, sfr, u, nelec, mass, dens, log=False):
        """ Following Navarro+ (1994) Eqn. 6 the most basic estimator of bolometric X-ray luminosity 
        in [erg/s] for individual gas cells, based only on their density and temperature. Assumes 
        simplified (primordial) high-temp cooling function, and only free-free (bremsstrahlung) 
        emission contribution from T>10^6 Kelvin gas. All inputs in code units. """
        hmassfrac = self.hydrogen_massfrac

        # calculate mean molecular weight
        meanmolwt = 4.0/(1.0 + 3.0 * hmassfrac + 4.0* hmassfrac * nelec.astype('float64')) 
        meanmolwt *= self.mass_proton

        # calculate temperature (K)
        temp = u.astype('float32')
        temp *= (self.gamma-1.0) / self.boltzmann * self.UnitEnergy_in_cgs / self.UnitMass_in_g * meanmolwt

        # Eqn. 6
        mass_g = mass.astype('float32') * (self.UnitMass_in_g) / self._sP.HubbleParam
        dens_g_cm3 = self.codeDensToPhys(dens, cgs=True) # g/cm^3

        Lx = 1.2e-24 / (meanmolwt)**2.0 * mass_g *  dens_g_cm3 * np.sqrt(temp/self.boltzmann_keV)

        # clip any cells on eEOS (SFR>0) to zero
        w = np.where(sfr > 0.0)
        Lx[w] = 0.0

        # implement a linear ramp from log(T)=6.0 to log(T)=5.8 over which we clip to zero
        temp = np.log10(temp)
        Lx *= np.clip( (temp-5.8)/0.2, 0.0, 1.0 )

        if log:
            Lx = logZeroSafe(Lx)
        return Lx # note: CANNOT truncate to float32 this clips high values at inf

    def calcEntropyCGS(self, u, dens, log=False):
        """ Calculate entropy as P/rho^gamma, converting rho from comoving to physical. """
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        # cosmological and unit system conversions
        dens_phys = dens.astype('float32') * self.HubbleParam**2.0 # remove all little h factors

        # pressure in [K/cm^3]
        pressure = u.astype('float32')
        pressure *= (self.gamma-1.0) * dens_phys * a3inv * \
                   self.UnitPressure_in_cgs / self.boltzmann

        # entropy in [K cm^2]
        dens_fac = self.UnitDensity_in_cgs/self.mass_proton*a3inv
        entropy  = pressure / (dens_phys * dens_fac)**self.gamma
      
        if log:
            entropy = logZeroSafe(entropy)
        return entropy

    def calcPressureCGS(self, u, dens, log=False):
        """ Calculate pressure as (gamma-1)*u*rho in physical 'cgs' [K/cm^3] units. """
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        dens_phys = dens.astype('float32') * self.HubbleParam**2.0 # remove all little h factors

        pressure = u.astype('float32') * ( dens_phys.astype('float32') * a3inv )
        pressure *= (self.gamma-1.0)

        # convert to CGS = 1 barye (ba) = 1 dyn/cm^2 = 0.1 Pa = 0.1 N/m^2 = 0.1 kg/m/s^2
        # and divide by boltzmann's constant -> [K/cm^3]
        pressure *= self.UnitPressure_in_cgs / self.boltzmann

        if log:
            pressure = logZeroSafe(pressure)
        return pressure

    def calcMagneticPressureCGS(self, b, log=False):
        """ Calculate magnetic pressure as B^2/8/pi in physical 'cgs' [K/cm^3] units. """

        # input b is PartType0/MagneticField 3-vector (code units)
        b = self.particleCodeBFieldToGauss(b) # to physical Gauss

        # magnetic pressure P_B in CGS units of [dyn/cm^2]
        P_B = (b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2]) / (8*np.pi) 

        P_B /= self.boltzmann # divide by boltzmann's constant -> [K/cm^3]

        if log:
            P_B = logZeroSafe(P_B)
        return P_B

    def calcSoundSpeedKmS(self, u, dens, log=False):
        """ Calculate sound speed as sqrt(gamma*Pressure/Density) in physical km/s. """
        pres = (self.gamma-1.0) * dens * u
        csnd = np.sqrt( self.gamma * pres / dens ) # code units, all scalefac and h cancel
        csnd = csnd.astype('float32')

        csnd *= (1.0e5/self.UnitVelocity_in_cm_per_s) # account for non-km/s code units

        if log:
            csnd = logZeroSafe(csnd)
        return csnd

    def codeDensToCritRatio(self, rho, baryon=None, log=False):
        """ Normalize code density by the critical (total/baryonic) density at some redshift. """
        assert self._sP.redshift is not None
        if baryon is None:
            raise Exception("Specify baryon True or False, note... change of behavior.")

        rho_crit = self.rhoCrit_z
        if baryon:
            rho_crit *= self._sP.omega_b

        ratio_crit = rho.astype('float32') / rho_crit

        if log:
            ratio_crit = logZeroSafe(ratio_crit)
        return ratio_crit

    def critRatioToCodeDens(self, ratioToCrit, baryon=None):
        """ Convert a ratio of the critical density at some redshift to a code density. """
        assert self._sP.redshift is not None
        if baryon is None:
            raise Exception("Specify baryon True or False, note... change of behavior.")

        code_dens = ratioToCrit.astype('float32') * self.rhoCrit_z

        if baryon:
            code_dens *= self._sP.omega_b

        return code_dens

    def codeMassToVirEnt(self, mass, log=False):
        """ Given a total halo mass, return a S200 (e.g. Pvir/rho_200crit^gamma). """
        virTemp = self.codeMassToVirTemp(mass, log=False)
        virU = self.TempToU(virTemp)
        r200crit = self.critRatioToCodeDens(np.array(200.0), baryon=True)

        s200 = self.calcEntropyCGS(virU, r200crit, log=log)

        return s200.astype('float32')

    def codeMassToVirVel(self, mass):
        """ Given a total halo mass [in code units], return a virial velocity (V200) in physical [km/s]. """
        assert self._sP.redshift is not None

        r200 = ( self.G * mass / 100.0 / self.H_z**2.0 )**(1.0/3.0)
        v200 = np.sqrt( self.G * mass / r200 )

        return v200.astype('float32')

    def codeM200R200ToV200InKmS(self, m200, r200):
        """ Given a (M200,R200) pair of a FoF group, compute V200 in physical [km/s]. """
        v200 = np.sqrt( self.G * m200 / r200 )
        v200 *= (1.0e5/self.UnitVelocity_in_cm_per_s) # account for non-km/s code units
        return v200

    def metallicityInSolar(self, metal, log=False):
        """ Given a code metallicity (M_Z/M_total), convert to value with respect to solar. """
        metal_solar = metal.astype('float32') / self.Z_solar

        metal_solar[metal_solar < 0] = 0.0 # clip possibly negative Illustris values at zero

        if log:
            return np.log10(metal_solar)
        return metal_solar

    def codeTimeStepToYears(self, TimeStep, Gyr=False):
        """ Convert a TimeStep/TimeStepHydro/TimeStepGrav (e.g. an integer times All.Timebase_interval) 
        for a comoving run to a physical time in years. """
        dtime = TimeStep / (np.sqrt(self.H2_z_fact) * self.H0_h1_s)
        dtime /= self.HubbleParam
        dtime /= self.s_in_yr

        if Gyr: dtime /= 1e9

        return dtime

    def scalefacToAgeLogGyr(self, scalefacs):
        """ Convert scalefactors (e.g. GFM_StellarFormationTime) to age in log(Gyr) given the 
        current age of the universe as specified by sP.redshift. """
        age = scalefacs.astype('float32')
        age.fill(np.nan) # set wind to age=nan
        w_stars = np.where(scalefacs >= 0.0)

        curUniverseAgeGyr = self.redshiftToAgeFlat(self._sP.redshift)
        birthRedshift = 1.0/scalefacs - 1.0
        birthRedshift = logZeroMin( curUniverseAgeGyr - self.redshiftToAgeFlat(birthRedshift) )

        age[w_stars] = birthRedshift[w_stars]
        return age

    def codeEnergyToErg(self, energy, log=False):
        """ Convert energy from code units (unitMass*unitLength^2/unitType^2) to [erg]. """
        energy_cgs = energy.astype('float32') * self.UnitEnergy_in_cgs / self._sP.HubbleParam
        
        if log:
            return logZeroNaN(energy_cgs)
        return energy_cgs

    # --- other ---

    def redshiftToAgeFlat(self, z):
        """ Calculate age of the universe [Gyr] at the given redshift (assuming flat cosmology).
            Analytical formula from Peebles, p.317, eq 13.2. """
        redshifts = np.array(z)
        if redshifts.ndim == 0:
            redshifts = np.array([z])

        with np.errstate(invalid='ignore'): # ignore nan comparison RuntimeWarning
            w = np.where( (redshifts >= 0.0) & np.isfinite(redshifts) )

        age = np.zeros(redshifts.size, dtype='float32')
        age.fill(np.nan) # leave negative/nan redshifts unset
        
        arcsinh_arg = np.sqrt( (1-self._sP.omega_m)/self._sP.omega_m ) * (1+redshifts[w])**(-3.0/2.0)
        age[w] = 2 * np.arcsinh(arcsinh_arg) / (self.H0_kmsMpc * 3 * np.sqrt(1-self._sP.omega_m))
        age[w] *= 3.085678e+19 / 3.15567e+7 / 1e9 # Gyr

        if len(age) == 1:
            return age[0]
        return age

    def ageFlatToRedshift(self, age):
        """ Calculate redshift from age of the universe [Gyr] (assuming flat cosmology).
            Inversion of analytical formula from redshiftToAgeFlat(). """
        with np.errstate(invalid='ignore'): # ignore nan comparison RuntimeWarning
            w = np.where( (age >= 0.0) & np.isfinite(age) )

        z = np.zeros(len(age), dtype='float32')
        z.fill(np.nan)

        sinh_arg = (self.H0_kmsMpc * 3 * np.sqrt(1-self._sP.omega_m))
        sinh_arg *= 3.15567e+7 * 1e9 * age[w] / 2.0 / 3.085678e+19

        z[w] = np.sinh(sinh_arg) / np.sqrt( (1-self._sP.omega_m)/self._sP.omega_m )
        z[w] = z[w]**(-2.0/3.0) - 1

        if len(z) == 1:
            return z[0]
        return z

    def meanmolwt(self, Y, Z):
        """ Mean molecular weight, from Monaco+ (2007) eqn 14, for hot halo gas.
            Y = helium fraction (0.25)
            Z = metallicity (non-log metal mass/total mass)
        """
        mu = 4.0 / (8 - 5*Y - 6*Z)
        return mu
