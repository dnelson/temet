"""
Various unit conversion and cosmological calculation utilities.

Includes conversion from 'code units' to physical units, conversions between different unit systems, and
derivations of common halo properties (e.g. virial temperature) and cosmological quantities
(e.g. age at a given redshift).
"""

import numpy as np

from ..util.helper import logZeroMin, logZeroNaN, logZeroSafe


class units:
    """Contains static methods which perform various unit conversions.

    Can also be instantiated with a redshift/sP, in which case contains the relevant unit
    system and redshift-dependent constants.
    """

    # units (default unit system)
    UnitLength_in_cm = 3.085678e21  # 1.0 kpc
    UnitMass_in_g = 1.989e43  # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0e5  # 1 km/sec

    UnitLength_str = r"ckpc/h"
    UnitMass_str = r"10$^{10}$ M$_{\rm sun}$/h"
    UnitVelocity_str = r"km/s"

    # derived units
    UnitTime_in_s = None
    UnitDensity_in_cgs = None
    UnitPressure_in_cgs = None
    UnitEnergy_in_cgs = None
    UnitTemp_in_cgs = None

    # non-cgs units
    UnitMass_in_Msun = None
    UnitTime_in_yr = None

    # constants
    boltzmann = 1.380650e-16  # cgs (erg/K)
    boltzmann_JK = 1.380650e-23  # joules/Kelvin
    boltzmann_keV = 11604505.0  # Kelvin/KeV
    planck_erg_s = 6.626e-27  # Planck constant, [erg*s]
    planck_eV_s = 4.135668e-15  # Planck constant, [eV*s]
    hc_kev_ang = 12.3981689  # h*c, [keV*angstrom]
    mass_proton = 1.672622e-24  # cgs
    mass_electron = 9.1095e-28  # cgs
    gamma = 1.666666667  # 5/3
    hydrogen_massfrac = 0.76  # XH (solar)
    helium_massfrac = 0.24  # Y (solar)
    mu = 0.6  # for ionized primordial (e.g. hot halo gas)
    Gravity = 6.6738e-8  # G in cgs, cm**3/g/s**2 (== GRAVITY)
    H0_h1_s = 3.24078e-18  # H0 (with h=1) in [1/s] (=H0_kmsMpc/HubbleParam/kpc_in_km) (=HUBBLE in Arepo)
    Z_solar = 0.0127  # solar metallicity = (massZ/massTot) in the sun (TNG/MCST)
    L_sun = 3.839e33  # solar luminosity [erg/s]
    Msun_in_g = 1.98892e33  # solar mass [g]
    c_cgs = 2.9979e10  # speed of light in [cm/s]
    c_km_s = 2.9979e5  # speed of light in [km/s]
    c_kpc_Gyr = 3.06595e5  # speed of light in [kpc/Gyr]
    sigma_thomson = 6.6524e-25  # thomson cross section [cm^2]
    electron_charge = 4.8032e-10  # esu [=cm*sqrt(dyne) = g^(1/2)cm^(3/2)s^(-1)]
    rydberg_ang = 0.00109737  # rydberg constant in 1/angstrom
    rydberg_freq = 3.28984e15  # Hz, i.e. rydberg constant * c
    Habing = 5.29e-14  # erg cm^-3

    # derived constants
    mag2cgs = None  # Lsun/Hz to cgs [erg/s/cm^2] at d=10pc
    c_ang_per_sec = None  # speed of light in [Angstroms/sec]

    # code/model parameters
    CourantFac = 0.3  # typical (used only in load:dt_courant)
    BH_eps_r = 0.2  # BH radiative efficiency, unchanged in Illustris and TNG models
    BH_eps_f_high = 0.10  # BH high-state efficiency, TNG fiducial model (is 0.05 for Illustris)
    BH_f_thresh = 0.05  # multiplier on the star-formation threshold modulating eps_f_low, TNG fiducial model
    N_SNII = 0.0118008  # winds: N_SNII per Msun formed (IMF integrated between CCSN mass limits), TNG fiducial
    E_SNII51 = 1.0  # winds: available energy per CC SN in units of 10^51 erg (=unity in Illustris/TNG)
    winds_tau = 0.1  # winds: thermal energy fraction
    winds_fZ = 0.25  # winds: Z-dependence reduction factor
    winds_Zref = 0.002  # winds: Z-dependence reference metallicity
    winds_gamma_Z = 2.0  # winds: Z-dependence reduction power
    winds_e = 3.6  # winds: energy factor, TNG fiducial model (1.09 in Illustris)
    winds_kappa = 7.4  # winds: velocity factor, TNG fiducial model (3.7 in Illustris)
    winds_vmin = 350.0  # winds: injection velocity floor, TNG fiducial model (0.0 in Illustris)

    # SFR/SH03 model parameters
    PhysDensThresh = 7.54654e-4  # SF threshold density (from CritOverDensity, code units, comoving) = 0.232 h^2/cm^3
    sh03_nH_thresh = 0.13  # (from CritOverDensity), SF density threshold [H atoms/cm^3] (TNG/Illustris value)
    sh03_A0 = 573.0  # FactorEVP, cloud evaporation efficiency (TNG/Illustris value)
    sh03_T_SN = 5.73e7  # TempSupernova, Kelvin (TNG/Illustris value)
    sh03_T_c = 1000.0  # TempClouds, Kelvin (TNG/Illustris value)
    sh03_beta = 0.22578  # FactorSN, mass fraction of stars which promptly explode as SNe (TNG value, 0.1 for Illustris)
    sh03_t_star = 3.27665  # Gyr (from MaxSfrTimescale) (TNG/Illustris value)

    # derived constants (code units without h factors)
    H0 = None  # km/s/kpc (hubble constant at z=0)
    H0_kmsMpc = None  # km/s/Mpc
    G = None  # kpc (km/s)**2 / 1e10 msun (== All.G)
    rhoCrit = None  # 1e10 msun / kpc**3 (critical density, z=0)
    rhoCrit_msunMpc3 = None  # msun / mpc^3 (critical density, z=0)
    Hubble = None  # 0.1 (== All.Hubble)

    # derived cosmology parameters
    f_b = None  # baryon fraction

    # redshift dependent values (code units without h factors)
    H2_z_fact = None  # H^2(z)
    Omega_z = None  # Omega_m(z) = Omega_m(0) * (1+z)^3 / H^2(z)
    H_z = None  # hubble constant at redshift z [km/s/kpc]
    H_of_a = None  # hubble constant, internal units (as computed in Arepo)
    rhoCrit_z = None  # critical density at redshift z
    scalefac = None  # a=1/(1+z)

    # as above but strict code units, e.g. with h factors
    rhoBack = None  # = 3 * Omega0 * All.Hubble**2 / (8*pi*All.G)

    # unit conversions
    s_in_yr = 3.155693e7
    pc_in_cm = 3.085680e18
    km_in_cm = 1e5
    kpc_in_ly = 3261.56  # lightyears in 1 pkpc
    arcsec_in_rad = 4.84814e-6  # 1 arcsecond in radian (rad / ")
    ster_in_arcsec2 = 4.25e10  # 1 steradian in square arcseconds
    ang_in_cm = 1.0e-8  # cm / angstrom (1 angstrom in cm)
    erg_in_J = 1e-7  # 1 erg in joules
    erg_in_kev = 6.2415e8  # 1 erg in keV
    eV_in_erg = 1.602e-12  # 1 eV in erg

    # derived unit conversions
    s_in_kyr = None
    s_in_Myr = None
    s_in_Gyr = None
    kpc_in_km = None
    Mpc_in_cm = None
    kmS_in_kpcYr = None
    kmS_in_kpcGyr = None
    kpc_in_cm = None
    msunKpc3_in_gCm3 = None

    def __init__(self, sP=None):
        """Compute derived and redshift dependent units and values."""
        self._sP = sP

        # Mpc for lengths instead of the usual kpc?
        if self._sP is not None and self._sP.mpcUnits:
            self.UnitLength_in_cm = 3.085678e24  # 1000.0 kpc
            self.UnitLength_str = "cMpc/h"

        # custom (non-standard) unit system in header? only for non-cosmological runs
        if self._sP.redshift is not None and not self._sP.comoving:
            keys = ["UnitMass_in_g", "UnitLength_in_cm", "UnitVelocity_in_cm_per_s"]
            header = self._sP.snapshotHeader()

            for key in keys:
                if key not in header:
                    continue  # not present
                if header[key] == getattr(self, key):
                    continue  # unchanged

                print("NOTE: Setting units.%s = %g from header! EXPERIMENTAL!" % (key, header[key]))

                # update numeric value, and clear string representation
                skey = key.split("_")[0] + "_str"

                setattr(self, key, header[key])
                setattr(self, skey, "")

                # try to create new string representation (can be generalized)
                if key == "UnitMass_in_g":
                    diff = np.abs(np.log10(header[key]) - np.log10(self.Msun_in_g))
                    if diff < 1e-4:
                        setattr(self, skey, "M$_{\\rm sun}$/h")
                    if np.abs(diff - 2.0) < 1e-4:
                        setattr(self, skey, "0.01 M$_{\\rm sun}$/h")
                if key == "UnitLength_in_cm":
                    diff = np.abs(np.log10(header[key]) - np.log10(self.pc_in_cm))
                    if diff < 1e-4:
                        setattr(self, skey, "cpc/h")

        # non-cosmological run?
        if self._sP.redshift is not None and np.isnan(self._sP.redshift):
            print("NOTE: Setting units.scalefac = 1 for non-cosmological run.")
            self.scalefac = 1.0  # nullify all comoving -> physical conversions

            # update unit string representations (remove 'c' for comoving, and '/h')
            if self.UnitLength_str[0] == "c":
                self.UnitLength_str = self.UnitLength_str[1:]
            if self.UnitLength_str[-2:] == "/h":
                self.UnitLength_str = self.UnitLength_str[0:-2]
            if self.UnitMass_str[-2:] == "/h":
                self.UnitMass_str = self.UnitMass_str[0:-2]

        # derived units
        self.UnitTime_in_s = self.UnitLength_in_cm / self.UnitVelocity_in_cm_per_s
        self.UnitDensity_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm**3.0
        self.UnitPressure_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm / self.UnitTime_in_s**2.0
        self.UnitEnergy_in_cgs = self.UnitMass_in_g * self.UnitLength_in_cm**2.0 / self.UnitTime_in_s**2.0
        self.UnitTemp_in_cgs = self.UnitEnergy_in_cgs / self.UnitMass_in_g

        # non-cgs units
        self.UnitMass_in_Msun = self.UnitMass_in_g / self.Msun_in_g
        self.UnitTime_in_yr = self.UnitTime_in_s / self.s_in_yr

        # derived unit conversions
        self.s_in_kyr = self.s_in_yr * 1e3
        self.s_in_Myr = self.s_in_yr * 1e6
        self.s_in_Gyr = self.s_in_yr * 1e9
        self.kpc_in_km = self.pc_in_cm * (1e3 / 1e5)
        self.Mpc_in_cm = self.pc_in_cm * 1e6

        self.kmS_in_kpcYr = self.s_in_Myr / self.kpc_in_km / 1e6  # Myr->yr
        self.kmS_in_kpcGyr = self.s_in_Myr / self.kpc_in_km * 1e3  # Myr->Gyr
        self.kpc_in_cm = self.kpc_in_km * 1e5
        self.msunKpc3_in_gCm3 = self.Msun_in_g / (self.kpc_in_cm) ** 3.0

        # derived constants (in code units without little h factors)
        self.H0 = self._sP.HubbleParam * 100 * 1e5 / (self.Mpc_in_cm)
        self.H0 = self.H0 / self.UnitVelocity_in_cm_per_s * self.UnitLength_in_cm
        self.G = self.Gravity / self.UnitLength_in_cm**3.0 * self.UnitMass_in_g * self.UnitTime_in_s**2.0

        self.rhoCrit = 3.0 * self.H0**2.0 / (8.0 * np.pi * self.G)  # code, z=0
        self.rhoCrit_msunMpc3 = self.rhoCrit * 1e10 * 1e9  # 10^10 msun -> msun, kpc^-3 -> Mpc^-3
        self.H0_kmsMpc = self.H0 * 1000.0

        # derived constants / cosmology parameters
        self.mag2cgs = np.log10(self.L_sun / (4.0 * np.pi * (10 * self.pc_in_cm) ** 2))
        self.c_ang_per_sec = self.c_cgs / self.ang_in_cm

        self.f_b = np.nan
        if self._sP.omega_m != 0.0 and self._sP.omega_b is not None:
            self.f_b = self._sP.omega_b / self._sP.omega_m

        self.Hubble = self.H0_h1_s * self.UnitTime_in_s
        self.rhoBack = 3 * self._sP.omega_m * self.Hubble**2 / (8 * np.pi * self.G)

        # redshift dependent values (code units)
        if self._sP.redshift is not None and not np.isnan(self._sP.redshift):
            self.H2_z_fact = (
                self._sP.omega_m * (1 + self._sP.redshift) ** 3.0
                + self._sP.omega_L
                + self._sP.omega_k * (1 + self._sP.redshift) ** 2.0
            )
            self.Omega_z = self._sP.omega_m * (1 + self._sP.redshift) ** 3.0 / self.H2_z_fact
            self.H_z = self.H0 * np.sqrt(self.H2_z_fact)
            self.H_of_a = self.Hubble * np.sqrt(self.H2_z_fact)
            self.rhoCrit_z = self.rhoCrit * self.H2_z_fact
            self.scalefac = 1.0 / (1 + self._sP.redshift)

    # --- unit conversions to/from code units ---

    def codeMassToMsun(self, mass):
        """Convert mass from code units (10**10 msun/h) to (msun)."""
        mass = np.atleast_1d(mass)
        mass_msun = mass.astype("float32") * (self.UnitMass_in_Msun) / self._sP.HubbleParam

        return mass_msun

    def codeMassToLogMsun(self, mass):
        """Convert mass from code units (10**10 msun/h) to (log msun)."""
        return logZeroNaN(self.codeMassToMsun(mass))

    def msunToCodeMass(self, mass):
        """Convert mass in [msun] to code units."""
        return mass / self.UnitMass_in_Msun * self._sP.HubbleParam

    def logMsunToCodeMass(self, mass):
        """Convert mass in (log msun) to code units."""
        return self.msunToCodeMass(10.0**mass)

    def codeMassToVirTemp(self, mass, meanmolwt=None, log=False):
        """Convert from halo mass in code units to virial temperature in Kelvin, at the specified redshift.

        (Barkana & Loeb (2001) eqn.26).
        """
        assert self._sP.redshift is not None
        if not meanmolwt:
            meanmolwt = self.meanmolwt(Y=0.25, Z=0.0)  # default is primordial

        # mass to msun
        print(mass)
        mass_msun = np.array(mass).astype("float64") * self.UnitMass_in_g / self.Msun_in_g

        little_h = 1.0  # do not multiply by h since mass_msun is already over h

        omega_m_z = (
            self._sP.omega_m
            * (1 + self._sP.redshift) ** 3.0
            / (
                self._sP.omega_m * (1 + self._sP.redshift) ** 3.0
                + self._sP.omega_L
                + self._sP.omega_k * (1 + self._sP.redshift) ** 2.0
            )
        )

        Delta_c = 18 * np.pi**2 + 82 * (omega_m_z - 1.0) - 39 * (omega_m_z - 1.0) ** 2.0

        Tvir = (
            1.98e4
            * (meanmolwt / 0.6)
            * (mass_msun / 1e8 * little_h) ** (2.0 / 3.0)
            * (self._sP.omega_m / omega_m_z * Delta_c / 18.0 / np.pi**2.0) ** (1.0 / 3.0)
            * (1.0 + self._sP.redshift)
            / 10.0
        )  # K

        if log:
            Tvir = logZeroSafe(Tvir)
        return Tvir.astype("float32")

    def codeMassOverTimeToMsunPerYear(self, mass_over_time_val):
        """Convert a code [mass/time] value (e.g. BH_Mdot) into [Msun/yr]. The usual 10.22 factor."""
        return mass_over_time_val * (self.UnitMass_in_Msun / self.UnitTime_in_yr)

    def codeBHMassToMdotEdd(self, mass, eps_r=None):
        """Convert a code mass (of a blackhole) into dM/dt_eddington in [Msun/yr].

        Also available directly as 'BH_MdotEddington' field.
        """
        mass_msun = self.codeMassToMsun(mass)

        if eps_r is None:
            eps_r = self.BH_eps_r  # BH radiative efficiency, unchanged in Illustris and TNG models

        # Mdot(Edd) = 4*pi*G*M_BH*m_p / (eps_r*sigma_T*c) in Msun/s
        mdot_edd = 4 * np.pi * self.Gravity * mass_msun * self.mass_proton / (eps_r * self.sigma_thomson * self.c_cgs)

        mdot_edd_msun_yr = mdot_edd * self.s_in_yr

        return mdot_edd_msun_yr

    def codeBHMassToLumEdd(self, mass):
        """Convert a code mass (of a blackhole) into its Eddington luminosity [erg/s]."""
        mass_msun = self.codeMassToMsun(mass).astype("float64")  # prevent overflow

        # L(Edd) = 4*pi*G*M_BH*m_p*c / sigma_T [msun/s * cm^2/s^2] = 1.26e38 * mass_msun [erg/s]
        lum_edd = 4 * np.pi * self.Gravity * mass_msun * self.mass_proton * self.c_cgs / self.sigma_thomson
        lum_edd *= self.Msun_in_g  # [g/s * cm^2/s^2] = [erg/s]

        return lum_edd

    def codeBHMassMdotToBolLum(self, mass, mdot, basic_model=False, obscuration=False):
        """Convert a (BH mass, BH mdot) pair to a bolometric luminosity [erg/s]."""
        mdot_edd = self.codeBHMassToMdotEdd(mass)
        lum_edd = self.codeBHMassToLumEdd(mass)

        mdot_msun_yr = self.codeMassOverTimeToMsunPerYear(mdot).astype("float64")  # prevent overflow

        # Weinberger+ (2018) Eqn 12, i.e. Churazov+ (2005)
        Lbol = np.zeros(mdot_edd.size, dtype="float64")
        w_high = np.where(mdot_msun_yr >= 0.1 * mdot_edd)
        w_low = np.where(mdot_msun_yr < 0.1 * mdot_edd)
        w_nan = np.where(np.isnan(mdot_msun_yr))

        assert len(w_high[0]) + len(w_low[0]) + len(w_nan[0]) == mdot_edd.size

        Lbol = self.BH_eps_r * mdot_msun_yr * self.Msun_in_g * self.c_cgs**2 / self.s_in_yr  # erg/s
        if not basic_model:
            # alternatively, we do not use the simplest option for "low" modes:
            Lbol[w_low] = (10.0 * (mdot_msun_yr[w_low] / mdot_edd[w_low])) ** 2 * 0.1 * lum_edd[w_low]

        # obscuration model? as in GFM_AGN_RADIATION (Hopkins+2007)
        if obscuration:
            assert self._sP.BHs in [1, 2]  # fiducial Illustris or TNG model
            BlackHoleFeedbackFactor = 0.1
            ObscurationFactor = 0.3
            ObscurationSlope = 0.07

            obs_fac = (1.0 - BlackHoleFeedbackFactor) * ObscurationFactor * (Lbol / 1e46) ** ObscurationSlope
            Lbol *= obs_fac

        # Vog+13 Eqn 29 'continuous' modulation for low luminosities
        if self._sP.BHs == 1:
            # in Illustris this may be more complicated
            import pdb

            pdb.set_trace()
        if self._sP.BHs == 2:
            # in TNG this is simple
            QuasarThreshold = 0.002
            edd_frac = mdot_msun_yr / mdot_edd
            w = np.where(edd_frac < QuasarThreshold)
            Lbol[w] = 0.0

        return Lbol

    def BH_chi(self, M_BH):
        """Return chi(M_BH) threshold for the fiducial Illustris/TNG model parameters. M_BH in Msun."""
        if self._sP.BHs == 1:
            # fiducial Illustris model
            return 0.2

        elif self._sP.BHs == 2:
            # fiducial TNG model
            chi0 = 0.002
            beta = 2.0
            chi_max = 0.1

            chi_bh = np.clip(chi0 * (M_BH / 1e8) ** beta, 0.0, chi_max)
        else:
            raise Exception("Unimplemented (or DMO or other strangeness).")

        return chi_bh

    def codeBHValsToFeedbackMode(self, bh_mass, bh_mdot, bh_mdot_bondi, bh_mdot_edd):
        """Return the feedback mode (0 = low/kinetic, 1 = high/quasar) of the BH, based on its mass and mdot."""
        mass_msun = self.codeMassToMsun(bh_mass)
        bh_chi = self.BH_chi(mass_msun)  # low-state/high-state threshold

        w_high = np.where(bh_mdot_bondi / bh_mdot_edd >= bh_chi)
        # w_low = np.where(bh_mdot_bondi / bh_mdot_edd < bh_chi)

        mode = np.zeros(bh_mass.size, dtype="float32")  # better not as int, confuses auxCat/nan's
        mode[w_high] = 1

        return mode

    def codeBHMassMdotToInstantaneousEnergy(self, bh_mass, bh_mdot, bh_density, bh_mdot_bondi, bh_mdot_edd):
        """Convert the instantaneous mass/mdot of a BH to the instantaneous energy injection rate [erg/s].

        All inputs are in code units.
        The energy being injected follows the (mode-dependent) feedback model of the simulation.
        """
        assert self._sP.BHs == 2  # denotes fiducial TNG model, otherwise generalize to Illustris and/or others

        # get mode separation
        mode = self.codeBHValsToFeedbackMode(bh_mass, bh_mdot, bh_mdot_bondi, bh_mdot_edd)
        w_low = np.where(mode == 0)
        w_high = np.where(mode == 1)

        assert len(w_high[0]) + len(w_low[0]) == bh_mass.size

        # energy
        mdot_g_s = self.codeMassOverTimeToMsunPerYear(bh_mdot).astype("float64") * self.Msun_in_g / self.s_in_yr  # g/s

        dEdt = np.zeros(bh_mass.size, dtype="float64")

        BH_eps_f_low = np.clip(bh_density / (self.BH_f_thresh * self.PhysDensThresh), None, 0.2)  # max of 0.2

        dEdt[w_high] = self.BH_eps_f_high * self.BH_eps_r * mdot_g_s[w_high] * self.c_cgs**2  # erg/s
        dEdt[w_low] = BH_eps_f_low[w_low] * mdot_g_s[w_low] * self.c_cgs**2  # erg/s

        return dEdt

    def codeMetallicityToWindSpecificEnergy(self, metal_code):
        """Convert the metallicity of a gas cell into the wind specific energy parameter (see TNG methods)."""
        assert self._sP.winds == 2  # otherwise generalize

        metal_fac = self.winds_fZ + (1.0 - self.winds_fZ) / (1 + (metal_code / self.winds_Zref) ** self.winds_gamma_Z)
        energy_w = self.winds_e * metal_fac * self.N_SNII * self.E_SNII51  # 10^51 erg/msun

        return energy_w

    def codeSfrZToWindEnergyRate(self, sfr_msunyr, metal_code):
        """Convert the SFR [Msun/yr] of a gas cell into dot{E}_SN of total wind energy available [10^51 erg/s]."""
        energy_w = self.codeMetallicityToWindSpecificEnergy(metal_code)
        dedt_w = energy_w * (sfr_msunyr / self.s_in_yr)  # 10^51 erg/s

        return dedt_w

    def codeSfrZToWindMomentumRate(self, sfr_msunyr, metal_code, dm_veldisp):
        """Convert the SFR [Msun/yr] of a gas cell into dot{p}_SN of total wind momentum available [10^51 g*cm/s^2]."""
        eta_w = self.codeSfrZSigmaDMToWindMassLoading(sfr_msunyr, metal_code, dm_veldisp)
        vel_w = self.sigmaDMToWindVel(dm_veldisp) * self.km_in_cm  # cm/s
        sfr = sfr_msunyr.astype("float64") * self.Msun_in_g / self.s_in_yr  # g/s
        dpdt_w = eta_w * vel_w * sfr / 1e51  # 10^51 g*cm/s^2

        return dpdt_w.astype("float32")

    def sigmaDMToWindVel(self, dm_veldisp):
        """Convert a code (3D) velocity dispersion (i.e. SubfindVelDisp), into the wind launch velocity [km/s]."""
        assert self._sP.winds == 2  # otherwise generalize

        veldisp_1d = dm_veldisp / np.sqrt(3)
        vel_wind = self.winds_kappa * veldisp_1d * (self.H0 / self.H_z) ** (1.0 / 3.0)
        vel_wind = np.clip(vel_wind, self.winds_vmin, None)

        return vel_wind

    def codeSfrZSigmaDMToWindMassLoading(self, sfr_msunyr, metal_code, dm_veldisp):
        """Convert a gas cell SFR [Msun/yr], metallicity [code], and 3D vel disp [km/s] into the wind mass loading.

        The return is dimensionless linear.
        """
        assert self._sP.winds == 2  # otherwise generalize

        energy_w = self.codeMetallicityToWindSpecificEnergy(metal_code)  # 10^51 erg/msun
        energy_w_erg_g = energy_w * (1e51 / self.Msun_in_g)
        vel_w_cgs = self.sigmaDMToWindVel(dm_veldisp) * 1e5  # cm/s
        eta_w = (
            (2.0 / vel_w_cgs**2) * energy_w_erg_g * (1 - self.winds_tau)
        )  # s^2/cm^2 * g cm^2/s^2 / g = dimensionless

        return eta_w

    def densToSH03TwoPhase(self, dens_in, sfr):
        """Convert a gas cell density to values corresponding to the two-phase state of the sub-cell gas.

        According to the Springel & Hernquist (2003) sub-grid ISM pressurization model.

        Args:
          dens_in (float or ndarray): hydrogen physical gas number density [H atoms/cm^3]
          sfr (float or ndarray): star formation rate [Msun/yr]

        Returns:
        a 2-tuple composed of

        - **x** (:py:class:`~numpy.ndarray`): the cold-phase (i.e. cold clouds)
          mass fraction, which is density-dependent, and always between zero and unity.
        - **T_h** (:py:class:`~numpy.ndarray`): the hot-phase temperature, which is
          density-dependent, and increases from ~1e5 to ~1e8 K.
        """
        assert "TNG" in self._sP.simName, "Check applicability to non-TNG models."

        dens = np.array(dens_in)

        # load KWH+96 lambda
        from ..load.data import dataBasePath

        lambda_tab = np.loadtxt(dataBasePath + "/kwh/kwh96_lambda.txt")

        # SH03 Eqn. 20
        A = self.sh03_A0 * (dens / self.sh03_nH_thresh) ** (-0.8)

        mu_cold = 4 / (1 + 3 * self.hydrogen_massfrac)  # fully neutral
        mu_hot = 4 / (3 + 5 * self.hydrogen_massfrac)  # fully ionized

        u_SN = self.sh03_T_SN * self.boltzmann / (mu_hot * (self.gamma - 1))  # erg
        u_c = self.sh03_T_c * self.boltzmann / (mu_cold * (self.gamma - 1))  # erg

        # SH03 Eqn. 11
        u_h = u_SN / (A + 1) + u_c

        T_h = u_h * (mu_hot * (self.gamma - 1)) / self.boltzmann

        lambda_net = np.interp(T_h, lambda_tab[:, 0], lambda_tab[:, 1])
        lambda_net *= self.hydrogen_massfrac**2  # KWH+96 convention
        lambda_net *= -1  # cooling -> positive values

        # SH03 Eqn. 16 (lambda_net at dens,u_h)
        cooling_net = lambda_net * self.s_in_Gyr * dens**2  # erg cm^3/s -> erg/Gyr/cm^3

        # [Gyr * erg/Gyr/cm^3 / cm^-3 / erg] = [unitless]
        y = self.sh03_t_star * cooling_net / (dens * (self.sh03_beta * u_SN - (1 - self.sh03_beta) * u_c))

        # tsfr = np.sqrt(self.sh03_nH_thresh / dens) * MaxSfrTimescale
        # y = tsfr / tcool * u_h / (FactorSN * EgySpecSN - (1 - FactorSN) * EgySpecCold)

        # SH03 Eqn. 18 (cold cloud fraction)
        x = 1 + 1 / (2 * y) - np.sqrt(1 / y + 1 / (4 * y**2))

        # clip from [0,1] and set to zero below star-formation threshold (use sfr directly to avoid
        # any possible mismatch between pressurized and unpressurized cells)
        x = np.clip(x, 0.0, 1.0)

        x[sfr == 0] = 0.0

        return x, T_h

    def logMsunToVirTemp(self, mass, meanmolwt=None, log=False):
        """Convert halo mass (in log msun, no little h) to virial temperature at specified redshift."""
        return self.codeMassToVirTemp(self.logMsunToCodeMass(mass), meanmolwt=meanmolwt, log=log)

    def codeLengthToComovingKpc(self, x):
        """Convert length/distance in code units to comoving kpc."""
        x_phys = np.array(x, dtype="float32") / self._sP.HubbleParam  # remove little h factor
        x_phys *= self.UnitLength_in_cm / self.kpc_in_cm  # account for non-kpc code lengths

        return x_phys

    def codeLengthToKpc(self, x):
        """Convert length/distance in code units to physical kpc."""
        assert self._sP.redshift is not None

        return self.codeLengthToComovingKpc(x) * self.scalefac  # comoving -> physical

    def codeLengthToMpc(self, x):
        """Convert length/distance in code units to physical Mpc."""
        return self.codeLengthToKpc(x) / 1000.0

    def codeLengthToPc(self, x):
        """Convert length/distance in code units to physical parsec."""
        return self.codeLengthToKpc(x) * 1000.0

    def codeLengthToComovingMpc(self, x):
        """Convert length/distance in code units to comoving Mpc."""
        return self.codeLengthToComovingKpc(x) / 1000.0

    def codeLengthToCm(self, x):
        """Convert length/distance in code units to cgs (cm)."""
        x_phys_cgs = np.array(x, dtype="float32") / self._sP.HubbleParam  # remove little h factor
        x_phys_cgs *= self.UnitLength_in_cm  # ckpc -> ccm
        x_phys_cgs *= self.scalefac  # comoving -> physical

        return x_phys_cgs

    def codeAreaToKpc2(self, x):
        """Convert an area [length^2] in code units to physical kpc^2."""
        assert self._sP.redshift is not None

        area = np.array(x, dtype="float32") / self._sP.HubbleParam**2  # remove little h factors
        area *= (self.UnitLength_in_cm / self.kpc_in_cm) ** 2  # account for non-kpc code lengths
        area *= self.scalefac**2  # comoving -> physical

        return area

    def codeAreaToMpc2(self, x):
        """Convert an area [length^2] in code units to physical Mpc^2."""
        return self.codeAreaToKpc2(x) / 1000.0**2

    def codeVolumeToCm3(self, x):
        """Convert a volume [length^3] in code units to physical cm^3 (cgs)."""
        assert self._sP.redshift is not None

        vol_cgs = np.array(x, dtype="float64") / self._sP.HubbleParam**3  # remove little h factors
        vol_cgs *= self.UnitLength_in_cm**3  # code (kpc or mpc) to cm
        vol_cgs *= self.scalefac**3  # comoving -> physical

        return vol_cgs

    def codeVolumeToKpc3(self, x):
        """Convert a volume [length^3] in code units to physical kpc^3."""
        assert self._sP.redshift is not None

        vol = np.array(x, dtype="float64") / self._sP.HubbleParam**3  # remove little h factors
        vol *= (self.UnitLength_in_cm / self.kpc_in_cm) ** 3  # account for non-kpc code lengths
        vol *= self.scalefac**3  # comoving -> physical

        return vol

    def codeVolumeToMpc3(self, x):
        """Convert a volume [length^3] in code units to physical Mpc^3."""
        return self.codeVolumeToKpc3(x) / 1000.0**3

    def physicalKpcToCodeLength(self, x):
        """Convert a length in [pkpc] to code units [typically ckpc/h]."""
        assert self._sP.redshift is not None

        x_comoving = np.array(x, dtype="float32") / self.scalefac
        x_comoving /= self.UnitLength_in_cm / self.kpc_in_cm  # account for non-kpc code lengths
        x_comoving *= self._sP.HubbleParam  # add little h factor

        return x_comoving

    def lightyearsToCodeLength(self, x):
        """Convert a length in [lightyears] to code units."""
        assert self._sP.redshift is not None

        x_comoving = np.array(x, dtype="float32") / self.scalefac / self.kpc_in_ly  # ckpc
        x_comoving /= self.UnitLength_in_cm / self.kpc_in_cm  # account for non-kpc code lengths
        x_comoving *= self._sP.HubbleParam  # add little h factor

        return x_comoving

    def physicalMpcToCodeLength(self, x):
        """Convert a length in [pMpc] to code units [typically ckpc/h]."""
        return self.physicalKpcToCodeLength(x * 1000.0)

    def particleCodeVelocityToKms(self, x):
        """Convert velocity field (for cells/particles, not group properties) into km/s."""
        assert self._sP.redshift is not None

        x_phys = x * np.sqrt(self.scalefac)
        x_phys *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units

        return x_phys.astype("float32")

    def groupCodeVelocityToKms(self, x):
        """Convert velocity vector (for groups, not subhalos nor particles) into km/s."""
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype="float32") / self.scalefac
        x_phys *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units

        return x_phys

    def subhaloCodeVelocityToKms(self, x):
        """Convert velocity vector (for subhalos, not groups nor particles) into km/s."""
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype="float32")
        x_phys *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units

        return x_phys

    def subhaloSpinToKpcKms(self, x):
        """Convert spin vector (for subhalos, not groups nor particles) into kpc km/s."""
        assert self._sP.redshift is not None

        x_phys = np.array(x, dtype="float32") / self._sP.HubbleParam
        x_phys *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units

        return x_phys

    def particleCodeBFieldToGauss(self, b):
        """Convert magnetic field 3-vector (for cells) into Gauss, input b is PartType0/MagneticField."""
        UnitMagneticField_in_cgs = np.float32(np.sqrt(self.UnitPressure_in_cgs))

        b_gauss = b * self._sP.HubbleParam  # remove little h factor
        b_gauss /= self.scalefac**2.0  # convert 'comoving' into physical

        b_gauss *= UnitMagneticField_in_cgs  # [Gauss] = [g^(1/2) * cm^(-1/2) * s^(-1)]
        return b_gauss

    def particleCodeDivBToGaussPerKpc(self, divb):
        """Convert magnetic field divergence into [Gauss/kpc] physical, input is PartType0/MagneticFieldDivergence."""
        UnitMagneticField_in_cgs = np.float32(np.sqrt(self.UnitPressure_in_cgs))

        divb_phys = divb * self._sP.HubbleParam**2.0  # remove little h factors
        divb_phys /= self.scalefac**3.0  # convert 'comoving' into physical

        divb_phys *= UnitMagneticField_in_cgs  # [Gauss] = [g^(1/2) * cm^(-1/2) * s^(-1)]
        divb_phys /= self.UnitLength_in_cm / self.kpc_in_cm  # account for non-kpc code lengths (could be checked)
        return divb_phys

    def codePotentialToEscapeVelKms(self, pot):
        """Convert Potential [(km/s)^2/a] into an escape velocity [km/s]."""
        pot_phys = pot / self._sP.scalefac
        with np.errstate(invalid="ignore"):  # ignore unbound (positive potential)
            vesc = np.sqrt(-2.0 * pot_phys)
        return vesc

    def particleAngMomVecInKpcKmS(self, pos, vel, mass, haloPos, haloVel):
        """Calculate particle angular momentum 3-vector in [Msun*kpc km/s].

        Takes input arrays of pos,vel,mass and the halo CM position and velocity to compute relative to.
        Includes Hubble correction.
        """
        # make copies of input arrays
        gas_mass = self.codeMassToMsun(mass.astype("float32"))
        gas_pos = pos.astype("float32")
        gas_vel = vel.astype("float32")

        # calculate position, relative to subhalo center (pkpc)
        for i in range(3):
            if haloPos.ndim == 1:  # scalar
                gas_pos[:, i] -= haloPos[i]
            else:
                gas_pos[:, i] -= haloPos[:, i]

        self._sP.correctPeriodicDistVecs(gas_pos)
        xyz = self.codeLengthToKpc(gas_pos)

        rad = np.sqrt(xyz[:, 0] ** 2.0 + xyz[:, 1] ** 2.0 + xyz[:, 2] ** 2.0)  # equals np.linalg.norm(xyz,2,axis=1)
        rad[rad == 0.0] = 1e-5

        # calculate momentum, correcting velocities for subhalo CM motion and hubble flow (Msun km/s)
        gas_vel = self.particleCodeVelocityToKms(gas_vel)

        for i in range(3):
            # SubhaloVel already peculiar, no scalefactor needed
            if haloVel.ndim == 1:  # scalar
                gas_vel[:, i] -= haloVel[i]
            else:
                gas_vel[:, i] -= haloVel[:, i]

        v_H = self.H_z * rad  # Hubble expansion velocity magnitude (km/s) at each position

        # add Hubble expansion velocity 3-vector at each position (km/s)
        for i in range(3):
            gas_vel[:, i] += xyz[:, i] / rad * v_H

        mom = np.zeros((gas_mass.size, 3), dtype="float32")

        for i in range(3):
            mom[:, i] = gas_mass * gas_vel[:, i]

        # calculate angular momentum of each particle, rr x pp
        ang_mom = np.cross(xyz, mom)

        return ang_mom

    def particleSpecAngMomMagInKpcKmS(self, pos, vel, mass, haloPos, haloVel, log=False):
        """Calculate particle *specific* angular momentum *magnitude* in [kpc km/s]."""
        ang_mom = self.particleAngMomVecInKpcKmS(pos, vel, mass, haloPos, haloVel)

        # magnitude
        ang_mom_mag = np.linalg.norm(ang_mom, 2, axis=1)

        # specific
        gas_mass = self.codeMassToMsun(mass.astype("float32"))
        ang_mom_mag /= gas_mass

        if log:
            ang_mom_mag = logZeroSafe(ang_mom_mag)

        return ang_mom_mag

    def particleRadialVelInKmS(self, pos, vel, haloPos, subhaloVel):
        """Calculate particle radial velocity in [km/s] (negative=inwards).

        Takes input arrays of pos,vel and the halo CM position and velocity to compute relative to.
        Includes Hubble correction.
        """
        # make copies of input arrays
        gas_pos = pos.astype("float32")
        gas_vel = vel.astype("float32")

        if gas_pos.size == 3:  # single particle
            gas_pos = np.reshape(gas_pos, (1, 3))
            gas_vel = np.reshape(gas_vel, (1, 3))

        # calculate position, relative to subhalo center (pkpc)
        for i in range(3):
            if haloPos.ndim == 1:  # scalar
                gas_pos[:, i] -= haloPos[i]
            else:
                gas_pos[:, i] -= haloPos[:, i]

        self._sP.correctPeriodicDistVecs(gas_pos)

        xyz = self.codeLengthToKpc(gas_pos)
        rad = np.linalg.norm(xyz, 2, axis=1)

        # correct velocities for subhalo CM motion
        gas_vel = self.particleCodeVelocityToKms(gas_vel)

        for i in range(3):
            # SubhaloVel already peculiar, no scalefactor needed (note: WRONG calculation if GroupVel is input here)
            if subhaloVel.ndim == 1:  # scalar
                gas_vel[:, i] -= subhaloVel[i]
            else:
                gas_vel[:, i] -= subhaloVel[:, i]

        # correct velocities for hubble flow (neglect mass growth term)
        np.clip(rad, self._sP.gravSoft, None)  # avoid division by zero

        # radial velocity (km/s), negative=inwards
        vrad = (gas_vel[:, 0] * xyz[:, 0] + gas_vel[:, 1] * xyz[:, 1] + gas_vel[:, 2] * xyz[:, 2]) / rad

        if self.H_z is None:
            # non-cosmological (idealized) simulation
            print("Note: skipping Hubble flow term in vrad for non-cosmological run.")
        else:
            # cosmological integration
            v_H = self.H_z * rad  # Hubble expansion velocity magnitude (km/s) at each position

            vrad += v_H  # radial velocity (km/s) with hubble expansion subtracted

        return vrad

    def particleRelativeVelInKmS(self, vel, subhaloVel):
        """Calculate particle velocity magnitude in [km/s], relative to a given reference frame motion.

        This reference is typically SubhaloVel. If not, must be in physical units.
        """
        # make copies of input arrays
        p_vel = vel.astype("float32")

        if p_vel.size == 3:  # single particle
            p_vel = np.reshape(p_vel, (1, 3))

        # correct velocities for subhalo CM motion
        p_vel = self.particleCodeVelocityToKms(p_vel)

        for i in range(3):
            # SubhaloVel already peculiar, no scalefactor needed (note: WRONG calculation if GroupVel is input here)
            if subhaloVel.ndim == 1:  # scalar
                p_vel[:, i] -= subhaloVel[i]
            else:
                p_vel[:, i] -= subhaloVel[:, i]

        return p_vel

    def codeDensToPhys(self, dens, scalefac=None, cgs=False, numDens=False, msunpc3=False, totKpc3=False):
        r"""Convert mass density comoving->physical and add little_h factors.

        Unless overridden by a parameter option, the default return units are :math:`[10^{10} M_\odot/\rm{kpc}^3]`.

        Args:
          dens (array[float]): density in code units, should be
            :math:`[10^{10} M_\odot/h / (ckpc/h)^3]` = :math:`[10^{10} M_\odot h^2 / ckpc^3]`.
          scalefac (array[float]): if provided, use this scale factor instead of the current snapshot value.
            Can be a single value, or a different value per density.
          cgs (bool): if True, return units are [g/cm^3].
          numDens (bool): if True and cgs == True, return units are [1/cm^3].
          msunpc3 (bool): if True, return units are [Msun/pc^3].
          totKpc3 (bool): if True, return units are [[orig units]/kpc^3].

        Returns:
          array[float]: densities in physical units, as specified above.
        """
        assert self._sP.redshift is not None
        if numDens and not cgs:
            raise Exception("Odd choice.")
        if totKpc3 and (cgs or numDens or msunpc3):
            raise Exception("Invalid combination.")
        if msunpc3 and (cgs or numDens or totKpc3):
            raise Exception("Invalid combination.")

        # remove cosmological factors -> [UnitDensity]
        if scalefac is None:
            scalefac = self.scalefac

        dens_phys = dens.astype("float32") * self._sP.HubbleParam**2 / scalefac**3

        if cgs:
            dens_phys *= self.UnitDensity_in_cgs  # e.g. [1e10 msun/kpc^3] -> [g/cm^3]
        if numDens:
            dens_phys /= self.mass_proton  # e.g. [g/cm^3] -> [1/cm^3]

        # otherwise, we are not converting to g/cm^3 or 1/cm^3
        if not cgs:
            dens_phys *= (
                self.kpc_in_cm / self.UnitLength_in_cm
            ) ** 3.0  # account for non-kpc units -> [UnitMass/kpc^3]

        if msunpc3:
            dens_phys *= 10  # 1e10 msun/kpc^3 -> msun/pc^3
        if totKpc3:
            # non-mass quantity input as numerator, assume it did not have an h factor
            dens_phys *= self._sP.HubbleParam

        return dens_phys

    def physicalDensToCode(self, dens, cgs=False, numDens=False):
        """Convert mass density in physical units to code units (comoving + w/ little h factors, in unit system).

        Input: dens in [msun/kpc^3] or [g/cm^3 if cgs==True] or [1/cm^3 if cgs==True and numDens==True].
        Output: dens in [10^10 Msun/h / (ckpc/h)^3] = [10^10 Msun h^2 / ckpc^3] comoving.
        """
        assert self._sP.redshift is not None
        if numDens and not cgs:
            raise Exception("Odd choice.")

        # add cosmological factors
        dens_code = dens.astype("float32") * self.scalefac**3 / self._sP.HubbleParam**2

        # convert into unit system
        if numDens:
            dens_code *= self.mass_proton  # [1/cm^3] -> [g/cm^3]

        if cgs:
            dens_code /= self.UnitDensity_in_cgs  # [g/cm^3] -> e.g. [1e10 msun/kpc^3]
        else:
            dens_code *= (
                self.UnitLength_in_cm / self.kpc_in_cm
            ) ** 3.0  # account for non-kpc units, e.g. -> [msun/kpc^3]
            dens_code /= self.UnitMass_in_Msun  # [msun/kpc^3] -> [10^10 msun/kpc^3]

        return dens_code

    def codeColDensToPhys(self, colDens, cgs=False, numDens=False, msunKpc2=False, totKpc2=False):
        r"""Convert a mass column density [mass/area] from comoving -> physical and remove little_h factors.

        Unless overridden by a parameter option, the default return units are :math:`[10^{10} M_{\odot}/kpc^2]`.

        Args:
          colDens (ndarray): column densities in code units, which should be
            :math:`[10^{10} M_{\odot}/h / (ckpc/h)^2]` = :math:`[10^{10} M_{\odot} * h / ckpc^2].`
          cgs (bool): if True, return units in [g/cm^2].
          numDens (bool): if True and cgs == True, return units in [1/cm^2], which is in fact [H atoms/cm^2].
          msunKpc2 (bool): return units in [Msun / kpc^2].
          totKpc2 (bool): return units in [[orig units] / kpc^2].

        Returns:
          ndarray[float32]: physical column densities, units depending on the above.
        """
        assert self._sP.redshift is not None
        if numDens and not cgs:
            raise Exception("Odd choice.")
        if (msunKpc2 or totKpc2) and (numDens or cgs):
            raise Exception("Invalid combination.")

        # convert to 'physical code units' of 10^10 Msun/kpc^2
        colDensPhys = colDens.astype("float64") * self._sP.HubbleParam / self.scalefac**2.0

        if cgs:
            UnitColumnDensity_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm**2.0
            colDensPhys *= UnitColumnDensity_in_cgs  # g/cm^2
        if numDens:
            colDensPhys /= self.mass_proton  # 1/cm^2
        if msunKpc2:
            colDensPhys *= self.UnitMass_in_g / self.Msun_in_g  # remove 10^10 factor
            colDensPhys *= (self.kpc_in_cm / self.UnitLength_in_cm) ** 2.0  # account for non-kpc units
        if totKpc2:
            # non-mass quantity input as numerator, assume it did not have an h factor
            colDensPhys *= self._sP.HubbleParam
            colDensPhys *= (self.kpc_in_cm / self.UnitLength_in_cm) ** 2.0  # account for non-kpc units

        return colDensPhys.astype("float32")

    def UToTemp(self, u, xe, log=False):
        """Convert (U,xe) pair in code units to temperature in Kelvin."""
        # hydrogen mass fraction default
        hmassfrac = self.hydrogen_massfrac

        # calculate mean molecular weight
        meanmolwt = 4.0 / (1.0 + 3.0 * hmassfrac + 4.0 * hmassfrac * xe.astype("float32"))
        meanmolwt *= self.mass_proton

        # calculate temperature (K)
        temp = u.astype("float32")
        temp *= (self.gamma - 1.0) / self.boltzmann * (self.UnitEnergy_in_cgs / self.UnitMass_in_g)
        temp *= meanmolwt

        if log:
            temp = logZeroSafe(temp)
        return temp

    def TempToU(self, temp, xe, log=False):
        """Convert temperature in Kelvin to InternalEnergy (u) in code units."""
        if np.max(temp) <= 10.0:
            raise Exception("Error: input temp probably in log, check.")

        # calculate mean molecular weight
        meanmolwt = 4.0 / (1.0 + 3.0 * self.hydrogen_massfrac + 4.0 * self.hydrogen_massfrac * xe)
        meanmolwt *= self.mass_proton

        # temp = (gamma-1.0) * u / units.boltzmann * units.UnitEnergy_in_cgs / units.UnitMass_in_g * meanmolwt
        u = temp * self.boltzmann * self.UnitMass_in_g / (self.UnitEnergy_in_cgs * meanmolwt * (self.gamma - 1.0))

        if log:
            u = logZeroSafe(u)
        return u.astype("float32")

    def coolingRateToCGS_unused(self, coolrate):
        """Convert code units (du/dt) to erg/s/g (cgs, specific). Unused."""
        coolrate_cgs = coolrate.astype("float32")
        coolrate_cgs *= (
            self.UnitEnergy_in_cgs * self.UnitTime_in_s ** (-1.0) * self.UnitMass_in_g ** (-1.0) * self._sP.HubbleParam
        )

        return coolrate_cgs

    def coolingRateToCGS(self, code_dens, code_gfmcoolrate):
        """Convert cooling/heating rate to specific CGS [erg/s/g].

        Input is PartType0/[Masses,Density,GFM_CoolingRate].
        """
        dens_cgs = self.codeDensToPhys(code_dens, cgs=True)  # g/cm^3
        ratefact = self.hydrogen_massfrac**2 / self.mass_proton**2 * dens_cgs  # 1/(g*cm^3)
        coolrate = code_gfmcoolrate * ratefact  # erg cm^3/s * (1/g/cm^3) = erg/s/g (i.e. specific rate)

        return coolrate  # positive = heating, negative = cooling

    def powellEnergyTermCGS(self, code_dens, code_divb, code_b, code_vel, code_vol):
        """The 'Powell heating/cooling' energy source term (rightmost in Eqn. 21 Pakmor & Springel arxiv:1212.1452)."""
        vel_kpc_s = self.particleCodeVelocityToKms(code_vel) / self.kpc_in_km  # kpc/s
        b_gauss = self.particleCodeBFieldToGauss(code_b)  # gauss
        b_gauss /= 4 * np.pi  # to heaviside-lorentz (Bc in Eqn)
        divb_gauss_kpc = self.particleCodeDivBToGaussPerKpc(code_divb)  # gauss/kpc
        # vol_kpc3 = self.codeVolumeToKpc3(code_vol)  # kpc^3

        # gauss*kpc/s
        bvel = b_gauss[:, 0] * vel_kpc_s[:, 0] + b_gauss[:, 1] * vel_kpc_s[:, 1] + b_gauss[:, 2] * vel_kpc_s[:, 2]
        # gauss/kpc * gauss*kpc/s = gauss^2/s = [erg/s/cm^3]
        energy_vol_rate = (-1.0 / self.scalefac) * divb_gauss_kpc * bvel

        dens_cgs = self.codeDensToPhys(code_dens, cgs=True)  # g/cm^3
        energy_rate = energy_vol_rate / dens_cgs  # [erg/s/g]

        return energy_rate  # positive = heating, negative = cooling

    def coolingTimeGyr(self, code_dens, code_gfmcoolrate, code_u):
        """Calculate a cooling time [Gyr] from gas cell properties.

        Inputs: three snapshot values (i.e. code units) of Density, GFM_CoolingRate, InternalEnergy.
        """
        dens_cgs = self.codeDensToPhys(code_dens, cgs=True)  # g/cm^3
        ratefact = self.hydrogen_massfrac**2 / self.mass_proton**2 * dens_cgs  # 1/(g*cm^3)
        coolrate = code_gfmcoolrate * ratefact  # erg cm^3/s * (1/g/cm^3) = erg/s/g (i.e. specific rate)
        u_cgs_spec = code_u * self.UnitVelocity_in_cm_per_s**2  # i.e. (km/s)^2 to (cm/s)^2, so specific erg/g
        t_cool = u_cgs_spec / (-1.0 * coolrate) / self.s_in_Gyr

        # if lambda_net is positive set t_cool=nan (i.e. actual net heating, perhaps from the background)
        w = np.where(code_gfmcoolrate >= 0.0)
        t_cool[w] = np.nan

        return t_cool.astype("float32")

    def tracerEntToCGS(self, ent, log=False):
        """Fix cosmological/unit system in TRACER_MC[MaxEnt], output in cgs [K cm^2]."""
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        # Note: dens=dens*a3inv but in the tracers only converted in dens^gamma not in the pressure
        # have to make this adjustment in loading tracers
        # for SFR, for gas and tracers, Pressure = GAMMA_MINUS1 * localSphP[i].Density * localSphP[i].Utherm;
        # for TRACER_MC, EntMax = SphP.Pressure / pow(SphP.Density * All.cf_a3inv, GAMMA);

        # fix Pressure
        ent_cgs = ent.astype("float32")
        ent_cgs *= a3inv * self.UnitPressure_in_cgs / self.boltzmann

        # fix Density
        ent_cgs /= (self.UnitDensity_in_cgs / self.mass_proton) ** self.gamma

        if log:
            ent_cgs = logZeroSafe(ent_cgs)
        return ent_cgs

    def calcXrayLumBolometric(self, sfr, u, xe, mass, dens, temp=None, log=False):
        """Estimate bolometric X-ray luminosity of gas [10^30 erg/s], assuming only free-free (bremsstrahlung) emission.

        Note: based only on gas density and temperature, and also assumes  simplified (primordial) high-temp cooling
        function, and only free-free emission contribution from T>10^6 Kelvin gas. All inputs in code units.
        """
        hmassfrac = self.hydrogen_massfrac

        # calculate mean molecular weight
        meanmolwt = 4.0 / (1.0 + 3.0 * hmassfrac + 4.0 * hmassfrac * xe.astype("float64"))
        meanmolwt *= self.mass_proton

        # calculate temperature (K)
        if temp is None:
            energy_fac = self.UnitEnergy_in_cgs / self.UnitMass_in_g
            temp = u * (self.gamma - 1.0) / self.boltzmann * energy_fac * meanmolwt

        # Eqn. 6
        mass_g = mass.astype("float64") * (self.UnitMass_in_g) / self._sP.HubbleParam
        dens_g_cm3 = self.codeDensToPhys(dens, cgs=True)  # g/cm^3

        Lx = 1.2e-24 / (meanmolwt) ** 2.0 * mass_g * dens_g_cm3 * np.sqrt(temp / self.boltzmann_keV)

        # clip any cells on eEOS (SFR>0) to zero
        w = np.where(sfr > 0.0)
        Lx[w] = 0.0

        # implement a linear ramp from log(T)=6.0 to log(T)=5.8 over which we clip to zero
        temp = np.log10(temp)
        Lx *= np.clip((temp - 5.8) / 0.2, 0.0, 1.0)

        Lx *= 1e-30  # work in this unit system of [10^30 erg/s] for xray to avoid overflows to inf

        if log:
            Lx = logZeroSafe(Lx)
        return Lx.astype("float32")

    def opticalDepthLineCenter(self, transition, dens_cgs, temp_K, cellsize_code):
        """Derive an optical depth tau_0 = n * sigma_0 * L for a given transition.

        Assume assuming the frequency is at line center (neglecting the Voigt profile shape). dens_cgs is the volume
        number density of the species of relevance [1/cm^3], temp is the cell temperature [linear K],
        and cellsize_code is the usual radius of the cell [code units], which we take as L/2.
        """
        trans = transition.lower()

        # todo: move into cosmo.cloudy, remove duplication in other parts of the codebase
        f12 = None

        if trans == "mgii2796":
            f12 = 0.3058  # oscillator strength of the transition
            wave0 = 2796.352  # 2803.5320 # line center wavelength [Ang]
            mass_amu = 24.305
        if trans == "mgii2803":
            f12 = 0.6155
            wave0 = 2803.5320
            mass_amu = 24.305
        if trans == "oviir":
            f12 = 6.96e-1
            wave0 = 21.6019
            mass_amu = 15.999
        if trans == "lya":
            f12 = 0.416
            wave0 = 1215.67
            mass_amu = 1.008
        if trans == "lyb":
            f12 = 0.07912
            wave0 = 1025.7223
            mass_amu = 1.008

        assert f12 is not None, "Unhandled."

        # Doppler width [Hz]
        nu0 = self.c_ang_per_sec / wave0  # line center frequency [Hz]
        Delta_vd = np.sqrt(2 * self.boltzmann * temp_K / (self.mass_proton * mass_amu * self.c_cgs**2)) * nu0

        # line center cross section [cm^2]
        sigma_0 = f12 * np.sqrt(np.pi) * self.electron_charge**2 / (self.mass_electron * self.c_cgs * Delta_vd)

        # note: could compute for an arbitrary frequency away from line center, as
        # sigma = sigma_0 * H(alpha,x) where alpha = Delta_vl / (2 * Delta_vd)
        #   and Delta_vl is the natural line width [Hz], x=(nu-nu0)/Delta_vd is the
        #   relative frequency of the incident photon in the observer's frame
        #   and H is the Voigt profile (see plot.cloudy.curveOfGrowth and Tasitsiomi+2006)

        # optical depth [unitless]
        length_cgs = self.codeLengthToKpc(cellsize_code * 2.0) * self.kpc_in_cm

        tau_0 = dens_cgs * length_cgs * sigma_0

        return tau_0

    def sfrToHalphaLuminosity(self, sfr):
        """Convert SFR from code units (Msun/yr) into H-alpha line luminosity [linear 10^30 erg/s].

        Just the usual linear conversion from Kennicutt.
        """
        halpha_lum = sfr / (7.9e-42 * 1e30)
        return halpha_lum

    def gasSfrMetalMassToS850Flux(self, sfr, metalmass, temp, dens, ismCut=True):
        """Convert SFR [code units, Msun/yr] and metal mass [Msun] into 850 micron submm flux [linear mJy].

        Simple model of Hayward+ (2013b) assuming a constant dust-to-metal ratio.
        Also enforce Torrey+12 'ISM' constraint using temp [log K] and dens [code units, comoving].
        """
        dust_to_metal = 0.4
        s850 = 0.81 * (sfr / 100) ** 0.43 * (dust_to_metal * metalmass / 1e8) ** 0.54

        if ismCut:
            # doesn't actually do anything (all SFR==0 above this cut)
            w = np.where(temp > 6.0 + 0.25 * np.log10(dens))  # unclear if dens is supposed to be comoving
            s850[w] = 0.0

        return s850

    def calcEntropyCGS(self, u, dens, log=False):
        """Calculate entropy as P/rho^gamma, converting rho from comoving to physical. Return [K cm^2]."""
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        # cosmological conversions
        dens_phys = dens.astype("float32") * self._sP.HubbleParam**2.0 * a3inv  # 10^10 msun / kpc^3

        # pressure in [K/cm^3], with unit system conversions
        pressure = u.astype("float32")
        pressure *= (self.gamma - 1.0) * dens_phys * self.UnitPressure_in_cgs / self.boltzmann

        # entropy in [K cm^2]
        dens_fac = self.UnitDensity_in_cgs / self.mass_proton * a3inv
        entropy = pressure / (dens_phys * dens_fac) ** self.gamma

        if log:
            entropy = logZeroSafe(entropy)
        return entropy

    def calcPressureCGS(self, u, dens, log=False):
        """Calculate pressure as (gamma-1)*u*rho in physical 'cgs' [K/cm^3] units."""
        assert self._sP.redshift is not None

        a3inv = 1.0 / self.scalefac**3.0

        dens_phys = dens.astype("float32") * self._sP.HubbleParam**2.0  # remove all little h factors

        pressure = u.astype("float32") * (dens_phys.astype("float32") * a3inv)
        pressure *= self.gamma - 1.0

        # convert to CGS = 1 barye (ba) = 1 dyn/cm^2 = 0.1 Pa = 0.1 N/m^2 = 0.1 kg/m/s^2
        # and divide by boltzmann's constant -> [K/cm^3]
        pressure *= self.UnitPressure_in_cgs / self.boltzmann

        if log:
            pressure = logZeroSafe(pressure)
        return pressure

    def calcMagneticPressureCGS(self, b, log=False):
        """Calculate magnetic pressure as B^2/8/pi in physical 'cgs' [K/cm^3] units."""
        # input b is PartType0/MagneticField 3-vector (code units)
        b = self.particleCodeBFieldToGauss(b)  # to physical Gauss

        # magnetic pressure P_B in CGS units of [dyn/cm^2] (is energy/volume)
        P_B = (b[:, 0] * b[:, 0] + b[:, 1] * b[:, 1] + b[:, 2] * b[:, 2]) / (8 * np.pi)

        P_B /= self.boltzmann  # divide by boltzmann's constant -> [K/cm^3]

        if log:
            P_B = logZeroNaN(P_B)
        return P_B

    def calcKineticEnergyDensityCGS(self, dens_code, vel_kms, log=False):
        """Calculate kinetic energy density (KE/volume = 1/2 * mv^2 / volume) in 'cgs' [K/cm^3] units.

        Inputs: dens_code in code units, vel_kms in physical km/s.
        """
        vel_cm_s = vel_kms * 1e5
        dens_g_cm3 = self.codeDensToPhys(dens_code, cgs=True, numDens=False)

        u_ke = 0.5 * dens_g_cm3 * vel_cm_s**2  # g/cm/s^2 = [dyn/cm^2] (is energy/volume)
        u_ke /= self.boltzmann  # divide by boltzmann's constant -> [K/cm^3]

        if log:
            u_ke = logZeroNaN(u_ke)
        return u_ke

    def calcSoundSpeedKmS(self, u, dens, log=False):
        """Calculate sound speed as sqrt(gamma*Pressure/Density) in physical km/s."""
        pres = (self.gamma - 1.0) * dens * u
        csnd = np.sqrt(self.gamma * pres / dens)  # code units, all scalefac and h cancel
        csnd = csnd.astype("float32")

        csnd *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units

        if log:
            csnd = logZeroNaN(csnd)
        return csnd

    def soundSpeedFromTemp(self, temp, log=False):
        """Calculate sound speed given temperature [in Kelvin] in physical km/s."""
        # calculate mean mass
        hmassfrac = self.hydrogen_massfrac
        xe = 1.1
        m = 4.0 / (1.0 + 3.0 * hmassfrac + 4.0 * hmassfrac * xe)
        m *= self.mass_proton

        csnd = np.sqrt(self.gamma * self.boltzmann * temp / m)  # cm/s
        csnd /= 1e5  # cm/s -> km/s

        if log:
            csnd = logZeroNaN(csnd)
        return csnd

    def calcSunyaevZeldovichYparam(self, mass, xe, temp):
        """Calculate per-cell (thermal) SZ y-parameter (McCarthy+2014 Eqn 2, Roncarelli+2007 Eqn 5, Kay+2012 Eqn 12).

        Args:
          mass (ndarray[float]): gas cell masses [code units].
          xe (ndarray[float]): gas electron number density fraction [code units, i.e. dimensionless linear].
          temp (ndarray[float]): gas temperature [linear K].

        Return:
          ndarray[float]: y-parameter in units of area [pkpc^2].
        """
        # Y_i = k * sigma_T / (m_e * c^2) * (n_e,i * m_i / rho_i) * T_i
        # Y_i = k * sigma_T / (m_e * c^2) * (m_i / mu_e / m_H) * T_i
        #   with mu_e=1.14 a reasonable mean molecular weight per free elctron in a fully ionized plasma with X=0.76

        # prefactor: [erg s^2 / K / g] = cm^2 / K
        consts = self.boltzmann * self.sigma_thomson / (self.mass_electron * self.c_cgs**2)

        # mass * ne/rho [dimensionless]
        massfac = (
            self.hydrogen_massfrac
            * xe
            * mass.astype("float64")
            * (self.UnitMass_in_g / self._sP.HubbleParam / self.mass_proton)
        )
        # massfac = mass * (self.UnitMass_in_g / self.mass_proton) / 1.14 # essentially identical

        Y = consts * temp * massfac  # cm^2
        Y /= (self.kpc_in_km * self.km_in_cm) ** 2  # kpc^2

        return Y.astype("float32")

    def calcKineticSZYParam(self, mass, xe, vel_los):
        """Calculate per-cell kinetic SZ y-parameter (e.g. Dolag+16 Eqn. 4, Altamura+23 Eqn. 2).

        Args:
            mass (ndarray[float]): gas cell masses [code units].
            xe (ndarray[float]): gas electron number density fraction [code units, i.e. dimensionless linear].
            vel_los (ndarray[float]): line-of-sight velocity [km/s].

        Return:
            ndarray[float]: y-parameter in units of area [pkpc^2].
        """
        # Y_i = -sigma_T / c * n_e_i * v_los_i

        # prefactor: [cm^2 / cm * s] = [cm s]
        consts = -1.0 * self.sigma_thomson / self.c_cgs

        # n_e [dimensionless]
        massfac = self.hydrogen_massfrac * xe * mass * (self.UnitMass_in_g / self._sP.HubbleParam / self.mass_proton)

        vel_los_cms = self.particleCodeVelocityToKms(vel_los) * 1e5  # km/s -> cm/s
        Y_kSZ = consts * massfac * vel_los_cms  # [cm^2]

        Y_kSZ /= (self.kpc_in_km * self.km_in_cm) ** 2  # [cm^2] -> [kpc^2]
        return Y_kSZ

    def codeDensToCritRatio(self, rho, baryon=False, log=False, redshiftZero=False):
        """Normalize code density by the critical (total/baryonic) density at some redshift.

        If redshiftZero, normalize by rho_crit,0 instead of rho_crit(z).
        """
        assert self._sP.redshift is not None

        rho_crit = self.rhoCrit_z
        if redshiftZero:
            rho_crit = self.rhoCrit
        if baryon:
            rho_crit *= self._sP.omega_b

        # normalize, note: codeDensToPhys() returns units [10^10 msun/kpc^3]
        ratio_crit = self.codeDensToPhys(rho) / rho_crit

        if log:
            ratio_crit = logZeroSafe(ratio_crit)
        return ratio_crit

    def critRatioToCodeDens(self, ratioToCrit, baryon=False):
        """Convert a ratio of the critical density at some redshift to a code density."""
        assert self._sP.redshift is not None

        phys_dens = ratioToCrit.astype("float32") * self.rhoCrit_z  # 10^10 msun / kpc^3
        code_dens = self.physicalDensToCode(phys_dens / self.UnitMass_in_Msun)

        if baryon:
            code_dens *= self._sP.omega_b

        return code_dens

    def codeMassToVirEnt(self, mass, log=False):
        """Given a total halo mass, return a S200 (e.g. Pvir/rho_200crit^gamma)."""
        virTemp = self.codeMassToVirTemp(mass, log=False)
        virNe = np.array([1.0])  # todo, want mu=0.6 for fully ionized
        virU = self.TempToU(virTemp, virNe)
        r200crit = self.critRatioToCodeDens(np.array(200.0), baryon=True)

        s200 = self.calcEntropyCGS(virU, r200crit, log=log)

        return s200.astype("float32")

    def codeMassToVirVel(self, mass):
        """Given a total halo mass [in code units], return a virial velocity (V200) in physical [km/s]."""
        assert self._sP.redshift is not None

        r200 = (self.G * mass / 100.0 / self.H_z**2.0) ** (1.0 / 3.0)
        v200 = np.sqrt(self.G * mass / r200)

        return v200.astype("float32")

    def codeMassToVirRad(self, mass):
        """Given a total halo mass [in code units], return a virial radius (r200) in physical [kpc]."""
        assert self._sP.redshift is not None

        r200 = (self.G * mass / 100.0 / self.H_z**2.0) ** (1.0 / 3.0)
        r200 = self.codeLengthToKpc(r200)

        return r200.astype("float32")

    def codeM200R200ToV200InKmS(self, m200, r200):
        """Given a (M200,R200) pair in code units for a FoF group, compute V200 in physical [km/s]."""
        assert self._sP.redshift is not None

        with np.errstate(invalid="ignore"):
            v200 = np.sqrt(self.G * m200 / r200 * self._sP.scalefac)  # little h factors cancel
        v200 *= 1.0e5 / self.UnitVelocity_in_cm_per_s  # account for non-km/s code units
        return v200

    def avgEnclosedDensityToFreeFallTime(self, rho_code):
        """Convert a mass density (code units) to gravitational free-fall time [linear Gyr].

        rho_code Should be the mean density interior to the radius R of a test mass within a halo.
        """
        # remove little h factors
        rho = rho_code.astype("float32") * self._sP.HubbleParam**2 / self.scalefac**3  # 1e10 msun / kpc^3
        rho /= (self.kpc_in_km) ** 2  # 1e10 msun / (km^2 * kpc)

        t_ff = np.sqrt(3 * np.pi / (32 * self.G * rho))  # G is [kpc (km/s)**2 / 1e10 msun] -> t_ff is [s]
        t_ff /= self.s_in_Gyr

        return t_ff

    def metallicityInSolar(self, metal, log=False):
        """Given a code metallicity (M_Z/M_total), convert to value with respect to solar."""
        metal_solar = metal.astype("float32") / self.Z_solar

        metal_solar = np.clip(metal_solar, 0.0, np.inf)  # clip possibly negative Illustris values at zero

        if log:
            return np.log10(metal_solar)
        return metal_solar

    def codeTimeStepToYears(self, TimeStep, Gyr=False):
        """Convert a TimeStep/TimeStepHydro/TimeStepGrav for a comoving run to a physical time in years.

        Note: these inputs are an integer times All.Timebase_interval.
        """
        dtime = TimeStep / (np.sqrt(self.H2_z_fact) * self.H0_h1_s)
        dtime /= self._sP.HubbleParam
        dtime /= self.s_in_yr

        if Gyr:
            dtime /= 1e9

        return dtime

    def scalefacToAgeLogGyr(self, scalefacs):
        """Convert scalefactors of formation (e.g. GFM_StellarFormationTime) to age in log(Gyr).

        Uses the current age of the universe as specified by sP.redshift.
        """
        age = scalefacs.astype("float32")
        age.fill(np.nan)  # set wind to age=nan
        w_stars = np.where(scalefacs >= 0.0)

        curUniverseAgeGyr = self.redshiftToAgeFlat(self._sP.redshift)
        birthRedshift = 1.0 / scalefacs - 1.0
        birthRedshift = logZeroMin(curUniverseAgeGyr - self.redshiftToAgeFlat(birthRedshift))

        age[w_stars] = birthRedshift[w_stars]
        return age

    def codeEnergyToErg(self, energy, log=False):
        """Convert energy from code units (unitMass*unitLength^2/unitTime^2) to [erg]. (for BH_CumEgy*)."""
        energy_cgs = energy.astype("float64") * self.UnitEnergy_in_cgs / self._sP.HubbleParam

        if log:
            return logZeroNaN(energy_cgs).astype("float32")
        return energy_cgs

    def codeEnergyRateToErgPerSec(self, energy_rate, log=False):
        """Convert energy/time from code units (unitEnergy/unitTime) to [erg/s]. (for Gas EnergyDissipation)."""
        energy_rate_cgs = energy_rate.astype("float64") * (1 / self._sP.scalefac)  # physical
        energy_rate_cgs *= self.UnitEnergy_in_cgs / self.UnitTime_in_s  # need float64 to avoid overflow

        if log:
            return logZeroNaN(energy_rate_cgs)
        return energy_rate_cgs

    def codeEnergyDensToErgPerCm3(self, energy_dens, log=False):
        """Convert energy/volume from -proper- code units to [erg/cm^3] (for RadiationEnergyDensity)."""
        edens_cgs = energy_dens.astype("float64") * self.UnitEnergy_in_cgs / self.UnitLength_in_cm**3
        edens_cgs /= self._sP.HubbleParam**2

        if log:
            return logZeroNaN(edens_cgs).astype("float32")
        return edens_cgs.astype("float32")

    def codeEnergyDensToHabing(self, energy_dens, log=False):
        """Convert energy/volume from -proper- code units to Habing units (for RadiationEnergyDensity)."""
        edens = self.codeEnergyDensToErgPerCm3(energy_dens, log=False)
        edens /= self.Habing

        if log:
            return logZeroNaN(edens).astype("float32")
        return edens.astype("float32")

    def lumToAbsMag(self, lum):
        """Convert from an input luminosity in units of [Lsun/Hz] to an AB absolute magnitude."""
        mag = -2.5 * np.log10(lum) - 48.60 - 2.5 * self.mag2cgs

        # mag2cgs converts from [Lsun/Hz] to cgs [erg/s/cm^2] at d=10pc (definition of absolute mag)
        # 48.60 sets the zero-point of 3631 Jy
        return mag

    def absMagToLuminosity(self, mag):
        """Convert from input AB absolute magnitudes to (linear) luminosity units of [Lsun/Hz]."""
        log_lum = (mag + 48.60 + 2.5 * self.mag2cgs) / (-2.5)
        lum = 10.0**log_lum
        return lum

    def absMagToApparent(self, absolute_mag, redshift=None):
        """Convert an absolute magnitude to apparent."""
        if redshift is None:
            redshift = self._sP.redshift

        d_L_cm = self.redshiftToLumDist(redshift) * 1e6  # Mpc -> pc

        apparent_mag = absolute_mag + 5.0 * (np.log10(d_L_cm) - 1.0)
        return apparent_mag

    def apparentMagToAbsolute(self, apparent_mag, redshift=None):
        """Convert an apparent magnitude to absolute."""
        if redshift is None:
            redshift = self._sP.redshift

        d_L_cm = self.redshiftToLumDist(redshift) * 1e6  # Mpc -> pc

        absolute_mag = apparent_mag - 5.0 * (np.log10(d_L_cm) - 1.0)
        return absolute_mag

    def photonWavelengthToErg(self, wavelength, redshift=None):
        """Convert a photon wavelength [rest-frame Ang] emitted at a redshift into photon energy [erg]."""
        if redshift is None:
            redshift = self._sP.redshift

        photon_fac = (wavelength * self.ang_in_cm / self.planck_erg_s / self.c_cgs) * (1.0 + redshift)
        return photon_fac

    def luminosityToFlux(self, lum, wavelength=None, redshift=None):
        """Convert a luminosity in [erg/s] to a flux [photon/s/cm^2] for e.g. line emission.

        At a given wavelength in [Angstroms] if not None, from a source at the given redshift.
        If wavelength is None, then output units are an energy flux e.g. [erg/s/cm^2].
        """
        if redshift is None:
            redshift = self._sP.redshift

        # flux F = L/(4*pi*d_L^2)*(lambda_L/h/c)*(1+z) in [photon/s/cm^2]
        d_L_cm = np.float64(self.redshiftToLumDist(redshift)) * self.Mpc_in_cm

        dist_fac = 4 * np.pi * d_L_cm**2.0  # cm^2

        photon_fac = 1.0  # photon/erg
        if wavelength is not None:
            photon_fac = (wavelength * self.ang_in_cm / self.planck_erg_s / self.c_cgs) * (1.0 + redshift)

        flux = lum / dist_fac * photon_fac
        return flux

    def fluxToLuminosity(self, flux, redshift=None):
        """Convert a flux in [erg/s/cm^2] to a luminosity [erg/s], from a source at the given redshift."""
        if redshift is None:
            redshift = self._sP.redshift

        d_L_cm = self.redshiftToLumDist(redshift) * self.Mpc_in_cm
        if isinstance(d_L_cm, np.ndarray):
            d_L_cm = d_L_cm.astype("float64")  # avoid overflow

        dist_fac = 4 * np.pi * d_L_cm**2.0

        lum = flux * dist_fac

        if lum.max() < np.finfo("float32").max and lum.min() > np.finfo("float32").min:
            lum = lum.astype("float32")

        return lum

    def fluxToSurfaceBrightness(self, flux, pxDimsCode, arcsec2=True, arcmin2=False, ster=False, kpc=False):
        """Convert a flux in e.g. [energy/s/cm^2] or [photon/s/cm^2] into a surface brightness.

        At a given redshift and for a certain pixel scale. pxDimsCode is a 2-tuple of the x and y
        dimensions of the pixel in code units, i.e. [ckpc/h]^2. Output e.g.: [photon/s/cm^2/arcsec^2]
        if arcsec2 == True, otherwise possibly [flux/arcmin^2], [flux/ster], or [flux/kpc^2].
        """
        assert self._sP.redshift is not None
        assert np.sum([arcsec2, arcmin2, ster, kpc]) in [1]  # choose one

        # surface brightness SB = F/Omega_px where the solid angle Omega_px = 2*pi*(1-cos(theta/2))
        #   where theta is the pixel size in radians, note this reduces to Omega_px = 2*pi^2 for
        #   small theta, i.e. just the area of a circle, and we instead do the area of the square pixel
        if not kpc:
            theta1 = self.codeLengthToAngularSize(pxDimsCode[0], arcsec=True)
            theta2 = self.codeLengthToAngularSize(pxDimsCode[1], arcsec=True)
            solid_angle = theta1 * theta2  # arcsec^2

        if ster:
            # convert [arcsec^2] -> [steradian]
            arcsec2_to_ster = (1 / self.arcsec_in_rad) ** 2.0
            solid_angle /= arcsec2_to_ster

        if arcmin2:
            # convert [arcsec^2] -> [arcmin^2]
            solid_angle /= 3600.0

        if kpc:
            # overwrite [arcsec^2] with [pkpc^2]
            theta1 = self.codeLengthToKpc(pxDimsCode[0])
            theta2 = self.codeLengthToKpc(pxDimsCode[1])
            solid_angle = theta1 * theta2

        return flux / solid_angle

    def synchrotronPowerPerFreq(
        self,
        gas_B,
        gas_vol,
        watts_per_hz=True,
        log=False,
        telescope="SKA",  # telescope/observing configurations from Vazza+ (2015) as below
        eta=1.0,  # radio between the u_dens in relativistic particles and the magnetic u_dens
        k=10,  # energy density ratio between (relativistic) protons and electrons
        gamma_min=300,  # lower limit for the Lorentz factor of the electrons
        gamma_max=15000,  # upper limit for the Lorentz factor of the electrons
        alpha=1.7,
    ):  # spectral index
        """Calculate synchrotron power per unit frequency (simple model) for gas cells.

        Inputs: gas_B and gas_vol the magnetic field 3-vector and volume, both in code units.
        Output units [Watts/Hz].
        Default model parameter assumptions of Xu+ (2012)/Marinacci+ (2017).
        """
        assert telescope in ["VLA", "LOFAR", "ASKAP", "SKA"]

        # v0 [Mhz], delta_nu [Mhz], beam [arcsec], rms noise [mJy/beam]
        telParams = {
            "VLA": [1400, 25, 35, 0.1],
            "LOFAR": [120, 32, 25, 0.25],
            "ASKAP": [1400, 300, 10, 0.01],
            "SKA": [120, 32, 10, 0.02],
        }

        nu0, delta_nu, beam, rms = telParams[telescope]

        # calculate magnetic energy density
        U_B = self.calcMagneticPressureCGS(gas_B) * self.boltzmann  # dyne/cm^2
        Bmag = np.sqrt(U_B * 8 * np.pi)  # sqrt(dyne)/cm

        # calculate larmor frequency
        nuL_Hz = self.electron_charge * Bmag / (2 * np.pi * self.mass_electron * self.c_cgs)  # 1/s
        nuL = nuL_Hz / 1e6  # Mhz

        # calculate n_0 normalization
        gamma_fac = gamma_min ** (1 - 2 * alpha) - gamma_max ** (1 - 2 * alpha)
        n_0 = eta / (1 + k) * Bmag**2 / (8 * np.pi * self.mass_electron * self.c_cgs**2) * (2 * alpha - 1) / gamma_fac

        # calculate power in erg/s/MHz/cm^3
        freq_fac = ((nu0 + delta_nu / 2) / nuL) ** (1 - alpha) - ((nu0 - delta_nu / 2) / nuL) ** (1 - alpha)
        P_sync = (2.0 / 3.0) * self.sigma_thomson * self.c_cgs * U_B * n_0 / (delta_nu * (1 - alpha)) * freq_fac

        # multiply by gas cell volumes and divide by Hz/Mhz -> [erg/s/Hz]
        gas_vol_cgs = self.codeVolumeToCm3(gas_vol)

        P_sync = P_sync.astype("float64") * gas_vol_cgs / 1e6

        if watts_per_hz:
            # convert from erg/s/Hz to W/Hz
            P_sync *= self.erg_in_J

        P_sync = P_sync.astype("float32")

        if log:
            return logZeroNaN(P_sync)
        return P_sync

    # --- cosmology ---

    def redshiftToAgeFlat(self, z):
        """Calculate age of the universe [Gyr] at the given redshift (assuming flat cosmology).

        Analytical formula from Peebles, p.317, eq 13.2.
        """
        redshifts = np.array(z)
        if redshifts.ndim == 0:
            redshifts = np.array([z])

        with np.errstate(invalid="ignore"):  # ignore nan comparison RuntimeWarning
            w = np.where((redshifts >= 0.0) & np.isfinite(redshifts))

        age = np.zeros(redshifts.size, dtype="float32")
        age.fill(np.nan)  # leave negative/nan redshifts unset

        arcsinh_arg = np.sqrt((1 - self._sP.omega_m) / self._sP.omega_m) * (1 + redshifts[w]) ** (-3.0 / 2.0)
        age[w] = 2 * np.arcsinh(arcsinh_arg) / (self.H0_kmsMpc * 3 * np.sqrt(1 - self._sP.omega_m))
        age[w] *= 3.085678e19 / 3.15576e7 / 1e9  # Gyr

        if len(age) == 1:
            return age[0]
        return age

    def ageFlatToRedshift(self, age):
        """Calculate redshift from age of the universe [Gyr] (assuming flat cosmology).

        Inversion of analytical formula from redshiftToAgeFlat().
        """
        with np.errstate(invalid="ignore"):  # ignore nan comparison RuntimeWarning
            w = np.where((age >= 0.0) & np.isfinite(age))

        z = np.zeros(len(age), dtype="float32")
        z.fill(np.nan)

        sinh_arg = self.H0_kmsMpc * 3 * np.sqrt(1 - self._sP.omega_m)
        sinh_arg *= 3.15567e7 * 1e9 * age[w] / 2.0 / 3.085678e19

        z[w] = np.sinh(sinh_arg) / np.sqrt((1 - self._sP.omega_m) / self._sP.omega_m)
        z[w] = z[w] ** (-2.0 / 3.0) - 1

        if len(z) == 1:
            return z[0]
        return z

    def redshiftToLookbackTime(self, z):
        """Calculate lookback time from z=0 to redshift in [Gyr], assuming flat cosmology."""
        tage = self.redshiftToAgeFlat(z)
        t_z0 = self.redshiftToAgeFlat(0.0)
        return t_z0 - tage

    def redshiftToComovingDist(self, z):
        """Convert redshift z to line of sight distance (in Mpc). Assumes flat."""
        from scipy.integrate import quad

        redshifts = np.array(z)
        if redshifts.ndim == 0:
            redshifts = np.array([z])

        dist = np.zeros(redshifts.size, dtype="float32")
        dist.fill(np.nan)  # leave negative/nan redshifts unset

        hubble_dist = self.c_cgs / self.H0_h1_s / self.Mpc_in_cm / self._sP.HubbleParam

        def _qfunc(zz, omegaM, omegaL):
            return 1.0 / np.sqrt((1.0 + zz) ** 2 * (omegaM * (1.0 + zz)) + omegaL)

        for i in range(len(dist)):
            dist[i] = quad(_qfunc, 0.0, redshifts[i], args=(self._sP.omega_m, self._sP.omega_L))[0]
            dist[i] *= hubble_dist

        if len(dist) == 1:
            return dist[0]
        return dist

    def redshiftToComovingVolume(self, z):
        """Calculate total comoving volume from the present to redshift z (all-sky) in [cGpc^3]."""
        return (4 * np.pi / 3) * self.redshiftToComovingDist(z) ** 3.0 / 1e9

    def comovingVolumeDeltaRedshift(self, z1, z2, arcSeqSq=None, arcMinSq=None, arcDegSq=None, mpc3=False):
        """Calcuate comoving volume between z1 and z2 (all-sky).

        If any of arcSeqSq, arcMinSq, or arcDegSq are input, then return for this solid angle on the sky instead of 4pi.
        If mpc3, return in units of [cMpc^3] instead of the default [cGpc^3].
        """
        from scipy.integrate import quad

        z1 = np.array(z1)
        z2 = np.array(z2)
        if z1.ndim == 0:
            z1 = np.array([z1])
            z2 = np.array([z2])
        assert z1.shape == z2.shape

        dist = np.zeros(z1.size, dtype="float32")
        dist.fill(np.nan)  # leave negative/nan redshifts unset

        hubble_dist = self.c_cgs / self.H0_h1_s / self.Mpc_in_cm / self._sP.HubbleParam
        solid_angle = 4 * np.pi  # all-sky
        gpc3_factor = 1e9

        if arcDegSq is not None:
            solid_angle = arcDegSq * (np.pi / 180) ** 2
        if arcMinSq is not None:
            solid_angle = arcMinSq * (np.pi / 180) ** 2 / 60**2
        if arcSeqSq is not None:
            solid_angle = arcSeqSq * (np.pi / 180) ** 2 / 3600**2
        if mpc3:
            gpc3_factor = 1.0

        def _qfunc(zz, omegaM, omegaL):
            num = (1.0 + zz) ** 2 * self.redshiftToAngDiamDist(zz) ** 2
            denom = np.sqrt((1.0 + zz) ** 2 * (omegaM * (1.0 + zz)) + omegaL)
            return num / denom

        for i in range(len(dist)):
            dist[i] = quad(_qfunc, z1[i], z2[i], args=(self._sP.omega_m, self._sP.omega_L))[0]
            dist[i] /= gpc3_factor
            dist[i] *= hubble_dist
            dist[i] *= solid_angle

        if len(dist) == 1:
            return dist[0]
        return dist

    def redshiftToAngDiamDist(self, z):
        """Convert redshift z to angular diameter distance (in Mpc).

        This equals the proper/physical transverse distance for theta=1 rad. Assumes flat. Peebles, p.325.
        """
        if z == 0.0:
            # absolute, 10 pc [in Mpc]
            return 10.0 / 1e6
        return self.redshiftToComovingDist(z) / (1.0 + z)

    def redshiftToLumDist(self, z):
        """Convert redshift z to luminosity distance (in Mpc).

        This then allows the conversion between luminosity and a flux at that redshift.
        """
        if not isinstance(z, np.ndarray) and z == 0.0:
            # absolute, 10 pc [in Mpc]
            return 10.0 / 1e6

        lumdist = self.redshiftToComovingDist(z) * (1.0 + z)
        if isinstance(lumdist, np.ndarray):
            # absolute, 10 pc [in Mpc]
            lumdist[np.where(z == 0)] = 10.0 / 1e6

        return lumdist

    def arcsecToAngSizeKpcAtRedshift(self, ang_diam, z=None):
        """Convert an angle in arcseconds to an angular/transverse size (in proper/physical kpc) at redshift z.

        Assumes flat cosmology.
        """
        if z is None:
            z = self._sP.redshift

        dA = self.redshiftToAngDiamDist(z)
        size_mpc = dA * ang_diam * self.arcsec_in_rad
        return size_mpc * 1000.0

    def arcsecToCodeLength(self, x_arcsec, z=None):
        """Convert an angle in arcseconds to an angular/tranverse size (in code length units)."""
        x_kpc = self.arcsecToAngSizeKpcAtRedshift(x_arcsec, z)
        return self.physicalKpcToCodeLength(x_kpc)

    def degToAngSizeKpcAtRedshift(self, ang_diam_deg, z):
        """Convert an angle in degrees to a physical size [kpc] at redshift z."""
        arcsec_per_deg = 60 * 60
        return self.arcsecToAngSizeKpcAtRedshift(ang_diam_deg * arcsec_per_deg, z)

    def codeLengthToAngularSize(self, length_codeunits, z=None, arcsec=True, arcmin=False, deg=False):
        """Convert a distance in code units (i.e. ckpc/h) to an angular scale in [arcsec] at a given redshift z.

        Assumes flat cosmology. If arcmin or deg is True, then [arcmin] or [deg].
        """
        if z is None:
            z = self._sP.redshift

        dA = self.redshiftToAngDiamDist(z)
        size_mpc = self.codeLengthToMpc(length_codeunits)
        ang_size = size_mpc / dA / self.arcsec_in_rad  # arcsec
        if arcmin:
            ang_size /= 60.0
        if deg:
            ang_size /= 3600.0

        return ang_size

    def physicalKpcToAngularSize(self, length_kpc, z=None, arcsec=True, arcmin=False, deg=False):
        """Convert a distance in physical kpc to an angular scale in [arcsec] at the given redshift z.

        Assumes flat cosmology. If arcmin or deg is True, then [arcmin] or [deg].
        """
        if z is None:
            z = self._sP.redshift

        dA = self.redshiftToAngDiamDist(z)
        ang_size = (length_kpc / 1000) / dA / self.arcsec_in_rad  # arcsec
        if arcmin:
            ang_size /= 60.0
        if deg:
            ang_size /= 3600.0

        return ang_size

    def magsToSurfaceBrightness(self, vals, pxSizeCode):
        """Convert magnitudes (probably on a grid) to surface brightness values, given a constant pxSizeCode."""
        pxSizeX = self.codeLengthToAngularSize(pxSizeCode[0], arcsec=True)
        pxSizeY = self.codeLengthToAngularSize(pxSizeCode[1], arcsec=True)
        pxAreaArcsecSq = pxSizeX * pxSizeY
        return vals + 2.5 * np.log10(pxAreaArcsecSq)

    def haloMassToOtherOverdensity(self, mass_orig, delta_orig=200, delta_new=500):
        """Convert a halo mass between two different spherical overdensity definitions, e.g. M200 to M500.

        Assumes NFW density profile and the c-M relation from Bullock by default.
        Based on https://arxiv.org/abs/astro-ph/0203169 (appendix).
        Input halo mass (and output) in units of linear Msun.
        """
        # derive concentration according to Bullock+
        Mstar = 2.77e12 / 0.6774  # msun
        scalefac = 1.0
        c = 9.0 * scalefac * (mass_orig / Mstar) ** (-0.13)

        def _convinv(x):
            """Fitting function from Hu & Kravtsov (2003)."""
            a2 = 0.5116
            a3 = -1.285 / 3
            a4 = -3.13e-3
            a5 = -3.52e-5
            p = a3 + a4 * np.log(x) + a5 * (np.log(x)) ** 2
            convinv = a2 * x ** (2 * p) + (3 / 4) ** 2
            convinv = 1.0 / np.sqrt(convinv) + 2 * x
            return convinv

        # M_target
        ratio = delta_orig / delta_new
        fval = np.log(1 + c) - c / (1 + c)
        fval = fval / ratio / c**3

        new_c = _convinv(fval)

        mtarget = mass_orig * ((c * new_c) ** (-3.0) / ratio)

        return mtarget

    def m200_to_m500(self, halo_mass):
        """Convert a halo mass from Delta=200 to Delta=500."""
        return self.haloMassToOtherOverdensity(halo_mass, delta_orig=200, delta_new=500)

    def m500_to_m200(self, halo_mass):
        """Convert a halo mass from Delta=500 to Delta=200."""
        return self.haloMassToOtherOverdensity(halo_mass, delta_orig=500, delta_new=200)

    # --- other ---

    def particleCountToMass(self, N_part, baryon=True, boxLength=None):
        """Convert the cube-root of a total particle count (e.g. 512, 1820) to its average mass.

        Uses the cosmology and volume of the box.
        If boxLength is specified, then should be a box side-length in cMpc/h units (e.g. 25, 205).
        Return in [Msun].
        """
        omega = self._sP.omega_b if baryon else self._sP.omega_m
        if boxLength is None:
            boxLength = self._sP.boxSize / 1000  # cMph/h

        vol = (boxLength / self._sP.HubbleParam) ** 3  # cMpc^3
        rhoAvg = self.rhoCrit_msunMpc3 * omega  # msun / cMpc^3
        totMass = vol * rhoAvg  # msun
        m_gas = totMass / N_part**3  # msun

        return m_gas

    def meanmolwt(self, Y, Z):
        """Mean molecular weight, from Monaco+ (2007) eqn 14, for hot halo gas.

        Y = helium fraction (0.25)
        Z = metallicity (non-log metal mass/total mass)
        """
        mu = 4.0 / (8 - 5 * Y - 6 * Z)
        return mu
