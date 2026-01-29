"""
Stellar population synthesis, evolution, photometrics (FSPS).
"""

from getpass import getuser
from os import environ, mkdir
from os.path import isdir, isfile

import h5py
import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from ..util.helper import iterable, logZeroMin, logZeroNaN, rootPath, trapsum
from ..util.rotation import rotateCoordinateArray, rotationMatrixFromVec
from ..util.simParams import simParams
from ..util.sphMap import sphMap


@jit(nopython=True, nogil=True, cache=True)
def _dust_tau_model_lum(
    N_H,
    Z_g,
    ages_logGyr,
    metals_log,
    masses_msun,
    wave,
    A_lambda_sol,
    redshift,
    beta,
    Z_solar,
    gamma,
    N_H0,
    f_scattering,
    metals,
    ages,
    wave_ang,
    spec,
    calzetti_case,
):
    """Helper for sps.dust_tau_model_mags(). Cannot JIT a class member function, so it sits here."""
    # accumulate per star attenuated luminosity (wavelength dependent):
    obs_lum = np.zeros(wave.size, dtype=np.float32)
    atten = np.zeros(wave.size, dtype=np.float32)
    gamma_term = np.zeros(wave.size, dtype=np.float32)

    gamma_w_lt200 = np.where(wave_ang < 2000.0)
    gamma_w_gt200 = np.where(wave_ang >= 2000.0)

    for i in range(N_H.size):
        # taking np.power() using gamma as a full array does ~6000 powers, need to avoid for efficiency
        gamma_val_lt200 = np.power(Z_g[i] / Z_solar, 1.35)
        gamma_val_gt200 = np.power(Z_g[i] / Z_solar, 1.6)

        gamma_term[gamma_w_lt200] = gamma_val_lt200
        gamma_term[gamma_w_gt200] = gamma_val_gt200

        # finish absorption tau and total tau
        tau_a = A_lambda_sol * np.power(1 + redshift, beta) * gamma_term * (N_H[i] / N_H0)

        tau_lambda = tau_a * f_scattering

        # attenuation as a function of wavelength
        atten *= 0.0
        atten += 1.0  # reset to one

        if calzetti_case == 3:
            # 'uniform scattering slab'
            atten[:] = np.exp(-tau_lambda)

        if calzetti_case == 5:
            # 'internal dust'
            # leave atten at 1.0 (no change) for tau_lambda->0, and use 1e-5 threshold to avoid
            # numerical truncation setting atten=0 for tau_lambda~0 (very small)
            w = np.where(tau_lambda >= 1e-5)
            atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]

        # bilinear interpolation: stellar population spectrum
        x = metals_log[i]
        y = ages_logGyr[i]

        ##x_ind = np.interp( x, metals, np.arange(metals.size) )
        ##y_ind = np.interp( y, ages,   np.arange(ages.size) )
        for x_ind0 in range(metals.size):
            if x < metals[x_ind0]:
                break
        for y_ind0 in range(ages.size):
            if y < ages[y_ind0]:
                break

        xt = (x - metals[x_ind0 - 1]) / (metals[x_ind0] - metals[x_ind0 - 1])
        yt = (y - ages[y_ind0 - 1]) / (ages[y_ind0] - ages[y_ind0 - 1])
        x_ind = (x_ind0 - 1) * (1 - xt) + x_ind0 * xt
        y_ind = (y_ind0 - 1) * (1 - yt) + y_ind0 * yt

        # set indices out of bounds on purpose if we are in extrapolation regime, which then causes
        # below x1_ind==x2_ind or y1_ind==y2_ind such that constant (not linear) extrapolation occurs
        if x < metals[0]:
            x_ind = -1.0
        if x > metals[-1]:
            x_ind = metals.size - 1
        if y < ages[0]:
            y_ind = -1.0
        if y > ages[-1]:
            y_ind = ages.size - 1

        # clip indices at [0,size] which leads to constant extrap (nearest grid edge value in that dim)
        x1_ind = np.int32(np.floor(x_ind))
        x2_ind = x1_ind + 1
        y1_ind = np.int32(np.floor(y_ind))
        y2_ind = y1_ind + 1

        if x1_ind < 0 or x2_ind < 0:
            x1_ind = 0
            x2_ind = 0
        if y1_ind < 0 or y2_ind < 0:
            y1_ind = 0
            y2_ind = 0
        if x1_ind > metals.size - 1 or x2_ind > metals.size - 1:
            x1_ind = metals.size - 1
            x2_ind = metals.size - 1
        if y1_ind > ages.size - 1 or y2_ind > ages.size - 1:
            y1_ind = ages.size - 1
            y2_ind = ages.size - 1

        spec_12 = spec[x1_ind, y2_ind, :]
        spec_21 = spec[x2_ind, y1_ind, :]
        spec_11 = spec[x1_ind, y1_ind, :]
        spec_22 = spec[x2_ind, y2_ind, :]

        x1 = metals[x1_ind]
        x2 = metals[x2_ind]
        y1 = ages[y1_ind]
        y2 = ages[y2_ind]

        # constant beyond edges, make denominator nonzero
        if x2_ind == x1_ind == metals.size - 1:
            x2 += metals[-1] - metals[-2]
        if x2_ind == x1_ind == 0:
            x1 -= metals[1] - metals[0]
        if y2_ind == y1_ind == ages.size - 1:
            y2 += ages[-1] - ages[-2]
        if y2_ind == y1_ind == 0:
            y1 -= ages[1] - ages[0]

        # interpolated 1D spectrum
        spectrum_local = (
            spec_11 * (x2 - x) * (y2 - y)
            + spec_21 * (x - x1) * (y2 - y)
            + spec_12 * (x2 - x) * (y - y1)
            + spec_22 * (x - x1) * (y - y1)
        )
        spectrum_local /= (x2 - x1) * (y2 - y1)

        # spectrum_local = np.clip(spectrum_local, 0.0, np.inf) # enforce everywhere positive
        w = np.where(spectrum_local < 0.0)
        spectrum_local[w] = 0.0

        # TODO: support for velocity shift

        # accumulate attenuated contribution of this stellar population
        obs_lum += (spectrum_local * masses_msun[i]) * atten

    # return full attenuated spectrum, for later convlution with some band
    return obs_lum


@jit(nopython=True, nogil=True, cache=True)
def _dust_tau_model_lum_indiv(
    N_H,
    Z_g,
    ages_logGyr,
    metals_log,
    masses_msun,
    wave,
    A_lambda_sol,
    redshift,
    beta,
    Z_solar,
    gamma,
    N_H0,
    f_scattering,
    metals,
    ages,
    wave_ang,
    spec,
    calzetti_case,
):
    """Helper for sps.dust_tau_model_mags(). Cannot JIT a class member function, so it sits here."""
    # accumulate per star attenuated luminosity (wavelength dependent):
    obs_lum_indiv = np.zeros((N_H.size, wave.size), dtype=np.float32)
    atten = np.zeros(wave.size, dtype=np.float32)
    gamma_term = np.zeros(wave.size, dtype=np.float32)

    gamma_w_lt200 = np.where(wave_ang < 2000.0)
    gamma_w_gt200 = np.where(wave_ang >= 2000.0)

    for i in range(N_H.size):
        # taking np.power() using gamma as a full array does ~6000 powers, need to avoid for efficiency
        gamma_val_lt200 = np.power(Z_g[i] / Z_solar, 1.35)
        gamma_val_gt200 = np.power(Z_g[i] / Z_solar, 1.6)

        gamma_term[gamma_w_lt200] = gamma_val_lt200
        gamma_term[gamma_w_gt200] = gamma_val_gt200

        # finish absorption tau and total tau
        tau_a = A_lambda_sol * np.power(1 + redshift, beta) * gamma_term * (N_H[i] / N_H0)

        tau_lambda = tau_a * f_scattering

        # attenuation as a function of wavelength
        atten *= 0.0
        atten += 1.0  # reset to one

        if calzetti_case == 3:
            # 'uniform scattering slab'
            atten[:] = np.exp(-tau_lambda)

        if calzetti_case == 5:
            # 'internal dust'
            # leave atten at 1.0 (no change) for tau_lambda->0, and use 1e-5 threshold to avoid
            # numerical truncation setting atten=0 for tau_lambda~0 (very small)
            w = np.where(tau_lambda >= 1e-5)
            atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]

        # bilinear interpolation: stellar population spectrum
        x = metals_log[i]
        y = ages_logGyr[i]

        ##x_ind = np.interp( x, metals, np.arange(metals.size) )
        ##y_ind = np.interp( y, ages,   np.arange(ages.size) )
        for x_ind0 in range(metals.size):
            if x < metals[x_ind0]:
                break
        for y_ind0 in range(ages.size):
            if y < ages[y_ind0]:
                break

        xt = (x - metals[x_ind0 - 1]) / (metals[x_ind0] - metals[x_ind0 - 1])
        yt = (y - ages[y_ind0 - 1]) / (ages[y_ind0] - ages[y_ind0 - 1])
        x_ind = (x_ind0 - 1) * (1 - xt) + x_ind0 * xt
        y_ind = (y_ind0 - 1) * (1 - yt) + y_ind0 * yt

        # set indices out of bounds on purpose if we are in extrapolation regime, which then causes
        # below x1_ind==x2_ind or y1_ind==y2_ind such that constant (not linear) extrapolation occurs
        if x < metals[0]:
            x_ind = -1.0
        if x > metals[-1]:
            x_ind = metals.size - 1
        if y < ages[0]:
            y_ind = -1.0
        if y > ages[-1]:
            y_ind = ages.size - 1

        # clip indices at [0,size] which leads to constant extrap (nearest grid edge value in that dim)
        x1_ind = np.int32(np.floor(x_ind))
        x2_ind = x1_ind + 1
        y1_ind = np.int32(np.floor(y_ind))
        y2_ind = y1_ind + 1

        if x1_ind < 0 or x2_ind < 0:
            x1_ind = 0
            x2_ind = 0
        if y1_ind < 0 or y2_ind < 0:
            y1_ind = 0
            y2_ind = 0
        if x1_ind > metals.size - 1 or x2_ind > metals.size - 1:
            x1_ind = metals.size - 1
            x2_ind = metals.size - 1
        if y1_ind > ages.size - 1 or y2_ind > ages.size - 1:
            y1_ind = ages.size - 1
            y2_ind = ages.size - 1

        spec_12 = spec[x1_ind, y2_ind, :]
        spec_21 = spec[x2_ind, y1_ind, :]
        spec_11 = spec[x1_ind, y1_ind, :]
        spec_22 = spec[x2_ind, y2_ind, :]

        x1 = metals[x1_ind]
        x2 = metals[x2_ind]
        y1 = ages[y1_ind]
        y2 = ages[y2_ind]

        # constant beyond edges, make denominator nonzero
        if x2_ind == x1_ind == metals.size - 1:
            x2 += metals[-1] - metals[-2]
        if x2_ind == x1_ind == 0:
            x1 -= metals[1] - metals[0]
        if y2_ind == y1_ind == ages.size - 1:
            y2 += ages[-1] - ages[-2]
        if y2_ind == y1_ind == 0:
            y1 -= ages[1] - ages[0]

        # interpolated 1D spectrum
        spectrum_local = (
            spec_11 * (x2 - x) * (y2 - y)
            + spec_21 * (x - x1) * (y2 - y)
            + spec_12 * (x2 - x) * (y - y1)
            + spec_22 * (x - x1) * (y - y1)
        )
        spectrum_local /= (x2 - x1) * (y2 - y1)

        # spectrum_local = np.clip(spectrum_local, 0.0, np.inf) # enforce everywhere positive
        w = np.where(spectrum_local < 0.0)
        spectrum_local[w] = 0.0

        # accumulate attenuated contribution of this stellar population
        obs_lum_indiv[i, :] = (spectrum_local * masses_msun[i]) * atten

    # return full attenuated spectra, for later convlution with bands
    return obs_lum_indiv


class sps:
    """Use pre-computed FSPS stellar photometrics tables to derive magnitudes for simulation stars."""

    basePath = rootPath + "/tables/fsps/"

    imfTypes = {"salpeter": 0, "chabrier": 1, "kroupa": 2}
    isoTracks = ["mist", "padova07", "parsec", "basti", "geneva"]
    stellarLib = ["miles", "basel", "csk"]  # unused herein, but we tend to assume MILES selected in FSPS
    dustModels = ["none", "cf00", "cf00_res_eff", "cf00b_res_conv", "cf00_res_conv", "cf00_res3_conv"]

    def __init__(
        self, sP, iso="padova07", imf="chabrier", dustModel="cf00_res_conv", order=3, redshifted=False, emlines=False
    ):
        """Load the pre-computed stellar photometrics table, computing if it does not yet exist.

        If redshifted, then band-magnitudes are attenuated and wavelength shifted into observer-frame
        at sP.redshift when initially computed (note: self.wave, self.wave_ang, self.spec are not, and
        are saved as-is, i.e. always rest-frame).
        Otherwise, band-magnitudes and spectra are in the rest-frame, regardless of sP.redshift.
        If emlines, include nebular emission line model for band-magnitudes and spectra, otherwise no.
        """
        import fsps

        assert iso in self.isoTracks
        assert imf in self.imfTypes
        assert dustModel in self.dustModels

        # if not redshifted and sP.redshift is not None and sP.redshift > 0.0 and getuser() != 'wwwrun':
        #    print(' WARNING: sP redshift = %.2f, but not redshifting SPS calculations!' % sP.redshift)

        self.sP = sP
        self.data = {}  # band magnitudes
        self.spec = {}  # spectra
        self.order = order  # bicubic interpolation by default (1 = bilinear)
        self.bands = fsps.find_filter("")  # do them all (previously 138, then 143, now 159)
        self.redshifted = redshifted
        self.emlines = emlines

        self.dust = dustModel.split("_")[0]
        self.dustModel = dustModel

        zStr = ""
        if redshifted:
            zStr = "_z=%.1f" % sP.redshift
            # print(' COMPUTING STELLAR MAGS/SPECTRA WITH REDSHIFT (z=%.1f)!' % sP.redshift)
            # cosmology in FSPS is hard-coded (sps_vars.f90), and this has been set to TNG values
            assert sP.omega_m == 0.3089 and sP.omega_L == 0.6911 and sP.HubbleParam == 0.6774
        if emlines:
            zStr += "_em"

        saveFilename = self.basePath + "mags_%s_%s_%s_bands-%d%s.hdf5" % (iso, imf, self.dust, len(self.bands), zStr)

        if not isdir(self.basePath):
            mkdir(self.basePath)

        # no saved table? compute now
        if not isfile(saveFilename):
            print(f" Compute new stellarPhotTable: [{iso = } imf={imf} dust={dustModel} bands={len(self.bands)}]...")
            self.computePhotTable(iso, imf, saveFilename)

        # load
        with h5py.File(saveFilename, "r") as f:
            self.bands = [b.decode("ascii") for b in f["bands"][()]]
            self.ages = f["ages_logGyr"][()]
            self.metals = f["metals_log"][()]
            self.wave = f["wave_nm"][()]
            self.spec = f["spec_lsun_hz"][()]

            self.emline_names = f["emline_names"]
            self.emline_wave = f["emline_wave_nm"]
            self.emline = f["emline_lsun"][()]

            for key in f:
                if "mags_" in key:
                    self.data[key] = f[key][()]

        # pre-compute for dust model
        self.prep_filters()
        self.prep_dust_models()

    def computePhotTable(self, iso, imf, saveFilename):
        """Compute a new photometrics table for the given (iso,imf,self.dust) using fsps."""
        import fsps

        if self.dust == "none":
            dust_type = 0
            dust1 = 0.0
            dust2 = 0.0
            dust_index = 0.0
            dust1_index = 0.0
            dust_tesc = 7.0  # log(yr)

        if self.dust == "bc00":
            # see Conroy+ (2009) or Charlot & Fall (2000) - note 'bc00' is just a typo for 'cf00' kept
            #   here with dust1=1.0 which, because the young populations have both attenuation terms
            #   applied in FSPS, in reality means dust1=1.0+dust2 in the below equation:
            # tau_dust(lambda) = tau_1 * (lambda/lambda_0)^alpha_1      t_ssp <= t_bc
            #                    tau_2 * (lambda/lambda_0)^alpha_2      t_ssp  > t_bc
            dust_type = 0  # powerlaw taking the above functional form
            dust1 = 1.0  # tau_1
            dust2 = 0.3  # tau_2
            dust_index = -0.7  # alpha_2
            dust1_index = -0.7  # alpha_1
            dust_tesc = 7.0  # t_bc [log(yr)] = 0.01 Gyr, timescale to escape/disrupt molecular birth cloud

        if self.dust == "cf00":
            # same as 'bc00', with the real citation name, and dust1 changed to 0.7 such that
            # tau_1 = 1.0 in the given functional form
            dust_type = 0  # powerlaw taking the above functional form
            dust1 = 0.7  # tau_1
            dust2 = 0.3  # tau_2
            dust_index = -0.7  # alpha_2
            dust1_index = -0.7  # alpha_1
            dust_tesc = 7.0  # t_bc

        if self.dust == "cf00b":
            # same as 'cf00' except no diffuse/old attenuation (e.g. assume this is separately taken
            # into account with a resolved dust computation)
            dust_type = 0  # powerlaw taking the above functional form
            dust1 = 1.0  # tau_1
            dust2 = 0.0  # tau_2
            dust_index = -0.7  # alpha_2
            dust1_index = -0.7  # alpha_1
            dust_tesc = 7.0  # t_bc

        # init
        pop = fsps.StellarPopulation(
            sfh=0,  # SSP
            zmet=1,  # integer index of metallicity value (modified later)
            add_neb_continuum=True,
            add_neb_emission=self.emlines,  # modifies get_mags()
            add_dust_emission=True,
            imf_type=self.imfTypes[imf],
            dust_type=dust_type,
            dust1=dust1,
            dust2=dust2,
            dust_index=dust_index,
            dust_tesc=dust_tesc,
            dust1_index=dust1_index,
        )

        assert pop.spec_library == b"miles"
        assert pop.isoc_library == b"pdva"  # padova07, otherwise generalize this

        # different tracks are available at discrete metallicities (linear mass_Z/mass_tot, not in solar!)
        if iso == "padova07":
            Zsolar = 0.019
            metals = [0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.0012,0.0016,0.002,0.0025,0.0031,0.0039,
                      0.0049,0.0061,0.0077,0.0096,0.012,0.015,0.019,0.024,0.03]  # fmt: skip
        if iso == "basti":
            Zsolar = 0.02
            metals = [0.0003, 0.0006, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.03, 0.04]
        if iso == "geneva":
            Zsolar = 0.02
            metals = [0.001, 0.004, 0.008, 0.02, 0.04]
        if iso == "mist":
            Zsolar = 0.0142
            metals = Zsolar * 10.0 ** np.array(
                [-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
            )
        if iso == "parsec":
            Zsolar = 0.0152
            metals = [0.0001,0.0002,0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.014,
                      0.017,0.02,0.03,0.04,0.06]  # fmt: skip

        assert len(metals) == pop.zlegend.size, "Likely mismatch of isochrone choice here and in FSPS."

        # get sizes of full spectra
        wave0, spec0 = pop.get_spectrum()  # Lsun/Hz, Angstroms

        # save struct and spectral array
        mags = {}

        for band in self.bands:
            mags[band] = np.zeros((pop.zlegend.size, pop.log_age.size), dtype="float32")
        spec = np.zeros((pop.zlegend.size, pop.log_age.size, wave0.size), dtype="float32")

        # get nebular emission line names, wavelengths
        line_wave = pop.emline_wavelengths
        line_file = environ["SPS_HOME"] + "/data/emlines_info.dat"
        with open(line_file) as f:
            line_file = [fline.strip() for fline in f.readlines()]

        line_file = [fline.split(",") for fline in line_file]
        line_wave_check = [float(fline[0]) for fline in line_file]
        line_names = [fline[1].encode("ascii") for fline in line_file]

        assert np.abs(line_wave - line_wave_check).max() <= 0.5  # make sure order is correct

        emline = np.zeros((pop.zlegend.size, pop.log_age.size, line_wave.size), dtype="float32")

        # loop over metallicites, compute band magnitudes (and full spectra) over an age grid for each
        for i in range(pop.zlegend.size):
            if getuser() != "wwwrun":
                print("  [%2d of %2d] Z = %g" % (i, pop.zlegend.size, pop.zlegend[i]))

            # update metallicity step
            pop.params["zmet"] = i + 1  # 1-indexed

            # if including nebular emission, update gas-phase metallicity and gas ionization parameter
            # note: for full self-consistency, should have Z_gas == Z_stars
            if self.emlines:
                pop.params["gas_logz"] = np.log10(pop.zlegend[i] / Zsolar)  # in units of log (Z_gas/Z_sun)
                pop.params["gas_logu"] = -2.0  # log ionization parameter, default (how to choose? obs scaling?)
                print("   warning: Inclusion of emission lines assumes default, constant U = -2.0.")

            # request magnitudes in all bands
            redshift = self.sP.redshift if self.redshifted else None
            x = pop.get_mags(bands=self.bands, redshift=redshift)  # by default, z=None (redshift zero)

            w, s = pop.get_spectrum(peraa=False)  # Lsun/Hz, Angstroms
            assert np.array_equal(w, wave0)  # we assume same wavelengths for all metal indices
            assert s.shape[0] == pop.log_age.size  # should be same age grid in isochrones

            # put magnitudes into (age,Z) grids split by band (either absolute or apparent)
            for bandNum, bandName in enumerate(self.bands):
                mags[bandName][i, :] = x[:, bandNum]

            # save spectral array (always rest-frame/absolute)
            for j in range(pop.log_age.size):
                spec[i, j, :] = s[j, :]

        # loop again, this time compute nebular line emission luminosities (modify pop each time)
        for i in range(pop.zlegend.size):
            if getuser() != "wwwrun":
                print("  [%2d of %2d] Z = %g (nebular)" % (i, pop.zlegend.size, pop.zlegend[i]))

            pop = fsps.StellarPopulation(
                sfh=0,  # SSP
                zmet=i + 1,
                add_neb_continuum=True,
                add_neb_emission=True,  # need True for emline properties
                gas_logz=np.log10(pop.zlegend[i] / Zsolar),
                gas_logu=-2.0,  # default
                add_dust_emission=True,
                imf_type=self.imfTypes[imf],
                dust_type=dust_type,
                dust1=dust1,
                dust2=dust2,
                dust_index=dust_index,
                dust_tesc=dust_tesc,
                dust1_index=dust1_index,
            )

            for j in range(pop.log_age.size):
                emline[i, j, :] = pop.emline_luminosity[j, :]

        # save
        with h5py.File(saveFilename, "w") as f:
            f["bands"] = [b.encode("ascii") for b in self.bands]
            f["ages_logGyr"] = np.array(pop.log_age - 9.0, dtype="float32")  # log(yr) -> log(Gyr)
            f["metals_log"] = np.array(np.log10(pop.zlegend), dtype="float32")  # linear -> log
            f["wave_nm"] = np.array(wave0 / 10.0, dtype="float32")  # Ang -> nm

            for key in mags:
                f["mags_" + key] = mags[key]  # AB absolute
            f["spec_lsun_hz"] = spec  # Lsun/Hz

            f["emline_wave_nm"] = line_wave / 10.0  # Ang -> nm
            f["emline_lsun"] = emline  # Lsun
            f["emline_names"] = line_names

        print("Saved: [%s]" % saveFilename)

    def filters(self, select=None):
        """Return name of available filters."""
        if select is not None:
            return [band for band in self.bands if select in band]
        return self.bands

    def has_filter(self, filterName):
        """Return True or False if the pre-computed table contains the specified filter/band."""
        return filterName.lower() in self.bands

    def prep_dust_models(self):
        """Do possibly expensive pre-calculations for (resolved) dust model."""
        if "_res" not in self.dustModel:
            return

        self.lambda_nm = {}
        self.A_lambda_sol = {}
        self.f_scattering = {}
        self.gamma = {}

        self.beta = -0.5
        self.N_H0 = 2.1e21  # neutral hydrogen column density [cm^-2]

        for band in self.bands:
            if "suprimecam" in band:
                continue  # missing transmission data

            # get wavelength array
            if "_eff" in self.dustModel:
                # get single (lambda_eff) luminosity attenuation factor for each star
                lambda_nm = np.array([self.lambda_eff[band]])

            if "_conv" in self.dustModel:
                # do full convolution of original stellar spectrum with tau(lambda)
                # lambda_nm = self.trans_lambda[band] # at the resolution of the transmission function
                lambda_nm = self.wave  # at the resolution of the stellar spectra

            # get tau^a factor from absorption (Cardelli 1989 equations 1-3b)
            x = 1 / (lambda_nm / 1000)  # inverse microns

            R_V = 3.1  # e.g. MW/LMC value (2.7 for SMC)
            a_x = np.zeros(lambda_nm.size, dtype="float32")
            b_x = np.zeros(lambda_nm.size, dtype="float32")

            # infrared regime (0.91 um < lambda < 3.3 um)
            w_lt = np.where(x < 1.1)

            a_x[w_lt] = 0.574 * x[w_lt] ** 1.61
            b_x[w_lt] = -0.527 * x[w_lt] ** 1.61

            # optical/NIR regime (0.3 um < lambda < 0.91)
            w_gt = np.where(x >= 1.1)
            y = x[w_gt] - 1.82

            a_x[w_gt] = (
                1
                + 0.17699 * y**1
                - 0.50447 * y**2
                - 0.02427 * y**3
                + 0.72085 * y**4
                + 0.01979 * y**5
                - 0.77530 * y**6
                + 0.32999 * y**7
            )
            b_x[w_gt] = (
                0
                + 1.41338 * y**1
                + 2.28305 * y**2
                + 1.07233 * y**3
                - 5.38434 * y**4
                - 0.62251 * y**5
                + 5.30260 * y**6
                - 2.09002 * y**7
            )

            # UV regime (0.125 um < lambda < 0.3 um)
            w_gt = np.where(x >= 3.3)
            w_gt59 = np.where(x >= 5.9)

            F_a = np.zeros(lambda_nm.size, dtype="float32")
            F_b = np.zeros(lambda_nm.size, dtype="float32")

            F_a[w_gt59] = -0.04473 * (x[w_gt59] - 5.9) ** 2 - 0.009779 * (x[w_gt59] - 5.9) ** 3
            F_b[w_gt59] = 0.21300 * (x[w_gt59] - 5.9) ** 2 + 0.120700 * (x[w_gt59] - 5.9) ** 3

            a_x[w_gt] = 1.752 - 0.316 * x[w_gt] - 0.104 / ((x[w_gt] - 4.67) ** 2 + 0.341) + F_a[w_gt]
            b_x[w_gt] = -3.09 + 1.825 * x[w_gt] + 1.206 / ((x[w_gt] - 4.62) ** 2 + 0.263) + F_b[w_gt]

            # far-UV regime (0.1 um < lambda < 0.125)
            w_gt = np.where(x >= 8.0)

            a_x[w_gt] = -1.073 - 0.628 * (x[w_gt] - 8) + 0.137 * (x[w_gt] - 8) ** 2 - 0.070 * (x[w_gt] - 8) ** 3
            b_x[w_gt] = 13.670 + 4.257 * (x[w_gt] - 8) - 0.420 * (x[w_gt] - 8) ** 2 + 0.374 * (x[w_gt] - 8) ** 3

            # outside scope
            w_gt = np.where(x > 10.0)  # these values are >1 and growing, but divergent as lambda->0
            a_x[w_gt] = a_x[np.where(x < 10.0)].max()
            b_x[w_gt] = b_x[np.where(x < 10.0)].max()
            w_lt = np.where(x < 0.3)  # these values anyways tend to zero
            a_x[w_lt] = 0.0
            b_x[w_lt] = 0.0

            self.A_lambda_sol[band] = a_x + b_x / R_V

            ww = np.where(self.A_lambda_sol[band] < 0.0)
            self.A_lambda_sol[band][ww] = 0.0  # clip negative values

            gamma = np.zeros(lambda_nm.size, dtype="float32")
            gamma[np.where(lambda_nm >= 200)] = 1.6
            gamma[np.where(lambda_nm < 200)] = 1.35

            # get full tau factor accounting for scattering (Calzetti 1994 internal dust model #5)
            h_lambda = np.zeros(lambda_nm.size, dtype="float32")
            omega_lambda = np.zeros(lambda_nm.size, dtype="float32")

            yy = np.log10(lambda_nm * 10.0)
            h_lambda = 1.0 - 0.561 * np.exp(-(np.abs(yy - 3.3112) ** 2.2) / 0.17)

            w_lt = np.where(lambda_nm <= 346.0)
            w_gt = np.where(lambda_nm > 346.0)

            omega_lambda[w_lt] = 0.43 + 0.366 * (1 - np.exp(-(yy[w_lt] - 3) * (yy[w_lt] - 3) / 0.2))
            omega_lambda[w_gt] = -0.48 * yy[w_gt] + 2.41

            # note these are only valid in (100 nm < lambda < 1000 nm) and the given functional forms
            # are not so well behaved outside this range, so we should probably enforce constant value
            # at the edges of this range if we were ever to extend the used bands
            h_lambda = np.clip(h_lambda, 0.0, 1.0)
            omega_lambda = np.clip(omega_lambda, 0.0, 1.0)

            self.f_scattering[band] = h_lambda * np.sqrt(1 - omega_lambda) + (1 - h_lambda) * (1 - omega_lambda)

            self.lambda_nm[band] = lambda_nm
            self.gamma[band] = gamma

    def prep_filters(self):
        """Extract filter properties in case we want them later."""
        import fsps

        self.lambda_eff = {}
        self.msun_ab = {}
        self.msun_vega = {}
        self.trans_lambda = {}
        self.trans_val = {}
        self.trans_normed = {}
        self.wave_ang = self.wave * 10.0

        for band in self.bands:
            if "suprimecam" in band or "ps1_" in band or "roman_f184" in band:
                continue  # missing transmission data

            f = fsps.get_filter(band)

            # get filter general properties
            self.lambda_eff[band] = f.lambda_eff / 10.0  # nm
            self.msun_ab[band] = f.msun_ab
            self.msun_vega[band] = f.msun_vega

            # get transmission of filter
            trans_lambda, trans_val = f.transmission
            self.trans_lambda[band] = np.array(trans_lambda) / 10.0  # nm, make sure to copy
            self.trans_val[band] = np.array(trans_val)

            # interpolate transmission function onto master wavelength grid
            trans_val = np.interp(self.wave, self.trans_lambda[band], self.trans_val[band])

            # normalize
            trans_norm = trapsum(self.wave_ang, trans_val / self.wave_ang)
            if trans_norm <= 0.0:
                trans_norm = 1.0  # band entirely outside wavelength array
            trans_val /= trans_norm
            trans_val[np.where(trans_val < 0.0)] = 0.0  # no negative values

            # pre-divide out Angstroms
            self.trans_normed[band] = trans_val / self.wave_ang

    def convertSpecToSDSSUnitsAndAttenuate(self, spec, output_wave=None):
        """Convert a spectrum from FSPS in [Lsun/Hz] to SDSS units [10^-17 erg/cm^2/s/Ang], possibly redshifted.

        If self.sP.redshift > 0, attenuate the spectrum by the luminosity distance.
        If self.sP.redshift == 0, the rest-frame spectrum is returned at an
        assumed distance of 10pc (i.e. absolute magnitudes). If output_wave is not None, should
        be in Angstroms.
        """
        # shift in wavelength (afterwards, self.wave_ang is now observed-frame instead of rest-frame)
        spec, wave = self.redshiftSpectrum(spec, output_wave=output_wave)

        # convert [Lsun/Hz] -> [Lsun/Ang]
        freq_Hz = self.sP.units.c_ang_per_sec / wave  # nu = c/lambda
        spec_perAng = freq_Hz**2 * spec / self.sP.units.c_ang_per_sec  # flux_nu = (lambda^2/c) * flux_lambda

        # if z=0 set dL=10pc (absolute), otherwise use the actual redshift, calculate luminosity distance
        dL = self.sP.units.redshiftToLumDist(self.sP.redshift)
        dL_cm = dL * self.sP.units.Mpc_in_cm

        # convert [Lsun/Ang] -> [erg/s/Ang] -> [erg/s/cm^2/Ang], i.e. luminosities into fluxes at this dist
        flux = spec_perAng * (self.sP.units.L_sun / (dL_cm * dL_cm)) / (4.0 * np.pi)
        flux *= 1e17

        # could rebin onto a wavelength grid more like SDSS observations (log wave spaced)
        # https://github.com/moustakas/impy/blob/master/lib/ppxf/ppxf_util.py
        return flux, wave

    def redshiftSpectrum(self, spec, output_wave=None):
        """Attenuate a spectrum for a given redshift.

        If self.sP.redshift > 0, attenuate the spectrum by the luminosity distance and redshift the wavelength.
        If self.sP.redshift == 0, the rest-frame spectrum is returned at an
        assumed distance of 10pc (i.e. absolute magnitudes). Note that the spectrum is sampled
        back onto the original output_wave points, which therefore become observer-frame instead of rest-frame.
        """
        flux = spec.copy()

        if output_wave is None:
            output_wave = self.wave_ang.copy()

        if not self.redshifted or self.sP.redshift == 0.0:
            return flux, output_wave

        # if z>0, redshift the wavelength axis
        wave_redshifted = self.wave_ang * (1.0 + self.sP.redshift)

        # and interpolate the shifted flux to the old, unshifted wavelength points
        # f = interp1d(wave_redshifted, flux, kind='linear', assume_sorted=True, bounds_error=False, fill_value=0.0)
        # flux_origwave = f(output_wave) # about 5x slower than below

        # todo: probably an error here if flux is a multi-star (2D) array, need to generalize
        flux_origwave = np.interp(output_wave, wave_redshifted, flux)

        if 0:
            # DEBUG
            import matplotlib.pyplot as plt

            # plot (A)
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\lambda$ [ Ang ]")
            ax.set_ylabel(r"$f_\lambda$ [ L$_\odot$ / hz ]")
            ax.set_xlim([800, 10000])
            ax.set_ylim([-15, -13])

            n = 10
            shift_back = 1.0  # 1.0 to disable, (1+z) to enable
            ax.plot(self.wave_ang[::n], np.log10(flux[::n]), ls="-", marker="o", label="flux")
            ax.plot(
                output_wave[::n] / shift_back,
                np.log10(flux_origwave[::n] - 7e-15),
                ls="-",
                marker="o",
                label="flux shifted",
            )

            ax.legend()
            fig.savefig("debug_redshiftSpectrum.pdf")
            plt.close(fig)

            # plot (B)
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\lambda$ [ Ang ]")
            ax.set_ylabel(r"$\Delta \lambda$ [ log Ang ]")
            ax.set_xlim([800, 10000])

            dwave = self.wave_ang - np.roll(self.wave_ang, 1)
            dwave[0] = dwave[1]
            dwave_output = output_wave - np.roll(output_wave, 1)
            dwave_output[0] = dwave_output[1]

            ax.plot(self.wave_ang[::n], np.log10(dwave[::n]), ls="-", marker="o", markersize=1.0, label="MILES")
            ax.plot(output_wave[::n], np.log10(dwave_output[::n]), ls="-", marker="o", markersize=1.0, label="OUTPUT")

            ax.legend()
            fig.savefig("debug_redshiftSpectrum_dwave.pdf")
            plt.close(fig)

        # account for (1+z)^2/(1+z) factors from redshifting of photon energies, arrival rates,
        # and bandwidth delta_freq, s.t. flux density per unit bandwidth goes as (1+z)
        # e.g. Peebles 3.87 or MoVdBWhite 10.85 (spec has units of Lsun/Hz, i.e. is f_nu)
        flux = flux_origwave * (1.0 + self.sP.redshift)

        return flux, output_wave

    def mags(self, band, ages_logGyr, metals_log, masses_logMsun):
        """Interpolate table to compute magnitudes [AB absolute] in requested band for input stars."""
        assert band.lower() in self.bands
        assert ages_logGyr.size == metals_log.size == masses_logMsun.size
        assert ages_logGyr.ndim == metals_log.ndim == masses_logMsun.ndim == 1

        # verify units
        assert np.max(metals_log) <= 0.0
        assert np.min(ages_logGyr) >= -10.0
        assert np.max(ages_logGyr) < 2.0
        assert np.max(masses_logMsun) < 12.0  # low-res
        # assert np.min(masses_logMsun) > 1.5 # we have stars as small as 1.5 at least

        # convert input interpolant point into fractional 2D (+bandNum) array indices
        # Note: we are clamping at [0,size-1], so no extrapolation (nearest grid edge value is returned)
        i1 = np.interp(metals_log, self.metals, np.arange(self.metals.size))
        i2 = np.interp(ages_logGyr, self.ages, np.arange(self.ages.size))

        iND = np.vstack((i1, i2))
        locData = self.data["mags_" + band.lower()]

        # do 2D interpolation on this band sub-table at the requested order
        mags = map_coordinates(locData, iND, order=self.order, mode="nearest")

        # account for population mass
        mags -= 2.5 * masses_logMsun

        return mags

    def mags_code_units(self, sP, band, gfm_sftime, gfm_metallicity, masses_code, retFullSize=False):
        """Do unit conversions (and wind particle filtering) on inputs, and return mags() results.

        If retFullSize is True, return same size as inputs with wind set to nan, otherwise filter
        out wind/nan values and compress return size.
        """
        wStars = np.where(gfm_sftime > 0.0)

        if len(wStars[0]) == 0:
            if retFullSize:
                return np.array([np.nan] * gfm_sftime.size)
            return None

        # unit conversions: ages, masses, metallicities
        ages_logGyr = self.sP.units.scalefacToAgeLogGyr(gfm_sftime[wStars])

        masses_logMsun = sP.units.codeMassToLogMsun(masses_code[wStars])

        metals_log = logZeroMin(gfm_metallicity[wStars])
        metals_log[metals_log < -20.0] = -20.0  # truncate at GFM_MIN_METAL

        # magnitudes for 1 solar mass SSPs
        stellarMags = self.mags(band, ages_logGyr, metals_log, masses_logMsun)

        # return an array of the same size of the input, with nan for wind entries
        if retFullSize:
            r = np.zeros(gfm_sftime.size, dtype="float32")
            r.fill(np.nan)
            r[wStars] = stellarMags
            return r

        return stellarMags

    def calcStellarLuminosities(self, sP, band, indRange=None, rotMatrix=None, rotCenter=None):
        """Compute luminosities in the given band, using  snapshot-stored values or on-the-fly calculations.

        Note that wind is returned as NaN luminosity, assuming it is filtered out elsewhere, e.g. in gridBox().

        Return:
          ndarray: linear luminosities in units of [Lsun/Hz].
        """
        assert isinstance(band, str)

        if "snap_" in band:
            # direct load snapshot saved stellar photometrics
            fields = ["sftime", "phot_" + band.split("snap_")[1]]

            stars = sP.snapshotSubset(partType="stars", fields=fields, indRange=indRange)

            wWind = np.where(stars["GFM_StellarFormationTime"] < 0.0)
            stars["GFM_StellarPhotometrics"][wWind] = np.nan

            mags = stars["GFM_StellarPhotometrics"]
        elif "_dustC" in band:
            # view direction dependent dust attenuation calculation on the fly
            from ..cosmo.hydrogen import hydrogenMass

            assert rotMatrix is not None and rotCenter is not None
            assert sP.subhaloInd is not None  # use to load gas based on -subhalo- id (e.g. called from within vis)

            subset = sP.haloOrSubhaloSubset(subhaloID=sP.subhaloInd)
            offset = subset["offsetType"][sP.ptNum("gas")]
            length = subset["lenType"][sP.ptNum("gas")]
            indRange_gas = [offset, offset + length - 1]  # inclusive

            # load gas
            fields = ["pos", "metal", "mass", "dens"]
            if sP.snapHasField("gas", "NeutralHydrogenAbundance"):
                fields.append("NeutralHydrogenAbundance")

            gas = sP.snapshotSubset("gas", fields=fields, indRange=indRange_gas)

            if sP.snapHasField("gas", "GFM_Metals"):
                gas["metals_H"] = sP.snapshotSubset("gas", "metals_H", indRange=indRange_gas)  # H only

            gas["Masses"] = hydrogenMass(gas, sP, totalNeutral=True)
            gas["Cellsize"] = sP.snapshotSubset("gas", "cellsize", indRange=indRange_gas)

            # load stars
            fields = ["initialmass", "sftime", "metallicity", "pos"]
            stars = sP.snapshotSubset(partType="stars", fields=fields, indRange=indRange)

            stars["GFM_StellarFormationTime"] = sP.units.scalefacToAgeLogGyr(stars["GFM_StellarFormationTime"])
            stars["GFM_InitialMass"] = sP.units.codeMassToMsun(stars["GFM_InitialMass"])

            stars["GFM_Metallicity"] = logZeroMin(stars["GFM_Metallicity"])
            stars["GFM_Metallicity"][np.where(stars["GFM_Metallicity"] < -20.0)] = -20.0

            w = np.isfinite(stars["GFM_StellarFormationTime"])  # remove wind
            pos_stars = stars["Coordinates"][w, :]
            ages_logGyr = stars["GFM_StellarFormationTime"][w]
            metals_log = stars["GFM_Metallicity"][w]
            masses_msun = stars["GFM_InitialMass"][w]

            # calculate resolved columns
            pos = gas["Coordinates"]
            hsml = 2.5 * gas["Cellsize"]
            mass_nh = gas["Masses"]
            quant_z = gas["GFM_Metallicity"]

            pxSize = 0.1  # physical kpc, should match to image pxScale for vis

            N_H, Z_g = self.resolved_dust_mapping(
                pos, hsml, mass_nh, quant_z, pos_stars, rotCenter, rotMatrix=rotMatrix, pxSize=pxSize
            )

            # compute stellar magnitudes, reshape back into full PT4 size
            bands = [band.replace("_dustC", "")]

            magsStars = self.dust_tau_model_mags(bands, N_H, Z_g, ages_logGyr, metals_log, masses_msun, ret_indiv=True)
            magsStars = magsStars[bands[0]]

            mags = np.zeros(stars["GFM_StellarFormationTime"].size, dtype=magsStars.dtype)
            mags.fill(np.nan)
            mags[w] = magsStars
        else:
            # load age,Z,mass_ini, use FSPS on the fly
            assert band in self.bands
            fields = ["initialmass", "sftime", "metallicity"]

            stars = sP.snapshotSubset(partType="stars", fields=fields, indRange=indRange)

            sftime = stars.get("GFM_StellarFormationTime", stars.get("StellarFormationTime"))
            metal = stars.get("GFM_Metallicity", stars.get("Metallicity"))
            imass = stars.get("GFM_InitialMass", stars.get("InitialMass"))

            if metal.ndim == 2:
                print("TODO REMOVE MCS ST9 HACK")
                metal = np.squeeze(metal[:, 0])  # bug in io_fields.c

            mags = self.mags_code_units(sP, band, sftime, metal, imass, retFullSize=True)

        # convert to luminosities in [Lsun/Hz]
        lums = np.zeros(mags.size, dtype="float32")
        lums.fill(np.nan)

        ww = np.isfinite(mags)
        lums[ww] = self.sP.units.absMagToLuminosity(mags[ww])

        return lums

    def dust_tau_model_mags(
        self,
        bands,
        N_H,
        Z_g,
        ages_logGyr,
        metals_log,
        masses_msun,
        ret_indiv=False,
        ret_full_spectrum=False,
        output_wave=None,
        rel_vel=None,
    ):
        """Spatially resolved dust attenuation on stellar spectra.

        For a set of stars characterized by their (age,Z,M) values as well as (N_H,Z_g)
        calculated from the resolved gas distribution, do the Model (C) attenuation on the
        full spectra, sum together, and convolve the resulting total L(lambda) with the
        transmission function of multiple bands, returning a dict of magnitudes, one for
        each band. If ret_indiv==True, then the individual magnitudes for every member star are
        instead returned separately. If ret_full_spectrum==True, the full aggregate spectrum,
        summed over all member stars, as a function of wavelength, is instead returned.
        If output_wave is not None, then this output spectrum is interpolated to the requested
        wavelength grid (in Angstroms, should be rest or observed frame depending on self.redshifted).
        If rel_vel is not None, then add LoS peculiar velocity shifts (physical km/s) of each star.
        Note: Will return apparent magnitudes if self.sP.redshift > 0, or absolute magnitudes if
        self.sP.redshift == 0. Will return redshifted (and attenuated) spectra if self.redshifted
        and self.sP.redshift is nonzero, otherwise rest-frame spectra.
        """
        assert N_H.size == Z_g.size == ages_logGyr.size == metals_log.size == masses_msun.size
        assert N_H.ndim == Z_g.ndim == ages_logGyr.ndim == metals_log.ndim == masses_msun.ndim == 1
        if rel_vel is not None:
            assert ret_full_spectrum
        if output_wave is not None:
            assert ret_full_spectrum

        bands = iterable(bands)
        for band in bands:
            assert band in self.bands

        calzetti_case = 3 if "_res3" in self.dustModel else 5

        r = {}

        if "_conv" in self.dustModel:
            # the aggregate spectrum is actually band-independent, get now from JITed helper function
            # band is any member of self.bands, since A_lambda_sol,gamma,f_scattering are all actually
            # band-independent in this case where self.lambda_nm == self.wave
            if ret_indiv or (rel_vel is not None):
                obs_lum = _dust_tau_model_lum_indiv(
                    N_H,
                    Z_g,
                    ages_logGyr,
                    metals_log,
                    masses_msun,
                    self.wave,
                    self.A_lambda_sol[band],
                    self.sP.redshift,
                    self.beta,
                    self.sP.units.Z_solar,
                    self.gamma[band],
                    self.N_H0,
                    self.f_scattering[band],
                    self.metals,
                    self.ages,
                    self.wave_ang,
                    self.spec,
                    calzetti_case,
                )
            else:
                obs_lum = _dust_tau_model_lum(
                    N_H,
                    Z_g,
                    ages_logGyr,
                    metals_log,
                    masses_msun,
                    self.wave,
                    self.A_lambda_sol[band],
                    self.sP.redshift,
                    self.beta,
                    self.sP.units.Z_solar,
                    self.gamma[band],
                    self.N_H0,
                    self.f_scattering[band],
                    self.metals,
                    self.ages,
                    self.wave_ang,
                    self.spec,
                    calzetti_case,
                )

        if ret_full_spectrum:
            # we want an aggregate summed spectrum for all member stars, not all individual spectra
            assert not ret_indiv

            if rel_vel is None:
                obs_lum_1d = obs_lum
            else:
                # support for an array of rel_vel in [km/s], positive or negative, giving the
                # relative velocity of each star particle such that its spectrum should be shifted
                # in wavelength space by the appropriate doppler shift, e.g.
                assert rel_vel.ndim == 1 and rel_vel.size == N_H.size
                assert obs_lum.ndim == 2

                # allocate 1d spectrum return
                obs_lum_1d = np.zeros(self.wave.size, dtype="float32")

                # (lambda_obs/lambda_emit = 1 + peculiar_velocity/c) such that if rel_vel>0 (receeding),
                # lambda_obs > lambda_emit, otherwise if rel_vel<0 then lambda_obs < lambda_edit
                doppler_factor = 1.0 + rel_vel / self.sP.units.c_km_s

                for i in range(rel_vel.size):
                    # create shifted wavelength grid
                    wave_shifted = self.wave * doppler_factor[i]

                    # interpolate fluxes onto original grid
                    f = interp1d(
                        wave_shifted,
                        obs_lum[i, :],
                        kind="linear",
                        assume_sorted=True,
                        bounds_error=False,
                        fill_value=0.0,
                    )

                    obs_lum_1d += f(self.wave)

            # unit conversion into SDSS spectra units, and attenuate by distance to sP.redshift
            spectrum, wave = self.convertSpecToSDSSUnitsAndAttenuate(obs_lum_1d, output_wave=output_wave)

            # early return before any band convolutions (the corresponding wavelengths are self.wave)
            return spectrum

        for band in bands:
            if "_eff" in self.dustModel:
                assert ret_indiv is False
                # the aggregate spectrum is band-dependent, but its calculation is very fast since
                # the lambda_nm is a single value instead of a ~6000 element array
                obs_lum = _dust_tau_model_lum(
                    N_H,
                    Z_g,
                    ages_logGyr,
                    metals_log,
                    masses_msun,
                    self.wave,
                    self.A_lambda_sol[band],
                    self.sP.redshift,
                    self.beta,
                    self.sP.units.Z_solar,
                    self.gamma[band],
                    self.N_H0,
                    self.f_scattering[band],
                    self.metals,
                    self.ages,
                    self.wave_ang,
                    self.spec,
                    calzetti_case,
                )

            # redshift if requested, so that band convolutions are observer-frame instead of rest-frame
            obs_lum, _ = self.redshiftSpectrum(obs_lum)

            # convolve with band (trapezoidal rule)
            if not ret_indiv:
                # return total band magnitude of all star particles combined
                obs_lum_conv = obs_lum * self.trans_normed[band]

                nn = self.wave.size
                band_lum = np.sum(
                    np.abs(self.wave_ang[1 : nn - 1] - self.wave_ang[0 : nn - 2])
                    * (obs_lum_conv[1 : nn - 1] + obs_lum_conv[0 : nn - 2])
                    * 0.5
                )

                assert band_lum > 0.0

                r[band] = self.sP.units.lumToAbsMag(band_lum)
            else:
                # return band magnitude individually for each star particle
                r[band] = np.zeros(obs_lum.shape[0], dtype="float32")

                for i in range(obs_lum.shape[0]):
                    obs_lum_conv = np.squeeze(obs_lum[i, :]) * self.trans_normed[band]

                    nn = self.wave.size
                    band_lum = np.sum(
                        np.abs(self.wave_ang[1 : nn - 1] - self.wave_ang[0 : nn - 2])
                        * (obs_lum_conv[1 : nn - 1] + obs_lum_conv[0 : nn - 2])
                        * 0.5
                    )

                    assert band_lum > 0.0

                    r[band][i] = band_lum
                r[band] = self.sP.units.lumToAbsMag(r[band])

            # optionally convert to apparent magnitude
            # TODO - should also require self.redshifted to be consistent with unresolved dust magnitudes
            if self.sP.redshift > 0:
                r[band] = self.sP.units.absMagToApparent(r[band])

        return r

    def resolved_dust_mapping(
        self, pos_in, hsml, mass_nh, quant_z, pos_stars_in, projCen, projVec=None, rotMatrix=None, pxSize=1.0
    ):
        """Compute line of sight quantities per star for a resolved dust attenuation calculation.

        Gas (pos,hsml,mass_nh,quant_z) and stars (pos_stars) are used for the gridding of the gas
        and the target (star) list. projVec is a [3]-vector, and the particles are rotated about
        projCen [3] such that it aligns with the projection direction. pxSize is in physical kpc.
        """
        assert projCen.size == 3
        assert projVec is not None or rotMatrix is not None

        # rotation
        axes = [0, 1]

        if rotMatrix is None:
            targetVec = np.array([0, 0, 1], dtype="float32")
            rotMatrix = rotationMatrixFromVec(projVec, targetVec)
            assert projVec.size == 3
        else:
            assert rotMatrix.size == 9

        pos, _ = rotateCoordinateArray(self.sP, pos_in, rotMatrix, projCen, shiftBack=True)
        pos_stars, extentStars = rotateCoordinateArray(self.sP, pos_stars_in, rotMatrix, projCen, shiftBack=True)

        # configure projection grid, note we use the symmetric covering extentStars around the
        # original projCen, although we could decrease the grid size by re-centering.
        # extentStars = np.array([ extentStars[1], extentStars[0] ]) # permute for T (no)
        extentStars += pxSize * 2.0

        boxSizeImg = np.array([extentStars[0], extentStars[1], self.sP.boxSize])
        pxSizeCode = self.sP.units.physicalKpcToCodeLength(pxSize)
        nPixels = np.int32(np.ceil(extentStars / pxSizeCode))[0:2]

        nThreads = 8

        if pos_in.shape[0] < 1e3 and pos_stars_in.shape[0] < 1e3:
            nThreads = 4
        if pos_in.shape[0] < 1e2 and pos_stars_in.shape[0] < 1e2:
            nThreads = 1

        # efficiency cut: neutral hydrogen mass in a cell less than 1e-6 times the target cell mass,
        # and in large (h > 2.5 * softening) cells, clip to zero to avoid sph deposition calculation
        massThreshold = 1e-6 * self.sP.targetGasMass
        sizeThreshold = 2.5 * self.sP.gravSoft

        ww = np.where((mass_nh < massThreshold) & (hsml > sizeThreshold))
        mass_nh[ww] = 0.0

        # get (N_H,Z_g) along line of sight to each star
        N_H, Z_g = sphMap(
            pos=pos,
            hsml=hsml,
            mass=mass_nh,
            quant=quant_z,
            axes=axes,
            ndims=3,
            boxSizeSim=self.sP.boxSize,
            boxSizeImg=boxSizeImg,
            boxCen=projCen,
            nPixels=nPixels,
            colDens=False,
            multi=True,
            nThreads=nThreads,
            posTarget=pos_stars,
        )

        # normalize out mass weights of metallicity
        w = np.where(N_H > 0.0)
        Z_g[w] /= N_H[w]

        # convert N_H units from mass to coldens_code to (neutral H atoms)/cm^2
        pixelArea = (boxSizeImg[0] / nPixels[0]) * (boxSizeImg[1] / nPixels[1])

        N_H /= pixelArea
        N_H = self.sP.units.codeColDensToPhys(N_H, cgs=True, numDens=True)

        return N_H, Z_g


def debug_check_redshifting(redshift=0.8):
    """Verify redshifting and apparent vs. absolute magnitudes.

    Check the band magnitudes (from FSPS) and the band magnitudes derived from our convolving our
    spectra with the bandpass filters manually.
    """
    sP = simParams(res=1820, run="tng", redshift=redshift)
    pop = sps(sP, "padova07", "chabrier", "none", redshifted=True)

    x_ind = 5
    y_ind = 42

    metal = np.array((pop.metals[x_ind],))
    age = np.array((pop.ages[y_ind],))

    # load lega-c dr2 wavelength grid
    # with h5py.File(expanduser("~") + "/obs/LEGAC/legac_dr2_spectra_wave.hdf5", "r") as f:
    #    output_wave = f["wavelength"][()]

    for band in ["wfc_acs_f814w"]:  # ['sdss_r','sdss_z','jwst_f444w','wfc_acs_f814w']:
        # select single spectrum, convolve with band, get band luminosity
        obs_lum = pop.spec[x_ind, y_ind, :].copy()
        obs_lum, _ = pop.redshiftSpectrum(obs_lum)  # , output_wave=output_wave)
        obs_lum_conv = obs_lum * pop.trans_normed[band]

        nn = pop.wave.size
        band_lum = np.sum(
            np.abs(pop.wave_ang[1 : nn - 1] - pop.wave_ang[0 : nn - 2])
            * (obs_lum_conv[1 : nn - 1] + obs_lum_conv[0 : nn - 2])
            * 0.5
        )

        band_mag_abs = sP.units.lumToAbsMag(band_lum)
        band_mag_app = sP.units.absMagToApparent(band_mag_abs)

        # compare with band magnitude from FSPS-derived band magnitudes
        mass_logmsun = np.array((np.log10(1.0),))

        check_mag = pop.mags(band, age, metal, mass_logmsun)

        print(band, band_mag_app, check_mag, "diff: ", band_mag_app - check_mag)


def debug_dust_plots():
    """Plot intermediate aspects of the resolved dust calculation."""
    import matplotlib.pyplot as plt

    sP = simParams(res=1820, run="tng", redshift=0.0)

    bands = ["sdss_u", "sdss_g"]  # ,'sdss_r','sdss_i','sdss_z','wfc_acs_f606w']
    iso = "padova07"
    imf = "chabrier"
    dust = "cf00_res_conv"  # _eff, _conv

    marker = "o" if "_eff" in dust else ""
    xm_label = r"FSPS Master $\lambda$ [nm]"
    xf_label = r"Filter $\lambda_{eff}$ [nm]" if "_eff" in dust else xm_label
    master_lambda_range = [0, 2000]

    pop = sps(sP, iso, imf, dust)

    for band in bands:
        # set up a calculation
        N_H = np.array([pop.N_H0 * 0.5])
        Z_g = np.array([sP.units.Z_solar * 0.8])
        age_logGyr = pop.ages[80]
        # age_logGyr = np.array(-3.9) # out of bounds left
        # age_logGyr = np.array(1.6) # out of bounds right
        mass_msun = 1e6
        metal_log = pop.metals[15]
        # print(band,N_H,Z_g,age_logGyr,mass_msun,metal_log)

        # go through a calculation
        tau_a = (
            pop.A_lambda_sol[band]
            * (1 + sP.redshift) ** pop.beta
            * (Z_g / sP.units.Z_solar) ** pop.gamma[band]
            * (N_H / pop.N_H0)
        )
        tau_lambda = tau_a * pop.f_scattering[band]

        atten = np.ones(tau_lambda.size, dtype="float32")
        w = np.where(tau_lambda >= 1e-5)
        atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]

        if pop.lambda_nm[band].size > 1:  # _conv models
            atten = np.interp(pop.wave, pop.lambda_nm[band], atten)

        # bilinear interpolation: stellar population spectrum
        x_ind = np.interp(metal_log, pop.metals, np.arange(pop.metals.size))
        y_ind = np.interp(age_logGyr, pop.ages, np.arange(pop.ages.size))
        # assert x_ind == np.int32(x_ind)
        # assert y_ind == np.int32(y_ind)
        x_ind = np.int32(x_ind)
        y_ind = np.int32(y_ind)

        spectrum_local = np.array(pop.spec[x_ind, y_ind, :])
        spectrum_local = np.clip(spectrum_local, 0.0, np.inf)  # enforce everywhere positive

        # accumulate attenuated contribution of this stellar population
        obs_lum = np.zeros(pop.wave.size, dtype="float32")
        obs_lum += (spectrum_local * mass_msun) * atten

        obs_lum_noatten = np.zeros(pop.wave.size, dtype="float32")
        obs_lum_noatten += spectrum_local * mass_msun

        # convolve with band (trapezoidal rule)
        obs_lum_conv = obs_lum * pop.trans_normed[band]
        obs_lum_noatten *= pop.trans_normed[band]

        nn = pop.wave.size
        band_lum = np.sum(
            np.abs(pop.wave_ang[1 : nn - 1] - pop.wave_ang[0 : nn - 2])
            * (obs_lum_conv[1 : nn - 1] + obs_lum_conv[0 : nn - 2])
            * 0.5
        )
        band_lum_noatten = np.sum(
            np.abs(pop.wave_ang[1 : nn - 1] - pop.wave_ang[0 : nn - 2])
            * (obs_lum_noatten[1 : nn - 1] + obs_lum_noatten[0 : nn - 2])
            * 0.5
        )

        assert band_lum > 0.0
        assert band_lum_noatten > 0.0

        result_mag = pop.sP.units.lumToAbsMag(band_lum)
        result_mag_noatten = pop.sP.units.lumToAbsMag(band_lum_noatten)

        # get magnitude without using our method of convolution
        ages_logGyr = np.array([age_logGyr])
        metals_log = np.array([metal_log])
        masses_msun = np.array([mass_msun])

        mag = pop.mags(band, ages_logGyr, metals_log, np.log10(masses_msun))

        # call our actual function and accelerated function to verify correctness
        NstarsTodo = 1000
        N_H = np.ones(NstarsTodo) * N_H
        Z_g = np.ones(NstarsTodo) * Z_g
        ages_logGyr = np.ones(NstarsTodo) * ages_logGyr
        metals_log = np.ones(NstarsTodo) * metals_log
        masses_msun = np.ones(NstarsTodo) * masses_msun

        mag_f = pop.dust_tau_model_mags(band, N_H, Z_g, ages_logGyr, metals_log, masses_msun)

        print(band, mag_f, result_mag - 2.5 * np.log10(NstarsTodo))  # ,result_mag_noatten,mag)

        # start figure
        fig = plt.figure(figsize=(22, 14))

        ax = fig.add_subplot(3, 4, 1)
        ax.set_xlabel(xf_label)
        ax.set_ylabel(r"$(A_\lambda / A_V)_\odot$")
        ax.plot(pop.lambda_nm[band], pop.A_lambda_sol[band], marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 2)
        ax.set_xlabel(xf_label)
        ax.set_ylabel("f_scattering")
        ax.plot(pop.lambda_nm[band], pop.f_scattering[band], marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 3)
        ax.set_xlabel(xf_label)
        ax.set_ylabel(r"$\gamma$")
        ax.plot(pop.lambda_nm[band], pop.gamma[band], marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 5)
        ax.set_xlabel(r"Filter $\lambda$ [nm]")
        ax.set_ylabel("Filter Transmission")
        ax.plot(pop.trans_lambda[band], pop.trans_val[band])

        ax = fig.add_subplot(3, 4, 6)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel("Filter Interp-Trans")
        ax.plot(pop.wave, pop.trans_normed[band] * pop.wave_ang)

        # plots of spectrum
        ax = fig.add_subplot(3, 4, 4)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel(r"log spec [L$_\odot$/Hz]")
        ax.plot(pop.wave, logZeroNaN(spectrum_local), label="fsps mag=%g" % mag)
        ax.legend()

        ax = fig.add_subplot(3, 4, 8)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xf_label)
        ax.set_ylabel("log spec*mass*atten")
        ax.plot(pop.wave, logZeroNaN(obs_lum), label="mag=%g" % result_mag)
        ax.legend()
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 11)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel("log spec*mass*atten conv")
        ax.plot(pop.wave, logZeroNaN(obs_lum_conv))

        ax = fig.add_subplot(3, 4, 12)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel("log spec*mass convolved")
        ax.plot(pop.wave, logZeroNaN(obs_lum_noatten), label="mag=%g" % result_mag_noatten)
        ax.legend()

        # other plots
        ax = fig.add_subplot(3, 4, 7)
        ax.set_xlabel(xf_label)
        ax.set_ylabel(r"$\tau_a$")
        ax.plot(pop.lambda_nm[band], tau_a, marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 9)
        ax.set_xlabel(xf_label)
        ax.set_ylabel(r"$\tau_\lambda$")
        ax.plot(pop.lambda_nm[band], tau_lambda, marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3, 4, 10)
        ax.set_xlabel(xf_label)
        ax.set_ylabel(r"L$_{obs}(\lambda)$ / L$_i(\lambda)$")
        ax.plot(pop.lambda_nm[band], atten, marker=marker)
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        fig.savefig("debug_%s_%s.pdf" % (dust, band))
        plt.close(fig)


def debug_check_rawspec():
    """Check spectral tables."""
    import matplotlib.pyplot as plt

    from ..plot.config import figsize

    zInd = 5
    ageInd = 30
    redshift = 0.8

    paths = [
        rootPath + "/tables/fsps/mags_padova07_chabrier_cf00_bands-143_z=0.5.hdf5",
        rootPath + "/tables/fsps/mags_padova07_chabrier_cf00_bands-143_z=0.5_em.hdf5",
    ]

    # start plot
    fig = plt.figure(figsize=(figsize[0] * 1.8, figsize[1]))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlabel(r"$\lambda$ [ Angstroms ]")
    ax1.set_ylabel(r"$f_\lambda$ [ L$_\odot$ / hz ]")
    ax1.set_xlim([8000 / (1 + redshift), 9000 / (1 + redshift)])  # rest

    ax2.set_xlabel(r"$\lambda$ [ Angstroms ]")
    ax2.set_ylabel(r"$f_\lambda$ [ L$_\odot$ / hz ]")
    ax2.set_xlim([2000, 10000])
    ax2.set_yscale("log")

    # load
    for path in paths:
        with h5py.File(path, "r") as f:
            spec = f["spec_lsun_hz"][()]
            wave = f["wave_nm"][()] * 10
            ages = f["ages_logGyr"][()]
            metals = f["metals_log"][()]

        label = "Z [%d], age [%d], %s" % (metals[zInd], ages[ageInd], path.split("143_")[1].replace(".hdf5", ""))
        xx = wave  # rest
        yy = spec[zInd, ageInd, :] * 1e14

        w = np.where((xx >= ax1.get_xlim()[0]) & (xx <= ax1.get_xlim()[1]))
        ax1.plot(xx[w], yy[w], ls="-", marker="o", markersize=1.5, label=label)

        w = np.where((xx >= ax2.get_xlim()[0]) & (xx <= ax2.get_xlim()[1]))
        ax2.plot(xx[w], yy[w], ls="-", label=label)

    # finish plot
    ax1.legend()
    fig.savefig("debug_rawspec.pdf")
    plt.close(fig)
