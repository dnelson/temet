"""
Observational data processing, reduction, and analysis (SDSS).
"""

import getpass
import json
import os
import pickle
import time
from datetime import date, datetime

import astropy.io.fits as pyfits
import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from prospect import fitting
from prospect.io import write_results
from prospect.likelihood import lnlike_phot, lnlike_spec, write_log
from prospect.models import sedmodel
from prospect.sources import CSPSpecBasis
from prospect.utils import smoothing
from scipy.interpolate import interp1d

from ..load.data import loadSDSSFits
from ..plot.config import figsize_sm
from ..util.helper import curRepoVersion, pSplitRange
from ..util.simParams import simParams


# config (don't change anything...)
zBin = "z0.0-0.1"
sdssCatName = "sdss_%s.hdf5" % zBin
sdssSpectraFitsCatName = "sdss_mcmc_fits_%s" % zBin
sdssSpecObjIDCatName = "sdss_objid_specobjid_z0.0-0.5.hdf5"
mockSpectraAuxcatName = "Subhalo_SDSSFiberSpectra_%s_p07c_cf00dust_res_conv_z"
spectraFitsAuxcatName = "Subhalo_SDSSFiberSpectraFits_%s-%s_p07c_cf00dust_res_conv_z"
spectralFitQuantities = ["zred", "mass", "logzsol", "tau", "tage", "dust1", "sigma_smooth"]
miles_fwhm_aa = 2.54  # spectral resolution of the MILES stellar library (FWHM/Ang)
sigma_to_fwhm = 2.355
indModulus = 100  # split individual galaxy results into this many subdirectories
percentiles = [16, 50, 84]
minPerMCMCFit = 25.0  # rough estimate, ~= (0.02sec/60) * (nburn.sum()+niter) * nwalkers (Odyssey)


def sdss_decompose_specobjid(id):
    """Convert 64-bit SpecObjID into its parts, returning a dict. DR13 convention."""
    r = {}
    bin = np.binary_repr(id, width=64)
    r["plate"] = int(bin[0:14], 2)  # bits 50-63
    r["fiberid"] = int(bin[14 : 14 + 12], 2)  # bits 38-49
    r["mjd"] = int(bin[14 + 12 : 14 + 12 + 14], 2) + 50000  # bits 24-37 minus 50000
    r["run2d"] = int(bin[14 + 12 + 14 : 14 + 12 + 14 + 14], 2)  # bits 10-23
    return r


def sdss_decompose_objid(id):
    """Convert 64-bit ObjID into its parts, returning a dict. DR13 convention."""
    r = {}
    bin = np.binary_repr(id, width=64)
    r["rerun"] = int(bin[5 : 5 + 11], 2)  # bits 48-58
    r["run"] = int(bin[5 + 11 : 5 + 11 + 16], 2)  # bits 32-47
    r["camcol"] = int(bin[5 + 11 + 16 : 5 + 11 + 16 + 3], 2)  # bits 29-31
    r["field"] = int(bin[5 + 11 + 16 + 3 + 1 : 5 + 11 + 16 + 3 + 1 + 12], 2)  # bits 16-27
    r["id"] = int(bin[5 + 11 + 16 + 3 + 1 + 12 :], 2)  # bits 0-15

    return r


def loadSDSSSpectrum(ind, fits=False):
    """Remotely acquire (via http) a single SDSS galaxy spectrum.

    Index corresponds to the sdss_z0.0-0.1.hdf5 datafile (z<0.1 targets).
    """
    basePath = os.path.expanduser("~") + "/obs/SDSS/"
    savePath = basePath + "spectra/"

    if not os.path.isdir(savePath):
        os.mkdir(savePath)

    # get target objid
    with h5py.File(basePath + sdssCatName, "r") as f:
        objid = f["objid"][ind]
        logMass = f["logMass_gran1"][ind]
        redshift = f["redshift"][ind]

    # get matching specobjid
    with h5py.File(basePath + sdssSpecObjIDCatName, "r") as f:
        objids = f["objid"][()]
        w = np.where(objids == objid)[0]
        specobjid = f["specobjid"][w[0]]

    print("[%d] objid: %d  specobjid: %d logMass: %.2f" % (ind, objid, specobjid, logMass))

    # calculate mjd, fiberid, plateid
    p = sdss_decompose_specobjid(specobjid)
    r = {"ind": ind, "objid": objid, "specobjid": specobjid, "logMass": logMass, "redshift": redshift}

    # construct url
    fmt = "fits" if fits else "csv"

    url_base = "https://dr13.sdss.org/optical/spectrum/view/data/format=%s/spec=lite?" % fmt
    url = url_base + "mjd=%d&fiberid=%d&plateid=%d" % (p["mjd"], p["fiberid"], p["plate"])

    # check for existence, download if does not already exist locally
    savePath = savePath + "%d/" % p["plate"]
    if not os.path.isdir(savePath):
        os.mkdir(savePath)

    saveFilename = savePath + "spec-%04d-%d-%04d.%s" % (p["plate"], p["mjd"], p["fiberid"], fmt)

    if not os.path.isfile(saveFilename):
        # acquire
        print(" " + url)
        try:
            req = requests.get(url, headers={}, timeout=10)
        except requests.exceptions.Timeout:
            print(" WARNING! Response timeout, skipping!")
            return None

        if req.status_code != 200:
            print(" WARNING! Response code = %d, skipping!" % req.status_code)
            return None

        # save (fits only for now)
        if fits:
            if "content-disposition" not in req.headers:
                return None
            # filename = req.headers['content-disposition'].split("filename=")[1]

            with open(saveFilename, "wb") as f:
                f.write(req.content)

    # load
    if not fits:
        # parse csv
        rows = req.text.split("\n")
        cols = ["wavelength", "flux", "bestfit", "skyflux"]
        for col in cols:
            r[col] = np.zeros(len(rows) - 2, dtype="float32")

        # first line is header, last line is empty
        for i, row in enumerate(rows[1:-1]):
            # wavelength = angstroms
            # flux = coadded calibrated flux [10^-17 erg/s/cm2/angstrom]
            # skyflux = subtracted sky flux [same units]
            # bestfit = pipeline best model fit used for classification and redshift (?)
            # ivar = inverse variance (one over simga-squared), is 0.0 for bad pixels that should be ignored
            r["wavelength"][i], r["flux"][i], r["bestfit"][i], r["skyflux"][i] = row.split(",")
    else:
        # open fits
        with pyfits.open(saveFilename) as hdus:
            spec = np.array(hdus[1].data)

        # return in same format/units as csv, except we now also have ivar and wdisp
        r["wavelength"] = 10.0 ** spec["loglam"]
        r["flux"] = spec["flux"]
        r["ivar"] = spec["ivar"]
        r["wdisp"] = spec["wdisp"]
        r["skyflux"] = spec["sky"]
        r["bestfit"] = spec["model"]

    return r


def loadSimulatedSpectrum(sP, ind, withVel=False, addRealism=False):
    """Load a single mock SDSS fiber spectrum fpr a simulated galaxy.

    If addRealism == True, use a [random] real SDSS fiber spectrum to
    convolve the mock with a realistic instrumental resolution, and add realistic noise. If
    withVel == True, use the mock spectra which account for peculiar stellar velocity.
    """
    basePath = os.path.expanduser("~") + "/obs/SDSS/"

    velStr = "Vel" if withVel else "NoVel"
    acName = mockSpectraAuxcatName % velStr

    # load mock spectrum
    spec = sP.auxCat(acName, indRange=[ind, ind + 1])

    assert spec["subhaloIDs"].size == 1
    subhaloID = spec["subhaloIDs"][0]

    # construct return (identical format as loadSDSSSpectrum)
    stellarMass = sP.groupCatSingle(subhaloID=subhaloID)["SubhaloMassInRadType"][sP.ptNum("stars")]
    logMass = sP.units.codeMassToLogMsun(stellarMass)

    r = {"ind": ind, "objid": None, "specobjid": None, "logMass": logMass, "redshift": sP.redshift}

    r["wavelength"] = spec[acName + "_attrs"]["wavelength"]
    r["flux"] = spec[acName]
    r["skyflux"] = None
    r["bestfit"] = None

    if addRealism:
        # determine a 'random' index into the SDSS z<0.1 catalog for this ind
        with h5py.File(basePath + sdssCatName, "r") as f:
            sdss_cat_size = f["objid"].size

        # load SDSS spectrum and de-redshift (simulated spectrum should be in rest-frame, if we are
        # fitting an actual z>0 redshifted simulated spectrum, think through the below again)
        np.random.seed(ind + 424242)
        sdss_cat_ind = np.random.choice(np.arange(sdss_cat_size))
        sdss_spec = loadSDSSSpectrum(sdss_cat_ind, fits=True)

        while sdss_spec["flux"].sum() == 0.0 or sdss_spec["wdisp"].max() == 0.0:
            # e.g. unplugged fiber or other catastropic failure, get another
            sdss_cat_ind += 1
            sdss_spec = loadSDSSSpectrum(sdss_cat_ind % sdss_cat_size, fits=True)

        if sdss_spec is None:
            return None

        sdss_spec["wavelength"] /= 1 + sdss_spec["redshift"]
        sdss_spec["flux"] /= 1 + sdss_spec["redshift"]

        # copy SDSS variance and dispersion estimates, interpolating onto the simulated wavelength grid
        r["ivar"] = interp1d(
            sdss_spec["wavelength"],
            sdss_spec["ivar"],
            kind="linear",
            bounds_error=False,
            fill_value=(sdss_spec["ivar"][0], sdss_spec["ivar"][-1]),
        )(r["wavelength"])
        r["wdisp"] = interp1d(
            sdss_spec["wavelength"],
            sdss_spec["wdisp"],
            kind="linear",
            bounds_error=False,
            fill_value=(sdss_spec["wdisp"][0], sdss_spec["wdisp"][-1]),
        )(r["wavelength"])

        # calculate the sdss dispersion in Angstroms, as a function of wavelength
        sigma_aa = sdss_spec["wdisp"] * np.gradient(sdss_spec["wavelength"])  # Ang
        # sigma_v = sP.units.c_km_s * (sigma_aa / sdss_spec['wavelength']) # km/s

        # interpolate the dispersion to the simulated wavelength grid (constant extrapolation)
        f = interp1d(
            sdss_spec["wavelength"], sigma_aa, kind="linear", bounds_error=False, fill_value=(sigma_aa[0], sigma_aa[-1])
        )
        smooth_res = f(r["wavelength"])

        w = np.where(smooth_res <= 0.0)  # avoid any catastrophes
        smooth_res[w] = np.median(smooth_res)

        # Replace by the quadrature difference with respect to MILES (TBD)
        # sigma_aa_miles = miles_fwhm_aa / sigma_to_fwhm
        # smooth_res = np.sqrt(np.clip(smooth_res**2 - sigma_aa_miles**2, 0, np.inf))

        # convolve the mock spectrum using sdss wdisp (decrease the resolution to the instrumental res)
        r["flux"] = smoothing.smoothspec(r["wavelength"], r["flux"], resolution=smooth_res, smoothtype="lsf")

        # clip observed sdss spectrum, no negative values
        w = np.where(sdss_spec["flux"] > 0.0)
        sdss_spec["flux"] = np.clip(sdss_spec["flux"], sdss_spec["flux"][w].min() / 10, np.inf)

        # clip observed ivar, remove zeros (these are masked out anyways in the fitting)
        w = np.where(sdss_spec["ivar"] == 0.0)
        sdss_spec["ivar"][w] = np.min(sdss_spec["ivar"][np.where(sdss_spec["ivar"] > 0.0)])

        # add Gaussian noise to the actual flux with the variance taken from ivar
        sdss_stddev_frac = 1.0 / np.sqrt(sdss_spec["ivar"]) / sdss_spec["flux"]

        f = interp1d(
            sdss_spec["wavelength"],
            sdss_stddev_frac,
            kind="linear",
            bounds_error=False,
            fill_value=(sdss_stddev_frac[0], sdss_stddev_frac[-1]),
        )
        interp_stddev_frac = f(r["wavelength"])
        interp_stddev_frac = np.clip(interp_stddev_frac, 0.001, 1.0)  # do SNR=1 at worst

        rnd_frac = np.random.normal(loc=0.0, scale=interp_stddev_frac)

        r["flux"] += rnd_frac * r["flux"]  # add random component
    else:
        # we fit a noise-less spectrum at its original (mock stellar library) resolution, need
        # to provide a wdisp estimate for MILES (this is constant in wavelength, while disp is
        # in units of pixels=dlog(lambda)
        r["wdisp"] = (miles_fwhm_aa / sigma_to_fwhm) / np.gradient(r["wavelength"])

        # give a non-zero (fake) stddev of 5% (roughly characteristic) to avoid over fitting
        frac_stddev = 0.05
        r["ivar"] = 1.0 / (frac_stddev * r["flux"]) ** 2.0

    return r


def getLSFSmoothing(spec, sP):
    """Get line spread function (LSF) as a function of wavelength for a given spectrum file.

    This method takes a spec file and returns the quadrature difference between the
    instrumental dispersion and the MILES dispersion, in km/s, as a function of wavelength.
    """
    # Get the SDSS instrumental resolution for this plate/mjd/fiber
    wave = spec["wavelength"]
    dlam = np.gradient(wave)
    sigma_aa = spec["wdisp"] * dlam
    sigma_v = sP.units.c_km_s * (sigma_aa / wave)

    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave, sigma_v = wave[good], sigma_v[good]

    # Get the miles velocity resolution function
    sigma_v_miles = sP.units.c_km_s * miles_fwhm_aa / sigma_to_fwhm / wave

    # Get the quadrature difference (zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))

    # Restrict to regions where MILES is used
    good = (wave > 3525.0) & (wave < 7500)

    # Get the quadrature difference between the instrumental and MILES resolution
    wave, delta_v = wave[good], dsv[good]

    if 0:
        # Write the file (out of date, now set in memory)
        lname = os.path.join(os.environ["SPS_HOME"], "data", "lsf.dat")
        with open(lname, "w") as out:
            for w, vel in zip(wave, delta_v):
                out.write(f"{w:4.2f}   {vel:4.2f}\n")
        out.close()

        print(" WROTE [%s] careful of overlap." % lname)

    return wave, delta_v


def load_obs(ind, run_params, doSim=None):
    """Construct observational object with a SDSS spectrum, ready for fitting.

    If doSim is not None, then instead of an actual SDSS spectrum, load and process a mock spectrum instead, in
    which case doSim should be a dict having keys 'withVel' and 'addRealism'.
    """
    tryCount = 0
    spec = None

    while spec is None and tryCount < 10:
        # make a few attempts since we are unreliably fetching data over HTTP
        if doSim is not None:
            spec = loadSimulatedSpectrum(doSim["sP"], ind=ind, withVel=doSim["withVel"], addRealism=doSim["addRealism"])

            for k in doSim.keys():
                # append any additional mock details to save into the output
                run_params[k] = doSim[k]
        else:
            spec = loadSDSSSpectrum(ind=ind, fits=True)

        tryCount += 1
        time.sleep(10.0)  # wait ten seconds

    if spec is None:
        return None  # failed

    # convert [10^-17 erg/cm^2/s/Ang] -> [10^-23 erg/cm^2/s/Hz] since flux_nu = (lambda^2/c) * flux_lambda
    # http://coolwiki.ipac.caltech.edu/index.php/Units, https://en.wikipedia.org/wiki/AB_magnitude
    # 3.34e4 = 1/(c * 1e3 * Jy_mks) = 1/(c * 1e3 * 1e-26) = 1/(3e18 ang/s * 1e3 * 1e-26)
    fac = 1e-17 * 3.34e4 * (spec["wavelength"]) ** 2.0 / 3631.0
    flux_maggies = spec["flux"] * fac
    # flux_Jy = flux_maggies * 3631.0
    # flux_nMgy = flux_maggies * 1e9

    # define observation
    obs = {}
    obs["wavelength"] = spec["wavelength"]  # vacuum Angstroms
    obs["spectrum"] = flux_maggies  # units of maggies

    if "ivar" in spec:
        with np.errstate(all="ignore"):
            obs["unc"] = 1.0 / np.sqrt(spec["ivar"]) * fac
    else:
        print(" Warning: No actual variance.")
        obs["unc"] = spec["flux"] * 0.05 * fac

    obs["maggies"] = None  # no broadband filter magnitudes
    obs["maggies_unc"] = None  # no associated uncertanties
    obs["filters"] = None  # no associated filters
    obs["phot_mask"] = None  # optional, no associated masks

    # lsf: such that we convolve the theoretical spectra with the instrument resolution (wave-dependent)
    sP = doSim["sP"] if doSim is not None else simParams(res=1820, run="tng")  # just for units

    obs["lsf_wave"], obs["lsf_delta_v"] = getLSFSmoothing(spec, sP)

    # deredshift (fit in rest-frame below)
    obs["wavelength"] /= 1 + spec["redshift"]
    obs["spectrum"] /= 1 + spec["redshift"]  # assuming spectrum is now f_nu

    # mask: bad variance or negative flux
    obs["mask"] = (spec["ivar"] != 0.0) & (spec["flux"] > 0.0)

    # mask: wavelength range
    wavelength_mask = (obs["wavelength"] > run_params["wlo"]) & (obs["wavelength"] < run_params["whi"])

    # mask: emission lines
    lines = """3715.0  3735.0  0.0  *[OII]      3726.0
               3720.0  3742.0  0.0  *[OII]      3728.8
               4065.0  4135.0  0.0  *Hdelta     4101.7
               4315.0  4370.0  0.0  *Hgamma     4340.5
               4840.0  4885.0  0.0  *Hbeta      4861.3
               4935.0  5040.0  0.0  *[OIII]     4958.9
               6285.0  6315.0  0.0  *[OI]       6300.3
               6510.0  6625.0  0.0  *Halpha+NII 6562.8
               6685.0  6770.0  0.0  *[SII]      6716.4
               7125.0  7142.0  0.0  [ArIII]     7135.8
               7315.0  7329.0  0.0  [OII]       7319.5
               7320.0  7335.0  0.0  [OII]       7330.2
               5185.0  5215.0  0.0  NI5199      5199.0
               5872.0  5916.0  0.0  NaD         5890.0"""

    emissionline_mask = np.zeros(len(obs["wavelength"]), dtype=bool)

    for line in lines.split("\n"):
        wave_min = float(line.split()[0])
        wave_max = float(line.split()[1])
        emissionline_mask |= (obs["wavelength"] > wave_min) & (obs["wavelength"] < wave_max)

    emissionline_mask = ~emissionline_mask  # such that 0=bad

    obs["mask"] &= wavelength_mask
    obs["mask"] &= emissionline_mask

    # auxiliary information
    for k in ["ind", "objid", "specobjid", "logMass", "redshift"]:
        obs[k] = spec[k]

    return obs


def _dust2_from_dust1(dust1=None, **extras):
    """Coupling function between dust1 and dust2 parameters."""
    return dust1 * 0.5


def load_model_params(redshift=None):
    """Return the set of model parameters.

    Includes details on which are fixed and which are free, the associated priors, and initial guesses.
    """
    from prospect.models import priors

    model_params = []

    if redshift is None:
        # no known redshift, fit freely
        model_params.append(
            {
                "name": "zred",
                "N": 1,
                "isfree": True,
                "init": 0.1,
                "units": "",
                "prior_function": priors.tophat,
                "prior_args": {"mini": 0.0, "maxi": 0.1},
            }
        )
    else:
        # redshift: fixed to sdss spectroscopic value (assumed to be 10pc if z=0, e.g. for absolute mags)
        # note we deredshift observed spectra, so this is only a residual redshift (and always left free)
        model_params.append(
            {
                "name": "zred",
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "init_disp": 1e-4,
                "disp_floor": 1e-4,
                "units": "residual redshift",
                "prior_function": priors.tophat,
                "prior_args": {"mini": -1e-3, "maxi": 1e-3},
            }
        )

        # set the lumdist parameter to get the spectral units (and thus masses) correct. note that, by
        # having both `lumdist` and `zred` we decouple the redshift from the distance (necessary since
        # zred represents only the residual redshift in the fitting)
        sP = simParams(res=1820, run="tng")  # for cosmology
        lumDist = sP.units.redshiftToLumDist(redshift)

        model_params.append({"name": "lumdist", "N": 1, "isfree": False, "init": lumDist, "units": "Mpc"})

    # --- SFH ---
    # FSPS parameter.  sfh=1 is a exponentially declining tau SFH, sfh=4 is a delayed-tau SFH
    model_params.append({"name": "sfh", "N": 1, "isfree": False, "init": 4, "units": "type"})

    # Normalization of the SFH.  If the ``mass_units`` parameter is not supplied,
    # this will be in surviving stellar mass.  Otherwise it is in the total stellar
    # mass formed.
    model_params.append(
        {
            "name": "mass",
            "N": 1,
            "isfree": True,
            "init": 4e10,
            "init_disp": 1e10,
            "units": r"M_\odot",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 1e6, "maxi": 1e12},
        }
    )

    model_params.append({"name": "mass_units", "N": 1, "isfree": False, "init": "mformed"})

    # Since we have zcontinuous=1 above, the metallicity is controlled by the
    # ``logzsol`` parameter.
    model_params.append(
        {
            "name": "logzsol",
            "N": 1,
            "isfree": True,
            "init": -0.4,
            "init_disp": 0.2,
            "units": r"$\log (Z/Z_\odot)$",
            "prior_function": priors.tophat,
            "prior_args": {"mini": -1.5, "maxi": 0.7},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "tau",
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "init_disp": 0.1,
            "units": "Gyr",
            "prior_function": priors.logarithmic,
            "prior_args": {"mini": 0.1, "maxi": 100},
        }
    )

    # FSPS parameter (could change max to == t_age at this redshift)
    model_params.append(
        {
            "name": "tage",
            "N": 1,
            "isfree": True,
            "init": 8.0,
            "init_disp": 2.0,
            "units": "Gyr",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.101, "maxi": 14.0},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "sfstart",
            "N": 1,
            "isfree": False,
            "init": 0.0,
            "units": "Gyr",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.1, "maxi": 14.0},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "tburst",
            "N": 1,
            "isfree": False,
            "init": 0.0,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.0, "maxi": 1.3},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "fburst",
            "N": 1,
            "isfree": False,
            "init": 0.0,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.0, "maxi": 0.5},
        }
    )

    # --- Dust ---------
    # FSPS parameter (0=bc00 plaw, 1=CCM, 4=Kreik and Conroy)
    model_params.append({"name": "dust_type", "N": 1, "isfree": False, "init": 0, "units": "index"})

    # FSPS parameter
    model_params.append(
        {
            "name": "dust1",
            "N": 1,
            "isfree": True,
            "init": 0.7,
            "reinit": True,
            "units": "",
            "init_disp": 0.2,
            "disp_floor": 0.1,
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.0, "maxi": 2.0},
        }
    )

    # FSPS parameter (could couple to dust1, e.g. some constant fraction, through depends_on)
    model_params.append(
        {
            "name": "dust2",
            "N": 1,
            "isfree": False,
            "init": 0.0,
            #'reinit': True,
            "depends_on": _dust2_from_dust1,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0.0, "maxi": 2.0},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "dust_index",
            "N": 1,
            "isfree": False,
            "init": -0.7,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": -1.5, "maxi": -0.5},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "dust1_index",
            "N": 1,
            "isfree": False,
            "init": -0.7,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": -1.5, "maxi": -0.5},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "dust_tesc",
            "N": 1,
            "isfree": False,
            "init": 7.0,
            "units": "log(Gyr)",
            "prior_function_name": None,
            "prior_args": None,
        }
    )

    # FSPS parameter
    model_params.append({"name": "add_dust_emission", "N": 1, "isfree": False, "init": True, "units": "index"})

    # FSPS parameter
    model_params.append(
        {"name": "duste_umin", "N": 1, "isfree": False, "init": 1.0, "units": "MMP83 local MW intensity"}
    )

    # --- Stellar Pops ------------
    # FSPS parameter
    model_params.append({"name": "tpagb_norm_type", "N": 1, "isfree": False, "init": 2, "units": "index"})

    # FSPS parameter
    model_params.append({"name": "add_agb_dust_model", "N": 1, "isfree": False, "init": True, "units": "index"})

    # FSPS parameter
    model_params.append({"name": "agb_dust", "N": 1, "isfree": False, "init": 1, "units": "index"})

    # --- Nebular Emission ------

    # Here is a really simple function that takes a **dict argument, picks out the
    # `logzsol` key, and returns the value.  This way, we can have gas_logz find
    # the value of logzsol and use it, if we uncomment the 'depends_on' line in the
    # `gas_logz` parameter definition.
    #
    # One can use this kind of thing to transform parameters as well (like making
    # them linear instead of log, or divide everything by 10, or whatever.) You can
    # have one parameter depend on several others (or vice versa).  Just remember
    # that a parameter with `depends_on` must always be fixed.
    # def stellar_logzsol(logzsol=0.0, **extras):
    #    return logzsol

    # FSPS parameter
    model_params.append({"name": "add_neb_emission", "N": 1, "isfree": False, "init": False})

    # FSPS parameter
    model_params.append(
        {
            "name": "gas_logz",
            "N": 1,
            "isfree": False,
            "init": 0.0,
            "units": r"log Z/Z_\odot",
            #                        'depends_on': stellar_logzsol,
            "prior_function": priors.tophat,
            "prior_args": {"mini": -2.0, "maxi": 0.5},
        }
    )

    # FSPS parameter
    model_params.append(
        {
            "name": "gas_logu",
            "N": 1,
            "isfree": False,
            "init": -2.0,
            "units": "",
            "prior_function": priors.tophat,
            "prior_args": {"mini": -4, "maxi": -1},
        }
    )

    # --- Kinematics --------
    model_params.append(
        {"name": "smoothtype", "N": 1, "isfree": False, "init": "vel", "units": "do velocity smoothing"}
    )

    model_params.append(
        {
            "name": "sigma_smooth",
            "N": 1,
            "isfree": True,
            "init": 200.0,
            "init_disp": 100.0,
            "disp_floor": 50.0,
            "units": "km/s",
            "prior_function": priors.logarithmic,
            "prior_args": {"mini": 50, "maxi": 400},
        }
    )

    model_params.append({"name": "fftsmooth", "N": 1, "isfree": False, "init": True, "units": "use fft for smoothing"})

    # --- Photometric Calibration ---------
    # model_params.append({'name': 'phot_jitter', 'N': 1,
    #                        'isfree': False,
    #                        'init': 0.0,
    #                        'units': 'mags',
    #                        'prior_function':priors.tophat,
    #                        'prior_args': {'mini':0.0, 'maxi':0.2}})

    # --- Spectroscopic Calibration --------
    # polyeqn = 'ln(f_tru/f_obs)_j=\sum_{i=1}^N poly_coeffs_{i-1} * ((lambda_j - lambda_min)/lambda_range)^i'
    # Set the order of the polynomial.  The highest order will be \lambda^npoly
    npoly = 2
    # for setting min/max on the polynomial coefficients.
    polymax = 0.1 / (np.arange(npoly) + 1)

    model_params.append(
        {
            "name": "poly_coeffs",
            "N": npoly,
            "isfree": False,
            "init": np.zeros(npoly),
            "init_disp": polymax / 5.0,
            "units": "",  # polyeqn,
            "prior_function": priors.tophat,
            "prior_args": {"mini": 0 - polymax, "maxi": polymax},
        }
    )

    model_params.append(
        {
            "name": "cal_type",
            "N": 1,
            "isfree": False,
            "init": "exp_poly",
            "units": "switch for whether to use exponential of polynomial for calibration",
        }
    )

    return model_params


def _indivSavePath(ind, doSim=None):
    """Return a save path, for mock if doSim==None, otherwise for SDSS."""
    if doSim is not None:
        sP = doSim["sP"]
        basePath = sP.derivPath + "/spectral_fits/snap_%03d/%d/" % (sP.snap, ind % indModulus)
        fileBase = basePath + "chains_v%d_r%d_%d" % (doSim["withVel"], doSim["addRealism"], ind)
    else:
        basePath = os.path.expanduser("~") + "/obs/SDSS/mcmc_fits_%s/%d/" % (zBin, ind % indModulus)
        fileBase = basePath + "chains_%d" % ind

    fileName = fileBase + ".hdf5"
    return fileName


def fitSingleSpectrum(ind, doSim=None):
    """Run MCMC fit against a particular SDSS spectrum.

    ind gives the index of the SDSS z<0.1 catalog (if doSim is None),
    otherwise the index of the sP SDSSFiberSpectra auxCat (if doSim is not None)
    in which case the identical fitting procedure is run against a mock instead of observed spectrum.
    """
    # test: NGC3937 z=0.02221 objid=1237667916491325445 ind=280692
    # plate=2515  mjd=54180 fiber=377 specobjid=2831741964808906752
    # test: NGC5227 z=0.01745 objid=1237651735230545966 ind=49455
    # plate=528   mjd=52022 fiber=137 specobjid=594512843009714176

    # configuration
    run_params = {
        "verbose": False,
        "debug": False,
        # Fitter parameters
        "nwalkers": 128,  # should be N*(nproc-1) where N is an even integer, Nproc=MPI tasks
        "nburn": [32, 32, 64, 64, 128, 128],
        "niter": 128,  # note: total number of retained samples = niter*nwalkers
        #'do_powell': False,
        #'ftol':0.5e-5, 'maxfev':5000,
        "initial_disp": 0.1,
        # Data manipulation parameters
        "logify_spectrum": False,
        "normalize_spectrum": False,
        "wlo": 3750.0,
        "whi": 7000.0,  # spectral libraries are too low resolution at higher wavelengths
        # SPS parameters
        "zcontinuous": 1,  # 1=continuous metallicity, 0=discretized zmet integers only
    }

    # already exists?
    outFileName = _indivSavePath(ind, doSim=doSim)

    if os.path.isfile(outFileName):
        print(" SKIP: [%s] already exists." % outFileName)
        return

    # load observational spectrum
    obs = load_obs(ind, run_params, doSim=doSim)

    if obs is None:
        print(" FAILED TO LOAD_OBS! SKIP [%d]." % outFileName)
        return

    # model
    # default: 7 free parameters: z_red, mass, logzsol, tau, tage, dust1, sigma_smooth
    model_params = load_model_params(redshift=obs["redshift"])
    model = sedmodel.SedModel(model_params)

    # SPS Model
    sps = CSPSpecBasis(zcontinuous=run_params["zcontinuous"], compute_vega_mags=False)

    assert sps.csp.libraries[1] == "miles"
    sps.csp.params["smooth_lsf"] = True
    sps.csp.set_lsf(obs["lsf_wave"], obs["lsf_delta_v"])

    # setup
    initial_theta = model.rectify_theta(model.initial_theta)

    hf = h5py.File(outFileName, "w")
    write_results.write_h5_header(hf, run_params, model)
    write_results.write_obs_to_h5(hf, obs)

    # no initial Powell guess
    postkwargs = {}
    pool = None  # MPI, disabled

    powell_guesses = None
    pdur = 0.0
    initial_center = initial_theta.copy()
    initial_prob = None

    # mcmc sample
    tstart = time.time()
    out = fitting.run_emcee_sampler(
        lnprobfn,
        initial_center,
        model,
        postargs=[model, obs, sps],
        postkwargs=postkwargs,
        initial_prob=initial_prob,
        pool=pool,
        hdf5=hf,
        **run_params,
    )
    esampler, burn_p0, burn_prob0 = out

    edur = time.time() - tstart
    print("done sampling in %.1f sec" % edur)

    # write results to hdf5
    write_results.write_hdf5(
        hf,
        run_params,
        model,
        obs,
        esampler,
        powell_guesses,
        toptimize=pdur,
        tsample=edur,
        sampling_initial_center=initial_center,
        post_burnin_center=burn_p0,
        post_burnin_prob=burn_prob0,
    )


def combineAndSaveSpectralFits(nSpec, objs=None, doSim=None):
    """Combine and save all of the individual MCMC hdf5 result files.

    For either the SDSS spectra sample or for the mock sample of a given sP.
    Same format as an auxCat. For mock samples, this file can then be loaded through load.auxCat().
    Note save size: condensed subhaloIDs only.
    """
    nFound = 0
    indRange = [0, nSpec - 1]

    if doSim is None:
        # SDSS: we save in the same format as the mocks, except in ~/obs/SDSS/
        field = sdssSpectraFitsCatName
        outFileName = os.path.expanduser("~") + "/obs/SDSS/%s.hdf5" % field
    else:
        # mocks: we save in the identical format as a normal auxCat
        sP = doSim["sP"]
        velStr = "Vel" if doSim["withVel"] else "NoVel"
        realStr = "Realism" if doSim["addRealism"] else "NoRealism"
        field = spectraFitsAuxcatName % (velStr, realStr)
        outFileName = sP.derivPath + "auxCat/%s_%03d.hdf5" % (field, sP.snap)

    assert not os.path.isfile(outFileName)

    # get metadata from first fit result
    attrs = {}

    with h5py.File(_indivSavePath(0, doSim=doSim), "r") as f:
        nFreeParams = f["sampling"]["initial_theta"].size

        for k in ["theta_labels"]:
            attrs[k] = f["sampling"].attrs[k].encode("ascii")
        for k in ["prospector_version", "model_params", "run_params"]:
            attrs[k] = f.attrs[k].encode("ascii")

    # allocate, condensed subhalo size, number of free parameters * 3 (for 2 percentiles each)
    r = np.zeros((nSpec, nFreeParams, 3), dtype="float32")
    r.fill(np.nan)

    print(" Concatenating into shape: ", r.shape)

    # start search
    for index in np.arange(indRange[0], indRange[1]):
        if index % np.max([1, int(nSpec / 100.0)]) == 0 and index <= nSpec:
            print("   %4.1f%%" % (float(index + 1) * 100.0 / nSpec))

        fileName = _indivSavePath(index, doSim=doSim)
        if not os.path.isfile(fileName):
            continue

        # load
        f = h5py.File(fileName, "r")

        if "sampling" not in f or "chain" not in f["sampling"]:
            print("CORRUPT, SKIP: %s" % fileName)
            f.close()
            continue

        chain = f["sampling"]["chain"][()]
        f.close()

        assert chain.ndim == 3 and chain.shape[2] == nFreeParams

        # flatten chain into a linear list of samples, calculate median and percentiles, save
        samples = chain.reshape((-1, nFreeParams))
        percs = np.percentile(samples, percentiles, axis=0)

        r[index, :, :] = percs.T  # (7,3) shape
        nFound += 1

    # save result
    print("Total # galaxy spectra: %d, found [%d] with results." % (nSpec, nFound))

    # save new dataset (or overwrite existing)
    with h5py.File(outFileName, "w") as f:
        f.create_dataset(field, data=r)

        if objs is not None:
            f.create_dataset(objs["name"], data=objs["ids"])

        for attrName, attrValue in attrs.items():
            f[field].attrs[attrName] = attrValue

        # save metadata and any additional descriptors as attributes
        f[field].attrs["CreatedOn"] = date.today().strftime("%d %b %Y")
        f[field].attrs["CreatedRev"] = curRepoVersion()
        f[field].attrs["CreatedBy"] = getpass.getuser()

    print(" Saved new [%s]." % outFileName)


def fitSDSSSpectra(pSplit):
    """Fit a pSplit work divided segment of the entire z<0.1 SDSS selection.

    Results are saved individually, one file per galaxy, in ~/obs/SDSS/mcmc_fits_{ZBIN}/{DIR}/.
    """
    f1 = os.path.expanduser("~") + "/obs/SDSS/sdss_%s.hdf5" % zBin

    basePath = os.path.expanduser("~") + "/obs/SDSS/mcmc_fits_%s/" % zBin
    if not os.path.isdir(basePath):
        os.mkdir(basePath)

    # get global list of all object IDs of interest
    with h5py.File(f1, "r") as f:
        objIDs = f["objid"][()]
        nObjIDs = f["objid"].size
        objIDRange = [0, nObjIDs - 1]

    if pSplit is None:
        # combine now (semi-loose: all the files that exist, need not be all)
        objs = {"name": "objid", "ids": objIDs}

        combineAndSaveSpectralFits(nObjIDs, objs=objs, doSim=None)
    else:
        # calculate now: divide range, decide indRange local to this task
        indRange = pSplitRange(objIDRange, pSplit[1], pSplit[0])
        hoursEst = minPerMCMCFit * (indRange[1] - indRange[0] + 1) / 60.0

        print(
            "Total # galaxies: %d, processing [%d] now, range [%d - %d]..."
            % (nObjIDs, indRange[1] - indRange[0] + 1, indRange[0], indRange[1])
        )
        print("Estimated time for this task: %.1f hours (%.1f days)..." % (hoursEst, hoursEst / 24))

        # process in a serial loop
        for index in np.arange(indRange[0], indRange[1]):
            savePath = basePath + str(index % indModulus) + "/"
            if not os.path.isdir(savePath):
                os.mkdir(savePath)

            fitSingleSpectrum(ind=index, doSim=None)


def fitMockSpectra(sP, pSplit, withVel=True, addRealism=True):
    """Fit a pSplit work divided segment of the mock SDSS fiber spectra for this snapshot.

    Results are saved individually in {SIM}/data.files/spectral_fits/snap_NNN/DIR/.
    """
    doSim = {"sP": sP, "withVel": withVel, "addRealism": addRealism}

    basePath = sP.derivPath + "/spectral_fits/"
    if not os.path.isdir(basePath):
        os.mkdir(basePath)
    basePath += "snap_%03d/" % sP.snap
    if not os.path.isdir(basePath):
        os.mkdir(basePath)

    # get global list of the number of spectra we have
    velStr = "Vel" if withVel else "NoVel"
    acName = mockSpectraAuxcatName % velStr
    spec = sP.auxCat(acName, onlyMeta=True)
    nSpec = spec["subhaloIDs"].size

    if pSplit is None:
        # combine now (semi-loose: all the files that exist, need not be all)
        doSim = {"sP": sP, "withVel": withVel, "addRealism": addRealism}
        objs = {"name": "subhaloIDs", "ids": spec["subhaloIDs"]}

        combineAndSaveSpectralFits(nSpec, objs=objs, doSim=doSim)
    else:
        # calculate now: divide range, decide indRange local to this task
        indRange = pSplitRange([0, nSpec - 1], pSplit[1], pSplit[0])
        hoursEst = minPerMCMCFit * (indRange[1] - indRange[0] + 1) / 60.0

        print(
            "Total # galaxy spectra: %d, processing [%d] now, range [%d - %d]..."
            % (nSpec, indRange[1] - indRange[0] + 1, indRange[0], indRange[1])
        )
        print("Estimated time for this task: %.1f hours (%.1f days)..." % (hoursEst, hoursEst / 24))

        # process in a serial loop
        for index in np.arange(indRange[0], indRange[1]):
            savePath = basePath + str(index % indModulus) + "/"
            if not os.path.isdir(savePath):
                os.mkdir(savePath)

            fitSingleSpectrum(ind=index, doSim=doSim)


def lnprobfn(theta, model=None, obs=None, sps=None, verbose=False):
    """Given a parameter vector theta, return the ln of the posterior."""
    lnp_prior = model.prior_product(theta)

    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    t1 = time.time()
    try:
        mu, phot, x = model.mean_model(theta, obs, sps=sps)
    except ValueError:
        return -np.infty
    d1 = time.time() - t1

    # Noise modeling
    # if spec_noise is not None:
    #    spec_noise.update(**model.params)
    # if phot_noise is not None:
    #    phot_noise.update(**model.params)
    vectors = {
        "spec": mu,
        "unc": obs["unc"],
        "sed": model._spec,
        "cal": model._speccal,
        "phot": phot,
        "maggies_unc": obs["maggies_unc"],
    }

    # Calculate likelihoods
    t2 = time.time()
    lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=None, **vectors)  # spec_noise=spec_noise
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=None, **vectors)  # phot_noise=phot_noise
    d2 = time.time() - t2

    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    return lnp_prior + lnp_phot + lnp_spec


def scida_sdss_spectra():
    """Demonstration of using scida to load and plot sdss-dr17.spectra.hdf5."""
    import dask.array as da
    import matplotlib.pyplot as plt
    import numpy as np
    import scida

    path = "/virgotng/mpia/obs/SDSS/"
    filename = "sdss-dr17-spectra.hdf5"

    ds = scida.load(path + filename)

    # class, z, and wave are small arrays, so we just load them into memory as numpy arrays
    cls = np.array(ds["class"])
    z = np.array(ds["z"])
    wave = np.array(ds["wave"])

    classes = {0: "GALAXY", 2: "QSO", 1: "STAR"}

    def average_bins(array, bin_size=1000):
        """Average bins in the first axis by a given bin_size."""
        a, b = array.shape
        remainder = a % bin_size
        # If there is a remainder, pad the array
        if remainder != 0:
            pad_size = bin_size - remainder
            padded_array = np.pad(array, ((0, pad_size), (0, 0)), mode="constant", constant_values=0)
        else:
            padded_array = array
        # Now, reshape and compute the mean along the new axis
        reshaped_array = padded_array.reshape(-1, bin_size, b)
        averaged_array = da.nanmean(reshaped_array, axis=1)

        return averaged_array

    def get_im(cl):
        # sub-select spectra in this class
        w = np.where(cls == cl)[0]

        # order by redshift
        z_loc = np.array(z)[w]
        inds = np.argsort(z_loc)  # ... and argsort is not properly supported by dask anyway
        flux = ds["flux"].rechunk((-1, 100))  # important to rechunk in first dimension where we use indices
        im2d = flux[w[inds], :]

        # compensate for distance
        # im2d *= (1+z_loc.reshape(z_loc.size,1))**4

        # reduce size by averaging 10000 spectra each
        bin_size = 2000
        im2d = average_bins(im2d, bin_size)
        im2d = im2d.compute()

        return im2d, [z_loc.min(), z_loc.max()]

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(14, 8))

    for i, (cl, label) in enumerate(classes.items()):
        # get image for plotting
        im2d, z_minmax = get_im(cl)

        # plot
        extent = [wave.min(), wave.max(), z_minmax[0], z_minmax[1]]
        im = axes[i].imshow(im2d, extent=extent, origin="lower", aspect="auto")

        axes[i].set_xlabel("Wavelength [%s]" % print(format(ds["wave"].units, "~L")))
        axes[i].set_ylabel("(Approximate) Redshift")
        axes[i].set_title(label)

    fig.colorbar(im, label="Flux [%s]" % print(format(ds["flux"].units, "~L")))  # $(1+z)^4$ *

    plt.savefig("sdss-dr17-spectra.png", dpi=150)


def plotSingleResult(ind, sps=None, doSim=None):
    """Examine and visualize the results of a single MCMC fit.

    Load the results of a single MCMC fit, print the answer, render a corner plot of the joint
    PDFs of all the parameters, and show the original spectrum as well as some ending model spectra.
    """
    from prospect.models import sedmodel
    from prospect.sources import CSPSpecBasis

    # mapping from sampling labels to pretty labels
    new_labels = {
        "zred": r"10$^4$ z$_{\rm res}$",
        "mass": r"M$_\star$ [10$^{10}$ M$_{\rm sun}$]",
        "logzsol": r"log(Z/Z$_{\rm sun}$)",
        "tau": r"$\tau_{\rm SFH}$ [Gyr]",
        "tage": r"$t_{\rm age}$ [Gyr]",
        "dust1": r"$\tau_{1}$",
        "sigma_smooth": r"$\sigma_{\rm disp}$ [km/s]",
    }

    # load a mock spectrum fit if doSim is input, otherwise load a SDSS spectrum fit
    fileName = _indivSavePath(ind, doSim=doSim)

    if not os.path.isfile(fileName):
        return

    print(fileName)

    # load chains from hdf5
    with h5py.File(fileName, "r") as f:
        chain = f["sampling"]["chain"][()]
        wave = f["obs"]["wavelength"][()]
        spec = f["obs"]["spectrum"][()]

        # variable-length null-terminated ASCII string pickles
        model_params = pickle.loads(f.attrs["model_params"])
        run_params = f.attrs["run_params"]
        if run_params[0] == "(":
            run_params = pickle.loads(run_params)  # pickled
        else:
            run_params = json.loads(run_params)  # json encoded
        # rstate = pickle.loads(f["sampling"].attrs["rstate"])

        # string encoded list
        theta_labels = f["sampling"].attrs["theta_labels"]
        theta_labels = theta_labels.replace("[", "").replace("]", "").replace('"', "")
        theta_labels = theta_labels.split(", ")

        # dt_sec = float(f["sampling"].attrs["sampling_duration"])

    # replace labels
    theta_labels_plot = [new_labels[l] for l in theta_labels]

    # fix any model parameter dependency functions
    for mp in model_params:
        if "depends_on" in mp and type(mp["depends_on"]) is type([]):
            import importlib

            module = importlib.import_module(mp["depends_on"][1])
            mp["depends_on"] = getattr(module, mp["depends_on"][0])

    # flatten chain into a linear list of samples
    ndim = chain.shape[2]
    samples = chain.reshape((-1, ndim))

    # print median as the answer, as well as standard percentiles
    percs = np.percentile(samples, percentiles, axis=0)
    for i, label in enumerate(theta_labels):
        print(label, percs[:, i])
    print("Number of (samples,free_params): ", samples.shape)

    # plot prep
    if doSim is None:
        orig_spec = loadSDSSSpectrum(ind, fits=True)
        saveStr = "sdss"
        label1 = "SDSS #%d z=%.3f" % (ind, orig_spec["redshift"])
        label2 = "SpecObjID " + str(orig_spec["specobjid"])
        label3 = "ObjID " + str(orig_spec["objid"])
    else:
        velStr = "Vel" if doSim["withVel"] else "NoVel"
        acName = mockSpectraAuxcatName % velStr
        subhaloID = doSim["sP"].auxCat(acName, indRange=[ind, ind + 1])["subhaloIDs"][0]

        sub = doSim["sP"].groupCatSingle(subhaloID=subhaloID)
        subMassStars = sub["SubhaloMassInRadType"][doSim["sP"].ptNum("stars")]
        logMassStars = doSim["sP"].units.codeMassToLogMsun(subMassStars)
        logMassTot = doSim["sP"].units.codeMassToLogMsun(sub["SubhaloMass"])
        logMetal = doSim["sP"].units.metallicityInSolar(sub["SubhaloStarMetallicity"], log=True)

        saveStr = "%s-%s_v%dr%d" % (doSim["sP"].simName, doSim["sP"].snap, doSim["withVel"], doSim["addRealism"])
        label1 = "%s #%d z=%.1f" % (doSim["sP"].simName, ind, doSim["sP"].redshift)
        label2 = "SubhaloIndex " + str(subhaloID)
        label3 = "MT = %.2f MS = %.2f Z = %.2f" % (logMassTot, logMassStars, logMetal)

        # for z=0 simulated spectra, for display, redshift to z=0.1
        if doSim["sP"].redshift == 0.0:
            target_z = 0.1
            dL_old_cm = doSim["sP"].units.redshiftToLumDist(0.0) * doSim["sP"].units.Mpc_in_cm
            dL_new_cm = doSim["sP"].units.redshiftToLumDist(target_z) * doSim["sP"].units.Mpc_in_cm
            spec *= (dL_old_cm) ** 2 / (dL_new_cm) ** 2

            wave_redshifted = wave * (1.0 + target_z)
            spec = np.interp1d(
                wave_redshifted, spec, kind="linear", assume_sorted=True, bounds_error=False, fill_value=np.nan
            )(wave)
            spec *= 1.0 + target_z

        # any negative values (due to noise) which were masked, set to nan for plot
        w = np.where(spec <= 0.0)
        spec[w] = np.nan

    # corner plot
    samples_plot = samples.copy()
    # adjust z_res and Mass for sig figs
    samples_plot[:, 0] *= 1e4  # residual redshift
    samples_plot[:, 1] /= 1e10  # mass

    fig = corner.corner(
        samples_plot,
        labels=theta_labels_plot,
        quantiles=[0.1, 0.5, 0.9],
        show_titles=True,
        title_kwargs={"fontsize": 13},
    )

    # reconstruct model
    model = sedmodel.SedModel(model_params)

    if sps is None:
        sps = CSPSpecBasis(zcontinuous=True, compute_vega_mags=False)

    # start spectrum figure
    left = 0.51
    bottom = 0.64
    width = 1 - left - 0.04
    height = 1 - bottom - 0.02

    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlabel(r"$\lambda$ [Angstroms]")
    ax.set_ylabel(r"$F_\lambda$ [$\mu$Mgy]")
    ax.set_ylim([5e-3, 2e-1])
    ax.set_xlim([3500, 9500])
    ax.set_yscale("log")

    lw = 1.0
    nModelsPlot = 3

    # add a number of models
    rng = np.random.default_rng(4242424)
    random_indices = rng.choice(np.arange(samples.shape[0]), size=nModelsPlot, replace=False)

    for i in random_indices:
        obs = {"wavelength": wave, "filters": [], "logify_spectrum": False}
        spec_model, phot, mfrac = model.mean_model(samples[i, :], obs=obs, sps=sps)

        ax.plot(wave, spec_model * 1e6, "-", lw=lw, alpha=0.3)

    # plot input spectrum
    ax.plot(wave, spec * 1e6, "-", lw=lw, color="black", label="Input Spectrum")

    # mark maximum wavelength used for fitting
    ax.plot([run_params["whi"], run_params["whi"]], [1e-1, 1.2e0], ":", color="black", alpha=0.5)

    # make inset zoomed-in
    ax_inset = inset_axes(ax, width="40%", height="40%", loc=4, borderpad=2.4)
    ax_inset.tick_params(labelsize=13)
    ax_inset.set_ylabel(r"$F_\lambda$ [$\mu$Mgy]")
    ax_inset.set_yscale("log")
    ax_inset.set_ylim([5e-2, 8e-2])
    ax_inset.set_xlim([5850, 5950])

    # plot models and then input spectrum
    for i in random_indices:
        obs = {"wavelength": wave, "filters": [], "logify_spectrum": False}
        spec_model, phot, mfrac = model.mean_model(samples[i, :], obs=obs, sps=sps)
        ax_inset.plot(wave, spec_model * 1e6, "-", lw=lw * 1.5, alpha=0.3)

    ax_inset.plot(wave, spec * 1e6, "-", lw=lw * 1.5, color="black")

    ax.annotate(
        label1,
        xy=(0.61, 0.57),
        xycoords="figure fraction",
        fontsize=24,
        color="#cccccc",
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.annotate(
        label2,
        xy=(0.61, 0.54),
        xycoords="figure fraction",
        fontsize=24,
        color="#cccccc",
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.annotate(
        label3,
        xy=(0.61, 0.51),
        xycoords="figure fraction",
        fontsize=24,
        color="#cccccc",
        horizontalalignment="left",
        verticalalignment="top",
    )

    # finish figure
    ax.legend()
    fig.savefig("fig_mcmcCornerModel_%s_%d.pdf" % (saveStr, ind))
    plt.close(fig)


def plotMultiSpectra(doSim, simInds, sdssInds):
    """Plot a few real, and a few mock, spectra."""
    run_params = {"wlo": 3750, "whi": 7000}

    fig = plt.figure(figsize=figsize_sm)
    ax = fig.add_subplot(111)

    ax.set_xlabel(r"$\lambda$ [Angstroms]")
    ax.set_ylabel(r"$F_\lambda$ [$\mu$Mgy]")
    ax.set_ylim([1e-3, 3e-1])
    ax.set_xlim([3700, 6700])
    ax.set_yscale("log")

    for sdssInd in sdssInds:
        spec = load_obs(sdssInd, run_params, doSim=None)

        # mask bad pixels
        # w = np.where(spec['mask'] == 0)
        # spec['spectrum'][w] = np.nan

        ax.plot(spec["wavelength"], spec["spectrum"] * 1e6, "-", label="SDSS %d" % spec["objid"])

    for ind in simInds:
        spec = load_obs(ind, run_params, doSim=doSim)

        ax.plot(spec["wavelength"], spec["spectrum"] * 1e6, "-", label="%s %d" % (doSim["sP"].simName, ind))

    # finish figure
    ax.legend(loc="upper left")
    fig.savefig("fig_plotMultiSpectra.pdf")
    plt.close(fig)


def sdssFitsVsMstar():
    """Plot the SDSS fit parameters vs Mstar."""
    # config
    sdss = loadSDSSFits()

    quants = {
        "dust1": [r"$\tau_{1,dust}$", [0.0, 3.0]],
        "mass": [r"Fiber M$_\star$ [ log M$_{\rm sun}$ ]", [8.0, 11.5]],
        "logzsol": [r"Stellar Metallicity [ Z / Z$_{\rm sun}$ ]", [-2.0, 1.0]],
        "tage": [r"Stellar Age [ Gyr ]", [0.0, 14.0]],
        "tau": [r"$\tau_{SFH}$ [ Gyr ]", [0.0, 5.0]],
        "sigma_smooth": [r"$\sigma_{\rm smooth}$ [ km/s ]", [0, 450]],
        "zred": [r"z$_{\rm residual}$", [-2e-3, 2e-3]],
    }

    # plot setup
    xlabel = r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ]"
    xlim = [8.0, 12.0]

    pdf = PdfPages("sdss_fits_z01_%s.pdf" % (datetime.now().strftime("%d-%m-%Y")))

    for quantName, p in quants.items():
        quantLabel, quantLim = p

        fig = plt.figure(figsize=figsize_sm)
        ax = fig.add_subplot(111)

        ax.set_xlim(xlim)
        ax.set_ylim(quantLim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(quantLabel)

        (l4,) = ax.plot(sdss[quantName]["xm"], sdss[quantName]["ym"], "-", color="green", lw=2.0, alpha=0.7)
        ax.fill_between(
            sdss[quantName]["xm"],
            sdss[quantName]["pm"][0, :],
            sdss[quantName]["pm"][4, :],
            color="green",
            interpolate=True,
            alpha=0.05,
        )
        ax.fill_between(
            sdss[quantName]["xm"],
            sdss[quantName]["pm"][1, :],
            sdss[quantName]["pm"][3, :],
            color="green",
            interpolate=True,
            alpha=0.1,
        )
        ax.fill_between(
            sdss[quantName]["xm"],
            sdss[quantName]["pm"][5, :],
            sdss[quantName]["pm"][6, :],
            color="green",
            interpolate=True,
            alpha=0.2,
        )

        pdf.savefig()
        plt.close(fig)

    pdf.close()
