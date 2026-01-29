"""
Observational data processing, reduction, and analysis (eROSITA).

 * aperture photometry on GAMA sources with apetool

  - multiple eef and arcsec apertures, could interpolate to fixed pkpc apertures
  - verification against Liu catalog values
  - random positions: assess residual background even after 'background subtraction'
  - could re-do anderson+ L_x vs M* plot

 * image stacking on GAMA sources

  - pretty clear detection down to M*=10.6
  - could do: Lx vs M* split on galaxy properties (other than SFR)
  - could do: radial profiles
  - could do: angular anisotropy signal (truong) - not clearly there
"""

from os.path import expanduser, isfile

import astropy.io.fits as pyfits
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass, gaussian_filter, rotate, shift
from scipy.stats import binned_statistic_2d

from ..plot.util import loadColorTable
from ..util.helper import dist_theta_grid, logZeroNaN, running_median
from ..util.match import match
from ..util.rotation import rotationMatrixFromAngle
from ..util.simParams import simParams


# config
basePath = expanduser("~") + "/obs/eFEDS/"
basePathOut = basePath + "0.5-2.0kev/"
basePathGAMA = expanduser("~") + "/obs/GAMA/"

clusterCatName = "eFEDS_clusters_V3.fits"

px_scale = 4.0  # arcsec per pixel in maps we generate (rebin == 80) (equals CDELT2)


def convert_liu_table():
    """Load Liu+2021 cluster table (currently v3) and convert to HDF5."""
    path = basePath + clusterCatName

    # define/allocate
    data = {}

    # open fits catalog file
    with pyfits.open(path) as f:
        # get metadata for array sizes
        #f0 = f[0].data  # no idea what it is

        for key in f[1].data.names:
            data[key] = np.array(f[1].data[key])

    data["ID"] = [bytes(name.strip(), encoding="ascii") for name in data["ID"]]

    with h5py.File(path.replace(".fits", ".hdf5"), "w") as f:
        for key in data:
            f[key] = data[key]

    print("Done.")


def make_apetool_inputcat(source="liu", param_eef=0.9):
    """Make a mllist input fits file for apetool, to do aperture photometry.

    Args:
      source (str): if 'liu' use the Liu+ cluster sample, if 'gama' the GAMA catalog, if 'random' then
        generate random sampling points in the survey footprint.
      param_eef (float): parameter specifying the aperture. If less than 1.0, interpreted as
        in units of EEF (enclosed energy fraction), otherwise interpreted as in units of arcsec.

    Returns:
      None
    """
    # extraction radius [EEF = enclosed energy fraction units]
    # param_eef = 0.9 #0.7

    # source removal radius [EEF units]
    param_rr = 0.95  # 0.8

    # load
    data = {}

    if source == "liu":
        with h5py.File(basePath + clusterCatName.replace(".fits", ".hdf5"), "r") as f:
            for key in ["DEC", "RA", "ID", "ID_SRC"]:
                data[key] = f[key][()]

    if source == "gama":
        data = gama_overlap()
        print("After catalog cross-matching, have [%d] sources to run." % data["ra"].size)

    if source == "random":
        # uniform random in [RA,DEC] rectangle bounds
        n = 100000
        rng = np.random.default_rng(424242)

        file = basePathOut + "events_merged_image.fits"
        _, _, extent = _load_map(file)

        # extent = [ra_max, ra_min, dec_min, dec_max]
        data["RA"] = rng.uniform(low=extent[1], high=extent[0], size=n)
        data["DEC"] = rng.uniform(low=extent[2], high=extent[3], size=n)

    # used from the catalog file, whereas command-line input "eefextract=0.6" is unused
    eef = np.zeros(data["RA"].size, dtype="float64") + param_eef
    rr = np.zeros(data["RA"].size, dtype="float64") + param_rr

    # write fits
    col_ra = pyfits.Column(name="RA", format="D", array=data["RA"])
    col_dec = pyfits.Column(name="DEC", format="D", array=data["DEC"])
    col_re = pyfits.Column(name="RE", format="D", array=eef)
    col_rr = pyfits.Column(name="RR", format="D", array=rr)

    cols = pyfits.ColDefs([col_ra, col_dec, col_re, col_rr])
    hdu = pyfits.BinTableHDU.from_columns(cols, name="Joined")
    rad_str = "eef%d" % (param_eef * 100) if param_eef < 1.0 else "arcsec%d" % param_eef
    hdu.writeto(basePath + "ape_inputcat_%s_%s.fits" % (source, rad_str), overwrite=True)
    print("Written.")


def _ecf():
    """Energy conversion factor.

    See Brunner+2021 Table D.1 [cm^2/erg] and http://hea-www.harvard.edu/HRC/calib/ecf/ecf.html.
    """
    #ecf_02_23_kev = 1.074e12
    ecf_05_2_kev = 1.185e12
    #ecf_23_5_kev = 1.147e11
    #ecf_5_8_kev = 2.776e10

    ecf = ecf_05_2_kev
    print("Using 0.5-2.0 keV ECF.")

    return ecf


def parse_apetool_output_cat(source="liu", param_eef=0.9):
    """Parse output of apetool photometry and plot results, e.g. L_X vs M*.

    Args:
      source (str): if 'liu' use the Liu+ cluster sample, if 'gama' the GAMA catalog, if 'random' then
        generate random sampling points in the survey footprint.
      param_eef (float): parameter specifying the aperture. If less than 1.0, interpreted as
        in units of EEF (enclosed energy fraction), otherwise interpreted as in units of arcsec.

    Returns:
      None
    """
    perc_levels = [16, 50, 84]

    rng = np.random.default_rng(424242)

    rad_str = "eef%d" % (param_eef * 100) if param_eef < 1.0 else "arcsec%d" % param_eef
    file = basePathOut + "mllist_ape_out_%s_%s.fits" % (source, rad_str)

    data = {}

    with pyfits.open(file) as f:
        for key in f[1].data.names:
            data[key] = f[1].data[key]

    # APE_CTS: counts at a position (source and background) extracted from the input images in certain energy bands
    # APE_BKG: background counts extracted from the ERMLDET source maps within certain energy bands.
    #   There is an option to remove nearby sources from the SRCMAP when extracting background counts.
    # APE_EXP: mean exposure time at the used-defined input positions
    # APE_EEF: Encircled Energy Fraction used to define the extraction radius
    # APE_RADIUS: Extraction radius in PIXELS (iff EEF!) or ARCSEC (if radius input for apetool > 1.0 i.e. in arcsec)
    # APE_POIS: Poisson probability that the extracted counts (source + background) are a fluctuation of the background.

    # background-subtracted source count rate
    data["countrate"] = (data["APE_CTS"] - data["APE_BKG"]) / data["APE_EXP"]  # counts/s
    print("Total counts: %g, total count rate: %g" % (data["APE_CTS"].sum(), data["countrate"].sum()))

    # flux [count/sec] / [cm^2/erg] -> [erg/s/cm^2]
    ecf = _ecf()
    data["flux"] = data["countrate"] / ecf

    # some sort of signal/noise ratio
    data["SN"] = np.zeros(data["APE_CTS"].size, dtype="float32")
    w = np.where(data["APE_BKG"] > 0)
    data["SN"][w] = data["APE_CTS"][w] / data["APE_BKG"][w]

    # source redshifts
    if source == "liu":
        with h5py.File(basePath + clusterCatName.replace(".fits", ".hdf5"), "r") as f:
            # Soft band (0.5-2keV) fluxes within 300 or 500 kpc (-1 indicates no reliable measurement)
            F_300kpc = f["F_300kpc"][()]  # 9px at z=0.2, 5.5px at z=0.4
            #F_500kpc = f["F_500kpc"][()]  # 15px at z=0.2, 9px at z=0.4
            z = f["z"][()]

        # which measurement to compare to?
        cat_flux = F_300kpc

    if source == "gama":
        # config
        xlim = [9.5, 12.0]
        ylim = [38, 45]

        # load
        gama = gama_overlap()
        z = gama["z"]

        # negative lums are noise fluctuations consistent with a detection threshold
        # find smallest flux with a reasonable S/N
        ww = np.where(data["SN"] >= 1.0)
        thresh1 = np.log10(np.nanpercentile(data["flux"][ww], 1))

        # find smallest flux/lum with reasonable POIS, call this a (single object) detection threshold
        ww = np.where((data["APE_POIS"] < 0.1) & (data["flux"] > 0))
        thresh2 = np.log10(np.nanmin(data["flux"][ww]))

        thresh3 = -14.0  # canonical 0.5-2.0kev value (Merloni+12, Pillepich+12)

        print("Single object detection thresholds: %.2f %.2f %.2f erg/s/arcsec^2" % (thresh1, thresh2, thresh3))

        # TODO: use 'random' to compute an average background flux, and subtract it off (for a given param_eef)

        # luminosities
        assert data["flux"].size == gama["z"].size

        sP = simParams(run="tng100-1")
        data["lum"] = sP.units.fluxToLuminosity(data["flux"], redshift=gama["z"])

        # (A) compute medians/errors in mstar bins
        binSize = 0.5
        nBins = int(np.ceil((xlim[1] - xlim[0]) / binSize))

        lum_percsA = np.zeros((nBins, 3), dtype="float32")
        lum_percsB = np.zeros((nBins, 3), dtype="float32")
        lum_mean = np.zeros(nBins, dtype="float32")

        lum_percsA.fill(np.nan)
        lum_percsB.fill(np.nan)
        lum_mean.fill(np.nan)

        for i in range(nBins):
            xmin = xlim[0] + binSize * (i + 0)
            xmax = xlim[0] + binSize * (i + 1)
            w = np.where((gama["mstar"] >= xmin) & (gama["mstar"] < xmax))
            print(f"[{i}] {xmin} < M* < {xmax}: {len(w[0]):5d} galaxies, total counts: {data['APE_CTS'][w].sum()}")

            if len(w[0]):
                # (A) include all zeros/negatives
                lum_percsA[i, :] = logZeroNaN(np.percentile(data["lum"][w], perc_levels))

                # (B) exclude any negative/zero luminosities
                lum_percsB[i, :] = np.nanpercentile(logZeroNaN(data["lum"][w]), perc_levels)

                # (C) redo "background-subtracted source count rate" equation, but stack first (in flux),
                # i.e. compute: total(counts) - total(bkg) for the stack, rather than per object
                lumdist_Mpc = sP.units.redshiftToLumDist(gama["z"][w])

                # L = F * 4 * pi * dL^2
                # F = CR * ECF = (C - B) / t * ECF = C*ECF/t - B*ECF/t
                # L = (C*ECF/t - B*ECF/t) * 4 * pi * dL^2
                # L_i = 4 * pi * ECF * (C_i - B_i) / t_i * dL_i^2
                # L_tot = 4 * pi * ECF * ( Sum[ C_i/t_i*dL_i^2 ] - Sum[ B_i/t_i*dL_i^2 ] )
                sum1 = np.sum(data["APE_CTS"][w] * lumdist_Mpc**2 / data["APE_EXP"][w])
                sum2 = np.sum(data["APE_BKG"][w] * lumdist_Mpc**2 / data["APE_EXP"][w])
                L_tot = 4 * np.pi * ecf * (sum1 - sum2) * sP.units.Mpc_in_cm
                lum_mean[i] = np.log10(L_tot / len(w[0]))  # erg/s

        xmid = xlim[0] + np.arange(nBins) * binSize + binSize / 2

        # nan lum_percs correspond are e.g. bottom percentile == 0
        w = np.where(~np.isfinite(lum_percsA))
        lum_percsA[w] = ylim[0]  # scale to bottom of plot

    if source == "random":
        # average flux?
        percs = np.percentile(data["flux"], perc_levels)
        print(f"All random samples, flux percs: {logZeroNaN(percs)} [erg/s/cm^2]")

        for poi_limit in [0.05, 0.1, 0.2, np.inf]:
            ww = np.where((data["APE_POIS"] < poi_limit) & (data["flux"] > 0))
            percs = np.percentile(data["flux"][ww], perc_levels)
            print(f"Random samples, nonzero and with POI < {poi_limit}, flux percs: {logZeroNaN(percs)} [erg/s/cm^2]")

        z = np.zeros(data["countrate"].size, dtype="float32")  # unused

    # some statistics
    if param_eef < 1.0:
        percs = np.percentile(data["APE_RADIUS"], perc_levels)  # pixels
        percs_arcsec = percs * px_scale
    else:
        percs_arcsec = np.percentile(data["APE_RADIUS"], perc_levels)  # ARCSEC!
        percs = percs_arcsec / px_scale

    print(f"Extraction EEF: {data['APE_EEF'].mean()}")
    print(f"Extraction radii [px] percentiles: {percs}")
    print(f"Extraction radii [arcsec] percentiles: {percs_arcsec}")
    print(f"Mean source redshift: {z.mean() = }")

    sP = simParams(run="tng100-1", redshift=z.mean())
    percs_kpc = sP.units.arcsecToAngSizeKpcAtRedshift(percs_arcsec)
    print(f"Extraction radii [kpc] at z: {percs_kpc}")
    print("Total number of sources we tried to measure: ", data["flux"].size)

    rad_str = "eef%d" % (param_eef * 100) if param_eef < 1.0 else "arcsec%d" % param_eef

    # compare with actual Liu+ catalog
    if source == "liu":
        # plot
        fig, ax = plt.subplots()
        ax.set_xlabel("catalog_flux [erg s$^{-1}$ cm$^{-2}$]")
        ax.set_ylabel("my_apetool_flux [erg s$^{-1}$ cm$^{-2}$]")
        ax.set_xlim([-14.5, -12.0])
        ax.set_ylim([-16.2, -12.0])

        # only include valid points from the original catalog
        ww = np.where((cat_flux > 0) & (data["flux"] > 0))
        ax.plot(np.log10(cat_flux[ww]), np.log10(data["flux"][ww]), "o", label="in both")
        print("Number of good measurements in both: ", len(ww[0]))

        # any points we miss, but existed in original?
        ww = np.where((cat_flux > 0) & (data["flux"] <= 0))
        yy = rng.uniform(low=-12.05, high=-12.2, size=len(ww[0]))
        ax.plot(np.log10(cat_flux[ww]), yy, "o", label="in original only")
        print("Number we missed, but exist in original cat: ", len(ww[0]))

        # any points we have measured, but were missing in original?
        ww = np.where((cat_flux < 0) & (data["flux"] > 0) & (data["APE_POIS"] < 0.1))
        xx = rng.uniform(low=-12.05, high=-12.2, size=len(ww[0]))
        ax.plot(xx, np.log10(data["flux"][ww]), "o", label="in myape only")
        print("Number we have, but missed in original cat: ", len(ww[0]))

        ax.plot([-14, -12], [-14, -12], "--", color="black", label="1-to-1")

        ax.legend(loc="best")
        fig.savefig("flux_comparison_liu_%s.pdf" % rad_str)
        plt.close(fig)

    # compare with Anderson+ ROSAT stacking
    if source == "gama":
        # some statistics
        ww = np.where((data["lum"]) < 0)
        print("Number of negative lums: ", len(ww[0]))

        ww = np.where((data["lum"] < 0) & (data["APE_POIS"] < 0.1))
        print("Number of negative lums with small POIS: ", len(ww[0]))

        # plot
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ]")
        ax.set_ylabel(r"L$_{\rm X}$ [ log erg/s ]")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # scatter plot all, color by POIS
        cmap = loadColorTable("magma", fracSubset=[0.1, 0.9])

        xx = gama["mstar"]
        yy = logZeroNaN(data["lum"])  # zeros -> NaN! excluded in percentiles below TODO
        cc = logZeroNaN(data["APE_POIS"])

        cMinMax = [-5.0, 0.0]

        sc = ax.scatter(xx, yy, c=cc, s=2.0, alpha=1.0, vmin=cMinMax[0], vmax=cMinMax[1], cmap=cmap, zorder=0)

        # (A) plot median including zeros/faint values
        yvals = lum_percsA[:, 1]
        yerr_low = lum_percsA[:, 1] - lum_percsA[:, 0]  # p50 - p14
        yerr_hi = lum_percsA[:, 2] - lum_percsA[:, 1]  # p84 - p50

        ax.errorbar(
            xmid,
            yvals,
            xerr=binSize / 2,
            yerr=[yerr_low, yerr_hi],
            color="#000000",
            ecolor="#000000",
            alpha=0.9,
            capsize=0.0,
            fmt="s",
            label="Median Bins (w/ zeros)",
        )

        # (B) plot median, only including actual detections (ish)
        yvals = lum_percsB[:, 1]
        yerr_low = lum_percsB[:, 1] - lum_percsB[:, 0]  # p50 - p14
        yerr_hi = lum_percsB[:, 2] - lum_percsB[:, 1]  # p84 - p50

        ax.errorbar(
            xmid + 0.02,
            yvals,
            xerr=binSize / 2,
            yerr=[yerr_low, yerr_hi],
            color="#000000",
            ecolor="#000000",
            alpha=0.9,
            capsize=0.0,
            fmt="o",
            label="Median Bins (only det)",
        )

        # (C) plot mean
        ax.errorbar(
            xmid + 0.04,
            lum_mean,
            xerr=binSize / 2,
            color="red",
            ecolor="red",
            alpha=0.9,
            capsize=0.0,
            fmt="o",
            label="Mean Stack",
        )

        # observational points from Anderson+
        from ..load.data import anderson2015

        a15 = anderson2015(sP)

        ax.errorbar(
            a15["stellarMass"],
            a15["xray_LumBol"],
            xerr=a15["stellarMass_err"],
            yerr=[a15["xray_LumBol_errDown"], a15["xray_LumBol_errUp"]],
            color="#000000",
            ecolor="#000000",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
            label=a15["label"],
        )

        # colorbar
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(sc, cax=cax, orientation="vertical")
        cb.set_alpha(1)  # fix stripes
        cb.draw_all()
        cb.ax.set_ylabel("log POIS")

        # legend and finish
        ax.legend(loc="best")
        fig.savefig(f"flux_comparison_gama_{rad_str}.pdf")
        plt.close(fig)

    # generate map to check uniformity
    if source == "random":
        # get extent = [ra_max, ra_min, dec_min, dec_max]
        file = basePathOut + "events_merged_image.fits"
        grid, _, extent = _load_map(file)

        aspect = grid.shape[1] / grid.shape[0]

        fig = plt.figure(figsize=[8 * aspect, 8])
        ax = fig.add_subplot(111)
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("DEC [deg]")

        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])

        # histogram
        nBinsX = 50
        nBinsY = int(nBinsX / aspect)

        h2d, _, _, _ = binned_statistic_2d(
            data["RA"],
            data["DEC"],
            data["flux"],
            statistic="mean",
            bins=[nBinsX, nBinsY],
            range=[[extent[1], extent[0]], [extent[2], extent[3]]],
        )
        h2d = h2d.T  # .T is normal convention for binned_statistic_2d return with imshow(origin='lower'),
        h2d = h2d[:, ::-1]  # then reverse x-direction to match axis reversed minmax

        h2d = logZeroNaN(h2d)
        norm = Normalize(vmin=-16.0, vmax=-15.0)  # flux [erg/s/cm^2]

        plt.imshow(h2d, extent=extent, norm=norm, origin="lower", interpolation="none", aspect="equal", cmap="viridis")

        # colorbar and finish
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel("Background/Random Sky Flux [erg/s/cm$^2$]")

        fig.savefig(f"flux_map_random_{rad_str}.pdf")
        plt.close(fig)


def gama_overlap():
    """Load GAMA catalog, find sources which are in the eFEDS footprint.

    Note this defines the input catalog for the aperture photometry, so changing here e.g. the
    additional catalogs we cross-match against requires new photometry to be run.
    """
    path = basePathGAMA + "LambdarInputCatUVOptNIR.fits"

    # open fits catalog file of Lambdar photometry
    with pyfits.open(path) as f:
        ra = f[1].data["RA"]  # deg
        dec = f[1].data["DEC"]  # deg
        gama_id = f[1].data["CATAID"]  # a few values of -999, meaning unclear

        #aperture_angle = f[1].data["ROTN2E"]  # orientation angle [deg]
        #aperture_maj = f[1].data["RADMAJ"]  # aperture semimajor axis length [arcsec]
        #aperture_min = f[1].data["RADMIn"]  # aperture semiminor axis length [arcsec]

    extent = [146.0, 126.0, -4.05, 7.05]  # eFEDS maps
    w = np.where((ra < extent[0]) & (ra > extent[1]) & (dec > extent[2]) & (dec < extent[3]) & (gama_id > 0))

    print("GAMA sources in eFEDS footprint: [%d of %d] = %.2f%%" % (len(w[0]), ra.size, (len(w[0]) / ra.size * 100)))

    gama_id = gama_id[w]
    ra = ra[w]
    dec = dec[w]

    # load stellar masses fits catalog
    path = basePathGAMA + "StellarMassesLambdar.fits"

    with pyfits.open(path) as f:
        gama_id2 = f[1].data["CATAID"]
        z = f[1].data["Z"]
        z_qual = f[1].data["nQ"]  # nQ, "use nQ > 2 for science"
        ppp = f[1].data["PPP"]  # dump any at 0 or maybe <~ 0.2
        mstar = f[1].data["logmstar"]  # log msun

    w2 = np.where((z_qual > 2) & (ppp > 0.1))
    gama_id2 = gama_id2[w2]
    z = z[w2]
    mstar = mstar[w2]

    # cross-match ids, construct catalog of available galaxies
    inds1, inds2 = match(gama_id, gama_id2)

    cat = {"gama_id": gama_id[inds1], "RA": ra[inds1], "DEC": dec[inds1], "z": z[inds2], "mstar": mstar[inds2]}

    # load MagPhys properties catalog
    path = basePathGAMA + "MagPhys.fits"

    with pyfits.open(path) as f:
        gama_id3 = f[1].data["CATAID"]
        ssfr_1gyr = f[1].data["sSFR_0_1Gyr_best_fit"]  # 1/yr
        mstar2 = f[1].data["mass_stellar_best_fit"]  # msun, unused
        metallicity = f[1].data["metalicity_Z_Zo_percentile50"]  # units?
        tform = f[1].data["tform_percentile50"]  # dex (yr), "of the oldest stars"
        age_masswt = f[1].data["agem_percentile50"]  # dex (yr)
        sfr_100myr = f[1].data["sfr18_percentile50"]  # dex(msun/yr)

    # cross-match ids
    inds_cat, inds3 = match(cat["gama_id"], gama_id3)

    # update properties of available galaxies
    for key in cat:
        cat[key] = cat[key][inds_cat]

    cat_new = {
        "ssfr_1gyr": ssfr_1gyr[inds3],
        "mstar_magphys": mstar2[inds3],
        "metallicity": metallicity[inds3],
        "tform": tform[inds3],
        "age_masswt": age_masswt[inds3],
        "sfr_100myr": sfr_100myr[inds3],
    }

    cat.update(cat_new)

    # load Sersic fit parameter catalog
    path = basePathGAMA + "SersicCatSDSS.fits"

    with pyfits.open(path) as f:
        gama_id4 = f[1].data["CATAID"]
        pa_r = f[1].data["GALPA_r"]  # also have u,g,r,z,i PAs [deg, CCW from x+, see notes]
        ellip_r = f[1].data["GALELLIP_r"]

    # cross-match ids
    inds_cat, inds4 = match(cat["gama_id"], gama_id4)
    assert inds_cat.size == cat["gama_id"].size  # must find all

    cat_new = {"pa_r": pa_r[inds4], "ellip_r": ellip_r[inds4]}
    cat.update(cat_new)

    # load ApMatched values (for PAs)
    path = basePathGAMA + "ApMatchedCat.fits"

    with pyfits.open(path) as f:
        gama_id5 = f[1].data["CATAID"]

        theta_image = f[1].data["THETA_IMAGE"]  # Position angle on image (counterclockwise, r band)
        theta_j2000 = f[1].data["THETA_J2000"]  # Position angle on sky (east of north, r band)

    # cross-match ids
    inds_cat, inds5 = match(cat["gama_id"], gama_id5)
    assert inds_cat.size == cat["gama_id"].size  # must find all

    cat_new = {"theta_image": theta_image[inds5], "theta_j2000": theta_j2000[inds5]}
    cat.update(cat_new)

    for key in cat.keys():
        assert cat[key].size == cat["z"].size

    return cat


def _load_map(file):
    """Helper."""
    # read
    with pyfits.open(file) as f:
        grid = np.array(f[0].data, dtype="float32")
        header = dict(f[0].header)
        # wcs = WCS(f[0].header)

    # fits/wcs fixes
    header["COMMENT"] = ""  # non-ascii characters
    header["RADESYSa"] = header.pop("RADECSYS")

    # coordinate system
    assert header["CTYPE1"] == "RA---SIN" and header["CTYPE2"] == "DEC--SIN"  # simple
    assert header["TIMEUNIT"] == "s"

    ra_min = header["CRVAL1"] + header["CDELT1"] * grid.shape[1] / 2
    ra_max = header["CRVAL1"] - header["CDELT1"] * grid.shape[1] / 2
    dec_min = header["CRVAL2"] - header["CDELT2"] * grid.shape[0] / 2
    dec_max = header["CRVAL2"] + header["CDELT2"] * grid.shape[0] / 2
    # radec = wcs.pixel_to_world(0,0)
    extent = [ra_max, ra_min, dec_min, dec_max]

    # print(f'{grid.shape = }')
    # print(f'{extent = }')
    # print(f'pixel scale = {header["CDELT2"]*60*60:.2f} arcsec/px')

    return grid, header, extent


def _vis_map(
    file, clabel, log=False, smooth=False, minmax=None, expcorrect=False, oplotClusters=False, oplotGAMA=False
):
    """Visualization helper: load and make a figure of a sky map in a fits file.

    Args:
      file (str): path to fits file.
      clabel (str): label for the map.
      log (bool): take log of grid.
      smooth (float): specify a float [pixel units] for Gaussian smoothing.
      minmax (float[2]): 2-tuple for scaling the displayed map.
      expcorrect (bool): if True, normalize the loaded map by the corresponding exposure map.
      oplotClusters (bool): overplot markers for the eFEDS Liu+ cluster catalog.
      oplotGAMA (bool): overplot markers for GAMA galaxies in the field, within a M* bin.
    """
    file_out = file.replace(".fits", ".pdf")

    # load
    grid, header, extent = _load_map(file)

    # exposure correct?
    if expcorrect:
        exp_file = file.replace("image", "expmap")
        with pyfits.open(exp_file) as f:
            expmap = np.array(f[0].data, dtype="float32")
            expheader = dict(f[0].header)
        for key in ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2"]:
            assert header[key] == expheader[key]

        # zero exposure times -> nan
        with np.errstate(invalid="ignore"):
            grid /= expmap
        file_out = file_out.replace(".pdf", "_expcorrected.pdf")
        print("Exposure corrected grid sum (countrate): ", np.nansum(grid))

    # data manipulation
    if smooth:
        grid = gaussian_filter(grid, smooth, mode="reflect", truncate=5.0)
    if log:
        grid = logZeroNaN(grid)

    if minmax is None:
        minmax = [np.nanmin(grid), np.nanmax(grid)]
    norm = Normalize(vmin=minmax[0], vmax=minmax[1])

    # plot
    aspect = grid.shape[1] / grid.shape[0]
    fig = plt.figure(figsize=[8 * aspect, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")

    # note! if interpolation='nearest', the raster image inside the pdf is resized according to the figsize
    plt.imshow(grid, extent=extent, norm=norm, origin="lower", interpolation="none", aspect="equal", cmap="viridis")

    # overplot markers?
    legend_elements = []

    def _add_markers(color, label, lw=1.0):
        # add markers
        circOpts = {"color": color, "alpha": 0.7, "linewidth": 0.1, "fill": False}
        legend_elements.append(Line2D([0], [0], marker="o", mew=1.0, mec=color, mfc="none", lw=0, label=label))

        for i in range(objs_ra.size):
            c = plt.Circle((objs_ra[i], objs_dec[i]), rad_deg[i], **circOpts)
            ax.add_artist(c)

    if oplotClusters:
        # load catalog
        with h5py.File(basePath + "eFEDS_clusters_V3.hdf5", "r") as f:
            objs_ra = f["RA"][()]
            objs_dec = f["DEC"][()]
            objs_z = f["z"][()]
            objs_F = f["F_500kpc"][()]  # 500 kpc flux

        w = np.where(objs_F > 0)
        rad_px = objs_F[w] * 1e14  # something for vis
        rad_deg = rad_px * header["CDELT2"]  # pixels are square
        objs_ra = objs_ra[w]
        objs_dec = objs_dec[w]
        objs_z = objs_z[w]

        _add_markers("red", label="eFEDS Clusters V3")
        file_out = file_out.replace(".pdf", "_liu.pdf")

    if oplotGAMA:
        # load catalog
        cat = gama_overlap()

        # simple mstar and z cut
        mstarBins = [[10.5, 11.5], [11.0, 11.5], [11.5, 12.5]]
        colors = ["orange", "yellow", "white"]

        z_max = 0.5  # 0.25

        for color, (mstar_min, mstar_max) in zip(colors, mstarBins):
            # select
            w = np.where((cat["mstar"] > mstar_min) & (cat["mstar"] < mstar_max) & (cat["z"] < z_max))
            print(f"GAMA: [{len(w[0])}] galaxies of [{cat['mstar'].size}] made [{mstar_min}-{mstar_max}] M* cut.")

            objs_ra = cat["RA"][w]
            objs_dec = cat["DEC"][w]
            rad_px = (cat["mstar"][w] - 10.5) / (12.5 - 10.5)  # [0,1]
            rad_px = rad_px / 2 + 0.5  # [0.5,1]
            rad_px *= 10  # 5-10px
            rad_deg = rad_px * header["CDELT2"]  # pixels are square

            _add_markers(color, "GAMA %.1f < M* < %.1f" % (mstar_min, mstar_max), lw=0.1)

            # testing: add EEF radii
            param_eef = 0.9

            rad_str = "eef%d" % (param_eef * 100) if param_eef < 1.0 else "arcsec%d" % param_eef
            file = basePathOut + "mllist_ape_out_gama_%s.fits" % rad_str

            if 1 and isfile(file):
                # load
                data = {}

                with pyfits.open(file) as f:
                    for key in f[1].data.names:
                        data[key] = f[1].data[key]

                # plot
                if param_eef < 1.0:
                    eef_size_deg = np.median(data["APE_RADIUS"]) * px_scale / (60 * 60)  # px->deg
                else:
                    eef_size_deg = np.median(data["APE_RADIUS"]) / (60 * 60)  # arcsec->deg

                rad_deg = np.zeros(rad_deg.size, dtype="float32")
                rad_deg += eef_size_deg

                _add_markers("black", "", lw=0.1)

        file_out = file_out.replace(".pdf", "_gama.pdf")

    # legend
    if len(legend_elements):
        ax.legend(handles=legend_elements, loc="lower left", fontsize="small")

    # colorbar and finish
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)

    fig.savefig(file_out)
    plt.close(fig)


def vis_map(type="final"):
    """Display a gridded image/map (made with eSASS evtool, expmap, erbackmap, ermask, etc)."""
    # raw counts
    if type == "counts":
        file = basePathOut + "events_merged_image.fits"
        _vis_map(file, "log events", log=True, minmax=[-1.8, -0.6], smooth=10)

    # raw count rate
    if type == "countrate":
        file = basePathOut + "events_merged_image.fits"
        _vis_map(file, "log events", log=True, minmax=[-5.0, -3.5], smooth=10, expcorrect=True)

    # exposure map
    if type == "expmap":
        file = basePathOut + "events_merged_expmap.fits"
        _vis_map(file, "log sec", log=True, minmax=[1.0, 3.3])

    # background map
    if type == "bkg":
        file = basePathOut + "bkg_map.fits"
        _vis_map(file, "counts")

    # source 'cheese' mask map
    if type == "cheese":
        file = basePathOut + "cheesemask.fits"
        _vis_map(file, "source mask")

    # count rate, source catalogs overplotted
    if type == "final":
        file = basePathOut + "events_merged_image.fits"
        _vis_map(
            file,
            "log counts/sec",
            log=True,
            minmax=[-5.0, -3.5],
            smooth=3,
            expcorrect=True,
            oplotClusters=True,
            oplotGAMA=True,
        )


def stack_map_cutouts(source="liu", bkg_subtract=True, reproject=True):
    """Using the pixel map, stack cutouts around sources.

    Args:
      source (str): if 'liu' use the Liu+ cluster sample, if 'gama' the GAMA catalog, if 'random' then
        generate random sampling points in the survey footprint.
      bkg_subtract (bool): if True, then also load the background map and subtract, prior to stacking.
      reproject (bool): if True, use the (expensive) reprojection algorithm. If 'interp', then use
        faster interpolation algorithm within reproject. If False, do a very simple
        shift approach, with no sub-pixel sampling, and no support for rotation.
    """
    from reproject import reproject_adaptive, reproject_interp

    # config
    stack_px = 40  # final map size, per dimension
    px_scale_target_kpc = 10.0  # pkpc, as compared to the native 4'' pixels = 7.6kpc for z=0.1
    exptime_thresh = 500.0  # sources in areas below this mean value are skipped [sec]

    sP = simParams(run="tng100-1")

    # load source catalog
    if source == "liu":
        # liu testing
        src_cat = {}
        with h5py.File(basePath + clusterCatName.replace(".fits", ".hdf5"), "r") as f:
            for key in ["DEC", "RA", "z", "Lbol_500kpc"]:
                src_cat[key] = f[key][()]
        src_cat["pa_r"] = np.zeros(src_cat["DEC"].size, dtype="float32")  # no rotations

    if source == "gama":
        # gama
        src_cat = gama_overlap()

        w = np.where(src_cat["pa_r"] < -1000)  # three bad values at -9999
        src_cat["pa_r"][w] = 0.0  # note: -90.0 < pa_r < 90.0

    if source == "random":
        # uniform random in [RA,DEC] rectangle bounds
        n = 10000
        rng = np.random.default_rng(424242)

        file = basePathOut + "events_merged_image.fits"
        _, _, extent = _load_map(file)

        # extent = [ra_max, ra_min, dec_min, dec_max]
        src_cat = {}
        src_cat["RA"] = rng.uniform(low=extent[1], high=extent[0], size=n)
        src_cat["DEC"] = rng.uniform(low=extent[2], high=extent[3], size=n)
        src_cat["pa_r"] = np.zeros(src_cat["DEC"].size, dtype="float32")  # no rotations
        src_cat["z"] = src_cat["pa_r"] + 0.1  # uniform

    # quick cache
    saveFilename = basePath + "stack_%s_bkgsub=%s_reproject=%s.hdf5" % (source, bkg_subtract, reproject)

    if isfile(saveFilename):
        # cached
        with h5py.File(saveFilename, "r") as f:
            stack = f["stack"][()]
        print(f"Loaded [{saveFilename}].")
    else:
        # load
        grid_cts, header_cts, _ = _load_map(basePathOut + "events_merged_image.fits")
        grid_exp, header_exp, _ = _load_map(basePathOut + "events_merged_expmap.fits")
        grid_bkg, header_bkg, _ = _load_map(basePathOut + "bkg_map.fits")

        wcs = WCS(header_cts)

        # background-subtracted source count rate
        with np.errstate(invalid="ignore"):
            if bkg_subtract:
                grid_countrate = (grid_cts - grid_bkg) / grid_exp  # counts/s
            else:
                grid_countrate = grid_cts / grid_exp  # counts/s

        # flux [count/sec] / [cm^2/erg] -> [erg/s/cm^2]
        grid_flux = grid_countrate / _ecf()

        # surface brightness [erg/s/cm^2/arcsec^2]
        px_area = np.abs(header_cts["CDELT1"] * 3600 * header_cts["CDELT2"] * 3600)  # arcsec^2
        grid_sb = grid_flux / px_area

        w = np.where(~np.isfinite(grid_sb))
        grid_sb[w] = 0.0  # clip, this is the unobserved boundaries of the map anyways

        # convert (ra,dec) coordinates to pixel coordinates
        px_x_coords, px_y_coords = wcs.wcs_world2pix(src_cat["RA"], src_cat["DEC"], 0)

        # allocate
        n = src_cat["RA"].size

        stack = np.zeros((stack_px, stack_px, n), dtype="float64")
        stack.fill(np.nan)

        for i in range(n):
            if i % int(n / 20) == 0:
                print("%d%%" % (i / n * 100))

            # (ra,dec) -> (x,y)
            x = px_x_coords[i]
            y = px_y_coords[i]

            j0 = int(np.round(x - stack_px / 2))
            j1 = j0 + stack_px
            i0 = int(np.round(y - stack_px / 2))
            i1 = i0 + stack_px

            # is position within the sampled map?
            if (
                (j0 < 0 or j0 > grid_sb.shape[1])
                or (j1 < 0 or j1 > grid_sb.shape[1])
                or (i0 < 0 or i0 > grid_sb.shape[0])
                or (i1 < 0 or i1 > grid_sb.shape[1])
            ):
                # cutout intersects edge of map
                continue

            loc_exptime = grid_exp[i0:i1, j0:j1]

            if loc_exptime.mean() < exptime_thresh:
                # heavily vignetted, or entirely off pointings
                continue

            # get local map cutout: simple local portion of original map
            if not reproject:
                loc_sb = grid_sb[i0:i1, j0:j1]

                # do rough rotation: pa_r directly from GAMA/GALFIT, "PA (x+, CCW)"
                angle = src_cat["pa_r"][i] + (src_cat["theta_j2000"][i] - src_cat["theta_image"][i])  # 'on-sky angle'
                angle *= -1.0  # undo pa

                loc_sb = rotate(loc_sb, angle, axes=(1, 0), reshape=False, mode="constant", cval=loc_sb.min() / 10)

            # use reproject to do cutout with optional transformation
            # cite: https://ui.adsabs.harvard.edu/abs/2020ascl.soft11023R/citations
            if reproject:
                output_wcs = WCS(naxis=2)

                # transformation
                output_wcs.wcs.ctype = wcs.wcs.ctype

                # reference position i.e. center
                output_wcs.wcs.crval = [src_cat["RA"][i], src_cat["DEC"][i]]

                # reference pixel (i.e. in exact center of grid)
                output_wcs.wcs.crpix = [stack_px / 2, stack_px / 2]  # [stack_px/2 + 0.5, stack_px/2 + 0.5]

                # pixel scale
                px_scale_target_deg = sP.units.physicalKpcToAngularSize(
                    px_scale_target_kpc, z=src_cat["z"][i], deg=True
                )
                output_wcs.wcs.cdelt = [px_scale_target_deg, px_scale_target_deg]
                # output_wcs.wcs.cdelt = wcs.wcs.cdelt # leave unchanged

                # linear transformation matrix, i.e. rotation
                # angle = -1.0 * src_cat['pa_r'][i] # directly from GAMA/GALFIT is "PA (x+, CCW)"
                angle = src_cat["pa_r"][i] + (src_cat["theta_j2000"][i] - src_cat["theta_image"][i])  # 'on-sky angle'
                angle *= -1.0  # undo pa

                output_wcs.wcs.pc = rotationMatrixFromAngle(angle)
                # output_wcs.wcs.pc = [[1.0,0.0], [0.0,1.0]] # identity

                # output grid size
                shape_out = (stack_px, stack_px)

                if reproject == "interp":
                    loc_sb, _ = reproject_interp((grid_sb, wcs), output_wcs, shape_out=shape_out)
                else:
                    loc_sb, _ = reproject_adaptive((grid_sb, wcs), output_wcs, shape_out=shape_out)

            # stamp
            stack[:, :, i] = loc_sb

        with h5py.File(saveFilename, "w") as f:
            f["stack"] = stack  # temp save

    # count valid
    bad_counts = np.count_nonzero(np.isnan(stack), axis=(0, 1))
    num_bad = len(np.where(bad_counts > 0)[0])
    print(f"Stackable sources in exposed area: {stack.shape[2] - num_bad}")

    # plot helper
    def _plot_stack(im, vminmax, name="", rel=False, recenter=False):
        """Helper."""
        fig = plt.figure()

        ylabel = "SB Flux [log erg/s/cm$^2$/arcsec$^2$]"

        if vminmax[1] > 0:
            ylabel = "Surface Brightness [log erg/s/kpc$^2$]"

        if recenter:
            # xxyy = np.arange(stack_px) + 0.5
            # xc = (im*xxyy).sum() / im.sum()
            im2 = im.copy()
            im2[np.isnan(im)] = np.nanmin(im) - 1.0

            cc = np.array(center_of_mass(im2))

            im = shift(im2, stack_px / 2 - cc, mode="constant", cval=np.nanmin(im) - 1.0)

        if rel:
            # subtract radially symmetric profile
            size_kpc = px_scale_target_kpc * stack_px
            dist, _ = dist_theta_grid(size_kpc, [stack_px, stack_px])

            w = np.where(~np.isnan(im))
            xx, yy, _ = running_median(dist[w], im[w], nBins=20)  # in log

            f = interp1d(xx, yy, kind="linear", bounds_error=False, fill_value="extrapolate")

            # we have our interpolating function for the average value at a given distance
            im = 10.0 ** (im) / 10.0 ** f(dist)  # SB -> SB/<SB> (linear)
            im = np.log10(im)

            # set bounds
            ylabel = r"$\phi$-Relative SB Flux [log]"
            if vminmax[1] > 0:
                ylabel = r"$\phi$-Relative SB [log]"
            vminmax = [-0.2, 0.2]

        norm = Normalize(vmin=vminmax[0], vmax=vminmax[1])
        extent = np.array([-stack_px / 2, stack_px / 2, -stack_px / 2, stack_px / 2]) * px_scale_target_kpc

        ax = fig.add_subplot(111)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")

        plt.imshow(im, origin="lower", norm=norm, extent=extent, interpolation="none", aspect="equal", cmap="viridis")

        # colorbar and finish
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel(ylabel)

        rStr = "simple"
        if reproject:
            rStr = "reproject"
        if reproject in ["interp"]:
            rStr = reproject
        fig.savefig("stack_map_cutouts_%s_%s_%s.pdf" % (source, rStr, name))
        plt.close(fig)

    # stack config
    if source == "liu":
        vminmax = [-18.4, -16.8]
    if source == "gama":
        vminmax = [-18.6, -17.8]
    if source == "random":
        vminmax = [-19.0, -18.4]

    vminmaxnorm = [36.0, 37.5]

    def _make_stacks(indiv_stacks, inds, name=""):
        """Helper. For the subset of inds images in indiv_stacks, make mean stack and mean normalized stack."""
        stack_mean = logZeroNaN(np.nanmean(indiv_stacks[:, :, inds], axis=2))
        print(f"[{name}] stacking [{inds.size}] galaxies.")

        # take into account the source redshift and e.g. convert flux->lum
        # (should probably call the latter SB, e.g. erg/s/arcsec^2)
        stack_norm = np.zeros((stack_px, stack_px, inds.size), dtype="float64")
        stack_norm.fill(np.nan)

        for i, ind in enumerate(inds):
            stack_norm[:, :, i] = sP.units.fluxToLuminosity(
                indiv_stacks[:, :, ind], redshift=src_cat["z"][ind]
            )  # units [erg/s/arcsec^2]
            assert np.count_nonzero(np.isinf(stack_norm[:, :, i])) == 0

            pxlenfac = sP.units.arcsecToAngSizeKpcAtRedshift(1.0, z=src_cat["z"][ind])  # pkpc per 1.0 arcsec

            stack_norm[:, :, i] /= pxlenfac**2  # [erg/s/arcsec^2] -> [erg/s/pkpc^2]

        stack_norm_mean = logZeroNaN(np.nanmean(stack_norm, axis=2))

        return stack_mean, stack_norm_mean

    # plot mean stack of all objects
    inds_all = np.arange(stack.shape[2])
    stack_mean, stack_norm_mean = _make_stacks(stack, inds=inds_all, name="all")

    _plot_stack(stack_mean, vminmax, "all")
    _plot_stack(stack_mean, vminmax, "all_rel", rel=True)

    _plot_stack(stack_norm_mean, vminmaxnorm, "all_norm")
    _plot_stack(stack_norm_mean, vminmaxnorm, "all_norm_rel", rel=True)

    _plot_stack(stack_mean, vminmax, "all_recen", recenter=True)
    _plot_stack(stack_norm_mean, vminmaxnorm, "all_norm_recen_rel", recenter=True, rel=True)

    # special plots by source
    if source == "gama":
        # test: restrict to z,M* bin
        assert src_cat["mstar"].size == stack.shape[2]

        inds = np.where((np.abs(src_cat["mstar"] - 11.0) < 0.2) & (np.abs(src_cat["z"] - 0.1) < 0.03))[0]

        stack_mean, stack_norm_mean = _make_stacks(stack, inds, name="Mstar11z03")

        _plot_stack(stack_mean, vminmax, "Mstar11z03")
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar11z03_norm")

        # restrict to M*=10.6 bin, relative
        inds = np.where(np.abs(src_cat["mstar"] - 10.6) < 0.2)[0]

        stack_mean, stack_norm_mean = _make_stacks(stack, inds, name="Mstar106")

        _plot_stack(stack_mean, vminmax, "Mstar106_rel", rel=True)
        _plot_stack(stack_mean, vminmax, "Mstar106", rel=False)

        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar106_norm_rel", rel=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar106_norm_recen_rel", rel=True, recenter=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar106_norm", rel=False)

        # restrict to small M*=10.6 bin, relative
        inds = np.where(np.abs(src_cat["mstar"] - 10.6) < 0.1)[0]

        stack_mean, stack_norm_mean = _make_stacks(stack, inds, name="Mstar106sm")

        _plot_stack(stack_mean, vminmax, "Mstar106sm_rel", rel=True)
        _plot_stack(stack_mean, vminmax, "Mstar106sm", rel=False)

        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar106sm_norm_rel", rel=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar106sm_norm", rel=False)

        # restrict to large M*=10.7 bin
        inds = np.where(np.abs(src_cat["mstar"] - 10.7) < 0.2)[0]

        stack_mean, stack_norm_mean = _make_stacks(stack, inds, name="Mstar107")

        _plot_stack(stack_mean, vminmax, "Mstar107_rel", rel=True)
        _plot_stack(stack_mean, vminmax, "Mstar107", rel=False)

        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar107_norm_rel", rel=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar107_norm", rel=False)

        # restrict to large M*=10.8 bin
        inds = np.where(np.abs(src_cat["mstar"] - 10.8) < 0.3)[0]

        stack_mean, stack_norm_mean = _make_stacks(stack, inds, name="Mstar108")

        _plot_stack(stack_mean, vminmax, "Mstar108_rel", rel=True)
        _plot_stack(stack_mean, vminmax, "Mstar108", rel=False)

        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar108_norm_rel", rel=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar108_norm_recen_rel", rel=True, recenter=True)
        _plot_stack(stack_norm_mean, vminmaxnorm, "Mstar108_norm", rel=False)

        # large M*=10.8 bin and ellipticity>0.3 (i.e. disk-like)
        for e_thresh in [0.3, 0.5, 0.6, 0.7]:
            inds = np.where((np.abs(src_cat["mstar"] - 10.8) < 0.3) & (src_cat["ellip_r"] > e_thresh))[0]

            name = "Mstar108e%d" % (e_thresh * 10)
            stack_mean, stack_norm_mean = _make_stacks(stack, inds, name=name)

            _plot_stack(stack_mean, vminmax, name + "_rel", rel=True)
            _plot_stack(stack_mean, vminmax, name, rel=False)

            _plot_stack(stack_norm_mean, vminmaxnorm, name + "_norm_rel", rel=True)
            _plot_stack(stack_norm_mean, vminmaxnorm, name + "_norm_recen_rel", rel=True, recenter=True)
            _plot_stack(stack_norm_mean, vminmaxnorm, name + "_norm", rel=False)


# current pipeline:
# note: seems neither 'region' nor 'central_position' really do anything for evtool, and all autoscaling
#       is based on the first .fits file of the eventfiles list only. so better to run evtool once just
#       to merge (image=false), then re-center, then run evtool again to image
# -- combine:
# evtool eventfiles="fm00_300008_020_EventList_c001.fits fm00_300007_020_EventList_c001.fits
#   fm00_300009_020_EventList_c001.fits fm00_300010_020_EventList_c001.fits"
#   outfile="fm00_merged_020_EventList_c001.fits"
# radec2xy fm00_merged_020_EventList_c001.fits 136.0 1.5
# -- make counts and exposure maps:
# evtool eventfiles="fm00_merged_020_EventList_c001.fits" emin=0.5 emax=2.0 outfile="events_merged_image.fits"
#   image=yes size="18000 10000" rebin="80" pattern=15 flag=0xc00fff30
# expmap inputdatasets="events_merged_image.fits" emin=0.5 emax=2.0 templateimage="events_merged_image.fits"
#   mergedmaps="events_merged_expmap.fits"
# ermask expimage="events_merged_expmap.fits" detmask="detmask.fits" threshold1=0.1 threshold2=1.0
# -- "local" source detection:
# erbox images="events_merged_image.fits" boxlist="boxlist_local.fits" emin=500 emax=2000
#   expimages="events_merged_expmap.fits" detmasks="detmask.fits" bkgima_flag=N ecf="1.2e12"
# -- background/mask maps, "map-based" source detection:
# erbackmap image="events_merged_image.fits" expimage="events_merged_expmap.fits" boxlist="boxlist_local.fits"
#   detmask="detmask.fits" bkgimage="bkg_map.fits" emin=500 emax=2000 cheesemask="cheesemask.fits"
# erbox images="events_merged_image.fits" boxlist="boxlist_map.fits" expimages="events_merged_expmap.fits"
#   detmasks="detmask.fits" bkgimages="bkg_map.fits" emin=500 emax=2000 ecf="1.2e12"
# -- characterize sources
# ermldet mllist="mllist.fits" boxlist="boxlist_map.fits" images="events_merged_image.fits"
#   expimages="events_merged_expmap.fits" detmasks="detmask.fits" bkgimages="bkg_map.fits"
#   extentmodel=beta srcimages="sourceimage.fits" emin=500 emax=2000
# -- generate psf map:
# apetool images="events_merged_image.fits" psfmaps="psf_map.fits" psfmapflag="yes"
# -- aperture photometry on user specified locations/apertures:
#    (note: seems that 'eefextract' parameter does nothing, and this is instead controlled in the input file)
# apetool mllist="mllist.fits" apelist="../ape_inputcat_liu.fits" apelistout="mllist_ape_out_liu.fits"
#   images="events_merged_image.fits" expimages="events_merged_expmap.fits" bkgimages="bkg_map.fits"
#   psfmaps="psf_map.fits" srcimages="sourceimage.fits" detmasks="detmask.fits" stackflag=yes
#   emin=500 emax=2000 eefextract=0.0 cutrad=15
# apetool mllist="mllist.fits" apelist="../ape_inputcat_gama_eef90.fits"
#   apelistout="mllist_ape_out_gama_eef90.fits" images="events_merged_image.fits"
#   expimages="events_merged_expmap.fits" bkgimages="bkg_map.fits" psfmaps="psf_map.fits"
#   srcimages="sourceimage.fits" detmasks="detmask.fits" stackflag=yes emin=500 emax=2000 eefextract=0.0 cutrad=15
# -- in arcsec instead of eef:
# apetool mllist="mllist.fits" apelist="../ape_inputcat_gama_arcsec20.fits"
#   apelistout="mllist_ape_out_gama_arcsec20.fits" images="events_merged_image.fits"
#   expimages="events_merged_expmap.fits" bkgimages="bkg_map.fits" psfmaps="psf_map.fits"
#   srcimages="sourceimage.fits" detmasks="detmask.fits" stackflag=yes emin=500 emax=2000 eefextract=0.0 cutrad=15
