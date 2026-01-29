"""
Diagnostic and production plots based on synthetic ray-traced absorption spectra.
"""

import glob
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from ..plot.config import figsize, linestyles, sKn, sKo
from ..plot.util import getWhiteBlackColors, loadColorTable, sampleColorTable, setAxisColors
from ..spectra.analysis import absorber_catalog, load_spectra_subset, wave_to_dv
from ..spectra.spectrum import (
    deposit_single_line,
    integrate_along_saved_rays,
    nRaysPerDim_def,
    raysType_def,
    spectra_filepath,
)
from ..spectra.util import (
    _equiv_width,
    _voigt_tau,
    create_wavelength_grid,
    line_params,
    lines,
    lsf_matrix,
    resample_spectrum,
    varconvolve,
)
from ..util.helper import closest, iterable, logZeroNaN
from ..util.units import units


def _cog(lines, N, b, nPts=5001):
    """Compute a local absorption spectrum for one or more line(s) to derive equivalent width.

    Args:
      lines (list[str]): names of transitions, e.g. ['LyA'], ['MgII 2803'] or ['CIV 1548','CIV 1550'].
        If multiple lines are given, the COG includes both i.e. for a doublet.
      N (float): log column density [cm^-2].
      b (float): Doppler parameter [km/s].
      nPts (int): number of points to sample the spectrum.
    """
    f, gamma, wave0_ang, _, _ = line_params(lines[0])

    wave_ang = np.linspace(wave0_ang - 40, wave0_ang + 40, nPts)  # 0.05 Ang spacing
    dvel = (wave_ang / wave0_ang - 1) * units.c_cgs / 1e5  # cm/s -> km/s

    # dwave_ang = wave_ang[1] - wave_ang[0]
    # wave_edges_ang = np.hstack(((wave_ang - dwave_ang/2),(wave_ang[-1] + dwave_ang/2)))

    tau = np.zeros(wave_ang.shape, dtype="float64")

    for line in lines:
        f, gamma, wave0_ang, _, _ = line_params(line)
        tau += _voigt_tau(wave_ang, 10.0**N, b, wave0_ang, f, gamma)

    if tau[0] > 1e-2 or tau[-1] > 1e-2:
        print("_cog(): Warning: optical depth at edges is large, consider increasing wave_ang range.")

    flux = np.exp(-1 * tau)
    EW = _equiv_width(tau, wave_ang)

    return flux, EW, dvel


def curve_of_growth(lines=("MgII 2803",), bvals=(5, 10, 15)):
    """Plot relationship between EW and column density (N) for the given transition(s).

    Args:
      lines (list[str]): names of transitions, e.g. ['LyA'], ['MgII 2803'] or ['CIV 1548','CIV 1550'].
        If multiple lines are given, the COG includes both i.e. for a doublet.
      bvals (list[float]): list of Doppler parameters [km/s] to vary across for the COG.
    """
    # plot flux
    fig, ax = plt.subplots()

    ax.set_xlabel("Velocity Offset [ km/s ]")
    ax.set_ylabel("Relative Flux")

    for _i, N in enumerate([13, 14, 15, 18, 20]):
        for j, b in enumerate(bvals):
            flux, EW, dvel = _cog(lines, N=N, b=b)
            print(f"{N = }, {b = }, {EW = :.3f}")

            label = f"{N = :.1f} cm$^{{-2}}$" if j == 0 else ""
            c = l.get_color() if j > 0 else None  # noqa: F821
            (l,) = ax.plot(dvel, flux, linestyle=linestyles[j], c=c, label=label)

    legend1 = ax.legend(loc="lower left")
    ax.add_artist(legend1)

    # legend two
    handles = []
    labels = []

    for j, b in enumerate(bvals):
        handles.append(plt.Line2D([0], [0], color="black", ls=linestyles[j]))
        labels.append(f"{b = :d} km/s")

    legend2 = ax.legend(handles, labels, loc="lower right")
    ax.add_artist(legend2)

    # finish plot
    fig.savefig("flux_%s.pdf" % "-".join(list(lines)))
    plt.close(fig)

    # plot cog
    fig, ax = plt.subplots()

    lineStr = "+".join(list(lines))
    ax.set_xlabel("Column Density [ log cm$^{-2}$ ]")
    ax.set_ylabel(lineStr + r" Equivalent Width [ $\AA$ ]")
    ax.set_yscale("log")
    ax.set_ylim([0.01, 10])

    cols = np.linspace(13.0, 20.0, 100)

    EW_cvals = [0.1, 0.5, 1.0, 3.0]

    for b in bvals:  # doppler parameter, km/s
        # draw some bands of constant EW
        xx = [cols.min(), cols.max()]
        for EW_cval in EW_cvals:
            ax.plot(xx, [EW_cval, EW_cval], "-", color="#444444", alpha=0.4)
            ax.fill_between(
                xx, [EW_cval * 0.9, EW_cval * 0.9], [EW_cval * 1.1, EW_cval * 1.1], color="#444444", alpha=0.05
            )

        # derive EWs as a function of column density
        EWs = np.zeros(cols.size, dtype="float32")
        for i, col in enumerate(cols):
            _, EW, _ = _cog(lines, N=col, b=b)
            EWs[i] = EW

        ax.plot(cols, EWs, label="b = %d km/s" % b)

    ax.legend(loc="upper left")
    fig.savefig("cog_%s.pdf" % "-".join(list(lines)))
    plt.close(fig)


def profile_single_line():
    """Voigt profile deposition of a single absorption line: create spectrum and plot."""
    # transition, instrument, and spectrum type
    line = "CIV 1548"
    instrument = None

    # config for 'this cell'
    N = 15.0  # log 1/cm^2
    b = 25.0  # km/s

    vel_los = 0.0  # 1000.0 # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_wavelength_grid(line=line, instrument=instrument)

    # deposit
    f, gamma, wave0, _, _ = line_params(line)

    z_doppler = vel_los / units.c_km_s
    z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1  # effective redshift

    deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff, debug=True)

    # compute flux
    flux_master = np.exp(-1 * tau_master)

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    ax1.set_xlabel("Wavelength [ Ang ]")
    ax1.set_ylabel("Relative Flux")
    ax1.plot(master_mid, flux_master, "o-", label="method A")
    # ax1.plot(wave_local, flux_local, '-', lw=lw, label='local')

    ax1.legend(loc="best")

    ax2.set_xlabel("Wavelength [ Ang ]")
    ax2.set_ylabel("Optical Depth $\\tau$")
    ax2.plot(master_mid, tau_master, "o-", label="method A")
    # ax2.plot(wave_local, tau_local, '-', lw=lw, label='local')

    ax2.legend(loc="best")
    fig.savefig("spectrum_single_%s.pdf" % line)
    plt.close(fig)


def profiles_multiple_lines(plotTau=True):
    """Deposit Voigt absorption profiles for a number of transitions: create spectrum and plot."""
    # transition, instrument, and spectrum type
    lineNames = ["LyA"] + [line for line in lines.keys() if "HI " in line]  # Lyman series
    instrument = "idealized"

    # config for 'this cell'
    N = 15.0  # log 1/cm^2
    b = 40.0  # km/s

    vel_los = 0.0  # 1000.0 # km/s
    z_cosmo = 0.0

    xlim = [800, 1300]

    if 1:
        # celine JWST cycle 2 proposal
        lineNames = ["NaI 5897", "NaI 5891"]
        instrument = "NIRSpec"
        N = 11.5
        z_cosmo = 0.9
        xlim = None

    # create master grid
    master_mid, master_edges, tau_master = create_wavelength_grid(instrument=instrument)

    # deposit
    z_doppler = vel_los / units.c_km_s
    z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1  # effective redshift

    for line in lineNames:
        f, gamma, wave0, _, _ = line_params(line)

        deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff)

    # compute flux
    flux_master = np.exp(-1 * tau_master)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1 + plotTau, 1, 1)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel("Wavelength [ Ang ]")
    ax.set_ylabel("Relative Flux")
    label = f"{N = :.1f} cm$^{{-2}}$, {b = :.1f} km/s"
    ax.plot(master_mid, flux_master, "-", label=label)

    ax.legend(loc="best")

    if plotTau:
        ax = fig.add_subplot(1 + plotTau, 1, 2)
        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_xlabel("Wavelength [ Ang ]")
        ax.set_ylabel(r"Optical Depth $\tau$")
        ax.plot(master_mid, tau_master, "-", label=label)

        ax.legend(loc="best")

    fig.savefig("spectrum_multi_%s.pdf" % ("-".join(lineNames)))
    plt.close(fig)


def profiles_multiple_lines_coldens():
    """Deposit Voigt absorption profiles for a number of transitions and N values: create spectrum and plot.

    Celine Peroux JWST cycle 2/3 proposal.
    """
    rng = np.random.default_rng(424244)

    # transition, instrument, and spectrum type
    lineNames = ["NaI 5897", "NaI 5891"]
    instrument = "NIRSpec"

    # physical config
    Nvals = [11.5, 12.0, 12.5]  # log 1/cm^2
    b = 5.0  # km/s
    SNR = 100.0

    z_cosmo = 0.9  # observed frame ~ 1.1 micron
    vel_los = 0.0  # 1000.0 # km/s

    ylim = None  # [0.7, 1.05]
    xlim = None

    _, lsf, _ = lsf_matrix(instrument)

    # plot
    fig = plt.figure(figsize=(14, 4))

    for i, N in enumerate(Nvals):
        ax = fig.add_subplot(1, 3, i + 1)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(r"Wavelength [ $\mu$m ]")
        ax.set_ylabel("Relative Flux")
        ax.set_title(r"log N$_{\rm NaI}$ = %.1f cm$^{{-2}}$" % N)
        ax.ticklabel_format(useOffset=False)

        # create master grid
        master_mid, master_edges, tau_master = create_wavelength_grid(instrument=instrument)

        # deposit
        z_doppler = vel_los / units.c_km_s
        z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1  # effective redshift

        for line in lineNames:
            f, gamma, wave0, _, _ = line_params(line)

            deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff)

        # convovle with LSF, then convert optical depth to relative flux
        tau_master = varconvolve(tau_master, lsf)
        flux_master = np.exp(-1 * tau_master)

        # down-sample to instrumental wavelength grid
        inst_mid, inst_waveedges, _ = create_wavelength_grid(instrument=instrument + "_inst")
        tau_inst = resample_spectrum(master_mid, tau_master, inst_waveedges)

        flux_inst = np.exp(-1 * tau_inst)

        # add noise? ("signal" is now 1.0)
        if SNR is not None:
            noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux_master.shape)
            flux_master += noise
            # achieved SNR = 1/stddev(noise)
            flux_master = np.clip(flux_master, 0, np.inf)  # clip negative values at zero

            noise2 = rng.normal(loc=0.0, scale=1 / SNR, size=flux_inst.shape)
            flux_inst += noise2
            flux_inst = np.clip(flux_inst, 0, np.inf)  # clip negative values at zero

        # plot
        master_mid /= 10000  # ang -> micron
        inst_mid /= 10000  # ang -> micron
        # ax.plot(master_mid, flux_master, '-', lw=lw)
        ax.plot(inst_mid, flux_inst, "-", color="black", drawstyle="steps")

    lineStr = "-".join([line.replace(" ", "") for line in lineNames])
    fig.savefig("spectrum_multi_%s_N%d_SNR%d_b%d.pdf" % (lineStr, len(Nvals), SNR, b))
    plt.close(fig)


def LyA_profiles_vs_coldens():
    """Reproduce Hummels+17 Figure 10 of LyA absorption profiles for various N_HI values."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from ..util.helper import loadColorTable

    line = "HI 1215"

    # config for 'this cell'
    N_vals = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # log 1/cm^2
    b = 22.0  # km/s

    vel_los = 0.0
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_wavelength_grid(line=line)

    # setup
    z_doppler = vel_los / units.c_km_s
    z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1  # effective redshift

    f, gamma, wave0, _, _ = line_params(line)

    # start plot
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot()

    ax.set_ylabel("Relative Flux")
    ax.set_xlabel("Wavelength [ Ang ]")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([wave0 - 1.0, wave0 + 1.0])

    # top x-axis (v/c = dwave/wave)
    ax2 = ax.twiny()
    ax2.set_xlabel(r"$\Delta$v [ km/s ]")

    dwave = np.array(ax.get_xlim()) - wave0  # ang
    dv = units.c_km_s * (dwave / wave0)

    ax2.set_xlim(dv)

    # colors
    cmap = loadColorTable("viridis")
    bounds = [N - 0.5 for N in N_vals] + [N_vals[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # loop over N values, compute a local spectrum for each and plot
    for N in N_vals:
        deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff, debug=True)

        # plot
        flux = np.exp(-1 * tau_master)
        ax.plot(master_mid, flux, "-", color=sm.to_rgba(N))

    # finish plot
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(sm, cax=cax, ticks=N_vals)
    cb.ax.set_ylabel(r"log N$_{\rm HI}$ [ cm$^{-2}$ ]")

    fig.savefig("LyA_absflux_vs_coldens.pdf")
    plt.close(fig)


def instrument_lsf(instrument):
    """Plot LSF(s) of a given instrument. For wavelength-dependent LSF matrices."""
    num = 6

    # get wavelength grid and wavelength-dependent LSF
    wave_mid, _, _ = create_wavelength_grid(instrument=instrument)
    lsf_mode, lsf, lsf_fwhm = lsf_matrix(instrument)

    print(f"{lsf_mode = }, {lsf.shape = }")

    # x-axis
    lsf_size = lsf.shape[1]
    cen_i = int(np.floor(lsf_size / 2))

    xx = np.arange(lsf_size, dtype="int32") - cen_i

    # start plot
    fig, axes = plt.subplots(ncols=1, nrows=3, height_ratios=[1, 1, 0.5], figsize=(8, 16.8))

    for ax in axes[0:2]:
        ax.set_xlabel("Pixel Number")
        if lsf_size < 100:
            ax.set_xticks(xx)
        ax.set_ylabel(f"{instrument} LSF")

        # first and second panels are identical, except second is y-log
        if ax == axes[1]:
            ax.set_yscale("log")

        # add a number of LSFs across the instrumental wavelength range
        for i in range(num):
            # evenly sample
            d_ind = lsf.shape[0] / num
            ind = int(i * d_ind + d_ind / 2)

            lsf_kernel = lsf[ind, :]

            label = r"$\rm{\lambda = %.1f \AA}$" % wave_mid[ind]

            if xx.size < 100:
                ax.plot(xx, lsf_kernel, "o-", label=label)
            else:
                ax.plot(xx, lsf_kernel, label=label)

        ax.plot([xx[cen_i], xx[cen_i]], [0, ax.get_ylim()[1]], "--", color="#ccc")

    # bottom panel: FWHM vs wave
    axes[-1].set_xlabel(r"Wavelength [ $\rm{\AA}$ ]")
    axes[-1].set_ylabel(r"FWHM [ $\rm{\AA}$ ]")

    axes[-1].plot(wave_mid, lsf_fwhm)

    # finish plot
    axes[0].legend(loc="upper right")
    fig.savefig("lsf_%s.pdf" % instrument)
    plt.close(fig)


def cos_disptab():
    """Handle DISPTAB files, wavelength grids, and LSF files for HST-COS (pre-processing step)."""
    from astropy.io import fits

    from ..util.helper import rootPath

    basePath = rootPath + "tables/hst/"

    files = glob.glob(basePath + "*.fits")

    # https://spacetelescope.github.io/hst_notebooks/notebooks/COS/LSF/LSF.html
    # https://www.stsci.edu/hst/instrumentation/cos/performance/spectral-resolution
    # https://hst-docs.stsci.edu/cosihb/chapter-5-spectroscopy-with-cos/5-5-spanning-the-gap-with-multiple-cenwave-settings#id-5.5SpanningtheGapwithMultipleCENWAVESettings-Table5.3

    aperture = "PSA"

    # NUV: have G185M from 1760-2127, (15 cenwave settings) 1664.2-2132.3 (0.035 spacing)
    #           G225M from 2070-2527, (13 cenwave settings) 2069.8-2523.0 (0.032 spacing)
    #           G285M from 2480-3229, (17 cenwave settings) 2476.4-3222.9 (0.037 spacing)
    #           G230L from 1334-3560 (four cenwave settings) 1349.9-3585.0 (0.19 spacing)
    # segments: A, B, and C
    # 63p1559jl_disp.fits is the only NUV disptab, the rest are all FUV

    # FUV: HSLA unifies G130M 892.5-1479.7 (0.00997 constant spacing), (8 cenwaves)
    #           unifies G160M 1374.5-1810.3 (0.01223 constant spacing), (6 cenwaves)
    #           unifies G140L 1026.8-2496.2 (0.083 constant spacing) (3 cenwaves)
    # segments: A and B

    # load all disptabs and print statistics
    # note: LIFE_ADJ == LP (wavelength grids are the same)
    for file in files:
        # load
        dtab = {}
        with fits.open(file) as f:
            for key in [col.name for col in f[1].data.columns]:
                dtab[key] = f[1].data[key]

        # gratings
        gratings = np.unique(dtab["OPT_ELEM"])
        gratings_minwave = dict.fromkeys(gratings, np.inf)
        gratings_maxwave = dict.fromkeys(gratings, 0.0)
        gratings_mindisp = dict.fromkeys(gratings, np.inf)
        gratings_maxdisp = dict.fromkeys(gratings, 0.0)

        # unique entries
        cenwaves = np.unique(dtab["CENWAVE"])
        segments = np.unique(dtab["SEGMENT"])

        for cenwave in cenwaves:
            for segment in segments:
                index = np.where(
                    (dtab["CENWAVE"] == cenwave) & (dtab["SEGMENT"] == segment) & (dtab["APERTURE"] == aperture)
                )[0]

                if len(index) == 0:
                    print(file, segment, cenwave, "SKIP")
                    continue

                grating = dtab["OPT_ELEM"][index][0]

                coeffs = dtab["COEFF"][index][0]
                coeffs = coeffs[::-1]

                # d_tv03 = dtab['D_TV03'][index]  # Offset from WCA to PSA in Thermal Vac. 2003 data
                # d_orbit = dtab['D'][index]  # Current offset from WCA to PSA
                # delta_d = d_tv03 - d_orbit

                # pixel indices for a given detector
                if "NUV" in segment:
                    pixel_inds = np.arange(1024)
                if "FUV" in segment:
                    pixel_inds = np.arange(16384)

                # sample polynomial mapping from pixel number to wavelength
                # w = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
                # and p[0] ~ 0.0, [1] ~ 0.0, such that p[2] is the constant ang per pixel, p[3] is the zeropoint
                wave = np.polyval(p=coeffs, x=pixel_inds)

                # print(file,grating,cenwave,segment,coeffs,wave.min(),wave.max())

                if wave.min() < gratings_minwave[grating]:
                    gratings_minwave[grating] = wave.min()
                if wave.max() > gratings_maxwave[grating]:
                    gratings_maxwave[grating] = wave.max()

                disp = coeffs[2]
                if disp < gratings_mindisp[grating]:
                    gratings_mindisp[grating] = disp
                if disp > gratings_maxdisp[grating]:
                    gratings_maxdisp[grating] = disp

        for grating in gratings:
            print(
                grating,
                gratings_minwave[grating],
                gratings_maxwave[grating],
                gratings_mindisp[grating],
                gratings_maxdisp[grating],
            )

    # note: different cenwave settings sample the LSF at overlapping wavelengths
    # switch as soon as possible in ascending cenwave order
    inst_maxwaves = {
        "G130M_1055_LP2": 940,
        "G130M_1096_LP2": 1067,
        "G130M_1222_LP4": 1134,
        "G130M_1291_LP4": 1144,
        "G130M_1300_LP4": 1154,
        "G130M_1309_LP4": 1163,
        "G130M_1318_LP4": 1172,
        "G130M_1327_LP4": np.inf,
        "G160M_1533_LP4": 1386,
        "G160M_1577_LP4": 1397,
        "G160M_1589_LP4": 1409,
        "G160M_1600_LP4": 1420,
        "G160M_1611_LP4": 1432,
        "G160M_1623_LP4": np.inf,
        "G140L_0800_LP4": 1118,
        "G140L_1105_LP4": 1293,
        "G140L_1280_LP4": np.inf,
    }

    save_lsf_waves = {g: [] for g in gratings}
    save_lsfs = {g: [] for g in gratings}

    # load LSFs (LP4 era for all except G130M at 1055 and 1096 which are LP2) and plot
    for grating in gratings:
        files = sorted(glob.glob(basePath + "aa_LSFTable_%s_*_cn.dat" % grating))

        for file in files:
            # read
            with open(file) as f:
                lines = f.readlines()

            # wavelengths where the LSFs are sampled, and the size (number of pixels) of each
            lsf_samples = np.array([float(wave) for wave in lines[0].split(" ")], dtype="float32")
            npx = len(lines) - 1

            lsfs = np.zeros((lsf_samples.size, npx), dtype="float32")

            instrument = file.split("LSFTable_")[1].split("_cn.dat")[0]
            print(instrument, lsfs.shape, lsf_samples.min(), lsf_samples.max())

            for i, line in enumerate(lines[1:]):
                lsfs[:, i] = np.array(line.split(" "), dtype="float32")

            # plot
            fig, ax = plt.subplots()

            ax.set_xlabel("Pixel Number")
            ax.set_ylabel("%s LSF" % instrument)
            ax.set_xlim([0, npx])
            ax.set_ylim([5e-6, 1e-1])
            ax.set_yscale("log")

            stride = lsfs.shape[0] // 8
            for i in np.arange(lsfs.shape[0])[::stride]:
                xx = np.arange(npx)  # number of pixels in each LSF
                ax.plot(xx, lsfs[i, :], "-", label=r"%d $\AA$" % lsf_samples[i])

            ax.legend(loc="best")
            fig.savefig("lsf_%s.pdf" % instrument)
            plt.close(fig)

            # save subset of LSFs into master set (for this grating)
            for i, wave in enumerate(lsf_samples):
                if wave > inst_maxwaves[instrument]:
                    break
                if instrument == "G140L_1280_LP4" and wave < inst_maxwaves["G140L_1105_LP4"]:
                    continue

                save_lsf_waves[grating].append(wave)
                save_lsfs[grating].append(lsfs[i, :])

    # LSFs are in pixel coordinates, leave as is (do not convert to wavelength space)
    # the cenwave dependence of the disptab solution is not important (constant) for FUV
    # while it is more important for NUV, but for NUV we anyways have only a single LSF file
    # that is independent of cenwave and even grating
    for grating in gratings:
        save_lsf_waves[grating] = np.array(save_lsf_waves[grating])

        # save
        saveFilename = f"COS-{grating}.txt"
        with open(saveFilename, "w") as f:
            for i, wave in enumerate(save_lsf_waves[grating]):
                line = "%.1f" % wave
                for j in range(save_lsfs[grating][i].size):
                    line += " %.6e" % save_lsfs[grating][i][j]
                f.write(line + "\n")

        print("Saved: [%s]." % saveFilename)

    # rewrite NUV into same format
    file = basePath + "nuv_model_lsf.dat"

    with open(file) as f:
        lines = f.readlines()

    lsf_samples = np.array([float(wave) for wave in lines[0].split(" ")], dtype="float32")
    lsfs = np.zeros((lsf_samples.size, len(lines) - 1), dtype="float32")

    for i, line in enumerate(lines[1:]):
        lsfs[:, i] = np.array(line.split(" "), dtype="float32")

    saveFilename = "COS-NUV.txt"
    with open(saveFilename, "w") as f:
        for i, wave in enumerate(lsf_samples):
            line = "%.1f" % wave
            for j in range(lsfs[i, :].size):
                line += " %.6e" % lsfs[i, j]
            f.write(line + "\n")

    print("Saved: [%s]." % saveFilename)


def spectra_gallery_indiv(
    sim,
    ion="Mg II",
    instrument="4MOST-HRS",
    nRaysPerDim=nRaysPerDim_def,
    raysType=raysType_def,
    EW_minmax=(0.1, 1.0),
    num=10,
    mode="random",
    inds=None,
    style="offset",
    solar=False,
    SNR=None,
    dv=False,
    xlim=None,
):
    """Plot a gallery of individual absorption profiles within a given EW range.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      ion (str, list[str]): space separated species name and ionic number e.g. 'Mg II', or list of such.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced) or 'voronoi_rndfullbox' (random).
      EW_minmax (list[float]): minimum and maximum EW to plot [Ang].
      num (int): how many individual spectra to show.
      mode (str): either 'random', 'evenly', or 'inds'.
      inds (list[int]): if mode is 'inds', then the list of specific spectra indices to plot. num is ignored.
      style (str): type of plot, 'stacked', 'grid', or '2d'.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead
        use the (constant) solar value.
      SNR (float): if not None, then add noise to achieve this signal to noise ratio.
      dv (bool): if False, x-axis in wavelength, else in velocity.
      xlim (str, list[float]): either 'full', a 2-tuple of [min,max], or automatic if None (default)
    """
    assert mode in ["random", "evenly", "inds", "all"]
    if mode == "inds":
        assert inds is not None
    if mode == "all":
        assert num is None and style == "2d"
    if mode in ["random", "evenly"]:
        assert inds is None

    # config
    ctName = "thermal"

    # load
    dv_window = 1000.0  # km/s, TODO should depend on transition, maybe inst

    if isinstance(xlim, list) and dv_window < xlim[1]:
        print(f"Increaing {dv_window = } to {xlim[1]} km/s to cover requested xlim.")
        dv_window = xlim[1]

    if isinstance(ion, list):
        # multiple ions, stack
        print(f"Loading multiple ions: {ion} ...")
        wave, flux, EW, _, lineNames = load_spectra_subset(
            sim, ion[0], instrument, mode, nRaysPerDim=nRaysPerDim, raysType=raysType, solar=solar, inds=inds
        )

        tau = -np.log(flux)

        for ion_loc in ion[1:]:
            wave_loc, flux_loc, _, _, lineNames_loc = load_spectra_subset(
                sim, ion_loc, instrument, mode, nRaysPerDim=nRaysPerDim, raysType=raysType, solar=solar, inds=inds
            )
            assert np.array_equal(wave, wave_loc)
            tau_loc = -np.log(flux_loc)
            tau += tau_loc

            if ion_loc == "H I":
                lineNames_loc = lineNames_loc[0:6]
            lineNames += lineNames_loc

        flux = np.exp(-1 * tau)

    else:
        # single ion load
        wave, flux, EW, _, lineNames = load_spectra_subset(
            sim,
            ion,
            instrument,
            mode,
            nRaysPerDim=nRaysPerDim,
            raysType=raysType,
            solar=solar,
            num=num,
            inds=inds,
            EW_minmax=EW_minmax,
            dv=dv_window if dv else 0.0,
        )

    # how many lines do we have? what is their span in wavelength?
    lines_wavemin = 0
    lines_wavemax = np.inf

    for line in lineNames:
        line = line.replace("_", " ")
        if line not in lines:
            continue  # old datasets
        lines_wavemin = np.clip(lines_wavemin, lines[line]["wave0"], np.inf)
        lines_wavemax = np.clip(lines_wavemax, 0, lines[line]["wave0"])

    # add noise? ("signal" is now 1.0)
    if SNR is not None:
        rng = np.random.default_rng(424242)
        noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux.shape)
        flux += noise
        # achieved SNR = 1/stddev(noise)
        flux = np.clip(flux, 0, np.inf)  # clip negative values at zero

    # determine wavelength (x-axis) bounds
    if str(xlim) == "full":
        xlim = [np.min(wave), np.max(wave)]
    elif isinstance(xlim, list):
        # input directly
        pass
    else:
        # automatic
        xlim = [np.inf, -np.inf]

        for i in range(flux.shape[0]):
            w = np.where(flux[i, :] < 0.99)[0]
            if len(w) == 0:
                continue

            xx_min = wave[w].min()
            xx_max = wave[w].max()

            if xx_min < xlim[0]:
                xlim[0] = xx_min
            if xx_max > xlim[1]:
                xlim[1] = xx_max

        dx = (xlim[1] - xlim[0]) * 0.01
        xlim[0] = np.floor((xlim[0] - dx) / 5) * 5
        xlim[1] = np.ceil((xlim[1] + dx) / 5) * 5

    if dv:
        # symmetrize
        xlim = [-np.max(np.abs(xlim)), np.max(np.abs(xlim))]

    # other common plot config
    title = r"%s ($\rm{z \simeq %.1f}$) %s" % (ion, sim.redshift, instrument)
    xlabel = r"$\Delta v$ [ km/s ]" if dv else "Wavelength [ Ang ]"
    ylabel = "Relative Flux"

    if num is None and inds is not None:
        num = len(inds)

    if style == "offset":
        # plot - single panel, with spectra vertically offset
        colors = sampleColorTable(ctName, num, bounds=[0.0, 0.9])
        figsize_loc = [figsize[0] * 0.6, figsize[1] * 1.5 * np.sqrt(num / 10)]

        fig, ax = plt.subplots(figsize=figsize_loc)

        if num > 1:
            ylabel += " (+ constant offset)"

        # determine flux (y-axis) bounds
        spacingFac = 1.0
        if EW_minmax is not None:
            if np.max(EW_minmax) <= 0.4:
                spacingFac = 0.5
            if np.max(EW_minmax) > 0.8:
                spacingFac = 1.1  # 0.8 #1.2
        ylim = [+spacingFac / 2, num * spacingFac + spacingFac / 5]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_yticks(np.arange(num + 1) * spacingFac)
        ax.set_yticklabels(["%d" % i for i in range(num + 1)])

        for i in range(flux.shape[0]):
            # vertical offset by 1.0 for each spectrum
            y_offset = (i + 1) * spacingFac - 1

            ax.step(wave, flux[i, :] + y_offset, "-", color=colors[i], where="mid")

            # label
            text_x = xlim[0] + (xlim[1] - xlim[0]) / 50
            text_y = y_offset + 1.0 - (num / 50) * spacingFac
            if SNR is not None:
                text_y -= (num / 50) * (5 / SNR)
            label = r"EW = %.2f$\AA$" % EW[i]

            ax.text(text_x, text_y, label, color=colors[i], alpha=0.6, fontsize=18, ha="left", va="top")

        # finish plot
        # ax.legend([plt.Line2D((0,1),(0,0),lw=0,marker='')], [title], fontsize=20, loc='upper right')
        ax.set_title(title)

    if style == "grid":
        # plot - (square) grid of many panels, each with one spectrum
        colors = sampleColorTable(ctName, num, bounds=[0.0, 0.9])
        n = int(np.sqrt(num))
        figsize_loc = [figsize[0] * 1.3 * (n / 10), figsize[0] * 1.3 * (n / 10)]

        gs = {"left": 0.06, "bottom": 0.06, "right": 0.95, "top": 0.95}

        fig = plt.figure(figsize=figsize_loc)
        axes = fig.subplots(nrows=n, ncols=n, sharex=True, sharey=True, gridspec_kw=gs)

        fontsize = 23 * (n / 10)
        fig.suptitle(title, fontsize=fontsize)
        fig.supxlabel(xlabel, fontsize=fontsize)
        fig.supylabel(ylabel, fontsize=fontsize)

        ylim = [-0.1, 1.1]
        xticks = [-600, 0, 600] if dv else None  # needs generalization, axes[i,j].get_xticks()[1::2]

        for i in range(n):
            for j in range(n):
                # axis config
                axes[i, j].set_xlim(xlim)
                axes[i, j].set_ylim(ylim)
                axes[i, j].set_yticks([0.0, 0.5, 1.0])
                axes[i, j].set_yticklabels(["0", "", "1"])

                if xticks is not None:
                    axes[i, j].set_xticks(xticks)

                # grid
                axes[i, j].plot(xlim, [0.5, 0.5], "-", color="#000", alpha=0.1)
                axes[i, j].plot([0, 0], ylim, "-", color="#000", alpha=0.1)

                # plot
                axes[i, j].step(wave, flux[i * n + j, :], "-", c=colors[i * n + j], where="mid")

        # fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)

    if style == "2d":
        # plot - single 2d panel, color indicating relative flux
        inds = np.argsort(EW)
        num = inds.size
        flux = flux[inds, :]
        EW = EW[inds]

        # plot
        fig, ax = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 1.5])
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"Equivalent Width [ $\rm{\AA}$ ]")
        cbar_label = "Relative Flux"

        if EW_minmax is None:  # for display only
            EW_minmax = [0.01, EW.max()]

        ylim = [np.log10(EW_minmax[0]), np.log10(EW_minmax[1])]
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        norm = Normalize(vmin=0.99, vmax=1.0)

        nbins = 1000
        bins = np.linspace(ylim[0], ylim[1], nbins)
        EW = np.log10(EW)

        # if wave.size > hueristic, downsample (in horizontal direction)
        n_x = wave.size
        while n_x > 2000:
            n_x = int(n_x / 2)

        h2d = np.zeros((nbins, n_x), dtype="float32")

        for i in range(nbins - 1):
            w = np.where((EW >= bins[i]) & (EW < bins[i + 1]))[0]

            flux_mean = np.mean(flux[w, :], axis=0)

            h2d[i, :] = flux_mean.reshape((n_x, flux_mean.size // n_x)).mean(-1)

        # show in log?
        if 0:
            h2d = logZeroNaN(h2d)
            norm = Normalize(vmin=-0.01, vmax=0.0)
            cbar_label += " [ log ]"

        s = ax.imshow(h2d, extent=extent, norm=norm, origin="lower", aspect="auto", cmap="plasma")

        yticks_all = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        yticks = []
        yticklabels = []
        for ytick in yticks_all:
            if ytick >= EW_minmax[0] and ytick <= EW_minmax[1]:
                yticks.append(ytick)
                yticklabels.append("%.1f" % ytick if ytick >= 0.1 else "%.2f" % ytick)
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels(yticklabels)

        # colorbar
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
        cb = fig.colorbar(s, cax=cax)
        cb.ax.set_ylabel(cbar_label)

        # mark transitions
        if 1:
            opts = {"color": "#000", "alpha": 0.8, "ha": "center", "va": "bottom"}
            ypos = np.linspace(ylim[0] + 0.1, ylim[0] + 0.1 * len(lineNames), len(lineNames))

            for line, yy in zip(lineNames, ypos):
                line = line.replace("_", " ")
                if line not in lines:
                    continue  # old datasets
                wave_z = lines[line]["wave0"] * (1 + sim.redshift)
                ax.text(wave_z, yy, line, **opts)

    snrStr = "_snr%d" % SNR if SNR is not None else ""
    ewStr = "_%.1f-%.1f" % (EW_minmax[0], EW_minmax[1]) if EW_minmax is not None else ""
    fig.savefig(
        "spectra_%s_%d_%s_%s%s_%d-%s%s_%s.pdf"
        % (sim.simName, sim.snap, ion.replace(" ", ""), instrument, ewStr, num, mode, snrStr, style)
    )
    plt.close(fig)


def _select_random_spectrum(file=None):
    """Select a random spectrum, from all those available.

    Args:
      file (str): if not None, the specific file to load. Otherwise, randomly select from those available.
    """
    basepath = "/virgotng/universe/IllustrisTNG/"
    rng = np.random.default_rng()  # 424242
    sims = ["TNG50-1"]

    if file is None:
        files = []
        for sim in sims:
            # sim = simParams(sim)
            # path = sim.postPath + 'AbsorptionSpectra/spectra*combined.hdf5'
            path = basepath + "%s/postprocessing/AbsorptionSpectra/spectra*combined.hdf5" % sim

            files += sorted(glob.glob(path))

        # pick one at random
        file = rng.choice(files)
        # print(file)

    # load all EWs of one transition (chosen at random)
    with h5py.File(file, "r") as f:
        # lines_all = [key.split('EW_')[1].replace('_',' ') for key in f.keys() if 'EW_' in key]
        lines = f.attrs["lineNames"]
        redshift = f.attrs["redshift"]

        line = rng.choice(lines)

        key = "EW_%s" % line.replace(" ", "_")
        if key not in f or "flux" not in f:
            # print('No EWs for line [%s], or flux is missing, in [%s].' % (line, file))
            return _select_random_spectrum()

        EWs = f[key][()]
        EWs /= 1 + redshift  # rest-frame

    if 0:
        # pick randomly from EWs (biased towards low EWs)
        EW_min = 0.1
        inds = np.where(EWs >= EW_min)[0]
        if len(inds) == 0:
            # print('No EWs >= %.2f Ang in [%s].' % (EW_min, file))
            return _select_random_spectrum()

        ind = rng.choice(inds)

    if 1:
        # pick a random EW, then find the closest match
        EW_rnd = rng.uniform(low=0.0, high=np.min([5.0, EWs.max()]))
        _, ind = closest(EWs, EW_rnd)

    return file, ind


def spectrum_plot_single(file=None, ind=None, full=True, saveFilename=None, pStyle="white", output_fmt="png"):
    """Plot a randomly selected spectrum.

    Args:
      file (str): if not None, the specific file to load.
      ind (int): if not None, the specific spectrum index to load.
      full (bool): show the full wavelength range in addition to zooming in on the absorption feature.
      saveFilename (str): if not None, the specific filename to save to (otherwise auto-generated).
      pStyle (str): either 'white' or 'black' for plot style.
      output_fmt (str): either 'pdf', 'png', or 'jpg'.
    """
    SNR_bounds = [5, 200]  # if not None, add noise with a random SNR in this range.

    rng = np.random.default_rng()

    if file is None or ind is None:
        file, ind = _select_random_spectrum(file)

    params = file.split("/")[-1].split("_")[1:-1]  # sim, redshift, config, inst, ion
    ion = params[4]
    config = params[2]

    # load and plot spectrum
    with h5py.File(file, "r") as f:
        wave = f["wave"][()]
        flux = f["flux"][ind]

        lines = f.attrs["lineNames"]
        instrument = f.attrs["instrument"]
        redshift = f.attrs["redshift"]
        # count = int(np.sqrt(f.attrs['count']))
        # ray_dir = np.where(f['ray_dir'][()] == 1)[0]
        simName = f.attrs["simName"]

        EWs = [f["EW_%s" % line][ind] for line in lines]

    EWs = np.array(EWs) / (1 + redshift)  # rest-frame

    # determine automatic bounds
    xlim = [np.inf, -np.inf]

    w = np.where(flux[:] < 0.99)[0]
    if len(w) == 0:
        xlim = [wave.min(), wave.max()]
    else:
        xlim = [wave[w].min(), wave[w].max()]

    dx = (xlim[1] - xlim[0]) * 0.01
    xlim[0] = np.floor((xlim[0] - dx) / 5) * 5
    xlim[1] = np.ceil((xlim[1] + dx) / 5) * 5

    # select dv window
    dv_val = 1000
    xlim = [-dv_val, dv_val]

    wave_dv, flux_dv = wave_to_dv(wave, flux, dv=dv_val)

    # adapt xlim automatically
    n_edge = np.max([1, wave_dv.size // 20])
    edge_flux_mean = np.max([np.mean(flux_dv[0:n_edge]), np.mean(flux_dv[-n_edge:])])

    if wave_dv.size < 50 or edge_flux_mean < 0.9:
        dv_val *= 2
        xlim = [-dv_val, dv_val]

        wave_dv, flux_dv = wave_to_dv(wave, flux, dv=dv_val)

    elif wave_dv.size > 200 and edge_flux_mean > 0.99:
        dv_val = 400
        xlim = [-dv_val, dv_val]

        wave_dv, flux_dv = wave_to_dv(wave, flux, dv=dv_val)

    # add noise (not actually consistent between zoom and full!)
    if SNR_bounds is not None:
        SNR = rng.uniform(low=SNR_bounds[0], high=SNR_bounds[1])

        noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux_dv.shape)
        flux_dv += noise  # achieved SNR = 1/stddev(noise)
        flux_dv = np.clip(flux_dv, 0, np.inf)  # clip negative values at zero

        noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux.shape)
        flux += noise  # achieved SNR = 1/stddev(noise)
        flux = np.clip(flux, 0, np.inf)  # clip negative values at zero

    # other common plot config
    title = r"%s ($\rm{z = %.1f}$) %s (%s)" % (ion, redshift, instrument, simName)
    ylabel = "Relative Flux"

    colors = loadColorTable("Set1").colors
    color = rng.choice(colors[0:5] + colors[6:-1])  # avoid yellow and gray

    for i in range(len(lines)):
        if EWs[i] > 0.1:
            break
    label1 = r"%s EW$_0$ = %.2f$\AA$ (SNR = %d)" % (lines[i].replace("_", " "), EWs[i], SNR)
    label2 = r"%s (#%d)" % (config, ind)

    # plotting two panels (zoomed and full), or just one (zoomed)?
    color1, color2, _, _ = getWhiteBlackColors(pStyle)

    if full:
        fig, (ax2, ax1) = plt.subplots(nrows=2, figsize=[figsize[0] * 0.8, figsize[1] * 1.0], facecolor=color1)
        for ax in (ax1, ax2):
            setAxisColors(ax, color2, color1)
    else:
        fig, ax1 = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 0.6], facecolor=color1)
        setAxisColors(ax1, color2, color1)

    # zoom: axis ranges
    ax1.set_xlim(xlim)
    if flux_dv.min() < 0.5:
        ylim = [-0.05, 1.01]
    else:
        ylim = [np.floor((flux_dv.min() - 0.05) * 10) / 10 - 0.01, 1.01]
    if SNR_bounds is not None:
        # ylim[1] += 1/SNR
        ylim[1] = np.max([1.0, flux_dv.max() + 0.01])
    if pStyle == "black":
        # movie
        ylim = [-0.05, 1.01]
        ax2.set_ylim(ylim)

    ax1.set_ylim(ylim)
    ax1.set_xlabel(r"$\Delta v$ [ km/s ]")
    ax1.set_ylabel(ylabel)

    # full: axis ranges
    if full:
        xlim = [wave.min(), wave.max()]
        ax2.set_xlim(xlim)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel("Wavelength [ Ang ]")
        ax2.plot(wave, flux, "-", color=color)

    ax1.step(wave_dv, flux_dv, "-", color=color, where="mid")

    ax = ax2 if full else ax1  # for annotations and title

    ax.set_title(title)
    ax.annotate(label1, xy=(0.02, 0.04), xycoords="axes fraction", fontsize=18, color="#555")

    ax.annotate(
        label2, xy=(0.98, 0.06), xycoords="axes fraction", ha="right", fontsize=14, color="#aaa", alpha=0.5, zorder=-10
    )

    # finish plot
    if saveFilename is None:
        saveFilename = "spectrum_%s_%d.%s" % ("_".join(params), ind, output_fmt)
    fig.savefig(saveFilename, facecolor=fig.get_facecolor())
    plt.close(fig)


def EW_distribution(
    sim_in,
    line="MgII 2796",
    instrument="SDSS-BOSS",
    redshifts=(0.5, 0.7, 1.0),
    xlim=None,
    solar=False,
    indivEWs=False,
    log=False,
    dX=False,
):
    """Plot the EW distribution (dN/dWdz) of a given absorption line.

    Args:
      sim_in (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (list[str]): specify wavelength range and resolution, must be known in `instruments` dict.
      redshifts (list[float]): list of redshifts to overplot.
      xlim (list[float]): min and max EW [Ang], or None for default.
      solar (bool): use the (constant) solar value instead of simulation-tracked metal abundances.
      indivEWs (bool): if True, then use/create absorber catalog, to handle multiple absorbers per sightline,
        otherwise use the available 'global' EWs, one per sightline.
      log (bool): plot log(EW) instead of linear EWs.
      dX (bool): if True, normalize by absorption distance dX instead of redshift path length dz.
    """
    sim = sim_in.copy()

    # plot config
    EW_min = 1e-2  # rest-frame ang

    if xlim is None:
        xlim = [0, 8]  # ang
        if log:
            xlim = [0.1, 10]  # log[ang]

    nBins = 40

    # load: loop over requested redshifts
    EWs = {}

    for redshift in redshifts:
        sim.setRedshift(redshift)

        ion = lines[line]["ion"]

        for inst in iterable(instrument):
            filepath = spectra_filepath(sim, ion, instrument=inst, solar=solar)

            if isfile(filepath):
                break

        if not isfile(filepath):
            continue

        print(filepath)

        # raw EWs (one per sightline), or re-processed EWs (one per individual absorber)?
        if indivEWs:
            EWs_orig, _, EWs_processed, _, _ = absorber_catalog(sim, ion, instrument=instrument, solar=solar)
            data = EWs_processed[line]
            count = EWs_orig[line].size
        else:
            with h5py.File(filepath, "r") as f:
                count = f.attrs["count"]
                data = f["EW_%s" % line.replace(" ", "_")][()]

        # exclude absolute zero EWs (i.e. no absorption)
        data = data[data > 0]

        # convert to rest-frame
        data /= 1 + sim.redshift

        # exclude unobservably small EWs
        data = data[data >= EW_min]

        EWs[redshift] = data

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 0.85, figsize[1] * 0.85])
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_xlabel(r"Rest-frame Equivalent Width [ $\rm{\AA}$ ]")
    if dX:
        ax.set_ylabel(r"d$^2 N$/d$X$d$W$ (%s)" % line)
    else:
        ax.set_ylabel(r"d$^2 N$/d$z$d$W$ (%s)" % line)
    ax.set_yscale("log")
    if log:
        ax.set_xscale("log")

    # loop over requested redshifts
    colors = []
    labels = []

    for redshift in EWs.keys():
        # load
        sim.setRedshift(redshift)
        x = EWs[redshift]

        # histogram
        hh, bin_edges = np.histogram(x, bins=nBins, range=xlim)

        if dX:
            # normalize by dX = absorption distance = N_sightlines * boxSizeInAbsorptionDistance
            hh = hh.astype("float32") / (count * sim.dX)
        else:
            # normalize by dz = total redshift path length = N_sightlines * boxSizeInDeltaRedshift
            hh = hh.astype("float32") / (count * sim.dz)

        # normalize by dW = equivalent width bin sizes [Ang]
        dW_norm = bin_edges[1:] - bin_edges[:-1]  # constant (linear)
        if log:
            dW_norm = 10.0 ** bin_edges[1:] - 10.0 ** bin_edges[:-1]  # variable (log)

        hh /= dW_norm

        l = ax.stairs(hh, edges=bin_edges)
        colors.append(l.get_edgecolor())
        labels.append("z = %.1f" % sim.redshift)

    # plot obs data
    if line == "MgII 2796" and not dX:
        # (Matejek+12 Table 3)
        # https://ui.adsabs.harvard.edu/abs/2012ApJ...761..112M/abstract
        # (!) see also https://ui.adsabs.harvard.edu/abs/2013ApJ...764....9M/abstract
        m13_x = [0.42, 0.94, 1.52, 2.11, 2.70, 4.34]
        m13_x_lower = [0.05, 0.64, 1.23, 1.82, 2.41, 3.00]
        m13_x_upper = [0.64, 1.23, 1.82, 2.41, 3.00, 5.68]
        m13_y = [1.570, 0.594, 0.291, 0.187, 0.083, 0.027]
        m13_yerr = [0.272, 0.119, 0.080, 0.064, 0.042, 0.011]
        m13_label = "Matejek+12 (1.9 < z < 6.3)"

        xerr = np.vstack((np.array(m13_x) - m13_x_lower, np.array(m13_x_upper) - m13_x))

        opts = {"color": "#333333", "ecolor": "#333333", "alpha": 0.6, "capsize": 0.0, "fmt": "s"}
        ax.errorbar(m13_x, m13_y, yerr=m13_yerr, xerr=xerr, label=m13_label, **opts)

        # Chen+17 (updated/finished sample of Matejek+, identical EW bins)
        # https://ui.adsabs.harvard.edu/abs/2017ApJ...850..188C/abstract
        c16_y = [1.539, 0.591, 0.298, 0.185, 0.134, 0.026]
        c16_yerr = [0.215, 0.082, 0.055, 0.042, 0.035, 0.007]
        c16_label = "Chen+17 (1.9 < z < 6.3)"

        opts = {"color": "#333333", "ecolor": "#333333", "alpha": 0.9, "capsize": 0.0, "fmt": "D"}
        ax.errorbar(m13_x, c16_y, yerr=c16_yerr, xerr=xerr, label=c16_label, **opts)

        # check: https://iopscience.iop.org/article/10.3847/1538-4357/abbb34/pdf

        # Sebastian+24 (E-XQR-30)
        s24_x = [5.15e-2, 1.03e-1, 2.0e-1, 3.95e-1, 7.75e-1, 1.51e0, 2.94e0]
        s24_y = [5.51e0, 4.12e0, 3.13e0, 1.55e0, 6.1e-1, 2.27e-1, 8.10e-2]
        s24_y0 = [4.48e0, 3.37e0, 2.80e0, 1.38e0, 5.17e-1, 1.86e-1, 6.43e-2]
        s24_yerr = np.array(s24_y) - np.array(s24_y0)
        s24_label = "Sebastian+24 (2 < z < 6)"

        opts = {"color": "#000", "ecolor": "#000", "alpha": 0.9, "capsize": 0.0, "fmt": "o"}
        ax.errorbar(s24_x, s24_y, yerr=s24_yerr, label=s24_label, **opts)

        # Zhu+13 (SDSS, fits via Table 1 and Eqn 3)
        z13_z = [0.48, 0.63, 0.78, 0.93, 1.08, 1.23, 1.38, 1.53, 1.68, 1.83, 1.98, 2.13]
        z13_N = [1.11, 1.06, 1.13, 1.25, 1.21, 1.22, 1.34, 1.33, 1.32, 1.65, 1.49, 1.55]
        z13_W = [0.51, 0.59, 0.63, 0.63, 0.68, 0.73, 0.71, 0.76, 0.76, 0.70, 0.70, 0.66]

        xx = np.linspace(0.6, 5.0, 100)

        for j, z_ind in enumerate([0, 2, 4]):
            yy = (z13_N[z_ind] / z13_W[z_ind]) * np.exp(-xx / z13_W[z_ind])  # Eqn 3
            ls = linestyles[j]
            ax.plot(xx, yy, ls=ls, color="#777", label="Zhu+13 (<z> = %.2f)" % z13_z[z_ind])

    if line == "CIV 1548" and dX:
        # Finlator+
        # Hasan, F.+ 2020
        h20_label = "Hasan+20 (1 < z < 2.5)"
        h20_W = [0.058, 0.078, 0.103, 0.131, 0.165, 0.207, 0.264, 0.330, 0.447, 0.680, 1.16, 2.12]
        h20_Werr = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.08, 0.17, 0.57, 0.52]) / 1.5
        h20_n = [14.0, 10.1, 7.9, 5.1, 4.7, 3.6, 2.8, 1.7, 1.0, 0.53, 0.24, 0.01]
        h20_nerr = [2.1, 1.3, 0.86, 0.72, 0.48, 0.40, 0.36, 0.22, 0.14, 0.05, 0.025, 0.004]

        opts = {"color": "#333333", "ecolor": "#333333", "alpha": 0.9, "capsize": 0.0, "fmt": "D"}
        ax.errorbar(h20_W, h20_n, yerr=h20_nerr, xerr=h20_Werr, label=h20_label, **opts)

        h20z_label = "Hasan+20 (2.5 < z < 4.7)"
        h20z_W = [0.058, 0.078, 0.103, 0.131, 0.165, 0.207, 0.264, 0.330, 0.45, 0.67, 1.05, 2.13]
        h20z_W_err = (
            np.vstack(
                (
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.08, 0.17, 0.3, 0.2],
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.08, 0.17, 0.5, 0.2],
                )
            )
            / 1.5
        )
        h20z_n = [12.9, 8.5, 7.2, 4.9, 3.6, 2.1, 1.9, 1.63, 0.84, 0.35, 0.06, 0.002]
        h20z_n_err = np.vstack(
            (
                [2.1, 1.3, 0.86, 0.72, 0.48, 0.40, 0.36, 0.22, 0.14, 0.05, 0.03, 0.0015],
                [2.1, 1.3, 0.86, 0.72, 0.48, 0.40, 0.36, 0.22, 0.14, 0.05, 0.05, 0.002],
            )
        )

        opts = {"color": "#000000", "ecolor": "#000000", "alpha": 0.9, "capsize": 0.0, "fmt": "o"}
        ax.errorbar(h20z_W, h20z_n, yerr=h20z_n_err, xerr=h20z_W_err, label=h20z_label, **opts)

    # simulation legend
    handles = [plt.Line2D([0], [0], color=color, ls="-") for color in colors]
    legend2 = ax.legend(handles, labels, loc="lower left")
    ax.add_artist(legend2)

    # finish plot
    ax.legend(loc="best")
    fig.savefig("EW_histogram_%s_%s%s.pdf" % (sim.simName, line.replace(" ", "-"), "_log" if log else ""))
    plt.close(fig)


def EW_vs_coldens(
    sim,
    line="CIV 1548",
    instrument="SDSS-BOSS",
    bvals=(5, 10, 25, 50, 100, 150),
    xlim=(12.5, 16.0),
    ylim=(-1.7, 0.5),
    solar=False,
    log=False,
    pSplit=None,
):
    """Plot the relationship between EW and column density, and compare to the CoG.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      bvals (list[float]): set of Doppler values to calculate and overplot CoG for.
      xlim (list[float]): min and max column density [log 1/cm^2].
      ylim (list[float]): min and max equivalent width [log Ang].
      solar (bool): use the (constant) solar value instead of simulation-tracked metal abundances.
      log (bool): plot log(EW) instead of linear EWs.
      pSplit (list[int]): if not None, then use this subset.
    """
    # load
    ion = lines[line]["ion"]
    filepath = spectra_filepath(sim, ion, instrument=instrument, solar=solar, pSplit=pSplit)

    with h5py.File(filepath, "r") as f:
        EW = f["EW_%s" % line.replace(" ", "_")][()]

    coldens = integrate_along_saved_rays(sim, "%s numdens" % ion, pSplit=pSplit)

    # convert to rest-frame
    EW /= 1 + sim.redshift

    # select and log
    with np.errstate(divide="ignore"):
        EW = np.log10(EW)
        coldens = np.log10(coldens)

        w = np.where((EW > ylim[0]) & (EW <= ylim[1]) & (coldens > xlim[0]) & (coldens <= xlim[1]))

    print("[%s] %d/%d sightlines selected." % (sim.simName, len(w[0]), len(EW)))

    EW = EW[w]
    coldens = coldens[w]

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 0.8, figsize[1] * 0.8])
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Column Density [ log cm$^{-2}$ ]")
    ax.set_ylabel(line + r" Equivalent Width [ log $\AA$ ]")

    # simulation sightlines
    min_contour = 0.1  # individual points shown outside this level
    nBins2D = [80, 50]

    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    ax.scatter(coldens, EW, marker=".", color="#333", alpha=0.8, zorder=0)

    zz, xc, yc = np.histogram2d(coldens, EW, bins=nBins2D, range=[xlim, ylim], density=True)

    zz = savgol_filter(zz.T, sKn, sKo)

    ax.contourf(
        xc[:-1],
        yc[:-1],
        zz,
        levels=[min_contour - 0.02, 100],
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors="#fff",
        zorder=1,
    )
    ax.contour(
        xc[:-1],
        yc[:-1],
        zz,
        levels=[min_contour, 0.5, 1, 2, 3],
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        colors="#333",
        zorder=2,
    )

    # overplot curve of growth
    Nvals = np.linspace(xlim[0], xlim[1], 100)

    for b in bvals:
        EWs = np.zeros(Nvals.size, dtype="float32")
        for i, N in enumerate(Nvals):
            _, EW, _ = _cog([line], N=N, b=b)
            EWs[i] = EW

        ax.plot(Nvals, np.log10(EWs), label="b = %d km/s" % b)

    # finish plot
    ax.legend(loc="upper left")
    fig.savefig(
        "EW_vs_coldens_%s_%s%s_%s.pdf" % (sim.simName, line.replace(" ", "-"), "_log" if log else "", instrument)
    )
    plt.close(fig)


def dNdz_evolution(sim_in, redshifts, line="MgII 2796", instrument="SDSS-BOSS", solar=False):
    """Plot the redshift evolution (i.e. dN/dz) and comoving incidence rate (dN/dX) of a given absorption line.

    Args:
      sim_in (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (list[str]): specify wavelength range and resolution, must be known in `instruments` dict.
      redshifts (list[float]): list of redshifts to overplot.
      solar (bool): use the (constant) solar value instead of simulation-tracked metal abundances.
      log (bool): plot log(EW) instead of linear EWs.
    """
    from ..load.data import zhu13mgii

    sim = sim_in.copy()

    # config
    z13 = zhu13mgii()
    # EW_thresholds = [0.3,1.0,3.0] # thresholds for EW for vs. redshift plot
    if "MgII" in line:
        EW_thresholds = z13["EW0"]  # match to obs data
        xlim = [0.0, 6.0]
        ylim = [5e-4, 2.0]

    if "CIV" in line:
        EW_thresholds = [0.05, 0.3, 0.6, 0.9]
        xlim = [np.min(redshifts) - 0.1, np.max(redshifts) + 0.1]
        ylim = [1e-4, 10.0]

    # load: loop over all available redshifts
    zz = []
    dNdz = {thresh: [] for thresh in EW_thresholds}
    dNdX = {thresh: [] for thresh in EW_thresholds}

    for redshift in redshifts:
        sim.setRedshift(redshift)
        ion = lines[line]["ion"]

        # check for existence across all specified instrument(s)
        for inst in iterable(instrument):
            filepath = spectra_filepath(sim, ion, instrument=inst, solar=solar)

            if isfile(filepath):
                break

        if not isfile(filepath):
            continue

        with h5py.File(filepath, "r") as f:
            print(filepath)
            count = f.attrs["count"]
            EWs = f["EW_%s" % line.replace(" ", "_")][()]

        # convert to rest-frame and store
        EWs /= 1 + sim.redshift

        # loop over requested thresholds
        for EW_thresh in EW_thresholds:
            num = len(np.where(EWs >= EW_thresh)[0])

            # normalize by dz = total redshift path length = N_sightlines * boxSizeInDeltaRedshift
            num_dz = float(num) / (count * sim.dz)

            # normalize by dX = total comoving path length
            num_dX = float(num) / (count * sim.dX)

            # store
            dNdz[EW_thresh].append(num_dz)
            dNdX[EW_thresh].append(num_dX)

        zz.append(redshift)

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 0.85, figsize[1] * 0.85])
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("d$N$/d$z$ (%s)" % line)
    ax.set_yscale("log")

    # plot the simulation dN/dz for each EW threshold
    colors = []
    for EW_thresh in EW_thresholds:
        (l,) = ax.plot(zz, dNdz[EW_thresh], "-")
        colors.append(l.get_color())

    # observational data
    if line == "MgII 2796":
        for i, EW0 in enumerate(z13["EW0"]):
            label = r"%s ($\rm{W_0 > %.1f \AA}$)" % (z13["label"], EW0)
            label = ""  # z13['label']+" (variable)" if i == 0 else ''
            typical_error = 1e-3
            ax.errorbar(
                z13["z"],
                z13["dNdz"][EW0],
                yerr=typical_error,
                color=colors[i],
                alpha=0.8,
                marker="s",
                linestyle="none",
                label=label,
            )

        # Zou+21 (z>2)
        z21_label = r"Zou+21 ($\rm{W_0 > 1.0 \AA}$)"
        z21_z = [2.6, 3.4, 4.6, 5.2, 6.1]
        z21_z_err1 = [3.0, 3.8, 5.0, 5.4, 6.2]
        z21_z_err2 = [2.2, 3.0, 4.2, 5.0, 6.0]
        z21_dNdz = [0.39, 0.67, 0.62, 0.19, 0.12]
        z21_dNdz_err1 = [0.54, 0.94, 0.89, 0.31, 0.22]
        z21_dNdz_err2 = [0.23, 0.40, 0.33, 0.07, 0.02]
        yerr1 = np.array(z21_dNdz_err1) - np.array(z21_dNdz)
        yerr2 = np.array(z21_dNdz) - np.array(z21_dNdz_err2)
        xerr1 = np.array(z21_z_err1) - np.array(z21_z)
        xerr2 = np.array(z21_z) - np.array(z21_z_err2)

        ax.errorbar(
            z21_z,
            z21_dNdz,
            yerr=np.vstack((yerr1, yerr2)),
            xerr=np.vstack((xerr1, xerr2)),
            color="#333",
            alpha=0.8,
            marker="D",
            linestyle="none",
            label=z21_label,
        )

        # Chen+17 (z>2)
        c17_label = r"Chen+17 ($\rm{W_0 > 1.0 \AA}$)"
        c17_z = [2.2, 2.7, 3.5, 4.8, 6.3]
        c17_z_err1 = [2.5, 3.0, 3.8, 5.3, 6.8]
        c17_z_err2 = [2.0, 2.5, 3.1, 4.3, 5.7]
        c17_dNdz = [0.73, 0.49, 0.46, 0.34, 0.19]
        c17_dNdz_err1 = [0.92, 0.63, 0.57, 0.44, 0.39]
        c17_dNdz_err2 = [0.57, 0.39, 0.37, 0.25, 0.02]

        yerr1 = np.array(c17_dNdz_err1) - np.array(c17_dNdz)
        yerr2 = np.array(c17_dNdz) - np.array(c17_dNdz_err2)
        xerr1 = np.array(c17_z_err1) - np.array(c17_z)
        xerr2 = np.array(c17_z) - np.array(c17_z_err2)

        ax.errorbar(
            c17_z,
            c17_dNdz,
            yerr=np.vstack((yerr1, yerr2)),
            xerr=np.vstack((xerr1, xerr2)),
            color="#333",
            alpha=0.8,
            marker="o",
            linestyle="none",
            label=c17_label,
        )

        # https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.1023C/abstract

        # Prochter+05

    if line == "CIV":  # also SiIV
        pass  # https://ui.adsabs.harvard.edu/abs/2022ApJ...924...12H/abstract
        # https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4940C/abstract

    # second legend
    handles = []
    labels = []

    for i, EW_thresh in enumerate(EW_thresholds):
        label = r"EW > %.1f$\,\rm{\AA}$" % EW_thresh
        handles.append(plt.Line2D([0], [0], color=colors[i], ls="-"))
        labels.append(label)

    legend2 = ax.legend(handles, labels, ncols=1, loc="lower right")
    ax.add_artist(legend2)

    # finish plot
    handles, labels = ax.get_legend_handles_labels()
    lExtra = [z13["label"] + " (variable)", sim.simName]
    hExtra = [
        plt.Line2D([0], [0], color="#333", lw=0, marker="s", alpha=0.8),
        plt.Line2D([0], [0], color="#333", alpha=1.0),
    ]
    ax.legend(handles + hExtra, labels + lExtra, ncols=1, loc=[0.41, 0.02], handlelength=1.2)
    fig.savefig("dNdz_evolution_%s_%s.pdf" % (sim.simName, line.replace(" ", "-")))
    plt.close(fig)

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 0.85, figsize[1] * 0.85])
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("d$N$/d$X$ (%s)" % line)
    ax.set_yscale("log")

    for EW_thresh in EW_thresholds:
        ax.plot(zz, dNdX[EW_thresh], "-")

    # observational data
    if line == "MgII 2796":
        # Zou+21

        # Codoreanu+17 (Table 2)
        c17_label = r"Codoreanu+17 ($\rm{0.3 \AA < W_0 < 0.6 \AA}$)"
        c17_z = [2.53, 3.41, 4.76]
        c17_dNdX = [0.30, 0.09, 0.11]
        c17_dNdX_err = [0.15, 0.08, 0.07]
        ax.errorbar(
            c17_z, c17_dNdX, yerr=c17_dNdX_err, color="#333", alpha=0.8, marker="o", linestyle="none", label=c17_label
        )

        c17_label = r"Codoreanu+17 ($\rm{1.0 \AA < W_0 < 4.0 \AA}$)"
        c17_z = [2.49, 3.43, 4.75]
        c17_dNdX = [0.09, 0.07, 0.08]
        c17_dNdX_err = [0.05, 0.06, 0.06]
        ax.errorbar(
            c17_z, c17_dNdX, yerr=c17_dNdX_err, color="#333", alpha=0.8, marker="s", linestyle="none", label=c17_label
        )

    if line == "CIV 1548":
        # Hasan+2020 (Table 4, Figure 7)
        h20_label = "Hasan+ (2020) - SDSS"
        h20_z = [1.34, 1.74, 1.94, 2.13, 2.30, 2.46, 2.64, 2.83, 3.01, 4.0]
        h20_dNdX_005A = [1.97, 2.17, 2.12, 2.11, 2.03, 2.02, 1.77, 2.13, 1.39, 1.23]
        h20_dNdX_03A = [0.80, 0.76, 0.75, 0.67, 0.61, 0.56, 0.59, 0.55, 0.40, 0.24]
        h20_dNdX_06A = [0.53, 0.39, 0.36, 0.34, 0.34, 0.19, 0.26, 0.20, 0.11, 0.06]
        h20_dNdX_005A_err = [0.19, 0.21, 0.20, 0.19, 0.18, 0.18, 0.16, 0.19, 0.12, 0.11]
        h20_dNdX_03A_err = [0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.06, 0.04]
        h20_dNdX_06A_err = [0.08, 0.07, 0.07, 0.07, 0.07, 0.05, 0.06, 0.06, 0.03, 0.02]

        label = r"%s ($\rm{W_0 > 0.05 \AA}$)" % (h20_label)
        ax.errorbar(h20_z, h20_dNdX_005A, yerr=h20_dNdX_005A_err, color=colors[0], marker="s", ls="none", label=label)
        label = r"%s ($\rm{W_0 > 0.3 \AA}$)" % (h20_label)
        ax.errorbar(h20_z, h20_dNdX_03A, yerr=h20_dNdX_03A_err, color=colors[1], marker="s", ls="none", label=label)
        label = r"%s ($\rm{W_0 > 0.6 \AA}$)" % (h20_label)
        ax.errorbar(h20_z, h20_dNdX_06A, yerr=h20_dNdX_06A_err, color=colors[2], marker="s", ls="none", label=label)

        # Anand+2025 (Figure 7)
        a25_label = "Anand+ (2025) - DESI"
        a25_z = [1.54, 1.77, 1.86, 1.93, 1.98, 2.03, 2.09, 2.14, 2.20, 2.26, 2.33, 2.41, 2.51, 2.66, 2.88, 3.85]
        a25_dNdX_06A = [ 1.80e-1, 1.72e-1, 1.75e-1, 1.63e-1, 1.59e-1, 1.52e-1, 1.50e-1, 1.47e-1, 1.41e-1, 1.33e-1,
                        1.25e-1, 1.22e-1, 1.07e-1, 8.47e-2, 7.67e-2, 4.81e-2]  # fmt: skip
        a25_dNdX_09A = [ 7.79e-2, 7.07e-2, 6.81e-2, 6.68e-2, 6.60e-2, 6.14e-2, 5.72e-2, 5.86e-2, 5.45e-2, 5.34e-2,
                        4.83e-2, 4.54e-2, 4.12e-2, 3.21e-2, 2.80e-2, 1.71e-2]  # fmt: skip

        label = r"%s ($\rm{W_0 > 0.6 \AA}$)" % (a25_label)
        ax.errorbar(a25_z, a25_dNdX_06A, color=colors[2], marker="o", ls="none", label=label)
        label = r"%s ($\rm{W_0 > 0.9 \AA}$)" % (a25_label)
        ax.errorbar(a25_z, a25_dNdX_09A, color=colors[3], marker="o", ls="none", label=label)

        # https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4940C/abstract

        # D'odorico+22 (from +13,+10)
        # d13_label = r'D\'Odorico+13 ($\rm{log N \ge 13.0 cm$^{-2}}$)'
        # d13_z = [2.1, 2.66, 3.44, 4.78, 5.18, 5.89]
        # d13_z_err1 = [0.3, 0.285, 1.0, 0.16, 0.46, 0.33]
        # d13_z_err2 = [0.34, 0.25, 0.5, 0.26, 0.22, 0.18]
        # d13_dNdX = [3.92, 4.22, 4.26, 4.36, 1.73, 0.86]
        # d13_dNdX_err1 = [0.33, 0.43, 0.42, 0.55, 0.24, 0.21]
        # d13_dNdX_err2 = [0.33, 0.42, 0.40, 0.58, 0.26, 0.21]

        # ax.errorbar(d13_z, d13_dNdX, yerr=np.vstack((d13_dNdX_err1,d13_dNdX_err2)),
        #            xerr=np.vstack((d13_z_err1,d13_z_err2)),
        #            color='#333', alpha=0.8, marker='o', linestyle='none', label=d13_label)

    # legends
    if 0:
        handles = []
        labels = []

        for i, EW_thresh in enumerate(EW_thresholds):
            label = r"%s (EW > %.1f$\,\rm{\AA}$)" % (sim.simName, EW_thresh)
            if EW_thresh < 0.1:
                label = r"%s (EW > %.2f$\,\rm{\AA}$)" % (sim.simName, EW_thresh)
            handles.append(plt.Line2D([0], [0], color=colors[i], ls="-"))
            labels.append(label)
    else:
        handles = [plt.Line2D([0], [0], color="black", ls="-")]
        labels = [sim.simName]

    legend2 = ax.legend(handles, labels, loc="upper right")
    ax.add_artist(legend2)

    ax.legend(loc="lower left")

    # finish plot
    fig.savefig("dNdX_evolution_%s_%s.pdf" % (sim.simName, line.replace(" ", "-")))
    plt.close(fig)


def n_cloud_distribution(sim, ion="Mg II", redshifts=(0.5, 0.7)):
    """Plot the N_cloud distribution of a given ion based on a rays statistics file.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      ion (str): the ion species.
      redshifts (list[float]): list of redshifts to overplot.
    """
    # config
    xlim = [0, 20]
    nBins = xlim[1]

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_xlabel("Number of Clouds per Sightline")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    # loop over requested redshifts
    for redshift in redshifts:
        # load
        saveFilename = spectra_filepath(sim, ion).replace("integral_", "stats_").replace("_combined", "")

        with h5py.File(saveFilename, "r") as f:
            n_clouds = f["n_clouds"][()]

        # histogram
        w = np.where(n_clouds > 0)
        print(f"At z = {redshift:0.1f} [{len(w[0])}] of [{n_clouds.size}] sightlines have at least one cloud.")
        hh, bin_edges = np.histogram(n_clouds[w], bins=nBins, range=xlim)

        ax.stairs(hh, edges=bin_edges, label="z = %.1f" % redshift)

    # finish plot
    ax.legend(loc="best")
    fig.savefig("N_clouds_histogram_%s.pdf" % ion.replace(" ", "_"))
    plt.close(fig)


def n_clouds_vs_EW(sim):
    """Plot relationship between number of discrete clouds along a sightline, and the EW of the transition.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
    """
    # config
    ion = "Mg II"
    instrument = "4MOST-HRS"

    xlog = False

    # load n_clouds
    saveFilename = spectra_filepath(sim, ion).replace("integral_", "stats_").replace("_combined", "")

    with h5py.File(saveFilename, "r") as f:
        n_clouds = f["n_clouds"][()]

    # load EWs
    filepath = spectra_filepath(sim, ion, instrument=instrument)

    EWs = {}
    inds = {}

    with h5py.File(filepath, "r") as f:
        for key in f:
            if "EW_" in key:
                EWs[key] = f[key][()]
        for pSplitInd in f["inds"]:
            inds[pSplitInd] = f["inds"][pSplitInd][()]

    # create subset of n_clouds matching measured EWs (concatenated spectra above threshold only)
    n_clouds = n_clouds[inds["global"]]

    # plot
    fig, ax = plt.subplots()

    ax.set_xlim([1e-2, 1e1] if xlog else [0, 10])
    ax.set_xlabel("Equivalent Width [ Ang ]")
    ax.set_ylabel("Number of Clouds per Sightline")
    if xlog:
        ax.set_xscale("log")

    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    ax.scatter(EWs["EW_total"], n_clouds, marker=".", zorder=0, label="z = %.1f" % sim.redshift)

    # finish plot
    ax.legend(loc="upper right")
    fig.savefig("N_clouds_vs_EW_%s.pdf" % ion.replace(" ", "_"))
    plt.close(fig)
