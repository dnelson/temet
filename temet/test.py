"""
Temporary stuff.
"""

import glob
import time
from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from illustris_python.util import partTypeNum
from matplotlib.backends.backend_pdf import PdfPages

from .plot.config import figsize
from .util import simParams
from .util.match import match


def check_spec():
    """Check OVI COS vs idealized spectra."""
    from temet.spectra.util import _equiv_width

    path = "/u/dnelson/sims.TNG/TNG50-1/postprocessing/AbsorptionSpectra/"
    file1 = "spectra_TNG50-1_z0.1_n1000d2-fullbox_COS-G130M_OVI_combined.hdf5"

    path2 = "/u/dnelson/sims.TNG/TNG50-1/data.files/spectra/"
    file2 = "spectra_TNG50-1_z0.1_n1000d2-fullbox_idealized_OVI_combined.hdf5"

    index = 10910

    # load
    with h5py.File(path + file1, "r") as f:
        wave_cos = f["wave"][()]
        tau_1031_cos = f["tau_OVI_1031"][index, :]
        tau_1037_cos = f["tau_OVI_1037"][index, :]
        print("ray_pos = ", f["ray_pos"][index])
    with h5py.File(path2 + file2, "r") as f:
        wave_ideal = f["wave"][()]
        tau_1031_ideal = f["tau_OVI_1031"][index, :]
        tau_1037_ideal = f["tau_OVI_1037"][index, :]

    # 1031
    wave_min = 1141
    wave_max = 1147

    w_cos = np.where((wave_cos >= wave_min) & (wave_cos <= wave_max))[0]
    w_ideal = np.where((wave_ideal >= wave_min) & (wave_ideal <= wave_max))[0]

    ew_cos_1031 = _equiv_width(tau_1031_cos[w_cos], wave_cos[w_cos])
    ew_ideal_1031 = _equiv_width(tau_1031_ideal[w_ideal], wave_ideal[w_ideal])

    print(f"EW COS 1031: {ew_cos_1031:.3f} A")
    print(f"EW Ideal 1031: {ew_ideal_1031:.3f} A")

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("TNG50-1 z=0.1 OVI Absorption Spectra: COS vs Idealized")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("Optical Depth tau")

    ax.plot(wave_cos[w_cos], tau_1031_cos[w_cos], "-", lw=2, label="COS 1031")
    ax.plot(wave_ideal[w_ideal], tau_1031_ideal[w_ideal], "-", lw=2, label="Ideal 1031")

    ax.legend(loc="upper right")
    fig.savefig("tng50_ovi_cos_vs_idealized_1031.pdf")

    # 1037
    wave_min = 1141 + 6
    wave_max = 1147 + 6

    w_cos = np.where((wave_cos >= wave_min) & (wave_cos <= wave_max))[0]
    w_ideal = np.where((wave_ideal >= wave_min) & (wave_ideal <= wave_max))[0]

    ew_cos_1037 = _equiv_width(tau_1037_cos[w_cos], wave_cos[w_cos])
    ew_ideal_1037 = _equiv_width(tau_1037_ideal[w_ideal], wave_ideal[w_ideal])

    print(f"EW COS 1037: {ew_cos_1037:.3f} A")
    print(f"EW Ideal 1037: {ew_ideal_1037:.3f} A")

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("TNG50-1 z=0.1 OVI Absorption Spectra: COS vs Idealized")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("Optical Depth tau")

    ax.plot(wave_cos[w_cos], tau_1037_cos[w_cos], "-", lw=2, label="COS 1037")
    ax.plot(wave_ideal[w_ideal], tau_1037_ideal[w_ideal], "-", lw=2, label="Ideal 1037")

    ax.legend(loc="upper right")
    fig.savefig("tng50_ovi_cos_vs_idealized_1037.pdf")


def plot_wofz():
    """Test wofz complex function."""
    from .spectra.util import wofz_complex_fn_realpart
    from .util.helper import faddeeva985

    u = np.linspace(-15, 15, 200)

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]:
        w = np.zeros(u.size, dtype="float32")
        w2 = np.zeros(u.size, dtype="float32")
        for i in range(u.size):
            w[i] = wofz_complex_fn_realpart(u[i], alpha)
            w2[i] = faddeeva985(u[i], alpha)

        (l,) = ax.plot(u, w, "--", lw=2, label="alpha = %.2f" % alpha)
        ax.plot(u, w2, ":", lw=2, color=l.get_color(), label="alpha = %.2f (985)" % alpha)
        err = np.max(np.abs(w - w2))
        print(alpha, w[0], w2[0], err)

    ax.set_xlabel("u")
    ax.set_ylabel("wofz(u)")
    ax.set_title("wofz Complex Function Real Part")
    ax.legend(loc="upper right")

    fig.savefig("wofz_test.pdf")
    plt.close(fig)

    # plot B
    fig, ax = plt.subplots(figsize=figsize)

    alpha = np.linspace(-4.0, 0.0, 200)
    alpha = 10.0**alpha

    for u in [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0]:
        w = np.zeros(alpha.size, dtype="float32")
        w2 = np.zeros(alpha.size, dtype="float32")
        for i in range(alpha.size):
            w[i] = wofz_complex_fn_realpart(u, alpha[i])
            w2[i] = faddeeva985(u, alpha[i])

        (l,) = ax.plot(np.log10(alpha), w, lw=2, label="u = %.2f" % u)
        ax.plot(np.log10(alpha), w2, ":", lw=2, color=ax.lines[-1].get_color(), label="u = %.2f (985)" % u)

        err = np.max(np.abs(w - w2))
        print(u, w[0], w2[0], err)

    # ax.set_xscale('log')
    ax.set_xlabel("log alpha")
    ax.set_ylabel("wofz(u)")
    ax.set_title("wofz Complex Function Real Part")
    ax.legend(loc="upper right")
    fig.savefig("wofz_test2.pdf")
    plt.close(fig)


def daniela_accretion_angles():
    """Combine three auxCat's and compute final angle."""
    sim = simParams(run="tng50-1", redshift=2.0)
    files = ["Subhalo_CGM_Inflow_Mean%s_%03d.hdf5" % (xyz, sim.snap) for xyz in ["X", "Y", "Z"]]

    # open output file
    outFile = files[0].replace("MeanX", "Angle_%s" % sim.simName)
    fOut = h5py.File(outFile, "w")

    # copy existing files
    for file in files:
        print(file)
        with h5py.File(sim.derivPath + "auxCat/" + file, "r") as f:
            for key in f:
                print(key)
                if key in fOut:
                    assert np.array_equal(f[key][()], fOut[key][()])
                else:
                    fOut[key] = f[key][()]

                for attr in f[key].attrs:
                    # print(attr)
                    fOut[key].attrs[attr] = f[key].attrs[attr]

    # load subhalo positions
    print("angle")
    subhalo_pos = sim.subhalos("SubhaloPos")

    # compute angle
    mean_x = fOut["Subhalo_CGM_Inflow_MeanX"][()]
    mean_y = fOut["Subhalo_CGM_Inflow_MeanY"][()]
    mean_z = fOut["Subhalo_CGM_Inflow_MeanZ"][()]

    angle = np.zeros((sim.numSubhalos, 3), dtype="float32")

    angle[:, 0] = mean_x - subhalo_pos[:, 0]
    angle[:, 1] = mean_y - subhalo_pos[:, 1]
    angle[:, 2] = mean_z - subhalo_pos[:, 2]

    sim.correctPeriodicDistVecs(angle)

    # normalize to unit vector and save
    norm = np.linalg.norm(angle, 2, axis=1)
    for i in range(3):
        angle[:, i] /= norm

    fOut["Subhalo_CGM_Inflow_Angle"] = angle

    fOut.close()

    print("Done.")


def varsha_nhi_metal():
    """Plot N_HI vs Z. Note: Z is the mass weighted mean along each LoS, not N_HI weighted!"""
    from .util.helper import running_median

    path = "/u/dnelson/sims.TNG/TNG100-1/data.files/auxCat/"
    f1 = "Box_Grid_nHI_025.hdf5"
    f2 = "Box_Grid_Z_025.hdf5"

    with h5py.File(path + f1, "r") as f:
        nhi = f["Box_Grid_nHI"][()].ravel()

    with h5py.File(path + f2, "r") as f:
        Z = f["Box_Grid_Z"][()].ravel()

    w = np.where(nhi > 18.0)[0]

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("TNG100-1 z=0")
    ax.set_xlabel("N$_{\\rm HI}$ [log cm$^{-2}$]")
    ax.set_ylabel("Metallicity [log Z$_{\\rm sun}$]")

    xm, ymm, sm, ym = running_median(nhi[w], Z[w], percs=[16, 50, 84], binSize=0.1)
    (l,) = ax.plot(xm, ym[1, :], "-", lw=3.0, alpha=0.7)
    ax.plot(xm, ym[0, :], ":", lw=3.0, color=l.get_color(), alpha=0.5)
    ax.plot(xm, ym[2, :], ":", lw=3.0, color=l.get_color(), alpha=0.5)

    fig.savefig("nhi_vs_Z_TNG100.pdf")
    plt.close(fig)


def simba_caesar_smf():
    """Compare Simba SMFs from the original (caesar) catalogs."""
    runs = ["m25n512", "m50n512", "m100n1024"]
    snap = 151

    # load
    mstar = {}
    for run in runs:
        print(run)
        with h5py.File("%s_%03d.hdf5" % (run, snap), "r") as f:
            if "masses.star_30kpc" in f["galaxy_data/dicts"]:
                key = "galaxy_data/dicts/masses.star_30kpc"
            else:
                key = "galaxy_data/dicts/masses.stellar_30kpc"

            mstar[run] = f[key][()]
            # print(f[key].attrs['unit'])

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Simba SMFs: Original Caesar Catalogs")
    ax.set_xlabel("Galaxy Stellar Mass [log Msun]")
    ax.set_ylabel("N / Mpc$^3$")
    ax.set_yscale("log")

    for run in runs:
        mass = np.log10(mstar[run])  # log msun

        h, bins = np.histogram(mass, bins=25, range=[7.5, 12.5])

        bins = bins[:-1] + 0.1  # mids
        boxsize = float(run.split("n")[0][1:])  # Mpc
        print(run, mass.min(), mass.max(), boxsize)
        h = h.astype("float32") / boxsize**3

        ax.plot(bins, h, "o-", lw=2, label=run)

    ax.legend(loc="best")
    fig.savefig("simba_smf.pdf")
    plt.close(fig)


def filter_list():
    """Check algorithm to remove elements with a given value from a list, keeping rest in order."""
    # setup
    cur_ind = 3
    xx = np.array([3, 0, 1, 2, 5, 3, -1, -1, -1, -1])

    # run
    print(xx)
    num_inds = len(np.where(xx != -1)[0])
    print(f"{num_inds = }")

    i = 0
    while i < num_inds:
        if xx[i] == cur_ind:
            for j in range(i, num_inds):
                xx[j] = xx[j + 1]
            num_inds -= 1
            continue
        i += 1

    print(xx)
    print(f"{num_inds = }")


def reorder_tracer_tracks(name="temp"):
    """Reshape from (20,nTr) to (Ntr,20)."""
    print(name)
    fname_old = f"tr_all_groups_99_{name}.hdf5"
    fname_new = f"tr_all_groups_99_{name}_new.hdf5"

    fold = h5py.File(fname_old, "r")
    fnew = h5py.File(fname_new, "w")

    fnew["redshifts"] = fold["redshifts"][()]
    fnew["snaps"] = fold["snaps"][()]

    # make dataset
    shape = np.roll(fold[name].shape, -1)  # (nSnaps,nTr) or (nSnaps,nTr,3) to (nTr,nSnaps) or (nTr,3,nSnaps)
    dset = fnew.create_dataset(name, shape, fold[name].dtype)

    for i in range(shape[-1]):
        print(i)
        if len(shape) == 2:
            # (nSnaps,nTr) -> (nTr,nSnaps)
            dset[..., i] = fold[name][i, ...]
        else:
            # (nSnaps,nTr,3) -> (nTr,3,nSnaps)
            for j in range(3):
                dset[..., j, i] = fold[name][i, ..., j]

    fold.close()
    fnew.close()
    print("Done.")


def hbt_check():
    """Check SubfindHBT."""
    sP = simParams(run="tng100-2", snap=50)

    hbtPath = sP.postPath + "SubfindHBT/subfind_hbt_%03d.hdf5" % sP.snap

    gNames = ["Group", "Matching", "SnapIndices", "Subhalo"]
    data = {}

    # load
    with h5py.File(hbtPath, "r") as f:
        for gName in gNames:
            data[gName] = {}
            for key in f[gName]:
                if key == "SubLinkHBT":  # nested group
                    continue
                print(gName, key)
                data[gName][key] = f[gName][key][()]

    # test matched properties, and property recovery from particles using indices
    ptNum = 0  # 4

    for haloID in [0, 100, 101, 102, 500, 5000, 50000, 120000]:
        offset = data["Group"]["GroupOffsetType"][haloID, ptNum]
        length = data["Group"]["GroupLenType"][haloID, ptNum]
        indices = data["SnapIndices"]["PartType%d" % ptNum][offset : offset + length]

        if length == 0:
            continue

        massType = data["Group"]["GroupMassType"][haloID, ptNum]
        pos = data["Group"]["GroupPos"][haloID]  # of minimum pot particle
        vel = data["Group"]["GroupVel"][haloID]  # mass-weighted mean over all types
        print(haloID, indices.shape, indices.min(), indices.max(), massType, pos, vel)

        # match
        matchID = data["Matching"]["GroupHBTToOrig"][haloID]
        halo = sP.halo(matchID)
        print(
            "match",
            matchID,
            halo["GroupLenType"][ptNum],
            halo["GroupMassType"][ptNum],
            halo["GroupPos"],
            halo["GroupVel"],
        )

        # particle data
        masses = sP.snapshotSubset(ptNum, "mass", inds=indices)
        print("particle mass sum", np.sum(masses, dtype="float32"))

    print("\nsubhalos:\n")

    for subID in [0, 14, 12345, 50000, 80000]:
        offset = data["Subhalo"]["SubhaloOffsetType"][subID, ptNum]
        length = data["Subhalo"]["SubhaloLenType"][subID, ptNum]
        indices = data["SnapIndices"]["PartType%d" % ptNum][offset : offset + length]

        if length == 0:
            continue

        massType = data["Subhalo"]["SubhaloMassType"][subID, ptNum]
        pos = data["Subhalo"]["SubhaloPos"][subID]  # of minimum pot particle
        vel = data["Subhalo"]["SubhaloVel"][subID]  # mass-weighted mean over all types
        print(subID, indices.shape, indices.min(), indices.max(), massType, pos, vel)

        # match
        matchID = data["Matching"]["SubhaloHBTToOrig"][subID]
        sub = sP.subhalo(matchID)
        print(
            "match",
            matchID,
            sub["SubhaloLenType"][ptNum],
            sub["SubhaloMassType"][ptNum],
            sub["SubhaloPos"],
            sub["SubhaloVel"],
        )

        # particle data
        masses = sP.snapshotSubset(ptNum, "mass", inds=indices)
        print("particle mass sum", np.sum(masses, dtype="float32"))

    import pdb

    pdb.set_trace()


def marc_sigma1():
    """Test."""
    from .plot import subhalos

    sP = simParams(run="tng300-1", redshift=0.0)

    xQuant = "sigma1kpc_stars"
    yQuant = "BH_mass"
    cQuant = "ssfr"

    xlim = [7.8, 11.0]
    ylim = None
    clim = [-2.5, -1.5]
    ctName = "seismic_r"
    maxPointsPerDex = 1000

    qRestrictions = [["mstar_30pkpc_log", 10.2, np.inf]]

    subhalos.median(
        sP,
        pdf=None,
        yQuants=[yQuant],
        xQuant=xQuant,
        xlim=xlim,
        ylim=ylim,
        scatterColor=cQuant,
        scatterPoints=True,
        markersize=30,
        qRestrictions=qRestrictions,
        clim=clim,
        ctName=ctName,
        maxPointsPerDex=maxPointsPerDex,
    )


def exportBoxGrids(sP, partType="dm", partField="mass", nCells=(64, 128, 256, 512)):
    """Export 3D uniform Cartesian grids, of different resolutions."""
    from .util.helper import logZeroSafe
    from .util.sphMap import sphGridWholeBox

    # config
    label, limits, takeLog = sP.simParticleQuantity(partType, partField)

    # load
    pos = sP.snapshotSubsetP(partType, "pos")
    mass = sP.snapshotSubsetP(partType, "mass") if partType != "dm" else sP.dmParticleMass
    hsml = sP.snapshotSubset(partType, "hsml")

    quant = None  # grid mass
    if partField != "mass":  # grid a different, mass-weighted quantity
        quant = sP.snapshotSubsetP(partType, partField)
        assert partField != "dens"  # do mass instead

    # make series of grids at progressively better resolution
    for nCell in nCells:
        grid = sphGridWholeBox(sP, pos, hsml, mass, quant, nCells=nCell)

        if partField == "mass":  # unit conversion
            pxVol = (sP.boxSize / nCell) ** 3  # code units (ckpc/h)^3
            grid = sP.units.codeDensToPhys(grid / pxVol) * 1e10  # Msun/kpc^3

        if takeLog:
            grid = logZeroSafe(grid)

        filename = "grid_%s_%d_%s_%s_%d.hdf5" % (sP.simName, sP.snap, partType, partField, nCell)

        with h5py.File(filename, "w") as f:
            f["grid"] = grid

        print("Saved: [%s]" % filename)


def bh_lum_agnrad_check():
    """Check that PartType0/GFM_AGNRadiation field/units are consistent with derived BH luminosities."""
    sP = simParams(run="tng50-1", redshift=3.0)
    haloID = 410

    subID = sP.halo(haloID)["GroupFirstSub"]

    # find BH
    # bh_subIDs = sP.bhs("subhalo_id", subhaloID=subID)

    bh_lum = sP.bhs("bh_bollum_basic_obscured", subhaloID=subID)  # erg/s

    # get a single gas cell, compute distance
    gas_rad = sP.snapshotSubset("gas", "rad_kpc", subhaloID=subID)

    w = np.where((gas_rad > 11) & (gas_rad < 12))[0][0]  # pick one

    rad_cm = gas_rad[w] * sP.units.UnitLength_in_cm  # physical cm

    # compute BH lum from PartType5 vs that inferred from the gas cell
    gas_agnrad = sP.gas("GFM_AGNRadiation", subhaloID=subID)[w]  # physical erg/s/cm^2

    gas_agnrad_lum = gas_agnrad * rad_cm**2  # erg/s

    print(bh_lum, gas_agnrad_lum, bh_lum / gas_agnrad_lum)  # ratio is 1.006


def check_zoom_variations():
    """Check TNG-Cluster tests: h30, h50, h3232, h3693."""
    redshift = 0.0

    # hInd : [variants] pairs
    sets = {
        3232: ["sf3", "sf3_n128", "sf3_n256", "sf3_n512"],  # L14* running
        3693: ["sf3", "sf3_m", "sf3s", "sf3s5008", "sf3none_kpc", "sf3none_m", "sf3s_kpc", "sf3_kpc", "L14_sf3"],
        50: ["sf2_n160", "sf2_n160s", "sf2_n160s_mpc", "sf2_n320", "sf2_n320s", "sf3"],
    }  # sf3_m running raven

    # hInd = 3693 we have the following:
    # sf3 = fiducial (mpc)
    # sf3s = only STEEPER (mpc)
    # sf3s5008 = only STEEPER and 5008 (mpc)
    # sf3none = nobugfixes (mpc) (only made it to z~3.1) (currently skipping here)
    # sf3none_kpc = nobugfixes (kpc)
    # sf3s_kpc (kpc, steeper only, no other bugfixes)
    # sf3_kpc = fiducial (kpc)
    # sf3none_m = nobugfixes, updated MaxSfrTimescale (mpc)
    # sf3_m = new fiducial (mpc), updated MaxSfrTimescale
    # L14_sf3 = new fiducial, finished on draco n640 (L14 is 5.4x L13 to z=0, n640 vs 483, draco vs raven)

    # hInd = 50 we have variations of CPU numbers, with and without steeper 's'
    #  - CPU numbers 160,320,fiducial run on draco/cobra/raven
    # sf3_m = fiducial (mpc), updated MaxSfrTimescale (125k cpuh on raven n384 to z=1 vs 83k for orig sf3 on n320)

    # hInd = 30 we have variations of CPU numbers (all L13) (partial hawk runs only)
    #  - sf3 fiducial: cobra run, 320 cores: Step 18343, Time: 0.2032 cum 114669.73  = 10.19khr
    #  - n256 hawk (updated MaxSfrTimescale): Step 28647, Time: 0.2032 cum 221221.09 = 15.73khr
    #  - n512 hawk (updated MaxSfrTimescale): Step 30756, Time: 0.2032 cum 131633.84 = 18.72khr
    #  - n1024 hawk (updated MaxSfrTimescale): Step 31778, Time: 0.2032 cum 91007.31 = 25.89khr

    # hInd = 3232 we have L13 and L14, variations of CPU numbers, all run on hawk (all new setup)
    #   L13_sf3_n128,256,512 (done on hawk)
    #    - n512:  Time: 0.2457 cum 25317.25 =  3.60khr, Time: 0.3100 cum 43854.43 =  6.24khr
    #   L14_sf3_n512,1024,2048 (partial on hawk):
    #    - n512:  Time: 0.2457 cum 71818.84 = 10.21khr, Time: 0.3100 cum 146728.38 = 20.87khr (3.4x L13)
    #    - n1024: Time: 0.2457 cum 46420.57 = 13.20khr, Time: 0.3100 cum 100744.45 = 28.66khr
    #    - n2048: Time: 0.2457 cum 73416.33 = 41.77khr...

    # current output:
    # 3232           sf3 mstar = [12.52 12.90] mbh = [10.33] size = [ 24.89] cpuKHours = 38.1
    # 3232      sf3_n128 mstar = [12.23 12.62] mbh = [10.34] size = [ 50.07] cpuKHours = 22.3
    # 3232      sf3_n256 mstar = [12.16 12.59] mbh = [10.36] size = [ 49.60] cpuKHours = 28.3
    # 3232      sf3_n512 mstar = [12.23 12.62] mbh = [10.35] size = [ 48.99] cpuKHours = 33.7
    # 3693           sf3 mstar = [12.55 12.92] mbh = [10.21] size = [ 24.08] cpuKHours = 31.9
    # 3693         sf3_m mstar = [11.89 12.55] mbh = [ 9.90] size = [ 79.76] cpuKHours = 37.1
    # 3693          sf3s mstar = [12.52 12.91] mbh = [10.19] size = [ 19.31] cpuKHours = 29.0
    # 3693      sf3s5008 mstar = [12.53 12.92] mbh = [10.21] size = [ 18.79] cpuKHours = 32.1
    # 3693   sf3none_kpc mstar = [11.91 12.51] mbh = [10.02] size = [ 85.34] cpuKHours = 51.4
    # 3693     sf3none_m mstar = [11.82 12.54] mbh = [ 9.63] size = [ 71.61] cpuKHours = 57.7
    # 3693      sf3s_kpc mstar = [11.89 12.59] mbh = [ 9.60] size = [ 73.95] cpuKHours = 40.9
    # 3693       sf3_kpc mstar = [11.88 12.49] mbh = [10.05] size = [ 62.66] cpuKHours = 37.0
    # 3693       L14_sf3 mstar = [11.93 12.61] mbh = [ 9.52] size = [ 48.11] cpuKHours = 205.1
    #   50      sf2_n160 mstar = [12.61 13.27] mbh = [11.12] size = [297.10] cpuKHours = 269.7
    #   50     sf2_n160s mstar = [12.64 13.33] mbh = [ 9.74] size = [250.89] cpuKHours = 154.1
    #   50 sf2_n160s_mpc mstar = [12.97 13.54] mbh = [ 9.84] size = [135.37] cpuKHours = 107.4
    #   50      sf2_n320 mstar = [12.58 13.26] mbh = [11.12] size = [255.85] cpuKHours = 346.6
    #   50     sf2_n320s mstar = [12.68 13.32] mbh = [ 9.79] size = [222.44] cpuKHours = 185.5
    #   50           sf3 mstar = [12.87 13.55] mbh = [ 9.66] size = [177.39] cpuKHours = 202.4

    for hInd, variants in sets.items():
        for variant_orig in variants:
            res = 14 if "L14_" in variant_orig else 13
            variant = variant_orig.replace("L14_", "")

            sP = simParams(run="tng_zoom", res=res, hInd=hInd, variant=variant, redshift=redshift)

            sub = sP.subhalo(0)
            group = sP.halo(0)

            mass_stellar = sP.units.codeMassToLogMsun(sub["SubhaloMassInRadType"][4])
            mass_bh = sP.units.codeMassToLogMsun(sub["SubhaloMassType"][5])
            mass_halostar = sP.units.codeMassToLogMsun(group["GroupMassType"][4])
            size = sP.units.codeLengthToKpc(sub["SubhaloHalfmassRadType"][4])

            print(
                "%4s %13s mstar = [%5.2f %5.2f] mbh = [%5.2f] size = [%6.2f] cpuKHours = %.1f"
                % (hInd, variant_orig, mass_stellar, mass_halostar, mass_bh, size, sP.cpuHours / 1000)
            )

    # current size of all L13 zooms: ~85TB, size of z=0 virtual snap: 2.0TB
    # projected size of L14: ~195TB in total, z=0 snapshot: 4.6TB (almost same as TNG300-1)


def minify_gergo_hydrogen_files():
    """Rewrite Gergo's hydrogen catalog files to avoid unneeded fields."""
    basePath = "/u/dnelson/sims.TNG/TNG100-1/postprocessing/hydrogen/"
    outPath = "/u/dnelson/sims.TNG/TNG100-1/data.files/"
    fields = ["MH", "MH2BR", "MH2KMT", "MH2GK"]
    # fields = ['MH','MH2BR','MH2KMT','MH2GK','MHIBR','MHIKMT','MHIGK'] # FULL!

    for i in range(100):
        fOut = h5py.File(outPath + "gas_%03d.hdf5" % i, "w")

        # open input file and rewrite
        with h5py.File(basePath + "gas_%03d.hdf5" % i, "r") as fIn:
            fOut.create_group("Header")
            for attr in fIn["Header"].attrs:
                fOut["Header"].attrs[attr] = fIn["Header"].attrs[attr]

            for field in fields:
                print(i, field)
                fOut.create_dataset(field, data=fIn[field][()], compression="gzip")
                # fOut[field] = fIn[field][()]

        fOut.close()


def half_Kband_radii():
    """Test for Hannah's paper."""
    sP = simParams(run="tng50-1", redshift=2.0)
    from .catalog.common import findHalfLightRadius

    subhaloIDs = [25821, 39745, 55106, 60750, 79350, 92271, 99303]

    mstar = sP.subhalos("mstar_30pkpc_log")
    subhaloIDs = np.where((mstar > 9.5) & (mstar < 9.51))[0]

    import pdb

    pdb.set_trace()

    for subhaloID in subhaloIDs:
        # groupcat vals
        rhalf_mass = sP.subhalo(subhaloID)["SubhaloHalfmassRadType"]
        rhalf_mass = sP.units.codeLengthToKpc(rhalf_mass)

        # stars
        rad = sP.snapshotSubset("stars_real", "rad_kpc", subhaloID=subhaloID)
        mags = sP.snapshotSubset("stars_real", "phot_K", subhaloID=subhaloID)
        mass = sP.snapshotSubset("stars_real", "mass", subhaloID=subhaloID)

        rhalf_mass2 = findHalfLightRadius(rad, mass, mags=False)
        rhalf = findHalfLightRadius(rad, mags)

        print(subhaloID, "stars: ", rhalf_mass[4], rhalf_mass2, rhalf)

        # gas
        rad_gas = sP.snapshotSubset("gas", "rad_kpc", subhaloID=subhaloID)
        mass_gas = sP.snapshotSubset("gas", "mass", subhaloID=subhaloID)
        sfr_gas = sP.snapshotSubset("gas", "sfr", subhaloID=subhaloID)

        rhalf_gas2 = findHalfLightRadius(rad_gas, mass_gas, mags=False)

        w = np.where(sfr_gas > 0)
        rad_gas_sfr = rad_gas[w]
        mass_gas_sfr = mass_gas[w]

        rhalf_gas_sfr = findHalfLightRadius(rad_gas_sfr, mass_gas_sfr, mags=False)

        print(subhaloID, "gas: ", rhalf_mass[0], rhalf_gas2, rhalf_gas_sfr)

        # baryon (stars + SFRing gas)
        baryon_rad = np.hstack((rad, rad_gas_sfr))
        baryon_mass = np.hstack((mass, mass_gas_sfr))

        rhalf_baryon = findHalfLightRadius(baryon_rad, baryon_mass, mags=False)

        print(subhaloID, "baryon: ", rhalf_baryon, end="\n\n")


def copy_eagle_config_param_attrs(snap=28):
    """Copy snapshot metadata from original EAGLE snaps into rewritten snaps."""
    loadPath = "/virgo/simulations/EagleDM/L0100N1504/DMONLY/data/"
    savePath = "/virgo/simulations/Illustris/Eagle-L68n1504DM/output/"

    nChunks = 40  # 256

    # find path
    dirName = glob.glob(loadPath + "snapshot_%03d_*" % snap)[0].split("/")[-1]

    def fileNameLoad(chunkNum):
        zStr = dirName.split("_")[-1]
        return loadPath + dirName + "/snap_%03d_%s.%d.hdf5" % (snap, zStr, chunkNum)

    def fileNameSave(chunkNum):
        return savePath + "snapdir_%03d/snap_%03d.%d.hdf5" % (snap, snap, chunkNum)

    # load attributes
    with h5py.File(fileNameLoad(0), "r") as f:
        config = dict(f["Config"].attrs)
        params = dict(f["RuntimePars"].attrs)

    # write attributes
    for i in range(nChunks):
        print("write: ", snap, i, flush=True)
        with h5py.File(fileNameSave(i), "a") as f:
            config_eagle = f.create_group("Config_Eagle")
            params_eagle = f.create_group("Parameters_Eagle")

            for key in config:
                config_eagle.attrs[key] = config[key]
            for key in params:
                params_eagle.attrs[key] = params[key]

    print("Done.")


def rewrite_sfrs_eagle(snap=28):
    """Rewrite particle-level SFR values from original EAGLE snaps into new snaps."""
    loadPath = "/virgo/simulations/Eagle/L0100N1504/REFERENCE/data/"
    savePath = "/virgo/simulations/Illustris/Eagle-L68n1504FP/output/"

    nChunks = 256

    # find path and number of gas particles
    dirName = glob.glob(loadPath + "snapshot_%03d_*" % snap)[0].split("/")[-1]

    def fileNameLoad(chunkNum):
        zStr = dirName.split("_")[-1]
        return loadPath + dirName + "/snap_%03d_%s.%d.hdf5" % (snap, zStr, chunkNum)

    def fileNameSave(chunkNum):
        return savePath + "snapdir_%03d/snap_%03d.%d.hdf5" % (snap, snap, chunkNum)

    with h5py.File(fileNameLoad(0), "r") as f:
        nGas = f["Header"].attrs["NumPart_Total"][0]

    # load ids from EAGLE
    ids = np.zeros(nGas, dtype="int64")

    offset = 0
    for i in range(nChunks):
        print("load a: ", i, offset, flush=True)
        with h5py.File(fileNameLoad(i), "r") as f:
            ids_loc = f["PartType0"]["ParticleIDs"][()]
            ids[offset : offset + ids_loc.size] = ids_loc
            offset += ids_loc.size

    assert offset == nGas

    # load existing IDs
    ids_new = np.zeros(nGas, dtype="int64")

    offset = 0
    for i in range(nChunks):
        print("load b: ", i, offset, flush=True)
        with h5py.File(fileNameSave(i), "r") as f:
            ids_loc = f["PartType0"]["ParticleIDs"][()]
            ids_new[offset : offset + ids_loc.size] = ids_loc
            offset += ids_loc.size

    assert offset == nGas

    # cross-match
    print("Matching...", flush=True)

    inds, _ = match(ids, ids_new)
    assert inds.size == ids.size

    ids = ids[inds]
    assert np.array_equal(ids, ids_new)
    ids = None
    ids_new = None

    # load SFRs and shuffle
    sfr = np.zeros(nGas, dtype="float32")

    offset = 0
    for i in range(nChunks):
        print("load c: ", i, offset, flush=True)
        with h5py.File(fileNameLoad(i), "r") as f:
            sfr_loc = f["PartType0"]["StarFormationRate"][()]
            sfr[offset : offset + sfr_loc.size] = sfr_loc
            offset += sfr_loc.size

    assert offset == nGas

    sfr = sfr[inds]

    # now can write shuffled sfr values
    offset = 0

    for i in range(nChunks):
        print("write: ", i, offset, flush=True)
        with h5py.File(fileNameSave(i), "a") as f:
            loc_size = f["PartType0"]["StarFormationRate"].size
            f["PartType0"]["StarFormationRate"][:] = sfr[offset : offset + loc_size]
            offset += loc_size

    assert offset == sfr.size
    print("Done.")


def eagle_fix_group_sfr(snap=28):
    """Recompute GroupSFR from particles."""
    sP = simParams(run="eagle", snap=snap)

    # gc
    gc = sP.groupCat(fieldsHalos=["GroupLen", "GroupLenType"])
    gc["GroupOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]

    nGroupsTot = sP.numHalos
    haloIDsTodo = np.arange(nGroupsTot, dtype="int32")
    print(nGroupsTot)

    ptNum = 0

    # load/allocate
    sfr = sP.gas("sfr")

    GroupSFR = np.zeros(nGroupsTot, dtype="float32")

    # loop over halos
    for i, haloID in enumerate(haloIDsTodo):
        if i % int(nGroupsTot / 20) == 0 and i <= nGroupsTot:
            print("  %4.1f%%" % (float(i + 1) * 100.0 / nGroupsTot))

        # slice starting/ending indices for gas local to this FoF
        i0 = gc["GroupOffsetType"][haloID, ptNum]
        i1 = i0 + gc["GroupLenType"][haloID, ptNum]
        assert i0 >= 0

        if i1 == i0:
            continue  # zero length of this type

        GroupSFR[i] = np.sum(sfr[i0:i1])

    # update groupcat
    savePath = "/virgo/simulations/Illustris/Eagle-L68n1504FP/output/"
    nChunks = 256
    offset = 0

    for i in range(nChunks):
        print("write: ", i, offset, flush=True)

        filename = "groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap, snap, i)
        with h5py.File(savePath + filename, "a") as f:
            if "GroupSFR" not in f["Group"]:
                continue
            loc_size = f["Group"]["GroupSFR"].size
            if loc_size == 0:
                continue

            f["Group"]["GroupSFR"][:] = GroupSFR[offset : offset + loc_size]
            offset += loc_size

    assert offset == GroupSFR.size
    print("Done.")


def eagle_fix_subhalo_sfr_fields(snap=28):
    """Recompute Subhalo*Sfr fields from particles."""
    sP = simParams(run="eagle", snap=snap)

    # gc
    fields = ["SubhaloLen", "SubhaloLenType", "SubhaloPos", "SubhaloHalfmassRadType", "SubhaloVmaxRad"]
    #'SubhaloGasMetalFractionsSfr','SubhaloGasMetalFractionsSfrWeighted',
    #'SubhaloGasMetallicitySfr','SubhaloGasMetallicitySfrWeighted',
    #'SubhaloSFR','SubhaloSFRinRad','SubhaloSFRinMaxRad','SubhaloSFRinHalfRad']

    gc = sP.groupCat(fieldsSubhalos=fields)
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]
    gc["RhalfStarsSq"] = gc["SubhaloHalfmassRadType"][:, 4] ** 2
    gc["2RhalfStarsSq"] = (2.0 * gc["SubhaloHalfmassRadType"][:, 4]) ** 2
    gc["VmaxRadSq"] = gc["SubhaloVmaxRad"] ** 2

    nSubhalosTot = sP.numSubhalos
    print(nSubhalosTot)

    pt = "gas"
    ptNum = 0

    # load/allocate
    loadFields = ["Coordinates", "GFM_Metals", "GFM_Metallicity", "Masses", "StarFormationRate"]

    gas = sP.snapshotSubsetP(pt, loadFields)
    nMetals = gas["GFM_Metals"].shape[1]

    SubhaloGasMetalFractionsSfr = np.zeros((nSubhalosTot, nMetals), dtype="float32")
    SubhaloGasMetalFractionsSfrWeighted = np.zeros((nSubhalosTot, nMetals), dtype="float32")
    SubhaloGasMetallicitySfr = np.zeros(nSubhalosTot, dtype="float32")
    SubhaloGasMetallicitySfrWeighted = np.zeros(nSubhalosTot, dtype="float32")

    SubhaloSFR = np.zeros(nSubhalosTot, dtype="float32")
    SubhaloSFRinHalfRad = np.zeros(nSubhalosTot, dtype="float32")
    SubhaloSFRinMaxRad = np.zeros(nSubhalosTot, dtype="float32")
    SubhaloSFRinRad = np.zeros(nSubhalosTot, dtype="float32")

    # loop over subhalos
    for i in range(nSubhalosTot):
        if i % int(nSubhalosTot / 20) == 0 and i <= nSubhalosTot:
            print("  %4.1f%%" % (float(i + 1) * 100.0 / nSubhalosTot))

        # slice starting/ending indices for gas local to this subhalo
        i0 = gc["SubhaloOffsetType"][i, ptNum]
        i1 = i0 + gc["SubhaloLenType"][i, ptNum]
        assert i0 >= 0

        if i1 == i0:
            continue  # zero length of this type

        # distances and selections
        rr = sP.periodicDistsSq(gc["SubhaloPos"][i, :], gas["Coordinates"][i0:i1, :])

        w_2rhalf = np.where(rr < gc["2RhalfStarsSq"][i])
        w_1rhalf = np.where(rr < gc["RhalfStarsSq"][i])
        w_maxrad = np.where(rr < gc["VmaxRadSq"][i])

        w_sfr = np.where(gas["StarFormationRate"][i0:i1] > 0.0)

        # derive quantities
        GasMassSfr = np.sum(gas["Masses"][i0:i1][w_sfr])
        Sfr = np.sum(gas["StarFormationRate"][i0:i1])

        GasMassMetalsSfr = np.sum(gas["GFM_Metals"][i0:i1][w_sfr] * gas["Masses"][i0:i1, np.newaxis][w_sfr], axis=0)
        GasMassMetalsSfrWeighted = np.sum(
            gas["GFM_Metals"][i0:i1] * gas["StarFormationRate"][i0:i1, np.newaxis], axis=0
        )
        GasMassMetallicitySfr = np.sum(gas["GFM_Metallicity"][i0:i1][w_sfr] * gas["Masses"][i0:i1][w_sfr])
        GasMassMetallicitySfrWeighted = np.sum(gas["GFM_Metallicity"][i0:i1] * gas["StarFormationRate"][i0:i1])

        if GasMassSfr > 0.0:
            SubhaloGasMetalFractionsSfr[i, :] = GasMassMetalsSfr / GasMassSfr
            SubhaloGasMetallicitySfr[i] = GasMassMetallicitySfr / GasMassSfr
        if Sfr > 0.0:
            SubhaloGasMetalFractionsSfrWeighted[i, :] = GasMassMetalsSfrWeighted / Sfr
            SubhaloGasMetallicitySfrWeighted[i] = GasMassMetallicitySfrWeighted / Sfr

        SubhaloSFR[i] = np.sum(gas["StarFormationRate"][i0:i1])
        SubhaloSFRinHalfRad[i] = np.sum(gas["StarFormationRate"][i0:i1][w_1rhalf])
        SubhaloSFRinRad[i] = np.sum(gas["StarFormationRate"][i0:i1][w_2rhalf])
        SubhaloSFRinMaxRad[i] = np.sum(gas["StarFormationRate"][i0:i1][w_maxrad])

        if 0:
            print(gc["SubhaloGasMetalFractionsSfr"][i, :])
            print(SubhaloGasMetalFractionsSfr[i, :])

            print(gc["SubhaloGasMetalFractionsSfrWeighted"][i, :])
            print(SubhaloGasMetalFractionsSfrWeighted[i, :])

            print(gc["SubhaloGasMetallicitySfr"][i], SubhaloGasMetallicitySfr[i])
            print(gc["SubhaloGasMetallicitySfrWeighted"][i], SubhaloGasMetallicitySfrWeighted[i])

            print(gc["SubhaloSFR"][i], SubhaloSFR[i])
            print(gc["SubhaloSFRinHalfRad"][i], SubhaloSFRinHalfRad[i])
            print(gc["SubhaloSFRinRad"][i], SubhaloSFRinRad[i])
            print(gc["SubhaloSFRinMaxRad"][i], SubhaloSFRinMaxRad[i])

    # update groupcat
    savePath = "/virgo/simulations/Illustris/Eagle-L68n1504FP/output/"
    nChunks = 256
    offset = 0

    for i in range(nChunks):
        print("write: ", i, offset, flush=True)

        filename = "groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap, snap, i)
        with h5py.File(savePath + filename, "a") as f:
            if "SubhaloSFR" not in f["Subhalo"]:
                continue
            loc_size = f["Subhalo"]["SubhaloSFR"].size
            if loc_size == 0:
                continue

            f["Subhalo"]["SubhaloGasMetalFractionsSfr"][:] = SubhaloGasMetalFractionsSfr[offset : offset + loc_size, :]
            f["Subhalo"]["SubhaloGasMetalFractionsSfrWeighted"][:] = SubhaloGasMetalFractionsSfrWeighted[
                offset : offset + loc_size, :
            ]

            f["Subhalo"]["SubhaloGasMetallicitySfr"][:] = SubhaloGasMetallicitySfr[offset : offset + loc_size]
            f["Subhalo"]["SubhaloGasMetallicitySfrWeighted"][:] = SubhaloGasMetallicitySfrWeighted[
                offset : offset + loc_size
            ]

            f["Subhalo"]["SubhaloSFR"][:] = SubhaloSFR[offset : offset + loc_size]
            f["Subhalo"]["SubhaloSFRinHalfRad"][:] = SubhaloSFRinHalfRad[offset : offset + loc_size]
            f["Subhalo"]["SubhaloSFRinRad"][:] = SubhaloSFRinRad[offset : offset + loc_size]
            f["Subhalo"]["SubhaloSFRinMaxRad"][:] = SubhaloSFRinMaxRad[offset : offset + loc_size]
            offset += loc_size

    assert offset == SubhaloSFR.size
    print("Done.")


def compare_subhalos_all_quantities(snap=28):
    """Plot diagnostic histograms."""
    nBins = 50

    sPs = []
    sPs.append(simParams(run="eagle", snap=snap))
    sPs.append(simParams(run="tng100-1", redshift=sPs[0].redshift))  # closest matching

    for sP in sPs:
        print(sP.simName, sP.redshift, sP.snap)

    # start pdf book
    pdf = PdfPages("compare_subhalos_%s_%d.pdf" % ("-".join(sP.simName for sP in sPs), snap))

    # get list of subhalo properties
    with h5py.File(sPs[-1].gcPath(sPs[-1].snap, chunkNum=0), "r") as f:
        fields_group = list(f["Group"].keys())
        fields_sub = list(f["Subhalo"].keys())

    for field in fields_sub + fields_group:
        # start plot
        if "Bfld" in field or field == "SubhaloFlag":
            continue
        print(field)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlabel(field + " [log]")
        ax.set_ylabel("log N")

        # loop over runs
        for _i, sP in enumerate(sPs):
            # load
            if field in fields_sub:
                vals = sP.subhalos(field)
            else:
                vals = sP.halos(field)

            # histogram and plot
            vals = vals.ravel()  # 1D for all multi-D

            if field not in ["SubhaloCM", "SubhaloGrNr", "SubhaloIDMostbound"]:
                vals = np.log10(vals)
            vals = vals[np.isfinite(vals)]

            label = sP.simName + " snap=%d (z=%.1f)" % (sP.snap, sP.redshift)
            ax.hist(vals, bins=nBins, alpha=0.6, label=label)

        # finish plot
        ax.legend(loc="best")
        pdf.savefig()
        plt.close(fig)

    # by type
    for field in fields_sub + fields_group:
        if field[-4:] != "Type":
            continue
        print(field)

        # load
        data = []
        labels = []

        for sP in sPs:
            if field in fields_sub:
                vals = sP.subhalos(field)
            else:
                vals = sP.halos(field)

            label = sP.simName + " snap=%d (z=%.1f)" % (sP.snap, sP.redshift)

            data.append(vals)
            labels.append(label)

        # separate plot for each type
        for pt in [0, 1, 4, 5]:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(field + " [Type=%d] [log]" % pt)
            ax.set_ylabel("log N")

            for data_loc, label in zip(data, labels):
                vals = np.squeeze(data_loc[:, pt])
                w = np.where(vals == 0)
                print(field, pt, " number of zeros: ", len(w[0]), " of ", vals.size)
                vals = np.log10(vals)
                vals = vals[np.isfinite(vals)]

                ax.hist(vals, bins=nBins, alpha=0.6, label=label)

            # finish plot
            ax.legend(loc="best")
            pdf.savefig()
            plt.close(fig)

    # finish
    pdf.close()


def lgal_cat_check():
    """Check Reza's L-Galaxies catalog."""
    from .util.helper import running_median

    sP = simParams(run="tng100-1", redshift=0.0)  # tng300-1

    # DMO positional
    pos1 = sP.dmoBox.subhalos("SubhaloPos")
    pos2 = sP.subhalos("LGal_Pos_Dark") * 1000  # mpc->kpc
    gal_type = sP.subhalos("LGal_Type_Dark")

    w = np.where(gal_type == 0)
    dists = sP.periodicDists(pos1[w], pos2[w])
    print("Max dist (should be small): ", dists.max(), " ckpc/h")

    # stellar mass
    mstar1 = sP.subhalos("SubhaloMassInRadType")[:, sP.ptNum("stars")]
    mstar2 = sP.subhalos("LGal_StellarMass")
    gal_type = sP.subhalos("LGal_Type")

    w = np.where((gal_type == 0) & (mstar2 > 0))
    print("Mean stellar mass ratio: ", np.mean(mstar1[w] / mstar2[w]))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel("Stellar Mass (LGal) [ log Msun ]")
    ax.set_ylabel("Stellar Mass Ratio (TNG/LGal) [ log ]")
    ax.set_xlim([8.5, 12.0])
    ax.set_ylim([-0.8, 0.6])

    xx = sP.units.codeMassToLogMsun(mstar2[w])
    yy = np.log10(mstar1[w] / mstar2[w])

    ax.plot(xx, yy, ".", markersize=1.0)

    xm, ym, sm = running_median(xx, yy, binSize=0.2)
    ax.plot(xm, ym, "-", lw=3.0, alpha=0.7)

    fig.savefig("out.pdf")
    plt.close(fig)


def swift_vs_arepo_performance():
    """L35n270TNG_NR test runs (weak scaling) in SWIFT and AREPO."""
    num_cores = [40, 80, 160, 320, 640]

    # core hours [kH]
    cputime_swift = [2.20, 3.52, 5.09, 9.27, 17.25]  # CF=0.1 ('required for sph')
    cputime_swift2 = [np.nan, np.nan, 3.70, np.nan, np.nan]  # CF=0.3
    cputime_arepo = [2.97, 3.19, 3.56, 4.73, 6.31]  # CF=0.3

    # plot
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("CPU Time [kHours]")

    ax.set_title("L35n270TNG_NR (TNG50-4-NR) Strong Scaling Test (to z=0)")
    ax.set_xlim([10, 1000])
    ax.set_ylim([2, 11])
    ax.set_xscale("log")

    ax_top = ax.twiny()
    ax_top.set_xscale(ax.get_xscale())
    ax_top.minorticks_off()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(num_cores)
    ax_top.set_xticklabels(num_cores)

    ax.plot(
        num_cores, cputime_arepo, "o-", label="AREPO (695f5dedc, public version, TreePM, PMGRID=512, full DP, CF=0.3)"
    )
    ax.plot(num_cores, cputime_swift, "o-", label="SWIFT (d9167a957, 11 Jan 2020, SPHENIX SPH/quintic, FMM, CF=0.1)")
    ax.plot(num_cores, cputime_swift2, "o-", label="SWIFT (CF=0.3)")

    ax.legend(loc="upper left", fontsize=18)
    fig.savefig("perf_swift_vs_arepo.pdf")
    plt.close(fig)


def check_tracer_pos():
    """Find a single tracer through some snapshots for debugging."""
    startSnap = 2912
    targetID = 102320028362
    sP = simParams(run="tng50-1", variant="subbox2", snap=startSnap)

    startSnap = 67
    sP = simParams(run="tng50-1", snap=startSnap)

    for i in range(5):
        sP.setSnap(startSnap - i - 1)
        print("snap = ", sP.snap, " redshift = ", sP.redshift)

        # load tracer IDs and find tracer by its ID
        trIDs = sP.snapshotSubsetP("tracer", "TracerID")

        trInd = np.where(trIDs == targetID)[0][0]

        # load parent ID of this tracer
        parID = sP.snapshotSubset("tracer", "ParentID", inds=[trInd])[0]

        print(" found tracer ind = %d (ParentID = %d)" % (trInd, parID))

        # locate parent
        for parType in ["gas"]:
            parIDs = sP.snapshotSubsetP(parType, "id")
            parInd = np.where(parIDs == parID)[0][0]

            print(" found parent [%s] ind = %d" % (parType, parInd))

            parPos = sP.snapshotSubset(parType, "pos", inds=[parInd])[0]
            parTemp = sP.snapshotSubset(parType, "temp", inds=[parInd])[0]
            print(" pos = ", parPos)
            print(" temp = ", parTemp)


def parse_rur_out():
    """Parse the HLRS XC40 resource usage file."""
    import datetime

    path = "/u/dnelson/sims.TNG/L35n2160TNG/output/txt-files/rur.out"

    with open(path) as f:
        lines = f.readlines()

    tot_joule = 0
    count = 0
    count_big = 0
    tot_time_hours = 0.0

    earliest_date = datetime.datetime.strptime("2100", "%Y")
    latest_date = datetime.datetime.strptime("1900", "%Y")

    ncores = 16320

    for line in lines:
        if "energy_used" in line and "min_accel_power" not in line:
            el = line.split()
            loc_joule = int(el[-1].replace("]", ""))
            tot_joule += loc_joule
        if "energy_used" in line and "min_accel_power" in line:
            el = line.split()
            loc_joule = int(el[21].replace(",", ""))
            tot_joule += loc_joule
        if "utime" in line:
            el = line.split()
            loc_time_musec = int(el[11].replace(",", ""))  # user time (summed over all processes)
            loc_time_hours = loc_time_musec / 1e6 / 60 / 60 / ncores
            count += 1
            if loc_time_hours >= 1.0:
                count_big += 1
            tot_time_hours += loc_time_hours
        if "APP_START" in line:
            loc_time = line.split()[11].replace("CET", "CEST")
            loc_dt = datetime.datetime.strptime(loc_time, "%Y-%m-%dT%H:%M:%SCEST")

            if loc_dt < earliest_date:
                earliest_date = loc_dt
            if loc_dt > latest_date:
                latest_date = loc_dt

    tot_kwh = tot_joule * 2.7e-7
    print("total MWH: %g (across %d jobs, %d of them >1 hour)" % (tot_kwh / 1e3, count, count_big))
    print("from [%s] to [%s]" % (earliest_date, latest_date))


def loadspeed_test():
    """Test NVMe cache on freyator."""
    sP = simParams(run="tng300-1", redshift=0.0)

    fields = [
        "SubhaloMassType",
        "SubhaloLenType",
        "SubhaloMassInRadType",
        "SubhaloGasMetalFractions",
        "SubhaloGasMetalFractionsSfr",
        "SubhaloLen",
    ]

    # direct h5py
    start_time = time.time()
    with h5py.File("/mnt/nvme/cache/%s/output/groups_099/fof_subhalo_tab_099.hdf5" % (sP.simNameAlt), "r") as f:
        y = {}
        for field in fields:
            print(field)
            y[field] = f["Subhalo"][field][()]
    print("single file: %.2f sec" % (time.time() - start_time))

    # load (possibly threaded)
    start_time = time.time()
    x = sP.groupCat(fieldsSubhalos=fields)
    print("done: %.2f sec" % (time.time() - start_time))

    # verify
    for key in fields:
        assert np.array_equal(x[key], y[key])


def francesca_voversigma():
    """Create plot of V/sigma from H-alpha kinematics data for Francesca."""
    # load
    sP = simParams(run="tng50-1", redshift=4.0)
    file = sP.postPath + "SlitKinematics/Subhalo_Halpha_BinnedSlitKinematics_%03d.hdf5" % sP.snap

    with h5py.File(file, "r") as f:
        v = f["Subhalo"]["Halpha_05ckpc_InRad_V_max_kms"][()]
        sigma = f["Subhalo"]["Halpha_05ckpc_InRad_sigmaV_binned_HalfRad2Rad"][()]

    mstar = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])

    mstar_bins = [[9.7, 10.3], [9.65, 10.35]]

    # plot
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"H-alpha based V$_{\rm rot}$ / $\sigma$")
    ax.set_ylabel("Number of Galaxies")

    nBins = 13

    ax.set_xlim([0, nBins])
    ax.set_ylim([0, 25])

    for mstar_bin in mstar_bins:
        w = np.where((mstar >= mstar_bin[0]) & (mstar < mstar_bin[1]))
        label = "%.2f < M$_\\star$ < %.2f" % (mstar_bin[0], mstar_bin[1])

        # hist,bins = np.histogram( (v/sigma)[w], bins=np.arange(nBins+1))
        # ax.plot(bins, hist, label=label, drawstyle='steps', alpha=0.5)

        plt.hist((v / sigma)[w], bins=nBins, range=[0, nBins], label=label, alpha=0.5)
        print(mstar_bin, (v / sigma)[w].mean(), np.median((v / sigma)[w]), np.std((v / sigma)[w]))

    ax.legend()
    fig.savefig("voversigma_%s_z=%d.pdf" % (sP.simName, sP.redshift))
    plt.close(fig)

    # count
    w = np.where(v / sigma > 7)
    print("voversigma values above seven: ", len(w[0]), w[0], v[w], sigma[w], (v / sigma)[w], mstar[w])


def plot_dist256():
    """Plot distance to 256th gas cell (i.e. BH accretion radius) vs M*."""
    from .util.helper import running_median

    # load
    sP = simParams(run="tng100-1", redshift=0.0)
    dist = sP.auxCat("Subhalo_Gas_Dist256")["Subhalo_Gas_Dist256"]
    mstar = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])

    dist = sP.units.codeLengthToKpc(dist)

    # plot
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel("Stellar Mass [ log M$_{\\rm sun}$ ]")
    ax.set_ylabel("Distance to 256th closest gas cell [ pkpc ]")

    ax.set_xlim([9.0, 12.0])
    ax.set_ylim([0, 20])

    ax.scatter(mstar, dist, 4.0, marker=".")

    xm, ym, sm = running_median(mstar, dist, binSize=0.1)
    ax.plot(xm, ym, "-", label="median")

    ax.legend()
    fig.savefig("check_dist256.pdf")
    plt.close(fig)


def check_load_memusage():
    """Check memory usage with snapshotSubset()."""
    import gc
    import multiprocessing as mp
    import tracemalloc

    from .util.helper import pSplitRange, reportMemory

    pSplitNum = 10
    ptTypes = [0, 1, 4, 5]

    sP = simParams(res=540, run="tng", redshift=0.0)
    pt = "gas"

    print("a: ", reportMemory() * 1024)  # base

    # allocate
    NumPart = sP.snapshotHeader()["NumPart"]
    NumPartTot = np.sum([NumPart[pt] for pt in ptTypes])

    indRange = [0, NumPart[sP.ptNum(pt)]]
    offset = 0

    tracemalloc.start()

    if 0:
        data = np.zeros(NumPart[sP.ptNum(pt)], dtype="float32")
        print("D: ", NumPart[sP.ptNum(pt)] * 4 / 1024**2)
        print("b: ", reportMemory() * 1024)

        for i in range(pSplitNum):
            # local range, snapshotSubset inclusive on last index
            locRange = pSplitRange(indRange, pSplitNum, i)
            locRange[1] -= 1

            expectedSize = (locRange[1] - locRange[0]) * 4 / 1024**2  # MB

            print("%d: " % i, reportMemory() * 1024, locRange, expectedSize, flush=True)

            # load
            data_loc = sP.snapshotSubsetP(pt, "mass", indRange=locRange)

            data[offset : offset + data_loc.shape[0]] = data_loc
            offset += data_loc.shape[0]

    if 1:
        data = np.zeros(NumPartTot, dtype="float32")
        print("D: ", NumPartTot * 4 / 1024**2, data.nbytes / 1024**2)
        print("b: ", reportMemory() * 1024)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        for stat in top_stats[:10]:
            print(stat)

        for pt in ptTypes:
            expectedSize = NumPart[pt] * 4 / 1024**2  # MB
            print("%da: " % pt, reportMemory() * 1024, expectedSize, flush=True)

            data_loc = sP.snapshotSubsetP(pt, "mass")
            print("%db: " % pt, reportMemory() * 1024, expectedSize, flush=True)
            data[offset : offset + NumPart[pt]] = data_loc

            print("%dc: " % pt, reportMemory() * 1024, expectedSize, flush=True)

            # hack: https://bugs.python.org/issue32759 (fixed only in python 3.8x)
            del data_loc
            mp.heap.BufferWrapper._heap = mp.heap.Heap()
            gc.collect()

            print("%dd: " % pt, reportMemory() * 1024, expectedSize, flush=True)

            offset += NumPart[pt]

    print("c: ", reportMemory() * 1024)

    # verify removing data_loc returns us to base+sizeof(data)
    del data_loc
    gc.collect()
    print("d: ", reportMemory() * 1024)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    for stat in top_stats[:10]:
        print(stat)


def check_groupcat_snap_rewrite(GrNr=0):
    """Check custom Subfind."""
    sP = simParams(res=2160, run="tng", snap=69)  # 69-99
    # sP = simParams(res=128,run='tng',snap=4,variant='0000')

    # final_save_file = sP.derivPath + "fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)

    fof = sP.groupCatSingle(haloID=GrNr)
    h = sP.snapshotHeader()

    np.random.seed(4242)
    sub_ids = np.arange(fof["GroupNsubs"]) + fof["GroupFirstSub"]
    np.random.shuffle(sub_ids)

    num_todo = 5
    sub_ids = [0, 1, 2, 3, 4, 5]

    for i in range(num_todo):
        print(i, sub_ids[i])

        # load
        sub = sP.groupCatSingle(subhaloID=sub_ids[i])

        dm = sP.snapshotSubset("dm", ["pos", "vel"], subhaloID=sub_ids[i])
        gas = sP.snapshotSubset("gas", ["mass", "pos", "sfr"], subhaloID=sub_ids[i])
        stars = sP.snapshotSubset("stars", ["mass", "pos"], subhaloID=sub_ids[i])
        bh = sP.snapshotSubset("bh", ["BH_Mdot", "pos", "mass"], subhaloID=sub_ids[i])

        gas_sfr = 0.0
        gas_mass = 0.0
        stars_mass = 0.0
        bh_mass = 0.0
        bh_mdot = 0.0
        dm_mass = dm["count"] * h["MassTable"][sP.ptNum("dm")]

        if gas["count"] > 0:
            gas_mass = gas["Masses"].sum()
            gas_sfr = gas["StarFormationRate"].sum()
        if stars["count"] > 0:
            stars_mass = stars["Masses"].sum()
        if bh["count"] > 0:
            bh_mass = bh["Masses"].sum()
            bh_mdot = bh["BH_Mdot"].sum()

        sub_mass = dm_mass + gas_mass + stars_mass + bh_mass

        cm = np.zeros(3, dtype="float64")
        for i in range(3):
            if dm["count"]:
                cm[i] += np.sum((dm["Coordinates"][:, i] - sub["SubhaloPos"][i]) * h["MassTable"][sP.ptNum("dm")])
            if gas["count"]:
                cm[i] += np.sum((gas["Coordinates"][:, i] - sub["SubhaloPos"][i]) * gas["Masses"])
            if stars["count"]:
                cm[i] += np.sum((stars["Coordinates"][:, i] - sub["SubhaloPos"][i]) * stars["Masses"])
            if bh["count"]:
                cm[i] += np.sum((bh["Coordinates"][:, i] - sub["SubhaloPos"][i]) * bh["Masses"])

        cm /= sub_mass
        cm += sub["SubhaloPos"]

        # compare
        print("lentype ", sub["SubhaloLenType"])
        print("su mass ", sub_mass, sub["SubhaloMass"])
        print("ga mass ", gas_mass, sub["SubhaloMassType"][sP.ptNum("gas")])
        print("gas sfr ", gas_sfr, sub["SubhaloSFR"])
        print("st mass ", stars_mass, sub["SubhaloMassType"][sP.ptNum("stars")])
        print("dm mass ", dm_mass, sub["SubhaloMassType"][sP.ptNum("dm")])
        print("bh mass ", bh_mass, sub["SubhaloMassType"][sP.ptNum("bh")])
        print("bh mdot ", bh_mdot, sub["SubhaloBHMdot"])
        print("cm      ", cm, sub["SubhaloCM"], end="\n\n")


def hsc_gri_composite():
    """Try to recreate HSC composite image based on (g,r,i) bands."""
    import skimage.io
    from astropy.io import fits

    # load
    files = [
        "cutout-HSC-G-8524-s17a_dud-180417-060421.fits",
        "cutout-HSC-R-8524-s17a_dud-180417-060434.fits",
        "cutout-HSC-I-8524-s17a_dud-180417-060445.fits",
    ]
    images = []

    for file in files:
        with fits.open(file) as hdu:
            images.append(hdu[1].data)

    band0_grid = images[2]  # I-band -> b
    band1_grid = images[1]  # R-band -> g
    band2_grid = images[0]  # G-band -> r

    nPixels = band0_grid.shape[::-1]

    # astropy lupton version
    from astropy.visualization import make_lupton_rgb

    image_lupton = make_lupton_rgb(images[2], images[1], images[0], Q=10, stretch=0.5)
    skimage.io.imsave("out_astropy.png", image_lupton)

    # mine
    grid_master = np.zeros((nPixels[1], nPixels[0], 3), dtype="float32")
    grid_master_u = np.zeros((nPixels[1], nPixels[0], 3), dtype="uint8")

    # lupton scheme
    fac = {"g": 1.0, "r": 1.0, "i": 1.0}  # RGB = gri
    lupton_alpha = 2.0  # 1/stretch
    lupton_Q = 8.0
    scale_min = 1e-4  # units of linear luminosity

    # make RGB array using arcsinh scaling following Lupton
    band0_grid *= fac["i"]
    band1_grid *= fac["r"]
    band2_grid *= fac["g"]

    inten = (band0_grid + band1_grid + band2_grid) / 3.0
    val = np.arcsinh(lupton_alpha * lupton_Q * (inten - scale_min)) / lupton_Q

    grid_master[:, :, 0] = band0_grid * val / inten
    grid_master[:, :, 1] = band1_grid * val / inten
    grid_master[:, :, 2] = band2_grid * val / inten

    # rescale and clip
    maxval = np.max(grid_master, axis=2)  # for every pixel, across the 3 bands

    w = np.where(maxval > 1.0)
    for i in range(3):
        grid_master[w[0], w[1], i] /= maxval[w]

    w = np.where((maxval < 0.0) | (inten < 0.0))
    for i in range(3):
        grid_master[w[0], w[1], i] = 0.0

    grid_master = np.clip(grid_master, 0.0, np.inf)

    # construct RGB
    for i in range(3):
        grid_master_u[:, :, i] = grid_master[:, :, i] * np.uint8(255)

    # save
    skimage.io.imsave("out.png", grid_master_u)


def check_tracer_tmax_vs_curtemp():
    """Can a tracer maxtemp ever be below the current parent gas cell temperature?"""
    # sP = simParams(res=11,run='zooms2_josh',redshift=2.25,hInd=2,variant='FPorig') # snap=52
    sP = simParams(res=11, run="zooms2_josh", redshift=2.25, hInd=2, variant="FP")
    # sP = simParams(res=11,run='zooms2_josh',snap=10,hInd=2,variant='FPorig')
    # sP = simParams(res=13,run='tng_zoom',redshift=2.0,hInd=50,variant='sf3')
    haloID = 0

    # load
    #'tracer_maxtemp') # change to 'FluidQuantities' for h2_L11_12_FP (only tmax stored)
    tmax = sP.snapshotSubset("tracer", "FluidQuantities")
    par_id = sP.snapshotSubset("tracer", "ParentID")
    tr_id = sP.snapshotSubset("tracer", "TracerID")

    gas_id = sP.snapshotSubset("gas", "id", haloID=haloID)
    gas_sfr = sP.snapshotSubset("gas", "sfr", haloID=haloID)
    gas_temp = sP.snapshotSubset("gas", "temp_log", haloID=haloID)
    star_id = sP.snapshotSubset("star", "id", haloID=haloID)

    # cross-match
    print("match...")
    gas_inds, tr_inds_gas = match(gas_id, par_id)
    star_inds, tr_inds_star = match(star_id, par_id)

    # fill
    tr_par_type = np.zeros(par_id.size, dtype="int16")
    tr_par_type.fill(-1)
    tr_par_type[tr_inds_gas] = 0
    tr_par_type[tr_inds_star] = 4

    tr_par_sfr = np.zeros(par_id.size, dtype="float32")
    tr_par_sfr.fill(-1.0)
    tr_par_sfr[tr_inds_gas] = gas_sfr[gas_inds]

    tr_par_temp = np.zeros(par_id.size, dtype="float32")
    tr_par_temp[tr_inds_gas] = gas_temp[gas_inds]

    # select
    print("tot tracers: ", par_id.size)

    w = np.where(tr_par_temp > tmax)
    print("current temp above tmax: ", len(w[0]))

    w = np.where((tr_par_temp > tmax) & (tr_par_type == 0))
    print("current temp above tmax (and in gas): ", len(w[0]))

    w = np.where((tr_par_temp > tmax) & (tr_par_type == 0) & (tr_par_sfr == 0))
    print("current temp above tmax (and in sfr==0 gas): ", len(w[0]))

    diffs = tr_par_temp[w] - tmax[w]
    # print('minmax delta_T(log): ', np.nanmin(diffs), np.nanmax(diffs))

    # load one snap back
    sP.setSnap(sP.snap - 1)

    tr_id_prev = sP.snapshotSubset("tracer", "TracerID")
    par_id_prev = sP.snapshotSubset("tracer", "ParentID")
    gas_id_prev = sP.snapshotSubset("gas", "id")
    gas_sfr_prev = sP.snapshotSubset("gas", "sfr")
    gas_temp_prev = sP.snapshotSubset("gas", "temp_log")

    # match tracers between snaps
    print("match tr...")
    tr_inds_cur, tr_inds_prev = match(tr_id, tr_id_prev)
    assert tr_inds_cur.size == tr_id.size  # must find all

    tmax_prev = tmax[tr_inds_cur]
    tr_par_sfr2 = tr_par_sfr[tr_inds_cur]
    par_id2 = par_id[tr_inds_cur]

    par_id_prev = par_id_prev[tr_inds_prev]
    tr_id_prev = tr_id_prev[tr_inds_prev]

    # match previous snap tracers to their parents (gas only, will only find some)
    print("match gas...")
    gas_id_prev_inds, tr_inds_prev_gas = match(gas_id_prev, par_id_prev)

    tmax_prev = tmax_prev[tr_inds_prev_gas]  # same tmax restricted to these gas matches
    tr_par_sfr2 = tr_par_sfr2[tr_inds_prev_gas]
    par_id2 = par_id2[tr_inds_prev_gas]

    par_id_prev = par_id_prev[tr_inds_prev_gas]
    tr_id_prev = tr_id_prev[tr_inds_prev_gas]

    # parent properties at previous snap
    tr_par_sfr_prev = gas_sfr_prev[gas_id_prev_inds]
    tr_par_temp_prev = gas_temp_prev[gas_id_prev_inds]

    # select
    w = np.where(tr_par_temp_prev > tmax_prev)
    print("current temp above tmax: ", len(w[0]))

    w = np.where((tr_par_temp_prev > tmax_prev) & (tr_par_sfr_prev == 0))
    print("current temp above tmax (and in sfr==0 gas at snap-1): ", len(w[0]))

    diffs = tr_par_temp_prev[w] - tmax_prev[w]
    print("minmax delta_T(log): ", diffs.min(), diffs.max())


def check_tracer_tmax_vs_curtemp2():
    """Followup, single tracer."""
    tracer_id = 175279761  # 269811444 #341052796# 215262662 #239990945 #205515254
    sP = simParams(res=11, run="zooms2_josh", redshift=2.25, hInd=2, variant="FPorig")  # snap=52

    # go back
    for i in range(3):
        if i > 0:
            sP.setSnap(sP.snap - 1)
        print("snap: ", sP.snap)

        # load
        tr_id = sP.snapshotSubset("tracer", "TracerID")
        ind_snap = np.where(tr_id == tracer_id)[0][0]
        inds = np.array([ind_snap])

        tmax = sP.snapshotSubset("tracer", "tracer_maxtemp", inds=inds)
        windc = sP.snapshotSubset("tracer", "tracer_windcounter", inds=inds)
        lst = sP.snapshotSubset("tracer", "tracer_laststartime", inds=inds)
        par_id = sP.snapshotSubset("tracer", "ParentID", inds=inds)[0]
        gas_ids = sP.snapshotSubset("gas", "id")

        print("tracer tmax: ", tmax, " windc: ", windc, " lst: ", lst)

        gas_ind = np.where(gas_ids == par_id)[0][0]
        inds = np.array([gas_ind])

        temp = sP.snapshotSubset("gas", "temp_log", inds=inds)
        sfr = sP.snapshotSubset("gas", "sfr", inds=inds)

        print("gas temp: ", temp, " sfr: ", sfr, " id:", gas_ids[gas_ind])


def check_colors_benedikt():
    """Test my colors vs snapshot."""
    from scipy.stats import binned_statistic_2d

    sP = simParams(res=1820, run="tng", redshift=0.0)

    # load
    mag_g_snap = sP.groupCat(fieldsSubhalos=["SubhaloStellarPhotometrics"])[:, 4]  # g-band

    acKey = "Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"
    ac = sP.auxCat(acKey)
    # bands = ac[acKey + "_attrs"]["bands"]
    mag_g_dust = ac[acKey][:, 1]  # g-band
    print(mag_g_dust.shape)
    mag_g_dust = mag_g_dust[:, 0]  # pick 1 projection at random

    # count valid
    w_snap = np.where(mag_g_snap < 0)
    w_dust = np.where(np.isfinite(mag_g_dust))

    print(len(w_snap[0]), len(w_dust[0]))
    print("snap: ", mag_g_snap[w_snap].min(), mag_g_snap[w_snap].max())
    print("dust: ", mag_g_dust[w_dust].min(), mag_g_dust[w_dust].max())

    # plot
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel("g_mag [snap]")
    ax.set_ylabel("g_mag [dust]")

    minmax = [-25, -5]

    ax.set_xlim(minmax)
    ax.set_ylim(minmax)

    nn, _, _, _ = binned_statistic_2d(
        mag_g_snap, mag_g_dust, np.zeros(mag_g_snap.size), "count", bins=[100, 100], range=[minmax, minmax]
    )
    nn = np.log10(nn.T)

    extent = [minmax[0], minmax[1], minmax[0], minmax[1]]
    im = plt.imshow(nn, extent=extent, origin="lower", interpolation="nearest", aspect="auto", cmap="viridis")

    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel("log Num gal")

    fig.savefig("mag_comp.pdf")
    plt.close(fig)


def guinevere_mw_sample():
    """Examine Milky Way sample for Guinevere's paper."""
    # get subhaloIDs
    sP_tng = simParams(res=1820, run="tng", redshift=0.0)
    sP_ill = simParams(res=1820, run="illustris", redshift=0.0)
    # sP_tng = simParams(res=512,run='tng',redshift=0.0,variant='0000')
    # sP_ill = simParams(res=512,run='tng',redshift=0.0,variant='0010')

    # data = np.genfromtxt(sP_tng.postPath + 'guinevere_cutouts/new_mw_sample_fgas.txt', delimiter=',', dtype='int32')
    data = np.genfromtxt(sP_tng.postPath + "guinevere_cutouts/new_mw_sample_fgas_sat.txt", delimiter=",", dtype="int32")
    subIDs_tng = data[:, 0]
    subIDs_ill = data[:, 1]

    w = np.where((subIDs_tng != -1) & (subIDs_ill != -1))
    print(len(w[0]))

    subIDs_tng = subIDs_tng[w]
    subIDs_ill = subIDs_ill[w]

    # load subhalo data
    masstype_tng = sP_tng.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])[subIDs_tng, :]
    masstype_ill = sP_ill.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])[subIDs_ill, :]

    is_cen_tng = sP_tng.groupCat(fieldsSubhalos=["central_flag"])[subIDs_tng]
    is_cen_ill = sP_ill.groupCat(fieldsSubhalos=["central_flag"])[subIDs_ill]

    fgas_tng = np.log10(masstype_tng[:, sP_tng.ptNum("gas")] / masstype_tng[:, sP_tng.ptNum("stars")])
    fgas_ill = np.log10(masstype_ill[:, sP_ill.ptNum("gas")] / masstype_ill[:, sP_ill.ptNum("stars")])

    wcen_tng = np.where(is_cen_tng == 1)
    wcen_ill = np.where(is_cen_ill == 1)
    wsat_tng = np.where(is_cen_tng == 0)
    wsat_ill = np.where(is_cen_ill == 0)

    assert wcen_tng[0].size + wsat_tng[0].size == is_cen_tng.size
    assert wcen_ill[0].size + wsat_ill[0].size == is_cen_ill.size

    msize = 7.0

    pdf = PdfPages("sample_check_25Mpc.pdf")

    for i in [0, 1]:
        fig, ax = plt.subplots()
        ax.set_xlabel(r"log $M_{\rm gas} / M_\star$ [ Illustris ]")
        ax.set_ylabel(r"log $M_{\rm gas} / M_\star$ [ TNG ]")

        ax.set_xlim([-2.2, 0.3])
        ax.set_ylim([-2.2, 0.3])

        if i == 0:
            wcen = wcen_tng
            wsat = wsat_tng
            label = "TNG"
        if i == 1:
            wcen = wcen_ill
            wsat = wsat_ill
            label = "ILL"

        ax.plot(fgas_ill[wcen], fgas_tng[wcen], "o", ms=msize, color="blue", alpha=0.7, label="Cen in %s" % label)
        ax.plot(fgas_ill[wsat], fgas_tng[wsat], "o", ms=msize, color="red", alpha=0.7, label="Sat in %s" % label)

        ax.legend(loc="upper left")
        pdf.savefig()
        plt.close(fig)

    pdf.close()


def vis_cholla_snapshot():
    """Testing."""
    basePath = "/u/dnelson/sims.idealized/gpu.cholla/"
    num = 999

    files = glob.glob(basePath + "/output/%d.h5.*" % num)

    # get size from first file and allocate
    with h5py.File(files[0], "r") as f:
        attrs = dict(f.attrs.items())
        fields = f.keys()

    for key in attrs:
        print(key, attrs[key])

    data = {}
    for field in fields:
        data[field] = np.zeros(attrs["dims"], dtype="float32")

    for file in files:
        print(file)
        with h5py.File(file, "r") as f:
            # get local dataset sizes and location
            offset = f.attrs["offset"]
            dims = f.attrs["dims_local"]
            assert dims[2] == 1  # 2D

            # read all datasets
            for field in f:
                data[field][offset[0] : offset[0] + dims[0], offset[1] : offset[1] + dims[1], 0] = f[field][()]

    # limits
    xlim = [0, attrs["domain"][0]]
    ylim = [0, attrs["domain"][1]]

    clims = {
        "Energy": [6.0, 7.2],
        "density": [1.0, 2.0],
        "momentum_x": [-1.0, 1.0],
        "momentum_y": [-1.0, 1.0],
        "momentum_z": [0.0, 1.0],
    }

    # start plot
    from matplotlib.colors import Normalize

    from .plot.util import loadColorTable

    for field in data:
        print("plotting: [%s]" % field)

        aspect = float(attrs["dims"][0]) / attrs["dims"][1]
        fig = plt.figure(figsize=[figsize[0] * aspect, figsize[1]])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cmap = loadColorTable("viridis")
        norm = Normalize(vmin=clims[field][0], vmax=clims[field][1], clip=False)
        zz = np.squeeze(data[field].T)  # 2D

        im = plt.imshow(
            zz,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
            aspect=1.0,
        )

        cb = plt.colorbar(im, ax=ax, pad=0)
        cb.ax.set_ylabel(field)

        fig.savefig("cholla_%d_%s.pdf" % (num, field))
        plt.close(fig)


def new_mw_fgas_sample():
    """Create new Milky Way sample of Guinevere."""
    from .cosmo.util import crossMatchSubhalosBetweenRuns

    sP_illustris = simParams(res=1820, run="illustris", redshift=0.0)
    sP_tng = simParams(res=1820, run="tng", redshift=0.0)

    # sP_illustris = simParams(res=512, run='tng', redshift=0.0, variant='0010')
    # sP_tng = simParams(res=512, run='tng', redshift=0.0, variant='0000')

    # mhalo = sP_tng.groupCat(fieldsSubhalos=['mhalo_200']) # [msun]
    mhalo = sP_tng.groupCat(fieldsSubhalos=["mhalo_subfind"])  # [msun]
    mstar = sP_tng.groupCat(fieldsSubhalos=["mstar_30pkpc"])  # [msun]
    # fgas  = sP_tng.groupCat(fieldsSubhalos=['fgas_2rhalf']) # m_gas/m_b within 2rhalfstars
    fgas, _, _, _ = sP_tng.simSubhaloQuantity("fgas2")

    inds_tng = np.where((mhalo >= 6e11) & (mhalo < 2e12) & (mstar >= 5e10) & (mstar < 1e11) & (fgas >= 0.01))[0]

    inds_ill_pos = crossMatchSubhalosBetweenRuns(sP_tng, sP_illustris, inds_tng, method="Positional")
    inds_ill_la = crossMatchSubhalosBetweenRuns(sP_tng, sP_illustris, inds_tng, method="Lagrange")

    header = "subhalo indices (z=0): TNG100-1, Illustris-1 (Lagrangian match), Illustris-1 (positional match)\n"
    with open("new_mw_sample_fgas.txt", "w") as f:
        f.write(header)
        for i in range(inds_tng.size):
            f.write("%d, %d, %d\n" % (inds_tng[i], inds_ill_la[i], inds_ill_pos[i]))

    mhalo_ill = sP_illustris.groupCat(fieldsSubhalos=["mhalo_200"])
    mstar_ill = sP_illustris.groupCat(fieldsSubhalos=["mstar_30pkpc"])
    fgas_ill, _, _, _ = sP_illustris.simSubhaloQuantity("fgas2")

    for i in range(inds_tng.size):
        if inds_tng[i] == -1 or inds_ill_la[i] == -1:
            print(i, "no match")
        else:
            ratio_mhalo = mhalo[inds_tng[i]] / mhalo_ill[inds_ill_la[i]]
            ratio_mstar = mstar[inds_tng[i]] / mstar_ill[inds_ill_la[i]]
            mhalo1 = np.log10(mhalo[inds_tng[i]])
            mhalo2 = np.log10(mhalo_ill[inds_ill_la[i]])
            mstar1 = np.log10(mstar[inds_tng[i]])
            mstar2 = np.log10(mstar_ill[inds_ill_la[i]])
            fgas1 = fgas[inds_tng[i]]
            fgas2 = fgas_ill[inds_ill_la[i]]

            print(i, mhalo1, mhalo2, mstar1, mstar2, ratio_mhalo, ratio_mstar, fgas1, fgas2)

    print("Done.")


def bh_details_check():
    """Check gaps in TNG100-1 blackhole_details.hdf5."""
    with open("out.txt") as f:
        lines = f.read()

    lines = lines.split("\n")
    mdot = np.zeros(len(lines) - 1, dtype="float32")
    scalefac = np.zeros(len(lines) - 1, dtype="float32")

    for i, line in enumerate(lines[:-1]):
        d = line.split(" ")
        mdot[i] = float(d[3])
        scalefac[i] = float(d[1])

    redshift = 1 / scalefac - 1

    inds = np.argsort(redshift)
    redshift = redshift[inds]
    scalefac = scalefac[inds]
    mdot = mdot[inds]

    hh, _ = np.histogram(scalefac, bins=400)
    ww = np.where(hh <= 3)
    print("near-empty bins: ", redshift[ww])

    if 1:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Mdot [msun/yr]")
        ax.set_yscale("log")
        ax.plot(redshift, mdot, lw=0.5)

        fig.savefig("check_details.pdf")
        plt.close(fig)

    if 1:
        fig, ax = plt.subplots(figsize=(18, 10))

        ax.set_xlabel("Redshift")
        ax.set_ylabel("Mdot [msun/yr]")
        ax.set_xlim([2.0, 2.2])
        ax.set_yscale("log")
        ax.plot(redshift, mdot, lw=0.5)

        fig.savefig("check_details_zoom.pdf")
        plt.close(fig)


def bh_mdot_subbox_test():
    """Check that BH accretion rate equals mass increase."""
    sP = simParams(res=455, run="tng", redshift=0.0, variant="subbox0")

    h = sP.snapshotHeader()
    print("num BHs: ", h["NumPart"][sP.ptNum("bh")])

    ids = sP.snapshotSubset("bh", "ids")
    print(ids.shape)

    id = ids[0]  # take first one

    numSnapsBack = 20

    tage_prev = None
    mass_prev = None

    for snap in range(sP.snap - numSnapsBack, sP.snap):
        sP.setSnap(snap)
        ids = sP.snapshotSubset("bh", "ids")
        w = np.where(ids == id)[0]
        assert len(w)

        dt = sP.tage - tage_prev if tage_prev is not None else 0.0
        if tage_prev is None:
            tage_prev = sP.tage
        dt_yr = dt * 1e9

        mdot = sP.snapshotSubset("bh", "BH_Mdot") * 10.22  # msun/yr
        medd = sP.snapshotSubset("bh", "BH_MdotEddington") * 10.22  # msun/yr
        mass = sP.snapshotSubset("bh", "BH_Mass")
        mass2 = sP.snapshotSubset("bh", "Masses")
        mass = sP.units.codeMassToMsun(mass)
        mass2 = sP.units.codeMassToMsun(mass2)

        if mass_prev is None:
            mdot_actual = 0.0
            mass_prev = mass[w]
        else:
            BlackHoleRadiativeEfficiency = 0.2
            mdot_actual = mdot[w] * dt_yr
            # mdot_adios = mdot / All.BlackHoleAccretionFactor # only if in BH_RADIO_MODE
            # note: many other modifications here including BH_PRESSURE_CRITERION and BH_EXACT_INTEGRATION...
            deltaM = (1 - BlackHoleRadiativeEfficiency) * mdot_actual
            mass_prev += deltaM

        print(snap, w, mdot_actual, medd[w], mass[w], mass2[w], dt_yr, mass_prev, mass_prev / mass[w])


def check_millennium():
    """Check re-write of Millennium simulation files."""
    basePath = "/u/dnelson/sims.millennium/Millennium1/output/"
    snap = 63

    objType = "Subhalo"  # Subhalo
    objID = 123456

    groupPath = basePath + "groups_%03d/fof_subhalo_tab_%03d.hdf5" % (snap, snap)
    snapPath = basePath + "snapdir_%03d/snap_%03d.hdf5" % (snap, snap)

    with h5py.File(groupPath, "r") as f:
        snap_off = f["Offsets/%s_Snap" % objType][objID]
        snap_len = f["%s/%sLen" % (objType, objType)][objID]
        obj_pos = f["%s/%sPos" % (objType, objType)][objID, :]
        obj_vel = f["%s/%sVel" % (objType, objType)][objID, :]

    print("%s [%d] found offset = %d, length = %d" % (objType, objID, snap_off, snap_len))

    with h5py.File(snapPath, "r") as f:
        pos = f["PartType1/Coordinates"][snap_off : snap_off + snap_len, :]
        vel = f["PartType1/Velocities"][snap_off : snap_off + snap_len, :]
        ids = f["PartType1/ParticleIDs"][snap_off : snap_off + snap_len]

    for i in range(3):
        xyz = ["x", "y", "z"][i]
        print("pos mean %s = %f" % (xyz, np.mean(pos[:, i], dtype="float64")))
        print("vel mean %s = %f" % (xyz, vel[:, i].mean()))
    print("obj pos = ", obj_pos)
    print("obj vel = ", obj_vel)
    print("ids first five:", ids[0:5])
    print("ids last five: ", ids[-5:])


def verifySimFiles(sP, groups=False, fullSnaps=False, subboxes=False):
    """Verify existence, permissions, and HDF5 structure of groups, full snaps, subboxes."""
    from illustris_python.snapshot import getNumPart

    assert groups or fullSnaps or subboxes
    assert sP.run in ["tng", "tng_dm"]

    nTypes = 6
    nFullSnapsExpected = 100
    nSubboxesExpected = 2 if sP.boxSize == 75000 else 3
    nSubboxSnapsExpected = {
        75000: {455: 2431, 910: 4380, 1820: 7908},
        35000: {270: 2333, 540: 4006, 1080: -1, 2160: -1},
        205000: {625: 2050, 1250: 3045, 2500: -1},
    }

    def checkSingleGroup(files):
        """Helper (count header and dataset shapes)."""
        nGroups_0 = 0
        nGroups_1 = 0
        nSubhalos_0 = 0
        nSubhalos_1 = 0
        nGroups_tot = 0
        nSubhalos_tot = 0

        # verify correct number of chunks
        assert nGroupFiles == len(files)
        assert nGroupFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file, "r") as f:
                nGroups_0 += f["Header"].attrs["Ngroups_ThisFile"]
                nSubhalos_0 += f["Header"].attrs["Nsubgroups_ThisFile"]

                if f["Header"].attrs["Ngroups_ThisFile"] > 0:
                    nGroups_1 += f["Group"]["GroupPos"].shape[0]
                if f["Header"].attrs["Nsubgroups_ThisFile"] > 0:
                    nSubhalos_1 += f["Subhalo"]["SubhaloPos"].shape[0]

                nGroups_tot = f["Header"].attrs["Ngroups_Total"]
                nSubhalos_tot = f["Header"].attrs["Nsubgroups_Total"]

        assert nGroups_0 == nGroups_tot
        assert nGroups_1 == nGroups_tot
        assert nSubhalos_0 == nSubhalos_tot
        assert nSubhalos_1 == nSubhalos_tot
        print(" [%2d] %d %d" % (i, nGroups_tot, nSubhalos_tot))

    def checkSingleSnap(files):
        """Helper (common for full and subbox snapshots) (count header and dataset shapes)."""
        nPart_0 = np.zeros(6, dtype="int64")
        nPart_1 = np.zeros(6, dtype="int64")
        nPart_tot = np.zeros(6, dtype="int64")

        # verify correct number of chunks
        assert nSnapFiles == len(files)
        assert nSnapFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file, "r") as f:
                for j in range(nTypes):
                    nPart_0[j] += f["Header"].attrs["NumPart_ThisFile"][j]

                    if f["Header"].attrs["NumPart_ThisFile"][j] > 0:
                        if j == 3:  # trMC
                            nPart_1[j] += f["PartType" + str(j)]["TracerID"].shape[0]
                        else:  # normal
                            nPart_1[j] += f["PartType" + str(j)]["Coordinates"].shape[0]

                nPart_tot = getNumPart(dict(f["Header"].attrs.items()))

        assert (nPart_0 == nPart_tot).all()
        assert (nPart_1 == nPart_tot).all()
        print(
            " [%2d] %d %d %d %d %d %d"
            % (i, nPart_tot[0], nPart_tot[1], nPart_tot[2], nPart_tot[3], nPart_tot[4], nPart_tot[5])
        )

    if groups:
        numDirs = len(glob(sP.simPath + "groups*"))
        nGroupFiles = 0
        print("Checking [%d] group directories..." % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + "/groups_%03d/*.hdf5" % i)
            if nGroupFiles == 0:
                nGroupFiles = len(files)

            checkSingleGroup(files)

        print("PASS GROUPS.")

    if fullSnaps:
        numDirs = len(glob(sP.simPath + "snapdir*"))
        nSnapFiles = 0
        print("Checking [%d] fullsnap directories..." % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + "/snapdir_%03d/*.hdf5" % i)
            if nSnapFiles == 0:
                nSnapFiles = len(files)

            checkSingleSnap(files)

        print("PASS FULL SNAPS.")

    if subboxes:
        numSubboxes = len(glob(sP.simPath + "subbox?"))
        assert numSubboxes == nSubboxesExpected

        for sbNum in range(numSubboxes):
            numDirs = len(glob(sP.simPath + "subbox" + str(sbNum) + "/snapdir*"))
            nSnapFiles = 0

            print(" SUBBOX [%d]: Checking [%d] subbox directories..." % (sbNum, numDirs))
            assert numDirs == nSubboxSnapsExpected[sP.boxSize][sP.res]

            for i in range(numDirs):
                # search for chunks and set number
                files = glob(sP.simPath + "/subbox%d/snapdir_subbox%d_%03d/*.hdf5" % (sbNum, sbNum, i))
                if nSnapFiles == 0:
                    nSnapFiles = len(files)

                checkSingleSnap(files)

            print("PASS SUBBOX [%d]." % sbNum)
        print("PASS ALL SUBBOXES.")


def illustris_api_check():
    """Check API."""
    import requests

    def get(path, params=None):
        # make HTTP GET request to path
        headers = {"api-key": "10d143a0ef27c6461f94b50275d45d6f"}
        r = requests.get(path, params=params, headers=headers)
        print(r.url)
        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers["content-type"] == "application/json":
            return r.json()  # parse json responses automatically

        if "content-disposition" in r.headers:
            filename = r.headers["content-disposition"].split("filename=")[1]
            with open(filename, "wb") as f:
                f.write(r.content)
            return filename  # return the filename string

        return r

    # test1
    # base_url = "http://www.tng-project.org/api/TNG100-1/snapshots/55/subhalos/195979/vis.png"
    # params = {'partType':'stars', 'partField':'mass', 'size':0.08, 'sizeType':'arcmin', 'nPixels':20, 'axes':'0,1'}
    # saved_filename = get(base_url, params)
    # print(saved_filename)
    # return

    # test2
    # base_url = "http://www.tng-project.org/api/TNG300-1/snapshots/50/subhalos/10/cutout.hdf5"
    # params = {'gas':'Coordinates,Density'}
    # saved_filename = get(base_url,params)
    # print(saved_filename)
    # return

    # test3
    # base_url = "http://www.illustris-project.org/api/Illustris-1/"
    # sim_metadata = get(base_url)
    # params = {'dm':'Coordinates'}

    # for i in [300]:#range(sim_metadata['num_files_snapshot']):
    #    file_url = base_url + "files/snapshot-135." + str(i) + ".hdf5"
    #    print(file_url)
    #    saved_filename = get(file_url, params)
    #    print('done')

    # test4 (Task 10)
    from io import BytesIO

    import matplotlib.image as mpimg

    ids = [41092, 338375, 257378, 110568, 260067]
    sub_count = 1
    fig = plt.figure(figsize=[15, 3])

    for id in ids:
        url = "http://www.tng-project.org/api/Illustris-1/snapshots/135/subhalos/" + str(id)
        sub = get(url)
        if "stellar_mocks" in sub["supplementary_data"]:
            png_url = sub["supplementary_data"]["stellar_mocks"]["image_fof"]
            response = get(png_url)

            plt.subplot(1, len(ids), sub_count)
            plt.text(0, -20, "ID=" + str(id), color="blue")
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            sub_count += 1

            file_object = BytesIO(response.content)
            plt.imshow(mpimg.imread(file_object))

    fig.savefig("out.png")


def checkStellarAssemblyMergerMass():
    """Check addition to StellarAssembly catalogs."""
    sP = simParams(res=2500, run="tng", snap=99)

    fName = sP.postPath + "StellarAssembly/stars_%03d_supp.hdf5" % sP.snap

    with h5py.File(fName, "r") as f:
        InSitu = f["InSitu"][()]
        MergerMass = f["MergerMass"][()]
        MergerSnap = f["MergerSnap"][()]

    w_exsitu = np.where(InSitu == 0)

    if 1:
        fName2 = sP.postPath + "StellarAssembly/stars_%03d.hdf5" % sP.snap
        with h5py.File(fName2, "r") as f:
            InSitu_prev = f["InSitu"][()]
        print(sP.simName, np.array_equal(InSitu, InSitu_prev))

    # plot
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(121)

    ax.set_xlabel("MergerSnap (InSitu==0)")
    ax.set_ylabel("N$_{\\rm stars}$")
    ax.hist(MergerSnap[w_exsitu], bins=100, range=[0, 99])

    ax = fig.add_subplot(122)
    ax.set_xlabel("MergerMass [log M$_{\\rm sun}$] (InSitu==0)")
    ax.set_ylabel("N$_{\\rm stars}$")

    vals = sP.units.codeMassToLogMsun(MergerMass[w_exsitu])
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=100)

    fig.savefig("check_stellarassembly_supp_%s_%d.pdf" % (sP.simName, sP.snap))
    plt.close(fig)


def checkColorCombos():
    """Check (r-i) from color TNG paper."""
    from .cosmo.color import loadSimGalColors
    from .util.helper import array_equal_nan

    sP = simParams(res=1820, run="tng", redshift=0.0)

    simColorsModel = "p07c_cf00dust_res_conv_ns1_rad30pkpc"

    colorData = loadSimGalColors(sP, simColorsModel)
    ui, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=["u", "i"], projs="random")
    ur, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=["u", "r"], projs="random")
    ri, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=["r", "i"], projs="random")

    ri2 = ui - ur  # u - i - (u - r) = r - i
    print(array_equal_nan(ri, ri2))

    import pdb

    pdb.set_trace()


def checkInfallTime():
    """Check infall times."""
    from .util.helper import closest

    sP = simParams(res=1820, run="tng", redshift=0.0)
    subhaloID = 131059
    treeName = "SubLink"

    vicente_answer = 69  # snapshot
    kiyun_answer_gyr = 6.35  # lookback Gyr

    # load
    sh = sP.groupCatSingle(subhaloID=subhaloID)
    parent_halo = sP.groupCatSingle(haloID=sh["SubhaloGrNr"])

    sub_mpb = sP.loadMPB(subhaloID, treeName=treeName)
    parent_mpb = sP.loadMPB(parent_halo["GroupFirstSub"], treeName=treeName)

    ind_par, ind_sub = match(parent_mpb["SnapNum"], sub_mpb["SnapNum"])

    # distance
    parent_r200 = parent_mpb["Group_R_Crit200"][ind_par]
    parent_pos = parent_mpb["SubhaloPos"][ind_par, :]
    sub_pos = sub_mpb["SubhaloPos"][ind_sub, :]

    dist = sP.periodicDists(parent_pos, sub_pos)

    # snap <-> time
    snapnum = parent_mpb["SnapNum"][ind_par]
    redshift = sP.snapNumToRedshift(snap=snapnum)
    tlookback = sP.units.redshiftToLookbackTime(redshift)

    _, kiyun_answer_ind = closest(tlookback, kiyun_answer_gyr)
    kiyun_answer = snapnum[kiyun_answer_ind]
    vicente_answer_ind = np.where(snapnum == vicente_answer)[0]
    vicente_answer_gyr = tlookback[vicente_answer_ind]

    print(parent_mpb["SnapNum"].size, sub_mpb["SnapNum"].size, ind_par.size, ind_sub.size)
    print(snapnum[0:10])
    print("parent r200: ", parent_r200[0:10])
    print("parent pos: ", parent_pos[0:10, :])
    print("sub pos: ", sub_pos[0:10, :])

    # what is infall?
    x = dist / parent_r200

    w = np.where(x <= 1.0)[0]
    my_answer = snapnum[w.max()]
    my_answer_gyr = tlookback[w.max()]
    print("first snapshot inside r200: ", my_answer, " lookback: ", my_answer_gyr)

    # plot
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(121)

    ax.set_xlabel("Snapshot Number")
    ax.set_ylabel("Radius [ckpc/h]")
    ax.plot(snapnum, parent_r200, "-", label="parent r200")
    ax.plot(snapnum, dist, "-", label="parent to sub distance")
    ax.plot([vicente_answer, vicente_answer], [10, 2000], ":", label="Vicente Infall Time")
    ax.plot([kiyun_answer, kiyun_answer], [10, 2000], ":", label="Kiyun Infall Time")
    ax.plot([my_answer, my_answer], [10, 2000], ":", label="My Answer")
    ax.legend()

    ax = fig.add_subplot(122)

    ax.set_xlabel("Lookback Time [Gyr]")
    ax.set_ylabel("Radius [ckpc/h]")
    ax.plot(tlookback, parent_r200, "-", label="parent r200")
    ax.plot(tlookback, dist, "-", label="parent to sub distance")
    ax.plot([vicente_answer_gyr, vicente_answer_gyr], [10, 2000], ":", label="Vicente Infall Time")
    ax.plot([kiyun_answer_gyr, kiyun_answer_gyr], [10, 2000], ":", label="Kiyun Infall Time")
    ax.plot([my_answer_gyr, my_answer_gyr], [10, 2000], ":", label="My Answer")
    ax.legend()

    fig.savefig("check_%s_snap-%d_subhalo-%d.pdf" % (sP.simName, sP.snap, subhaloID))
    plt.close(fig)


def lagrangeMatching():
    """Test L75n1820TNG -> L75n1820FP matching."""
    sP = simParams(res=1820, run="tng", redshift=0.0)
    sP_illustris = simParams(res=1820, run="illustris", redshift=0.0)

    # load matching
    matchFilePath = sP.postPath + "/SubhaloMatchingToIllustris/"
    matchFileName = matchFilePath + "LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5" % sP.snap

    with h5py.File(matchFileName, "r") as f:
        inds_tng = f["SubhaloIndexFrom"][()]
        inds_illustris = f["SubhaloIndexTo"][()]
        scores = f["Score"][()]

    # get indices of TNG centrals
    cen_inds_tng = sP.cenSatSubhaloIndices(cenSatSelect="cen")

    # load stellar masses and positions
    mstar_illustris = sP_illustris.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
    mstar_illustris = sP_illustris.units.codeMassToLogMsun(mstar_illustris[:, 4])

    mstar_tng = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
    mstar_tng = sP.units.codeMassToLogMsun(mstar_tng[:, 4])

    pos_illustris = sP_illustris.groupCat(fieldsSubhalos=["SubhaloPos"])
    pos_tng = sP.groupCat(fieldsSubhalos=["SubhaloPos"])

    # print matches for TNG centrals 0-10, 100-110, 1000-1010
    doInds = list(range(0, 10)) + list(range(100, 110)) + list(range(1000, 1010))
    doInds = list(range(70, 95))

    for i in doInds:
        w = np.where(inds_tng == cen_inds_tng[i])[0]

        if not len(w):
            print(i, cen_inds_tng[i], "tng central not in catalog")
            continue

        assert len(w) == 1
        w = w[0]

        ind_tng = inds_tng[w]
        ind_illustris = inds_illustris[w]
        score = scores[w]

        print(
            i,
            ind_tng,
            ind_illustris,
            score,
            mstar_tng[ind_tng],
            mstar_illustris[ind_illustris],
            pos_illustris[ind_illustris, :],
            pos_tng[ind_tng, :],
        )


def checkSublinkIntermediateFiles():
    """Check _first* and _second* descendant links."""
    sP = simParams(res=2500, run="tng")
    subLinkPath = "/home/extdylan/data/sims.TNG/L205n2500TNG_temp/postprocessing/trees/SubLink/"
    snaps = sP.snapNumToRedshift(all=True)
    print("num snaps: %d" % snaps.size)

    nSubgroups = np.zeros(snaps.size, dtype="int64")

    print("get subgroup dimensions from actual run")
    for i in range(51):  # range(snaps.size):
        sP.setSnap(i)
        nSubgroups[i] = sP.groupCatHeader()["Nsubgroups_Total"]
        print(" [%2d] %d" % (i, nSubgroups[i]))

    print("verify sublink")
    for i in range(50):  # snaps.size):
        print(" [%2d]" % i)
        if path.isfile(subLinkPath + "_first_%03d.hdf5" % i):
            with h5py.File(subLinkPath + "_first_%03d.hdf5" % i) as f:
                first_size = f["DescendantIndex"].size
                first_desc_index_max = f["DescendantIndex"][()].max()
            if first_size != nSubgroups[i]:
                print(" FAIL _first_%03d.hdf5 does not correspond" % i)
            if i < snaps.size - 1:
                if first_desc_index_max >= nSubgroups[i + 1]:
                    print(" FAIL _first_%03d.hdf5 points to nonexistent sub" % i)
        else:
            print("  skip first missing")

        if path.isfile(subLinkPath + "_second_%03d.hdf5" % i):
            with h5py.File(subLinkPath + "_second_%03d.hdf5" % i) as f:
                second_size = f["DescendantIndex"].size
                second_desc_index_max = f["DescendantIndex"][()].max()
            if second_size != nSubgroups[i]:
                print(" FAIL _second_%03d.hdf5 does not correspond" % i)
            if i < snaps.size - 2:
                if second_desc_index_max >= nSubgroups[i + 2]:
                    print(" FAIL _second_%03d.hdf5 points to nonexistent sub" % i)
        else:
            print("  skip second missing")

    import pdb

    pdb.set_trace()


def domeTestData():
    """Write out test data files for planetarium vendors."""
    from datetime import datetime

    sP = simParams(res=1820, run="illustris", redshift=0.0)
    shFields = ["SubhaloPos", "SubhaloVel", "SubhaloMass", "SubhaloSFR"]

    gc = sP.groupCat(fieldsSubhalos=shFields)

    def _writeAttrs(f):
        # header
        h = f.create_group("Header")
        h.attrs["SimulationName"] = sP.simName
        h.attrs["SimulationRedshift"] = sP.redshift
        h.attrs["SimulationBoxSize"] = sP.boxSize
        h.attrs["SimulationRef"] = "http://www.illustris-project.org/api/" + sP.simName
        h.attrs["CreatedBy"] = "Dylan Nelson"
        h.attrs["CreatedOn"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # XDMF equivalent type metadata
        h.attrs["field_pos_x"] = "/SubhaloPos[:,0]"
        h.attrs["field_pos_y"] = "/SubhaloPos[:,1]"
        h.attrs["field_pos_z"] = "/SubhaloPos[:,2]"
        h.attrs["field_vel_x"] = "/SubhaloVel[:,0]"
        h.attrs["field_vel_y"] = "/SubhaloVel[:,1]"
        h.attrs["field_vel_z"] = "/SubhaloVel[:,2]"

        h.attrs["field_color_avail"] = "/SubhaloMass,/SubhaloSFR"
        h.attrs["field_color_default"] = "/SubhaloMass"
        h.attrs["field_color_default_min"] = 0.01
        h.attrs["field_color_default_max"] = 1000.0
        h.attrs["field_color_default_func"] = "log"

        # dataset attributes
        f["SubhaloPos"].attrs["Description"] = "Galaxy Position"
        f["SubhaloVel"].attrs["Description"] = "Galaxy Velocity"
        f["SubhaloMass"].attrs["Description"] = "Galaxy Total Mass"
        f["SubhaloSFR"].attrs["Description"] = "Galaxy Star Formation Rate"

        f["SubhaloPos"].attrs["Units"] = "ckpc/h"
        f["SubhaloVel"].attrs["Units"] = "km/s"
        f["SubhaloMass"].attrs["Units"] = "10^10 Msun/h"
        f["SubhaloSFR"].attrs["Units"] = "Msun/yr"

    def _writeFile(fileName, gc, shFields):
        f = h5py.File(fileName, "w")

        for key in shFields:
            f[key] = gc[key]
            f[key].attrs["Min"] = gc[key].min()
            f[key].attrs["Max"] = gc[key].max()
            f[key].attrs["Mean"] = gc[key].mean()

        _writeAttrs(f)
        f.close()

    # "10 million points" (all subhalos)
    if 1:
        fileName = "domeTestData_4million_%s_z%d.hdf5" % (sP.simName, sP.redshift)
        _writeFile(fileName, gc, shFields)

    # "1 million points" (10^9 halo mass cut)
    gcNew = {}
    gcNew = {}

    w = np.where(sP.units.codeMassToLogMsun(gc["SubhaloMass"]) >= 9.0)
    for key in shFields:
        if gc[key].ndim == 1:
            gcNew[key] = gc[key][w]
        else:
            gcNew[key] = np.zeros((len(w[0]), gc[key].shape[1]), dtype=gc[key].dtype)
            for i in range(gc[key].shape[1]):
                gcNew[key][:, i] = gc[key][w, i]

    if 1:
        fileName = "domeTestData_1million_%s_z%d.hdf5" % (sP.simName, sP.redshift)
        _writeFile(fileName, gcNew, shFields)


def checkIllustrisMetalRatioVsSolar():
    """Check corrupted GFM_Metals content vs solar expectation."""
    from .cosmo.cloudy import cloudyIon

    element = "O"
    ionNum = "VI"
    sP = simParams(res=910, run="tng", redshift=0.0)
    nBins = 400
    indRange = [0, 500000]

    ion = cloudyIon(sP, redshiftInterp=True)
    metal = sP.snapshotSubset("gas", "metal", indRange=indRange)

    metal_mass_fraction_1 = (metal / ion.solar_Z) * ion._solarMetalAbundanceMassRatio(element)
    metal_mass_fraction_2 = 1.0 * sP.snapshotSubset("gas", "metals_" + element, indRange=indRange)
    metal_mass_fraction_3 = ion._solarMetalAbundanceMassRatio(element)

    metal_1b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, solarAbunds=True)
    metal_2b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, solarAbunds=False)
    metal_3b = ion.calcGasMetalAbundances(
        sP, element, ionNum, indRange=indRange, solarAbunds=True, solarMetallicity=True
    )

    metal_mass_fraction_1 = np.log10(metal_mass_fraction_1)
    metal_mass_fraction_2 = np.log10(metal_mass_fraction_2)
    metal_mass_fraction_3 = np.log10(metal_mass_fraction_3)
    metal_1b = np.log10(metal_1b)
    metal_2b = np.log10(metal_2b)
    metal_3b = np.log10(metal_3b)

    # plot metal mass fractions
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlabel("log metal_mass_fraction")
    ax.set_ylabel(r"N$_{\rm tr}$")
    ax.set_yscale("log")

    plt.hist(metal_mass_fraction_1, nBins, facecolor="red", alpha=0.8)
    plt.hist(metal_mass_fraction_2, nBins, facecolor="green", alpha=0.8)
    plt.plot([metal_mass_fraction_3, metal_mass_fraction_3], [1e1, 1e4], color="blue", alpha=0.8)

    fig.savefig("checkIllustrisMetalRatioVsSolar_12.pdf")
    plt.close(fig)

    # plot metal ion mass fractions
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlabel("log metal_mass_fraction_in_ion")
    ax.set_ylabel(r"N$_{\rm tr}$")
    ax.set_yscale("log")

    plt.hist(metal_1b, nBins, facecolor="red", alpha=0.8)
    plt.hist(metal_2b, nBins, facecolor="green", alpha=0.8)
    plt.hist(metal_3b, nBins, facecolor="blue", alpha=0.8)

    fig.savefig("checkIllustrisMetalRatioVsSolar_34.pdf")
    plt.close(fig)


def checkTracerLoad():
    """Check new code to load tracers from snapshots."""
    # basePath = '/n/home07/dnelson/dev.prime/realizations/L25n32_trTest/output/'
    basePath = "/n/home07/dnelson/sims.zooms/128_20Mpc_h0_L9/output/"

    fieldsGroups = ["GroupMass", "GroupLenType", "GroupMassType", "GroupNsubs"]
    fieldsSubs = ["SubhaloMass", "SubhaloMassType", "SubhaloLenType"]

    fields = {
        "gas": ["Masses", "ParticleIDs"],
        "dm": ["Velocities", "ParticleIDs"],  # Potential in L25n32_trTest only, not sims.zooms
        "bhs": ["Masses", "ParticleIDs"],  # L25n32_trTest only, not sims.zooms
        "stars": ["Masses", "ParticleIDs"],
        "trmc": ["TracerID", "ParentID"],
    }

    parTypes = ["gas", "stars", "bhs"]

    # sim specifications
    class sP_old:
        snap = 50  # 4
        simPath = basePath
        run = "testing"
        trMCFields = None

    sP_new = sP_old()
    sP_new.snap = 99  # 5 # new version of snap4 moved to fake snap5

    # load group catalogs
    # gc_old = sP_old.groupCat(fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)
    gc_new = sP_new.groupCat(fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)

    # load snapshots
    h_new = sP_new.snapshotHeader()
    h_old = sP_old.snapshotHeader()
    assert (h_new["NumPart"] != h_old["NumPart"]).sum() == 0

    snap_old = {}
    snap_new = {}

    for ptName, fieldList in fields.items():
        # skip bhs or stars if none exist
        if h_new["NumPart"][partTypeNum(ptName)] == 0:
            continue

        snap_old[ptName] = {}
        snap_new[ptName] = {}

        for key in fieldList:
            snap_old[ptName][key] = sP_old.snapshotSubset(ptName, key)
            snap_new[ptName][key] = sP_new.snapshotSubset(ptName, key)

    # compare
    # assert gc_old['halos']['count'] == gc_new['halos']['count']
    # assert gc_old['subhalos']['count'] == gc_new['subhalos']['count']

    # for key in fieldsGroups:
    #    assert np.array_equal( gc_old['halos'][key], gc_new['halos'][key] )
    # for key in fieldsSubs:
    #    assert np.array_equal( gc_old['subhalos'][key], gc_new['subhalos'][key] )

    # check all particle type properties are same (including that same tracers have same parents)
    for ptName, fieldList in fields.items():
        idFieldName = "ParticleIDs" if ptName != "trmc" else "TracerID"

        if ptName not in snap_old:
            continue

        pt_sort_old = np.argsort(snap_old[ptName][idFieldName])
        pt_sort_new = np.argsort(snap_new[ptName][idFieldName])

        for key in fieldList:
            assert np.array_equal(snap_old[ptName][key][pt_sort_old], snap_new[ptName][key][pt_sort_new])

    # make offset tables for Groups/Subhalos by hand
    gc_new_off = {"halos": {}, "subhalos": {}}

    for tName in parTypes:
        tNum = partTypeNum(tName)
        shCount = 0

        gc_new_off["halos"][tName] = np.insert(np.cumsum(gc_new["halos"]["GroupLenType"][:, tNum]), 0, 0)
        gc_new_off["subhalos"][tName] = np.zeros(gc_new["subhalos"]["count"], dtype="int32")

        for k in range(gc_new["header"]["Ngroups_Total"]):
            if gc_new["halos"]["GroupNsubs"][k] == 0:
                continue

            gc_new_off["subhalos"][tName][shCount] = gc_new_off["halos"][tName][k]

            shCount += 1
            for _m in np.arange(1, gc_new["halos"]["GroupNsubs"][k]):
                gc_new_off["subhalos"][tName][shCount] = (
                    gc_new_off["subhalos"][tName][shCount - 1] + gc_new["subhalos"]["SubhaloLenType"][shCount - 1, tNum]
                )
                shCount += 1

    # new content (verify Group and Subhalo counts)
    gcSets = {"subhalos": "SubhaloLenType"}  # , 'halos':'GroupLenType' }

    for name1, name2 in gcSets.items():
        gc_new_totTr = gc_new[name1][name2][:, 3].sum()
        gc_new_count = 0

        if name1 == "halos":
            gcNumTot = gc_new["header"]["Ngroups_Total"]
        if name1 == "subhalos":
            gcNumTot = gc_new["header"]["Nsubgroups_Total"]
        if name1 == "halos":
            massName = "GroupMassType"
        if name1 == "subhalos":
            massName = "SubhaloMassType"

        for i in range(gcNumTot):
            locTrCount = 0
            savTrCount = gc_new[name1][name2][i, 3]

            # get indices and ids for group members (gas/bhs)
            for tName in parTypes:
                tNum = partTypeNum(tName)

                inds_type_start = gc_new_off[name1][tName][i]
                inds_type_end = inds_type_start + gc_new[name1][name2][i, tNum]

                if tName in snap_new:
                    ids_type = snap_new[tName]["ParticleIDs"][inds_type_start:inds_type_end]

                    # verify mass
                    mass_type = snap_new[tName]["Masses"][inds_type_start:inds_type_end]
                    assert np.abs(mass_type.sum() - gc_new[name1][massName][i, tNum]) < 1e-4

                    if ids_type.size == 0:
                        continue

                    # crossmatch member gas/stars/bhs to all ParentIDs of tracers
                    ia, ib = match(ids_type, snap_new["trmc"]["ParentID"])
                    if ia is not None:
                        locTrCount += ia.size

            gc_new_count += locTrCount

            # does the number of re-located children tracers equal the LenType value?
            print(name1, i, locTrCount, savTrCount)
            assert locTrCount == savTrCount

        print(name1, gc_new_totTr, gc_new_count)
        assert gc_new_totTr == gc_new_count


def enrichChecks():
    """Check GFM_WINDS_DISCRETE_ENRICHMENT comparison runs."""
    # config
    # sP1 = simParams(res=256, run='L12.5n256_discrete_dm0.0', redshift=0.0)
    ##sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.0001', redshift=0.0)
    # sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.00001', redshift=0.0)

    sP1 = simParams(res=1820, run="tng", redshift=0.0)
    sP2 = simParams(res=1820, run="illustris", redshift=0.0)

    nBins = 100  # 60 for 128, 100 for 256

    pdf = PdfPages("enrichChecks_" + sP1.simName + "_" + sP2.simName + ".pdf")

    # (1) - enrichment counter
    if 0:
        ec1 = sP1.snapshotSubset("stars", "GFM_EnrichCount")
        ec2 = sP2.snapshotSubset("stars", "GFM_EnrichCount")

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)

        ax.set_title("")
        ax.set_xlabel("Number of Enrichments per Star")
        ax.set_ylabel("N$_{\\rm stars}$")

        hRange = [0, max(ec1.max(), ec2.max())]
        plt.hist(ec1, nBins, range=hRange, facecolor="red", alpha=0.7, label=sP1.simName)
        plt.hist(ec2, nBins, range=hRange, facecolor="green", alpha=0.7, label=sP2.simName)

        ax.legend(loc="upper right")
        pdf.savefig()
        plt.close(fig)

    # (2) final stellar masses
    if 0:
        mstar1 = sP1.snapshotSubset("stars", "mass")
        mstar2 = sP2.snapshotSubset("stars", "mass")
        mstar1 = sP1.units.codeMassToLogMsun(mstar1)
        mstar2 = sP2.units.codeMassToLogMsun(mstar2)

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)

        ax.set_title("")
        ax.set_xlabel("Final Stellar Masses [ log M$_{\\rm sun}$ z=0 ]")
        ax.set_ylabel("N$_{\\rm stars}$")

        hRange = [min(mstar1.min(), mstar2.min()), max(mstar1.max(), mstar2.max())]
        plt.hist(mstar1, nBins, range=hRange, facecolor="red", alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor="green", alpha=0.7, label=sP2.simName)

        ax.plot([sP1.targetGasMass, sP1.targetGasMass], [1, 1e8], ":", color="black", alpha=0.7, label="target1")
        ax.plot([sP2.targetGasMass, sP2.targetGasMass], [1, 1e8], ":", color="black", alpha=0.7, label="target2")

        ax.legend(loc="upper right")
        pdf.savefig()
        plt.close(fig)

    # (2b) initial stellar masses
    if 1:
        mstar1 = sP1.snapshotSubset("stars", "mass_ini")
        mstar2 = sP2.snapshotSubset("stars", "mass_ini")
        mstar1 = np.log10(mstar1 / sP1.targetGasMass)
        mstar2 = np.log10(mstar2 / sP2.targetGasMass)

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)
        ax.set_yscale("log")

        ax.set_title("")
        ax.set_xlabel("Initial Stellar Masses / targetGasMass [ log z=0 ]")
        ax.set_ylabel("N$_{\\rm stars}$")

        hRange = [min(mstar1.min(), mstar2.min()), max(mstar1.max(), mstar2.max())]
        plt.hist(mstar1, nBins, range=hRange, facecolor="red", alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor="green", alpha=0.7, label=sP2.simName)

        ax.legend(loc="upper right")
        pdf.savefig()
        plt.close(fig)

    # (3) final gas metallicities
    if 0:
        zgas1 = sP1.snapshotSubset("gas", "GFM_Metallicity")
        zgas2 = sP2.snapshotSubset("gas", "GFM_Metallicity")
        zgas1 = np.log10(zgas1)
        zgas2 = np.log10(zgas2)

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)
        ax.set_yscale("log")

        ax.set_title("")
        ax.set_xlabel("Final Gas Metallicities [ log code z=0 ]")
        ax.set_ylabel("N$_{\\rm cells}$")

        hRange = [min(zgas1.min(), zgas2.min()), max(zgas1.max(), zgas2.max())]
        plt.hist(zgas1, nBins, range=hRange, facecolor="red", alpha=0.7, label=sP1.simName)
        plt.hist(zgas2, nBins, range=hRange, facecolor="green", alpha=0.7, label=sP2.simName)

        ax.legend(loc="upper right")
        pdf.savefig()
        plt.close(fig)

    # (4) final/initial stellar masses
    if 0:
        mstar1_final = sP1.snapshotSubset("stars", "mass")
        mstar2_final = sP2.snapshotSubset("stars", "mass")
        mstar1_ini = sP1.snapshotSubset("stars", "mass_ini")
        mstar2_ini = sP2.snapshotSubset("stars", "mass_ini")

        ratio1 = mstar1_final / mstar1_ini
        ratio2 = mstar2_final / mstar2_ini

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)
        ax.set_yscale("log")

        ax.set_title("")
        ax.set_xlabel("(Final / Initial) Stellar Masses [ z=0 ]")
        ax.set_ylabel("N$_{\\rm stars}$")

        hRange = [min(ratio1.min(), ratio2.min()), max(ratio1.max(), ratio2.max())]
        plt.hist(ratio1, nBins, range=hRange, facecolor="red", alpha=0.7, label=sP1.simName)
        plt.hist(ratio2, nBins, range=hRange, facecolor="green", alpha=0.7, label=sP2.simName)

        ax.legend(loc="upper right")
        pdf.savefig()
        plt.close(fig)

    pdf.close()


def checkMusic():
    """Check MUSIC initial conditions splitting."""
    import illustris_python as il

    basePath = "/n/home07/dnelson/sims.zooms2/ICs/fullbox/output/"
    fileBase = "ics_2048"  #'ics'
    gName = "PartType1"
    hKeys = ["NumPart_ThisFile", "NumPart_Total", "NumPart_Total_HighWord"]

    # load parent
    print("Parent:\n")

    with h5py.File(basePath + fileBase + "_temp.hdf5", "r") as f:
        # header
        for hKey in hKeys:
            print(" ", hKey, f["Header"].attrs[hKey], f["Header"].attrs[hKey].dtype)

        nPart = il.snapshot.getNumPart(f["Header"].attrs)
        print("  nPart: ", nPart)

        # datasets
        for key in f[gName].keys():
            print(" ", key, f[gName][key].shape, f[gName][key].dtype)

    # load split
    print("\n---")
    nPartSum = np.zeros(6, dtype="int64")

    files = sorted(glob.glob(basePath + fileBase + ".*.hdf5"))
    for file in files:
        print("\n" + file)

        with h5py.File(file) as f:
            # header
            for hKey in hKeys:
                print(" ", hKey, f["Header"].attrs[hKey], f["Header"].attrs[hKey].dtype)

            nPart = il.snapshot.getNumPart(f["Header"].attrs)
            print("  nPart: ", nPart)
            nPartSum += f["Header"].attrs["NumPart_ThisFile"]

            # datasets
            for key in f[gName].keys():
                print(" ", key, f[gName][key].shape, f[gName][key].dtype)

    print("\n nPartSum: ", nPartSum, "\n")

    # compare data
    parent = {}
    children = {}
    dsets = ["ParticleIDs", "Coordinates", "Velocities"]

    for key in dsets:
        print(key)

        with h5py.File(basePath + fileBase + "_temp.hdf5", "r") as f:
            print("parent load: ", f[gName][key].shape, f[gName][key].dtype)
            parent[key] = f[gName][key][:]

        for file in files:
            print(file)
            with h5py.File(file) as f:
                if key not in children:
                    children[key] = f[gName][key][:]
                else:
                    children[key] = np.concatenate((children[key], f[gName][key][:]), axis=0)

        print(key, parent[key].shape, children[key].shape, parent[key].dtype, children[key].dtype)
        print("", np.allclose(parent[key], children[key]), np.array_equal(parent[key], children[key]))

        parent = {}
        children = {}
