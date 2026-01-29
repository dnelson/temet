"""
Definitions of custom catalog fields, based on postprocessing/ datasets.
"""

from os.path import isfile

import h5py
import numpy as np

from .groupcat import catalog_field, groupOrderedValsToSubhaloOrdered


# ---------------------------- postprocessing/stellarassembly -----------------------------------------------


@catalog_field(aliases=["massfrac_exsitu2", "massfrac_insitu", "massfrac_insitu2"])
def massfrac_exsitu(sim, field):
    """Postprocessing/StellarAssembly: ex-situ or in-situ stellar mass fraction.
    Within the stellar half mass radius, unless '2' in field name, in which case within 2*rhalf."""
    inRadStr = "_in_rad" if "2" in field else ""
    filePath = sim.postPath + "/StellarAssembly/galaxies%s_%03d.hdf5" % (inRadStr, sim.snap)

    dNameNorm = "StellarMassTotal"
    dNameMass = "StellarMassInSitu" if "_insitu" in field else "StellarMassExSitu"

    if isfile(filePath):
        with h5py.File(filePath, "r") as f:
            mass_type = f[dNameMass][()]
            mass_norm = f[dNameNorm][()]

        # take fraction and set Mstar=0 cases to nan silently
        wZeroMstar = np.where(mass_norm == 0.0)
        wNonzeroMstar = np.where(mass_norm > 0.0)

        vals = mass_type
        vals[wNonzeroMstar] /= mass_norm[wNonzeroMstar]
        vals[wZeroMstar] = np.nan
    else:
        print("WARNING: [%s] does not exist, empty return." % filePath)
        vals = np.zeros(sim.numSubhalos, dtype="float32")
        vals.fill(np.nan)

    return vals


massfrac_exsitu.label = lambda sim, f: r"%s Stellar Mass Fraction" % ("Ex-Situ" if "_exsitu" in f else "In-Situ")
massfrac_exsitu.units = ""  # linear dimensionless
massfrac_exsitu.limits = [0.0, 1.0]
massfrac_exsitu.log = False

# ---------------------------- postprocessing/mergerhistory -----------------------------------------------


@catalog_field(multi="num_mergers_", alias="num_mergers")
def num_mergers_(sim, field):
    """Postprocessing/MergerHistory: number of major/minor mergers, within different time ranges."""
    # num_mergers, num_mergers_{major,minor}, num_mergers_{major,minor}_{250myr,500myr,gyr,z1,z2}
    filePath = sim.postPath + "/MergerHistory/MergerHistory_%03d.hdf5" % (sim.snap)

    if not isfile(filePath):
        filePath = sim.postPath + "/MergerHistory/merger_history_%03d.hdf5" % (sim.snap)

    typeStr = ""
    timeStr = "Total"

    if "_minor" in field:
        typeStr = "Minor"  # 1/10 < mu < 1/4
    if "_major" in field:
        typeStr = "Major"  # mu > 1/4
    if "_250myr" in field:
        timeStr = "Last250Myr"
    if "_500myr" in field:
        timeStr = "Last500Myr"
    if "_gyr" in field:
        timeStr = "LastGyr"
    if "_z1" in field:
        timeStr = "SinceRedshiftOne"
    if "_z2" in field:
        timeStr = "SinceRedshiftTwo"

    fieldLoad = "Num%sMergers%s" % (typeStr, timeStr)
    if field == "mergers_mean_fgas":
        fieldLoad = "MeanGasFraction"
    if field == "mergers_mean_z":
        fieldLoad = "MeanRedshift"
    if field == "mergers_mean_mu":
        fieldLoad = "MeanMassRatio"

    if isfile(filePath):
        with h5py.File(filePath, "r") as f:
            vals = f[fieldLoad][()].astype("float32")  # uint32 for counts

        w = np.where(vals == -1)
        if len(w[0]):
            vals[w] = np.nan
    else:
        print("WARNING: [%s] does not exist, empty return." % filePath)
        vals = np.zeros(sim.numSubhalos, dtype="float32")
        vals.fill(np.nan)

    return vals


num_mergers_.label = lambda sim, f: r"Number of Mergers (%s)" % ("-".join(f.split("_")[2:]))
num_mergers_.units = ""  # linear dimensionless
num_mergers_.limits = [0, 10]
num_mergers_.log = False


@catalog_field
def mergers_mean_fgas(sim, field):
    """Postprocessing/MergerHistory: mean property ('cold' i.e. star-forming gas fraction) of mergers.
    Weighted by the maximum stellar mass of the secondary progenitors."""
    filePath = sim.postPath + "/MergerHistory/MergerHistory_%03d.hdf5" % (sim.snap)

    with h5py.File(filePath, "r") as f:
        vals = f["MeanGasFraction"][()]
        vals[vals == -1] = np.nan

    return vals


mergers_mean_fgas.label = "Mean Gas Fraction of Mergers"
mergers_mean_fgas.units = ""  # linear dimensionless
mergers_mean_fgas.limits = [-2.0, 0.0]
mergers_mean_fgas.log = True


@catalog_field
def mergers_mean_z(sim, field):
    """Postprocessing/MergerHistory: mean property (redshift) of all mergers this subhalo gas undergone.
    Weighted by the maximum stellar mass of the secondary progenitors."""
    filePath = sim.postPath + "/MergerHistory/MergerHistory_%03d.hdf5" % (sim.snap)

    with h5py.File(filePath, "r") as f:
        vals = f["MeanRedshift"][()]
        vals[vals == -1] = np.nan

    return vals


mergers_mean_z.label = "Redshift"
mergers_mean_z.units = ""  # linear dimensionless
mergers_mean_z.limits = [0.0, 6.0]
mergers_mean_z.log = False


@catalog_field
def mergers_mean_mu(sim, field):
    """Postprocessing/MergerHistory: mean property (stellar mass ratio) of mergers.
    Weighted by the maximum stellar mass of the secondary progenitors."""
    filePath = sim.postPath + "/MergerHistory/MergerHistory_%03d.hdf5" % (sim.snap)

    with h5py.File(filePath, "r") as f:
        vals = f["MeanMassRatio"][()]
        vals[vals == -1] = np.nan

    return vals


mergers_mean_mu.label = "Mean Stellar Mass Ratio of Mergers"
mergers_mean_mu.units = ""  # linear dimensionless
mergers_mean_mu.limits = [0.0, 1.0]
mergers_mean_mu.log = False

# ---------------------------- postprocessing/lgalaxies -----------------------------------------------


@catalog_field(multi="lgal_")
def lgal_(sim, field):
    """Postprocessing/L-Galaxies: (H15) run on dark matter only analog, automatically cross-matched to the
    TNG run such that return has the same shape as sP.numSubhalos (unmatched TNG subs = NaN).
    Examples: LGal_StellarMass, LGal_HotGasMass, LGal_Type, LGal_XrayLum, ...
    Note: if '_orig' appended, e.g. LGal_StellarMass_orig, then no matching, full LGal return."""
    fieldName = field.split("_")[1]
    filePath = sim.postPath + "/LGalaxies/LGalaxies_%03d.hdf5" % sim.snap

    if isfile(filePath):
        # load
        with h5py.File(filePath, "r") as f:
            # find field with capitalized name
            for key in f["Galaxy"].keys():
                if key.lower() == fieldName.lower():
                    fieldName = key
                    break

            data = f["/Galaxy/%s/" % fieldName][()]
            if "_orig" not in field:
                if "_dark" in field:
                    match_ids = f["Galaxy/SubhaloIndex_TNG-Dark"][()]
                    numSubhalos = sim.dmoBox.numSubhalos
                else:
                    match_ids = f["Galaxy/SubhaloIndex_TNG"][()]
                    numSubhalos = sim.numSubhalos

        # optionally cross-match
        if "_orig" in field:
            vals = data
        else:
            w = np.where(match_ids >= 0)
            shape = [numSubhalos] if data.ndim == 1 else [numSubhalos, data.shape[1]]
            vals = np.zeros(shape, dtype=data.dtype)

            if data.dtype in ["float32", "float64"]:
                vals.fill(np.nan)
            else:
                vals.fill(-1)  # Len, DisruptOn, Type

            vals[match_ids[w]] = data[w]
    else:
        print("WARNING: [%s] does not exist, empty return." % filePath)
        vals = np.zeros(sim.numSubhalos, dtype="float32")
        vals.fill(np.nan)

    return vals


lgal_.label = lambda sim, f: r"L-Galaxies (%s)" % (f.split("_", max=1)[1])
lgal_.units = ""  # variable (todo)
lgal_.limits = []  # variable (todo)
lgal_.log = False  # variable (todo)

# ---------------------------- postprocessing/coolcore_criteria -----------------------------------------------


def _coolcore_load(sim, field):
    """Helper function to load coolcore_criteria data."""
    filePath = sim.postPath + "/released/coolcore_criteria.hdf5"

    with h5py.File(filePath, "r") as f:
        HaloIDs = f["HaloIDs"][()]
        data = f[field][:, sim.snap]

    # expand from value per primary target to value per subhalo
    vals = np.zeros(sim.numSubhalos, dtype="float32")
    vals.fill(np.nan)

    vals[sim.halos("GroupFirstSub")[HaloIDs]] = data

    return vals


@catalog_field
def coolcore_flag(sim, field):
    """Postprocessing/coolcore_criteria: flag (0=SCC, 1=WCC, 2=NCC) based on Lehle+24 tcool fiducial definition."""
    return _coolcore_load(sim, "centralCoolingTime_flag")


coolcore_flag.label = "Cool-core Flag (0=CC, 1=WCC, 2=NCC)"
coolcore_flag.units = ""  # linear dimensionless
coolcore_flag.limits = [0.0, 2.0]
coolcore_flag.log = False


@catalog_field(alias="tcool0")
def coolcore_tcool(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 central cooling time."""
    return _coolcore_load(sim, "centralCoolingTime")


coolcore_tcool.label = r"Central $t_{\rm cool}$"
coolcore_tcool.units = "Gyr"
coolcore_tcool.limits = [0.0, 10.0]
coolcore_tcool.log = False


@catalog_field(alias="K0")
def coolcore_entropy(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 central cooling time."""
    return _coolcore_load(sim, "centralEntropy")


coolcore_entropy.label = r"Central $K_0$"
coolcore_entropy.units = "keV cm$^2$"
coolcore_entropy.limits = [1.0, 2.5]
coolcore_entropy.log = True


@catalog_field
def coolcore_ne(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 central electron number density."""
    return _coolcore_load(sim, "centralNumDens")


coolcore_ne.label = r"Central $n_e$"
coolcore_ne.units = "cm$^{-3}$"
coolcore_ne.limits = [-3.0, 1.0]
coolcore_ne.log = True


@catalog_field
def coolcore_ne_slope(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 central slope of number density."""
    return _coolcore_load(sim, "slopeNumDens")


coolcore_ne_slope.label = r"n_{\rm e} slope ($\alpha$"
coolcore_ne_slope.units = ""  # linear dimensionless
coolcore_ne_slope.limits = [0.0, 1.0]
coolcore_ne_slope.log = False


@catalog_field
def coolcore_c_phys(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 X-ray concentration (40kpc vs 400kpc), physical."""
    return _coolcore_load(sim, "concentrationPhys")


coolcore_c_phys.label = r"C_{\rm phys}"
coolcore_c_phys.units = ""  # linear dimensionless
coolcore_c_phys.limits = [0.0, 1.0]
coolcore_c_phys.log = False


@catalog_field
def coolcore_c_scaled(sim, field):
    """Postprocessing/coolcore_criteria: Lehle+24 X-ray concentration (40kpc vs 400kpc), scaled."""
    return _coolcore_load(sim, "concentrationScaled")


coolcore_c_scaled.label = r"C_{\rm phys}"
coolcore_c_scaled.units = ""  # linear dimensionless
coolcore_c_scaled.limits = [0.0, 1.0]
coolcore_c_scaled.log = False


@catalog_field(aliases=["peakoffset_xray_x", "peakoffset_xray_y", "peakoffset_xray_z"])
def peakoffset_xray(sim, field):
    """Postprocessing/released: Nelson+24 offsets of X-ray peaks [pkpc]."""
    filePath = sim.postPath + "/released/XrayOffsets_%03d.hdf5" % sim.snap

    with h5py.File(filePath, "r") as f:
        data = f["Subhalo_XrayOffset_2D"][()]

    # convert code lengths to pkpc
    data = sim.units.codeLengthToKpc(data)

    # expand from value per primary target to value per subhalo
    pri_target = sim.halos("GroupPrimaryZoomTarget")
    HaloIDs = np.where(pri_target == 1)[0]
    assert HaloIDs.size == data.shape[0]

    vals = np.zeros(sim.numSubhalos, dtype="float32")
    vals.fill(np.nan)

    # choose viewing direction
    xyz_index = {"xray": 0, "x": 0, "y": 1, "z": 2}[field.split("_")[-1]]
    data = data[:, xyz_index]

    vals[sim.halos("GroupFirstSub")[HaloIDs]] = data

    return vals


peakoffset_xray.label = r"$\Delta x_{\rm X-ray}$"
peakoffset_xray.units = r"$\rm{kpc}$"
peakoffset_xray.limits = [-1.5, 2.5]
peakoffset_xray.log = True


@catalog_field(aliases=["peakoffset_sz_x", "peakoffset_sz_y", "peakoffset_sz_z"])
def peakoffset_sz(sim, field):
    """Postprocessing/released: Nelson+24 offsets of SZ peaks [pkpc]."""
    filePath = sim.postPath + "/released/SZOffsets_%03d.hdf5" % sim.snap

    with h5py.File(filePath, "r") as f:
        data = f["Subhalo_SZOffset_2D"][()]

    # convert code lengths to pkpc
    data = sim.units.codeLengthToKpc(data)

    # expand from value per primary target to value per subhalo
    pri_target = sim.halos("GroupPrimaryZoomTarget")
    HaloIDs = np.where(pri_target == 1)[0]
    assert HaloIDs.size == data.shape[0]

    vals = np.zeros(sim.numSubhalos, dtype="float32")
    vals.fill(np.nan)

    # choose viewing direction
    xyz_index = {"xray": 0, "x": 0, "y": 1, "z": 2}[field.split("_")[-1]]
    data = data[:, xyz_index]

    vals[sim.halos("GroupFirstSub")[HaloIDs]] = data

    return vals


peakoffset_sz.label = r"$\Delta x_{\rm SZ}$"
peakoffset_sz.units = r"$\rm{kpc}$"
peakoffset_sz.limits = [-1.5, 2.5]
peakoffset_sz.log = True

# ---------------------------- postprocessing/circularities -----------------------------------------------


@catalog_field(aliases=["fcirc", "fcirc_10re_eps07o"])
def fcirc_all_eps07o(sim, field):
    """Postprocessing/circularities/: fraction of disk-stars (circularity > 0.7), all stars in subhalo."""
    sel_str = "allstars" if "all_" in field else "10Re"
    basePath = sim.postPath + "/circularities/circularities_aligned_%s" % sel_str
    dName = "CircAbove07Frac"

    filePath = basePath + "_L75n1820TNG%03d.hdf5" % sim.snap

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["done"][()])
        vals = np.squeeze(f[dName][()])

    # for unprocessed subgroups, replace values with NaN
    print(" [%s] keeping only %d of %d non-NaN (done==1)" % (field, len(np.where(done == 1)[0]), vals.size))
    vals[done == 0] = np.nan

    return vals


fcirc_all_eps07o.label = r"$\rm{f_{circ, \epsilon > 0.7}}$"
fcirc_all_eps07o.units = ""  # linear dimensionless
fcirc_all_eps07o.limits = [0.0, 0.8]
fcirc_all_eps07o.log = False


@catalog_field(aliases=["fcirc_10re_eps07m"])
def fcirc_all_eps07m(sim, field):
    """Postprocessing/circularities/: fraction of disk-stars (circularity > 0.7), all stars in subhalo."""
    sel_str = "allstars" if "all_" in field else "10Re"
    basePath = sim.postPath + "/circularities/circularities_aligned_%s" % sel_str
    dName = "CircAbove07MinusBelowNeg07Frac"

    filePath = basePath + "_L75n1820TNG%03d.hdf5" % sim.snap

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["done"][()])
        vals = np.squeeze(f[dName][()])

    # for unprocessed subgroups, replace values with NaN
    print(" [%s] keeping only %d of %d non-NaN (done==1)" % (field, len(np.where(done == 1)[0]), vals.size))
    vals[done == 0] = np.nan

    return vals


fcirc_all_eps07m.label = r"$\rm{f_{circ, \epsilon > 0.7 - \epsilon < -0.7}}$"
fcirc_all_eps07m.units = ""  # linear dimensionless
fcirc_all_eps07m.limits = [0.0, 0.8]
fcirc_all_eps07m.log = False

# ---------------------------- postprocessing/galskitkinematics -----------------------------------------------


@catalog_field
def slit_vrot_halpha(sim, field):
    """Postprocessing/galskitkinematics/: Disk rotation velocity based on H-alpha emission."""
    basePath = sim.postPath + "/galslitkinematics/"
    filePath = basePath + "Subhalo_Halpha_slitKinematics_%03d.hdf5" % sim.snap
    dName = "Halpha_in_InRad_V_max_kms"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


slit_vrot_halpha.label = r"$V_{\rm rot, H\alpha}$"
slit_vrot_halpha.units = "km/s"
slit_vrot_halpha.limits = [50, 400]
slit_vrot_halpha.log = False


@catalog_field
def slit_vrot_stars(sim, field):
    """Postprocessing/galskitkinematics/: Disk rotation velocity based on stellar light."""
    basePath = sim.postPath + "/galslitkinematics/"
    filePath = basePath + "Subhalo_BuserVLum_slitKinematics_%03d.hdf5" % sim.snap
    dName = "BuserVLum_in_InRad_V_max_kms"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


slit_vrot_stars.label = r"$V_{\rm rot, V-band}$"
slit_vrot_stars.units = "km/s"
slit_vrot_stars.limits = [50, 400]
slit_vrot_stars.log = False


@catalog_field
def slit_vsigma_halpha(sim, field):
    """Postprocessing/galskitkinematics/: Disk velocity dispersion based on H-alpha emission."""
    basePath = sim.postPath + "/galslitkinematics/"
    filePath = basePath + "Subhalo_Halpha_slitKinematics_%03d.hdf5" % sim.snap
    dName = "Halpha_in_InRad_sigmaV_HalfRad2Rad_kms"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


slit_vsigma_halpha.label = r"$\sigma_{\rm vel, los, H\alpha}$"
slit_vsigma_halpha.units = "km/s"
slit_vsigma_halpha.limits = [0, 100]
slit_vsigma_halpha.log = False


@catalog_field
def slit_vsigma_stars(sim, field):
    """Postprocessing/galskitkinematics/: Disk velocity dispersion based on stellar light."""
    basePath = sim.postPath + "/galslitkinematics/"
    filePath = basePath + "Subhalo_BuserVLum_slitKinematics_%03d.hdf5" % sim.snap
    dName = "BuserVLum_in_InRad_sigmaV_HalfRad2Rad_kms"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


slit_vsigma_stars.label = r"$\sigma_{\rm vel, los, V-band}$"
slit_vsigma_stars.units = "km/s"
slit_vsigma_stars.limits = [0, 100]
slit_vsigma_stars.log = False


@catalog_field
def slit_voversigma_halpha(sim, field):
    """Postprocessing/galskitkinematics/: Disk rotational support (V/sigma) based on H-alpha emission."""
    vrot = sim.subhalos(field.replace("_voversigma", "_vrot"))
    sigma = sim.subhalos(field.replace("_voversigma", "_vsigma"))

    with np.errstate(invalid="ignore"):
        vals = vrot
        w = np.where(sigma > 0.0)
        vals[w] /= sigma[w]
        w = np.where(sigma == 0.0)
        vals[w] = np.nan

    return vals


slit_voversigma_halpha.label = r"$V_{\rm rot}$ / $\sigma_{\rm vel,los}$ [H-alpha]"
slit_voversigma_halpha.units = ""  # linear dimensionless
slit_voversigma_halpha.limits = [0, 12]
slit_voversigma_halpha.log = False


@catalog_field
def slit_voversigma_stars(sim, field):
    """Postprocessing/galskitkinematics/: Disk rotational support (V/sigma) based on stellar light."""
    vrot = sim.subhalos(field.replace("_voversigma", "_vrot"))
    sigma = sim.subhalos(field.replace("_voversigma", "_vsigma"))

    with np.errstate(invalid="ignore"):
        vals = vrot
        w = np.where(sigma > 0.0)
        vals[w] /= sigma[w]
        w = np.where(sigma == 0.0)
        vals[w] = np.nan

    return vals


slit_voversigma_stars.label = r"$V_{\rm rot}$ / $\sigma_{\rm vel,los}$ [V-band]"
slit_voversigma_stars.units = ""  # linear dimensionless
slit_voversigma_stars.limits = [0, 6]
slit_voversigma_stars.log = False

# ---------------------------- postprocessing/galsizes -----------------------------------------------


@catalog_field
def size2d_halpha(sim, field):
    """Postprocessing/galsizes/: Galaxy disk size (from 2D projection) based on H-alpha emission."""
    basePath = sim.postPath + "/galsizes/"
    filePath = basePath + "Subhalo_Sizes_GalProjs_%03d.hdf5" % sim.snap
    dName = "Halpha_HalfLightRadii_pkpc_2D_GalProjs_in_all_12"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


size2d_halpha.label = r"Projected 2D Size r$_{\rm H-alpha,1/2}$"
size2d_halpha.units = "kpc"
size2d_halpha.limits = [0, 1.2]
size2d_halpha.log = True


@catalog_field
def size2d_stars(sim, field):
    """Postprocessing/galsizes/: Galaxy disk size (from 2D projection) based on stellar light."""
    basePath = sim.postPath + "/galsizes/"
    filePath = basePath + "Subhalo_Sizes_GalProjs_%03d.hdf5" % sim.snap
    dName = "BuserVLum_HalfLightRadii_pkpc_2D_GalProjs_in_all_12"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


size2d_stars.label = r"Projected 2D Size r$_{\rm V-band,1/2}$"
size2d_stars.units = "kpc"
size2d_stars.limits = [0, 1.2]
size2d_stars.log = True


@catalog_field
def diskheight2d_halpha(sim, field):
    """Postprocessing/galsizes/: Galaxy disk scale-height (from 2D projection) based on H-alpha emission."""
    basePath = sim.postPath + "/galsizes/"
    filePath = basePath + "Subhalo_DiskHeights_GalProjs_%03d.hdf5" % sim.snap
    dName = "Halpha_HalfLightDiskHeights_pkpc_2D_GalProjs_in_InRad_13"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


diskheight2d_halpha.label = r"Projected 2D Disk Height h$_{\rm H-alpha,1/2}$"
diskheight2d_halpha.units = "kpc"
diskheight2d_halpha.limits = [-1.0, 0.2]
diskheight2d_halpha.log = True


@catalog_field
def diskheight2d_stars(sim, field):
    """Postprocessing/galsizes/: Galaxy disk scale-height (from 2D projection) based on stellar light."""
    basePath = sim.postPath + "/galsizes/"
    filePath = basePath + "Subhalo_DiskHeights_GalProjs_%03d.hdf5" % sim.snap
    dName = "BuserVLum_HalfLightDiskHeights_pkpc_2D_GalProjs_in_InRad_13"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        done = np.squeeze(f["/Subhalo/Done"][()])
        vals = np.squeeze(f["/Subhalo/" + dName][()])

    # for unprocessed subgroups, replace values with NaN
    w = np.where((done != 1) | (vals == -999))
    vals[w] = np.nan

    assert sim.numSubhalos == vals.size == vals.shape[0]


diskheight2d_stars.label = r"Projected 2D Disk Height h$_{\rm V-band,1/2}$"
diskheight2d_stars.units = "kpc"
diskheight2d_stars.limits = [-1.0, 0.2]
diskheight2d_stars.log = True


@catalog_field
def diskheightnorm2d_halpha(sim, field):
    """Postprocessing/galsizes/: Galaxy disk normalized scale-height (h/r) based on H-alpha emission."""
    height = sim.subhalos(field.replace("diskheightnorm2d_", "diskheight2d_"))
    size = sim.subhalos(field.replace("diskheightnorm2d_", "size2d_"))

    with np.errstate(invalid="ignore"):
        vals = height
        w = np.where(size > 0.0)
        vals[w] /= size[w]
        w = np.where(size == 0.0)
        vals[w] = np.nan

    return vals


diskheightnorm2d_halpha.label = r"Normalized Disk Height $\rm{(h/r)_{H-alpha}}$"
diskheightnorm2d_halpha.units = ""  # linear dimensionless
diskheightnorm2d_halpha.limits = [0.0, 0.9]
diskheightnorm2d_halpha.log = False


@catalog_field
def diskheightnorm2d_stars(sim, field):
    """Postprocessing/galsizes/: Galaxy disk normalized scale-height (h/r) based on stellar light."""
    height = sim.subhalos(field.replace("diskheightnorm2d_", "diskheight2d_"))
    size = sim.subhalos(field.replace("diskheightnorm2d_", "size2d_"))

    with np.errstate(invalid="ignore"):
        vals = height
        w = np.where(size > 0.0)
        vals[w] /= size[w]
        w = np.where(size == 0.0)
        vals[w] = np.nan

    return vals


diskheightnorm2d_stars.label = r"Normalized Disk Height $\rm{(h/r)_{V-band}}$"
diskheightnorm2d_stars.units = ""  # linear dimensionless
diskheightnorm2d_stars.limits = [0.0, 0.9]
diskheightnorm2d_stars.log = False

# ---------------------------- data.files/pillepich/ -----------------------------------------------


@catalog_field
def mstar_out_10kpc(sim, field):
    """data.files/pillepich/: Stellar mass outside of 10 kpc."""
    basePath = sim.derivPath + "/pillepich/"
    filePath = basePath + "Group_StellarMasses_%03d.hdf5" % sim.snap
    dName = "Group/Masses_stars_NoWt_sum_out_r10kpc"

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        vals_group = np.squeeze(f[dName][()])  # in Msun, by group

    vals = groupOrderedValsToSubhaloOrdered(vals_group)

    return vals


mstar_out_10kpc.label = r"Stellar Mass (r > 10 pkpc)"
mstar_out_10kpc.units = "Msun"
mstar_out_10kpc.limits = [6.0, 12.0]
mstar_out_10kpc.log = True

# ---------------------------- tracers ------------------------------------------------------


def _tr_load(sim, quant, filePath, fieldName):
    """Helper function to load tracer tracks auxcat data."""
    from ..tracer.evolution import ACCMODES
    from ..tracer.montecarlo import defParPartTypes

    ACCMODES["ALL"] = len(ACCMODES)  # add 'all' mode last
    defParPartTypes.append("all")  # add 'all' parent type last

    quant = quant[3:]  # remove 'tr_'
    mode = "all"  # default unless specified
    par = "all"  # default unless specified
    if "mode=" in quant and "par=" in quant:
        assert quant.index("mode=") <= quant.index("par=")  # parType request must be second

    if "par=" in quant:
        par = quant.split("par=")[1].split("_")[0]
        quant = quant.split("_par=")[0]
    if "mode=" in quant:
        mode = quant.split("mode=")[1].split("_")[0]
        quant = quant.split("_mode=")[0]

    assert mode.upper() in ACCMODES.keys() and par in defParPartTypes
    modeInd = ACCMODES.keys().index(mode.upper())
    parInd = defParPartTypes.index(par)

    with h5py.File(filePath, "r") as f:
        # load data
        vals = f[fieldName][:, parInd, modeInd]

    return vals


@catalog_field(multi="tr_zacc_mean_", alias="tr_zacc_mean")
def tr_zacc_mean_(sim, field):
    """Tracer tracks quantity: mean accretion redshift. (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_zAcc_mean"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    return vals


tr_zacc_mean_.label = r"Tracer Mean z$_{\rm acc}$"
tr_zacc_mean_.units = ""  # linear dimensionless
tr_zacc_mean_.limits = [0, 3.5]
tr_zacc_mean_.log = False


@catalog_field(multi="tr_zacc_mean_over_zform_", alias="tr_zacc_mean_over_zform")
def tr_zacc_mean_over_zform_(sim, field):
    """Tracer tracks quantity: mean accretion redshift normalized by halo formation redshift.
    (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_zAcc_mean_over_zForm"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    # normalize
    acField = "Subhalo_SubLink_zForm_mm5"
    vals /= sim.auxCat(fields=[acField])[acField]

    return vals


tr_zacc_mean_over_zform_.label = r"Tracer Mean z$_{\rm acc}$ / z$_{\rm form,halo}$"
tr_zacc_mean_over_zform_.units = ""  # linear dimensionless
tr_zacc_mean_over_zform_.limits = [0, 3.5]
tr_zacc_mean_over_zform_.log = False


@catalog_field(multi="tr_dthalo_mean_", alias="tr_dthalo_mean")
def tr_dthalo_mean_(sim, field):
    """Tracer tracks quantity: mean halo-crossing time. (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_dtHalo_mean"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    return vals


tr_dthalo_mean_.label = r"Tracer Mean $\Delta {\rm t}_{\rm halo}$"
tr_dthalo_mean_.units = r"Gyr"
tr_dthalo_mean_.limits = [-0.2, 0.6]
tr_dthalo_mean_.log = True


@catalog_field(multi="tr_angmom_tacc_", alias="tr_angmom_tacc")
def tr_angmom_tacc_(sim, field):
    """Tracer tracks quantity: angular momentum at accretion time (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_angMom_tAcc"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    return vals


tr_angmom_tacc_.label = r"Tracer Mean j$_{\rm spec}$ at $t_{\rm acc}$"
tr_angmom_tacc_.units = r"kpc km/s"
tr_angmom_tacc_.limits = [3.0, 5.0]
tr_angmom_tacc_.log = False  # auxCat() angmom vals are in log


@catalog_field(multi="tr_entr_tacc_", alias="tr_entr_tacc")
def tr_entr_tacc_(sim, field):
    """Tracer tracks quantity: entropy at accretion time (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_entr_tAcc"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    return vals


tr_entr_tacc_.label = r"Tracer Mean S$_{\rm gas}$ at $t_{\rm acc}$"
tr_entr_tacc_.units = r"K cm$^2$"
tr_entr_tacc_.limits = [7.0, 9.0]
tr_entr_tacc_.log = False  # auxCat() entr vals are in log


@catalog_field(multi="tr_temp_tacc_", alias="tr_temp_tacc")
def tr_temp_tacc_(sim, field):
    """Tracer tracks quantity: temperature at accretion time (Optionally, specify 'mode=X' and/or 'par=Y')."""
    fieldName = "Subhalo_Tracers_temp_tAcc"
    filePath = sim.derivPath + "auxCat/%s_%03d.hdf5" % (fieldName, sim.snap)

    vals = _tr_load(sim, field, filePath, fieldName)

    return vals


tr_temp_tacc_.label = r"Tracer Mean T$_{\rm gas}$ at $t_{\rm acc}$"
tr_temp_tacc_.units = r"K"
tr_temp_tacc_.limits = [4.8, 6.0]
tr_temp_tacc_.log = False  # auxCat() temp vals are in log

# ---------------------------- postprocessing/disperse -----------------------------------------------


@catalog_field
def d_minima(sim, field):
    """Postprocessing/disperse/: Distance to nearest cosmic web structure (minima = void), from disperse."""
    basePath = sim.postPath + "/disperse/output_upskl/stel_subhalo/"
    filePath = basePath + "subhalo_%s_S%d_M8-5_STEL.hdf5" % (sim.simName.split("-")[0], sim.snap)

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        vals = f["d_minima"][()]
        sub_ids = f["subhalo_ID"][()]

    assert np.array_equal(sub_ids, np.arange(sim.numSubhalos))  # sanity check

    vals = sim.units.codeLengthToMpc(vals)


d_minima.label = r"Distance to Nearest Void"
d_minima.units = r"Mpc"
d_minima.limits = [-2.0, 2.0]
d_minima.log = True


@catalog_field
def d_node(sim, field):
    """Postprocessing/disperse/: Distance to nearest cosmic web structure (node = halo), from disperse."""
    basePath = sim.postPath + "/disperse/output_upskl/stel_subhalo/"
    filePath = basePath + "subhalo_%s_S%d_M8-5_STEL.hdf5" % (sim.simName.split("-")[0], sim.snap)

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        vals = f["d_node"][()]
        sub_ids = f["subhalo_ID"][()]

    assert np.array_equal(sub_ids, np.arange(sim.numSubhalos))  # sanity check

    vals = sim.units.codeLengthToMpc(vals)


d_node.label = r"Distance to Nearest Halo"
d_node.units = r"Mpc"
d_node.limits = [-2.0, 2.0]
d_node.log = True


@catalog_field
def d_skel(sim, field):
    """Postprocessing/disperse/: Distance to nearest cosmic web structure (skel = filament), from disperse."""
    basePath = sim.postPath + "/disperse/output_upskl/stel_subhalo/"
    filePath = basePath + "subhalo_%s_S%d_M8-5_STEL.hdf5" % (sim.simName.split("-")[0], sim.snap)

    assert isfile(filePath), "File %s not found!" % filePath

    with h5py.File(filePath, "r") as f:
        vals = f["d_skel"][()]
        sub_ids = f["subhalo_ID"][()]

    assert np.array_equal(sub_ids, np.arange(sim.numSubhalos))  # sanity check

    vals = sim.units.codeLengthToMpc(vals)


d_skel.label = r"Distance to Nearest Filament"
d_skel.units = r"Mpc"
d_skel.limits = [-2.0, 2.0]
d_skel.log = True
