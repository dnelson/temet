"""
Definitions of custom catalog fields.
"""

import numpy as np

from ..cosmo.color import gfmBands, vegaMagCorrections
from ..plot.quantities import bandMagRange
from ..util.helper import logZeroNaN, running_median_clipped
from .groupcat import catalog_field


# -------------------- subhalos: meta -------------------------------------------------------------


@catalog_field(aliases=["subhalo_index", "id", "index"])
def subhalo_id(sim, field):
    """Subhalo ID/index."""
    assert "_log" not in field

    return np.arange(sim.numSubhalos)


subhalo_id.label = "Subhalo ID"
subhalo_id.units = ""  # dimensionless
subhalo_id.limits = [0, 7]
subhalo_id.log = True


@catalog_field(aliases=["cen_flag", "is_cen", "is_central"])
def central_flag(sim, field):
    """Subhalo central flag (1 if central, 0 if not)."""
    assert "_log" not in field

    GroupFirstSub = sim.halos("GroupFirstSub")
    GroupFirstSub = GroupFirstSub[np.where(GroupFirstSub >= 0)]

    # satellites given zero
    flag = np.zeros(sim.numSubhalos, dtype="int16")
    flag[GroupFirstSub] = 1

    return flag


central_flag.label = "Central Flag (0=no, 1=yes)"
central_flag.units = ""  # dimensionless
central_flag.limits = [0, 1]
central_flag.log = False


@catalog_field
def contam_frac(sim, field):
    """Subhalo contamination fraction (low-res DM to total DM particle count)."""
    SubhaloLenType = sim.subhalos("SubhaloLenType")

    n_lowres = SubhaloLenType[:, sim.ptNum("dmlowres")]
    n_hires = SubhaloLenType[:, sim.ptNum("dm")]

    with np.errstate(invalid="ignore"):
        frac = n_lowres / (n_lowres + n_hires)

    return frac


contam_frac.label = "Low-res Contamination Fraction"
contam_frac.units = ""  # dimensionless
contam_frac.limits = [0, 1]
contam_frac.log = False


@catalog_field
def redshift(sim, field):
    """Redshift, i.e. constant for all subhalos."""
    vals = np.zeros(sim.numSubhalos, dtype="float32") + sim.redshift

    return vals


redshift.label = "Redshift"
redshift.units = ""  # dimensionless
redshift.limits = [0, 4]
redshift.log = False

# -------------------- subhalos: halo-related properties ------------------------------------------


def mhalo_lim(sim, f):
    """Limits for halo masses."""
    lim = [11.0, 14.0]
    if sim.boxSize > 200000:
        lim = [11.0, 15.0]
    if sim.boxSize < 50000:
        lim = [10.5, 13.5]
    return lim


def _mhalo_load(sim, field):
    """Helper for the halo mass fields below."""
    haloField = "Group_M_Crit200"  # default for 'mhalo'

    if "200" in field:
        haloField = "Group_M_Crit200"
    if "500" in field:
        haloField = "Group_M_Crit500"
    if "vir" in field:
        haloField = "Group_M_TopHat200"  # misleading name

    halos = sim.groupCat(fieldsHalos=[haloField, "GroupFirstSub"])
    GrNr = sim.subhalos("SubhaloGrNr")

    mhalo = halos[haloField][GrNr]

    if "_code" not in field:
        mhalo = sim.units.codeMassToMsun(mhalo)

    if "_parent" not in field:
        # satellites given nan (by default)
        mask = np.zeros(GrNr.size, dtype="int16")
        mask[halos["GroupFirstSub"]] = 1

        mhalo[mask == 0] = np.nan

    return mhalo


@catalog_field(aliases=["mhalo_200_code", "mhalo_200_parent", "m200", "m200c", "mhalo"])
def mhalo_200(sim, field):
    r"""Parent halo total mass (:math:`\rm{M_{200,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default),
    unless '_parent' is specified in the field name, in which case satellites are given
    the same host halo mass as their central."""
    return _mhalo_load(sim, field)


mhalo_200.label = lambda sim, f: r"Halo Mass $\rm{M_{200c%s}}$" % (",parent" if "_parent" in f else "")
mhalo_200.units = lambda sim, f: r"$\rm{M_{sun}}$" if "_code" not in f else "code_mass"
mhalo_200.limits = mhalo_lim
mhalo_200.log = True


@catalog_field(aliases=["mhalo_500_code", "mhalo_500_parent", "m500", "m500c"])
def mhalo_500(sim, field):
    r"""Parent halo total mass (:math:`\rm{M_{500,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _mhalo_load(sim, field)


mhalo_500.label = lambda sim, f: r"Halo Mass $\rm{M_{500c%s}}$" % (",parent" if "_parent" in f else "")
mhalo_500.units = lambda sim, f: r"$\rm{M_{sun}}$" if "_code" not in f else "code_mass"
mhalo_500.limits = mhalo_lim
mhalo_500.log = True


@catalog_field(aliases=["mhalo_vir_code", "mhalo_vir_parent"])
def mhalo_vir(sim, field):
    r"""Parent halo total mass (:math:`\rm{M_{vir}}`). Defined by :math:`\rm{M_{\Delta}}`
    where :math:`\Delta` is the overdensity based on spherical tophat collapse.
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _mhalo_load(sim, field)


mhalo_vir.label = lambda sim, f: r"Halo Mass $\rm{M_{vir%s}}$" % (",parent" if "_parent" in f else "")
mhalo_vir.units = lambda sim, f: r"$\rm{M_{sun}}$" if "_code" not in f else "code_mass"
mhalo_vir.limits = mhalo_lim
mhalo_vir.log = True


def _rhalo_load(sim, field):
    """Helper for the halo radii loads below."""
    rField = "Group_R_Crit200" if "200" in field else "Group_R_Crit500"

    halos = sim.groupCat(fieldsHalos=[rField, "GroupFirstSub"])
    GrNr = sim.subhalos("SubhaloGrNr")

    rad = halos[rField][GrNr]

    if "_code" not in field:
        rad = sim.units.codeLengthToKpc(rad)

    # satellites given nan
    if "_parent" not in field:
        mask = np.zeros(GrNr.size, dtype="int16")
        mask[halos["GroupFirstSub"]] = 1
        wSat = np.where(mask == 0)
        rad[wSat] = np.nan

    return rad


@catalog_field(aliases=["rhalo_200_code", "rhalo_200_parent", "rhalo", "r200", "rhalo_200", "rvir"])
def rhalo_200(sim, field):
    r"""Parent halo virial radius (:math:`\rm{R_{200,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _rhalo_load(sim, field)


rhalo_200.label = lambda sim, f: r"$\rm{R_{halo,200c%s}}$" % (",parent" if "_parent" in f else "")
rhalo_200.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
rhalo_200.limits = [1.0, 3.0]
rhalo_200.log = True


@catalog_field(aliases=["rhalo_500_code", "rhalo_500_parent", "r500", "rhalo_500"])
def rhalo_500(sim, field):
    r"""Parent halo :math:`\rm{R_{500,crit}}` radius.
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _rhalo_load(sim, field)


rhalo_500.label = lambda sim, f: r"$\rm{R_{halo,500c%s}}$" % (",parent" if "_parent" in f else "")
rhalo_500.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
rhalo_500.limits = [1.0, 3.0]
rhalo_500.log = True


@catalog_field(aliases=["v200", "vvir"])
def vhalo(sim, field):
    r"""Parent halo virial velocity (:math:`\rm{V_{200}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    gc = sim.groupCat(fieldsSubhalos=["mhalo_200_code", "rhalo_200_code"])
    return sim.units.codeM200R200ToV200InKmS(gc["mhalo_200_code"], gc["rhalo_200_code"])


vhalo.label = r"$\rm{v_{200,halo}}$"
vhalo.units = r"$\rm{km/s}$"
vhalo.limits = [0, 200]
vhalo.log = False


@catalog_field(aliases=["halo_nsubs", "nsubs", "numsubs"])
def halo_numsubs(sim, field):
    """Total number of subhalos in parent dark matter halo. A value of one implies only a
    central subhalo exists, while a value of two indicates a central and one satellite,
    and so on. Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    haloField = "GroupNsubs"
    halos = sim.groupCat(fieldsHalos=[haloField, "GroupFirstSub"])
    GrNr = sim.subhalos("SubhaloGrNr")

    num = halos[haloField][GrNr].astype("float32")  # int dtype

    # satellites given nan
    mask = np.zeros(GrNr.size, dtype="int16")
    mask[halos["GroupFirstSub"]] = 1
    wSat = np.where(mask == 0)
    num[wSat] = np.nan

    return num


halo_numsubs.label = r"$\rm{N_{sub}}$ in Halo"
halo_numsubs.units = ""  # dimensionless
halo_numsubs.limits = [0.0, 2.0]
halo_numsubs.log = True


@catalog_field(alias="tvir")
def virtemp(sim, partType, fields, args):
    """Virial temperature of the parent halo (satellites have NaN)."""
    mass = sim.groupCat(fieldsSubhalos=["mhalo_200_code"])
    tvir = sim.units.codeMassToVirTemp(mass)
    return tvir.astype("float32")


virtemp.label = r"$\rm{T_{vir}}$"
virtemp.units = r"$\rm{K}$"
virtemp.limits = [4.0, 7.0]
virtemp.log = True


@catalog_field(aliases=["rdist_code", "rdist", "rdist_rvir", "distance_code", "distance_rvir"])
def distance(sim, field):
    """Radial distance of satellites to center of parent halo (centrals have zero)."""
    gc = sim.groupCat(fieldsHalos=["GroupPos", "Group_R_Crit200"], fieldsSubhalos=["SubhaloPos", "SubhaloGrNr"])

    parInds = gc["subhalos"]["SubhaloGrNr"]
    dist = sim.periodicDists(gc["halos"]["GroupPos"][parInds, :], gc["subhalos"]["SubhaloPos"])

    if "_rvir" not in field and "_code" not in field:
        dist = sim.units.codeLengthToKpc(dist)

    if "_rvir" in field:
        with np.errstate(invalid="ignore"):
            dist /= gc["halos"]["Group_R_Crit200"][parInds]

    return dist


distance.label = lambda sim, f: r"Radial Distance" if "_rvir" not in f else r"R / R$_{\rm vir,host}$"
distance.units = lambda sim, f: "code_length" if "_code" in f else ("" if "_rvir" in f else r"$\rm{kpc}$")
distance.limits = lambda sim, f: [0.0, 2.0] if "_rvir" in f else [1.0, 3.5]
distance.log = lambda sim, f: True if "_rvir" not in f else False  # linear for rvir normalized

# -------------------- subhalos: masses -----------------------------------------------------------


@catalog_field
def mhalo_subfind(sim, field):
    """Parent dark matter (sub)halo total mass, defined by the gravitationally bound mass as determined by Subfind."""
    mhalo = sim.subhalos("SubhaloMass")
    return sim.units.codeMassToMsun(mhalo)


mhalo_subfind.label = r"Subhalo Mass $\rm{M_{grav}}$"
mhalo_subfind.units = r"$\rm{M_{sun}}$"
mhalo_subfind.limits = mhalo_lim
mhalo_subfind.log = True


@catalog_field
def mstar1(sim, field):
    """Galaxy stellar mass, measured within the stellar half mass radius."""
    mass = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("stars")]
    return sim.units.codeMassToMsun(mass)


mstar1.label = r"$\rm{M_{\star}}$"  # (<r_{\star,1/2})
mstar1.units = r"$\rm{M_{sun}}$"
mstar1.limits = lambda sim, f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar1.log = True


@catalog_field
def mstar2(sim, field):
    """Galaxy stellar mass, measured within *twice* the stellar half mass radius."""
    mass = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]
    return sim.units.codeMassToMsun(mass)


mstar2.label = r"$\rm{M_{\star}}$"  # (<2r_{\star,1/2})
mstar2.units = r"$\rm{M_{sun}}$"
mstar2.limits = lambda sim, f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar2.log = True


@catalog_field
def mstar_tot(sim, field):
    """Galaxy stellar mass, total subhalo/subfind value."""
    mass = sim.subhalos("SubhaloMassType")[:, sim.ptNum("stars")]
    return sim.units.codeMassToMsun(mass)


mstar_tot.label = r"$\rm{M_{\star}}$"  # (subfind)
mstar_tot.units = r"$\rm{M_{sun}}$"
mstar_tot.limits = lambda sim, f: [9.0, 11.5] if sim.boxSize > 50000 else [8.0, 12.0]
mstar_tot.log = True


@catalog_field
def mstar_fof(sim, field):
    """Galaxy stellar mass, total halo (FoF) value. All satellites have same value as their central."""
    halo_mstar = sim.halos("GroupMassType")[:, sim.ptNum("stars")]
    mass = halo_mstar[sim.subhalos("SubhaloGrNr")]
    return sim.units.codeMassToMsun(mass)


mstar_fof.label = r"$\rm{M_{\star,fof}}$"
mstar_fof.units = r"$\rm{M_{sun}}$"
mstar_fof.limits = lambda sim, f: [9.5, 12.0] if sim.boxSize > 50000 else [8.5, 12.5]
mstar_fof.log = True


@catalog_field
def mgas1(sim, field):
    """Galaxy gas mass (all phases), measured within the stellar half mass radius."""
    mass = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("gas")]
    return sim.units.codeMassToMsun(mass)


mgas1.label = r"$\rm{M_{gas}}$"  # (<r_{\star,1/2})
mgas1.units = r"$\rm{M_{sun}}$"
mgas1.limits = lambda sim, f: [8.0, 11.0] if sim.boxSize > 50000 else [7.0, 10.5]
mgas1.log = True


@catalog_field
def mgas2(sim, field):
    """Galaxy gas mass (all phases), measured within *twice* the stellar half mass radius."""
    mass = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("gas")]
    return sim.units.codeMassToMsun(mass)


mgas2.label = r"$\rm{M_{gas}}$"  # (<2r_{\star,1/2})
mgas2.units = r"$\rm{M_{sun}}$"
mgas2.limits = lambda sim, f: [8.0, 11.0] if sim.boxSize > 50000 else [7.0, 10.5]
mgas2.log = True


@catalog_field
def mgas_tot(sim, field):
    """Galaxy gas mass, total subhalo/subfind value."""
    mass = sim.subhalos("SubhaloMassType")[:, sim.ptNum("gas")]
    return sim.units.codeMassToMsun(mass)


mgas_tot.label = r"$\rm{M_{gas}}$"  # (subfind)
mgas_tot.units = r"$\rm{M_{sun}}$"
mgas_tot.limits = lambda sim, f: [9.0, 11.5] if sim.boxSize > 50000 else [8.0, 12.0]
mgas_tot.log = True


@catalog_field
def mdm_tot(sim, field):
    """Galaxy DM mass, total subhalo/subfind value."""
    mass = sim.subhalos("SubhaloMassType")[:, sim.ptNum("dm")]
    return sim.units.codeMassToMsun(mass)


mdm_tot.label = r"$\rm{M_{DM}}$"  # (subfind)
mdm_tot.units = r"$\rm{M_{sun}}$"
mdm_tot.limits = lambda sim, f: [10.0, 12.5] if sim.boxSize > 50000 else [9.0, 13.0]
mdm_tot.log = True


@catalog_field(alias="mstar_100kpc")
def mstar_100pkpc(sim, field):
    """Galaxy stellar mass, measured within a fixed 3D aperture of 100 physical kpc."""
    acField = "Subhalo_Mass_100pkpc_Stars"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mstar_100pkpc.label = r"$\rm{M_{\star, <30kpc}}$"
mstar_100pkpc.units = r"$\rm{M_{sun}}$"
mstar_100pkpc.limits = lambda sim, f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar_100pkpc.log = True
mstar_100pkpc.auxcat = True


@catalog_field(alias="mstar_30kpc")
def mstar_30pkpc(sim, field):
    """Galaxy stellar mass, measured within a fixed 3D aperture of 30 physical kpc."""
    acField = "Subhalo_Mass_30pkpc_Stars"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mstar_30pkpc.label = r"$\rm{M_{\star, <30kpc}}$"
mstar_30pkpc.units = r"$\rm{M_{sun}}$"
mstar_30pkpc.limits = lambda sim, f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar_30pkpc.log = True
mstar_30pkpc.auxcat = True


@catalog_field
def mstar_5pkpc(sim, field):
    """Galaxy stellar mass, measured within a fixed 3D aperture of 5 physical kpc."""
    acField = "Subhalo_Mass_5pkpc_Stars"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mstar_5pkpc.label = r"$\rm{M_{\star, <5kpc}}$"
mstar_5pkpc.units = r"$\rm{M_{sun}}$"
mstar_5pkpc.limits = [8.0, 12.0]
mstar_5pkpc.log = True
mstar_5pkpc.auxcat = True


@catalog_field
def mgas_5pkpc(sim, field):
    """Galaxy gas mass, measured within a fixed 3D aperture of 5 physical kpc."""
    acField = "Subhalo_Mass_5pkpc_Gas"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mgas_5pkpc.label = r"$\rm{M_{gas, <5kpc}}$"
mgas_5pkpc.units = r"$\rm{M_{sun}}$"
mgas_5pkpc.limits = [7.5, 10.5]
mgas_5pkpc.log = True
mgas_5pkpc.auxcat = True


@catalog_field
def mdm_5pkpc(sim, field):
    """Galaxy dark matter mass, measured within a fixed 3D aperture of 5 physical kpc."""
    acField = "Subhalo_Mass_5pkpc_DM"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mdm_5pkpc.label = r"$\rm{M_{DM, <5kpc}}$"
mdm_5pkpc.units = r"$\rm{M_{sun}}$"
mdm_5pkpc.limits = [8.0, 12.0]
mdm_5pkpc.log = True
mdm_5pkpc.auxcat = True


@catalog_field
def mtot_5pkpc(sim, field):
    """Galaxy total mass (gas + stars + DM + BHs), measured within a fixed 3D aperture of 5 physical kpc."""
    mass = np.zeros(sim.numSubhalos, dtype="float32")
    for pt in ["Gas", "Stars", "DM", "BH"]:
        acField = "Subhalo_Mass_5pkpc_%s" % pt
        mass += sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mtot_5pkpc.label = r"$\rm{M_{total, <5kpc}}$"
mtot_5pkpc.units = r"$\rm{M_{sun}}$"
mtot_5pkpc.limits = [9.0, 12.0]
mtot_5pkpc.log = True
mtot_5pkpc.auxcat = True


@catalog_field
def mstar_mtot_ratio_5pkpc(sim, field):
    """Ratio of galaxy stellar mass, to total mass, both measured within a 3D aperture of 5 physical kpc."""
    mstar = sim.subhalos("mstar_5pkpc")
    mtot = sim.subhalos("mtot_5pkpc")

    return mstar / mtot


mstar_mtot_ratio_5pkpc.label = r"$\rm{M_{\star} / M_{total} (<5kpc)}$"
mstar_mtot_ratio_5pkpc.units = ""  # dimensionless
mstar_mtot_ratio_5pkpc.limits = [0.0, 0.8]
mstar_mtot_ratio_5pkpc.log = False
mstar_mtot_ratio_5pkpc.auxcat = True


@catalog_field
def mstar2_mhalo200_ratio(sim, field):
    """Galaxy stellar mass to halo mass ratio, the former defined as
    within twice the stellar half mass radius, the latter as M_200_Crit."""
    mstar = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]
    mhalo = sim.subhalos("mhalo_200_code")

    w = np.where(mhalo == 0)  # low mass halos with no central
    mhalo[w] = np.nan

    with np.errstate(invalid="ignore"):
        ratio = mstar / mhalo

    return ratio


mstar2_mhalo200_ratio.label = r"$\rm{M_{\star, <2r_{\star}} / M_{halo,200c}}$"
mstar2_mhalo200_ratio.units = ""  # dimensionless
mstar2_mhalo200_ratio.limits = [-3.0, -1.0]
mstar2_mhalo200_ratio.log = True


@catalog_field(alias="mstar_mhalo_ratio")
def mstar30pkpc_mhalo200_ratio(sim, field):
    """Galaxy stellar mass to halo mass ratio, the former measured within a
    fixed 3D aperture of 30 physical kpc, the latter taken as M_200_Crit."""
    acField = "Subhalo_Mass_30pkpc_Stars"
    mstar = sim.auxCat(acField)[acField]
    mhalo = sim.subhalos("mhalo_200_code")

    w = np.where(mhalo == 0)  # low mass halos with no central
    mhalo[w] = np.nan

    with np.errstate(invalid="ignore"):
        ratio = mstar / mhalo

    return ratio


mstar30pkpc_mhalo200_ratio.label = r"$\rm{M_{\star, <30pkpc} / M_{halo,200c}}$"
mstar30pkpc_mhalo200_ratio.units = ""  # dimensionless
mstar30pkpc_mhalo200_ratio.limits = [-3.0, -1.0]
mstar30pkpc_mhalo200_ratio.log = True
mstar30pkpc_mhalo200_ratio.auxcat = True


@catalog_field
def mstar_r500(sim, field):
    r"""Subhalo stellar mass (i.e. central+ICL, but no sats), measured within :math:`\rm{R_{500c}}`."""
    acField = "Subhalo_Mass_r500_Stars_FoF"
    mass = sim.auxCat(acField, expandPartial=True)[acField]
    return sim.units.codeMassToMsun(mass)


mstar_r500.label = r"$\rm{M_{\star, <r500}}$"
mstar_r500.units = r"$\rm{M_{sun}}$"
mstar_r500.limits = [8.0, 12.0]
mstar_r500.log = True
mstar_r500.auxcat = True


@catalog_field
def mgas_r500(sim, field):
    r"""Subhalo gas mass (all phases), measured within :math:`\rm{R_{500c}}`."""
    acField = "Subhalo_Mass_r500_Gas_FoF"
    mass = sim.auxCatSplit(acField, expandPartial=True)[acField]
    return sim.units.codeMassToMsun(mass)


mgas_r500.label = r"$\rm{M_{gas, <r500}}$"
mgas_r500.units = r"$\rm{M_{sun}}$"
mgas_r500.limits = [8.0, 12.0]
mgas_r500.log = True
mgas_r500.auxcat = True


@catalog_field
def mgas_halo(sim, field):
    """Halo-scale gas mass, measured within each FoF."""
    acField = "Subhalo_Mass_FoF_Gas"
    mass = sim.auxCatSplit(acField, expandPartial=True)[acField]
    return sim.units.codeMassToMsun(mass)


mgas_halo.label = r"$\rm{M_{gas, halo}}$"
mgas_halo.units = r"$\rm{M_{sun}}$"
mgas_halo.limits = lambda sim, f: [11.0, 14.5] if sim.boxSize > 50000 else [10.0, 13.5]
mgas_halo.log = True
mgas_halo.auxcat = True


@catalog_field
def mhi(sim, field):
    """Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within
    the entire subhalo (all gravitationally bound gas)."""
    acField = "Subhalo_Mass_HI"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mhi.label = r"$\rm{M_{HI, grav}}$"
mhi.units = r"$\rm{M_{sun}}$"
mhi.limits = lambda sim, f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi.log = True
mhi.auxcat = True


@catalog_field
def mhi2(sim, field):
    """Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within
    twice the stellar half mass radius."""
    acField = "Subhalo_Mass_2rstars_HI"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mhi2.label = r"$\rm{M_{HI, <2r_{\star}}}$"
mhi2.units = r"$\rm{M_{sun}}$"
mhi2.limits = lambda sim, f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi2.log = True
mhi2.auxcat = True


@catalog_field
def mhi_30pkpc(sim, field):
    """Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within
    a fixed 3D aperture of 30 physical kpc."""
    acField = "Subhalo_Mass_30pkpc_HI"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mhi_30pkpc.label = r"$\rm{M_{HI, <30kpc}}$"
mhi_30pkpc.units = r"$\rm{M_{sun}}$"
mhi_30pkpc.limits = lambda sim, f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi_30pkpc.log = True
mhi_30pkpc.auxcat = True


@catalog_field
def mhi_halo(sim, field):
    """Halo-scale atomic HI gas mass (BR06 molecular H2 model), measured within each FoF."""
    acField = "Subhalo_Mass_FoF_HI"
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)


mhi_halo.label = r"$\rm{M_{HI, halo}}$"
mhi_halo.units = r"$\rm{M_{sun}}$"
mhi_halo.limits = lambda sim, f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi_halo.log = True
mhi_halo.auxcat = True

# -------------------- subhalos: mass fractions -----------------------------------------------------------


@catalog_field(alias="fgas1_alt")
def fgas1(sim, field):
    """Galaxy gas mass fraction (all phases), measured within the stellar half mass radius."""
    mgas = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("gas")]
    mstar = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("stars")]

    with np.errstate(invalid="ignore"):
        if "_alt" in field:
            fgas = mgas / mstar  # alternative definition
        else:
            fgas = mgas / (mgas + mstar)

    return fgas


fgas1.label = (
    lambda sim, f: r"$\rm{f_{gas} = M_{\rm gas} / (M_{gas} + M_\star) (<r_{\star})}$"
    if "_alt" not in f
    else r"$\rm{f_{gas} = M_{\rm gas} / M_\star (<r_{\star})}$"
)
fgas1.units = ""  # dimensionless
fgas1.limits = [-3.0, 0.0]
fgas1.log = True


@catalog_field(alias="fgas2_alt")
def fgas2(sim, field):
    """Galaxy gas mass fraction (all phases), measured within twice the stellar half mass radius."""
    mgas = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("gas")]
    mstar = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]

    with np.errstate(invalid="ignore"):
        if "_alt" in field:
            fgas = mgas / mstar  # alternative definition
        else:
            fgas = mgas / (mgas + mstar)

    return fgas


fgas2.label = (
    lambda sim, f: r"$\rm{f_{gas} = M_{\rm gas} / (M_{gas} + M_\star) (<2r_{\star})}$"
    if "_alt" not in f
    else r"$\rm{f_{gas} = M_{\rm gas} / M_\star (<2r_{\star})}$"
)
fgas2.units = ""  # dimensionless
fgas2.limits = [-3.0, 0.0]
fgas2.log = True


@catalog_field(alias="fgas_alt")
def fgas(sim, field):
    """Galaxy gas mass fraction (all phases), measured within the entire subhalo."""
    mgas = sim.subhalos("SubhaloMassType")[:, sim.ptNum("gas")]
    mstar = sim.subhalos("SubhaloMassType")[:, sim.ptNum("stars")]

    with np.errstate(invalid="ignore"):
        if "_alt" in field:
            fgas = mgas / mstar  # alternative definition
        else:
            fgas = mgas / (mgas + mstar)

    return fgas


fgas.label = (
    lambda sim, f: r"$\rm{f_{gas} = M_{\rm gas} / (M_{gas} + M_\star)}$"
    if "_alt" not in f
    else r"$\rm{f_{gas} = M_{\rm gas} / M_\star}$"
)
fgas.units = ""  # dimensionless
fgas.limits = [-3.0, 0.0]
fgas.log = True


@catalog_field
def fdm(sim, field):
    """Galaxy DM fraction, measured within the entire subhalo."""
    mdm = sim.subhalos("SubhaloMassType")[:, sim.ptNum("dm")]
    mtot = sim.subhalos("SubhaloMass")

    with np.errstate(invalid="ignore"):
        fdm = mdm / mtot

    return fdm


fdm.label = r"$\rm{f_{dm} = M_{\rm dm} / M_{\rm tot}}$"
fdm.units = ""  # dimensionless
fdm.limits = [-3.0, 0.0]
fdm.log = True


@catalog_field
def fdm1(sim, field):
    """Galaxy DM fraction, measured within the stellar half mass radius."""
    mdm = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("dm")]
    mtot = sim.subhalos("SubhaloMassInHalfRad")

    with np.errstate(invalid="ignore"):
        fdm = mdm / mtot

    return fdm


fdm1.label = r"$\rm{f_{dm} = M_{\rm dm} / M_{\rm tot}}$"
fdm1.units = ""  # dimensionless
fdm1.limits = [-3.0, 0.0]
fdm1.log = True


@catalog_field
def fdm2(sim, field):
    """Galaxy DM fraction, measured within twice the stellar half mass radius."""
    mdm = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("dm")]
    mtot = sim.subhalos("SubhaloMassInRad")

    with np.errstate(invalid="ignore"):
        fdm = mdm / mtot

    return fdm


fdm2.label = r"$\rm{f_{dm} = M_{\rm dm} / M_{\rm tot}}$"
fdm2.units = ""  # dimensionless
fdm2.limits = [-3.0, 0.0]
fdm2.log = True

# -------------------- star formation rates -------------------------------------------------------


@catalog_field
def sfr1(sim, field):
    """Galaxy star formation rate (instantaneous, within one times the stellar half mass radius)."""
    return sim.subhalos("SubhaloSFRinHalfRad")  # units correct


sfr1.label = r"$\rm{SFR_{<r_{\star},instant}}$"
sfr1.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr1.limits = [-2.5, 1.0]
sfr1.log = True


@catalog_field
def sfr2(sim, field):
    """Galaxy star formation rate (instantaneous, within twice the stellar half mass radius)."""
    return sim.subhalos("SubhaloSFRinRad")  # units correct


sfr2.label = r"$\rm{SFR_{<2r_{\star},instant}}$"
sfr2.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr2.limits = [-2.5, 1.0]
sfr2.log = True


@catalog_field
def sfr1_surfdens(sim, field):
    """Galaxy star formation rate surface density (instantaneous, within one times the stellar half mass radius)."""
    sfr = sim.subhalos("SubhaloSFRinHalfRad")
    aperture = sim.units.codeLengthToKpc(sim.subhalos("SubhaloHalfmassRadType")[:, sim.ptNum("stars")])
    area = np.pi * aperture**2

    with np.errstate(invalid="ignore"):
        vals = sfr / area  # Msun/yr/kpc^2

    return vals


sfr1_surfdens.label = r"$\rm{\Sigma_{SFR,<r_{\star},instant}}$"
sfr1_surfdens.units = r"$\rm{M_{sun}\, yr^{-1}\, kpc^{-2}}$"
sfr1_surfdens.limits = [-2.5, 2.0]
sfr1_surfdens.log = True


@catalog_field
def sfr2_surfdens(sim, field):
    """Galaxy star formation rate surface density (instantaneous, within twice the stellar half mass radius)."""
    sfr = sim.subhalos("SubhaloSFRinRad")
    aperture = 2.0 * sim.units.codeLengthToKpc(sim.subhalos("SubhaloHalfmassRadType")[:, sim.ptNum("stars")])
    area = np.pi * aperture**2

    with np.errstate(invalid="ignore"):
        vals = sfr / area  # Msun/yr/kpc^2

    return vals


sfr2_surfdens.label = r"$\rm{\Sigma_{SFR,<2r_{\star},instant}}$"
sfr2_surfdens.units = r"$\rm{M_{sun}\, yr^{-1}\, kpc^{-2}}$"
sfr2_surfdens.limits = [-3.5, 1.0]
sfr2_surfdens.log = True


@catalog_field
def ssfr(sim, field):
    """Galaxy specific star formation rate [1/yr] (sSFR, instantaneous, both SFR and M* within 2rhalfstars)."""
    sfr = sim.subhalos("SubhaloSFRinRad")
    mstar = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]
    mstar = sim.units.codeMassToMsun(mstar)

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr


ssfr.label = r"$\rm{sSFR_{<2r_{\star},instant}}$"
ssfr.units = r"$\rm{yr^{-1}}$"
ssfr.limits = [-12.0, -8.0]
ssfr.log = True


@catalog_field
def ssfr_gyr(sim, field):
    """Galaxy specific star formation rate [1/Gyr] (sSFR, instantaneous, both SFR and M* within 2rhalfstars)."""
    return sim.subhalos("ssfr") * 1e9


ssfr_gyr.label = r"$\rm{sSFR_{<2r_{\star}}}$"
ssfr_gyr.units = r"$\rm{Gyr^{-1}}$"
ssfr_gyr.limits = [-3.0, 1.0]
ssfr_gyr.log = True


@catalog_field
def delta_sfms(sim, field):
    """Offset from the star-formation main sequence (SFMS), taken as the clipped sim median, in dex."""
    mstar = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]
    sfr = sim.subhalos("SubhaloSFRinRad")
    mstar = sim.units.codeMassToMsun(mstar)

    with np.errstate(invalid="ignore"):  # mstar==0 values generate ssfr==nan
        log_ssfr = logZeroNaN(sfr / mstar * 1e9)  # 1/yr to 1/Gyr

    # construct SFMS (in sSFR) values as a function of stellar mass (skip zeros, clip 10% tails)
    # fix minVal and maxVal for consistent bins
    binSize = 0.2  # dex
    mstar = logZeroNaN(mstar)
    med_mstar, med_log_ssfr, mstar_bins = running_median_clipped(
        mstar, log_ssfr, binSize=binSize, minVal=6.0, maxVal=12.0, skipZerosX=True, skipZerosY=True, clipPercs=[10, 90]
    )

    # constant value beyond the end of the MS
    with np.errstate(invalid="ignore"):
        w = np.where(med_mstar >= 10.5)

    ind_last_sfms_bin = w[0][0] - 1
    med_log_ssfr[w[0][0] :] = med_log_ssfr[ind_last_sfms_bin]

    # for every subhalo, locate the value to compare to (its mstar bin)
    inds = np.searchsorted(mstar_bins, mstar, side="left") - 1
    comp_log_ssfr = med_log_ssfr[inds]

    vals = log_ssfr - comp_log_ssfr  # dex

    return vals


delta_sfms.label = r"$\rm{\Delta SFMS}$"
delta_sfms.units = "dex"
delta_sfms.limits = [-1.5, 1.5]
delta_sfms.log = False

# -------------------- metallicities  ------------------------------------------------


@catalog_field
def z_stars(sim, field):
    """Galaxy stellar metallicity, mass-weighted average of all star particles within 2rhalfstars."""
    vals = sim.subhalos("SubhaloStarMetallicity")
    vals = sim.units.metallicityInSolar(vals)
    return vals


z_stars.label = r"$\rm{Z_{\star}}$"
z_stars.units = r"$\rm{Z_\odot}$"  # solar
z_stars.limits = [-2.0, 0.5]
z_stars.log = True


@catalog_field
def z_gas(sim, field):
    """Galaxy gas metallicity, mass-weighted average of all gas cells within 2rhalfstars."""
    vals = sim.subhalos("SubhaloGasMetallicity")
    vals = sim.units.metallicityInSolar(vals)
    return vals


z_gas.label = r"$\rm{Z_{gas}}$"
z_gas.units = r"$\rm{Z_\odot}$"  # solar
z_gas.limits = [-1.0, 0.4]
z_gas.log = True


@catalog_field
def z_gas_sfr(sim, field):
    """Galaxy gas metallicity, SFR-weighted average of all gas cells within 2rhalfstars."""
    vals = sim.subhalos("SubhaloGasMetallicitySfrWeighted")
    vals = sim.units.metallicityInSolar(vals)
    return vals


z_gas_sfr.label = r"$\rm{Z_{gas}}}$"
z_gas_sfr.units = r"$\rm{Z_\odot}$"  # solar
z_gas_sfr.limits = [-2.0, 0.5]
z_gas_sfr.log = True

# -------------------- sizes ------------------------------------------------


@catalog_field(aliases=["rhalf_stars", "size_stars_code", "rhalf_stars_code"])
def size_stars(sim, field):
    """Stellar half mass radius."""
    radtype = sim.subhalos("SubhaloHalfmassRadType")
    rad = radtype[:, sim.ptNum("stars")]

    if "_code" not in field:
        rad = sim.units.codeLengthToKpc(rad)

    return rad


size_stars.label = r"r$_{\rm 1/2,\star}$"
size_stars.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
size_stars.limits = lambda sim, f: [0.2, 1.6] if sim.redshift < 1 else [-0.4, 1.4]
size_stars.log = True


@catalog_field(aliases=["size_stars_rvir_ratio"])
def re_rvir_ratio(sim, field):
    """Stellar half mass radius normalized by parent halo virial radius (r200c)."""
    radtype = sim.subhalos("SubhaloHalfmassRadType")
    rad = radtype[:, sim.ptNum("stars")]

    rad_norm = sim.subhalos("rhalo_200_code")
    with np.errstate(invalid="ignore"):
        rad /= rad_norm

    return rad


re_rvir_ratio.label = r"r$_{\rm 1/2,\star}$ / R$_{\rm vir,halo}$"
re_rvir_ratio.units = ""  # dimensionless
re_rvir_ratio.limits = lambda sim, f: [-2.5, -1.5]
re_rvir_ratio.log = True


@catalog_field(aliases=["rhalf_gas", "size_gas_code", "rhalf_gas_code"])
def size_gas(sim, field):
    """Gas half mass radius."""
    radtype = sim.subhalos("SubhaloHalfmassRadType")
    rad = radtype[:, sim.ptNum("gas")]

    if "_code" not in field:
        rad = sim.units.codeLengthToKpc(rad)

    return rad


size_gas.label = r"r$_{\rm 1/2,gas}$"
size_gas.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
size_gas.limits = lambda sim, f: [1.0, 2.8]
size_gas.log = True


@catalog_field
def surfdens1_stars(sim, field):
    """Galaxy stellar surface density (within the stellar half mass radius)."""
    mstar = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("stars")]
    mass = sim.units.codeMassToMsun(mstar)
    aperture = sim.units.codeLengthToKpc(sim.subhalos("SubhaloHalfmassRadType")[:, sim.ptNum("stars")])
    area = np.pi * aperture**2

    with np.errstate(invalid="ignore"):
        vals = mass / area  # Msun/kpc^2

    return vals


surfdens1_stars.label = r"$\rm{\Sigma_{*,<r_{\star}}}$"
surfdens1_stars.units = r"$\rm{M_{sun}\, kpc^{-2}}$"
surfdens1_stars.limits = [6.5, 9.0]
surfdens1_stars.log = True


@catalog_field
def surfdens2_stars(sim, field):
    """Galaxy stellar surface density (within twice the stellar half mass radius)."""
    mstar = sim.subhalos("SubhaloMassInRadType")[:, sim.ptNum("stars")]
    mass = sim.units.codeMassToMsun(mstar)
    aperture = 2.0 * sim.units.codeLengthToKpc(sim.subhalos("SubhaloHalfmassRadType")[:, sim.ptNum("stars")])
    area = np.pi * aperture**2

    with np.errstate(invalid="ignore"):
        vals = mass / area  # Msun/kpc^2

    return vals


surfdens2_stars.label = r"$\rm{\Sigma_{*,<2r_{\star}}}$"
surfdens2_stars.units = r"$\rm{M_{sun}\, kpc^{-2}}$"
surfdens2_stars.limits = [6.5, 9.0]
surfdens2_stars.log = True


@catalog_field
def surfdens1_dm(sim, field):
    """Galaxy DM surface density (within the stellar half mass radius)."""
    mstar = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("dm")]
    mass = sim.units.codeMassToMsun(mstar)
    aperture = sim.units.codeLengthToKpc(sim.subhalos("SubhaloHalfmassRadType")[:, sim.ptNum("stars")])
    area = np.pi * aperture**2

    with np.errstate(invalid="ignore"):
        vals = mass / area  # Msun/kpc^2

    return vals


surfdens1_dm.label = r"$\rm{\Sigma_{DM,<r_{\star}}}$"
surfdens1_dm.units = r"$\rm{M_{sun}\, kpc^{-2}}$"
surfdens1_dm.limits = [6.5, 9.0]
surfdens1_dm.log = True

# -------------------- general subhalo properties ------------------------------------------------


@catalog_field(aliases=["vc", "vmax"])
def vcirc(sim, field):
    """Maximum value of the spherically-averaged 3D circular velocity curve
    (i.e. galaxy circular velocity)."""
    return sim.subhalos("SubhaloVmax")  # units correct


vcirc.label = r"$\rm{V_{circ}}$"
vcirc.units = r"$\rm{km/s}$"
vcirc.limits = [1.8, 2.8]
vcirc.log = True


@catalog_field(alias="vmag")
def velmag(sim, field):
    """The magnitude of the current velocity of the subhalo through the box,
    in the simulation reference frame."""
    vel = sim.subhalos("SubhaloVel")
    vel = sim.units.subhaloCodeVelocityToKms(vel)
    vmag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2)

    return vmag


velmag.label = r"$\rm{|V|_{subhalo}}$"
velmag.units = r"$\rm{km/s}$"
velmag.limits = [1.5, 3.5]
velmag.log = True


@catalog_field(alias="smag")
def spinmag(sim, field):
    """The magnitude of the subhalo spin vector, computed as the mass weighted
    sum of all subhalo particles/cells."""
    spin = sim.subhalos("SubhaloSpin")
    spin = sim.units.subhaloSpinToKpcKms(spin)
    smag = np.sqrt(spin[:, 0] ** 2 + spin[:, 1] ** 2 + spin[:, 2] ** 2)

    return smag


spinmag.label = r"$\rm{|S|_{subhalo}}$"
spinmag.units = r"$\rm{kpc km/s}$"
spinmag.limits = [2.0, 4.0]
spinmag.log = True

# -------------------- subhalo photometrics  ------------------------------------------------------


@catalog_field(aliases=["m_u", "m_b", "m_r"])
def m_v(sim, field):
    """V-band (B-band, r-band, ...) magnitude (StellarPhotometrics from snapshot). AB system. No dust."""
    assert "_log" not in field
    bandName = field.split("_")[1].upper()
    if bandName not in gfmBands:
        bandName = bandName.lower()

    vals = sim.subhalos("SubhaloStellarPhotometrics")
    mags = vals[:, gfmBands[bandName]].copy()  # careful with mutable cache

    # fix zero values
    w = np.where(mags > 1e10)
    mags[w] = np.nan

    # Vega corrections
    if bandName in vegaMagCorrections:
        mags += vegaMagCorrections[bandName]

    return mags


m_v.label = lambda sim, f: r"M$_{\rm %s}$" % f.split("_")[1].upper()
m_v.units = "abs AB mag"
m_v.limits = [-24, -16]
m_v.log = False


@catalog_field(aliases=["color_vb"])
def color_uv(sim, field):
    """Integrated photometric/broadband galaxy colors, from snapshot. AB system. No dust."""
    assert "_log" not in field
    bandNames = field.split("color_")[1].upper()

    mags_0 = sim.subhalos("M_" + bandNames[0])
    mags_1 = sim.subhalos("M_" + bandNames[1])

    colors = mags_0 - mags_1

    return colors


color_uv.label = lambda sim, f: r"(%s-%s) color" % (f.split("color_")[1][0].upper(), f.split("color_")[1][1].upper())
color_uv.units = "mag"
color_uv.limits = lambda sim, f: bandMagRange(f.split("color_")[1], sim=sim)
color_uv.log = False
