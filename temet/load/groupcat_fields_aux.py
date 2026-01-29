"""
Definitions of custom catalog fields, based on auxCat/ datasets.
"""

import numpy as np

from ..cosmo.clustering import isolationCriterion3D
from ..cosmo.color import loadColors
from .groupcat import catalog_field, groupOrderedValsToSubhaloOrdered


# ---------------------------- auxcat: environment ------------------------------------------------


@catalog_field(multi="isolated3d,")
def isolated_flag_(sim, field):
    """Isolated flag (1 if 'isolated', according to criterion, 0 if not, -1 if unprocessed)."""
    # e.g. 'isolated3d,mstar30kpc,max,in_300pkpc'
    _, quant, max_type, dist = field.split(",")
    dist = float(dist.split("in_")[1].split("pkpc")[0])

    ic3d = isolationCriterion3D(sim, dist)
    icName = "flag_iso_%s_%s" % (quant, max_type)

    return ic3d[icName]


isolated_flag_.label = "Isolated?"
isolated_flag_.units = ""  # dimensionless
isolated_flag_.limits = [-1, 1]
isolated_flag_.log = False


@catalog_field(aliases=["d5_mstar_gt7", "d5_mstar_gt8"])
def d5_mstar_gthalf(sim, field):
    """Environment: distance to 5th nearest neighbor (subhalo) that has a stellar mass at least half
    of our own (default unless specified)."""
    acField = "Subhalo_Env_d5_MstarRel_GtHalf"  # include galaxies with Mstar > 0.5*of this subhalo
    if "_gt8" in field:
        acField = "Subhalo_Env_d5_Mstar_Gt8"  # for galaxies with Mstar > 10^8 Msun
    if "_gt7" in field:
        acField = "Subhalo_Env_d5_Mstar_Gt7"  # for galaxies with Mstar > 10^7 Msun

    ac = sim.auxCat(fields=[acField], expandPartial=True)
    return ac[acField]


d5_mstar_gthalf.label = r"$d_{5}$"
d5_mstar_gthalf.units = "code_length"
d5_mstar_gthalf.limits = [1.0, 4.0]
d5_mstar_gthalf.log = True


@catalog_field(aliases=["delta5_mstar_gt7", "delta5_mstar_gt8"])
def delta5_mstar_gthalf(sim, field):
    """Environment: overdensity based on 5th nearest neighbor (subhalo)."""
    d5 = sim.subhalos(field.replace("delta5_", "d5_"))

    # compute dimensionless overdensity (rho/rho_mean-1)
    N = 5
    rho_N = N / (4 / 3 * np.pi * d5**3)  # local galaxy volume density
    delta_N = rho_N / np.nanmean(rho_N) - 1.0

    return delta_N


delta5_mstar_gthalf.label = r"$\delta_{5}$"
delta5_mstar_gthalf.units = ""  # linear dimensionless
delta5_mstar_gthalf.limits = [-1.0, 2.0]
delta5_mstar_gthalf.log = True


@catalog_field(aliases=["num_ngb_gt7", "num_ngb_gt8", "num_ngb_gttenth"])
def num_ngb_gthalf(sim, field):
    """Environment: counts of nearby neighbor subhalos, within a given 3D specture, satisfying some
    minimum (relative) stellar mass criterion."""
    relStr = "MstarRel_GtHalf"  # include galaxies with Mstar > 0.5*of this subhalo
    if "_gttenth" in field:
        relStr = "MstarRel_GtTenth"  # include galaxies with Mstar > 0.1*of this subhalo
    if "_gt7" in field:
        relStr = "Mstar_Gt7"  # for galaxies with Mstar > 10^7 Msun
    if "_gt8" in field:
        relStr = "Mstar_Gt8"  # for galaxies with Mstar > 10^8 Msun

    distStr = "2rvir"
    acField = "Subhalo_Env_Count_%s_%s" % (relStr, distStr)

    ac = sim.auxCat(fields=[acField], expandPartial=True)[acField]  # int dtype
    vals = ac.astype("float32")
    vals[vals == -1.0] = np.nan  # works?

    return vals


num_ngb_gthalf.label = r"$\rm{N_{ngb,subhalos}}$"
num_ngb_gthalf.units = ""  # linear dimensionless
num_ngb_gthalf.limits = [0, 100]
num_ngb_gthalf.log = False

# ---------------------------- auxcat: color ------------------------------------------------------


@catalog_field(multi="color_")
def color_(sim, field):
    """Photometric/broadband colors (e.g. 'color_C_gr', 'color_A_ur')."""
    return loadColors(sim, field)


color_.label = lambda sim, f: "(%s-%s) color" % (f.split("_")[2][0], f.split("_")[2][1])
color_.units = "mag"
color_.limits = [-0.4, 1.0]
color_.log = False


@catalog_field(multi="mag_")
def mag_(sim, field):
    """Photometric/broadband magnitudes (e.g. 'mag_C_g', 'mag_A_r')."""
    return loadColors(sim, field)


mag_.label = lambda sim, f: r"$\rm{M_{%s}}$" % (f.split("_")[2])
mag_.units = "mag"  # AB
mag_.limits = [-19, -23]
mag_.log = False

# ---------------------------- auxcat: (gas) masses -----------------------------------------------------


@catalog_field
def mass_ovi(sim, field):
    """Total gas mass in sub-species: OVI."""
    # todo: could generalize to e.g. all ions
    acField = "Subhalo_Mass_OVI"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_ovi.label = r"$\rm{M_{OVI}}$"
mass_ovi.units = r"$\rm{M_{sun}}$"
mass_ovi.limits = [5.0, 10.0]
mass_ovi.log = True


@catalog_field
def mass_ovii(sim, field):
    """Total gas mass in sub-species: OVII."""
    acField = "Subhalo_Mass_OVII"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_ovii.label = r"$\rm{M_{OVII}}$"
mass_ovii.units = r"$\rm{M_{sun}}$"
mass_ovii.limits = [5.0, 10.0]
mass_ovii.log = True


@catalog_field
def mass_oviii(sim, field):
    """Total gas mass in sub-species: OVIII."""
    acField = "Subhalo_Mass_OVIII"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_oviii.label = r"$\rm{M_{OVIII}}$"
mass_oviii.units = r"$\rm{M_{sun}}$"
mass_oviii.limits = [5.0, 10.0]
mass_oviii.log = True


@catalog_field
def mass_o(sim, field):
    """Total gas mass in sub-species: O."""
    acField = "Subhalo_Mass_AllGas_Oxygen"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_o.label = r"$\rm{M_{O,gas}}$"
mass_o.units = r"$\rm{M_{sun}}$"
mass_o.limits = [5.0, 10.0]
mass_o.log = True


@catalog_field
def mass_z(sim, field):
    """Total gas mass in sub-species: Z."""
    acField = "Subhalo_Mass_AllGas_Metal"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_z.label = r"$\rm{M_{Z,gas}}$"
mass_z.units = r"$\rm{M_{sun}}$"
mass_z.limits = [6.0, 10.0]
mass_z.log = True


@catalog_field
def mass_halogas(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass."""
    acField = "Subhalo_Mass_HaloGas"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogas.label = r"$\rm{M_{halo,gas}}$"
mass_halogas.units = r"$\rm{M_{sun}}$"
mass_halogas.limits = [8.0, 14.0]
mass_halogas.log = True


@catalog_field
def mass_halogasfof(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass. FoF-scope, centrals only."""
    acField = "Subhalo_Mass_HaloGasFoF"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogasfof.label = r"$\rm{M_{halo,gas}}$"
mass_halogasfof.units = r"$\rm{M_{sun}}$"
mass_halogasfof.limits = [8.0, 14.0]
mass_halogasfof.log = True


@catalog_field
def mass_halogas_cold(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass. Only cold (log T < 4.5 K), star-forming gas at eEOS temp."""
    acField = "Subhalo_Mass_HaloGas_Cold"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogas_cold.label = r"$\rm{M_{halo,gas,cold}}$"
mass_halogas_cold.units = r"$\rm{M_{sun}}$"
mass_halogas_cold.limits = [7.0, 13.0]
mass_halogas_cold.log = True


@catalog_field
def mass_halogas_sfcold(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass. Only cold (log T < 4.5 K), star-forming gas at cold temp."""
    acField = "Subhalo_Mass_HaloGas_SFCold"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogas_sfcold.label = r"$\rm{M_{halo,gas,sfcold}}$"
mass_halogas_sfcold.units = r"$\rm{M_{sun}}$"
mass_halogas_sfcold.limits = [7.0, 13.0]
mass_halogas_sfcold.log = True


@catalog_field
def mass_halogasfof_cold(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass. FoF-scope, centrals only.
    Only cold (log T < 4.5 K), star-forming gas at eEOS temp."""
    acField = "Subhalo_Mass_HaloGasFoF_Cold"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogasfof_cold.label = r"$\rm{M_{halo,gas,cold}}$"
mass_halogasfof_cold.units = r"$\rm{M_{sun}}$"
mass_halogasfof_cold.limits = [7.0, 13.0]
mass_halogasfof_cold.log = True


@catalog_field
def mass_halogasfof_sfcold(sim, field):
    """Total halo (0.15 < r/rvir < 1.0) gas mass. FoF-scope, centrals only.
    Only cold (log T < 4.5 K), star-forming gas at cold temp."""
    acField = "Subhalo_Mass_HaloGasFoF_SFCold"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_halogasfof_sfcold.label = r"$\rm{M_{halo,gas,sfcold}}$"
mass_halogasfof_sfcold.units = r"$\rm{M_{sun}}$"
mass_halogasfof_sfcold.limits = [7.0, 13.0]
mass_halogasfof_sfcold.log = True


@catalog_field
def frac_halogas_cold(sim, field):
    """Fraction of halo (0.15 < r/rvir < 1.0) gas mass that is cold (log T < 4.5 K), star-forming gas at eEOS temp."""
    mass_subset = sim.subhalos("mass_halogas_cold")
    mass_total = sim.subhalos("mass_halogas")

    with np.errstate(invalid="ignore"):
        frac = mass_subset / mass_total

    return frac


frac_halogas_cold.label = r"$\rm{M_{halo,gas,cold} / M_{halo,gas}}$"
frac_halogas_cold.units = ""  # dimensionless
frac_halogas_cold.limits = [0, 1]
frac_halogas_cold.log = False


@catalog_field
def frac_halogas_sfcold(sim, field):
    """Fraction of halo (0.15 < r/rvir < 1.0) gas mass that is cold (log T < 4.5 K), star-forming gas at cold temp."""
    mass_subset = sim.subhalos("mass_halogas_sfcold")
    mass_total = sim.subhalos("mass_halogas")

    with np.errstate(invalid="ignore"):
        frac = mass_subset / mass_total

    return frac


frac_halogas_sfcold.label = r"$\rm{M_{halo,gas,sfcold} / M_{halo,gas}}$"
frac_halogas_sfcold.units = ""  # dimensionless
frac_halogas_sfcold.limits = [0, 1]
frac_halogas_sfcold.log = False


@catalog_field
def frac_halogasfof_cold(sim, field):
    """Fraction of halo (0.15 < r/rvir < 1.0) gas mass that is cold (log T < 4.5 K),
    star-forming gas at eEOS temp. FoF-scope, centrals only."""
    mass_subset = sim.subhalos("mass_halogasfof_cold")
    mass_total = sim.subhalos("mass_halogasfof")

    with np.errstate(invalid="ignore"):
        frac = mass_subset / mass_total

    return frac


frac_halogasfof_cold.label = r"$\rm{M_{halo,gas,cold} / M_{halo,gas}}$"
frac_halogasfof_cold.units = ""  # dimensionless
frac_halogasfof_cold.limits = [0, 1]
frac_halogasfof_cold.log = False


@catalog_field
def frac_halogasfof_sfcold(sim, field):
    """Fraction of halo (0.15 < r/rvir < 1.0) gas mass that is cold (log T < 4.5 K),
    star-forming gas at cold temp. FoF-scope, centrals only."""
    mass_subset = sim.subhalos("mass_halogasfof_sfcold")
    mass_total = sim.subhalos("mass_halogasfof")

    with np.errstate(invalid="ignore"):
        frac = mass_subset / mass_total

    return frac


frac_halogasfof_sfcold.label = r"$\rm{M_{halo,gas,sfcold} / M_{halo,gas}}$"
frac_halogasfof_sfcold.units = ""  # dimensionless
frac_halogasfof_sfcold.limits = [0, 1]
frac_halogasfof_sfcold.log = False

# ---------------------------- auxcat: (other) masses -----------------------------------------------------


@catalog_field
def mass_smbh(sim, field):
    """Largest SMBH mass in each subhalo. Avoids summing multiple SMBH masses, if more than one present."""
    acField = "Subhalo_BH_Mass_largest"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])


mass_smbh.label = r"$\rm{M_{SMBH}}$"
mass_smbh.units = r"$\rm{M_{sun}}$"
mass_smbh.limits = [6.0, 10.0]
mass_smbh.log = True


@catalog_field
def smbh_mdot(sim, field):
    """Largest SMBH Mdot in each subhalo."""
    acField = "Subhalo_BH_Mdot_largest"
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassOverTimeToMsunPerYear(ac[acField])


smbh_mdot.label = r"$\rm{\dot{M}_{SMBH}}$"
smbh_mdot.units = r"$\rm{M_{sun} / yr}$"
smbh_mdot.limits = [-4.0, 0.0]
smbh_mdot.log = True


@catalog_field(aliases=["l_bol", "l_agn"])
def smbh_lum(sim, field):
    """Bolometric luminosity of largest SMBH in each subhalo."""
    acField = "Subhalo_BH_Mass_largest"
    m_smbh = sim.auxCat(fields=[acField])[acField]

    acField = "Subhalo_BH_Mdot_largest"
    smbh_mdot = sim.auxCat(fields=[acField])[acField]

    return sim.units.codeBHMassMdotToBolLum(m_smbh, smbh_mdot)


smbh_lum.label = r"$\rm{L_{AGN,bol}}$"
smbh_lum.units = r"$\rm{erg / s}$"
smbh_lum.limits = [37.0, 42.0]
smbh_lum.log = True

# ---------------------------- auxcat: mass fractions -----------------------------------------------------


@catalog_field
def fgas_r200(sim, field):
    """Gas mass fraction (= Mgas/Mtot) within the virial radius, r200c. FoF-scope approximation."""
    acField = "Subhalo_Mass_r200_Gas"
    M_gas = sim.auxCat(acField, expandPartial=True)[acField]
    M_tot = sim.subhalos("mhalo_200_code")

    # correct for non-global r200 calculation
    if "_Global" not in acField:
        M_gas *= 1.12  # mean shift derived from L75n455TNG z=0
        print("Warning: correcting [%s] for non-global r200 calculation (~10%% difference)" % acField)

    with np.errstate(invalid="ignore", divide="ignore"):
        vals = M_gas / M_tot

    return vals


fgas_r200.label = r"$\rm{f_{gas}(<r_{200c})}$"
fgas_r200.units = ""  # dimensionless
fgas_r200.limits = [0.0, 0.2]
fgas_r200.log = False


@catalog_field
def fgas_r500(sim, field):
    """Gas mass fraction (= Mgas/Mtot) within r500c. FoF-scope approximation."""
    M_gas = sim.subhalos("mgas_r500")
    M_tot = sim.subhalos("mhalo_500")

    with np.errstate(invalid="ignore", divide="ignore"):
        vals = M_gas / M_tot

    return vals


fgas_r500.label = r"$\rm{f_{gas}(<r_{500c})}$"
fgas_r500.units = ""  # dimensionless
fgas_r500.limits = [0.0, 0.2]
fgas_r500.log = False

# ---------------------------- auxcat: sfr --------------------------------------------------------


@catalog_field(aliases=["sfr_30pkpc", "sfr_30pkpc_instant"])
def sfr(sim, field):
    """Galaxy star formation rate (instantaneous, within 30pkpc aperture)."""
    acField = "Subhalo_GasSFR_30pkpc"
    sfr = sim.auxCat(acField)[acField]  # units correct
    return sfr


sfr.label = r"$\rm{SFR_{<30kpc},instant}$"
sfr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr.limits = [-2.5, 1.0]
sfr.log = True
sfr.auxcat = True


@catalog_field
def sfr_10myr(sim, field):
    """Star formation rate (full subhalo) averaged over the past 10 Myr."""
    acField = "Subhalo_StellarMassFormed_10myr"

    dt_yr = 1e6 * 10  # 10 Myr

    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_10myr.label = r"$\rm{SFR_{10Myr}}$"
sfr_10myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_10myr.limits = [-2.5, 1.0]
sfr_10myr.log = True


@catalog_field
def sfr_30pkpc_10myr(sim, field):
    """Star formation rate (30pkpc) averaged over the past 10 Myr."""
    acField = "Subhalo_StellarMassFormed_10myr_30pkpc"

    dt_yr = 1e6 * 10  # 10 Myr

    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_30pkpc_10myr.label = r"$\rm{SFR_{<30kpc,10Myr}}$"
sfr_30pkpc_10myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_30pkpc_10myr.limits = [-2.5, 1.0]
sfr_30pkpc_10myr.log = True


@catalog_field
def sfr_50myr(sim, field):
    """Star formation rate (full subhalo) averaged over the past 50 Myr."""
    acField = "Subhalo_StellarMassFormed_50myr"

    dt_yr = 1e6 * 50  # 50 Myr

    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_50myr.label = r"$\rm{SFR_{50Myr}}$"
sfr_50myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_50myr.limits = [-2.5, 1.0]
sfr_50myr.log = True


@catalog_field
def sfr_30pkpc_50myr(sim, field):
    """Star formation rate (30pkpc) averaged over the past 50 Myr."""
    acField = "Subhalo_StellarMassFormed_50myr_30pkpc"

    dt_yr = 1e6 * 50  # 50 Myr
    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_30pkpc_50myr.label = r"$\rm{SFR_{<30kpc,50Myr}}$"
sfr_30pkpc_50myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_30pkpc_50myr.limits = [-2.5, 1.0]
sfr_30pkpc_50myr.log = True


@catalog_field
def sfr_100myr(sim, field):
    """Star formation rate (full subhalo) averaged over the past 100 Myr."""
    acField = "Subhalo_StellarMassFormed_100myr"

    dt_yr = 1e6 * 100  # 100 Myr

    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_100myr.label = r"$\rm{SFR_{100Myr}}$"
sfr_100myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_100myr.limits = [-2.5, 1.0]
sfr_100myr.log = True


@catalog_field
def sfr_30pkpc_100myr(sim, field):
    """Star formation rate (30pkpc) averaged over the past 100 Myr."""
    acField = "Subhalo_StellarMassFormed_100myr_30pkpc"

    dt_yr = 1e6 * 100  # 100 Myr
    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr  # msun/yr

    return vals


sfr_30pkpc_100myr.label = r"$\rm{SFR_{<30kpc,100Myr}}$"
sfr_30pkpc_100myr.units = r"$\rm{M_{sun}\, yr^{-1}}$"
sfr_30pkpc_100myr.limits = [-2.5, 1.0]
sfr_30pkpc_100myr.log = True


@catalog_field(alias="sfr_10_100")
def sfr_10_100_ratio(sim, field):
    """Ratio of SFR_10Myr / SFR_100Myr (full subhalo) as a burstyness indicator."""
    sfr_10 = sim.subhalos("sfr_10myr")
    sfr_100 = sim.subhalos("sfr_100myr")

    with np.errstate(invalid="ignore"):
        ratio = sfr_10 / sfr_100

    return ratio


sfr_10_100_ratio.label = r"$\rm{SFR_{10Myr}}$ / $\rm{SFR_{100Myr}}$"
sfr_10_100_ratio.units = r""  # dimensionless
sfr_10_100_ratio.limits = [-1.0, 4.0]
sfr_10_100_ratio.log = True


@catalog_field(alias="ssfr_30pkpc_instant")
def ssfr_30pkpc(sim, field):
    """Galaxy specific star formation rate [1/yr] (sSFR, instantaneous, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc")
    mstar = sim.subhalos("mstar_30pkpc")

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr


ssfr_30pkpc.label = r"$\rm{sSFR}$"
ssfr_30pkpc.units = r"$\rm{yr^{-1}}$"
ssfr_30pkpc.limits = [-12.0, -8.0]
ssfr_30pkpc.log = True


@catalog_field
def ssfr_30pkpc_10myr(sim, field):
    """Galaxy specific star formation rate [1/yr] (sSFR, 10 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_10myr")
    mstar = sim.subhalos("mstar_30pkpc")

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr


ssfr_30pkpc_10myr.label = r"$\rm{sSFR, 10Myr}$"
ssfr_30pkpc_10myr.units = r"$\rm{yr^{-1}}$"
ssfr_30pkpc_10myr.limits = [-12.0, -8.0]
ssfr_30pkpc_10myr.log = True


@catalog_field
def ssfr_30pkpc_50myr(sim, field):
    """Galaxy specific star formation rate [1/yr] (sSFR, 50 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_50myr")
    mstar = sim.subhalos("mstar_30pkpc")

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr


ssfr_30pkpc_50myr.label = r"$\rm{sSFR, 50Myr}$"
ssfr_30pkpc_50myr.units = r"$\rm{yr^{-1}}$"
ssfr_30pkpc_50myr.limits = [-12.0, -8.0]
ssfr_30pkpc_50myr.log = True


@catalog_field
def ssfr_30pkpc_100myr(sim, field):
    """Galaxy specific star formation rate [1/yr] (sSFR, 100 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_100myr")
    mstar = sim.subhalos("mstar_30pkpc")

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr


ssfr_30pkpc_100myr.label = r"$\rm{sSFR, 100Myr}$"
ssfr_30pkpc_100myr.units = r"$\rm{yr^{-1}}$"
ssfr_30pkpc_100myr.limits = [-12.0, -8.0]
ssfr_30pkpc_100myr.log = True


@catalog_field(alias="sfr_surfdens_30pkpc_instant")
def sfr_surfdens(sim, field):
    """Star formation surface density (instantaneous, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc")
    area = np.pi * (30.0) ** 2  # kpc^2
    vals = sfr / area  # msun/yr/kpc^2
    return vals


sfr_surfdens.label = r"$\rm{\Sigma_{SFR}}$"
sfr_surfdens.units = r"$\rm{M_\odot \, yr^{-1} \, kpc^{-2}}$"
sfr_surfdens.limits = [-7.0, -1.0]
sfr_surfdens.log = True


@catalog_field
def sfr_surfdens_30pkpc_10myr(sim, field):
    """Star formation surface density (10 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_10myr")
    area = np.pi * (30.0) ** 2  # kpc^2
    vals = sfr / area  # msun/yr/kpc^2
    return vals


sfr_surfdens_30pkpc_10myr.label = r"$\rm{\Sigma_{SFR}}$"
sfr_surfdens_30pkpc_10myr.units = r"$\rm{M_\odot \, yr^{-1} \, kpc^{-2}}$"
sfr_surfdens_30pkpc_10myr.limits = [-7.0, -1.0]
sfr_surfdens_30pkpc_10myr.log = True


@catalog_field
def sfr_surfdens_30pkpc_50myr(sim, field):
    """Star formation surface density (50 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_50myr")
    area = np.pi * (30.0) ** 2  # kpc^2
    vals = sfr / area  # msun/yr/kpc^2
    return vals


sfr_surfdens_30pkpc_50myr.label = r"$\rm{\Sigma_{SFR}}$"
sfr_surfdens_30pkpc_50myr.units = r"$\rm{M_\odot \, yr^{-1} \, kpc^{-2}}$"
sfr_surfdens_30pkpc_50myr.limits = [-7.0, -1.0]
sfr_surfdens_30pkpc_50myr.log = True


@catalog_field
def sfr_surfdens_30pkpc_100myr(sim, field):
    """Star formation surface density (100 Myr, SFR and M* within 30kpc)."""
    sfr = sim.subhalos("sfr_30pkpc_100myr")
    area = np.pi * (30.0) ** 2  # kpc^2
    vals = sfr / area  # msun/yr/kpc^2
    return vals


sfr_surfdens_30pkpc_100myr.label = r"$\rm{\Sigma_{SFR}}$"
sfr_surfdens_30pkpc_100myr.units = r"$\rm{M_\odot \, yr^{-1} \, kpc^{-2}}$"
sfr_surfdens_30pkpc_100myr.limits = [-7.0, -1.0]
sfr_surfdens_30pkpc_100myr.log = True

# ---------------------------- auxcat: gas observables --------------------------------------------


@catalog_field
def szy_r500c_3d(sim, field):
    """Sunyaev Zeldovich y-parameter within r500c (3d)."""
    acField = "Subhalo_SZY_R500c_3D"
    ac = sim.auxCat(fields=[acField])

    # unit conversion [kpc^2] -> [Mpc^2]
    vals = ac[acField] * 1e-6

    return vals


szy_r500c_3d.label = r"$\rm{Y_{SZ,r500}^{3d}}$"
szy_r500c_3d.units = r"$\rm{Mpc^2}$"
szy_r500c_3d.limits = [-6.0, -3.0]
szy_r500c_3d.log = True


@catalog_field
def szy_r500c_2d(sim, field):
    """Sunyaev Zeldovich y-parameter within r500c (2d)."""
    acField = "Subhalo_SZY_R500c_2D_d=r200"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = 10.0 ** ac[acField] * 1e-6  # log pkpc^2 -> linear pMpc^2

    vals = vals[:, 0]  # select first view direction

    return vals


szy_r500c_2d.label = r"$\rm{Y_{SZ,r500}^{2d}}$"
szy_r500c_2d.units = r"$\rm{Mpc^2}$"
szy_r500c_2d.limits = [-6.0, -3.0]
szy_r500c_2d.log = True


@catalog_field
def xray_r500(sim, field):
    """Bolometric X-ray luminosity (simple free-free model), within r500c."""
    # note: computed per group, e.g. for centrals only
    acField = "Group_XrayBolLum_Crit500"
    ac = sim.auxCat(fields=[acField])[acField]

    vals = ac.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]
    vals = groupOrderedValsToSubhaloOrdered(vals, sim)

    return vals


xray_r500.label = r"L$_{\rm X}$ Bolometric ($R_{500c}$)"
xray_r500.units = r"$\rm{erg/s}$"
xray_r500.limits = [37, 42]
xray_r500.log = True


@catalog_field
def xray_subhalo(sim, field):
    """Bolometric X-ray luminosity (simple free-free model), within full subhalo."""
    acField = "Subhalo_XrayBolLum"
    ac = sim.auxCat(fields=[acField])[acField]

    vals = ac.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]

    return vals


xray_subhalo.label = r"L$_{\rm X}$ Bolometric"
xray_subhalo.units = r"$\rm{erg/s}$"
xray_subhalo.limits = [37, 42]
xray_subhalo.log = True


@catalog_field
def xray_05_2kev_r500(sim, field):
    """X-ray luminosity 0.5-2.0 keV (APEC model), within r500c in 3D."""
    acField = "Subhalo_XrayLum_0.5-2.0kev"
    ac = sim.auxCat(fields=[acField])[acField]

    vals = ac.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]

    return vals


xray_05_2kev_r500.label = r"L$_{\rm X,r500}^{\rm 0.5-2 keV}$"
xray_05_2kev_r500.units = r"$\rm{erg/s}$"
xray_05_2kev_r500.limits = [37, 42]
xray_05_2kev_r500.log = True


@catalog_field
def xray_05_2kev_r500_halo(sim, field):
    """X-ray luminosity 0.5-2.0 keV (APEC model), within r500c in 3D (FoF-scope)."""
    acField = "Group_XrayLum_0.5-2.0kev_Crit500"
    ac = sim.auxCat(fields=[acField])[acField]

    vals = ac.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]
    vals = groupOrderedValsToSubhaloOrdered(vals, sim)

    return vals


xray_05_2kev_r500_halo.label = r"L$_{\rm X,r500}^{\rm 0.5-2 keV}$"
xray_05_2kev_r500_halo.units = r"$\rm{erg/s}$"
xray_05_2kev_r500_halo.limits = [37, 42]
xray_05_2kev_r500_halo.log = True


@catalog_field
def xray_01_24kev_r500_halo(sim, field):
    """X-ray luminosity 0.1-2.4 keV (APEC model), within r500c in 3D (FoF-scope)."""
    acField = "Group_XrayLum_0.1-2.4kev_Crit500"
    ac = sim.auxCat(fields=[acField])[acField]

    vals = ac.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]
    vals = groupOrderedValsToSubhaloOrdered(vals, sim)

    return vals


xray_01_24kev_r500_halo.label = r"L$_{\rm X,r500}^{\rm 0.1-2.4 keV}$"
xray_01_24kev_r500_halo.units = r"$\rm{erg/s}$"
xray_01_24kev_r500_halo.limits = [37, 42]
xray_01_24kev_r500_halo.log = True


@catalog_field
def xraylum_r500c_2d(sim, field):
    """X-ray luminosity (0.5-2.0 keV) within r500c (2d)."""
    acField = "Subhalo_XrayLum_0.5-2.0kev_R500c_2D_d=r200"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = 10.0 ** ac[acField].astype("float64")  # log erg/s -> linear erg/s
    vals = vals[:, 0]  # select first view direction

    return vals


xraylum_r500c_2d.label = r"$\rm{L_{X,r500}^{2d}}$"
xraylum_r500c_2d.units = r"$\rm{erg/s}$"
xraylum_r500c_2d.limits = [41.0, 46.0]
xraylum_r500c_2d.log = True


@catalog_field(aliases=["xray_peak_offset_2d_rvir", "xray_peak_offset_2d_r500"])
def xray_peak_offset_2d(sim, field):
    """Spatial offset between X-ray (0.5-2.0 keV) emission peak and galaxy (SubhaloPos). In 2D projection."""
    acField = "Subhalo_XrayOffset_2D"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = ac[acField][:, 0]  # select first view direction

    # what kind of distance?
    if "_rvir" in field or "_r500" in field:
        rField = "Group_R_Crit500" if "500" in field else "Group_R_Crit500"

        halos = sim.groupCat(fieldsHalos=[rField, "GroupFirstSub"])
        GrNr = sim.subhalos("SubhaloGrNr")

        rad = halos[rField][GrNr]

        vals /= rad  # linear, relative to halo radius
    else:
        vals = sim.units.codeLengthToKpc(vals)  # code -> pkpc

    return vals


xray_peak_offset_2d.label = (
    lambda sim, f: r"$\rm{\Delta_{X-ray,galaxy}^{2d}}$"
    if f.endswith("_2d")
    else (
        r"$\rm{\Delta_{X-ray,galaxy} / R_{vir}}$" if f.endswith("_rvir") else r"$\rm{\Delta_{X-ray,galaxy} / R_{500}}$"
    )
)
xray_peak_offset_2d.units = lambda sim, f: r"$\rm{kpc}$" if f.endswith("_2d") else ""  # linear dimensionless
xray_peak_offset_2d.limits = lambda sim, f: [0.0, 2.5] if f.endswith("_2d") else [-2.0, 0.0]
xray_peak_offset_2d.log = True


@catalog_field
def tcool_halo_ovi(sim, field):
    """Mean cooling time of halo gas, weighted by OVI mass."""
    acField = "Subhalo_CoolingTime_OVI_HaloGas"
    vals = sim.auxCat(fields=[acField])[acField]  # Gyr

    return vals


tcool_halo_ovi.label = r"$\rm{t_{cool,halo,OVI}}$"
tcool_halo_ovi.units = r"$\rm{Gyr}$"
tcool_halo_ovi.limits = [-0.5, 1.5]
tcool_halo_ovi.log = True


@catalog_field(aliases=["p_sync_ska_eta43", "p_sync_ska_alpha15"])
def p_sync_ska(sim, field):
    """Synchrotron power radio emission (SKA model)."""
    acField = "Subhalo_SynchrotronPower_SKA"
    if field.endswith("_eta43"):
        acField += "_eta43"
    elif field.endswith("_alpha15"):
        acField += "_alpha15"

    vals = sim.auxCat(fields=[acField])[acField]

    return vals


p_sync_ska.label = r"$\rm{P_{sync,SKA}}$"
p_sync_ska.units = r"$\rm{W / Hz}$"
p_sync_ska.limits = [16, 26]
p_sync_ska.log = True


@catalog_field(aliases=["p_sync_vla_eta43", "p_sync_vla_alpha15"])
def p_sync_vla(sim, field):
    """Synchrotron power radio emission (VLA model)."""
    acField = "Subhalo_SynchrotronPower_VLA"
    if field.endswith("_eta43"):
        acField += "_eta43"
    elif field.endswith("_alpha15"):
        acField += "_alpha15"

    vals = sim.auxCat(fields=[acField])[acField]

    return vals


p_sync_ska.label = r"$\rm{P_{sync,VLA}}$"
p_sync_ska.units = r"$\rm{W / Hz}$"
p_sync_ska.limits = [16, 26]
p_sync_ska.log = True

# ---------------------------- auxcat: gas emission (cloudy-based) -----------------------------------


@catalog_field
def lum_civ1551_outercgm(sim, field):
    """CIV 1551 luminosity in the outer CGM."""
    acField = "Subhalo_CIV1551_Lum_OuterCGM"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = ac[acField].astype("float64") * 1e30  # 1e30 erg/s -> erg/s

    return vals


lum_civ1551_outercgm.label = r"$\rm{L_{CIV 1551} (R_{200c}/2 - R_{200c})}$"
lum_civ1551_outercgm.units = r"$\rm{erg/s}$"
lum_civ1551_outercgm.limits = [36.0, 45.0]
lum_civ1551_outercgm.log = True


@catalog_field
def lum_civ1551_innercgm(sim, field):
    """CIV 1551 luminosity in the inner CGM."""
    acField = "Subhalo_CIV1551_Lum_InnerCGM"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = ac[acField].astype("float64") * 1e30  # 1e30 erg/s -> erg/s

    return vals


lum_civ1551_innercgm.label = r"$\rm{L_{CIV 1551} (20 kpc - R_{200c}/2)}$"
lum_civ1551_innercgm.units = r"$\rm{erg/s}$"
lum_civ1551_innercgm.limits = [36.0, 45.0]
lum_civ1551_innercgm.log = True


@catalog_field
def lum_heii1640_outercgm(sim, field):
    """HeII 1640 luminosity in the outer CGM."""
    acField = "Subhalo_HeII1640_Lum_OuterCGM"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = ac[acField].astype("float64") * 1e30  # 1e30 erg/s -> erg/s

    return vals


lum_heii1640_outercgm.label = r"$\rm{L_{HeII 1640} (R_{200c}/2 - R_{200c})}$"
lum_heii1640_outercgm.units = r"$\rm{erg/s}$"
lum_heii1640_outercgm.limits = [36.0, 45.0]
lum_heii1640_outercgm.log = True


@catalog_field
def lum_heii1640_innercgm(sim, field):
    """HeII 1640 luminosity in the inner CGM."""
    acField = "Subhalo_HeII1640_Lum_InnerCGM"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    vals = ac[acField].astype("float64") * 1e30  # 1e30 erg/s -> erg/s

    return vals


lum_heii1640_innercgm.label = r"$\rm{L_{HeII 1640} (20 kpc - R_{200c}/2)}$"
lum_heii1640_innercgm.units = r"$\rm{erg/s}$"
lum_heii1640_innercgm.limits = [36.0, 45.0]
lum_heii1640_innercgm.log = True


@catalog_field
def mg2_lum(sim, field):
    """MgII emission, total luminosity."""
    acField = "Subhalo_MgII_Lum_DustDepleted"
    vals = sim.auxCat(acField)[acField]
    vals = vals.astype("float64") * 1e30  # 1e30 erg/s -> erg/s

    return vals


mg2_lum.label = r"$\rm{L_{MgII}}$"
mg2_lum.units = r"$\rm{erg/s}$"
mg2_lum.limits = [37, 42]
mg2_lum.log = True


@catalog_field
def mg2_lumsize(sim, field):
    """MgII emission, size (half-light radius)."""
    acField = "Subhalo_MgII_LumSize_DustDepleted"
    vals = sim.auxCat(acField)[acField]
    vals = sim.units.codeLengthToKpc(vals)  # code -> kpc

    return vals


mg2_lumsize.label = r"L$_{\rm MgII}$ Half-light Radius"
mg2_lumsize.units = r"kpc"
mg2_lumsize.limits = [1, 10]
mg2_lumsize.log = False


@catalog_field
def mg2_lumsize_rel(sim, field):
    """MgII emission, size (half-light radius) relative to stellar half mass radius."""
    acField = "Subhalo_MgII_LumSize_DustDepleted"
    vals = sim.auxCat(acField)[acField]

    rhalf_stars = sim.subhalos("rhalf_stars_code")
    vals /= rhalf_stars

    return vals


mg2_lumsize_rel.label = r"L$_{\rm MgII}$ Half-light Radius / R$_{\rm 1/2,\star}$"
mg2_lumsize_rel.units = r""  # dimensionless
mg2_lumsize_rel.limits = [-0.5, 0.5]
mg2_lumsize_rel.log = True


@catalog_field
def mg2_m20(sim, field):
    """MgII emission, M20 statistic."""
    acField = "Subhalo_MgII_Emission_Grid2D_M20"
    vals = np.squeeze(sim.auxCat(acField)[acField])

    return vals


mg2_m20.label = r"MgII Emission M$_{\rm 20}$ Index"
mg2_m20.units = r""  # dimensionless
mg2_m20.limits = [-3.0, 0.5]
mg2_m20.log = False


@catalog_field
def mg2_concentration(sim, field):
    """MgII emission, concentration statistic."""
    acField = "Subhalo_MgII_LumConcentration_DustDepleted"
    vals = sim.auxCat(acField)[acField]

    return vals


mg2_concentration.label = r"MgII Emission Concentration (C)"
mg2_concentration.units = r""  # dimensionless
mg2_concentration.limits = [2.0, 5.0]
mg2_concentration.log = False


@catalog_field(multi="mg2_shape_")
def mg2_shape_(sim, field):
    """MgII emission, shape (axis ratio) statistic  (at some isophotal level)."""
    acField = "Subhalo_MgII_Emission_Grid2D_Shape"
    ac = sim.auxCat(acField)[acField]

    isophot_level = float(field.split("mg2_shape_")[1])
    isophot_inds = np.where(ac[acField + "_attrs"]["isophot_levels"] == isophot_level)[0]
    assert len(isophot_inds) == 1, "Failed to find shape at requested isophot level."

    vals = ac[acField][:, isophot_inds[0]]

    return vals


mg2_shape_.label = r"MgII Emission Shape (Axis Ratio)"
mg2_shape_.units = r""  # dimensionless
mg2_shape_.limits = [0.95, 2.4]
mg2_shape_.log = False


@catalog_field(multi="mg2_area_")
def mg2_area_(sim, field):
    """MgII emission, total area (at some isophotal level)."""
    acField = "Subhalo_MgII_Emission_Grid2D_Area"
    ac = sim.auxCat(acField)[acField]

    isophot_level = float(field.split("mg2_area_")[1])
    isophot_inds = np.where(ac[acField + "_attrs"]["isophot_levels"] == isophot_level)[0]
    assert len(isophot_inds) == 1, "Failed to find shape at requested isophot level."

    vals = ac[acField][:, isophot_inds[0]]
    vals = sim.units.codeAreaToKpc2(vals)  # (ckpc/h)^2 -> kpc^2

    return vals


mg2_area_.label = r"MgII Emission Area"
mg2_area_.units = r"kpc$^2$"
mg2_area_.limits = [1.0, 4.0]
mg2_area_.log = True


@catalog_field(multi="mg2_gini_")
def mg2_gini_(sim, field):
    """MgII emission, Gini statistic (at some isophotal level)."""
    acField = "Subhalo_MgII_Emission_Grid2D_Gini"
    ac = sim.auxCat(acField)[acField]

    isophot_level = float(field.split("mg2_gini_")[1])
    isophot_inds = np.where(ac[acField + "_attrs"]["isophot_levels"] == isophot_level)[0]
    assert len(isophot_inds) == 1, "Failed to find shape at requested isophot level."

    vals = ac[acField][:, isophot_inds[0]]

    return vals


mg2_gini_.label = r"MgII Emission Gini Coefficient"
mg2_gini_.units = r""  # dimensionless
mg2_gini_.limits = [0.0, 1.0]
mg2_gini_.log = False

# ---------------------------- auxcat: metallicity ---------------------------------------------------


@catalog_field
def z_stars_masswt(sim, field):
    """Stellar metallicity (no radial restriction), mass weighted."""
    acField = "Subhalo_StellarZ_NoRadCut_MassWt"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])


z_stars_masswt.label = r"$\rm{Z_{\star,masswt}}$"
z_stars_masswt.units = r"$\rm{Z_{\odot}}$"
z_stars_masswt.limits = [-3.0, 0.5]
z_stars_masswt.log = True


@catalog_field
def z_stars_1kpc_masswt(sim, field):
    """Stellar metallicity (within 1 kpc, fof-scope), mass weighted."""
    acField = "Subhalo_StellarZ_1kpc_FoF_MassWt"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])


z_stars_1kpc_masswt.label = r"$\rm{Z_{\star,masswt}}$"
z_stars_1kpc_masswt.units = r"$\rm{Z_{\odot}}$"
z_stars_1kpc_masswt.limits = [-3.0, 0.5]
z_stars_1kpc_masswt.log = True


@catalog_field
def z_stars_2rhalfstarsfof_masswt(sim, field):
    """Stellar metallicity (within 2rhalfstars of fof, and fof-scope), mass weighted."""
    acField = "Subhalo_StellarZ_2rhalfstars-FoF_MassWt"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])


z_stars_2rhalfstarsfof_masswt.label = r"$\rm{Z_{\star,masswt}}$"
z_stars_2rhalfstarsfof_masswt.units = r"$\rm{Z_{\odot}}$"
z_stars_2rhalfstarsfof_masswt.limits = [-3.0, 0.5]
z_stars_2rhalfstarsfof_masswt.log = True


@catalog_field
def z_stars_fof_masswt(sim, field):
    """Stellar metallicity (full fof-scope), mass weighted. All satellites have same value as their central."""
    acField = "Subhalo_StellarZ_FoF_MassWt"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])


z_stars_fof_masswt.label = r"$\rm{Z_{\star,masswt}}$"
z_stars_fof_masswt.units = r"$\rm{Z_{\odot}}$"
z_stars_fof_masswt.limits = [-3.0, 0.5]
z_stars_fof_masswt.log = True


@catalog_field
def z_gas_sfrwt(sim, field):
    """Gas-phase metallicity (no radial restriction), mass weighted."""
    acField = "Subhalo_GasZ_NoRadCut_SfrWt"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])


z_gas_sfrwt.label = r"$\rm{Z_{gas,sfrwt}}$"
z_gas_sfrwt.units = r"$\rm{Z_{\odot}}$"
z_gas_sfrwt.limits = [-3.0, 0.5]
z_gas_sfrwt.log = True


@catalog_field
def z_stars_halo(sim, field):
    """Mean stellar metallicity in the halo (0.15 < r/rvir < 1.0), mass weighted."""
    fieldName1 = "Subhalo_Mass_HaloStars"
    fieldName2 = "Subhalo_Mass_HaloStars_Metal"
    ac1 = sim.auxCat(fields=[fieldName1])[fieldName1]  # code mass units
    ac2 = sim.auxCat(fields=[fieldName2])[fieldName2]  # code mass units

    metallicity_mass_ratio = ac2 / ac1
    vals = sim.units.metallicityInSolar(metallicity_mass_ratio)
    return vals


z_stars_halo.label = r"$\rm{Z_{stars,halo}}$"
z_stars_halo.units = r"$\rm{Z_{\odot}}$"
z_stars_halo.limits = [-3.0, 1.0]
z_stars_halo.log = True


@catalog_field
def z_gas_halo(sim, field):
    """Mean gas metallicity in the halo (0.15 < r/rvir < 1.0), mass weighted."""
    fieldName1 = "Subhalo_Mass_HaloGas"
    fieldName2 = "Subhalo_Mass_HaloGas_Metal"
    ac1 = sim.auxCat(fields=[fieldName1])[fieldName1]  # code mass units
    ac2 = sim.auxCat(fields=[fieldName2])[fieldName2]  # code mass units

    metallicity_mass_ratio = ac2 / ac1
    vals = sim.units.metallicityInSolar(metallicity_mass_ratio)
    return vals


z_gas_halo.label = r"$\rm{Z_{gas,halo}}$"
z_gas_halo.units = r"$\rm{Z_{\odot}}$"
z_gas_halo.limits = [-3.0, 1.0]
z_gas_halo.log = True

# ---------------------------- auxcat: stellar/gas kinematics --------------------------------------------


@catalog_field
def veldisp(sim, field):
    """Stellar velocity dispersion (3D), within the stellar half mass radius."""
    acField = "Subhalo_VelDisp3D_Stars_1rhalfstars"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp.label = r"$\rm{\sigma_{\star}}$"
veldisp.units = r"$\rm{km/s}$"
veldisp.limits = [1.0, 3.0]
veldisp.log = True


@catalog_field
def veldisp1d(sim, field):
    """Stellar velocity dispersion (1D, from 3D), within the stellar half mass radius."""
    acField = "Subhalo_VelDisp1D_Stars_1rhalfstars"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp1d.label = r"$\rm{\sigma_{\star, 1D}}$"
veldisp1d.units = r"$\rm{km/s}$"
veldisp1d.limits = [1.0, 3.0]
veldisp1d.log = True


@catalog_field
def veldisp1d_05re(sim, field):
    """Stellar velocity dispersion (1D, from 3D), within 0.5 times the stellar half mass radius."""
    acField = "Subhalo_VelDisp1D_Stars_05rhalfstars"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp1d_05re.label = r"$\rm{\sigma_{\star, 1D}}$"
veldisp1d_05re.units = r"$\rm{km/s}$"
veldisp1d_05re.limits = [1.0, 2.8]
veldisp1d_05re.log = True


@catalog_field
def veldisp1d_10pkpc(sim, field):
    """Stellar velocity dispersion (1D, in z-direction), within 10pkpc."""
    acField = "Subhalo_VelDisp1Dz_Stars_10pkpc"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp1d_10pkpc.label = r"$\rm{\sigma_{\star, 1D}}$"
veldisp1d_10pkpc.units = r"$\rm{km/s}$"
veldisp1d_10pkpc.limits = [1.0, 2.8]
veldisp1d_10pkpc.log = True


@catalog_field
def veldisp1d_4pkpc2d(sim, field):
    """Stellar velocity dispersion (1D, in z-direction), within 4pkpc (~SDSS fiber low-z) in 2D."""
    acField = "Subhalo_VelDisp1Dz_Stars_4pkpc2D"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp1d_4pkpc2d.label = r"$\rm{\sigma_{\star, 1D}}$"
veldisp1d_4pkpc2d.units = r"$\rm{km/s}$"
veldisp1d_4pkpc2d.limits = [1.0, 2.8]
veldisp1d_4pkpc2d.log = True


@catalog_field
def veldisp_gas_01r500c_xray(sim, field):
    """Gas velocity dispersion (1D, in z-direction), weighted by 0.2-2 keV X-ray luminosity, within 0.1r500c."""
    acField = "Subhalo_VelDisp1Dz_XrayWt_010r500c"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]


veldisp_gas_01r500c_xray.label = r"$\rm{\sigma_{gas, 1D, X-ray, <0.1\,r500c}}$"
veldisp_gas_01r500c_xray.units = r"$\rm{km/s}$"
veldisp_gas_01r500c_xray.limits = [100, 300]  # [1.5, 3.0]
veldisp_gas_01r500c_xray.log = False  # True


@catalog_field
def gas_vrad_2rhalf(sim, field):
    """Mean gas radial velocity within the galaxy (< 2rhalfstars), mass-weighted."""
    acField = "Subhalo_Gas_RadialVel_2rhalfstars_massWt"
    vals = sim.auxCat(acField)[acField]  # physical km/s (negative = inwards)

    return vals


gas_vrad_2rhalf.label = r"Gas v$_{\rm rad,ISM}$"
gas_vrad_2rhalf.units = r"$\rm{km/s}$"
gas_vrad_2rhalf.limits = [-300, 300]
gas_vrad_2rhalf.log = False


@catalog_field
def gas_vrad_halo(sim, field):
    """Mean gas radial velocity the halo (0.15 < r/rvir < 1), mass-weighted."""
    acField = "Subhalo_Gas_RadialVel_halo_massWt"
    vals = sim.auxCat(acField)[acField]  # physical km/s (negative = inwards)

    return vals


gas_vrad_halo.label = r"Gas v$_{\rm rad,halo}$"
gas_vrad_halo.units = r"$\rm{km/s}$"
gas_vrad_halo.limits = [-150, 150]
gas_vrad_halo.log = False

# ---------------------------- auxcat: virshock ------------------------------------------------------


@catalog_field
def rshock(sim, field):
    """Virial shock radius, fiducial model choice."""
    return sim.subhalos("rshock_ShocksMachNum_m2p2_kpc")


rshock.label = r"$\rm{R_{shock}}$"
rshock.units = r"$\rm{kpc}$"
rshock.limits = [1.6, 3.2]
rshock.log = True


@catalog_field
def rshock_rvir(sim, field):
    """Virial shock radius, fiducial model choice. Normalized."""
    return sim.subhalos("rshock_ShocksMachNum_m2p2")


rshock_rvir.label = r"$\rm{R_{shock} / R_{vir}}$"
rshock_rvir.units = ""  # linear dimensionless
rshock_rvir.limits = [0.0, 4.0]
rshock_rvir.log = False


@catalog_field(multi="rshock_")
def rshock_(sim, field):
    """Virial shock radius. [pkpc or rvir]."""
    # "rshock_{Temp,Entropy,RadVel,ShocksMachNum,ShocksEnergyDiss}_mXpY_{kpc,rvir}"
    maps = {
        "temp": "Temp",
        "entropy": "Entropy",
        "radvel": "RadVel",
        "shocksmachnum": "ShocksMachNum",
        "shocksenergydiss": "ShocksEnergyDiss",
    }

    fieldName = maps[field.split("_")[1]]
    methodPerc = field.split("_")[2]

    acField = "Subhalo_VirShockRad_%s_400rad_16ns" % fieldName
    methodInd = int(methodPerc[1])
    percInd = int(methodPerc[3])

    # load and expand
    ac = sim.auxCat(acField)

    rr = ac[acField][:, methodInd, percInd]  # rvir units, linear

    vals = np.zeros(sim.numSubhalos, dtype="float32")
    vals.fill(np.nan)
    vals[ac["subhaloIDs"]] = rr

    if "_kpc" in field:
        r200 = sim.subhalos("r200")  # pkpc
        vals *= r200

    return vals


rshock_.label = lambda sim, f: r"$\rm{R_{shock}" if "_kpc" in f else r"$\rm{R_{shock} / R_{vir}}$"
rshock_.units = lambda sim, f: r"$\rm{kpc}$" if "_kpc" in f else ""  # linear dimensionless
rshock_.limits = lambda sim, f: [1.6, 3.2] if "_kpc" in f else [0.0, 4.0]
rshock_.log = lambda sim, f: True if "_kpc" in f else False

# ---------------------------- auxcat: gas sizes/shapes ------------------------------------------------------


@catalog_field
def size_halpha(sim, field):
    """Half-light radius of H-alpha emission, based on SFR empirical relation."""
    acField = "Subhalo_Gas_Halpha_HalfRad"
    ac = sim.auxCat(acField)
    vals = sim.units.codeLengthToKpc(ac[acField])

    return vals


size_halpha.label = r"r$_{\rm 1/2,H\alpha}$"
size_halpha.units = r"$\rm{kpc}$"
size_halpha.limits = [0.0, 1.5]
size_halpha.log = True


@catalog_field
def size_halpha_em(sim, field):
    """Half-light radius of H-alpha emission, based on cloudy."""
    acField = "Subhalo_Gas_H-alpha_HalfRad"
    ac = sim.auxCat(acField)
    vals = sim.units.codeLengthToKpc(ac[acField])

    return vals


size_halpha_em.label = r"r$_{\rm 1/2,H\alpha}$"
size_halpha_em.units = r"$\rm{kpc}$"
size_halpha_em.limits = [0.0, 1.5]
size_halpha_em.log = True


@catalog_field
def shape_q_sfrgas(sim, field):
    """Iterative ellipsoid shape measurement: axis ratio (q) of star-forming gas."""
    acField = "Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]
    vals = vals[:, 0]

    return vals


shape_q_sfrgas.label = r"q$_{\rm SFRgas}$"
shape_q_sfrgas.units = ""  # dimensionless
shape_q_sfrgas.limits = [0.1, 0.9]
shape_q_sfrgas.log = False


@catalog_field
def shape_s_sfrgas(sim, field):
    """Iterative ellipsoid shape measurement: sphericity (s) of star-forming gas."""
    acField = "Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]
    vals = vals[:, 1]

    return vals


shape_s_sfrgas.label = r"s$_{\rm SFRgas}$"
shape_s_sfrgas.units = ""  # dimensionless
shape_s_sfrgas.limits = [0.1, 0.9]
shape_s_sfrgas.log = False


@catalog_field
def shape_ratio_sfrgas(sim, field):
    """Iterative ellipsoid shape measurement: ratio (s/q) of star-forming gas."""
    acField = "Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]

    vals = vals[:, 1] / vals[:, 0]

    return vals


shape_ratio_sfrgas.label = r"(s/q)$_{\rm SFRgas}$"
shape_ratio_sfrgas.units = ""  # dimensionless
shape_ratio_sfrgas.limits = [0.1, 0.9]
shape_ratio_sfrgas.log = False

# ---------------------------- auxcat: stellar sizes/shapes ------------------------------------------------------


@catalog_field(multi="re_stars_", alias="re_stars")
def re_stars_(sim, field):
    """Half light radii (effective optical radii R_e) of optical light from stars, in a given band.
    Testing: z-axis 2D (random) projection."""
    acField = "Subhalo_HalfLightRad_p07c_cf00dust_z"
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    # find requested band
    band = field.split("re_stars_")[1].lower()

    bands = ac[acField + "_attrs"]["bands"]
    if isinstance(bands[0], (list, np.ndarray)):
        bands = bands[0]  # remove nested
    bands = list(bands)
    assert band.encode("utf-8") in bands

    bandInd = bands.index(band.encode("utf-8"))

    vals = ac[acField][:, bandInd]

    if "_code" not in field:
        vals = sim.units.codeLengthToKpc(vals)

    return vals


re_stars_.label = r"R$_{\rm e,\star}$"
re_stars_.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
re_stars_.limits = [0.0, 1.5]
re_stars_.log = True


@catalog_field
def r80_stars(sim, field):
    """3D radius enclosing 80% of stellar mass, non-standard."""
    acField = "Subhalo_Stars_R80"
    vals = sim.auxCat(fields=[acField])[acField]
    vals = sim.units.codeLengthToKpc(vals)

    return vals


r80_stars.label = r"R$_{\rm 80,\star}$"
r80_stars.units = lambda sim, f: r"$\rm{kpc}$"
r80_stars.limits = [0.0, 1.5]
r80_stars.log = True


@catalog_field
def sigma1kpc_stars(sim, field):
    """Stellar surface density within a central 1 pkpc (2D projected) aperture."""
    acField = "Subhalo_Mass_1pkpc_2D_Stars"
    vals = sim.auxCat(fields=[acField])[acField]
    vals = sim.units.codeMassToMsun(vals)

    area = np.pi * 1.0**2  # kpc^2
    vals /= area  # msun/kpc^2

    return vals


sigma1kpc_stars.label = r"$\Sigma_{\rm 1,\star}$"
sigma1kpc_stars.units = lambda sim, f: r"$\rm{M_{\odot}\, kpc^{-2}}$"
sigma1kpc_stars.limits = [6.5, 11.0]
sigma1kpc_stars.log = True


@catalog_field(alias="rhalf_stars_fof_code")
def rhalf_stars_fof(sim, field):
    """Stellar half-mass radius, computed from all FoF-scope stars."""
    acField = "Subhalo_Stars_R50_FoF"
    ac = sim.auxCat(acField, expandPartial=True)  # saved only for centrals

    # assign same value to all subhalos of each halo
    grnr = sim.subhalos("SubhaloGrNr")
    firstsub = sim.halos("GroupFirstSub")[grnr]
    vals = ac[acField][firstsub]

    if "_code" not in field:
        vals = sim.units.codeLengthToKpc(vals)

    return vals


rhalf_stars_fof.label = r"r$_{\rm 1/2,\star}$"
rhalf_stars_fof.units = lambda sim, f: r"$\rm{kpc}$" if "_code" not in f else "code_length"
rhalf_stars_fof.limits = [0.0, 1.5]
rhalf_stars_fof.log = True


@catalog_field
def shape_q_stars(sim, field):
    """Iterative ellipsoid shape measurement: axis ratio (q) of stars."""
    acField = "Subhalo_EllipsoidShape_Stars_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]
    vals = vals[:, 0]

    return vals


shape_q_stars.label = r"q$_{\rm stars}$"
shape_q_stars.units = ""  # dimensionless
shape_q_stars.limits = [0.1, 0.9]
shape_q_stars.log = False


@catalog_field
def shape_s_stars(sim, field):
    """Iterative ellipsoid shape measurement: sphericity (s) of stars."""
    acField = "Subhalo_EllipsoidShape_Stars_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]
    vals = vals[:, 1]

    return vals


shape_s_stars.label = r"s$_{\rm stars}$"
shape_s_stars.units = ""  # dimensionless
shape_s_stars.limits = [0.1, 0.9]
shape_s_stars.log = False


@catalog_field
def shape_ratio_stars(sim, field):
    """Iterative ellipsoid shape measurement: ratio (s/q) of stars."""
    acField = "Subhalo_EllipsoidShape_Stars_2rhalfstars_shell"
    vals = sim.auxCat(acField)[acField]

    vals = vals[:, 1] / vals[:, 0]

    return vals


shape_ratio_stars.label = r"(s/q)$_{\rm stars}$"
shape_ratio_stars.units = ""  # dimensionless
shape_ratio_stars.limits = [0.1, 0.9]
shape_ratio_stars.log = False


@catalog_field(aliases=["krot_stars1", "krot_stars2"])
def krot_stars(sim, field):
    """Galaxy disk kappa, fraction of stars in ordered rotation."""
    acField = "Subhalo_StellarRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 0]  # index 0 of 4
    vals = np.squeeze(vals)

    return vals


krot_stars.label = r"$\kappa_{\rm stars, rot}$"
krot_stars.units = ""  # dimensionless
krot_stars.limits = [0.1, 0.8]
krot_stars.log = False


@catalog_field(aliases=["krot_gas1", "krot_gas2"])
def krot_gas(sim, field):
    """Galaxy disk kappa, fraction of gas in ordered rotation."""
    acField = "Subhalo_GasRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 0]  # index 0 of 4
    vals = np.squeeze(vals)

    return vals


krot_gas.label = r"$\kappa_{\rm gas, rot}$"
krot_gas.units = ""  # dimensionless
krot_gas.limits = [0.1, 1.0]
krot_gas.log = False


@catalog_field(aliases=["krot_oriented_stars1", "krot_oriented_stars2"])
def krot_oriented_stars(sim, field):
    """Galaxy disk kappa, fraction of stars in ordered rotation (J_z > 0)."""
    acField = "Subhalo_StellarRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 1]  # index 1 of 4
    vals = np.squeeze(vals)

    return vals


krot_oriented_stars.label = r"$\kappa_{\rm stars, rot} (J_z > 0)$"
krot_oriented_stars.units = ""  # dimensionless
krot_oriented_stars.limits = [0.1, 0.8]
krot_oriented_stars.log = False


@catalog_field(aliases=["krot_oriented_gas1", "krot_oriented_gas2"])
def krot_oriented_gas(sim, field):
    """Galaxy disk kappa, fraction of gas in ordered rotation (J_z > 0)."""
    acField = "Subhalo_GasRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 1]  # index 1 of 4
    vals = np.squeeze(vals)

    return vals


krot_oriented_gas.label = r"$\kappa_{\rm gas, rot} (J_z > 0)$"
krot_oriented_gas.units = ""  # dimensionless
krot_oriented_gas.limits = [0.1, 0.8]
krot_oriented_gas.log = False


@catalog_field(aliases=["arot_stars1", "arot_stars2"])
def arot_stars(sim, field):
    """Galaxy disk, fraction of counter-rotating stars."""
    acField = "Subhalo_StellarRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 2]  # index 2 of 4
    vals = np.squeeze(vals)

    return vals


arot_stars.label = r"$M_{\rm stars, counter-rot} / M_{\rm stars, total}$"
arot_stars.units = ""  # dimensionless
arot_stars.limits = [0.0, 0.6]
arot_stars.log = False


@catalog_field(aliases=["arot_gas1", "arot_gas2"])
def arot_gas(sim, field):
    """Galaxy disk, fraction of counter-rotating gas."""
    acField = "Subhalo_GasRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 2]  # index 2 of 4
    vals = np.squeeze(vals)

    return vals


arot_gas.label = r"$M_{\rm gas, counter-rot} / M_{\rm gas, total}$"
arot_gas.units = ""  # dimensionless
arot_gas.limits = [0.0, 0.4]
arot_gas.log = False


@catalog_field(aliases=["specangmom_stars1", "specangmom_stars2"])
def specangmom_stars(sim, field):
    """Galaxy disk, specific angular momentum of stars."""
    acField = "Subhalo_StellarRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 3]  # index 3 of 4
    vals = np.squeeze(vals)

    return vals


specangmom_stars.label = r"$j_{\rm stars}$"
specangmom_stars.units = r"kpc km/s"
specangmom_stars.limits = [1.0, 5.0]
specangmom_stars.log = True


@catalog_field(aliases=["specangmom_gas1", "specangmom_gas2"])
def specangmom_gas(sim, field):
    """Galaxy disk, specific angular momentum of gas."""
    acField = "Subhalo_GasRotation"
    if field.endswith("2"):
        acField += "_2rhalfstars"
    if field.endswith("1"):
        acField += "_1rhalfstars"

    vals = sim.auxCat(acField)[:, 3]  # index 3 of 4
    vals = np.squeeze(vals)

    return vals


specangmom_gas.label = r"$j_{\rm gas}$"
specangmom_gas.units = r"kpc km/s"
specangmom_gas.limits = [2.0, 5.0]
specangmom_gas.log = True


@catalog_field
def m_bulge_counter_rot(sim, field):
    """M_bulge estimator: twice the counter-rotating stellar mass within the stellar half-mass radius."""
    acField = "Subhalo_StellarRotation_1rhalfstars"

    # load auxCat and groupCat masses
    Arot = np.squeeze(sim.auxCat(acField)[:, 2])  # counter-rotating mass fraction relative to total

    assert np.nanmin(Arot) >= 0.0 and np.nanmax(Arot) <= 1.0

    masses = sim.subhalos("SubhaloMassInHalfRadType")[:, sim.ptNum("stars")]

    # multiply 2 x (massfrac) x (stellar mass) and convert to solar masses
    vals = sim.units.codeMassToMsun(2.0 * Arot * masses)

    return vals


m_bulge_counter_rot.label = r"$M_{\rm bulge}$"
m_bulge_counter_rot.units = r"$M_\odot$"
m_bulge_counter_rot.limits = [8.0, 10.5]
m_bulge_counter_rot.log = True

# ---------------------------- auxcat: magnetic fields --------------------------------------------


@catalog_field(alias="bmag_sfrgt0_volwt")
def bmag_sfrgt0_masswt(sim, field):
    """Mean magnetic field amplitude in the ISM (star-forming gas), full subhalo, mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_SFingGas_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_sfrgt0_masswt.label = r"$\rm{|B|_{ISM}}$"
bmag_sfrgt0_masswt.units = r"$\rm{\mu G}$"
bmag_sfrgt0_masswt.limits = [0.0, 2.0]
bmag_sfrgt0_masswt.log = True


@catalog_field(alias="bmag_2rhalf_volwt")
def bmag_2rhalf_masswt(sim, field):
    """Mean magnetic field amplitude in the ISM (within 2rhalfstars), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_2rhalfstars_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_2rhalf_masswt.label = r"$\rm{|B|_{ISM}}$"
bmag_2rhalf_masswt.units = r"$\rm{\mu G}$"
bmag_2rhalf_masswt.limits = [0.0, 2.0]
bmag_2rhalf_masswt.log = True


@catalog_field(alias="bmag_halo_volwt")
def bmag_halo_masswt(sim, field):
    """Mean magnetic field amplitude in the halo (0.15 < r/rvir < 1.0), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_halo_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_halo_masswt.label = r"$\rm{|B|_{halo}}$"
bmag_halo_masswt.units = r"$\rm{\mu G}$"
bmag_halo_masswt.limits = [-1.5, 0.0]
bmag_halo_masswt.log = True


@catalog_field(alias="bmag_r500_volwt")
def bmag_r500_masswt(sim, field):
    """Mean magnetic field amplitude within r500c, mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_fof_r500_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_r500_masswt.label = r"$\rm{|B|_{r500c}}$"
bmag_r500_masswt.units = r"$\rm{\mu G}$"
bmag_r500_masswt.limits = [-1.5, 0.0]
bmag_r500_masswt.log = True


@catalog_field(alias="bmag_halfr500_volwt")
def bmag_halfr500_masswt(sim, field):
    """Mean magnetic field amplitude within 0.5 * r500c, mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_fof_halfr500_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_halfr500_masswt.label = r"$\rm{|B|_{0.5r500c}}$"
bmag_halfr500_masswt.units = r"$\rm{\mu G}$"
bmag_halfr500_masswt.limits = [-1.0, 0.5]
bmag_halfr500_masswt.log = True


@catalog_field(alias="bmag_volwt")
def bmag_masswt(sim, field):
    """Mean magnetic field amplitude (full subhalo), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Bmag_subhalo_%s" % wtStr
    ac = sim.auxCat(acField, expandPartial=True)

    vals = ac[acField] * 1e6  # Gauss -> microGauss

    return vals


bmag_masswt.label = r"$\rm{|B|}$"
bmag_masswt.units = r"$\rm{\mu G}$"
bmag_masswt.limits = [-1.0, 0.5]
bmag_masswt.log = True


@catalog_field(alias="pratio_halo_volwt")
def pratio_halo_masswt(sim, field):
    """Ratio of magnetic to thermal gas pressure in the halo (0.15 < r/rvir < 1.0), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Pratio_halo_%s" % wtStr
    vals = sim.auxCat(acField)[acField]

    return vals


pratio_halo_masswt.label = r"$P_{\rm B} / P_{\rm gas}$ (halo)"
pratio_halo_masswt.units = ""  # dimensionless
pratio_halo_masswt.limits = [-2.0, 1.0]
pratio_halo_masswt.log = True


@catalog_field(alias="pratio_2rhalf_volwt")
def pratio_2rhalf_masswt(sim, field):
    """Ratio of magnetic to thermal gas pressure in the galaxy (r < 2rhalfstars), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_Pratio_2rhalfstars_%s" % wtStr
    vals = sim.auxCat(acField)[acField]

    return vals


pratio_2rhalf_masswt.label = r"$P_{\rm B} / P_{\rm gas}$ (ISM)"
pratio_2rhalf_masswt.units = ""  # dimensionless
pratio_2rhalf_masswt.limits = [-2.0, 1.0]
pratio_2rhalf_masswt.log = True


@catalog_field(alias="bke_ratio_halo_volwt")
def bke_ratio_halo_masswt(sim, field):
    """Ratio of magnetic to kinetic energy in the halo (0.15 < r/rvir < 1.0), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_uB_uKE_ratio_halo_%s" % wtStr
    vals = sim.auxCat(acField)[acField]

    return vals


bke_ratio_halo_masswt.label = r"$u_{\rm B} / u_{\rm KE}$ (halo)"
bke_ratio_halo_masswt.units = ""  # dimensionless
bke_ratio_halo_masswt.limits = [-2.0, 1.0]
bke_ratio_halo_masswt.log = True


@catalog_field(alias="bke_ratio_2rhalf_volwt")
def bke_ratio_2rhalf_masswt(sim, field):
    """Ratio of magnetic to kinetic energy in the galaxy (r < 2rhalfstars), mass or volume weighted."""
    if "_masswt" in field:
        wtStr = "massWt"
    if "_volwt" in field:
        wtStr = "volWt"

    acField = "Subhalo_uB_uKE_ratio_2rhalfstars_%s" % wtStr
    vals = sim.auxCat(acField)[acField]

    return vals


bke_ratio_2rhalf_masswt.label = r"$u_{\rm B} / u_{\rm KE}$ (ISM)"
bke_ratio_2rhalf_masswt.units = ""  # dimensionless
bke_ratio_2rhalf_masswt.limits = [-2.0, 1.0]
bke_ratio_2rhalf_masswt.log = True


@catalog_field
def ptot_gas_halo(sim, field):
    """Total gas thermal pressure in the halo (0.15 < r/rvir < 1.0)."""
    acField = "Subhalo_Ptot_gas_halo"
    vals = sim.auxCat(acField)[acField]

    return vals


ptot_gas_halo.label = r"P$_{\rm tot,gas}$"
ptot_gas_halo.units = r"$\rm{K/cm^3}$"
ptot_gas_halo.limits = [5.0, 7.0]
ptot_gas_halo.log = True


@catalog_field
def ptot_b_halo(sim, field):
    """Total gas magnetic pressure in the halo (0.15 < r/rvir < 1.0)."""
    acField = "Subhalo_Ptot_B_halo"
    vals = sim.auxCat(acField)[acField]

    return vals


ptot_b_halo.label = r"P$_{\rm tot,B}$"
ptot_b_halo.units = r"$\rm{K/cm^3}$"
ptot_b_halo.limits = [5.0, 7.0]
ptot_b_halo.log = True

# -------------------- smbhs -----------------------------------------------------------


@catalog_field
def m_bh(sim, field):
    """Supermassive black hole mass (dynamical)."""
    # 'total' black hole mass in this subhalo
    # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here
    # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
    vals = sim.subhalos("SubhaloMassType")[:, sim.ptNum("bhs")]
    vals = sim.units.codeMassToMsun(vals)

    return vals


m_bh.label = r"$\rm{M_{BH}}$"
m_bh.units = r"$\rm{M_{\odot}}$"
m_bh.limits = [6.0, 9.0]
m_bh.log = True


@catalog_field
def m_bh(sim, field):
    """Supermassive black hole mass (actual, i.e. starting from the seed mass)."""
    # 'total' black hole mass in this subhalo
    # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here
    # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
    vals = sim.subhalos("SubhaloBHMass")
    vals = sim.units.codeMassToMsun(vals)

    return vals


m_bh.label = r"$\rm{M_{BH}}$"
m_bh.units = r"$\rm{M_{\odot}}$"
m_bh.limits = [6.0, 9.0]
m_bh.log = True


@catalog_field
def bh_mdot_edd(sim, field):
    """Blackhole mass accretion rate normalized by its Eddington rate (for most massive BH in each subhalo)."""
    fields = ["Subhalo_BH_Mdot_largest", "Subhalo_BH_MdotEdd_largest"]
    ac = sim.auxCat(fields=fields)

    vals = ac["Subhalo_BH_Mdot_largest"] / ac["Subhalo_BH_MdotEdd_largest"]

    return vals


bh_mdot_edd.label = r"$\rm{\dot{M}_{BH} / \dot{M}_{Edd}}$"
bh_mdot_edd.units = ""  # dimensionless
bh_mdot_edd.limits = [-5.0, 0.0]
bh_mdot_edd.log = True


@catalog_field
def bh_bollum(sim, field):
    """Blackhole bolometric luminosity (for most massive BH in each subhalo)."""
    acField = "Subhalo_BH_BolLum_largest"
    vals = sim.auxCat(acField)[acField]

    return vals


bh_bollum.label = r"Blackhole $L_{\rm bol}$"
bh_bollum.units = r"erg/s"
bh_bollum.limits = [41.0, 46.0]
bh_bollum.log = True


@catalog_field
def bh_bollum_basic(sim, field):
    """Blackhole bolometric luminosity, basic model (for most massive BH in each subhalo)."""
    acField = "Subhalo_BH_BolLum_basic_largest"
    vals = sim.auxCat(acField)[acField]

    return vals


bh_bollum_basic.label = r"Blackhole $L_{\rm bol}$ [basic]"
bh_bollum_basic.units = r"erg/s"
bh_bollum_basic.limits = [41.0, 46.0]
bh_bollum_basic.log = True


@catalog_field
def bh_eddratio(sim, field):
    """Blackhole Eddington ratio (for most massive BH in each subhalo)."""
    acField = "Subhalo_BH_EddRatio_largest"
    vals = sim.auxCat(acField)[acField]

    return vals


bh_eddratio.label = r"Blackhole $\lambda_{\rm edd}$"
bh_eddratio.units = r""  # dimensionless
bh_eddratio.limits = [-4.0, 0.0]
bh_eddratio.log = True


@catalog_field
def bh_dedt(sim, field):
    """Blackhole energy injection rate (for most massive BH in each subhalo)."""
    acField = "Subhalo_BH_dEdt_largest"
    vals = sim.auxCat(acField)[acField]

    return vals


bh_dedt.label = r"Blackhole $\dot{E}_{\rm BH}$"
bh_dedt.units = r"erg/s"
bh_dedt.limits = [42.0, 45.0]
bh_dedt.log = True


@catalog_field
def bh_mode(sim, field):
    """Blackhole feedback mode [0=low/kinetic, 1=high/quasar] (for most massive BH in each subhalo)."""
    acField = "Subhalo_BH_mode"
    vals = sim.auxCat(acField)[acField]

    return vals


bh_mode.label = r"Blackhole Mode [ 0=low/kinetic, 1=high/quasar ]"
bh_mode.units = r""  # dimensionless
bh_mode.limits = [-0.1, 1.1]
bh_mode.log = False


@catalog_field
def bh_cumegy_low(sim, field):
    """Cumulative energy injected in the low (kinetic) accretion mode."""
    acField = "Subhalo_BH_CumEgyInjection_Low"
    vals = sim.auxCat(acField)[acField]
    vals = sim.units.codeEnergyToErg(vals)

    return vals


bh_cumegy_low.label = r"BH $\int$ E$_{\rm injected,low}$"
bh_cumegy_low.units = r"erg"
bh_cumegy_low.limits = [54, 61]
bh_cumegy_low.log = True


@catalog_field
def bh_cumegy_high(sim, field):
    """Cumulative energy injected in the high (thermal/quasar) accretion mode."""
    acField = "Subhalo_BH_CumEgyInjection_High"
    vals = sim.auxCat(acField)[acField]
    vals = sim.units.codeEnergyToErg(vals)

    return vals


bh_cumegy_high.label = r"BH $\int$ E$_{\rm injected,high}$"
bh_cumegy_high.units = r"erg"
bh_cumegy_high.limits = [58, 62]
bh_cumegy_high.log = True


@catalog_field
def bh_cummass_low(sim, field):
    """Cumulative mass accretion in the low (kinetic) accretion mode."""
    acField = "Subhalo_BH_CumMassGrowth_Low"
    vals = sim.auxCat(acField)[acField]
    vals = sim.units.codeMassToMsun(vals)

    return vals


bh_cummass_low.label = r"$\int$ M$_{\rm growth,low}$"
bh_cummass_low.units = r"$M_\odot$"
bh_cummass_low.limits = [0.0, 7.0]
bh_cummass_low.log = True


@catalog_field
def bh_cummass_high(sim, field):
    """Cumulative mass accretion in the high (thermal/quasar) accretion mode."""
    acField = "Subhalo_BH_CumMassGrowth_High"
    vals = sim.auxCat(acField)[acField]
    vals = sim.units.codeMassToMsun(vals)

    return vals


bh_cummass_high.label = r"$\int$ M$_{\rm growth,high}$"
bh_cummass_high.units = r"$M_\odot$"
bh_cummass_high.limits = [5.0, 9.0]
bh_cummass_high.log = True


@catalog_field
def bh_cumegy_ratio(sim, field):
    """Ratio of cumulative energy injected in low vs. high accretion modes."""
    acFields = ["Subhalo_BH_CumEgyInjection_High", "Subhalo_BH_CumEgyInjection_Low"]
    ac = sim.auxCat(acFields)

    # fix ac[fields[1]]=0 values such that vals is zero, which is then specially colored
    w = np.where(ac[acFields[1]] == 0.0)[0]
    if len(w):
        ac[acFields[1]][w] = 1.0
        ac[acFields[0]][w] = 0.0

    vals = ac[acFields[0]] / ac[acFields[1]]

    return vals


bh_cumegy_ratio.label = r"BH $\int$ E$_{\rm injected,high}$ / $\int$ E$_{\rm injected,low}$"
bh_cumegy_ratio.units = r""  # dimensionless
bh_cumegy_ratio.limits = [0.0, 4.0]
bh_cumegy_ratio.log = True


@catalog_field
def bh_cumegy_ratioinv(sim, field):
    """Ratio of cumulative energy injected in high vs. low accretion modes."""
    acFields = ["Subhalo_BH_CumMassGrowth_Low", "Subhalo_BH_CumMassGrowth_High"]
    ac = sim.auxCat(acFields)

    # fix ac[fields[1]]=0 values such that vals is zero, which is then specially colored
    w = np.where(ac[acFields[1]] == 0.0)[0]
    if len(w):
        ac[acFields[1]][w] = 1.0
        ac[acFields[0]][w] = 0.0

    vals = ac[acFields[0]] / ac[acFields[1]]

    return vals


bh_cumegy_ratioinv.label = r"BH $\int$ E$_{\rm injected,low}$ / $\int$ E$_{\rm injected,high}$"
bh_cumegy_ratioinv.units = r""  # dimensionless
bh_cumegy_ratioinv.limits = [-4.0, 0.0]
bh_cumegy_ratioinv.log = True


@catalog_field
def bh_cummass_ratio(sim, field):
    """Ratio of cumulative mass accretion in low vs. high accretion modes."""
    acFields = ["Subhalo_BH_CumMassGrowth_High", "Subhalo_BH_CumMassGrowth_Low"]
    ac = sim.auxCat(acFields)

    # fix ac[fields[1]]=0 values such that vals is zero, which is then specially colored
    w = np.where(ac[acFields[1]] == 0.0)[0]
    if len(w):
        ac[acFields[1]][w] = 1.0
        ac[acFields[0]][w] = 0.0

    vals = ac[acFields[0]] / ac[acFields[1]]

    return vals


bh_cummass_ratio.label = r"BH $\int$ M$_{\rm growth,high}$ / $\int$ M$_{\rm growth,low}$"
bh_cummass_ratio.units = r""  # dimensionless
bh_cummass_ratio.limits = [1.0, 5.0]
bh_cummass_ratio.log = True

# ---------------------------- auxcat: outflows ------------------------------------------------------


@catalog_field
def wind_vel(sim, field):
    """Wind model: velocity at injection, from star-forming gas."""
    acField = "Subhalo_Gas_Wind_vel"
    vals = sim.auxCat(acField)[acField]

    return vals


wind_vel.label = r"Wind Injection Velocity"
wind_vel.units = r"km/s"
wind_vel.limits = [1.0, 3.0]
wind_vel.log = True


@catalog_field
def wind_etam(sim, field):
    """Wind model: mass loading at injection, from star-forming gas."""
    acField = "Subhalo_Gas_Wind_etaM"
    vals = sim.auxCat(acField)[acField]

    return vals


wind_etam.label = r"Wind Mass Loading $\eta_{\rm M}$"
wind_etam.units = r""  # dimensionless
wind_etam.limits = [-1.0, 2.0]
wind_etam.log = True


@catalog_field
def wind_dedt(sim, field):
    """Wind model: energy injection rate, from star-forming gas."""
    acField = "Subhalo_Gas_Wind_dEdt"
    vals = sim.auxCat(acField)[acField]
    vals = vals.astype("float64") * 1e51  # unit conversion: remove 10^51 factor

    return vals


wind_dedt.label = r"Wind Energy Injection Rate $\dot{E}_{\rm SN}$"
wind_dedt.units = r"erg/s"
wind_dedt.limits = [39.0, 42.0]
wind_dedt.log = True


@catalog_field
def wind_dpdt(sim, field):
    """Wind model: momemtum injection rate, from star-forming gas."""
    acField = "Subhalo_Gas_Wind_dPdt"
    vals = sim.auxCat(acField)[acField]
    vals = vals.astype("float64") * 1e51  # unit conversion: remove 10^51 factor

    return vals


wind_dpdt.label = r"Wind Momentum Injection Rate $\dot{P}_{\rm SN}$"
wind_dpdt.units = r"erg/s"
wind_dpdt.limits = [39.0, 42.0]
wind_dpdt.log = True


@catalog_field(multi="etam_")
def etam_(sim, field):
    """Outflows: mass loading factor (given a selected timescale, radius, and velocity cut)."""
    _, sfr_timescale, rad, vcut = field.split("_")

    fieldName = "Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-%s" % sfr_timescale
    ac = sim.auxCat(fields=[fieldName], expandPartial=True)

    # figure out which (radius,vcut) selection
    radBins = ac[fieldName + "_attrs"]["rad"]
    vcutVals = ac[fieldName + "_attrs"]["vcut_vals"]
    radBinsMid = (radBins[:-1] + radBins[1:]) / 2

    vcutInd = list(vcutVals).index(float(vcut.replace("kms", "")))

    if rad == "all":
        # last bin accumulates across all radii
        radInd = len(radBins) - 1
    else:
        radInd = list(radBinsMid).index(float(rad.replace("kpc", "")))

    vals = ac[fieldName][:, radInd, vcutInd]
    return vals


etam_.label = r"Mass Loading $\eta_{\rm M}$"
etam_.units = r""  # dimensionless
etam_.limits = [0.0, 2.0]
etam_.log = True


@catalog_field(multi="etae_")
def etae_(sim, field):
    """Outflows: energy loading factor (given a selected radius, and velocity cut)."""
    _, rad, vcut = field.split("_")

    fieldName = "Subhalo_EnergyLoadingSN_SubfindWithFuzz"
    ac = sim.auxCat(fields=[fieldName], expandPartial=True)

    # figure out which (radius,vcut) selection
    radBins = ac[fieldName + "_attrs"]["rad"]
    vcutVals = ac[fieldName + "_attrs"]["vcut_vals"]
    radBinsMid = (radBins[:-1] + radBins[1:]) / 2

    radInd = list(radBinsMid).index(float(rad.replace("kpc", "")))
    vcutInd = list(vcutVals).index(float(vcut.replace("kms", "")))

    vals = ac[fieldName][:, radInd, vcutInd]
    return vals


etae_.label = r"Energy Loading $\eta_{\rm E}$"
etae_.units = r""  # dimensionless
etae_.limits = [-1.5, 2.5]
etae_.log = True


@catalog_field(multi="etap_")
def etap_(sim, field):
    """Outflows: momentum loading factor (given a selected radius, and velocity cut)."""
    _, rad, vcut = field.split("_")

    fieldName = "Subhalo_MomentumLoadingSN_SubfindWithFuzz"
    ac = sim.auxCat(fields=[fieldName], expandPartial=True)

    # figure out which (radius,vcut) selection
    radBins = ac[fieldName + "_attrs"]["rad"]
    vcutVals = ac[fieldName + "_attrs"]["vcut_vals"]
    radBinsMid = (radBins[:-1] + radBins[1:]) / 2

    radInd = list(radBinsMid).index(float(rad.replace("kpc", "")))
    vcutInd = list(vcutVals).index(float(vcut.replace("kms", "")))

    vals = ac[fieldName][:, radInd, vcutInd]
    return vals


etap_.label = r"Momentum Loading $\eta_{\rm P}$"
etap_.units = r""  # dimensionless
etap_.limits = [-1.5, 2.5]
etap_.log = True


@catalog_field(multi="vout_")
def vout_(sim, field):
    """Outflows: outflow velocity (given a selected percentile and radius)."""
    _, perc, rad = field.split("_")

    fieldName = "Subhalo_OutflowVelocity_SubfindWithFuzz"
    ac = sim.auxCat(fields=[fieldName], expandPartial=True)

    # figure out which (radius,perc) selection
    radBins = ac[fieldName + "_attrs"]["rad"]
    percs = ac[fieldName + "_attrs"]["percs"]

    if rad == "all":
        # last bin accumulates across all radii
        radInd = len(radBins) - 1
    else:
        # all other bins addressed by their midpoint (e.g. '10kpc')
        radBinsMid = (radBins[:-1] + radBins[1:]) / 2
        radInd = list(radBinsMid).index(float(rad.replace("kpc", "")))

    percInd = list(percs).index(int(perc))

    vals = ac[fieldName][:, radInd, percInd]
    return vals


vout_.label = r"Outflow Velocity $v_{\rm out}$"
vout_.units = r"$\rm{km/s}$"
vout_.limits = [1.5, 3.5]
vout_.log = True

# ---------------------------- auxcat: other ------------------------------------------------------


@catalog_field(alias="z_form")
def zform(sim, field):
    """Formation redshift (of the halo), at which the subhalo had half of its current mass."""
    acField = "Subhalo_SubLink_zForm_mm5"
    ac = sim.auxCat(fields=[acField])

    return ac[acField]


zform.label = r"$\rm{z_{form}}$"
zform.units = ""  # linear dimensionless
zform.limits = [0.0, 4.0]
zform.log = False


@catalog_field
def stellar_zform_vimos(sim, field):
    """Stellar formation redshift (mass-weighted mean), using the VIMOS slit aperture of stars."""
    acField = "Subhalo_StellarZform_VIMOS_Slit"
    ac = sim.auxCat(fields=[acField])

    return ac[acField]


stellar_zform_vimos.label = r"$\rm{z_{form,\star}}$"
stellar_zform_vimos.units = ""  # linear dimensionless
stellar_zform_vimos.limits = [0.5, 6.0]
stellar_zform_vimos.log = False


@catalog_field
def stellarage(sim, field):
    """Mean stellar age (mass-weighted), all stars in subhalo."""
    acField = "Subhalo_StellarAge_NoRadCut_MassWt"
    vals = sim.auxCat(fields=[acField])[acField]

    return vals


stellarage.label = r"$\rm{log t_{age,\star}$"
stellarage.units = r"$\rm{Gyr}$"
stellarage.limits = [0.0, 1.0]
stellarage.log = True


@catalog_field
def stellarage_4pkpc(sim, field):
    """Mean stellar age (r-band luminosity weighted), including only stars within a 4 pkpc aperture."""
    acField = "Subhalo_StellarAge_4pkpc_rBandLumWt"
    vals = sim.auxCat(fields=[acField])[acField]

    return vals


stellarage_4pkpc.label = r"$\rm{log t_{age,\star}$"
stellarage_4pkpc.units = r"$\rm{Gyr}$"
stellarage_4pkpc.limits = [0.0, 1.0]
stellarage_4pkpc.log = True


@catalog_field(
    aliases=["fiber_zred", "fiber_mass", "fiber_logzsol", "fiber_tau", "fiber_tage", "fiber_dust1", "fiber_dust2"]
)
def fiber_(sim, field):
    """Mock SDSS fiber spectrum MCMC fit quantities."""
    # withVel=True, addRealism=True, dustModel=p07c_cf00dust_res_conv, directions=z
    import json

    from ..util.match import match

    acField = "Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z"
    ac = sim.auxCat(fields=[acField])

    acInds = {
        "fiber_zred": 0,
        "fiber_mass": 1,
        "fiber_logzsol": 2,
        "fiber_tau": 3,
        "fiber_tage": 4,
        "fiber_dust1": 5,
        "fiber_dust2": 6,
    }
    acInd = acInds[field]

    # verify index
    field_names = json.loads(ac[acField + "_attrs"]["theta_labels"])
    assert field_names[acInd] == field.split("fiber_")[1]

    # non-dense in subhaloIDs, crossmatch and leave missing at nan
    subhaloIDs_snap = np.arange(sim.numSubhalos)

    gc_inds, _ = match(subhaloIDs_snap, ac["subhaloIDs"])
    assert gc_inds.size == ac["subhaloIDs"].size

    vals = np.zeros(len(subhaloIDs_snap), dtype="float32")
    vals.fill(np.nan)

    vals[gc_inds] = np.squeeze(ac[acField][:, acInd, 1])  # last index 1 = median

    return vals


fiber_.label = r""  # variable (todo)
fiber_.units = r""  # variable (todo)
fiber_.limits = [0.0, 1.0]  # variable (todo)
fiber_.log = True  # variable (todo)


@catalog_field(alias="temp_halo_volwt")
def temp_halo(sim, field):
    """Mean gas temperature in the halo (0.15 < r/rvir < 1), mass or volume weighted."""
    wtStr = "massWt" if "_volwt" not in field else "volWt"
    acField = "Subhalo_Temp_halo_%s" % wtStr
    vals = sim.auxCat(fields=[acField])[acField]

    return vals


temp_halo.label = r"$\rm{log T_{halo}}$"
temp_halo.units = r"$\rm{K}$"
temp_halo.limits = [4.0, 8.0]
temp_halo.log = True


@catalog_field(alias="nh_halo_volwt")
def nh_halo(sim, field):
    """Mean hydrogen number density in the halo (0.15 < r/rvir < 1), mass or volume weighted."""
    wtStr = "massWt" if "_volwt" not in field else "volWt"
    acField = "Subhalo_nH_halo_%s" % wtStr
    vals = sim.auxCat(fields=[acField])[acField]

    return vals


nh_halo.label = r"$\rm{log n_{H,halo}}$"
nh_halo.units = r"$\rm{cm^{-3}}$"
nh_halo.limits = [-5.0, -1.0]
nh_halo.log = True


@catalog_field(alias="nh_2rhalf_volwt")
def nh_2rhalf(sim, field):
    """Mean hydrogen number density in the galaxy (< 2rhalfstars), mass or volume weighted."""
    wtStr = "massWt" if "_volwt" not in field else "volWt"
    acField = "Subhalo_nH_ISM_%s" % wtStr
    vals = sim.auxCat(fields=[acField])[acField]

    return vals


nh_2rhalf.label = r"$\rm{log n_{H,ISM}}$"
nh_2rhalf.units = r"$\rm{cm^{-3}}$"
nh_2rhalf.limits = [-2.5, 1.0]
nh_2rhalf.log = True
