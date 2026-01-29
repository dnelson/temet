"""
Definitions of custom snapshot fields.
"""

from functools import partial
from getpass import getuser
from os.path import getsize, isfile

import h5py
import numpy as np

from ..util.helper import closest, numPartToChunkLoadSize, pSplitRange
from .snapshot import _haloOrSubhaloIndRange, snap_field


# -------------------- general/all particle types -------------------------------------------------


@snap_field
def mass_msun(sim, partType, field, args):
    """Particle/cell mass in solar masses."""
    mass = sim.snapshotSubset(partType, "mass", **args)

    return sim.units.codeMassToMsun(mass)


mass_msun.label = "[pt] Mass"
mass_msun.units = r"$\rm{M_{sun}}$"
mass_msun.limits = [5.0, 7.0]
mass_msun.limits_halo = [3.0, 5.0]
mass_msun.log = True


@snap_field(alias="masses")
def mass(sim, partType, field, args):
    """Particle/cell mass in code units, supporting constant-mass DM particles."""
    if sim.isPartType(partType, "dm"):
        mass = sim.snapshotSubset(partType, "pos_x", **args)  # make sure size is correct
        mass[:] = sim.dmParticleMass

        return mass

    # otherwise, normal load
    return None


mass.label = "[pt] Mass"
mass.units = "code_mass"
mass.limits = [-3.0, 0.0]
mass.limits_halo = [-6.0, -2.0]
mass.log = True


@snap_field
def highres_massfrac(sim, partType, field, args):
    """High-resolution mass normalized by total mass (applicable only to zooms)."""
    assert sim.isPartType(partType, "gas")
    mass = sim.snapshotSubset(partType, "mass", **args)
    mass_highres = sim.snapshotSubset(partType, "HighResGasMass", **args)

    return mass_highres / mass


mass.label = "High-Res Mass Fraction"
mass.units = ""  # dimensionless
mass.limits = [-3.0, 0.0]
mass.limits_halo = [-6.0, -2.0]
mass.log = True


@snap_field(alias="vmag")
def velmag(sim, partType, field, args):
    """Magnitude of the gas velocity 3-vector."""
    vel = sim.snapshotSubset(partType, "vel", **args)
    vel = sim.units.particleCodeVelocityToKms(vel)

    vmag = np.sqrt(vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1] + vel[:, 2] * vel[:, 2])

    return vmag


velmag.label = "Velocity Magnitude"
velmag.units = r"$\rm{km/s}$"
velmag.limits = [0, 1000]
velmag.limits_halo = [0, 400]
velmag.log = False


@snap_field(alias="subfind_volume")
def subfind_vol(sim, partType, field, args):
    """Particle 'volume' based on SubfindHsml of the N=64 nearest DM particles."""
    hsml = sim.snapshotSubset(partType, "SubfindHsml", **args)
    vol = (4.0 / 3.0) * np.pi * hsml**3.0

    return vol


subfind_vol.label = "Subfind Volume"
subfind_vol.units = "code_volume"
subfind_vol.limits = [-6.0, 6.0]
subfind_vol.limits_halo = [-4.0, 6.0]
subfind_vol.log = True


@snap_field(alias="gravpotential")
def gravpot(sim, partType, field, args):
    """Gravitational potential (from stored value in snapshot)."""
    pot = sim.snapshotSubset(partType, "Potential", **args)

    return pot * sim.units.scalefac


gravpot.label = "Gravitational Potential"
gravpot.units = r"$\rm{(km/s)^2}$"
gravpot.limits = [-1e6, 1e5]
gravpot.limits_halo = [-1e6, -1e2]
gravpot.log = False


@snap_field(alias="escapevel")
def vesc(sim, partType, field, args):
    """Escape velocity, based on the stored gravitational Potential."""
    pot = sim.snapshotSubset(partType, "Potential", **args)

    return sim.units.codePotentialToEscapeVelKms(pot)


vesc.label = "Escape Velocity"
vesc.units = r"$\rm{km/s}$"
vesc.limits = [0, 2000]
vesc.limits_halo = [200, 1000]
vesc.log = False


@snap_field(alias="dt_yr")
def dt(sim, partType, field, args):
    """Particle/cell total actual/effective timestep, from stored snapshot value."""
    dt = sim.snapshotSubset(partType, "TimeStep", **args)

    return sim.units.codeTimeStepToYears(dt)


dt.label = "[pt] Timestep"
dt.units = r"$\rm{yr}$"
dt.limits = [1.0, 6.0]
dt.limits_halo = [1.0, 5.0]
dt.log = True

# -------------------- gas ------------------------------------------------------------------------


@snap_field(alias="temperature")
def temp(sim, partType, field, args):
    """Gas temperature."""
    if sim.snapHasField(partType, "GrackleTemperature"):
        return sim.snapshotSubset(partType, "GrackleTemperature", **args)

    u = sim.snapshotSubset(partType, "u", **args)
    xe = sim.snapshotSubset(partType, "xe", **args)

    return sim.units.UToTemp(u, xe, log=False)


temp.label = "Gas Temperature"
temp.units = r"$\rm{K}$"
temp.limits = [2.0, 8.0]
temp.limits_halo = [3.5, 8.0]
temp.log = True


@snap_field
def temp_old(sim, partType, field, args):
    """Gas temperature (uncorrected values for TNG runs)."""
    u = sim.snapshotSubset(partType, "InternalEnergyOld", **args)
    xe = sim.snapshotSubset(partType, "xe", **args)

    return sim.units.UToTemp(u, xe, log=False)


temp_old.label = "Gas Temperature (Uncorrected)"
temp_old.units = r"$\rm{K}$"
temp_old.limits = [2.0, 8.0]
temp_old.limits_halo = [3.5, 8.0]
temp_old.log = True


@snap_field
def temp_sfcold(sim, partType, field, args):
    """Gas temperature, where star-forming gas is set to the sub-grid (constant)
    cold-phase temperature, instead of eEOS 'effective' temperature."""
    assert sim.eEOS in [1, 2]

    temp = sim.snapshotSubset(partType, "temp", **args)
    sfr = sim.snapshotSubset(partType, "sfr", **args)

    w = np.where(sfr > 0.0)
    if sim.eEOS == 1:
        temp[w] = sim.units.sh03_T_c
    elif sim.eEOS == 2:
        temp[w] = 1e4  # K (see Rahmati+16, Wijers+19, i.e. accepted EAGLE eEOS convention)

    return temp


temp_sfcold.label = "Gas Temperature"
temp_sfcold.units = r"$\rm{K}$"
temp_sfcold.limits = [3.5, 7.2]
temp_sfcold.limits_halo = [3.5, 8.0]
temp_sfcold.log = True


@snap_field
def temp_sfhot(sim, partType, field, args):
    """Gas temperature, where star-forming gas is set to the sub-grid (constant)
    hot-phase temperature, instead of the eEOS 'effective' temperature. Use with caution."""
    assert sim.eEOS == 1
    temp = sim.snapshotSubset(partType, "temp", **args)
    nh = sim.snapshotSubset(partType, "nh", **args)
    sfr = sim.snapshotSubset(partType, "sfr", **args)

    _, T_h = sim.units.densToSH03TwoPhase(nh, sfr)

    w = np.where(sfr > 0.0)
    temp[w] = T_h[w]

    return temp


temp_sfhot.label = "Gas Temperature"
temp_sfhot.units = r"$\rm{K}$"
temp_sfhot.limits = [3.5, 7.2]
temp_sfhot.limits_halo = [3.5, 8.0]
temp_sfhot.log = True


@snap_field
def twophase_coldfrac(sim, partType, field, args):
    """Cold-phase mass (or density) fraction, for the SH03 two-phase ISM model.
    Note: is exactly 0.0 for non-starforming (SFR==0) gas cells, and typically of order ~0.9 for SFR>0 cells."""
    assert sim.eEOS == 1

    nh = sim.snapshotSubset(partType, "nh", **args)
    sfr = sim.snapshotSubset(partType, "sfr", **args)

    coldfrac, _ = sim.units.densToSH03TwoPhase(nh, sfr)

    return coldfrac


twophase_coldfrac.label = r"SH03 Cold-phase Mass Fraction"
twophase_coldfrac.units = ""  # dimensionless
twophase_coldfrac.limits = [0, 1]
twophase_coldfrac.limits_halo = [0, 1]
twophase_coldfrac.log = False


@snap_field(alias="nelec")
def ne(sim, partType, field, args):
    """Electron number density, derived from (fractional) ElectronAbundance, handling runs without cooling."""
    assert sim.isPartType(partType, "gas")

    if sim.snapHasField(partType, "ElectronAbundance"):
        # normal run
        xe = sim.snapshotSubset(partType, "ElectronAbundance", **args)
    else:
        # no cooling run: assume fully ionized primordial composition
        xe = sim.snapshotSubset(partType, "u", **args)  # make sure size is correct
        xe[:] = 1.0 / (1 + 2 * sim.units.helium_massfrac)

    nelec = xe * sim.snapshotSubset(partType, "nh", **args)

    return nelec


ne.label = r"Electron Number Density $\rm{n_e}$"
ne.units = r"$\rm{cm^{-3}}$"
ne.limits = [-9.0, -3.0]
ne.limits_halo = [-6.0, 0.0]
ne.log = True


@snap_field
def ne_twophase(sim, partType, field, args):
    """Electron number density, where for star-forming gas cells we override the naive snapshot value,
    which is unphysically high, with a value based on the SH03 hot-phase mass only."""
    assert sim.eEOS == 1

    ne = sim.snapshotSubset(partType, "ne", **args)

    # compute hot-phase fraction (is 1.0 for SFR==0 cells, and of order ~0.1 for SFR>0 cells)
    hot_frac = 1.0 - sim.snapshotSubset(partType, "twophase_coldfrac", **args)

    return ne * hot_frac


ne_twophase.label = r"Hot-phase Electron Number Density $\rm{n_e}$"
ne_twophase.units = r"$\rm{cm^{-3}}$"
ne_twophase.limits = [-9.0, -3.0]
ne_twophase.limits_halo = [-6.0, 0.0]
ne_twophase.log = True


@snap_field(alias="hdens")
def nh(sim, partType, field, args):
    """Hydrogen number density, derived from total Density."""
    dens = sim.snapshotSubset(partType, "dens", **args)
    dens = sim.units.codeDensToPhys(dens, cgs=True, numDens=True)  # 1/cm^3

    nh = dens * sim.units.hydrogen_massfrac  # constant 0.76 assumed

    return nh


nh.label = r"Gas Hydrogen Density $\rm{n_H}$"
nh.units = r"$\rm{cm^{-3}}$"
nh.limits = [-9.0, 3.0]
nh.limits_halo = [-5.0, 0.0]
nh.log = True


@snap_field(alias="massfrac_hi")
def xhi(sim, partType, field, args):
    """Neutral hydrogen fraction, relative to the total gas mass. No H2 model applied."""
    if sim.snapHasField(partType, "NeutralHydrogenAbundance"):
        nh0_frac = sim.snapshotSubset(partType, "NeutralHydrogenAbundance", **args)
        return sim.units.hydrogen_massfrac * nh0_frac

    if sim.snapHasField(partType, "HIMassFraction"):  # grackle
        return sim.snapshotSubset(partType, "HIMassFraction", **args)


xhi.label = r"Neutral Hydrogen Mass Fraction $\rm{x_{HI}}$"
xhi.units = ""  # dimensionless
xhi.limits = [-9.0, 0.0]
xhi.limits_halo = [-5.0, 0.0]
xhi.log = True


@snap_field
def numdens(sim, partType, field, args):
    """Total gas number density, derived from total Density."""
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.codeDensToPhys(dens, cgs=True, numDens=True)


numdens.label = "Gas Number Density"
numdens.units = r"$\rm{cm^{-3}}$"
numdens.limits = [-9.0, 3.0]
numdens.limits_halo = [-5.0, 0.0]
numdens.log = True


@snap_field
def dens_critratio(sim, partType, field, args):
    """Mass density to critical density."""
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.codeDensToCritRatio(dens, baryon=False, log=False)


dens_critratio.label = r"$\rm{\rho_{gas} / \rho_{crit}}$"
dens_critratio.units = ""  # dimensionless
dens_critratio.limits = [-6.0, 5.0]
dens_critratio.limits_halo = [-1.0, 6.0]
dens_critratio.log = True


@snap_field
def dens_critb(sim, partType, field, args):
    """Mass density to critical baryon density."""
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.codeDensToCritRatio(dens, baryon=True, log=False)


dens_critb.label = r"$\rm{\rho_{gas} / \rho_{crit,b}}$"
dens_critb.units = ""  # dimensionless
dens_critb.limits = [-2.0, 9.0]
dens_critb.limits_halo = [3.0, 9.0]
dens_critb.log = True


@snap_field(aliases=["ent", "entr"])
def entropy(sim, partType, field, args):
    """Gas entropy, derived from (u,dens)."""
    u = sim.snapshotSubset(partType, "u", **args)
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.calcEntropyCGS(u, dens, log=False)


entropy.label = "Gas Entropy"
entropy.units = r"$\rm{K\ cm^2}$"
entropy.limits = [8.0, 11.0]
entropy.limits_halo = [9.0, 11.0]
entropy.log = True


@snap_field(alias="bfieldmag")
def bmag(sim, partType, field, args):
    """Magnitude of the gas magnetic field 3-vector in Gauss."""
    b = sim.snapshotSubset(partType, "MagneticField", **args)
    b = sim.units.particleCodeBFieldToGauss(b)

    bmag = np.sqrt(b[:, 0] * b[:, 0] + b[:, 1] * b[:, 1] + b[:, 2] * b[:, 2])

    return bmag


bmag.label = "Magnetic Field Strength"
bmag.units = r"$\rm{Gauss}$"
bmag.limits = [-15.0, -3.0]
bmag.limits_halo = [-9.0, -2.0]
bmag.log = True


@snap_field(alias="bfieldmag_ug")
def bmag_ug(sim, partType, field, args):
    """Magnitude of the gas magnetic field 3-vector in micro-Gauss."""
    return sim.snapshotSubset(partType, "bmag", **args) * 1e6


bmag_ug.label = "Magnetic Field Strength"
bmag_ug.units = r"$\rm{\mu G}$"
bmag_ug.limits = [-9.0, 3.0]
bmag_ug.limits_halo = [-3.0, 2.0]
bmag_ug.log = True


@snap_field
def b2(sim, partType, field, args):
    """Magnitude squared of the gas magnetic field 3-vector (in code units)."""
    b = sim.snapshotSubset(partType, "MagneticField", **args)

    b2 = b[:, 0] * b[:, 0] + b[:, 1] * b[:, 1] + b[:, 2] * b[:, 2]

    return b2


b2.label = "Magnetic Field Squared"
b2.units = "code units"
b2.limits = [-3.0, 9.0]
b2.limits_halo = [-3.0, 5.0]
b2.log = True


@snap_field(alias="vel_alfven")
def va(sim, partType, field, args):
    """Magnetic Alfven-wave velocity."""
    bmag = sim.snapshotSubset(partType, "bmag", **args)  # G

    # note: density should be the sum of all charged particle species, not just electrons
    rho = sim.snapshotSubset(partType, "ne", **args) * sim.units.mass_electron  # g/cm^3

    va = bmag / np.sqrt(4 * np.pi * rho) * 1e-5  # cgs -> km/s

    return va


va.label = "Alfven Velocity"
va.units = r"$\rm{km/s}$"
va.limits = [-9.0, 3.0]
va.limits_halo = [-3.0, 2.0]
va.log = True


@snap_field(aliases=["vel_sound", "csound", "csnd"])
def cs(sim, partType, field, args):
    """Gas sound speed (hydro only version)."""
    u = sim.snapshotSubset(partType, "u", **args)
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.calcSoundSpeedKmS(u, dens)


cs.label = "Sound Speed"
cs.units = r"$\rm{km/s}$"
cs.limits = [-6.0, 3.0]
cs.limits_halo = [-1.0, 2.0]
cs.log = True


@snap_field(aliases=["jeans_mass"])
def mjeans(sim, partType, field, args):
    """Local (per-cell) Jeans mass."""
    cs = sim.snapshotSubset(partType, "cs", **args).astype("float64")  # km/s
    cs *= 1e5  # km/s -> cm/s
    dens = sim.snapshotSubset(partType, "dens", **args)
    dens = sim.units.codeDensToPhys(dens, cgs=True)  # g/cm^3

    mJ = np.pi ** (5 / 2) * cs**3 / (6 * sim.units.Gravity ** (3 / 2) * dens ** (1 / 2))  # g

    mJ /= sim.units.Msun_in_g  # Msun

    return mJ.astype("float32")


mjeans.label = "Jeans Mass"
mjeans.units = r"$\rm{M_{sun}}$"
mjeans.limits = [0.0, 6.0]
mjeans.limits_halo = [2.0, 6.0]
mjeans.log = True


@snap_field(alias="tff_cell")
def tff_local(sim, partType, field, args):
    """Local (per-cell) free-fall time."""
    dens = sim.snapshotSubset(partType, "dens", **args)
    dens = sim.units.codeDensToPhys(dens, cgs=True)  # g/cm^3

    tff = np.sqrt(3 * np.pi / (32 * sim.units.Gravity * dens))  # s
    tff /= sim.units.s_in_yr  # yr

    return tff


tff_local.label = "Free-fall Time"
tff_local.units = r"$\rm{yr}$"
tff_local.limits = [0.0, 6.0]
tff_local.limits_halo = [2.0, 6.0]
tff_local.log = True


@snap_field(alias="vol")
def volume(sim, partType, field, args):
    """Gas cell volume."""
    if not sim.snapHasField(partType, "Volume"):
        # PartType0/Volume eliminated in newer outputs, calculate if necessary
        mass = sim.snapshotSubset(partType, "mass", **args)
        dens = sim.snapshotSubset(partType, "dens", **args)
        vol = mass / dens
        return vol

    # otherwise, normal load
    return None


volume.label = "Gas Cell Volume"
volume.units = "(code_length)$^3$"
volume.limits = [-6.0, 6.0]
volume.log = True


@snap_field(alias="vol_cm3")
def volume_cm3(sim, partType, field, args):
    """Gas cell volume [cm^3]."""
    return sim.units.codeVolumeToCm3(sim.snapshotSubset(partType, "volume", **args))


volume_cm3.label = "Gas Cell Volume"
volume_cm3.units = r"$\rm{cm^3}$"
volume_cm3.limits = [55.0, 65.0]
volume_cm3.limits_halo = [55.0, 62.0]
volume_cm3.log = True


@snap_field(alias="vol_kpc3")
def volume_kpc3(sim, partType, field, args):
    """Gas cell volume [kpc^3]."""
    return sim.units.codeVolumeToKpc3(sim.snapshotSubset(partType, "volume", **args))


volume_kpc3.label = "Gas Cell Volume"
volume_kpc3.units = r"$\rm{kpc^3}$"
volume_kpc3.limits = [-6.0, 6.0]
volume_kpc3.limits_halo = [-6.0, 2.0]
volume_kpc3.log = True


@snap_field(alias="cellrad")
def cellsize(sim, partType, field, args):
    """Gas cell 'size' i.e. 'cell radius', defined as the radius of the volume equivalent sphere."""
    vol = sim.snapshotSubset(partType, "volume", **args)
    rcell = (vol * 3.0 / (4 * np.pi)) ** (1.0 / 3.0)

    return rcell


cellsize.label = "Gas Cell Size"
cellsize.units = "code_length"
cellsize.limits = [-2.0, 3.0]
cellsize.limits_halo = [-2.0, 1.0]
cellsize.log = True


@snap_field(alias="cellrad_kpc")
def cellsize_kpc(sim, partType, field, args):
    """Gas cell size [kpc]."""
    rcell = sim.snapshotSubset(partType, "cellsize", **args)

    return sim.units.codeLengthToKpc(rcell)


cellsize_kpc.label = "Gas Cell Size"
cellsize_kpc.units = r"$\rm{kpc}$"
cellsize_kpc.limits = [-2.0, 3.0]
cellsize_kpc.limits_halo = [-2.0, 1.0]
cellsize_kpc.log = True


@snap_field(alias="cellrad_ckpc")
def cellsize_ckpc(sim, partType, field, args):
    """Gas cell size [comoving kpc]."""
    rcell = sim.snapshotSubset(partType, "cellsize", **args)

    return sim.units.codeLengthToComovingKpc(rcell)


cellsize_ckpc.label = "Gas Cell Size"
cellsize_ckpc.units = r"$\rm{ckpc}$"
cellsize_ckpc.limits = [-2.0, 3.0]
cellsize_ckpc.limits_halo = [-2.0, 1.0]
cellsize_ckpc.log = True


@snap_field
def hsml(sim, partType, field, args):
    """Smoothing length i.e. characteristic size, possibly for visualization purposes."""
    assert args["inds"] is None  # otherwise generalize
    from ..vis.render import defaultHsmlFac, getHsmlForPartType

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    useSnapHsml = sim.isPartType(partType, "stars")
    hsml = getHsmlForPartType(sim, partType, indRange=indRange, useSnapHsml=useSnapHsml)
    hsml *= defaultHsmlFac(partType)

    return hsml


hsml.label = "Smoothing Length"
hsml.units = "code_length"
hsml.limits = [-1.0, 4.0]
hsml.limits_halo = [-1.0, 2.0]
hsml.log = True


@snap_field(alias="baryon_frac")
def f_b(sim, partType, field, args):
    """Baryon fraction, defined as (gas+stars)/(gas+stars+DM), all estimated locally, then
    normalized to the cosmic baryon fraction."""
    assert sim.isPartType(partType, "gas")  # otherwise generalize
    from ..util.treeSearch import calcHsml, calcQuantReduction

    pt_pos = sim.snapshotSubset(partType, "pos", **args)

    # DM
    if sim.snapHasField(partType, "SubfindDMDensity"):
        dens_dm = sim.snapshotSubset(partType, "SubfindDMDensity", **args)
    else:
        # derive if not stored
        dm_pos = sim.snapshotSubset("dm", "pos")  # global load and tree
        SubfindHsml = calcHsml(dm_pos, sim.boxSize, posSearch=pt_pos, nNGB=64, treePrec=dm_pos.dtype)
        dens_dm = 64 * sim.dmParticleMass / (4 / 3 * np.pi * SubfindHsml**3)  # code mass / code volume

    # stars
    stars_pos = sim.snapshotSubset("stars", "pos")  # global load and tree
    stars_mass = sim.snapshotSubset("stars", "mass")
    StellarHsml = calcHsml(stars_pos, sim.boxSize, posSearch=pt_pos, nNGB=32, treePrec=stars_pos.dtype)
    totmass_stars = calcQuantReduction(
        stars_pos, stars_mass, StellarHsml, op="sum", boxSizeSim=sim.boxSize, posSearch=pt_pos, treePrec=stars_pos.dtype
    )
    dens_stars = totmass_stars / (4 / 3 * np.pi * StellarHsml**3)  # code mass / code volume

    # gas
    dens_gas = sim.snapshotSubset(partType, "Density", **args)

    # f_b
    dens_b = dens_gas + dens_stars
    dens_tot = dens_gas + dens_stars + dens_dm

    return dens_b / dens_tot / sim.units.f_b


f_b.label = r"$\rm{f_{b} / f_{b,cosmic}}$"
f_b.units = ""  # dimensionless
f_b.limits = [0.0, 2.0]
f_b.log = False


@snap_field(aliases=["gas_pres", "gas_pressure", "pres", "p_gas", "p_thermal"])
def pressure(sim, partType, field, args):
    """Gas *thermal* pressure."""
    u = sim.snapshotSubset(partType, "u", **args)
    dens = sim.snapshotSubset(partType, "dens", **args)

    return sim.units.calcPressureCGS(u, dens)


pressure.label = "Gas Pressure"
pressure.units = r"$\rm{K\ cm^{-3}}$"
pressure.limits = [-1.0, 7.0]
pressure.limits_halo = [0.0, 5.0]
pressure.log = True


@snap_field(aliases=["mag_pres", "magnetic_pressure", "p_magnetic", "p_b"])
def pressure_mag(sim, partType, field, args):
    """Gas *magnetic* pressure."""
    b = sim.snapshotSubset(partType, "MagneticField", **args)

    return sim.units.calcMagneticPressureCGS(b)


pressure_mag.label = "Gas Magnetic Pressure"
pressure_mag.units = r"$\rm{K\ cm^{-3}}$"
pressure_mag.limits = [-1.0, 7.0]
pressure_mag.limits_halo = [0.0, 5.0]
pressure_mag.log = True


@snap_field(aliases=["pres_ratio"])
def pressure_ratio(sim, partType, field, args):
    r"""Ratio of gas magnetic to thermal pressure (:math:`\beta^{-1}`)."""
    P_gas = sim.snapshotSubset(partType, "p_gas", **args)
    P_mag = sim.snapshotSubset(partType, "p_magnetic", **args)

    return P_mag / P_gas


pressure_ratio.label = r"$\rm{\beta^{-1} = P_{B} / P_{gas}}$"
pressure_ratio.units = ""  # dimensionless
pressure_ratio.limits = [-2.5, 2.5]
pressure_ratio.limits_halo = [-2.0, 2.0]
pressure_ratio.log = True


@snap_field
def beta(sim, partType, field, args):
    r"""Ratio of gas thermal to magnetic pressure (plasma :math:`\beta`)."""
    P_gas = sim.snapshotSubset(partType, "p_gas", **args)
    P_mag = sim.snapshotSubset(partType, "p_magnetic", **args)

    return P_gas / P_mag


beta.label = r"$\rm{\beta = P_{gas} / P_{B}}$"
beta.units = ""  # dimensionless
beta.limits = [-2.5, 2.5]
beta.limits_halo = [-2.0, 2.0]
beta.log = True


@snap_field(aliases=["p_tot", "pres_tot", "pres_total", "pressure_total"])
def pressure_tot(sim, partType, field, args):
    """Total (thermal+magnetic) gas pressure."""
    P_gas = sim.snapshotSubset(partType, "p_gas", **args)
    P_mag = sim.snapshotSubset(partType, "p_magnetic", **args)

    return P_gas + P_mag


pressure_tot.label = r"Gas Total Pressure = $\rm{P_{gas} + P_{B}}$"
pressure_tot.units = r"$\rm{K\ cm^{-3}}$"
pressure_tot.limits = [-6.0, 8.0]
pressure_tot.limits_halo = [-2.0, 7.0]
pressure_tot.log = True


@snap_field(aliases=["u_ke", "kinetic_edens", "kinetic_energydens"])
def u_kinetic(sim, partType, field, args):
    """Kinetic, as opposed to thermal, energy density."""
    dens_code = sim.snapshotSubset(partType, "Density", **args)
    vel_kms = sim.snapshotSubset(partType, "velmag", **args)

    return sim.units.calcKineticEnergyDensityCGS(dens_code, vel_kms)


u_kinetic.label = "Gas Kinetic Energy Density"
u_kinetic.units = r"$\rm{erg\ cm^{-3}}$"
u_kinetic.limits = [-6.0, 8.0]
u_kinetic.limits_halo = [-2.0, 7.0]
u_kinetic.log = True


@snap_field(aliases=["uratio_b_ke", "u_b_ke_ratio", "b_ke_edens_ratio"])
def uratio_mag_ke(sim, partType, field, args):
    """Ratio of gas magnetic to kinetic energy density."""
    u_kinetic = sim.snapshotSubset(partType, "u_kinetic", **args)
    u_magnetic = sim.snapshotSubset(partType, "p_b", **args)

    return u_magnetic / u_kinetic


uratio_mag_ke.label = r"Gas ($\rm{u_{mag} / u_{ke}}$) Ratio"
uratio_mag_ke.units = ""  # dimensionless
uratio_mag_ke.limits = [-4.0, 4.0]
uratio_mag_ke.limits_halo = [-3.0, 3.0]
uratio_mag_ke.log = True


@snap_field(alias="cooltime")
def tcool(sim, partType, field, args):
    """Gas cooling time (computed from saved GFM_CoolingRate), is np.nan if cell has net heating."""
    dens = sim.snapshotSubset(partType, "Density", **args)
    u = sim.snapshotSubset(partType, "InternalEnergy", **args)
    coolrate = sim.snapshotSubset(partType, "GFM_CoolingRate", **args)

    cooltime = sim.units.coolingTimeGyr(dens, coolrate, u)

    # also set eEOS gas to np.nan
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    w = np.where(sfr > 0.0)
    cooltime[w] = np.nan

    return cooltime


tcool.label = "Gas Cooling Time"
tcool.units = r"$\rm{Gyr}$"
tcool.limits = [-8.0, 2.0]
tcool.limits_halo = [-8.0, 2.5]
tcool.log = True


@snap_field(alias="coolingrate")
def coolrate(sim, partType, field, args):
    """Gas specific cooling rate (computed from saved GFM_CoolingRate), is np.nan if cell has net heating."""
    dens = sim.snapshotSubset(partType, "Density", **args)
    coolrate = sim.snapshotSubset(partType, "GFM_CoolingRate", **args)

    coolheat = sim.units.coolingRateToCGS(dens, coolrate)

    w = np.where(coolheat >= 0.0)
    coolheat[w] = np.nan  # cell is heating, so cooling rate is undefined

    return -1.0 * coolheat  # convention: positive


coolrate.label = "Gas Cooling Rate"
coolrate.units = r"$\rm{erg/s/g}$"
coolrate.limits = [-12.0, 2.0]
coolrate.limits_halo = [-8.0, 3.0]
coolrate.log = True


@snap_field(alias="heatingrate")
def heatrate(sim, partType, field, args):
    """Gas specific heating rate (computed from saved GFM_CoolingRate), is np.nan if cell has net cooling."""
    dens = sim.snapshotSubset(partType, "Density", **args)
    coolrate = sim.snapshotSubset(partType, "GFM_CoolingRate", **args)

    coolheat = sim.units.coolingRateToCGS(dens, coolrate)

    w = np.where(coolheat <= 0.0)
    coolheat[w] = np.nan  # cell is cooling, so heating rate is undefined

    return coolheat  # convention: positive


heatrate.label = "Gas Heating Rate"
heatrate.units = r"$\rm{erg/s/g}$"
heatrate.limits = [-14.0, 0.0]
heatrate.limits_halo = [-10.0, -1.0]
heatrate.log = True


@snap_field
def netcoolrate(sim, partType, field, args):
    """Gas net specific cooling rate (computed from saved GFM_CoolingRate)."""
    dens = sim.snapshotSubset(partType, "Density", **args)
    coolrate = sim.snapshotSubset(partType, "GFM_CoolingRate", **args)

    coolheat = sim.units.coolingRateToCGS(dens, coolrate)

    return coolheat  # convention: negative is cooling, positive is heating


netcoolrate.label = "Gas Net Cooling Rate"
netcoolrate.units = r"$\rm{erg/s/g}$"
netcoolrate.limits = [-1e3, 1e1]
netcoolrate.limits_halo = [-1e3, 1e1]
netcoolrate.log = False


@snap_field
def coolrate_powell(sim, partType, field, args):
    """Gas 'cooling rate' of Powell source term, specific (computed from saved DivB, GFM_CoolingRate)."""
    dens = sim.snapshotSubset(partType, "Density", **args)
    divb = sim.snapshotSubset(partType, "MagneticFieldDivergence", **args)
    bfield = sim.snapshotSubset(partType, "MagneticField", **args)
    vel = sim.snapshotSubset(partType, "Velocities", **args)
    vol = sim.snapshotSubset(partType, "Volume", **args)

    coolheat = sim.units.powellEnergyTermCGS(dens, divb, bfield, vel, vol)

    w = np.where(coolheat >= 0.0)
    coolheat[w] = np.nan  # cooling only

    return -1.0 * coolheat  # convention: positive


coolrate_powell.label = "Powell Cooling Rate"
coolrate_powell.units = r"$\rm{erg/s/g}$"
coolrate_powell.limits = [-16.0, 2.0]
coolrate_powell.limits_halo = [-12.0, 1.5]
coolrate_powell.log = True


@snap_field(alias="dt_hydro_yr")
def dt_hydro(sim, partType, field, args):
    """Gas cell hydrodynamical (Courant) timestep."""
    soundspeed = sim.snapshotSubset("gas", "soundspeed", **args)
    cellrad = sim.snapshotSubset("gas", "cellrad", **args)
    cellrad_kpc = sim.units.codeLengthToKpc(cellrad)
    cellrad_km = cellrad_kpc * sim.units.kpc_in_km

    dt_hydro_s = sim.units.CourantFac * cellrad_km / soundspeed
    dt_yr = dt_hydro_s / sim.units.s_in_yr

    return dt_yr


dt_hydro.label = "Gas Courant Timestep"
dt_hydro.units = r"$\rm{yr}$"
dt_hydro.limits = [1.0, 6.0]
dt_hydro.limits_halo = [1.0, 5.0]
dt_hydro.log = True


@snap_field(aliases=["depletion_time", "tau_dep"])
def tdep(sim, partType, field, args):
    """Gas cell depletion time: cells with zero sfr given nan."""
    mass = sim.units.codeMassToMsun(sim.snapshotSubset("gas", "mass", **args))
    sfr = sim.snapshotSubset("gas", "sfr", **args)

    t = np.zeros(mass.size, dtype="float32")
    t.fill(np.nan)

    w = np.where(sfr > 0)
    t[w] = mass[w] / sfr[w] / 1e9

    return t


tdep.label = "Gas Depletion Time"
tdep.units = r"$\rm{Gyr}$"
tdep.limits = [0.0, 12.0]
tdep.limits_halo = [0.0, 10.0]
tdep.log = False


@snap_field(multi=True)
def tau0_(sim, partType, field, args):
    """Optical depth to a certain line, at line center."""
    transition = field.split("_")[1].lower()  # e.g. "tau0_mgii2796", "tau0_mgii2803", "tau0_lya"
    print(field, transition)
    if "mgii" in transition:
        baseSpecies = "Mg II"
    elif "ovii" in transition:
        baseSpecies = "O VII"
    elif "ly" in transition:
        baseSpecies = "H I"  # note: uses internal hydrogen model, could use e.g. 'nhi_gk' (popping)
    else:
        raise Exception("Not handled.")

    temp = sim.snapshotSubset(partType, "temp_sfcold", **args)  # K
    dens = sim.snapshotSubset(partType, "%s numdens" % baseSpecies, **args)  # linear 1/cm^3
    cellsize = sim.snapshotSubset(partType, "cellsize", **args)  # code

    return sim.units.opticalDepthLineCenter(transition, dens, temp, cellsize)


tau0_.label = lambda sim, pt, f: r"Optical Depth $\rm{\tau_{%s,0}}$" % f.split("tau0_")[1]
tau0_.units = ""  # dimensionless
tau0_.limits = [-2.0, 3.0]
tau0_.limits_halo = [-2.0, 6.0]
tau0_.log = True


@snap_field(aliases=["dens_z", "dens_metal"])
def metaldens(sim, partType, field, args):
    """Total metal mass density."""
    dens = sim.snapshotSubset(partType, "dens", **args)
    dens *= sim.snapshotSubset(partType, "metallicity", **args)

    dens = sim.units.codeDensToPhys(dens, cgs=True)

    return dens


metaldens.label = "Metal Density"
metaldens.units = r"$\rm{g\ cm^{-3}}$"
metaldens.limits = [-36.0, -24.0]
metaldens.limits_halo = [-32.0, -24.0]
metaldens.log = True


@snap_field(multi=True)
def metaldens_(sim, partType, field, args):
    """Metal mass density for a given species, e.g. 'metaldens_O'."""
    species = field.replace("metaldens_", "").capitalize()

    dens = sim.snapshotSubset(partType, "dens", **args)
    dens *= sim.snapshotSubset(partType, "metals_" + species, **args)

    dens = sim.units.codeDensToPhys(dens, cgs=True)

    return dens


metaldens_.label = lambda sim, pt, f: "%s Metal Density" % f.replace("metaldens_", "")
metaldens_.units = r"$\rm{g\ cm^{-3}}$"
metaldens_.limits = [-40.0, -26.0]
metaldens_.limits_halo = [-36.0, -26.0]
metaldens_.log = True


@snap_field
def h_massfrac(sim, partType, field, args):
    """Total hydrogen (H) mass fraction. This is a custom helper for MCST runs where this value
    is not directly stored for gas, but is split based on sub-species and ionization state."""
    assert sim.isPartType(partType, "gas")
    assert sim.snapHasField(partType, "ElementFraction")

    frac = sim.snapshotSubset(partType, "HMMassFraction", **args)  # hydrogen anion (w/ extra electron)
    frac += sim.snapshotSubset(partType, "HIMassFraction", **args)  # neutral hydrogen
    frac += sim.snapshotSubset(partType, "HIIMassFraction", **args)  # single ionized hydrogen
    # frac += sim.snapshotSubset(partType, 'H2IMassFraction', **args) # neutral molecular hydrogen
    # frac += sim.snapshotSubset(partType, 'H2IIMassFraction', **args) # singly ionized molecular hydrogen

    return frac


h_massfrac.label = "H Mass Fraction"
h_massfrac.units = r""  # dimensionless
h_massfrac.limits = [0.0, 1.0]
h_massfrac.limits_halo = [0.0, 1.0]
h_massfrac.log = False


@snap_field
def he_massfrac(sim, partType, field, args):
    """Total helium (He) mass fraction. This is a custom helper for MCST runs where this value
    is not directly stored for gas, but is split based on sub-species and ionization state."""
    assert sim.isPartType(partType, "gas")
    assert sim.snapHasField(partType, "ElementFraction")

    frac = sim.snapshotSubset(partType, "HeIMassFraction", **args)  # neutral helium
    frac += sim.snapshotSubset(partType, "HeIIMassFraction", **args)  # once ionized helium
    frac += sim.snapshotSubset(partType, "HeIIIMassFraction", **args)  # twice ionized helium

    return frac


he_massfrac.label = "He Mass Fraction"
he_massfrac.units = r""  # dimensionless
he_massfrac.limits = [0.0, 1.0]
he_massfrac.limits_halo = [0.0, 1.0]
he_massfrac.log = False

# -------------------- gas observables ------------------------------------------------------------


@snap_field(aliases=["sz_yparam", "yparam"])
def sz_y(sim, partType, field, args):
    """(Thermal) Sunyaev-Zeldovich y-parameter (per gas cell)."""
    temp = sim.snapshotSubset(partType, "temp_sfcold", **args)
    xe = sim.snapshotSubset(partType, "ElectronAbundance", **args)
    mass = sim.snapshotSubset(partType, "Masses", **args)

    return sim.units.calcSunyaevZeldovichYparam(mass, xe, temp)


sz_y.label = "Sunyaev-Zeldovich y-parameter"
sz_y.units = r"$\rm{kpc^2}$"
sz_y.limits = [-12.0, -4.0]
sz_y.limits_halo = [-10.0, -4.0]
sz_y.log = True


@snap_field(aliases=["ksz_yparam", "ksz_y"])
def ksz_y(sim, partType, field, args):
    """(Kinetic) Sunyaev-Zeldovich y-parameter (per gas cell)."""
    xe = sim.snapshotSubset(partType, "ElectronAbundance", **args)
    mass = sim.snapshotSubset(partType, "Masses", **args)
    vel_los = sim.snapshotSubset(partType, "vel_z", **args)

    if sim.refVel is not None:
        vel_los -= sim.refVel[2]  # move into halo center of motion frame

    return sim.units.calcKineticSZYParam(mass, xe, vel_los)


ksz_y.label = "Kinetic SZ y-parameter"
ksz_y.units = r"$\rm{kpc^2/g}$"
ksz_y.limits = [-12.0, -4.0]
ksz_y.limits_halo = [-10.0, -4.0]
ksz_y.log = True


@snap_field(aliases=["frm_y", "frm_z"])
def frm_x(sim, partType, field, args):
    """Faraday rotation measure -integrand- (ne*B_parallel) in [rad m^-2]. Must be integrated through
    all cells along a line of sight, as sum(integrand*dl) where dl is the pathlength through each cell in pc.
    Requires the B-field component 'along the line-of-sight' which must be axis-aligned and specified."""
    projDir = field.split("_")[1]

    # Heesen+2023 Eqn. 2: RM = 0.81 * los_integral( (ne/cm^3) * (b_parallel/uG) * (dr/pc) ) in [rad m^-2]
    # see Prochaska+2019 Eqn S17: want a (1+z)^-2 factor for z>0?
    b = sim.snapshotSubset(partType, "b_%s" % projDir, **args)
    b = sim.units.particleCodeBFieldToGauss(b) * 1e6  # uG

    ne = sim.snapshotSubset(partType, "ne_twophase", **args)  # cm^-3

    frm = 0.812 * ne * b
    return frm


frm_x.label = "Faraday RM"
frm_x.units = r"$\rm{rad m^{-2}}$"
frm_x.limits = [-2.0, 2.0]
frm_x.limits_halo = [-4.0, -4.0]
frm_x.log = True


@snap_field(aliases=["p_sync_ska", "p_sync_ska_eta43", "p_sync_ska_alpha15", "p_sync_vla"])
def p_sync(sim, partType, field, args):
    """Radio synchrotron power (simple model)."""
    b = sim.snapshotSubset(partType, "MagneticField", **args)
    vol = sim.snapshotSubset(partType, "Volume", **args)

    modelArgs = {}
    if "_ska" in field:
        modelArgs["telescope"] = "SKA"
    if "_vla" in field:
        modelArgs["telescope"] = "VLA"
    if "_eta43" in field:
        modelArgs["eta"] = 4.0 / 3.0
    if "_alpha15" in field:
        modelArgs["alpha"] = 1.5

    return sim.units.synchrotronPowerPerFreq(b, vol, watts_per_hz=True, log=False, **modelArgs)


p_sync.label = "Synchrotron Power"
p_sync.units = r"$\rm{W/Hz}$"
p_sync.limits = [7.0, 26.0]
p_sync.limits_halo = [8.0, 26.0]
p_sync.log = True


@snap_field(aliases=["halpha", "sfr_halpha"])
def halpha_lum(sim, partType, field, args):
    """H-alpha line luminosity, simple model: linear conversion from SFR."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)

    return sim.units.sfrToHalphaLuminosity(sfr)


halpha_lum.label = r"$\rm{L_{H\alpha}}$"
halpha_lum.units = r"$\rm{10^{30}\ erg/s}$"  # 1e30 unit system to avoid overflow
halpha_lum.limits = [5.0, 12.0]
halpha_lum.limits_halo = [8.0, 12.0]
halpha_lum.log = True


@snap_field(aliases=["submm_flux", "s850um_flux_ismcut", "submm_flux_ismcut"])
def s850um_flux(sim, partType, field, args):
    """850 micron (sub-mm) flux (simple model)."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    metalmass = sim.snapshotSubset(partType, "metalmass_msun", **args)

    if "_ismcut" in field:
        temp = sim.snapshotSubset(partType, "temp_log", **args)
        dens = sim.snapshotSubset(partType, "Density", **args)
        ismCut = True
    else:
        temp = None
        dens = None
        ismCut = False

    return sim.units.gasSfrMetalMassToS850Flux(sfr, metalmass, temp, dens, ismCut=ismCut)


s850um_flux.label = r"$\rm{S_{850\mu m}}$"
s850um_flux.units = r"$\rm{mJy}$"
s850um_flux.limits = [-8.0, -2.0]
s850um_flux.limits_halo = [-5.0, -2.0]
s850um_flux.log = True


@snap_field(alias="xray")
def xray_lum(sim, partType, field, args):
    """Bolometric x-ray luminosity, simple bremsstrahlung (free-free) emission model only."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    dens = sim.snapshotSubset(partType, "dens", **args)
    mass = sim.snapshotSubset(partType, "mass", **args)
    u = sim.snapshotSubset(partType, "u", **args)
    xe = sim.snapshotSubset(partType, "xe", **args)

    return sim.units.calcXrayLumBolometric(sfr, u, xe, mass, dens)


xray_lum.label = r"$\rm{L_{X,bolometric}}$"
xray_lum.units = r"$\rm{10^{30}\ erg/s}$"  # 1e30 unit system to avoid overflow
xray_lum.limits = [2.0, 10.0]
xray_lum.limits_halo = [5.0, 10.0]
xray_lum.log = True


@snap_field(
    aliases=[
        "xray_lum_05-2kev",
        "xray_flux_05-2kev",
        "xray_lum_05-2kev_nomet",
        "xray_flux_05-2kev_nomet",
        "xray_counts_erosita",
        "xray_counts_chandra",
        "xray_lum_0.1-2.4kev",
        "xray_lum_0.5-2.0kev",
        "xray_lum_0.3-7.0kev",
        "xray_lum_0.5-5.0kev",
        "xray_lum_0.5-8.0kev",
        "xray_lum_2.0-10.0kev",
    ]
)
def xray_lum_apec(sim, partType, field, args):
    """X-ray luminosity/flux/counts (the latter for a given instrumental configuration).
    If a decimal point '.' in field, using my APEC-based tables, otherwise using XSPEC-based tables (from Nhut)."""
    from ..cosmo.xray import xrayEmission

    instrument = field.replace("xray_", "")
    if "." not in instrument:
        # XSPEC-based table conventions
        instrument = instrument.replace("-", "_").replace("kev", "")
        instrument = instrument.replace("lum_", "Luminosity_")
        instrument = instrument.replace("flux_", "Flux_")
        instrument = instrument.replace("_nomet", "_NoMet")
        instrument = instrument.replace("counts_erosita", "Count_Erosita_05_2_2ks")  # only available config
        instrument = instrument.replace("counts_chandra", "Count_Chandra_03_5_100ks")  # as above
    else:
        # APEC-based table conventions
        instrument = instrument.replace("lum_", "emis_")

    xray = xrayEmission(sim, instrument, use_apec=("." in field))

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    xray = xray.calcGasEmission(sim, instrument, indRange=indRange)

    if "lum_" in field:
        xray = (xray / 1e30).astype("float32")  # 10^30 erg/s unit system to avoid overflow

    return xray


def xray_lum_apec_metadata(sim, pt, f, ret):
    """Helper to determine xray_* field metadata."""
    label = "X"

    if "05-2kev" in f:
        label = "X, 0.5-2 keV"
    if "05-2kev_nomet" in f:
        label = "X, 0.5-2 keV, no-Z"
    if "0.1-2.4kev" in f:
        label = "X, 0.5-2 keV"
    if "0.5-2.0kev" in f:
        label = "X, 0.5-2 keV"
    if "0.3-7.0kev" in f:
        label = "X, 0.3-7 keV"
    if "0.5-5.0kev" in f:
        label = "X, 0.5-5 keV"
    if "0.5-8.0kev" in f:
        label = "X, 0.5-8 keV"
    if "2.0-10.0kev" in f:
        label = "X, 2-10 keV"

    if "lum_" in f:
        label = r"$\rm{L_{" + label + "}}$"
        units = r"$\rm{10^{30}\ erg/s}$"
        limits = [2.0, 10.0]
        limits_halo = [5.0, 11.0]

    elif "flux_" in f:
        label = r"$\rm{F_{" + label + "}}$"
        units = r"$\rm{erg/s/cm^{2}}$"
        limits = [-30.0, -14.0]
        limits_halo = [-20.0, -12.0]

    elif "counts_" in f:
        label = "%s X-ray Count Rate" % f.replace("xray_counts_", "")
        units = r"$\rm{s^{-1}}$"  # dimensionless
        limits = [-6.0, 2.0]
        limits_halo = [-6.0, 2.0]

    if ret == "label":
        return label
    if ret == "units":
        return units
    if ret == "limits":
        return limits
    if ret == "limits_halo":
        return limits_halo


xray_lum_apec.label = partial(xray_lum_apec_metadata, ret="label")
xray_lum_apec.units = partial(xray_lum_apec_metadata, ret="units")
xray_lum_apec.limits = partial(xray_lum_apec_metadata, ret="limits")
xray_lum_apec.limits_halo = partial(xray_lum_apec_metadata, ret="limits_halo")
xray_lum_apec.log = True


@snap_field(alias="hi_column")
def n_hi(sim, partType, field, args):
    """Experimental: assign a N_HI (column density) to every cell based on a (xy) grid projection."""
    for k in ["inds", "indRange", "haloID", "subhaloID"]:
        assert args[k] is None  # otherwise generalize

    # cache
    savePath = sim.derivPath + "cache/hi_column_%s_%03d.hdf5" % (partType, sim.snap)

    if isfile(savePath):
        with h5py.File(savePath, "r") as f:
            data = f["hi_column"][()]
        print("Loaded: [%s]" % savePath)
    else:
        # config
        acFieldName = "Box_Grid_nHI_GK_depth10"
        boxWidth = 10000.0  # only those in slice, columns don't apply to others
        z_bounds = [sim.boxSize * 0.5 - boxWidth / 2, sim.boxSize * 0.5 + boxWidth / 2]

        # load z coords
        pos_z = sim.snapshotSubset(partType, "pos_z", **args)

        data = np.zeros(pos_z.shape[0], dtype="float32")  # allocate
        data.fill(np.nan)

        w = np.where((pos_z > z_bounds[0]) & (pos_z < z_bounds[1]))
        pos_z = None

        # load x,y coords and find grid indices
        pos_x = sim.snapshotSubset(partType, "pos_x", **args)[w]
        pos_y = sim.snapshotSubset(partType, "pos_y", **args)[w]

        grid = sim.auxCat(acFieldName)[acFieldName]
        grid = 10.0**grid  # log -> linear
        pxSize = sim.boxSize / grid.shape[0]

        x_ind = np.floor(pos_x / pxSize).astype("int64")
        y_ind = np.floor(pos_y / pxSize).astype("int64")

        data[w] = grid[y_ind, x_ind]

        # save
        with h5py.File(savePath, "w") as f:
            f["hi_column"] = data
        print("Saved: [%s]" % savePath)

    return data


n_hi.label = r"$\rm{N_{HI}}$"
n_hi.units = r"$\rm{cm^{-2}}$"
n_hi.limits = [15.0, 22.0]
n_hi.log = True


@snap_field(aliases=["mass_hi", "h i mass", "hi mass", "h i numdens"])
def hi_mass(sim, partType, field, args):
    """Hydrogen model: atomic H (neutral subtracting molecular) mass calculation."""
    from ..cosmo.hydrogen import hydrogenMass

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    # hydrogen model mass calculation (todo: generalize to different molecular models)
    data = hydrogenMass(None, sim, atomic=True, indRange=indRange)

    if "numdens" in field:
        data /= sim.snapshotSubset(partType, "volume", **args)
        data = sim.units.codeDensToPhys(data, cgs=True, numDens=True)  # linear [H atoms/cm^3]

    return data


hi_mass.label = r"$\rm{M_{HI}}$"
hi_mass.units = "code_mass"
hi_mass.limits = [-2.0, 6.0]
hi_mass.limits_halo = [1.0, 6.0]
hi_mass.log = True


@snap_field(aliases=["mass_h2", "h2mass"])
def h2_mass(sim, partType, field, args):
    """Hydrogen model: molecular H (neutral subtracting atomic) mass calculation."""
    from ..cosmo.hydrogen import hydrogenMass

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    # choose molecular model: could generalize this field to accept different options
    molecularModel = "BL06"
    print("Note: using [%s] model for H2 by default." % molecularModel)

    data = hydrogenMass(None, sim, molecular=molecularModel, indRange=indRange)

    return data


h2_mass.label = r"$\rm{M_{H2}}$"
h2_mass.units = "code_mass"
h2_mass.limits = [-3.0, 5.0]
h2_mass.limits_halo = [0.0, 5.0]
h2_mass.log = True


@snap_field(aliases=["mhi_br", "mhi_gk", "mhi_kmt", "mh2_br", "mh2_gk", "mh2_kmt"])
def mass_h_popping(sim, partType, field, args):
    """Pre-computed atomic (HI) and molecular (H2) gas cell masses (from Popping+2019)."""
    if field == "mass_h_popping":
        raise Exception("Specify HI or H2, and molecular model.")

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    path = sim.postPath + "hydrogen/gas_%03d.hdf5" % sim.snap
    # model = field.split("_")[1].upper()  # BR, GK, or KMT

    if not isfile(path):
        print("Warning: [%s] from [%s] does not exist, empty return." % (field, path))
        return None

    with h5py.File(path, "r") as f:
        # dataset naming convention
        key = field.replace("_", "").upper()

        if key in f:
            # old storage: MH, MH2*, and MHI* all explicitly
            if indRange is None:
                masses = f[key][()]
            else:
                masses = f[key][indRange[0] : indRange[1] + 1]
        else:
            # more compact storage: only MH and MH2*, where MHI must be derived
            assert "MHI" in key
            if indRange is None:
                MH = f["MH"][()]
                masses = MH - f[key.replace("HI", "H2")][()]
            else:
                MH = f["MH"][indRange[0] : indRange[1] + 1]
                masses = MH - f[key.replace("HI", "H2")][indRange[0] : indRange[1] + 1]

    return masses


mass_h_popping.label = lambda sim, pt, f: r"$\rm{M_{HI,%s}}$" % f.split("_")[1].upper()
mass_h_popping.units = "code_mass"
mass_h_popping.limits = [-16.0, -4.0]
mass_h_popping.limits_halo = [-12.0, -4.0]
mass_h_popping.log = True


@snap_field(aliases=["nhi_br", "nhi_gk", "nhi_kmt", "nh2_br", "nh2_gk", "nh2_kmt"])
def numdens_h_popping(sim, partType, field, args):
    """Pre-computed atomic (HI) and molecular (H2) gas cell number densities (from Popping+2019)."""
    if field == "numdens_h_popping":
        raise Exception("Specify HI or H2, and molecular model.")

    mass_field = field.replace("nhi_", "mhi_").replace("nh2_", "mh2_")
    masses = sim.snapshotSubset(partType, mass_field, **args)

    dens = masses / sim.snapshotSubset(partType, "volume", **args)
    dens = sim.units.codeDensToPhys(dens, cgs=True, numDens=True)  # [H atoms/cm^3]

    return dens


numdens_h_popping.label = lambda sim, pt, f: r"$\rm{n_{HI,%s}}$" % f.split("_")[1].upper()
numdens_h_popping.units = r"$\rm{cm^{-3}}$"
numdens_h_popping.limits = [-14.0, -1.0]
numdens_h_popping.limits_halo = [-12.0, 0.0]
numdens_h_popping.log = True


@snap_field(aliases=["mhi_gd14", "mhi_gk11", "mhi_k13", "mhi_s14", "mh2_gd14", "mh2_gk11", "mh2_k13", "mh2_s14"])
def mass_h_diemer(sim, partType, field, args):
    """Pre-computed atomic (HI) and molecular (H2) gas cell masses (from Diemer+2019)."""
    if field == "mass_h_diemer":
        raise Exception("Specify HI or H2, and molecular model.")

    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])

    path = sim.postPath + "hydrogen/diemer_%03d.hdf5" % sim.snap
    key = "f_mol_" + field.split("_")[1].upper()

    with h5py.File(path, "r") as f:
        if indRange is None:
            f_mol = f[key][()]
            f_neutral_H = f["f_neutral_H"][()]
        else:
            f_mol = f[key][indRange[0] : indRange[1] + 1]
            f_neutral_H = f["f_neutral_H"][indRange[0] : indRange[1] + 1]

    # file contains f_mol, for M_H2 = Mass_gas * f_neutral_H * f_mol,
    # while for M_HI = MasS_gas * f_neutral_H * (1-f_mol)
    mass = sim.snapshotSubset(partType, "mass", **args)

    if "mh2_" in field:
        mass = mass * f_neutral_H * f_mol
    if "mhi_" in field:
        mass = mass * f_neutral_H * (1.0 - f_mol)

    return mass


mass_h_diemer.label = lambda sim, pt, f: r"$\rm{M_{HI,%s}}$" % f.split("_")[1].upper()
mass_h_diemer.units = "code_mass"
mass_h_diemer.limits = [-16.0, -4.0]
mass_h_diemer.limits_halo = [-12.0, -4.0]
mass_h_diemer.log = True


@snap_field(aliases=["nhi_gd14", "nhi_gk11", "nhi_k13", "nhi_s14", "nh2_gd14", "nh2_gk11", "nh2_k13", "nh2_s14"])
def numdens_h_diemer(sim, partType, field, args):
    """Pre-computed atomic (HI) and molecular (H2) gas cell number densities (from Diemer+2019)."""
    if field == "numdens_h_diemer":
        raise Exception("Specify HI or H2, and molecular model.")

    mass_field = field.replace("nhi_", "mhi_").replace("nh2_", "mh2_")
    masses = sim.snapshotSubset(partType, mass_field, **args)

    dens = masses / sim.snapshotSubset(partType, "volume", **args)
    dens = sim.units.codeDensToPhys(dens, cgs=True, numDens=True)  # [H atoms/cm^3]

    return dens


numdens_h_diemer.label = lambda sim, pt, f: r"$\rm{n_{HI,%s}}$" % f.split("_")[1].upper()
numdens_h_diemer.units = r"$\rm{cm^{-3}}$"
numdens_h_diemer.limits = [-14.0, -1.0]
numdens_h_diemer.limits_halo = [-12.0, 0.0]
numdens_h_diemer.log = True


def _cloudy_load(sim, partType, field, args):
    """Helper caching loader, for all of the following CLOUDY-based fields."""
    from ..cosmo.cloudy import cloudyEmission, cloudyIon

    if "flux" in field or "lum" in field:
        lineName, prop = field.rsplit(" ", 1)
        lineName = lineName.replace("-", " ")  # e.g. "O--8-16.0067A" -> "O  8 16.0067A"
        dustDepletion = False
        if "_dustdepleted" in prop:  # e.g. "MgII lum_dustdepleted"
            dustDepletion = True
            prop = prop.replace("_dustdepleted", "")
    else:
        solarAbunds = False
        element, ionNum, prop = (
            field.split()
        )  # e.g. "O VI mass", "Mg II frac", "C IV numdens", or "Si II numdens_solar"
        if "_solar" in prop:
            solarAbunds = True
            prop = prop.replace("_solar", "")

    assert sim.isPartType(partType, "gas")
    assert prop in ["mass", "frac", "flux", "lum", "lum2phase", "numdens"]

    # indRange subset herein (do not change args dict, could be used on other fields)
    indRangeOrig = args["indRange"]
    assert args["inds"] is None  # custom field, old code, nowadays should not ever be passed inds, only indRange

    # haloID or subhaloID subset
    if args["haloID"] is not None or args["subhaloID"] is not None:
        assert indRangeOrig is None and args["inds"] is None
        subset = sim.haloOrSubhaloSubset(haloID=args["haloID"], subhaloID=args["subhaloID"])
        offset = subset["offsetType"][sim.ptNum(partType)]
        length = subset["lenType"][sim.ptNum(partType)]
        indRangeOrig = [offset, offset + length - 1]  # inclusive below

    # check memory cache (only simplest support at present, for indRange returns of global cache)
    cache_key = "snap%d_%s_%s" % (sim.snap, partType, field.replace(" ", "_"))

    if cache_key in sim.data:
        if indRangeOrig is not None:
            print(
                "NOTE: Returning [%s] from cache, indRange [%d - %d]!" % (cache_key, indRangeOrig[0], indRangeOrig[1])
            )
            return sim.data[cache_key][indRangeOrig[0] : indRangeOrig[1] + 1]
        if args["inds"] is not None:
            print("NOTE: Returning [%s] from cache, [%d] discrete indices!" % (cache_key, args["inds"].size))
            return sim.data[cache_key][args["inds"]]

        # if key exists but neither indRange or inds specified, we return this (possibly custom subset)
        print("CAUTION: Cached return [%s], and indRange is None, returning all of sim.data field." % cache_key)
        return sim.data[cache_key]

    # full snapshot-level caching, create during normal usage but not web (always use if exists)
    useCache = True
    createCache = True

    if getuser() == "wwwrun":
        createCache = False
    if args["haloID"] is not None or args["subhaloID"] is not None:
        createCache = False
    if hasattr(sim, "createCloudyCache") and not sim.createCloudyCache:
        createCache = False

    sbStr = "sb%d_" % sim.subbox if sim.subbox is not None else ""
    cacheFile = sim.cachePath + "cached_%s_%s_%s%d.hdf5" % (partType, field.replace(" ", "-"), sbStr, sim.snap)
    indRangeAll = [0, sim.numPart[sim.ptNum(partType)]]

    if useCache:
        # does not exist yet, and should create?
        nChunks = numPartToChunkLoadSize(indRangeAll[1])

        # does a cache file already exist? only use it if the calculation finished
        if isfile(cacheFile):
            remakeCacheFile = False

            if getsize(cacheFile) < 2000:
                print("Warning: Found cache file [%s], but size is abnormally small, remaking." % cacheFile)
                remakeCacheFile = True
            else:
                with h5py.File(cacheFile, "r") as f:
                    nChunksDone = f.attrs["nChunksDone"] if "nChunksDone" in f.attrs else nChunks  # fallback
                    nChunks = f.attrs["nChunksTotal"] if "nChunksTotal" in f.attrs else nChunks  # fallback

                if nChunksDone < nChunks:
                    print(
                        "Warning: Found cache file [%s], but only has [%d] of [%d] chunks done, remaking."
                        % (cacheFile.split(sim.cachePath)[1], nChunksDone, nChunks)
                    )
                    remakeCacheFile = True

        if createCache and (not isfile(cacheFile) or remakeCacheFile):
            # compute for indRange == None (whole snapshot) with a reasonable pSplit
            print(
                "Creating [%s] for [%d] particles in [%d] chunks (set sP.createCloudyCache = False to disable)."
                % (cacheFile.split(sim.derivPath)[1], indRangeAll[1], nChunks)
            )

            # create file and init ionization calculator
            with h5py.File(cacheFile, "w") as f:
                f.attrs["nChunksDone"] = 0
                f.attrs["nChunksTotal"] = nChunks
                f.create_dataset("field", (indRangeAll[1],), dtype="float32")

            if prop in ["mass", "frac", "numdens"]:
                ion = cloudyIon(sim, el=element, redshiftInterp=True)
            else:
                emis = cloudyEmission(sim, line=lineName, redshiftInterp=True)
                wavelength = emis.lineWavelength(lineName)

            # process chunked
            for i in range(nChunks):
                indRangeLocal = pSplitRange(indRangeAll, nChunks, i)

                # indRange is inclusive for snapshotSubset(), so skip saving the very last
                # element, which is included in the next return of pSplitRange()
                indRangeLocal[1] = int(indRangeLocal[1] - 1)

                if indRangeLocal[0] == indRangeLocal[1]:
                    continue  # we are done

                if prop in ["mass", "frac", "numdens"]:
                    # either ionization fractions, or total mass in the ion
                    values = ion.calcGasMetalAbundances(
                        sim, element, ionNum, indRange=indRangeLocal, solarAbunds=solarAbunds, parallel=True
                    )

                    if prop == "mass":
                        values *= sim.snapshotSubset(partType, "Masses", indRange=indRangeLocal)

                    if prop == "numdens":
                        values *= sim.snapshotSubset(partType, "numdens", indRange=indRangeLocal)
                        values /= ion.atomicMass(element)  # [H atoms/cm^3] to [ions/cm^3]

                elif prop == "lum":
                    # by default, gas temperature is 'temp_sfcold' i.e. star-forming gas is set to 1000 K
                    values = emis.calcGasLineLuminosity(
                        sim, lineName, indRange=indRangeLocal, dustDepletion=dustDepletion
                    )
                    values /= 1e30  # 10^30 erg/s unit system to avoid overflow

                elif prop == "lum2phase":
                    # for star-forming gas, include contributions from both the cold and hot phases,
                    # with their respective mass fractions
                    values = emis.calcGasLineLuminosity(
                        sim, lineName, indRange=indRangeLocal, dustDepletion=dustDepletion, sfGasTemp="both"
                    )
                    values /= 1e30  # 10^30 erg/s unit system to avoid overflow

                elif prop == "flux":
                    # emission flux
                    lum = emis.calcGasLineLuminosity(sim, lineName, indRange=indRangeLocal, dustDepletion=dustDepletion)
                    values = sim.units.luminosityToFlux(lum, wavelength=wavelength)  # [photon/s/cm^2] @ sim.redshift

                with h5py.File(cacheFile, "a") as f:
                    f.attrs["nChunksDone"] = f.attrs["nChunksDone"] + 1
                    f["field"][indRangeLocal[0] : indRangeLocal[1] + 1] = values

                print(" [%2d] saved %d - %d" % (i, indRangeLocal[0], indRangeLocal[1]), flush=True)
            print("Saved: [%s]." % cacheFile.split(sim.derivPath)[1])

        # load from existing cache if it exists
        if isfile(cacheFile):
            # if getuser() != 'wwwrun':
            #    print('Loading [%s] [%s] from [%s].' % (partType,field,cacheFile.split(sim.derivPath)[1]))

            with h5py.File(cacheFile, "r") as f:
                assert f["field"].size == indRangeAll[1]
                if indRangeOrig is None and args["inds"] is None:
                    values = f["field"][()]
                elif indRangeOrig is not None:
                    values = f["field"][indRangeOrig[0] : indRangeOrig[1] + 1]
                elif args["inds"] is not None:
                    indRange = [np.min(args["inds"]), np.max(args["inds"])]
                    values = f["field"][indRange[0] : indRange[1] + 1]
                    return values[args["inds"] - np.min(args["inds"])]

    if not useCache or not isfile(cacheFile):
        # don't use cache, or tried to use and it doesn't exist yet, so run computation now
        if prop in ["mass", "frac", "numdens"]:
            ion = cloudyIon(sim, el=element, redshiftInterp=True)
        else:
            emis = cloudyEmission(sim, line=lineName, redshiftInterp=True)
            wavelength = emis.lineWavelength(lineName)

        if prop in ["mass", "frac", "numdens"]:
            # either ionization fractions, or total mass in the ion
            values = ion.calcGasMetalAbundances(sim, element, ionNum, indRange=indRangeOrig)
            if prop == "mass":
                values *= sim.snapshotSubset(partType, "Masses", indRange=indRangeOrig)
            if prop == "numdens":
                values *= sim.snapshotSubset(partType, "numdens", indRange=indRangeOrig)
                values /= ion.atomicMass(element)  # [H atoms/cm^3] to [ions/cm^3]
        elif prop == "lum":
            values = emis.calcGasLineLuminosity(sim, lineName, indRange=indRangeOrig, dustDepletion=dustDepletion)
            values /= 1e30  # 10^30 erg/s unit system to avoid overflow
        elif prop == "lum2phase":
            values = emis.calcGasLineLuminosity(
                sim, lineName, indRange=indRangeOrig, dustDepletion=dustDepletion, sfGasTemp="both"
            )
            values /= 1e30  # 10^30 erg/s unit system to avoid overflow
        elif prop == "flux":
            # emission flux
            lum = emis.calcGasLineLuminosity(sim, lineName, indRange=indRangeOrig, dustDepletion=dustDepletion)
            values = sim.units.luminosityToFlux(lum, wavelength=wavelength)  # [photon/s/cm^2]

    return values


@snap_field(multi=" mass")
def cloudy_mass_(sim, partType, field, args):
    """CLOUDY-based photoionization calculation: **total ionic mass** (e.g. 'O VI mass', 'Mg II mass'),
    for any known ion name and excited level number. Note: uses spaces in field name."""
    return _cloudy_load(sim, partType, field, args)


cloudy_mass_.label = lambda sim, pt, f: "%s %s Ionic Mass" % (f.split()[0], f.split()[1])
cloudy_mass_.units = r"$\rm{M_{sun}}$"
cloudy_mass_.limits = [1.0, 7.0]
cloudy_mass_.limits_halo = [2.0, 6.0]
cloudy_mass_.log = True


@snap_field(multi=" frac")
def cloudy_frac_(sim, partType, field, args):
    """CLOUDY-based photoionization calculation: **ionic mass fraction** (e.g. 'O VI frac', 'Mg II frac'),
    for a given ion name and excited level number. Note: uses spaces in field name."""
    return _cloudy_load(sim, partType, field, args)


cloudy_frac_.label = lambda sim, pt, f: "%s %s Ionization Fraction" % (f.split()[0], f.split()[1])
cloudy_frac_.units = ""  # dimensionless
cloudy_frac_.limits = [-10.0, -2.0]
cloudy_frac_.limits_halo = [-10.0, -4.0]
cloudy_frac_.log = True


@snap_field(multi=" numdens")
def cloudy_numdens_(sim, partType, field, args):
    """CLOUDY-based photoionization calculation: **ionic number density** (e.g. 'O VI numdens', 'Mg II numdens'),
    for a given ion name and excited level number. Note: uses spaces in field name."""
    return _cloudy_load(sim, partType, field, args)


cloudy_numdens_.label = lambda sim, pt, f: r"$\rm{n_{%s%s}}$" % (f.split()[0], f.split()[1])
cloudy_numdens_.units = r"$\rm{cm^{-3}}$"
cloudy_numdens_.limits = [-14.0, -4.0]
cloudy_numdens_.limits_halo = [-12.0, -6.0]
cloudy_numdens_.log = True


@snap_field(multi=" flux")
def cloudy_flux_(sim, partType, field, args):
    """CLOUDY-based photoionization calculation: **ion line emission flux** (e.g. 'H-alpha flux', 'O--6-1037.62A'),
    for a given line name. Note: uses spaces in field name."""
    return _cloudy_load(sim, partType, field, args)


cloudy_flux_.label = lambda sim, pt, f: "%s Line Flux" % (f.replace(" flux", "").replace("-", " "))
cloudy_flux_.units = r"$\rm{photon/s/cm^2}$"
cloudy_flux_.limits = [-30.0, -15.0]
cloudy_flux_.limits_halo = [-25.0, -15.0]
cloudy_flux_.log = True


@snap_field(multi=" lum")
def cloudy_lum_(sim, partType, field, args):
    """CLOUDY-based photoionization calculation: **ion line luminosity** (e.g. 'MgII lum', 'CVI lum'),
    for a given line name. Note: uses spaces in field name."""
    return _cloudy_load(sim, partType, field, args)


cloudy_lum_.label = lambda sim, pt, f: "%s Luminosity" % (f.replace(" lum", "").replace("_dustdepleted", " "))
cloudy_lum_.units = r"$\rm{10^{30}\ erg/s}$"
cloudy_lum_.limits = [-15.0, 10.0]
cloudy_lum_.limits_halo = [-5.0, 10.0]
cloudy_lum_.log = True


@snap_field(multi=True)
def ionmassratio_(sim, partType, field, args):
    """Ratio between two ionic masses, e.g. 'ionmassratio_O6_O8'."""
    from ..cosmo.cloudy import cloudyIon

    ion = cloudyIon(sP=None)
    ion1, ion2, _ = field.split("_")

    mass1 = sim.snapshotSubset(partType, "%s mass" % ion.formatWithSpace(ion1), **args)
    mass2 = sim.snapshotSubset(partType, "%s mass" % ion.formatWithSpace(ion2), **args)
    return mass1 / mass2


ionmassratio_.label = lambda sim, pt, f: "(%s / %s) Mass Ratio" % (f.split("_")[1], f.split("_")[2])
ionmassratio_.units = ""  # dimensionless
ionmassratio_.limits = [-3.0, 3.0]
ionmassratio_.limits_halo = [-2.0, 2.0]
ionmassratio_.log = True

# -------------------- gas (wind model) -----------------------------------------------------------


@snap_field(aliases=["wind_edot", "sn_dedt", "sn_edot", "sf_dedt", "sf_edot"])
def wind_dedt(sim, partType, field, args):
    """TNG/SH03 wind model: feedback energy injection rate."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    metal = sim.snapshotSubset(partType, "metal", **args)

    return sim.units.codeSfrZToWindEnergyRate(sfr, metal)


wind_dedt.label = "Wind Energy Injection Rate"
wind_dedt.units = r"$\rm{10^{51}\ erg/s}$"  # 1e51 unit system to avoid overflow
wind_dedt.limits = [-16.0, -10.0]
wind_dedt.limits_halo = [-15.0, -10.0]
wind_dedt.log = True


@snap_field(aliases=["wind_pdot", "sn_dpdt", "sn_pdot", "sf_dpdt", "sf_pdot"])
def wind_dpdt(sim, partType, field, args):
    """TNG/SH03 wind model: feedback momentum injection rate."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    metal = sim.snapshotSubset(partType, "metal", **args)
    dm_sigma = sim.snapshotSubset(partType, "SubfindVelDisp", **args)

    return sim.units.codeSfrZToWindMomentumRate(sfr, metal, dm_sigma)


wind_dpdt.label = "Wind Momentum Injection Rate"
wind_dpdt.units = r"$\rm{10^{51}\ g\,cm\,s^{-2}}$"  # 1e51 unit system to avoid overflow
wind_dpdt.limits = [-24.0, -18.0]
wind_dpdt.limits_halo = [-23.0, -18.0]
wind_dpdt.log = True


@snap_field(alias="wind_launchvel")
def wind_vel(sim, partType, field, args):
    """TNG/SH03 wind model: launch velocity."""
    dm_sigma = sim.snapshotSubset(partType, "SubfindVelDisp", **args)

    return sim.units.sigmaDMToWindVel(dm_sigma)


wind_vel.label = "Wind Launch Velocity"
wind_vel.units = r"$\rm{km/s}$"
wind_vel.limits = [300, 2000]
wind_vel.limits_halo = [300, 5000]
wind_vel.log = True


@snap_field(aliases=["wind_massloading", "wind_etam"])
def wind_eta(sim, partType, field, args):
    """TNG/SH03 wind model: mass loading factor (at launch)."""
    sfr = sim.snapshotSubset(partType, "sfr", **args)
    metal = sim.snapshotSubset(partType, "metal", **args)
    dm_sigma = sim.snapshotSubset(partType, "SubfindVelDisp", **args)

    return sim.units.codeSfrZSigmaDMToWindMassLoading(sfr, metal, dm_sigma)


wind_eta.label = r"Wind Mass Loading $\rm{\eta_m}$"
wind_eta.units = ""  # dimensionless
wind_eta.limits = [-2.0, 2.0]
wind_eta.limits_halo = [-1.0, 1.5]
wind_eta.log = True

# -------------------- gas (mcst model) -----------------------------------------------------------


@snap_field(aliases=["raddens_fuv", "rad_fuv_habing", "raddens_fuv_habing"])
def rad_fuv(sim, partType, field, args):
    """Radiation energy density in FUV band (6 - 13.6 eV)."""
    edens_bands = sim.snapshotSubset(partType, "RadiationEnergyDensity", **args)
    edens = edens_bands[:, 0]  # FUV band is first index (generalize)

    return sim.units.codeEnergyDensToErgPerCm3(edens)


rad_fuv.label = r"FUV Radiation Energy Density"
rad_fuv.units = lambda sim, pt, f: r"erg cm$^{-3}$" if "_habing" not in f else r"$\rm{Habing}$"
rad_fuv.limits = lambda sim, pt, f: [-20, -10] if "_habing" not in f else [-6, 2]
rad_fuv.limits_halo = lambda sim, pt, f: [-14, -8] if "_habing" not in f else [-4, 4]
rad_fuv.log = True


@snap_field(aliases=["raddens_lw", "rad_lw_habing", "raddens_lw_habing"])
def rad_lw(sim, partType, field, args):
    """Radiation energy density in Lyman-Werner band (11.2 - 12.3 eV)."""
    edens_bands = sim.snapshotSubset(partType, "RadiationEnergyDensity", **args)
    edens = edens_bands[:, 1]  # LW is second index (generalize)

    if "_habing" in field:
        return sim.units.codeEnergyDensToHabing(edens)
    return sim.units.codeEnergyDensToErgPerCm3(edens)


rad_lw.label = r"LW Radiation Energy Density"
rad_lw.units = lambda sim, pt, f: r"erg cm$^{-3}$" if "_habing" not in f else r"$\rm{Habing}$"
rad_lw.units = lambda sim, pt, f: r"erg cm$^{-3}$" if "_habing" not in f else r"$\rm{Habing}$"
rad_lw.limits = lambda sim, pt, f: [-20, -10] if "_habing" not in f else [-6, 2]
rad_lw.limits_halo = lambda sim, pt, f: [-14, -8] if "_habing" not in f else [-4, 4]
rad_lw.log = True


@snap_field
def rad_fuv_lw_ratio(sim, partType, field, args):
    """Ratio of radiation energy density in FUV to Lyman-Werner band."""
    edens_bands = sim.snapshotSubset(partType, "RadiationEnergyDensity", **args)
    edens_fuv = edens_bands[:, 0]  # FUV band is first index (generalize)
    edens_lw = edens_bands[:, 1]  # LW is second index

    return edens_fuv / edens_lw


rad_fuv_lw_ratio.label = r"(FUV / LW) Radiation Ratio"
rad_fuv_lw_ratio.units = ""  # dimensionless
rad_fuv_lw_ratio.limits = [-1.0, 1.0]
rad_fuv_lw_ratio.limits_halo = [-1.0, 1.0]
rad_fuv_lw_ratio.log = True


@snap_field
def rad_fuv_uvb_ratio(sim, partType, field, args):
    """Ratio of local to UVB radiation energy density in FUV band."""
    edens_fuv = sim.snapshotSubset(partType, "rad_fuv", **args)

    # could make this sim independent in the future
    path = sim.arepoPath + sim.params["GrackleDataFile"].decode()
    path = path.replace("arepo7", "arepo8")  # hack for now

    with h5py.File(path, "r") as f:
        uvb_z = f["UVBEnergyDens/Redshift"][()]
        uvb_edens_z = f["UVBEnergyDens/EnergyDensity_6.0-13.6eV"][()]  # [log erg/cm^3]

    # find closest redshift, get (single constant) UVB energy density
    z_found, z_ind = closest(uvb_z, sim.redshift)
    edens_uvb = 10.0 ** uvb_edens_z[z_ind]

    return edens_fuv / edens_uvb


rad_fuv_uvb_ratio.label = r"(Local / UVB) FUV Radiation Ratio"
rad_fuv_uvb_ratio.units = ""  # dimensionless
rad_fuv_uvb_ratio.limits = [-1.0, 1.0]
rad_fuv_uvb_ratio.limits_halo = [-1.0, 1.0]
rad_fuv_uvb_ratio.log = True

# -------------------- gas/stars ------------------------------------------------------------------


@snap_field(alias="metal_solar")
def z_solar(sim, partType, field, args):
    """Metallicity in solar units."""
    metal = sim.snapshotSubset(partType, "metal", **args)  # (metal mass / total mass) ratio

    return sim.units.metallicityInSolar(metal)


z_solar.label = "[pt] Metallicity"
z_solar.units = r"$\rm{Z_{sun}}$"
z_solar.limits = [-3.5, 1.0]
z_solar.limits_halo = [-2.0, 1.0]
z_solar.log = True


@snap_field
def sn_iaii_ratio_fe(sim, partType, field, args):
    """GFM_MetalsTagged: ratio of iron mass [linear] produced in SNIa versus SNII."""
    metals_FeSNIa = sim.snapshotSubset(partType, "metals_FeSNIa", **args)
    metals_FeSNII = sim.snapshotSubset(partType, "metals_FeSNII", **args)
    return metals_FeSNIa / metals_FeSNII


sn_iaii_ratio_fe.label = r"[pt] Mass Ratio $\rm{Fe_{SNIa} / {Fe}_{SNII}}$"
sn_iaii_ratio_fe.units = ""  # dimensionless
sn_iaii_ratio_fe.limits = [-4.0, 6.0]
sn_iaii_ratio_fe.limits_halo = [-3.0, 5.5]
sn_iaii_ratio_fe.log = True


@snap_field
def sn_iaii_ratio_metals(sim, partType, field, args):
    """GFM_MetalsTagged: ratio of total metal mass [linear] produced in SNIa versus SNII."""
    metals_SNIa = sim.snapshotSubset(partType, "metals_SNIa", **args)
    metals_SNII = sim.snapshotSubset(partType, "metals_SNII", **args)
    return metals_SNIa / metals_SNII


sn_iaii_ratio_metals.label = r"[pt] Mass Ratio $\rm{Z_{SNIa} / Z_{SNII}}$"
sn_iaii_ratio_metals.units = ""  # dimensionless
sn_iaii_ratio_metals.limits = [-5.0, 6.0]
sn_iaii_ratio_metals.limits_halo = [-4.0, 5.0]
sn_iaii_ratio_metals.log = True


@snap_field
def sn_ia_agb_ratio_metals(sim, partType, field, args):
    """GFM_MetalsTagged: ratio of total metal mass [linear] produced in SNIa versus AGB."""
    metals_SNIa = sim.snapshotSubset(partType, "metals_SNIa", **args)
    metals_AGB = sim.snapshotSubset(partType, "metals_AGB", **args)
    return metals_SNIa / metals_AGB


sn_ia_agb_ratio_metals.label = r"[pt] Mass Ratio $\rm{Z_{SNIa} / Z_{AGB}}$"
sn_ia_agb_ratio_metals.units = ""  # dimensionless
sn_ia_agb_ratio_metals.limits = [-2.0, 2.0]
sn_ia_agb_ratio_metals.limits_halo = [-1.5, 1.5]
sn_ia_agb_ratio_metals.log = True


@snap_field(multi=True)
def numratio_(sim, partType, field, args):
    """Metal abundance number density ratio e.g. 'numratio_Si_H', relative to solar, i.e. [Si/H]."""
    from ..cosmo.cloudy import cloudyIon

    el1, el2, _ = field.split("_")

    ion = cloudyIon(sP=None)
    el1_massratio = sim.snapshotSubset(partType, "metals_" + el1, **args)
    el2_massratio = sim.snapshotSubset(partType, "metals_" + el2, **args)
    el_ratio = el1_massratio / el2_massratio

    return ion._massRatioToRelSolarNumDensRatio(el_ratio, el1, el2)


numratio_.label = lambda sim, pt, f: r"$\rm{[%s/%s]_{[pt]}}$" % (f.split("_")[1], f.split("_")[2])
numratio_.units = ""  # dimensionless
numratio_.limits = [-4.0, 4.0]
numratio_.limits_halo = [-3.0, 1.0]
numratio_.log = True


@snap_field(multi=True)
def massratio_(sim, partType, field, args):
    """Metal abundance mass ratio e.g. 'massratio_Si_H', absolute (not relative to solar)."""
    el1, el2, _ = field.split("_")

    el1_massratio = sim.snapshotSubset(partType, "metals_" + el1, **args)
    el2_massratio = sim.snapshotSubset(partType, "metals_" + el2, **args)

    return el1_massratio / el2_massratio


massratio_.label = lambda sim, pt, f: r"Mass Ratio $\rm{(%s/%s)_{[pt]}}$" % (f.split("_")[1], f.split("_")[2])
massratio_.units = ""  # dimensionless
massratio_.limits = [-5.0, 0.0]
massratio_.limits_halo = [-4.0, 1.0]
massratio_.log = True


@snap_field(aliases=["mass_z", "mass_metal", "metalmass_msun"])
def metalmass(sim, partType, field, args):
    """Total metal mass (convert GFM_Metals from fraction to mass)."""
    masses = sim.snapshotSubset(partType, "Masses", **args)
    masses *= sim.snapshotSubset(partType, "metallicity", **args)

    if "_msun" in field:
        masses = sim.units.codeMassToMsun(masses)

    return masses


metalmass.label = "Metal Mass"
metalmass.units = lambda sim, pt, f: r"$\rm{M_{sun}}$" if "_msun" in f else "code_mass"
metalmass.limits = [3.0, 7.0]
metalmass.limits_halo = [4.0, 7.0]
metalmass.log = True


@snap_field(multi=True)
def metalmass_(sim, partType, field, args):
    """Metal mass for a given species (convert GFM_Metals from fraction to mass),
    e.g. 'metalmass_O' or 'metalmass_Mg' or 'metalmass_Fe_msun'."""
    species = field.replace("metalmass_", "").replace("_msun", "").capitalize()

    masses = sim.snapshotSubset(partType, "Masses", **args)
    masses *= sim.snapshotSubset(partType, "metals_" + species, **args)

    if "_msun" in field:
        masses = sim.units.codeMassToMsun(masses)

    return masses


metalmass_.label = lambda sim, pt, f: "[pt] %s Metal Mass" % f.split("metalmass_")[1].replace("_msun", "")
metalmass_.units = lambda sim, pt, f: r"$\rm{M_{\rm sun}}$" if "_msun" in f else "code_mass"
metalmass_.limits = [2.0, 6.0]
metalmass_.limits_halo = [3.0, 6.0]
metalmass_.log = True

# -------------------- stars ----------------------------------------------------------------------


@snap_field(alias="stellar_age")
def star_age(sim, partType, field, args):
    """Age of stellar population (conversion of GFM_StellarFormationTime)."""
    birthTime = sim.snapshotSubset(partType, "birthtime", **args)
    w = np.where(birthTime == 0)  # anyways, wind particles only
    birthTime[w] = 1e-30  # avoid divide by zero
    birthRedshift = 1.0 / birthTime - 1.0

    age = sim.tage - sim.units.redshiftToAgeFlat(birthRedshift)

    return age


star_age.label = "Stellar Age"
star_age.units = r"$\rm{Gyr}$"
star_age.limits = lambda sim, pt, f: [0.0, np.clip(np.floor(sim.tage), 1, 12)]
star_age.log = False


@snap_field(alias="z_formation")
def z_form(sim, partType, field, args):
    """Formation redshift of stellar population (conversion of GFM_StellarFormationTime)."""
    birthTime = sim.snapshotSubset(partType, "birthtime", **args)

    z = 1.0 / birthTime - 1.0

    return z


z_form.label = "Stellar Formation Redshift"
z_form.units = ""  # dimensionless
z_form.limits = lambda sim, pt, f: [np.clip(np.floor(sim.redshift), 0, 5), 6.0]
z_form.log = False

# -------------------- stars (MCST) ---------------------------------------------------------------


@snap_field
def stellar_masshist(sim, partType, field, args):
    """Stellar mass histogram (binned by mass) for a given snapshot. MCST model."""
    assert 0  # needs verification
    StellarArray = sim.snapshotSubset(partType, "StellarArray", **args)

    STAR_BIN_SIZE = 4
    N_BINS = 64

    Nbin_per_element = N_BINS / STAR_BIN_SIZE
    fullblock = np.array([15], dtype="uint64")

    result = np.zeros((StellarArray.shape[0], N_BINS), dtype="uint64")

    # loop over bins
    for i in range(N_BINS):
        el_loc = int(i // Nbin_per_element)
        nibble_shift = np.uint64(STAR_BIN_SIZE * (i % Nbin_per_element))

        StellarArray_loc = np.right_shift(StellarArray[:, el_loc], nibble_shift)
        result[:, i] = np.bitwise_and(fullblock, StellarArray_loc)

        # loop over star particles
        # for j in np.arange(StellarArray.shape[0]):
        #    result[j,i] = np.bitwise_and(fullblock,np.right_shift(StellarArray[j,el_loc],nibble_shift))[0]

    return result


# -------------------- black holes ----------------------------------------------------------------


@snap_field(aliases=["bh_bollum", "bh_bollum_obscured"])
def bh_lbol(sim, partType, field, args):
    """Black hole bolometric luminosity (optionally with obscuration)."""
    bh_mass = sim.snapshotSubset(partType, "BH_Mass", **args)
    bh_mdot = sim.snapshotSubset(partType, "BH_Mdot", **args)

    return sim.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot, obscuration=("_obscured" in field))


bh_lbol.label = r"$\rm{L_{bol}}$"
bh_lbol.units = r"$\rm{erg/s}$"
bh_lbol.limits = [38.0, 46.0]
bh_lbol.limits_halo = [39.0, 46.0]
bh_lbol.log = True


@snap_field(aliases=["bh_bollum_basic", "bh_bollum_basic_obscured"])
def bh_lbol_basic(sim, partType, field, args):
    """Black hole bolometric luminosity (simple model, optionally with obscuration)."""
    bh_mass = sim.snapshotSubset(partType, "BH_Mass", **args)
    bh_mdot = sim.snapshotSubset(partType, "BH_Mdot", **args)

    return sim.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot, basic_model=True, obscuration=("_obscured" in field))


bh_lbol_basic.label = r"$\rm{L_{bol}}$"
bh_lbol_basic.units = r"$\rm{erg/s}$"
bh_lbol_basic.limits = [38.0, 46.0]
bh_lbol_basic.limits_halo = [39.0, 46.0]
bh_lbol_basic.log = True


@snap_field(aliases=["ledd", "lumedd", "edd_ratio", "bh_ledd", "eddington_lum"])
def bh_lumedd(sim, partType, field, args):
    """Black hole Eddington luminosity."""
    bh_mass = sim.snapshotSubset(partType, "BH_Mass", **args)

    return sim.units.codeBHMassToLumEdd(bh_mass)


bh_lumedd.label = r"$\rm{L_{edd}}$"
bh_lumedd.units = r"$\rm{erg/s}$"
bh_lumedd.limits = [38.0, 46.0]
bh_lumedd.limits_halo = [40.0, 46.0]
bh_lumedd.log = True


@snap_field(aliases=["eddington_ratio", "lambda_edd", "edd_ratio"])
def bh_eddratio(sim, partType, field, args):
    """Black hole bolometric luminosity (optionally with obscuration)."""
    bh_mdot = sim.snapshotSubset(partType, "BH_Mdot", **args)
    bh_mdot_edd = sim.snapshotSubset(partType, "BH_MdotEddington", **args)

    return bh_mdot / bh_mdot_edd  # = (lum_bol / lum_edd)


bh_eddratio.label = r"$\rm{\lambda_{edd} = L_{bol} / L_{edd}}$"
bh_eddratio.units = ""  # dimensionless
bh_eddratio.limits = [-10.0, 0.0]
bh_eddratio.limits_halo = [-8.0, 0.0]
bh_eddratio.log = True


@snap_field
def bh_mode(sim, partType, field, args):
    """Black hole accretion/feedback mode (0=low/kinetic, 1=high/quasar)."""
    bh_mass = sim.snapshotSubset(partType, "BH_Mass", **args)
    bh_mdot = sim.snapshotSubset(partType, "BH_Mdot", **args)
    bh_mdot_edd = sim.snapshotSubset(partType, "BH_MdotEddington", **args)
    bh_mdot_bondi = sim.snapshotSubset(partType, "BH_MdotBondi", **args)

    return sim.units.codeBHValsToFeedbackMode(bh_mass, bh_mdot, bh_mdot_bondi, bh_mdot_edd)


bh_mode.label = "BH mode (0=low, 1=high)"
bh_mode.units = ""  # dimensionless
bh_mode.limits = [0, 1]
bh_mode.limits_halo = [0, 1]
bh_mode.log = False


@snap_field(alias="bh_edot")
def bh_dedt(sim, partType, field, args):
    """Black hole feedback energy injection rate."""
    bh_mass = sim.snapshotSubset(partType, "BH_Mass", **args)
    bh_mdot = sim.snapshotSubset(partType, "BH_Mdot", **args)
    bh_mdot_edd = sim.snapshotSubset(partType, "BH_MdotEddington", **args)
    bh_mdot_bondi = sim.snapshotSubset(partType, "BH_MdotBondi", **args)
    bh_dens = sim.snapshotSubset(partType, "BH_Density", **args)

    return sim.units.codeBHMassMdotToInstantaneousEnergy(bh_mass, bh_mdot, bh_dens, bh_mdot_bondi, bh_mdot_edd)


bh_dedt.label = "BH Energy Injection Rate"
bh_dedt.units = r"$\rm{erg/s}$"
bh_dedt.limits = [30.0, 46.0]
bh_dedt.limits_halo = [36.0, 46.0]
bh_dedt.log = True

# -------------------- halo-centric fields (all particle types) -----------------------------------
# note: such fields currently require an explicit haloID or subhaloID. in the future, could
#       generalize to full snapshots, with an inverse mapping of indRange to subhaloIDs,
#       checking the case of indRange==None correctly mapping to all, then replacing
#       single halo/subhalo loads with full catalog loads, then looping over each ID, and
#       for each calling the appropriate unit function with the [sub]halo particle subset
#       and [sub]halo position. would require a decision for satellite subhalos, i.e. are
#       the properties relative to themselves, or their host halo.


@snap_field(aliases=["pos_rel_kpc", "pos_rel_rvir"])
def pos_rel(sim, partType, field, args):
    """3D (xyz) position, relative to the halo/subhalo center."""
    assert args["haloID"] is not None or args["subhaloID"] is not None

    pos = sim.snapshotSubset(partType, "pos", **args)

    if isinstance(pos, dict) and pos["count"] == 0:
        return pos  # no particles of type, empty return

    # get haloID and load halo regardless, even for non-centrals
    # take center position as subhalo center (same as group center for centrals)
    if args["subhaloID"] is None:
        halo = sim.halo(args["haloID"])
        haloPos = halo["GroupPos"]
    if args["subhaloID"] is not None:
        sub = sim.subhalo(args["subhaloID"])
        halo = sim.halo(sub["SubhaloGrNr"])
        haloPos = sub["SubhaloPos"]

    if sim.refPos is not None:
        haloPos = sim.refPos  # allow override

    # compute
    for j in range(3):
        pos[:, j] -= haloPos[j]

    sim.correctPeriodicDistVecs(pos)

    # units: pkpc, code lengths, or in terms of r200
    if "_kpc" in field:
        pos = sim.units.codeLengthToKpc(pos)
    if "_rvir" in field:
        pos /= halo["Group_R_Crit200"]

    return pos


pos_rel.label = (
    lambda sim, pt, f: "Halocentric Position" if "_rvir" not in f else r"Halocentric Position / R$_{\rm vir}$"
)
pos_rel.units = lambda sim, pt, f: r"$\rm{kpc}$" if "_kpc" in f else "" if "_rvir" in f else "code_length"
pos_rel.limits = [-1e3, 1e3]
pos_rel.limits_halo = [-1e3, 1e3]
pos_rel.log = False


@snap_field(aliases=["vrel", "halo_vrel", "halo_relvel", "relative_vel"])
def vel_rel(sim, partType, field, args):
    """3D (xyz) velocity, relative to the halo/subhalo motion."""
    vel = sim.snapshotSubset(partType, "vel", **args)

    if isinstance(vel, dict) and vel["count"] == 0:
        return vel  # no particles of type, empty return

    # get reference velocity
    if sim.isZoom and args["subhaloID"] is None and args["haloID"] is None:
        args["subhaloID"] = sim.zoomSubhaloID
        print(f"WARNING: Using {sim.zoomSubhaloID =} for zoom run to compute [{field}]!")

    if args["haloID"] is None and args["subhaloID"] is None:
        assert sim.refVel is not None
        print(f"WARNING: Using refVel in non-zoom run to compute [{field}]!")
        refVel = sim.refVel
    else:
        # take central subhalo velocity of (host) halo
        shID = sim.halo(args["haloID"])["GroupFirstSub"] if args["subhaloID"] is None else args["subhaloID"]
        firstSub = sim.subhalo(shID)
        refVel = firstSub["SubhaloVel"]

    if sim.refVel is not None:
        refVel = sim.refVel  # allow override

    return sim.units.particleRelativeVelInKmS(vel, refVel)


vel_rel.label = "[pt] Halo-Relative Velocity"
vel_rel.units = r"$\rm{km/s}$"
vel_rel.limits = [-1000, 1000]
vel_rel.limits_halo = [-300, 300]
vel_rel.log = False


@snap_field(aliases=["vrelmag", "halo_vrelmag", "relative_vmag"])
def vel_rel_mag(sim, partType, field, args):
    """Magnitude of velocity, relative to the halo/subhalo motion."""
    vel = sim.snapshotSubset(partType, "vel_rel", **args)
    return np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2)


vel_rel_mag.label = "[pt] Halo-Relative Velocity"
vel_rel_mag.units = r"$\rm{km/s}$"
vel_rel_mag.limits = [0, 1000]
vel_rel_mag.limits_halo = [0, 300]
vel_rel_mag.log = False


@snap_field(aliases=["halo_rad", "rad_r500", "rad_rvir", "halo_rad_r500", "halo_rad_rvir"])
def rad(sim, partType, field, args):
    """3D radial distance from (parent) halo center."""
    pos = sim.snapshotSubset(partType, "pos", **args)

    if isinstance(pos, dict) and pos["count"] == 0:
        return pos  # no particles of type, empty return

    # get (host) halo center position
    if sim.isZoom and args["subhaloID"] is None and args["haloID"] is None and sim.refPos is None:
        args["subhaloID"] = sim.zoomSubhaloID
        print(f"WARNING: Using {sim.zoomSubhaloID = } for zoom run to compute [{field}]!")

    if args["haloID"] is None and args["subhaloID"] is None:
        assert sim.refPos is not None
        print(f"Note: Using refPos to compute [{field}]!")
        haloPos = sim.refPos
        halo = sim.halo(sim.refSubhalo["SubhaloGrNr"])
    else:
        haloID = args["haloID"]
        if args["subhaloID"] is not None:
            haloID = sim.subhalo(args["subhaloID"])["SubhaloGrNr"]

        halo = sim.halo(haloID)
        haloPos = halo["GroupPos"]  # note: is identical to SubhaloPos of GroupFirstSub

    # compute
    rr = sim.periodicDists(haloPos, pos)

    # what kind of distance?
    if "_rvir" in field:
        rr /= halo["Group_R_Crit200"]
    if "_r500" in field:
        rr /= halo["Group_R_Crit500"]

    return rr


def _rad_label(sim, pt, f):
    if "_rvir" in f:
        return r"[pt] Radial Distance / $\rm{R_{vir}}$"
    if "_r500" in f:
        return r"[pt] Radial Distance / $\rm{R_{500}}$"
    return "[pt] Radial Distance"


def _rad_units(sim, pt, f):
    if "_rvir" in f:
        return ""
    if "_r500" in f:
        return ""
    return "code_length"


rad.label = _rad_label
rad.units = _rad_units
rad.limits = lambda sim, pt, f: [-2.5, 3.0] if ("_rvir" in f or "_r500" in f) else [0, 5.0]
rad.limits_halo = lambda sim, pt, f: [-2.5, 0.5] if ("_rvir" in f or "_r500" in f) else [0, 3.0]
rad.log = True


@snap_field(aliases=["halo_rad_kpc", "rad_kpc_linear"])
def rad_kpc(sim, partType, field, args):
    """3D radial distance from (parent) halo center in [kpc]."""
    rr = sim.snapshotSubset(partType, "halo_rad", **args)
    return sim.units.codeLengthToKpc(rr)


rad_kpc.label = "[pt] Radial Distance"
rad_kpc.units = r"$\rm{kpc}$"
rad_kpc.limits = lambda sim, pt, f: [0.0, 5.0] if "_linear" not in f else [0.0, 5000]
rad_kpc.limits_halo = lambda sim, pt, f: [0.0, 3.0] if "_linear" not in f else [0.0, 800]
rad_kpc.log = lambda sim, pt, f: True if "_linear" not in f else False


@snap_field(aliases=["dist_2dz_r200", "dist_2dz_r500"])
def dist_2dz(sim, partType, field, args):
    """2D distance (i.e. impact parameter), projecting along z-hat, from (parent) halo center."""
    pos = sim.snapshotSubset(partType, "pos", **args)
    pos = pos[:, 0:2]

    if isinstance(pos, dict) and pos["count"] == 0:
        return pos  # no particles of type, empty return

    # get (host) halo center position, or position of reference
    if sim.isZoom and args["subhaloID"] is None and args["haloID"] is None and sim.refPos is None:
        args["subhaloID"] = sim.zoomSubhaloID
        print(f"WARNING: Using {sim.zoomSubhaloID = } for zoom run to compute [{field}]!")

    if args["haloID"] is None and args["subhaloID"] is None:
        assert sim.refPos is not None
        print(f"WARNING: Using refPos in non-zoom run to compute [{field}]!")
        haloPos = sim.refPos[0:2]
        halo = sim.halo(sim.refSubhalo["SubhaloGrNr"])
    else:
        haloID = args["haloID"]
        if args["subhaloID"] is not None:
            haloID = sim.subhalo(args["subhaloID"])["SubhaloGrNr"]

        halo = sim.halo(haloID)
        haloPos = halo["GroupPos"]  # note: is identical to SubhaloPos of GroupFirstSub
        haloPos = haloPos[0:2]

    # compute
    rr = sim.periodicDists2D(haloPos, pos)

    # what kind of distance?
    if "_rvir" in field:
        rr /= halo["Group_R_Crit200"]
    if "_r500" in field:
        rr /= halo["Group_R_Crit500"]

    return rr


@snap_field(aliases=["halo_vrad", "radvel", "halo_radvel", "vrad_vvir", "halo_vrad_vvir", "halo_radvel_vvir"])
def vrad(sim, partType, field, args):
    """Radial velocity, relative to the central subhalo and its motion, including hubble correction.
    Optionally normalized by the halo virial velocity. Convention: negative = in, positive = out."""
    pos = sim.snapshotSubset(partType, "pos", **args)
    vel = sim.snapshotSubset(partType, "vel", **args)

    if isinstance(pos, dict) and pos["count"] == 0:
        return pos  # no particles of type, empty return

    # get position and velocity of reference
    if sim.isZoom and args["subhaloID"] is None and args["haloID"] is None and sim.refPos is None:
        args["subhaloID"] = sim.zoomSubhaloID
        print(f"WARNING: Using {sim.zoomSubhaloID = } for zoom run to compute [{field}]!")

    if args["haloID"] is None and args["subhaloID"] is None:
        if sim.refPos is not None and sim.refVel is not None:
            print(f"Note: Using refPos and refVel to compute [{field}]!")
            refPos = sim.refPos
            refVel = sim.refVel
        else:
            # full box, replicate (todo: can directly use e.g. 'parent_halo_GroupPos' but need
            # support for 'parent_subhalo_*' quantities taking central values for satellites)
            halo_id = sim.snapshotSubset(partType, "halo_id", **args)
            subhalo_id = sim.halos("GroupFirstSub")[halo_id]
            refPos = sim.subhalos("SubhaloPos")[subhalo_id]
            refVel = sim.subhalos("SubhaloVel")[subhalo_id]

            # particles outside of FoFs
            w = np.where(halo_id < 0)
            refPos[w] = np.nan
            refVel[w] = np.nan
    else:
        haloID = args["haloID"]
        if args["subhaloID"] is not None:  # for subhalos, take host halo
            haloID = sim.subhalo(args["subhaloID"])["SubhaloGrNr"]

        # need velocity of subhalo, take central of this halo
        shID = args["subhaloID"]
        if shID is None:
            shID = sim.halo(haloID)["GroupFirstSub"]

        firstSub = sim.subhalo(shID)
        refPos = firstSub["SubhaloPos"]
        refVel = firstSub["SubhaloVel"]

    # compute
    vv = sim.units.particleRadialVelInKmS(pos, vel, refPos, refVel)

    if "_vvir" in field:
        # normalize by halo v200
        mhalo = sim.halo(haloID)["Group_M_Crit200"]
        vv /= sim.units.codeMassToVirVel(mhalo)

    return vv


vrad.label = lambda sim, pt, f: "[pt] Radial Velocity" if "_vvir" not in f else r"[pt] Radial Velocity / $\rm{V_{200}}$"
vrad.units = lambda sim, pt, f: r"$\rm{km/s}$" if "_vvir" not in f else ""
vrad.limits = lambda sim, pt, f: [-1000, 1000] if "_vvir" not in f else [-2.0, 2.0]
vrad.limits_halo = lambda sim, pt, f: [-300, 300] if "_vvir" not in f else [-1.0, 1.0]
vrad.log = False


@snap_field(aliases=["j", "specj", "specangmom", "angmom_mag", "specj_mag", "specangmom_mag"])
def angmom(sim, partType, field, args):
    """Angular momentum, relative to the central subhalo and its motion, including hubble correction,
    either the 3-vector or the specific magnitude (if field contains '_mag')."""
    assert args["haloID"] is not None or args["subhaloID"] is not None

    pos = sim.snapshotSubset(partType, "pos", **args)
    vel = sim.snapshotSubset(partType, "vel", **args)
    mass = sim.snapshotSubset(partType, "mass", **args)

    if isinstance(pos, dict) and pos["count"] == 0:
        return pos  # no particles of type, empty return

    # reference position and velocity
    if sim.isZoom and args["subhaloID"] is None and args["haloID"] is None:
        args["subhaloID"] = sim.zoomSubhaloID
        print(f"WARNING: Using {sim.zoomSubhaloID = } for zoom run to compute [{field}]!")

    shID = args["subhaloID"]
    if shID is None:
        shID = sim.halo(args["haloID"])["GroupFirstSub"]
    firstSub = sim.subhalo(shID)

    refPos = firstSub["SubhaloPos"]
    refVel = firstSub["SubhaloVel"]

    # compute
    if "_mag" in field:
        return sim.units.particleSpecAngMomMagInKpcKmS(pos, vel, mass, refPos, refVel)

    return sim.units.particleAngMomVecInKpcKmS(pos, vel, mass, refPos, refVel)


angmom.label = lambda sim, pt, f: "[pt] Angular Momentum" if "_mag" in f else "[pt] Specific Angular Momentum"
angmom.units = lambda sim, pt, f: r"$\rm{kpc\ km/s}$" if "_mag" in f else r"$\rm{M_{sun}\ kpc\ km/s}$"
angmom.limits_halo = lambda sim, pt, f: [2.0, 6.0] if "_mag" in f else [-1e12, 1e12]
angmom.log = lambda sim, pt, f: True if "_mag" in f else False


@snap_field(alias="enclosedmass")
def menc(sim, partType, field, args):
    """Enclosed mass, i.e. total halo mass within the radial distance of each particle/cell."""
    assert args["haloID"] is not None or args["subhaloID"] is not None

    # allocate for radii and masses of all particle types
    if args["haloID"] is not None:
        lenType = sim.halo(args["haloID"])["GroupLenType"]
    else:
        lenType = sim.subhalo(args["subhaloID"])["SubhaloLenType"]

    numPartTot = np.sum(lenType[sim.ptNum(pt)] for pt in sim.partTypes)

    rad = np.zeros(numPartTot, dtype="float32")
    mass = np.zeros(numPartTot, dtype="float32")
    mask = np.zeros(numPartTot, dtype="int16")

    # load
    offset = 0
    for pt in sim.partTypes:
        numPartType = lenType[sim.ptNum(pt)]
        if numPartType == 0:
            continue

        rad[offset : offset + numPartType] = sim.snapshotSubset(pt, "rad", **args)
        mass[offset : offset + numPartType] = sim.snapshotSubset(pt, "mass", **args)

        if sim.isPartType(pt, partType):
            mask[offset : offset + numPartType] = 1
        offset += numPartType

    # sort and cumulative sum
    inds = np.argsort(rad)
    # radtype = rad[np.where(mask == 1)]
    indstype = np.argsort(rad[np.where(mask == 1)])
    mass = mass[inds]
    mask = mask[inds]
    cum_mass = np.cumsum(mass, dtype="float64")

    # extract enclosed mass for our particle type, shuffle back into original order
    mass_enc = np.zeros(indstype.size, dtype="float32")
    mass_enc[indstype] = cum_mass[np.where(mask == 1)]

    return mass_enc


menc.label = "Enclosed Mass"
menc.units = "code_mass"
menc.limits_halo = [-1.0, 4.0]
menc.log = True


@snap_field(alias="enclosedmass_msun")
def menc_msun(sim, partType, field, args):
    """Enclosed mass, in solar masses."""
    menc = sim.snapshotSubset(partType, "menc", **args)

    return sim.units.codeMassToMsun(menc)


menc_msun.label = "Enclosed Mass"
menc_msun.units = r"$\rm{M_{sun}}$"
menc_msun.limits_halo = [9.0, 14.0]
menc_msun.log = True


@snap_field(aliases=["tfreefall", "freefalltime"])
def tff(sim, partType, field, args):
    """Gravitational free-fall time."""
    menc = sim.snapshotSubset(partType, "menc", **args)
    rad = sim.snapshotSubset(partType, "rad", **args)

    enclosed_vol = 4 * np.pi * rad**3 / 3  # code units
    enclosed_meandens = menc / enclosed_vol

    return sim.units.avgEnclosedDensityToFreeFallTime(enclosed_meandens)


tff.label = "Gravitational Free-Fall Time"
tff.units = r"$\rm{Gyr}$"
tff.limits_halo = [-2.0, 1.0]
tff.log = True


@snap_field
def tcool_tff(sim, partType, field, args):
    """Ratio of gas cooling time to gravitational free-fall time."""
    tcool = sim.snapshotSubset(partType, "tcool", **args)
    tff = sim.snapshotSubset(partType, "tff", **args)

    return tcool / tff


tcool_tff.label = r"$\rm{t_{cool} / t_{ff}}$"
tcool_tff.units = ""  # dimensionless
tcool_tff.limits_halo = [-1.0, 2.0]
tcool_tff.log = True


@snap_field
def menc_vesc(sim, partType, field, args):
    """Gravitational escape velocity (based on enclosed mass)."""
    menc = sim.snapshotSubset(partType, "menc", **args)
    rad = sim.snapshotSubset(partType, "rad", **args)

    vesc = np.sqrt(2 * sim.units.G * menc / rad)  # code units
    vesc *= 1.0e5 / sim.units.UnitVelocity_in_cm_per_s  # account for non-km/s code units

    return vesc


menc_vesc.label = "Escape Velocity"
menc_vesc.units = r"$\rm{km/s}$"
menc_vesc.limits_halo = [1.0, 2.6]
menc_vesc.log = True


@snap_field
def delta_rho(sim, partType, field, args):
    """Ratio of density to local mean density, delta_rho/<rho>, based on a spherically symmetric,
    halo-centric mass density profile. This is a special case of the below."""
    from scipy.stats import binned_statistic

    from ..util.helper import logZeroNaN

    mass = sim.snapshotSubset(partType, "mass", **args)
    rad = sim.snapshotSubset(partType, "rad", **args)
    rad = logZeroNaN(rad)

    bins = np.linspace(0.0, 3.6, 19)  # log code dist, 0.2 dex bins, ~1 kpc - 3 Mpc
    totvol_bins = 4 / 3 * np.pi * ((10.0 ** bins[1:]) ** 3 - (10.0 ** bins[:-1]) ** 3)  # (ckpc/h)^3
    bin_cens = (bins[1:] + bins[:-1]) / 2

    totmass_bins, _, _ = binned_statistic(rad, mass, "sum", bins=bins)

    # interpolate mass-density to the distance of each particle/cell
    avg_rho = np.interp(rad, bin_cens, totmass_bins / totvol_bins)

    avg_rho[avg_rho == 0] = np.min(avg_rho[avg_rho > 0])  # clip to nonzero as we divide

    # return ratio
    dens = sim.snapshotSubset(partType, "dens", **args)  # will fail for stars/DM, can generalize
    ratio = (dens / avg_rho).astype("float32")

    return ratio


delta_rho.label = r"$\delta \rho / <\rho>$"
delta_rho.units = ""  # dimensionless
delta_rho.limits_halo = [-1.0, 1.0]
delta_rho.log = True


@snap_field(multi=True)
def delta_(sim, partType, field, args):
    """Ratio of any particle/cell property to its local average, based on a spherically symmetric,
    halo-centric radial profile."""
    from scipy.interpolate import interp1d
    from scipy.stats import binned_statistic

    from ..util.helper import logZeroNaN

    propName = field.split("_")[1]

    prop = sim.snapshotSubset(partType, propName, **args)
    rad = sim.snapshotSubset(partType, "rad", **args)
    rad = logZeroNaN(rad)

    bins = np.linspace(0.0, 3.6, 19)  # log code dist, 0.2 dex bins, ~1 kpc - 3 Mpc
    bin_cens = (bins[1:] + bins[:-1]) / 2

    avg_prop_binned, _, _ = binned_statistic(rad, prop, "mean", bins=bins)

    # if any bins were empty, avg_prop_binned has nan entries
    w_nan = np.where(np.isnan(avg_prop_binned))
    if len(w_nan[0]):
        # linear extrapolate/interpolate (in log quantity) to fill them
        w_finite = np.where(~np.isnan(avg_prop_binned))
        f_interp = interp1d(
            bin_cens[w_finite],
            np.log10(avg_prop_binned[w_finite]),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        avg_prop_binned[w_nan] = 10.0 ** f_interp(bin_cens[w_nan])

    # interpolate mass-density to the distance of each particle/cell
    avg_prop = np.interp(rad, bin_cens, avg_prop_binned)

    avg_prop[avg_prop == 0] = np.min(avg_prop[avg_prop > 0])  # clip to nonzero as we divide

    if np.count_nonzero(avg_prop < 0):
        print("WARNING: avg_prop has negative entries, unexpected.")

    ratio = (prop / avg_prop).astype("float32")

    return ratio


delta_.label = lambda sim, pt, f: r"$\rm{\delta %s / <%s>}$" % (f.split("delta_")[1], f.split("delta_")[1])
delta_.units = ""  # dimensionless
delta_.limits = [-1.0, 1.0]
delta_.log = True

# -------------------- halo-centric metadata ------------------------------------------------------


@snap_field(aliases=["subid", "subhaloid"])
def subhalo_id(sim, partType, field, args):
    """Parent subhalo ID, per particle/cell."""
    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])
        indRange[1] += 1  # inverseMapPartIndicesToSubhaloIDs() is numpy convention i.e. excludes last index

    # make explicit list of indices
    if indRange is not None:
        inds = np.arange(indRange[0], indRange[1] + 1)
    else:
        inds = np.arange(0, sim.numPart[sim.ptNum(partType)])

    # inverse map back to parent [sub]halo ID
    return sim.inverseMapPartIndicesToSubhaloIDs(inds, partType)


subhalo_id.label = "Subhalo ID"
subhalo_id.units = ""  # dimensionless
subhalo_id.limits = [0, 1e7]
subhalo_id.log = True


@snap_field(alias="haloid")
def halo_id(sim, partType, field, args):
    """Parent halo ID, per particle/cell."""
    indRange = args["indRange"]
    if args["haloID"] is not None or args["subhaloID"] is not None:
        indRange = _haloOrSubhaloIndRange(sim, partType, haloID=args["haloID"], subhaloID=args["subhaloID"])
        indRange[1] += 1

    # make explicit list of indices
    if indRange is not None:
        inds = np.arange(indRange[0], indRange[1])
    else:
        inds = np.arange(0, sim.numPart[sim.ptNum(partType)])

    # inverse map back to parent [sub]halo ID
    return sim.inverseMapPartIndicesToHaloIDs(inds, partType)


halo_id.label = "Halo ID"
halo_id.units = ""  # dimensionless
halo_id.limits = [0, 1e7]
halo_id.log = True


@snap_field
def sat_member(sim, partType, field, args):
    """True (1) if particle/cell belongs to a satellite subhalo, False (0) otherwise (central/inner fuzz/outer fuzz)."""
    subhaloIDs = sim.snapshotSubset(partType, "subhalo_id", **args)
    haloIDs = sim.snapshotSubset(partType, "halo_id", **args)
    GroupFirstSub = sim.halos("GroupFirstSub")[haloIDs]

    data = np.zeros(GroupFirstSub.size, dtype="int8")
    w = np.where((subhaloIDs != GroupFirstSub) & (subhaloIDs != -1))
    data[w] = 1

    return data


sat_member.label = "Satellite Member"
sat_member.units = ""  # dimensionless
sat_member.limits = [0, 1]
sat_member.log = False


@snap_field(multi=True)
def parent_subhalo_(sim, partType, field, args):
    """Any property of the parent subhalo, per particle/cell."""
    parentField = field.split("parent_subhalo_")[1]
    parentIDs = sim.snapshotSubset(partType, "subhalo_id", **args)
    parentProp = sim.subhalos(parentField)

    data = parentProp[parentIDs].astype("float32")

    # set nan for any particles/cells not in a parent
    w = np.where(parentIDs == -1)
    data[w] = np.nan

    return data


parent_subhalo_.label = lambda sim, pt, f: "Parent Subhalo [%s]" % f.split("parent_subhalo_")[1]
parent_subhalo_.units = ""  # TODO: call simSubhaloQuantity in lambda and retrieve
parent_subhalo_.limits = [0, 1e7]  # TODO: as above
parent_subhalo_.log = True


@snap_field(multi=True)
def parent_halo_(sim, partType, field, args):
    """Any property of the parent halo, per particle/cell."""
    parentField = field.split("parent_halo_")[1]
    parentIDs = sim.snapshotSubset(partType, "halo_id", **args)
    parentProp = sim.halos(parentField)

    data = parentProp[parentIDs].astype("float32")

    # set nan for any particles/cells not in a parent
    w = np.where(parentIDs == -1)
    data[w] = np.nan

    return data


parent_halo_.label = lambda sim, pt, f: "Parent Halo [%s]" % f.split("parent_halo_")[1]
parent_halo_.units = ""  # TODO: call simSubhaloQuantity in lambda and retrieve
parent_halo_.limits = [0, 1e7]  # TODO: as above
parent_halo_.log = True
