"""
Definitions of custom catalog fields.
"""
import h5py
import numpy as np
#from functools import partial
#from getpass import getuser
#from os.path import isfile, isdir, getsize
#from os import mkdir

from .groupcat import catalog_field
from ..util.helper import logZeroNaN
from ..cosmo.color import gfmBands, vegaMagCorrections
from ..plot.quantities import bandMagRange

# -------------------- subhalos: meta -------------------------------------------------------------

@catalog_field(aliases=['subhalo_index','id','index'])
def subhalo_id(sim, partType, field, args):
    """ Subhalo ID/index. """
    assert '_log' not in field

    return np.arange(sim.numSubhalos)

subhalo_id.label = 'Subhalo ID'
subhalo_id.units = '' # dimensionless
subhalo_id.limits = [0, 7]
subhalo_id.log = True

@catalog_field(aliases=['cen_flag','is_cen','is_central'])
def central_flag(sim, partType, field, args):
    """ Subhalo central flag (1 if central, 0 if not). """
    assert '_log' not in field

    GroupFirstSub = sim.halos('GroupFirstSub')
    GroupFirstSub = GroupFirstSub[np.where(GroupFirstSub >= 0)]

    # satellites given zero
    flag = np.zeros(sim.numSubhalos, dtype='int16')
    flag[GroupFirstSub] = 1

    return flag

central_flag.label = 'Central Flag (0=no, 1=yes)'
central_flag.units = '' # dimensionless
central_flag.limits = [0, 1]
central_flag.log = False

# -------------------- subhalos: halo-related properties ------------------------------------------

def mhalo_lim(sim,pt,f):
    """ Limits for halo masses. """
    lim = [11.0, 14.0]
    if sim.boxSize > 200000: lim = [11.0, 15.0]
    if sim.boxSize < 50000: lim = [10.5, 13.5]
    return lim

def _mhalo_load(sim, partType, field, args):
    """ Helper for the halo mass fields below. """
    if '200' in field:
        haloField = 'Group_M_Crit200'
    if '500' in field:
        haloField = 'Group_M_Crit500'
    if 'vir' in field:
        haloField = 'Group_M_TopHat200' # misleading name

    halos = sim.groupCat(fieldsHalos=[haloField,'GroupFirstSub'])
    GrNr = sim.subhalos('SubhaloGrNr')

    mhalo = halos[haloField][GrNr]

    if '_code' not in field:
        mhalo = sim.units.codeMassToMsun(mhalo)

    if '_parent' not in field:
        # satellites given nan (by default)
        mask = np.zeros(GrNr.size, dtype='int16')
        mask[halos['GroupFirstSub']] = 1

        mhalo[mask == 0] = np.nan

    return mhalo

@catalog_field(aliases=['mhalo_200_code','mhalo_200_parent','m200','m200c'])
def mhalo_200(sim, partType, field, args):
    """ Parent halo total mass (:math:`\\rm{M_{200,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default), 
    unless '_parent' is specified in the field name, in which case satellites are given 
    the same host halo mass as their central."""
    return _mhalo_load(sim, partType, field, args)

mhalo_200.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{200c%s}}$' % (',parent' if '_parent' in f else '')
mhalo_200.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_200.limits = mhalo_lim
mhalo_200.log = True

@catalog_field(aliases=['mhalo_500_code','mhalo_500_parent','m500','m500c'])
def mhalo_500(sim, partType, field, args):
    """ Parent halo total mass (:math:`\\rm{M_{500,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _mhalo_load(sim, partType, field, args)

mhalo_500.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{500c%s}}$' % (',parent' if '_parent' in f else '')
mhalo_500.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_500.limits = mhalo_lim
mhalo_500.log = True

@catalog_field(aliases=['mhalo_vir_code','mhalo_vir_parent'])
def mhalo_vir(sim, partType, field, args):
    """ Parent halo total mass (:math:`\\rm{M_{vir}}`). Defined by :math:`\\rm{M_{\\Delta}}` 
    where :math:`\\Delta` is the overdensity based on spherical tophat collapse.
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _mhalo_load(sim, partType, field, args)

mhalo_vir.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{vir%s}}$' % (',parent' if '_parent' in f else '')
mhalo_vir.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_vir.limits = mhalo_lim
mhalo_vir.log = True

def _rhalo_load(sim, partType, field, args):
    """ Helper for the halo radii loads below. """
    rField = 'Group_R_Crit200' if '200' in field else 'Group_R_Crit500'

    halos = sim.groupCat(fieldsHalos=[rField,'GroupFirstSub'])
    GrNr = sim.subhalos('SubhaloGrNr')

    rad = halos[rField][GrNr]

    if '_code' not in field:
        rad = sim.units.codeLengthToKpc(rad)

    # satellites given nan
    if '_parent' not in field:
        mask = np.zeros(GrNr.size, dtype='int16')
        mask[halos['GroupFirstSub']] = 1
        wSat = np.where(mask == 0)
        rad[wSat] = np.nan

    return rad

@catalog_field(aliases=['rhalo_200_code','rhalo_200_parent','rhalo','r200','rhalo_200','rvir'])
def rhalo_200(sim, partType, field, args):
    """ Parent halo virial radius (:math:`\\rm{R_{200,crit}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _rhalo_load(sim, partType, field, args)

rhalo_200.label = lambda sim,pt,f: r'$\rm{R_{halo,200c%s}}$' % (',parent' if '_parent' in f else '')
rhalo_200.units = lambda sim,pt,f: r'$\rm{kpc}$' if '_code' not in f else 'code_length'
rhalo_200.limits = [1.0, 3.0]
rhalo_200.log = True

@catalog_field(aliases=['rhalo_500_code','rhalo_500_parent','r500','rhalo_500'])
def rhalo_500(sim, partType, field, args):
    """ Parent halo :math:`\\rm{R_{500,crit}}` radius.
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    return _rhalo_load(sim, partType, field, args)

rhalo_500.label = lambda sim,pt,f: r'$\rm{R_{halo,500c%s}}$' % (',parent' if '_parent' in f else '')
rhalo_500.units = lambda sim,pt,f: r'$\rm{kpc}$' if '_code' not in f else 'code_length'
rhalo_500.limits = [1.0, 3.0]
rhalo_500.log = True

@catalog_field(aliases=['v200','vvir'])
def vhalo(sim, partType, field, args):
    """ Parent halo virial velocity (:math:`\\rm{V_{200}}`).
    Only defined for centrals: satellites are assigned a value of nan (excluded by default)."""
    gc = sim.groupCat(fieldsSubhalos=['mhalo_200_code','rhalo_200_code'])
    return sim.units.codeM200R200ToV200InKmS(gc['mhalo_200_code'], gc['rhalo_200_code'])

vhalo.label = r'$\rm{v_{200,halo}}$'
vhalo.units = r'$\rm{km/s}$'
vhalo.limits = [0, 200]
vhalo.log = False

@catalog_field(aliases=['halo_nsubs','nsubs','numsubs'])
def halo_numsubs(sim, partType, field, args):
    """ Total number of subhalos in parent dark matter halo. A value of one implies only a 
    central subhalo exists, while a value of two indicates a central and one satellite, 
    and so on. Only defined for centrals: satellites are assigned a value of nan (excluded by default). """
    haloField = 'GroupNsubs'
    halos = sim.groupCat(fieldsHalos=[haloField,'GroupFirstSub'])
    GrNr = sim.subhalos('SubhaloGrNr')

    num = halos[haloField][GrNr].astype('float32') # int dtype

    # satellites given nan
    mask = np.zeros(GrNr.size, dtype='int16')
    mask[halos['GroupFirstSub']] = 1
    wSat = np.where(mask == 0)
    num[wSat] = np.nan

    return num

halo_numsubs.label = r'$\rm{N_{sub}}$ in Halo'
halo_numsubs.units = '' # dimensionless
halo_numsubs.limits = [0.0, 2.0]
halo_numsubs.log = True

@catalog_field(alias='tvir')
def virtemp(sim, partType, fields, args):
    """ Virial temperature of the parent halo (satellites have NaN). """
    mass = sim.groupCat(fieldsSubhalos=['mhalo_200_code'])
    tvir = sim.units.codeMassToVirTemp(mass)
    return tvir.astype('float32')

virtemp.label = r'$\rm{T_{vir}}$'
virtemp.units = r'$\rm{K}$'
virtemp.limits = [4.0, 7.0]
virtemp.log = True

@catalog_field(aliases=['rdist_code','rdist','rdist_rvir','distance_code','distance_rvir'])
def distance(sim, partType, field, args):
    """ Radial distance of satellites to center of parent halo (centrals have zero). """
    gc = sim.groupCat(fieldsHalos=['GroupPos','Group_R_Crit200'], 
                      fieldsSubhalos=['SubhaloPos','SubhaloGrNr'])
    
    parInds = gc['subhalos']['SubhaloGrNr']
    dist = sim.periodicDists(gc['halos']['GroupPos'][parInds,:], gc['subhalos']['SubhaloPos'])

    if '_rvir' not in field and '_code' not in field:
        dist = sim.units.codeLengthToKpc(dist)

    if '_rvir' in field:
        with np.errstate(invalid='ignore'):
            dist /= gc['halos']['Group_R_Crit200'][parInds]

    return dist

distance.label = lambda sim,pt,f: r'Radial Distance' if '_rvir' not in f else r'R / R$_{\\rm vir,host}$'
distance.units = lambda sim,pt,f: 'code_length' if '_code' in f else ('' if '_rvir' in f else r'$\rm{kpc}$')
distance.limits = lambda sim,pt,f: [0.0, 2.0] if '_rvir' in f else [1.0, 3.5]
distance.log = lambda sim,pt,f: True if '_rvir' not in f else False # linear for rvir normalized

# -------------------- subhalos: masses -----------------------------------------------------------

@catalog_field
def mhalo_subfind(sim, partType, field, args):
    """ Parent dark matter (sub)halo total mass, defined by the gravitationally bound mass as determined by Subfind. """
    mhalo = sim.subhalos('SubhaloMass')
    return sim.units.codeMassToMsun(mhalo)

mhalo_subfind.label = r'Subhalo Mass $\rm{M_{grav}}$'
mhalo_subfind.units = r'$\rm{M_{sun}}$'
mhalo_subfind.limits = mhalo_lim
mhalo_subfind.log = True

@catalog_field
def mstar1(sim, partType, field, args):
    """ Galaxy stellar mass, measured within the stellar half mass radius. """
    mass = sim.subhalos('SubhaloMassInHalfRadType')[:,sim.ptNum('stars')]
    return sim.units.codeMassToMsun(mass)

mstar1.label = r'$\rm{M_{\star}}$' # (<r_{\star,1/2})
mstar1.units = r'$\rm{M_{sun}}$'
mstar1.limits = lambda sim,pt,f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar1.log = True

@catalog_field
def mstar2(sim, partType, field, args):
    """ Galaxy stellar mass, measured within *twice* the stellar half mass radius. """
    mass = sim.subhalos('SubhaloMassInRadType')[:,sim.ptNum('stars')]
    return sim.units.codeMassToMsun(mass)

mstar2.label = r'$\rm{M_{\star}}$' # (<2r_{\star,1/2})
mstar2.units = r'$\rm{M_{sun}}$'
mstar2.limits = lambda sim,pt,f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar2.log = True

@catalog_field
def mstar_tot(sim, partType, field, args):
    """ Galaxy stellar mass, total subhalo/subfind value. """
    mass = sim.subhalos('SubhaloMassType')[:,sim.ptNum('stars')]
    return sim.units.codeMassToMsun(mass)

mstar_tot.label = r'$\rm{M_{\star}}$' # (subfind)
mstar_tot.units = r'$\rm{M_{sun}}$'
mstar_tot.limits = lambda sim,pt,f: [9.0, 11.5] if sim.boxSize > 50000 else [8.0, 12.0]
mstar_tot.log = True

@catalog_field
def mgas1(sim, partType, field, args):
    """ Galaxy gas mass (all phases), measured within the stellar half mass radius. """
    mass = sim.subhalos('SubhaloMassInHalfRadType')[:,sim.ptNum('gas')]
    return sim.units.codeMassToMsun(mass)

mgas1.label = r'$\rm{M_{gas}}$' # (<r_{\star,1/2})
mgas1.units = r'$\rm{M_{sun}}$'
mgas1.limits = lambda sim,pt,f: [8.0, 11.0] if sim.boxSize > 50000 else [7.0, 10.5]
mgas1.log = True

@catalog_field
def mgas2(sim, partType, field, args):
    """ Galaxy gas mass (all phases), measured within *twice* the stellar half mass radius. """
    mass = sim.subhalos('SubhaloMassInRadType')[:,sim.ptNum('gas')]
    return sim.units.codeMassToMsun(mass)

mgas2.label = r'$\rm{M_{gas}}$' # (<2r_{\star,1/2})
mgas2.units = r'$\rm{M_{sun}}$'
mgas2.limits = lambda sim,pt,f: [8.0, 11.0] if sim.boxSize > 50000 else [7.0, 10.5]
mgas2.log = True

@catalog_field(alias='mstar_30kpc')
def mstar_30pkpc(sim, partType, field, args):
    """ Galaxy stellar mass, measured within a fixed 3D aperture of 30 physical kpc. """
    acField = 'Subhalo_Mass_30pkpc_Stars'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mstar_30pkpc.label = r'$\rm{M_{\star, <30kpc}}$'
mstar_30pkpc.units = r'$\rm{M_{sun}}$'
mstar_30pkpc.limits = lambda sim,pt,f: [9.0, 11.0] if sim.boxSize > 50000 else [8.0, 11.5]
mstar_30pkpc.log = True
mstar_30pkpc.auxcat = True

@catalog_field
def mstar_5pkpc(sim, partType, field, args):
    """ Galaxy stellar mass, measured within a fixed 3D aperture of 5 physical kpc. """
    acField = 'Subhalo_Mass_5pkpc_Stars'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mstar_5pkpc.label = r'$\rm{M_{\star, <5kpc}}$'
mstar_5pkpc.units = r'$\rm{M_{sun}}$'
mstar_5pkpc.limits = [8.0, 12.0]
mstar_5pkpc.log = True
mstar_5pkpc.auxcat = True

@catalog_field
def mgas_5pkpc(sim, partType, field, args):
    """ Galaxy gas mass, measured within a fixed 3D aperture of 5 physical kpc. """
    acField = 'Subhalo_Mass_5pkpc_Gas'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mgas_5pkpc.label = r'$\rm{M_{gas, <5kpc}}$'
mgas_5pkpc.units = r'$\rm{M_{sun}}$'
mgas_5pkpc.limits = [7.5, 10.5]
mgas_5pkpc.log = True
mgas_5pkpc.auxcat = True

@catalog_field
def mdm_5pkpc(sim, partType, field, args):
    """ Galaxy dark matter mass, measured within a fixed 3D aperture of 5 physical kpc. """
    acField = 'Subhalo_Mass_5pkpc_DM'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mdm_5pkpc.label = r'$\rm{M_{DM, <5kpc}}$'
mdm_5pkpc.units = r'$\rm{M_{sun}}$'
mdm_5pkpc.limits = [8.0, 12.0]
mdm_5pkpc.log = True
mdm_5pkpc.auxcat = True

@catalog_field
def mtot_5pkpc(sim, partType, field, args):
    """ Galaxy total mass (gas + stars + DM + BHs), measured within a fixed 3D aperture of 5 physical kpc. """
    mass = np.zeros(sim.numSubhalos, dtype='float32')
    for pt in ['Gas','Stars','DM','BH']:
        acField = 'Subhalo_Mass_5pkpc_%s' % pt
        mass += sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mtot_5pkpc.label = r'$\rm{M_{total, <5kpc}}$'
mtot_5pkpc.units = r'$\rm{M_{sun}}$'
mtot_5pkpc.limits = [9.0, 12.0]
mtot_5pkpc.log = True
mtot_5pkpc.auxcat = True

@catalog_field
def mstar_mtot_ratio_5pkpc(sim, partType, field, args):
    """ Ratio of galaxy stellar mass, to total mass, both measured within a 3D aperture of 5 physical kpc. """
    mstar = sim.subhalos('mstar_5pkpc')
    mtot = sim.subhalos('mtot_5pkpc')

    return (mstar / mtot)

mstar_mtot_ratio_5pkpc.label = r'$\rm{M_{\star} / M_{total} (<5kpc)}$'
mstar_mtot_ratio_5pkpc.units = '' # dimensionless
mstar_mtot_ratio_5pkpc.limits = [0.0, 0.8]
mstar_mtot_ratio_5pkpc.log = False
mstar_mtot_ratio_5pkpc.auxcat = True

@catalog_field
def mstar2_mhalo200_ratio(sim, partType, field, args):
    """ Galaxy stellar mass to halo mass ratio, the former defined as 
    within twice the stellar half mass radius, the latter as M_200_Crit. """
    mstar = sim.subhalos('SubhaloMassInRadType')[:,sim.ptNum('stars')]
    mhalo = sim.subhalos('mhalo_200_code')

    w = np.where(mhalo == 0) # low mass halos with no central
    mhalo[w] = np.nan

    with np.errstate(invalid='ignore'):
        ratio = mstar / mhalo

    return ratio

mstar2_mhalo200_ratio.label = r'$\rm{M_{\star, <2r_{\star}} / M_{halo,200c}}$'
mstar2_mhalo200_ratio.units = '' # dimensionless
mstar2_mhalo200_ratio.limits = [-3.0, -1.0]
mstar2_mhalo200_ratio.log = True

@catalog_field(alias='mstar_mhalo_ratio')
def mstar30pkpc_mhalo200_ratio(sim, partType, field, args):
    """ Galaxy stellar mass to halo mass ratio, the former measured within a 
    fixed 3D aperture of 30 physical kpc, the latter taken as M_200_Crit. """
    acField = 'Subhalo_Mass_30pkpc_Stars'
    mstar = sim.auxCat(acField)[acField]
    mhalo = sim.subhalos('mhalo_200_code')

    w = np.where(mhalo == 0) # low mass halos with no central
    mhalo[w] = np.nan

    with np.errstate(invalid='ignore'):
        ratio = mstar / mhalo

    return ratio

mstar30pkpc_mhalo200_ratio.label = r'$\rm{M_{\star, <30pkpc} / M_{halo,200c}}$'
mstar30pkpc_mhalo200_ratio.units = '' # dimensionless
mstar30pkpc_mhalo200_ratio.limits = [-3.0, -1.0]
mstar30pkpc_mhalo200_ratio.log = True
mstar30pkpc_mhalo200_ratio.auxcat = True

@catalog_field
def mstar_r500(sim, partType, field, args):
    """ Subhalo stellar mass, measured within :math:`\\rm{R_{500c}}`. """
    acField = 'Subhalo_Mass_r500_Stars_FoF'
    mass = sim.auxCat(acField, expandPartial=True)[acField]
    return sim.units.codeMassToMsun(mass)

mstar_r500.label = r'$\rm{M_{\star, <r500}}$'
mstar_r500.units = r'$\rm{M_{sun}}$'
mstar_r500.limits = [8.0, 12.0]
mstar_r500.log = True
mstar_r500.auxcat = True

@catalog_field
def mgas_r500(sim, partType, field, args):
    """ Subhalo gas mass (all phases), measured within :math:`\\rm{R_{500c}}`. """
    acField = 'Subhalo_Mass_r500_Gas_FoF'
    mass = sim.auxCatSplit(acField, expandPartial=True)[acField]
    return sim.units.codeMassToMsun(mass)

mgas_r500.label = r'$\rm{M_{gas, <r500}}$'
mgas_r500.units = r'$\rm{M_{sun}}$'
mgas_r500.limits = [8.0, 12.0]
mgas_r500.log = True
mgas_r500.auxcat = True

@catalog_field
def mhi(sim, partType, field, args):
    """ Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within 
    the entire subhalo (all gravitationally bound gas). """
    acField = 'Subhalo_Mass_HI'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mhi.label = r'$\rm{M_{HI, grav}}$'
mhi.units = r'$\rm{M_{sun}}$'
mhi.limits = lambda sim,pt,f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi.log = True
mhi.auxcat = True

@catalog_field
def mhi2(sim, partType, field, args):
    """ Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within 
    twice the stellar half mass radius. """
    acField = 'Subhalo_Mass_2rstars_HI'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mhi2.label = r'$\rm{M_{HI, <2r_{\star}}}$'
mhi2.units = r'$\rm{M_{sun}}$'
mhi2.limits = lambda sim,pt,f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi2.log = True
mhi2.auxcat = True

@catalog_field
def mhi_30pkpc(sim, partType, field, args):
    """ Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within 
    a fixed 3D aperture of 30 physical kpc. """
    acField = 'Subhalo_Mass_30pkpc_HI'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mhi_30pkpc.label = r'$\rm{M_{HI, <30kpc}}$'
mhi_30pkpc.units = r'$\rm{M_{sun}}$'
mhi_30pkpc.limits = lambda sim,pt,f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi_30pkpc.log = True
mhi_30pkpc.auxcat = True

@catalog_field
def mhi_halo(sim, partType, field, args):
    """ Halo-scale atomic HI gas mass (BR06 molecular H2 model), measured within each FoF. """
    acField = 'Subhalo_Mass_FoF_HI'
    mass = sim.auxCat(acField)[acField]
    return sim.units.codeMassToMsun(mass)

mhi_halo.label = r'$\rm{M_{HI, halo}}$'
mhi_halo.units = r'$\rm{M_{sun}}$'
mhi_halo.limits = lambda sim,pt,f: [8.0, 11.5] if sim.boxSize > 50000 else [7.0, 10.5]
mhi_halo.log = True
mhi_halo.auxcat = True

# -------------------- star formation rates -------------------------------------------------------

@catalog_field(alias='sfr_30pkpc')
def sfr(sim, partType, field, args):
    """ Galaxy star formation rate (instantaneous, within 30pkpc aperture). """
    acField = 'Subhalo_GasSFR_30pkpc'
    sfr = sim.auxCat(acField)[acField] # units correct
    return sfr

sfr.label = r'$\rm{SFR_{<30kpc}}$'
sfr.units = r'$\rm{M_{sun} yr^{-1}}$'
sfr.limits = [-2.5, 1.0]
sfr.log = True
sfr.auxcat = True

@catalog_field
def sfr2(sim, partType, field, args):
    """ Galaxy star formation rate (instantaneous, within twice the stellar half mass radius). """
    return sim.subhalos('SubhaloSFRinRad') # units correct

sfr.label = r'$\rm{SFR_{<2r_{\star}}}$'
sfr.units = r'$\rm{M_{sun} yr^{-1}}$'
sfr.limits = [-2.5, 1.0]
sfr.log = True

@catalog_field
def ssfr(sim, partType, field, args):
    """ Galaxy specific star formation rate [1/yr] (sSFR, instantaneous, both SFR and M* within 2rhalfstars). """
    sfr = sim.subhalos('SubhaloSFRinRad')
    mstar = sim.subhalos('SubhaloMassInRadType')[:,sim.ptNum('stars')]

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr

ssfr.label = r'$\rm{sSFR_{<2r_{\star}}}$'
ssfr.units = r'$\rm{yr^{-1}}$'
ssfr.limits = [-12.0, -8.0]
ssfr.log = True

@catalog_field
def ssfr_30pkpc(sim, partType, field, args):
    """ Galaxy specific star formation rate [1/yr] (sSFR, instantaneous, SFR within 2rhalfstars, M* within 30kpc). """
    sfr = sim.subhalos('SubhaloSFRinRad')
    mstar = sim.subhalos('mstar_30pkpc')

    # set mstar==0 subhalos to nan
    w = np.where(mstar == 0.0)[0]
    if len(w):
        mstar[w] = 1.0
        sfr[w] = np.nan

    ssfr = sfr / mstar
    return ssfr

ssfr_30pkpc.label = r'$\rm{sSFR}$'
ssfr_30pkpc.units = r'$\rm{yr^{-1}}$'
ssfr_30pkpc.limits = [-12.0, -8.0]
ssfr_30pkpc.log = True

@catalog_field
def ssfr_gyr(sim, partType, field, args):
    """ Galaxy specific star formation rate [1/Gyr] (sSFR, instantaneous, both SFR and M* within 2rhalfstars). """
    return sim.subhalos('ssfr') * 1e9

ssfr_gyr.label = r'$\rm{sSFR_{<2r_{\star}}}$'
ssfr_gyr.units = r'$\rm{Gyr^{-1}}$'
ssfr_gyr.limits = [-3.0, 1.0]
ssfr_gyr.log = True

@catalog_field
def ssfr_30pkpc_gyr(sim, partType, field, args):
    """ Galaxy specific star formation rate [1/Gyr] (sSFR, instantaneous, SFR within 2rhalfstars, M* within 30kpc). """
    return sim.subhalos('ssfr') * 1e9

ssfr_30pkpc_gyr.label = r'$\rm{sSFR}$'
ssfr_30pkpc_gyr.units = r'$\rm{Gyr^{-1}}$'
ssfr_30pkpc_gyr.limits = [-3.0, 1.0]
ssfr_30pkpc_gyr.log = True

# -------------------- general subhalo properties  ------------------------------------------------

@catalog_field(aliases=['vc','vmax'])
def vcirc(sim, partType, field, args):
    """ Maximum value of the spherically-averaged 3D circular velocity curve 
    (i.e. galaxy circular velocity). """
    return sim.subhalos('SubhaloVmax') # units correct

vcirc.label = r'$\rm{V_{circ}}$'
vcirc.units = r'$\rm{km/s}$'
vcirc.limits = [1.8, 2.8]
vcirc.log = True

@catalog_field(alias='vmag')
def velmag(sim, partType, field, args):
    """ The magnitude of the current velocity of the subhalo through the box, 
    in the simulation reference frame. """
    vel = sim.subhalos('SubhaloVel')
    vel = sim.units.subhaloCodeVelocityToKms(vel)
    vmag = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)

    return vmag

velmag.label = r'$\rm{|V|_{subhalo}}$'
velmag.units = r'$\rm{km/s}$'
velmag.limits = [1.5, 3.5]
velmag.log = True

@catalog_field(alias='smag')
def spinmag(sim, partType, field, args):
    """ The magnitude of the subhalo spin vector, computed as the mass weighted 
    sum of all subhalo particles/cells. """
    spin = sim.subhalos('SubhaloSpin')
    spin = sim.units.subhaloSpinToKpcKms(spin)
    smag = np.sqrt(spin[:,0]**2 + spin[:,1]**2 + spin[:,2]**2)

    return smag

spinmag.label = r'$\rm{|S|_{subhalo}}$'
spinmag.units = r'$\rm{kpc km/s}$'
spinmag.limits = [2.0, 4.0]
spinmag.log = True

@catalog_field(aliases=['rhalf_stars','size_stars_code','rhalf_stars_code'])
def size_stars(sim, partType, field, args):
    """ Stellar half mass radius. """
    radtype = sim.subhalos('SubhaloHalfmassRadType')
    rad = radtype[:,sim.ptNum('stars')]

    if '_code' not in field:
        rad = sim.units.codeLengthToKpc(rad)

    return rad

size_stars.label = r'r$_{\\rm 1/2,\star}$'
size_stars.units = lambda sim,pt,f: r'$\rm{kpc}$' if '_code' not in f else 'code_length'
size_stars.limits = lambda sim,pt,f: [0.0, 1.8] if sim.redshift < 1 else [-0.4, 1.4]
size_stars.log = True

# -------------------- subhalo photometrics  ------------------------------------------------------

@catalog_field(aliases=['m_u','m_b'])
def m_v(sim, partType, field, args):
    """ V-band magnitude (StellarPhotometrics from snapshot). No dust. """
    assert '_log' not in field
    bandName = field.split('_')[1].upper()

    vals = sim.subhalos('SubhaloStellarPhotometrics')
    mags = vals[:,gfmBands[bandName]]

    # fix zero values
    w = np.where(mags > 1e10)
    mags[w] = np.nan

    # Vega corrections
    if bandName in vegaMagCorrections:
        mags += vegaMagCorrections[bandName]

    return mags

m_v.label = lambda sim,pt,f: r'M$_{\\rm %s}$' % f.split("_")[1].upper()
m_v.units = 'abs AB mag'
m_v.limits = [-24, -16]
m_v.log = False

@catalog_field(aliases=['color_vb'])
def color_uv(sim, partType, field, args):
    """ Integrated photometric/broadband galaxy colors, from snapshot. No dust. """
    assert '_log' not in field
    bandNames = field.split('color_')[1].upper()

    mags_0 = sim.subhalos('M_' + bandNames[0])
    mags_1 = sim.subhalos('M_' + bandNames[1])

    colors = mags_0 - mags_1

    return colors

color_uv.label = lambda sim,pt,f: r'(%s-%s) color' % (f.split('color_')[1][0].upper(), f.split('color_')[1][1].upper())
color_uv.units = 'mag'
color_uv.limits = lambda sim,pt,f: bandMagRange(f.split('color_')[1], sim=sim)
color_uv.log = False
