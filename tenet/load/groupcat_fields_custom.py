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
    if '_200' in field:
        haloField = 'Group_M_Crit200'
    if '_500' in field:
        haloField = 'Group_M_Crit500'
    if '_vir' in field:
        haloField = 'Group_M_Tophat200' # misleading name

    gc = sim.groupCat(fieldsHalos=[haloField,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])

    mhalo = gc['halos'][haloField][gc['subhalos']]

    if '_code' not in field:
        mhalo = sim.units.codeMassToMsun(mhalo)

    if '_parent' not in field:
        # satellites given nan (by default)
        mask = np.zeros(gc['subhalos'].size, dtype='int16')
        mask[gc['halos']['GroupFirstSub']] = 1

        mhalo[mask == 0] = np.nan

    return mhalo

@catalog_field(aliases=['mhalo_200_code','mhalo_200_parent'])
def mhalo_200(sim, partType, field, args):
    """ Parent halo mass (:math:`\\rm{M_{200,crit}}`), with satellites assigned nan, 
    unless '_parent' is specified in the field name, in which case satellites are given 
    the same host halo mass as their central."""
    return _mhalo_load(sim, partType, field, args)

mhalo_200.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{200,crit%s}}$' % (',parent' if '_parent' in f else '')
mhalo_200.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_200.limits = mhalo_lim
mhalo_200.log = True

@catalog_field(aliases=['mhalo_500_code','mhalo_500_parent'])
def mhalo_500(sim, partType, field, args):
    """ Parent halo mass (:math:`\\rm{M_{500,crit}}`), satellites assigned nan by default. """
    return _mhalo_load(sim, partType, field, args)

mhalo_500.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{500,crit%s}}$' % (',parent' if '_parent' in f else '')
mhalo_500.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_500.limits = mhalo_lim
mhalo_500.log = True

@catalog_field(aliases=['mhalo_vir_code','mhalo_vir_parent'])
def mhalo_vir(sim, partType, field, args):
    """ Parent halo mass (:math:`\\rm{M_{vir}}`), satellites assigned nan by default. """
    return _mhalo_load(sim, partType, field, args)

mhalo_vir.label = lambda sim,pt,f: r'Halo Mass $\rm{M_{vir%s}}$' % (',parent' if '_parent' in f else '')
mhalo_vir.units = lambda sim,pt,f: r'$\rm{M_{sun}}$' if '_code' not in f else 'code_mass'
mhalo_vir.limits = mhalo_lim
mhalo_vir.log = True
