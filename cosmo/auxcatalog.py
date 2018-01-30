"""
cosmo/auxcatalog.py
  Cosmological simulations - auxiliary catalog for additional derived galaxy/halo properties.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
from functools import partial

from cosmo.util import periodicDistsSq, snapNumToRedshift, subhaloIDListToBoundingPartIndices, \
  inverseMapPartIndicesToSubhaloIDs, inverseMapPartIndicesToHaloIDs, correctPeriodicDistVecs, \
  cenSatSubhaloIndices
from util.helper import logZeroMin, logZeroNaN, logZeroSafe
from util.helper import pSplit as pSplitArr, pSplitRange, numPartToChunkLoadSize

""" Relatively 'hard-coded' analysis decisions that can be changed. For reference, they are attached 
    as metadata attributes in the auxCat file. """

# Subhalo_*: parameters for computations done over each Subfind subhalo

# Box_*: parameters for whole box (halo-independent) computations
boxGridSizeHI     = 1.5 # code units, e.g. ckpc/h
boxGridSizeMetals = 5.0 # code units, e.g. ckpc/h

# todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
# eliminate the entire specialized ufunc logic herein
userCustomFields = ['Krot','radvel']

def fofRadialSumType(sP, pSplit, ptProperty, rad, method='B', ptType='all'):
    """ Compute total/sum of a particle property (e.g. mass) for those particles enclosed within one of 
        the SO radii already computed and available in the group catalog (input as a string). Methods A 
        and B restrict this calculation to FoF particles only, whereas method C does a full particle 
        search over the entire box in order to compute the total/sum for each FoF halo. If ptType='all', 
        then do for all types (dm,gas,stars), otherwise just for the single specified type.

      Method A: do individual halo loads per halo, one loop over all halos.
      Method B: do a full snapshot load per type, then halo loop and slice per FoF, to cut down on I/O ops. 
      Method C: per type: full snapshot load, spherical aperture search per FoF (brute-force global).
      Method D: per type: full snapshot load, construct octtree, spherical aperture search per FoF (global).
    """

    # config
    if ptType == 'all':
        # (0=dm, 1=gas+wind, 2=stars-wind), instead of making a half-empty Nx6 array
        ptSaveTypes = {'dm':0, 'gas':1, 'stars':2}
        ptLoadTypes = {'dm':sP.ptNum('dm'), 'gas':sP.ptNum('gas'), 'stars':sP.ptNum('stars')}
        assert ptProperty == 'Masses' # not yet implemented any other fields which apply to all partTypes
    else:
        ptSaveTypes = {ptType:0}
        ptLoadTypes = {ptType:sP.ptNum(ptType)}

    desc   = "Mass by type enclosed within a radius of "+rad+" (only FoF particles included). "
    desc  += "Type indices: " + " ".join([t+'='+str(i) for t,i in ptSaveTypes.iteritems()]) + "."
    select = "All FoF halos."

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupPos','GroupLen','GroupLenType',rad])
    gc['halos']['GroupOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']

    h = cosmo.load.snapshotHeader(sP)

    nGroupsTot = gc['header']['Ngroups_Total']
    haloIDsTodo = np.arange(nGroupsTot, dtype='int32')

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = subhaloIDListToBoundingPartIndices(sP, haloIDsTodo, groups=True)

    if pSplit is not None:
        ptSplit = ptType if ptType != 'all' else 'gas'

        # subdivide the global [variable ptType!] particle set, then map this back into a division of 
        # group IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange( indRange[ptSplit], pSplit[1], pSplit[0] )

        invGroups = inverseMapPartIndicesToHaloIDs(sP, gasSplit, ptSplit, debug=True)

        if pSplit[0] == pSplit[1] - 1:
            invGroups[1] = nGroupsTot
        else:
            assert invGroups[1] != -1

        haloIDsTodo = np.arange( invGroups[0], invGroups[1] )
        indRange = subhaloIDListToBoundingPartIndices(sP, haloIDsTodo, groups=True)

    nHalosDo = len(haloIDsTodo)

    # info
    print(' ' + desc)
    print(' Total # Halos: %d, processing [%d] halos...' % (nGroupsTot,nHalosDo))

    # allocate return, NaN indicates not computed
    r = np.zeros( (nHalosDo, len(ptSaveTypes.keys())), dtype='float32' )
    r.fill(np.nan)

    # square radii, and use sq distance function
    gc['halos'][rad] = gc['halos'][rad] * gc['halos'][rad]

    if method == 'A':
        # loop over all halos
        for i, haloID in enumerate(haloIDsTodo):
            if i % int(nHalosDo/50) == 0 and i <= nHalosDo:
                print(' %4.1f%%' % (float(i+1)*100.0/nHalosDo))

            # For each type:
            #   1. Load pos (DM), pos/mass (gas), pos/mass/sftime (stars) for this FoF.
            #   2. Calculate periodic distances, (DM: count num within rad, sum massTable*num)
            #      gas/stars: sum mass of those within rad (gas = gas+wind, stars=real stars only)

            # DM
            if 'dm' in ptLoadTypes:
                dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], haloID=haloID, sq=False)

                if dm['count']:
                    rDM = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], dm['Coordinates'], sP )
                    wDM = np.where( rDM <= gc['halos'][rad][haloID] )

                    r[i, ptSaveTypes['dm']] = len(wDM[0]) * h['MassTable'][ptLoadTypes['dm']]

            # GAS
            if 'gas' in ptLoadTypes:
                gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos',ptProperty], haloID=haloID)
                assert gas[ptProperty].ndim == 1

                if gas['count']:
                    rGas = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], gas['Coordinates'], sP )
                    wGas = np.where( rGas <= gc['halos'][rad][haloID] )

                    r[i, ptSaveTypes['gas']] = np.sum( gas[ptProperty][wGas] )

            # STARS
            if 'stars' in ptLoadTypes:
                stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','sftime',ptProperty], haloID=haloID)
                assert stars[ptProperty].ndim == 1

                if stars['count']:
                    rStars = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], stars['Coordinates'], sP )
                    wWind  = np.where( (rStars <= gc['halos'][rad][haloID]) & (stars['GFM_StellarFormationTime'] < 0.0) )
                    wStars = np.where( (rStars <= gc['halos'][rad][haloID]) & (stars['GFM_StellarFormationTime'] >= 0.0) )

                    r[i, ptSaveTypes['gas']] += np.sum( stars[ptProperty][wWind] )
                    r[i, ptSaveTypes['stars']] = np.sum( stars[ptProperty][wStars] )

    if method == 'B':
        # (A): DARK MATTER
        if 'dm' in ptLoadTypes:
            print(' [DM]')
            dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], sq=False, indRange=indRange['dm'])

            if ptProperty == 'Masses':
                dm[ptProperty] = np.zeros( dm['count'], dtype='float32' ) + h['MassTable'][ptLoadTypes['dm']]
            else:
                dm[ptProperty] = cosmo.load.snapshotSubset(sP, partType='dm', fields=ptProperty, haloSubset=True)

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for dm local to this FoF
                i0 = gc['halos']['GroupOffsetType'][haloID,ptLoadTypes['dm']] - indRange['dm'][0]
                i1 = i0 + gc['halos']['GroupLenType'][haloID,ptLoadTypes['dm']]

                assert i0 >= 0 and i1 <= (indRange['dm'][1]-indRange['dm'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], dm['Coordinates'][i0:i1,:], sP )
                ww = np.where( rr <= gc['halos'][rad][haloID] )

                r[i, ptSaveTypes['dm']] = np.sum( dm[ptProperty][i0:i1][ww] )
            del dm

        # (B): GAS
        if 'gas' in ptLoadTypes:
            print(' [GAS]')
            gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos'], sq=False, indRange=indRange['gas'])
            gas[ptProperty] = cosmo.load.snapshotSubset(sP, partType='gas', fields=ptProperty, indRange=indRange['gas'])
            assert gas[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for gas local to this FoF
                i0 = gc['halos']['GroupOffsetType'][haloID,ptLoadTypes['gas']] - indRange['gas'][0]
                i1 = i0 + gc['halos']['GroupLenType'][haloID,ptLoadTypes['gas']]

                assert i0 >= 0 and i1 <= (indRange['gas'][1]-indRange['gas'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], gas['Coordinates'][i0:i1,:], sP )
                ww = np.where( rr <= gc['halos'][rad][haloID] )

                r[i, ptSaveTypes['gas']] = np.sum( gas[ptProperty][i0:i1][ww] )
            del gas

        # (C): STARS
        if 'stars' in ptLoadTypes:
            print(' [STARS]')
            stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','sftime'], sq=False, indRange=indRange['stars'])
            stars[ptProperty] = cosmo.load.snapshotSubset(sP, partType='stars', fields=ptProperty, indRange=indRange['stars'])
            assert stars[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for stars local to this FoF
                i0 = gc['halos']['GroupOffsetType'][haloID,ptLoadTypes['stars']] - indRange['stars'][0]
                i1 = i0 + gc['halos']['GroupLenType'][haloID,ptLoadTypes['stars']]

                assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = periodicDistsSq( gc['halos']['GroupPos'][haloID,:], stars['Coordinates'][i0:i1,:], sP )
                wWind  = np.where( (rr <= gc['halos'][rad][haloID]) & (stars['GFM_StellarFormationTime'][i0:i1] < 0.0) )
                wStars = np.where( (rr <= gc['halos'][rad][haloID]) & (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) )

                r[i, ptSaveTypes['gas']] += np.sum( stars[ptProperty][i0:i1][wWind] )
                r[i, ptSaveTypes['stars']] = np.sum( stars[ptProperty][i0:i1][wStars] )

    if method == 'C':
        # proceed with loads as in B and simply do rr calculation against all particles
        raise Exception('Not implemented.')

    if method == 'D':
        # use our numba tree implementation for 3d ball searches (todo)
        raise Exception('Not implemented.')

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'ptProperty'  : ptProperty.encode('ascii'),
             'subhaloIDs'  : haloIDsTodo}

    r = np.squeeze(r)

    return r, attrs

def _radialRestriction(sP, nSubsTot, rad):
    """ Handle an input 'rad' specification and return the min/max/2d/3d details to apply. """
    radRestrictIn2D = False
    radSqMin = np.zeros( nSubsTot, dtype='float32' ) # leave at zero unless modified below

    if isinstance(rad, float):
        # constant scalar, convert [pkpc] -> [ckpc/h] (code units) at this redshift
        rad_pkpc = sP.units.physicalKpcToCodeLength(rad)
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += rad_pkpc * rad_pkpc
    elif rad is None:
        # no radial restriction (all particles in subhalo)
        radSqMax = np.zeros( nSubsTot, dtype='float32' )
        radSqMax += sP.boxSize**2.0
    elif rad == 'p10':
        # load group m200_crit values
        gcLoad = cosmo.load.groupCat(sP, fieldsHalos=['Group_M_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentM200 = gcLoad['halos'][gcLoad['subhalos']]

        # r_cut = 27.3 kpc/h * (M200crit / (10^15 Msun/h))^0.29 from Puchwein+ (2010) Eqn 1
        r_cut = 27.3 * (parentM200/1e5)**(0.29) / sP.HubbleParam
        radSqMax = r_cut * r_cut
    elif rad == '30h':
        # hybrid, minimum of [constant scalar 30 pkpc] and [the usual, 2rhalf,stars]
        rad_pkpc = sP.units.physicalKpcToCodeLength(30.0)

        gcLoad = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * gcLoad['subhalos'][:,sP.ptNum('stars')]

        ww = np.where(twiceStellarRHalf > rad_pkpc)
        twiceStellarRHalf[ww] = rad_pkpc
        radSqMax = twiceStellarRHalf**2.0
    elif rad == 'r015_1rvir_halo':
        # classic 'halo' definition, 0.15rvir < r < 1.0rvir (meaningless for non-centrals)
        gcLoad = cosmo.load.groupCat(sP, fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
        radSqMin = (0.15 * parentR200)**2
    elif rad == 'r200crit':
        # within the virial radius (r200,crit definition) (centrals only)
        gcLoad = cosmo.load.groupCat(sP, fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
    elif rad == '2rhalfstars':
        # classic Illustris galaxy definition, r < 2*r_{1/2,mass,stars}
        gcLoad = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * gcLoad['subhalos'][:,sP.ptNum('stars')]

        radSqMax = twiceStellarRHalf**2
    elif rad == '1rhalfstars':
        # inner galaxy definition, r < 1*r_{1/2,mass,stars}
        gcLoad = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 1.0 * gcLoad['subhalos'][:,sP.ptNum('stars')]

        radSqMax = twiceStellarRHalf**2
    elif rad == 'sdss_fiber':
        # SDSS fiber is 3" diameter, convert to physical radius at this redshift for all z>0
        # for z=0.0 snapshots only, for this purpose we fake the angular diameter distance at z=0.1
        fiber_z = sP.redshift if sP.redshift > 0.0 else 0.1
        fiber_arcsec = 3.0 # note: 2.0 for BOSS, 3.0 for legacy SDSS
        fiber_diameter = sP.units.arcsecToAngSizeKpcAtRedshift(fiber_arcsec, z=fiber_z)
        print(' Set SDSS fiber diameter [%.2f pkpc] at redshift [z = %.2f]. NOTE: 2D restriction!' % \
            (fiber_diameter,fiber_z))

        # convert [pkpc] -> [ckpc/h] (code units) at this redshift
        fiber_diameter = sP.units.physicalKpcToCodeLength(fiber_diameter)

        radSqMax = np.zeros( nSubsTot, dtype='float32' )
        radSqMax += (fiber_diameter / 2.0)**2.0
        radRestrictIn2D = True
    elif rad == 'sdss_fiber_4pkpc':
        # keep old 4pkpc 'fiber radius' approximation but with 2D
        rad_pkpc = sP.units.physicalKpcToCodeLength(4.0)

        radSqMax = np.zeros( nSubsTot, dtype='float32' )
        radSqMax += rad_pkpc * rad_pkpc
        radRestrictIn2D = True

    assert radSqMax.size == nSubsTot
    assert radSqMin.size == nSubsTot

    return radRestrictIn2D, radSqMin, radSqMax

def subhaloRadialReduction(sP, pSplit, ptType, ptProperty, op, rad, 
                           ptRestriction=None, weighting=None, scope='subfind', minStellarMass=None):
    """ Compute a reduction operation (either total/sum or weighted mean) of a particle property (e.g. mass) 
        for those particles of a given type enclosed within a fixed radius (input as a scalar, in physical 
        kpc, or as a string specifying a particular model for a variable cut radius). 
        Restricted to subhalo particles only if scope=='subfind' (default), or FoF particles if scope=='fof'.
        If scope=='global', currently a full non-chunked snapshot load and brute-force distance 
        computations to all particles for each subhalo.
        ptRestriction applies a further cut to which particles are included (None, 'sfrgt0', 'sfreq0').
        if minStellarMass is not None, then only process subhalos with mstar_30pkpc_log above this value.
    """
    assert op in ['sum','mean','max','ufunc']
    assert scope in ['subfind','fof','global']
    if op == 'ufunc': assert ptProperty in userCustomFields

    # determine ptRestriction
    if ptType == 'stars':
        assert ptRestriction is None # otherwise we have a collision
        ptRestriction = 'real_stars'

    # config
    ptLoadType = sP.ptNum(ptType)

    desc   = "Quantity [%s] enclosed within a radius of [%s] for [%s]." % (ptProperty,rad,ptType)
    if ptRestriction is not None:
        desc += " (restriction = %s). " % ptRestriction
    if weighting is not None:
        desc += " (weighting = %s). " % weighting
    if scope == 'subfind': desc  +=" (only subhalo particles included). "
    if scope == 'fof': desc  +=" (all parent FoF particles included). "
    if scope == 'global': desc  +=" (all global particles included). "
    select = "All Subhalos."

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['subhalos']['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']
    nSubsTot = gc['header']['Nsubgroups_Total']

    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    if scope == 'fof':
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        GroupLenType = cosmo.load.groupCat(sP, fieldsHalos=['GroupLenType'])['halos']
        GroupOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        SubhaloGrNr = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloGrNr'])['subhalos']

        gc['subhalos']['SubhaloLenType'] = GroupLenType[SubhaloGrNr,:]
        gc['subhalos']['SubhaloOffsetType'] = GroupOffsetType[SubhaloGrNr,:]

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if pSplit is not None and scope != 'global':
        # subdivide the global [variable ptType!] particle set, then map this back into a division of 
        # subhalo IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange( indRange[ptType], pSplit[1], pSplit[0] )

        invSubs = inverseMapPartIndicesToSubhaloIDs(sP, gasSplit, ptType, debug=True, flagFuzz=False)

        if pSplit[0] == pSplit[1] - 1:
            invSubs[1] = nSubsTot
        else:
            assert invSubs[1] != -1

        subhaloIDsTodo = np.arange( invSubs[0], invSubs[1] )
        indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    indRange = indRange[ptType] # choose index range for the requested particle type

    if scope == 'global':
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = cosmo.load.snapshotHeader(sP)
        indRange = [0, h['NumPart'][sP.ptNum(ptType)]-1]
        i0 = 0 # never changes
        i1 = indRange[1] # never changes

    # stellar mass select
    if minStellarMass is not None:
        masses = cosmo.load.groupCat(sP, fieldsSubhalos=['mstar_30pkpc_log'])
        masses = masses[subhaloIDsTodo]
        wSelect = np.where( masses >= minStellarMass )

        subhaloIDsTodo = subhaloIDsTodo[wSelect]
        select += ' (Only with stellar mass >= %.2f)' % minStellarMass

    nSubsDo = len(subhaloIDsTodo)

    # info
    print(' ' + desc)
    print(' Total # Subhalos: %d, processing [%d] subhalos...' % (nSubsTot,nSubsDo))

    # determine radial restriction for each subhalo
    radRestrictIn2D, radSqMin, radSqMax = _radialRestriction(sP, nSubsTot, rad)

    if radRestrictIn2D:
        Nside = 'z-axis'
        print(' Requested: radRestrictIn2D! Using hard-coded projection direction of [%s]!' % Nside)

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = []

    if rad is not None:
        fieldsLoad.append('pos')
    if ptRestriction == 'real_stars':
        fieldsLoad.append('sftime')
    if ptRestriction in ['sfrgt0','sfreq0']:
        fieldsLoad.append('sfr')

    if ptProperty == 'Krot':
        if 'pos' not in fieldsLoad: fieldsLoad.append('pos')
        fieldsLoad.append('vel')
        fieldsLoad.append('mass')
    if ptProperty == 'radvel':
        if 'pos' not in fieldsLoad: fieldsLoad.append('pos')
        if 'vel' not in fieldsLoad: fieldsLoad.append('vel')
        gc['subhalos']['SubhaloVel'] = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloVel'], sq=True)

    particles = {}
    if len(fieldsLoad):
        particles = cosmo.load.snapshotSubset(sP, partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

    if op != 'ufunc':
        # todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
        # eliminate the entire specialized ufunc logic herein
        particles[ptProperty] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=[ptProperty], indRange=indRange)

    if 'count' not in particles:
        particles['count'] = particles[ particles.keys()[0] ].shape[0]

    # allocate, NaN indicates not computed except for mass where 0 will do
    if op == 'ufunc': 
        if ptProperty == 'Krot':
            r = np.zeros( (nSubsDo,4), dtype='float32' )
        else:
            r = np.zeros( nSubsDo, dtype='float32' )
    else:
        if particles[ptProperty].ndim == 1:
            r = np.zeros( nSubsDo, dtype='float32' )
        else:
            r = np.zeros( (nSubsDo,particles[ptProperty].shape[1]), dtype='float32' )

    if op not in ['sum']:
        r.fill(np.nan) # set NaN value for subhalos with e.g. no particles for op=mean

    # load weights
    if weighting is None:
        particles['weights'] = np.zeros( particles['count'], dtype='float32' )
        particles['weights'] += 1.0 # uniform
    else:
        assert op not in ['sum'] # meaningless

        if 'bandLum' in weighting:
            # prepare sps interpolator
            from cosmo.stellarPop import sps
            pop = sps(sP, 'padova07', 'chabrier', 'cf00')

            # load additional fields, snapshot wide
            fieldsLoadMag = ['initialmass','metallicity']
            magsLoad = cosmo.load.snapshotSubset(sP, partType=ptType, fields=fieldsLoadMag, indRange=indRange)

            # request magnitudes in this band
            band = weighting.split("-")[1]
            mags = pop.mags_code_units(sP, band, particles['GFM_StellarFormationTime'], 
                                     magsLoad['GFM_Metallicity'], 
                                     magsLoad['GFM_InitialMass'], retFullSize=True)

            # use the (linear) luminosity in this band as the weight
            particles['weights'] = np.power(10.0, -0.4 * mags)

        else:
            # use a particle quantity as weights (e.g. 'mass', 'volume', 'O VI mass')
            particles['weights'] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=weighting, indRange=indRange)

    assert particles['weights'].ndim == 1 and particles['weights'].size == particles['count']

    # loop over subhalos
    printFac = 100.0 if (sP.res > 512 or scope == 'global') else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo:
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

        # slice starting/ending indices for stars local to this FoF
        if scope != 'global':
            i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
            i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,ptLoadType]

        assert i0 >= 0 and i1 <= (indRange[1]-indRange[0]+1)

        if i1 == i0:
            continue # zero length of this type

        # use squared radii and sq distance function
        validMask = np.ones( i1-i0, dtype=np.bool )

        if rad is not None:

            if not radRestrictIn2D:
                # apply in 3D
                rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                      particles['Coordinates'][i0:i1,:], sP )
            else:
                # apply in 2D projection, limited support for now, just Nside='z-axis'
                # otherwise, for any more complex projection, need to apply it here, and anyways 
                # for nProj>1, this validMask selection logic becomes projection dependent, so 
                # need to move it inside the range(nProj) loop, which is definitely doable
                assert Nside == 'z-axis'
                p_inds = [0,1] # x,y
                pt_2d = gc['subhalos']['SubhaloPos'][subhaloID,:]
                pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                vecs_2d = np.zeros( (i1-i0, 2), dtype=particles['Coordinates'].dtype )
                vecs_2d[:,0] = particles['Coordinates'][i0:i1,p_inds[0]]
                vecs_2d[:,1] = particles['Coordinates'][i0:i1,p_inds[1]]

                rr = periodicDistsSq( pt_2d, vecs_2d, sP ) # handles 2D

            validMask &= (rr <= radSqMax[subhaloID])
            validMask &= (rr >= radSqMin[subhaloID])

        if ptRestriction == 'real_stars':
            validMask &= (particles['GFM_StellarFormationTime'][i0:i1] >= 0.0)
        if ptRestriction == 'sfrgt0':
            validMask &= (particles['StarFormationRate'][i0:i1] > 0.0)
        if ptRestriction == 'sfreq0':
            validMask &= (particles['StarFormationRate'][i0:i1] == 0.0)

        wValid = np.where(validMask)

        if len(wValid[0]) == 0:
            continue # zero length of particles satisfying radial cut and restriction

        # user function reduction operations
        if op == 'ufunc':
            # ufunc: kappa rot
            if ptProperty == 'Krot':
                # minimum two star particles
                if len(wValid[0]) < 2:
                    continue

                stars_pos  = np.squeeze( particles['Coordinates'][i0:i1,:][wValid,:] )
                stars_vel  = np.squeeze( particles['Velocities'][i0:i1,:][wValid,:] )
                stars_mass = particles['Masses'][i0:i1][wValid].reshape( (len(wValid[0]),1) )

                # velocity of stellar CoM
                sub_stellarMass = stars_mass.sum()
                sub_stellarCoM_vel = np.sum(stars_mass * stars_vel, axis=0) / sub_stellarMass

                # positions relative to most bound star, velocities relative to stellar CoM vel
                for j in range(3):
                    stars_pos[:,j] -= stars_pos[0,j]
                    stars_vel[:,j] -= sub_stellarCoM_vel[j]

                correctPeriodicDistVecs( stars_pos, sP )
                stars_pos = sP.units.codeLengthToKpc( stars_pos ) # kpc
                stars_vel = sP.units.particleCodeVelocityToKms( stars_vel ) # km/s
                stars_rad_sq = stars_pos[:,0]**2.0 + stars_pos[:,1]**2.0 + stars_pos[:,2]**2.0

                # total kinetic energy
                sub_K = 0.5 * np.sum(stars_mass * stars_vel**2.0)

                # specific stellar angular momentum
                stars_J = stars_mass * np.cross(stars_pos, stars_vel, axis=1)
                sub_stellarJ = np.sum(stars_J, axis=0)
                sub_stellarJ_mag = np.linalg.norm(sub_stellarJ)
                sub_stellarJ /= sub_stellarJ_mag # to unit vector
                stars_Jz_i = np.dot(stars_J, sub_stellarJ)

                # kinetic energy in rot (exclude first star with zero radius)
                stars_R_i = np.sqrt(stars_rad_sq - np.dot(stars_pos,sub_stellarJ)**2.0)

                stars_mass = stars_mass.reshape( stars_mass.size )
                sub_Krot = 0.5 * np.sum( (stars_Jz_i[1:]/stars_R_i[1:])**2.0 / stars_mass[1:])

                # restricted to those stars with the same rotation orientation as the mean
                w = np.where( (stars_Jz_i > 0.0) & (stars_R_i > 0.0) )
                sub_Krot_oriented = np.nan
                if len(w[0]):
                    sub_Krot_oriented = 0.5 * np.sum( (stars_Jz_i[w]/stars_R_i[w])**2.0 / stars_mass[w])

                # mass fraction of stars with counter-rotation
                w = np.where( (stars_Jz_i < 0.0) )
                mass_frac_counter = stars_mass[w].sum() / sub_stellarMass

                r[i,0] = sub_Krot / sub_K                   # \kappa_{star, rot}
                r[i,1] = sub_Krot_oriented / sub_K          # \kappa_{star, rot oriented}
                r[i,2] = mass_frac_counter                  # M_{star,counter} / M_{star,total}
                r[i,3] = sub_stellarJ_mag / sub_stellarMass # j_star [kpc km/s]

            # ufunc: radial velocity
            if ptProperty == 'radvel':
                gas_pos  = np.squeeze( particles['Coordinates'][i0:i1,:][wValid,:] )
                gas_vel  = np.squeeze( particles['Velocities'][i0:i1,:][wValid,:] )
                gas_weights = np.squeeze( particles['weights'][i0:i1][wValid] )

                haloPos = gc['subhalos']['SubhaloPos'][subhaloID,:]
                haloVel = gc['subhalos']['SubhaloVel'][subhaloID,:]

                vrad = sP.units.particleRadialVelInKmS(gas_pos, gas_vel, haloPos, haloVel)

                r[i] = np.average(vrad, weights=gas_weights)

            # ufunc processed and value stored, skip to next subhalo
            continue

        # standard reduction operation
        if particles[ptProperty].ndim == 1:
            # scalar
            loc_val = particles[ptProperty][i0:i1][wValid]
            loc_wt  = particles['weights'][i0:i1][wValid]

            if op == 'sum': r[i] = np.sum( loc_val )
            if op == 'max': r[i] = np.max( loc_val )
            if op == 'mean':
                if loc_wt.sum() == 0.0: loc_wt = np.zeros( loc_val.size, dtype='float32' ) + 1.0 # if all zero weights
                r[i] = np.average( loc_val , weights=loc_wt )
        else:
            # vector (e.g. pos, vel, Bfield)
            for j in range(particles[ptProperty].shape[1]):
                loc_val = particles[ptProperty][i0:i1,j][wValid]
                loc_wt  = particles['weights'][i0:i1][wValid]

                if op == 'sum': r[i,j] = np.sum( loc_val )
                if op == 'max': r[i,j] = np.max( loc_val )
                if op == 'mean':
                    if loc_wt.sum() == 0.0: loc_wt = np.zeros( loc_val.size, dtype='float32' ) + 1.0 # if all zero weights
                    r[i,j] = np.average( loc_val , weights=loc_wt )

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'ptProperty'  : ptProperty.encode('ascii'),
             'rad'         : str(rad).encode('ascii'),
             'weighting'   : str(weighting).encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    return r, attrs

def _pSplitBounds(sP, pSplit, Nside, indivStarMags):
    """ For a given pSplit = [thisTaskNum,totNumOfTasks], determine an efficient work split and 
    return the required processing for this task, in the form of the list of subhaloIDs to 
    process and the global snapshot index range required in load to cover these subhalos. """
    nSubsTot = cosmo.load.groupCatHeader(sP)['Nsubgroups_Total']
    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if isinstance(Nside, (int,long)) and Nside > 1:
        # special case: just do a few special case subhalos at high Nside for demonstration
        assert sP.res == 1820 and sP.run == 'tng' and sP.snap == 99 and pSplit is None
        #gcDemo = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'])
        #massesDemo = sP.units.codeMassToLogMsun( gcDemo['halos']['Group_M_Crit200'] )
        #ww = np.where( (massesDemo >= 11.9) & (massesDemo < 12.1) ) # ww[0]= [597, 620, 621...]
        #ww = np.where( (massesDemo >= 13.4) & (massesDemo < 13.5) ) # ww[0]= [34, 52, ...]
        # two massive + three MW-mass halos, SubhaloSFR = [0.2, 5.2, 1.7, 5.0, 1.1] Msun/yr
        subhaloIDsTodo = [172649,208781,412332,415496,415628] # gc['halos']['GroupFirstSub'][inds]

        indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if pSplit is not None:
        if 0:
            # split up subhaloIDs in round-robin scheme (equal number of massive/centrals per job)
            # works perfectly for balance, but retains global load of all haloSubset particles
            modSplit = subhaloIDsTodo % pSplit[1]
            subhaloIDsTodo = np.where(modSplit == pSplit[0])[0]

        if 0:
            # do contiguous subhalo ID division and reduce global haloSubset load 
            # to the particle sets which cover the subhalo subset of this pSplit, but the issue is 
            # that early tasks take all the large halos and all the particles, very imbalanced
            subhaloIDsTodo = pSplitArr( subhaloIDsTodo, pSplit[1], pSplit[0] )

            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

        if 1:
            # subdivide the global gas particle set, then map this back into a division of 
            # subhalo IDs which will be better work-load balanced among tasks
            gasSplit = pSplitRange( indRange['gas'], pSplit[1], pSplit[0] )

            invSubs = inverseMapPartIndicesToSubhaloIDs(sP, gasSplit, 'gas', debug=True, flagFuzz=False)

            if pSplit[0] == pSplit[1] - 1:
                invSubs[1] = nSubsTot
            
            assert invSubs[1] != -1

            if invSubs[1] == invSubs[0]:
                # split is actually zero size, this is ok
                return [], {'gas':[0,1],'stars':[0,1]}

            subhaloIDsTodo = np.arange( invSubs[0], invSubs[1] )
            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if indivStarMags:
        # make subhalo-strict bounding index range and compute number of PT4 particles we will do
        if invSubs[0] > 0:
            # except for first pSplit, move coverage to include the last subhalo of the previous 
            # split, then increment the indRange[0] by the length of that subhalo. in this way we 
            # avoid any gaps for full Pt4 coverage
            subhaloIDsTodo_extended = np.arange( invSubs[0]-1, invSubs[1] )

            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo_extended, strictSubhalos=True)

            lastPrevSub = cosmo.load.groupCatSingle(sP, subhaloID=invSubs[0]-1)
            indRange['stars'][0] += lastPrevSub['SubhaloLenType'][ sP.ptNum('stars') ]
        else:
            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo, strictSubhalos=True)

    return subhaloIDsTodo, indRange

def _findHalfLightRadius(rad,mags):
    """ Helper function, linearly interpolate in rr (squared radii) to find the half light 
    radius, given the magnitudes mags[i] corresponding to each star particle at rad[i]. 
    Will give 3D or 2D radii exactly if rad is input 3D or 2D. """
    assert rad.size == mags.size

    # convert individual mags to luminosities
    lums = np.power(10.0, -0.4 * mags)
    totalLum = np.nansum( lums )

    sort_inds = np.argsort(rad)

    rad = rad[sort_inds]
    lums = lums[sort_inds]

    # cumulative sum luminosities in radial-distance order
    w = np.where(~np.isfinite(lums)) # wind particles have mags==nan -> lums==nan
    lums[w] = 0.0

    lums_cum = np.cumsum( lums )

    # locate radius where sum equals half of total (half-light radius)
    w = np.where(lums_cum >= 0.5*totalLum)[0]
    w1 = np.min(w)

    # linear interpolation in linear(rad) and linear(lum), find radius where lums_cum = totalLum/2
    if w1 == 0:
        # half of total luminosity could be within the radius of the first star
        r1 = lums_cum[w1]
        halfLightRad = (0.5*totalLum - 0.0)/(r1-0.0) * (rad[w1]-0.0) + 0.0

        assert (halfLightRad >= 0.0 and halfLightRad <= rad[w1]) or np.isnan(halfLightRad)
    else:
        # more generally valid case
        w0 = w1 - 1
        assert w0 >= 0 and w1 < lums.size
        
        r0 = lums_cum[w0]
        r1 = lums_cum[w1]
        halfLightRad = (0.5*totalLum - r0)/(r1-r0) * (rad[w1]-rad[w0]) + rad[w0]

        assert halfLightRad >= rad[w0] and halfLightRad <= rad[w1]

    return halfLightRad

def subhaloStellarPhot(sP, pSplit, iso=None, imf=None, dust=None, Nside=1, rad=None, modelH=True, 
                       sizes=False, indivStarMags=False, fullSubhaloSpectra=False):
    """ Compute the total band-magnitudes (or instead half-light radii if sizes==True), per subhalo, 
    under the given assumption of an iso(chrone) model, imf model, dust model, and radial restrction. 
    If using a dust model, include multiple projection directions per subhalo. If indivStarMags==True, 
    then save the magnitudes for every Pt4 (wind->NaN) in all subhalos. If fullSubhaloSpectra==True, 
    then save a full spectrum vs wavelength for every subhalo. """
    from cosmo.stellarPop import sps
    from healpy.pixelfunc import nside2npix, pix2vec
    from cosmo.hydrogen import hydrogenMass
    from vis.common import rotationMatrixFromVec, rotateCoordinateArray, momentOfInertiaTensor, \
      rotationMatricesFromInertiaTensor, meanAngMomVector

    # mutually exclusive options, at most one can be enabled
    assert sum([sizes,indivStarMags,np.clip(fullSubhaloSpectra,0,1)]) in [0,1]

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust)

    # which bands? for now, to change, just recompute from scratch
    bands = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
    bands += ['wfc_acs_f606w']
    bands += ['des_y']
    bands += ['jwst_f150w']

    if indivStarMags: bands = ['sdss_r']

    nBands = len(bands)

    if fullSubhaloSpectra:
        # set nBands to size of wavelength grid within SDSS spectral range
        sdss_min_ang = 3000.0
        sdss_max_ang = 10000.0
        ww = np.where( (pop.wave_ang >= sdss_min_ang) & (pop.wave_ang <= sdss_max_ang) )[0]
        sdss_min_ind = ww.min()
        sdss_max_ind = ww.max() + 1
        nBands = sdss_max_ind - sdss_min_ind

        # only for resolved dust models do we currently calculate full spectra of every star particle
        assert '_res' in dust

        # if fullSubhaloSpectra == 2, we include the peculiar motions of stars (e.g. veldisp)
        # in which case the rel_vel here is overwritten on a per subhalo basis below
        rel_vel_los = None

    # which projections?
    nProj = 1
    efrDirs = False

    if '_res' in dust or sizes is True:
        if isinstance(Nside, (int,long)):
            # numeric Nside -> healpix vertices as projection vectors
            nProj = nside2npix(Nside)
            projVecs = pix2vec(Nside,range(nProj),nest=True)
            projVecs = np.transpose( np.array(projVecs, dtype='float32') ) # Nproj,3
        else:
            # string Nside -> custom projection vectors
            if Nside == 'edgeon_faceon_rnd':
                if '_res' in dust:
                    # 2D: edge-on, face-on, edge-on-smallest, edge-on-random, random for 2D, and then again for 3D
                    nProj = 5 * 2 
                else:
                    # 2D: edge-on, face-on, edge-on-smallest, edge-on-random, random, and +1 for 3D half-light rad
                    nProj = 5 + 1

                Nside = Nside.encode('ascii') # for hdf5 attr save
                projVecs = np.zeros( (nProj-1,3), dtype='float32' ) # derive per subhalo
                efrDirs = True
                assert (sizes is True) or ('_res' in dust) # only cases where efr logic exists for now
            elif Nside == 'z-axis':
                # single projection in the direction of the z-axis of the simulation box
                nProj = 1
                projVecs = np.array( [0,0,1], dtype='float32' ).reshape(1,3) # [nProj,3]
            elif Nside == None:
                pass # no 2D radii
            else:
                assert 0 # unhandled

    # prepare catalog metadata
    desc = "Stellar light emission (total magnitudes) by subhalo"
    if sizes: desc = "Stellar half light radii (code units) by subhalo"
    if indivStarMags: desc = "Star particle individual magnitudes"
    if fullSubhaloSpectra:
        desc = "Optical spectra by subhalo, [%d] wavelength points between [%.1f Ang] and [%.1f Ang]." % \
            (nBands,sdss_min_ang,sdss_max_ang)
    else:
        desc  += ", in multiple rest-frame bands."

    select = "All Subfind subhalos"
    if indivStarMags: select = "All PartType4 particles in all subhalos"
    select += " (numProjectionsPer = %d) (%s)." % (nProj, Nside)

    print(' %s\n %s' % (desc,select))

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType','SubhaloHalfmassRadType','SubhaloPos'])
    gc['subhalos']['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']
    nSubsTot = gc['header']['Nsubgroups_Total']

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange = _pSplitBounds(sP, pSplit, Nside, indivStarMags)

    nSubsDo = len(subhaloIDsTodo)
    partInds = None

    print(' Total # Subhalos: %d, processing [%d] in [%d] bands and [%d] projections...' % \
        (nSubsTot,nSubsDo,nBands,nProj))

    # allocate
    if indivStarMags:
        # compute number of PT4 particles we will do (cover full PT4 size)
        nPt4Tot = cosmo.load.snapshotHeader(sP)['NumPart'][sP.ptNum('stars')]
        nPt4Do = indRange['stars'][1] - indRange['stars'][0] + 1

        if pSplit is None: nPt4Do = nPt4Tot
        if pSplit is not None and pSplit[0] == pSplit[1] - 1:
            nPt4Do = nPt4Tot - indRange['stars'][0]

        # allocate for individual particles
        r = np.zeros( (nPt4Do, nBands, nProj), dtype='float32' )

        # store global (snapshot) indices of particles we process 
        partInds = np.arange(indRange['stars'][0], nPt4Do+indRange['stars'][0], dtype='int64')
        assert partInds.size == nPt4Do
        print(' Total # PT4 particles: %d, processing [%d] now, range [%d - %d]...' % \
            (nPt4Tot,nPt4Do,indRange['stars'][0],indRange['stars'][0]+nPt4Do))
    else:
        # allocate one save per subhalo
        r = np.zeros( (nSubsDo, nBands, nProj), dtype='float32' )
        
    r.fill(np.nan)

    # radial restriction
    radRestrictIn2D, radSqMin, radSqMax = _radialRestriction(sP, nSubsTot, rad)
    assert radSqMin.max() == 0.0 # not handled here

    # global load of all stars in all groups in snapshot
    starsLoad = ['initialmass','sftime','metallicity']
    if '_res' in dust or rad is not None or sizes is not None: starsLoad += ['pos']
    if sizes: starsLoad += ['mass']
    if fullSubhaloSpectra == 2: starsLoad += ['vel','masses'] # masses is the current weight for LOS mean vel

    stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=starsLoad, indRange=indRange['stars'])

    printFac = 100.0 if sP.res > 512 else 10.0

    # non-resolved dust: loop over all requested bands first
    if '_res' not in dust:
        if sizes:
            gas = cosmo.load.snapshotSubset(sP, 'gas', fields=['pos','mass','sfr'], indRange=indRange['gas'])

        for bandNum, band in enumerate(bands):
            print('  %02d/%02d [%s]' % (bandNum+1,len(bands),band))

            # request magnitudes in this band for all stars
            mags = pop.mags_code_units(sP, band, stars['GFM_StellarFormationTime'], 
                                                 stars['GFM_Metallicity'], 
                                                 stars['GFM_InitialMass'], retFullSize=True)

            # loop over subhalos
            for i, subhaloID in enumerate(subhaloIDsTodo):
                if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo:
                    print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

                # slice starting/ending indices for stars local to this subhalo
                i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange['stars'][0]
                i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

                assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

                if i1 == i0:
                    continue # zero length of this type
                
                # radius restriction: use squared radii and sq distance function
                validMask = np.ones( i1-i0, dtype=np.bool )
                if rad is not None:
                    rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                          stars['Coordinates'][i0:i1,:], sP )
                    validMask &= (rr <= radSqMax[subhaloID])
                wValid = np.where(validMask)
                if len(wValid[0]) == 0:
                    continue # zero length of particles satisfying radial cut and restriction

                magsLocal = mags[i0:i1][wValid] # wind particles still here, and have NaN

                if not sizes and not indivStarMags:
                    # convert mags to luminosities, sum together
                    totalLum = np.nansum( np.power(10.0, -0.4 * magsLocal) )

                    # convert back to a magnitude in this band
                    if totalLum > 0.0:
                        r[i,bandNum,0] = -2.5 * np.log10( totalLum )
                elif indivStarMags:
                    # save raw magnitudes per particle (wind/outside subhalo entries left at NaN)
                    saveInds = np.arange(i0, i1)
                    r[saveInds[wValid],bandNum,0] = magsLocal
                elif sizes:
                    # require at least 2 stars for size calculation
                    if len(wValid[0]) < 2:
                        continue

                    # slice starting/ending indices for -gas- local to this subhalo
                    i0g = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('gas')] - indRange['gas'][0]
                    i1g = i0g + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('gas')]

                    assert i0g >= 0 and i1g <= (indRange['gas'][1]-indRange['gas'][0]+1)

                    # calculate projection directions for this subhalo
                    projCen = gc['subhalos']['SubhaloPos'][subhaloID,:]

                    if efrDirs:
                        # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                        rHalf = gc['subhalos']['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
                        shPos = gc['subhalos']['SubhaloPos'][subhaloID,:]

                        gasLocal = { 'Masses' : gas['Masses'][i0g:i1g], 
                                     'Coordinates' : np.squeeze(gas['Coordinates'][i0g:i1g,:]),
                                     'StarFormationRate' : gas['StarFormationRate'][i0g:i1g],
                                     'count' : (i1g - i0g) }
                        starsLocal = { 'Masses' : stars['Masses'][i0:i1],
                                       'Coordinates' : np.squeeze(stars['Coordinates'][i0:i1,:]),
                                       'GFM_StellarFormationTime' : stars['GFM_StellarFormationTime'][i0:i1],
                                       'count' : (i1 - i0) }

                        I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
                        rots = rotationMatricesFromInertiaTensor(I)

                        rotMatrices = [rots['edge-on'], rots['face-on'], rots['edge-on-smallest'],
                                       rots['edge-on-random'], rots['identity']]

                    # get interpolated 2D half light radii
                    for projNum in range(nProj-1):
                        # rotate coordinates
                        pos_stars = np.squeeze(stars['Coordinates'][i0:i1,:][wValid,:])
                        pos_stars_rot, _ = rotateCoordinateArray(sP, pos_stars, rotMatrices[projNum], 
                                                                 projCen, shiftBack=False)

                        # calculate 2D radii as rr2d (.A1 convert matrix to flattened ndarray)
                        x_2d = pos_stars_rot[:,0].A1 # realize axes=[0,1]
                        y_2d = pos_stars_rot[:,1].A1 # realize axes=[0,1]
                        rr2d = np.sqrt( x_2d*x_2d + y_2d*y_2d )

                        r[i,bandNum,projNum] = _findHalfLightRadius(rr2d,magsLocal)

                    # calculate radial distance of each star particle if not yet already
                    if rad is None:
                        rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                              stars['Coordinates'][i0:i1,:], sP )
                    rr = np.sqrt(rr[wValid])

                    # get interpolated 3D half light radius
                    r[i,bandNum,-1] = _findHalfLightRadius(rr,magsLocal)

    # or, resolved dust: loop over all subhalos first
    if '_res' in dust:
        # prep: resolved dust attenuation uses simulated gas distribution in each subhalo
        gas = cosmo.load.snapshotSubset(sP, 'gas', fields=['pos','metal','mass','nh'], indRange=indRange['gas'])
        gas['GFM_Metals'] = cosmo.load.snapshotSubset(sP, 'gas', 'metals_H', indRange=indRange['gas']) # H only

        # prep: override 'Masses' with neutral hydrogen mass (model or snapshot value), free some memory
        if modelH:
            gas['Density'] = cosmo.load.snapshotSubset(sP, 'gas', 'dens', indRange=indRange['gas'])
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutral=True)
            gas['Density'] = None
        else:
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutralSnap=True)

        gas['GFM_Metals'] = None
        gas['NeutralHydrogenAbundance'] = None
        gas['Cellsize'] = cosmo.load.snapshotSubset(sP, 'gas', 'cellsize', indRange=indRange['gas'])

        # prep: unit conversions on stars (age,mass,metallicity)
        stars['GFM_StellarFormationTime'] = sP.units.scalefacToAgeLogGyr(stars['GFM_StellarFormationTime'])
        stars['GFM_InitialMass'] = sP.units.codeMassToMsun(stars['GFM_InitialMass'])

        stars['GFM_Metallicity'] = logZeroMin(stars['GFM_Metallicity'])
        stars['GFM_Metallicity'][np.where(stars['GFM_Metallicity'] < -20.0)] = -20.0

        if sizes:
            gas['StarFormationRate'] = cosmo.load.snapshotSubset(sP, 'gas', fields=['sfr'], indRange=indRange['gas'])

        # outer loop over all subhalos
        if not fullSubhaloSpectra: print(' Bands: [%s].' % ', '.join(bands))

        for i, subhaloID in enumerate(subhaloIDsTodo):
            print('[%d] subhalo = %d' % (i,subhaloID))
            if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo:
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

            # slice starting/ending indices for stars local to this subhalo
            i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange['stars'][0]
            i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

            assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

            if i1 == i0:
                continue # zero length of this type
            
            # radius restriction: use squared radii and sq distance function
            validMask = np.ones( i1-i0, dtype=np.bool )

            if rad is not None:
                if not radRestrictIn2D:
                    # apply in 3D
                    rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                          stars['Coordinates'][i0:i1,:], sP )
                else:
                    # apply in 2D projection, limited support for now, just Nside='z-axis'
                    # otherwise, for any more complex projection, need to apply it here, and anyways 
                    # for nProj>1, this validMask selection logic becomes projection dependent, so 
                    # need to move it inside the range(nProj) loop, which is definitely doable
                    assert Nside == 'z-axis' and nProj == 1 and np.array_equal(projVecs,[[0,0,1]])
                    p_inds = [0,1] # x,y
                    pt_2d = gc['subhalos']['SubhaloPos'][subhaloID,:]
                    pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                    vecs_2d = np.zeros( (i1-i0, 2), dtype=stars['Coordinates'].dtype )
                    vecs_2d[:,0] = stars['Coordinates'][i0:i1,p_inds[0]]
                    vecs_2d[:,1] = stars['Coordinates'][i0:i1,p_inds[1]]

                    rr = periodicDistsSq( pt_2d, vecs_2d, sP ) # handles 2D

                validMask &= (rr <= radSqMax[subhaloID])

            validMask &= np.isfinite(stars['GFM_StellarFormationTime'][i0:i1] ) # remove wind

            wValid = np.where(validMask)[0]

            if len(wValid) == 0:
                continue # zero length of particles satisfying radial cut and real stars restriction

            if len(wValid) < 2 and sizes:
                continue # require at least 2 stars for size calculation

            ages_logGyr = stars['GFM_StellarFormationTime'][i0:i1][wValid]
            metals_log  = stars['GFM_Metallicity'][i0:i1][wValid]
            masses_msun = stars['GFM_InitialMass'][i0:i1][wValid]
            pos_stars   = stars['Coordinates'][i0:i1,:][wValid,:]
            
            assert ages_logGyr.shape == metals_log.shape == masses_msun.shape
            assert pos_stars.shape[0] == ages_logGyr.size and pos_stars.shape[1] == 3

            if fullSubhaloSpectra == 2:
                # derive mean stellar LOS velocity of selected stars, and LOS peculiar velocities of each
                # limited support for now, just Nside='z-axis', otherwise for any more complex projection, 
                # need to apply it here, and anyways for nProj>1, this vel_stars calculation becomes 
                # projection dependent, so need to move it inside the range(nProj) loop, as above
                assert Nside == 'z-axis' and nProj == 1 and np.array_equal(projVecs,[[0,0,1]])
                p_ind = 2 # z

                vel_stars = stars['Velocities'][i0:i1,:][wValid,:]
                masses_stars = stars['Masses'][i0:i1][wValid]

                vel_stars = sP.units.particleCodeVelocityToKms(vel_stars)

                # mass weighted, this could be light weighted... anyways a change to this represents a 
                # constant wavelength shift, e.g. is fit out as a residual redshift
                mean_vel_los = np.average( vel_stars[:,p_ind] , weights=masses_stars )

                rel_vel_los = vel_stars[:,p_ind] - mean_vel_los

            # slice starting/ending indices for -gas- local to this subhalo
            i0g = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('gas')] - indRange['gas'][0]
            i1g = i0g + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('gas')]

            assert i0g >= 0 and i1g <= (indRange['gas'][1]-indRange['gas'][0]+1)
 
            # calculate projection directions for this subhalo
            projCen = gc['subhalos']['SubhaloPos'][subhaloID,:]

            if efrDirs:
                # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                rHalf = gc['subhalos']['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
                shPos = gc['subhalos']['SubhaloPos'][subhaloID,:]

                gasLocal = { 'Masses' : gas['Masses'][i0g:i1g], 
                             'Coordinates' : np.squeeze(gas['Coordinates'][i0g:i1g,:]),
                             'StarFormationRate' : gas['StarFormationRate'][i0g:i1g] }
                starsLocal = { 'Masses' : stars['Masses'][i0:i1],
                               'Coordinates' : np.squeeze(stars['Coordinates'][i0:i1,:]),
                               'GFM_StellarFormationTime' : stars['GFM_StellarFormationTime'][i0:i1] }

                I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
                rots = rotationMatricesFromInertiaTensor(I)

                rotMatrices = [rots['edge-on'], rots['face-on'], rots['edge-on-smallest'],
                               rots['edge-on-random'], rots['identity']]
                rotMatrices.extend(rotMatrices) # append to itself, now has (5 2d + 5 3d) = 10 elements

            # loop over all different viewing directions
            for projNum in range(nProj):
                # at least 2 gas cells exist in subhalo?
                if i1g > i0g+1:
                    # subsets
                    pos     = gas['Coordinates'][i0g:i1g,:]
                    hsml    = 2.5 * gas['Cellsize'][i0g:i1g]
                    mass_nh = gas['Masses'][i0g:i1g]
                    quant_z = gas['GFM_Metallicity'][i0g:i1g]

                    # compute line of sight integrated quantities (choose appropriate projection)
                    if efrDirs:
                        N_H, Z_g = pop.resolved_dust_mapping(pos, hsml, mass_nh, quant_z, pos_stars, 
                                                             projCen, rotMatrix=rotMatrices[projNum])
                    else:
                        projVec = projVecs[projNum,:]
                        N_H, Z_g = pop.resolved_dust_mapping(pos, hsml, mass_nh, quant_z, 
                                                             pos_stars, projCen, projVec=projVec)
                else:
                    # set columns to zero
                    N_H = np.zeros( len(wValid), dtype='float32' )
                    Z_g = np.zeros( len(wValid), dtype='float32' )

                if sizes:
                    # compute attenuated stellar luminosity for each star particle in each band
                    magsLocal = pop.dust_tau_model_mags(bands,N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                                                        ret_indiv=True)

                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        # do 2D radii for first set of efr, then 3D radii for second set of efr
                        if projNum < nProj/2:
                            # rotate coordinates
                            pos_stars_rot, _ = rotateCoordinateArray(sP, pos_stars, rotMatrices[projNum], 
                                                                     projCen, shiftBack=False)

                            # calculate 2D radii as rr2d (.A1 convert matrix to flattened ndarray)
                            x_2d = pos_stars_rot[:,0].A1 # realize axes=[0,1]
                            y_2d = pos_stars_rot[:,1].A1 # realize axes=[0,1]
                            rr2d = np.sqrt( x_2d*x_2d + y_2d*y_2d )

                            # get interpolated 2D half light radii
                            r[i,bandNum,projNum] = _findHalfLightRadius(rr2d,magsLocal[band])

                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = periodicDistsSq( projCen, stars['Coordinates'][i0:i1,:], sP )
                            rrLocal = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i,bandNum,projNum] = _findHalfLightRadius(rrLocal,magsLocal[band])

                elif indivStarMags:
                    # compute attenuated stellar luminosity for each star particle in each band
                    magsLocal = pop.dust_tau_model_mags(bands,N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                                                        ret_indiv=True)

                    saveInds = np.arange(i0, i1)
                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        # save raw magnitudes per particle (wind/outside subhalos left at NaN)
                        r[saveInds[wValid],bandNum,projNum] = magsLocal[band]

                elif fullSubhaloSpectra:
                    # request stacked spectrum of all stars, optionally handle doppler velocity shifting
                    spectrum = pop.dust_tau_model_mags(bands,N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                                                        ret_full_spectrum=True, rel_vel=rel_vel_los)

                    # unit conversion into SDSS spectra units, and redshift to sP.redshift
                    spectrum = pop.convertSpecToSDSSUnitsAndRedshift(spectrum)

                    # save spectrum within valid wavelength range
                    r[i,:,projNum] = spectrum[sdss_min_ind:sdss_max_ind]
                else:
                    # compute total attenuated stellar luminosity in each band
                    magsLocal = pop.dust_tau_model_mags(bands,N_H,Z_g,ages_logGyr,metals_log,masses_msun)

                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        r[i,bandNum,projNum] = magsLocal[band]

    # prepare save
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'bands'       : [b.encode('ascii') for b in bands],
             'dust'        : dust.encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    if partInds is not None:
        attrs['partInds'] = partInds

    if fullSubhaloSpectra:
        # save wavelength grid and details of redshifting
        attrs['wavelength'] = pop.wave_ang[sdss_min_ind : sdss_max_ind]
        attrs['spectraUnits'] = '10^-17 erg/cm^2/s/Ang'.encode('ascii')
        attrs['spectraLumDistMpc'] = sP.units.redshiftToLumDist( sP.redshift )
        attrs['spectraFiberDiameterCode'] = fiber_diameter

    if '_res' in dust:
        # save projection details
        attrs['nProj'] = nProj
        attrs['Nside'] = Nside
        attrs['projVecs'] = projVecs
        attrs['modelH'] = modelH

    # remove nProj and/or nBands dimensions if unity (never remove nSubsDo dimension)
    if r.shape[2] == 1: r = np.squeeze(r, axis=(2))
    if r.shape[1] == 1: r = np.squeeze(r, axis=(1)) 

    return r, attrs

def mergerTreeQuant(sP, pSplit, treeName, quant, smoothing=None):
    """ For every subhalo, compute an assembly/related quantity using a merger tree. """
    from scipy.signal import medfilt
    assert quant in ['zForm']
    assert pSplit is None # not implemented

    def _ma(X, windowSize):
        """ Running mean. Endpoints are copied unmodified in bwhalf region. """
        r = np.zeros( X.size, dtype=X.dtype )
        bwhalf = int(windowSize/2.0)

        cumsum = np.cumsum(np.insert(X, 0, 0)) 
        r[bwhalf:-bwhalf] = (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
        r[0:bwhalf] = X[0:bwhalf]
        r[-bwhalf:] = X[-bwhalf:]
        return r

    def _mm(X, windowSize):
        """ Running median. """
        return medfilt(X, windowSize)

    # process smoothing request [method,windowSize,windowVal,order]
    assert len(smoothing) == 3
    assert smoothing[2] == 'snap' # todo: e.g. Gyr, scalefac

    # prepare catalog metadata
    desc   = "Merger tree quantity (%s) using smoothing [%s]." % (quant,'_'.join([str(s) for s in smoothing]))
    select = "All Subfind subhalos."

    # load snapshot and subhalo information
    redshifts = snapNumToRedshift(sP, all=True)

    gcH = cosmo.load.groupCatHeader(sP)
    nSubsTot = gcH['Nsubgroups_Total']

    ids = np.arange(nSubsTot, dtype='int32')

    # allocate return, NaN indicates not computed (e.g. not in tree at sP.snap)
    r = np.zeros( nSubsTot, dtype='float32' )
    r.fill(np.nan)

    # load all trees at once
    fields = ['SnapNum']

    if quant == 'zForm':
        fields.append('SubhaloMass')

    mpbs = cosmo.mergertree.loadMPBs(sP, ids, fields=fields, treeName=treeName)

    # loop over subhalos
    printFac = 100.0 if sP.res > 512 else 10.0

    for i in range(nSubsTot):
        if i % int(nSubsTot/printFac) == 0 and i <= nSubsTot:
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsTot))

        if i not in mpbs:
            continue # subhalo ID i not in tree at sP.snap

        # todo: could generalize here into generic reduction operations over a given tree field
        # e.g. 'max', 'min', 'mean' of 'SubhaloSFR', 'SubhaloGasMetallicity', ... in addition to 
        # more specialized calculations such as formation time
        loc_vals = mpbs[i]['SubhaloMass']
        loc_snap = mpbs[i]['SnapNum']

        if loc_snap.size < smoothing[1]+1:
            continue

        # smoothing
        if smoothing[0] == 'mm': # moving median window of size N snapshots
            loc_vals2 = _mm(loc_vals, windowSize=smoothing[1])

        if smoothing[0] == 'ma': # moving average window of size N snapshots
            loc_vals2 = _ma(loc_vals, windowSize=smoothing[1])

        if smoothing[0] == 'poly': # polynomial fit of Nth order
            coeffs = np.polyfit(loc_snap, loc_vals, smoothing[1])
            loc_vals2 = np.polyval(coeffs, loc_snap) # resample to original X-pts

        assert loc_vals2.shape == loc_vals.shape

        # where does half of max of [smoothed] total mass occur?
        halfMaxVal = loc_vals2.max() * 0.5

        #if smoothing[0] == 'poly': # root find on the polynomial coefficients (not so simple)
        #coeffs[-1] -= halfMaxVal # shift such that we find the M=halfMaxVal not M=0 roots
        #roots = np.polynomial.polynomial.polyroots(coeffs[::-1]) # there are many

        w = np.where(loc_vals2 >= halfMaxVal)[0]
        assert len(w) # by definition

        # linearly interpolate between snapshots
        snap0 = loc_snap[w].min()
        ind0 = w.max() # lowest snapshot where mass exceeds halfMaxVal
        ind1 = ind0 + 1 # lower snapshot (earlier in time)

        assert snap0 == loc_snap[ind0]

        if ind0 == loc_vals.size-1:
            # only at first tree entry
            z_form = redshifts[loc_snap[ind0]]
        else:
            assert ind0 >= 0 and ind0 < loc_vals.size-1
            assert ind1 > 0 and ind1 <= loc_vals.size-1

            z0 = redshifts[loc_snap[ind0]]
            z1 = redshifts[loc_snap[ind1]]
            m0 = loc_vals2[ind0]
            m1 = loc_vals2[ind1]

            assert m0 >= halfMaxVal and m1 <= halfMaxVal

            # linear interpolation, find redshift where mass=halfMaxVal
            z_form = (halfMaxVal-m0)/(m1-m0) * (z1-z0) + z0
            assert z_form >= z0 and z_form <= z1

        assert z_form >= 0.0
        r[i] = z_form

    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')
    
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    return r, attrs

def tracerTracksQuant(sP, pSplit, quant, op, time, norm=None):
    """ For every subhalo, compute a assembly/accretion/related quantity using the tracker 
    tracks through time. """
    from tracer.tracerEvo import tracersTimeEvo, tracersMetaOffsets, trValsAtAccTimes, \
      accTime, accMode, ACCMODES, mpbValsAtRedshifts, mpbValsAtAccTimes
    from tracer.tracerMC import defParPartTypes, fields_in_log

    assert pSplit is None # not implemented
    assert op in ['mean'] #,'sample']
    assert quant in ['angmom','entr','temp','acc_time_1rvir','acc_time_015rvir','dt_halo']
    assert time is None or time in ['acc_time_1rvir','acc_time_015rvir']
    assert norm is None or norm in ['tvir_tacc','tvir_cur']

    def _nansum(x):
        """ Helper. """
        N = np.count_nonzero(~np.isnan(x))

        r = np.nan
        if N > 0:
            r = np.nansum(x)

        return r, N

    # prepare catalog metadata
    desc   = "Tracer tracks quantity (%s) using [%s] over [%s]." % (quant,op,time)
    select = "All Subfind subhalos."

    # load snapshot and subhalo information
    nSubsTot = cosmo.load.groupCatHeader(sP)['Nsubgroups_Total']
    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    nSubsDo = len(subhaloIDsTodo)

    # allocate return, NaN indicates not computed (e.g. not in tree at sP.snap)
    nTypes = len(defParPartTypes) + 1 # store separately by z=0 particle type, +1 for 'all' combined
    nModes = len(ACCMODES) + 1 # store separately by mode, +1 for 'all' combined

    r = np.zeros( (nSubsDo,nTypes,nModes), dtype='float32' )
    N = np.zeros( (nSubsDo,nTypes,nModes), dtype='int32' ) # accumulate bin counts
    r.fill(np.nan)

    # load 1D reduction of the tracer tracks corresponding to the requested quantity, per tracer,
    # taking into account the time specification. note that 'accretion times' as the quantity 
    # are already just 1D, while the others need to be selected at a specific time, or averaged 
    # over a specific time window, reducing the (Nsnaps,Ntr) shaped tracks into a 1D (Ntr) shape
    if quant == 'acc_time_1rvir':
        assert time is None and norm is None
        tracks = accTime(sP, rVirFac=1.0)

    elif quant == 'acc_time_015rvir':
        assert time is None and norm is None
        tracks = accTime(sP, rVirFac=0.15)

    elif quant == 'dt_halo':
        assert time is None and norm is None
        age_universe_rad100rvir = sP.units.redshiftToAgeFlat( accTime(sP, rVirFac=1.0) )
        age_universe_rad015rvir = sP.units.redshiftToAgeFlat( accTime(sP, rVirFac=0.15) )
        tracks = age_universe_rad015rvir - age_universe_rad100rvir
        assert np.nanmin(tracks) >= 0.0 # negative would contradict definition

        # handful of zeros, set to nan
        w = np.where(tracks == 0.0)[0]
        if len(w):
            print(' Note: setting [%d] of [%d] dt_halo==0 to nan.' % (len(w),tracks.size))
            tracks[w] = np.nan
    else:
        if time is None:
            # full tracks (what are we going to do with these?)
            tracks = tracersTimeEvo(sP, quant, all=True) #snapStep=None
            # do e.g. a mean across all time
            assert norm is None
            import pdb; pdb.set_trace()

        elif time == 'acc_time_1rvir':
            tracks = trValsAtAccTimes(sP, quant, rVirFac=1.0)
            
        elif time == 'acc_time_015rvir':
            tracks = trValsAtAccTimes(sP, quant, rVirFac=0.15)

    assert tracks.ndim == 1

    # remove log if needed
    if quant in fields_in_log:
        tracks = 10.0**tracks

    # normalization?
    if norm is not None:
        # what MPB property to normalize by, and take it at which time (e.g. Tvir at tAcc)
        norm_field, norm_time = norm.split("_")

        if norm_time == 'tacc':
            norm_vals = mpbValsAtAccTimes(sP, norm_field, rVirFac=1.0)
        if norm_time == 'z0':
            norm_vals = mpbValsAtRedshifts(sP, norm_field, 0.0)
        if norm_time == 'cur':
            norm_vals = mpbValsAtRedshifts(sP, norm_field, sP.redshift)

        assert tracks.shape == norm_vals.shape
        if tracks.max() > 20.0 and norm_vals.max() < 20.0: assert 0 # check log of norm_vals

        tracks /= norm_vals

    # load mode decomposition, metadata offsets
    mode = accMode(sP)

    meta, nTracerTot = tracersMetaOffsets(sP, all='Subhalo')

    # loop over subhalos
    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/100)]) == 0 and i <= nSubsDo:
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

        # loop over modes
        for modeNum, modeName in enumerate(ACCMODES.keys()):
            modeVal = ACCMODES[modeName]
            #print('   [%s] %s (val=%d)' % (modeNum,modeName,modeVal))

            # loop over partTypes
            for j, ptName in enumerate(defParPartTypes):
                # slice starting/ending indices for tracers local to this subhalo
                i0 = meta[ptName]['offset'][subhaloID]
                i1 = i0 + meta[ptName]['length'][subhaloID]

                if i1 == i0:
                    continue # zero length of this type

                # mode segregation
                if modeNum < len(ACCMODES):
                    w_mode = np.where( mode[i0:i1] == modeVal )[0]
                else:
                    w_mode = np.arange(i1-i0) # 'all', use all

                if w_mode.size == 0:
                    continue # no tracers of this mode in this subhalo

                # local slice, mode dependent
                loc_vals = tracks[i0:i1][w_mode]

                # should never overwrite anything
                assert np.isnan( r[i,j,modeNum] )
                assert N[i,j,modeNum] == 0

                # store intermediate value (e.g. sum for mean) and counts for later calculation of op
                if op == 'mean':
                    r[i,j,modeNum], N[i,j,modeNum] = _nansum(loc_vals)

                #print(j,ptName,N[i,j,modeNum],r[i,j,modeNum],i1-i0,w_mode.size)

            # do op for all part types together, for this mode
            assert np.isnan( r[i,j+1,modeNum] )
            assert N[i,j+1,modeNum] == 0

            if op == 'mean':                
                r[i,j+1,modeNum], _ = _nansum( r[i,0:j+1,modeNum] )
                N[i,j+1,modeNum] = np.sum( N[i,0:j+1,modeNum] )
                #print(j+1,'all',N[i,j+1,modeNum],r[i,j+1,modeNum])
            
        # do op for all modes together, for each partType+all
        #print('   [%s] ALL MODES' % (modeNum+1))

        for j in range(len(defParPartTypes)+1):
            assert np.isnan( r[i,j,modeNum+1] ) # should never overwrite anything
            assert N[i,j,modeNum+1] == 0

            r[i,j,modeNum+1], _ = _nansum( r[i,j,0:modeNum+1] )
            N[i,j,modeNum+1] = np.sum( N[i,j,0:modeNum+1] )
            #print(j,' - ',N[i,j,modeNum+1],r[i,j,modeNum+1])

    # now, normalize element by element
    w = np.where(N == 0)
    assert len( np.where(np.isnan(r[w]))[0] ) == len(w[0]) # all zero counts should have nan value

    w = np.where(N > 0)
    assert len( np.where(np.isnan(r[w]))[0] ) == 0 # all nonzero counts should have finite value

    r[w] /= N[w]

    # restore log for consistency
    if quant in fields_in_log:
        r = logZeroNaN(r)

    attrs = {'Description'  : desc.encode('ascii'), 
             'Selection'    : select.encode('ascii')}
             #'accModes'     : ACCMODES, # encoding errors, would need to do more carefully
             #'parPartTypes' : defParPartTypes}

    return r, attrs

def wholeBoxColDensGrid(sP, pSplit, species):
    """ Compute a 2D grid of gas column densities [cm^-2] covering the entire simulation box. For 
        example to derive the neutral hydrogen CDDF. The grid has dimensions of boxGridDim x boxGridDim 
        and so a grid cell size of (sP.boxSize/boxGridDim) in each dimension. Strategy is a chunked 
        load of the snapshot files, for each using SPH-kernel deposition to distribute the mass of 
        the requested species (e.g. HI, CIV) in all gas cells onto the grid.
    """
    assert pSplit is None # not implemented

    from cosmo import hydrogen
    from cosmo.load import snapshotHeader
    from util.sphMap import sphMapWholeBox
    from cosmo.cloudy import cloudyIon

    # adjust projection depth
    projDepthCode = sP.boxSize

    if '_depth10' in species:
        projDepthCode = 10000.0 # 10 Mpc/h
        species = species.split("_depth10")[0]  

    # check
    hDensSpecies = ['HI','HI_noH2']
    zDensSpecies = ['O VI','O VI 10','O VI 25','O VI solar','O VII','O VIII']

    if species not in hDensSpecies + zDensSpecies + ['Z']:
        raise Exception('Not implemented.')

    # config
    h = snapshotHeader(sP)
    nChunks = numPartToChunkLoadSize( h['NumPart'][sP.ptNum('gas')] )
    axes    = [0,1] # x,y projection
    
    # info
    h = cosmo.load.snapshotHeader(sP)

    if species in zDensSpecies:
        boxGridSize = boxGridSizeMetals
    else:
        boxGridSize = boxGridSizeHI

    # adjust grid size
    if species == 'O VI 10':
        boxGridSize = 10.0 # test, x2 bigger
    if species == 'O VI 25':
        boxGridSize = 2.5 # test, x2 smaller

    boxGridDim = round(sP.boxSize / boxGridSize)
    chunkSize = int(h['NumPart'][sP.ptNum('gas')] / nChunks)

    if species in hDensSpecies + zDensSpecies:
        desc = "Square grid of integrated column densities of ["+species+"] in units of [cm^-2]. "
    if species == 'Z':
        desc = "Square grid of mean gas metallicity in units of [log solar]."

    if species == 'HI':
        desc += "Atomic only, H2 calculated and removed."
    if species == 'HI_noH2':
        desc += "All neutral hydrogen included, any contribution of H2 ignored."

    select = "Grid dimensions: %dx%d pixels (cell size = %06.2f codeunits) along axes=[%d,%d]." % \
             (boxGridDim,boxGridDim,boxGridSize,axes[0],axes[1])

    print(' '+desc)
    print(' '+select)
    print(' Total # Snapshot Load Chunks: '+str(nChunks)+' ('+str(chunkSize)+' cells per load)')

    # specify needed data load, and allocate accumulation array(s)
    if species in hDensSpecies:
        fields = ['Coordinates','Masses','Density','metals_H','NeutralHydrogenAbundance']

        r = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )

    if species in zDensSpecies:
        fields = ['Coordinates','Masses','Density']

        r = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )

    if species == 'Z':
        fields = ['Coordinates','Masses','Density','GFM_Metallicity']

        rM = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )
        rZ = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )

    # loop over chunks (we are simply accumulating, so no need to load everything at once)
    for i in np.arange(nChunks):
        # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
        indRange = [i*chunkSize, (i+1)*chunkSize-1]
        if i == nChunks-1: indRange[1] = h['NumPart'][sP.ptNum('gas')]-1
        print('  [%2d] %9d - %d' % (i,indRange[0],indRange[1]))

        # load
        gas = cosmo.load.snapshotSubset(sP, 'gas', fields, indRange=indRange)

        # calculate smoothing size (V = 4/3*pi*h^3)
        vol = gas['Masses'] / gas['Density']
        hsml = (vol * 3.0 / (4*np.pi))**(1.0/3.0)
        hsml = hsml.astype('float32')

        if species in hDensSpecies:
            # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
            mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2' or species=='HI3'),
                                                 totalNeutral=(species=='HI_noH2'))

            # simplified models (difference is quite small in CDDF)
            #mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
            #mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

            # grid gas mHI using SPH kernel, return in units of [10^10 Msun * h / ckpc^2]
            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mHI, quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True, sliceFac=1.0)

            r += ri

        if species in zDensSpecies:
            # calculate metal ion mass, and grid column densities
            element = species.split()[0]
            ionNum  = species.split()[1]

            ion = cloudyIon(sP, el=element, redshiftInterp=True)

            if len(species.split()) == 3 and species.split()[2] == 'solar':
                # assume solar abundances
                mMetal = gas['Masses'] * ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange,
                                                                    assumeSolarAbunds=True)
            else:
                # default (use cached ion masses)
                mMetal = cosmo.load.snapshotSubset(sP, 'gas', '%s %s mass' % (element,ionNum), indRange=indRange)
            
            # determine projection depth fraction
            boxWidthFrac = projDepthCode / sP.boxSize

            # project
            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mMetal, quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True, sliceFac=boxWidthFrac)

            r += ri

        if species == 'Z':
            # grid total gas mass using SPH kernel, return in units of [10^10 Msun / h]
            rMi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=gas['Masses'], quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False, sliceFac=1.0)

            # grid total gas metal mass
            mMetal = gas['Masses'] * gas['GFM_Metallicity']

            rZi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mMetal, quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False, sliceFac=1.0)

            rM += rMi
            rZ += rZi

    # finalize
    if species in hDensSpecies+zDensSpecies:
        # column density: convert units from [code column density, above] to [H atoms/cm^2] and take log
        rr = sP.units.codeColDensToPhys(r, cgs=True, numDens=True)

        if species in zDensSpecies:
            ion = cosmo.cloudy.cloudyIon(None)
            rr /= ion.atomicMass(species.split()[0]) # [H atoms/cm^2] to [ions/cm^2]

        rr = np.log10(rr)

    if species == 'Z':
        # metallicity: take Z = mass_tot/mass_gas for each pixel, normalize by solar, take log
        rr = rZ / rM
        rr = np.log10( rr / sP.units.Z_solar )

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def wholeBoxCDDF(sP, pSplit, species, omega=False):
    """ Compute the column density distribution function (CDDF, i.e. histogram) of column densities 
        given a full box colDens grid. If omega == True, then instead compute the single number 
        Omega_species = rho_species / rho_crit,0. """
    assert pSplit is None # not implemented
    from cosmo.load import auxCat
    from cosmo.hydrogen import calculateCDDF

    if omega:
        mass = cosmo.load.snapshotSubset(sP, 'gas', species + ' mass')
        code_dens = np.sum(mass) / sP.boxSize**3 # code units
        rr = sP.units.codeDensToCritRatio(code_dens, redshiftZero=True)
        desc = 'Omega_%s = (rho_%s / rho_crit,z=0)' % (species,species)
        attrs = {'Description' : desc.encode('ascii'), 
                 'Selection'   : 'All gas cells in box.'.encode('ascii')}
        return rr, attrs

    # config
    binSize   = 0.1 # log cm^-2
    binMinMax = [11.0, 24.0] # log cm^-2

    desc   = "Column density distribution function (CDDF) for ["+species+"]. "
    desc  += "Return has shape [2,nBins] where the first slice gives n [cm^-2], the second fN [cm^-2]."
    select = "Binning min: [%g] max: [%g] size: [%g]." % (binMinMax[0], binMinMax[1], binSize)

    # load
    acField = 'Box_Grid_n'+species
    ac = auxCat(sP, fields=[acField])

    depthFrac = 10000.0/sP.boxSize if '_depth10' in species else 1.0

    # calculate
    fN, n = calculateCDDF(ac[acField], binMinMax[0], binMinMax[1], binSize, sP, depthFrac=depthFrac)

    rr = np.vstack( (n,fN) )
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def subhaloRadialProfile(sP, pSplit, ptType, ptProperty, op, scope, weighting=None, 
                        proj2D=None, ptRestriction=None, subhaloIDsTodo=None):
    """ Compute a radial profile (either total/sum or weighted mean) of a particle property (e.g. mass) 
        for those particles of a given type. If scope=='global', then all snapshot particles are used, 
        and we do the accumulation in a chunked snapshot load. Self/other halo terms are decided based 
        on subfind membership, unless scope=='global_fof', then on group membership. If scope=='fof' or 
        'subfind' then restrict to FoF/subhalo particles only, respectively, and do a restricted load 
        according to pSplit. In this case, only the self-halo term is computed. If scope=='subfind_global' 
        then only the other-halo term is computed, approximating the particle distribution using an 
        already computed subhalo-based accumlation auxCat, e.g. 'Subhalo_Mass_OVI'. Weighting is 
        currently unsupported, but in the future e.g. mass weighted mean temperature profiles would be 
        possible. If proj2D is None, do 3D profiles, otherwise 2-tuple specifying (i) integer coordinate 
        axis in [0,1,2] to project along, and (ii) depth in code units (None for full box). 
        If subhaloIDsTodo is not None, then process this explicit list of subhalos.
    """
    from scipy.stats import binned_statistic

    assert op in ['sum','mean','median','count']
    assert scope in ['global','global_fof','subfind','fof','subfind_global']

    def _binned_statistic_weighted(x, values, statistic, bins, weights=None, weights_w=None):
        """ If weights == None, straight passthrough to scipy.stats.binned_statistic(). Otherwise, 
        compute once for values*weights, again for weights alone, then normalize and return. 
        If weights_w is not None, apply this np.where() result to the weights array. """
        if weights is None:
            return binned_statistic(x, values, statistic=statistic, bins=bins)

        weights_loc = weights[weights_w] if weights_w is not None else weights

        valwt_sum, bin_edges, bin_number = binned_statistic(x, values*weights_loc, statistic=statistic, bins=bins)
        wt_sum, _, _ = binned_statistic(x, weights_loc, statistic=statistic, bins=bins)

        return (valwt_sum/wt_sum), bin_edges, bin_number

    # config
    radMin = 0.0 # log code units
    radMax = 4.0 # log code units
    radNumBins = 100
    minHaloMass = 10.8 # log m200crit
    cenSatSelect = 'cen'

    # determine ptRestriction
    if ptType == 'stars':
        assert ptRestriction is None # otherwise we have a collision
        ptRestriction = 'real_stars'

    # config
    ptLoadType = sP.ptNum(ptType)

    desc   = "Quantity [%s] radial profile for [%s] from [%.1f - %.1f] with [%d] bins." % \
      (ptProperty,ptType,radMin,radMax,radNumBins)
    if ptRestriction is not None:
        desc += " (restriction = %s)." % ptRestriction
    if weighting is not None:
        desc += " (weighting = %s)." % weighting
    if proj2D is not None:
        assert len(proj2D) == 2
        proj2Daxis, proj2Ddepth = proj2D

        if proj2Daxis == 0: p_inds = [1,2,3]
        if proj2Daxis == 1: p_inds = [0,2,1]
        if proj2Daxis == 2: p_inds = [0,1,2]
        proj2D_halfDepth = proj2Ddepth / 2 # code units

        depthStr = 'fullbox' if proj2Ddepth is None else '%.1f' % proj2Ddepth
        desc += " (2D projection axis = %d, depth = %s)." % (proj2Daxis,depthStr)

    desc  +=" (scope = %s). " % scope
    if subhaloIDsTodo is None:
        select = "Subhalos [%s] above a min m200crit halo mass of [%.1f]." % (cenSatSelect,minHaloMass)
    else:
        select = "Subhalos [%d] specifically input." % len(subhaloIDsTodo)

    # load group information and make selection
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloPos','SubhaloLenType'])['subhalos']
    gc['header'] = cosmo.load.groupCatHeader(sP)

    nChunks = 1 # chunk load disabled by default

    # need for scope=='subfind' and scope=='global' (for self/other halo terms)
    gc['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    if scope in ['fof','global_fof']:
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        # for scope=='global_fof' take all FoF particles for the respective halo terms
        GroupLenType = cosmo.load.groupCat(sP, fieldsHalos=['GroupLenType'])['halos']
        GroupOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        SubhaloGrNr = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloGrNr'])['subhalos']

        gc['SubhaloLenType'] = GroupLenType[SubhaloGrNr,:]
        gc['SubhaloOffsetType'] = GroupOffsetType[SubhaloGrNr,:]

    if scope in ['global','global_fof']:
        # enable chunk loading
        h = cosmo.load.snapshotHeader(sP)
        nChunks = numPartToChunkLoadSize( h['NumPart'][sP.ptNum(ptType)] )
        chunkSize = int(h['NumPart'][sP.ptNum(ptType)] / nChunks)

    nSubsTot = gc['header']['Nsubgroups_Total']
    
    # no explicit ID list input, choose subhalos to process now
    if subhaloIDsTodo is None:
        subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

        # css select
        if cenSatSelect is not None:
            cssInds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
            subhaloIDsTodo = subhaloIDsTodo[cssInds]

        # halo mass select
        if minHaloMass is not None:
            halo_masses = cosmo.load.groupCat(sP, fieldsSubhalos=['mhalo_200_log'])
            halo_masses = halo_masses[cssInds]
            wSelect = np.where( halo_masses >= minHaloMass )

            subhaloIDsTodo = subhaloIDsTodo[wSelect]

    nSubsTot = subhaloIDsTodo.size

    if scope in ['global','global_fof']:
        # default particle load range is set inside chunkLoadLoop
        print(' Total # Snapshot Load Chunks: '+str(nChunks)+' ('+str(chunkSize)+' particles per load)')
        indRange = None
        prevMaskInd = 0

        if pSplit is not None:
            # subdivide subhaloIDsTodo directly
            subhaloIDsTodo = pSplitArr(subhaloIDsTodo, pSplit[1], pSplit[0])

    if scope in ['subfind','fof']:
        # if no task parallelism (pSplit), set default particle load ranges
        indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

        if pSplit is not None:
            # subdivide the global [variable ptType!] particle set, then map this back into a division of 
            # subhalo IDs which will be better work-load balanced among tasks
            gasSplit = pSplitRange( indRange[ptType], pSplit[1], pSplit[0] )

            invSubs = inverseMapPartIndicesToSubhaloIDs(sP, gasSplit, ptType, debug=True, flagFuzz=False)

            if pSplit[0] == pSplit[1] - 1:
                invSubs[1] = gc['header']['Nsubgroups_Total']
            else:
                assert invSubs[1] != -1

            # we process a sparse set of subhalo ids, so intersect with the previous (mass/css) selection
            subhaloIDsSpanned = np.arange( invSubs[0], invSubs[1] )
            subhaloIDsTodo = np.intersect1d( subhaloIDsTodo, subhaloIDsSpanned )
            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

        indRange = indRange[ptType] # choose index range for the requested particle type

    nSubsDo = len(subhaloIDsTodo)

    # info
    print(' ' + desc)
    print(' ' + select)
    print(' Total # Subhalos: %d, in mass bin [%d], processing [%d] subhalos now...' % \
        (gc['header']['Nsubgroups_Total'],nSubsTot,nSubsDo))

    # determine radial binning
    rad_bin_edges = np.linspace(radMin,radMax,radNumBins) # bin edges, including inner and outer boundary
    rad_bin_edges = np.hstack([radMin-1.0,rad_bin_edges]) # include an inner bin complete to r=0

    rbins_sq = np.log10( (10.0**rad_bin_edges)**2 ) # we work in squared distances for speed
    rad_bins_code = 0.5*(rad_bin_edges[1:] + rad_bin_edges[:-1]) # bin centers [log]
    rad_bins_pkpc = sP.units.codeLengthToKpc( 10.0**rad_bins_code )

    radMaxSqCode = (10.0**radMax)**2

    # bin (spherical shells in 3D, circular annuli in 2D) volumes/areas [code units]
    r_outer = 10.0**rad_bin_edges[1:]
    r_inner = 10.0**rad_bin_edges[:-1]
    r_inner[0] = 0.0

    bin_volumes_code = 4.0/3.0 * np.pi * (r_outer**3.0 - r_inner**3.0)
    bin_areas_code = np.pi * (r_outer**2.0 - r_inner**2.0) # 2D annuli e.g. if proj2D is not None

    # allocation: for global particle scope: [all, self-halo, other-halo, diffuse]
    # or for subfind/fof scope: [self-halo] only, or for subfind_global scope: [other-halo] only
    numProfTypes = 4 if scope in ['global','global_fof'] else 1 

    # allocate, NaN indicates not computed except for mass where 0 will do
    r = np.zeros( (nSubsDo,radNumBins,numProfTypes), dtype='float32' )
    if numProfTypes == 1: r = np.squeeze(r, axis=2)

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = ['pos']

    if ptRestriction == 'real_stars':
        fieldsLoad.append('sftime')
    if ptRestriction in ['sfrgt0','sfreq0']:
        fieldsLoad.append('sfr')

    if ptProperty == 'radvel':
        if 'pos' not in fieldsLoad: fieldsLoad.append('pos')
        if 'vel' not in fieldsLoad: fieldsLoad.append('vel')
        gc['SubhaloVel'] = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloVel'], sq=True)

    # so long as scope is not 'global', load the full particle set we need for these subhalos now
    if scope not in ['global','global_fof','subfind_global']:
        particles = cosmo.load.snapshotSubset(sP, partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

        if ptProperty not in userCustomFields:
            particles[ptProperty] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=[ptProperty], indRange=indRange)
            assert particles[ptProperty].ndim == 1

        if 'count' not in particles:
            particles['count'] = particles[ particles.keys()[0] ].shape[0]

        # load weights
        if weighting is not None:
            assert op not in ['sum'] # meaningless
            assert op in ['mean'] # currently only supported

            # use particle masses or volumes (linear) as weights
            particles['weights'] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=weighting, indRange=indRange)

            assert particles['weights'].ndim == 1 and particles['weights'].size == particles['count']

    # chunk load: loop (possibly just once if chunk load is disabled)
    for chunkNum in range(nChunks):

        # load chunk now (we are simply accumulating, so no need to load everything at once)
        if scope in ['global','global_fof']:
            # calculate load indices (snapshotSubset is inclusive on last index, make sure we get to the end)
            indRange = [chunkNum*chunkSize, (chunkNum+1)*chunkSize-1]
            if chunkNum == nChunks-1: indRange[1] = h['NumPart'][ptLoadType]-1
            print('  [%2d] %9d - %d' % (chunkNum,indRange[0],indRange[1]))

            particles = cosmo.load.snapshotSubset(sP, partType=ptType, fields=fieldsLoad, sq=False, 
                                                 indRange=indRange)

            if ptProperty not in userCustomFields:
                particles[ptProperty] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=[ptProperty], 
                                                                  indRange=indRange)
                assert particles[ptProperty].ndim == 1

            assert 'count' in particles
            assert weighting is None # load not implemented

        # if approximating the particle distribution with pre-accumulated subhalo based values, load now
        if scope in ['subfind_global']:
            assert nChunks == 1 # no need for more
            assert ptRestriction is None # cannot apply particle restriction to subhalos
            particles = {}

            if len(ptProperty.split(" ")) == 3:
                species, ion, prop = ptProperty.split(" ")
                acName = 'Subhalo_%s_%s' % (prop.capitalize(), species+ion)
                print(' subfind_global: loading [%s] as effective particle data...' % acName)
            else:
                assert 0 # handle other cases, could e.g. call subhaloRadialReduction directly as:
                # (subhaloRadialReduction(sP,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof')

            particles[ptProperty] = cosmo.load.auxCat(sP, acName)[acName]
            particles['count'] = particles[ptProperty].size
            particles['Coordinates'] = gc['SubhaloPos']

            # for now assume full subhalo coverage, otherwise need to handle and remap subhalo positions 
            # based on ac['subhaloIDs'], e.g. for a subhaloRadialReduction(..., scope='fof', css='cen')
            assert particles['count'] == gc['header']['Nsubgroups_Total']
        else:
            indRangeSize = indRange[1] - indRange[0] + 1 # size of load

        if scope in ['global','global_fof']:
            # make mask corresponding to particles in all subhalos (other-halo term)
            subhalo_particle_mask = np.zeros( indRangeSize, dtype='int16' )

            # loop from where we previously stopped, to the end of all subhalos
            for i in range(prevMaskInd,gc['header']['Nsubgroups_Total']):
                ind0 = gc['SubhaloOffsetType'][i,ptLoadType] - indRange[0]
                ind1 = ind0 + gc['SubhaloLenType'][i,ptLoadType]

                # out of loaded chunk, stop marking for now
                if ind0 >= indRangeSize:
                    break

                # is this subhalo entirely outside the current chunk? [0,indRangeSize]
                if ind1 <= 0:
                    continue 

                # clip indices to be local to this loaded chunk
                ind0 = np.max([ind0, 0])
                ind1 = np.min([ind1, indRangeSize])

                # stamp
                subhalo_particle_mask[ind0:ind1] = 1

            prevMaskInd = i - 1

        # loop over subhalos
        for i, subhaloID in enumerate(subhaloIDsTodo):
            if i % np.max([1,int(nSubsDo/10.0)]) == 0 and i <= nSubsDo:
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

            # slice starting/ending indices for stars local to this subhalo/FoF
            i0 = 0
            i1 = particles['count'] # set default i1 to full chunk size for 'global' scope

            if scope in ['subfind','fof']:
                i0 = gc['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
                i1 = i0 + gc['SubhaloLenType'][subhaloID,ptLoadType]

                assert i0 >= 0 and i1 <= indRangeSize

                if i1 == i0:
                    continue # zero length of this type

            # use squared radii and sq distance function
            validMask = np.ones( i1-i0, dtype=np.bool )

            if proj2D is None:
                # apply in 3D
                rr = periodicDistsSq( gc['SubhaloPos'][subhaloID,:], particles['Coordinates'][i0:i1,:], sP )
            else:
                # apply in 2D projection, along the specified axis
                pt_2d = gc['SubhaloPos'][subhaloID,:]
                pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                vecs_2d = np.zeros( (i1-i0, 2), dtype=particles['Coordinates'].dtype )
                vecs_2d[:,0] = particles['Coordinates'][i0:i1,p_inds[0]]
                vecs_2d[:,1] = particles['Coordinates'][i0:i1,p_inds[1]]

                rr = periodicDistsSq( pt_2d, vecs_2d, sP ) # handles 2D

                # enforce depth restriction
                if proj2Ddepth is not None:
                    dist_projDir = particles['Coordinates'][i0:i1,p_inds[2]].copy() # careful of view
                    dist_projDir -= gc['SubhaloPos'][subhaloID,p_inds[2]]
                    correctPeriodicDistVecs(dist_projDir, sP)
                    validMask &= (np.abs(dist_projDir) <= proj2D_halfDepth)

            if scope in ['subfind_global']:
                # do not self count, we are accumulating the other-halo term
                validMask[subhaloID] = 0

            validMask &= (rr <= radMaxSqCode)

            if ptRestriction == 'real_stars':
                validMask &= (particles['GFM_StellarFormationTime'][i0:i1] >= 0.0)
            if ptRestriction == 'sfrgt0':
                validMask &= (particles['StarFormationRate'][i0:i1] > 0.0)
            if ptRestriction == 'sfreq0':
                validMask &= (particles['StarFormationRate'][i0:i1] == 0.0)

            wValid = np.where(validMask)

            if len(wValid[0]) == 0:
                continue # zero length of particles satisfying radial cut and restriction

            # log(radius), with any zero value set to small (included in first bin)
            loc_rr_log = logZeroSafe( rr[wValid], zeroVal=radMin-1.0 )
            loc_wt = particles['weights'][i0:i1][wValid] if weighting is not None else None

            if ptProperty not in userCustomFields:
                loc_val = particles[ptProperty][i0:i1][wValid]
            else:
                # user function reduction operations, set loc_val now
                if ptProperty == 'radvel':
                    p_pos  = np.squeeze( particles['Coordinates'][i0:i1,:][wValid,:] )
                    p_vel  = np.squeeze( particles['Velocities'][i0:i1,:][wValid,:] )

                    haloPos = gc['SubhaloPos'][subhaloID,:]
                    haloVel = gc['SubhaloVel'][subhaloID,:]

                    loc_val = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)

            # weighted histogram (or other op) of rr_log distances
            if scope in ['global','global_fof']:
                # (1) all
                result, _, _ = _binned_statistic_weighted(loc_rr_log, loc_val, statistic=op, bins=rbins_sq, weights=loc_wt)
                r[i,:,0] += result

                # (2) self-halo
                restoreSelf = False
                if gc['SubhaloLenType'][subhaloID,ptLoadType]:
                    is0 = gc['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
                    is1 = is0 + gc['SubhaloLenType'][subhaloID,ptLoadType]

                    # update mask to specifically mark this halo (do not include in the other-halo term)
                    if not ((is0 < 0 and is1 <= 0) or (is0 >= indRangeSize and is1 > indRangeSize)):
                        is0 = np.max([is0, 0])
                        is1 = np.min([is1, indRangeSize])
                        subhalo_particle_mask[is0:is1] = 2
                        restoreSelf = True

                # extract mask portion corresponding to current valid particle selection
                loc_mask = subhalo_particle_mask[wValid]

                w = np.where(loc_mask == 2)

                if len(w[0]):
                    # this subhalo at least partially in the currently loaded data
                    result, _, _ = _binned_statistic_weighted(loc_rr_log[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,1] += result

                # (3) other-halo
                w = np.where(loc_mask == 1)

                if len(w[0]):
                    result, _, _ = _binned_statistic_weighted(loc_rr_log[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,2] += result

                if restoreSelf:
                    subhalo_particle_mask[is0:is1] = 1 # restore

                # (4) diffuse
                w = np.where(loc_mask == 0)

                if len(w[0]):
                    result, _, _ = _binned_statistic_weighted(loc_rr_log[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,3] += result
            else:
                # subhalo/fof scope, we have only the self-term available, or 'subfind_global' technique
                result, _, _ = _binned_statistic_weighted(loc_rr_log, loc_val, statistic=op, bins=rbins_sq, weights=loc_wt)
                r[i,:] += result

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'ptProperty'  : ptProperty.encode('ascii'),
             'weighting'   : str(weighting).encode('ascii'),
             'rad_bins_code'    : rad_bins_code,
             'rad_bins_pkpc'    : rad_bins_pkpc,
             'rad_bin_edges'    : rad_bin_edges,
             'bin_volumes_code' : bin_volumes_code,
             'bin_areas_code'   : bin_areas_code,
             'subhaloIDs'       : subhaloIDsTodo}

    return r, attrs

# this dictionary contains a mapping between all auxCatalogs and their generating functions, where the 
# first sP,pSplit inputs are stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group_Mass_Crit500_Type' : \
     partial(fofRadialSumType,ptProperty='Masses',ptType='all',rad='Group_R_Crit500'),
   'Group_XrayBolLum_Crit500' : \
     partial(fofRadialSumType,ptProperty='xray_lum',ptType='gas',rad='Group_R_Crit500'),

   'Subhalo_Mass_30pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0),
   'Subhalo_Mass_100pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=100.0),
   'Subhalo_Mass_min_30pkpc_2rhalf_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='30h'),
   'Subhalo_Mass_puchwein10_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='p10'),
   'Subhalo_Mass_SFingGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestriction='sfrgt0'),

   'Subhalo_Mass_OV' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O V mass',op='sum',rad=None),
   'Subhalo_Mass_OVI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VI mass',op='sum',rad=None),
   #'Group_Mass_OVI' : \
   #  partial(subhaloRadialReduction,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof'),
   'Subhalo_Mass_OVII' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VII mass',op='sum',rad=None),
   'Subhalo_Mass_OVIII' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VIII mass',op='sum',rad=None),

   'Subhalo_Mass_AllGas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad=None),
   'Subhalo_Mass_AllGas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None),
   'Subhalo_Mass_AllGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None),
   'Subhalo_Mass_SF0Gas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad=None,ptRestriction='sfreq0'),
   'Subhalo_Mass_SF0Gas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestriction='sfreq0'),
   'Subhalo_Mass_SF0Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestriction='sfreq0'),
   'Subhalo_Mass_HaloGas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloStars_Metal' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metalmass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloStars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='mass',op='sum',rad='r015_1rvir_halo'),

   'Subhalo_Mass_50pkpc_Gas': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=50.0),
   'Subhalo_Mass_50pkpc_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=50.0),
   'Subhalo_Mass_250pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=250.0),
   'Subhalo_Mass_250pkpc_Gas_Global' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=250.0,scope='global'),
   'Subhalo_Mass_250pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=250.0),
   'Subhalo_Mass_250pkpc_Stars_Global' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=250.0,scope='global'),
   'Subhalo_Mass_r200_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad='r200crit'),
   'Subhalo_Mass_r200_Gas_Global' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad='r200crit',scope='global',minStellarMass=9.0),

   'Subhalo_CoolingTime_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',rad='r015_1rvir_halo',ptRestriction='sfreq0'),
   'Subhalo_CoolingTime_OVI_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',weighting='O VI mass',rad='r015_1rvir_halo',ptRestriction='sfreq0'),

   'Subhalo_XrayBolLum' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad=None),
   'Subhalo_XrayBolLum_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad='2rhalfstars'),
   'Subhalo_BH_Mass_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mass',op='max',rad=None),
   'Subhalo_BH_Mdot_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mdot',op='max',rad=None),
   'Subhalo_BH_MdotEdd_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_MdotEddington',op='max',rad=None),

   'Subhalo_SynchrotronPower_SKA' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska',op='sum',rad=None),
   'Subhalo_SynchrotronPower_SKA_eta43' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska_eta43',op='sum',rad=None),
   'Subhalo_SynchrotronPower_SKA_alpha15' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska_alpha15',op='sum',rad=None),
   'Subhalo_SynchrotronPower_VLA' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_vla',op='sum',rad=None),

   'Subhalo_StellarRotation' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad=None),
   'Subhalo_StellarRotation_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad='2rhalfstars'),
   'Subhalo_StellarRotation_1rhalfstars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad='1rhalfstars'),
   'Subhalo_GasRotation' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Krot',op='ufunc',rad=None),
   'Subhalo_GasRotation_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Krot',op='ufunc',rad='2rhalfstars'),

   'Subhalo_StellarAge_NoRadCut_MassWt'       : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='mass'),
   'Subhalo_StellarAge_NoRadCut_rBandLumWt' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=4.0,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_SDSSFiber_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_SDSSFiber4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='sdss_fiber_4pkpc',weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad=4.0,weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_SDSSFiber_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_SDSSFiber4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber_4pkpc',weighting='bandLum-sdss_r'),

   'Subhalo_StellarMeanVel' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='vel',op='mean',rad=None,weighting='mass'),

   'Subhalo_Bmag_SFingGas_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='mass',ptRestriction='sfrgt0'),
   'Subhalo_Bmag_SFingGas_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='volume',ptRestriction='sfrgt0'),
   'Subhalo_Bmag_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Bmag_2rhalfstars_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='volume'),
   'Subhalo_Bmag_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Bmag_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='volume'),

   'Subhalo_Temp_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='temp_linear',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Temp_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='temp_linear',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_nH_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_nH_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_nH_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Gas_RadialVel_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='radvel',op='ufunc',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Gas_RadialVel_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='radvel',op='ufunc',rad='2rhalfstars',weighting='mass'),

   'Subhalo_Pratio_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Pratio_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Pratio_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_uB_uKE_ratio_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_uB_uKE_ratio_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_uB_uKE_ratio_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='r015_1rvir_halo',weighting='volume'),

   'Subhalo_Ptot_gas_halo' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_gas_linear',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Ptot_B_halo' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_b_linear',op='sum',rad='r015_1rvir_halo'),


   'Subhalo_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none'),
   'Subhalo_StellarPhot_p07c_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00'),
   'Subhalo_StellarPhot_p07c_cf00dust_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', rad=30.0),
   'Subhalo_StellarPhot_p07k_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='none'),
   'Subhalo_StellarPhot_p07k_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='cf00'),
   'Subhalo_StellarPhot_p07s_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='none'),
   'Subhalo_StellarPhot_p07s_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='cf00'),

   'Subhalo_StellarPhot_p07c_cf00dust_res_eff_ns1' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_eff', Nside=1),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=1),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=1, rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00b_dust_res_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00b_res_conv', Nside=1, rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00dust_res3_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res3_conv', Nside=1, rad=30.0),

   'Subhalo_StellarPhot_p07c_ns8_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=8),
   'Subhalo_StellarPhot_p07c_ns4_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=4),
   'Subhalo_StellarPhot_p07c_ns8_demo_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=8, rad=30.0),
   'Subhalo_StellarPhot_p07c_ns4_demo_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=4, rad=30.0),

   'Subhalo_HalfLightRad_p07c_nodust' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside=None, sizes=True),
   'Subhalo_HalfLightRad_p07c_nodust_efr' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside='edgeon_faceon_rnd', sizes=True), 
   'Subhalo_HalfLightRad_p07c_cf00dust_efr' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='edgeon_faceon_rnd', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_efr_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='edgeon_faceon_rnd', rad=30.0, sizes=True),   
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='edgeon_faceon_rnd', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='edgeon_faceon_rnd', rad=30.0, sizes=True),

   'Particle_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', indivStarMags=True, Nside='z-axis'),

   'Subhalo_SDSSFiberSpectra_NoVel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis'),
   'Subhalo_SDSSFiberSpectra_Vel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=2, Nside='z-axis'),

   'Box_Grid_nHI'            : partial(wholeBoxColDensGrid,species='HI'),
   'Box_Grid_nHI_noH2'       : partial(wholeBoxColDensGrid,species='HI_noH2'),
   'Box_Grid_Z'              : partial(wholeBoxColDensGrid,species='Z'),
   'Box_Grid_nOVI'           : partial(wholeBoxColDensGrid,species='O VI'),
   'Box_Grid_nOVI_10'        : partial(wholeBoxColDensGrid,species='O VI 10'),
   'Box_Grid_nOVI_25'        : partial(wholeBoxColDensGrid,species='O VI 25'),
   'Box_Grid_nOVI_solar'     : partial(wholeBoxColDensGrid,species='O VI solar'),
   'Box_Grid_nOVII'          : partial(wholeBoxColDensGrid,species='O VII'),
   'Box_Grid_nOVIII'         : partial(wholeBoxColDensGrid,species='O VIII'),

   'Box_CDDF_nHI'            : partial(wholeBoxCDDF,species='HI'),
   'Box_CDDF_nHI_noH2'       : partial(wholeBoxCDDF,species='HI_noH2'),
   'Box_CDDF_nOVI'           : partial(wholeBoxCDDF,species='OVI'),
   'Box_CDDF_nOVI_10'        : partial(wholeBoxCDDF,species='OVI_10'),
   'Box_CDDF_nOVI_25'        : partial(wholeBoxCDDF,species='OVI_25'),
   'Box_CDDF_nOVI_solar'     : partial(wholeBoxCDDF,species='OVI_solar'),
   'Box_CDDF_nOVII'          : partial(wholeBoxCDDF,species='OVII'),
   'Box_CDDF_nOVIII'         : partial(wholeBoxCDDF,species='OVIII'),

   'Box_Grid_nOVI_depth10'           : partial(wholeBoxColDensGrid,species='O VI_depth10'),
   'Box_Grid_nOVI_10_depth10'        : partial(wholeBoxColDensGrid,species='O VI 10_depth10'),
   'Box_Grid_nOVI_25_depth10'        : partial(wholeBoxColDensGrid,species='O VI 25_depth10'),
   'Box_Grid_nOVI_solar_depth10'     : partial(wholeBoxColDensGrid,species='O VI solar_depth10'),
   'Box_Grid_nOVII_depth10'          : partial(wholeBoxColDensGrid,species='O VII_depth10'),
   'Box_Grid_nOVIII_depth10'         : partial(wholeBoxColDensGrid,species='O VIII_depth10'),
   'Box_CDDF_nOVI_depth10'           : partial(wholeBoxCDDF,species='OVI_depth10'),
   'Box_CDDF_nOVI_10_depth10'        : partial(wholeBoxCDDF,species='OVI_10_depth10'),
   'Box_CDDF_nOVI_25_depth10'        : partial(wholeBoxCDDF,species='OVI_25_depth10'),
   'Box_CDDF_nOVI_solar_depth10'     : partial(wholeBoxCDDF,species='OVI_solar_depth10'),
   'Box_CDDF_nOVII_depth10'          : partial(wholeBoxCDDF,species='OVII_depth10'),
   'Box_CDDF_nOVIII_depth10'         : partial(wholeBoxCDDF,species='OVIII_depth10'),

   'Box_Omega_HI'                    : partial(wholeBoxCDDF,species='H I',omega=True),
   'Box_Omega_H2'                    : partial(wholeBoxCDDF,species='H 2',omega=True),
   'Box_Omega_OVI'                   : partial(wholeBoxCDDF,species='O VI',omega=True),
   'Box_Omega_OVII'                  : partial(wholeBoxCDDF,species='O VII',omega=True),
   'Box_Omega_OVIII'                 : partial(wholeBoxCDDF,species='O VIII',omega=True),

   'Subhalo_SubLink_zForm_mm5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['mm',5,'snap']),
   'Subhalo_SubLink_zForm_ma5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['ma',5,'snap']),
   'Subhalo_SubLink_zForm_poly7' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['poly',7,'snap']),

   'Subhalo_Tracers_zAcc_mean'   : partial(tracerTracksQuant,quant='acc_time_1rvir',op='mean',time=None),
   'Subhalo_Tracers_dtHalo_mean' : partial(tracerTracksQuant,quant='dt_halo',op='mean',time=None),
   'Subhalo_Tracers_angmom_tAcc' : partial(tracerTracksQuant,quant='angmom',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_entr_tAcc'   : partial(tracerTracksQuant,quant='entr',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_temp_tAcc'   : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_tempTviracc_tAcc' : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir',norm='tvir_tacc'),
   'Subhalo_Tracers_tempTvircur_tAcc' : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir',norm='tvir_cur'),

   'Subhalo_BH_CumEgyInjection_Low' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumEgyInjection_RM',op='sum',rad=None),
   'Subhalo_BH_CumEgyInjection_High' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumEgyInjection_QM',op='sum',rad=None),
   'Subhalo_BH_CumMassGrowth_Low' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumMassGrowth_RM',op='sum',rad=None),
   'Subhalo_BH_CumMassGrowth_High' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumMassGrowth_QM',op='sum',rad=None),

   'Subhalo_RadProfile3D_Global_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global_fof'), 
   'Subhalo_RadProfile3D_SubfindGlobal_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind_global'), 
   'Subhalo_RadProfile3D_Subfind_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind'),
   'Subhalo_RadProfile3D_FoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='fof'),

   'Subhalo_RadProfile2Dz_2Mpc_Global_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global_fof',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_Subfind_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_FoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_Global_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global_fof',proj2D=[2,2000]),
   'Subhalo_RadProfile3D_Global_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global_fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_Global_Gas_O_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass_O',op='sum',scope='global'),
   'Subhalo_RadProfile3D_Global_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_Global_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global'),
   'Subhalo_RadProfile2Dz_2Mpc_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global',proj2D=[2,2000]),
  }

# this list contains the names of auxCatalogs which are computed manually (e.g. require more work than 
# a single generative function), but are then saved in the same format and so can be loaded normally
manualFieldNames = \
[   'Subhalo_SDSSFiberSpectraFits_NoVel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_NoVel-Realism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z'
]
