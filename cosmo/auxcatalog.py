"""
cosmo/auxcatalog.py
  Cosmological simulations - auxiliary catalog for additional derived galaxy/halo properties.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from functools import partial
from os.path import expanduser
from getpass import getuser

from cosmo.util import snapNumToRedshift, subhaloIDListToBoundingPartIndices, \
  inverseMapPartIndicesToSubhaloIDs, inverseMapPartIndicesToHaloIDs, correctPeriodicDistVecs
from util.helper import logZeroMin, logZeroNaN, logZeroSafe, weighted_std_binned
from util.helper import pSplit as pSplitArr, pSplitRange, numPartToChunkLoadSize
from util.rotation import rotateCoordinateArray, rotationMatrixFromVec, momentOfInertiaTensor, \
  rotationMatricesFromInertiaTensor, ellipsoidfit

# generative functions
from projects.outflows_analysis import instantaneousMassFluxes, massLoadingsSN, outflowVelocities

""" Relatively 'hard-coded' analysis decisions that can be changed. For reference, they are attached 
    as metadata attributes in the auxCat file. """

# Subhalo_*: parameters for computations done over each Subfind subhalo

# Box_*: parameters for whole box (halo-independent) computations
boxGridSizeHI     = 1.5 # code units, e.g. ckpc/h
boxGridSizeMetals = 5.0 # code units, e.g. ckpc/h

# todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
# eliminate the entire specialized ufunc logic herein
userCustomFields = ['Krot','radvel','losvel','losvel_abs','shape_ellipsoid']

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
    desc  += "Type indices: " + " ".join([t+'='+str(i) for t,i in ptSaveTypes.items()]) + "."
    select = "All FoF halos."

    # load group information
    gc = sP.groupCat(fieldsHalos=['GroupPos','GroupLen','GroupLenType',rad])
    gc['GroupOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']

    h = sP.snapshotHeader()

    nGroupsTot = sP.numHalos
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
    gc[rad] = gc[rad] * gc[rad]

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
                dm = sP.snapshotSubsetP(partType='dm', fields=['pos'], haloID=haloID, sq=False)

                if dm['count']:
                    rDM = sP.periodicDistsSq( gc['GroupPos'][haloID,:], dm['Coordinates'] )
                    wDM = np.where( rDM <= gc[rad][haloID] )

                    r[i, ptSaveTypes['dm']] = len(wDM[0]) * h['MassTable'][ptLoadTypes['dm']]

            # GAS
            if 'gas' in ptLoadTypes:
                gas = sP.snapshotSubsetP(partType='gas', fields=['pos',ptProperty], haloID=haloID)
                assert gas[ptProperty].ndim == 1

                if gas['count']:
                    rGas = sP.periodicDistsSq( gc['GroupPos'][haloID,:], gas['Coordinates'] )
                    wGas = np.where( rGas <= gc[rad][haloID] )

                    r[i, ptSaveTypes['gas']] = np.sum( gas[ptProperty][wGas] )

            # STARS
            if 'stars' in ptLoadTypes:
                stars = sP.snapshotSubsetP(partType='stars', fields=['pos','sftime',ptProperty], haloID=haloID)
                assert stars[ptProperty].ndim == 1

                if stars['count']:
                    rStars = sP.periodicDistsSq( gc['GroupPos'][haloID,:], stars['Coordinates'] )
                    wWind  = np.where( (rStars <= gc[rad][haloID]) & (stars['GFM_StellarFormationTime'] < 0.0) )
                    wStars = np.where( (rStars <= gc[rad][haloID]) & (stars['GFM_StellarFormationTime'] >= 0.0) )

                    r[i, ptSaveTypes['gas']] += np.sum( stars[ptProperty][wWind] )
                    r[i, ptSaveTypes['stars']] = np.sum( stars[ptProperty][wStars] )

    if method == 'B':
        # (A): DARK MATTER
        if 'dm' in ptLoadTypes:
            print(' [DM]')
            dm = sP.snapshotSubsetP(partType='dm', fields=['pos'], sq=False, indRange=indRange['dm'])

            if ptProperty == 'Masses':
                dm[ptProperty] = np.zeros( dm['count'], dtype='float32' ) + h['MassTable'][ptLoadTypes['dm']]
            else:
                dm[ptProperty] = sP.snapshotSubsetP(partType='dm', fields=ptProperty, haloSubset=True)

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for dm local to this FoF
                i0 = gc['GroupOffsetType'][haloID,ptLoadTypes['dm']] - indRange['dm'][0]
                i1 = i0 + gc['GroupLenType'][haloID,ptLoadTypes['dm']]

                assert i0 >= 0 and i1 <= (indRange['dm'][1]-indRange['dm'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = sP.periodicDistsSq( gc['GroupPos'][haloID,:], dm['Coordinates'][i0:i1,:] )
                ww = np.where( rr <= gc[rad][haloID] )

                r[i, ptSaveTypes['dm']] = np.sum( dm[ptProperty][i0:i1][ww] )
            del dm

        # (B): GAS
        if 'gas' in ptLoadTypes:
            print(' [GAS]')
            gas = sP.snapshotSubsetP(partType='gas', fields=['pos'], sq=False, indRange=indRange['gas'])
            gas[ptProperty] = sP.snapshotSubsetP(partType='gas', fields=ptProperty, indRange=indRange['gas'])
            assert gas[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for gas local to this FoF
                i0 = gc['GroupOffsetType'][haloID,ptLoadTypes['gas']] - indRange['gas'][0]
                i1 = i0 + gc['GroupLenType'][haloID,ptLoadTypes['gas']]

                assert i0 >= 0 and i1 <= (indRange['gas'][1]-indRange['gas'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = sP.periodicDistsSq( gc['GroupPos'][haloID,:], gas['Coordinates'][i0:i1,:] )
                ww = np.where( rr <= gc[rad][haloID] )

                r[i, ptSaveTypes['gas']] = np.sum( gas[ptProperty][i0:i1][ww] )
            del gas

        # (C): STARS
        if 'stars' in ptLoadTypes:
            print(' [STARS]')
            stars = sP.snapshotSubsetP( partType='stars', fields=['pos','sftime'], sq=False, indRange=indRange['stars'])
            stars[ptProperty] = sP.snapshotSubsetP( partType='stars', fields=ptProperty, indRange=indRange['stars'])
            assert stars[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if i % int(nHalosDo/10) == 0 and i <= nHalosDo:
                    print('  %4.1f%%' % (float(i+1)*100.0/nHalosDo))

                # slice starting/ending indices for stars local to this FoF
                i0 = gc['GroupOffsetType'][haloID,ptLoadTypes['stars']] - indRange['stars'][0]
                i1 = i0 + gc['GroupLenType'][haloID,ptLoadTypes['stars']]

                assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

                if i1 == i0:
                    continue # zero length of this type

                rr = sP.periodicDistsSq( gc['GroupPos'][haloID,:], stars['Coordinates'][i0:i1,:] )
                wWind  = np.where( (rr <= gc[rad][haloID]) & (stars['GFM_StellarFormationTime'][i0:i1] < 0.0) )
                wStars = np.where( (rr <= gc[rad][haloID]) & (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) )

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
    slit_code = None # used to return aperture geometry for weighted inclusions

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
        gcLoad = sP.groupCat(fieldsHalos=['Group_M_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentM200 = gcLoad['halos'][gcLoad['subhalos']]

        # r_cut = 27.3 kpc/h * (M200crit / (10^15 Msun/h))^0.29 from Puchwein+ (2010) Eqn 1
        r_cut = 27.3 * (parentM200/1e5)**(0.29) / sP.HubbleParam
        radSqMax = r_cut * r_cut
    elif rad == '30h':
        # hybrid, minimum of [constant scalar 30 pkpc] and [the usual, 2rhalf,stars]
        rad_pkpc = sP.units.physicalKpcToCodeLength(30.0)

        subHalfmassRadType = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * subHalfmassRadType[:,sP.ptNum('stars')]

        ww = np.where(twiceStellarRHalf > rad_pkpc)
        twiceStellarRHalf[ww] = rad_pkpc
        radSqMax = twiceStellarRHalf**2.0
    elif rad == '10pkpc_slice':
        # slice at 10 +/- 2 pkpc
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += (sP.units.physicalKpcToCodeLength(12.0))**2
        radSqMin += (sP.units.physicalKpcToCodeLength(8.0))**2
    elif rad == 'r015_1rvir_halo':
        # classic 'halo' definition, 0.15rvir < r < 1.0rvir (meaningless for non-centrals)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
        radSqMin = (0.15 * parentR200)**2
    elif rad == 'r200crit':
        # within the virial radius (r200,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
    elif rad == '2rhalfstars':
        # classic Illustris galaxy definition, r < 2*r_{1/2,mass,stars}
        subHalfmassRadType = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * subHalfmassRadType[:,sP.ptNum('stars')]

        radSqMax = twiceStellarRHalf**2
    elif rad == '1rhalfstars':
        # inner galaxy definition, r < 1*r_{1/2,mass,stars}
        subHalfmassRadType = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 1.0 * subHalfmassRadType[:,sP.ptNum('stars')]

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
    elif rad == 'legac_slit':
        # slit is 1" x 8" minimum (length is variable and depends on galaxy size? TODO)
        slit_arcsec = np.array([1.0, 4.0])
        slit_kpc = sP.units.arcsecToAngSizeKpcAtRedshift(slit_arcsec, z=sP.redshift) # arcsec -> pkpc
        slit_code = sP.units.physicalKpcToCodeLength(slit_kpc) # pkpc -> ckpc/h

        radSqMax = np.zeros( (nSubsTot,2), dtype='float32' ) # second dim: (x,y) in projection
        radSqMax[:,0] += (slit_code[0]/2.0)**2
        radSqMax[:,1] += (slit_code[1]/2.0)**2
        radRestrictIn2D = True

    assert radSqMax.size == nSubsTot or (radSqMax.size == nSubsTot*2 and radSqMax.ndim == 2)
    assert radSqMin.size == nSubsTot or (radSqMIn.size == nSubsTot*2 and radSqMin.ndim == 2)

    return radRestrictIn2D, radSqMin, radSqMax, slit_code

def _pSplitBounds(sP, pSplit, Nside, indivStarMags, minStellarMass):
    """ For a given pSplit = [thisTaskNum,totNumOfTasks], determine an efficient work split and 
    return the required processing for this task, in the form of the list of subhaloIDs to 
    process and the global snapshot index range required in load to cover these subhalos. """
    nSubsTot = sP.numSubhalos
    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    # stellar mass select
    if minStellarMass is not None:
        masses = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])
        masses = masses[subhaloIDsTodo]
        with np.errstate(invalid='ignore'):
            wSelect = np.where( masses >= minStellarMass )

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if isinstance(Nside, int) and Nside > 1:
        # special case: just do a few special case subhalos at high Nside for demonstration
        assert sP.res == 1820 and sP.run == 'tng' and sP.snap == 99 and pSplit is None
        #gcDemo = sP.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'])
        #massesDemo = sP.units.codeMassToLogMsun( gcDemo['halos']['Group_M_Crit200'] )
        #ww = np.where( (massesDemo >= 11.9) & (massesDemo < 12.1) ) # ww[0]= [597, 620, 621...]
        #ww = np.where( (massesDemo >= 13.4) & (massesDemo < 13.5) ) # ww[0]= [34, 52, ...]
        # two massive + three MW-mass halos, SubhaloSFR = [0.2, 5.2, 1.7, 5.0, 1.1] Msun/yr
        subhaloIDsTodo = [172649,208781,412332,415496,415628] # gc['halos']['GroupFirstSub'][inds]

        indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    invSubs = [0,0]

    if pSplit is not None:
        if 0:
            # split up subhaloIDs in round-robin scheme (equal number of massive/centrals per job)
            # works perfectly for balance, but retains global load of all haloSubset particles
            modSplit = subhaloIDsTodo % pSplit[1]
            subhaloIDsTodo = np.where(modSplit == pSplit[0])[0]

        # already only have a subset of all subhalos? (from minStellarMass)
        if subhaloIDsTodo.size < nSubsTot:
            # do contiguous subhalo ID division and reduce global haloSubset load 
            # to the particle sets which cover the subhalo subset of this pSplit, but the issue is 
            # that early tasks take all the large halos and all the particles, very imbalanced
            subhaloIDsTodo = pSplitArr( subhaloIDsTodo, pSplit[1], pSplit[0] )

            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)
        else:
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

            lastPrevSub = sP.groupCatSingle(subhaloID=invSubs[0]-1)
            indRange['stars'][0] += lastPrevSub['SubhaloLenType'][ sP.ptNum('stars') ]
        else:
            indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo, strictSubhalos=True)

    return subhaloIDsTodo, indRange

def subhaloRadialReduction(sP, pSplit, ptType, ptProperty, op, rad, 
                           ptRestrictions=None, weighting=None, scope='subfind', minStellarMass=None):
    """ Compute a reduction operation (either total/sum or weighted mean) of a particle property (e.g. mass) 
        for those particles of a given type enclosed within a fixed radius (input as a scalar, in physical 
        kpc, or as a string specifying a particular model for a variable cut radius). 
        Restricted to subhalo particles only if scope=='subfind' (default), or FoF particles if scope=='fof'.
        If scope=='global', currently a full non-chunked snapshot load and brute-force distance 
        computations to all particles for each subhalo.
        ptRestrictions apply further cuts to which particles are included, as a dictionary, where each 
        key is the particle field to apply the restriction to, and the value is a list of two entries, 
        the first being the inequality to apply ('gt','lt','eq') and the second the numeric value to compare to.
        If minStellarMass is not None, then only process subhalos with mstar_30pkpc_log above this value.
    """
    assert op in ['sum','mean','max','ufunc','halfrad']
    assert scope in ['subfind','fof','global']
    if op == 'ufunc': assert ptProperty in userCustomFields

    # determine ptRestriction
    if ptType == 'stars':
        if ptRestrictions is None:
            ptRestrictions = {}
        ptRestrictions['GFM_StellarFormationTime'] = ['gt',0.0] # real stars

    # config
    ptLoadType = sP.ptNum(ptType)

    desc   = "Quantity [%s] enclosed within a radius of [%s] for [%s]." % (ptProperty,rad,ptType)
    if ptRestrictions is not None:
        desc += " (restriction = %s). " % ','.join([r for r in ptRestrictions])
    if weighting is not None:
        desc += " (weighting = %s). " % weighting
    if scope == 'subfind': desc  +=" (only subhalo particles included). "
    if scope == 'fof': desc  +=" (all parent FoF particles included). "
    if scope == 'global': desc  +=" (all global particles included). "
    select = "All Subhalos."
    if minStellarMass is not None: select += ' (Only with stellar mass >= %.2f)' % minStellarMass

    # load group information
    gc = sP.groupCat(fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']
    nSubsTot = sP.numSubhalos

    if scope == 'fof':
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        GroupLenType = sP.groupCat(fieldsHalos=['GroupLenType'])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=['SubhaloGrNr'])

        gc['SubhaloLenType'] = GroupLenType[SubhaloGrNr,:]
        gc['SubhaloOffsetType'] = GroupOffsetType[SubhaloGrNr,:]

    # determine radial restriction for each subhalo
    radRestrictIn2D, radSqMin, radSqMax, _ = _radialRestriction(sP, nSubsTot, rad)

    if radRestrictIn2D:
        Nside = 'z-axis'
        print(' Requested: radRestrictIn2D! Using hard-coded projection direction of [%s]!' % Nside)

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange = _pSplitBounds(sP, pSplit, None, False, minStellarMass)
    nSubsDo = len(subhaloIDsTodo)

    indRange = indRange[ptType] # choose index range for the requested particle type

    if scope == 'global':
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = [0, h['NumPart'][sP.ptNum(ptType)]-1]
        i0 = 0 # never changes
        i1 = indRange[1] # never changes

    # info
    username = getuser()
    if username == 'dnelson':
        print(' ' + desc)
        print(' Total # Subhalos: %d, processing [%d] subhalos...' % (nSubsTot,nSubsDo))

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = []

    if rad is not None or op == 'halfrad':
        fieldsLoad.append('pos')

    if ptRestrictions is not None:
        for restrictionField in ptRestrictions:
            fieldsLoad.append(restrictionField)

    if ptProperty == 'Krot':
        fieldsLoad.append('pos')
        fieldsLoad.append('vel')
        fieldsLoad.append('mass')
        allocSize = (nSubsDo,4)

    if ptProperty == 'radvel':
        fieldsLoad.append('pos')
        fieldsLoad.append('vel')
        gc['SubhaloVel'] = sP.groupCat(fieldsSubhalos=['SubhaloVel'])
        allocSize = (nSubsDo,)

    if ptProperty == 'shape_ellipsoid':
        gc['SubhaloRhalfStars'] = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])[:,sP.ptNum('stars')]
        ellipsoid_rin  = 1.8 # rhalfstars
        ellipsoid_rout = 2.2 # rhalfstars
        fieldsLoad.append('pos')
        allocSize = (nSubsDo,2) # q,s

    fieldsLoad = list(set(fieldsLoad)) # make unique

    particles = {}
    if len(fieldsLoad):
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

    if op != 'ufunc':
        # todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
        # eliminate the entire specialized ufunc logic herein
        particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)

    if 'count' not in particles:
        key = list(particles.keys())[0]
        particles['count'] = particles[key].shape[0]

    # allocate, NaN indicates not computed except for mass where 0 will do
    dtype = particles[ptProperty].dtype if ptProperty in particles.keys() else 'float32' # for custom
    assert dtype in ['float32','float64'] # otherwise check, when does this happen?

    if op == 'ufunc': 
        r = np.zeros( allocSize, dtype=dtype )
    else:
        if particles[ptProperty].ndim == 1:
            r = np.zeros( nSubsDo, dtype=dtype )
        else:
            r = np.zeros( (nSubsDo,particles[ptProperty].shape[1]), dtype=dtype )

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
            magsLoad = sP.snapshotSubset(partType=ptType, fields=fieldsLoadMag, indRange=indRange)

            # request magnitudes in this band
            band = weighting.split("-")[1]
            mags = pop.mags_code_units(sP, band, particles['GFM_StellarFormationTime'], 
                                     magsLoad['GFM_Metallicity'], 
                                     magsLoad['GFM_InitialMass'], retFullSize=True)

            # use the (linear) luminosity in this band as the weight
            particles['weights'] = sP.units.absMagToLuminosity(mags)

        else:
            # use a particle quantity as weights (e.g. 'mass', 'volume', 'O VI mass')
            particles['weights'] = sP.snapshotSubset(partType=ptType, fields=weighting, indRange=indRange)

    assert particles['weights'].ndim == 1 and particles['weights'].size == particles['count']

    # loop over subhalos
    printFac = 100.0 if (sP.res > 512 or scope == 'global') else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo and username == 'dnelson':
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo)) 

        # slice starting/ending indices for stars local to this FoF
        if scope != 'global':
            i0 = gc['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
            i1 = i0 + gc['SubhaloLenType'][subhaloID,ptLoadType]

        assert i0 >= 0 and i1 <= (indRange[1]-indRange[0]+1)

        if i1 == i0:
            continue # zero length of this type

        # use squared radii and sq distance function
        validMask = np.ones( i1-i0, dtype=np.bool )

        if rad is not None or op == 'halfrad':

            if not radRestrictIn2D:
                # apply in 3D
                rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], particles['Coordinates'][i0:i1,:] )
            else:
                # apply in 2D projection, limited support for now, just Nside='z-axis'
                # otherwise, for any more complex projection, need to apply it here, and anyways 
                # for nProj>1, this validMask selection logic becomes projection dependent, so 
                # need to move it inside the range(nProj) loop, which is definitely doable
                assert Nside == 'z-axis'
                p_inds = [0,1] # x,y
                pt_2d = gc['SubhaloPos'][subhaloID,:]
                pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                vecs_2d = np.zeros( (i1-i0, 2), dtype=particles['Coordinates'].dtype )
                vecs_2d[:,0] = particles['Coordinates'][i0:i1,p_inds[0]]
                vecs_2d[:,1] = particles['Coordinates'][i0:i1,p_inds[1]]

                rr = sP.periodicDistsSq( pt_2d, vecs_2d ) # handles 2D

            if rad is not None:
                validMask &= (rr <= radSqMax[subhaloID])
                validMask &= (rr >= radSqMin[subhaloID])

        # apply particle-level restrictions
        if ptRestrictions is not None:
            for restrictionField in ptRestrictions:
                inequality, val = ptRestrictions[restrictionField]

                if inequality == 'gt':
                    validMask &= (particles[restrictionField][i0:i1] > val)
                if inequality == 'lt':
                    validMask &= (particles[restrictionField][i0:i1] <= val)
                if inequality == 'eq':
                    validMask &= (particles[restrictionField][i0:i1] == val)

        wValid = np.where(validMask)

        if len(wValid[0]) == 0:
            continue # zero length of particles satisfying radial cut and restriction

        # user function reduction operations
        if op in ['ufunc','halfrad']:
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

                haloPos = gc['SubhaloPos'][subhaloID,:]
                haloVel = gc['SubhaloVel'][subhaloID,:]

                vrad = sP.units.particleRadialVelInKmS(gas_pos, gas_vel, haloPos, haloVel)

                r[i] = np.average(vrad, weights=gas_weights)

            # ufunc: 'half radius' of the quantity
            if op == 'halfrad':
                loc_val = particles[ptProperty][i0:i1][wValid]
                loc_rad = np.sqrt( rr[wValid] )
                r[i] = _findHalfLightRadius(loc_rad, mags=None, vals=loc_val)

            # shape measurement via iterative ellipsoid fitting
            if ptProperty == 'shape_ellipsoid':
                scale_rad = gc['SubhaloRhalfStars'][subhaloID]
                if scale_rad == 0.0:
                    continue

                loc_val = particles['Coordinates'][i0:i1, :][wValid]
                loc_wt  = particles['weights'][i0:i1][wValid] # mass

                # positions relative to subhalo center, and normalized by stellar half mass radius
                for j in range(3):
                    loc_val[:,j] -= gc['SubhaloPos'][subhaloID,j]

                sP.correctPeriodicDistVecs(loc_val)
                loc_val /= scale_rad

                # fit, and save ratios of second and third axes lengths to major axis
                q, s, _, _ = ellipsoidfit(loc_val, loc_wt, scale_rad, ellipsoid_rin, ellipsoid_rout)
                r[i,0] = q
                r[i,1] = s

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

def _findHalfLightRadius(rad,mags,vals=None):
    """ Helper function, linearly interpolate in rr (squared radii) to find the half light 
    radius, given the magnitudes mags[i] corresponding to each star particle at rad[i]. 
    Will give 3D or 2D radii exactly if rad is input 3D or 2D. """
    if mags is not None: assert rad.size == mags.size

    if vals is None:
        # convert individual mags to luminosities [arbitrary units]
        lums = np.power(10.0, -0.4 * mags)
    else:
        # take input values unchanged (assume e.g. linear masses or light quantities already)
        assert vals.size == rad.size
        lums = vals.copy()

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

def subhaloStellarPhot(sP, pSplit, iso=None, imf=None, dust=None, Nside=1, rad=None, modelH=True, bands=None,
                       sizes=False, indivStarMags=False, fullSubhaloSpectra=False, redshifted=False, emlines=False,
                       seeing=None, minStellarMass=None):
    """ Compute the total band-magnitudes (or instead half-light radii if sizes==True), per subhalo, 
    under the given assumption of an iso(chrone) model, imf model, dust model, and radial restrction. 
    If using a dust model, include multiple projection directions per subhalo. If indivStarMags==True, 
    then save the magnitudes for every Pt4 (wind->NaN) in all subhalos. If fullSubhaloSpectra==True, 
    then save a full spectrum vs wavelength for every subhalo. If redshifted is True, then all the 
    stellar spectra/magnitudes are computed at sP.redshift and the band filters are then applied, 
    resulting in e.g apparent magnitudes, otherwise the stars are assumed to be at z=0, spectra 
    are e.g. rest-frame and magnitudes are absolute. If emlines, then include nebular emission lines 
    in either band-magnitudes or full spectra, otherwise exclude.
    If seeing is not None, then instead of a binary inclusion/exclusion of each star particle 
    based on the 'rad' aperture, include all stars weighted by the fraction of their light which 
    enters the 'rad' aperture, assuming it is spread by atmospheric seeing into a Gaussian with a 
    sigma of seeing [units of arcseconds at sP.redshift].
    If minStellarMass is not None, then only process subhalos with mstar_30pkpc_log above this value. """
    from cosmo.stellarPop import sps
    from healpy.pixelfunc import nside2npix, pix2vec
    from cosmo.hydrogen import hydrogenMass

    # mutually exclusive options, at most one can be enabled
    assert sum([sizes,indivStarMags,np.clip(fullSubhaloSpectra,0,1)]) in [0,1]

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust, redshifted=redshifted, emlines=emlines)

    # which bands? for now, to change, just recompute from scratch
    if bands is None:
        bands = []
        bands += ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
        #bands += ['wfcam_y','wfcam_j','wfcam_h','wfcam_k'] # UKIRT IR wide
        bands += ['wfc_acs_f606w','wfc3_ir_f125w','wfc3_ir_f140w','wfc3_ir_f160w'] # HST IR wide
        bands += ['jwst_f070w','jwst_f090w','jwst_f115w','jwst_f150w','jwst_f200w','jwst_f277w','jwst_f356w','jwst_f444w'] # JWST IR (NIRCAM) wide

        if indivStarMags or sizes: bands = ['sdss_r','jwst_f150w']

    nBands = len(bands)

    if fullSubhaloSpectra:
        # set nBands to size of wavelength grid within SDSS/LEGA-C spectral range
        if 'sdss' in rad:
            spec_min_ang = 3000.0
            spec_max_ang = 10000.0

            output_wave = pop.wave_ang # at intrinsic stellar library model resolution / wavelength grid

            # enforced in rest-frame if redshifted is False, otherwise in observed-frame (because if redshifted is True, 
            # then pop.wave_ang corresponds to observed-frame wavelengths for the spectra returned by pop.dust_tau_model_mags)
            ww = np.where( (pop.wave_ang >= spec_min_ang) & (pop.wave_ang <= spec_max_ang) )[0]
            spec_min_ind = ww.min()
            spec_max_ind = ww.max() + 1
            nBands = spec_max_ind - spec_min_ind

        if 'legac' in rad:
            # load lega-c dr2 wavelength grid
            with h5py.File(expanduser("~") + '/obs/legac_dr2_spectra_wave.hdf5','r') as f:
                output_wave = f['wavelength'][()]

            spec_min_ang = output_wave.min()
            spec_max_ang = output_wave.max()

            spec_min_ind = 0
            spec_max_ind = output_wave.size
            nBands = output_wave.size

        # only for resolved dust models do we currently calculate full spectra of every star particle
        assert '_res' in dust

        # if fullSubhaloSpectra == 2, we include the peculiar motions of stars (e.g. veldisp)
        # in which case the rel_vel here is overwritten on a per subhalo basis below
        rel_vel_los = None

    # which projections?
    nProj = 1
    efrDirs = False

    if '_res' in dust or sizes is True:
        if isinstance(Nside, int):
            # numeric Nside -> healpix vertices as projection vectors
            nProj = nside2npix(Nside)
            projVecs = pix2vec(Nside,range(nProj),nest=True)
            projVecs = np.transpose( np.array(projVecs, dtype='float32') ) # Nproj,3
            projDesc = '2D projections.'
        else:
            # string Nside -> custom projection vectors
            if Nside == 'efr2d':
                projDesc = '2D: edge-on, face-on, edge-on-smallest, edge-on-random, random.'
                nProj = 5

                Nside = Nside.encode('ascii') # for hdf5 attr save
                projVecs = np.zeros( (nProj,3), dtype='float32' ) # derive per subhalo
                efrDirs = True
                assert (sizes is True) or ('_res' in dust) # only cases where efr logic exists for now
            elif Nside == 'z-axis':
                projDesc = '2D: single projection along z-axis of simulation box.'
                nProj = 1
                projVecs = np.array( [0,0,1], dtype='float32' ).reshape(1,3) # [nProj,3]                
            elif Nside == None:
                projDesc = '3D projection.'
                pass # no 2D radii
            else:
                assert 0 # unhandled

    # prepare catalog metadata
    desc = "Stellar light emission (total AB magnitudes) by subhalo, multiple bands."
    if sizes: desc = "Stellar half light radii (code units) by subhalo, multiple bands. " + projDesc
    if indivStarMags: desc = "Star particle individual AB magnitudes, multiple bands."
    if fullSubhaloSpectra:
        desc = "Optical spectra by subhalo, [%d] wavelength points between [%.1f Ang] and [%.1f Ang]." % \
            (nBands,spec_min_ang,spec_max_ang)
    if redshifted:
        desc += " Redshifted, observed-frame bands/wavelengths, apparent magnitudes/luminosities."
    else:
        desc += " Unredshifted, rest-frame bands/wavelengths, absolute magnitudes/luminosities."
    if seeing is not None:
        desc += ' Weighted contributions incorporating atmospheric seeing of [%.1f arcsec].' % seeing

    select = "All Subfind subhalos"
    if minStellarMass is not None: select += ' (Only with stellar mass >= %.2f)' % minStellarMass
    if indivStarMags: select = "All PartType4 particles in all subhalos"
    select += " (numProjectionsPer = %d) (%s)." % (nProj, Nside)

    print(' %s\n %s' % (desc,select))

    # load group information
    gc = sP.groupCat(fieldsSubhalos=['SubhaloLenType','SubhaloHalfmassRadType','SubhaloPos'])
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']
    nSubsTot = sP.numSubhalos

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange = _pSplitBounds(sP, pSplit, Nside, indivStarMags, minStellarMass)

    nSubsDo = len(subhaloIDsTodo)
    partInds = None

    print(' Total # Subhalos: %d, processing [%d] in [%d] bands and [%d] projections...' % \
        (nSubsTot,nSubsDo,nBands,nProj))

    # allocate
    if indivStarMags:
        # compute number of PT4 particles we will do (cover full PT4 size)
        nPt4Tot = sP.snapshotHeader()['NumPart'][sP.ptNum('stars')]
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
    radRestrictIn2D, radSqMin, radSqMax, radRestrict_sizeCode = _radialRestriction(sP, nSubsTot, rad)
    assert radSqMin.max() == 0.0 # not handled here

    # spread light of stars into gaussians based on atmospheric seeing?
    if seeing is not None:
        assert rad is not None # meaningless
        assert '_res' in dust # otherwise generalize
        assert radRestrictIn2D # only makes sense in 2d projection
        if indivStarMags or sizes: raise Exception('What does it mean?')

        nint = 100 # integration accuracy parameter
        seeing_pkpc =  sP.units.arcsecToAngSizeKpcAtRedshift(seeing, z=sP.redshift) # arcsec -> pkpc
        seeing_code = sP.units.physicalKpcToCodeLength(seeing_pkpc) # pkpc -> ckpc/h

        seeing_const1 = 1.0/(2*np.pi*seeing_code**2)
        seeing_const2 = (-1.0/(2*seeing_code**2))
        def _seeing_func(x, y):
            """ 2D Gaussian, integrand for determining overlap with collecting aperture. """
            return seeing_const1 * np.exp((x*x+y*y)*seeing_const2)

    # global load of all stars in all groups in snapshot
    starsLoad = ['initialmass','sftime','metallicity']
    if '_res' in dust or rad is not None or sizes is not None: starsLoad += ['pos']
    if sizes: starsLoad += ['mass']
    if fullSubhaloSpectra == 2: starsLoad += ['vel','masses'] # masses is the current weight for LOS mean vel

    stars = sP.snapshotSubsetP(partType='stars', fields=starsLoad, indRange=indRange['stars'])

    printFac = 100.0 if sP.res > 512 else 10.0

    # non-resolved dust: loop over all requested bands first
    if '_res' not in dust:
        if sizes:
            gas = sP.snapshotSubsetP('gas', fields=['pos','mass','sfr'], indRange=indRange['gas'])

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
                i0 = gc['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange['stars'][0]
                i1 = i0 + gc['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

                assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

                if i1 == i0:
                    continue # zero length of this type
                
                # radius restriction: use squared radii and sq distance function
                validMask = np.ones( i1-i0, dtype=np.bool )
                if rad is not None:
                    assert radSqMax.ndim == 1 # otherwise generalize like below for '_res'
                    rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], stars['Coordinates'][i0:i1,:] )
                    validMask &= (rr <= radSqMax[subhaloID])
                wValid = np.where(validMask)
                if len(wValid[0]) == 0:
                    continue # zero length of particles satisfying radial cut and restriction

                magsLocal = mags[i0:i1][wValid] # wind particles still here, and have NaN

                if not sizes and not indivStarMags:
                    # convert mags to luminosities, sum together
                    totalLum = np.nansum( sP.units.absMagToLuminosity(magsLocal) )

                    # convert back to a magnitude in this band
                    if totalLum > 0.0:
                        r[i,bandNum,0] = sP.units.lumToAbsMag(totalLum)
                elif indivStarMags:
                    # save raw magnitudes per particle (wind/outside subhalo entries left at NaN)
                    saveInds = np.arange(i0, i1)
                    r[saveInds[wValid],bandNum,0] = magsLocal
                elif sizes:
                    # require at least 2 stars for size calculation
                    if len(wValid[0]) < 2:
                        continue

                    # slice starting/ending indices for -gas- local to this subhalo
                    i0g = gc['SubhaloOffsetType'][subhaloID,sP.ptNum('gas')] - indRange['gas'][0]
                    i1g = i0g + gc['SubhaloLenType'][subhaloID,sP.ptNum('gas')]

                    assert i0g >= 0 and i1g <= (indRange['gas'][1]-indRange['gas'][0]+1)

                    # calculate projection directions for this subhalo
                    projCen = gc['SubhaloPos'][subhaloID,:]

                    if efrDirs:
                        # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                        rHalf = gc['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
                        shPos = gc['SubhaloPos'][subhaloID,:]

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
                    else:
                        # construct rotation matrices for each specified projection vector direction
                        if Nside is not None:
                            rotMatrices = []
                            for projNum in range(nProj):
                                targetVec = projVecs[projNum,:]
                                rotMatrices.append( rotationMatrixFromVec(projVecs[projNum,:], targetVec) )

                    # get interpolated 2D half light radii
                    for projNum in range(nProj):
                        # rotate coordinates
                        pos_stars = np.squeeze(stars['Coordinates'][i0:i1,:][wValid,:])

                        if Nside is not None:
                            # calculate 2D radii as rr2d
                            pos_stars_rot, _ = rotateCoordinateArray(sP, pos_stars, rotMatrices[projNum], 
                                                                     projCen, shiftBack=False)

                            x_2d = pos_stars_rot[:,0] # realize axes=[0,1]
                            y_2d = pos_stars_rot[:,1] # realize axes=[0,1]
                            rr2d = np.sqrt( x_2d*x_2d + y_2d*y_2d )

                            r[i,bandNum,projNum] = _findHalfLightRadius(rr2d,magsLocal)
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], pos_stars )
                            rr = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i,bandNum,projNum] = _findHalfLightRadius(rr,magsLocal)

    # or, resolved dust: loop over all subhalos first
    if '_res' in dust:
        # prep: resolved dust attenuation uses simulated gas distribution in each subhalo
        loadFields = ['pos','metal','mass']
        if sP.snapHasField('gas', 'NeutralHydrogenAbundance'):
            loadFields.append('NeutralHydrogenAbundance')
        gas = sP.snapshotSubsetP('gas', fields=loadFields, indRange=indRange['gas'])
        if sP.snapHasField('gas', 'GFM_Metals'):
            gas['metals_H'] = sP.snapshotSubsetP('gas', 'metals_H', indRange=indRange['gas']) # H only

        # prep: override 'Masses' with neutral hydrogen mass (model or snapshot value), free some memory
        if modelH:
            gas['Density'] = sP.snapshotSubsetP('gas', 'dens', indRange=indRange['gas'])
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutral=True)
            gas['Density'] = None
        else:
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutralSnap=True)

        gas['metals_H'] = None
        gas['NeutralHydrogenAbundance'] = None
        gas['Cellsize'] = sP.snapshotSubsetP('gas', 'cellsize', indRange=indRange['gas'])

        # prep: unit conversions on stars (age,mass,metallicity)
        stars['GFM_StellarFormationTime'] = sP.units.scalefacToAgeLogGyr(stars['GFM_StellarFormationTime'])
        stars['GFM_InitialMass'] = sP.units.codeMassToMsun(stars['GFM_InitialMass'])

        stars['GFM_Metallicity'] = logZeroMin(stars['GFM_Metallicity'])
        stars['GFM_Metallicity'][np.where(stars['GFM_Metallicity'] < -20.0)] = -20.0

        if sizes:
            gas['StarFormationRate'] = sP.snapshotSubsetP('gas', fields=['sfr'], indRange=indRange['gas'])

        # outer loop over all subhalos
        if not fullSubhaloSpectra: print(' Bands: [%s].' % ', '.join(bands))

        for i, subhaloID in enumerate(subhaloIDsTodo):
            #print('[%d] subhalo = %d' % (i,subhaloID))
            if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo:
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

            # slice starting/ending indices for stars local to this subhalo
            i0 = gc['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange['stars'][0]
            i1 = i0 + gc['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

            assert i0 >= 0 and i1 <= (indRange['stars'][1]-indRange['stars'][0]+1)

            if i1 == i0:
                continue # zero length of this type
            
            # radius restriction: use squared radii and sq distance function
            validMask = np.ones( i1-i0, dtype=np.bool )

            if rad is not None:
                if not radRestrictIn2D:
                    # apply in 3D
                    assert radSqMax.ndim == 1

                    rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], stars['Coordinates'][i0:i1,:] )
                    validMask &= (rr <= radSqMax[subhaloID])
                else:
                    # apply in 2D projection, limited support for now, just Nside='z-axis'
                    # otherwise, for any more complex projection, need to apply it here, and anyways 
                    # for nProj>1, this validMask selection logic becomes projection dependent, so 
                    # need to move it inside the range(nProj) loop, which is definitely doable
                    assert Nside == 'z-axis' and nProj == 1 and np.array_equal(projVecs,[[0,0,1]])
                    p_inds = [0,1] # x,y
                    pt_2d = gc['SubhaloPos'][subhaloID,:]
                    pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                    vecs_2d = np.zeros( (i1-i0, 2), dtype=stars['Coordinates'].dtype )
                    vecs_2d[:,0] = stars['Coordinates'][i0:i1,p_inds[0]]
                    vecs_2d[:,1] = stars['Coordinates'][i0:i1,p_inds[1]]

                    # if doing individual weights based on seeing-spread overlap with aperture, 
                    # truncate contributions to stars at distances >= 5 sigma
                    sigmaPad = 0.0 if seeing is None else 5.0 * seeing_code

                    if radSqMax.ndim == 1:
                        # radial / circular aperture
                        rr = sP.periodicDistsSq( pt_2d, vecs_2d ) # handles 2D
                        rr = np.sqrt(rr)

                        validMask &= (rr <= (np.sqrt(radSqMax[subhaloID])+sigmaPad))
                    else:
                        # rectangular aperture in projected (x,y), e.g. slit
                        xDist = vecs_2d[:,0] - pt_2d[0]
                        yDist = vecs_2d[:,1] - pt_2d[1]
                        correctPeriodicDistVecs(xDist, sP)
                        correctPeriodicDistVecs(yDist, sP)

                        validMask &= ( (xDist <= (np.sqrt(radSqMax[subhaloID,0])+sigmaPad)) & \
                                       (yDist <= (np.sqrt(radSqMax[subhaloID,1])+sigmaPad)) )

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

            if seeing is not None:
                # derive seeing-overlap of aperture based weights
                assert radSqMax.ndim == 2 # otherwise generalize for circular integrals as well
                seeing_weights = np.zeros( ages_logGyr.size, dtype='float32' )

                # re-use previous distance computation
                for j in range(seeing_weights.size):
                    # collecting aperture is centered at (0,0) i.e. at the subhalo center
                    # shift gaussian representing each star's seeing-distributed light to the origin
                    x_min = -xDist[j] - radRestrict_sizeCode[0]*0.5
                    x_max = -xDist[j] + radRestrict_sizeCode[0]*0.5
                    y_min = -yDist[j] - radRestrict_sizeCode[1]*0.5
                    y_max = -yDist[j] + radRestrict_sizeCode[1]*0.5

                    # by hand grid sampling of 2D gaussian within the aperture area
                    # (much faster than scipy.integrate.dblquad)
                    seeing_x, seeing_y = np.meshgrid( np.linspace(x_min,x_max,nint+1), np.linspace(y_min,y_max,nint+1) )
                    wt = np.sum( _seeing_func(seeing_x, seeing_y) ) * (x_max-x_min)/nint * (y_max-y_min)/nint
                    seeing_weights[j] = wt

                assert seeing_weights.min() >= 0.0 and seeing_weights.max() <= 1.0

                # enforce weights by modulating masses of the populations
                masses_msun *= seeing_weights

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
            i0g = gc['SubhaloOffsetType'][subhaloID,sP.ptNum('gas')] - indRange['gas'][0]
            i1g = i0g + gc['SubhaloLenType'][subhaloID,sP.ptNum('gas')]

            assert i0g >= 0 and i1g <= (indRange['gas'][1]-indRange['gas'][0]+1)
 
            # calculate projection directions for this subhalo
            projCen = gc['SubhaloPos'][subhaloID,:]

            if efrDirs:
                # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                rHalf = gc['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
                shPos = gc['SubhaloPos'][subhaloID,:]

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
            else:
                # construct rotation matrices for each specified projection vector direction
                rotMatrices = []
                for projNum in range(projVecs.shape[0]):
                    targetVec = projVecs[projNum,:]
                    rotMatrices.append( rotationMatrixFromVec(projVecs[projNum,:], targetVec) )

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
                    N_H, Z_g = pop.resolved_dust_mapping(pos, hsml, mass_nh, quant_z, pos_stars, 
                                                         projCen, rotMatrix=rotMatrices[projNum])
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
                        if Nside is not None:
                            # rotate coordinates
                            pos_stars_rot, _ = rotateCoordinateArray(sP, pos_stars, rotMatrices[projNum], 
                                                                     projCen, shiftBack=False)

                            # calculate 2D radii as rr2d
                            x_2d = pos_stars_rot[:,0] # realize axes=[0,1]
                            y_2d = pos_stars_rot[:,1] # realize axes=[0,1]
                            rr2d = np.sqrt( x_2d*x_2d + y_2d*y_2d )

                            # get interpolated 2D half light radii
                            r[i,bandNum,projNum] = _findHalfLightRadius(rr2d,magsLocal[band])
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq( projCen, stars['Coordinates'][i0:i1,:] )
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
                                                        ret_full_spectrum=True, output_wave=output_wave, rel_vel=rel_vel_los)

                    # save spectrum within valid wavelength range
                    r[i,:,projNum] = spectrum[spec_min_ind:spec_max_ind]
                else:
                    # compute total attenuated stellar luminosity in each band
                    magsLocal = pop.dust_tau_model_mags(bands,N_H,Z_g,ages_logGyr,metals_log,masses_msun)

                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        r[i,bandNum,projNum] = magsLocal[band]

    # prepare save
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'dust'        : dust.encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    if partInds is not None:
        attrs['partInds'] = partInds

    if fullSubhaloSpectra:
        # save wavelength grid and details of redshifting
        attrs['wavelength'] = output_wave[spec_min_ind : spec_max_ind] # rest-frame
        #if redshifted:
        #    attrs['wavelength'] *= (1 + sP.redshift) # save in observed-frame
        attrs['spectraLumDistMpc'] = sP.units.redshiftToLumDist( sP.redshift )
        if 'sdss' in rad:
            attrs['spectraUnits'] = '10^-17 erg/cm^2/s/Ang'.encode('ascii')
            attrs['spectraFiberDiameterCode'] = fiber_diameter
        if 'legac' in rad:
            r *= 1e2 # 1e-17 to 1e-19 unit prefix (just convention)
            attrs['spectraUnits'] = '10^-19 erg/cm^2/s/Ang'.encode('ascii')
            attrs['slitSizeCode'] = [radSqMax[0,0],radSqMax[0,1]]
    else:
        attrs['bands'] = [b.encode('ascii') for b in bands],

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
    """ For every subhalo, compute an assembly/related quantity using a merger tree. 
    If smoothing is not None, then a tuple request [method,windowSize,windowVal,order]. """
    from scipy.signal import medfilt
    assert quant in ['zForm','isSat_atForm','rad_rvir_atForm','dmFrac_atForm']
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

    # prepare catalog metadata
    desc   = "Merger tree quantity (%s)." % quant
    if smoothing is not None:
        desc += " Using smoothing [%s]." % '_'.join([str(s) for s in smoothing])
    select = "All Subfind subhalos."

    # load snapshot and subhalo information
    redshifts = snapNumToRedshift(sP, all=True)

    nSubsTot = sP.numSubhalos
    ids = np.arange(nSubsTot, dtype='int32') # currently, always process all

    # choose tree fields to load, and validate smoothing request
    groupFields = None
    fields = ['SnapNum']

    if quant == 'zForm':
        fields += ['SubhaloMass']
        mpb_valkey = 'SubhaloMass'

        dtype = 'float32'
        assert len(smoothing) == 3
        assert smoothing[2] == 'snap' # todo: e.g. Gyr, scalefac
    elif quant == 'isSat_atForm':
        fields += ['SubfindID','SubhaloGrNr']
        groupFields = ['GroupFirstSub']
        mpb_valkey = 'SubfindID'

        dtype = 'int16'
        assert smoothing is None
    elif quant == 'rad_rvir_atForm':
        fields += ['SubhaloGrNr','SubhaloPos']
        groupFields = ['GroupPos','Group_R_Crit200']
        mpb_valkey = 'SubhaloGrNr'

        dtype = 'float32'
        assert smoothing is None
    elif quant == 'dmFrac_atForm':
        fields += ['SubhaloMass','SubhaloMassType']
        mpb_valkey = 'SubhaloMassType'
        dmPtNum = sP.ptNum('dm')

        dtype = 'float32'
        assert smoothing is None

    if groupFields is not None:
        # we also need group properties, at all snapshots, so load now
        cacheKey = 'mtq_%s' % quant
        if cacheKey in sP.data:
            print('Loading [%s] from sP.data cache...' % cacheKey)
            groups = sP.data[cacheKey]
        else:
            groups = {}
            prevSnap = sP.snap

            for snap in sP.validSnapList():
                sP.setSnap(snap)
                if snap % 10 == 0: print('%d%%' % (float(snap)/len(sP.validSnapList())*100), end=', ')
                groups[snap] = sP.groupCat(fieldsHalos=groupFields, sq=False)['halos']

            sP.setSnap(prevSnap)
            sP.data[cacheKey] = groups
            print('Saved [%s] into sP.data cache.' % cacheKey)

    # allocate return, NaN indicates not computed (e.g. not in tree at sP.snap)
    r = np.zeros( nSubsTot, dtype=dtype )
    r.fill(np.nan)

    # load all trees at once
    mpbs = cosmo.mergertree.loadMPBs(sP, ids, fields=fields, treeName=treeName)

    # loop over subhalos
    printFac = 100.0 if sP.res > 512 else 10.0

    for i in range(nSubsTot):
        #if i % int(nSubsTot/printFac) == 0 and i <= nSubsTot:
        #    print('   %4.1f%%' % (float(i+1)*100.0/nSubsTot))

        if i not in mpbs:
            continue # subhalo ID i not in tree at sP.snap

        # todo: could generalize here into generic reduction operations over a given tree field
        # e.g. 'max', 'min', 'mean' of 'SubhaloSFR', 'SubhaloGasMetallicity', ... in addition to 
        # more specialized calculations such as formation time
        loc_vals = mpbs[i][mpb_valkey]
        loc_snap = mpbs[i]['SnapNum']

        # smoothing
        if smoothing is not None:
            if loc_snap.size < smoothing[1]+1:
                continue

            if smoothing[0] == 'mm': # moving median window of size N snapshots
                loc_vals = _mm(loc_vals, windowSize=smoothing[1])

            if smoothing[0] == 'ma': # moving average window of size N snapshots
                loc_vals = _ma(loc_vals, windowSize=smoothing[1])

            if smoothing[0] == 'poly': # polynomial fit of Nth order
                coeffs = np.polyfit(loc_snap, loc_vals, smoothing[1])
                loc_vals = np.polyval(coeffs, loc_snap) # resample to original X-pts

        # general quantities
        # (currently none)

        # custom quantities: 'at formation' (at end of MPB)
        if quant == 'isSat_atForm':
            #subpar = loc_vals[-1]
            subid  = loc_vals[-1] #mpbs[i]['SubfindID'][-1]
            subgrnr = mpbs[i]['SubhaloGrNr'][-1]
            subgrnr_snap = loc_snap[-1]
            grfirstsub = groups[subgrnr_snap][subgrnr]

            if grfirstsub == subid:
                # at the MPB last snapshot, GroupFirstSub[SubhaloGrNr] points to this same subhalo, 
                # as recorded by SubfindID at this snapshot, so we are a central
                r[i] = 0
            else:
                # GroupFirstSub points elsewhere
                r[i] = 1

        if quant == 'rad_rvir_atForm':
            sub_pos = mpbs[i]['SubhaloPos'][-1,:]
            subgrnr = loc_vals[-1]
            subgrnr_snap = loc_snap[-1]

            par_pos = groups[subgrnr_snap]['GroupPos'][subgrnr,:]
            par_rvir = groups[subgrnr_snap]['Group_R_Crit200'][subgrnr] 
            dist = sP.periodicDists(sub_pos, par_pos)

            if dist == 0.0:
                # mostly the case for centrals
                r[i] = 0.0
            elif par_rvir == 0.0:
                # can be zero for small groups (why?)
                r[i] = np.inf
            else:
                r[i] = dist / par_rvir

        if quant == 'dmFrac_atForm':
            sub_masstype = loc_vals[-1,:]
            sub_mass = mpbs[i]['SubhaloMass'][-1]

            r[i] = sub_masstype[dmPtNum] / sub_mass

        # custom quantities
        if quant == 'zForm':
            # where does half of max of [smoothed] total mass occur?
            halfMaxVal = loc_vals.max() * 0.5

            #if smoothing[0] == 'poly': # root find on the polynomial coefficients (not so simple)
            #coeffs[-1] -= halfMaxVal # shift such that we find the M=halfMaxVal not M=0 roots
            #roots = np.polynomial.polynomial.polyroots(coeffs[::-1]) # there are many
            w = np.where(loc_vals >= halfMaxVal)[0]
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
                m0 = loc_vals[ind0]
                m1 = loc_vals[ind1]

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
    subhaloIDsTodo = np.arange(sP.numSubhalos, dtype='int32')

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
    from util.sphMap import sphMapWholeBox
    from cosmo.cloudy import cloudyIon

    # adjust projection depth
    projDepthCode = sP.boxSize

    if '_depth10' in species:
        projDepthCode = 10000.0 # 10 cMpc/h
        species = species.split("_depth10")[0]  
    if '_depth125' in species:
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac) # 12.5 cMpc
        species = species.split("_depth125")[0]  

    # check
    hDensSpecies = ['HI','HI_noH2']
    zDensSpecies = ['O VI','O VI 10','O VI 25','O VI solar','O VII','O VIII','O VII solarz','O VII 10 solarz']

    if species not in hDensSpecies + zDensSpecies + ['Z']:
        raise Exception('Not implemented.')

    # config
    h = snapshotHeader(sP)
    nChunks = numPartToChunkLoadSize( h['NumPart'][sP.ptNum('gas')] )
    axes    = [0,1] # x,y projection
    
    # info
    h = sP.snapshotHeader()

    if species in zDensSpecies:
        boxGridSize = boxGridSizeMetals
    else:
        boxGridSize = boxGridSizeHI

    # adjust grid size
    if species in ['O VI 10','O VII 10 solarz']:
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
        gas = sP.snapshotSubsetP('gas', fields, indRange=indRange)

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

            if species[-5:] == 'solar':
                # assume solar abundances
                mMetal = gas['Masses'] * ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange,
                                                                    assumeSolarAbunds=True)
            elif species[-6:] == 'solarz':
                # assume solar abundances and solar metallicity
                mMetal = gas['Masses'] * ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange,
                                                                    assumeSolarAbunds=True,assumeSolarMetallicity=True)
            else:
                # default (use cached ion masses)
                mMetal = sP.snapshotSubset('gas', '%s %s mass' % (element,ionNum), indRange=indRange)
            
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
    from cosmo.hydrogen import calculateCDDF

    if omega:
        mass = sP.snapshotSubset('gas', species + ' mass')
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
    ac = sP.auxCat(fields=[acField])

    # depth
    projDepthCode = sP.boxSize
    if '_depth10' in species:
        projDepthCode = 10000.0
    if '_depth125' in species: 
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac)

    # calculate
    depthFrac = projDepthCode/sP.boxSize

    fN, n = calculateCDDF(ac[acField], binMinMax[0], binMinMax[1], binSize, sP, depthFrac=depthFrac)

    rr = np.vstack( (n,fN) )
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def subhaloRadialProfile(sP, pSplit, ptType, ptProperty, op, scope, weighting=None, 
                         proj2D=None, ptRestrictions=None, subhaloIDsTodo=None):
    """ Compute a radial profile (either total/sum or weighted mean) of a particle property (e.g. mass) 
        for those particles of a given type. If scope=='global', then all snapshot particles are used, 
        and we do the accumulation in a chunked snapshot load. Self/other halo terms are decided based 
        on subfind membership, unless scope=='global_fof', then on group membership. If scope=='fof' or 
        'subfind' then restrict to FoF/subhalo particles only, respectively, and do a restricted load 
        according to pSplit. In this case, only the self-halo term is computed. If scope=='subfind_global' 
        then only the other-halo term is computed, approximating the particle distribution using an 
        already computed subhalo-based accumlation auxCat, e.g. 'Subhalo_Mass_OVI'. If proj2D is None, 
        do 3D profiles, otherwise 2-tuple specifying (i) integer coordinate axis in [0,1,2] to project 
        along or 'face-on' or 'edge-on', and (ii) depth in code units (None for full box). 
        If subhaloIDsTodo is not None, then process this explicit list of subhalos.
    """
    from scipy.stats import binned_statistic

    assert op in ['sum','mean','median','count',np.std]
    assert scope in ['global','global_fof','subfind','fof','subfind_global']

    def _binned_statistic_weighted(x, values, statistic, bins, weights=None, weights_w=None):
        """ If weights == None, straight passthrough to scipy.stats.binned_statistic(). Otherwise, 
        compute once for values*weights, again for weights alone, then normalize and return. 
        If weights_w is not None, apply this np.where() result to the weights array. """
        if weights is None:
            return binned_statistic(x, values, statistic=statistic, bins=bins)

        weights_loc = weights[weights_w] if weights_w is not None else weights

        if statistic == 'mean':
            # weighted mean (nan for bins where wt_sum == 0)
            valwt_sum, bin_edges, bin_number = binned_statistic(x, values*weights_loc, statistic='sum', bins=bins)
            wt_sum, _, _ = binned_statistic(x, weights_loc, statistic='sum', bins=bins)

            return (valwt_sum/wt_sum), bin_edges, bin_number

        if statistic == np.std:
            # weighted standard deviation (note: numba accelerated)
            std = weighted_std_binned(x, values, weights, bins)
            return std, None, None

    # config (hard-coded for oxygen/outflows projects at the moment, could be generalized)
    if sP.boxSize in [75000,205000]:
        radMin = 0.0 # log code units
        radMax = 4.0 # log code units
        radNumBins = 100
        minHaloMass = 10.8 # log m200crit
    if sP.boxSize in [20000,35000]:
        radMin = -0.5 # log code units
        radMax = 3.0 # log code units
        radNumBins = 100
        minHaloMass = 9.0 # log m200crit

    cenSatSelect = 'cen'

    # determine ptRestriction
    if ptType == 'stars':
        if ptRestrictions is None:
            ptRestrictions = {}
        ptRestrictions['GFM_StellarFormationTime'] = ['gt',0.0] # real stars

    # config
    ptLoadType = sP.ptNum(ptType)

    desc   = "Quantity [%s] radial profile for [%s] from [%.1f - %.1f] with [%d] bins." % \
      (ptProperty,ptType,radMin,radMax,radNumBins)
    if ptRestrictions is not None:
        desc += " (restriction = %s)." % ','.join([r for r in ptRestrictions])
    if weighting is not None:
        desc += " (weighting = %s)." % weighting
    if proj2D is not None:
        assert len(proj2D) == 2
        proj2Daxis, proj2Ddepth = proj2D

        if proj2Daxis == 0: p_inds = [1,2,3] # seems wrong (unused), should be e.g. [1,2,0]
        if proj2Daxis == 1: p_inds = [0,2,1]
        if proj2Daxis == 2: p_inds = [0,1,2]
        if isinstance(proj2Daxis, str):
            p_inds = [0,1,2] # by convention, after rotMatrix is applied, index 2 is the projection direction

        proj2D_halfDepth = proj2Ddepth / 2 if proj2Ddepth is not None else None # code units

        depthStr = 'fullbox' if proj2Ddepth is None else '%.1f' % proj2Ddepth
        desc += " (2D projection axis = %s, depth = %s)." % (proj2Daxis,depthStr)

    desc  +=" (scope = %s). " % scope
    if subhaloIDsTodo is None:
        select = "Subhalos [%s] above a min m200crit halo mass of [%.1f]." % (cenSatSelect,minHaloMass)
    else:
        select = "Subhalos [%d] specifically input." % len(subhaloIDsTodo)

    # load group information and make selection
    gc = sP.groupCat(fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['header'] = sP.groupCatHeader()

    nChunks = 1 # chunk load disabled by default

    # need for scope=='subfind' and scope=='global' (for self/other halo terms)
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']

    if scope in ['fof','global_fof']:
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        # for scope=='global_fof' take all FoF particles for the respective halo terms
        GroupLenType = sP.groupCat(fieldsHalos=['GroupLenType'])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=['SubhaloGrNr'])

        gc['SubhaloLenType'] = GroupLenType[SubhaloGrNr,:]
        gc['SubhaloOffsetType'] = GroupOffsetType[SubhaloGrNr,:]

    if scope in ['global','global_fof']:
        # enable chunk loading
        h = sP.snapshotHeader()
        nChunks = numPartToChunkLoadSize( h['NumPart'][sP.ptNum(ptType)] )
        chunkSize = int(h['NumPart'][sP.ptNum(ptType)] / nChunks)

    nSubsTot = gc['header']['Nsubgroups_Total']
    
    # no explicit ID list input, choose subhalos to process now
    if subhaloIDsTodo is None:
        subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

        # css select
        if cenSatSelect is not None:
            cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
            subhaloIDsTodo = subhaloIDsTodo[cssInds]

        # halo mass select
        if minHaloMass is not None:
            halo_masses = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])
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

    if ptRestrictions is not None:
        for restrictionField in ptRestrictions:
            fieldsLoad.append(restrictionField)

    if proj2D is not None and isinstance(proj2Daxis, str):
        # needed for moment of intertia tensor for rotations
        assert ptType == 'gas' # otherwise need to make a separate load to fill gasLocal
        fieldsLoad.append('mass')
        fieldsLoad.append('sfr')
        gc['SubhaloHalfmassRadType'] = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])

    if ptProperty == 'radvel':
        fieldsLoad.append('pos')
        fieldsLoad.append('vel')
        gc['SubhaloVel'] = sP.groupCat(fieldsSubhalos=['SubhaloVel'])

    if ptProperty in ['losvel','losvel_abs']:
        assert proj2D is not None # some 2D projection direction must be defined

        if proj2Daxis in [0,1,2]:
            # load component along one of the cartesian axes (line of sight direction)
            vel_key = 'vel_%s' % ['x','y','z'][proj2Daxis]
            fieldsLoad.append(vel_key)
            gc['SubhaloVel'] = sP.groupCat(fieldsSubhalos=['SubhaloVel'])[:,proj2Daxis]
        else:
            # load full velocity 3-vector for later, per-subhalo rotation
            fieldsLoad.append('vel')
            gc['SubhaloVel'] = sP.groupCat(fieldsSubhalos=['SubhaloVel'])

    fieldsLoad = list(set(fieldsLoad))

    # so long as scope is not 'global', load the full particle set we need for these subhalos now
    if scope not in ['global','global_fof','subfind_global']:
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

        if ptProperty not in userCustomFields:
            particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)
            assert particles[ptProperty].ndim == 1

        if 'count' not in particles:
            key = list(particles.keys())[0]
            particles['count'] = particles[key].shape[0]

        # load weights
        if weighting is not None:
            assert op not in ['sum'] # meaningless
            assert op in ['mean',np.std] # currently only supported

            # use particle masses or volumes (linear) as weights
            particles['weights'] = sP.snapshotSubsetP(partType=ptType, fields=weighting, indRange=indRange)

            assert particles['weights'].ndim == 1 and particles['weights'].size == particles['count']

    # chunk load: loop (possibly just once if chunk load is disabled)
    for chunkNum in range(nChunks):

        # load chunk now (we are simply accumulating, so no need to load everything at once)
        if scope in ['global','global_fof']:
            # calculate load indices (snapshotSubset is inclusive on last index, make sure we get to the end)
            indRange = [chunkNum*chunkSize, (chunkNum+1)*chunkSize-1]
            if chunkNum == nChunks-1: indRange[1] = h['NumPart'][ptLoadType]-1
            print('  [%2d] %9d - %d' % (chunkNum,indRange[0],indRange[1]))

            particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

            if ptProperty not in userCustomFields:
                particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)
                assert particles[ptProperty].ndim == 1

            assert 'count' in particles
            assert weighting is None # load not implemented

        # if approximating the particle distribution with pre-accumulated subhalo based values, load now
        if scope in ['subfind_global']:
            assert nChunks == 1 # no need for more
            assert ptRestrictions is None # cannot apply particle restriction to subhalos
            particles = {}

            if len(ptProperty.split(" ")) == 3:
                species, ion, prop = ptProperty.split(" ")
                acName = 'Subhalo_%s_%s' % (prop.capitalize(), species+ion)
                print(' subfind_global: loading [%s] as effective particle data...' % acName)
            else:
                assert 0 # handle other cases, could e.g. call subhaloRadialReduction directly as:
                # (subhaloRadialReduction(sP,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof')

            particles[ptProperty] = sP.auxCat(acName)[acName]
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

                if op == np.std and i1 - i0 == 1:
                    continue # need at least 2 of this type

            # rotation?
            rotMatrix = None

            if proj2D is not None and isinstance(proj2Daxis, str) and (i1-i0 > 1): # at least 2 particles
                # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                rHalf = gc['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
                shPos = gc['SubhaloPos'][subhaloID,:]

                # local particle set: even if we are computing global radial profiles
                i0g = gc['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
                i1g = i0g + gc['SubhaloLenType'][subhaloID,ptLoadType]

                gasLocal = { 'Masses' : particles['Masses'][i0g:i1g], 
                             'Coordinates' : np.squeeze(particles['Coordinates'][i0g:i1g,:]),
                             'StarFormationRate' : particles['StarFormationRate'][i0g:i1g],
                             'count' : i1g-i0g }
                starsLocal = {'count':0}

                I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos, useStars=False)

                rots = rotationMatricesFromInertiaTensor(I)
                rotMatrix = rots[proj2Daxis]

                # rotate coordinates (velocities handled below)
                particles_pos = particles['Coordinates'][i0:i1,:].copy()
                particles_pos, _ = rotateCoordinateArray(sP, particles_pos, rotMatrix, shPos)

            else:
                particles_pos = particles['Coordinates'][i0:i1,:]

            # use squared radii and sq distance function
            validMask = np.ones( i1-i0, dtype=np.bool )

            if proj2D is None:
                # apply in 3D
                rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], particles_pos )
            else:
                # apply in 2D projection, along the specified axis
                pt_2d = gc['SubhaloPos'][subhaloID,:]
                pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                vecs_2d = np.zeros( (i1-i0, 2), dtype=particles['Coordinates'].dtype )
                vecs_2d[:,0] = particles_pos[:,p_inds[0]]
                vecs_2d[:,1] = particles_pos[:,p_inds[1]]

                rr = sP.periodicDistsSq( pt_2d, vecs_2d ) # handles 2D

                # enforce depth restriction
                if proj2Ddepth is not None:
                    dist_projDir = particles_pos[:,p_inds[2]].copy() # careful of view
                    dist_projDir -= gc['SubhaloPos'][subhaloID,p_inds[2]]
                    correctPeriodicDistVecs(dist_projDir, sP)
                    validMask &= (np.abs(dist_projDir) <= proj2D_halfDepth)

            if scope in ['subfind_global']:
                # do not self count, we are accumulating the other-halo term
                validMask[subhaloID] = 0

            validMask &= (rr <= radMaxSqCode)

            # apply particle-level restrictions
            if ptRestrictions is not None:
                for restrictionField in ptRestrictions:
                    inequality, val = ptRestrictions[restrictionField]

                    if inequality == 'gt':
                        validMask &= (particles[restrictionField][i0:i1] > val)
                    if inequality == 'lt':
                        validMask &= (particles[restrictionField][i0:i1] <= val)
                    if inequality == 'eq':
                        validMask &= (particles[restrictionField][i0:i1] == val)

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
                    p_pos  = np.squeeze( particles_pos[wValid,:] )
                    p_vel  = np.squeeze( particles_pos[wValid,:] )

                    haloPos = gc['SubhaloPos'][subhaloID,:]
                    haloVel = gc['SubhaloVel'][subhaloID,:]

                    loc_val = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)

                if ptProperty in ['losvel','losvel_abs']:
                    if rotMatrix is None:
                        p_vel   = sP.units.particleCodeVelocityToKms( particles[vel_key][i0:i1][wValid] )
                        assert p_vel.ndim == 1 # otherwise, do the following (old)
                        #p_vel   = p_vel[:,p_inds[2]]
                        haloVel = sP.units.subhaloCodeVelocityToKms( gc['SubhaloVel'][subhaloID] )#[p_inds[2]]
                    else:
                        p_vel   = sP.units.particleCodeVelocityToKms( np.squeeze(particles['Velocities'][i0:i1,:][wValid,:]) )
                        haloVel = sP.units.subhaloCodeVelocityToKms( gc['SubhaloVel'][subhaloID,:] )

                        p_vel = np.array( np.transpose( np.dot(rotMatrix, p_vel.transpose()) ) )
                        p_vel = np.squeeze(p_vel[:,p_inds[2]]) # slice index 2 by convention of rotMatrix

                        haloVel = np.array( np.transpose( np.dot(rotMatrix, haloVel.transpose()) ) )[p_inds[2]][0]

                    loc_val = p_vel - haloVel
                    if ptProperty == 'losvel_abs':
                        loc_val = np.abs(loc_val)

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

# common particle-level restrictions
sfrgt0 = {'StarFormationRate':['gt',0.0]}
sfreq0 = {'StarFormationRate':['eq',0.0]}

# this dictionary contains a mapping between all auxCatalogs and their generating functions, where the 
# first sP,pSplit inputs are stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group_Mass_Crit500_Type' : \
     partial(fofRadialSumType,ptProperty='Masses',ptType='all',rad='Group_R_Crit500'),
   'Group_XrayBolLum_Crit500' : \
     partial(fofRadialSumType,ptProperty='xray_lum',ptType='gas',rad='Group_R_Crit500'),

   # subhalo: masses
   'Subhalo_Mass_30pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0),
   'Subhalo_Mass_100pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=100.0),
   'Subhalo_Mass_min_30pkpc_2rhalf_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='30h'),
   'Subhalo_Mass_puchwein10_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='p10'),
   'Subhalo_Mass_SFingGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestrictions=sfrgt0),

   'Subhalo_Mass_30pkpc_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=30.0),
   'Subhalo_Mass_100pkpc_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=100.0),
   'Subhalo_Mass_2rstars_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad='2rhalfstars'),
   'Subhalo_Mass_FoF_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None,scope='fof'),
   'Subhalo_Mass_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None),

   'Subhalo_Mass_10pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_Mass_10pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_Mass_10pkpc_DM' : \
     partial(subhaloRadialReduction,ptType='dm',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_EscapeVel_10pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='vesc',op='mean',rad='10pkpc_slice'),

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
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad=None,ptRestrictions=sfreq0),
   'Subhalo_Mass_SF0Gas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestrictions=sfreq0),
   'Subhalo_Mass_SF0Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestrictions=sfreq0),
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

   # subhalo
   'Subhalo_CoolingTime_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',rad='r015_1rvir_halo',ptRestrictions=sfreq0),
   'Subhalo_CoolingTime_OVI_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',weighting='O VI mass',rad='r015_1rvir_halo',ptRestrictions=sfreq0),

   'Subhalo_StellarMassFormed_10myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.01]}),
   'Subhalo_StellarMassFormed_50myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.05]}),
   'Subhalo_StellarMassFormed_100myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.1]}),
   'Subhalo_GasSFR_30pkpc': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='sfr',op='sum',rad=30.0),

   'Subhalo_Gas_SFR_HalfRad': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='sfr',op='halfrad',rad=None),
   'Subhalo_Gas_HI_HalfRad': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='halfrad',rad=None),

   'Subhalo_XrayBolLum' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad=None),
   'Subhalo_XrayBolLum_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad='2rhalfstars'),
   'Subhalo_S850um' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='s850um_flux',op='sum',rad=None),
   'Subhalo_S850um_25pkpc' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='s850um_flux',op='sum',rad=25.0),
   'Subhalo_BH_Mass_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mass',op='max',rad=None),
   'Subhalo_BH_Mdot_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mdot',op='max',rad=None),
   'Subhalo_BH_MdotEdd_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_MdotEddington',op='max',rad=None),
   'Subhalo_BH_BolLum_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_BolLum',op='max',rad=None),
   'Subhalo_BH_BolLum_basic_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_BolLum_basic',op='max',rad=None),
   'Subhalo_BH_EddRatio_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_EddRatio',op='max',rad=None),
   'Subhalo_BH_dEdt_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_dEdt',op='max',rad=None),

   'Subhalo_Gas_Wind_vel' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_vel',op='mean',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_dEdt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_dEdt',op='sum',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_dPdt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_dPdt',op='sum',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_etaM' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_etaM',op='mean',rad='2rhalfstars'),

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

   'Subhalo_EllipsoidShape_Stars_2rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='shape_ellipsoid',op='ufunc',weighting='mass',rad=None),
   'Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='shape_ellipsoid',op='ufunc',weighting='mass',ptRestrictions=sfrgt0,rad=None),

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
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='mass',ptRestrictions=sfrgt0),
   'Subhalo_Bmag_SFingGas_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='volume',ptRestrictions=sfrgt0),
   'Subhalo_Bmag_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Bmag_2rhalfstars_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='volume'),
   'Subhalo_Bmag_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Bmag_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='volume'),

   'Subhalo_B2_2rhalfstars_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='b2',op='mean',rad='2rhalfstars',weighting='volume'),
   'Subhalo_B2_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='b2',op='mean',rad=None, weighting='volume'),

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

   # light: rest-frame/absolute
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
   'Subhalo_HalfLightRad_p07c_nodust_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside='z-axis', sizes=True),
   'Subhalo_HalfLightRad_p07c_nodust_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside='efr2d', sizes=True), 
   'Subhalo_HalfLightRad_p07c_cf00dust_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='efr2d', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_efr2d_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='efr2d', rad=30.0, sizes=True),   
   'Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', rad=100.0, sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='efr2d', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='efr2d', rad=30.0, sizes=True),

   'Particle_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', indivStarMags=True, Nside='z-axis'),

   # spectral mocks
   'Subhalo_SDSSFiberSpectra_NoVel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis'),
   'Subhalo_SDSSFiberSpectra_Vel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=2, Nside='z-axis'),

   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z_restframe' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_Seeing_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, seeing=0.4, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_Em_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, emlines=True, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_Em_Seeing_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit',
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv',
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, emlines=True, seeing=0.4, minStellarMass=9.8),


   # UVJ: Martina Donnari
   'Subhalo_StellarPhot_UVJ_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j']),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_5pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad=5.0),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad=30.0),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_2rhalf'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad='2rhalfstars'),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j']),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_5pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad=5.0),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad=30.0),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_2rhalf'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad='2rhalfstars'),

   # light: redshifted/apparent
   'Subhalo_StellarPhot_p07c_nodust_red'   : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='none', redshifted=True),
   'Subhalo_StellarPhot_p07c_cf00dust_red' : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='cf00', redshifted=True),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_rad30pkpc_red' : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', rad=30.0, redshifted=True),

   'Particle_StellarPhot_p07c_nodust_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='none', indivStarMags=True, redshifted=True),
   'Particle_StellarPhot_p07c_cf00dust_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='cf00', indivStarMags=True, redshifted=True),
   'Particle_StellarPhot_p07c_cf00dust_res_conv_z_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='cf00_res_conv', indivStarMags=True, Nside='z-axis', redshifted=True),

   'Subhalo_HalfLightRad_p07c_nodust_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside=None, sizes=True, redshifted=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_z_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', sizes=True, redshifted=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', sizes=True, redshifted=True),

   # fullbox
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
   'Box_CDDF_nOVII'          : partial(wholeBoxCDDF,species='OVII'),
   'Box_CDDF_nOVIII'         : partial(wholeBoxCDDF,species='OVIII'),

   'Box_Grid_nOVI_depth10'           : partial(wholeBoxColDensGrid,species='O VI_depth10'),
   'Box_Grid_nOVI_10_depth10'        : partial(wholeBoxColDensGrid,species='O VI 10_depth10'),
   'Box_Grid_nOVI_25_depth10'        : partial(wholeBoxColDensGrid,species='O VI 25_depth10'),
   'Box_Grid_nOVII_depth10'          : partial(wholeBoxColDensGrid,species='O VII_depth10'),
   'Box_Grid_nOVIII_depth10'         : partial(wholeBoxColDensGrid,species='O VIII_depth10'),
   'Box_CDDF_nOVI_depth10'           : partial(wholeBoxCDDF,species='OVI_depth10'),
   'Box_CDDF_nOVI_10_depth10'        : partial(wholeBoxCDDF,species='OVI_10_depth10'),
   'Box_CDDF_nOVI_25_depth10'        : partial(wholeBoxCDDF,species='OVI_25_depth10'),
   'Box_CDDF_nOVII_depth10'          : partial(wholeBoxCDDF,species='OVII_depth10'),
   'Box_CDDF_nOVIII_depth10'         : partial(wholeBoxCDDF,species='OVIII_depth10'),

   'Box_Grid_nOVI_solar_depth10'     : partial(wholeBoxColDensGrid,species='O VI solar_depth10'),
   'Box_CDDF_nOVI_solar_depth10'     : partial(wholeBoxCDDF,species='OVI_solar_depth10'),
   'Box_Grid_nOVII_solarz_depth10'    : partial(wholeBoxColDensGrid,species='O VII solarz_depth10'),
   'Box_CDDF_nOVII_solarz_depth10'    : partial(wholeBoxCDDF,species='OVII_solarz_depth10'),
   'Box_Grid_nOVIII_solarz_depth10'   : partial(wholeBoxColDensGrid,species='O VIII solarz_depth10'),
   'Box_CDDF_nOVIII_solarz_depth10'   : partial(wholeBoxCDDF,species='OVIII_solarz_depth10'),
   'Box_Grid_nOVII_solarz_depth125'    : partial(wholeBoxColDensGrid,species='O VII solarz_depth125'),
   'Box_CDDF_nOVII_solarz_depth125'    : partial(wholeBoxCDDF,species='OVII_solarz_depth125'),
   'Box_Grid_nOVII_10_solarz_depth125'    : partial(wholeBoxColDensGrid,species='O VII 10 solarz_depth125'),
   'Box_CDDF_nOVII_10_solarz_depth125'    : partial(wholeBoxCDDF,species='OVII_10_solarz_depth125'),

   'Box_Omega_HI'                    : partial(wholeBoxCDDF,species='H I',omega=True),
   'Box_Omega_H2'                    : partial(wholeBoxCDDF,species='H 2',omega=True),
   'Box_Omega_OVI'                   : partial(wholeBoxCDDF,species='O VI',omega=True),
   'Box_Omega_OVII'                  : partial(wholeBoxCDDF,species='O VII',omega=True),
   'Box_Omega_OVIII'                 : partial(wholeBoxCDDF,species='O VIII',omega=True),

    # temporal
   'Subhalo_SubLink_zForm_mm5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['mm',5,'snap']),
   'Subhalo_SubLink_zForm_ma5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['ma',5,'snap']),
   'Subhalo_SubLink_zForm_poly7' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['poly',7,'snap']),

   'Subhalo_SubLinkGal_isSat_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='isSat_atForm'),
   'Subhalo_SubLinkGal_dmFrac_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='dmFrac_atForm'),
   'Subhalo_SubLinkGal_rad_rvir_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='rad_rvir_atForm'),

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

   # radial profiles
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

   'Subhalo_RadProfile3D_GlobalFoF_MgII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='Mg II mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_MgII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='Mg II mass',op='sum',scope='global_fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global'),
   'Subhalo_RadProfile2Dz_2Mpc_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global',proj2D=[2,2000]),
   'Subhalo_RadProfile3D_FoF_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='fof',proj2D=[2,None]),

   'Subhalo_RadProfile3D_FoF_SFR' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='sfr',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_SFR' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='sfr',op='sum',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile3D_FoF_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile3D_FoF_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='fof',proj2D=[2,None]),

   'Subhalo_RadProfile3D_FoF_Gas_Bmag' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='bmag',op='mean',scope='fof'),

   'Subhalo_RadProfile3D_FoF_Gas_Metallicity' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',scope='fof'),
   'Subhalo_RadProfile3D_FoF_Gas_Metallicity_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',weighting='sfr',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metallicity' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metallicity_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',weighting='sfr',scope='fof',proj2D=[2,None]),

   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVel_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel_abs',op='mean',weighting='sfr',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVel_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel_abs',op='mean',weighting='sfr',scope='fof',proj2D=['edge-on',None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=['edge-on',None]),
   'Subhalo_RadProfile2Dfaceon_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=['face-on',None]),

   # outflows/inflows
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz'),
   'Subhalo_RadialMassFlux_Global_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='global'),
   'Subhalo_RadialMassFlux_Global_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='global'),

   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_MgII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Mg II mass'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_SiII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Si II mass'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_NaI' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Na I mass'),

   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',
                                                            rawMass=True,fluxMass=False,proj2D=True),
   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',
                                                             rawMass=True,fluxMass=False,proj2D=True),
   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas_SiII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',
                                                             rawMass=True,fluxMass=False,proj2D=True,massField='Si II mass'),

   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr' : partial(massLoadingsSN,sfr_timescale=100,outflowMethod='instantaneous'),
   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-50myr' : partial(massLoadingsSN,sfr_timescale=50,outflowMethod='instantaneous'),
   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-10myr' : partial(massLoadingsSN,sfr_timescale=10,outflowMethod='instantaneous'),

   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-100myr' : partial(massLoadingsSN,sfr_timescale=100,outflowMethod='instantaneous',massField='MgII'),
   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-50myr' : partial(massLoadingsSN,sfr_timescale=50,outflowMethod='instantaneous',massField='MgII'),
   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-10myr' : partial(massLoadingsSN,sfr_timescale=10,outflowMethod='instantaneous',massField='MgII'),

   'Subhalo_RadialEnergyFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',fluxKE=True),
   'Subhalo_RadialEnergyFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',fluxKE=True),
   'Subhalo_RadialMomentumFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',fluxP=True),
   'Subhalo_RadialMomentumFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',fluxP=True),
   'Subhalo_EnergyLoadingSN_SubfindWithFuzz' : partial(massLoadingsSN,outflowMethod='instantaneous',fluxKE=True),
   'Subhalo_MomentumLoadingSN_SubfindWithFuzz' : partial(massLoadingsSN,outflowMethod='instantaneous',fluxP=True),

   'Subhalo_OutflowVelocity_SubfindWithFuzz' : partial(outflowVelocities),
   'Subhalo_OutflowVelocity_MgII_SubfindWithFuzz' : partial(outflowVelocities,massField='MgII'),
   'Subhalo_OutflowVelocity_SiII_SubfindWithFuzz' : partial(outflowVelocities,massField='SiII'),
   'Subhalo_OutflowVelocity_NaI_SubfindWithFuzz' : partial(outflowVelocities,massField='NaI'),

   'Subhalo_OutflowVelocity2DProj_SubfindWithFuzz' : partial(outflowVelocities,proj2D=True),
   'Subhalo_OutflowVelocity2DProj_SiII_SubfindWithFuzz' : partial(outflowVelocities,proj2D=True,massField='SiII'),
  }

# this list contains the names of auxCatalogs which are computed manually (e.g. require more work than 
# a single generative function), but are then saved in the same format and so can be loaded normally
manualFieldNames = \
[   'Subhalo_SDSSFiberSpectraFits_NoVel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_NoVel-Realism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z'
]
