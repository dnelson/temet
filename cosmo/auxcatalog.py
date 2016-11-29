"""
cosmo/auxcatalog.py
  Cosmological simulations - auxiliary catalog for additional derived galaxy/halo properties.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
from functools import partial

from illustris_python.util import partTypeNum
from cosmo.util import periodicDistsSq, snapNumToRedshift
from util.helper import logZeroMin, logZeroNaN
from util.helper import pSplit as pSplitArr

""" Relatively 'hard-coded' analysis decisions that can be changed. For reference, they are attached 
    as metadata attributes in the auxCat file. """

# Group_*: parameters for computations done over each FoF halo
minNumPartGroup = 100 # only consider halos above a minimum total number of particles? (0=disabled)
                      # note: only affects fofRadialSumType() at present

# Subhalo_*: parameters for computations done over each Subfind subhalo

# Box_*: parameters for whole box (halo-independent) computations
boxGridSizeHI     = 1.5 # code units, e.g. ckpc/h
boxGridSizeMetals = 5.0 # code units, e.g. ckpc/h

def fofRadialSumType(sP, pSplit, ptProperty, rad, method='B'):
    """ Compute total/sum of a particle property (e.g. mass) for those particles enclosed within one of 
        the SO radii already computed and available in the group catalog (input as a string). Methods A 
        and B restrict this calculation to FoF particles only, whereas method C does a full particle 
        search over the entire box in order to compute the total/sum for each FoF halo.

      Method A: do individual halo loads per halo, one loop over all halos.
      Method B: do a full snapshot load per type, then halo loop and slice per FoF, to cut down on I/O ops. 
      Method C: per type: full snapshot load, construct the global tree, spherical aperture search per FoF.
    """
    assert pSplit is None # not implemented
    if ptProperty != 'mass':
        raise Exception('Not implemented.')

    # config
    ptSaveTypes = {'dm':0, 'gas':1, 'stars':2} # instead of making a half-empty Nx6 array
    ptLoadTypes = {'dm':partTypeNum('dm'), 'gas':partTypeNum('gas'), 'stars':partTypeNum('stars')}

    desc   = "Mass by type enclosed within a radius of "+rad+" (only FoF particles included). "
    desc  += "Type indices: " + " ".join([t+'='+str(i) for t,i in ptSaveTypes.iteritems()]) + "."
    select = "All FoF halos with GroupLen >= %d." % minNumPartGroup

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupPos','GroupLen','GroupLenType',rad])
    gc['halos']['GroupOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']

    h = cosmo.load.snapshotHeader(sP)

    # allocate return (0=gas+wind, 1=dm, 2=stars-wind), NaN indicates not computed
    r = np.zeros( (gc['header']['Ngroups_Total'], len(ptSaveTypes.keys())), dtype='float32' )
    r.fill(np.nan)

    # info
    Ngroups_Process = len( np.where(gc['halos']['GroupLen'] >= minNumPartGroup)[0] )

    print(' Total # Halos: '+str(gc['header']['Ngroups_Total'])+', above threshold ('+\
          str(minNumPartGroup)+' particles) = '+str(Ngroups_Process))

    # mark halos to be processed (sum will be zero if there are no particles of a give type)
    wProcess = np.where(gc['halos']['GroupLen'] >= minNumPartGroup)[0]
    r[wProcess,:] = 0.0

    # square radii, and use sq distance function
    gc['halos'][rad] = gc['halos'][rad] * gc['halos'][rad]

    if method == 'A':
        # loop over all halos
        for i in np.arange(gc['header']['Ngroups_Total']):
            if i % int(Ngroups_Process/50) == 0 and i <= Ngroups_Process:
                print(' %4.1f%%' % (float(i+1)*100.0/Ngroups_Process))

            if gc['halos']['GroupLen'][i] < minNumPartGroup:
                continue

            # For each type:
            #   1. Load pos (DM), pos/mass (gas), pos/mass/sftime (stars) for this FoF.
            #   2. Calculate periodic distances, (DM: count num within rad, sum massTable*num)
            #      gas/stars: sum mass of those within rad (gas = gas+wind, stars=real stars only)

            # DM
            dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], haloID=i, sq=False)

            if dm['count']:
                rDM = periodicDistsSq( gc['halos']['GroupPos'][i,:], dm['Coordinates'], sP )
                wDM = np.where( rDM <= gc['halos'][rad][i] )

                r[i, ptSaveTypes['dm']] = len(wDM[0]) * h['MassTable'][ptLoadTypes['dm']]

            # GAS
            gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos','mass'], haloID=i)

            if gas['count']:
                rGas = periodicDistsSq( gc['halos']['GroupPos'][i,:], gas['Coordinates'], sP )
                wGas = np.where( rGas <= gc['halos'][rad][i] )

                r[i, ptSaveTypes['gas']] = np.sum( gas['Masses'][wGas] )

            # STARS
            stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','mass','sftime'], haloID=i)

            if stars['count']:
                rStars = periodicDistsSq( gc['halos']['GroupPos'][i,:], stars['Coordinates'], sP )
                wWind  = np.where( (rStars <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'] < 0.0) )
                wStars = np.where( (rStars <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'] >= 0.0) )

                r[i, ptSaveTypes['gas']] += np.sum( stars['Masses'][wWind] )
                r[i, ptSaveTypes['stars']] = np.sum( stars['Masses'][wStars] )

    if method == 'B':
        # (A): DARK MATTER
        print(' [DM]')
        dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], sq=False, haloSubset=True)

        # loop over halos
        for i in wProcess:
            if i % int(Ngroups_Process/10) == 0 and i <= Ngroups_Process:
                print('  %4.1f%%' % (float(i+1)*100.0/Ngroups_Process))

            # slice starting/ending indices for dm local to this FoF
            i0 = gc['halos']['GroupOffsetType'][i,ptLoadTypes['dm']]
            i1 = i0 + gc['halos']['GroupLenType'][i,ptLoadTypes['dm']]

            if i1 == i0:
                continue # zero length of this type

            rr = periodicDistsSq( gc['halos']['GroupPos'][i,:], dm['Coordinates'][i0:i1,:], sP )
            ww = np.where( rr <= gc['halos'][rad][i] )

            r[i, ptSaveTypes['dm']] = len(ww[0]) * h['MassTable'][ptLoadTypes['dm']]

        # (B): GAS
        print(' [GAS]')
        del dm
        gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos','mass'], haloSubset=True)

        # loop over halos
        for i in wProcess:
            if i % int(Ngroups_Process/10) == 0 and i <= Ngroups_Process:
                print('  %4.1f%%' % (float(i+1)*100.0/Ngroups_Process))

            # slice starting/ending indices for gas local to this FoF
            i0 = gc['halos']['GroupOffsetType'][i,ptLoadTypes['gas']]
            i1 = i0 + gc['halos']['GroupLenType'][i,ptLoadTypes['gas']]

            if i1 == i0:
                continue # zero length of this type

            rr = periodicDistsSq( gc['halos']['GroupPos'][i,:], gas['Coordinates'][i0:i1,:], sP )
            ww = np.where( rr <= gc['halos'][rad][i] )

            r[i, ptSaveTypes['gas']] = np.sum( gas['Masses'][i0:i1][ww] )

        # (C): STARS
        print(' [STARS]')
        del gas
        stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','mass','sftime'], haloSubset=True)

        # loop over halos
        for i in wProcess:
            if i % int(Ngroups_Process/10) == 0 and i <= Ngroups_Process:
                print('  %4.1f%%' % (float(i+1)*100.0/Ngroups_Process))

            # slice starting/ending indices for stars local to this FoF
            i0 = gc['halos']['GroupOffsetType'][i,ptLoadTypes['stars']]
            i1 = i0 + gc['halos']['GroupLenType'][i,ptLoadTypes['stars']]

            if i1 == i0:
                continue # zero length of this type

            rr = periodicDistsSq( gc['halos']['GroupPos'][i,:], stars['Coordinates'][i0:i1,:], sP )
            wWind  = np.where( (rr <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'][i0:i1] < 0.0) )
            wStars = np.where( (rr <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) )

            r[i, ptSaveTypes['gas']] += np.sum( stars['Masses'][i0:i1][wWind] )
            r[i, ptSaveTypes['stars']] = np.sum( stars['Masses'][i0:i1][wStars] )

    if method == 'C':
        # scipy methods may work ('toroidal geometry' for periodic BC searches), but probably not fast
        raise Exception('Not implemented.')

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return r, attrs

def subhaloRadialReduction(sP, pSplit, ptType, ptProperty, op, rad, weighting=None):
    """ Compute a reduction operation (either total/sum or weighted mean) of a particle property (e.g. mass) 
        for those particles of a given type enclosed within a fixed radius (input as a scalar, in physical 
        kpc, or as a string specifying a particular model for a variable cut radius). 
        Restricted to subhalo particles only.
    """
    assert op in ['sum','mean']

    # determine ptRestriction
    ptRestriction = None

    if ptType == 'stars':
        ptRestriction = 'real_stars'

    if '_sfrgt0' in ptProperty:
        ptProperty, ptRestriction = ptProperty.split("_") # takes the form ptProp_sfrgt0

    # config
    ptLoadType = sP.ptNum(ptType)

    desc   = "Quantity [%s] enclosed within a radius of [%s] for [%s]." % (ptProperty,rad,ptType)
    if ptRestriction is not None:
        desc += " (restriction = %s). " % ptRestriction
    if weighting is not None:
        desc += " (weighting = %s). " % weighting
    desc  +=" (only subhalo particles included). "
    select = "All Subhalos."

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['subhalos']['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']
    nSubsTot = gc['header']['Nsubgroups_Total']

    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    # task parallelism
    offsets_pt = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
    indRange = [0, offsets_pt[:,ptLoadType].max()]

    if pSplit is not None:
        # do contiguous subhalo ID division and reduce global haloSubset load 
        # to the particle sets which cover the subhalo subset of this pSplit
        subhaloIDsTodo = pSplitArr( subhaloIDsTodo, pSplit[1], pSplit[0] )

        first_sub = subhaloIDsTodo[0]
        last_sub = subhaloIDsTodo[-1]

        first_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=first_sub)['SubhaloGrNr']
        last_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=last_sub)['SubhaloGrNr']

        indRange = offsets_pt[ [first_sub_groupID,last_sub_groupID+1], ptLoadType]

    nSubsDo = len(subhaloIDsTodo)

    # info
    print(' ' + desc)
    print(' Total # Subhalos: %d, processing [%d] subhalos...' % (nSubsTot,nSubsDo))

    # determine radial restriction for each subhalo
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
    elif rad == '2rhalfstars':
        # classic Illustris galaxy definition, r < 2*r_{1/2,mass,stars}
        gcLoad = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * gcLoad['subhalos'][:,sP.ptNum('stars')]

        radSqMax = twiceStellarRHalf**2

    assert radSqMax.size == nSubsTot
    assert radSqMin.size == nSubsTot

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = []

    if rad is not None:
        fieldsLoad.append('pos')
    if ptRestriction == 'real_stars':
        fieldsLoad.append('sftime')
    if ptRestriction == 'sfrgt0':
        fieldsLoad.append('sfr')

    particles = cosmo.load.snapshotSubset(sP, partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)
    particles[ptProperty] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=[ptProperty], indRange=indRange)

    # allocate, NaN indicates not computed except for mass where 0 will do
    if particles[ptProperty].ndim == 1:
        r = np.zeros( nSubsDo, dtype='float32' )
    else:
        r = np.zeros( (nSubsDo,particles[ptProperty].shape[1]), dtype='float32' )

    if ptProperty not in ['mass','Masses']:
        r.fill(np.nan)

    # load weights
    if weighting is None:
        particles['weights'] = np.zeros( particles[ptProperty].shape[0], dtype='float32' )
        particles['weights'] += 1.0 # uniform
    else:
        assert weighting in ['mass','volume'] or 'bandLum' in weighting
        assert op not in ['sum'] # meaningless

        if weighting in ['mass','volume']:
            # use particle masses or volumes (linear) as weights
            particles['weights'] = cosmo.load.snapshotSubset(sP, partType=ptType, fields=weighting, indRange=indRange)

        if 'bandLum' in weighting:
            # prepare sps interpolator
            from cosmo.stellarPop import sps
            pop = sps(sP, 'padova07', 'chabrier', 'bc00')

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

    assert particles['weights'].ndim == 1 and particles['weights'].size == particles[ptProperty].shape[0]

    # loop over subhalos
    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/100)]) == 0 and i <= nSubsDo:
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

        # slice starting/ending indices for stars local to this FoF
        i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,ptLoadType] - indRange[0]
        i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,ptLoadType]

        assert i0 >= 0 and i1 <= (indRange[1]-indRange[0])

        if i1 == i0:
            continue # zero length of this type

        # use squared radii and sq distance function
        validMask = np.ones( i1-i0, dtype=np.bool )

        if rad is not None:
            rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                  particles['Coordinates'][i0:i1,:], sP )
            validMask &= (rr <= radSqMax[subhaloID])
            validMask &= (rr >= radSqMin[subhaloID])

        if ptRestriction == 'real_stars':
            validMask &= (particles['GFM_StellarFormationTime'][i0:i1] >= 0.0)

        if ptRestriction == 'sfrgt0':
            validMask &= (particles['StarFormationRate'][i0:i1] > 0.0)

        wValid = np.where(validMask)

        if len(wValid[0]) == 0:
            continue # zero length of particles satisfying radial cut and restriction

        if particles[ptProperty].ndim == 1:
            # scalar
            loc_val = particles[ptProperty][i0:i1][wValid]
            loc_wt  = particles['weights'][i0:i1][wValid]

            if op == 'sum': r[i] = np.sum( loc_val )
            if op == 'mean': r[i] = np.average( loc_val , weights=loc_wt )
        else:
            # vector (e.g. pos, vel, Bfield)
            for j in range(particles[ptProperty].shape[1]):
                loc_val = particles[ptProperty][i0:i1,j][wValid]
                loc_wt  = particles['weights'][i0:i1][wValid]

                if op == 'sum': r[i,j] = np.sum( loc_val )
                if op == 'mean': r[i,j] = np.average( loc_val , weights=loc_wt )

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'ptProperty'  : ptProperty.encode('ascii'),
             'rad'         : str(rad).encode('ascii'),
             'weighting'   : str(weighting).encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    return r, attrs

def subhaloStellarPhot(sP, pSplit, iso=None, imf=None, dust=None, Nside=1, rad=None, modelH=True):
    """ Compute the total band-magnitudes, per subhalo, under the given assumption of 
    an iso(chrone) model, imf model, and dust model. """
    from cosmo.stellarPop import sps
    from healpy.pixelfunc import nside2npix, pix2vec
    from cosmo.hydrogen import hydrogenMass

    # which bands? for now, to change, just recompute from scratch
    bands = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
    bands += ['wfc_acs_f606w']
    bands += ['des_y']
    bands += ['jwst_f150w']

    nBands = len(bands)

    # which projections?
    nProj = 1

    if '_res' in dust:
        nProj = nside2npix(Nside)
        projVecs = pix2vec(Nside,range(nProj),nest=True)
        projVecs = np.transpose( np.array(projVecs, dtype='float32') ) # Nproj,3

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust)

    # prepare catalog metadata
    desc   = "Stellar photometrics (light emission) totals by subhalo, in multiple rest-frame bands."
    select = "All Subfind subhalos (numProjectionsPer = %d)." % nProj

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType','SubhaloHalfmassRadType','SubhaloPos'])
    gc['subhalos']['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    # allocate return, NaN indicates not computed (no star particles)
    nSubsTot = gc['header']['Nsubgroups_Total']
    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    if Nside > 1:
        # special case: just do a few special case subhalos at high Nside for demonstration
        assert sP.res == 1820 and sP.run == 'tng' and sP.snap == 99 and pSplit is None
        #gcDemo = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'])
        #massesDemo = sP.units.codeMassToLogMsun( gcDemo['halos']['Group_M_Crit200'] )
        #ww = np.where( (massesDemo >= 11.9) & (massesDemo < 12.1) ) # ww[0]= [597, 620, 621...]
        #ww = np.where( (massesDemo >= 13.4) & (massesDemo < 13.5) ) # ww[0]= [34, 52, ...]
        # two massive + three MW-mass halos, SubhaloSFR = [0.2, 5.2, 1.7, 5.0, 1.1] Msun/yr
        subhaloIDsTodo = [172649,208781,412332,415496,415628] # gc['halos']['GroupFirstSub'][inds]

    #if sP.res == 455 and sP.run == 'tng' and sP.snap == 99:
    #    # special case: debugging
    #    subhaloIDsTodo = [0,19051,19052] #[0]

    # task parallelism: cosmo.load.snapshotSubset(haloSubset) manually:
    offsets_pt = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
    indRange_stars = [0, offsets_pt[:,sP.ptNum('stars')].max()]
    indRange_gas = [0, offsets_pt[:,sP.ptNum('gas')].max()]

    if pSplit is not None:
        if 1:
            # split up subhaloIDs in round-robin scheme (equal number of massive/centrals per job)
            # works well, but retains global load of all haloSubset particles
            modSplit = subhaloIDsTodo % pSplit[1]
            subhaloIDsTodo = np.where(modSplit == pSplit[0])[0]

        if 0:
            # do contiguous subhalo ID division and reduce global haloSubset load 
            # to the particle sets which cover the subhalo subset of this pSplit
            subhaloIDsTodo = pSplitArr( subhaloIDsTodo, pSplit[1], pSplit[0] )

            first_sub = subhaloIDsTodo[0]
            last_sub = subhaloIDsTodo[-1]

            first_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=first_sub)['SubhaloGrNr']
            last_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=last_sub)['SubhaloGrNr']

            indRange_stars = offsets_pt[ [first_sub_groupID,last_sub_groupID+1], sP.ptNum('stars')]
            indRange_gas = offsets_pt[ [first_sub_groupID,last_sub_groupID+1], sP.ptNum('gas')]

    nSubsDo = len(subhaloIDsTodo)

    r = np.zeros( (nSubsDo, nBands, nProj), dtype='float32' )
    r = np.squeeze(r)
    r.fill(np.nan)

    print(' Total # Subhalos: %d, processing [%d] in [%d] bands and [%d] projections...' % \
        (nSubsTot,nSubsDo,nBands,nProj))

    # radial restriction
    if rad is not None and isinstance(rad, float):
        # constant scalar, convert [pkpc] -> [ckpc/h] (code units) at this redshift
        rad_pkpc = sP.units.physicalKpcToCodeLength(rad)
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += rad_pkpc * rad_pkpc

    # global load of all stars in all groups in snapshot
    starsLoad = ['initialmass','sftime','metallicity']
    if '_res' in dust or rad is not None: starsLoad += ['pos']

    stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=starsLoad, indRange=indRange_stars)

    # non-resolved dust: loop over all requested bands first
    if '_res' not in dust:
        for bandNum, band in enumerate(bands):
            print('  %02d/%02d [%s]' % (bandNum+1,len(bands),band))

            # request magnitudes in this band for all stars
            mags = pop.mags_code_units(sP, band, stars['GFM_StellarFormationTime'], 
                                                 stars['GFM_Metallicity'], 
                                                 stars['GFM_InitialMass'], retFullSize=True)

            # loop over subhalos
            for i, subhaloID in enumerate(subhaloIDsTodo):
                if i % np.max([1,int(nSubsDo/10)]) == 0 and i <= nSubsDo:
                    print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

                # slice starting/ending indices for stars local to this subhalo
                i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange_stars[0]
                i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

                assert i0 >= 0 and i1 <= (indRange_stars[1]-indRange_stars[0])

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

                magsLocal = mags[i0:i1][wValid] # wind particles have NaN

                # convert mags to luminosities, sum together
                totalLum = np.nansum( np.power(10.0, -0.4 * magsLocal) )

                # convert back to a magnitude in this band
                if totalLum > 0.0:
                    r[i,bandNum] = -2.5 * np.log10( totalLum )

    # or, resolved dust: loop over all subhalos first
    if '_res' in dust:
        # prep: resolved dust attenuation uses simulated gas distribution in each subhalo
        gas = cosmo.load.snapshotSubset(sP, 'gas', fields=['pos','metal','mass','nh'], indRange=indRange_gas)
        gas['GFM_Metals'] = cosmo.load.snapshotSubset(sP, 'gas', 'metals_H', indRange=indRange_gas) # H only

        # prep: override 'Masses' with neutral hydrogen mass (model or snapshot value), free some memory
        if modelH:
            gas['Density'] = cosmo.load.snapshotSubset(sP, 'gas', 'dens', indRange=indRange_gas)
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutral=True)
            gas['Density'] = None
        else:
            gas['Masses'] = hydrogenMass(gas, sP, totalNeutralSnap=True)

        gas['GFM_Metals'] = None
        gas['NeutralHydrogenAbundance'] = None
        gas['Cellsize'] = cosmo.load.snapshotSubset(sP, 'gas', 'cellsize', indRange=indRange_gas)

        # prep: unit conversions on stars (age,mass,metallicity)
        stars['GFM_StellarFormationTime'] = sP.units.scalefacToAgeLogGyr(stars['GFM_StellarFormationTime'])
        stars['GFM_InitialMass'] = sP.units.codeMassToMsun(stars['GFM_InitialMass'])

        stars['GFM_Metallicity'] = logZeroMin(stars['GFM_Metallicity'])
        stars['GFM_Metallicity'][np.where(stars['GFM_Metallicity'] < -20.0)] = -20.0

        # outer loop over all subhalos
        print(' Bands: [%s].' % ', '.join(bands))

        for i, subhaloID in enumerate(subhaloIDsTodo):
            print('[%d] subhalo = %d' % (i,subhaloID))
            if i % np.max([1,int(nSubsDo/100)]) == 0 and i <= nSubsDo:
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

            # slice starting/ending indices for stars local to this subhalo
            i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('stars')] - indRange_stars[0]
            i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('stars')]

            assert i0 >= 0 and i1 <= (indRange_stars[1]-indRange_stars[0])

            if i1 == i0:
                continue # zero length of this type
            
            # radius restriction: use squared radii and sq distance function
            validMask = np.ones( i1-i0, dtype=np.bool )

            if rad is not None:
                rr = periodicDistsSq( gc['subhalos']['SubhaloPos'][subhaloID,:], 
                                      stars['Coordinates'][i0:i1,:], sP )
                validMask &= (rr <= radSqMax[subhaloID])

            validMask &= np.isfinite(stars['GFM_StellarFormationTime'][i0:i1] ) # remove wind

            wValid = np.where(validMask)

            if len(wValid[0]) == 0:
                continue # zero length of particles satisfying radial cut and real stars restriction

            ages_logGyr = stars['GFM_StellarFormationTime'][i0:i1]
            metals_log  = stars['GFM_Metallicity'][i0:i1]
            masses_msun = stars['GFM_InitialMass'][i0:i1]
            pos_stars   = stars['Coordinates'][i0:i1,:]

            if len(wValid[0]) > 1:
                ages_logGyr = ages_logGyr[wValid]
                metals_log  = metals_log[wValid]
                masses_msun = masses_msun[wValid]
                pos_stars   = np.squeeze( pos_stars[wValid,:] )
            
            assert ages_logGyr.shape == metals_log.shape == masses_msun.shape
            assert pos_stars.shape[0] == ages_logGyr.size and pos_stars.shape[1] == 3

            # slice starting/ending indices for -gas- local to this subhalo
            i0g = gc['subhalos']['SubhaloOffsetType'][subhaloID,sP.ptNum('gas')] - indRange_gas[0]
            i1g = i0g + gc['subhalos']['SubhaloLenType'][subhaloID,sP.ptNum('gas')]

            assert i0g >= 0 and i1g <= (indRange_gas[1]-indRange_gas[0])

            # loop over all different viewing directions
            for projNum in range(nProj):
                # at least 2 gas cells exist in subhalo?
                if i1g > i0g+1:
                    # projection
                    projCen = gc['subhalos']['SubhaloPos'][subhaloID,:]
                    projVec = projVecs[projNum,:]

                    # subsets
                    pos     = gas['Coordinates'][i0g:i1g,:]
                    hsml    = 2.5 * gas['Cellsize'][i0g:i1g]
                    mass_nh = gas['Masses'][i0g:i1g]
                    quant_z = gas['GFM_Metallicity'][i0g:i1g]

                    # compute line of sight integrated quantities
                    N_H, Z_g = pop.resolved_dust_mapping(pos, hsml, mass_nh, quant_z, 
                                                         pos_stars, projCen, projVec)
                else:
                    # set columns to zero
                    N_H = np.zeros( len(wValid[0]), dtype='float32' )
                    Z_g = np.zeros( len(wValid[0]), dtype='float32' )

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

    if '_res' in dust:
        attrs['nProj'] = nProj
        attrs['Nside'] = Nside
        attrs['projVecs'] = projVecs
        attrs['modelH'] = modelH

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
    for i in range(nSubsTot):
        if i % int(nSubsTot/10) == 0 and i <= nSubsTot:
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

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    return r, attrs

def tracerTracksQuant(sP, pSplit, quant, op, time):
    """ For every subhalo, compute a assembly/accretion/related quantity using the tracker 
    tracks through time. """
    from tracer.tracerEvo import tracersTimeEvo, tracersMetaOffsets, trValsAtAccTimes, \
      accTime, accMode, ACCMODES
    from tracer.tracerMC import defParPartTypes, fields_in_log

    assert pSplit is None # not implemented
    assert op in ['mean'] #,'sample']
    assert quant in ['angmom','entr','acc_time_1rvir','acc_time_015rvir']
    assert time is None or time in ['acc_time_1rvir','acc_time_015rvir']

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
        assert time is None
        tracks = accTime(sP, rVirFac=1.0) #snapStep
    elif quant == 'acc_time_015rvir':
        assert time is None
        tracks = accTime(sP, rVirFac=0.15) #snapStep
    else:
        if time is None:
            # full tracks (what are we going to do with these?)
            tracks = tracersTimeEvo(sP, quant, all=True) #snapStep=None
            # do e.g. a mean across all time
            import pdb; pdb.set_trace()
        elif time == 'acc_time_1rvir':
            tracks = trValsAtAccTimes(sP, quant, rVirFac=1.0)
        elif time == 'acc_time_015rvir':
            tracks = trValsAtAccTimes(sP, quant, rVirFac=0.15)

    assert tracks.ndim == 1

    # remove log if needed
    if quant in fields_in_log:
        tracks = 10.0**tracks

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

    hDensSpecies = ['HI','HI_noH2','HI2','HI3']
    zDensSpecies = ['O VI','O VI 10','O VI 25','O VI solar']

    if species not in hDensSpecies + zDensSpecies + ['Z']:
        raise Exception('Not implemented.')

    # config
    h = snapshotHeader(sP)

    nChunks = np.max( [4, int(h['NumPart'][sP.ptNum('gas')]**(1.0/3.0) / 10.0)] )
    axes    = [0,1] # x,y projection
    
    # info
    h = cosmo.load.snapshotHeader(sP)

    if species in zDensSpecies:
        boxGridSize = boxGridSizeMetals
    else:
        boxGridSize = boxGridSizeHI

    # DEBUG
    if species == 'HI2':
        boxGridSize = 1.0 # test, 50% smaller
    if species == 'HI3':
        boxGridSize = 0.5 # test, x3 smaller
    if species == 'O VI 10':
        boxGridSize = 10.0 # test, x2 bigger
    if species == 'O VI 25':
        boxGridSize = 2.5 # test, x2 smaller
    # END DEBUG

    boxGridDim = round(sP.boxSize / boxGridSize)
    chunkSize = int(h['NumPart'][partTypeNum('gas')] / nChunks)

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
        if i == nChunks-1: indRange[1] = h['NumPart'][partTypeNum('gas')]-1
        print('  [%2d] %9d - %d' % (i,indRange[0],indRange[1]))

        # load
        gas = cosmo.load.snapshotSubset(sP, 'gas', fields, indRange=indRange)

        # calculate smoothing size (V = 4/3*pi*h^3)
        vol = gas['Masses'] / gas['Density']
        hsml = (vol * 3.0 / (4*np.pi))**(1.0/3.0)
        #hsml = getHsmlForPartType(sP, 'gas', indRange=indRange) # duplicate loading

        if species in hDensSpecies:
            # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
            mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2' or species=='HI3'),
                                                 totalNeutral=(species=='HI_noH2'))

            # simplified models (difference is quite small in CDDF)
            #mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
            #mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

            # grid gas mHI using SPH kernel, return in units of [10^10 Msun * h / ckpc^2]
            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mHI, quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True)

            r += ri

        if species in zDensSpecies:
            # calculate metal ion mass, and grid column densities
            element = species.split()[0]
            ionNum  = species.split()[1]

            ion = cloudyIon(sP, el=element, redshiftInterp=True)

            aSA = False
            if len(species.split()) == 3 and species.split()[2] == 'solar':
                aSA = True # assume solar abundances

            mMetal = gas['Masses'] * ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange,
                                                                assumeSolarAbunds=aSA)

            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mMetal, quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True)

            r += ri

        if species == 'Z':
            # grid total gas mass using SPH kernel, return in units of [10^10 Msun / h]
            rMi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=gas['Masses'], quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False)

            # grid total gas metal mass
            mMetal = gas['Masses'] * gas['GFM_Metallicity']

            rZi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mMetal, quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False)

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

def wholeBoxCDDF(sP, pSplit, species):
    """ Compute the column density distribution function (CDDF, i.e. histogram) of column densities 
        given a full box colDens grid."""
    assert pSplit is None # not implemented
    from cosmo.load import auxCat
    from cosmo.hydrogen import calculateCDDF

    # config
    binSize   = 0.1 # log cm^-2
    binMinMax = [11.0, 24.0] # log cm^-2

    desc   = "Column density distribution function (CDDF) for ["+species+"]. "
    desc  += "Return has shape [2,nBins] where the first slice gives n [cm^-2], the second fN [cm^-2]."
    select = "Binning min: [%g] max: [%g] size: [%g]." % (binMinMax[0], binMinMax[1], binSize)

    # load
    ac = auxCat(sP, fields=['Box_Grid_n'+species])

    # calculate
    fN, n = calculateCDDF(ac['Box_Grid_n'+species], binMinMax[0], binMinMax[1], binSize, sP)

    rr = np.vstack( (n,fN) )
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

# this dictionary contains a mapping between all auxCatalogs and their generating functions, where the 
# first sP,pSplit inputs are stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group_Mass_Crit500_Type' : \
     partial(fofRadialSumType,ptProperty='mass',rad='Group_R_Crit500'),

   'Subhalo_Mass_30pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0),
   'Subhalo_Mass_min_30pkpc_2rhalf_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='30h'),
   'Subhalo_Mass_puchwein10_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='p10'),
   'Subhalo_Mass_SFingGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass_sfrgt0',op='sum',rad=None),

   'Subhalo_StellarAge_NoRadCut_MassWt'       : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='mass'),
   'Subhalo_StellarAge_NoRadCut_rBandLumWt' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=4.0,weighting='bandLum-sdss_r'),

   'Subhalo_StellarMeanVel' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='vel',op='mean',rad=None,weighting='mass'),

   'Subhalo_Bmag_SFingGas_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag_sfrgt0',op='mean',rad=None,weighting='mass'),
   'Subhalo_Bmag_SFingGas_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag_sfrgt0',op='mean',rad=None,weighting='volume'),
   'Subhalo_Bmag_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Bmag_2rhalfstars_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='volume'),
   'Subhalo_Bmag_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Bmag_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_Pratio_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Pratio_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='volume'),

   'Subhalo_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none'),
   'Subhalo_StellarPhot_p07c_bc00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='bc00'), # temporary testing
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
   'Subhalo_StellarPhot_p07c_ns8_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=8),
   'Subhalo_StellarPhot_p07c_ns4_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=4),

   'Box_Grid_nHI'            : partial(wholeBoxColDensGrid,species='HI'),
   'Box_Grid_nHI2'           : partial(wholeBoxColDensGrid,species='HI2'),
   'Box_Grid_nHI3'           : partial(wholeBoxColDensGrid,species='HI3'),
   'Box_Grid_nHI_noH2'       : partial(wholeBoxColDensGrid,species='HI_noH2'),
   'Box_Grid_Z'              : partial(wholeBoxColDensGrid,species='Z'),

   'Box_Grid_nOVI'           : partial(wholeBoxColDensGrid,species='O VI'),
   'Box_Grid_nOVI_10'        : partial(wholeBoxColDensGrid,species='O VI 10'),
   'Box_Grid_nOVI_25'        : partial(wholeBoxColDensGrid,species='O VI 25'),
   'Box_Grid_nOVI_solar'     : partial(wholeBoxColDensGrid,species='O VI solar'),

   'Box_CDDF_nHI'            : partial(wholeBoxCDDF,species='HI'),
   'Box_CDDF_nHI2'           : partial(wholeBoxCDDF,species='HI2'),
   'Box_CDDF_nHI3'           : partial(wholeBoxCDDF,species='HI3'),
   'Box_CDDF_nHI_noH2'       : partial(wholeBoxCDDF,species='HI_noH2'),

   'Box_CDDF_nOVI'           : partial(wholeBoxCDDF,species='OVI'),
   'Box_CDDF_nOVI_10'        : partial(wholeBoxCDDF,species='OVI_10'),
   'Box_CDDF_nOVI_25'        : partial(wholeBoxCDDF,species='OVI_25'),
   'Box_CDDF_nOVI_solar'     : partial(wholeBoxCDDF,species='OVI_solar'),

   'Subhalo_SubLink_zForm_mm5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['mm',5,'snap']),
   'Subhalo_SubLink_zForm_ma5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['ma',5,'snap']),
   'Subhalo_SubLink_zForm_poly7' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['poly',7,'snap']),

   'Subhalo_Tracers_zAcc_mean'   : partial(tracerTracksQuant,quant='acc_time_1rvir',op='mean',time=None),
   'Subhalo_Tracers_angmom_tAcc' : partial(tracerTracksQuant,quant='angmom',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_entr_tAcc'   : partial(tracerTracksQuant,quant='entr',op='mean',time='acc_time_1rvir')
  }
