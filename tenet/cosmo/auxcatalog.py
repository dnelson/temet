"""
Cosmological simulations - auxiliary catalog for additional derived galaxy/halo properties.
The functions here are rarely called directly. Instead they are typically invoked from 
within a particular auxCat request.
"""
import numpy as np
import h5py
from os.path import expanduser
from getpass import getuser

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter

from ..util.helper import logZeroMin, logZeroNaN, logZeroSafe, binned_statistic_weighted
from ..util.helper import pSplit as pSplitArr, pSplitRange, numPartToChunkLoadSize, mvbe
from ..util.sphMap import sphMap
from ..util.treeSearch import calcParticleIndices, buildFullTree, calcHsml, calcQuantReduction
from ..util.rotation import rotateCoordinateArray, rotationMatrixFromVec, momentOfInertiaTensor, \
  rotationMatricesFromInertiaTensor, ellipsoidfit

# todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
# eliminate the entire specialized ufunc logic herein
userCustomFields = ['Krot','radvel','losvel','losvel_abs','shape_ellipsoid','shape_ellipsoid_1r',
                    'tff','tcool_tff']

def fofRadialSumType(sP, pSplit, ptProperty, rad, method='B', ptType='all'):
    """ Compute total/sum of a particle property (e.g. mass) for those particles enclosed within one of 
    the SO radii already computed and available in the group catalog. Use one of four methods:

    * Method A: do individual halo loads per halo, one loop over all halos.
    * Method B: do a full snapshot load per type, then halo loop and slice per FoF, to cut down on I/O ops. 
    * Method C: per type: full snapshot load, spherical aperture search per FoF (brute-force global).
    * Method D: per type: full snapshot load, construct octtree, spherical aperture search per FoF (global).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      rad (str): a radius available in the group catalog, e.g. 'Group_R_Crit200' [code units].
      method (str): see above. **Note!** Methods A and B restrict this calculation to FoF particles only, 
        whereas method C does a full particle search over the entire box in order to compute the total/sum 
        for each FoF halo.
      ptType (str): if 'all', then sum over all types (dm, gas, and stars), otherwise just for the single 
        specified type.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, result for each subhalo.
      - **attrs** (dict): metadata.

    Warning:
      This was an early example of a catalog generating function, and is left mostly for reference as a 
      particularly simple example. In practice, its functionality can be superseded by 
      :py:func:`subhaloRadialReduction`.
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
    indRange = sP.subhaloIDListToBoundingPartIndices(haloIDsTodo, groups=True)

    if pSplit is not None:
        ptSplit = ptType if ptType != 'all' else 'gas'

        # subdivide the global [variable ptType!] particle set, then map this back into a division of 
        # group IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange( indRange[ptSplit], pSplit[1], pSplit[0] )

        invGroups = sP.inverseMapPartIndicesToHaloIDs(gasSplit, ptSplit, debug=True)

        if pSplit[0] == pSplit[1] - 1:
            invGroups[1] = nGroupsTot
        else:
            assert invGroups[1] != -1

        haloIDsTodo = np.arange( invGroups[0], invGroups[1] )
        indRange = sP.subhaloIDListToBoundingPartIndices(haloIDsTodo, groups=True)

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
            if nHalosDo >= 50 and i % int(nHalosDo/50) == 0 and i <= nHalosDo:
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
                if nHalosDo >= 10 and i % int(nHalosDo/10) == 0 and i <= nHalosDo:
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
                if nHalosDo >= 10 and i % int(nHalosDo/10) == 0 and i <= nHalosDo:
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
                if nHalosDo >= 10 and i % int(nHalosDo/10) == 0 and i <= nHalosDo:
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
    """ Handle an input 'rad' specification of a radial restriction.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      nSubsTot (int): total number of subhalos at this snapshot.
      rad (float or str): should match one of the options below, which specifies the details which are
        applied to particle/cell distances in order to achieve the aperture/radial restriction.

    Return:
      a 4-tuple composed of
      
      - **radRestrictIn2D** (bool): apply the cut to 2d projected, instead of 3d, distances.
      - **radSqMin** (list[float]): for each subhalo, the minimum distance to consider, squared.
      - **radSqMax** (list[float]): for each subhalo, the maximum distance to consider, squared.
      - **slit_code** (list[float]): if not None, represents 2d x,y aperture geometry of a slit.
    """
    radRestrictIn2D = False
    radSqMin = np.zeros( nSubsTot, dtype='float32' ) # leave at zero unless modified below
    radSqMax = None
    slit_code = None # used to return aperture geometry for weighted inclusions

    if isinstance(rad, float):
        # constant scalar, convert [pkpc] -> [ckpc/h] (code units) at this redshift
        rad_code = sP.units.physicalKpcToCodeLength(rad)
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += rad_code * rad_code
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
        rad_code = sP.units.physicalKpcToCodeLength(30.0)

        subHalfmassRadType = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        twiceStellarRHalf = 2.0 * subHalfmassRadType[:,sP.ptNum('stars')]

        ww = np.where(twiceStellarRHalf > rad_code)
        twiceStellarRHalf[ww] = rad_code
        radSqMax = twiceStellarRHalf**2.0
    elif rad == '10pkpc_shell':
        # shell at 10 +/- 2 pkpc
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += (sP.units.physicalKpcToCodeLength(12.0))**2
        radSqMin += (sP.units.physicalKpcToCodeLength(8.0))**2
    elif rad == 'rvir_shell':
        # shell at 1.0rvir +/- 0.1 rvir
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.1 * parentR200)**2
        radSqMin = (0.9 * parentR200)**2
    elif rad == 'r015_1rvir_halo':
        # classic 'halo' definition, 0.15rvir < r < 1.0rvir (meaningless for non-centrals)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
        radSqMin = (0.15 * parentR200)**2
    elif rad in ['r200crit','rvir']:
        # within the virial radius (r200,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (1.00 * parentR200)**2
    elif rad in ['2r200crit','2rvir']:
        # within twice the virial radius (r200,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit200'], fieldsSubhalos=['SubhaloGrNr'])
        parentR200 = gcLoad['halos'][gcLoad['subhalos']]

        radSqMax = (2.00 * parentR200)**2
    elif rad in ['0.5r500crit', 'r500crit']:
        # within the (r500,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=['Group_R_Crit500'], fieldsSubhalos=['SubhaloGrNr'])
        parentR500 = gcLoad['halos'][gcLoad['subhalos']]

        if rad == 'r500crit':
            radSqMax = (1.0 * parentR500)**2
        elif rad== '0.5r500crit':
            radSqMax = (0.5 * parentR500)**2
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
    elif rad == '1pkpc_2d':
        # 1 pkpc in 2D projection (e.g. for Sigma_1)
        rad_code = sP.units.physicalKpcToCodeLength(1.0)
        radSqMax = np.zeros( nSubsTot, dtype='float32' ) 
        radSqMax += rad_code * rad_code

        radRestrictIn2D = True
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

    assert radSqMax is not None, 'Unrecognized [%s] rad specification.' % rad
    assert radSqMax.size == nSubsTot or (radSqMax.size == nSubsTot*2 and radSqMax.ndim == 2)
    assert radSqMin.size == nSubsTot or (radSqMin.size == nSubsTot*2 and radSqMin.ndim == 2)

    return radRestrictIn2D, radSqMin, radSqMax, slit_code

def pSplitBounds(sP, pSplit, minStellarMass=None, minHaloMass=None, indivStarMags=False, 
                  partType=None, cenSatSelect=None, equalSubSplit=True):
    """ For a given pSplit, determine an efficient work split and 
    return the required processing for this task, in the form of the list of subhaloIDs to 
    process and the global snapshot index range required in load to cover these subhalos.

    Args:
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      minStellarMass (float): apply lower limit on ``mstar_30pkpc_log``.
      minHaloMass (float): apply lower limit on ``mhalo_200_log``.
      indivStarMags (bool): make sure return covers the full PartType4 size.
      partType (str or None): if not None, use this to decide particle-based split, otherwise use 'gas'.
      cenSatSelect (str or None): if not None, restrict to 'cen', 'sat', or 'all'.
      equalSubSplit (bool): subdivide a pSplit based on equal numbers of subhalos, rather than particles. 

    Return:
      a 3-tuple composed of
      
      - **subhaloIDsTodo** (list[int]): the list of subhalo IDs to process by this task.
      - **indRange** (dict): the index range for the particle load required to cover these subhalos.
      - **nSubsSelection** (int): the number of subhalos to be processed.
    """
    nSubsTot = sP.numSubhalos
    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    # stellar mass select
    if minStellarMass is not None:
        if str(minStellarMass) == '100stars':
            # sP dependent: one hundred stellar particles
            minStellarMass = sP.units.codeMassToLogMsun( sP.targetGasMass * 100 )
            minStellarMass = np.round(minStellarMass[0] * 10) / 10 # round to 0.1

        masses = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])
        with np.errstate(invalid='ignore'):
            wSelect = np.where( masses >= minStellarMass )

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # m200 halo mass select
    if minHaloMass is not None:
        if str(minHaloMass) in ['1000dm','10000dm']:
            # sP dependent: one thousand dm particles
            numDM = 10000 if minHaloMass == '10000dm' else 1000
            minHaloMass = sP.units.codeMassToLogMsun( sP.dmParticleMass * numDM )
            minHaloMass = np.round(minHaloMass[0] * 10) / 10 # round to 0.1

        halo_masses = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])
        if minStellarMass is not None:
            halo_masses = halo_masses[wSelect]

        with np.errstate(invalid='ignore'):
            wSelect = np.where( halo_masses >= minHaloMass )
        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # cen/sat select?
    if cenSatSelect is not None:
        cssSubIDs = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        subhaloIDsTodo = np.intersect1d(subhaloIDsTodo, cssSubIDs)

    nSubsSelection = subhaloIDsTodo.size

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo)

    invSubs = [0,0]

    if pSplit is not None:
        if 0:
            # split up subhaloIDs in round-robin scheme (equal number of massive/centrals per job)
            # works perfectly for balance, but retains global load of all haloSubset particles
            modSplit = subhaloIDsTodo % pSplit[1]
            subhaloIDsTodo = np.where(modSplit == pSplit[0])[0]

        if equalSubSplit:
            # do contiguous subhalo ID division and reduce global haloSubset load 
            # to the particle sets which cover the subhalo subset of this pSplit, but the issue is 
            # that early tasks take all the large halos and all the particles, very imbalanced
            subhaloIDsTodo = pSplitArr( subhaloIDsTodo, pSplit[1], pSplit[0] )

            indRange = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo)
        else:
            # subdivide the global cell/particle set, then map this back into a division of 
            # subhalo IDs which will be better work-load balanced among tasks
            ptType = partType if partType is not None else 'gas'
            gasSplit = pSplitRange( indRange[ptType], pSplit[1], pSplit[0] )

            invSubs = sP.inverseMapPartIndicesToSubhaloIDs(gasSplit, ptType, flagFuzz=False)

            if pSplit[0] == pSplit[1] - 1:
                invSubs[1] = nSubsTot
            
            assert invSubs[1] != -1

            if invSubs[1] == invSubs[0]:
                # split is actually zero size, this is ok
                return [], {'gas':[0,1],'stars':[0,1],'dm':[0,1]}, nSubsSelection

            invSubIDs = np.arange( invSubs[0], invSubs[1] )
            subhaloIDsTodo = np.intersect1d(subhaloIDsTodo, invSubIDs)
            indRange = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo)

    if indivStarMags:
        # make subhalo-strict bounding index range and compute number of PT4 particles we will do
        if invSubs[0] > 0:
            # except for first pSplit, move coverage to include the last subhalo of the previous 
            # split, then increment the indRange[0] by the length of that subhalo. in this way we 
            # avoid any gaps for full PT4 coverage
            subhaloIDsTodo_extended = np.arange( invSubs[0]-1, invSubs[1] )

            indRange = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo_extended, strictSubhalos=True)

            lastPrevSub = sP.groupCatSingle(subhaloID=invSubs[0]-1)
            indRange['stars'][0] += lastPrevSub['SubhaloLenType'][ sP.ptNum('stars') ]
        else:
            indRange = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo, strictSubhalos=True)

    return subhaloIDsTodo, indRange, nSubsSelection

def findHalfLightRadius(rad, vals, frac=0.5, mags=True):
    """ Linearly interpolate in rr (squared radii) to find the half light radius, given a list of 
    values[i] corresponding to each particle at rad[i].

    Args:
      rad (:py:class:`~numpy.ndarray`): list of **squared** radii.
      vals (:py:class:`~numpy.ndarray`): list of values.
      frac (float): if 0.5, then half-light radius, otherwise e.g. 0.2 for r20.
      mags (bool): input ``vals`` are magnitudes, i.e. conversion to linear luminosity needed.
    
    Return:
      float: half-light radius in 3D or 2D (if rad is input 3D or 2D).
    """
    assert rad.size == vals.size

    # take input values unchanged (assume e.g. linear masses or light quantities already)
    lums = vals.copy()

    if mags:
        # convert individual mags to luminosities [arbitrary units]
        lums = np.power(10.0, -0.4 * lums)
   
    radii = rad.copy()
    totalLum = np.nansum( lums )

    sort_inds = np.argsort(radii)

    radii = radii[sort_inds]
    lums = lums[sort_inds]

    # cumulative sum luminosities in radial-distance order
    w = np.where(~np.isfinite(lums)) # wind particles have mags==nan -> lums==nan
    lums[w] = 0.0

    lums_cum = np.cumsum( lums )

    # locate radius where sum equals half of total (half-light radius)
    w = np.where(lums_cum >= frac*totalLum)[0]
    if len(w) == 0:
        return np.nan

    w1 = np.min(w)

    # linear interpolation in linear(rad) and linear(lum), find radius where lums_cum = totalLum/2
    if w1 == 0:
        # half of total luminosity could be within the radius of the first star
        r1 = lums_cum[w1]
        halfLightRad = (frac*totalLum - 0.0)/(r1-0.0) * (radii[w1]-0.0) + 0.0

        assert (halfLightRad >= 0.0 and halfLightRad <= radii[w1]) or np.isnan(halfLightRad)
    else:
        # more generally valid case
        w0 = w1 - 1
        assert w0 >= 0 and w1 < lums.size
        
        r0 = lums_cum[w0]
        r1 = lums_cum[w1]
        halfLightRad = (frac*totalLum - r0)/(r1-r0) * (radii[w1]-radii[w0]) + radii[w0]

        assert halfLightRad >= radii[w0] and (halfLightRad-radii[w1]) < 1e-4

    return halfLightRad

def _process_custom_func(sP,op,ptProperty,gc,subhaloID,particles,rr,i0,i1,wValid,opts):
    """ Handle custom logic for user defined functions and other non-standard 'reduction' 
    operations, for a given particle set (e.g. those of a single subhalo). """

    # ufunc: kappa rot
    if ptProperty == 'Krot':
        # minimum two star particles
        if len(wValid[0]) < 2:
            return [np.nan, np.nan, np.nan, np.nan]

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

        sP.correctPeriodicDistVecs(stars_pos)
        stars_pos = sP.units.codeLengthToKpc(stars_pos) # kpc
        stars_vel = sP.units.particleCodeVelocityToKms(stars_vel) # km/s
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

        r0 = sub_Krot / sub_K                   # \kappa_{star, rot}
        r1 = sub_Krot_oriented / sub_K          # \kappa_{star, rot oriented}
        r2 = mass_frac_counter                  # M_{star,counter} / M_{star,total}
        r3 = sub_stellarJ_mag / sub_stellarMass # j_star [kpc km/s]

        return [r0,r1,r2,r3]

    # ufunc: radial velocity
    if ptProperty == 'radvel':
        gas_pos  = np.squeeze( particles['Coordinates'][i0:i1,:][wValid,:] )
        gas_vel  = np.squeeze( particles['Velocities'][i0:i1,:][wValid,:] )
        gas_weights = np.squeeze( particles['weights'][i0:i1][wValid] )

        haloPos = gc['SubhaloPos'][subhaloID,:]
        haloVel = gc['SubhaloVel'][subhaloID,:]

        vrad = sP.units.particleRadialVelInKmS(gas_pos, gas_vel, haloPos, haloVel)
        if gas_weights.ndim == 0 and vrad.ndim == 1: gas_weights = [gas_weights]

        vrad_avg = np.average(vrad, weights=gas_weights)
        return vrad_avg

    # shape measurement via iterative ellipsoid fitting
    if ptProperty in ['shape_ellipsoid','shape_ellipsoid_1r']:
        scale_rad = gc['SubhaloRhalfStars'][subhaloID]

        if scale_rad == 0:
            return np.nan

        loc_val = particles['Coordinates'][i0:i1, :][wValid]
        loc_wt  = particles['weights'][i0:i1][wValid] # mass

        # positions relative to subhalo center, and normalized by stellar half mass radius
        for j in range(3):
            loc_val[:,j] -= gc['SubhaloPos'][subhaloID,j]

        sP.correctPeriodicDistVecs(loc_val)
        loc_val /= scale_rad

        if ptProperty == 'shape_ellipsoid':
            ellipsoid_rin  = 1.8 # rhalfstars
            ellipsoid_rout = 2.2 # rhalfstars
        if ptProperty == 'shape_ellipsoid_1r':
            ellipsoid_rin  = 0.8 # rhalfstars
            ellipsoid_rout = 1.2 # rhalfstars

        # fit, and save ratios of second and third axes lengths to major axis
        q, s, _, _ = ellipsoidfit(loc_val, loc_wt, scale_rad, ellipsoid_rin, ellipsoid_rout)

        return [q,s]

    # ufunc: 'half radius' (enclosing 50%) of the quantity, or 80%, etc
    if op == 'halfrad':
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt( rr[wValid] )

        rhalf = findHalfLightRadius(loc_rad, loc_val, mags=False)
        return rhalf

    if op == 'rad80':
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt( rr[wValid] )

        r80 = findHalfLightRadius(loc_rad, loc_val, frac=0.8, mags=False)
        return r80

    # ufunc: '3D concentration' C = 5*log(r80/r20), e.g. Rodriguez-Gomez+2019 Eqn 16
    if op == 'concentration':
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt( rr[wValid] )

        r20 = findHalfLightRadius(loc_rad, loc_val, frac=0.2, mags=False)
        r80 = findHalfLightRadius(loc_rad, loc_val, frac=0.8, mags=False)

        c = 5 * np.log10(r80/r20)
        return c

    # distance to 256th closest particle
    if op == 'dist256':
        rr_loc = np.sort( rr[wValid] )
        dist = np.sqrt( np.take(rr_loc, 256, mode='clip') )

        return dist

    # 2d gridding: deposit quantity onto a grid and derive a summary statistic
    if op.startswith('grid2d_'):
        # prepare grid
        pos = np.squeeze( particles['Coordinates'][i0:i1,:][wValid,:] )
        hsml = np.squeeze( particles['hsml'][i0:i1][wValid] )
        mass = np.squeeze( particles[ptProperty][i0:i1][wValid] )
        quant = None

        boxCen = gc['SubhaloPos'][subhaloID,:]

        # allocate return
        result = np.zeros( len(opts['isophot_levels']), dtype='float32' )
        result.fill(np.nan)

        if mass.size == 1:
            return result # cannot grid one cell

        # run grid
        grid = sphMap(pos, hsml, mass, quant, opts['axes'], opts['boxSizeImg'], sP.boxSize, boxCen, 
                      opts['nPixels'], ndims=3)

        # post-process grid (any/all functionality of vis.common.gridBox() that we care about)
        if opts['smoothFWHM'] is not None:
            grid = gaussian_filter(grid, opts['sigma_xy'], mode='reflect', truncate=5.0)

        if ' lum' in ptProperty: # erg/s/cm^2 -> erg/s/cm^2/arcsec^2
            grid = sP.units.fluxToSurfaceBrightness(grid, opts['pxSizesCode'], arcsec2=True)

        grid = logZeroNaN(grid) # log [erg/s/cm^2/arcsec^2]

        # derive quantity
        for j, isoval in enumerate(opts['isophot_levels']):
            if np.isinf(isoval):
                mask = np.ones( grid.shape, dtype='bool' )
            else:
                with np.errstate(invalid='ignore'):
                    mask = ( (grid > isoval) & (grid < isoval+1.0) )

            ww = np.where(mask)

            if op.endswith('_shape'):
                if len(ww[0]) <= 3:
                    continue # singular matrix for mvbe

                points = np.vstack( (opts['xxyy'][ww[0]], opts['xxyy'][ww[1]]) ).T

                # compute minimum volume bounding ellipsoid (minimum area ellipse in 2D)
                axislengths, theta, cen = mvbe(points)

                if 0:
                    # debug plot
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    from matplotlib.patches import Ellipse

                    figsize = np.array([14,10]) * 0.8
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)

                    # plot
                    plt.imshow(mask, cmap='viridis', aspect=mask.shape[0]/mask.shape[1])
                    ax.autoscale(False)

                    pxscale = (opts['nPixels'][0] / opts['gridSizeCodeUnits'])
                    minoraxis_px = 2 * axislengths.min() * pxscale
                    majoraxis_px = 2 * axislengths.max() * pxscale
                    cen_px = cen * pxscale

                    e = Ellipse( cen_px, majoraxis_px, minoraxis_px, theta, lw=2.0, fill=False, color='red' )
                    ax.add_artist(e)

                    ax.scatter(points[:,1]*pxscale, points[:,0]*pxscale, 1.0, marker='x', color='green')

                    fig.savefig('mask_%.1f.pdf' % isoval)
                    plt.close(fig)

                result[j] = (axislengths.max() / axislengths.min()) # a/b > 1

            if op.endswith('_area'):
                result[j] = len(ww[0]) * opts['pxAreaCode'] # (ckpc/h)^2

            if op.endswith('_gini'):
                # Gini coefficient (Rodriguez-Gomez+2019 Eqn 9)
                n = len(ww[0])

                if n < 2:
                    continue # too few pixels

                Xi = np.sort(np.abs(10.0**grid[ww] * 1e10)) # linear, arbitrary scaling to move closer to one
                denom = np.nanmean(Xi) * n * (n-1)
                num = np.sum( (2 * np.arange(1,n+1) - n - 1) * Xi )
                gini = num / denom
                assert gini >= 0.0 and gini <= 1.0

                result[j] = gini

            if op.endswith('_m20'):
                # M20 coefficient (Rodriguez-Gomez+2019 Sec 4.4.2)
                I = 10.0**grid[ww] * 1e10 # linear, arbitrary scaling to move closer to one
                I[np.isnan(I)] = 0.0

                x = opts['xxyy'][ww[0]]
                y = opts['xxyy'][ww[1]]

                # calculate centroid
                M_00 = np.sum(I)
                M_10 = np.sum(x * I)
                M_01 = np.sum(y * I)

                xc = M_10 / M_00
                yc = M_01 / M_00

                # calculate second total central moment
                M_20 = np.sum(x**2 * I)
                M_02 = np.sum(y**2 * I)

                mu_20 = M_20 - xc * M_10
                mu_02 = M_02 - yc * M_01
                second_moment_tot = mu_20 + mu_02

                if second_moment_tot <= 0:
                    continue # negative second moment

                # calculate threshold pixel value
                sorted_vals = np.sort(I.ravel())
                lumfrac = np.cumsum(sorted_vals) / np.nansum(sorted_vals)
                thresh = sorted_vals[np.where(lumfrac >= 0.8)[0]]

                if len(thresh) == 0:
                    continue # too few pixels

                # calculate second moment of these brightest 20% of pixels
                I_20 = I.copy()
                I_20[I < thresh[0]] = 0.0

                M_10 = np.sum(x * I_20)
                M_01 = np.sum(y * I_20)
                M_20 = np.sum(x**2 * I_20)
                M_02 = np.sum(y**2 * I_20)

                mu_20 = M_20 - xc * M_10
                mu_02 = M_02 - yc * M_01

                second_moment_20 = mu_20 + mu_02

                if second_moment_20 <= 0:
                    continue # negative moment

                m20 = np.log10(second_moment_20 / second_moment_tot)

                result[j] = m20

        return result

    raise Exception('Unhandled op.')

def subhaloRadialReduction(sP, pSplit, ptType, ptProperty, op, rad, 
                           ptRestrictions=None, weighting=None, scope='subfind', 
                           minStellarMass=None, minHaloMass=None, cenSatSelect=None):
    """ Compute a reduction operation (total/sum, weighted mean, etc) of a particle property (e.g. mass) 
    for those particles of a given type, optionally enclosed within a given radius.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle type e.g. 'gas', 'stars', 'dm', 'bhs'.
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'sum', 'mean', 'max' or custom user-defined string.
      rad (float or str): if a scalar, then [physical kpc], otherwise a string label for a given 
        radial restriction specification e.g. 'rvir' or '2rhalfstars' (see :py:func:`_radialRestriction`).
      ptRestrictions (dict): apply cuts to which particles/cells are included. Each key,val pair in the dict 
        specifies a particle/cell field string in key, and a [min,max] pair in value, where e.g. np.inf can be 
        used as a maximum to enforce a minimum threshold only.
      weighting (str): if not None, then use this additional particle/cell property as the weight.
      scope (str): Calculation is restricted to subhalo particles only if ``scope=='subfind'`` (default), 
        or FoF particles if ``scope=='fof'``. If ``scope=='global'``, currently a full non-chunked snapshot load 
        and brute-force distance computations to all particles for each subhalo (can change to tree method).
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    ops_basic = ['sum','mean','max']
    ops_custom = ['ufunc','halfrad','rad80','dist256','concentration',
      'grid2d_isophot_shape','grid2d_isophot_area','grid2d_isophot_gini','grid2d_m20']
    assert op in ops_basic + ops_custom
    assert scope in ['subfind','fof','global']
    if op == 'ufunc': assert ptProperty in userCustomFields
    assert minStellarMass is None or minHaloMass is None # cannot have both

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
    if minHaloMass is not None: select += ' (Only with halo mass >= %s)' % minHaloMass
    if cenSatSelect is not None: select += ' (Only [%s] subhalos)' % cenSatSelect

    # load group information
    gc = sP.groupCat(fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']
    nSubsTot = sP.numSubhalos

    if nSubsTot == 0:
        return np.nan, {} # e.g. snapshots so early there are no subhalos

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
    subhaloIDsTodo, indRange, nSubsSelected = pSplitBounds(sP, pSplit, minStellarMass=minStellarMass, 
        minHaloMass=minHaloMass, cenSatSelect=cenSatSelect)
    nSubsDo = len(subhaloIDsTodo)

    if ptType not in indRange:
        return np.nan, {} # e.g. snapshots so early there are no stars

    indRange = indRange[ptType] # choose index range for the requested particle type

    if scope == 'global':
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = [0, h['NumPart'][sP.ptNum(ptType)]-1]
        i0 = 0 # never changes
        i1 = indRange[1] # never changes

    # info
    username = getuser()
    if username != 'wwwrun':
        print(' ' + desc)
        print(' Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now...' % \
            (nSubsTot,nSubsSelected,nSubsDo))

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = []

    if rad is not None or op in ['halfrad','rad80','dist256','concentration']:
        fieldsLoad.append('pos')

    if ptRestrictions is not None:
        for restrictionField in ptRestrictions:
            fieldsLoad.append(restrictionField)

    allocSize = None

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

    if ptProperty in ['shape_ellipsoid','shape_ellipsoid_1r']:
        gc['SubhaloRhalfStars'] = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])[:,sP.ptNum('stars')]
        fieldsLoad.append('pos')
        allocSize = (nSubsDo,2) # q,s

    opts = None # todo: can move to function argument
    if 'grid2d' in op:
        fieldsLoad.append('pos')
        fieldsLoad.append('hsml')

        # hard-code constant grid parameters (can generalize)
        opts = {'isophot_levels' : [-17.5, -18.0, -18.5, -19.0, -19.5, -20.0], # erg/s/cm^2/arcsec^2
                'axes' : [0,1], # random orientations
                'quant' : None, # distribute e.g. mass or light
                'gridExtentKpc' : 100.0,
                'smoothFWHM' : None, # disabled
                'nPixels' : [250,250]}

        # hard-code instrumental related grid parameters (can generalize)
        if 1:
            # MUSE UDF
            opts['pxScaleKpc'] = sP.units.arcsecToAngSizeKpcAtRedshift(0.2) # MUSE 0.2"/px
            opts['smoothFWHM'] = sP.units.arcsecToAngSizeKpcAtRedshift(0.7) # ~MUSE UDF (arcsec, non-AO seeing)

        if op.endswith('_gini') or op.endswith('_m20'):
            # actual pixel size/counts important
            nPixels = int(np.round(opts['gridExtentKpc']/opts['pxScaleKpc']))
            opts['nPixels'] = [nPixels, nPixels]

        if 'isophot_' not in op:
            # quantities invariant to pixel selection, or where we don't want to explore multiple levels
            opts['isophot_levels'] = [-np.inf]

        opts['gridSizeCodeUnits'] = sP.units.physicalKpcToCodeLength(opts['gridExtentKpc'])
        opts['boxSizeImg'] = opts['gridSizeCodeUnits'] * np.array([1.0,1.0,1.0])
        opts['pxSizesCode'] = opts['boxSizeImg'][0:1] / opts['nPixels']
        opts['pxAreaCode'] = np.product(opts['pxSizesCode'])

        # compute pixel coordinates
        opts['xxyy'] = np.linspace(opts['pxScaleKpc']/2, 
                                   opts['gridSizeCodeUnits']-opts['pxScaleKpc']/2,
                                   opts['nPixels'][0])

        if opts['smoothFWHM'] is not None:
            # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
            pxScaleXY = np.array(opts['boxSizeImg'])[opts['axes']] / opts['nPixels']
            opts['sigma_xy'] = (opts['smoothFWHM'] / 2.3548) / pxScaleXY

        allocSize = (nSubsDo,len(opts['isophot_levels']))
        if len(opts['isophot_levels']) == 1:
            allocSize = (nSubsDo)

    fieldsLoad = list(set(fieldsLoad)) # make unique

    particles = {}
    if len(fieldsLoad):
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

    if op != 'ufunc':
        # todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can 
        # eliminate the entire specialized ufunc logic herein
        particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)

    if 'grid2d' in op:
        # ptProperty to be gridded is a luminosity? convert lum -> flux now
        if ' lum' in ptProperty: # 1e30 erg/s -> erg/s/cm^2
            particles[ptProperty] *= 1e30
            particles[ptProperty] = sP.units.luminosityToFlux(particles[ptProperty], wavelength=None)
            
    if 'count' not in particles:
        key = list(particles.keys())[0]
        particles['count'] = particles[key].shape[0]

    # allocate, NaN indicates not computed except for mass where 0 will do
    dtype = particles[ptProperty].dtype if ptProperty in particles.keys() else 'float32' # for custom
    assert dtype in ['float32','float64'] # otherwise check, when does this happen?

    if allocSize is not None:
        r = np.zeros( allocSize, dtype=dtype )
    else:
        if particles[ptProperty].ndim in [0,1]:
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
            from ..cosmo.stellarPop import sps
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
        if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo and username != 'wwwrun':
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

        rr = None
        if 'Coordinates' in particles:

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

                if radSqMax.ndim == 1:
                    # radial / circular aperture
                    validMask &= (rr <= radSqMax[subhaloID])
                    validMask &= (rr >= radSqMin[subhaloID])
                else:
                    # rectangular aperture in projected (x,y), e.g. slit
                    xDist = vecs_2d[:,0] - pt_2d[0]
                    yDist = vecs_2d[:,1] - pt_2d[1]
                    sP.correctPeriodicDistVecs(xDist)
                    sP.correctPeriodicDistVecs(yDist)

                    validMask &= (xDist <= np.sqrt( radSqMax[subhaloID,0]) ) 
                    validMask &= (yDist <= np.sqrt( radSqMax[subhaloID,1]) )

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
        if op in ops_custom:
            r[i,...] = _process_custom_func(sP,op,ptProperty,gc,subhaloID,particles,rr,i0,i1,wValid,opts)

            # ufunc processed and value stored, skip to next subhalo
            continue

        # standard reduction operation
        if particles[ptProperty].ndim == 1:
            # scalar
            loc_val = particles[ptProperty][i0:i1][wValid]
            loc_wt  = particles['weights'][i0:i1][wValid]

            if op == 'sum':
                r[i] = np.sum( loc_val )
            if op == 'max':
                r[i] = np.max( loc_val )
            if op == 'mean':
                if loc_wt.sum() == 0.0:
                    loc_wt = np.zeros( loc_val.size, dtype='float32' ) + 1.0 # if all zero weights
                r[i] = np.average( loc_val , weights=loc_wt )
        else:
            # vector (e.g. pos, vel, Bfield)
            for j in range(particles[ptProperty].shape[1]):
                loc_val = particles[ptProperty][i0:i1,j][wValid]
                loc_wt  = particles['weights'][i0:i1][wValid]

                if op == 'sum':
                    r[i,j] = np.sum( loc_val )
                if op == 'max':
                    r[i,j] = np.max( loc_val )
                if op == 'mean':
                    if loc_wt.sum() == 0.0:
                        loc_wt = np.zeros( loc_val.size, dtype='float32' ) + 1.0 # if all zero weights

                    r[i,j] = np.average( loc_val , weights=loc_wt )

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'ptProperty'  : ptProperty.encode('ascii'),
             'rad'         : str(rad).encode('ascii'),
             'weighting'   : str(weighting).encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    if 'grid2d' in op:
        for key in ['isophot_levels','axes','gridExtentKpc','pxScaleKpc','smoothFWHM','nPixels']:
            attrs[key] = opts[key]

    return r, attrs

def subhaloStellarPhot(sP, pSplit, iso=None, imf=None, dust=None, Nside=1, rad=None, modelH=True, bands=None,
                       sizes=False, indivStarMags=False, fullSubhaloSpectra=False, redshifted=False, emlines=False,
                       seeing=None, minStellarMass=None, minHaloMass=None):
    """ Compute the total band-magnitudes (or half-light radii if ``sizes==True``), per subhalo, 
    under a given assumption of an iso(chrone) model, imf model, dust model, and radial restrction. 
    If using a dust model, can include multiple projection directions per subhalo.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      iso (str): isochrone library, as in :py:class:`cosmo.stellarPop.sps` initialization.
      imf (str): stellar IMF, as in :py:class:`cosmo.stellarPop.sps` initialization.
      dust (str or None): dust model, as in :py:class:`cosmo.stellarPop.sps` initialization.
      Nside (int or str or None): if None, then no 2D projections are done, should be used if e.g. if no 
        viewing angle dependent dust is requested. If an integer, then a Healpix specification, which 
        defines the multiple viewing angles per subhalo. If str, one of 'efr2d' (several edge-on and face-on 
        viewing angles), or 'z-axis' (single projection per subhalo along z).
      rad (float or str): if a scalar, then [physical kpc], otherwise a string label for a given 
        radial restriction specification e.g. 'rvir' or '2rhalfstars' (see :py:func:`_radialRestriction`).
      modelH (bool): use our model for neutral hydrogen masses (for extinction), instead of snapshot values.
      bands (list[str] or None): over-ride default list of broadbands to compute.
      sizes (bool): instead of band-magnitudes, save half-light radii.
      indivStarMags (bool): save the magnitudes for every PT4 (wind->NaN) in all subhalos.
      fullSubhaloSpectra (bool): save a full spectrum vs wavelength for every subhalo.
      redshifted (bool): all the stellar spectra/magnitudes are computed at sP.redshift and the band filters 
        are then applied, resulting in apparent magnitudes. If False (default), stars are assumed to be at 
        z=0, spectra are rest-frame and magnitudes are absolute.
      emlines (bool): include nebular emission lines.
      seeing (float or None): if not None, then instead of a binary inclusion/exclusion of each star particle 
        based on the ``rad`` aperture, include all stars weighted by the fraction of their light which 
        enters the ``rad`` aperture, assuming it is spread by atmospheric seeing into a Gaussian with a 
        sigma of seeing [units of arcseconds at sP.redshift].
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    from ..cosmo.stellarPop import sps
    from healpy.pixelfunc import nside2npix, pix2vec
    from ..cosmo.hydrogen import hydrogenMass

    # mutually exclusive options, at most one can be enabled
    assert sum([sizes,indivStarMags,np.clip(fullSubhaloSpectra,0,1)]) in [0,1]
    assert minStellarMass is None or minHaloMass is None # cannot have both

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust, redshifted=redshifted, emlines=emlines)

    # which bands? for now, to change, just recompute from scratch
    if bands is None:
        bands = []
        bands += ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
        #bands += ['wfcam_y','wfcam_j','wfcam_h','wfcam_k'] # UKIRT IR wide
        #bands += ['wfc_acs_f606w','wfc3_ir_f125w','wfc3_ir_f140w','wfc3_ir_f160w'] # HST IR wide
        #bands += ['jwst_f070w','jwst_f090w','jwst_f115w','jwst_f150w','jwst_f200w','jwst_f277w','jwst_f356w','jwst_f444w'] # JWST IR (NIRCAM) wide
        #if indivStarMags: bands = ['sdss_r']

    if str(bands) == 'all':
        bands = pop.bands

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
            with h5py.File(expanduser("~") + '/obs/LEGAC/legac_dr2_spectra_wave.hdf5','r') as f:
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
    if minHaloMass is not None: select += ' (Only with halo mass >= %s)' % minHaloMass
    if indivStarMags: select = "All PartType4 particles in all subhalos"
    select += " (numProjectionsPer = %d) (%s)." % (nProj, Nside)

    print(' %s\n %s' % (desc,select))

    # load group information
    gc = sP.groupCat(fieldsSubhalos=['SubhaloLenType','SubhaloHalfmassRadType','SubhaloPos'])
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']
    nSubsTot = sP.numSubhalos

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange, nSubsSelected = pSplitBounds(sP, pSplit, minStellarMass=minStellarMass, 
        minHaloMass=minHaloMass, equalSubSplit=False, indivStarMags=indivStarMags)

    nSubsDo = len(subhaloIDsTodo)
    partInds = None

    print(' Total # Subhalos: %d, [%d] in selection, now processing [%d] in [%d] bands and [%d] projections...' % \
        (nSubsTot,nSubsSelected,nSubsDo,nBands,nProj))

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
                    print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo), flush=True)

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

                            r[i,bandNum,projNum] = findHalfLightRadius(rr2d,magsLocal)
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], pos_stars )
                            rr = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i,bandNum,projNum] = findHalfLightRadius(rr,magsLocal)

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
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo), flush=True)

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
                        sP.correctPeriodicDistVecs(xDist)
                        sP.correctPeriodicDistVecs(yDist)

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
                            r[i,bandNum,projNum] = findHalfLightRadius(rr2d,magsLocal[band])
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq( projCen, stars['Coordinates'][i0:i1,:] )
                            rrLocal = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i,bandNum,projNum] = findHalfLightRadius(rrLocal,magsLocal[band])

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
            #attrs['spectraFiberDiameterCode'] = fiber_diameter
        if 'legac' in rad:
            r *= 1e2 # 1e-17 to 1e-19 unit prefix (just convention)
            attrs['spectraUnits'] = '10^-19 erg/cm^2/s/Ang'.encode('ascii')
            attrs['slitSizeCode'] = [radSqMax[0,0],radSqMax[0,1]]
    else:
        attrs['bands'] = [b.encode('ascii') for b in bands]

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

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      treeName (str): specify merger tree.
      quant (str): specify subhalo quantity (available in merger tree) or custom user-defined quantity.
      smoothing (None or list): if not None, then smooth the quantity along the time dimension 
        according to the tuple specification ``[method,windowSize,windowVal,order]`` where 
        ``method`` should be ``mm`` (moving median of ``windowSize``), ``ma`` (moving average 
        of ``windowSize``), or ``poly`` (poly fit of ``order`` N). The window size is given 
        in units of ``windowVal`` which can be only ``snap``.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
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
    redshifts = sP.snapNumToRedshift(all=True)

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
            # if computing these auxcats over many snapshots, use sP.setSnap() to preserve mtq_* cache
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
    mpbs = sP.loadMPBs(ids, fields=fields, treeName=treeName)

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
    """ For every subhalo, compute a assembly/accretion/related quantity using the pre-computed 
    tracker track catalogs (through time).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      quant (str): specify tracer tracks catalog quantity.
      op (str): statistical operation to apply across all the child tracers of the subhalo.
      time (str or None): sample quantity at what time? e.g. 'acc_time_1rvir'.
      norm (str or None): normalize quantity by a second quantity, e.g. 'tvir_tacc'.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    from ..tracer.tracerEvo import tracersTimeEvo, tracersMetaOffsets, trValsAtAccTimes, \
      accTime, accMode, ACCMODES, mpbValsAtRedshifts, mpbValsAtAccTimes
    from ..tracer.tracerMC import defParPartTypes, fields_in_log

    assert pSplit is None # not implemented
    assert op in ['mean'] #,'sample']
    assert quant in ['angmom','entr_log','temp','acc_time_1rvir','acc_time_015rvir','dt_halo']
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

def subhaloCatNeighborQuant(sP, pSplit, quant, op, rad=None, subRestrictions=None, subRestrictionsRel=None, 
                            minStellarMass=None, minHaloMass=None, cenSatSelect=None):
    """ For every subhalo search for spatially nearby neighbors, using a tree, and compute some 
    reduction operation over them. In this case, the search radius is globally constant and/or 
    set per subhalo. Alternatively, perform an adaptive search such that we find at least N>=1 
    neighbor, and similarly compute a reduction operation over their properties.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      quant (str): subhalo quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'min', 'max', 'mean', 'median', and 'sum' compute over the requested 
        quant for all nearby subhalos within the search, excluding this subhalo. 'closest_rad' returns 
        the distance of the closest neighbor satisfying the requested restrictions. 'd[3,5,10]_rad' returns 
        the distance to the 3rd, 5th, or 10th nearest neighbor, respectively. 'closest_quant' 
        returns the quant of this closest neighbor. 'count' returns the number of identified neighbors.
      rad (str or float): physical kpc if float, else a string as recognized by :py:func:`_radialRestriction`.
      subRestrictions (list): apply cuts to which subhalos are searched over. Each item in the list is a 
        3-tuple consisting of {field name, min value, max value}, where e.g. np.inf can be used as a 
        maximum to enforce a minimum threshold only.
        This is the only which modifies the search target sample, as opposite to the search origin sample.
      subRestrictionsRel (dict): as above, but every field is understood to be relative to the current
        subhalo value, which is the normalization, e.g. {'gt':1.0} requires that neighbors have a strictly 
        larger value, while {'lt':0.5} requires neighbors have a value half as large or smaller.
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d or 2d array, containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    assert op in ['min','max','mean','median','sum',
                  'closest_rad','d3_rad','d5_rad','d10_rad','closest_quant','count']
    if op == 'closest_quant': assert quant is not None
    if op in ['closest_rad','count']: assert quant is None

    nSubsTot = sP.numSubhalos

    if nSubsTot == 0:
        return np.nan, {} # e.g. snapshots so early there are no subhalos

    # determine radial restriction for each subhalo
    radRestrictIn2D, radSqMin, radSqMax, _ = _radialRestriction(sP, nSubsTot, rad)

    maxSearchRad = np.sqrt(radSqMax) # code units

    if radRestrictIn2D:
        assert 0 # generalize below, currently everything in 3D

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, _, nSubsSelected = pSplitBounds(sP, pSplit, minStellarMass=minStellarMass, 
        minHaloMass=minHaloMass, cenSatSelect=cenSatSelect)
    nSubsDo = len(subhaloIDsTodo)

    # info
    desc = "[%s] of quantity [%s] enclosed within a radius of [%s]." % (op,quant,rad)
    if subRestrictions is not None:
        for rField, rFieldMin, rFieldMax in subRestrictions:
            desc += " (%s %s %s)" % (rField,rFieldMin,rFieldMax)
    if subRestrictionsRel is not None:
        for rField, rFieldMin, rFieldMax in subRestrictionsRel:
            desc += " (rel: %s %s %s)" % (rField,rFieldMin,rFieldMax)

    select = "All Subhalos."
    if minStellarMass is not None: select += ' (Only with stellar mass >= %.2f)' % minStellarMass
    if minHaloMass is not None: select += ' (Only with halo mass >= %s)' % minHaloMass
    if cenSatSelect is not None: select += ' (Only [%s] subhalos)' % cenSatSelect

    username = getuser()
    if username != 'wwwrun':
        print(' ' + desc)
        print(' Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now...' % \
            (nSubsTot,nSubsSelected,nSubsDo))

    # decide fields, and load all subhalos in snapshot
    fieldsLoad = ['SubhaloPos', 'id']

    if quant is not None:
        fieldsLoad.append(quant)

    if subRestrictions is not None:
        for rField, _, _ in subRestrictions:
            fieldsLoad.append(rField)

    if subRestrictionsRel is not None:
        for rField, _, _ in subRestrictionsRel:
            fieldsLoad.append(rField)

    fieldsLoad = list(set(fieldsLoad)) # make unique

    gc = sP.subhalos(fieldsLoad)

    # start all valid mask for search targets
    validMask = np.ones(nSubsTot, dtype=np.bool)

    # if we will apply (locally variable) restrictions
    if subRestrictionsRel is not None:
        # then the quantities must be non-nan and non-zero for the subhalos we are processing 
        # (efficiency improvement only)
        mask = np.ones( nSubsDo, dtype='bool' )

        for rField, _, _ in subRestrictionsRel:
            # mark invalid subhalos
            mask &= np.isfinite(gc[rField][subhaloIDsTodo])
            mask &= (gc[rField][subhaloIDsTodo] != 0)

        wTodoValid = np.where(mask)

        subhaloIDsTodo = subhaloIDsTodo[wTodoValid]
        nSubsDo = len(subhaloIDsTodo)

        print(' Note: to make relative cuts on [%s] leaves [%d] subhalos to be processed.' % \
            (', '.join([rField for rField,_,_ in subRestrictionsRel]), nSubsDo))

        # similarly, we can apply a global pre-filter to the search targets, based on the 
        # absolute min/max values of the subhalos to be processed
        for rField, rMin, rMax in subRestrictionsRel:
            global_min = gc[rField][subhaloIDsTodo].min() * rMin
            global_max = gc[rField][subhaloIDsTodo].max() * rMax

            ww = np.where( (gc[rField] < global_min) | (gc[rField] > global_max) )
            validMask[ww] = 0

        print(' Note: most conservative application of relative cuts leaves [%d] subhalos to search over.' % \
            np.count_nonzero(validMask))

    # apply (globally constant) restriction to those subhalos included in searches?
    if subRestrictions is not None:
        for rField, rFieldMin, rFieldMax in subRestrictions:
            with np.errstate(invalid='ignore'):
                validMask &= ( (gc[rField] > rFieldMin) & (gc[rField] < rFieldMax) )

    wValid = np.where(validMask)

    print(' After any subRestrictions, searching over [%d] of [%d] subhalos.' % (len(wValid[0]),nSubsTot))

    # take subset
    gc_search = {}

    for key in fieldsLoad:
        gc_search[key] = gc[key][wValid]

    # create inverse mapping (subhaloID -> gc_search index)
    gc_search_index = np.zeros( nSubsTot, dtype='int32' ) - 1
    gc_search_index[gc_search['id']] = np.arange(gc_search['id'].size)

    # initial guess (iterative)
    if op in ['d3_rad','d5_rad','d10_rad']:
        target_ngb_num = int(op.replace('_rad','')[1])
        prevMaxSearchRad = 1000.0

    # allocate, NaN indicates not computed
    dtype = gc[quant].dtype if quant in gc.keys() else 'float32' # for custom
    if op == 'count': dtype = 'int32'

    shape = [nSubsDo]
    if quant is not None and gc[quant].ndim > 1:
        shape.append(gc[quant].shape[1])

    r = np.zeros( shape, dtype=dtype )
    r.fill(np.nan) # set NaN value for un-processed subhalos

    # build tree
    tree = buildFullTree(gc_search['SubhaloPos'], boxSizeSim=sP.boxSize, treePrec='float64', verbose=True)

    # define all valid mask
    loc_search_mask = np.ones(gc_search['SubhaloPos'].shape[0], dtype='bool')

    # loop over subhalos
    printFac = 100.0 if sP.res > 512 else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo and username != 'wwwrun':
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo)) 

        loc_search_pos = gc_search['SubhaloPos']

        if quant is not None:
            loc_search_quant = gc_search[quant]

        # apply (locally relative) restriction to those subhalos included in searches?
        if subRestrictionsRel is not None:
            # reset mask for each subhalo
            loc_search_mask.fill(1)

            # apply each requested restriction
            for rField, rFieldMin, rFieldMax in subRestrictionsRel:
                # can contain nan (e.g. mstar_30pkpc_log), in which case we say this fails the restriction
                with np.errstate(invalid='ignore'):
                    # compute the relative value we apply the restriction on
                    relative_val = gc_search[rField] / gc[rField][subhaloID]

                    loc_search_mask &= ( (relative_val > rFieldMin) & (relative_val < rFieldMax) )

            if np.count_nonzero(loc_search_mask) == 0:
                continue # no subhalos satisfy this relative restriction

        # closest radius search?
        if op in ['closest_rad', 'closest_quant']:
            posSearch = gc['SubhaloPos'][subhaloID,:].reshape((1,3))

            dist, index = calcHsml(loc_search_pos, sP.boxSize, posSearch=posSearch, 
                                  posMask=loc_search_mask, nearest=True, tree=tree)

            if 0:
                # debug verify
                wValid = np.where(loc_search_mask)[0]
                dists = sP.periodicDists(gc['SubhaloPos'][subhaloID,:], loc_search_pos[wValid])
                dists[dists == 0] = np.inf
                index2 = np.argmin(dists)
                dist2 = dists[index2]
                assert index == wValid[index2] and np.abs(dist-dist2) < 1e-3

            if op == 'closest_rad':
                r[i] = dist[0]
            if op == 'closest_quant':
                r[i] = loc_search_quant[index[0]]

            continue

        if op in ['d3_rad','d5_rad','d10_rad']:
            # distance to the 3rd, 5th, 10th closest neighbor
            loc_inds = []

            if np.count_nonzero(loc_search_mask) < target_ngb_num + 1:
                continue # not enough global satisfying subhalos to find locals

            iter_num = 0
            while len(loc_inds) < target_ngb_num + 1:
                # iterative search
                loc_inds = calcParticleIndices(loc_search_pos, gc['SubhaloPos'][subhaloID,:], 
                                               prevMaxSearchRad, boxSizeSim=sP.boxSize, 
                                               posMask=loc_search_mask, tree=tree)

                # if size was too small, increase
                if loc_inds is None:
                    loc_inds = []

                if len(loc_inds) < target_ngb_num + 1:
                    prevMaxSearchRad *= 1.25

                iter_num += 1
                if iter_num > 100:
                    assert 0 # can continue, but this is catastropic

            if 0:
                # debug verify
                wValid = np.where(loc_search_mask)[0]
                loc_dists = sP.periodicDists(gc['SubhaloPos'][subhaloID,:], loc_search_pos[wValid])
                loc_inds2 = np.where(loc_dists <= prevMaxSearchRad)[0]
                assert np.array_equal(np.sort(loc_inds),np.sort(wValid[loc_inds2]))

            # if size was excessive, reduce for next time
            if len(loc_inds) > target_ngb_num * 2:
                prevMaxSearchRad /= 1.25

            dists = sP.periodicDists(gc['SubhaloPos'][subhaloID], loc_search_pos[loc_inds])
            dists = np.sort(dists)

            r[i] = dists[target_ngb_num] # includes r=0 for ourself
            
            continue

        # standard reductions: tree search within given search radius
        loc_inds = calcParticleIndices(loc_search_pos, gc['SubhaloPos'][subhaloID,:], 
                                       maxSearchRad[subhaloID], boxSizeSim=sP.boxSize, 
                                       posMask=loc_search_mask, tree=tree)

        if loc_inds is None:
            # no neighbors within radius
            if op == 'count':
                r[i] = 0

            continue

        if 0:
            # debug verify
            wValid = np.where(loc_search_mask)[0]
            loc_dists = sP.periodicDists(gc['SubhaloPos'][subhaloID,:], loc_search_pos[wValid])
            loc_inds2 = np.where(loc_dists <= maxSearchRad[subhaloID])[0]
            assert np.array_equal(np.sort(loc_inds),np.sort(wValid[loc_inds2]))

        if op == 'count':
            r[i] = len(loc_inds) - 1 # do not count self

            continue

        # do not include this subhalo in any statistic
        loc_vals = loc_search_quant.copy()

        if gc_search_index[subhaloID] >= 0:
            loc_vals[gc_search_index[subhaloID]] = np.nan

        # take subset corresponding to identified neighbors
        loc_vals = loc_vals[loc_inds]

        if np.count_nonzero(np.isfinite(loc_vals)) == 0:
            continue

        # store result
        if op == 'sum':
            r[i] = np.nansum(loc_vals)
        if op == 'max':
            r[i] = np.nanmax(loc_vals)
        if op == 'min':
            r[i] = np.nanmin(loc_vals)
        if op == 'mean':
            r[i] = np.nanmean(loc_vals)
        if op == 'median':
            r[i] = np.nanmedian(loc_vals)

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'rad'         : str(rad).encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    return r, attrs

def wholeBoxColDensGrid(sP, pSplit, species, gridSize=None, onlySFR=False, allSFR=False):
    """ Compute a 2D grid of gas column densities [cm^-2] covering the entire simulation box. For 
    example to derive the neutral hydrogen CDDF. Strategy is a chunked load of the snapshot files, 
    for each using SPH-kernel deposition to distribute the mass of the requested species (e.g. HI, 
    CIV) in all gas cells onto the grid.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      species (str): the gas species/sub-component to grid.
      gridSize (int or None): if specified, override the default grid cell size [code units].
      onlySFR (bool): if True, only include SFR > 0 gas cells.
      allSFR (bool): if True, assume that all gas with SFR > 0 gas a unity fraction of the given 
        species. E.g. for H2, assume star formation gas cells are entirely made of molecular hydrogen.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 2d array, gridded column densities.
      - **attrs** (dict): metadata.
    """
    assert pSplit is None # not implemented

    from ..cosmo import hydrogen
    from ..util.sphMap import sphMapWholeBox
    from ..util.helper import reportMemory
    from ..cosmo.cloudy import cloudyIon

    # hard-coded parameters for whole box (halo-independent) computations, could generalize
    boxGridSizeHI     = 1.5 # code units, e.g. ckpc/h
    boxGridSizeMetals = 5.0 # code units, e.g. ckpc/h

    # adjust projection depth
    projDepthCode = sP.boxSize

    if '_depth10' in species:
        projDepthCode = 10000.0 # 10 cMpc/h
        species = species.split("_depth10")[0] 
    if '_depth20' in species:
        projDepthCode = 20000.0 # 20 cMpc/h
        species = species.split("_depth20")[0] 
    if '_depth5' in species:
        projDepthCode = 5000.0 # 5 cMpc/h
        species = species.split("_depth5")[0] 
    if '_depth1' in species:
        projDepthCode = 1000.0 # 1 cMpc/h
        species = species.split("_depth1")[0] 
    if '_depth125' in species:
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac) # 12.5 cMpc
        species = species.split("_depth125")[0]

    # check
    hDensSpecies   = ['HI','HI_noH2']
    preCompSpecies = ['MH2_BR', 'MH2_GK', 'MH2_KMT', 'MHI_BR', 'MHI_GK', 'MHI_KMT',
                      'MH2_GD14', 'MH2_GK11', 'MH2_K13', 'MH2_S14',
                      'MHI_GD14', 'MHI_GK11', 'MHI_K13', 'MHI_S14']
    zDensSpecies   = ['O VI','O VI 10','O VI 25','O VI solar','O VII','O VIII','O VII solarz','O VII 10 solarz']

    if species not in hDensSpecies + zDensSpecies + preCompSpecies + ['Z']:
        raise Exception('Not implemented.')

    # config
    h = sP.snapshotHeader()
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

    if gridSize is not None:
        print(' Seting gridSize = %f [code units]' % gridSize)
        boxGridSize = gridSize

    boxGridDim = round(sP.boxSize / boxGridSize)
    chunkSize = int(h['NumPart'][sP.ptNum('gas')] / nChunks)

    if species in hDensSpecies + zDensSpecies + preCompSpecies:
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

    if species in preCompSpecies:
        fields = ['Coordinates', 'Masses', species]

        r = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )

    if species == 'Z':
        fields = ['Coordinates','Masses','Density','GFM_Metallicity']

        rM = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )
        rZ = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )

    if onlySFR or allSFR:
        fields += ['StarFormationRate']

    # determine projection depth fraction
    boxWidthFrac = projDepthCode / sP.boxSize

    # loop over chunks (we are simply accumulating, so no need to load everything at once)
    for i in np.arange(nChunks):
        # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
        indRange = [i*chunkSize, (i+1)*chunkSize-1]
        if i == nChunks-1: indRange[1] = h['NumPart'][sP.ptNum('gas')]-1
        print('  [%2d] %9d - %d' % (i,indRange[0],indRange[1]), reportMemory())

        # load
        gas = sP.snapshotSubsetP('gas', fields, indRange=indRange)

        # calculate smoothing size (V = 4/3*pi*h^3)
        if 'Masses' in gas and 'Density' in gas:
            vol = gas['Masses'] / gas['Density']
            hsml = (vol * 3.0 / (4*np.pi))**(1.0/3.0)
        else:
            # equivalent calculation
            hsml = sP.snapshotSubsetP('gas', 'cellsize', indRange=indRange)

        hsml = hsml.astype('float32')

        # modifications
        if onlySFR:
            # only SFR>0 gas contributes
            w = np.where(gas['StarFormationRate'] == 0)
            gas[species][w] = 0.0
            gas['Masses'][w] = 0.0

        if allSFR:
            # SFR>0 gas has a fraction=1 of the given species
            assert species in preCompSpecies # otherwise handle

        if species in hDensSpecies:
            # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
            mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2' or species=='HI3'),
                                                 totalNeutral=(species=='HI_noH2'))

            # simplified models (difference is quite small in CDDF)
            #mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
            #mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

            # grid gas mHI using SPH kernel, return in units of [10^10 Msun * h / ckpc^2]
            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mHI, quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True, sliceFac=boxWidthFrac)

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

        if species in preCompSpecies:
            # anything directly loaded from the snapshots, return in units of [10^10 Msun * h / ckpc^2]
            nThreads = 8
            if boxGridDim > 60000: nThreads = 4
            if boxGridDim > 100000: nThreads = 2

            if allSFR:
                w = np.where(gas['StarFormationRate'] > 0)
                gas[species][w] = gas['Masses'][w]

            ri = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=gas[species], quant=None, 
                                axes=axes, nPixels=boxGridDim, sP=sP, colDens=True, nThreads=nThreads, sliceFac=boxWidthFrac)

            r += ri

        if species == 'Z':
            # grid total gas mass using SPH kernel, return in units of [10^10 Msun / h]
            rMi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=gas['Masses'], quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False, sliceFac=boxWidthFrac)

            # grid total gas metal mass
            mMetal = gas['Masses'] * gas['GFM_Metallicity']

            rZi = sphMapWholeBox(pos=gas['Coordinates'], hsml=hsml, mass=mMetal, quant=None, 
                                 axes=axes, nPixels=boxGridDim, sP=sP, colDens=False, sliceFac=boxWidthFrac)

            rM += rMi
            rZ += rZi

    # finalize
    if species in hDensSpecies+zDensSpecies+preCompSpecies:
        # column density: convert units from [code column density, above] to [H atoms/cm^2] and take log
        rr = sP.units.codeColDensToPhys(r, cgs=True, numDens=True)

        if species in zDensSpecies:
            ion = cloudyIon(None)
            rr /= ion.atomicMass(species.split()[0]) # [H atoms/cm^2] to [ions/cm^2]

        if 'MH2' in species:
            print('Converting [H atoms/cm^2] to [H molecules/cm^2].')
            rr /= 2 # [H atoms/cm^2] to [H molecules/cm^2]

        rr = np.log10(rr)

    if species == 'Z':
        # metallicity: take Z = mass_tot/mass_gas for each pixel, normalize by solar, take log
        rr = rZ / rM
        rr = np.log10( rr / sP.units.Z_solar )

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def wholeBoxCDDF(sP, pSplit, species, gridSize=None, omega=False):
    """ Compute the column density distribution function (CDDF, i.e. histogram) of column densities 
    given a full box column density grid. 

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      species (str): the gas species/sub-component to grid.
      gridSize (int or None): if specified, override the default grid cell size [code units].
      omega (bool): if True, then instead compute the single number 
        Omega_species = rho_species / rho_crit,0.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 1d array, the CDDF.
      - **attrs** (dict): metadata.

    Note:
      There is unfortunate duplication/lack of generality between this function and 
      :py:func:`wholeBoxColDensGrid` (e.g. in the projection depth specification) which is always called. 
      To define a new catalog for a CDDF, it must be specified twice: the actual grid, and the CDDF.
    """
    assert pSplit is None # not implemented
    from ..cosmo.hydrogen import calculateCDDF

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
    binMinMax = [11.0, 28.0] # log cm^-2

    desc   = "Column density distribution function (CDDF) for ["+species+"]. "
    desc  += "Return has shape [2,nBins] where the first slice gives n [cm^-2], the second fN [cm^-2]."
    select = "Binning min: [%g] max: [%g] size: [%g]." % (binMinMax[0], binMinMax[1], binSize)

    # load
    acField = 'Box_Grid_n'+species
    if gridSize is not None:
        acField += '_gridSize=%.1f' % gridSize

    ac = sP.auxCat(fields=[acField])

    # depth
    projDepthCode = sP.boxSize
    if '_depth1' in species:
        projDepthCode = 1000.0
    if '_depth10' in species: # must be after '_depth1'...
        projDepthCode = 10000.0
    if '_depth5' in species:
        projDepthCode = 5000.0
    if '_depth20' in species:
        projDepthCode = 20000.0
    if '_depth125' in species: 
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac)
    assert not ('_depth' in species and projDepthCode == sP.boxSize) # handle

    # calculate
    depthFrac = projDepthCode/sP.boxSize

    fN, n = calculateCDDF(ac[acField], binMinMax[0], binMinMax[1], binSize, sP, depthFrac=depthFrac)

    rr = np.vstack( (n,fN) )
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def subhaloRadialProfile(sP, pSplit, ptType, ptProperty, op, scope, weighting=None, 
                         proj2D=None, ptRestrictions=None, subhaloIDsTodo=None,
                         radMin=-1.0, radMax=3.7, radNumBins=100, radRvirUnits=False, Nside=None, Nngb=None, 
                         minHaloMass=None, minStellarMass=None, cenSatSelect='cen'):
    """ Compute subhalo radial profiles (e.g. total/sum, weighted mean, and so on) of a particle property 
    (e.g. mass) for those particles of a given type.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle type, e.g. 'gas', 'stars', 'dm', 'bhs'.
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'min', 'max', 'mean', 'median', and 'sum' compute over the requested 
        quant for all nearby subhalos within the search, excluding this subhalo. 'closest_rad' returns 
        the distance of the closest neighbor satisfying the requested restrictions. 'd[3,5,10]_rad' returns 
        the distance to the 3rd, 5th, or 10th nearest neighbor, respectively. 'closest_quant' 
        returns the quant of this closest neighbor. 'count' returns the number of identified neighbors.
      scope (str): which particles/cells are included for each profile.
        If ``scope=='global'``, then all snapshot particles are used, and we do the accumulation in a 
        chunked snapshot load. Self/other halo terms are decided based on subfind membership, unless 
        scope=='global_fof', then on group membership.
        If ``scope=='fof'`` or 'subfind' then restrict to FoF/subhalo particles only, respectively, and do a 
        restricted load according to pSplit. In this case, only the self-halo term is computed.
        If ``scope=='subfind_global'`` then only the other-halo term is computed, approximating the particle 
        distribution using an already computed subhalo-based accumlation auxCat, e.g. 'Subhalo_Mass_OVI'. 
        If ``scope=='global_spatial'``, then use pSplit to decompose the work via a spatial subset of the box,
        such that we are guaranteed to have access to all the particles/cells within radMax of all halos 
        being processed, to enable more complex global scope operations.
      weighting (str): if not None, then use this additional particle/cell property as the weight.
      proj2D (bool): if not None, do 3D profiles, otherwise 2-tuple specifying (i) integer coordinate axis in 
        [0,1,2] to project along or 'face-on' or 'edge-on', and (ii) depth in code units (None for full box). 
      ptRestrictions (dict): apply cuts to which particles/cells are included. Each key,val pair in the dict 
        specifies a particle/cell field string in key, and a [min,max] pair in value, where e.g. np.inf can be 
        used as a maximum to enforce a minimum threshold only.
      subhaloIDsTodo (list): if not None, then process this explicit list of subhalos.
      radMin (int): minimum radius for profiles [log code units].
      radMax (int): maximum radius for profiles [log code units].
      radNumBins (int): number of radial bins for profiles.
      radRvirunits (bool): if True, change radMin and radMax to be bins linear in units of rvir of each halo.
      Nside (int or None): if not None, should be a healpix parameter (2,4,8,etc). In this case, we do not 
        compute a spherically averaged radial profile per halo, but instead a spherical healpix sampled set of 
        shells/radial profiles, where the quantity sampling at each point uses a SPH-kernel with ``Nngb``.
      Nngb (int or None): must be specified, if and only if ``Nside`` is also specified. The neighbor number.
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of
      
      - **result** (:py:class:`~numpy.ndarray`): 2d array, radial profile for each subhalo.
      - **attrs** (dict): metadata.

    Note:
        For scopes `global` and `global_fof`, four profiles are saved: [all, self-halo, other-halo, diffuse],
        otherwise only a single 'all' profile is computed.
    """
    from ..projects.rshock import healpix_shells_points

    assert op in ['sum','mean','median','min','max','count','kernel_mean',np.std] # todo: or is a lambda
    assert scope in ['global','global_fof','global_spatial','subfind','fof','subfind_global']

    if scope in ['global','global_fof']:
        assert op in ['sum'] # not generalized to non-accumulation stats w/ chunk loading

    if Nside is not None:
        assert Nngb is not None
        assert op in ['kernel_mean'] # can generalize, calcQuantReduction() accepts other operations
        assert weighting is None # otherwise need to add support in calcQuantReduction()
        assert ptRestrictions is None # otherwise need to rearrange order below
        assert proj2D is None
        assert cenSatSelect == 'cen' # otherwise generalize r/rvir scaling of sample points

    useTree = True if scope == 'global_spatial' else False # can be generalized, or made a parameter

    # determine ptRestriction
    if ptType == 'stars':
        if ptRestrictions is None:
            ptRestrictions = {}
        ptRestrictions['GFM_StellarFormationTime'] = ['gt',0.0] # real stars

    # config
    ptLoadType = sP.ptNum(ptType)

    radDesc = 'log code units' if not radRvirUnits else 'linear rvir units'
    desc = "Quantity [%s] radial profile for [%s] from [%.1f - %.1f] %s, with [%d] bins." % \
        (ptProperty,ptType,radMin,radMax,radDesc,radNumBins)
    if not radRvirUnits:
        desc += " Note: first/inner-most bin is extra, and extends from r=0 to r=%.1f." % radMin
    if Nside is not None:
        desc = "Quantity [%s,%s] spherical healpix sampling, [%.1f - %.1f] r/rvir with [%d] bins. " % \
        (ptType,ptProperty,radMin,radMax,radNumBins)
        desc += "Nside = [%d] Nngb = [%d]." % (Nside,Nngb)
    if ptRestrictions is not None:
        desc += " (restriction = %s)." % ','.join([r for r in ptRestrictions])

    if weighting is not None:
        desc += " (weighting = %s)." % weighting

        assert op not in ['sum'] # meaningless
        assert op in ['mean',np.std] # currently only supported

    if proj2D is not None:
        assert len(proj2D) == 2
        assert scope != 'global_spatial' # otherwise generalize i0g,i1g indices below
        proj2Daxis, proj2Ddepth = proj2D

        if proj2Daxis == 0: p_inds = [1,2,3] # seems wrong (unused), should be e.g. [1,2,0]
        if proj2Daxis == 1: p_inds = [0,2,1]
        if proj2Daxis == 2: p_inds = [0,1,2]
        if isinstance(proj2Daxis, str):
            p_inds = [0,1,2] # by convention, after rotMatrix is applied, index 2 is the projection direction

        proj2D_halfDepth = proj2Ddepth / 2 if proj2Ddepth is not None else None # code units

        depthStr = 'fullbox' if proj2Ddepth is None else '%.1f' % proj2Ddepth
        desc += " (2D projection axis = %s, depth = %s)." % (proj2Daxis,depthStr)

    desc +=" (scope = %s). " % scope
    if subhaloIDsTodo is None:
        select = "Subhalos [%s]." % cenSatSelect
        if minStellarMass is not None: select += ' (Only with stellar mass >= %.2f)' % minStellarMass
        if minHaloMass is not None: select += ' (Only with halo mass >= %s)' % minHaloMass
    else:
        nSubsSelected = len(subhaloIDsTodo)
        select = "Subhalos [%d] specifically input." % nSubsSelected

    # load group information and make selection
    gc = sP.groupCat(fieldsSubhalos=['SubhaloPos','SubhaloLenType'])
    gc['header'] = sP.groupCatHeader()

    # no explicit ID list input, choose subhalos to process now
    pSplitSpatial = None
    load_inds = None

    if scope == 'global_spatial':
        # for spatial subdivision, disable the normal subhalo-based subdivision
        pSplitSpatial = pSplit
        pSplit = None
        indRange = None

    if subhaloIDsTodo is None:
        subhaloIDsTodo, indRange_scoped, nSubsSelected = pSplitBounds(sP, pSplit, 
            partType='dm', minStellarMass=minStellarMass, minHaloMass=minHaloMass, cenSatSelect=cenSatSelect, 
            equalSubSplit=False)
    else:
        assert pSplit is None # otherwise check, don't think we actually subdivide the work
        indRange_scoped = sP.subhaloIDListToBoundingPartIndices(subhaloIDsTodo)

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

        # default particle load range is set inside chunkLoadLoop
        print(' Total # Snapshot Load Chunks: '+str(nChunks)+' ('+str(chunkSize)+' particles per load)')
        indRange = None
        prevMaskInd = 0

    if scope in ['subfind','fof']:
        # non-global load, use restricted index range covering our input/selected/pSplit subhaloIDsTodo
        indRange = indRange_scoped[ptType]

    # determine radial binning
    if Nside is None:
        # normal radial profiles
        if radRvirUnits:
            # radMin, radMax in linear rvir units
            rad_bin_edges = np.linspace(radMin, radMax, radNumBins+1) # bin edges, including inner and outer boundary
            rbins_sq = rad_bin_edges**2

            # load virial radii (code units)
            gc['Subhalo_Rvir'] = sP.subhalos('rhalo_200_code')
        else:
            # radMin, radMax in log code units
            rad_bin_edges = np.linspace(radMin,radMax,radNumBins+1) # bin edges, including inner and outer boundary
            rad_bin_edges = np.hstack([radMin-1.0,rad_bin_edges]) # include an inner bin complete to r=0

            rbins_sq = np.log10( (10.0**rad_bin_edges)**2 ) # we work in squared distances for speed
            rad_bins_code = 0.5*(rad_bin_edges[1:] + rad_bin_edges[:-1]) # bin centers [log]
            rad_bins_pkpc = sP.units.codeLengthToKpc( 10.0**rad_bins_code )

            radMaxCode = 10.0**radMax
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
    else:
        # spherical sampling: get sample points (centered at 0,0,0 in units of rvir)
        pts, nProj, rad_bins_rvir, _ = healpix_shells_points(nRad=radNumBins, Nside=Nside, radMin=radMin, radMax=radMax)

        nRad = rad_bins_rvir.size

        # load virial radii (code units)
        gc['Subhalo_Rvir'] = sP.subhalos('rhalo_200_code')

    if pSplitSpatial:
        # spatial decomposition: determine extent and child subhalos
        assert np.abs(pSplitSpatial[1]**(1/3) - np.round(pSplitSpatial[1]**(1/3))) < 1e-6, 'pSplitSpatial: Total number of jobs should have integer cube root, e.g. 8, 27, 64.'
        nPerDim = int(pSplitSpatial[1]**(1/3))
        extent = sP.boxSize / nPerDim

        ijk = np.unravel_index(pSplitSpatial[0], (nPerDim,nPerDim,nPerDim))
        xmin = ijk[0] * extent
        xmax = (ijk[0]+1) * extent
        ymin = ijk[1] * extent
        ymax = (ijk[1]+1) * extent
        zmin = ijk[2] * extent
        zmax = (ijk[2]+1) * extent

        print(' pSplitSpatial: [%d of %d] ijk (%d %d %d) extent [%g] x [%.1f - %.1f] y [%.1f - %.1f] z [%.1f - %.1f]' % \
            (pSplitSpatial[0],pSplitSpatial[1],ijk[0],ijk[1],ijk[2],extent,xmin,xmax,ymin,ymax,zmin,zmax))

        # which subhalos?
        pos = gc['SubhaloPos'][subhaloIDsTodo]
        w_spatial = np.where( (pos[:,0]>xmin) & (pos[:,0]<=xmax) & \
                              (pos[:,1]>ymin) & (pos[:,1]<=ymax) & \
                              (pos[:,2]>zmin) & (pos[:,2]<=zmax) )

        subhaloIDsTodo = subhaloIDsTodo[w_spatial]

        # generate list of particle indices sufficient for (subhalos,binning specifications)
        mask = np.zeros(sP.numPart[sP.ptNum(ptType)], dtype='int8')
        mask += 1 # all required

        print(' pSplitSpatial:', end='')
        for ind, axis in enumerate(['x','y','z']):
            print(' slice[%s]...' % axis, end='')
            dists = sP.snapshotSubsetP(ptType, 'pos_'+axis, float32=True)

            dists = (ijk[ind] + 0.5) * extent - dists # 1D, along axis, from center of subregion
            sP.correctPeriodicDistVecs(dists)

            # compute maxdist (in code units): the largest 1d distance we need for the calculation
            if Nside is None and not radRvirUnits:
                maxdist = extent / 2 + np.ceil(10.0**radMax) # radMax in log code units
            else:
                # radMax in linear rvir units
                radMaxCode = np.nanmax(gc['Subhalo_Rvir'][subhaloIDsTodo]) * radMax * 1.05
                radMaxSqCode = radMaxCode**2
                maxdist = extent / 2 + radMaxCode

            w_spatial = np.where(np.abs(dists) > maxdist)
            mask[w_spatial] = 0 # outside bounding box along this axis

        load_inds = np.nonzero(mask)[0]
        print('\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%' % \
            (load_inds.size/mask.size*100, 1/pSplitSpatial[1]*100))

        dists = None
        w_spatial = None
        mask = None

    nSubsDo = len(subhaloIDsTodo)

    # info
    print(' ' + desc)
    print(' ' + select)
    print(' Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now...' % \
        (gc['header']['Nsubgroups_Total'],nSubsSelected,nSubsDo))

    # allocate
    if Nside is None:
        # normal profiles" NaN indicates not computed except for mass where 0 will do
        r = np.zeros( (nSubsDo,rad_bin_edges.size-1,numProfTypes), dtype='float32' )
        if numProfTypes == 1: r = np.squeeze(r, axis=2)
    else:
        # spherical sampling
        r = np.zeros( (nSubsDo,nProj,nRad), dtype='float32' )

    #if op not in ['sum']: # does not work with r[i,:] += below!
    #    r.fill(np.nan) # set NaN value for subhalos with e.g. no particles for op=mean

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

    # so long as scope is not global, load the full particle set we need for these subhalos now
    if scope not in ['global','global_fof','global_spatial','subfind_global'] or \
      (scope == 'global_spatial' and pSplitSpatial is None):
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

        if ptProperty not in userCustomFields:
            particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)
            assert particles[ptProperty].ndim == 1

        if 'count' not in particles:
            particles['count'] = particles[list(particles.keys())[0]].shape[0]

        # load weights, e.g. use particle masses or volumes (linear) as weights
        if weighting is not None:
            particles['weights'] = sP.snapshotSubsetP(partType=ptType, fields=weighting, indRange=indRange)

            assert particles['weights'].ndim == 1 and particles['weights'].size == particles['count']

    # if spatial decomposition, load the full particle set we need for these subhalos now
    if pSplitSpatial:
        # use snapshotSubsetC() to avoid ever having the global arrays in memory
        particles = {}

        for field in fieldsLoad:
            data = sP.snapshotSubsetC(partType=ptType, field=field, inds=load_inds, sq=False)
            for key in data: # includes 'count'
                particles[key] = data[key]

        if ptProperty not in userCustomFields:
            particles[ptProperty] = sP.snapshotSubsetC(partType=ptType, field=ptProperty, inds=load_inds)
            assert particles[ptProperty].ndim == 1

        # load weights
        if weighting is not None:
            particles['weights'] = sP.snapshotSubsetC(partType=ptType, field=weighting, inds=load_inds)

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
        
        # construct a global octtree to accelerate searching?
        tree = None
        if useTree:
            tree = buildFullTree(particles['Coordinates'], boxSizeSim=sP.boxSize, treePrec='float64', verbose=True)

        if ptProperty in userCustomFields and Nside is not None: # allocate for local halocentric calculations
            loc_val = np.zeros( particles['count'], dtype='float32' )

        if scope in ['global','global_fof','subfind','fof']:
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
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo), flush=True)

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

            # healpix spherical sampling? calculate now
            if Nside is not None:
                # shift sampling points to be halo local, account for periodic BCs
                pts_loc = pts.copy()
                pts_loc *= gc['Subhalo_Rvir'][subhaloID]
                pts_loc += gc['SubhaloPos'][subhaloID][np.newaxis, :]

                sP.correctPeriodicPosVecs(pts_loc)

                # derive hsml (one per sample point)
                loc_hsml = calcHsml(particles['Coordinates'], sP.boxSize, posSearch=pts_loc, nNGB=Nngb, tree=tree)

                # property
                if ptProperty not in userCustomFields:
                    loc_val = particles[ptProperty]
                elif ptProperty == 'radvel':
                    # take tree search spatial subset
                    radMaxCode = gc['Subhalo_Rvir'][subhaloID] * radMax * 1.05 # maxRad in linear rvir units

                    loc_inds = calcParticleIndices(particles['Coordinates'], gc['SubhaloPos'][subhaloID,:], 
                                                   radMaxCode, boxSizeSim=sP.boxSize, tree=tree)

                    # TODO: de-duplicate 'radvel' logic from below
                    p_pos = np.squeeze( particles['Coordinates'][loc_inds,:] )
                    p_vel = np.squeeze( particles['Velocities'][loc_inds,:] )

                    haloPos = gc['SubhaloPos'][subhaloID,:]
                    haloVel = gc['SubhaloVel'][subhaloID,:]

                    # only compute particleRadialVelInKmS() on the local subset for efficiency
                    loc_val *= 0
                    loc_val[loc_inds] = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)
                else:
                    raise Exception('Unhandled.')

                # sample (note: cannot modify e.g. subset loc_pos, must correspond to the constructed tree)
                loc_pos = particles['Coordinates']

                result = calcQuantReduction(loc_pos, loc_val, loc_hsml, op, sP.boxSize, posSearch=pts_loc, tree=tree)
                result = np.reshape(result, (nProj,nRad))

                # stamp and continue to next subhalo
                r[i,:,:] = result

                continue

            # tree based search?
            if tree is not None:
                maxSearchRad = radMaxCode
                if radRvirUnits:
                    maxSearchRad = gc['Subhalo_Rvir'][subhaloID] * radMax * 1.05

                loc_inds = calcParticleIndices(particles['Coordinates'], gc['SubhaloPos'][subhaloID,:], 
                                               maxSearchRad, boxSizeSim=sP.boxSize, tree=tree)

                if loc_inds is None:
                    continue # zero particles of this type within search radius
                loc_size = loc_inds.size
            else:
                loc_inds = np.s_[i0:i1] # slice object (will create view)
                loc_size = i1 - i0

            # particle pos subset
            particles_pos = particles['Coordinates'][loc_inds]

            # rotation?
            rotMatrix = None

            if proj2D is not None and isinstance(proj2Daxis, str) and (loc_size > 1): # at least 2 particles
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
                particles_pos = particles_pos.copy()
                particles_pos, _ = rotateCoordinateArray(sP, particles_pos, rotMatrix, shPos)

            # use squared radii and sq distance function
            validMask = np.ones( particles_pos.shape[0], dtype=np.bool )

            if proj2D is None:
                # apply in 3D
                rr = sP.periodicDistsSq( gc['SubhaloPos'][subhaloID,:], particles_pos )
            else:
                # apply in 2D projection, along the specified axis
                pt_2d = gc['SubhaloPos'][subhaloID,:]
                pt_2d = [ pt_2d[p_inds[0]], pt_2d[p_inds[1]] ]
                vecs_2d = np.zeros( (particles_pos.shape[0], 2), dtype=particles_pos.dtype )
                vecs_2d[:,0] = particles_pos[:,p_inds[0]]
                vecs_2d[:,1] = particles_pos[:,p_inds[1]]

                rr = sP.periodicDistsSq( pt_2d, vecs_2d ) # handles 2D

                # enforce depth restriction
                if proj2Ddepth is not None:
                    dist_projDir = particles_pos[:,p_inds[2]].copy() # careful of view
                    dist_projDir -= gc['SubhaloPos'][subhaloID,p_inds[2]]
                    sP.correctPeriodicDistVecs(dist_projDir)
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
                        validMask &= (particles[restrictionField][loc_inds] > val)
                    if inequality == 'lt':
                        validMask &= (particles[restrictionField][loc_inds] <= val)
                    if inequality == 'eq':
                        validMask &= (particles[restrictionField][loc_inds] == val)

            wValid = np.where(validMask)

            if len(wValid[0]) == 0:
                continue # zero length of particles satisfying radial cut and restriction

            if radRvirUnits:
                # linear(radius), in units of rvir
                loc_rr = rr[wValid] / gc['Subhalo_Rvir'][subhaloID]
            else:
                # log(radius), with any zero value set to small (included in first bin)
                loc_rr = logZeroSafe( rr[wValid], zeroVal=radMin-1.0 )

            loc_wt = particles['weights'][loc_inds][wValid] if weighting is not None else None

            if ptProperty not in userCustomFields:
                loc_val = particles[ptProperty][loc_inds][wValid]
            else:
                # user function reduction operations, set loc_val now
                if ptProperty == 'radvel':
                    p_pos  = np.squeeze( particles_pos[wValid,:] )
                    p_vel  = np.squeeze( particles['Velocities'][loc_inds,:][wValid,:] )

                    haloPos = gc['SubhaloPos'][subhaloID,:]
                    haloVel = gc['SubhaloVel'][subhaloID,:]

                    loc_val = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)

                if ptProperty in ['losvel','losvel_abs']:
                    if rotMatrix is None:
                        p_vel   = sP.units.particleCodeVelocityToKms( particles[vel_key][loc_inds][wValid] )
                        assert p_vel.ndim == 1 # otherwise, do the following (old)
                        #p_vel   = p_vel[:,p_inds[2]]
                        haloVel = sP.units.subhaloCodeVelocityToKms( gc['SubhaloVel'][subhaloID] )#[p_inds[2]]
                    else:
                        p_vel   = sP.units.particleCodeVelocityToKms( np.squeeze(particles['Velocities'][loc_inds,:][wValid,:]) )
                        haloVel = sP.units.subhaloCodeVelocityToKms( gc['SubhaloVel'][subhaloID,:] )

                        p_vel = np.array( np.transpose( np.dot(rotMatrix, p_vel.transpose()) ) )
                        p_vel = np.squeeze(p_vel[:,p_inds[2]]) # slice index 2 by convention of rotMatrix

                        haloVel = np.array( np.transpose( np.dot(rotMatrix, haloVel.transpose()) ) )[p_inds[2]][0]

                    loc_val = p_vel - haloVel
                    if ptProperty == 'losvel_abs':
                        loc_val = np.abs(loc_val)

                if ptProperty in ['tff','tcool_tff']:
                    # do per-halo load (not scalable)
                    if scope == 'subhalo':
                        loc_val = sP.snapshotSubset(ptType, ptProperty, subhaloID=subhaloID)
                    elif scope == 'fof':
                        haloID = sP.subhalo(subhaloID)['SubhaloGrNr']
                        loc_val = sP.snapshotSubset(ptType, ptProperty, haloID=haloID)
                    loc_val = loc_val[wValid]

            # weighted histogram (or other op) of rr_log distances
            if scope in ['global','global_fof']:
                # (1) all
                result, _, _ = binned_statistic_weighted(loc_rr, loc_val, statistic=op, bins=rbins_sq, weights=loc_wt)
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
                    result, _, _ = binned_statistic_weighted(loc_rr[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,1] += result

                # (3) other-halo
                w = np.where(loc_mask == 1)

                if len(w[0]):
                    result, _, _ = binned_statistic_weighted(loc_rr[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,2] += result

                if restoreSelf:
                    subhalo_particle_mask[is0:is1] = 1 # restore

                # (4) diffuse
                w = np.where(loc_mask == 0)

                if len(w[0]):
                    result, _, _ = binned_statistic_weighted(loc_rr[w], loc_val[w], statistic=op, bins=rbins_sq, weights=loc_wt, weights_w=w)
                    r[i,:,3] += result
            else:
                # subhalo/fof/global_spatial scope, only compute the self-term, or 'subfind_global' technique
                result, _, _ = binned_statistic_weighted(loc_rr, loc_val, statistic=op, bins=rbins_sq, weights=loc_wt)
                r[i,:] += result

    # return
    if Nside is None:
        attrs = {'Description' : desc.encode('ascii'),
                 'Selection'   : select.encode('ascii'),
                 'ptType'      : ptType.encode('ascii'),
                 'ptProperty'  : ptProperty.encode('ascii'),
                 'weighting'   : str(weighting).encode('ascii'),
                 'rad_bin_edges' : rad_bin_edges,
                 'subhaloIDs'    : subhaloIDsTodo}

        if not radRvirUnits:
            attrs['rad_bins_code'] = rad_bins_code
            attrs['rad_bins_pkpc'] = rad_bins_pkpc
            attrs['bin_volumes_code'] = bin_volumes_code
            attrs['bin_areas_code'] = bin_areas_code
    else:
        attrs = {'Description'   : desc.encode('ascii'),
                 'Selection'     : select.encode('ascii'),
                 'ptType'        : ptType.encode('ascii'),
                 'ptProperty'    : ptProperty.encode('ascii'),
                 'weighting'     : str(weighting).encode('ascii'),
                 'Nside'         : Nside,
                 'Nngb'          : Nngb,
                 'op'            : op,
                 'rad_bins_rvir' : rad_bins_rvir,
                 'subhaloIDs'    : subhaloIDsTodo}

    return r, attrs
