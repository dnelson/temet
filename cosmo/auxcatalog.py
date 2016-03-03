"""
cosmo/auxcatalog.py
  Cosmological simulations - auxiliary catalog for additional derived galaxy/halo properties.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
import pdb
from illustris_python.util import partTypeNum
from cosmo.util import correctPeriodicDistVecs
from functools import partial

# only compute quantities above a minimum total number of particles threshold? (0=disabled)
# this could be changed, and for reference is attached as an attribute in the auxCat file
minNumPartGroup = 100

def fofRadialSumType(sP, ptProperty, rad, method='B'):
    """ Compute total/sum of a particle property (e.g. mass) for those particles enclosed within one of 
        the SO radii already computed and available in the group catalog (input as a string). Methods A 
        and B restrict this calculation to FoF particles only, whereas method C does a full particle 
        search over the entire box in order to compute the total/sum for each FoF halo.

      Method A: do individual halo loads per halo, one loop over all halos.
      Method B: do a full snapshot load per type, then halo loop and slice per FoF, to cut down on I/O ops. 
      Method C: per type: full snapshot load, construct the global tree, spherical aperture search per FoF.
    """
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

    # square radii, and use custom distance function
    gc['halos'][rad] = gc['halos'][rad] * gc['halos'][rad]

    def periodicDistsSq(pt, vecs, sP):
        """ As cosmo.util.periodicDists() but specialized, without error checking, and no sqrt. """
        xDist = vecs[:,0] - pt[0]
        yDist = vecs[:,1] - pt[1]
        zDist = vecs[:,2] - pt[2]

        correctPeriodicDistVecs(xDist, sP)
        correctPeriodicDistVecs(yDist, sP)
        correctPeriodicDistVecs(zDist, sP)

        return xDist*xDist + yDist*yDist + zDist*zDist

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
        dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], sq=False)

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
        gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos','mass'])

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
        stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','mass','sftime'])

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

    return r, desc, select

# this dictionary contains a mapping between all auxCatalog fields and their generating functions, 
# where the first sP input is stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group/Mass_Crit500_Type' : partial(fofRadialSumType,ptProperty='mass',rad='Group_R_Crit500'),
   'second'                  : 0
  }