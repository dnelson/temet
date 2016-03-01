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
from cosmo.util import periodicDists
from functools import partial

# only compute quantities above a minimum total number of particles threshold? (0=disabled)
# this could be changed, and for reference is attached as an attribute in the auxCat file
minNumPartGroup = 100

def fofRadialSumType(sP, ptProperty, rad):
    """ Compute total/sum of a particle property (e.g. mass) which exists for all particle types, 
        for those particles enclosed within one of the SO radii already computed and available in 
        the group catalog (input as a string). We do this restricted to FoF particles, not with a 
        global snapshot load. """

    ptSaveTypes = {'dm':0, 'gas':1, 'stars':2}
    ptLoadTypes = {'dm':partTypeNum('dm'), 'gas':partTypeNum('gas'), 'stars':partTypeNum('stars')}

    # allocate return (0=gas+wind, 1=dm, 2=stars-wind), load group positions and SO radii
    gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupPos','GroupLen','GroupLenType',rad])
    h = cosmo.load.snapshotHeader(sP)

    r = np.zeros( (gc['header']['Ngroups_Total'], len(ptSaveTypes.keys())), dtype='float32' )
    r.fill(np.nan) # NaN indicates not computed

    # loop over all halos
    Ngroups_Process = len( np.where(gc['halos']['GroupLen'] >= minNumPartGroup)[0] )

    print(' Total # Halos: '+str(gc['header']['Ngroups_Total'])+', above threshold ('+\
          str(minNumPartGroup)+') = '+str(Ngroups_Process))

    for i in np.arange(gc['header']['Ngroups_Total']):
        if i % int(Ngroups_Process/50) == 0:
            print(' %2d%%' % np.floor(float(i+1)*100.0/Ngroups_Process))

        if gc['halos']['GroupLen'][i] < minNumPartGroup:
            continue

        # mark halo as processed (sum will be zero if there are no particles of a give type)
        r[i,:] = 0.0

        # For each type:
        #   1. Load pos (DM), pos/mass (gas), pos/mass/sftime (stars) for this FoF.
        #   2. Calculate periodic distances, (DM: count num within rad, sum massTable*num)
        #      gas/stars: sum mass of those within rad (gas = gas+wind, stars=real stars only)

        # DM
        dm = cosmo.load.snapshotSubset(sP, partType='dm', fields=['pos'], haloID=i, sq=False)

        if dm['count']:
            rDM = periodicDists( gc['halos']['GroupPos'][i,:], dm['Coordinates'], sP )
            wDM = np.where( rDM <= gc['halos'][rad][i] )

            r[i, ptSaveTypes['dm']] = len(wDM[0]) * h['MassTable'][ptLoadTypes['dm']]

        # GAS
        gas = cosmo.load.snapshotSubset(sP, partType='gas', fields=['pos','mass'], haloID=i)

        if gas['count']:
            rGas = periodicDists( gc['halos']['GroupPos'][i,:], gas['Coordinates'], sP )
            wGas = np.where( rGas <= gc['halos'][rad][i] )

            r[i, ptSaveTypes['gas']] = np.sum( gas['Masses'][wGas] )

        # STARS
        stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['pos','mass','sftime'], haloID=i)

        if stars['count']:
            rStars = periodicDists( gc['halos']['GroupPos'][i,:], stars['Coordinates'], sP )
            wWind  = np.where( (rStars <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'] < 0.0) )
            wStars = np.where( (rStars <= gc['halos'][rad][i]) & (stars['GFM_StellarFormationTime'] >= 0.0) )

            r[i, ptSaveTypes['gas']] += np.sum( stars['Masses'][wWind] )
            r[i, ptSaveTypes['stars']] = np.sum( stars['Masses'][wStars] )

        #print(i,gc['halos'][rad][i],r[i,:])
        #print(i,gc['halos']['GroupPos'][i,:],dm['Coordinates'][:,0].min(),dm['Coordinates'][:,0].max(),
        #        dm['Coordinates'][:,1].min(),dm['Coordinates'][:,1].max(),
        #        dm['Coordinates'][:,2].min(),dm['Coordinates'][:,2].max())

    return r

# this dictionary contains a mapping between all auxCatalog fields and their generating functions, 
# where the first sP input is stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group/Mass_Crit500_Type' : partial(fofRadialSumType,ptProperty='mass',rad='Group_R_Crit500'),
   'second'                  : 0
  }