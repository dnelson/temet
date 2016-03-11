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

""" Relatively 'hard-coded' analysis decisions that can be changed. For reference, they are attached 
    as metadata attributes in the auxCat file. """
# Group/*: parameters for computations done over each FoF halo
minNumPartGroup = 100 # only consider halos above a minimum total number of particles? (0=disabled)

# Subhalo/*: parameters for computations done over each Subfind subhalo

# Box/*: parameters for whole box (halo-independent) computations
boxGridSize = 5.0 # code units, e.g. ckpc/h

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

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return r, attrs

def wholeBoxColDensGrid(sP, species):
    """ Compute a 2D grid of gas column densities [cm^-2] covering the entire simulation box. For 
        example to derive the neutral hydrogen CDDF. The grid has dimensions of boxGridDim x boxGridDim 
        and so a grid cell size of (sP.boxSize/boxGridDim) in each dimension. Strategy is a chunked 
        load of the snapshot files, for each using SPH-kernel deposition to distribute the mass of 
        the requested species (e.g. HI, CIV) in all gas cells onto the grid.
    """
    from cosmo import hydrogen
    from util.sphMap import sphMap

    if species not in ['HI','HI_noH2','Z']:
        raise Exception('Not implemented.')

    # config
    nChunks = 20    # split loading into how many pieces
    axes    = [0,1] # x,y projection
    
    # info
    h = cosmo.load.snapshotHeader(sP)

    boxGridDim = round(sP.boxSize / boxGridSize)
    chunkSize = int(h['NumPart'][partTypeNum('gas')] / nChunks)

    if species == 'HI':
        desc = "Square grid of integrated column densities of [HI] in units of [cm^-2]. "
        desc += "Atomic only, H2 calculated and removed."
    if species == 'HI_noH2':
        desc = "Square grid of integrated column densities of [HI] in units of [cm^-2]. "
        desc += "All neutral hydrogen included, any contribution of H2 ignored."
    if species == 'Z':
        desc = "Square grid of mean gas metallicity in units of [log solar]."

    select = "Grid dimensions: %dx%d pixels (cell size = %06.2f codeunits) along axes=[%d,%d]." % \
             (boxGridDim,boxGridDim,boxGridSize,axes[0],axes[1])

    print(' '+desc)
    print(' '+select)
    print(' Total # Snapshot Load Chunks: '+str(nChunks)+' ('+str(chunkSize)+' cells per load)')

    # specify needed data load, and allocate accumulation array(s)
    if species in ['HI','HI_noH2']:
        fields = ['Coordinates','Density','Masses','metals_H','NeutralHydrogenAbundance']

        r = np.zeros( (boxGridDim,boxGridDim), dtype='float32' )
    if species == 'Z':
        fields = ['Coordinates','Density','Masses','GFM_Metallicity']

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
        gas['hsml'] = (vol * 3.0 / (4*np.pi))**(1.0/3.0)

        if species in ['HI','HI_noH2']:
            # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
            mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI'), totalNeutral=(species=='HI_noH2'))

            # grid gas mHI using SPH kernel, return in units of [10^10 Msun * h / ckpc^2]
            ri = sphMap( pos=gas['Coordinates'], hsml=gas['hsml'], mass=mHI, quant=None, axes=axes, 
                         ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=sP.boxSize*np.array([1.0,1.0,1.0]), 
                         boxCen=sP.boxSize*np.array([0.5,0.5,0.5]), nPixels=[boxGridDim,boxGridDim], 
                         colDens=True )

            r += ri

        if species == 'Z':
            # grid total gas mass using SPH kernel, return in units of [10^10 Msun / h]
            rMi = sphMap( pos=gas['Coordinates'], hsml=gas['hsml'], mass=gas['Masses'], quant=None, axes=axes, 
                          ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=sP.boxSize*np.array([1.0,1.0,1.0]), 
                          boxCen=sP.boxSize*np.array([0.5,0.5,0.5]), nPixels=[boxGridDim,boxGridDim] )

            # grid total gas metal mass
            mMetal = gas['Masses'] * gas['GFM_Metallicity']

            rZi = sphMap( pos=gas['Coordinates'], hsml=gas['hsml'], mass=mMetal, quant=None, axes=axes, 
                          ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=sP.boxSize*np.array([1.0,1.0,1.0]), 
                          boxCen=sP.boxSize*np.array([0.5,0.5,0.5]), nPixels=[boxGridDim,boxGridDim] )

            rM += rMi
            rZ += rZi

    # finalize
    if species in ['HI','HI_noH2']:
        # column density: convert units from [code column density, above] to [atoms/cm^2] and take log
        rr = sP.units.codeColDensToPhys(r, cgs=True, numDens=True)
        rr = np.log10(rr)

    if species == 'Z':
        # metallicity: take Z = mass_tot/mass_gas for each pixel, normalize by solar, take log
        rr = rZ / rM
        rr = np.log10( rr / sP.units.Z_solar )

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

def wholeBoxCDDF(sP, species):
    """ Compute the column density distribution function (CDDF, i.e. histogram) of column densities 
        given a full box colDens grid."""
    from cosmo.load import auxCat
    from cosmo.hydrogen import calculateCDDF

    # config
    binSize   = 0.1 # log cm^-2
    binMinMax = [11.0, 24.0] # log cm^-2

    desc   = "Column density distribution function (CDDF) for ["+species+"]. "
    desc  += "Return has shape [2,nBins] where the first slice gives n_HI [cm^-2], the second fN_HI [cm^-2]."
    select = "Binning min: [%g] max: [%g] size: [%g]." % (binMinMax[0], binMinMax[1], binSize)

    # load
    ac = auxCat(sP, fields=['Box/Grid_n'+species])

    # calculate
    fN_HI, n_HI = calculateCDDF(ac['Box/Grid_n'+species], binMinMax[0], binMinMax[1], binSize, sP)

    rr = np.vstack( (n_HI,fN_HI) )
    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii')}

    return rr, attrs

# this dictionary contains a mapping between all auxCatalog fields and their generating functions, 
# where the first sP input is stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group/Mass_Crit500_Type' : partial(fofRadialSumType,ptProperty='mass',rad='Group_R_Crit500'),

   'Box/Grid_nHI'            : partial(wholeBoxColDensGrid,species='HI'),
   'Box/Grid_nHI_noH2'       : partial(wholeBoxColDensGrid,species='HI_noH2'),
   'Box/Grid_Z'              : partial(wholeBoxColDensGrid,species='Z'),

   'Box/CDDF_nHI'            : partial(wholeBoxCDDF,species='HI'),
   'Box/CDDF_nHI_noH2'       : partial(wholeBoxCDDF,species='HI_noH2')
  }
