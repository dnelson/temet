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
from cosmo.util import correctPeriodicDistVecs

""" Relatively 'hard-coded' analysis decisions that can be changed. For reference, they are attached 
    as metadata attributes in the auxCat file. """
# Group_*: parameters for computations done over each FoF halo
minNumPartGroup = 100 # only consider halos above a minimum total number of particles? (0=disabled)

# Subhalo_*: parameters for computations done over each Subfind subhalo

# Box_*: parameters for whole box (halo-independent) computations
boxGridSizeHI     = 1.5 # code units, e.g. ckpc/h
boxGridSizeMetals = 5.0 # code units, e.g. ckpc/h

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

def subhaloStellarPhot(sP, iso=None, imf=None, dust=None):
    """ Compute the total band-magnitudes, per subhalo, under the given assumption of 
    an iso(chrone) model, imf model, and dust model. """
    from cosmo.stellarPop import sps

    # which bands? for now, to change, just recompute from scratch
    bands = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
    bands += ['wfc_acs_f606w']
    bands += ['des_y']
    bands += ['jwst_f150w']

    nBands = len(bands)

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust)

    # prepare catalog metadata
    desc   = "Stellar photometrics (light emission) totals by subhalo, in multiple rest-frame bands."
    select = "All Subfind subhalos."

    # load group information
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType','SubhaloHalfmassRadType'])
    gc['subhalos']['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    # allocate return, NaN indicates not computed (no star particles)
    nSubsTot = gc['header']['Nsubgroups_Total']

    r = np.zeros( (nSubsTot, nBands), dtype='float32' )
    r.fill(np.nan)

    print(' Total # Subhalos: %d, processing all [%s] subhalos in [%d] bands...' % (nSubsTot,nSubsTot,nBands))

    # global load of all stars in snapshot
    stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=['initialmass','sftime','metallicity'])

    # loop over all requested bands
    for bandNum, band in enumerate(bands):
        print('  %02d/%02d [%s]' % (bandNum+1,len(bands),band))

        # request magnitudes in this band for all stars
        mags = pop.mags_code_units(sP, band, stars['GFM_StellarFormationTime'], 
                                             stars['GFM_Metallicity'], 
                                             stars['GFM_InitialMass'], retFullPt4Size=True)

        # loop over halos
        for i in range(nSubsTot):
            if i % int(nSubsTot/10) == 0 and i <= nSubsTot:
                print('   %4.1f%%' % (float(i+1)*100.0/nSubsTot))

            # slice starting/ending indices for stars local to this FoF
            i0 = gc['subhalos']['SubhaloOffsetType'][i,sP.ptNum('stars')]
            i1 = i0 + gc['subhalos']['SubhaloLenType'][i,sP.ptNum('stars')]

            if i1 == i0:
                continue # zero length of this type
            
            magsLocal = mags[i0:i1] # wind particles have NaN

            # convert to luminosities, sum together, convert back to a magnitude in this band
            totalLum = np.nansum( np.power(10.0, -0.4 * magsLocal) )

            if totalLum > 0.0:
                r[i,bandNum] = -2.5 * np.log10( totalLum )

    attrs = {'Description' : desc.encode('ascii'), 
             'Selection'   : select.encode('ascii'),
             'bands'       : [b.encode('ascii') for b in bands]}

    return r, attrs

def wholeBoxColDensGrid(sP, species):
    """ Compute a 2D grid of gas column densities [cm^-2] covering the entire simulation box. For 
        example to derive the neutral hydrogen CDDF. The grid has dimensions of boxGridDim x boxGridDim 
        and so a grid cell size of (sP.boxSize/boxGridDim) in each dimension. Strategy is a chunked 
        load of the snapshot files, for each using SPH-kernel deposition to distribute the mass of 
        the requested species (e.g. HI, CIV) in all gas cells onto the grid.
    """
    from cosmo import hydrogen
    from util.sphMap import sphMapWholeBox
    from cosmo.cloudy import cloudyIon
    #from vis.common import getHsmlForPartType, loadMassAndQuantity, gridOutputProcess

    hDensSpecies = ['HI','HI_noH2','HI2','HI3']
    zDensSpecies = ['O VI','O VI 10','O VI 25','O VI solar']

    if species not in hDensSpecies + zDensSpecies + ['Z']:
        raise Exception('Not implemented.')

    # config
    nChunks = 50    # split loading into how many pieces
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

def wholeBoxCDDF(sP, species):
    """ Compute the column density distribution function (CDDF, i.e. histogram) of column densities 
        given a full box colDens grid."""
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

# this dictionary contains a mapping between all auxCatalog fields and their generating functions, 
# where the first sP input is stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group_Mass_Crit500_Type' : partial(fofRadialSumType,ptProperty='mass',rad='Group_R_Crit500'),

   'Subhalo_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none'),
   'Subhalo_StellarPhot_p07c_bc00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='bc00'),
   'Subhalo_StellarPhot_p07k_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='none'),
   'Subhalo_StellarPhot_p07k_bc00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='bc00'),
   'Subhalo_StellarPhot_p07s_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='none'),
   'Subhalo_StellarPhot_p07s_bc00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='bc00'),

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
   'Box_CDDF_nOVI_solar'     : partial(wholeBoxCDDF,species='OVI_solar')
  }
