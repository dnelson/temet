"""
common.py
  Visualizations: common routines.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import hashlib
import h5py
from os.path import isfile, isdir, expanduser
from os import mkdir

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from util.sphMap import sphMap
from util.treeSearch import calcHsml
from util.helper import loadColorTable, logZeroMin
from cosmo.load import snapshotSubset, snapshotHeader, snapHasField, subboxVals
from cosmo.load import groupCat, groupCatSingle, groupCatHeader, groupCatOffsetListIntoSnap
from cosmo.util import periodicDists
from cosmo.cloudy import cloudyIon
from illustris_python.util import partTypeNum

# all frames output here (current directory if empty string)
saveBasePath = expanduser("~") + '/data3/frames/'
#saveBasePath = expanduser("~") + '/Dropbox/odyssey/'

# configure certain behavior types
volDensityFields = ['density']
colDensityFields = ['coldens','coldens_msunkpc2','HI','HI_segmented']
totSumFields     = ['mass','mass2']
velLOSFieldNames = ['vlos','v_los','vel_los','velocity_los','vel_line_of_sight']

def getHsmlForPartType(sP, partType, indRange=None):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """

    _, sbStr, _ = subboxVals(sP.subbox)
    irStr = '' if indRange is None else '.%d-%d' % (indRange[0],indRange[1])
    saveFilename = sP.derivPath + 'hsml/hsml.%s%d.%s%s.hdf5' % \
                   (sbStr, sP.snap, partType, irStr)

    if not isdir(sP.derivPath + 'hsml/'):
        mkdir(sP.derivPath + 'hsml/')

    # cache?
    useCache = sP.isPartType(partType, 'stars') or \
              (sP.isPartType(partType, 'dm') and not snapHasField(sP, partType, 'SubfindHsml'))

    if useCache and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            hsml = f['hsml'][()]
        print(' loaded: [%s]' % saveFilename.split(sP.derivPath)[1])

    else:
        # dark matter
        if sP.isPartType(partType, 'dm'):
            if not snapHasField(sP, partType, 'SubfindHsml'):
                pos = snapshotSubset(sP, partType, 'pos', indRange=indRange)
                treePrec = 'single' if pos.dtype == np.float32 else 'double'
                hsml = calcHsml(pos, sP.boxSize, nNGB=64, nNGBDev=4, treePrec=treePrec)
            else:
                hsml = snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)

        # gas
        if sP.isPartType(partType, 'gas'):
            hsml = snapshotSubset(sP, partType, 'cellrad', indRange=indRange)

        # stars
        if sP.isPartType(partType, 'stars'):
            # SubfindHsml is a density estimator of the local DM, don't genreally use for stars
            pos = snapshotSubset(sP, partType, 'pos', indRange=indRange)
            treePrec = 'double' #'single' if pos.dtype == np.float32 else 'double'
            hsml = calcHsml(pos, sP.boxSize, nNGB=64, nNGBDev=2, treePrec=treePrec)

        # save
        if useCache:
            with h5py.File(saveFilename,'w') as f:
                f['hsml'] = hsml
            print(' saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    return hsml

    raise Exception('Unimplemented partType.')

def meanAngMomVector(sP, subhaloID, shPos=None, shVel=None):
    """ Calculate the 3-vector (x,y,z) of the mean angular momentum of either the star-forming gas 
    or the inner stellar component, for rotation and projection into disk face/edge-on views. """
    sh = groupCatSingle(sP, subhaloID=subhaloID)

    # allow center pos/vel to be input (e.g. mpb smoothed values)
    if shPos is None: shPos = sh['SubhaloPos']
    if shVel is None: shVel = sh['SubhaloVel']

    fields = ['Coordinates','Masses','StarFormationRate','Velocities']
    gas = snapshotSubset(sP, 'gas', fields, subhaloID=subhaloID)

    # star forming gas only
    wGas = np.where(gas['StarFormationRate'] > 0.0)[0]

    # add stars within 1 times the stellar half mass radius
    fields.remove('StarFormationRate')
    stars = snapshotSubset(sP, 'stars', fields, subhaloID=subhaloID)

    rad = periodicDists(shPos, stars['Coordinates'], sP)

    wStars = np.where(rad <= sh['SubhaloHalfmassRadType'][sP.ptNum('stars')])[0]

    # return default vector in case of total failure
    if len(wGas) + len(wStars) == 0:
        print('meanAngMomVector(): No star-forming gas or stars in radius, returning [1,0,0].')
        return np.array([1.0,0.0,0.0], dtype='float32')

    # combine gas and stars with restrictions (no gas or no stars is ok)
    pos  = np.vstack( (gas['Coordinates'][wGas,:], stars['Coordinates'][wStars,:]) )
    mass = np.hstack( (gas['Masses'][wGas], stars['Masses'][wStars]) )
    vel  = np.vstack( (gas['Velocities'][wGas,:], stars['Velocities'][wStars,:]) )

    ang_mom = sP.units.particleAngMomVecInKpcKmS( pos, vel, mass, shPos, shVel )

    # calculate mean angular momentum unit 3-vector
    ang_mom_mean = np.mean(ang_mom, axis=0)
    ang_mom_mean /= np.linalg.norm(ang_mom_mean,2)

    return ang_mom_mean

def rotationMatrixFromVec(vec, target_vec=(0,0,1)):
    """ Calculate 3x3 rotation matrix to align input vec with a target vector. By default this is the 
    z-axis, such that with vec the angular momentum vector of the galaxy, an (x,y) projection will 
    yield a face on view, and an (x,z) projection will yield an edge on view. """

    # verify we have unit vectors
    vec /= np.linalg.norm(vec,2)
    target_vec /= np.linalg.norm(target_vec,2)

    v = np.cross(vec,target_vec)
    s = np.linalg.norm(v,2)
    c = np.dot(vec,target_vec)

    # v_x is the skew-symmetric cross-product matrix of v
    I = np.identity(3)
    v_x = np.asmatrix( np.array( [ [0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0] ] ) )

    # R * (x,y,z)_unrotated == (x,y,z)_rotated
    R = I + v_x + v_x**2 * (1-c/s**2.0)

    return R.astype('float32')

def rotationMatrixFromAngleDirection(angle, direction):
    """ Calculate 3x3 rotation matrix for input angle about an axis defined by the input direction 
    about the origin. """

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    direction /= np.lingalg.norm(direction,2)

    # rotation matrix about unit vector
    R = np.diag( [cos_a, cos_a, cos_a] )
    R += np.outer(direction, direction) * (1.0 - cos_a)
    direction *= sin_a

    R += np.array( [[0.0, -direction[2], direction[1]], 
                    [direction[2], 0.0, -direction[0]], 
                    [-direction[1], direction[0], 0.0]] )

    return R.astype('float32')

def loadMassAndQuantity(sP, partType, partField, indRange=None):
    """ Load the field(s) needed to make a projection type grid, with any unit preprocessing. """
    # mass/weights
    if partType in ['gas','stars']:
        mass = snapshotSubset(sP, partType, 'mass', indRange=indRange)
    elif partType == 'dm':
        mass = sP.dmParticleMass

    # neutral hydrogen mass model
    if partField in ['HI','HI_segmented']:
        nh0_frac = snapshotSubset(sP, partType, 'NeutralHydrogenAbundance', indRange=indRange)

        # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
        #mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2'), totalNeutral=(species=='HI_noH2'))
        # simplified models (difference is quite small in CDDF)
        ##mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
        ##mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

        mass *= sP.units.hydrogen_massfrac * nh0_frac

    # metal ion mass
    if ' ' in partField:
        element = partField.split()[0]
        ionNum  = partField.split()[1]

        ion = cloudyIon(sP, el=element, redshiftInterp=True)
        mass *= ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange)

        mass[mass < 0] = 0.0 # clip -eps values to 0.0

    # quantity relies on a non-trivial computation / load of another quantity
    partFieldLoad = partField

    if partField in ['vrad','vrad_vvir']:
        partFieldLoad = 'vel'
    if partField in velLOSFieldNames:
        partFieldLoad = 'vel'

    # quantity and column density normalization
    normCol = False

    if partFieldLoad in volDensityFields+colDensityFields+totSumFields or ' ' in partFieldLoad:
        # distribute mass and calculate column/volume density grid
        quant = None

        if partFieldLoad in volDensityFields+colDensityFields or ' ' in partFieldLoad:
            normCol = True
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        quant = snapshotSubset(sP, partType, partFieldLoad, indRange=indRange)

    # quantity pre-processing (need to remove log for means)
    if partField in ['temp','temperature','ent','entr','entropy','P_gas','P_B']:
        quant = 10.0**quant

    if partField in ['vrad','vrad_vvir']:
        raise Exception('Not implemented (and remove duplication with tracerMC somehow?)')

    if partField in velLOSFieldNames:
        quant = sP.units.particleCodeVelocityToKms(quant) # could add hubble expansion

    if partField in ['TimebinHydro']: # cast integers to float
        quant = np.float32(quant)

    # protect against scalar/0-dimensional (e.g. single particle) arrays
    if quant is not None and quant.size == 1 and quant.ndim == 0:
        quant = np.array([quant])

    return mass, quant, normCol

def gridOutputProcess(sP, grid, partType, partField, boxSizeImg):
    """ Perform any final unit conversions on grid output and set field-specific plotting configuration. """
    config = {}

    if sP.isPartType(partType,'dm'):    ptStr = 'DM'
    if sP.isPartType(partType,'gas'):   ptStr = 'Gas'
    if sP.isPartType(partType,'stars'): ptStr = 'Stellar'

    # volume densities
    if partField in volDensityFields:
        grid /= boxSizeImg[2] # mass/area -> mass/volume (normalizing by projection ray length)

    if partField in ['dens','density']:
        grid  = logZeroMin( sP.units.codeDensToPhys( grid, cgs=True, numDens=True ) )
        config['label']  = 'Mean %s Volume Density [log cm$^{-3}$]' % ptStr
        config['ctName'] = 'jet'

    # total sum fields
    if partField == 'mass':
        grid  = logZeroMin( sP.units.codeMassToLogMsun(grid) )
        config['label']  = 'Total %s Mass [log M$_{\\rm sun}$]' % ptStr
        config['ctName'] = 'jet'

    if partField == 'mass2':
        grid  = logZeroMin( sP.units.codeMassToLogMsun(grid) )
        config['label']  = 'Total %s Mass SQ [who knows]' % ptStr
        config['ctName'] = 'jet'

    # column densities
    if partField == 'coldens':
        grid  = logZeroMin( sP.units.codeColDensToPhys( grid, cgs=True, numDens=True ) )
        config['label']  = '%s Column Density [log cm$^{-2}$]' % ptStr
        config['ctName'] = 'cubehelix'

    if partField in ['coldens_msunkpc2']:
        if partField == 'coldens_msunkpc2':
            grid  = logZeroMin( sP.units.codeColDensToPhys( grid, msunKpc2=True ) )
            config['label']  = '%s Column Density [log M$_{\\rm sun}$ kpc$^{-2}$]' % ptStr

        if sP.isPartType(partType,'dm'):    config['ctName'] = 'dmdens'
        if sP.isPartType(partType,'gas'):   config['ctName'] = 'magma'
        if sP.isPartType(partType,'stars'): config['ctName'] = 'gray' #'cubehelix'

    if partField in ['HI','HI_segmented'] or ' ' in partField:
        if ' ' in partField:
            ion = cloudyIon(None)
            grid /= ion.atomicMass(partField.split()[0]) # [H atoms/cm^2] to [ions/cm^2]

        grid = logZeroMin( sP.units.codeColDensToPhys(grid, cgs=True, numDens=True) )
        config['label']  = 'N$_{\\rm ' + partField + '}$ [log cm$^{-2}$]'
        config['ctName'] = 'viridis'

    if partField == 'HI_segmented':
        config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
        config['ctName'] = 'HI_segmented'

    # gas: mass-weighted quantities
    if partField in ['temp','temperature']:
        grid = logZeroMin( grid )
        config['label']  = 'Temperature [log K]'
        config['ctName'] = 'jet'

    if partField in ['ent','entr','entropy']:
        grid = logZeroMin( grid )
        config['label']  = 'Entropy [log K cm$^2$]'
        config['ctName'] = 'jet'

    if partField in ['bmag']:
        grid = logZeroMin( grid )
        config['label']  = 'Mean Magnetic Field Magnitude [log G]'
        config['ctName'] = 'jet'

    # todo: sqrt(magnetic energy / volume)

    # gas: pressures
    if partField in ['P_gas']:
        grid = logZeroMin( grid )
        config['label']  = 'Gas Pressure [log K cm$^{-3}$]'
        config['ctName'] = 'viridis'

    if partField in ['P_B']:
        grid = logZeroMin( grid )
        config['label']  = 'Magnetic Pressure [log K cm$^{-3}$]'
        config['ctName'] = 'viridis'

    if partField in ['pressure_ratio']:
        grid = logZeroMin( grid )
        config['label']  = 'Pressure Ratio [log P$_{\\rm B}$ / P$_{\\rm gas}$]'
        config['ctName'] = 'Spectral_r' # RdYlBu, Spectral

    # metallicities
    if partField in ['metal','Z']:
        grid = logZeroMin( grid )
        config['label']  = '%s Metallicity [log M$_{\\rm Z}$ / M$_{\\rm tot}$]' % ptStr
        config['ctName'] = 'gist_earth'

    if partField in ['metal_solar','Z_solar']:
        grid = logZeroMin( grid )
        config['label']  = '%s Metallicity [log Z$_{\\rm \odot}$]' % ptStr
        config['ctName'] = 'gist_earth'

    # velocities (mass-weighted)
    if partField in ['vmag','velmag']:
        config['label']  = '%s Velocity Magnitude [km/s]' % ptStr
        config['ctName'] = 'afmhot' # same as pm/f-34-35-36 (illustris)

    if partField in velLOSFieldNames:
        config['label']  = '%s Line of Sight Velocity [km/s]' % ptStr
        config['ctName'] = 'RdBu_r' # bwr, coolwarm, RdBu_r

    if partField == 'vrad':
        config['label']  = '%s Radial Velocity [km/s]' % ptStr
        config['ctName'] = 'brewer-brownpurple'

    if partField == 'vrad_vvir':
        config['label']  = '%s Radial Velocity / Halo v$_{200}$' % ptStr
        config['ctName'] = 'brewer-brownpurple'

    # stars
    if partField in ['star_age','stellar_age']:
        config['label']  = 'Stellar Age [Gyr]'
        config['ctName'] = 'blgrrd_black0'

    # debugging
    if partField in ['TimeStep']:
        grid = logZeroMin( grid )
        config['label']  = 'log (%s TimeStep)' % ptStr
        config['ctName'] = 'viridis_r'

    if partField in ['TimebinHydro']:
        config['label']  = 'TimebinHydro'
        config['ctName'] = 'viridis'

    # failed to find?
    if 'label' not in config:
        raise Exception('Unrecognized field ['+partField+'].')

    return grid, config

def gridBox(sP, method, partType, partField, nPixels, axes, 
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, 
            forceRecalculate=False, smoothFWHM=None, **kwargs):
    """ Caching gridding/imaging of a simulation box. """
    m = hashlib.sha256('nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d.%g.rot-%s' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1], 
         hsmlFac, str(rotMatrix))).hexdigest()[::4]

    _, sbStr, _ = subboxVals(sP.subbox)

    saveFilename = sP.derivPath + 'grids/%s/%s.%s%d.%s.%s.%s.hdf5' % \
                   (sbStr.replace("_","/"), method, sbStr, sP.snap, partType, partField.replace(' ','_'), m)

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')
    if not isdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/")):
        mkdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/"))

    # no particles of type exist? blank grid return (otherwise die in getHsml and wind removal)
    h = snapshotHeader(sP)

    def emptyReturn():
        print('Warning: No particles, returning empty for [%s]!' % saveFilename.split(sP.derivPath)[1])
        grid = np.zeros( nPixels, dtype='float32' )
        grid, config = gridOutputProcess(sP, grid, partType, partField, boxSizeImg)
        return grid, config

    if h['NumPart'][sP.ptNum(partType)] <= 2:
        return emptyReturn()

    # map
    if not forceRecalculate and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid_master = f['grid'][...]
        print('Loaded: [%s]' % saveFilename.split(sP.derivPath)[1])
    else:
        # will we use a complete load or a subset particle load?
        indRange = None

        # non-zoom simulation and hInd specified (plotting around a single halo): do FoF restricted load
        if not sP.isZoom and sP.hInd is not None and '_global' not in method:
            sh = groupCatSingle(sP, subhaloID=sP.hInd)
            gr = groupCatSingle(sP, haloID=sh['SubhaloGrNr'])

            if not sP.groupOrdered:
                raise Exception('Want to do a group-ordered load but cannot.')

            # calculate indRange
            pt = sP.ptNum(partType)
            startInd = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup'][sh['SubhaloGrNr'],pt]
            indRange = [startInd, startInd + gr['GroupLenType'][pt] - 1]

        # if indRange is still None (full snapshot load), we will proceed chunked, unless we need
        # a full tree construction to calculate hsml values
        grid_dens  = np.zeros( nPixels, dtype='float32' )
        grid_quant = np.zeros( nPixels, dtype='float32' )
        nChunks = 1

        disableChunkLoad = (sP.isPartType(partType,'stars') or sP.isPartType(partType,'dm')) \
                           and not snapHasField(sP, partType, 'SubfindHsml')

        if indRange is None and sP.subbox is None and not disableChunkLoad:
            nChunks = np.max( [1, int(h['NumPart'][partTypeNum(partType)]**(1.0/3.0) / 10.0)] )
            chunkSize = int(h['NumPart'][partTypeNum(partType)] / nChunks)
            print(' gridBox(): proceeding for (%s %s) with [%d] chunks...' % (partType,partField,nChunks))

        for chunkNum in np.arange(nChunks):
            # only if nChunks>1 do we here modify indRange
            if nChunks > 1:
                # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
                indRange = [chunkNum*chunkSize, (chunkNum+1)*chunkSize-1]
                if chunkNum == nChunks-1: indRange[1] = h['NumPart'][sP.ptNum(partType)]-1
                print('  [%2d] %11d - %d' % (chunkNum,indRange[0],indRange[1]))

            # load: 3D positions
            pos = snapshotSubset(sP, partType, 'pos', indRange=indRange)

            # rotation? shift points to subhalo center, rotate, and shift back
            if rotMatrix is not None:
                if rotCenter is None:
                    # use subhalo center at this snapshot
                    sh = groupCatSingle(sP, subhaloID=sP.hInd)
                    rotCenter = sh['SubhaloPos']

                    if not sP.isZoom and sP.hInd is None:
                        raise Exception('Rotation in periodic box must be about a halo center.')

                for i in range(3):
                    pos[:,i] -= rotCenter[i]

                pos = np.transpose( np.dot(rotMatrix, pos.transpose()) )

                for i in range(3):
                    pos[:,i] += rotCenter[i]

            # load: mass/weights, quantity, and normalization required
            mass, quant, normCol = loadMassAndQuantity(sP, partType, partField, indRange=indRange)

            # rotation? handle for view dependent quantities (e.g. velLOS)
            if partField in velLOSFieldNames:
                # first compensate for subhalo CM motion (if this is a halo plot)
                if sP.isZoom or sP.hInd is not None:
                    sh = groupCatSingle(sP, subhaloID=sP.zoomSubhaloID if sP.isZoom else sP.hInd)
                    for i in range(3):
                        # SubhaloVel already peculiar, quant converted already in loadMassAndQuantity()
                        quant[:,i] -= sh['SubhaloVel'][i] 

                # slice corresponding to (optionally rotated) LOS component
                if rotMatrix is None:
                    quant = quant[:,3-axes[0]-axes[1]]
                else:
                    quant = np.transpose( np.dot(rotMatrix, quant.transpose()) )
                    quant = np.squeeze( np.array(quant[:,2]) )

            assert quant is None or quant.ndim == 1 # must be scalar

            # stars requested in run with winds? if so, load SFTime to remove contaminating wind particles
            wMask = None

            if partType == 'stars' and sP.winds:
                sftime = snapshotSubset(sP, partType, 'sftime', indRange=indRange)
                wMask = np.where(sftime > 0.0)[0]
                if len(wMask) <= 2 and nChunks == 1:
                    return emptyReturn()

                mass = mass[wMask]
                pos  = pos[wMask,:]
                if quant is not None:
                    quant = quant[wMask]

            # render
            if method in ['sphMap','sphMap_global']:
                # particle by particle orthographic splat using standard SPH cubic spline kernel
                hsml = getHsmlForPartType(sP, partType, indRange=indRange)
                hsml *= hsmlFac

                if wMask is not None:
                    hsml = hsml[wMask]

                if sP.isPartType(partType, 'stars'):
                    # use a minimum/maximum size for stars in outskirts
                    #hsml[hsml < 0.05*sP.gravSoft] = 0.05*sP.gravSoft
                    #hsml[hsml > 2.0*sP.gravSoft] = 2.0*sP.gravSoft # can decouple, leads to strageness
                    # adaptively clip in proportion to pixel scale of image, depending on ~pixel number
                    pxScale = np.max(boxSizeImg[axes] / nPixels)
                    #clipAboveNumPx = 30.0*(np.max(nPixels)/1920)
                    #clipAboveToPx  = np.max([3.0, 6.0-2*1920/np.max(nPixels)])
                    #hsml[hsml > clipAboveNumPx*pxScale] = clipAboveToPx*pxScale
                    clipPx  = np.max([3.0, 6.0/(1920/np.max(nPixels))])
                    hsml[hsml > clipPx*pxScale] = clipPx*pxScale

                grid_d, grid_q = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                                         boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, boxCen=boxCenter, 
                                         nPixels=nPixels, colDens=normCol, multi=True )

                grid_dens  += grid_d
                grid_quant += grid_q
            else:
                raise Exception('Not implemented.')

        # normalize quantity
        grid_master = grid_dens

        if quant is not None:
            w = np.where(grid_dens > 0.0)
            grid_quant[w] /= grid_dens[w]
            grid_master = grid_quant

        # save
        with h5py.File(saveFilename,'w') as f:
            f['grid'] = grid_master
        print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    # handle units and come up with units label
    grid_master, config = gridOutputProcess(sP, grid_master, partType, partField, boxSizeImg)

    # temporary: something a bit peculiar here, request an entirely different grid and 
    # clip the line of sight to zero (or nan) where log(n_HI)<19.0 cm^(-2)
    if partField in velLOSFieldNames:
        grid_nHI, _ = gridBox(sP, method, 'gas', 'HI_segmented', nPixels, axes, 
                           boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, smoothFWHM=smoothFWHM)

        grid_master[grid_nHI < 19.0] = np.nan

    # smooth down to some resolution by convolving with a Gaussian?
    if smoothFWHM is not None:
        # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
        sigma_xy = (smoothFWHM / 2.3548) / (boxSizeImg[axes] / nPixels) 
        print('fwhm: ',smoothFWHM,sigma_xy)
        grid_master = gaussian_filter(grid_master, sigma_xy, mode='reflect', truncate=5.0)

    return grid_master, config

def addBoxMarkers(p, conf, ax):
    """ Factor out common annotation/markers to overlay. """
    if 'plotHalos' in p and p['plotHalos'] > 0:
        # plotting N most massive halos
        h = groupCatHeader(p['sP'])

        if h['Ngroups_Total'] > 0:
            gc = groupCat(p['sP'], fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

            for j in range(np.min( [p['plotHalos'], h['Ngroups_Total']] )):
                xPos = gc['halos']['GroupPos'][j,p['axes'][0]]
                yPos = gc['halos']['GroupPos'][j,p['axes'][1]]
                rad  = gc['halos']['Group_R_Crit200'][j] * 1.0

                c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                ax.add_artist(c)

    if 'rVirFracs' in p and p['rVirFracs']:
        # plot circles for N fractions of the virial radius
        xPos = p['boxCenter'][0]
        yPos = p['boxCenter'][1]

        if p['relCoords']:
            xPos = 0.0
            yPos = 0.0

        for rVirFrac in p['rVirFracs']:
            rad  = rVirFrac * p['haloVirRad']

            c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
            ax.add_artist(c)

    if 'labelZ' in p and p['labelZ']:
        p0 = plt.Line2D((0,0),(0,0), linestyle='')

        legend = ax.legend([p0], ["z$\,$=$\,$%.2f" % p['sP'].redshift], loc='upper right')
        legend.get_texts()[0].set_color('white')
        ax.add_artist(legend)

    if 'labelSim' in p and p['labelSim']:
        p0 = plt.Line2D((0,0),(0,0), linestyle='')

        legend = ax.legend([p0], [p['sP'].simName], loc='lower right')
        legend.get_texts()[0].set_color('white')

    if 'labelScale' in p and p['labelScale']:
        scaleBarLen = (p['extent'][1]-p['extent'][0])*0.15 # 15% of plot width
        scaleBarLen = 100.0 * np.ceil(scaleBarLen/100.0) # round to nearest 100 code units

        # if scale bar is more than 50% of width, reduce by 10x
        if scaleBarLen >= 0.5 * (p['extent'][1]-p['extent'][0]):
            scaleBarLen /= 10.0

        unitStrs = ['cpc','ckpc','cMpc','cGpc'] # comoving
        unitInd = 1 if p['sP'].mpcUnits is False else 2

        scaleBarStr = "%g %s" % (scaleBarLen, unitStrs[unitInd])
        if scaleBarLen > 900: # use Mpc label
            scaleBarStr = "%g %s" % (scaleBarLen/1000.0, unitStrs[unitInd+1])
        if scaleBarLen < 1: # use pc label
            scaleBarStr = "%g %s" % (scaleBarLen*1000.0, unitStrs[unitInd-1])

        x0 = p['extent'][0] + (p['extent'][1]-p['extent'][0])*0.03 # upper left
        x1 = x0 + p['sP'].units.codeLengthToComovingKpc(scaleBarLen) # actually plot size in ckpc (no h)
        yy = p['extent'][3] - (p['extent'][3]-p['extent'][2])*0.03
        yt = p['extent'][3] - (p['extent'][3]-p['extent'][2])*0.06

        ax.plot( [x0,x1], [yy,yy], '-', color='white', lw=2.0, alpha=1.0)
        ax.text( np.mean([x0,x1]), yt, scaleBarStr, color='white', alpha=1.0, 
                 size='x-large', ha='center', va='center') # same size as legend text

    if 'labelHalo' in p and p['labelHalo']:
        assert p['sP'].hInd is not None

        if not p['sP'].isZoom:
            # periodic box: write properties of the hInd we are centered on
            subhaloID = p['hInd']
        else:
            # zoom run: write properties of zoomTargetHalo
            subhaloID = p['sP'].zoomSubhaloID

        subhalo = groupCatSingle(p['sP'], subhaloID=p['hInd'])
        halo = groupCatSingle(p['sP'], haloID=subhalo['SubhaloGrNr'])

        haloMass = p['sP'].units.codeMassToLogMsun(halo['Group_M_Crit200'])
        stellarMass = p['sP'].units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][p['sP'].ptNum('stars')])

        str1 = "log M$_{\\rm halo}$ = %.1f" % haloMass
        str2 = "log M$_{\\rm star}$ = %.1f" % stellarMass

        p0 = plt.Line2D((0,0),(0,0), linestyle='')
        p1 = plt.Line2D((0,0),(0,0), linestyle='')

        legend = ax.legend([p0,p1], [str1,str2], fontsize='xx-large', loc='lower right')
        legend.get_texts()[0].set_color('white')
        legend.get_texts()[1].set_color('white')

def setAxisColors(ax, color2):
    """ Factor out common axis color commands. """
    ax.title.set_color(color2)
    ax.yaxis.label.set_color(color2)
    ax.xaxis.label.set_color(color2)

    for s in ['bottom','left','top','right']:
        ax.spines[s].set_color(color2)
    for a in ['x','y']:
        ax.tick_params(axis=a, colors=color2)

def addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                       rowHeight, colWidth, bottomNorm, leftNorm):
    """ Add colorbar(s) with custom positioning and labeling, either below or above panels. """
    if not conf.colorbars:
        return

    factor  = 0.80 # bar length, fraction of column width, 1.0=whole
    height  = 0.04 # colorbar height, fraction of entire figure
    hOffset = 0.4  # padding between image and top of bar (fraction of bar height)
    tOffset = 0.15 # padding between top of bar and top of text label (fraction of bar height)
    lOffset = 0.01 # padding between colorbar edges and end label (frac of bar width)

    height *= heightFac

    if barAreaTop == 0.0:
        # bottom
        bottomNormBar = barAreaBottom - height*(hOffset+1.0)
        textTopY = -tOffset
        textMidY = 0.5
    else:
        # top
        bottomNormBar = (1.0-barAreaTop) + height*hOffset
        textTopY = 1.0 + tOffset
        textMidY = 0.45 # pixel adjust down by 1 hack

    leftNormBar = leftNorm + 0.5*colWidth*(1-factor)   
    posBar = [leftNormBar, bottomNormBar, colWidth*factor, height]

    # add bounding axis and draw colorbar
    cax = fig.add_axes(posBar)
    cax.set_axis_off()

    colorbar = plt.colorbar(cax=cax, orientation='horizontal')
    colorbar.outline.set_edgecolor(color2)

    # label, centered and below/above
    cax.text(0.5, textTopY, config['label'], color=color2, transform=cax.transAxes, 
             size='x-large', ha='center', va='top' if barAreaTop == 0.0 else 'bottom')

    # tick labels, 5 evenly spaced inside bar
    valLimits = plt.gci().get_clim()

    colorsA = [(1,1,1),(0.9,0.9,0.9),(0.8,0.8,0.8),(0.2,0.2,0.2),(0,0,0)]
    colorsB = ['white','white','white','black','black']

    formatStr = "%.1f" if np.max(np.abs(valLimits)) < 100.0 else "%d"

    cax.text(0.0+lOffset, textMidY, formatStr % (1.0*valLimits[0]+0.0*valLimits[1]), 
        color=colorsB[0], size='x-large', ha='left', va='center', transform=cax.transAxes)
    cax.text(0.25, textMidY, formatStr % (0.75*valLimits[0]+0.25*valLimits[1]), 
        color=colorsB[1], size='x-large', ha='center', va='center', transform=cax.transAxes)
    cax.text(0.5, textMidY, formatStr % (0.5*valLimits[0]+0.5*valLimits[1]), 
        color=colorsB[2], size='x-large', ha='center', va='center', transform=cax.transAxes)
    cax.text(0.75, textMidY, formatStr % (0.25*valLimits[0]+0.75*valLimits[1]), 
        color=colorsB[3], size='x-large', ha='center', va='center', transform=cax.transAxes)
    cax.text(1.0-lOffset, textMidY, formatStr % (0.0*valLimits[0]+1.0*valLimits[1]), 
        color=colorsB[4], size='x-large', ha='right', va='center', transform=cax.transAxes)

def renderMultiPanel(panels, conf):
    """ Generalized plotting function which produces a single multi-panel plot with one panel for 
        each of panels, all of which can vary in their configuration. 
    Global plot configuration options:
      plotStyle    : open (show axes), edged (no axes), open_black, edged_black
      rasterPx     : each panel will have this number of pixels if making a raster (png) output, 
                     but note also it controls the relative size balance of raster/vector (e.g. fonts)
      colorbars    : True or False
      saveFilename : output file name (extension determines type e.g. pdf or png)

    Each panel in panels must be a dictionary containing the following keys:
      sP         : simParams() object specifying the box, e.g. run, res, and redshift
      partType   : dm, gas, stars, tracerMC, ...
      partField  : coldens, coldens_cgs, temp, velmag, entr, ...
      method     : sphMap, voronoi_const, voronoi_grads, ...
      nPixels    : number of pixels per dimension of images when projecting
      axes       : e.g. [0,1] is x,y
      boxSizeImg : (x,y,z) extent of the imaging box in simulation units
      boxCenter  : (x,y,z) coordinates of the imaging box center in simulation units
      extent     : (axis0_min,axis0_max,axis1_min,axis1_max) in simulation units
    """
    assert conf.plotStyle in ['open','open_black','edged','edged_black']

    color1 = 'black' if '_black' in conf.plotStyle else 'white'
    color2 = 'white' if '_black' in conf.plotStyle else 'black'

    # plot sizing and arrangement
    sizeFac = conf.rasterPx / mpl.rcParams['savefig.dpi']
    nRows   = np.floor(np.sqrt(len(panels)))
    nCols   = np.ceil(len(panels) / nRows)
    aspect  = nRows/nCols

    if conf.plotStyle in ['open','open_black']:
        # start plot
        fig = plt.figure(facecolor=color1)

        widthFacCBs = 1.167 if conf.colorbars else 1.0
        fig.set_size_inches(widthFacCBs*sizeFac*nRows/aspect, sizeFac*nRows)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            # grid projection for image
            grid, config = gridBox(**p)

            # create this panel, and label axes and title
            ax = fig.add_subplot(nRows,nCols,i+1)

            sP = p['sP']
            idStr = ' (id=' + str(sP.hInd) + ')' if not sP.isZoom and sP.hInd is not None else ''
            ax.set_title('%s z=%3.1f%s' % (sP.simName,sP.redshift,idStr))
            ax.set_xlabel( ['x','y','z'][p['axes'][0]] + ' [ ckpc/h ]')
            ax.set_ylabel( ['x','y','z'][p['axes'][1]] + ' [ ckpc/h ]')

            setAxisColors(ax, color2)

            # rotation? indicate transformation with axis labels
            if p['rotMatrix'] is not None:
                old_1 = np.zeros( 3, dtype='float32' )
                old_2 = np.zeros( 3, dtype='float32' )
                old_1[p['axes'][0]] = 1.0
                old_2[p['axes'][1]] = 1.0

                new_1 = np.transpose( np.dot(p['rotMatrix'], old_1) )
                new_2 = np.transpose( np.dot(p['rotMatrix'], old_2) )

                ax.set_xlabel( 'rotated: %4.2fx %4.2fy %4.2fz  [ ckpc/h ]' % (new_1[0], new_1[1], new_1[2]))
                ax.set_ylabel( 'rotated: %4.2fx %4.2fy %4.2fz  [ ckpc/h ]' % (new_2[0], new_2[1], new_2[2]))

            # color mapping and place image
            vMM = p['valMinMax'] if 'valMinMax' in p else None
            cmap = loadColorTable(config['ctName'], valMinMax=vMM)
           
            #cmap.set_bad(color='#000000',alpha=1.0) # use black for nan pixels
            #grid = np.ma.array(grid, mask=np.isnan(grid))

            plt.imshow(grid, extent=p['extent'], cmap=cmap, aspect=1.0)
            ax.autoscale(False)
            if 'valMinMax' in p: plt.clim( p['valMinMax'] )

            addBoxMarkers(p, conf, ax)

            # colobar
            if conf.colorbars:
                cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
                setAxisColors(cax, color2)

                cb = plt.colorbar(cax=cax)
                cb.outline.set_edgecolor(color2)
                cb.ax.set_ylabel(config['label'])

        fig.tight_layout()
        if nRows == 1 and nCols == 3: plt.subplots_adjust(top=0.97,bottom=0.06) # fix degenerate case

        fig.savefig(conf.saveFilename, facecolor=fig.get_facecolor())

    if conf.plotStyle in ['edged','edged_black']:
        # colorbar plot area sizing
        barAreaHeight = np.max([0.035,0.1 / nRows]) if conf.colorbars else 0.0
        
        # check uniqueness of panel (partType,partField,valMinMax)'s
        pPartTypes   = set()
        pPartFields  = set()
        pValMinMaxes = set()

        for p in panels:
            pPartTypes.add(p['partType'])
            pPartFields.add(p['partField'])
            pValMinMaxes.add(str(p['valMinMax']))

        # if all panels in the entire figure are the same, we will do 1 single colorbar
        oneGlobalColorbar = False

        if len(pPartTypes) == 1 and len(pPartFields) == 1 and len(pValMinMaxes):
            if None not in pValMinMaxes:
                oneGlobalColorbar = True

        if nRows == 2 and not oneGlobalColorbar:
            # two rows, special case, colors on top and bottom, every panel can be different
            barAreaTop = 1.0 * barAreaHeight
            barAreaBottom = 1.0 * barAreaHeight
        else:
            # colorbars on the bottom of the plot, one per column (columns should be same field/valMinMax)
            barAreaTop = 0.0
            barAreaBottom = barAreaHeight

        if nRows > 2:
            # verify that each column contains the same field and valMinMax
            pass

        # start plot
        fig = plt.figure(frameon=False, tight_layout=False, facecolor=color1)

        width_in  = sizeFac * np.ceil(nCols)
        height_in = sizeFac * np.ceil(nRows) * (1/(1.0-barAreaTop-barAreaBottom))

        fig.set_size_inches(width_in, height_in)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            # grid projection for image
            grid, config = gridBox(**p)

            # set axes coordinates and add
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            rowHeight = (1.0 - barAreaTop - barAreaBottom) / np.ceil(nRows)
            colWidth   = 1.0 / np.ceil(nCols)
            bottomNorm  = (1.0 - barAreaTop) - rowHeight * (curRow+1)
            leftNorm = colWidth * curCol

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]

            ax = fig.add_axes(pos, axisbg=color1)
            ax.set_axis_off()
            setAxisColors(ax, color2)

            # color mapping and place image
            vMM = p['valMinMax'] if 'valMinMax' in p else None
            cmap = loadColorTable(config['ctName'], valMinMax=vMM)

            plt.imshow(grid, extent=p['extent'], cmap=cmap, aspect='equal')
            ax.autoscale(False) # disable re-scaling of axes with any subsequent ax.plot()
            if 'valMinMax' in p: plt.clim( p['valMinMax'] )

            addBoxMarkers(p, conf, ax)

            # colobar(s)
            heightFac = np.max([1.0/nRows, 0.35])

            if oneGlobalColorbar:
                continue

            if nRows == 2:
                # both above and below, one per column
                if curRow == 0:
                    addCustomColorbars(fig, ax, conf, config, heightFac, 0.0, barAreaTop, color2, 
                                       rowHeight, colWidth, bottomNorm, leftNorm)

                if curRow == nRows-1:
                    addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, 0.0, color2, 
                                       rowHeight, colWidth, bottomNorm, leftNorm)
            
            if nRows == 1 or (nRows > 2 and curRow == nRows-1):
                # only below, one per column
                addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                                   rowHeight, colWidth, bottomNorm, leftNorm)

        # global colorbar?
        if oneGlobalColorbar:
            addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                               rowHeight, 0.4, bottomNorm, 0.3)

        fig.savefig(conf.saveFilename, facecolor=fig.get_facecolor())

    plt.close(fig)
