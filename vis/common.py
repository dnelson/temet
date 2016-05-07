"""
common.py
  Visualizations: common routines.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import hashlib
import h5py
from os.path import isfile, isdir
from os import mkdir

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.sphMap import sphMap
from util.helper import loadColorTable, logZeroSafe
from cosmo.load import snapshotSubset, groupCat, groupCatSingle
from cosmo.load import groupCatOffsetListIntoSnap
from cosmo.cloudy import cloudyIon

volDensityFields = ['density']
colDensityFields = ['coldens','coldens_msunkpc2','HI']
totSumFields     = ['mass']

def getHsmlForPartType(sP, partType, indRange=None):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """
    # dark matter
    if sP.isPartType(partType, 'dm'):
        print('WARNING: TEMPORARY TODO, use CalcHSML for DM (with caching).')
        hsml = snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)
        return hsml

    # gas
    if sP.isPartType(partType, 'gas'):
        hsml = snapshotSubset(sP, partType, 'cellrad', indRange=indRange)
        return hsml

    # stars
    if sP.isPartType(partType, 'stars'):
        print('WARNING: TEMPORARY TODO, use CalcHSML for stars (with caching).')
        #hsml = 0.25 * snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)
        #hsml[hsml > 0.1] = 0.1 # can decouple, leads to strageness/interestingness
        hsml = 0.5 * snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)
        return hsml

    raise Exception('Unimplemented partType.')

def meanAngMomVector(sP, subhaloID):
    """ Calculate the 3-vector (x,y,z) of the mean angular momentum of either the star-forming gas 
    or the inner stellar component, for rotation and projection into disk face/edge-on views. """
    # star forming gas
    sh = groupCatSingle(sP, subhaloID=subhaloID)

    fields = ['Coordinates','Masses','StarFormationRate','Velocities']
    gas = snapshotSubset(sP, 'gas', fields, subhaloID=subhaloID)

    w = np.where(gas['StarFormationRate'] > 0.0)[0]

    if not len(w):
        raise Exception('No star-forming gas.')

    gas['Coordinates'] = gas['Coordinates'][w,:]
    gas['Masses']      = gas['Masses'][w]
    gas['Velocities']  = gas['Velocities'][w,:]

    ang_mom = sP.units.particleAngMomVecInKpcKmS( gas['Coordinates'], gas['Velocities'], gas['Masses'], 
                                                  sh['SubhaloPos'], sh['SubhaloVel'] )

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

    return R

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

    return R

def loadMassAndQuantity(sP, partType, partField, indRange=None):
    """ Load the field(s) needed to make a projection type grid, with any unit preprocessing. """
    # mass/weights
    if partType in ['gas','stars']:
        mass = snapshotSubset(sP, partType, 'mass', indRange=indRange)
    elif partType == 'dm':
        mass = sP.dmParticleMass

    # neutral hydrogen mass model
    if partField == 'HI':
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

        ion = cloudyIon(sP, el=element, redshiftInterp=False)
        mass *= ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange)

    # quantity
    normCol = False

    if partField in volDensityFields+colDensityFields+totSumFields or ' ' in partField:
        # distribute mass and calculate column/volume density grid
        quant = None

        if partField in volDensityFields+colDensityFields:
            normCol = True
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        quant = snapshotSubset(sP, partType, partField, indRange=indRange)

    # unit pre-processing (only need to remove log for means)
    if partField in ['temp','temperature','ent','entr','entropy']:
        quant = 10.0**quant

    return mass, quant, normCol

def gridOutputProcess(sP, grid, partType, partField, boxSizeImg):
    """ Perform any final unit conversions on grid output and set field-specific plotting configuration. """
    config = {}

    # volume densities
    if partField in volDensityFields:
        grid /= boxSizeImg[2] # mass/area -> mass/volume (normalizing by projection ray length)

    if partField == 'density':
        grid  = logZeroSafe( sP.units.codeDensToPhys( grid, cgs=True, numDens=True ) )
        config['label']  = 'Mean Volume Density [log cm$^{-3}$]'
        config['ctName'] = 'jet'

    # total sum fields
    if partField == 'mass':
        grid  = logZeroSafe( sP.units.codeMassToLogMsun(grid) )
        config['label']  = 'Total Mass [log M$_{\\rm sun}$]'
        config['ctName'] = 'jet'

    # column densities
    if partField == 'coldens':
        grid  = logZeroSafe( sP.units.codeColDensToPhys( grid, cgs=True, numDens=True ) )
        config['label']  = 'Total Column Density [log cm$^{-2}$]'
        config['ctName'] = 'viridis'

    if partField == 'coldens_msunkpc2':
        grid  = logZeroSafe( sP.units.codeColDensToPhys( grid, msunKpc2=True ) )
        config['label']  = 'Total Column Density [log M$_{\\rm sun}$ kpc$^{-2}$]'

        if sP.isPartType(partType,'gas'):   config['ctName'] = 'cubehelix'
        if sP.isPartType(partType,'stars'): config['ctName'] = 'afmhot'

    if partField == 'HI' or ' ' in partField:
        grid = logZeroSafe( sP.units.codeColDensToPhys(grid, cgs=True, numDens=True) )
        config['label']  = 'N$_{\\rm ' + partField + '}$ [log cm$^{-2}$]'
        config['ctName'] = 'viridis'

    # mass-weighted quantities
    if partField in ['temp','temperature']:
        grid  = logZeroSafe( grid )
        config['label']  = 'Temperature [log K]'
        config['ctName'] = 'jet'

    if partField in ['ent','entr','entropy']:
        grid  = logZeroSafe( grid )
        config['label']  = 'Entropy [log K cm$^2$]'
        config['ctName'] = 'jet'

    if partField == 'velmag':
        print('TODO: handle scale factor if this is particle-level data')
        config['label']  = 'Velocity Magnitude [km/s]'
        config['ctName'] = 'jet'

    # failed to find?
    if 'label' not in config:
        raise Exception('Unrecognized field ['+partField+'].')

    return grid, config

def gridBox(sP, method, partType, partField, nPixels, axes, 
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, **kwargs):
    """ Caching gridding/imaging of a simulation box. """
    m = hashlib.sha256('nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d.%g.rot-%s' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1], 
         hsmlFac, str(rotMatrix))).hexdigest()[::4]

    saveFilename = sP.derivPath + 'grids/%s.%d.%s.%s.%s.hdf5' % \
                   (method, sP.snap, partType, partField.replace(' ','_'), m)

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')

    # map
    if isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid = f['grid'][...]
        print('Loaded: [%s]' % saveFilename.split(sP.derivPath)[1])
    else:
        # will we use a complete load or a subset particle load?
        indRange = None

        # non-zoom simulation and hInd specified (plotting around a single halo): do FoF restricted load
        if not sP.isZoom and sP.hInd is not None:
            sh = groupCatSingle(sP, subhaloID=sP.hInd)

            if not sP.groupOrdered:
                raise Exception('Want to do a group-ordered load but cannot.')

            # calculate indRange
            pt = sP.ptNum(partType)
            startInd = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup'][sh['SubhaloGrNr'],pt]
            indRange = [startInd, startInd + sh['SubhaloLenType'][pt] - 1]

        # load: 3D positions
        pos = snapshotSubset(sP, partType, 'pos', indRange=indRange)

        # rotation? shift points to subhalo center, rotate, and shift back
        if rotMatrix is not None:
            sh = groupCatSingle(sP, subhaloID=sP.hInd)

            if not sP.isZoom and sP.hInd is None:
                raise Exception('Rotation in periodic box must be about a halo center.')

            for i in range(3):
                pos[:,i] -= sh['SubhaloPos'][i]

            pos = np.transpose( np.dot(rotMatrix, pos.transpose()) )

            for i in range(3):
                pos[:,i] += sh['SubhaloPos'][i]

        # load: mass/weights, quantity, and normalization required
        mass, quant, normCol = loadMassAndQuantity(sP, partType, partField, indRange=indRange)

        if method == 'sphMap':
            # particle by particle orthographic splat using standard SPH cubic spline kernel
            hsml = getHsmlForPartType(sP, partType, indRange=indRange) * hsmlFac

            grid = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                           boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, boxCen=boxCenter, nPixels=nPixels, 
                           colDens=normCol )
        else:
            raise Exception('Not implemented.')

        # save
        with h5py.File(saveFilename,'w') as f:
            f['grid'] = grid
        print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    # handle units and come up with units label
    grid, config = gridOutputProcess(sP, grid, partType, partField, boxSizeImg)

    return grid, config

def addBoxMarkers(p, ax):
    """ Factor out common annotation/markers to overlay. """
    if 'plotHalos' in p and p['plotHalos'] > 0:
        # debug plotting N most massive halos
        gc = groupCat(p['sP'], fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

        for j in range(p['plotHalos']):
            xPos = gc['halos']['GroupPos'][j,p['axes'][0]]
            yPos = gc['halos']['GroupPos'][j,p['axes'][1]]
            rad  = gc['halos']['Group_R_Crit200'][j] * 1.0

            c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
            ax.add_artist(c)

    if 'rVirFracs' in p:
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

def renderMultiPanel(panels, plotStyle, rasterPx, saveFilename):
    """ Generalized plotting function which produces a single multi-panel plot with one panel for 
        each of panels, all of which can vary in their configuration. 
      plotStyle    : open, edged
      rasterPx     : each panel will have this number of pixels if making a raster (png) output, 
                     but note also it controls the relative size balance of raster/vector (e.g. fonts)
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

    if plotStyle == 'open':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig = plt.figure(figsize=(1.167*sizeFac*nRows/aspect,sizeFac*nRows))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = p['sP']

            # grid projection for image
            grid, config = gridBox(**p)

            # create this panel, and label
            ax = fig.add_subplot(nRows,nCols,i+1)

            idStr = ''
            if not sP.isZoom and sP.hInd is not None:
                idStr = ' (id=' + str(sP.hInd) + ')'

            ax.set_title('%s z=%3.1f %s %s%s' % (sP.simName,sP.redshift,p['partType'],p['partField'],idStr))

            ax.set_xlabel( ['x','y','z'][p['axes'][0]] + ' [ ckpc/h ]')
            ax.set_ylabel( ['x','y','z'][p['axes'][1]] + ' [ ckpc/h ]')

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
            cmap = loadColorTable(config['ctName'])
            
            plt.imshow(grid, extent=p['extent'], cmap=cmap, aspect=1.0)
            if 'valMinMax' in p: plt.clim( p['valMinMax'] )

            addBoxMarkers(p, ax)

            # colobar
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel(config['label'])

        fig.tight_layout()
        fig.savefig(saveFilename)

    if plotStyle == 'edged':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        fig = plt.figure(frameon=False, tight_layout=False)
        barAreaHeight = 0.1
        
        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig.set_size_inches(sizeFac*nCols, sizeFac*nRows*(1/(1.0-barAreaHeight)))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = p['sP']

            # grid projection for image
            grid, config = gridBox(**p)

            # set axes and place image
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            rowHeight  = (1.0 - barAreaHeight) / nRows
            colWidth   = 1.0 / nCols
            bottomNorm  = 1.0 - rowHeight * (curRow+1)
            leftNorm = colWidth * curCol

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]
            ax = fig.add_axes(pos)
            ax.set_axis_off()

            # color mapping and place image
            cmap = loadColorTable(config['ctName'])

            plt.imshow(grid, extent=p['extent'], cmap=cmap, aspect=1.0)
            if 'valMinMax' in p: plt.clim( p['valMinMax'] )

            addBoxMarkers(p, ax)

            # colobar
            factor  = 0.95 # bar length, fraction of column width, 1.0=whole
            height  = 0.04 # colorbar height, fraction of entire figure
            hOffset = 0.4  # padding between image and start of bar (fraction of height)

            leftNormBar   = leftNorm + 0.5*colWidth*(1-factor)
            bottomNormBar = barAreaHeight - height*hOffset - height

            posBar = [leftNormBar, bottomNormBar, colWidth*factor, height]

            cax = fig.add_axes(posBar)
            cax.set_axis_off()
            cb = plt.colorbar(cax=cax, orientation='horizontal')
            cb.ax.set_ylabel(config['label'])

        fig.savefig(saveFilename)

    plt.close(fig)
