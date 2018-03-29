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
from vis.lic import line_integral_convolution

from util.sphMap import sphMap
from util.treeSearch import calcHsml
from util.helper import loadColorTable, logZeroMin, logZeroNaN
from cosmo.load import snapshotSubset, snapshotHeader, snapHasField, subboxVals
from cosmo.load import groupCat, groupCatSingle, groupCatHeader, groupCatOffsetListIntoSnap
from cosmo.util import periodicDists, correctPeriodicDistVecs, correctPeriodicPosVecs
from cosmo.cloudy import cloudyIon, cloudyEmission, getEmissionLines
from illustris_python.util import partTypeNum

# all frames output here (current directory if empty string)
savePathDefault = expanduser("~") + '/' #+ '/Dropbox/odyssey/'

# configure certain behavior types
volDensityFields = ['density']
colDensityFields = ['coldens','coldens_msunkpc2','coldens_sq_msunkpc2','HI','HI_segmented',
                    'xray','xray_lum','p_sync_ska','coldens_msun_ster','sfr_msunyrkpc2']
totSumFields     = ['mass','sfr']
velLOSFieldNames = ['vel_los','vel_los_sfrwt','velsigma_los','velsigma_los_sfrwt']
velCompFieldNames = ['vel_x','vel_y','velocity_x','velocity_y']

def getHsmlForPartType(sP, partType, nNGB=64, indRange=None, snapHsmlForStars=False):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """
    _, sbStr, _ = subboxVals(sP.subbox)
    irStr = '' if indRange is None else '.%d-%d' % (indRange[0],indRange[1])
    shStr = '' if snapHsmlForStars is False else '.sv'
    ngStr = '' if nNGB == 64 else '.ngb%d' % nNGB
    saveFilename = sP.derivPath + 'hsml/hsml.%s%d.%s%s%s%s.hdf5' % \
                   (sbStr, sP.snap, partType, irStr, shStr, ngStr)

    if not isdir(sP.derivPath + 'hsml/'):
        mkdir(sP.derivPath + 'hsml/')

    # cache?
    useCache = (sP.isPartType(partType, 'stars') and not snapHsmlForStars) or \
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
                nNGBDev = int( np.sqrt(nNGB)/2 )
                hsml = calcHsml(pos, sP.boxSize, nNGB=nNGB, nNGBDev=nNGBDev, treePrec=treePrec)
            else:
                hsml = snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)

        # gas
        if sP.isPartType(partType, 'gas'):
            hsml = snapshotSubset(sP, partType, 'cellrad', indRange=indRange)

        # stars
        if sP.isPartType(partType, 'stars'):
            # SubfindHsml is a density estimator of the local DM, don't generally use for stars
            if snapHsmlForStars:
                hsml = snapshotSubset(sP, partType, 'SubfindHsml', indRange=indRange)
            else:
                pos = snapshotSubset(sP, partType, 'pos', indRange=indRange)
                treePrec = 'double' #'single' if pos.dtype == np.float32 else 'double'
                hsml = calcHsml(pos, sP.boxSize, nNGB=nNGB, nNGBDev=1, treePrec=treePrec)

        # save
        if useCache:
            with h5py.File(saveFilename,'w') as f:
                f['hsml'] = hsml
            print(' saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    return hsml.astype('float32')

    raise Exception('Unimplemented partType.')

def defaultHsmlFac(partType, partField):
    """ Helper, set default hsmlFac for a given partType/partField if not input. """
    if partType == 'gas':
        return 2.5 # times cellsize
    if partType == 'stars':
        return 1.0 # times nNGB=32 CalcHsml search
    if partType == 'dm':
        return 0.5 # times SubfindHsml or nNGB=64 CalcHsml search

    raise Exception('Unrecognized partType [%s].' % partType)

def clipStellarHSMLs(hsml, sP, pxScale, nPixels, method=2):
    """ Clip input stellar HSMLs/sizes to minimum/maximum values. Work in progress. """

    # use a minimum/maximum size for stars in outskirts
    if method == 0:
        # constant based on numerical resolution
        hsml[hsml < 0.05*sP.gravSoft] = 0.05*sP.gravSoft
        hsml[hsml > 2.0*sP.gravSoft] = 2.0*sP.gravSoft # can decouple, leads to strageness

        #print(' [m0] stellar hsml clip above [%.1f px] below [%.1f px]' % (2.0*sP.gravSoft,0.05*sP.gravSoft))

    # adaptively clip in proportion to pixel scale of image, depending on ~pixel number
    if method == 1:
        # adaptive technique 2 (used for Gauss proposal stellar composite figure)
        clipAboveNumPx = 30.0*(np.max(nPixels)/1920)
        clipAboveToPx  = np.max([5.0, 6.0-2*1920/np.max(nPixels)]) # was 3.0 not 5.0 before composite tests
        hsml[hsml > clipAboveNumPx*pxScale] = clipAboveToPx*pxScale

        #print(' [m1] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method == 2:
        # adaptive technique 1 (preferred) (used for TNG subbox movies)
        #minClipVal = 4.0 # was 3.0 before composite tests # previous
        minClipVal = 30.0*(np.max(nPixels)/1920) # testing for tng.methods2

        #if 'sdss_g' in partField:
        #    minClipVal = 20.0
        #    print(' set minClipVal from 3 to 20 for Blue-channel')

        clipAboveNumPx = np.max([minClipVal, minClipVal*2/(1920/np.max(nPixels))])
        clipAboveToPx = clipAboveNumPx # coupled
        hsml[hsml > clipAboveNumPx*pxScale] = clipAboveToPx*pxScale

        #print(' [m2] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method is None:
        pass
        #print(' hsml clip DISABLED!')

    return hsml

def meanAngMomVector(sP, subhaloID, shPos=None, shVel=None):
    """ Calculate the 3-vector (x,y,z) of the mean angular momentum of either the star-forming gas 
    or the inner stellar component, for rotation and projection into disk face/edge-on views. """
    sh = groupCatSingle(sP, subhaloID=subhaloID)

    # allow center pos/vel to be input (e.g. mpb smoothed values)
    if shPos is None: shPos = sh['SubhaloPos']
    if shVel is None: shVel = sh['SubhaloVel']

    fields = ['Coordinates','Masses','StarFormationRate','Velocities']
    gas = snapshotSubset(sP, 'gas', fields, subhaloID=subhaloID)

    # star forming gas only within a radial restriction
    wGas = []

    if gas['count']:
        rad = periodicDists(shPos, gas['Coordinates'], sP)
        wGas = np.where( (rad <= 1.0*sh['SubhaloHalfmassRadType'][sP.ptNum('stars')]) & \
                         (gas['StarFormationRate'] > 0.0) )[0]

    # add (actual) stars within 1 times the stellar half mass radius
    wStars = []

    fields.remove('StarFormationRate')
    fields.append('GFM_StellarFormationTime')
    stars = snapshotSubset(sP, 'stars', fields, subhaloID=subhaloID)

    if stars['count']:
        rad = periodicDists(shPos, stars['Coordinates'], sP)

        wStars = np.where( (rad <= 1.0*sh['SubhaloHalfmassRadType'][sP.ptNum('stars')]) & \
                           (stars['GFM_StellarFormationTime'] >= 0.0) )[0]

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

def momentOfInertiaTensor(sP, gas=None, stars=None, rHalf=None, shPos=None, subhaloID=None):
    """ Calculate the moment of inertia tensor (3x3 matrix) for a subhalo or halo, given a load 
    of its member gas and stars (at least within 2*rHalf==shHalfMassRadStars) and center position shPos. """
    if subhaloID is not None:
        assert all(v is None for v in [gas,stars,rHalf])
        # load required particle data for this subhalo
        subhalo = groupCatSingle(sP, subhaloID=subhaloID)
        rHalf = subhalo['SubhaloHalfmassRadType'][sP.ptNum('stars')]
        shPos = subhalo['SubhaloPos']

        gas = snapshotSubset(sP, 'gas', fields=['mass','pos','sfr'], subhaloID=subhaloID)
        stars = snapshotSubset(sP, 'stars', fields=['mass','pos','sftime'], subhaloID=subhaloID)
    else:
        assert all(v is not None for v in [gas,stars,rHalf,shPos])

    if not gas['count'] and not stars['count']:
        print('Warning! momentOfInteriaTensor() no stars or gas in subhalo...')
        return np.identity(3)

    useStars = True

    if gas['count'] and len(gas['Masses']) > 1:
        rad_gas = periodicDists(shPos, gas['Coordinates'], sP)
        wGas = np.where( (rad_gas <= 0.5*rHalf) & (gas['StarFormationRate'] > 0.0) )[0]

        if len(wGas) >= 50:
            useStars = False

    if useStars:
        # restrict to real stars
        wValid = np.where( stars['GFM_StellarFormationTime'] > 0.0 )

        if len(wValid[0]) <= 1:
            return np.identity(3)

        stars['Masses'] = stars['Masses'][wValid]
        stars['Coordinates'] = np.squeeze(stars['Coordinates'][wValid,:])

        # use all stars within 1*rHalf
        rad_stars = periodicDists(shPos, stars['Coordinates'], sP)
        wStars = np.where( (rad_stars <= 1.0*rHalf) )

        if len(wStars[0]) <= 1:
            return np.identity(3)

        masses = stars['Masses'][wStars]
        xyz = stars['Coordinates'][wStars,:]
    else:
        # use all star-forming gas cells within 2*rHalf
        wGas = np.where( (rad_gas <= 2.0*rHalf) & (gas['StarFormationRate'] > 0.0) )[0]

        masses = gas['Masses'][wGas]
        xyz = gas['Coordinates'][wGas,:]

    # shift
    xyz = np.squeeze(xyz)

    for i in range(3):
        xyz[:,i] -= shPos[i]

    # if coordinates wrapped box boundary before shift:
    correctPeriodicDistVecs(xyz, sP)

    # construct moment of inertia
    I = np.zeros( (3,3), dtype='float32' )

    I[0,0] = np.sum( masses * (xyz[:,1]*xyz[:,1] + xyz[:,2]*xyz[:,2]) )
    I[1,1] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,2]*xyz[:,2]) )
    I[2,2] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) )
    I[0,1] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,1]) )
    I[0,2] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,2]) )
    I[1,2] = -1 * np.sum( masses * (xyz[:,1]*xyz[:,2]) )
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    return I

def rotationMatricesFromInertiaTensor(I):
    """ Calculate 3x3 rotation matrix by a diagonalization of the moment of inertia tensor. 
    Note the resultant rotation matrices are hard-coded for projection with axes=[0,1] e.g. along z. """

    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(I)

    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]

    # permute the eigenvectors into this order, which is the rotation matrix which orients the 
    # principal axes to the cartesian x,y,z axes, such that if axes=[0,1] we have face-on
    new_matrix = np.matrix( (rotation_matrix[:,sort_inds[0]],
                             rotation_matrix[:,sort_inds[1]],
                             rotation_matrix[:,sort_inds[2]]) )

    # make a random edge on view
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.pi / 2
    psi = 0

    A_00 =  np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    A_01 =  np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    A_02 =  np.sin(psi)*np.sin(theta)
    A_10 = -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    A_11 = -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    A_12 =  np.cos(psi)*np.sin(theta)
    A_20 =  np.sin(theta)*np.sin(phi)
    A_21 = -np.sin(theta)*np.cos(phi)
    A_22 =  np.cos(theta)

    random_edgeon_matrix = np.matrix( ((A_00, A_01, A_02), (A_10, A_11, A_12), (A_20, A_21, A_22)) )

    # prepare return with a few other useful versions of this rotation matrix
    r = {}
    r['face-on'] = new_matrix
    r['edge-on'] = np.matrix( ((1,0,0),(0,0,1),(0,-1,0)) ) * r['face-on']
    r['edge-on-smallest'] = np.matrix( ((0,1,0),(0,0,1),(1,0,0)) ) * r['face-on']
    r['edge-on-random'] = random_edgeon_matrix * r['face-on']
    r['phi'] = phi
    r['identity'] = np.matrix( np.identity(3) )

    return r

def rotationMatrixFromVec(vec, target_vec=(0,0,1)):
    """ Calculate 3x3 rotation matrix to align input vec with a target vector. By default this is the 
    z-axis, such that with vec the angular momentum vector of the galaxy, an (x,y) projection will 
    yield a face on view, and an (x,z) projection will yield an edge on view. """

    # verify we have unit vectors
    vec /= np.linalg.norm(vec,2)
    target_vec /= np.linalg.norm(target_vec,2)

    I = np.identity(3)

    if np.array_equal(vec, target_vec):
        # identity rotation
        return np.asmatrix(I)

    v = np.cross(vec,target_vec)
    s = np.linalg.norm(v,2)
    c = np.dot(vec,target_vec)

    # v_x is the skew-symmetric cross-product matrix of v
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

def rotateCoordinateArray(sP, pos, rotMatrix, rotCenter, shiftBack=True):
    """ Rotate a [N,3] array of Coordinates about rotCenter according to rotMatrix. """
    pos_in = np.array(pos) # do not modify input

    # shift
    for i in range(3):
        pos_in[:,i] -= rotCenter[i]

    # if coordinates wrapped box boundary before shift:
    correctPeriodicDistVecs(pos_in, sP)

    # rotate
    pos_in = np.transpose( np.dot(rotMatrix, pos_in.transpose()) )

    if shiftBack:
        for i in range(3):
            pos_in[:,i] += rotCenter[i]

    # return a symmetric extent which covers the origin-centered particle distribution, which is hard to 
    # recover after we wrap the coordinates back into the box
    extent = np.zeros( 3, dtype='float32' )
    for i in range(3):
        right = pos_in[:,i].max() - rotCenter[i]
        left = rotCenter[i] - pos_in[:,i].min()
        extent[i] = 2.0 * np.max([left,right])

    # place all coordinates back inside [0,sP.boxSize] if necessary:
    if shiftBack:
        correctPeriodicPosVecs(pos_in, sP)

    pos_in = np.asarray(pos_in) # matrix to ndarray

    return pos_in, extent

def stellar3BandCompositeImage(bands, sP, method, nPixels, axes, projType, projParams, boxCenter, boxSizeImg, 
                               hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM):
    """ Generate 3-band RGB composite using starlight in three different passbands. Work in progress. """
    assert len(bands) == 3
    assert projType == 'ortho'

    print('Generating stellar composite with bands: [%s %s %s]' % (bands[0],bands[1],bands[2]))

    band0_grid_mag, _ = gridBox(sP, method, 'stars', 'stellarBand-'+bands[0], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM)
    band1_grid_mag, _ = gridBox(sP, method, 'stars', 'stellarBand-'+bands[1], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM)
    band2_grid_mag, _ = gridBox(sP, method, 'stars', 'stellarBand-'+bands[2], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM)
    #band2b_grid_mag, _ = gridBox(sP, method, 'stars', 'stellarBand-sdss_g', nPixels, axes, boxCenter, 
    #                             boxSizeImg, hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM)

    ww = np.where(band0_grid_mag < 99) # these left at zero
    band0_grid = band0_grid_mag.astype('float32') * 0.0
    band0_grid[ww] = np.power(10.0, -0.4 * band0_grid_mag[ww])

    ww = np.where(band1_grid_mag < 99)
    band1_grid = band1_grid_mag.astype('float32') * 0.0
    band1_grid[ww] = np.power(10.0, -0.4 * band1_grid_mag[ww])

    ww = np.where(band2_grid_mag < 99)
    band2_grid = band2_grid_mag.astype('float32') * 0.0
    band2_grid[ww] = np.power(10.0, -0.4 * band2_grid_mag[ww])

    #ww = np.where(band2b_grid_mag < 99)
    #band2b_grid = band2b_grid_mag.astype('float32') * 0.0
    #band2b_grid[ww] = np.power(10.0, -0.4 * band2b_grid_mag[ww])

    grid_master = np.zeros( (nPixels[1], nPixels[0], 3), dtype='float32' )
    grid_master_u = np.zeros( (nPixels[1], nPixels[0], 3), dtype='uint8' )

    if 0:
        #fac0 = 1.1 # i (red channel)
        #fac1 = 1.0 # r (green channel)
        #fac2 = 1.3 # g (blue channel)
        lupton_alpha = 0.5
        lupton_Q = 0.5
        scale_min = 1.0 # units of linear luminosity

        # make RGB array using arcsinh scaling following Lupton
        #band0_grid *= fac0
        #band1_grid *= fac1
        #band2_grid *= fac2

        inten = (band0_grid + band1_grid + band2_grid) / 3.0
        val = np.arcsinh( lupton_alpha * lupton_Q * (inten - scale_min) ) / lupton_Q

        ww = np.where(inten < scale_min)[0]
        print(' clipping %d of %d' % (len(ww),inten.size))

        inten[inten < scale_min] = 1e100 # since we divide by inten below, this sets the grid here to zero

        grid_master[:,:,0] = band0_grid * val / inten
        grid_master[:,:,1] = band1_grid * val / inten
        grid_master[:,:,2] = band2_grid * val / inten

        # rescale and clip
        #max_rgbval = np.amax(grid_master, axis=2)
        #min_rgbval = np.amin(grid_master, axis=2)

        ######ww_max = np.where(max_rgbval > 1.0)
        #####ww_min = np.where((min_rgbval < 0.0) | (inten < 0.0))
        #ww_max = max_rgbval > 1.0
        #ww_min = min_rgbval < 0.0

        for i in range(3): # rescale each channel individually to one (must reach white)
            maxVal = np.max( grid_master[:,:,i] )
            print('channel [%d] max = %g' % (i,maxVal))
            grid_master[:,:,i] /= maxVal
            grid_master_u[:,:,i] = grid_master[:,:,i] * np.uint8(255)
            #grid_master[ww_max,i] = grid_master[ww_max,i] / max_rgbval[ww_max]
            #grid_master[ww_min,i] = 0.0

    if 0:
        #fac = (1/res)**2 * (pxScale)**2 (maybe)
        dranges = {'snap_K' : [400, 80000], # red
                   'snap_B' : [20, 13000], # green
                   'snap_U' : [13, 20000], # blue
                   \
                   '2mass_ks' : [40, 8000], # red
                   'b' : [2, 3300], # green
                   'u' : [1, 1500], # blue
                   \
                   'wfc_acs_f814w' : [60, 8000], # red
                   'wfc_acs_f606w' : [20, 50000], # green
                   'wfc_acs_f475w' : [3, 20000], # blue
                   \
                   'jwst_f070w' : [4000, 30000], # red  #[400, 30000]
                   'jwst_f115w' : [2000, 85000], # green  #[200, 85000]
                   'jwst_f200w': [1000, 75000], # blue  #[100, 75000]
                   \
                   'sdss_z' : [100, 1000], # red
                   'sdss_i' : [30, 5000], # red
                   'sdss_r' : [30, 6000], # green
                   'sdss_g' : [1, 7000], # blue
                   'sdss_u' : [5, 3000]} # blue

        for i in range(3):
            drange = dranges[bands[i]]
            drange = np.array(drange) * 1.0 #fac
            drange_log = np.log10( drange )

            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid

            print(' ',i,bands[i],drange,grid_loc.mean(),grid_loc.min(),grid_loc.max())

            grid_log = np.log10( np.clip( grid_loc, drange[0], drange[1] ) )
            grid_stretch = (grid_log - drange_log[0]) / (drange_log[1]-drange_log[0])

            grid_master[:,:,i] = grid_stretch
            grid_master_u[:,:,i] = grid_stretch * np.uint8(255)
            #import pdb; pdb.set_trace()
    if 1:
        pxArea = (boxSizeImg[axes[1]] / nPixels[0]) * (boxSizeImg[axes[0]] / nPixels[1])
        pxArea0 = (80.0/960)**2.0 # at which the following ranges were calibrated
        resFac = 1.0 #(512.0/sP.res)**2.0

        minValLog = np.array([2.8,2.8,2.8]) # previous: 3.3
        minValLog = np.log10( (10.0**minValLog) * (pxArea/pxArea0*resFac) )

        #maxValLog = np.array([5.71, 5.68, 5.36])*0.9 # jwst f200w, f115w, f070w # previous
        maxValLog = np.array([5.60, 5.68, 5.36])*1.0 # little less clipping, more yellow/red color

        maxValLog = np.log10( (10.0**maxValLog) * (pxArea/pxArea0*resFac) )
        #print('pxArea*res mod: ',(pxArea/pxArea0*resFac))

        for i in range(3):
            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid

            if 0:
                # testing: add extra partial strength sdss_g band into the blue channel
                if i == 2:
                    grid_loc += 0.5*band2b_grid
                    print(' extra blue channel added float')

            # handle zero values
            ww = np.where(grid_loc == 0.0)
            grid_loc[ww] = grid_loc[np.where(grid_loc > 0.0)].min() * 0.1 # 10x less than min
            grid_log = np.log10( grid_loc )

            # clip and stretch within [minValLog,maxValLog]
            grid_log = np.clip( grid_log, minValLog[i], maxValLog[i] )
            grid_stretch = (grid_log - minValLog[i]) / (maxValLog[i]-minValLog[i])

            grid_master[:,:,i] = grid_stretch
            grid_master_u[:,:,i] = grid_stretch * np.uint8(255)

            #print(' grid: ',i,grid_stretch.min(),grid_stretch.max(),grid_master_u[:,:,i].min(),grid_master_u[:,:,i].max())

        # add in extra blue channel
        if 0:
            print(' extra blue channel added int')
            grid_loc = band2b_grid
            ww = np.where(grid_loc == 0.0)
            grid_loc[ww] = grid_loc[np.where(grid_loc > 0.0)].min() * 0.1 # 10x less than min
            grid_log = np.log10( grid_loc )
            minValLog = grid_log.max()-3
            grid_log = np.clip( grid_log, minValLog, grid_log.max() )
            grid_stretch = (grid_log - minValLog) / (grid_log.max()-minValLog)
            grid_master_u[:,:,2] = np.float32(grid_master_u[:,:,2]) + grid_stretch * 255.0
            grid_master_u[:,:,2] = np.uint8( np.clip( grid_master_u[:,:,2], 0, 255 ) )

        # saturation adjust
        if 0:
            satVal = 1.5 # 0.0 -> b&w, 0.5 -> reduce color saturation by half, 1.0 -> unchanged
            R = grid_master_u[:,:,0]
            G = grid_master_u[:,:,1]
            B = grid_master_u[:,:,2]
            P = np.sqrt( R*R*0.299 + G*G*0.587 + B*B*0.144 ) # standard luminance weights

            ww = np.where((B > 150))
            #grid_master_u[:,:,0] = np.uint8(np.clip( P + (R-P)*satVal, 0, 255 ))
            #grid_master_u[:,:,1] = np.uint8(np.clip( P + (G-P)*satVal, 0, 255 ))
            B[ww] = np.uint8(np.clip( P[ww] + (B[ww]-P[ww])*satVal, 0, 255 ))
            grid_master_u[:,:,2] = B
            print(' adjusted saturation')

        # contrast adjust
        if 1:
            C = 20.0
            F = 259*(C+255) / (255*(259-C))
            for i in range(3):
                new_i = F * (np.float32(grid_master_u[:,:,i]) - 128) + 128
                grid_master_u[:,:,i] = np.uint8( np.clip( new_i, 0, 255 ) )
            #print(' adjusted contrast ',F)

    # DEBUG: dump 16 bit tiff without clipping
    if 0:
        im = np.zeros( (nPixels[0], nPixels[1], 3), dtype='uint16' )

        for i in range(3):
            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid
            
            ww = np.where(grid_loc == 0.0)
            grid_loc[ww] = grid_loc[np.where(grid_loc > 0.0)].min() * 0.1 # 10x less than min
            grid_loc = np.log10( grid_loc )

            # rescale log(lum) into [0,65535]
            mVal = np.uint16(65535)
            grid_out = (grid_loc - grid_loc.min()) / (grid_loc.max()-grid_loc.min()) * mVal
            im[:,:,i] = grid_out
            print(' tiff: ',i,grid_loc.min(),grid_loc.max())

        import skimage.io
        skimage.io.imsave('out_%s.tif' % '-'.join(bands), im, plugin='tifffile')
    # END DEBUG

    config = {'ctName':'gray', 'label':'Stellar Composite [%s]' % ', '.join(bands)}
    return grid_master_u, config

def loadMassAndQuantity(sP, partType, partField, indRange=None):
    """ Load the field(s) needed to make a projection type grid, with any unit preprocessing. """
    # mass/weights
    from cosmo.stellarPop import sps

    if partType in ['dm']:
        mass = sP.dmParticleMass
    else:
        mass = snapshotSubset(sP, partType, 'mass', indRange=indRange).astype('float32')

    # neutral hydrogen mass model (do column densities)
    if partField in ['HI','HI_segmented']:
        nh0_frac = snapshotSubset(sP, partType, 'NeutralHydrogenAbundance', indRange=indRange)

        # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
        #mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2'), totalNeutral=(species=='HI_noH2'))
        # simplified models (difference is quite small in CDDF)
        ##mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
        ##mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

        mass *= sP.units.hydrogen_massfrac * nh0_frac

    # elemental mass fraction (do column densities)
    if 'metals_' in partField:
        elem_mass_frac = snapshotSubset(sP, partType, partField, indRange=indRange)
        mass *= elem_mass_frac

    # metal ion mass (do column densities) [e.g. "O VI", "O VI mass", "O VI frac", "O VI fracmass"]
    if ' ' in partField:
        element = partField.split()[0]
        ionNum  = partField.split()[1]

        if 1:
            # use cache
            mass = snapshotSubset(sP, 'gas', '%s %s mass' % (element,ionNum), indRange=indRange)
        else:
            # previous (always calculate on the fly)
            ion = cloudyIon(sP, el=element, redshiftInterp=True)
            mass *= ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange)

        mass[mass < 0] = 0.0 # clip -eps values to 0.0

    # other total sum fields (replace mass)
    if partField in ['xray','xray_lum']:
        # xray: replace 'mass' with x-ray luminosity [10^-30 erg/s], which is then accumulated into a 
        # total Lx [erg/s] per pixel, and normalized by spatial pixel size into [erg/s/kpc^2]
        mass = snapshotSubset(sP, partType, 'xray', indRange=indRange)

    if partField in ['sfr_msunyrkpc2']:
        mass = snapshotSubset(sP, partType, 'sfr', indRange=indRange)

    # flux/surface brightness (replace mass)
    if 'sb_' in partField: # e.g. ['sb_H-alpha','sb_Lyman-alpha','sb_OVIII']
        # zero contribution from SFing gas cells?
        zeroSfr = False
        if '_sf0' in partField:
            partField = partField.split("_sf0")[0]
            zeroSfr = True
        if '_ster' in partField: partField = partField.replace("_ster","")

        lineName = partField.split("_")[1].replace("-"," ") # e.g. "O--8-16.0067A" -> "O  8 16.0067A"

        # compute line emission flux for each gas cell in [photon/s/cm^2]
        if 1:
            # use cache
            assert not zeroSfr # not implemented in cache
            mass = snapshotSubset(sP, 'gas', '%s flux' % lineName, indRange=indRange)
        else:
            e_interp = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            lum = e_interp.calcGasLineLuminosity(sP, lineName, indRange=indRange)
            wavelength = e_interp.lineWavelength(lineName)
            mass = sP.units.luminosityToFlux(lum, wavelength=wavelength)
            assert mass.min() >= 0.0
            assert np.count_nonzero( np.isnan(mass) ) == 0

        if zeroSfr:
            sfr = snapshotSubset(sP, partType, 'sfr', indRange=indRange)
            w = np.where(sfr > 0.0)
            mass[w] = 0.0

    # single stellar band, replace mass array with linear luminosity of each star particle
    if 'stellarBand-' in partField:
        bands = partField.split("-")[1:]
        assert len(bands) == 1

        pop = sps(sP)
        mass = pop.calcStellarLuminosities(sP, bands[0], indRange=indRange)

    # quantity relies on a non-trivial computation / load of another quantity
    partFieldLoad = partField

    if partField in velLOSFieldNames + velCompFieldNames:
        partFieldLoad = 'vel'

    if partField in ['masspart','particle_mass']:
        partFieldLoad = 'mass'

    # weighted quantity, but using some property other than mass for the weighting (replace now)
    if partField in ['vel_los_sfrwt','velsigma_los_sfrwt']:
        mass = snapshotSubset(sP, partType, 'sfr', indRange=indRange)

    # quantity and column density normalization
    normCol = False

    if partFieldLoad in volDensityFields+colDensityFields+totSumFields or \
      ' ' in partFieldLoad or 'metals_' in partFieldLoad or 'stellarBand-' in partFieldLoad or \
      'sb_' in partFieldLoad:
        # distribute mass and calculate column/volume density grid
        quant = None

        if partFieldLoad in volDensityFields+colDensityFields or \
        (' ' in partFieldLoad and 'mass' not in partFieldLoad and 'frac' not in partFieldLoad):
            normCol = True
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        quant = snapshotSubset(sP, partType, partFieldLoad, indRange=indRange)

    # quantity pre-processing (need to remove log for means)
    if partField in ['temp','temperature','ent','entr','entropy','P_gas','P_B']:
        quant = 10.0**quant

    if partField in ['coldens_sq_msunkpc2']:
        # DM annihilation radiation (see Schaller 2015, Eqn 2 for real units)
        # load density estimate, square, convert back to effective mass (still col dens like)
        dm_vol = snapshotSubset(sP, partType, 'subfind_volume', indRange=indRange)
        mass = (mass / dm_vol)**2.0 * dm_vol

    if partField in velLOSFieldNames + velCompFieldNames:
        quant = sP.units.particleCodeVelocityToKms(quant) # could add hubble expansion

    if partField in ['TimebinHydro','id']: # cast integers to float
        quant = np.float32(quant)

    # protect against scalar/0-dimensional (e.g. single particle) arrays
    if quant is not None and quant.size == 1 and quant.ndim == 0:
        quant = np.array([quant])

    return mass, quant, normCol

def gridOutputProcess(sP, grid, partType, partField, boxSizeImg, nPixels, projType, method=None):
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

    # total sum fields (also of sub-components e.g. "O VI mass")
    if partField == 'mass' or ' mass' in partField:
        grid  = logZeroMin( sP.units.codeMassToMsun(grid) )
        subStr = ' '+' '.join(partField.split()[:-1]) if ' mass' in partField else ''
        config['label']  = 'Total %s%s Mass [log M$_{\\rm sun}$]' % (ptStr,subStr)
        config['ctName'] = 'perula'

    if partField in ['masspart','particle_mass']:
        grid  = logZeroMin( sP.units.codeMassToMsun(grid) )
        config['label']  = '%s Particle Mass [log M$_{\\rm sun}$]' % ptStr
        config['ctName'] = 'perula'

    if partField == 'sfr':
        grid = logZeroMin( grid )
        config['label'] = 'Star Formation Rate [log M$_{\\rm sun}$/yr]'
        config['ctName'] = 'inferno'

    if partField == 'sfr_msunyrkpc2':
        grid = logZeroMin( sP.units.codeColDensToPhys( grid, totKpc2=True ) )
        config['label'] = 'Star Formation Surface Density [log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'inferno'

    # fractional total sum of sub-component relative to total (note: for now, grid is pure mass)
    if ' fracmass' in partField:
        grid  = logZeroMin( sP.units.codeMassToMsun(grid) )
        compStr = ' '.join(partField.split()[:-1])
        config['label']  = '%s Mass / Total %s Mass [log]' % (compStr,ptStr)
        config['ctName'] = 'perula'

    # column densities
    if partField == 'coldens':
        grid  = logZeroMin( sP.units.codeColDensToPhys( grid, cgs=True, numDens=True ) )
        config['label']  = '%s Column Density [log cm$^{-2}$]' % ptStr
        config['ctName'] = 'cubehelix'

    if partField in ['coldens_msunkpc2','coldens_sq_msunkpc2']:
        if partField == 'coldens_msunkpc2':
            grid  = logZeroMin( sP.units.codeColDensToPhys( grid, msunKpc2=True ) )
            config['label']  = '%s Column Density [log M$_{\\rm sun}$ kpc$^{-2}$]' % ptStr
        if partField == 'coldens_sq_msunkpc2':
            # note: units are fake for now
            grid  = logZeroMin( sP.units.codeColDensToPhys( grid, msunKpc2=True ) )
            config['label'] = 'DM Annihilation Radiation [log GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ kpc$^{-2}$]'

        if sP.isPartType(partType,'dm'):    config['ctName'] = 'dmdens_tng' #'gray_r' (pillepich.stellar)
        if sP.isPartType(partType,'gas'):   config['ctName'] = 'gasdens_tng4' # 'perula' (methods2)
        if sP.isPartType(partType,'gas'):   config['plawScale'] = 1.0 # default
        if sP.isPartType(partType,'stars'): config['ctName'] = 'gray' # copper

    if partField in ['coldens_msun_ster']:
        assert projType == 'equirectangular' # otherwise generalize
        # grid is (code mass) / pixelArea where pixelArea is incorrectly constant as:
        pxArea = (2*np.pi/nPixels[0]) * (np.pi/nPixels[1]) # steradian
        grid *= pxArea # remove normalization

        dlat = np.pi/nPixels[1]
        lats = np.linspace(0.0+dlat/2, np.pi-dlat/2, nPixels[1]) # rad, 0 to np.pi from z-axis
        pxAreasByLat = np.sin(lats) * pxArea # infinite at poles, 0 at equator

        for i in range(nPixels[1]): # normalize separately for each latitude
            grid[i,:] /= pxAreasByLat[i]

        grid = logZeroMin( sP.units.codeMassToMsun(grid) ) # log(msun/ster)
        config['label'] = '%s Column Density [log M$_{\\rm sun}$ ster$^{-2}$]' % ptStr

        if sP.isPartType(partType,'dm'):    config['ctName'] = 'dmdens_tng'
        if sP.isPartType(partType,'gas'):   config['ctName'] = 'gray' #'gasdens_tng4'
        if sP.isPartType(partType,'stars'): config['ctName'] = 'gray'

    # hydrogen/metal/ion column densities
    if (' ' in partField and ' mass' not in partField and ' frac' not in partField):
        assert 'sb_' not in partField
        ion = cloudyIon(sP=None)
        grid /= ion.atomicMass(partField.split()[0]) # [H atoms/cm^2] to [ions/cm^2]

        grid = logZeroMin( sP.units.codeColDensToPhys(grid, cgs=True, numDens=True) )
        config['label']  = 'N$_{\\rm ' + partField + '}$ [log cm$^{-2}$]'
        config['ctName'] = 'viridis'
        if partField == 'O VII': config['ctName'] = 'magma'
        if partField == 'O VIII': config['ctName'] = 'plasma'

    if '_ionmassratio' in partField:
        ion = cloudyIon(sP=None)
        ion1, ion2, _ = partField.split('_')

        grid  = logZeroMin( grid )
        config['label']  = '%s / %s Mass Ratio [log]' % (ion.formatWithSpace(ion1),ion.formatWithSpace(ion2))
        config['ctName'] = 'Spectral'
        config['plawScale'] = 0.6

    if partField in ['HI','HI_segmented']:
        grid = logZeroMin( sP.units.codeColDensToPhys(grid, cgs=True, numDens=True) )

        if partField == 'HI':
            config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
            config['ctName'] = 'viridis'
        if partField == 'HI_segmented':
            config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
            config['ctName'] = 'HI_segmented'

    if partField in ['xray','xray_lum']:
        grid = logZeroMin( sP.units.codeColDensToPhys( grid*1e30, totKpc2=True ) ) # return 1e30 factor
        config['label']  = 'Gas Bolometric L$_{\\rm X}$ [log erg s$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'inferno'

    if partField in ['p_sync_ska']:
        grid = logZeroMin( sP.units.codeColDensToPhys( grid, totKpc2=True ) )
        config['label']  = 'Gas Synchrotron Emission, SKA [log W Hz$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'perula'

    if 'metals_' in partField:
        # all of GFM_Metals as well as GFM_MetalsTagged (projected as column densities)
        grid = logZeroMin( sP.units.codeColDensToPhys(grid, msunKpc2=True) )
        metalName = partField.split("_")[1]

        mStr = '-Metals' if metalName in ['SNIa','SNII','AGB','NSNS'] else ''
        config['label'] = '%s %s%s Column Density [log cm$^{-2}$]' % (ptStr,metalName,mStr)
        config['ctName'] = 'cubehelix'

        # testing:
        if '_minIP' in method: config['ctName'] = 'gray' # minIP: do dark on light
        if '_maxIP' in method: config['ctName'] = 'gray_r' # maxIP: do light on dark
        #config['plawScale'] = 1.0

    if 'sb_' in partField:
        # surface brightness map, based on fluxes, i.e. [erg/s/cm^2] -> [erg/s/cm^2/arcsec]
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]

        ster = True if '_ster' in partField else False
        arcsec2 = not ster

        grid = logZeroMin( sP.units.fluxToSurfaceBrightness(grid, pxSizesCode, arcsec2=arcsec2, ster=ster) )
        uLabel = 'arcsec$^{-2}$'
        if ster: uLabel = 'ster$^{-1}$'

        lineName = partField.replace("_ster","").split("sb_")[1].replace("-"," ")
        if lineName[-1] == 'A': lineName = lineName[:-1] + '$\AA$' # Angstrom
        config['label']  = '%s Surface Brightness [log photon s$^{-1}$ cm$^{-2}$ %s]' % (lineName,uLabel)
        config['ctName'] = 'inferno'

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
        config['label']  = 'Magnetic Field Magnitude [log G]'
        config['ctName'] = 'Spectral_r'

    if partField in ['bmag_uG']:
        config['label']  = 'Magnetic Field Magnitude [log $\mu$G]'
        config['ctName'] = 'Spectral_r'
        config['plawScale'] = 0.4

    if partField in ['bfield_x','bfield_y','bfield_z']:
        grid = sP.units.particleCodeBFieldToGauss(grid) * 1e6 # linear micro-Gauss
        dirStr = partField.split("_")[1].lower()
        config['label']  = 'B$_{\\rm %s}$ [$\mu$G]' % dirStr
        config['ctName'] = 'PuOr' # is brewer-purpleorange

    # gas: shock finder
    if partField in ['dedt','energydiss','shocks_dedt','shocks_energydiss']:
        grid = logZeroMin( sP.units.codeEnergyRateToErgPerSec(grid) )
        config['label']  = 'Shocks Dissipated Energy [log erg/s]'
        config['ctName'] = 'plasma'
        config['plawScale'] = 0.7

    if partField in ['machnum','shocks_machnum']:
        #grid = logZeroMin( grid )
        config['label']  = 'Shock Mach Number' # [log]'
        config['ctName'] = 'hot'
        config['plawScale'] = 0.7

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
        config['label']  = '%s Metallicity [log Z$_{\\rm sun}$]' % ptStr
        config['ctName'] = 'viridis'
        config['plawScale'] = 1.0

    if partField in ['SN_IaII_ratio_Fe']:
        grid = logZeroMin( grid )
        config['label']  = '%s Mass Ratio Fe$_{\\rm SNIa}$ / Fe$_{\\rm SNII}$ [log]' % ptStr
        config['ctName'] = 'Spectral'
    if partField in ['SN_IaII_ratio_metals']:
        grid = logZeroMin( grid )
        config['label']  = '%s Mass Ratio Z$_{\\rm SNIa}$ / Z$_{\\rm SNII}$ [log]' % ptStr
        config['ctName'] = 'Spectral'
        config['cmapCenVal'] = 0.0
    if partField in ['SN_Ia_AGB_ratio_metals']:
        grid = logZeroMin( grid )
        config['label']  = '%s Mass Ratio Z$_{\\rm SNIa}$ / Z$_{\\rm AGB}$ [log]' % ptStr
        config['ctName'] = 'Spectral'

    # velocities (mass-weighted)
    if partField in ['vmag','velmag']:
        config['label']  = '%s Velocity Magnitude [km/s]' % ptStr
        config['ctName'] = 'afmhot' # same as pm/f-34-35-36 (illustris)

    if partField in ['vel_los','vel_los_sfrwt']:
        config['label']  = '%s Line of Sight Velocity [km/s]' % ptStr
        config['ctName'] = 'RdBu_r' # bwr, coolwarm, RdBu_r

    if partField in ['vel_x','vel_y','velocity_x','velocity_y']:
        velDirection = partField.split("_")[1]
        config['label'] = '%s %s-Velocity [km/s]' % (ptStr,velDirection)
        config['ctName'] = 'RdBu_r'

    if partField in ['velsigma_los','velsigma_los_sfrwt']:
        grid = np.sqrt(grid) # variance -> sigma
        config['label']  = '%s Line of Sight Velocity Dispersion [km/s]' % ptStr
        config['ctName'] = 'PuBuGn_r' # hot, magma

    if partField in ['vrad','halo_vrad','radvel','halo_radvel']:
        config['label']  = '%s Radial Velocity [km/s]' % ptStr
        config['ctName'] = 'PRGn' # brewer purple-green diverging

    if partField == 'vrad_vvir':
        config['label']  = '%s Radial Velocity / Halo v$_{200}$' % ptStr
        config['ctName'] = 'PRGn' # brewer purple-green diverging

    if partField in ['specangmom_mag','specj_mag']:
        grid = logZeroMin( grid )
        config['label'] = '%s Specific Angular Momentum Magnitude [log kpc km/s]' % ptStr
        config['ctName'] = 'cubehelix'

    # stars
    if partField in ['star_age','stellar_age']:
        config['label']  = 'Stellar Age [Gyr]'
        config['ctName'] = 'blgrrd_black0'

    if 'stellarBand-' in partField:
        # convert linear luminosities back to magnitudes
        ww = np.where(grid == 0.0)
        w2 = np.where(grid > 0.0)
        grid[w2] = -2.5 * np.log10( grid[w2] )
        grid[ww] = 99.0

        bandName = partField.split("stellarBand-")[1]
        config['label'] = 'Stellar %s Luminosity [mag]' % bandName
        config['ctName'] = 'gray_r'

    if 'stellarComp-' in partField:
        print('todo')
        import pdb; pdb.set_trace()

    # all particle types
    if partField in ['potential']:
        config['label']  = '%s Gravitational Potential [slog km$^2$/s$^2$]' % ptStr
        config['ctName'] = 'RdGy_r'

        grid /= sP.scalefac # remove a factor
        w_neg = np.where(grid <= 0.0)
        w_pos = np.where(grid > 0.0)
        grid[w_pos] = logZeroMin( grid[w_pos] )
        grid[w_neg] = -logZeroMin( -grid[w_neg] )

    if partField in ['id']:
        config['label']  = '%s Particle ID [log]' % ptStr
        config['ctName'] = 'afmhot'
        grid = logZeroMin( grid )

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

def gridBox(sP, method, partType, partField, nPixels, axes, projType, projParams, 
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, 
            forceRecalculate=False, smoothFWHM=None, **kwargs):
    """ Caching gridding/imaging of a simulation box. """
    optionalStr = ''
    if projType != 'ortho':
        optionalStr += '_%s-%s' % (projType, '_'.join( [str(k)+'='+str(v) for k,v in projParams.iteritems()] ))

    m = hashlib.sha256('nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d.%g.rot-%s%s' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1], 
         hsmlFac, str(rotMatrix), optionalStr)).hexdigest()[::4]

    _, sbStr, _ = subboxVals(sP.subbox)

    # if loaded/gridded data is the same, just processed differently, don't save twice
    partFieldSave = partField.replace(' fracmass',' mass')
    partFieldSave = partFieldSave.replace(' ','_') # convention for filenames

    saveFilename = sP.derivPath + 'grids/%s/%s.%s%d.%s.%s.%s.hdf5' % \
                   (sbStr.replace("_","/"), method, sbStr, sP.snap, partType, partFieldSave, m)

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')
    if not isdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/")):
        mkdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/"))

    # no particles of type exist? blank grid return (otherwise die in getHsml and wind removal)
    h = snapshotHeader(sP)

    def emptyReturn():
        print('Warning: No particles, returning empty for [%s]!' % saveFilename.split(sP.derivPath)[1])
        grid = np.zeros( nPixels, dtype='float32' )
        grid, config = gridOutputProcess(sP, grid, partType, partField, boxSizeImg, nPixels, projType, method)
        return grid, config

    if h['NumPart'][sP.ptNum(partType)] <= 2:
        return emptyReturn()

    # generate a 3-band composite stellar image from 3 bands
    if 'stellarComp-' in partField:
        bands = partField.split("-")[1:]        
        return stellar3BandCompositeImage(bands, sP, method, nPixels, axes, projType, projParams, boxCenter, boxSizeImg, 
                                          hsmlFac, rotMatrix, rotCenter, forceRecalculate, smoothFWHM)

    # map
    if not forceRecalculate and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid_master = f['grid'][...]
        print('Loaded: [%s]' % saveFilename.split(sP.derivPath)[1])
    else:
        # will we use a complete load or a subset particle load?
        indRange = None
        boxSizeSim = sP.boxSize
        boxSizeImgMap = boxSizeImg
        boxCenterMap = boxCenter

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

        # quantity is computed with respect to a pre-existing grid? load now
        refGrid = None
        if partField in ['velsigma_los','velsigma_los_sfrwt']:
            partFieldRef = partField.replace('sigma','') # e.g. 'velsigma_los_sfrwt' -> 'vel_los_sfrwt'
            projParamsLoc = dict(projParams)
            projParamsLoc['noclip'] = True
            refGrid, _ = gridBox(sP, method, partType, partFieldRef, nPixels, axes, projType, projParamsLoc, 
                                 boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, smoothFWHM=smoothFWHM)

        # if indRange is still None (full snapshot load), we will proceed chunked, unless we need
        # a full tree construction to calculate hsml values
        grid_dens  = np.zeros( nPixels[::-1], dtype='float32' )
        grid_quant = np.zeros( nPixels[::-1], dtype='float32' )
        nChunks = 1

        # if doing a minimum intensity projection, pre-fill grid_quant with infinity as we 
        # accumulate per chunk by using a minimum reduction between the master grid and each chunk grid
        if '_minIP' in method: grid_quant.fill(np.inf)

        disableChunkLoad = (sP.isPartType(partType,'dm') and not snapHasField(sP, partType, 'SubfindHsml')) or \
                           sP.isPartType(partType,'stars') # use custom CalcHsml always for stars now

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

                pos, _ = rotateCoordinateArray(sP, pos, rotMatrix, rotCenter)

            # load: sizes (hsml) and manipulate as needed
            if 'stellarBand-' in partField or (partType == 'stars' and 'coldens' in partField):
                hsml = getHsmlForPartType(sP, partType, indRange=indRange, nNGB=32, snapHsmlForStars=False)
            else:
                hsml = getHsmlForPartType(sP, partType, indRange=indRange)

            hsml *= hsmlFac # modulate hsml values by hsmlFac

            if sP.isPartType(partType, 'stars'):
                pxScale = np.max(np.array(boxSizeImg)[axes] / nPixels)
                hsml = clipStellarHSMLs(hsml, sP, pxScale, nPixels, method=None)

                if 1:
                    #print(' custom AGE HSMLFAC MOD!') # (now default behavior)
                    age_min = 1.0
                    age_max = 3.0
                    max_mod = 2.0
                    # load stellar ages
                    ages = snapshotSubset(sP, 'stars', 'stellar_age', indRange=indRange)
                    # ramp such that hsml*=1.0 at <=1Gyr, linear to hsml*=2.0 at >=4 Gyr
                    rampFac = np.ones( ages.size, dtype='float32' )
                    ages = np.clip( (ages-age_min)/(age_max-age_min), 0.0, 1.0 ) * max_mod
                    rampFac += ages
                    hsml *= rampFac

            # load: mass/weights, quantity, and render specifications required
            mass, quant, normCol = loadMassAndQuantity(sP, partType, partField, indRange=indRange)

            if 0:
                print('WARNING: Selectively setting mass=0 based on some particle criterion!')
                dens = snapshotSubset(sP, partType, 'numdens', indRange=indRange)
                temp = snapshotSubset(sP, partType, 'temp', indRange=indRange)
                w = np.where( (temp>5.5) & (dens>1e-3) )
                mask = np.zeros( dens.size, dtype='int16' )
                mask[w] = 1
                w = np.where(mask == 0)
                mass[w] = 0.0

            # non-orthographic projection? project now, converting pos from a 3-vector into a 2-vector
            hsml_1 = None

            if projType == 'equirectangular':
                assert axes == [0,1] # by convention
                assert projParams['fov'] == 360.0
                assert nPixels[0] == nPixels[1]*2 # we expect to make a 2:1 aspect ratio image

                hsml_orig = hsml.copy()

                # shift pos to boxCenter
                for i in range(3):
                    pos[:,i] -= boxCenter[i]
                correctPeriodicDistVecs(pos, sP)

                # cartesian to spherical coordinates
                s_rad = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
                s_lat = np.arctan2(pos[:,2], np.sqrt(pos[:,0]**2 + pos[:,1]**2)) # latitude (phi) in [-pi/2,pi/2], defined from XY plane up
                s_long = np.arctan2(pos[:,1], pos[:,0]) # longitude (lambda) in [-pi,pi]

                # restrict to sphere, instead of cube, to avoid differential ray lengths
                w = np.where(s_rad > sP.boxSize/2)
                mass[w] = 0.0

                # hsml: convert from kpc to deg (compute angular diameter)
                w = np.where(hsml_orig > 2*s_rad)
                hsml_orig[w] = 1.999*s_rad[w] # otherwise outside arcsin

                hsml = 2 * np.arcsin(hsml_orig / (2*s_rad))

                # handle differential distortion along x/y directions
                hsml_1 = hsml.astype('float32') # hsml_1 (hsml_y) unmodified
                hsml = hsml / np.cos(s_lat) # hsml_0 (i.e. hsml_x) only
                hsml = hsml.astype('float32')

                # we will project in this space, periodic on the boundaries
                pos = np.zeros( (s_rad.size,2), dtype=pos.dtype )
                pos[:,0] = s_long + np.pi # [0,2pi]
                pos[:,1] = s_lat + np.pi/2 # [0,pi]

                boxSizeImgMap = [2*np.pi, np.pi]
                boxCenterMap  = [np.pi, np.pi/2]
                boxSizeSim    = [2*np.pi, np.pi, 0.0] # periodic on projected coordinate system extent

            # rotation? handle for view dependent quantities (e.g. velLOS) (any 3-vector really...)
            if partField in velLOSFieldNames + velCompFieldNames:
                # first compensate for subhalo CM motion (if this is a halo plot)
                if sP.isZoom or sP.hInd is not None:
                    sh = groupCatSingle(sP, subhaloID=sP.zoomSubhaloID if sP.isZoom else sP.hInd)
                    for i in range(3):
                        # SubhaloVel already peculiar, quant converted already in loadMassAndQuantity()
                        quant[:,i] -= sh['SubhaloVel'][i] 

                if partField in velLOSFieldNames:
                    # slice corresponding to (optionally rotated) LOS component
                    sliceIndNoRot = 3-axes[0]-axes[1]
                    sliceIndRot = 2

                if partField in velCompFieldNames:
                    # slice corresponding to (optionally rotated) _x or _y velocity component
                    if '_x' in partField: sliceIndRot = 0
                    if '_y' in partField: sliceIndRot = 1
                    sliceIndNoRot = sliceIndRot

                # do slice (convert 3-vector into scalar)
                if rotMatrix is None:
                    quant = quant[:,sliceIndNoRot]
                else:
                    quant = np.transpose( np.dot(rotMatrix, quant.transpose()) )
                    quant = np.squeeze( np.array(quant[:,sliceIndRot]) )
                    quant = quant.astype('float32') # rotMatrix was posssibly in double

            assert quant is None or quant.ndim == 1 # must be scalar

            # stars requested in run with winds? if so, load SFTime to remove contaminating wind particles
            wMask = None

            if partType == 'stars' and sP.winds:
                sftime = snapshotSubset(sP, partType, 'sftime', indRange=indRange)
                wMask = np.where(sftime > 0.0)[0]
                if len(wMask) <= 2 and nChunks == 1:
                    return emptyReturn()

                mass = mass[wMask]
                hsml = hsml[wMask]
                pos  = pos[wMask,:]
                if quant is not None:
                    quant = quant[wMask]

            # render
            if method in ['sphMap','sphMap_global','sphMap_minIP','sphMap_maxIP','sphMap_LIC']:
                # particle by particle (unordered) splat using standard SPH cubic spline kernel

                # further sub-method specification?
                maxIntProj = True if '_maxIP' in method else False
                minIntProj = True if '_minIP' in method else False

                # render
                grid_d, grid_q = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                                         boxSizeSim=boxSizeSim, boxSizeImg=boxSizeImgMap, boxCen=boxCenterMap, 
                                         nPixels=nPixels, hsml_1=hsml_1, colDens=normCol, multi=True, 
                                         maxIntProj=maxIntProj, minIntProj=minIntProj, refGrid=refGrid )

            elif method in ['histo']:
                # simple 2D histogram, particles assigned to the bin which contains them
                from scipy.stats import binned_statistic_2d
                assert hsml_1 is None # not supported

                stat = 'sum'
                if '_minIP' in method: stat = 'min'
                if '_maxIP' in method: stat = 'max'

                xMinMax = [-boxSizeImgMap[0]/2, +boxSizeImgMap[0]/2]
                yMinMax = [-boxSizeImgMap[1]/2, +boxSizeImgMap[1]/2]

                # make pos periodic relative to boxCenterMap, and slice in axes[2] dimension
                for i in range(3):
                    pos[:,i] -= boxCenterMap[i]
                correctPeriodicDistVecs(pos, sP)

                zvals = np.squeeze( pos[:,3-axes[0]-axes[1]] )
                w = np.where( np.abs(zvals) <= boxSizeImgMap[2] * 0.5 )

                xvals = np.squeeze( pos[w,axes[0]] )
                yvals = np.squeeze( pos[w,axes[1]] )
                mass  = mass[w]

                # compute mass sum grid
                grid_d, _, _, _ = binned_statistic_2d(xvals, yvals, mass, stat, bins=nPixels, range=[xMinMax,yMinMax])
                grid_d = grid_d.T

                if normCol:
                    pixelArea = (boxSizeImg[0] / nPixels[0]) * (boxSizeImg[1] / nPixels[1])
                    grid_d /= pixelArea

                # mass-weighted quantity? compute mass*quant sum grid
                grid_q = np.zeros( grid_d.shape, dtype=grid_d.dtype )

                if quant is not None:
                    quant = quant[w]

                if quant is not None and partField != 'velsigma_los':
                    grid_q, _, _, _ = binned_statistic_2d(xvals, yvals, mass*quant, stat, bins=nPixels, range=[xMinMax,yMinMax])
                    grid_q = grid_q.T

                # special behavior
                #def _weighted_std(values, weights):
                #    """ Return weighted standard deviation. Would enable e.g. velsigma_los_sfrwt, except that user 
                #        functions in binned_statistic_2d() cannot accept any arguments beyond the list of values in a bin. 
                #        So we cannot do a weighted f(), without rewriting the internals therein. """
                #    avg = np.average(values, weights=weights)
                #    delta = values - avg
                #    var = np.average(delta*delta, weights=weights)
                #    return np.sqrt(var)
                if partField == 'velsigma_los':
                    assert nChunks == 1 # otherwise not supported
                    # refGrid loaded but not used, we let np.std() re-compute the per pixel mean
                    grid_q, binx, biny, inds = binned_statistic_2d(xvals, yvals, quant, np.std, bins=nPixels, range=[xMinMax,yMinMax])
                    grid_q = grid_q.T
                    grid_q *= grid_d # pre-emptively undo normalization below == unweighted stddev(los_vel)
                else:
                    assert refGrid is None # not supported except for this one field

            else:
                # todo: e.g. external calls to ArepoVTK for voronoi_* based visualization
                raise Exception('Not implemented.')

            # accumulate for chunked processing
            if '_minIP' in method:
                w = np.where(grid_q < grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            elif '_maxIP' in method:
                w = np.where(grid_q > grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            else:
                grid_dens  += grid_d
                grid_quant += grid_q

        # normalize quantity
        grid_master = grid_dens

        if quant is not None:
            # multi=True, so global normalization by per pixel 'mass' now
            w = np.where(grid_dens > 0.0)
            grid_quant[w] /= grid_dens[w]
            grid_master = grid_quant

        # save
        with h5py.File(saveFilename,'w') as f:
            f['grid'] = grid_master
        print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    # handle units and come up with units label
    grid_master, config = gridOutputProcess(sP, grid_master, partType, partField, boxSizeImg, nPixels, projType, method)

    # temporary: something a bit peculiar here, request an entirely different grid and 
    # clip the line of sight to zero (or nan) where log(n_HI)<19.0 cm^(-2)
    if 0 and partField in velLOSFieldNames:
        print('Clipping LOS velocity, visible at log(n_HI) > 19.0 only.')
        grid_nHI, _ = gridBox(sP, method, 'gas', 'HI_segmented', nPixels, axes, projType, projParams, 
                              boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, smoothFWHM=smoothFWHM)

        grid_master[grid_nHI < 19.0] = np.nan

    if 1 and partField in velLOSFieldNames:
        if 'noclip' not in projParams:
            print('Clipping LOS velocity, visible at SFR surface density > 0.01 msun/yr/kpc^2 only.')
            grid_sfrsd, _ = gridBox(sP, method, 'gas', 'sfr_msunyrkpc2', nPixels, axes, projType, projParams, 
                                  boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, smoothFWHM=smoothFWHM)

            grid_master[grid_sfrsd < -3.0] = np.nan

    # temporary: similar, truncate stellar_age projection at a stellar column density of 
    # ~log(3.2) [msun/kpc^2] equal to the bottom of the color scale for the illustris/tng sb0 box renders
    if partField == 'stellar_age':
        grid_stellarColDens, _ = gridBox(sP, method, 'stars', 'coldens_msunkpc2', nPixels, axes, projType, projParams, 
                                         boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

        w = np.where(grid_stellarColDens < 3.0)
        grid_master[w] = 0.0 # black

    # temporary: similar, fractional total mass sum of a sub-component relative to the full, request 
    # the 'full' mass grid of this particle type now and normalize
    if ' fracmass' in partField:
        grid_totmass, _ = gridBox(sP, method, partType, 'mass', nPixels, axes, projType, projParams, 
                                  boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, smoothFWHM=smoothFWHM)
        grid_master = logZeroMin( 10.0**grid_master / 10.0**grid_totmass )

    # temporary: line integral convolution test
    if 'licMethod' in kwargs and kwargs['licMethod'] is not None:
        # temp config
        vecSliceWidth = kwargs['licSliceDepth']
        pixelFrac = kwargs['licPixelFrac']
        field_pt = kwargs['licPartType']
        field_name = kwargs['licPartField']

        # compress vector grids along third direction to more thin slice
        boxSizeImgLoc = np.array(boxSizeImg)
        boxSizeImgLoc[3-axes[0]-axes[1]] = sP.units.physicalKpcToCodeLength(vecSliceWidth)

        # load two grids of vector length in plot-x and plot-y directions
        vel_field = np.zeros( (nPixels[0], nPixels[1], 2), dtype='float32' )
        field_x = field_name + '_' + ['x','y','z'][axes[0]]
        field_y = field_name + '_' + ['x','y','z'][axes[1]]

        vel_field[:,:,0], _ = gridBox(sP, method.split("_LIC")[0], field_pt, field_x, nPixels, axes, projType, projParams, 
                                      boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter)

        vel_field[:,:,1], _ = gridBox(sP, method.split("_LIC")[0], field_pt, field_y, nPixels, axes, projType, projParams, 
                                      boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter)

        # smoothing kernel
        from scipy.stats import norm
        gauss_kernel = norm.pdf(np.linspace(-3, 3, 25*2))
        # TODO: this 50 should likely scale with nPixels to maintain same image (check)
        # TODO: Perlin noise

        # create noise field and do LIC
        np.random.seed(424242)

        if kwargs['licMethod'] == 1:
            # first is half pixels black, second is 99% pixels black (make into parameter)
            noise_50 = (np.random.random(nPixels) < pixelFrac)

            grid_master = line_integral_convolution(noise_50, vel_field, gauss_kernel)

        if kwargs['licMethod'] == 2:
            # noise field biased by the data field, or the data field itself somehow...
            noise_template = (np.random.random(nPixels) < pixelFrac)
            noise_50 = noise_template #* grid_master

            lic_output = line_integral_convolution(noise_50, vel_field, gauss_kernel)
            lic_output = np.clip(lic_output, 1e-8, 1.0)

            # multiple the LIC field [0,1] by the logged, or the linear, actual data field
            print(lic_output.min(), lic_output.max())
            #grid_master *= lic_output
            grid_master = logZeroMin(10.0**grid_master * lic_output)

    # smooth down to some resolution by convolving with a Gaussian?
    if smoothFWHM is not None:
        # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
        sigma_xy = (smoothFWHM / 2.3548) / (boxSizeImg[axes] / nPixels) 
        print('fwhm: ',smoothFWHM,sigma_xy)
        grid_master = gaussian_filter(grid_master, sigma_xy, mode='reflect', truncate=5.0)

    return grid_master, config

def addBoxMarkers(p, conf, ax):
    """ Factor out common annotation/markers to overlay. """
    fontsize = int(conf.rasterPx[0] / 100.0 * conf.nCols * 1.0)

    def _addCirclesHelper(p, ax, pos, radii, numToAdd):
        """ Helper function to add a number of circle markers for halos/subhalos, within the panel. """
        countAdded = 0
        gcInd = 0

        while countAdded < numToAdd:
            xyzPos = pos[gcInd,:][ [p['axes'][0], p['axes'][1], 3-p['axes'][0]-p['axes'][1]] ]
            xyzDist = xyzPos - p['boxCenter']
            correctPeriodicDistVecs(xyzDist, p['sP'])
            xyzDistAbs = np.abs(xyzDist)

            # in bounds?
            if ( (xyzDistAbs[0] <= p['boxSizeImg'][0]/2) & \
                 (xyzDistAbs[1] <= p['boxSizeImg'][1]/2) & \
                 (xyzDistAbs[2] <= p['boxSizeImg'][2]/2) ):
                # draw and count
                countAdded += 1

                xPos = pos[gcInd,p['axes'][0]]
                yPos = pos[gcInd,p['axes'][1]]
                rad  = radii[gcInd] * 1.0

                # our plot coordinate system is true simulation coordinates, except without 
                # any periodicity, e.g. relative to boxCenter but restored (negatives or >boxSize ok)
                if xPos > p['extent'][1]: xPos -= p['boxSizeImg'][0]
                if yPos > p['extent'][3]: yPos -= p['boxSizeImg'][1]
                if xPos < p['extent'][0]: xPos += p['boxSizeImg'][0]
                if yPos < p['extent'][2]: yPos += p['boxSizeImg'][1]

                if 'relCoords' in p and p['relCoords']:
                    xPos = xyzDist[0]
                    yPos = xyzDist[1]

                if p['axesUnits'] == 'kpc':
                    xPos = p['sP'].units.codeLengthToKpc(xPos)
                    yPos = p['sP'].units.codeLengthToKpc(yPos)
                    rad  = p['sP'].units.codeLengthToKpc(rad)
                if p['axesUnits'] == 'mpc':
                    xPos = p['sP'].units.codeLengthToMpc(xPos)
                    yPos = p['sP'].units.codeLengthToMpc(yPos)
                    rad  = p['sP'].units.codeLengthToMpc(rad)
                assert p['axesUnits'] not in ['deg','arcmin'] # todo

                c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                ax.add_artist(c)

            gcInd += 1
            if gcInd >= radii.size:
                print('Warning: Ran out of halos to add, only [%d of %d]' % (countAdded,numToAdd))
                break

    if 'plotHalos' in p and p['plotHalos'] > 0:
        # plotting N most massive halos in visible area
        h = groupCatHeader(p['sP'])

        if h['Ngroups_Total'] > 0:
            gc = groupCat(p['sP'], fieldsHalos=['GroupPos','Group_R_Crit200'])['halos']

            _addCirclesHelper(p, ax, gc['GroupPos'], gc['Group_R_Crit200'], p['plotHalos'])

    if 'plotSubhalos' in p and p['plotSubhalos'] > 0:
        # plotting N most massive child subhalos in visible area
        h = groupCatHeader(p['sP'])

        if h['Ngroups_Total'] > 0:
            haloInd = groupCatSingle(p['sP'], subhaloID=p['hInd'])['SubhaloGrNr']
            halo = groupCatSingle(p['sP'], haloID=haloInd)

            if halo['GroupFirstSub'] != p['hInd']:
                print('Warning: Rendering subhalo circles around a non-central subhalo!')
        
            subInds = np.arange( halo['GroupFirstSub']+1, halo['GroupFirstSub']+halo['GroupNsubs'] )

            gc = groupCat(p['sP'], fieldsSubhalos=['SubhaloPos','SubhaloHalfmassRad'])['subhalos']
            gc['SubhaloPos'] = gc['SubhaloPos'][subInds,:]
            gc['SubhaloHalfmassRad'] = gc['SubhaloHalfmassRad'][subInds]

            _addCirclesHelper(p, ax, gc['SubhaloPos'], gc['SubhaloHalfmassRad'], p['plotSubhalos'])

    if 'rVirFracs' in p and p['rVirFracs']:
        # plot circles for N fractions of the virial radius
        xPos = p['boxCenter'][0]
        yPos = p['boxCenter'][1]

        if p['relCoords']:
            xPos = 0.0
            yPos = 0.0

        if p['axesUnits'] == 'kpc':
            xPos = p['sP'].units.codeLengthToKpc(xPos)
            yPos = p['sP'].units.codeLengthToKpc(yPos)
        if p['axesUnits'] == 'mpc':
            xPos = p['sP'].units.codeLengthToMpc(xPos)
            yPos = p['sP'].units.codeLengthToMpc(yPos)

        for rVirFrac in p['rVirFracs']:
            rad = rVirFrac

            if p['fracsType'] == 'rVirial':
                rad *= p['haloVirRad']
            if p['fracsType'] == 'rHalfMass':
                rad *= p['galHalfMass']
            if p['fracsType'] == 'rHalfMassStars':
                rad *= p['galHalfMassStars']
                if rad == 0.0:
                    print('Warning: Drawing frac [%.1f %s] is zero, use halfmass.' % (rVirFrac,p['fracsType']))
                    rad = rVirFrac * p['galHalfMass']
            if p['fracsType'] == 'codeUnits':
                rad *= 1.0
            if p['fracsType'] == 'pkpc':
                rad = p['sP'].units.physicalKpcToCodeLength(rad)

            if p['axesUnits'] == 'kpc': rad = p['sP'].units.codeLengthToKpc(rad)
            if p['axesUnits'] == 'mpc': rad = p['sP'].units.codeLengthToMpc(rad)

            c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
            ax.add_artist(c)

    if 'labelZ' in p and p['labelZ']:
        if p['sP'].redshift >= 0.99:
            zStr = "z$\,$=$\,$%.1f" % p['sP'].redshift
        else:
            zStr = "z$\,$=$\,$%.2f" % p['sP'].redshift

        xt = p['extent'][1] - (p['extent'][1]-p['extent'][0])*(0.02) # upper right
        yt = p['extent'][3] - (p['extent'][3]-p['extent'][2])*(0.02)
        ax.text( xt, yt, zStr, color='white', alpha=1.0, 
                 size=fontsize, ha='right', va='top') # same size as legend text

    if 'labelScale' in p and p['labelScale']:
        scaleBarLen = (p['extent'][1]-p['extent'][0])*0.10 # 10% of plot width
        if conf.rasterPx[0] >= 1000: scaleBarLen *= 2 # 20% of plot width if going to be visually small
        scaleBarLen /= p['sP'].HubbleParam # ckpc/h -> ckpc (or cMpc/h -> cMpc)
        scaleBarLen = 100.0 * np.ceil(scaleBarLen/100.0) # round to nearest 100 code units (kpc)

        # if scale bar is more than 40% of width, reduce by 10x
        if scaleBarLen >= 0.4 * (p['extent'][1]-p['extent'][0]):
            scaleBarLen /= 10.0

        # if scale bar is more than 1 Mpc (or 10Mpc), round to nearest 1 Mpc (or 10 Mpc)
        for roundScale in [10000.0, 1000.0]:
            if scaleBarLen >= roundScale:
                scaleBarLen = roundScale * np.round(scaleBarLen/roundScale)

        # actually plot size in code units (e.g. ckpc/h)
        scaleBarPlotLen = scaleBarLen * p['sP'].HubbleParam

        if p['labelScale'] == 'physical':
            # convert size from comoving to physical, which influences only the label below
            scaleBarLen = np.round(scaleBarLen * p['sP'].units.scalefac)

        # label
        cmStr = 'c' if (p['sP'].redshift > 0.0 and p['labelScale'] != 'physical') else ''
        unitStrs = [cmStr+'pc',cmStr+'kpc',cmStr+'Mpc',cmStr+'Gpc'] # comoving (drop 'c' if at z=0)
        unitInd = 1 if p['sP'].mpcUnits is False else 2

        scaleBarStr = "%d %s" % (scaleBarLen, unitStrs[unitInd])
        if scaleBarLen > 900: # use Mpc label
            scaleBarStr = "%g %s" % (scaleBarLen/1000.0, unitStrs[unitInd+1])
        if scaleBarLen < 1: # use pc label
            scaleBarStr = "%g %s" % (scaleBarLen*1000.0, unitStrs[unitInd-1])

        lw = 2.0 * (conf.nCols/2)
        y_off = 0.03

        if conf.rasterPx[0] >= 1000:
            lw = 3.0
            y_off = 0.04

        if conf.rasterPx[0] > 2000:
            lw = 4.0
            y_off = 0.05

        yt_frac = y_off * (2 + conf.rasterPx[0]/5e3)

        x0 = p['extent'][0] + (p['extent'][1]-p['extent'][0])*(y_off * 960.0/conf.rasterPx[0]) # upper left
        x1 = x0 + scaleBarPlotLen
        yy = p['extent'][3] - (p['extent'][3]-p['extent'][2])*(y_off * 960.0/conf.rasterPx[0])
        yt = p['extent'][3] - (p['extent'][3]-p['extent'][2])*(yt_frac * 960.0/conf.rasterPx[0])

        if p['axesUnits'] in ['deg','arcmin','arcsec']:
            deg = (p['axesUnits'] == 'deg')
            amin = (p['axesUnits'] == 'arcmin')
            asec = (p['axesUnits'] == 'arcsec')
            x0 = p['sP'].units.codeLengthToAngularSize(x0, deg=deg, arcmin=amin, arcsec=asec)
            x1 = p['sP'].units.codeLengthToAngularSize(x1, deg=deg, arcmin=amin, arcsec=asec)
            yy = p['sP'].units.codeLengthToAngularSize(yy, deg=deg, arcmin=amin, arcsec=asec)
            yt = p['sP'].units.codeLengthToAngularSize(yt, deg=deg, arcmin=amin, arcsec=asec)
        if p['axesUnits'] == 'kpc':
            x0 = p['sP'].units.codeLengthToKpc(x0)
            x1 = p['sP'].units.codeLengthToKpc(x1)
            yy = p['sP'].units.codeLengthToKpc(yy)
            yt = p['sP'].units.codeLengthToKpc(yt)
        if p['axesUnits'] == 'mpc':
            x0 = p['sP'].units.codeLengthToMpc(x0)
            x1 = p['sP'].units.codeLengthToMpc(x1)
            yy = p['sP'].units.codeLengthToMpc(yy)
            yt = p['sP'].units.codeLengthToMpc(yt)

        ax.plot( [x0,x1], [yy,yy], '-', color='white', lw=lw, alpha=1.0)
        ax.text( np.mean([x0,x1]), yt, scaleBarStr, color='white', alpha=1.0, size=fontsize, ha='center', va='center')

    # text in a combined legend?
    legend_labels = []

    if 'labelSim' in p and p['labelSim']:
        legend_labels.append( p['sP'].simName )

    if 'labelHalo' in p and p['labelHalo']:
        assert p['sP'].hInd is not None

        if not p['sP'].isZoom:
            # periodic box: write properties of the hInd we are centered on
            subhaloID = p['hInd']
        else:
            # zoom run: write properties of zoomTargetHalo
            subhaloID = p['sP'].zoomSubhaloID

        subhalo = groupCatSingle(p['sP'], subhaloID=subhaloID)
        halo = groupCatSingle(p['sP'], haloID=subhalo['SubhaloGrNr'])

        haloMass = p['sP'].units.codeMassToLogMsun(halo['Group_M_Crit200'])
        stellarMass = p['sP'].units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][p['sP'].ptNum('stars')])

        str1 = "log M$_{\\rm halo}$ = %.1f" % haloMass
        str2 = "log M$_{\\rm star}$ = %.1f" % stellarMass

        if p['labelHalo'] == 'Mstar':
            # just Mstar
            str2 = "log M$^\star$ = %.1f" % stellarMass
            legend_labels.append( str2 )
        else:
            # both Mhalo and Mstar
            legend_labels.append( str1 )
            if np.isfinite(stellarMass): legend_labels.append( str2 )

    if 'labelCustom' in p and p['labelCustom']:
        for label in p['labelCustom']:
            legend_labels.append( label )

    # draw legend
    if len(legend_labels):
        legend_lines = [plt.Line2D((0,0),(0,0), linestyle='') for _ in legend_labels]
        legend = ax.legend(legend_lines, legend_labels, fontsize=fontsize, loc='lower left', 
                           handlelength=0, handletextpad=0)

        for text in legend.get_texts(): text.set_color('white')

def addVectorFieldOverlay(p, conf, ax):
    """ Add quiver or streamline overlay on top to visualization vector field data. """
    if 'vecOverlay' not in p or not p['vecOverlay']:
        return

    field_pt = None

    if p['vecOverlay'] == 'bfield':
        assert p['rotMatrix'] is None # otherwise need to handle like los-vel/velComps
        field_pt = 'gas'
        field_name = 'bfield'

    if '_vel' in p['vecOverlay']:
        # we are handling rotation properly for the velocity field (e.g. 'gas_vel', 'stars_vel', 'dm_vel')
        field_pt, field_name = p['vecOverlay'].split("_")

    assert field_pt is not None

    field_x = field_name + '_' + ['x','y','z'][p['axes'][0]]
    field_y = field_name + '_' + ['x','y','z'][p['axes'][1]]
    nPixels = [40,40]
    qStride = 3 # total number of ticks per axis is nPixels[i]/qStride
    vecSliceWidth = 5.0 # pkpc
    smoothFWHM = None

    # compress vector grids along third direction to more thin slice
    boxSizeImg = np.array(p['boxSizeImg'])
    boxSizeImg[3-p['axes'][0]-p['axes'][1]] = p['sP'].units.physicalKpcToCodeLength(vecSliceWidth)

    # load two grids of vector length in plot-x and plot-y directions
    grid_x, _ = gridBox(p['sP'], p['method'], field_pt, field_x, nPixels, p['axes'], p['projType'], p['projParams'], 
                        p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'], 
                        smoothFWHM=smoothFWHM)

    grid_y, _ = gridBox(p['sP'], p['method'], field_pt, field_y, nPixels, p['axes'], p['projType'], p['projParams'], 
                        p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'],
                        smoothFWHM=smoothFWHM)

    # load a grid of any quantity to use to color map the strokes
    grid_c, conf_c = gridBox(p['sP'], p['method'], p['vecColorPT'], p['vecColorPF'], nPixels, p['axes'], p['projType'], p['projParams'], 
                             p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'],
                             smoothFWHM=smoothFWHM)

    # create a unit vector at the position of each pixel
    grid_mag = np.sqrt(grid_x**2.0 + grid_y**2.0)

    w = np.where(grid_mag == 0.0) # protect against zero magnitude
    grid_mag[w] = grid_mag.max() * 1e10 # set grid_x,y to zero in these cases

    grid_x /= grid_mag
    grid_y /= grid_mag

    # create arrow starting (tail) positions
    pxScale = p['boxSizeImg'][p['axes']] / p['nPixels']
    xx = np.linspace( p['extent'][0] + pxScale[0]/2, p['extent'][1] - pxScale[0]/2, nPixels[0] )
    yy = np.linspace( p['extent'][2] + pxScale[1]/2, p['extent'][3] - pxScale[1]/2, nPixels[1] )

    # prepare for streamline variable thickness
    maxSize = 4.0
    minSize = 0.5
    uniSize = 1.0

    grid_c2 = grid_c
    if p['vecOverlay'] == 'bfield':
        # do a unit conversion such that we could actually make a quantitative streamplot (in progress)
        grid_c2 = 10.0**grid_c * 1e12 # [log G] -> [linear pG]

    grid_s = (maxSize - minSize)/(grid_c2.max() - grid_c2.min()) * (grid_c2 - grid_c2.min()) + minSize

    # set normalization?
    norm = None
    if p['vecMinMax'] is not None:
        norm = mpl.colors.Normalize(vmin=p['vecMinMax'][0], vmax=p['vecMinMax'][1])

    # (A) plot white quivers
    if p['vecMethod'] == 'A':
        assert norm is None
        q = ax.quiver(xx[::qStride], yy[::qStride], grid_x[::qStride,::qStride], grid_y[::qStride,::qStride], 
                      color='white', angles='xy', pivot='mid')

    # (B) plot colored quivers
    if p['vecMethod'] == 'B':
        assert norm is None # don't yet know how to handle
        q = ax.quiver(xx[::qStride], yy[::qStride], grid_x[::qStride,::qStride], grid_y[::qStride,::qStride],
                      grid_c[::qStride,::qStride], angles='xy', pivot='mid')
        # legend for quiver length: (in progress)
        #ax.quiverkey(q, 1.1, 1.05, 10.0, 'label', labelpos='E', labelsep=0.1,  coordinates='figure')

    # (C) plot white streamlines, uniform thickness
    if p['vecMethod'] == 'C':
        ax.streamplot(xx, yy, grid_x, grid_y, density=[1.0,1.0], linewidth=None, color='white')

    # (D) plot white streamlines, thickness scaled by color quantity
    if p['vecMethod'] == 'D':
        ax.streamplot(xx, yy, grid_x, grid_y, density=[1.0,1.0], linewidth=grid_s, color='white')

    # (E) plot colored streamlines, uniform thickness
    if p['vecMethod'] == 'E':
        ax.streamplot(xx, yy, grid_x, grid_y, density=[1.0,1.0], 
                      linewidth=uniSize, color=grid_c, cmap='afmhot', norm=norm)

    # (F) plot colored streamlines, thickness also proportional to color quantity
    if p['vecMethod'] == 'F':
        ax.streamplot(xx, yy, grid_x, grid_y, density=[1.0,1.0], 
                      linewidth=grid_s, color=grid_c, cmap='afmhot', norm=norm)

def setAxisColors(ax, color2):
    """ Factor out common axis color commands. """
    ax.title.set_color(color2)
    ax.yaxis.label.set_color(color2)
    ax.xaxis.label.set_color(color2)

    for s in ['bottom','left','top','right']:
        ax.spines[s].set_color(color2)
    for a in ['x','y']:
        ax.tick_params(axis=a, colors=color2)

def setColorbarColors(cb, color2):
    """ Factor out common colorbar color commands. """
    cb.ax.yaxis.label.set_color(color2)
    cb.outline.set_edgecolor(color2)
    cb.ax.yaxis.set_tick_params(color=color2)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color2)

def addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                       rowHeight, colWidth, bottomNorm, leftNorm, cmap=None):
    """ Add colorbar(s) with custom positioning and labeling, either below or above panels. """
    if not conf.colorbars:
        return

    factor  = 0.80 # bar length, fraction of column width, 1.0=whole
    height  = 0.04 # colorbar height, fraction of entire figure
    hOffset = 0.4  # padding between image and top of bar (fraction of bar height)
    tOffset = 0.20 # padding between top of bar and top of text label (fraction of bar height)
    lOffset = 0.02 # padding between colorbar edges and end label (frac of bar width)

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

    if 'vecMinMax' in config:
        #norm = mpl.colors.Normalize(vmin=config['vecMinMax'][0], vmax=config['vecMinMax'][1])
        colorbar = mpl.colorbar.ColorbarBase(cax, cmap=config['ctName'], orientation='horizontal')
        valLimits = config['vecMinMax'] #colorbar.get_clim()
    else:
        colorbar = plt.colorbar(cax=cax, orientation='horizontal')
        valLimits = plt.gci().get_clim()

    colorbar.outline.set_edgecolor(color2)

    # label, centered and below/above
    textsize = int(conf.rasterPx[0] / 100.0 * conf.nCols * 1.0) # xx-large

    cax.text(0.5, textTopY, config['label'], color=color2, transform=cax.transAxes, 
             size=textsize, ha='center', va='top' if barAreaTop == 0.0 else 'bottom')

    # tick labels, 5 evenly spaced inside bar
    colorsA = [(1,1,1),(0.9,0.9,0.9),(0.8,0.8,0.8),(0.2,0.2,0.2),(0,0,0)]
    colorsB = ['white','white','white','black','black']

    formatStr = "%.1f" if np.max(np.abs(valLimits)) < 100.0 else "%d"

    cax.text(0.0+lOffset, textMidY, formatStr % (1.0*valLimits[0]+0.0*valLimits[1]), 
        color=colorsB[0], size=textsize, ha='left', va='center', transform=cax.transAxes)
    cax.text(0.25, textMidY, formatStr % (0.75*valLimits[0]+0.25*valLimits[1]), 
        color=colorsB[1], size=textsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(0.5, textMidY, formatStr % (0.5*valLimits[0]+0.5*valLimits[1]), 
        color=colorsB[2], size=textsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(0.75, textMidY, formatStr % (0.25*valLimits[0]+0.75*valLimits[1]), 
        color=colorsB[3], size=textsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(1.0-lOffset, textMidY, formatStr % (0.0*valLimits[0]+1.0*valLimits[1]), 
        color=colorsB[4], size=textsize, ha='right', va='center', transform=cax.transAxes)

def renderMultiPanel(panels, conf):
    """ Generalized plotting function which produces a single multi-panel plot with one panel for 
        each of panels, all of which can vary in their configuration. 
    Global plot configuration options:
      plotStyle    : open (show axes), edged (no axes), open_black, edged_black
      rasterPx     : 2-tuple, each panel will have this number of pixels if making a raster (png) output, 
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
    sizeFac = np.array(conf.rasterPx) / mpl.rcParams['savefig.dpi']
    nRows   = np.floor(np.sqrt(len(panels))) if not hasattr(conf,'nRows') else conf.nRows 
    nCols   = np.ceil(len(panels) / nRows) if not hasattr(conf,'nCols') else conf.nCols
    aspect  = nRows/nCols

    conf.nCols = nCols

    if conf.plotStyle in ['open','open_black']:
        # start plot
        fig = plt.figure(facecolor=color1)

        widthFacCBs = 1.167 if conf.colorbars else 1.0
        fig.set_size_inches(widthFacCBs*sizeFac[0]*nRows/aspect, sizeFac[1]*nRows)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p['boxSizeImg'] is None: continue # blank panel
            # grid projection for image
            grid, config = gridBox(**p)

            # create this panel, and label axes and title
            ax = fig.add_subplot(nRows,nCols,i+1)

            sP = p['sP']
            idStr = ' (id=' + str(sP.hInd) + ')' if not sP.isZoom and sP.hInd is not None else ''
            ax.set_title('%s z=%d%s' % (sP.simName,sP.redshift,idStr))
            if sP.redshift != int(sP.redshift): ax.set_title('%s z=%3.1f%s' % (sP.simName,sP.redshift,idStr))
            if sP.redshift/0.1 != int(sP.redshift/0.1): ax.set_title('%s z=%4.2f%s' % (sP.simName,sP.redshift,idStr))

            axStrs = {'code':'[ ckpc/h ]', 'kpc':'[ pkpc ]', 'mpc':'[ Mpc ]', 'arcmin':'[ arcminutes ]', 'deg':'[ degrees ]', 
                      'rad_pi':' [ radians / $\pi$ ]'}
            axStr = axStrs[ p['axesUnits'] ]
            ax.set_xlabel( ['x','y','z'][p['axes'][0]] + ' ' + axStr)
            ax.set_ylabel( ['x','y','z'][p['axes'][1]] + ' ' + axStr)
            if p['axesUnits'] in ['arcsec','arcmin','deg']:
                ax.set_xlabel( '$\\alpha$ ' + axStr) # e.g. right ascension
                ax.set_ylabel( '$\delta$ ' + axStr) # e.g. declination
            if p['axesUnits'] in ['rad_pi']:
                ax.set_xlabel( '$\\theta$ ' + axStr) # e.g. longitude
                ax.set_ylabel( '$\phi$ ' + axStr) # e.g. latitude

            setAxisColors(ax, color2)

            # rotation? indicate transformation with axis labels
            if p['rotMatrix'] is not None:
                old_1 = np.zeros( 3, dtype='float32' )
                old_2 = np.zeros( 3, dtype='float32' )
                old_1[p['axes'][0]] = 1.0
                old_2[p['axes'][1]] = 1.0

                new_1 = np.transpose( np.dot(p['rotMatrix'], old_1) )
                new_2 = np.transpose( np.dot(p['rotMatrix'], old_2) )

                ax.set_xlabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_1[0], new_1[1], new_1[2], axStr))
                ax.set_ylabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_2[0], new_2[1], new_2[2], axStr))

            # color mapping (handle defaults and overrides)
            vMM = p['valMinMax'] if 'valMinMax' in p else None
            plaw = p['plawScale'] if 'plawScale' in p else None
            if 'plawScale' in config: plaw = config['plawScale']
            if 'plawScale' in p: plaw = p['plawScale']
            cenVal = p['cmapCenVal'] if 'cmapCenVal' in p else None
            if 'cmapCenVal' in config: cenVal = config['cmapCenVal']
            ctName = p['ctName'] if 'ctName' in p else config['ctName']

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal)
           
            cmap.set_bad(color='#000000',alpha=1.0) # use black for nan pixels
            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            if p['axesUnits'] == 'code':
                pExtent = p['extent'] 
            if p['axesUnits'] == 'kpc':
                pExtent = p['sP'].units.codeLengthToKpc(p['extent'])
            if p['axesUnits'] == 'mpc':
                pExtent = p['sP'].units.codeLengthToMpc(p['extent'])
            if p['axesUnits'] == 'arcmin':
                pExtent = p['sP'].units.codeLengthToAngularSize(p['extent'], arcmin=True)
            if p['axesUnits'] == 'deg':
                if p['sP'].redshift == 0.0: p['sP'].redshift = 0.1
                pExtent = p['sP'].units.codeLengthToAngularSize(p['extent'], deg=True)
                # shift to arbitrary center at (0,0)
                if pExtent[0] == 0.0 and pExtent[2] == 0.0:
                    assert pExtent[1] == pExtent[3] # box, not halo, imaging
                    pExtent -= pExtent[1]/2
            if p['projType'] == 'equirectangular':
                assert p['axesUnits'] == 'rad_pi'
                pExtent = [0, 2, 0, 1] # in units if pi

            plt.imshow(grid, extent=pExtent, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])

            ax.autoscale(False)
            if 'valMinMax' in p and cmap is not None:
                plt.clim( p['valMinMax'] )

            addBoxMarkers(p, conf, ax)

            addVectorFieldOverlay(p, conf, ax)

            # colobar
            if conf.colorbars:
                cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
                setAxisColors(cax, color2)

                cb = plt.colorbar(cax=cax)
                cb.outline.set_edgecolor(color2)
                cb.ax.set_ylabel(config['label'])

            # towards font-size invariance with changing rasterPx
            fontsize = int(conf.rasterPx[0] / 100.0 * 1.2)
            padding = conf.rasterPx[0] / 240.0
            ax.tick_params(axis='x', which='major', labelsize=fontsize)
            ax.tick_params(axis='y', which='major', labelsize=fontsize)
            ax.xaxis.label.set_size(fontsize)
            ax.yaxis.label.set_size(fontsize)
            ax.title.set_fontsize(fontsize)
            ax.tick_params(axis='both', which='major', pad=padding)

            if conf.colorbars:
                cb.ax.tick_params(axis='y', which='major', labelsize=fontsize)
                cb.ax.yaxis.label.set_size(fontsize)
                cb.ax.tick_params(axis='both', which='major', pad=padding)

        fig.tight_layout()
        if nRows == 1 and nCols == 3: plt.subplots_adjust(top=0.97,bottom=0.06) # fix degenerate case

    if conf.plotStyle in ['edged','edged_black']:
        # colorbar plot area sizing
        barAreaHeight = np.max([0.035,0.12 / nRows]) if conf.colorbars else 0.0
        if nRows == 1 and nCols == 1 and conf.colorbars: barAreaHeight = 0.05
        
        # check uniqueness of panel (partType,partField,valMinMax)'s
        pPartTypes   = set()
        pPartFields  = set()
        pValMinMaxes = set()

        for p in panels:
            pPartTypes.add(p['partType'])
            pPartFields.add(p['partField'])
            if 'valMinMax' in p: pValMinMaxes.add(str(p['valMinMax']))

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

        if 'vecColorbar' in p and p['vecColorbar'] and not oneGlobalColorbar:
            raise Exception('Only support vecColorbar addition with oneGlobalColorbar type configuration.')

        # start plot
        fig = plt.figure(frameon=False, tight_layout=False, facecolor=color1)

        width_in  = sizeFac[0] * np.ceil(nCols)
        height_in = sizeFac[1] * np.ceil(nRows) * (1/(1.0-barAreaTop-barAreaBottom))

        fig.set_size_inches(width_in, height_in)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p['boxSizeImg'] is None: continue # blank panel
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

            ax = fig.add_axes(pos, facecolor=color1)
            ax.set_axis_off()
            setAxisColors(ax, color2)

            # color mapping (handle defaults and overrides)
            vMM = p['valMinMax'] if 'valMinMax' in p else None
            plaw = p['plawScale'] if 'plawScale' in p else None
            if 'plawScale' in config: plaw = config['plawScale']
            if 'plawScale' in p: plaw = p['plawScale']
            cenVal = p['cmapCenVal'] if 'cmapCenVal' in p else None
            if 'cmapCenVal' in config: cenVal = config['cmapCenVal']
            ctName = p['ctName'] if 'ctName' in p else config['ctName']

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal)

            cmap.set_bad(color='#000000',alpha=1.0) # use black for nan pixels
            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            plt.imshow(grid, extent=p['extent'], cmap=cmap, aspect=grid.shape[0]/grid.shape[1])
            ax.autoscale(False) # disable re-scaling of axes with any subsequent ax.plot()
            if 'valMinMax' in p and cmap is not None:
                plt.clim( p['valMinMax'] )

            addBoxMarkers(p, conf, ax)

            addVectorFieldOverlay(p, conf, ax)

            # colobar(s)
            if oneGlobalColorbar:
                continue

            heightFac = np.max([1.0/nRows, 0.35])

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

        # one global colorbar? centered at bottom
        if oneGlobalColorbar:
            heightFac = np.max([1.0/nRows, 0.35])
            if nRows == 1: heightFac *= 1.0 #0.5 # reduce
            if nRows == 2: heightFac *= 1.3 # increase

            if 'vecColorbar' not in p or not p['vecColorbar']:
                # normal
                addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                                   rowHeight, 0.4, bottomNorm, 0.3)
            else:
                # normal, offset to the left
                addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                                   rowHeight, 0.4, bottomNorm, 0.05)

                # colorbar for the vector field visualization, offset to the right
                _, vConfig = gridOutputProcess(p['sP'], np.zeros(2), p['vecColorPT'], p['vecColorPF'], [1,1], 'ortho', 1.0)
                vConfig['vecMinMax'] = p['vecMinMax']
                vConfig['ctName'] = p['vecColormap']

                addCustomColorbars(fig, ax, conf, vConfig, heightFac, barAreaBottom, barAreaTop, color2, 
                                   rowHeight, 0.4, bottomNorm, 0.55)

    fig.savefig(conf.saveFilename, facecolor=fig.get_facecolor())
    plt.close(fig)
