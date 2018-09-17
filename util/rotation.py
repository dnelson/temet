"""
util/rotation.py
  Find rotation matrices (moment of inertia tensors) to place galaxies edge-on/face-on, do coordinate rotations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

from cosmo.util import periodicDists, correctPeriodicDistVecs, correctPeriodicPosVecs

def meanAngMomVector(sP, subhaloID, shPos=None, shVel=None):
    """ Calculate the 3-vector (x,y,z) of the mean angular momentum of either the star-forming gas 
    or the inner stellar component, for rotation and projection into disk face/edge-on views. """
    sh = sP.groupCatSingle(subhaloID=subhaloID)

    # allow center pos/vel to be input (e.g. mpb smoothed values)
    if shPos is None: shPos = sh['SubhaloPos']
    if shVel is None: shVel = sh['SubhaloVel']

    fields = ['Coordinates','Masses','StarFormationRate','Velocities']
    gas = sP.snapshotSubset('gas', fields, subhaloID=subhaloID)

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
    stars = sP.snapshotSubset('stars', fields, subhaloID=subhaloID)

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

def momentOfInertiaTensor(sP, gas=None, stars=None, rHalf=None, shPos=None, subhaloID=None, useStars=True):
    """ Calculate the moment of inertia tensor (3x3 matrix) for a subhalo or halo, given a load 
    of its member gas and stars (at least within 2*rHalf==shHalfMassRadStars) and center position shPos. 
    If useStars == True, then switch to stars if not enough SFing gas present, otherwise never use stars. """
    if subhaloID is not None:
        assert all(v is None for v in [gas,stars,rHalf])
        # load required particle data for this subhalo
        subhalo = sP.groupCatSingle(subhaloID=subhaloID)
        rHalf = subhalo['SubhaloHalfmassRadType'][sP.ptNum('stars')]
        shPos = subhalo['SubhaloPos']

        gas = sP.snapshotSubset('gas', fields=['mass','pos','sfr'], subhaloID=subhaloID)
        stars = sP.snapshotSubset('stars', fields=['mass','pos','sftime'], subhaloID=subhaloID)
    else:
        assert all(v is not None for v in [gas,stars,rHalf,shPos])

    if not gas['count'] and not stars['count']:
        print('Warning! momentOfInteriaTensor() no stars or gas in subhalo...')
        return np.identity(3)

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
        if gas['count'] == 1:
            return np.identity(3)
            
        wGas = np.where( (rad_gas <= 2.0*rHalf) & (gas['StarFormationRate'] > 0.0) )[0]

        masses = gas['Masses'][wGas]
        xyz = gas['Coordinates'][wGas,:]

    # shift
    xyz = np.squeeze(xyz)

    if xyz.ndim == 1:
        xyz = np.reshape( xyz, (1,3) )

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
    r['edge-on'] = np.matrix( ((1,0,0),(0,0,1),(0,-1,0)) ) * r['face-on'] # disk along x-hat
    r['edge-on-smallest'] = np.matrix( ((0,1,0),(0,0,1),(1,0,0)) ) * r['face-on']
    r['edge-on-y'] = np.matrix( ((0,0,1),(1,0,0),(0,-1,0)) ) * r['face-on'] # disk along y-hat
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
    about the origin. Input angle in degrees. """
    angle_rad = np.radians(angle)
    
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    direction /= np.linalg.norm(direction,2)

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
