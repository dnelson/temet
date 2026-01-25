"""
Find rotation matrices (moment of inertia tensors) to place galaxies edge-on/face-on, do coordinate rotations.
"""

import numpy as np
from numba import jit, prange

from ..util.helper import cache


def meanAngMomVector(sP, subhaloID, shPos=None, shVel=None):
    """Calculate the mean angular momentum 3-vector, for rotation and projection into disk face/edge-on views.

    Of either the star-forming gas or the inner stellar component.
    """
    sh = sP.groupCatSingle(subhaloID=subhaloID)

    # allow center pos/vel to be input (e.g. mpb smoothed values)
    if shPos is None:
        shPos = sh["SubhaloPos"]
    if shVel is None:
        shVel = sh["SubhaloVel"]

    fields = ["Coordinates", "Masses", "StarFormationRate", "Velocities"]
    gas = sP.snapshotSubset("gas", fields, subhaloID=subhaloID)

    # star forming gas only within a radial restriction
    wGas = []

    if gas["count"]:
        rad = sP.periodicDists(shPos, gas["Coordinates"])
        wGas = np.where(
            (rad <= 1.0 * sh["SubhaloHalfmassRadType"][sP.ptNum("stars")]) & (gas["StarFormationRate"] > 0.0)
        )[0]

    # add (actual) stars within 1 times the stellar half mass radius
    wStars = []

    fields.remove("StarFormationRate")
    fields.append("GFM_StellarFormationTime")
    stars = sP.snapshotSubset("stars", fields, subhaloID=subhaloID)

    if stars["count"]:
        rad = sP.periodicDists(shPos, stars["Coordinates"])

        wStars = np.where(
            (rad <= 1.0 * sh["SubhaloHalfmassRadType"][sP.ptNum("stars")]) & (stars["GFM_StellarFormationTime"] >= 0.0)
        )[0]

    # return default vector in case of total failure
    if len(wGas) + len(wStars) == 0:
        print("meanAngMomVector(): No star-forming gas or stars in radius, returning [1,0,0].")
        return np.array([1.0, 0.0, 0.0], dtype="float32")

    # combine gas and stars with restrictions (no gas or no stars is ok)
    pos = np.vstack((gas["Coordinates"][wGas, :], stars["Coordinates"][wStars, :]))
    mass = np.hstack((gas["Masses"][wGas], stars["Masses"][wStars]))
    vel = np.vstack((gas["Velocities"][wGas, :], stars["Velocities"][wStars, :]))

    ang_mom = sP.units.particleAngMomVecInKpcKmS(pos, vel, mass, shPos, shVel)

    # calculate mean angular momentum unit 3-vector
    ang_mom_mean = np.mean(ang_mom, axis=0)
    ang_mom_mean /= np.linalg.norm(ang_mom_mean, 2)

    return ang_mom_mean


@cache
def momentOfInertiaTensor(
    sP, gas=None, stars=None, rHalf=None, shPos=None, subhaloID=None, useStars=True, onlyStars=False
):
    """Calculate the moment of inertia tensor (3x3 matrix) for a subhalo or halo.

    Given a load of its member gas and stars (at least within 2*rHalf==shHalfMassRadStars) and center position shPos.
    If useStars == True, then switch to stars if not enough SFing gas present, otherwise never use stars.
    If onlyStars == True, use stars alone to determine.
    """
    if subhaloID is not None:
        assert all(v is None for v in [gas, stars, rHalf])
        # load required particle data for this subhalo
        subhalo = sP.groupCatSingle(subhaloID=subhaloID)
        rHalf = subhalo["SubhaloHalfmassRadType"][sP.ptNum("stars")]
        shPos = subhalo["SubhaloPos"]

        gas = sP.snapshotSubset("gas", fields=["mass", "pos", "sfr"], subhaloID=subhaloID)

        stars = {"count": 0}
        if subhalo["SubhaloLenType"][sP.ptNum("stars")] == 0:
            useStars = False

        if useStars:
            stars = sP.snapshotSubset("stars", fields=["mass", "pos", "sftime"], subhaloID=subhaloID)
    else:
        assert all(v is not None for v in [gas, stars, rHalf, shPos])

    if "count" not in gas:
        gas["count"] = gas["Masses"].size
    if "count" not in stars:
        stars["count"] = stars["Masses"].size

    if not gas["count"] and not stars["count"]:
        print("Warning! momentOfInteriaTensor() no stars or gas in subhalo...")
        return np.identity(3)

    if gas["count"] and len(gas["Masses"]) > 1 and not onlyStars:
        rad_gas = sP.periodicDists(shPos, gas["Coordinates"])
        wGas = np.where((rad_gas <= 0.5 * rHalf) & (gas["StarFormationRate"] > 0.0))[0]

        if len(wGas) >= 50:
            useStars = False

    if not stars["count"]:
        assert not onlyStars, "No stars, but onlyStars=True."
        useStars = False

    if useStars:
        # restrict to real stars
        wValid = np.where(stars["GFM_StellarFormationTime"] > 0.0)

        if len(wValid[0]) <= 1:
            return np.identity(3)

        stars["Masses"] = stars["Masses"][wValid]
        stars["Coordinates"] = np.squeeze(stars["Coordinates"][wValid, :])

        # use all stars within 1*rHalf
        rad_stars = sP.periodicDists(shPos, stars["Coordinates"])
        wStars = np.where(rad_stars <= 1.0 * rHalf)

        if len(wStars[0]) <= 1:
            return np.identity(3)

        masses = stars["Masses"][wStars]
        xyz = stars["Coordinates"][wStars, :]
    else:
        # use all star-forming gas cells within 2*rHalf
        if gas["count"] == 1:
            return np.identity(3)

        wGas = np.where((rad_gas <= 2.0 * rHalf) & (gas["StarFormationRate"] > 0.0))[0]

        masses = gas["Masses"][wGas]
        xyz = gas["Coordinates"][wGas, :]

    # shift
    xyz = np.squeeze(xyz)

    if xyz.ndim == 1:
        xyz = np.reshape(xyz, (1, 3))

    for i in range(3):
        xyz[:, i] -= shPos[i]

    # if coordinates wrapped box boundary before shift:
    sP.correctPeriodicDistVecs(xyz)

    # construct moment of inertia
    I = np.zeros((3, 3), dtype="float32")

    I[0, 0] = np.sum(masses * (xyz[:, 1] * xyz[:, 1] + xyz[:, 2] * xyz[:, 2]))
    I[1, 1] = np.sum(masses * (xyz[:, 0] * xyz[:, 0] + xyz[:, 2] * xyz[:, 2]))
    I[2, 2] = np.sum(masses * (xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1]))
    I[0, 1] = -1 * np.sum(masses * (xyz[:, 0] * xyz[:, 1]))
    I[0, 2] = -1 * np.sum(masses * (xyz[:, 0] * xyz[:, 2]))
    I[1, 2] = -1 * np.sum(masses * (xyz[:, 1] * xyz[:, 2]))
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I


def rotationMatricesFromInertiaTensor(I):
    """Calculate 3x3 rotation matrix by a diagonalization of the moment of inertia tensor.

    Note the resultant rotation matrices are hard-coded for projection with axes=[0,1] e.g. along z.
    """
    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(I)

    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]

    # permute the eigenvectors into this order, which is the rotation matrix which orients the
    # principal axes to the cartesian x,y,z axes, such that if axes=[0,1] we have face-on
    new_matrix = np.matrix(
        (rotation_matrix[:, sort_inds[0]], rotation_matrix[:, sort_inds[1]], rotation_matrix[:, sort_inds[2]])
    )

    # make a random edge on view
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.pi / 2
    psi = 0

    A_00 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
    A_01 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    A_02 = np.sin(psi) * np.sin(theta)
    A_10 = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
    A_11 = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
    A_12 = np.cos(psi) * np.sin(theta)
    A_20 = np.sin(theta) * np.sin(phi)
    A_21 = -np.sin(theta) * np.cos(phi)
    A_22 = np.cos(theta)

    random_edgeon_matrix = np.matrix(((A_00, A_01, A_02), (A_10, A_11, A_12), (A_20, A_21, A_22)))

    # prepare return with a few other useful versions of this rotation matrix
    r = {}
    r["face-on"] = new_matrix
    r["edge-on"] = np.matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))) * r["face-on"]  # disk along x-hat
    r["edge-on-smallest"] = np.matrix(((0, 1, 0), (0, 0, 1), (1, 0, 0))) * r["face-on"]
    r["edge-on-y"] = np.matrix(((0, 0, 1), (1, 0, 0), (0, -1, 0))) * r["face-on"]  # disk along y-hat
    r["edge-on-random"] = random_edgeon_matrix * r["face-on"]
    r["phi"] = phi
    r["identity"] = np.matrix(np.identity(3))

    return r


def rotationMatrixFromVec(i_v_in, target_vec=None):
    """Calculate 3x3 rotation matrix to align input vec with a target vector.

    By default this is the z-axis, such that with vec the angular momentum vector of the galaxy,
    an (x,y) projection will yield a face on view, and an (x,z) projection will yield an edge on view.
    """
    if target_vec is None:
        target_vec = np.asarray([0.0, 0.0, 1.0])

    # Normalize vector length
    i_v = np.copy(i_v_in)
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, target_vec)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, target_vec)
    rsin = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    mat = (
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

    return mat


def rotationMatrixFromAngleDirection(angle, direction):
    """Calculate 3x3 rotation matrix for input angle about an axis defined by the input direction.

    Args:
      angle (float): rotation angle in degrees.
      direction (ndarray[float][3] or list[float][3]): 3-vector defining direction of rotation, about the origin.
    """
    assert not isinstance(angle, np.ndarray)  # single value
    assert len(direction) == 3  # 3-tuple or ndarray of length 3

    angle_rad = np.radians(angle)

    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    direction /= np.linalg.norm(direction, 2)

    # rotation matrix about unit vector
    R = np.diag([cos_a, cos_a, cos_a])
    R += np.outer(direction, direction) * (1.0 - cos_a)
    direction *= sin_a

    R += np.array(
        [[0.0, -direction[2], direction[1]], [direction[2], 0.0, -direction[0]], [-direction[1], direction[0], 0.0]]
    )

    return R.astype("float32")


def rotationMatrixFromAngle(angle):
    """Calculate 2x2 rotation matrix for input angle, in degrees, CCW from the positive x-axis."""
    assert not isinstance(angle, np.ndarray)  # single value

    angle_rad = np.radians(angle)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return R.astype("float32")


def rotateCoordinateArray(sP, pos, rotMatrix, rotCenter, shiftBack=True):
    """Rotate a [N,3] array of Coordinates about rotCenter according to rotMatrix."""
    pos_in = np.array(pos)  # do not modify input

    # shift
    for i in range(3):
        pos_in[:, i] -= rotCenter[i]

    # if coordinates wrapped box boundary before shift:
    sP.correctPeriodicDistVecs(pos_in)

    # rotate
    pos_in = np.transpose(np.dot(rotMatrix, pos_in.transpose()))

    if shiftBack:
        for i in range(3):
            pos_in[:, i] += rotCenter[i]

    # return a symmetric extent which covers the origin-centered particle distribution, which is hard to
    # recover after we wrap the coordinates back into the box
    extent = np.zeros(3, dtype="float32")
    for i in range(3):
        right = pos_in[:, i].max() - rotCenter[i]
        left = rotCenter[i] - pos_in[:, i].min()
        extent[i] = 2.0 * np.max([left, right])

    # place all coordinates back inside [0,sP.boxSize] if necessary:
    if shiftBack:
        sP.correctPeriodicPosVecs(pos_in)

    pos_in = np.asarray(pos_in)  # matrix to ndarray

    return pos_in, extent


@jit(nopython=True, parallel=True)
def perspectiveProjection(n, f, l, r, b, t, pos, hsml, axes):
    """
    Transforms coordinates using a perspective projection, taking into account finite sizes.

    Based on the Perspective Projection Matrix using the ratio of
    similar triangles (http://www.songho.ca/opengl/gl_projectionmatrix.html).

    The truncated pyramid frustrum is defined by:
      n (float): The distance to the near plane along the line of sight direction.
      f (float): The distance to the far plane along the line of sight direction.
      [l, r] ([float, float]): The range of "x-axis" coordinates along the near plane.
      [b, t] ([float, float]): The range of "y-axis" coordinates along the near plane.

      (l,b)/(r,t) thus correspond to the bottom-left/top-right corners of the frustum projected onto the near plane.

    The Perspective Projection Matrix computed from this set of parameters is thereafter used to transform:
      pos (ndarray[float][N,3]): array of 3-coordinates for the particles; camera is situated at z=0.
      hsml (ndarray[float][N]): smoothing lengths.
      axes (list[int][2]): the axis of the projection plane, e.g. [0,1] for a projection along the z-axis.

    Returns:
      tPos (ndarray[float][N,3]): Transformed coordinates. Since the z-component is always mapped to the near
        plane in a perspective projection, only the x and y components are transformed here. The z component is
        left as is for use in, e.g., sphMap(). As a result, the transformed coordinates cannot be unprojected.
      tHsml (ndarray[float][N]): Transformed hsml; equal to original hsml along the near plane, and scales
        inversely with projection distance farther away from camera.

    Notes:
      * The tranformed values of hsml will be negative if the point is behind the camera.
    """
    tPos = np.zeros(pos.shape, dtype=pos.dtype)
    tHsml = np.zeros(hsml.shape, dtype=hsml.dtype)
    axis2 = 3 - axes[0] - axes[1]  # axis corresponding to the projection direction

    for j in prange(pos.shape[0]):
        x = ((2 * n) * (pos[j][axes[0]]) / (r - l) + (r + l) * (pos[j][axis2]) / (r - l)) / (-pos[j][axis2])
        x *= (r - l) / 2
        x += (r + l) / 2
        tPos[j][0] = x
        y = ((2 * n) * (pos[j][axes[1]]) / (t - b) + (t + b) * (pos[j][axis2]) / (t - b)) / (-pos[j][axis2])
        y *= (t - b) / 2
        y += (t + b) / 2
        tPos[j][1] = y
        tPos[j][2] = pos[j][axis2]

        tHsml[j] = n * hsml[j] / (-pos[j][axis2])

    return (tPos, tHsml)


def ellipsoidfit(pos_in, mass, scalerad, rin, rout, weighted=False):
    """Iterative algorithm to derive the shape (axes ratios) of a set of particles/cells.

    Their distribution is given by pos_in and mass, within a radial shell defined by rin, rout.
    Positions should be unitless, normalized by scalerad (a scaling factor, could be e.g. r200 or half mass radius).
    Originally from Eddie Chua.
    """
    pos = pos_in.copy()
    assert pos.shape[0] == mass.size

    convcrit = 1e-2  # convergence criterion
    minNumParticles = 20  # quit if we drop below this
    minEigenval = 1e-4  # quit if we drop below this
    maxNumIters = 100  # stop if no convergence after N iterations

    q = 1.0
    s = 1.0

    conv = 10
    count = 0

    mass_loc = []
    axes = np.eye(3)  # holds final rotation

    # iterate until convergence or until an early exit
    while conv > convcrit:
        count += 1

        # restrict to particles within required radii
        distsq = pos[:, 0] ** 2 + (pos[:, 1] / q) ** 2 + (pos[:, 2] / s) ** 2
        w = np.where((distsq > rin**2) & (distsq <= rout**2))

        if len(w[0]) < minNumParticles:
            # print('early exit, particle num dropped to [%d]' % len(w[0]))
            break

        pos_loc = pos[w[0], :]
        mass_loc = mass[w]

        # use weighted shape tensor?
        a2 = 1.0

        if weighted:
            a2 = pos_loc[:, 0] ** 2 + (pos_loc[:, 1] / q) ** 2 + (pos_loc[:, 2] / s) ** 2

        # calculate shape tensor
        M = np.zeros((3, 3), dtype="float32")

        for i in range(3):
            M[i, i] += np.sum(mass_loc * pos_loc[:, i] ** 2 / a2)

            for j in range(i):  # range(3) ??
                val = np.sum(mass_loc * pos_loc[:, i] * pos_loc[:, j] / a2)
                M[i, j] += val
                M[j, i] += val

        # normalize
        M *= scalerad**2 / np.sum(mass_loc)

        # get (normalized) eigenvectors and corresponding eigenvalues, sorting from largest to smallest
        eigvals, eigvecs = np.linalg.eigh(M)

        sort_inds = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_inds]
        eigvecs = eigvecs[:, sort_inds]  # columns are the principal axes

        rot_matrix = (
            eigvecs.T
        )  # actual rotation matrix is the transpose of this, since the transformation is I'= R I R.T

        # check rotation matrix using similarity transformation
        # the convention is M' = V.T M V, which should correspond to the eigenvalues on the diagonal
        if not np.allclose(np.dot(eigvecs.T, np.dot(M, eigvecs)), np.diag(eigvals), atol=1e-4):
            # print('Issue: ', np.dot(eigvecs.T, np.dot(M,eigvecs)), ' vs ', eigvals)
            return np.nan, np.nan, 0, axes  # complete failure

        # are any of the eigenvalues very small? the next iteration will then return imaginary values
        if (eigvals < minEigenval).any():
            # print('eigenvalues too small, stopping early')
            break

        if (eigvals < 1e-6).any():
            # print('Eigenvalues too small, stopping.')
            return np.nan, np.nan, 0, axes  # complete failure

        # obtain q and s from the eigenvalues
        q_new = np.sqrt(eigvals[1] / eigvals[0])
        s_new = np.sqrt(eigvals[2] / eigvals[0])

        # rotate positions
        pos = np.dot(pos, rot_matrix.T)

        # rotate the identity matrix as well, so that we can obtain the final rotation from the original frame
        axes = np.dot(axes, rot_matrix.T)

        # check convergence
        conv = np.max([np.abs((q_new - q) / q), np.abs((s_new - s) / s)])

        q, s = q_new, s_new

        if count > maxNumIters:
            break

        assert q <= 1.0 and s <= 1.0
        # print(' [%2d] q = %.3f s = %.3f ratio=%.2f n = %5d conv = %.3f' % (count,q,s,s/q,mass_loc.size,conv))

        # iterations done
        assert np.allclose(np.dot(pos_in, axes), pos, rtol=1e-2, atol=1e-3)

    if q == 1.0 and s == 1.0:
        # never completed one iteration, lack of particles
        return np.nan, np.nan, 0, axes

    return q, s, mass_loc.size, axes
