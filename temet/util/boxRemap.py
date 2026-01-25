"""
Implementation of 'a volume and local structure preserving mapping of periodic boxes' (Carlson, White 2010).

http://mwhite.berkeley.edu/BoxRemap/.
"""

from math import fmod

import numpy as np
from numba import jit

from ..util.helper import closest, rootPath


class Plane:
    """Plane class for half-space representation."""

    def __init__(self, p, n):
        """Init plane."""
        self.a = n[0]
        self.b = n[1]
        self.c = n[2]
        self.d = -1.0 * (p[0] * n[0] + p[1] * n[1] + p[2] * n[2])  # -dot(p,n)

    def normal(self):
        """Normal vector."""
        ell = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return [self.a / ell, self.b / ell, self.c / ell]

    def test(self, x, y, z):
        """Point-plane comparison: >0, <0 or ==0 depending on if point is above, below, or on the plane."""
        return self.a * x + self.b * y + self.c * z + self.d


class Cell:
    """Cell class for holding faces/planes."""

    def __init__(self, ix=0, iy=0, iz=0):
        """Init cell."""
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.faces = []

    def contains(self, x, y, z):
        """Check if cell contains point (x,y,z)."""
        for f in self.faces:
            if f.test(x, y, z) < 0:
                return False
        return True


class Cuboid:
    """Main cuboid remapping class."""

    def __init__(self, u1=(1, 0, 0), u2=(0, 1, 0), u3=(0, 0, 1)):
        """Initialize by passing a 3x3 invertible integer matrix."""
        if self._triple_scalar_product(u1, u2, u3) != 1:
            raise Exception("Invalid lattice vectors: u1 = %s, u2 = %s, u3 = %s" % (u1, u2, u3))
            self.e1 = np.array([1, 0, 0])
            self.e2 = np.array([0, 1, 0])
            self.e3 = np.array([0, 0, 1])
        else:
            s1 = self._square(u1)
            s2 = self._square(u2)
            d12 = self._dot(u1, u2)
            d23 = self._dot(u2, u3)
            d13 = self._dot(u1, u3)
            alpha = -d12 / s1
            gamma = -(alpha * d13 + d23) / (alpha * d12 + s2)
            beta = -(d13 + gamma * d12) / s1
            self.e1 = u1
            self.e2 = u2 + alpha * u1
            self.e3 = u3 + beta * u1 + gamma * u2

        self.L1 = self._length(self.e1)
        self.L2 = self._length(self.e2)
        self.L3 = self._length(self.e3)
        self.n1 = self.e1 / self.L1
        self.n2 = self.e2 / self.L2
        self.n3 = self.e3 / self.L3
        self.cells = []

        v0 = np.array([0, 0, 0])
        self.v = [
            v0,
            v0 + self.e3,
            v0 + self.e2,
            v0 + self.e2 + self.e3,
            v0 + self.e1,
            v0 + self.e1 + self.e3,
            v0 + self.e1 + self.e2,
            v0 + self.e1 + self.e2 + self.e3,
        ]

        # Compute bounding box of cuboid
        xs = [vk[0] for vk in self.v]
        ys = [vk[1] for vk in self.v]
        zs = [vk[2] for vk in self.v]
        vmin = [min(xs), min(ys), min(zs)]
        vmax = [max(xs), max(ys), max(zs)]

        # Extend to nearest integer coordinates
        ixmin = int(np.floor(vmin[0]))
        ixmax = int(np.ceil(vmax[0]))
        iymin = int(np.floor(vmin[1]))
        iymax = int(np.ceil(vmax[1]))
        izmin = int(np.floor(vmin[2]))
        izmax = int(np.ceil(vmax[2]))

        # Determine which cells (and which faces within those cells) are non-trivial
        for ix in range(ixmin, ixmax):
            for iy in range(iymin, iymax):
                for iz in range(izmin, izmax):
                    shift = [-ix, -iy, -iz]
                    faces = [
                        Plane(self.v[0] + shift, +self.n1),
                        Plane(self.v[4] + shift, -self.n1),
                        Plane(self.v[0] + shift, +self.n2),
                        Plane(self.v[2] + shift, -self.n2),
                        Plane(self.v[0] + shift, +self.n3),
                        Plane(self.v[1] + shift, -self.n3),
                    ]

                    c = Cell(ix, iy, iz)
                    skipcell = False
                    for f in faces:
                        r = self.UnitCubeTest(f)
                        if r == +1:
                            # Unit cube is completely above this plane; this cell is empty
                            continue
                        elif r == 0:
                            # Unit cube intersects this plane; keep track of it
                            c.faces.append(f)
                        elif r == -1:
                            skipcell = True
                            break

                    if skipcell or len(c.faces) == 0:
                        continue
                    else:
                        self.cells.append(c)

        # For the identity remapping, use exactly one cell
        if len(self.cells) == 0:
            self.cells.append(Cell())

    def _dot(self, u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def _triple_scalar_product(self, u, v, w):
        return (
            u[0] * (v[1] * w[2] - v[2] * w[1]) + u[1] * (v[2] * w[0] - v[0] * w[2]) + u[2] * (v[0] * w[1] - v[1] * w[0])
        )

    def _square(self, v):
        return v[0] ** 2 + v[1] ** 2 + v[2] ** 2

    def _length(self, v):
        return np.sqrt(self._square(v))

    def UnitCubeTest(self, P):
        """Return +1, 0, or -1 if the unit cube is above, below, or intersecting the plane."""
        above = 0
        below = 0
        for a, b, c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            s = P.test(a, b, c)
            if s > 0:
                above = 1
            elif s < 0:
                below = 1
        return above - below

    def GetCells(self):
        """Create and return flattened, ndarray representation of cells and faces/planes."""
        nCellsTot = len(self.cells)

        nFacesTot = 0
        faceOff = np.zeros(nCellsTot, dtype="int32")
        faceLen = np.zeros(nCellsTot, dtype="int32")
        cell_ix = np.zeros(nCellsTot, dtype="int32")
        cell_iy = np.zeros(nCellsTot, dtype="int32")
        cell_iz = np.zeros(nCellsTot, dtype="int32")

        # loop over all cells
        for i, c in enumerate(self.cells):
            # create offset table for faces
            faceOff[i] = nFacesTot
            faceLen[i] = len(c.faces)

            cell_ix[i] = c.ix
            cell_iy[i] = c.iy
            cell_iz[i] = c.iz

            nFacesTot += len(c.faces)

        # allocate face/plane information
        wOffset = 0

        face_a = np.zeros(nFacesTot, dtype="float64")
        face_b = np.zeros(nFacesTot, dtype="float64")
        face_c = np.zeros(nFacesTot, dtype="float64")
        face_d = np.zeros(nFacesTot, dtype="float64")

        # loop over all cells, a second time
        for c in self.cells:
            # loop over local faces/planes
            for face in c.faces:
                face_a[wOffset] = face.a
                face_b[wOffset] = face.b
                face_c[wOffset] = face.c
                face_d[wOffset] = face.d

                wOffset += 1

        assert wOffset == nFacesTot

        return nCellsTot, nFacesTot, faceOff, faceLen, cell_ix, cell_iy, cell_iz, face_a, face_b, face_c, face_d

    def GetN123(self):
        """Return remapping numbers."""
        return self.n1, self.n2, self.n3

    def Transform(self, x, y, z):
        """Transform a point (x,y,z) in [0,1] cubical space to remapped cuboid space."""
        for c in self.cells:
            if c.contains(x, y, z):
                x += c.ix
                y += c.iy
                z += c.iz
                p = [x, y, z]
                return (self._dot(p, self.n1), self._dot(p, self.n2), self._dot(p, self.n3))
        raise Exception("(%g, %g, %g) not contained in any cell" % (x, y, z))

    def InverseTransform(self, r1, r2, r3):
        """Inverse transform a point (r1,r2,r3) in remapped cuboid space to [0,1] cubical space."""
        p = r1 * self.n1 + r2 * self.n2 + r3 * self.n3
        x1 = fmod(p[0], 1) + (p[0] < 0)
        x2 = fmod(p[1], 1) + (p[1] < 0)
        x3 = fmod(p[2], 1) + (p[2] < 0)
        return [x1, x2, x3]


@jit(nopython=True, nogil=True, cache=True)
def CuboidTransformPoint(
    x,
    y,
    z,
    nCellsTot,
    nFacesTot,
    faceOff,
    faceLen,
    cell_ix,
    cell_iy,
    cell_iz,
    face_a,
    face_b,
    face_c,
    face_d,
    n1,
    n2,
    n3,
):
    """Do Cuboid transform of a single (x,y,z) point given the flattened input configuration. Return 3-tuple."""
    for i in range(nCellsTot):
        # loop over all child faces, evaluate c.contains()
        contains = True

        for j in range(faceOff[i], faceOff[i] + faceLen[i]):
            # point-plane comparison
            test = face_a[j] * x + face_b[j] * y + face_c[j] * z + face_d[j]

            if test < 0:
                contains = False
                break

        if not contains:
            continue

        # cell contains (x,y,z), begin transformation
        x_new = x + cell_ix[i]
        y_new = y + cell_iy[i]
        z_new = z + cell_iz[i]

        # final (x,y,z)_out is (x,y,z)_new dotted with n1, n2, and n3, respectively
        x_new2 = x_new * n1[0] + y_new * n1[1] + z_new * n1[2]
        y_new2 = x_new * n2[0] + y_new * n2[1] + z_new * n2[2]
        z_new2 = x_new * n3[0] + y_new * n3[1] + z_new * n3[2]

        return x_new2, y_new2, z_new2
    return -1, -1, -1  # failure


@jit(nopython=True, nogil=True, cache=True)
def CuboidTransformArray(
    pos_in,
    pos_out,
    nCellsTot,
    nFacesTot,
    faceOff,
    faceLen,
    cell_ix,
    cell_iy,
    cell_iz,
    face_a,
    face_b,
    face_c,
    face_d,
    n1,
    n2,
    n3,
):
    """Do Cuboid transform of a [N,3] coordinate array, given the flattened input configuration.

    Write results to pos_out.
    """
    for i in range(pos_in.shape[0]):
        x_in = pos_in[i, 0]
        y_in = pos_in[i, 1]
        z_in = pos_in[i, 2]

        x_new, y_new, z_new = CuboidTransformPoint(
            x_in,
            y_in,
            z_in,
            nCellsTot,
            nFacesTot,
            faceOff,
            faceLen,
            cell_ix,
            cell_iy,
            cell_iz,
            face_a,
            face_b,
            face_c,
            face_d,
            n1,
            n2,
            n3,
        )

        if x_new == -1:
            # failed due to floating point rounding issues, add small perturbation and redo
            x_in += 1e-12
            y_in += 1e-12
            z_in += 1e-12

            x_new, y_new, z_new = CuboidTransformPoint(
                x_in,
                y_in,
                z_in,
                nCellsTot,
                nFacesTot,
                faceOff,
                faceLen,
                cell_ix,
                cell_iy,
                cell_iz,
                face_a,
                face_b,
                face_c,
                face_d,
                n1,
                n2,
                n3,
            )

        pos_out[i, 0] = x_new
        pos_out[i, 1] = y_new
        pos_out[i, 2] = z_new


def findCuboidRemapInds(remapRatio, nPixels=None):
    """Find the closest remapping matrix (3x3) to achieve the input remapping ratio of [x,y,z] relative extents.

    Also input nPixels [x,y] of the final image to be made, in order to calculate newBoxSize.
    """
    assert len(remapRatio) == 3, "Error: remapRatio should have three elements."
    assert np.abs(1.0 - np.prod(remapRatio)) < 1e-3, "Error: Check L1*L2*L3 == 1 constraint."

    # load pre-computed mapping posibilities
    file = rootPath + "tables/box_remap_N7.txt"  # e1,e2,e3,u11,u12,u13,u21,u22,u23,u31,u32,u33,periodicity
    data = np.loadtxt(file, comments="#", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # skip last (periodicity)

    # calculate closest matching edge length set (use abs(xyz) distance metric)
    dists = np.abs(data[:, 0] - remapRatio[0]) + np.abs(data[:, 1] - remapRatio[1]) + np.abs(data[:, 2] - remapRatio[2])
    min_dist, ind = closest(dists, 0.0)

    # construct return
    u11 = data[ind, 3]
    u12 = data[ind, 4]
    u13 = data[ind, 5]
    u21 = data[ind, 6]
    u22 = data[ind, 7]
    u23 = data[ind, 8]
    u31 = data[ind, 9]
    u32 = data[ind, 10]
    u33 = data[ind, 11]

    remapMatrix = np.vstack(([u11, u12, u13], [u21, u22, u23], [u31, u32, u33]))

    newBoxSize = np.array([data[ind, 0], data[ind, 1], data[ind, 2]])  # [e1,e2,e3]

    # print('remapRatio: ', remapRatio, ' e123: ',newBoxSize,' found distance:', min_dist, ' with index: ',ind)
    assert min_dist < 0.1, "Warning: Inaccurate remapping matrix chosen (can disable)."

    # adjust box size, increasing smaller xy dimension to match the pixel aspect ratio, and so keep the output image
    # size exactly as requested (resulting in a black border somewhere)
    if nPixels is not None:
        # 2d
        if newBoxSize[0] < newBoxSize[1]:
            newBoxSize[0] = newBoxSize[1] * (float(nPixels[0]) / nPixels[1])
        else:
            newBoxSize[1] = newBoxSize[0] * (float(nPixels[1]) / nPixels[0])

    return remapMatrix, newBoxSize


def remapPositions(sP, pos, remapRatio, nPixels):
    """Remap an array of points from a cubic periodic box into a remapped cuboid volume.

    Input: [N,3] coordinate array from the original cubic periodic domain of side-length sP.boxSize, into
    a non-periodic, rectangular domain with the relative shape of remapRatio in the (x,y,z) dimensions.
    """
    remapMatrix, newBoxSize = findCuboidRemapInds(remapRatio, nPixels)
    newBoxSize = np.array(newBoxSize) * sP.boxSize

    # init
    u1 = remapMatrix[0, :]
    u2 = remapMatrix[1, :]
    u3 = remapMatrix[2, :]
    C = Cuboid(u1, u2, u3)

    nCellsTot, nFacesTot, faceOff, faceLen, cell_ix, cell_iy, cell_iz, face_a, face_b, face_c, face_d = C.GetCells()
    n1, n2, n3 = C.GetN123()

    # normalize to [0,1], leave input array intact
    pos_in = pos.copy() / sP.boxSize

    # allocate return and remap
    pos_remapped = np.zeros(pos.shape, dtype=pos.dtype)

    CuboidTransformArray(
        pos_in,
        pos_remapped,
        nCellsTot,
        nFacesTot,
        faceOff,
        faceLen,
        cell_ix,
        cell_iy,
        cell_iz,
        face_a,
        face_b,
        face_c,
        face_d,
        n1,
        n2,
        n3,
    )

    # expand back into boxSize * remapRatio
    pos_remapped *= sP.boxSize

    return pos_remapped, newBoxSize
