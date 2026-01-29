"""
Algorithms and methods related to the use and analysis of the Voronoi mesh.
"""

import time
from os.path import isfile

import h5py
import numpy as np
from numba import jit


def loadSingleHaloVPPP(sP, haloID):
    """Load Voronoi connectivity information for a single FoF halo."""
    subset = sP.haloOrSubhaloSubset(haloID=haloID)

    indStart = subset["offsetType"][sP.ptNum("gas")]
    indStop = indStart + subset["lenType"][sP.ptNum("gas")]

    # read neighbor list for all gas cells of this halo
    filename = sP.derivPath + "voronoi/mesh_%02d.hdf5" % sP.snap

    with h5py.File(filename, "r") as f:
        num_ngb = f["num_ngb"][indStart:indStop]
        offset_ngb = f["offset_ngb"][indStart:indStop]

        tot_ngb = num_ngb.sum()
        assert offset_ngb[0] + tot_ngb == offset_ngb[-1] + num_ngb[-1]

        ngb_inds = f["ngb_inds"][offset_ngb[0] : offset_ngb[0] + tot_ngb]

    # make mesh indices and offsets halo-local
    ngb_inds -= indStart
    offset_ngb -= offset_ngb[0]

    # flag any mesh neighbors which are beyond halo-scope (outside fof) as -1
    w = np.where((ngb_inds < 0) | (ngb_inds >= num_ngb.size))

    ngb_inds[w] = -1

    # return (n_ngb[ncells], ngb_list[n_ngb.sum()], ngb_offset[ncells])
    return num_ngb, ngb_inds, offset_ngb


def loadGlobalVPPP(sP):
    """Load global Voronoi connectivity information for a snapshot."""
    filename = sP.derivPath + "voronoi/mesh_%02d.hdf5" % sP.snap

    with h5py.File(filename, "r") as f:
        num_ngb = f["num_ngb"][()]
        offset_ngb = f["offset_ngb"][()]
        ngb_inds = f["ngb_inds"][()]

    return num_ngb, ngb_inds, offset_ngb


@jit(nopython=True, nogil=True)  # , cache=True)
def _contiguousVoronoiCells(num_ngb, offset_ngb, ngb_inds, prop_val, identity, mode, propThresh):
    """Identify contiguous (naturally connected) subsets of the input Voronoi mesh cells.

    Only those with thresh_mask == True are assigned. Identity output.
    """
    ncells = num_ngb.size

    # process cells in a sorted order (ascending if mode==1/lt, descending if mode==0/gt)
    # note: result is independent of this processing order if we include the merge stage
    sort_inds = np.argsort(prop_val, kind="mergesort")

    count = 0

    # loop over all cells
    for i in range(ncells):
        # which cell?
        if mode == 0:
            cell_index_i = sort_inds[ncells - i - 1]
        if mode == 1:
            cell_index_i = sort_inds[i]

        # skip cells which do not satisfy threshold
        if mode == 0 and prop_val[cell_index_i] < propThresh:
            continue

        if mode == 1 and prop_val[cell_index_i] > propThresh:
            continue

        # loop over all natural neighbors of this cell
        for j in range(num_ngb[cell_index_i]):
            # index of this voronoi neighbor
            ngb_index = offset_ngb[cell_index_i] + j
            cell_index_j = ngb_inds[ngb_index]

            # if neighbor is not in FoF-scope particle load, skip
            if cell_index_j == -1:
                continue

            # if the neighbor does not satisfy threshold, skip
            if mode == 0 and prop_val[cell_index_j] < propThresh:
                continue

            if mode == 1 and prop_val[cell_index_j] > propThresh:
                continue

            # if the neighbor belongs to an existing object? then assign to this cell, and exit
            if identity[cell_index_j] >= 0:
                identity[cell_index_i] = identity[cell_index_j]
                break

        # no neighbors already assigned, so start a new object now
        if identity[cell_index_i] < 0:
            identity[cell_index_i] = count
            count += 1

    # iterate: merge step
    assert np.max(identity) + 1 == count

    for _niter in range(1000):
        changes_count = 0

        for i in range(ncells):
            # skip cells which do not below to any object
            if identity[i] < 0:
                continue

            # loop over all natural neighbors of this cell
            for j in range(num_ngb[i]):
                # index of this voronoi neighbor
                ngb_index = offset_ngb[i] + j
                cell_index_j = ngb_inds[ngb_index]

                if cell_index_j < 0 or identity[cell_index_j] < 0:
                    continue

                # neighbor belongs to an object with a lower id? merge cell (i) into this object
                if identity[cell_index_j] < identity[i]:
                    identity[i] = identity[cell_index_j]
                    changes_count += 1

        # converged?
        if changes_count == 0:
            break

    # debug check: every assigned cell should have neighbors with the same identity, or identity == -1
    for i in range(ncells):
        # skip cells which do not below to any object
        if identity[i] < 0:
            continue

        # loop over all natural neighbors of this cell
        for j in range(num_ngb[i]):
            # index of this voronoi neighbor
            ngb_index = offset_ngb[i] + j
            cell_index_j = ngb_inds[ngb_index]

            if cell_index_j < 0 or identity[cell_index_j] < 0:
                continue

            assert identity[i] == identity[cell_index_j]

    # identity IDs are now sparse, condense
    c = np.zeros(count, dtype=np.int32) - 1

    for i in range(identity.size):
        if identity[i] >= 0:
            c[identity[i]] = 1

    count = 0

    # create mapping (old ID -> new ID)
    for i in range(c.size):
        if c[i] > 0:
            # old ID still exists, assign a new dense ID, and recount
            c[i] = count
            count += 1

    # relabel cells
    for i in range(ncells):
        if identity[i] >= 0:
            assert c[identity[i]] >= 0
            identity[i] = c[identity[i]]

    assert np.max(identity) + 1 == count

    return count


def voronoiThresholdSegmentation(sP, haloID, propName, propThresh, propThreshComp):
    """Apply a Voronoi thresholded segmentation algorithm to all gas cells in a halo.

    Identifies spatially connected collectins, in the sense that they are Voronoi natural neighbors, and which satisfy
    a threshold criterion on a particular gas property, e.g. log(T) < 4.5.
    """
    assert propThresh != 0.0, "Better to use +/- 1e-10 for instance, to avoid issues around zero."

    saveFilename = sP.derivPath + "voronoi/segmentation_%d_h%d_%s_%s_%s.hdf5" % (
        sP.snap,
        haloID,
        propName.replace(" ", "-"),
        propThreshComp,
        propThresh,
    )

    # check for existence of pre-existing save
    if isfile(saveFilename):
        objects = {}
        props = {}

        with h5py.File(saveFilename, "r") as f:
            for key in f["objects"]:
                objects[key] = f["objects"][key][()]
            for key in f["props"]:
                props[key] = f["props"][key][()]

        return objects, props

    # load mesh
    num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID)
    prop_val = sP.snapshotSubset("gas", propName, haloID=haloID)

    ncells = num_ngb.size

    assert prop_val.shape[0] == num_ngb.size

    # shift property threshold to zero
    if propThreshComp == "gt":
        w_thresh = np.where(prop_val > propThresh)
        mode = 0
    if propThreshComp == "lt":
        w_thresh = np.where(prop_val < propThresh)
        mode = 1

    frac_cells = len(w_thresh[0]) / ncells * 100
    print(
        "segmentation: %.1f%% of the total [%d] cells satisfy threshold (%s %s %s)"
        % (frac_cells, ncells, propName, propThreshComp, propThresh)
    )

    # run segmentation algorithm
    identity = np.zeros(ncells, dtype="int32") - 1

    start_time = time.time()

    count = _contiguousVoronoiCells(num_ngb, offset_ngb, ngb_inds, prop_val, identity, mode, propThresh)

    print("segmentation found [%d] objects, took [%.1f] sec" % (count, time.time() - start_time))

    # due to small round off differences, we take the answer of identity to determine which cells satisfy the threshold
    w_thresh = np.where(identity >= 0)

    # count number of cells per object
    lengths = np.bincount(identity[w_thresh])
    assert lengths.size == count

    # zero objects? early return
    if count == 0:
        with h5py.File(saveFilename, "w") as f:
            f["objects/count"] = 0
            f["props/dummy"] = 0

        print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])
        return {"count": 0}, {"dummy": 0}

    # create mapping from (assigned cells) -> (parent object index)
    assert lengths.min() > 0
    bins = np.arange(identity[w_thresh].max() + 1)
    obj_inds = np.searchsorted(bins, identity[w_thresh])

    # create ordered list of cell indices per object, and offsets
    cell_inds = w_thresh[0][np.argsort(obj_inds, kind="mergesort")]  # stable

    offsets = np.zeros(count, dtype="int32")
    offsets[1:] = np.cumsum(lengths)[:-1]

    # per object: compute volume, radius, mass (tot), prop (tot), prop (mean), prop (median)
    print("loading additional data for properties...")
    mass = sP.snapshotSubset("gas", "mass", haloID=haloID)
    dens = sP.snapshotSubset("gas", "nh", haloID=haloID)
    temp = sP.snapshotSubset("gas", "temp_sfcold", haloID=haloID)
    bmag = sP.snapshotSubset("gas", "bmag", haloID=haloID)
    rcell = sP.snapshotSubset("gas", "cellsize_kpc", haloID=haloID)
    vol = sP.snapshotSubset("gas", "volume", haloID=haloID)
    specj = sP.snapshotSubset("gas", "specj_mag", haloID=haloID)
    pos = sP.snapshotSubset("gas", "pos", haloID=haloID)
    vrel = sP.snapshotSubset("gas", "vrel", haloID=haloID)
    vrad = sP.snapshotSubset("gas", "vrad", haloID=haloID)
    sfr = sP.snapshotSubset("gas", "sfr", haloID=haloID)
    metal = sP.snapshotSubset("gas", "z_solar", haloID=haloID)
    beta = sP.snapshotSubset("gas", "beta", haloID=haloID)

    mg2_mass = sP.snapshotSubset("gas", "Mg II mass", haloID=haloID)
    hi_mass = sP.snapshotSubset("gas", "MHI_GK", haloID=haloID)
    if hi_mass is None:
        hi_mass = sP.snapshotSubset("gas", "hi mass", haloID=haloID)  # simpler model

    props = {
        "vol": np.zeros(count, dtype="float32"),  # code units
        "mass": np.zeros(count, dtype="float32"),  # code units
        "dens_mean": np.zeros(count, dtype="float32"),  # linear 1/cm^3
        "temp_mean": np.zeros(count, dtype="float32"),  # linear K
        "bmag_mean": np.zeros(count, dtype="float32"),  # physical Gauss
        "rcell_mean": np.zeros(count, dtype="float32"),  # pkpc
        "rcell_min": np.zeros(count, dtype="float32"),  # pkpc
        "sfr_tot": np.zeros(count, dtype="float32"),  # msun/yr
        "metal_mean": np.zeros(count, dtype="float32"),  # linear solar
        "beta_mean": np.zeros(count, dtype="float32"),  # linear
        "mg2_mass": np.zeros(count, dtype="float32"),  # code units
        "hi_mass": np.zeros(count, dtype="float32"),  # code units
        "specj_tot": np.zeros(count, dtype="float32"),  # kpc km/s
        "vrad_mean": np.zeros(count, dtype="float32"),  # km/s
        "prop_tot": np.zeros(count, dtype="float32"),  # code
        "prop_mean": np.zeros(count, dtype="float32"),  # code
        "prop_median": np.zeros(count, dtype="float32"),  # code
        "cen": np.zeros((count, 3), dtype="float32"),  # code xyz
        "cen_masswt": np.zeros((count, 3), dtype="float32"),  # code xyz
        "cen_denswt": np.zeros((count, 3), dtype="float32"),  # code xyz
        "cen_propwt": np.zeros((count, 3), dtype="float32"),  # code xyz
        "vrel": np.zeros((count, 3), dtype="float32"),  # km/s xyz
        "vrel_masswt": np.zeros((count, 3), dtype="float32"),  # km/s xyz
        "vrel_denswt": np.zeros((count, 3), dtype="float32"),  # km/s xyz
        "vrel_propwt": np.zeros((count, 3), dtype="float32"),
    }  # km/s xyz

    for i in range(count):
        if i % np.max([int(count / 10), 1]) == 0:
            print("%d%%" % ((i + 1) / count * 100))
        loc_inds = cell_inds[offsets[i] : offsets[i] + lengths[i]]
        assert identity[loc_inds].min() == i and identity[loc_inds].max() == i

        props["vol"][i] = vol[loc_inds].sum()
        props["mass"][i] = mass[loc_inds].sum()
        props["dens_mean"][i] = dens[loc_inds].mean()
        props["temp_mean"][i] = temp[loc_inds].mean()
        props["bmag_mean"][i] = bmag[loc_inds].mean()
        props["rcell_mean"][i] = rcell[loc_inds].mean()
        props["rcell_min"][i] = rcell[loc_inds].min()
        props["sfr_tot"][i] = sfr[loc_inds].sum()
        props["metal_mean"][i] = metal[loc_inds].mean()
        props["beta_mean"][i] = beta[loc_inds].mean()
        props["mg2_mass"][i] = mg2_mass[loc_inds].sum()
        props["hi_mass"][i] = hi_mass[loc_inds].sum()
        props["specj_tot"][i] = specj[loc_inds].sum()
        props["vrad_mean"][i] = vrad[loc_inds].mean()

        props["cen"][i, :] = np.average(pos[loc_inds, :], axis=0)
        props["cen_masswt"][i, :] = np.average(pos[loc_inds, :], axis=0, weights=mass[loc_inds])
        props["cen_denswt"][i, :] = np.average(pos[loc_inds, :], axis=0, weights=dens[loc_inds])
        props["cen_propwt"][i, :] = np.average(pos[loc_inds, :], axis=0, weights=prop_val[loc_inds])

        props["vrel"][i, :] = np.average(vrel[loc_inds, :], axis=0)
        props["vrel_masswt"][i, :] = np.average(vrel[loc_inds, :], axis=0, weights=mass[loc_inds])
        props["vrel_denswt"][i, :] = np.average(vrel[loc_inds, :], axis=0, weights=dens[loc_inds])
        props["vrel_propwt"][i, :] = np.average(vrel[loc_inds, :], axis=0, weights=prop_val[loc_inds])

        props["prop_tot"][i] = prop_val[loc_inds].sum()
        props["prop_mean"][i] = np.nanmean(prop_val[loc_inds])
        props["prop_median"][i] = np.nanmedian(prop_val[loc_inds])

    props["radius"] = (props["vol"] * 3.0 / (4 * np.pi)) ** (1.0 / 3.0)  # code units

    halo_cen = sP.groupCatSingle(haloID=haloID)["GroupPos"]
    props["distance"] = sP.periodicDists(halo_cen, props["cen"])  # code units

    # save
    objects = {
        "count": count,
        "lengths": lengths,
        "offsets": offsets,
        "obj_inds": obj_inds,
        "cell_inds": cell_inds,
        "identity": identity,
    }

    with h5py.File(saveFilename, "w") as f:
        for key in objects:
            f["objects/%s" % key] = objects[key]
        for key in props:
            f["props/%s" % key] = props[key]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    # note: obj_lens and obj_offsets access into cell_inds in the usual offset table fashion, such that
    # cell_inds[obj_offsets[i] : obj_offsets[i]+obj_lens[i]] gives the list of cell indices (halo local) in this object
    return objects, props


def _circumsphereCenter(p1, p2, p3, p4):
    """For four points in 3D making up a tetra, compute the circumsphere center.

    We solve for the coordinates (xc,yc,zc) of the circumcenter, using the fact that
    the vectors P12 = (x2-x1,y2-y1,z2-z1), P13, and P14 are secants of the sphere.
    Each forms a right triangle with the diameter through P1.
    This produces the linear system:

    (x2-x1) * xc + (y2-y1) * yc + ( z2-z1) * zc = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    (x3-x1) * xc + (y3-y1) * yc + ( z3-z1) * zc = (x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2
    (x4-x1) * xc + (y4-y1) * yc + ( z4-z1) * zc = (x4-x1)**2 + (y4-y1)**2 + (z4-z1)**2
    """
    cen = np.zeros(3, dtype=np.float64)  # TODO: move alloc outside of this func (reuse)

    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    z21 = p2[2] - p1[2]
    x31 = p3[0] - p1[0]
    y31 = p3[1] - p1[1]
    z31 = p3[2] - p1[2]
    x41 = p4[0] - p1[0]
    y41 = p4[1] - p1[1]
    z41 = p4[2] - p1[2]

    rhs1 = x21**2 + y21**2 + z21**2
    rhs2 = x31**2 + y31**2 + z31**2
    rhs3 = x41**2 + y41**2 + z41**2

    # x21 * xc + y21 * yc + z21 * zc = rhs1  [1]
    # x31 * xc + y31 * yc + z31 * zc = rhs2  [2]
    # x41 * xc + y41 * yc + z41 * zc = rhs3  [3]

    # [ [x21,y21,z21],[x31,y31,z31],[x41,y41,z41] ] * [xc,yc,zc].T = [rhs1,rhs2,rhs3].T
    # [xc,yc,zc] = inv(A) * rhs.T

    # construct inv(A) = (1/det_A) * [[A,B,C],[D,E,F],[G,H,I]].T = (1/det_A) * [[A,D,G],[B,E,H],[C,F,I]]
    A = +(y31 * z41 - z31 * y41)
    B = -(x31 * z41 - z31 * x41)
    C = +(x31 * y41 - y31 * x41)

    D = -(y21 * z41 - z21 * y41)
    E = +(x21 * z41 - z21 * x41)
    F = -(x21 * y41 - y21 * x41)

    G = +(y21 * z31 - z21 * y31)
    H = -(x21 * z31 - z21 * x31)
    I = +(x21 * y31 - y21 * x31)

    det_A = x21 * A + y21 * B + z21 * C

    assert det_A != 0.0  # otherwise degenerate geometry

    inv_det_A = 1.0 / det_A

    cen[0] = A * rhs1 + D * rhs2 + G * rhs3
    cen[1] = B * rhs1 + E * rhs2 + H * rhs3
    cen[2] = C * rhs1 + F * rhs2 + I * rhs3

    cen *= inv_det_A

    radius = 0.5 * np.sqrt(cen[0] ** 2 + cen[1] ** 2 + cen[2] ** 2)

    cen[0] = p1[0] + 0.5 * cen[0]
    cen[1] = p1[1] + 0.5 * cen[1]
    cen[2] = p1[2] + 0.5 * cen[2]

    return radius, cen


def _testCoplanarity4(p1, p2, p3, p4):
    """Check if the four points in 3D all lie on a plane."""
    eps_tol = 1e-12

    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]

    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = -a * p1[0] - b * p1[1] - c * p1[2]

    # equation of the plane defined by the first three points:
    # a*x + b*y + c*z + d = 0

    # check if fourth point satisfies this equation
    if np.abs(a * p4[0] + b * p4[1] + c * p4[2] + d) < eps_tol:
        return True

    return False


def test_cen():
    """Test circumsphere center and coplanarity functions."""
    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 0.0, 0.0]
    p3 = [0.0, 1.0, 0.0]
    p4 = [0.0, 0.0, 10.0]

    rad, cen = _circumsphereCenter(p1, p2, p3, p4)
    print(rad, cen)

    print(_testCoplanarity4(p1, p2, p3, p4))

    for i in range(10000):
        p1 = np.random.uniform(low=-200.0, high=200.0, size=3)
        p2 = np.random.uniform(low=-200.0, high=200.0, size=3)
        p3 = np.random.uniform(low=-200.0, high=200.0, size=3)
        p4 = np.random.uniform(low=-200.0, high=200.0, size=3)

        print(i, _circumsphereCenter(p1, p2, p3, p4), _testCoplanarity4(p1, p2, p3, p4))
        assert not _testCoplanarity4(p1, p2, p3, p4)


def plotFacePolygon(cell_cen, neighbor_cen, vertices):
    """Plot helper."""
    # import matplotlib
    # matplotlib.use('TkAgg') # set in matplotlibrc first

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    ax.scatter3D(cell_cen[0], cell_cen[1], cell_cen[2], color="black")
    ax.scatter3D(neighbor_cen[0], neighbor_cen[1], neighbor_cen[2], color="blue")
    ax.plot3D([cell_cen[0], neighbor_cen[0]], [cell_cen[1], neighbor_cen[1]], [cell_cen[2], neighbor_cen[2]])

    plt.show()


def _findCellFaces(i, num_ngb, ngb_inds, offset_ngb, pos):
    """Determine faces of a Voronoi cell."""
    max_n_tetra_per_face = 100

    # loop over neighbors
    for j in range(num_ngb[i]):
        # index of this neighbor (local cell ordered)
        ind_j = ngb_inds[offset_ngb[i] + j]  # 1st neighbor, defines face

        if ind_j == -1:
            continue

        tetra = np.zeros((max_n_tetra_per_face, 4), dtype="int32")
        tetra_cen = np.zeros((max_n_tetra_per_face, 3), dtype="float64")

        n_tetra = 0
        print("j = %2d check (num_ngb = %2d)" % (ind_j, num_ngb[ind_j]))

        # find another neighbor which is also a neighbor of (i) and (j)
        for k in range(num_ngb[ind_j]):
            # index of this candidate third cell (local cell ordered)
            ind_k = ngb_inds[offset_ngb[ind_j] + k]

            if ind_k == -1:
                continue

            print(" k = %2d check (index = %2d num_ngb = %2d)" % (k, ind_k, num_ngb[ind_k]))

            for l in range(num_ngb[ind_k]):
                ind_l = ngb_inds[offset_ngb[ind_k] + l]

                # no link back to original cell (i), skip
                if ind_l != i:
                    continue

                # we have verified (k) links back to original cell (i)

                # since (k) and (j) both neighbor (i), we have a triangle
                # defined by local cell ordered indices {i,ind_j,ind_k}
                # need a fourth to construct a tetra, and thus one vertex of this face
                # is given by the circumsphere center of this tetra
                print("  tri: [%2d,%2d,%2d]" % (i, ind_j, ind_k))

                # note: this fourth must still be a neighbor of (j), so search there
                for m in range(num_ngb[ind_j]):
                    ind_m = ngb_inds[offset_ngb[ind_j] + m]

                    if ind_m == -1:
                        continue

                    if ind_m == i or ind_m == ind_k:
                        continue

                    # to avoid duplicate tetras with differing orientations:
                    if ind_m > ind_k:
                        continue

                    # have a candidate
                    count = 0
                    count_check = 0

                    for n in range(num_ngb[ind_m]):
                        ind_n = ngb_inds[offset_ngb[ind_m] + n]

                        if ind_n == i:
                            count += 1
                        if ind_n == ind_k:
                            count += 1
                        if ind_n == ind_j:
                            count_check += 1

                    assert count_check == 1

                    if count == 2:
                        # have found a fourth index {ind_m} forming a tetra
                        # {i,ind_j,ind_k,ind_m}
                        print("  tetra: [%2d,%2d,%2d,%2d] m = %2d" % (i, ind_j, ind_k, ind_m, m))
                        tetra[n_tetra, :] = (i, ind_j, ind_k, ind_m)
                        n_tetra += 1
                    # else:
                    #    print('  no tetra, count = %d' % count)

            print(" k = %2d done" % k)

        print("j = %2d done (n_tetra = %d)" % (ind_j, n_tetra))

        # compute circumsphere centers of each tetra
        for tetra_ind in range(n_tetra):
            p1 = pos[tetra[tetra_ind, 0], :]
            p2 = pos[tetra[tetra_ind, 1], :]
            p3 = pos[tetra[tetra_ind, 2], :]
            p4 = pos[tetra[tetra_ind, 3], :]
            _, tetra_cen[tetra_ind, :] = _circumsphereCenter(p1, p2, p3, p4)

        # check: have a set of (x,y,z) vertices, which should all lie on a plane
        p1 = tetra_cen[0, :]
        p2 = tetra_cen[1, :]
        p3 = tetra_cen[2, :]

        for tetra_ind in range(n_tetra):
            p4 = tetra_cen[tetra_ind, :]
            assert _testCoplanarity4(p1, p2, p3, p4)

        # center of this polygon need not coincide with the midpoint between (i) and (j)
        face_cen = np.mean(tetra_cen[0:n_tetra, :], axis=0)
        # ij_midpoint = 0.5 * (pos[i, :] + pos[ind_j, :])

        # face is complete, are points ordered in some fashion? we want to walk along the
        # edge of the face (or equivalently the triangles making up the face) by considering
        # adjacent vertices

        # compute the angle from any interior point on the face (i.e. the geometrical center)
        # to every vertex, then we can sort by this angle to order the points
        angles = np.zeros(n_tetra, dtype="float64")

        # shift face center to origin (TEMPORARY)
        tetra_cen[:, 0] -= face_cen[0]
        tetra_cen[:, 1] -= face_cen[1]
        tetra_cen[:, 2] -= face_cen[2]

        for tetra_ind in range(n_tetra):
            p = tetra_cen[tetra_ind, :]
            arg = p[0] * face_cen[0] + p[1] * face_cen[1] + p[2] * face_cen[2]
            denom = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * (face_cen[0] ** 2 + face_cen[1] ** 2 + face_cen[2] ** 2)
            angles[tetra_ind] = np.degrees(np.arccos(arg / np.sqrt(denom)))

        assert 0  # todo: finish

        # plot face polygon
        # plotFacePolygon(pos[i,:]-face_cen, pos[ind_j,:]-face_cen, tetra_cen[0:n_tetra,:])


def voronoiSliceWithPlane():
    """Testing."""
    from ..util.simParams import simParams

    sP = simParams(run="tng50-1", redshift=0.5)
    haloID = 19

    # load
    num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID)

    pos = sP.snapshotSubset("gas", "pos", haloID=haloID)

    # test
    cell_ind = 0
    _findCellFaces(cell_ind, num_ngb, ngb_inds, offset_ngb, pos)

    # todo: finish
