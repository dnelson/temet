"""
Algorithms and methods related to Delaunay tetrahedral meshes, and ray-tracing.
"""

import math
import time
from os.path import isfile

import h5py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from scipy.spatial import Delaunay

from ..util.helper import logZeroMin, pSplit
from ..util.rotation import rotateCoordinateArray


try:
    from cuda.pathfinder import DynamicLibNotFoundError
    from numba import cuda
except (ImportError, DynamicLibNotFoundError):

    def cuda(f, device=None):
        """Dummy decorator."""
        raise Exception("Error: Numba CUDA not available. Tetrahedral rendering requires CUDA.")

    cuda.jit = cuda

# --- gpu kernels ---


@cuda.jit(device=True)
def ray_single_tetra_intersect(ray_org_x, ray_org_y, ray_org_z, ray_dir_z, points, tetra_indices, zmin, zmax):
    """Compute ray length inside a single tetrahedron."""
    faces = ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 3, 2, 0))

    t_entry = -1e20
    t_exit = 1e20

    p0 = cuda.local.array(3, dtype=np.float64)
    p1 = cuda.local.array(3, dtype=np.float64)
    p2 = cuda.local.array(3, dtype=np.float64)
    other = cuda.local.array(3, dtype=np.float64)
    v1 = cuda.local.array(3, dtype=np.float64)
    v2 = cuda.local.array(3, dtype=np.float64)
    normal = cuda.local.array(3, dtype=np.float64)

    ray_org = (ray_org_x, ray_org_y, ray_org_z)

    # accelerate cases where the ray never intersects this tetra
    p0[0] = points[tetra_indices[0], 0]
    p0[1] = points[tetra_indices[0], 1]
    p0[2] = points[tetra_indices[0], 2]
    p1[0] = points[tetra_indices[1], 0]
    p1[1] = points[tetra_indices[1], 1]
    p1[2] = points[tetra_indices[1], 2]
    p2[0] = points[tetra_indices[2], 0]
    p2[1] = points[tetra_indices[2], 1]
    p2[2] = points[tetra_indices[2], 2]
    other[0] = points[tetra_indices[3], 0]
    other[1] = points[tetra_indices[3], 1]
    other[2] = points[tetra_indices[3], 2]

    if p0[2] < zmin and p1[2] < zmin and p2[2] < zmin and other[2] < zmin:
        return 0.0
    if p0[2] > zmax and p1[2] > zmax and p2[2] > zmax and other[2] > zmax:
        return 0.0

    if p0[0] < ray_org[0] and p1[0] < ray_org[0] and p2[0] < ray_org[0] and other[0] < ray_org[0]:
        return 0.0
    if p0[0] > ray_org[0] and p1[0] > ray_org[0] and p2[0] > ray_org[0] and other[0] > ray_org[0]:
        return 0.0

    if p0[1] < ray_org[1] and p1[1] < ray_org[1] and p2[1] < ray_org[1] and other[1] < ray_org[1]:
        return 0.0
    if p0[1] > ray_org[1] and p1[1] > ray_org[1] and p2[1] > ray_org[1] and other[1] > ray_org[1]:
        return 0.0

    # consider intersections with each face
    for i in range(4):
        idx0 = tetra_indices[faces[i][0]]
        idx1 = tetra_indices[faces[i][1]]
        idx2 = tetra_indices[faces[i][2]]
        idx_other = tetra_indices[faces[i][3]]

        p0[0] = points[idx0, 0]
        p0[1] = points[idx0, 1]
        p0[2] = points[idx0, 2]
        p1[0] = points[idx1, 0]
        p1[1] = points[idx1, 1]
        p1[2] = points[idx1, 2]
        p2[0] = points[idx2, 0]
        p2[1] = points[idx2, 1]
        p2[2] = points[idx2, 2]
        other[0] = points[idx_other, 0]
        other[1] = points[idx_other, 1]
        other[2] = points[idx_other, 2]

        for d in range(3):
            v1[d] = p1[d] - p0[d]
            v2[d] = p2[d] - p0[d]

        normal[0] = v1[1] * v2[2] - v1[2] * v2[1]
        normal[1] = v1[2] * v2[0] - v1[0] * v2[2]
        normal[2] = v1[0] * v2[1] - v1[1] * v2[0]

        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

        if norm < 1e-12:
            continue

        for d in range(3):
            normal[d] /= norm

        # ensure outward normal orientation
        dot_check = normal[0] * (other[0] - p0[0]) + normal[1] * (other[1] - p0[1]) + normal[2] * (other[2] - p0[2])
        if dot_check > 0.0:
            for d in range(3):
                normal[d] = -normal[d]

        # ray-plane math, optimized for orthographic z-axis ray
        denom = normal[2] * ray_dir_z
        t_num = normal[0] * (p0[0] - ray_org[0]) + normal[1] * (p0[1] - ray_org[1]) + normal[2] * (p0[2] - ray_org[2])

        if abs(denom) < 1e-12:
            # ray is parallel to the plane
            dot_outside = (
                normal[0] * (ray_org[0] - p0[0]) + normal[1] * (ray_org[1] - p0[1]) + normal[2] * (ray_org[2] - p0[2])
            )
            if dot_outside > 0.0:
                return 0.0
            continue

        t = t_num / denom

        if denom > 0.0:
            if t < t_exit:
                t_exit = t
        else:
            if t > t_entry:
                t_entry = t

    if t_entry > t_exit or t_exit < 0.0:
        return 0.0

    actual_entry = max(0.0, t_entry)
    max_dist = zmax - zmin
    actual_exit = min(max_dist, t_exit)

    if actual_entry > actual_exit:
        return 0.0

    return actual_exit - actual_entry


@cuda.jit
def _orthographic_integral_kernel(
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    npixels_x,
    npixels_y,
    voxel_size_x,
    voxel_size_y,
    ngrid,
    points,
    tetrahedra,
    quantities,
    bucket_offsets,
    bucket_lengths,
    bucket_contents,
    image_out,
):
    px, py = cuda.grid(2)

    if px < npixels_x and py < npixels_y:
        # map pixel centers to box coordinates
        ray_x = xmin + (px + 0.5) * (xmax - xmin) / npixels_x
        ray_y = ymin + (py + 0.5) * (ymax - ymin) / npixels_y
        ray_z = zmin
        ray_dir_z = 1.0

        # determine which bucket cell this ray falls in
        bucket_index_x = int((ray_x - xmin) / voxel_size_x)
        bucket_index_y = int((ray_y - ymin) / voxel_size_y)
        bucket_index = bucket_index_x + bucket_index_y * ngrid

        # consider only the tetra that overlap with (i.e. are in) this bucket
        bucket_start = bucket_offsets[bucket_index]

        los_integral = 0.0
        num_tetra = bucket_lengths[bucket_index]

        # accumulate (segment length * cell quantity) i.e. column density
        for m in range(num_tetra):
            t_index = bucket_contents[bucket_start + m]
            length = ray_single_tetra_intersect(ray_x, ray_y, ray_z, ray_dir_z, points, tetrahedra[t_index], zmin, zmax)

            if length > 0.0:
                los_integral += length * quantities[t_index]

        image_out[py, px] = los_integral


def _render_gpu(
    bounds,
    npixels,
    voxel_size,
    ngrid,
    points,
    tetra_indices,
    quantities,
    bucket_offsets,
    bucket_lengths,
    bucket_contents,
    numSplits,
):
    """
    Executes the orthographic pipeline to compute line-of-sight quantity integrals.

    This function sets up the GPU data and launches the kernel.
    """
    h_image = np.zeros((npixels[1], npixels[0]), dtype=np.float32)

    # gpu version:
    d_points = cuda.to_device(points)
    d_tetra = cuda.to_device(tetra_indices)
    d_quantities = cuda.to_device(quantities)
    d_bucket_offsets = cuda.to_device(bucket_offsets)
    # d_bucket_lengths = cuda.to_device(bucket_lengths)
    d_bucket_contents = cuda.to_device(bucket_contents)
    d_image = cuda.to_device(h_image)

    threads_per_block = (8, 8)  # 128 threads per block (1024 max)
    blocks_x = (npixels[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (npixels[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    # print(f"Launching kernel {blocks_per_grid = }, {threads_per_block = }.")
    assert numSplits == 1, "Not actually saving memory yet, need to send only a subset of bucket_contents."

    # process tetra in potentially multiple passes (to save memory)
    for i in range(numSplits):
        bucket_inds = pSplit(np.arange(bucket_lengths.size), numSplits, i)

        split_mask = np.zeros_like(bucket_lengths, dtype=np.bool_)
        split_mask[bucket_inds] = True

        d_bucket_lengths = cuda.to_device(bucket_lengths * split_mask)

        # print(f"Pass {i + 1}/{numSplits}: {split_mask.sum()} buckets.")

        _orthographic_integral_kernel[blocks_per_grid, threads_per_block](
            bounds[0],  # xmin
            bounds[1],  # xmax
            bounds[2],  # ymin
            bounds[3],  # ymax
            bounds[4],  # zmin
            bounds[5],  # zmax
            npixels[0],  # x
            npixels[1],  # y
            voxel_size[0],  # x
            voxel_size[1],  # y
            ngrid,
            d_points,
            d_tetra,
            d_quantities,
            d_bucket_offsets,
            d_bucket_lengths,
            d_bucket_contents,
            d_image,
        )

        # transfer back and accumulate
        h_image += d_image.copy_to_host()

    return h_image


# --- cpu jitted helpers ---


@njit
def _tetra_volume(p0, p1, p2, p3):
    """Compute the volume of a single tetrahedron defined by its 4 vertices."""
    v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    v3 = (p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2])

    cross = (v2[1] * v3[2] - v2[2] * v3[1], v2[2] * v3[0] - v2[0] * v3[2], v2[0] * v3[1] - v2[1] * v3[0])

    dot = v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]

    volume = abs(dot) / 6.0
    return volume


@njit
def _tetra_volumes(pos, inds):
    """Compute the volumes of multiple tetrahedra defined by their vertex positions and indices."""
    num_tetra = inds.shape[0]
    volumes = np.zeros(num_tetra, dtype=np.float64)

    for i in range(num_tetra):
        idx0, idx1, idx2, idx3 = inds[i]
        p0 = pos[idx0]
        p1 = pos[idx1]
        p2 = pos[idx2]
        p3 = pos[idx3]
        volumes[i] = _tetra_volume(p0, p1, p2, p3)

    return volumes


@njit
def _build_2d_grid(points, tets_indices, nx, ny, grid_min, grid_max, voxel_size):
    """
    Builds the 2D uniform grid bucket structures on the CPU.
    """
    num_tets = tets_indices.shape[0]

    # step 1: count number of tetra that overlap each bucket (2D cell column, z-axis aligned)
    grid_counts = np.zeros(nx * ny, dtype=np.int32)
    mask = np.zeros(num_tets, dtype=np.bool_)

    for i in range(num_tets):
        # four point indices for this tetra
        idx0 = tets_indices[i, 0]
        idx1 = tets_indices[i, 1]
        idx2 = tets_indices[i, 2]
        idx3 = tets_indices[i, 3]

        # spatial coordinates of these vertices
        x0, y0, z0 = points[idx0, 0], points[idx0, 1], points[idx0, 2]
        x1, y1, z1 = points[idx1, 0], points[idx1, 1], points[idx1, 2]
        x2, y2, z2 = points[idx2, 0], points[idx2, 1], points[idx2, 2]
        x3, y3, z3 = points[idx3, 0], points[idx3, 1], points[idx3, 2]

        # overlap grid in z-direction?
        min_z = min(z0, z1, z2, z3)
        max_z = max(z0, z1, z2, z3)

        if max_z < grid_min[2] or min_z > grid_max[2]:
            continue

        # calculate AABB bounding box (2d)
        min_x = min(x0, x1, x2, x3)
        max_x = max(x0, x1, x2, x3)
        min_y = min(y0, y1, y2, y3)
        max_y = max(y0, y1, y2, y3)

        start_x = max(0, int((min_x - grid_min[0]) / voxel_size[0]))
        end_x = min(nx - 1, int((max_x - grid_min[0]) / voxel_size[0]))
        start_y = max(0, int((min_y - grid_min[1]) / voxel_size[1]))
        end_y = min(ny - 1, int((max_y - grid_min[1]) / voxel_size[1]))

        # add this tetra to the count of every bucket cell it overlaps
        for g_y in range(start_y, end_y + 1):
            for g_x in range(start_x, end_x + 1):
                cell_1d = g_x + g_y * nx
                grid_counts[cell_1d] += 1
                mask[i] = True

    # cumulative sum for offsets
    grid_offsets = np.zeros(nx * ny + 1, dtype=np.int64)
    grid_offsets[1:] = np.cumsum(grid_counts)

    # subset tetra to those that overlap at least one bucket (to save memory)
    print("Fraction of tetra that overlap grid:", mask.sum() / num_tets)

    tets_indices = tets_indices[mask]
    num_tets = tets_indices.shape[0]

    # step 2: record tetra indices in each bucket
    working_counts = np.zeros(nx * ny, dtype=np.int32)
    bucket_content = np.zeros(grid_offsets[-1], dtype=np.int32)

    # n_buckets = np.zeros(num_tets, dtype=np.int32)

    for i in range(num_tets):
        idx0 = tets_indices[i, 0]
        idx1 = tets_indices[i, 1]
        idx2 = tets_indices[i, 2]
        idx3 = tets_indices[i, 3]

        x0, y0, z0 = points[idx0, 0], points[idx0, 1], points[idx0, 2]
        x1, y1, z1 = points[idx1, 0], points[idx1, 1], points[idx1, 2]
        x2, y2, z2 = points[idx2, 0], points[idx2, 1], points[idx2, 2]
        x3, y3, z3 = points[idx3, 0], points[idx3, 1], points[idx3, 2]

        # overlap grid in z-direction?
        min_z = min(z0, z1, z2, z3)
        max_z = max(z0, z1, z2, z3)

        if max_z < grid_min[2] or min_z > grid_max[2]:
            continue

        # AABB bounding box (2d)
        min_x = min(x0, x1, x2, x3)
        max_x = max(x0, x1, x2, x3)
        min_y = min(y0, y1, y2, y3)
        max_y = max(y0, y1, y2, y3)

        start_x = max(0, int((min_x - grid_min[0]) / voxel_size[0]))
        end_x = min(nx - 1, int((max_x - grid_min[0]) / voxel_size[0]))
        start_y = max(0, int((min_y - grid_min[1]) / voxel_size[1]))
        end_y = min(ny - 1, int((max_y - grid_min[1]) / voxel_size[1]))

        # todo: can count how many buckets this tetra is added to
        # if it's a large number >> 1, we could instead add it (only) to a special bucket, that all rays must consider
        for g_y in range(start_y, end_y + 1):
            for g_x in range(start_x, end_x + 1):
                cell_1d = g_x + g_y * nx
                slot = working_counts[cell_1d]
                working_counts[cell_1d] += 1
                bucket_content[grid_offsets[cell_1d] + slot] = i
                # n_buckets[i] += 1
                # mask[i] = True

    # print("Maximum number of buckets for a single tetra: ", n_buckets.max())
    return grid_offsets, grid_counts, bucket_content, mask


def _get_tetra(sim):
    """Helper function for below, return the tetra based on the simulation initial conditions."""
    sim_ics = sim.copy()
    sim_ics.setSnap("ics")

    # load tetra, check memory cache (for time interp vis) and then disk cache
    cache_key = "snap%s_dm_tetra_inds" % sim.snap
    if cache_key in sim.data:
        tetra_inds = sim.data[cache_key]
        # print(" loaded from memory cache: [%s]" % cache_key)
    else:
        # proceed to disk cache
        cacheFile = sim.cachePath + "sorted_dm_tetra_ics.hdf5"
        if not isfile(cacheFile):
            # make new
            dmIDs_ics = sim_ics.snapshotSubsetP("dm", "ids")

            # sort
            sort_inds = np.argsort(dmIDs_ics)

            # load positions
            pos_ics = sim_ics.dm("pos")[sort_inds]

            # delaunay triangualation
            start_time = time.time()

            mesh = Delaunay(pos_ics)  # , qhull_options="QJ")
            tetra_inds = mesh.simplices

            print(" making cache: delaunay done, took [%g] sec." % (time.time() - start_time))

            if mesh.coplanar.size > 0:
                print(f"WARNING: Delaunay triangulation found {mesh.coplanar.size} coplanar points.")

            with h5py.File(cacheFile, "w") as f:
                f["tetra_inds"] = tetra_inds

            dmIDs_ics = None
            pos_ics = None
        else:
            # load from cache
            with h5py.File(cacheFile, "r") as f:
                tetra_inds = f["tetra_inds"][:]

        # add to memory cache
        sim.data[cache_key] = tetra_inds

    # load DM IDs at this snapshot (memory caching for time interp vis)
    cache_key = "snap%s_dm_IDs" % sim.snap
    if cache_key in sim.data:
        dm_ids = sim.data[cache_key]
        # print(" loaded from memory cache: [%s]" % cache_key)
    else:
        dm_ids = sim.snapshotSubsetP("dm", "ids")

    cache_key = "snap%s_dm_IDs_sort_inds" % sim.snap
    if cache_key in sim.data:
        sort_inds = sim.data[cache_key]
        # print(" loaded from memory cache: [%s]" % cache_key)
    else:
        sort_inds = np.argsort(dm_ids)
        sim.data[cache_key] = sort_inds

    # load DM positions at this snapshot and shuffle into sorted ID order
    # note: could be 'loading' time-interpolated positions for vis
    pos = sim.dm("pos")[sort_inds]

    # tetra connectivity remains unchanged
    return pos, tetra_inds


def render_tetra(sim, bounds, npixels, rotMatrix=None, rotCenter=None, verbose=True):
    """Main renderer."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # load
    start_time = time.time()

    points_data, tetra_data = _get_tetra(sim)

    # compute volumes of all tetra
    volumes = _tetra_volumes(points_data, tetra_data)

    # convert to mass density
    quant_data = sim.dmParticleMass / 4 / volumes  # code units

    if verbose:
        print(f"Load and volumes of [{volumes.size}] tetra in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()

    # rotation requested?
    if rotMatrix is not None:
        points_data, _ = rotateCoordinateArray(sim, points_data, rotMatrix, rotCenter)

    # estimate number of tetra per bucket cell
    w = np.where(
        (points_data[:, 0] >= xmin)
        & (points_data[:, 0] <= xmax)
        & (points_data[:, 1] >= ymin)
        & (points_data[:, 1] <= ymax)
    )
    N_pts = len(w[0])
    ngrid = int(math.sqrt(N_pts / 10000))  # heuristic
    ngrid = np.clip(ngrid, 10, 100)  # sanity limits

    # build acceleration grid
    grid_min = np.array([xmin, ymin, zmin], dtype=np.float32)
    grid_max = np.array([xmax, ymax, zmax], dtype=np.float32)
    voxel_size = (grid_max[0:2] - grid_min[0:2]) / np.array([ngrid, ngrid], dtype=np.float32)

    bucket_offsets, bucket_lengths, bucket_contents, mask = _build_2d_grid(
        points_data, tetra_data, ngrid, ngrid, grid_min, grid_max, voxel_size
    )

    # subset tetra to those that overlap grid
    tetra_data = tetra_data[mask]
    quant_data = quant_data[mask]

    # estimate total memory that will be needed on the GPU
    total_bytes_needed = (
        points_data.nbytes
        + tetra_data.nbytes
        + quant_data.nbytes
        + bucket_offsets.nbytes
        + bucket_lengths.nbytes
        + bucket_contents.nbytes
        + (npixels[0] * npixels[1] * 4)  # output image, float32
    )

    # check available GPU memory, decide if we need to split the work
    context = cuda.current_context()
    free_bytes, total_bytes = context.get_memory_info()

    numSplits = np.clip(int(np.ceil(total_bytes_needed * 1.1 / free_bytes)), 1, 64)  # sanity limits

    if verbose:
        print(f"Bucketing [{ngrid}x{ngrid}] completed in {time.time() - start_time:.2f} seconds.")
        print(f"Bucket <size>: [{bucket_lengths.mean():.1f}] dupe_frac = [{bucket_lengths.sum() / volumes.size:.2f}]")

        print(f"Total GPU memory needed: {total_bytes_needed / 1024**3:.2f} GB")
        print(f"GPU memory available: {free_bytes / 1024**3:.2f} GB free out of {total_bytes / 1024**3:.2f} GB total")

        start_time = time.time()

    # render (GPU)
    im = _render_gpu(
        [xmin, xmax, ymin, ymax, zmin, zmax],
        npixels,
        voxel_size,
        ngrid,
        points_data,
        tetra_data,
        quant_data,
        bucket_offsets,
        bucket_lengths,
        bucket_contents,
        numSplits,
    )

    if verbose:
        print(f"Ray-tracing completed in {time.time() - start_time:.2f} seconds.")

    return im


def render_test(sim):
    """Test render and plot."""
    import matplotlib.pyplot as plt

    # config
    size = 20.0  # code units
    cen = sim.halo(0)["GroupPos"]
    xmin, xmax = cen[0] - size, cen[0] + size
    ymin, ymax = cen[1] - size / 2, cen[1] + size / 2
    zmin, zmax = cen[2] - size, cen[2] + size
    np_x, np_y = 512, 512  # 256, 512, 1024px take 13, 28, 97 sec (h219612_L14)

    saveFilename = f"cache_tetra_{sim.simName}_{sim.snap}_{size}_{np_x}x{np_y}.hdf5"
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            im = f["image"][:]
        print(f"Loaded existing image from {saveFilename}")
    else:
        # render
        im = render_tetra(sim, [xmin, xmax, ymin, ymax, zmin, zmax], [np_x, np_y], verbose=True)

        # save
        with h5py.File(saveFilename, "w") as f:
            f["image"] = im
        print(f"Saved image to {saveFilename}")

    # unit conversions, [code mass / code length^2] -> [Msun / kpc^2]
    im = sim.units.codeColDensToPhys(im, msunKpc2=True)
    im = logZeroMin(im)

    vmm = np.percentile(im, [1, 99.9])

    # plot
    fig, ax = plt.subplots()

    plt.imshow(im, extent=[xmin, xmax, ymin, ymax], vmin=vmm[0], vmax=vmm[1], cmap="inferno", origin="lower")

    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(r"Dark Matter Column Density [log M$_{\rm sun}$ kpc$^{-2}$]")

    fig.savefig(f"tetra_{sim.simName}_{sim.snap}_{size}_{np_x}x{np_y}.pdf")
    plt.close(fig)
