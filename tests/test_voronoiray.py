""" Test and benchmark util.voronoiRay. """
import time

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.voronoi import loadGlobalVPPP
from temet.util.voronoiRay import (
    buildFullTree,
    rayTrace,
    trace_ray_through_voronoi_mesh_treebased,
    trace_ray_through_voronoi_mesh_with_connectivity,
)

@pytest.mark.requires_data
def test_benchmark_voronoiray(benchmark, plot=False):
    """ Benchmark: run a large number of rays using the threaded-code. """
    # config
    sP = sim(run='tng50-4', redshift=0.5)
    ray_dir = [0.0, 0.0, 1.0]
    n_rays = 100

    # loop over:
    total_dls = [5000.0, 10000.0] # [1000.0, 5000.0]
    nThreads = [1, 2, 4] #[1,1,2,4,8,16,32,64]

    # load global cell positions
    print('Loading...')
    pos = sP.snapshotSubsetP('gas', 'pos') # code

    quant = np.zeros( pos.shape[0], dtype='float32') + 0.5
    quant2 = np.zeros( pos.shape[0], dtype='float32') + 2.0
    mode = 'full' #'quant_weighted_mean' #'count'

    # init random number generator, create rays
    rng = np.random.default_rng(424242)
    ray_pos = rng.uniform(low=0.0, high=sP.boxSize, size=(n_rays,3))

    # build tree
    tree = buildFullTree(pos,sP.boxSize,pos.dtype,verbose=True)

    # start scaling plot
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    # closure for compute kernel: loop over requested path lengths
    def _run():
        for total_dl in total_dls:
            times = []

            # loop over requested numbers of threads
            for nThread in nThreads:
                start_time = time.time()

                # run and time
                result = rayTrace(sP,ray_pos,ray_dir,total_dl,pos,quant=quant,quant2=quant2,mode=mode,nThreads=nThread,tree=tree)

                total_time = (time.time() - start_time)
                print(f'[{nThread = :2d}] [{total_dl = :.1f}] ray-tracing total {total_time = :.3f} sec.')
                times.append(total_time)

                if mode == 'full':
                    # split output, sanity check
                    offsets, lengths, r_dx, r_ind = result
                    assert lengths.sum() == r_dx.size

            # add to plot
            if plot:
                ax.set_xlabel('Number of Threads')
                ax.set_ylabel('Time [sec]')
                ax.plot(nThreads[1:], times[1:], 'o-', label='dl = %d' % total_dl)

    # run benchmark
    benchmark(_run)

    # finish scaling plot
    if plot:
        ax.legend(loc='upper right')
        fig.savefig('benchmark_test_raytracing_%s.pdf' % mode)

        ax.set_yscale('log')
        fig.savefig('benchmark_test_raytracing_%s_log.pdf' % mode)
        plt.close(fig)

@pytest.mark.requires_data
def test_voronoiray_treebased(compare=True):
    """ Test tree-based Voronoi ray-tracing via comparison with explicit mesh results.

    Run a large number of rays through the (fullbox) Voronoi mesh, in each case comparing the
    results from pre-computed vs. tree-based approaches, for correctness (and speed).

    Args:
      compare (bool): if True, then run both methods and assert they have the same return.
        Otherwise, run only the tree-based method for timing.

    Returns:
      None
    """
    # config
    sP = sim(run='tng50-4', redshift=0.5)

    projAxis = 2 # z, to simplify vellos for now

    num_rays = 100
    verify = False

    # load global cell positions
    cell_pos = sP.snapshotSubsetP('gas', 'pos') # code

    # construct neighbor tree
    tree = buildFullTree(cell_pos, boxSizeSim=sP.boxSize, treePrec=cell_pos.dtype, verbose=True)
    NextNode, length, center, sibling, nextnode = tree

    # load mesh neighbor connectivity
    if compare:
        num_ngb, ngb_inds, offset_ngb = loadGlobalVPPP(sP)

    # init random number generator and counters
    rng = np.random.default_rng(424242)

    N_intersects = 0
    total_pathlength = 0.0
    time_a = 0.0
    time_b = 0.0

    # allocate (internal ray arrays)
    #max_steps = 10000
    #master_dx = np.zeros(max_steps, dtype=np.float32) # pathlength for each ray segment
    #master_ind = np.zeros(max_steps, dtype=np.int64) # index
    #prev_cell_inds = np.zeros(max_steps, dtype=np.int64) - 1
    #prev_cell_cen = np.zeros(max_steps, dtype=np.float32)

    for i in range(num_rays):
        # ray direction
        ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
        ray_dir[projAxis] = 1.0

        # ray starting position and length (random)
        ray_pos = rng.uniform(low=0.0, high=sP.boxSize, size=3)

        total_dl = rng.uniform(low=sP.boxSize/100, high=sP.boxSize/2)

        print(f'[{i:3d}] {ray_pos = } {total_dl = }')

        # (A) ray-trace with precomputed connectivity method
        start_time = time.time()

        if compare:
            master_dx, master_ind = \
              trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, num_ngb, ngb_inds, offset_ngb, 
                                                               ray_pos, ray_dir, total_dl, sP.boxSize, 
                                                               debug=0, verify=verify, fof_scope_mesh=False)

        time_a += (time.time() - start_time) # accumulate

        # (B) ray-trace with tree-based method
        start_time = time.time()

        master_dx2, master_ind2 = \
          trace_ray_through_voronoi_mesh_treebased(cell_pos, NextNode, length, center, sibling, nextnode, 
                                                   ray_pos, ray_dir, total_dl, sP.boxSize, 
                                                   #master_dx, master_ind, prev_cell_inds, prev_cell_cen, 
                                                   debug=0, verify=verify)

        time_b += (time.time() - start_time)

        # compare
        N_intersects += master_dx2.size
        total_pathlength += total_dl

        if compare:
            assert np.allclose(master_dx,master_dx2)
            assert np.array_equal(master_ind,master_ind2)

    # stats
    avg_intersections = N_intersects / sP.units.codeLengthToMpc(total_pathlength) # per pMpc
    avg_time_a = time_a / num_rays / N_intersects # per intersection (w/ connectivity)
    avg_time_b = time_b / num_rays / N_intersects # per intersection (tree-based)

    time_1000_crossings_a = avg_time_a * avg_intersections * sP.units.codeLengthToMpc(sP.boxSize) * 1000
    time_1000_crossings_b = avg_time_b * avg_intersections * sP.units.codeLengthToMpc(sP.boxSize) * 1000

    print(f'For {num_rays = }, have {N_intersects = } and {total_pathlength = :.2f}')
    print(f'Time per ray, w/ connectivity: [{time_a/num_rays:.2f}] sec, tree-based: [{time_b/num_rays:.2f}] sec')
    print(f'Mean intersections per pMpc: [{avg_intersections:.2f}]')
    print(f'Time for 1000x full box crossings: [{time_1000_crossings_a:.2f}] vs [{time_1000_crossings_b:.2f}] sec')
