"""
Algorithms and methods related to ray-tracing through a Voronoi mesh.
"""
import numpy as np
import time
import threading
from numba import jit

from ..util.helper import periodicDistsN, pSplitRange
from ..util.sphMap import _NEAREST_POS
from ..util.voronoi import loadSingleHaloVPPP, loadGlobalVPPP
from ..util.treeSearch import buildFullTree, _treeSearchNearest, _treeSearchNearestSingle

@jit(nopython=True, nogil=True, cache=True)
def _periodic_wrap_point(pos, pos_ref, boxSize, boxHalf):
    """ If pos is more than a half-box away from pos_ref, wrap it into the same octant.

    Args:
      pos (float[3]): x,y,z coordinates to wrap. Modified in place.
      pos_ref (float[3]): x,y,z coordinates to use as reference point.
      boxSize (float): the simulation box size, for periodic boundaries.
      boxHalf (float): half of the simulation box size, for periodic boundaries.

    Return:
      None
    """
    dx = pos[0] - pos_ref[0]
    dy = pos[1] - pos_ref[1]
    dz = pos[2] - pos_ref[2]

    if dx > boxHalf:
        pos[0] -= boxSize
    if dx < -boxHalf:
        pos[0] += boxSize
    if dy > boxHalf:
        pos[1] -= boxSize
    if dy < -boxHalf:
        pos[1] += boxSize
    if dz > boxHalf:
        pos[2] -= boxSize
    if dz < -boxHalf:
        pos[2] += boxSize

    return

@jit(nopython=True, nogil=True, cache=True)
def trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, num_ngb, ngb_inds, offset_ngb,
                                   ray_pos_in, ray_dir, total_dl, boxSize, debug, verify, fof_scope_mesh):
    """ For a single ray, specified by its starting location, direction, and length, ray-trace through 
    a Voronoi mesh as specified by the pre-computed natural neighbor connectivity information.

    Args:
      cell_pos (ndarray[float]): Voronoi cell center positions [N,3].
      num_ngb (array[int]): Voronoi mesh connectivity, first return of :func:`util.voronoi.loadSingleHaloVPPP` 
        or :func:`util.voronoi.loadGlobalVPPP` .
      ngb_inds (array[int]): Voronoi mesh connectivity, second return of :func:`util.voronoi.loadSingleHaloVPPP` 
        or :func:`util.voronoi.loadGlobalVPPP` .
      offset_ngb (array[int]): Voronoi mesh connectivity, third return of :func:`util.voronoi.loadSingleHaloVPPP` 
        or :func:`util.voronoi.loadGlobalVPPP` .
      ray_pos_in (float[3]): x,y,z coordinates of ray starting position.
      ray_dir (float[3]): normalized unit vector specifying ray direction.
      total_dl (float): pathlength to integrate ray (code units i.e. same as cell_pos).
      boxSize (float): simulation box size, for periodic boundaries (code units).
      debug (bool): if True, >=1, or >=2, print increasing verbose debugging information (disabled for @jit).
      verify (bool): if True, do brute-force distance verification every step of parent Voronoi cell.
      fof_scope_mesh (bool): if True, indicate that we have loaded and are working with a fof-scope set of cell 
        data and mesh connectivity, i.e. not a correct not periodic mesh at the edges.

    Return:
      a 2-tuple composed of

      - **dx** (ndarray[float]): per-cell path length, in order, for each intersected cell (code/input units).
      - **ind** (ndarray[float]): per-cell index, in order, for each intersected cell.
    """

    # path length accumulated
    boxHalf = boxSize / 2
    ray_pos = ray_pos_in.copy()

    dl = 0.0
    n_step = 0

    # locate starting cell
    dists = periodicDistsN(ray_pos, cell_pos, boxSize)
    cur_cell_ind = np.where(dists == dists.min())[0][0]
    prev_cell_ind = -1
    
    #if debug: print(f'Starting cell index [{cur_cell_ind}] at distance = {dists[cur_cell_ind]:.2f} ckpc/h, {total_dl = :.3f}.')

    # allocate
    max_steps = 10000

    master_dx = np.zeros(max_steps, dtype=np.float32) # pathlength for each ray segment
    master_ind = np.zeros(max_steps, dtype=np.int64) # index

    # while total dl does not exceed request, start tracing through mesh
    while 1:
        # current Voronoi cell
        cur_cell_pos = cell_pos[cur_cell_ind].copy()
        _periodic_wrap_point(cur_cell_pos, ray_pos, boxSize, boxHalf)
        
        #if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

        local_dl = np.inf
        next_ngb_index = -1

        if verify:
            dists = periodicDistsN(ray_pos, cell_pos, boxSize)
            mindist_cell_ind = np.where(dists == dists.min())[0][0]
            # due to round-off, answer should be ambiguous between previous and current cell (we sit on the face)
            if mindist_cell_ind not in [prev_cell_ind,cur_cell_ind]:
                if fof_scope_mesh:
                    dist_to_halo_cen = 2.0 #sP.periodicDists(ray_pos, halo['GroupPos']) / halo['Group_R_Crit200']
                    # note: still fail if we start too early i.e. before fof-scope!
                    assert dist_to_halo_cen > 1.0 and dl > total_dl/2 # otherwise check
                    #if debug: print(' -- NOTE: Termination! Leaving fof-scope mesh.')
                    break
                else:
                    assert 0 # should not occur

        # loop over all natural neighbors
        for i in range(num_ngb[cur_cell_ind]):
            # neighbor properties
            ngb_index = ngb_inds[offset_ngb[cur_cell_ind]+i]
            ngb_pos = cell_pos[ngb_index].copy()
            _periodic_wrap_point(ngb_pos, ray_pos, boxSize, boxHalf)

            if ngb_index == -1: # outside of fof-scope mesh
                assert fof_scope_mesh # otherwise should not occur
                #if debug > 1: print(f' [{i:2d} of {num_ngb[cur_cell_ind]:2d}] with {ngb_index = } skip')
                continue

            #if debug > 1: print(f' [{i:2d} of {num_ngb[cur_cell_ind]:2d}] with {ngb_index = } and {ngb_pos = }')

            # edge midpoint, i.e. a point on the Voronoi face plane shared with this neighbor
            m = 0.5 * (ngb_pos + cur_cell_pos)

            # the vector from the current ray position to m
            c = m - ray_pos

            # the vector from the current cell to the neighbor, which is a normal vector to the face plane
            q = ngb_pos - cur_cell_pos

            # test intersection of a ray and a plane. because the dot product of two perpendicular vectors is
            # zero, we can write (p-m).n = 0 for some point p on the face, because (p-m) is a vector on the face.
            # then, we have the parametric ray equation ray_pos+ray_dir*s = p for some scalar distance s. If the 
            # ray and plane intersect, this point p is the same in both equations, so substituting and 
            # re-arranging for s we solve for s = c.q / (ray_dir.q)
            cdotq = np.sum(c * q)
            ddotq = np.sum(ray_dir * q)

            # s gives the point where the ray (ray_pos + s*ray_dir)
            # intersects the plane perpendicular to q containing c, i.e. the Voronoi face with this neighbor
            if cdotq > 0:
                # standard case, ray_pos is inside the cell, calculate length to intersection
                s = cdotq / ddotq
            else:
                # point is on the wrong side of face (i.e. outside), could be due to numerical roundoff error
                # (if distance to the face is ~eps), or because we are not in the cell we think we are
                # (if distance to the face is large)
                if ddotq > 0:
                    # direciton is away from cell, so it was supposed to have intersected this face?
                    # set s=0 i.e. there is no local pathlength
                    #if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} > 0, direction is out of cell (set next, local_dl=0)')
                    s = 0
                    assert 0 # check when/how this really happens
                else:
                    # direction is into the cell, so it must have entered the cell through this face (ignore)
                    #if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} < 0, direction is into cell (ignore)')
                    s = np.inf

                # if np.abs(ddotq) < eps, then the plane and ray are parallel
                #   - if the ray and face perfectly coincide, there is an infinity of intersection solutions
                #   - or, if the ray is off the face, there is no intersection
                # either way, we treat this as a non-intersection
                if np.abs(ddotq) < 1e-10:
                    assert 0 # check when/how this really happens
                
            if s >= 0 and s < local_dl:
                # we have a valid intersection, and it is closer than all previous intersections, so mark this
                # neighbor as the new best candidate for the exit face
                next_ngb_index = ngb_index
                local_dl = s
                #if debug > 1: print(f'  -- new next neighbor: [{ngb_index}] with {local_dl = }')

        # have we exceeded the requested total pathlength?
        assert local_dl > 0
        assert np.isfinite(local_dl)

        if dl+local_dl >= total_dl:
            # integrate final cell partially, to end of ray
            local_dl = total_dl - dl

        # calculate local pathlength, update ray position
        dl += local_dl
        ray_pos += ray_dir*local_dl

        # wrap ray_pos if it has left the box
        for i in range(3):
            ray_pos[i] = _NEAREST_POS(ray_pos[i], boxSize)

        #if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, next cell index = {next_ngb_index}')

        # accumulate (all code units and/or same units as input)
        master_dx[n_step] = local_dl
        master_ind[n_step] = cur_cell_ind
        n_step += 1

        if dl >= (total_dl - 1e-10):
            # finished
            #if debug > 1: print('  -- reached total pathlength, terminating.')
            break

        # do we have a next valid neighbor? if not, exit
        if next_ngb_index == -1:
            assert fof_scope_mesh # otherwise should not happen
            #if debug > 1: print('  -- next neighbor is outside FoF scope, terminating.')
            break

        # move to next cell
        prev_cell_ind = cur_cell_ind
        cur_cell_ind = next_ngb_index

    # reduce arrays to used size
    master_dx = master_dx[0:n_step]
    master_ind = master_ind[0:n_step]

    return master_dx, master_ind

@jit(nopython=True, nogil=True, cache=True)
def trace_ray_through_voronoi_mesh_treebased(cell_pos, NextNode, length, center, sibling, nextnode, 
                                             ray_pos_in, ray_dir, total_dl, boxSize, 
                                             #master_dx, master_ind, prev_cell_inds, prev_cell_cen, 
                                             debug, verify):
    """ For a single ray, specified by its starting location, direction, and length, ray-trace through 
    a Voronoi mesh using neighbor searches in a pre-computed tree.

    Args:
      cell_pos (ndarray[float]): Voronoi cell center positions [N,3].
      NextNode (array[int]): neighbor tree, first return of :func:`util.treeSearch.buildFullTree`.
      length (array[int]): neighbor tree, second return of :func:`util.treeSearch.buildFullTree`.
      center (array[int]): neighbor tree, third return of :func:`util.treeSearch.buildFullTree`.
      sibling (array[int]): neighbor tree, fourth return of :func:`util.treeSearch.buildFullTree`.
      nextnode (array[int]): neighbor tree, fifth return of :func:`util.treeSearch.buildFullTree`.
      ray_pos_in (float[3]): x,y,z coordinates of ray starting position.
      ray_dir (float[3]): normalized unit vector specifying ray direction.
      total_dl (float): pathlength to integrate ray (code units i.e. same as cell_pos).
      boxSize (float): simulation box size, for periodic boundaries (code units).
      debug (bool): if True, >=1, or >=2, print increasing verbose debugging information (disabled for @jit).
      verify (bool): if True, do brute-force distance verification every step of parent Voronoi cell.

    Return:
      a 2-tuple composed of

      - **dx** (ndarray[float]): per-cell path length, in order, for each intersected cell (code/input units).
      - **ind** (ndarray[float]): per-cell index, in order, for each intersected cell.
    """
    abs_tol = 0.01

    # path length accumulated
    ray_pos = ray_pos_in.copy()
    dl = 0.0
    n_step = 0

    boxHalf = boxSize / 2

    # allocate
    max_steps = 10000
    
    master_dx = np.zeros(max_steps, dtype=np.float32) # pathlength for each ray segment
    master_ind = np.zeros(max_steps, dtype=np.int64) # index

    # for bisection stack: indices of previous failed candidate cell(s)
    prev_cell_inds = np.zeros(max_steps, dtype=np.int64) - 1
    prev_cell_cen = np.zeros(max_steps, dtype=np.float32)

    num_prev_inds = 0

    # locate starting cell
    cur_cell_ind, h_guess = _treeSearchNearestSingle(ray_pos,cell_pos,boxSize,NextNode,length,center,sibling,nextnode,h=1.0)

    # where will we terminate ray, globally?
    ray_end = ray_pos + ray_dir * total_dl

    end_cell_ind, h_guess = _treeSearchNearestSingle(ray_end,cell_pos,boxSize,NextNode,length,center,sibling,nextnode,h_guess)
    end_cell_pos = cell_pos[end_cell_ind]

    # note: by definition end_cell_ind == cur_cell_ind if total_dl = BoxSize
    #if debug: print(f'Starting cell index [{cur_cell_ind}], ending cell index [{end_cell_ind}], {total_dl = :.3f}.')

    if verify:
        # verify start
        dists = periodicDistsN(ray_pos, cell_pos, boxSize)
        mindist_cell_ind = np.where(dists == dists.min())[0][0]
        assert mindist_cell_ind == cur_cell_ind
        # verify end
        dists = periodicDistsN(ray_end, cell_pos, boxSize)
        mindist_cell_ind = np.where(dists == dists.min())[0][0]
        assert mindist_cell_ind == end_cell_ind

    prev_cell_ind = -1 # for verify only

    # loop while still intersecting cells
    finished = False

    while not finished:
        # current Voronoi cell
        cur_cell_pos = cell_pos[cur_cell_ind]

        #if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

        #if n_step > 10:
        #    import pdb; pdb.set_trace()

        if verify:
            dists = periodicDistsN(ray_pos, cell_pos, boxSize)
            mindist_cell_ind = np.where(dists == dists.min())[0][0]
            # due to round-off, answer should be ambiguous between previous and current cell (we sit on the face)
            assert mindist_cell_ind in [prev_cell_ind,cur_cell_ind]

        # ending target for this segment is the global ray endpoint, unless we have previously failed a 
        # bisection and thus have a closer guess
        end_cell_local_ind = -1

        raylength_left = 0.0
        raylength_right = total_dl - dl # total remaining length

        # clip maximum rightward step to a quarter of the boxsize to avoid distanat periodic issues
        if raylength_right > 0.25 * boxSize:
            raylength_right = 0.25 * boxSize

        # bisection acceleration: from the last ray position, we can use the 'closest' failed 
        # distance as a (closer) starting point
        i = 0
        while i < num_prev_inds:
            if prev_cell_inds[i] == cur_cell_ind:
                # avoid self, shift all following entries to overwrite this one
                #if debug > 1: print(f' -- remove self [{i}] [{prev_cell_inds[i]}] from prev_cell_inds stack!')
                for j in range(i,num_prev_inds):
                    prev_cell_inds[j] = prev_cell_inds[j+1]
                num_prev_inds -= 1
                continue
            i += 1

        if num_prev_inds > 0 and prev_cell_inds[num_prev_inds-1] >= 0:
            # set first ray_end_local to known (shorter) result
            raylength_right = 2 * prev_cell_cen[num_prev_inds-1]
            end_cell_local_ind = prev_cell_inds[num_prev_inds-1]

            #if debug > 1: print(' -- prev_cell_inds stack: ', prev_cell_inds[0:num_prev_inds+2])
            #if debug > 1: print(' -- prev_cell_cen stack: ', prev_cell_cen[0:num_prev_inds+2])
            # pop from stack
            prev_cell_cen[num_prev_inds-1] = 0.0 # for safety only
            prev_cell_inds[num_prev_inds-1] = -1 # for safety only
            num_prev_inds -= 1
            #if debug: print(f' -- set {raylength_right = } from {num_prev_inds = } index {end_cell_local_ind}')

        assert raylength_right > 0.0 # otherwise used an empty prev_cell_cen value?

        # while the cell containing the end of the segment is not a natural neighbor of the current cell
        local_dl = np.inf

        for n_iter in range(1000):
            # set distance along ray as midpoint between bracketing
            assert n_iter < 500 and (raylength_right - raylength_left) > 1e-10 # otherwise failure

            # new test position along ray (ray_end_local can be outside box, which is ok for __treeSearchNearestSingle)
            raylength_cen = 0.5 * (raylength_left + raylength_right)

            ray_end_local = ray_pos + ray_dir * raylength_cen

            #if debug > 1: print(f' ({n_iter:2d}) L+R midpoint = {(raylength_left+raylength_right)*0.5:.5f}')

            # locate parent cell of this point
            if n_iter > 0 or end_cell_local_ind == -1:
                # only skip this tree-research, possibly, on first iteration if we have a saved index
                # from a previous bisection
                end_cell_local_ind, dist_end_local = _treeSearchNearestSingle(ray_end_local,cell_pos,boxSize,NextNode,length,center,sibling,nextnode,h_guess)

            # is this parent the same as the current cell? then we have skipped over the neighbor (from the right)
            # or, we have yet to exit the current cell (from the left)
            if end_cell_local_ind == cur_cell_ind:
                if (raylength_right - raylength_left) > 1e-5:
                    # skipped: modify starting point (bisection), and continue loop
                    raylength_left = (raylength_right + raylength_left) * 0.5

                    #if debug > 1: print(f'  !! still in current cell, move left, new [L={raylength_left:.4f} R={raylength_right:.4f}] and re-search')
                    continue
                else:
                    # yet to exit: we are very close to the face
                    raylength_right += abs_tol
                    #if debug > 1: print(f'  !! still in current cell, add abs_tol to R, new [L={raylength_left:.4f} R={raylength_right:.4f}] and re-search')
                    continue

            # position of parent cell of this point
            end_cell_pos_local = cell_pos[end_cell_local_ind].copy()

            _periodic_wrap_point(end_cell_pos_local, cur_cell_pos, boxSize, boxHalf)

            if verify:
                dists = periodicDistsN(ray_end_local, cell_pos, boxSize)
                min_index_verify = np.where(dists == dists.min())[0][0]
                assert min_index_verify == end_cell_local_ind

            # edge midpoint, i.e. a point on the Voronoi face plane shared with this neighbor, if the 
            # current and final cells are actually natural neighbors
            m = 0.5 * (end_cell_pos_local + cur_cell_pos)

            # the vector from the current ray position to m
            ray_pos_wrapped = ray_pos.copy()
            _periodic_wrap_point(ray_pos_wrapped, cur_cell_pos, boxSize, boxHalf)

            c = m - ray_pos_wrapped

            # the vector from the current cell to the neighbor, which is a normal vector to the face plane
            q = end_cell_pos_local - cur_cell_pos

            # test intersection of a ray and a plane. because the dot product of two perpendicular vectors is
            # zero, we can write (p-m).n = 0 for some point p on the face, because (p-m) is a vector on the face.
            # then, we have the parametric ray equation ray_pos+ray_dir*s = p for some scalar distance s. If the 
            # ray and plane intersect, this point p is the same in both equations, so substituting and 
            # re-arranging for s we solve for s = c.q / (ray_dir.q)
            cdotq = np.sum(c * q)
            ddotq = np.sum(ray_dir * q)

            # s gives the point where the ray (ray_pos + s*ray_dir)
            # intersects the plane perpendicular to q containing c, i.e. the Voronoi face with this neighbor
            if cdotq > 0:
                # standard case, ray_pos is inside the cell, calculate length to intersection
                s = cdotq / ddotq
            else:
                # point is on the wrong side of face (i.e. outside), could be due to numerical roundoff error
                # (if distance to the face is ~eps), or because we are not in the cell we think we are
                # (if distance to the face is large)
                if ddotq > 0:
                    # direciton is away from cell, so it was supposed to have intersected this face?
                    # set s=0 i.e. there is no local pathlength
                    #if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} > 0, direction is out of cell (set next, local_dl=0)')
                    s = 0
                    assert 0 # check when/how this really happens
                else:
                    # direction is into the cell, so it must have entered the cell through this face (ignore)
                    #if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} < 0, direction is into cell (ignore)')
                    s = np.inf

                # if np.abs(ddotq) < eps, then the plane and ray are parallel
                #   - if the ray and face perfectly coincide, there is an infinity of intersection solutions
                #   - or, if the ray is off the face, there is no intersection
                # either way, we treat this as a non-intersection
                if np.abs(ddotq) < 1e-10:
                    assert 0 # check when/how this really happens
                
            if s >= 0 and s <= local_dl:
                # we have a valid intersection, and it is closer than all previous intersections, so mark this
                # neighbor as the new best candidate for the exit face
                local_dl = s
                #if debug > 1: print(f'  -- new next neighbor: [{end_cell_local_ind}] with {local_dl = }')
            else:
                # should not occur, we thought we had here a valid intersection
                assert 0

            # candidate new ray position must be within the current, or next, cell
            # if not, we intersected the face-plane outside of the extent of the face polygon
            # and there is in fact a closer face intersection (closer natural neighbor)
            cand_new_ray_pos = ray_pos + ray_dir*local_dl

            cand_index, h_guess = _treeSearchNearestSingle(cand_new_ray_pos,cell_pos,boxSize,NextNode,length,center,sibling,nextnode,h_guess)

            #if debug > 1: print(f' ({n_iter:2d}) {cand_new_ray_pos = } {cand_index = } [L={raylength_left:.4f} R={raylength_right:.4f}]')

            if cand_index not in [cur_cell_ind,end_cell_local_ind]:
                # set new endpoint as candidate ray position (inside some other natural neighbor)
                ray_search_length = np.sum( (cand_new_ray_pos - ray_pos) * ray_dir )
                if ray_search_length > raylength_left:
                    # move right, such that new 0.5*(L+R) is at this point
                    raylength_right = 2*ray_search_length - raylength_left
                else:
                    # move left (should this ever actually happen?)
                    assert 0

                # update stack (avoid duplicates)
                if num_prev_inds == 0 or prev_cell_inds[num_prev_inds-1] != cand_index:
                    prev_cell_inds[num_prev_inds] = cand_index
                    prev_cell_cen[num_prev_inds] = raylength_cen
                    prev_cell_cen[num_prev_inds] += abs_tol # avoid roundoff issues
                    #if debug > 1: print(f' -- adding {num_prev_inds = } index {cand_index} cen {raylength_cen}')
                    num_prev_inds += 1

                #if debug > 1: print(f'  !! neighbor is incorrect, set new [L={raylength_left:.4f} R={raylength_right:.4f}] and re-search')
                continue

            # yes: found a natural neighbor, the correct next cell
            #if debug > 1: print(f' -- {end_cell_local_ind = } matches, finding ray-face intersection...')

            if verify:
                dists = periodicDistsN(m, cell_pos, boxSize)
                min_index_verify = np.where(dists == dists.min())[0][0]
                assert min_index_verify in [cur_cell_ind,end_cell_local_ind]

            # calculate local pathlength, update ray position
            assert local_dl > 0
            assert np.isfinite(local_dl)
            dl += local_dl
            ray_pos += ray_dir*local_dl

            #if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, next cell index = {end_cell_local_ind}')

            # update lengths i.e. maximum search distance along the ray at which to start the next bisection(s)
            for i in range(num_prev_inds):
                prev_cell_cen[i] -= local_dl

            # accumulate (all code units and/or same units as input)
            if n_step >= master_dx.size:
                print('RAY ALLOC FAILURE')
                assert 0

            master_dx[n_step] = local_dl
            master_ind[n_step] = cur_cell_ind
            n_step += 1

            # are we finished unexpectedly?
            assert dl < total_dl
            assert (cur_cell_ind != end_cell_ind) or (total_dl == boxSize)

            # update cur_cell_ind (global ending cell always remains the same)
            prev_cell_ind = cur_cell_ind # for verify only
            cur_cell_ind = end_cell_local_ind

            # is the next cell the end?
            if cur_cell_ind == end_cell_ind:
                # remaining path-length
                local_dl = total_dl - dl
                #if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

                dl += local_dl
                ray_pos += ray_dir*local_dl

                # accumulate (all code units and/or same units as input)
                master_dx[n_step] = local_dl
                master_ind[n_step] = cur_cell_ind
                n_step += 1

                # we are done with the entire ray integration
                #if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, finished.')
                finished = True

            # wrap ray_pos if it has left the box
            for i in range(3):
                ray_pos[i] = _NEAREST_POS(ray_pos[i], boxSize)

            # terminate bisection search, move on to next
            break

    # for safety only
    prev_cell_cen[0] = 0.0
    prev_cell_inds[0] = -1

    # reduce arrays to used size
    master_dx = master_dx[0:n_step]
    master_ind = master_ind[0:n_step]

    return master_dx, master_ind

@jit(nopython=True, nogil=True, cache=True)
def _rayTraceFull(pos, NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, boxSize, ind0, ind1):
    """ Helper for below. Run a series of rays, from ind0 to ind1, and return concatenated results. 
    Full return mode. """
    n_rays = ind1 - ind0 + 1

    # estimate alloc size as (n_rays * avg_intersections_per_ray * safety_fac)
    avg_intersections_per_boxlength = boxSize / pos.shape[0]**(1/3)
    n_alloc = int(n_rays * (total_dl / avg_intersections_per_boxlength)) * 100

    if total_dl < boxSize/10:
        # likely fof-scope and/or halo-centered, e.g. in an overdense region
        n_alloc *= 2

    # allocate (return per ray)
    offsets = np.zeros(n_rays, dtype=np.int32)
    lengths = np.zeros(n_rays, dtype=np.int32)

    # allocate (internal ray arrays) (testing efficiency only)
    #max_steps = 10000
    #master_dx = np.zeros(max_steps, dtype=np.float32) # pathlength for each ray segment
    #master_ind = np.zeros(max_steps, dtype=np.int64) # index
    #prev_cell_inds = np.zeros(max_steps, dtype=np.int64) - 1
    #prev_cell_cen = np.zeros(max_steps, dtype=np.float32)

    # possibly iterate for allocation
    i = -1
    while i != n_rays - 1:
        # allocate for full (dx,ind) lists for each ray, along with per-ray offset/length info
        offset = 0

        r_dx = np.zeros(n_alloc, dtype=np.float32)
        r_ind = np.zeros(n_alloc, dtype=np.int64)

        for i in range(n_rays):
            ray_pos_local = ray_pos[ind0+i,:]
            #print(i,n_rays,ray_pos_local,flush=True)

            dx, ind = trace_ray_through_voronoi_mesh_treebased(pos, NextNode, length, center, sibling, nextnode, 
                                                 ray_pos_local, ray_dir, total_dl, boxSize, 
                                                 #master_dx, master_ind, prev_cell_inds, prev_cell_cen, 
                                                 debug=0, verify=False)

            # stamp
            if offset+dx.size >= n_alloc:
                print('WARNING: REALLOC FOR RAY RETURN.', n_alloc, n_alloc*2)
                n_alloc *= 2
                break

            offsets[i] = offset
            lengths[i] = dx.size
            r_dx[offset:offset+dx.size] = dx
            r_ind[offset:offset+dx.size] = ind
            offset += dx.size

    # return 4-tuple
    r_dx = r_dx[0:offset]
    r_ind = r_ind[0:offset]

    return offsets, lengths, r_dx, r_ind

@jit(nopython=True, nogil=True, cache=True)
def _rayTraceReduced(pos, NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, boxSize, 
                     ind0, ind1, quant, quant2, mode):
    """ Helper for below. Run a series of rays, from ind0 to ind1, and return concatenated results. 
    Reduced mode i.e. compute one answer (number) per ray. """
    n_rays = ind1 - ind0 + 1

    # allocate for some statistical operation producing one float answer per ray
    r_answer = np.zeros(n_rays, dtype=np.float32)

    for i in range(n_rays):
        ray_pos_local = ray_pos[ind0+i,:]

        dx, ind = trace_ray_through_voronoi_mesh_treebased(pos, NextNode, length, center, sibling, nextnode, 
                                             ray_pos_local, ray_dir, total_dl, boxSize, debug=0, verify=False)

        if mode == 1:
            # intersection count per ray
            r_answer[i] = dx.size
        elif mode == 2:
            # sum of dx
            r_answer[i] = np.sum(dx)
        elif mode == 3:
            # sum of quant
            r_answer[i] = np.sum(quant[ind])
        elif mode == 4:
            # sum of quant*dx (if quant is a number or mass density, this is the integrated column or surface density)
            r_answer[i] = np.sum(quant[ind] * dx)
        elif mode == 5:
            # mean quant of all intersected cells
            r_answer[i] = np.mean(quant[ind])
        elif mode == 6:
            # mean quant (e.g. temp), weighted by quant2 (e.g. mass), of all intersected cells
            # note however the pathlength through each cell is not considered
            r_answer[i] = np.sum(quant[ind]*quant2[ind]) / np.sum(quant2[ind])
        elif mode == 7:
            # mean quant (e.g. temp) weighted by quant2*dx (e.g. dens*dl = column density), of all intersected cells
            r_answer[i] = np.sum(quant[ind]*quant2[ind]*dx) / np.sum(quant2[ind]*dx)

    return r_answer

def rayTrace(sP, ray_pos, ray_dir, total_dl, pos, quant=None, quant2=None, mode='full', nThreads=20, tree=None):
    """ For a given set of particle coordinates, assumed to be Voronoi cell center positions, perform a 
    tree-based ray-tracing through this implicit Voronoi mesh, for a number of specified ray_pos initial ray 
    positions, in a given direction, for a total pathlength. The operation while tracing is specified by mode.

    Args:
      sP (:py:class:`util.simParams.simParams`): simParams instance.
      ray_pos (ndarray[float][M,3]): starting ray positions in (x,y,z) space.
      ray_dir (float[3]): normalized unit vector specifying (the same) direction for all rays.
      total_dl (float): path length to integrate all rays (same units as pos).
      pos (ndarray[float][N,3]): array of 3-coordinates for the gas cells.
      quant (ndarray[float][N]): if not None, a quantity with the same size as pos to operate on.
      quant2 (ndarray[float][N]): if not None, a second quantity with the same size as pos to operate on, e.g. for weighting.
      mode (str): one of the following modes of operation:
        **full**: here
        **count**: here
      nThreads (int): do multithreaded calculation (mem required=nThreads times more).
      tree (list or None) if not None, should be a list of all the needed tree arrays (pre-computed), 
                        i.e the exact return of :py:func:`util.treeSearch.buildFullTree`.
    """
    modes = {'full':0, 'count':1, 'dx_sum':2, 'quant_sum':3, 'quant_dx_sum':4, 
             'quant_mean':5, 'quant_weighted_mean':6, 'quant_weighted_dx_mean':7}
    assert mode in modes
    if 'quant' in mode: assert quant is not None, 'Quant required given requested mode.'
    if 'quant_weighted' in mode: assert quant2 is not None, 'Quant2 required as the weights.'

    assert pos.ndim == 2 and pos.shape[1] == 3, 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    if isinstance(ray_dir,list): ray_dir = np.array(ray_dir)
    assert ray_dir.ndim == 1 and ray_dir.size == 3, 'Strange ray_dir.'
    assert quant is None or (quant.ndim == 1 and quant.size == pos.shape[0]), 'Strange quant shape.'
    assert quant2 is None or (quant2.ndim == 1 and quant2.size == pos.shape[0]), 'Strange quant2 shape.'
    
    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos,sP.boxSize,pos.dtype)
    else:
        NextNode, length, center, sibling, nextnode = tree # split out list

    n_rays = ray_pos.shape[0]

    # help numba
    mode = modes[mode] # convert string to number
    if quant is None: quant = np.zeros(1, dtype='float32') # unused, but for type inference
    if quant2 is None: quant2 = np.zeros(1, dtype='float32')

    # single threaded?
    # ----------------
    if nThreads == 1 or n_rays < nThreads:
        ind0 = 0
        ind1 = n_rays - 1

        if mode == modes['full']:
            result = _rayTraceFull(pos,NextNode,length,center,sibling,nextnode,
                                   ray_pos,ray_dir,total_dl,sP.boxSize,ind0,ind1)
        else:
            result = _rayTraceReduced(pos,NextNode,length,center,sibling,nextnode,
                                      ray_pos,ray_dir,total_dl,sP.boxSize,ind0,ind1,
                                      quant,quant2,mode)

        return result

    # else, multithreaded
    # -------------------
    class rtThread(threading.Thread):
        """ Subclass Thread() to provide local storage which can be retrieved after 
            this thread terminates and added to the global return. """
        def __init__(self, threadNum, nThreads):
            super(rtThread, self).__init__()

            # determine local slice
            self.ind0, self.ind1 = pSplitRange([0, n_rays-1], nThreads, threadNum, inclusive=True)

            # copy others into local space (non-self inputs to _calc() appears to prevent GIL release)
            self.ray_dir  = ray_dir
            self.total_dl = total_dl
            self.boxSize  = sP.boxSize
            self.mode     = mode

            # create views to other arrays
            self.pos     = pos
            self.quant   = quant
            self.quant2  = quant2

            self.ray_pos = ray_pos

            self.NextNode  = NextNode
            self.length    = length
            self.center    = center
            self.sibling   = sibling
            self.nextnode  = nextnode

        def run(self):
            # call JIT compiled kernel
            if self.mode == 0:
                self.result = _rayTraceFull(self.pos,self.NextNode,self.length,self.center,self.sibling,self.nextnode,
                                            self.ray_pos,self.ray_dir,self.total_dl,self.boxSize,self.ind0,self.ind1)
            else:
                self.result = _rayTraceReduced(self.pos,self.NextNode,self.length,self.center,self.sibling,self.nextnode,
                                               self.ray_pos,self.ray_dir,self.total_dl,self.boxSize,self.ind0,self.ind1,
                                               self.quant,self.quant2,self.mode)

    # create threads
    threads = [rtThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    # all threads are done, determine return size and allocate
    if mode == modes['full']:
        # offsets, lengths, r_dx, r_ind = result
        n_alloc = np.sum([thread.result[2].size for thread in threads])

        # allocate
        offsets = np.zeros(n_rays, dtype=threads[0].result[0].dtype)
        lengths = np.zeros(n_rays, dtype=threads[0].result[1].dtype)

        r_dx = np.zeros(n_alloc, dtype=threads[0].result[2].dtype)
        r_ind = np.zeros(n_alloc, dtype=threads[0].result[3].dtype)

        # add the result array from each thread to the global
        offset = 0

        for thread in threads:
            loc_offsets, loc_lengths, loc_r_dx, loc_r_ind = thread.result
            # thread local to global offset
            offsets[thread.ind0 : thread.ind1 + 1] = loc_offsets + offset
            lengths[thread.ind0 : thread.ind1 + 1] = loc_lengths

            r_dx[offset : offset + loc_r_dx.size] = loc_r_dx
            r_ind[offset : offset + loc_r_dx.size] = loc_r_ind
            offset += loc_r_dx.size

        return offsets, lengths, r_dx, r_ind
    else:
        n_alloc = np.sum([thread.result.size for thread in threads])
        assert n_alloc == n_rays

        # allocate
        result = np.zeros( n_alloc, dtype=threads[0].result.dtype)

        # add the result array from each thread to the global
        for thread in threads:
            result[thread.ind0 : thread.ind1 + 1] = thread.result

    return result

def benchmark_test_raytracing():
    """ Run a large number of rays using the threaded-code. """
    import matplotlib.pyplot as plt
    from ..plot.config import figsize, lw
    from ..util.simParams import simParams

    # config
    sP = simParams(run='tng100-3', redshift=0.5)
    ray_dir = [0.0, 0.0, 1.0]
    n_rays = 10000

    # loop over:
    total_dls = [5000.0, 10000.0] # [1000.0, 5000.0]
    nThreads = [1,1,2,4,8,16,32]

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
    #fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111)

    # loop over requested path lengths
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
        #ax.set_xlabel('Number of Threads')
        #ax.set_ylabel('Time [sec]')
        #ax.plot(nThreads[1:], times[1:], 'o-', lw=lw, label='dl = %d' % total_dl)

    # finish scaling plot
    #ax.legend(loc='upper right')
    #fig.savefig('benchmark_test_raytracing_%s.pdf' % mode)

    #ax.set_yscale('log')
    #fig.savefig('benchmark_test_raytracing_%s_log.pdf' % mode)
    #plt.close(fig)

def benchmark_test_voronoi(compare=True):
    """ Run a large number of rays through the (fullbox) Voronoi mesh, in each case comparing the 
    results from pre-computed vs. tree-based approaches, for correctness (and speed).

    Args:
      compare (bool): if True, then run both methods and assert they have the same return.
        Otherwise, run only the tree-based method for timing.

    Returns:
      None
    """
    from ..util.simParams import simParams

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

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
