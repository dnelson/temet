"""
Algorithms and methods related to ray-tracing through a Voronoi mesh.
"""
import numpy as np
import time

from util.sphMap import _NEAREST_POS
from util.voronoi import loadSingleHaloVPPP, loadGlobalVPPP
from util.treeSearch import buildFullTree, _treeSearchNearest

def _periodic_wrap_point(pos, pos_ref, boxSize, boxHalf):
    """ If pos is more than a half-box away from pos_ref, wrap it into the same octant. """
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

def trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, cell_vellos, cell_temp, cell_dens, 
                                   num_ngb, ngb_inds, offset_ngb,
                                   ray_pos_in, ray_dir, total_dl, sP, debug, verify, fof_scope_mesh):
    """ For a single ray, specified by its starting location, direction, and length, ray-trace through 
    a Voronoi mesh as specified by the pre-computed natural neighbor connectivity information. """

    # path length accumulated
    boxHalf = sP.boxSize / 2
    ray_pos = ray_pos_in.copy()

    dl = 0.0
    n_step = 0

    # locate starting cell
    dists = sP.periodicDists(ray_pos, cell_pos)
    cur_cell_ind = np.where(dists == dists.min())[0][0]
    prev_cell_ind = -1
    
    if debug: print(f'Starting cell index [{cur_cell_ind}] at distance = {dists[cur_cell_ind]:.2f} ckpc/h, {total_dl = :.3f}.')

    # allocate
    max_steps = 10000
    
    master_dens   = np.zeros(max_steps, dtype='float32') # density for each ray segment
    master_dx     = np.zeros(max_steps, dtype='float32') # pathlength for each ray segment
    master_temp   = np.zeros(max_steps, dtype='float32') # temp for each ray segment
    master_vellos = np.zeros(max_steps, dtype='float32') # line of sight velocity

    # while total dl does not exceed request, start tracing through mesh
    while 1:
        # current Voronoi cell
        cur_cell_pos = cell_pos[cur_cell_ind].copy()
        _periodic_wrap_point(cur_cell_pos, ray_pos, sP.boxSize, boxHalf)
        
        if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

        local_dl = np.inf
        next_ngb_index = -1

        if verify:
            dists = sP.periodicDists(ray_pos, cell_pos)
            mindist_cell_ind = np.where(dists == dists.min())[0][0]
            # due to round-off, answer should be ambiguous between previous and current cell (we sit on the face)
            if mindist_cell_ind not in [prev_cell_ind,cur_cell_ind]:
                if fof_scope_mesh:
                    dist_to_halo_cen = sP.periodicDists(ray_pos, halo['GroupPos']) / halo['Group_R_Crit200']
                    # note: still fail if we start too early i.e. before fof-scope!
                    assert dist_to_halo_cen > 1.0 and dl > total_dl/2 # otherwise check
                    if debug: print(' -- NOTE: Termination! Leaving fof-scope mesh.')
                    break
                else:
                    assert 0 # should not occur

        # loop over all natural neighbors
        for i in range(num_ngb[cur_cell_ind]):
            # neighbor properties
            ngb_index = ngb_inds[offset_ngb[cur_cell_ind]+i]
            ngb_pos = cell_pos[ngb_index].copy()
            _periodic_wrap_point(ngb_pos, ray_pos, sP.boxSize, boxHalf)

            if ngb_index == -1: # outside of fof-scope mesh
                assert fof_scope_mesh # otherwise should not occur
                if debug > 1: print(f' [{i:2d} of {num_ngb[cur_cell_ind]:2d}] with {ngb_index = } skip')
                continue

            if debug > 1: print(f' [{i:2d} of {num_ngb[cur_cell_ind]:2d}] with {ngb_index = } and {ngb_pos = }')

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
                    if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} > 0, direction is out of cell (set next, local_dl=0)')
                    s = 0
                    assert 0 # check when/how this really happens
                else:
                    # direction is into the cell, so it must have entered the cell through this face (ignore)
                    if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} < 0, direction is into cell (ignore)')
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
                if debug > 1: print(f'  -- new next neighbor: [{ngb_index}] with {local_dl = }')

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
            ray_pos[i] = _NEAREST_POS(ray_pos[i], sP.boxSize)

        if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, next cell index = {next_ngb_index}')

        # accumulate
        master_dens[n_step] = cell_dens[cur_cell_ind] # ions/cm^3
        master_dx[n_step] = local_dl # code!
        master_temp[n_step] = cell_temp[cur_cell_ind] # K
        master_vellos[n_step] = cell_vellos[cur_cell_ind] # km/s

        if dl >= (total_dl - 1e-10):
            # integrate final cell partially
            if debug > 1: print('  -- reached total pathlength, terminating.')
            break

        # do we have a next valid neighbor? if not, exit
        if next_ngb_index == -1:
            assert fof_scope_mesh # otherwise should not happen
            if debug > 1: print('  -- next neighbor is outside FoF scope, terminating.')
            break

        # move to next cell
        prev_cell_ind = cur_cell_ind
        cur_cell_ind = next_ngb_index
        n_step += 1

    # reduce arrays to used size
    master_dens   = master_dens[0:n_step]
    master_dx     = master_dx[0:n_step]
    master_temp   = master_temp[0:n_step]
    master_vellos = master_vellos[0:n_step]

    return master_dens, master_dx, master_temp, master_vellos

def _locate_nearest_cell(xyz,pos,posMask,boxSizeSim,NextNode,length,center,sibling,nextnode,h_guess=1.0):
    """ Iterate on tree-search until we have at least one neighbor, return nearest. """
    NumPart = pos.shape[0]
    loc_index = -1
    iter_num = 0

    while loc_index == -1:
        loc_index, loc_dist2 = _treeSearchNearest(xyz,h_guess,NumPart,boxSizeSim,pos,posMask,
                                                  NextNode,length,center,sibling,nextnode)
        h_guess *= 2.0
        iter_num += 1

        if iter_num > 1000:
            print('ERROR: Failed to converge.')
            break

    return loc_index, np.sqrt(loc_dist2)

def trace_ray_through_voronoi_mesh_treebased(cell_pos, cell_vellos, cell_temp, cell_dens, 
                                             NextNode, length, center, sibling, nextnode, 
                                             ray_pos_in, ray_dir, total_dl, sP, debug, verify):
    """ For a single ray, specified by its starting location, direction, and length, ray-trace through 
    a Voronoi mesh using neighbor searches in a pre-computed tree. """

    # if a bisection along a ray converges to this level (in code units), we have iterated to an 
    # actual face which gives the correct next natural neighbor. in this case we stop, as our
    # criterion won't otherwise find it (the midpoint of the two cells, although in the face-plane, 
    # is not actually in the face polygon, but is instead inside a different natural neighbor of the 
    # current cell)
    abs_tol = 0.01

    # path length accumulated
    ray_pos = ray_pos_in.copy()
    dl = 0.0
    n_step = 0

    # no masking of search candidates
    posMask = np.ones(cell_pos.shape[0], dtype=np.bool)
    boxSize = sP.boxSize
    boxHalf = sP.boxSize / 2

    # allocate
    max_steps = 10000
    
    master_dens   = np.zeros(max_steps, dtype='float32') # density for each ray segment
    master_dx     = np.zeros(max_steps, dtype='float32') # pathlength for each ray segment
    master_temp   = np.zeros(max_steps, dtype='float32') # temp for each ray segment
    master_vellos = np.zeros(max_steps, dtype='float32') # line of sight velocity

    # for bisection stack: indices of previous failed candidate cell(s)
    prev_cell_inds = np.zeros(max_steps, dtype='int64') - 1
    prev_cell_cen = np.zeros(max_steps, dtype='float32')

    num_prev_inds = 0

    iter_counter  = np.zeros(max_steps, dtype='int32')

    # locate starting cell
    #cur_cell_ind, dist = _locate_nearest_cell(ray_pos,cell_pos)
    cur_cell_ind, h_guess = _locate_nearest_cell(ray_pos,cell_pos,posMask,boxSize,NextNode,length,center,sibling,nextnode,h_guess=1.0)

    # where will we terminate ray, globally?
    ray_end = ray_pos + ray_dir * total_dl

    #end_cell_ind, dist_end = _locate_nearest_cell(ray_end,cell_pos)
    end_cell_ind, h_guess = _locate_nearest_cell(ray_end,cell_pos,posMask,boxSize,NextNode,length,center,sibling,nextnode,h_guess)
    end_cell_pos = cell_pos[end_cell_ind]

    if debug: print(f'Starting cell index [{cur_cell_ind}], ending cell index [{end_cell_ind}], {total_dl = :.3f}.')

    if verify:
        # verify start
        dists = sP.periodicDists(ray_pos, cell_pos)
        mindist_cell_ind = np.where(dists == dists.min())[0][0]
        assert mindist_cell_ind == cur_cell_ind
        # verify end
        dists = sP.periodicDists(ray_end, cell_pos)
        mindist_cell_ind = np.where(dists == dists.min())[0][0]
        assert mindist_cell_ind == end_cell_ind

    prev_cell_ind = -1 # for verify only

    # loop while still intersecting cells
    finished = False

    while not finished:
        # current Voronoi cell
        cur_cell_pos = cell_pos[cur_cell_ind]

        if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

        if verify:
            dists = sP.periodicDists(ray_pos, cell_pos)
            mindist_cell_ind = np.where(dists == dists.min())[0][0]
            # due to round-off, answer should be ambiguous between previous and current cell (we sit on the face)
            assert mindist_cell_ind in [prev_cell_ind,cur_cell_ind]

        # ending target for this segment is the global ray endpoint, unless we have previously failed a 
        # bisection and thus have a closer guess
        end_cell_local_ind = -1

        raylength_left = 0.0
        raylength_right = total_dl - dl # total remaining length

        # bisection acceleration: from the last ray position, we can use the 'closest' failed 
        # distance as a (closer) starting point
        while num_prev_inds > 0 and prev_cell_inds[num_prev_inds-1] == cur_cell_ind:
            # avoid self, pop from stack
            if debug > 1: print(' -- remove self from prev_cell_inds stack!')
            prev_cell_cen[num_prev_inds-1] = 0.0 # for safety only
            prev_cell_inds[num_prev_inds-1] = -1 # for safety only
            num_prev_inds -= 1

        if num_prev_inds > 0 and prev_cell_inds[num_prev_inds-1] >= 0:
            # set first ray_end_local to known (shorter) result
            raylength_right = 2 * prev_cell_cen[num_prev_inds-1]
            end_cell_local_ind = prev_cell_inds[num_prev_inds-1]

            if debug > 1: print(' -- prev_cell_inds stack: ', prev_cell_inds[0:num_prev_inds+2])
            if debug > 1: print(' -- prev_cell_cen stack: ', prev_cell_cen[0:num_prev_inds+2])
            # pop from stack
            prev_cell_cen[num_prev_inds-1] = 0.0 # for safety only
            prev_cell_inds[num_prev_inds-1] = -1 # for safety only
            num_prev_inds -= 1
            if debug: print(f' -- set {raylength_right = } from {num_prev_inds = } index {end_cell_local_ind}')

        assert raylength_right > 0.0 # otherwise used an empty prev_cell_cen value?

        # while the cell containing the end of the segment is not a natural neighbor of the current cell
        local_dl = np.inf

        for n_iter in range(1000):
            # set distance along ray as midpoint between bracketing
            iter_counter[n_step] = n_iter

            assert n_iter < 100 and (raylength_right - raylength_left) > 1e-10 # otherwise failure

            # new test position along ray (ray_end_local can be outside box, which is ok for _locate_nearest_cell)
            raylength_cen = 0.5 * (raylength_left + raylength_right)

            ray_end_local = ray_pos + ray_dir * raylength_cen

            if debug > 1: print(f' ({n_iter:2d}) L+R midpoint = {(raylength_left+raylength_right)*0.5:.5f}')

            # locate parent cell of this point
            if n_iter > 0 or end_cell_local_ind == -1:
                # only skip this tree-research, possibly, on first iteration if we have a saved index
                # from a previous bisection
                #end_cell_local_ind, dist_end_local = _locate_nearest_cell(ray_end_local,cell_pos)
                end_cell_local_ind, dist_end_local = _locate_nearest_cell(ray_end_local,cell_pos,posMask,boxSize,NextNode,length,center,sibling,nextnode,h_guess)

            end_cell_pos_local = cell_pos[end_cell_local_ind].copy()

            _periodic_wrap_point(end_cell_pos_local, cur_cell_pos, sP.boxSize, boxHalf)

            if verify:
                dists = sP.periodicDists(ray_end_local, cell_pos)
                min_index_verify = np.where(dists == dists.min())[0][0]
                assert min_index_verify == end_cell_local_ind

            # is this parent the same as the current cell? then we have skipped over the neighbor
            if end_cell_local_ind == cur_cell_ind:
                # no: modify starting point (bisection), and continue loop
                raylength_left = (raylength_right + raylength_left) * 0.5

                if debug > 1: print(f'  !! neighbor was skipped, set new [L={raylength_left:.4f} R={raylength_right:.4f}] and re-search')
                continue

            # edge midpoint, i.e. a point on the Voronoi face plane shared with this neighbor, if the 
            # current and final cells are actually natural neighbors
            m = 0.5 * (end_cell_pos_local + cur_cell_pos)

            # locate parent cell of this point: is it in one of the two cells?
            #cand_cell_ind, dist_cand = _locate_nearest_cell(m,cell_pos)
            cand_cell_ind, h_guess = _locate_nearest_cell(m,cell_pos,posMask,boxSize,NextNode,length,center,sibling,nextnode,h_guess)

            if verify:
                dists = sP.periodicDists(m, cell_pos)
                min_index_verify = np.where(dists == dists.min())[0][0]
                # note: if we are actually at the natural neighbor, then cand_cell_ind could be either
                # end_cell_local_ind or cur_cell_ind, so likely we should also allow cur_cell_ind here
                assert min_index_verify in [cand_cell_ind,end_cell_local_ind]

            if debug: print(f' ({n_iter:2d}) {ray_end_local = } {cand_cell_ind = } {end_cell_local_ind = } [L={raylength_left:.4f} R={raylength_right:.4f}]')

            # if it isn't in either cell, and our bisection has not yet converged, then continue
            # but if our bisection has converged, then assume that the parent cell of the 
            # local ray end position is a natural neighbor whose shared face does not, unfortunately, 
            # contain the midpoint of the line segment between the two cell centers
            if cand_cell_ind not in [cur_cell_ind,end_cell_local_ind] and \
                ((raylength_right - raylength_left) > abs_tol):
                # update stack (avoid duplicates)
                if num_prev_inds == 0 or prev_cell_inds[num_prev_inds-1] != end_cell_local_ind:
                    prev_cell_inds[num_prev_inds] = end_cell_local_ind
                    prev_cell_cen[num_prev_inds] = raylength_cen
                    if debug > 1: print(f' -- adding {num_prev_inds = } index {end_cell_local_ind} cen {raylength_cen}')
                    num_prev_inds += 1

                # no: modify end point (bisection), and continue loop
                raylength_right = (raylength_right + raylength_left) * 0.5

                if debug > 1: print(f' -- {end_cell_local_ind = } not a match, bisecting [L={raylength_left:.4f} R={raylength_right:.4f}]...')
                continue

            # yes: found -a- natural neighbor, which may be the correct next cell
            if debug > 1: print(f' -- {end_cell_local_ind = } matches, finding ray-face intersection...')

            if verify and ((raylength_right - raylength_left) > abs_tol):
                dists = sP.periodicDists(m, cell_pos)
                min_index_verify = np.where(dists == dists.min())[0][0]
                assert min_index_verify in [cur_cell_ind,end_cell_local_ind]

            # the vector from the current ray position to m
            ray_pos_wrapped = ray_pos.copy()
            _periodic_wrap_point(ray_pos_wrapped, cur_cell_pos, sP.boxSize, boxHalf)

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
                    if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} > 0, direction is out of cell (set next, local_dl=0)')
                    s = 0
                    assert 0 # check when/how this really happens
                else:
                    # direction is into the cell, so it must have entered the cell through this face (ignore)
                    if debug > 2: print(f'  -- cdotq <= 0! {ddotq = :g} < 0, direction is into cell (ignore)')
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
                local_dl = s
                if debug > 1: print(f'  -- new next neighbor: [{end_cell_local_ind}] with {local_dl = }')
            else:
                # should not occur, we thought we had here a valid intersection
                assert 0

            # candidate new ray position must be within the current, or next, cell
            # if not, we intersected the face-plane outside of the extent of the face polygon
            # and there is in fact a closer face intersection (closer natural neighbor)
            cand_new_ray_pos = ray_pos + ray_dir*local_dl

            #cand_index, _ = _locate_nearest_cell(cand_new_ray_pos,cell_pos)
            cand_index, h_guess = _locate_nearest_cell(cand_new_ray_pos,cell_pos,posMask,boxSize,NextNode,length,center,sibling,nextnode,h_guess)

            if cand_index not in [cur_cell_ind,end_cell_local_ind]:
                # set new endpoint as candidate ray position (inside some other natural neighbor)
                ray_search_length = np.sum( (cand_new_ray_pos - ray_pos) * ray_dir )
                if ray_search_length > raylength_left:
                    # move right, such that new 0.5*(L+R) is at this point
                    raylength_right = 2*ray_search_length - raylength_left
                else:
                    # move left (todo, should this ever actually happen?)
                    assert 0

                if debug > 1: print(f'  !! neighbor is incorrect, set new [L={raylength_left:.4f} R={raylength_right:.4f}] and re-search')
                continue

            # calculate local pathlength, update ray position
            assert local_dl > 0
            assert np.isfinite(local_dl)
            dl += local_dl
            ray_pos += ray_dir*local_dl

            if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, next cell index = {end_cell_local_ind}')

            # update lengths i.e. maximum search distance along the ray at which to start the next bisection(s)
            for i in range(num_prev_inds):
                prev_cell_cen[i] -= local_dl

            # accumulate
            master_dens[n_step] = cell_dens[cur_cell_ind] # ions/cm^3
            master_dx[n_step] = local_dl # code!
            master_temp[n_step] = cell_temp[cur_cell_ind] # K
            master_vellos[n_step] = cell_vellos[cur_cell_ind] # km/s

            # are we finished unexpectedly?
            assert dl < total_dl
            assert cur_cell_ind != end_cell_ind # dl should exceed total_dl first

            # update cur_cell_ind (global ending cell always remains the same)
            prev_cell_ind = cur_cell_ind # for verify only
            cur_cell_ind = end_cell_local_ind
            n_step += 1

            # is the next cell the end?
            if cur_cell_ind == end_cell_ind:
                # remaining path-length
                local_dl = total_dl - dl
                if debug: print(f'[{n_step:3d}] {dl = :7.3f} {ray_pos = } {cur_cell_ind = }')

                dl += local_dl
                ray_pos += ray_dir*local_dl

                # accumulate
                master_dens[n_step] = cell_dens[cur_cell_ind] # ions/cm^3
                master_dx[n_step] = local_dl # code!
                master_temp[n_step] = cell_temp[cur_cell_ind] # K
                master_vellos[n_step] = cell_vellos[cur_cell_ind] # km/s

                # we are done with the entire ray integration
                if debug: print(f' ** accumulate {cur_cell_ind = }, {local_dl = :.3f}, finished.')
                finished = True

            # wrap ray_pos if it has left the box
            for i in range(3):
                ray_pos[i] = _NEAREST_POS(ray_pos[i], sP.boxSize)

            # terminate bisection search, move on to next
            break

    # reduce arrays to used size
    master_dens   = master_dens[0:n_step]
    master_dx     = master_dx[0:n_step]
    master_temp   = master_temp[0:n_step]
    master_vellos = master_vellos[0:n_step]

    iter_counter = iter_counter[0:n_step]

    if debug: print('iter_counter: ', iter_counter)

    return master_dens, master_dx, master_temp, master_vellos

def benchmark_test_voronoi(compare=True):
    """ Run a large number of rays through the (fullbox) Voronoi mesh, in each case comparing the 
    results from pre-computed vs. tree-based approaches, for correctness (and speed). """
    from util.simParams import simParams

    # config
    sP = simParams(run='tng50-3', redshift=0.5)

    projAxis = 2 # z, to simplify vellos for now

    num_rays = 100
    verify = False

    # load global cell properties (pos,vel,species dens,temp)
    velLosField = 'vel_'+['x','y','z'][projAxis]

    cell_pos    = sP.snapshotSubsetP('gas', 'pos') # code
    cell_vellos = sP.snapshotSubsetP('gas', velLosField) # code
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear') # K
    cell_dens   = sP.snapshotSubset('gas', 'nh') # ions/cm^3

    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

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
            master_dens, master_dx, master_temp, master_vellos = \
              trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, cell_vellos, cell_temp, cell_dens, 
                                           num_ngb, ngb_inds, offset_ngb, ray_pos, ray_dir, total_dl, 
                                           sP, debug=0, verify=verify, fof_scope_mesh=False)

        time_a += (time.time() - start_time) # accumulate
        
        # (B) ray-trace with tree-based method
        start_time = time.time()

        master_dens2, master_dx2, master_temp2, master_vellos2 = \
          trace_ray_through_voronoi_mesh_treebased(cell_pos, cell_vellos, cell_temp, cell_dens, 
                                       NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, 
                                       sP, debug=0, verify=verify)

        time_b += (time.time() - start_time)

        # compare
        N_intersects += master_dens2.size
        total_pathlength += total_dl

        if compare:
            assert np.allclose(master_dens2,master_dens)
            assert np.allclose(master_dx,master_dx2)
            assert np.allclose(master_temp,master_temp2)
            assert np.allclose(master_vellos,master_vellos2)

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
