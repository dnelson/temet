"""
obs/galaxySample.py
  Observations: re-create various mock galaxy samples to match surveys/datasets.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os.path import isfile, isdir
from os import mkdir

from util.loadExtern import werk2013
from cosmo.util import redshiftToSnapNum
from cosmo.load import groupCat, groupCatSingle, snapshotSubset
from vis.common import getHsmlForPartType
from util.sphMap import sphMap
from util.helper import logZeroNaN, closest
from cosmo.cloudy import cloudyIon

def obsMatchedSample(sP, datasetName='COS-Halos', numRealizations=100):
    """ Get a matched sample of simulated galaxies which match an observational abs. line dataset. """
    if datasetName == 'COS-Halos':
        # load
        gals, logM, redshift, sfr, sfr_err, sfr_limit, R, _, _, _ = werk2013()
        logM_err = 0.2 # dex, assumed
        R_err = 1.0 # kpc, assumed
        sfr_err[sfr_limit] = 0.0 # upper limits

        # define how we will create this sample, by matching on what quantities
        propList = ['mstar_30pkpc_log','ssfr_30pkpc_log','central_flag',
                    'isolated3d_mstar_30pkpc_max_in_300pkpc']

        # set up required properties and limit types
        shape = (len(gals), numRealizations)

        props = {}
        props['mstar_30pkpc_log'] = np.zeros( shape, dtype='float32' )
        props['ssfr_30pkpc_log'] = np.zeros( shape, dtype='float32' )

        props['central_flag'] = np.zeros( shape, dtype='int16' )
        props['central_flag'].fill(1) # realization independent, always required

        props['isolated3d_mstar_30pkpc_max_in_300pkpc'] = np.zeros( shape, dtype='int16' )
        props['isolated3d_mstar_30pkpc_max_in_300pkpc'].fill(1) # realization independent, always required

        props['impact_parameter'] = np.zeros( shape, dtype='float32' )

        limits = {}
        for propName in propList:
            # set 0=compute in distance (default), 1=upper limit, 2=lower limit, 3=exact match required
            limits[propName] = np.zeros( shape, dtype='int16' )

        limits['ssfr_30pkpc_log'][np.where(sfr_limit),:] = 1 # realization independent, upper limit
        limits['central_flag'][:] = 3 # realization/galaxy indepedent, exact
        limits['isolated3d_mstar_30pkpc_max_in_300pkpc'][:] = 3 # realization/galaxy indepedent, exact

        # create realizations by adding appropriate noise to obs
        np.random.seed(424242)
        for i in range(numRealizations):
            impact_param_random_err = np.random.normal(loc=0.0, scale=R_err, size=len(gals))
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))
            sfr_rnd_err = np.random.normal(loc=0.0, scale=sfr_err, size=len(gals))
            ssfr = (sfr + sfr_rnd_err) / 10.0**(logM + mass_random_err_log)

            props['mstar_30pkpc_log'][:,i] = logM + mass_random_err_log
            props['ssfr_30pkpc_log'][:,i] = np.log10( ssfr )
            props['impact_parameter'][:,i] = R + impact_param_random_err

    if datasetName == 'other_todo':
        assert 0

    # save file exists?
    saveFilename = sP.derivPath + "obsMatchedSample_%s_%d.hdf5" % (datasetName,numRealizations)

    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # no, compute now
    r = {}

    if len(redshift) > 1:
        # match each galaxy redshift to a simulation snapshot
        r['snaps'] = redshiftToSnapNum(redshift, sP)
    else:
        # match all observed galaxies to the single simulation redshift 'z'
        r['snaps'] = [ redshiftToSnapNum(redshift, sP) ]

    # save the final subfind IDs we select via matching
    r['selected_inds'] = np.zeros( shape, dtype='int32' )

    for propName in propList:
        r[propName] = np.zeros( shape, dtype=props[propName].dtype )

    for propName in set(props.keys()) - set(propList):
        r[propName] = props[propName] # straight copy, not used in selection (i.e. impact parameter)

    origSnap = sP.snap

    for snap in np.unique(r['snaps']):
        # which galaxies correspond to this snap?
        w_loc = np.where(r['snaps'] == snap)
        print('[snap %3d] N = %d' % (snap,len(w_loc[0])))

        # load catalog properties at this redshift
        sP.setSnap(snap)
        sim_props = groupCat(sP, fieldsSubhalos=propList)

        # loop over observed galaxies
        for gal_num in w_loc[0]:
            # loop over requested realizations
            print(' [%2d] [' % gal_num,end='')
            for realization in range(numRealizations):
                # subset of absolutely consistent simulated systems (i.e. handle limits)
                mask = np.ones( sim_props[propList[0]].size, dtype=np.bool )
                print('.',end='')

                for propName in propList:
                    loc_limit = limits[propName][gal_num,realization]
                    loc_prop_val = props[propName][gal_num,realization]

                    if loc_limit == 1: # upper limit
                        mask &= (sim_props[propName] < loc_prop_val)
                    if loc_limit == 2: # lower limit
                        mask &= (sim_props[propName] > loc_prop_val)
                    if loc_limit == 3: # exact, i.e. only integer makes sense
                        mask &= (sim_props[propName] == loc_prop_val)

                w_mask = np.where(mask)

                # L1 norm (Manhattan distance metric) for remaining properties (i.e. handle non-limits)
                dists = np.zeros( w_mask[0].size )

                for propName in propList:
                    loc_limit = limits[propName][gal_num,realization]
                    loc_prop_val = props[propName][gal_num,realization]

                    if loc_limit == 0:
                        dists += np.abs( sim_props[propName][w_mask] - loc_prop_val )

                # select minimum distance in the space of properties to be matched
                w_nan = np.isnan(dists)
                dists[w_nan] = np.nanmax(dists) + 1.0 # filter out nan arising from simulated galaxies

                r['selected_inds'][gal_num,realization] = w_mask[0][ dists.argmin() ]

            # store properties as matched
            for propName in propList:
                r[propName][gal_num,:] = sim_props[propName][ r['selected_inds'][gal_num,:] ]

            print(']')
            print(' [%d] selected_inds = %d, %d...' % \
                (gal_num,r['selected_inds'][gal_num,0],r['selected_inds'][gal_num,1]))

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]

    return r

def addIonColumnPerSystem(sP, sim_sample, config='COS-Halos'):
    """ Compute gridded column densities around a sample of simulated galaxies and attached a 
    single column value to each in analogy to the observational dataset. """
    if config == 'COS-Halos':
        # impact parameters
        #impact_parameters = werk2013()[6]
        #assert impact_parameters.size == sim_sample['selected_inds'].shape[0]

        # grid parameters
        partType  = 'gas'
        ionName   = 'O VI'
        projDepth = sP.units.codeLengthToKpc(2000.0) # pkpc in projection direction, same as rad profiles
        gridSize  = 800.0 # pkpc, box sidelength
        gridRes   = 2.0 # pkpc, pixel size
        axes      = [0,1] # x,y

        nPixels = [int(gridSize / gridRes), int(gridSize / gridRes)] # square

    if config == 'other_todo':
        assert 0

    # save file exists?
    saveFilename = sP.derivPath + "obsMatchedColumns_%s_%d.hdf5" % (config,sim_sample['selected_inds'].size)

    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        r.update(sim_sample)
        return r

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')
    if not isdir(sP.derivPath + 'grids/%s/' % config):
        mkdir(sP.derivPath + 'grids/%s/' % config)

    # no, compute now
    r = {}
    r['column'] = np.zeros( sim_sample['selected_inds'].shape, dtype='float32' )

    origSnap = sP.snap
    ion = cloudyIon(None)
    np.random.seed(4242 + sim_sample['selected_inds'].size)

    def _gridFilePath():
        gridPath = sP.derivPath + 'grids/%s/' % config
        gridFile = gridPath + 'snap-%d_ind-%d_axes-%d%d.hdf5' % (snap,ind,axes[0],axes[1])
        return gridFile

    for snap in np.unique(sim_sample['snaps']):
        # which realized galaxies (a unique set) are in this snap?
        w_loc = np.where(sim_sample['snaps'] == snap)

        inds_all = sim_sample['selected_inds'][w_loc,:].ravel()
        inds_uniq = np.unique(inds_all)

        print('[snap %3d] N_all = %d N_unique = %d' % (snap,inds_all.size,inds_uniq.size))

        # check which grids are already made / if any are missing
        allExist = True
        inds_todo = []

        for ind in inds_uniq:
            if not isfile(_gridFilePath()):
                inds_todo.append(ind)
                allExist = False

        if allExist:
            print(' all subhalos done, skipping this snapshot.')
            continue

        # process: global load all particle data needed (calculate OVI on the fly, no cache)
        print(' loading mass...')
        mass = snapshotSubset(sP, partType, '%s mass' % ionName).astype('float32')
        print(' loading pos...')
        pos = snapshotSubset(sP, partType, 'pos')
        print(' loading hsml...')
        hsml = getHsmlForPartType(sP, partType)

        assert mass.min() >= 0.0

        # loop over all inds
        for i, ind in enumerate(inds_todo):
            # configure grid for this galaxy
            boxSizeImg = sP.units.physicalKpcToCodeLength( np.array([gridSize, gridSize, projDepth]) )
            boxCenter  = groupCatSingle(sP, subhaloID=ind)['SubhaloPos']

            boxSizeImg = boxSizeImg[ [axes[0], axes[1], 3-axes[0]-axes[1]] ]
            boxCenter  = boxCenter[ [axes[0], axes[1], 3-axes[0]-axes[1]] ]

            # call projection
            print(' projecting [%3d of %3d] ind = %d' % (i,len(inds_todo),ind))

            grid_d = sphMap( pos=pos, hsml=hsml, mass=mass, quant=None, axes=axes, ndims=3, 
                             boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, boxCen=boxCenter, 
                             nPixels=nPixels, colDens=True, multi=False )

            grid_d /= ion.atomicMass(ionName.split()[0]) # [H atoms/cm^2] to [ions/cm^2]
            grid_d = logZeroNaN( sP.units.codeColDensToPhys(grid_d,cgs=True,numDens=True) ) # [log 1/cm^2]

            # save
            with h5py.File(_gridFilePath(),'w') as f:
                f['grid'] = grid_d

    # create 2d distance mask and in order to select correct distance 'ring'
    zz = np.indices(nPixels)
    dist_mask = np.sqrt( ((np.flip(zz[1],1) - zz[1])/2)**2 + ((np.flip(zz[0],0) - zz[0])/2)**2 )
    dist_mask_local = dist_mask * gridRes # for now, constant per halo

    # loop over the snapshot set again
    for snap in np.unique(sim_sample['snaps']):
        # which realized galaxies (a unique set) are in this snap?
        w_loc = np.where(sim_sample['snaps'] == snap)

        # all grids now exist, process them to extract single column values per galaxy
        for gal_num in w_loc[0]:
            # loop through realizations and load the grid of each
            inds = np.squeeze( sim_sample['selected_inds'][gal_num,:] )
            print(' [%2d] compute final column for each realization...' % gal_num)

            for realization_num, ind in enumerate(inds):
                with h5py.File(_gridFilePath(),'r') as f:
                    grid = f['grid'][()]

                # find the unique distance in the mask closest to the requested b parameter
                target_impact_parameter = sim_sample['impact_parameter'][gal_num,realization_num]
                discrete_dist_val, _ = closest(dist_mask_local, target_impact_parameter)

                # find all the pixels that share this [float32] distance value
                w_dist = np.where(dist_mask_local == discrete_dist_val)

                # randomly choose a pixel at the correct distance and save
                valid_values = grid[w_dist]
                chosen_val = np.random.choice(valid_values)
                assert np.isfinite(chosen_val) # otherwise, choose again

                r['column'][gal_num,realization_num] = chosen_val

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]

    r.update(sim_sample)
    return r