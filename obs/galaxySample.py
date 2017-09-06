"""
obs/galaxySample.py
  Observations: re-create various mock galaxy samples to match surveys/datasets.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os.path import isfile

from util.loadExtern import werk2013
from cosmo.util import redshiftToSnapNum
from cosmo.load import groupCat

def obsMatchedSample(sP, datasetName='COS-Halos', numRealizations=100):
    """ Get a matched sample of simulated galaxies which match an observational abs. line dataset. """
    if datasetName == 'COS-Halos':
        # load
        gals, logM, redshift, sfr, sfr_err, sfr_limit, R, _, _, _ = werk2013()
        logM_err = 0.2 # dex, assumed
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
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))
            sfr_rnd_err = np.random.normal(loc=0.0, scale=sfr_err, size=len(gals))
            ssfr = (sfr + sfr_rnd_err) / 10.0**(logM + mass_random_err_log)

            props['mstar_30pkpc_log'][:,i] = logM + mass_random_err_log
            props['ssfr_30pkpc_log'][:,i] = np.log10( ssfr )

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
