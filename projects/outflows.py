"""
projects/oxygen.py
  Plots: Outflows paper (TNG50 presentation).
  in prep.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic, binned_statistic_2d
from os.path import isfile, isdir
from os import mkdir

from util import simParams
from plot.config import *
from util.helper import loadColorTable, running_median, logZeroNaN, iterable, closest
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.mergertree import loadMPBs
from plot.general import plotHistogram1D, plotPhaseSpace2D
from plot.quantities import simSubhaloQuantity
from tracer.tracerMC import match3
from vis.common import gridBox

def halo_selection(sP, minM200=11.5):
    """ Make a quick halo selection above some mass limit and sorted based on energy 
    injection in the low BH state between this snapshot and the previous. """
    snap = sP.snap
    tage = sP.tage
    
    r = {}

    # quick caching
    saveFilename = '/u/dnelson/temp_haloselect_%.1f.hdf5' % minM200
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # halo selection: all centrals above 10^12 Mhalo
    m200 = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w = np.where(m200 >= minM200)
    subInds = w[0]

    print('Halo selection [%d] objects (m200 >= %.1f).' % (len(subInds),minM200))

    # load mergertree for mapping the subhalos between adjacent snapshots
    mpbs = loadMPBs(sP, subInds, fields=['SnapNum','SubfindID'])

    prevInds = np.zeros( subInds.size, dtype='int32' ) - 1

    for i in range(subInds.size):
        if subInds[i] not in mpbs:
            continue
        mpb = mpbs[subInds[i]]
        w = np.where(mpb['SnapNum'] == sP.snap-1)
        if len(w[0]) == 0:
            continue # skipped sP.snap-1 in the MPB
        prevInds[i] = mpb['SubfindID'][w]

    # restrict to valid matches
    w = np.where(prevInds >= 0)
    print('Using [%d] of [%d] snapshot adjacent matches through the MPBs.' % (len(w[0]),prevInds.size))

    prevInds = prevInds[w]
    subInds = subInds[w]

    # compute a delta(BH_CumEgyInjection_RM) between this snapshot and the last
    bh_egyLow_cur,_,_,_ = simSubhaloQuantity(sP, 'BH_CumEgy_low')

    sP.setSnap(snap-1)
    bh_egyLow_prev,_,_,_ = simSubhaloQuantity(sP, 'BH_CumEgy_low') # erg

    dt_myr = (tage - sP.tage) * 1000
    sP.setSnap(snap)

    bh_dedt_low = (bh_egyLow_cur[subInds] - bh_egyLow_prev[prevInds]) / dt_myr # erg/myr

    w = np.where(bh_dedt_low < 0.0)
    bh_dedt_low[w] = 0.0 # bad MPB track? CumEgy counter should be monotonically increasing

    # sort halo sample based on recent BH energy injection in low-state
    sort_inds = np.argsort(bh_dedt_low)[::-1] # highest to lowest

    r['subInds'] = subInds[sort_inds]
    r['m200'] = m200[r['subInds']]
    r['bh_dedt_low'] = bh_dedt_low

    # get fof halo IDs
    haloInds = groupCat(sP, fieldsSubhalos=['SubhaloGrNr'])['subhalos']
    r['haloInds'] = haloInds[r['subInds']]

    # save cache
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved [%s].' % saveFilename)

    return r

def selection_subbox_overlap(sP, sbNum, sel, verbose=False):
    """ Determine intersection with a halo selection and evolving tracks through a given subbox. """
    path = sP.postPath + 'SubboxSubhaloList/subbox%d_%d.hdf5' % (sbNum, sP.snap)

    with h5py.File(path,'r') as f:
        sbSubIDs = f['SubhaloIDs'][()]
        sbEverInFlag = f['EverInSubboxFlag'][()]

        numInside = sbEverInFlag[sel['subInds']].sum()
        if verbose:
            print('number of selected halos ever inside [subbox %d]: %d' % (sbNum, numInside))

        if numInside == 0:
            return None

        # cross-match to locate target subhalos in these datasets
        sel_inds, subbox_inds = match3(sel['subInds'], sbSubIDs)

        # load remaining datasets
        subboxScaleFac = f['SubboxScaleFac'][()]
        minEdgeDistRedshifts = f['minEdgeDistRedshifts'][()]
        sbMinEdgeDist = f['SubhaloMinEdgeDist'][subbox_inds,:]
        minSBsnap = f['SubhaloMinSBSnap'][()][subbox_inds]
        maxSBsnap = f['SubhaloMaxSBSnap'][()][subbox_inds]

        subhaloPos = f['SubhaloPos'][subbox_inds,:,:]

        # extended information available?
        extInfo = {}
        for key in f:
            if 'SubhaloStars_' in key or 'SubhaloGas_' in key or 'SubhaloBH_' in key:
                extInfo[key] = f[key][subbox_inds,:,:]

        extInfo['mostBoundID'] = f['SubhaloMBID'][subbox_inds,:]

    if verbose:
        z2Ind = np.where(minEdgeDistRedshifts == 2.0)
        subboxRedshift = 1.0/subboxScaleFac - 1

        for i, selInd in enumerate(sel_inds):
            print('[%d] selInd [%3d] subInd [%6d] haloInd [%4d] sbIndex [%6d] m200 [%.1f] minDist(z2->0) = %9.2f snapRange [%4d - %4d] redshift [%.2f - %.2f]' % \
                (i,selInd,sel['subInds'][selInd],sel['haloInds'][selInd],subbox_inds[i],sel['m200'][selInd],sbMinEdgeDist[i,z2Ind],
                    minSBsnap[i],maxSBsnap[i],subboxRedshift[minSBsnap[i]],subboxRedshift[maxSBsnap[i]]))
            #interval = 200
            #print(' x: ', subhaloPos[i,::interval,0])
            #print(' y: ', subhaloPos[i,::interval,1])
            #print(' z: ', subhaloPos[i,::interval,2])
            #print(' redshift: ', subboxRedshift[::interval])

    return sel_inds, subbox_inds, minSBsnap, maxSBsnap, subhaloPos, subboxScaleFac, extInfo

def _getHaloEvoDataOneSnap(snap, sP, haloInds, minSnap, maxSnap, centerPos, scalarFields, loadFields, 
                           histoNames1D, histoNames2D, apertures, limits, histoNbins):
    """ Multiprocessing target: load and process all data for one subbox/normal snap, returning results. """
    sP.setSnap(snap)
    if (sP.isSubbox and snap % 100 == 0) or (not sP.isSubbox): print('snap: ',snap)

    data = {'snap':snap}

    # particle data load
    for ptType in scalarFields.keys():
        data[ptType] = {}
        fieldsToLoad = list(set( ['Coordinates'] + scalarFields[ptType] + loadFields[ptType] )) # unique

        x = snapshotSubset(sP, ptType, fieldsToLoad)

        if x['count'] == 0:
            continue

        # subhalo loop
        for i, haloInd in enumerate(haloInds):
            data_loc = {}
            subPos = centerPos[i,snap,:]

            if snap < minSnap[i] or snap > maxSnap[i]:
                continue

            # localize to this subhalo
            dists_sq = (x['Coordinates'][:,0]-subPos[0])**2 + (x['Coordinates'][:,1]-subPos[1])**2 + (x['Coordinates'][:,2]-subPos[2])**2

            # scalar fields: select relevant particles and save
            w = np.where(dists_sq <= apertures['scalar']**2)

            if len(w[0]) > 0:
                for key in scalarFields[ptType]:
                    if ptType == 'bhs':
                        data_loc[key] = x[key][w].max() # MAX
                    if ptType == 'gas':
                        data_loc[key] = x[key][w].sum() # TOTAL (unused)

            if len(histoNames1D[ptType]) + len(histoNames2D[ptType]) == 0:
                continue

            # common computations
            if ptType == 'gas':
                # first compute subVel using gas
                w = np.where( (dists_sq <= apertures['sfgas']**2) & (x['StarFormationRate'] > 0.0) )
                subVel = np.mean( x['Velocities'][w,:], axis=1 )
                # todo: may need to smooth vel in time?
                # todo: alternatively, use MBID pos/vel evolution
                # or, we have the Potential saved in subboxes, could use particle with min(Potential) inside rad

            rad = np.sqrt(dists_sq) # i.e. 'rad', code units, [ckpc/h]
            radlog = np.log10(rad)
            vrad = sP.units.particleRadialVelInKmS(x['Coordinates'], x['Velocities'], subPos, subVel)

            vrel = sP.units.particleRelativeVelInKmS(x['Velocities'], subVel)
            vrel = np.sqrt( vrel[:,0]**2 + vrel[:,1]**2 + vrel[:,2]**2 )

            vals = {'rad':rad, 'radlog':radlog, 'vrad':vrad, 'vrel':vrel}

            if ptType == 'gas':
                vals['numdens'] = np.log10( x['numdens'] )
                vals['temp'] = np.log10( x['temp_linear'] )
                vals['templinear'] = x['temp_linear']

            # 2D histograms: compute and save
            for histoName in histoNames2D[ptType]:
                xaxis, yaxis, color = histoName.split('_')

                xlim = limits[xaxis]
                ylim = limits[yaxis]

                xvals = vals[xaxis]
                yvals = vals[yaxis]

                if color == 'massfrac':
                    # mass distribution in this 2D plane
                    weight = x['Masses']
                    zz, _, _ = np.histogram2d(xvals, yvals, bins=[histoNbins, histoNbins], range=[xlim,ylim], normed=True, weights=weight)
                else:
                    # each pixel colored according to its mean value of a third quantity
                    weight = vals[color]
                    zz, _, _, _ = binned_statistic_2d(xvals, yvals, weight, 'mean', bins=[histoNbins, histoNbins], range=[xlim,ylim])

                zz = zz.T
                if color != 'vrad':
                    zz = logZeroNaN(zz)

                data_loc[histoName] = zz

            # 1D histograms (and X as a function of Y relationships): compute and save
            for histoName in histoNames1D[ptType]:
                xaxis, yaxis = histoName.split('_')
                xlim  = limits[xaxis]
                xvals = vals[xaxis]

                data_loc[histoName] = np.zeros( (len(apertures['histo1d']), histoNbins), dtype='float32' )

                # loop over apertures (always code units)
                for j, aperture in enumerate(apertures['histo1d']):
                    w = np.where(dists_sq <= aperture**2)

                    if yaxis == 'count':
                        # 1d histogram of a quantity
                        hh, _ = np.histogram(xvals[w], bins=histoNbins, range=xlim, density=True)
                    else:
                        # median yval (i.e. vrad) in bins of xval, which is typically e.g. radius
                        yvals = vals[yaxis]
                        hh, _, _ = binned_statistic(xvals[w], yvals[w], statistic='median', range=xlim, bins=histoNbins)

                    data_loc[histoName][j,:] = hh

            data[ptType][haloInd] = data_loc # add dict for this subhalo to the byPartType dict, with haloInd as key

    return data

def save_haloevo_data(sP, haloInds, haloIndsSnap, centerPos, minSnap, maxSnap, largeLimits=False):
    """ For one or more halos (defined by their evolving centerPos locations), iterate over all subbox/normal snapshots and 
    record a number of properties for each halo at each timestep. minSnap/maxSnap define the range over which to consider 
    each halo. sP can be a fullbox or subbox, which sets the data origin. One save file is made per halo. """
    import multiprocessing as mp
    from functools import partial
    import time

    # config
    scalarFields = {'gas'  :['StarFormationRate'], 
                    'stars':[],
                    'bhs'  :['BH_CumEgyInjection_QM','BH_CumEgyInjection_RM','BH_Mass']}
    if not np.array_equal(haloInds,[296]): # just legacy
        scalarFields['bhs'] += ['BH_Mdot', 'BH_MdotEddington', 'BH_MdotBondi', 'BH_Progs']

    histoNames1D = {'gas'  :['rad_numdens','rad_temp','rad_vrad','rad_vrel','temp_vrad',
                             'radlog_numdens','radlog_temp','radlog_vrad','radlog_vrel',
                             'vrad_count','vrel_count','temp_count'],
                    'stars':[],
                    'bhs'  :[]}
    histoNames2D = {'gas'  :['rad_vrad_massfrac','rad_vrel_massfrac','rad_vrad_templinear',
                             'numdens_temp_massfrac','numdens_temp_vrad',
                             'radlog_vrad_massfrac','radlog_vrel_massfrac','radlog_vrad_templinear'],
                    'stars':['rad_vrad_massfrac','radlog_vrad_massfrac'],
                    'bhs'  :[]}
    loadFields =   {'gas'  :['mass','vel','temp_linear','numdens'],
                    'stars':['mass','vel'],
                    'bhs'  :[]} # anything needed to achieve histograms

    histoNbins = 300

    apertures = {'scalar'  : 30.0 , # code units, within which scalar quantities are accumulated
                 'sfgas'   : 20.0,  # code units, select SFR>0 gas within this aperture to calculate subVel
                 'histo1d' : [10,50,100,1000]} # code units, for1D histograms/relations

    limits = {'rad'     : [0.0, 800.0],
              'radlog'  : [0.0, 3.0],
              'vrad'    : [-400, 800],
              'vrel'    : [0, 800],
              'numdens' : [-8.0, 2.0],
              'temp'    : [3.0, 8.0]}

    if largeLimits:
        # e.g. looking at M_halo > 12 with the action of the low-state BH winds, expand limits
        limits['rad'] = [0, 1200]
        limits['vrad'] = [-1000, 2000]
        limits['vrel'] = [0, 3000]

    # existence check, immediate load and return if so
    sbStr = '_'+sP.variant if 'subbox' in sP.variant else ''
    hashStr = hashlib.sha256('%s_%s_%s_%s_%d_%s_%s' % \
                (str(scalarFields),str(histoNames1D),str(histoNames2D),str(loadFields),
                 histoNbins,str(apertures),str(limits))).hexdigest()[::4]

    savePath = sP.derivPath + '/haloevo/'

    if not isdir(savePath):
        mkdir(savePath)

    savePath = savePath + 'evo_%d_h%d%s_%s.hdf5'

    if len(haloInds) == 1:
        # single halo: try to load and return available data
        data = {}
        saveFilename = savePath % (haloIndsSnap,haloInds[0],sbStr,hashStr)

        if isfile(saveFilename):
            with h5py.File(saveFilename,'r') as f:
                for group in f.keys():
                    data[group] = {}
                    for dset in f[group].keys():
                        data[group][dset] = f[group][dset][()]
            return data

    # thread parallelize by snapshot
    nThreads = 32 if sP.isSubbox else 1 # assume ~full node memory usage when analyzing full boxes
    pool = mp.Pool(processes=nThreads)
    func = partial(_getHaloEvoDataOneSnap, sP=sP, haloInds=haloInds, minSnap=minSnap, maxSnap=maxSnap,
                   centerPos=centerPos, scalarFields=scalarFields, loadFields=loadFields, 
                   histoNames1D=histoNames1D, histoNames2D=histoNames2D, apertures=apertures, 
                   limits=limits, histoNbins=histoNbins)

    #snaps = range( np.min(minSnap),np.max(maxSnap)+1 ) #[2687]
    snaps = range( 1,np.max(maxSnap)+1 ) #[2687]

    start_time = time.time()

    if nThreads > 1:
        results = pool.map(func, snaps)
    else:
        results = []
        for snap in snaps:
            results.append( func(snap) )

    print('[%d] snaps and [%d] halos with [%d] threads took [%.2f] sec' % (len(snaps),len(haloInds),nThreads,time.time()-start_time))

    # save each individually
    numSnaps = centerPos.shape[1]

    for i, haloInd in enumerate(haloInds):
        data = {}

        # allocate a save data structure for this halo alone
        for ptType in scalarFields.keys():
            data[ptType] = {}
            for field in scalarFields[ptType]:
                data[ptType][field] = np.zeros( numSnaps , dtype='float32' )

            for name in histoNames2D[ptType]:
                data[ptType]['histo2d_'+name] = \
                  np.zeros( (numSnaps,histoNbins,histoNbins), dtype='float32' )
            for name in histoNames1D[ptType]:
                data[ptType]['histo1d_'+name] = \
                  np.zeros( (numSnaps,len(apertures['histo1d']),histoNbins), dtype='float32' )

        data['global'] = {}
        data['global']['mask'] = np.zeros( numSnaps, dtype='int16' ) # 1 = in subbox
        data['global']['mask'][minSnap[i] : maxSnap[i] + 1] = 1
        data['limits'] = limits
        data['apertures'] = apertures

        # stamp by snapshot
        for result in results:
            snap = result['snap']

            for ptType in scalarFields.keys():
                # nothing for this halo/ptType combination (i.e. out of minSnap/maxSnap bounds)
                if haloInd not in result[ptType]:
                    continue

                for field in scalarFields[ptType]:
                    if field not in result[ptType][haloInd]: continue
                    data[ptType][field][snap] = result[ptType][haloInd][field]

                for name in histoNames2D[ptType]:
                    if name not in result[ptType][haloInd]: continue
                    data[ptType]['histo2d_'+name][snap,:,:] = result[ptType][haloInd][name]

                for name in histoNames1D[ptType]:
                    if name not in result[ptType][haloInd]: continue
                    data[ptType]['histo1d_'+name][snap,:,:] = result[ptType][haloInd][name]

        # save
        saveFilename = savePath % (haloIndsSnap,haloInd,sbStr,hashStr)
        with h5py.File(saveFilename,'w') as f:
            for key in data:
                group = f.create_group(key)
                for dset in data[key]:
                    group[dset] = data[key][dset]
        print('Saved [%s].' % saveFilename)

    return data

def save_haloevo_data_subbox(sP, sbNum, selInds, minM200=11.5):
    """ For one or more halos, iterate over all subbox snapshots and record a number of 
    properties for those halo at each subbox snap. Halos are specified by selInds, which 
    index the result of selection_subbox_overlap() which intersects the SubboxSubhaloList 
    catalog with the simple mass selection returned by halo_selection(). """
    sel = halo_selection(sP, minM200=minM200)

    sel_inds, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel)

    # indices, position evolution tracks, and min/max subbox snapshots for each
    selInds = iterable(selInds)

    haloInds = sel['haloInds'][sel_inds[selInds]]
    centerPos = subhaloPos[ selInds,:,:] # ndim == 3
    minSnap = minSBsnap[selInds]
    maxSnap = maxSBsnap[selInds]

    # compute and save, or return, time evolution data
    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    largeLimits = False if minM200 == 11.5 else True

    return save_haloevo_data(sP_sub, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)

def save_haloevo_data_fullbox(sP, haloInds):
    """ For one or more halos, iterate over all fullbox snapshots and record a number of 
    properties for those halo at each snasphot. Use SubLink MPB for positioning, extrapolating 
    back to snapshot zero. """
    from cosmo.mergertree import mpbPositionComplete

    posSet  = []
    minSnap = []
    maxSnap = []

    # acquire complete positional tracks at all snapshots
    for haloInd in haloInds:
        halo = groupCatSingle(sP, haloID=haloInd)
        snaps, pos = mpbPositionComplete(sP, halo['GroupFirstSub'])

        posSet.append(pos)
        minSnap.append(0)
        maxSnap.append(sP.snap)

        assert np.array_equal(snaps, range(0,sP.snap+1)) # otherwise handle

    centerPos = np.zeros( (len(posSet),posSet[0].shape[0],posSet[0].shape[1]), dtype=posSet[0].dtype )
    for i, pos in enumerate(posSet):
        centerPos[i,:,:] = pos

    # compute and save, or return, time evolution data
    largeLimits = True if sP.units.codeMassToLogMsun(halo['Group_M_Crit200']) > 12.1 else False

    return save_haloevo_data(sP, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)

def _render_single_subbox_image(snap, sP_sub, minSBsnap, maxSBsnap, subhaloPos):
    """ Multipricessing pool target. """
    # constant config
    method     = 'sphMap'
    axes       = [0,1]
    nPixels    = [600,600]
    hsmlFac    = 2.5
    rotMatrix  = None
    rotCenter  = None
    projType   = 'ortho'
    projParams = {}
    boxSize    = 2000.0 # code units

    if snap < minSBsnap or snap > maxSBsnap:
        return

    sP_sub.setSnap(snap)

    # configure render at this snapshot
    subPos = subhaloPos[snap,:]
    boxCenter = subPos[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering
    boxSizeImg = boxSize * np.array([1.0, 1.0, 1.0]) # same width, height, and depth

    # call gridBox
    partType = 'gas'
    partField = 'coldens_msunkpc2'
    gridBox(sP_sub, method, partType, partField, nPixels, axes, projType, projParams,
        boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

def prerender_subbox_images(sP, sbNum, selInd, minM200=11.5):
    """ Testing subbox. """
    import multiprocessing as mp
    from functools import partial

    assert minM200 == 11.5

    # load
    sel = halo_selection(sP, minM200=minM200)

    _, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel)

    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    # loop over all snapshots of relevance
    snaps = range(minSBsnap.min(),maxSBsnap.max()+1)

    # thread parallelize by snapshot
    nThreads = 4
    pool = mp.Pool(processes=nThreads)
    func = partial(_render_single_subbox_image, sP_sub=sP_sub, minSBsnap=minSBsnap[selInd], 
                   maxSBsnap=maxSBsnap[selInd], subhaloPos=subhaloPos[selInd,:,:])

    if nThreads > 1:
        pool.map(func, snaps)
    else:
        for snap in snaps:
            func(snap)

def vis_subbox_data(sP, sbNum, selInd, extended=False):
    """ Visualize subbox data. 3x2 panel image sequence, or 5x3 if extended == True. """

    def _histo2d_helper(gs,i,pt,xaxis,yaxis,color,clim):
        """ Add one panel of a 2D phase diagram. """
        ax = plt.subplot(gs[i])

        xlim   = data['limits'][xaxis]
        ylim   = data['limits'][yaxis]
        xlabel = labels[xaxis]
        ylabel = labels[yaxis]
        clabel = labels[color]

        colNorm = False
        if color == 'massfracnorm':
            colNorm = True
            color = 'massfrac'

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # plot 2d histogram
        key = 'histo2d_%s_%s_%s' % (xaxis,yaxis,color)
        zz = data[pt][key][snap,:,:]

        if colNorm:
            zz = 10.0**zz
            for i in range(zz.shape[1]):
                if np.count_nonzero(np.isfinite(zz[:,i])) == 0: continue
                colMax = np.nanmax(zz[:,i])
                if np.isfinite(colMax): zz[:,i] /= colMax
            zz = logZeroNaN(zz)

        norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

        # colorbar
        cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.ax.set_ylabel(clabel)

        # legend (lower right)
        y0 = 0.04
        yd = 0.04

        x0 = 0.85 if extended else 0.89
        if xaxis in ['rad','radlog'] and yaxis == 'vrel':
            # lower right is occupied, move to upper right
            y0 = 1 - y0
            yd = -yd

        legend_labels = ['snap = %6d' % snap, 'zred  = %6.3f' % redshifts[snap], 't/gyr = %6.3f' % tage[snap]]
        textOpts = {'fontsize':fontsizeTime, 'horizontalalignment':'center', 'verticalalignment':'center'}
        for j, label in enumerate(legend_labels):
            ax.text(x0, y0+yd*j, label, transform=ax.transAxes, **textOpts)

        return ax

    def _histo1d_helper(gs,i,pt,xaxis,yaxis,ylim):
        """ Add one panel of a 1D profile, all the available apertures by default. """
        ax = plt.subplot(gs[i])
        
        key    = 'histo1d_%s_%s' % (xaxis,yaxis)
        xlim   = data['limits'][xaxis]
        xlabel = labels[xaxis]
        ylabel = labels[yaxis]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # reconstruct bin midpoints
        histoNbins = data[pt][key].shape[2]
        xx = np.linspace(xlim[0], xlim[1], histoNbins+1)[:-1] + (xlim[1]-xlim[0])/histoNbins/2

        # plot
        for j, aperture in enumerate(data['apertures']['histo1d']):
            if xaxis in ['rad','radlog'] and j < len(data['apertures']['histo1d'])-1:
                continue # no need for radially restricted radial profiles

            yy = data[pt][key][snap,j,:]
            if yaxis == 'count': yy = logZeroNaN(yy)

            label = 'r < %04d ckpc/h' % aperture if xaxis not in ['rad','radlog'] else ''
            ax.plot(xx, yy, '-', lw=lw, alpha=1.0, label=label)

        # for radial profiles, overplot a few at selected moments in time as they are passed
        if xaxis in ['rad','radlog']:
            cur_redshift = redshifts[snap]
            freeze_redshifts = [6.0, 2.0, 1.0]
            aperture_ind = len(data['apertures']['histo1d']) - 1 # full profile

            for z in freeze_redshifts:
                if cur_redshift <= z:
                    _, z_ind = closest(redshifts, z)
                    yy = data[pt][key][z_ind,aperture_ind,:]
                    ax.plot(xx, yy, '-', lw=lw, alpha=0.3, label='z = %d' % z)

        # legend
        ax.legend(loc='upper right', prop={'size':fontsizeLegend})
        return ax

    def _image_helper(gs, i, pt, field, axes, boxSize):
        """ Add one panel of a gridBox() rendered image projection. """
        ax = plt.subplot(gs[i])

        # render config
        method     = 'sphMap'
        nPixels    = [600,600]
        hsmlFac    = 2.5
        rotMatrix  = None
        rotCenter  = None
        projType   = 'ortho'
        projParams = {}
        boxSizeImg = np.array([boxSize,boxSize,boxSize])
        boxCenter  = subhaloPos[selInd,snap,:]

        # load/render
        sP_sub = simParams(res=sP.res, run=sP.run, snap=snap, variant='subbox%d' % sbNum)

        boxCenter  = boxCenter[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering
        grid, config = gridBox(sP_sub, method, pt, field, nPixels, axes, projType, projParams, 
                               boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

        # plot config
        valMinMax = [5.5, 8.0] # todo generalize
        if boxSize >= 500.0: valMinMax[0] -= 1.0
        cmap = loadColorTable(config['ctName'], valMinMax=valMinMax)

        extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
                   boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]
        extent[0:2] -= boxCenter[0] # make coordinates relative
        extent[2:4] -= boxCenter[1]

        ax.set_xlabel( ['x','y','z'][axes[0]] + ' [ckpc/h]')
        ax.set_ylabel( ['x','y','z'][axes[1]] + ' [ckpc/h]')

        # plot
        plt.imshow(grid, extent=extent, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])
        ax.autoscale(False)
        plt.clim(valMinMax)

        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel(config['label'])

    def _lineplot_helper(gs,i):
        """ Add one panel of a composite line plot of various component masses and BH energies. """
        ax = plt.subplot(gs[i])
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Masses [log M$_{\\rm sun}$]')

        # blackhole mass
        yy = sP.units.codeMassToLogMsun( data['bhs']['BH_Mass'] )
        l, = ax.plot(redshifts[w], yy[w], '-', lw=lw, alpha=0.5, label='BH Mass')
        ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=l.get_color())
        ax.set_xlim(ax.get_xlim()[::-1]) # time increasing to the right
        for t in ax.get_yticklabels(): t.set_color(l.get_color())

        # additional info from extended SubboxSubhaloList?
        linestyles = ['-','--',':'] # for the three apertures
        apertures = ['30pkpc', '30ckpc/h', '50ckpc/h']

        for key in ['SubhaloGas_Mass','SubhaloStars_Mass']:
            if key in extInfo:
                c = ax._get_lines.prop_cycler.next()['color']
                for j, aperture in enumerate(apertures):
                    yy = sP.units.codeMassToLogMsun( extInfo[key][selInd,j,:])
                    yy -= 3.0 # 3 dex offset to get within range of M_BH
                    label = key.split('Subhalo')[1] + '/$10^3$' if j == 0 else ''
                    l, = ax.plot(redshifts[w], yy[w], linestyles[j], lw=lw, alpha=0.5, label=label, color=c)
                    ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=c)

        # blackhole energetics
        ax2 = ax.twinx()
        ax2.set_ylabel('BH $\Delta$ E$_{\\rm low}$ (dotted), E$_{\\rm high}$ (solid) [ log erg ]')
        c = ax._get_lines.prop_cycler.next()['color']
        ax2.plot(redshifts[w], dy_high[w], '-', lw=lw, alpha=0.7, color=c)
        ax2.plot(redshifts[w], dy_low[w], ':', lw=lw, alpha=0.7, color=c)
        ax2.plot(redshifts[snap], dy_high[snap], 'o', markersize=14.0, color=c, alpha=0.6)
        ax2.plot(redshifts[snap], dy_low[snap], 'o', markersize=14.0, color=c, alpha=0.6)
        for t in ax2.get_yticklabels(): t.set_color(c)

        handles, labels = ax.get_legend_handles_labels()
        sExtra = [plt.Line2D((0,1), (0,0), color='black', marker='', lw=lw, linestyle=ls) for ls in linestyles]
        ax.legend(handles+sExtra, labels+apertures, loc='lower right', prop={'size':fontsizeLegend})

    def _lineplot_helper2(gs,i):
        """ Add one panel of a composite line plot SFR(t) and BH_Mdot(t). """
        ax = plt.subplot(gs[i])
        ax.set_xlabel('Redshift')
        ax.set_ylabel('SFR or $\dot{M}_{\\rm BH}$ [log M$_{\\rm sun}$ / yr]')

        # BH mdot
        c = ax._get_lines.prop_cycler.next()['color']
        ax.plot(redshifts[w], bh_mdot[w], '-', lw=lw, alpha=0.7, color=c, label='BH_Mdot')
        ax.plot(redshifts[snap], bh_mdot[snap], 'o', markersize=14.0, color=c, alpha=0.6)
        ax.set_xlim(ax.get_xlim()[::-1]) # time increasing to the right

        # gas SFR
        linestyles = ['-','--',':'] # for the three apertures
        apertures = ['30pkpc', '30ckpc/h', '50ckpc/h']

        key = 'SubhaloGas_SFR'
        if key in extInfo:
            c = ax._get_lines.prop_cycler.next()['color']
            for j, aperture in enumerate(apertures):
                yy = logZeroNaN( extInfo[key][selInd,j,:] )
                label = 'Gas_SFR' if j == 0 else ''
                l, = ax.plot(redshifts[w], yy[w], linestyles[j], lw=lw, alpha=0.5, label=label, color=c)
                ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=c)

        handles, labels = ax.get_legend_handles_labels()
        sExtra = [plt.Line2D((0,1), (0,0), color='black', marker='', lw=lw, linestyle=ls) for ls in linestyles]
        ax.legend(handles+sExtra, labels+apertures, loc='lower right', prop={'size':fontsizeLegend})

    # config
    lw = 2.0
    fontsizeTime   = 22.0
    fontsizeLegend = 16.0
    cmap = loadColorTable('viridis')

    labels = {'radlog'       : 'Radius [log ckpc/h]',
              'rad'          : 'Radius [ckpc/h]',
              'vrad'         : 'Radial Velocity [km/s]',
              'vrel'         : 'Halo-Rel Velocity Mag [km/s]',
              'numdens'      : 'Gas Density [log cm$^{-3}$]',
              'temp'         : 'Gas Temperature [log K]',
              'templinear'   : 'Gas Temperature [log K]',
              'massfrac'     : 'Relative Mass Fraction [log]',
              'massfracnorm' : 'Conditional Mass Fraction [log]',
              'count'        : 'PDF'}

    lim1 = [-4.0, -1.5] # massfrac
    lim2 = [-2.5, 0.0] # massfracnorm
    lim3 = [-5.0, -1.5] # histo1d
    lim4 = [4.0, 6.5] # temp

    # load
    data = save_subbox_data(sP, sbNum, selInd)

    sel = halo_selection(sP, minM200=11.5)
    _, _, _, _, subhaloPos, subboxScaleFac, extInfo = selection_subbox_overlap(sP, sbNum, sel)

    # derive blackhole differential energetics (needed for _lineplot_helpers)
    redshifts = 1.0 / subboxScaleFac - 1.0
    tage = sP.units.redshiftToAgeFlat(redshifts)
    dtyr = np.diff(tage * 1e9, n=1)
    dtyr = np.append(dtyr, dtyr[-1])

    yy_high = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_QM'], log=True)
    yy_low  = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_RM'], log=True)
    dy_high = np.diff(10.0**yy_high, n=1) # dy[i] is delta_E between snap=i and snap=i+1
    dy_low  = np.diff(10.0**yy_low, n=1)
    dy_high = logZeroNaN( np.append(dy_high, dy_high[-1]) ) # restore to original size, then log
    dy_low  = logZeroNaN( np.append(dy_low, dy_low[-1]) )
    bh_mdot = np.diff(sP.units.codeMassToMsun(data['bhs']['BH_Mass']) / dtyr[::-1], n=1)
    bh_mdot = logZeroNaN( np.append(bh_mdot, bh_mdot[-1]) )
    wmax = np.where(bh_mdot == np.nanmax(bh_mdot)) # spurious spike due to seed event
    bh_mdot[wmax] = np.nan

    # make plot booklets / movies
    w = np.where(data['global']['mask'] == 1)

    for snap in w[0]: #[2687]:

        # typical configuration (3x2 panels, 1.5 aspect ratio)
        if not extended:
            # plot setup
            fig = plt.figure(figsize=[38.4, 21.6]) # produce 3840x2160px image at dpi==100
            gs = gridspec.GridSpec(2,3)

            # upper left: histo2d [radlog,vrad] massfrac
            _histo2d_helper(gs,i=0,pt='gas',xaxis='radlog',yaxis='vrad',color='massfracnorm',clim=lim2)

            # upper center: histo2d [radlog,vrad] mean temperature
            _histo2d_helper(gs,i=1,pt='gas',xaxis='radlog',yaxis='vrad',color='templinear',clim=lim4)

            # upper right: [rad,vrelmag] mass frac
            _histo2d_helper(gs,i=2,pt='gas',xaxis='rad',yaxis='vrel',color='massfracnorm',clim=lim2)

            # lower left: 2dhisto [dens,temp] mean vrad
            _histo2d_helper(gs,i=3,pt='gas',xaxis='numdens',yaxis='temp',color='vrad',clim=[-100,300])

            # lower center: gas dens image, 200 kpc, xy
            _image_helper(gs,i=4,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=200.0)

            # lower right: (top) masses/blackhole energetics, (bottom) vrad histograms
            gs_local = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[5], height_ratios=[2,1], hspace=0.20)
            _lineplot_helper(gs_local,i=0)
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='vrad',yaxis='count',ylim=lim3)

        # extended configuration (5x3 panels, 1.67 aspect ratio)
        if extended:
            # plot setup
            fig = plt.figure(figsize=[38.4/0.75, 19.8/0.75]) # produce 3840x1920px image at dpi==75
            gs = gridspec.GridSpec(3,5)

            # upper col0: histo2d [rad,vrad] massfrac
            _histo2d_helper(gs,i=0,pt='gas',xaxis='radlog',yaxis='vrad',color='massfracnorm',clim=lim2)

            # upper col1: histo2d [rad,vrad] mean temperature
            _histo2d_helper(gs,i=1,pt='gas',xaxis='radlog',yaxis='vrad',color='templinear',clim=lim4)

            # upper col2: histo2d [rad,vrelmag] mass frac
            _histo2d_helper(gs,i=2,pt='gas',xaxis='rad',yaxis='vrel',color='massfracnorm',clim=lim2)

            # upper col3: gas radial density profile, vrad vs. temp relation
            gs_local = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.35)
            _histo1d_helper(gs_local,i=0,pt='gas',xaxis='radlog',yaxis='numdens',ylim=[-6.0,1.0])
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='temp',yaxis='vrad',ylim=[-200, 400])

            # upper col4: SFR and BH Mdot
            _lineplot_helper2(gs,i=4)

            # center col0: 2dhisto [dens,temp] mass frac
            _histo2d_helper(gs,i=5,pt='gas',xaxis='numdens',yaxis='temp',color='massfrac',clim=[-4.0,-0.5])

            # center col1: 2dhisto [rad,vrad] temp
            _histo2d_helper(gs,i=6,pt='gas',xaxis='rad',yaxis='vrad',color='templinear',clim=lim4)

            # center col2: 2dhisto [rad,vrad] mass frac (stars)
            _histo2d_helper(gs,i=7,pt='stars',xaxis='rad',yaxis='vrad',color='massfracnorm',clim=lim2)

            # center col3: gas vrad 1d histograms
            _histo1d_helper(gs,i=8,pt='gas',xaxis='vrad',yaxis='count',ylim=lim3)
           
            # center col4: gas vrel/temp 1d histograms
            gs_local = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[9], hspace=0.25)
            _histo1d_helper(gs_local,i=0,pt='gas',xaxis='vrel',yaxis='count',ylim=lim3)
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='temp',yaxis='count',ylim=[-2.0, 0.5])

            # bottom col0: 2dhisto [dens,temp] mean vrad
            _histo2d_helper(gs,i=10,pt='gas',xaxis='numdens',yaxis='temp',color='vrad',clim=[-100,300])

            # bottom col1: gas dens image, 200 kpc, xz
            _image_helper(gs,i=11,pt='gas',field='coldens_msunkpc2',axes=[0,2],boxSize=200.0)

            # bottom col2: gas dens image, 200 kpc, xy
            _image_helper(gs,i=12,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=200.0)

            # bottom col3: gas dens image, 2000 kpc, xy
            _image_helper(gs,i=13,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=2000.0)

            # bottom col4: line plot, masses and blackhole diagnostics
            _lineplot_helper(gs,i=14)

        # finish
        fig.tight_layout()
        fig.savefig('vis_%s_sbNum-%d_selInd-%d%s_%04d.png' % \
            (sP.simName,sbNum,selInd,'_extended' if extended else '',snap), 
            dpi=(75 if extended else 100))
        plt.close(fig)

def explore_vrad_selection(sP):
    """ Testing. A variety of plots looking at halo-centric gas/wind radial velocities. For entire selection. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    sel = halo_selection(sP, minM200=12.0)
    haloIndsPlot = sel['haloInds']

    # plot: booklet of 1D vrad profiles
    if 0:
        numPerPage = 5
        numPages = haloIndsPlot.size / numPerPage
        pdf = PdfPages('histo1d_vrad.pdf')

        for i in range(numPages):
            haloIDs = [haloIndsPlot[(i+0)*numPerPage : (i+1)*numPerPage]] # fof scope
            plotHistogram1D([sP], haloIDs=haloIDs, ptType='gas', ptProperty='vrad', 
                sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

        pdf.close()

    # plot: booklets of 2D phase diagrams
    if 0:
        pdf = PdfPages('phase2d_vrad_numdens.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 1:
        pdf = PdfPages('phase2d_vrelmag_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, 
                yQuant='vrelmag', ylim=[0,3000], nBins=nBins, clim=clim)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad_kpc_linear.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloID=haloID, pdf=pdf, 
                yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_rad_sfreq0.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, sfreq0=True, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_rad_wind.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='wind_real', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_temp_c=rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['rad'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_numdens_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_dens_temp_c=vrad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
                meancolors=['vrad'], weights=None, haloID=haloID, clim=vrad_lim, pdf=pdf)
        pdf.close()

    if haloInds is not None:
        pdf.close()

def explore_vrad_halos(sP, haloIndsPlot):
    """ Testing. A variety of plots looking at halo-centric gas/wind radial velocities. For input halos. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    pdf = PdfPages('halos_%s.pdf' % ('-'.join([str(i) for i in haloIndsPlot])))

    # plot: booklet of 1D vrad profiles
    numPerPage = 5
    numPages = len(haloIndsPlot) / numPerPage

    for i in range(numPages):
        haloIDs = [haloIndsPlot[(i+0)*numPerPage : (i+1)*numPerPage]] # fof scope
        plotHistogram1D([sP], haloIDs=haloIDs, ptType='gas', ptProperty='vrad', 
            sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

    # plot: booklets of 2D phase diagrams
    for haloID in haloIndsPlot:
        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, 
            yQuant='vrelmag', ylim=[0,3000], nBins=nBins, clim=clim)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloID=haloID, pdf=pdf, 
            yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, sfreq0=True, **commonOpts)

        plotPhaseSpace2D(sP, partType='wind_real', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['rad'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
            meancolors=['vrad'], weights=None, haloID=haloID, clim=vrad_lim, pdf=pdf)

    pdf.close()
   
def outflow_rates(sP):
    pass

def sample_comparison_z2_sins_ao(sP):
    """ Compare available galaxies vs. the SINS-AO sample of ~35 systems. """
    from util.loadExtern import foersterSchreiber2018

    # config
    xlim = [9.0, 12.0]
    ylim = [-2.5, 4.0]

    msize = 4.0 # marker size for simulated points
    binSize = 0.2 # in M* for median line
    fullSubhaloSFR = True # use total SFR in subhalo, otherwise within 2rhalf

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
    ax = fig.add_subplot(111)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel('Star Formation Rate [ log M$_{\\rm sun}$ / yr ]')
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')

    # load simulation points
    sfrField = 'SubhaloSFR' if fullSubhaloSFR else 'SubhaloSFRinRad'
    fieldsSubhalos = ['SubhaloMassInRadType',sfrField,'central_flag']

    gc = groupCat(sP, fieldsSubhalos=fieldsSubhalos)

    xx_code = gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')]
    xx = sP.units.codeMassToLogMsun( xx_code )

    yy = gc['subhalos'][sfrField]

    # centrals only above some mass limit
    with np.errstate(invalid='ignore'):
        ww = np.where( (xx > xlim[0]+0.2) & gc['subhalos']['central_flag'] )

    w_nonzero = np.where(yy[ww] > 0.0)
    w_zero = np.where(yy[ww] == 0.0)

    l, = ax.plot(xx[ww][w_nonzero], np.log10(yy[ww][w_nonzero]), 'o', markersize=msize, label=sP.simName)
    ax.plot(xx[ww][w_zero], np.zeros(len(w_zero[0])) + ylim[0]+0.1, 'D', markersize=msize, color=l.get_color(), alpha=0.5)

    # median line and 1sigma band
    xm, ym, sm = running_median(xx[ww][w_nonzero],np.log10(yy[ww][w_nonzero]),binSize=binSize,skipZeros=True)
    l, = ax.plot(xm[:-1], ym[:-1], '-', lw=2.0, alpha=0.4, color=l.get_color())

    y_down = np.array(ym[:-1]) - sm[:-1]
    y_up   = np.array(ym[:-1]) + sm[:-1]
    ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

    # observational points (put on top at the end)
    fs = foersterSchreiber2018()
    l1, = ax.plot(fs['Mstar'], np.log10(fs['SFR']), 's', color='#444444', label=fs['label'])

    # second legend
    legend2 = ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('sample_comparison_%s_sfrFullSub=%s.pdf' % (sP.simName,fullSubhaloSFR))
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def paperPlots():
    """ Construct all the final plots for the paper. """
    redshift = 0.73 # last snapshot, 58

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    #TNG50_2 = simParams(res=1080,run='tng',redshift=redshift) # on /isaac/ptmp/
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)

    if 0:
        # vrad plots, entire selection
        explore_vrad(TNG50)

    if 0:
        # vrad plots, single halo
        subInd = 389836 # first one in subbox0 intersecting with >11.5 selection
        haloInd = groupCatSingle(TNG50, subhaloID=subInd)['SubhaloGrNr']
        explore_vrad_halos(TNG50, haloIndsPlot=[haloInd])

    if 0:
        # print out subbox intersections with selection
        sel = halo_selection(TNG50, minM200=11.5)
        for sbNum in [0,1,2]:
            _ = selection_subbox_overlap(TNG50, sbNum, sel, verbose=True)

    if 0:
        # save data (phase diagrams) through time
        save_haloevo_data_subbox(TNG50, sbNum=0, selInds=[2,3])
        #prerender_subbox_images(TNG50, sbNum=0, selInd=0)

    if 0:
        # vis
        vis_subbox_data(TNG50, sbNum=0, selInd=0, extended=False)

    if 1:
        # load fullbox test, first 20 halos of 12.0 selection all at once...
        sel = halo_selection(TNG50, minM200=12.0)
        save_haloevo_data_fullbox(TNG50, haloInds=sel['haloInds'][0:20])

    if 0:
        # sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG50.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG50)
