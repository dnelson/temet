"""
plot/oxygen.py
  Plots: Outflows paper (TNG50 presentation).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d
from os.path import isfile

from util import simParams
from plot.config import *
from util.helper import loadColorTable, running_median, logZeroNaN, iterable
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.mergertree import loadMPBs
from plot.general import plotHistogram1D, plotPhaseSpace2D
from plot.quantities import simSubhaloQuantity
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from cosmo.util import cenSatSubhaloIndices
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

def _get_subbox_data_onesnap(snap, sP_sub, minSBsnap, maxSBsnap, subhaloPos, scalarFields, loadFields, 
                             histoNames1D, histoNames2D, apertures, limits, histoNbins):
    """ Multiprocessing pool target, load and process all data for one subbox snapshot, returning results. """
    if snap < minSBsnap or snap > maxSBsnap:
        return None

    sP_sub.setSnap(snap)
    if snap % 100 == 0: print(snap)

    subPos = subhaloPos[snap,:]
    subVel = 0 # derived below, or could load from extended SubboxSubhaloList if available

    data = {'snap':snap}

    # particle data load
    for ptType in scalarFields.keys():
        data[ptType] = {}
        fieldsToLoad = list(set( ['Coordinates'] + scalarFields[ptType] + loadFields[ptType] )) # unique

        x = snapshotSubset(sP_sub, ptType, fieldsToLoad)

        if x['count'] == 0:
            continue

        # localize to this subhalo
        dists_sq = (x['Coordinates'][:,0]-subPos[0])**2 + (x['Coordinates'][:,1]-subPos[1])**2 + (x['Coordinates'][:,2]-subPos[2])**2

        # scalar fields: select relevant particles and save
        w = np.where(dists_sq <= apertures['scalar']**2)

        if len(w[0]) > 0:
            for key in scalarFields[ptType]:
                if ptType == 'bhs':
                    data[ptType][key] = x[key][w].max() # MAX
                if ptType == 'gas':
                    data[ptType][key] = x[key][w].sum() # TOTAL (unused)

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
        vrad = sP_sub.units.particleRadialVelInKmS(x['Coordinates'], x['Velocities'], subPos, subVel)

        vrel = sP_sub.units.particleRelativeVelInKmS(x['Velocities'], subVel)
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

            data[ptType][histoName] = zz

        # 1D histograms (and X as a function of Y relationships): compute and save
        for histoName in histoNames1D[ptType]:
            xaxis, yaxis = histoName.split('_')
            xlim  = limits[xaxis]
            xvals = vals[xaxis]

            if yaxis == 'count':
                # 1d histogram within an aperture
                w = np.where(dists_sq <= apertures['histo1d']**2)
                hh, _ = np.histogram(xvals[w], bins=histoNbins, range=xlim, normed=True)
            else:
                # median yval (i.e. vrad) in bins of xval, which is typically e.g. radius
                yvals = vals[yaxis]
                hh, _, _ = binned_statistic(xvals, yvals, statistic='median', range=xlim, bins=histoNbins)

            data[ptType][histoName] = hh

    return data

def save_subbox_data(sP, sbNum, selInd):
    """ Testing subbox. """
    import multiprocessing as mp
    from functools import partial
    import time

    minM200 = 11.5
    sel = halo_selection(sP, minM200=minM200)

    # for one halo, track through full subbox range and save lots of interesting data
    assert minM200 == 11.5

    _, _, minSBsnap, maxSBsnap, subhaloPos, subboxScaleFac, _ = selection_subbox_overlap(sP, sbNum, sel)

    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    # config
    scalarFields = {'gas'  :['StarFormationRate'], 
                    'stars':[],
                    'bhs'  :['BH_CumEgyInjection_QM','BH_CumEgyInjection_RM','BH_Mass']}
    histoNames1D = {'gas'  :['rad_numdens','rad_temp','rad_vrad','rad_vrel','temp_vrad',
                             'radlog_numdens','radlog_temp','radlog_vrad','radlog_vrel',
                             'vrad_count','vrel_count','temp_count'],
                    'stars':[],
                    'bhs'  :[]}
    histoNames2D = {'gas'  :['rad_vrad_massfrac','rad_vrel_massfrac','rad_vrad_templinear','numdens_temp_massfrac','numdens_temp_vrad',
                             'radlog_vrad_massfrac','radlog_vrel_massfrac','radlog_vrad_templinear'],
                    'stars':['rad_vrad_massfrac','radlog_vrad_massfrac'],
                    'bhs'  :[]}
    loadFields =   {'gas'  :['mass','vel','temp_linear','numdens'],
                    'stars':['mass','vel'],
                    'bhs'  :[]} # anything needed to achieve histograms

    histoNbins = 300

    apertures = {'scalar'  : 30.0 , # code units, within which scalar quantities are accumulated
                 'sfgas'   : 20.0,  # code units, select SFR>0 gas within this aperture to calculate subVel
                 'histo1d' : 100.0} # code units, within which 1D histograms are calculated

    limits = {'rad'     : [0.0, 800.0],
              'radlog'  : [0.0, 3.0],
              'vrad'    : [-400, 800],
              'vrel'    : [0, 800],
              'numdens' : [-8.0, 2.0],
              'temp'    : [3.0, 8.0]}

    if minM200 == 12.0:
        # looking at M_halo > 12 with the action of the low-state BH winds
        limits['rad'] = [0, 1200]
        limits['vrad'] = [-1000, 2000]
        limits['vrel'] = [0, 3000]

    # existence check, immediate load and return if so
    data = {}

    saveFilename = sP.derivPath + 'subhalo_evo_sb%d_sel%d.hdf5' % (sbNum,selInd)
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for group in f.keys():
                data[group] = {}
                for dset in f[group].keys():
                    data[group][dset] = f[group][dset][()]
        return data

    # thread parallelize by snapshot
    nThreads = 32
    pool = mp.Pool(processes=nThreads)
    func = partial(_get_subbox_data_onesnap, sP_sub=sP_sub, minSBsnap=minSBsnap[selInd], maxSBsnap=maxSBsnap[selInd],
                   subhaloPos=subhaloPos[selInd,:,:], scalarFields=scalarFields, loadFields=loadFields, 
                   histoNames1D=histoNames1D, histoNames2D=histoNames2D, apertures=apertures, 
                   limits=limits, histoNbins=histoNbins)

    snaps = range(minSBsnap.min(),maxSBsnap.max()+1) #[2687]

    start_time = time.time()

    if nThreads > 1:
        results = pool.map(func, snaps)
    else:
        results = []
        for snap in snaps:
            results.append( func(snap) )

    print('[%d] snapshots with [%d] threads took [%.2f] sec' % (len(snaps),nThreads,time.time()-start_time))

    # allocate
    for ptType in scalarFields.keys():
        data[ptType] = {}
        for field in scalarFields[ptType]:
            data[ptType][field] = np.zeros( subboxScaleFac.size , dtype='float32' )

        for name in histoNames2D[ptType]:
            data[ptType]['histo2d_'+name] = np.zeros( (subboxScaleFac.size,histoNbins,histoNbins), dtype='float32' )
        for name in histoNames1D[ptType]:
            data[ptType]['histo1d_'+name] = np.zeros( (subboxScaleFac.size,histoNbins), dtype='float32' )

    data['global'] = {}
    data['global']['mask'] = np.zeros( subboxScaleFac.size, dtype='int16' ) # 1 = in subbox
    data['global']['mask'][minSBsnap[selInd] : maxSBsnap[selInd] + 1] = 1
    data['limits'] = limits

    # stamp
    for result in results:
        if result is None:
            continue

        snap = result['snap']

        for ptType in scalarFields.keys():
            for field in scalarFields[ptType]:
                if field not in result[ptType]: continue
                data[ptType][field][snap] = result[ptType][field]

            for name in histoNames2D[ptType]:
                if name not in result[ptType]: continue
                data[ptType]['histo2d_'+name][snap,:,:] = result[ptType][name]

            for name in histoNames1D[ptType]:
                if name not in result[ptType]: continue
                data[ptType]['histo1d_'+name][snap,:] = result[ptType][name]

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in data:
            group = f.create_group(key)
            for dset in data[key]:
                group[dset] = data[key][dset]
    print('Saved [%s].' % saveFilename)

    return data

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

def vis_subbox_data(sP, sbNum, selInd):
    """ Visualize subbox data. """

    def _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i,legendTopRight=False):
        """ Add one panel of a 2D phase diagram. """
        ax = fig.add_subplot(nRows,nCols,i+1)

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
        if legendTopRight:
            # lower right is occupied, move to upper right
            y0 = 1 - y0
            yd = -yd

        legend_labels = ['snap = %6d' % snap, 'zred  = %6.3f' % redshifts[snap], 't/gyr = %6.3f' % tage[snap]]
        for j, label in enumerate(legend_labels):
            ax.text(0.89, y0+yd*j, label, fontsize=22.0, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        return ax

    def _histo1d_helper(pt,xaxis,yaxis,ylim,i):
        """ Add one panel of a 1D profile. """
        ax = fig.add_subplot(nRows,nCols,i+1)
        
        key    = 'histo1d_%s_%s' % (xaxis,yaxis)
        xlim   = data['limits'][xaxis]
        xlabel = labels[xaxis]
        ylabel = labels[yaxis]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # reconstruct bin midpoints
        histoNbins = data[pt][key].shape[1]
        xx = np.linspace(xlim[0], xlim[1], histoNbins+1)[:-1] + (xlim[1]-xlim[0])/histoNbins/2

        # plot
        yy = data[pt][key][snap,:]
        if yaxis == 'count': yy = logZeroNaN(yy)

        ax.plot(xx, yy, '-', label=ylabel)

        return ax

    def _image_helper(partType, partField, axes, boxSize, i):
        """ Helper. """
        method     = 'sphMap'
        nPixels    = [600,600]
        hsmlFac    = 2.5
        rotMatrix  = None
        rotCenter  = None
        projType   = 'ortho'
        projParams = {}
        boxSizeImg = np.array([boxSize,boxSize,boxSize])
        boxCenter  = subhaloPos[selInd,snap,:]

        sP_sub = simParams(res=sP.res, run=sP.run, snap=snap, variant='subbox%d' % sbNum)

        boxCenter  = boxCenter[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering
        grid, config = gridBox(sP_sub, method, partType, partField, nPixels, axes, projType, projParams, 
                               boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

        valMinMax = [5.5, 8.0] # todo generalize
        if boxSize >= 500.0: valMinMax[0] -= 1.0
        cmap = loadColorTable(config['ctName'], valMinMax=valMinMax)

        extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
                   boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]
        extent[0:2] -= boxCenter[0] # make coordinates relative
        extent[2:4] -= boxCenter[1]

        ax = fig.add_subplot(nRows,nCols,i+1)
        ax.set_xlabel( ['x','y','z'][axes[0]] + ' [ckpc/h]')
        ax.set_ylabel( ['x','y','z'][axes[1]] + ' [ckpc/h]')
        plt.imshow(grid, extent=extent, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])
        ax.autoscale(False)
        plt.clim(valMinMax)

        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel(config['label'])

    # config
    nRows = 2
    nCols = 3

    labels = {'radlog'       : 'Radius [log ckpc/h]',
              'rad'          : 'Radius [ckpc/h]',
              'vrad'         : 'Radial Velocity [km/s]',
              'vrel'         : 'Halo-Rel Velocity Mag [km/s]',
              'numdens'      : 'Gas Number Density [log cm$^{-3}$]',
              'temp'         : 'Gas Temperature [log K]',
              'templinear'   : 'Gas Temperature [log K]',
              'massfrac'     : 'Relative Mass Fraction [log]',
              'massfracnorm' : 'Conditional Mass Fraction [log]',
              'count'        : 'PDF'}

    # load
    data = save_subbox_data(sP, sbNum, selInd)

    sel = halo_selection(sP, minM200=11.5)
    sel_inds, subbox_inds, minSBsnap, maxSBsnap, subhaloPos, subboxScaleFac, extInfo = selection_subbox_overlap(sP, sbNum, sel)

    redshifts = 1.0 / subboxScaleFac - 1.0
    tage = sP.units.redshiftToAgeFlat(redshifts)

    # derive blackhole differential energetics
    yy_high = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_QM'], log=True)
    yy_low  = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_RM'], log=True)
    dy_high = np.diff(10.0**yy_high, n=1) # dy[i] is delta_E between snap=i and snap=i+1
    dy_low  = np.diff(10.0**yy_low, n=1)
    dy_high = logZeroNaN( np.append(dy_high, np.nan) ) # restore to original size, then log
    dy_low  = logZeroNaN( np.append(dy_low, np.nan) )

    # make plot booklets / movies
    w = np.where(data['global']['mask'] == 1)

    for snap in [2687]: #w[0][::-1]:
        print(snap)

        # plot setup
        # todo use Gridspec: https://stackoverflow.com/questions/37737538/merge-matplotlib-subplots-with-shared-x-axis
        cmap = loadColorTable('viridis')
        lw = 2.0
        fig = plt.figure(figsize=[38.4, 21.6]) # produce 3840x2160px image at dpi==100

        # upper right: histo2d [rad,vrad] massfrac
        pt    = 'gas'
        xaxis = 'radlog'
        yaxis = 'vrad'
        color = 'massfracnorm'
        clim  = [-2.5, 0.0] #[-4.0, -1.5] for colNorm==False

        _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=0)

        # upper center: histo2d [rad,vrad] mean temperature
        pt    = 'gas'
        xaxis = 'radlog'
        yaxis = 'vrad'
        color = 'templinear'
        clim  = [4.0, 6.5]

        _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=1)

        # upper right: [vrad,vrelmag] mass frac
        pt    = 'gas'
        xaxis = 'rad'
        yaxis = 'vrel'
        color = 'massfracnorm'
        clim  = [-2.5, 0.0]

        _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=2,legendTopRight=True)

        # lower left
        if 0:
            # 2dhisto [rad,vrad] mass frac (stars)
            pt    = 'stars'
            xaxis = 'rad'
            yaxis = 'vrad'
            color = 'massfracnorm'
            clim  = [-2.5, 0.0]

            _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=3)

        if 1:
            # 2dhisto [dens,temp] mass frac
            pt    = 'gas'
            xaxis = 'numdens'
            yaxis = 'temp'
            color = 'massfrac'
            clim   = [-4.0, 0.0]

            _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=3)

        # lower center
        if 0:
            # 2dhisto [dens,temp] mean vrad
            pt    = 'gas'
            xaxis = 'numdens'
            yaxis = 'temp'
            color = 'vrad'
            clim  = [-100, 300]

            _histo2d_helper(pt,xaxis,yaxis,color,clim,snap,i=4)

        if 1:
            # gas dens image, 200 kpc, xy
            axes    = [0,1]
            boxSize = 200.0
            ptType  = 'gas'
            ptField = 'coldens_msunkpc2'

            _image_helper(ptType,ptField,axes,boxSize,i=4)

        # lower right
        if 0:
            # gas dens image, 200 kpc, xz
            axes    = [0,2]
            boxSize = 2000.0
            ptType  = 'gas'
            ptField = 'coldens_msunkpc2'

            _image_helper(ptType,ptField,axes,boxSize,i=5)

        if 0:
            # gas radial density profile
            pt    = 'gas'
            xaxis = 'radlog'
            yaxis = 'numdens'
            ylim  = data['limits']['numdens']

            _histo1d_helper(pt,xaxis,yaxis,ylim,i=5)

        if 1:
            # gas vrad histogram
            pt    = 'gas'
            xaxis = 'vrad'
            yaxis = 'count'
            ylim  = [-6.0, -1.0]

            _histo1d_helper(pt,xaxis,yaxis,ylim,i=5)

        if 0:
            # line plot, masses and blackhole diagnostics
            ax = fig.add_subplot(2,3,6)
            ax.set_xlabel('Redshift')
            ax.set_ylabel('Masses [log M$_{\\rm sun}$]')

            # blackhole mass
            yy = sP.units.codeMassToLogMsun( data['bhs']['BH_Mass'] )
            l, = ax.plot(redshifts[w], yy[w], '-', alpha=0.5, label='BH Mass')
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
                        l, = ax.plot(redshifts[w], yy[w], linestyles[j], alpha=0.5, label=label, color=c)
                        ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=c)

            # blackhole energetics
            ax2 = ax.twinx()
            ax2.set_ylabel('BH $\Delta$ E$_{\\rm low}$ (dotted), E$_{\\rm high}$ (solid) [ log erg ]')
            c = ax._get_lines.prop_cycler.next()['color']
            ax2.plot(redshifts[w], dy_high[w], '-', alpha=0.7, color=c)
            ax2.plot(redshifts[w], dy_low[w], ':', alpha=0.7, color=c)
            ax2.plot(redshifts[snap], dy_high[snap], 'o', markersize=14.0, color=c, alpha=0.6)
            ax2.plot(redshifts[snap], dy_low[snap], 'o', markersize=14.0, color=c, alpha=0.6)
            for t in ax2.get_yticklabels(): t.set_color(c)

            handles, labels = ax.get_legend_handles_labels()
            sExtra = [plt.Line2D((0,1), (0,0), color='black', marker='', lw=lw, linestyle=ls) for ls in linestyles]
            ax.legend(handles+sExtra, labels+apertures, loc='lower right')

        # finish
        fig.tight_layout()
        fig.savefig('vis_%s_sbNum-%d_selInd-%d_%04d.png' % (sP.simName,sbNum,selInd,snap))
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

    if 1:
        # save data (phase diagrams) through time
        save_subbox_data(TNG50, sbNum=0, selInd=0)
        #prerender_subbox_images(TNG50, sbNum=0, selInd=0)

    if 0:
        # vis
        vis_subbox_data(TNG50, sbNum=0, selInd=0)

    if 0:
        # sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG50.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG50)
