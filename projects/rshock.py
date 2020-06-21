"""
projects/rshock.py
  Plots: TNG virial shock radius paper.
  in prep.
"""
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d
from os.path import isfile, isdir
from os import mkdir
import healpy

from util import simParams
from util.helper import running_median, logZeroNaN, logZeroSafe, loadColorTable, last_nonzero
from util.treeSearch import calcParticleIndices, buildFullTree, calcHsml, calcQuantReduction
from plot.config import *
from plot.general import plotStackedRadialProfiles1D, plotHistogram1D, plotPhaseSpace2D
from vis.halo import renderSingleHalo
from vis.box import renderBox

def plotHealpixShells(rad, data, label, rads=None, clim=None, ctName='viridis', saveFilename='plot.pdf'):
    """ Plot a series of healpix shell samplings. """
    fig = plt.figure(figsize=(9,16))

    xlim = [rad.min(), rad.max()]
    ylim = [0, 12]

    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('r / r$_{\\rm vir}$')
    ax.set_ylabel('Angular Direction ($\\theta, \\phi$)')

    ax.set_yticks(np.arange(ylim[0]+1,ylim[1]+1))
    #ax.set_xticks(list(ax.get_xticks()) + xlim)

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False) if clim is not None else None
    cmap = loadColorTable(ctName)

    plt.imshow(data, extent=extent, cmap=cmap, norm=norm, aspect='auto') # origin='lower'

    if rads is not None:
        for i, rad in enumerate(rads):
            ax.plot([rad,rad],ylim,'-',lw=lw,label='#%d'%i)
        l = ax.legend(loc='upper left')
        for text in l.get_texts(): text.set_color('white')

    # colorbar
    cax = make_axes_locatable(ax).append_axes('top', size='3%', pad=0.2)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.set_title(label)
    cax.xaxis.set_ticks_position('top')

    # finish
    fig.savefig(saveFilename)
    plt.close(fig)

def healpix_shells_points(nRad, Nside, radMin=0.0, radMax=5.0):
    """ Return a set of spherical shell sample points as defined by healpix. """

    # generate one set sample positions on unit sphere
    nProj = healpy.nside2npix(Nside)
    projVecs = np.array(healpy.pix2vec(Nside, range(nProj), nest=True)).T # [nProj,3]

    # broadcast into nRad shells, radial coordinates in units of rvir
    samplePoints = np.repeat( projVecs[:,np.newaxis,:], nRad, axis=1 ) # [nProj,nRad,3]

    radPts = np.linspace(radMin, radMax, nRad, endpoint=True)
    
    # shift shells to radial distances
    pts = samplePoints * radPts[np.newaxis,:,np.newaxis]

    pts = np.reshape(pts, (nProj*nRad,3) ) # [N,3] for tree/search operations

    # bin sizes
    radBinSize = (radPts[1] - radPts[0]) # r/rvir
    thetaBinSize = np.sqrt(180**2 / (3*np.pi*Nside**2)) # deg
    thetaBinSizeRvir = np.tan(np.deg2rad(thetaBinSize)) # angular spacing @ rvir, in units of rvir

    return pts, nProj, radPts, radBinSize

def healpix_thresholded_radius(radPts, h2d, thresh_perc, inequality, saveBase=None):
    """ Derive a radius (rshock) by a thresholded histogramming/voting of a healpix sampling input. 
    If saveBase is not None, then also dump debug plots. """
    assert inequality in ['<','>']

    # config
    thresh_minrad    = 0.5 # r/rvir
    local_windowsize = 0.5 # +/- in units of rvir
    combineBins      = 4 # radial smoothing (in histogramming)

    # copy input, derive threshold value
    q2d = h2d.copy()

    w_rad = np.where(radPts < thresh_minrad)
    q2d[:,w_rad] = np.nan
    w_zero = np.where(q2d == 0) # e.g. shocks_dedt, shocks_mach for unflagged cells
    q2d[w_zero] = np.nan

    thresh = np.nanpercentile(q2d, thresh_perc)

    # select pixels on value>threshold or value<threshold
    if inequality == '<':
        q2d[np.isnan(q2d)] = thresh + 1 # do not select nan
        w_thresh = np.where(q2d < thresh)
    if inequality == '>':
        q2d[np.isnan(q2d)] = thresh - 1
        w_thresh = np.where(q2d > thresh)

    # collect all radii of these satisfying pixels, restricted to beyond minimum distance
    rad_thresh = radPts[w_thresh[1]] # second dimension only
    rad_thresh = rad_thresh[np.where(rad_thresh >= thresh_minrad)]

    # make 2d mask
    result_mask = np.zeros(h2d.shape, 'int32')
    result_mask[w_thresh] = 1

    w_rad = np.where(radPts < thresh_minrad)
    result_mask[:,w_rad] = 0

    # rshock: answer 1, localize histogram peak, take median rad of nearby thresholded pixels
    hist1d, xx = np.histogram(rad_thresh, bins=radPts[::combineBins])

    radPtsHist = xx[:-1] + (xx[1]-xx[0])*combineBins/2

    ind = np.argmax(hist1d)
    w = np.where( (rad_thresh > xx[ind]-local_windowsize) & (rad_thresh <= xx[ind]+local_windowsize) )
    rshock1 = np.nanmedian( rad_thresh[w] )

    # rshock: answer 2, median of all thresholded pixels (no peak localization)
    rshock2 = np.nanmedian( rad_thresh )

    # go 'outside in', from large radii towards r->0, and find the first (farthest) pixel satisfying thresh
    # each ray gets one 'vote' for a radius
    inds = last_nonzero(result_mask, axis=1)
    w = np.where(inds >= 0)
    rad_thresh_ray = radPts[inds[w]]

    # rshock: answer 3, localize around ray-based max, but again median all thresholded pixels
    hist1dray, xx = np.histogram(rad_thresh_ray, bins=radPts[::combineBins])

    ind_raymax = np.argmax(hist1dray)
    w = np.where( (rad_thresh > xx[ind_raymax]-local_windowsize) & (rad_thresh <= xx[ind_raymax]+local_windowsize) )
    rshock3 = np.nanmedian( rad_thresh[w] )

    # rshock: answer 4, localize around ray-based max, and use only ray-satisfying pixels
    w = np.where( (rad_thresh_ray > xx[ind_raymax]-local_windowsize) & (rad_thresh_ray <= xx[ind_raymax]+local_windowsize) )
    rshock4 = np.nanmedian( rad_thresh_ray[w] )

    # rshock: answer 5, median of all ray-satisfying pixels
    rshock5 = np.nanmedian( rad_thresh_ray )

    print(' thresh: [%4.1f = %6.3f] rshock [%.2f %.2f] ray: [%.2f %.2f %.2f peak=%.2f]' % \
        (thresh_perc,thresh,rshock1,rshock2,rshock3,rshock4,rshock5,xx[ind_raymax]))

    rshock_vals = [rshock1, rshock2, rshock3, rshock4, rshock5]

    if saveBase is not None:
        # plot 2d mask
        label = 'Quantity %s Threshold' % (inequality)
        plotHealpixShells(radPts, result_mask, label=label, clim=[0,1], ctName='gray', rads=rshock_vals, \
            saveFilename=saveBase+'_mask2d.pdf')

        # plot 1d histo
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.minorticks_on()
        ax.xaxis.grid(which='major', linestyle='-', linewidth=1.0, alpha=0.3, color='black')
        ax.xaxis.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.1, color='black')
        
        ax.set_xlabel('r / r$_{\\rm vir}$')
        ax.set_ylabel('Count of Pixels %s Thresh' % inequality)

        l, = ax.plot(radPtsHist, hist1d, '-', lw=lw, drawstyle='steps-mid')

        for i, rshock in enumerate(rshock_vals):
            ax.plot([rshock,rshock],[0,ax.get_ylim()[1]], '-', lw=lw, label='#%d'%i)

        ax.legend(loc='upper right')
        fig.savefig(saveBase+'_1d.pdf')
        plt.close(fig)

        # plot 1d ray histo
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.minorticks_on()
        ax.xaxis.grid(which='major', linestyle='-', linewidth=1.0, alpha=0.3, color='black')
        ax.xaxis.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.1, color='black')
        
        ax.set_xlabel('r / r$_{\\rm vir}$')
        ax.set_ylabel('Count of Rays Voting Here')

        l, = ax.plot(radPtsHist, hist1dray, '-', lw=lw, drawstyle='steps-mid')

        for i, rshock in enumerate(rshock_vals):
            ax.plot([rshock,rshock],[0,ax.get_ylim()[1]], '-', lw=lw, label='#%d'%i)

        fig.savefig(saveBase+'_ray1d.pdf')
        plt.close(fig)

    return rshock_vals

def local_gas_subset(sP, haloID=0, maxRadR200=5.2, useTree=True):
    """ Obtain and cache a set of gas cells in the vicinity of a halo.
    Temporary, move into auxCat with tree. """
    from util.helper import reportMemory

    gas_local = {}

    # cache
    cacheFilename = sP.derivPath + '/cache/rshock_subset_%d_h%d_r%.1f.hdf5' % (sP.snap,haloID,maxRadR200)

    if isfile(cacheFilename):
        print('Loading: [%s]' % cacheFilename)
        with h5py.File(cacheFilename,'r') as f:
            for key in f:
                gas_local[key] = f[key][()]

        return gas_local

    # metadata
    halo = sP.halo(haloID)
    print('Creating new local gas subset, halo mass: ', sP.units.codeMassToLogMsun(halo['Group_M_Crit200']), reportMemory())

    haloPos = halo['GroupPos']
    maxRad = maxRadR200 * halo['Group_R_Crit200']

    # global load
    pos = sP.snapshotSubsetP('gas', 'pos', float32=True)

    if useTree:
        savePath = sP.derivPath + 'tree/'
        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + 'tree32_%s_%d_gas.hdf5' % (sP.simName,sP.snap)

        if not isfile(saveFilename):
            # construct tree
            print('Start build of global oct-tree...', reportMemory(), flush=True)
            start_time = time.time()
            tree = buildFullTree(pos, boxSizeSim=sP.boxSize, treePrec='float32', verbose=True)
            print('Tree finished [%.1f sec].' % (time.time()-start_time))

            with h5py.File(saveFilename,'w') as f:
                for i, item in enumerate(tree):
                    f['item_%d' % i] = item
            print('Saved: [%s]' % saveFilename)
        else:
            # load previously saved tree
            print('Loading: [%s]' % saveFilename)
            with h5py.File(saveFilename,'r') as f:
                tree = []
                for item in f:
                    tree.append(f[item][()])

        # tree search
        print('Start tree search...')
        start_time = time.time()
        loc_inds = calcParticleIndices(pos, haloPos, maxRad, boxSizeSim=sP.boxSize, tree=tree)
        print('Tree search finished [%.1f sec].' % (time.time()-start_time))

        if 0:
            # brute-force verify
            print('Start brute-force verify...')
            start_time = time.time()
            dists = sP.periodicDistsN(haloPos, pos)
            loc_inds2 = np.where(dists <= maxRad)[0]

            zz = np.argsort(loc_inds)
            zz = loc_inds[zz]
            assert np.array_equal(zz,loc_inds2)
            print('Verify finished [%.1f sec].' % (time.time()-start_time))
    else:
        # avoid tree, simply do brute-force distance search
        print('Start brute-force search...')
        start_time = time.time()

        dists = sP.periodicDistsN(haloPos, pos)
        loc_inds = np.where(dists <= maxRad)[0]

        print('Brute-force search finished [%.1f sec].' % (time.time()-start_time))

    # take local subset
    gas_local['inds'] = loc_inds
    gas_local['pos'] = pos[loc_inds]

    gas_local['rad'] = sP.periodicDists(haloPos, gas_local['pos'])
    gas_local['rad'] /= halo['Group_R_Crit200']

    # shocks_dedt, temp, entr
    gas_local['shocks_dedt'] = sP.snapshotSubsetC('gas', 'shocks_dedt', inds=loc_inds)
    gas_local['shocks_dedt'] = sP.units.codeEnergyRateToErgPerSec(gas_local['shocks_dedt'])
    gas_local['shocks_mach'] = sP.snapshotSubsetC('gas', 'shocks_machnum', inds=loc_inds)

    gas_local['temp'] = sP.snapshotSubsetC('gas', 'temp', inds=loc_inds)
    gas_local['entr'] = sP.snapshotSubsetC('gas', 'entr', inds=loc_inds)

    # vrad
    sub = sP.subhalo(halo['GroupFirstSub'])
    sP.refPos = sub['SubhaloPos']
    sP.refVel = sub['SubhaloVel']

    gas_local['vrad'] = sP.snapshotSubsetC('gas', 'vrad', inds=loc_inds)

    # save cache
    with h5py.File(cacheFilename,'w') as f:
        for key in gas_local:
            f[key] = gas_local[key]
    print('Saved: [%s]' % cacheFilename)

    # run algorithm
    return gas_local

def virialShockRadius(sP, haloID):
    """ Given input gas cell set of required properties, extending at least beyond the 
    virial shock radius in extent (i.e. beyond fof-scope), derive rshock. 
    Use healpix shell sampling of gas quantities, and also generate debug plots. """    

    # config
    nRad = 400
    Nside = 16

    clim_percs = [5,99] # colorbar range

    # get sample points, shift to box coords, halo centered (handle periodic boundaries)
    pts, nProj, radPts, radBinSize = healpix_shells_points(nRad=nRad, Nside=Nside)

    halo = sP.halo(haloID)
    pts *= halo['Group_R_Crit200']
    pts += halo['GroupPos'][np.newaxis, :]

    sP.correctPeriodicPosVecs(pts)

    # load a particle subset
    p = local_gas_subset(sP, haloID=haloID)

    # construct tree
    print('build tree...')
    tree = buildFullTree(p['pos'], boxSizeSim=sP.boxSize, treePrec='float32')

    # derive hsml (one per sample point)
    nNGB = 20
    nNGBDev = 1

    print('calc hsml...')
    hsml = calcHsml(p['pos'], sP.boxSize, posSearch=pts, nNGB=nNGB, nNGBDev=nNGBDev, tree=tree)

    # sample and plot different quantiites
    for field in ['shocks_mach']: #['shocks_dedt','temp','entr','vrad']:
        print('Total number of samples [%d], running [%s]...' % (pts.shape[0],field))

        saveBase = 'healpix_%s_z%d_h%d_ns%d_nr%d_%s' % (sP.simName,sP.redshift,haloID,Nside,nRad,field)

        if field == 'shocks_dedt':
            op = 'kernel_mean' # 'mean' # kernel_mean
            quant = p[field] / 1e30
            label = 'Shock Energy Dissipation [ log $10^{30}$ erg/s ]'
            label2 = 'log( $\dot{E}_{\\rm shock}$ / <$\dot{E}_{\\rm shock}$> )'
            label3 = '$\partial$$\dot{E}_{\\rm shock}$/$\partial$r [ log $10^{30}$ erg/s kpc$^{-1}$ ]'

        if field == 'shocks_mach':
            op = 'kernel_mean'
            quant = p[field]
            label = 'Shock Mach Number [ linear ]'
            label2 = 'log( $\mathcal{M}_{\\rm shock}$ / <$\mathcal{M}_{\\rm shock}$> )'
            label3 = '$\partial$$\mathcal{M}_{\\rm shock}$/$\partial$r [ linear kpc$^{-1}$ ]'

        if field == 'temp':
            op = 'kernel_mean'
            quant = 10.0**p[field]
            label = 'Temperature [ log K ]'
            label2 = 'log( $\delta$T / <T> )'
            label3 = '$\partial$T/$\partial$r [ log K kpc$^{-1}$ ]'

        if field == 'entr':
            op = 'kernel_mean'
            quant = 10.0**p[field]
            label = 'Entropy [ log K cm$^2$ ]'
            label2 = 'log( $\delta$S / <S> )'
            label3 = '$\partial$S/$\partial$r [ log K cm$^2$ kpc$^{-1}$ ]'

        if field == 'vrad':
            op = 'kernel_mean'
            quant = p[field]
            label = 'Radial Velocity [ km/s ]'
            label2 = 'log( v$_{\\rm rad}$ / <v$_{\\rm rad}$> )'
            label3 = '$\partial$v$_{\\rm rad}$/$\partial$r [ km/s kpc$^{-1}$ ]'

        result = calcQuantReduction(p['pos'], quant, hsml, op, sP.boxSize, posSearch=pts, tree=tree)
        result = np.reshape(result, (nProj,nRad))

        # unit conversions
        if field in ['temp','entr','shocks_dedt']:
            #result = logZeroSafe(result, zeroVal=result[result>0].min()) # for shocks_dedt
            result = logZeroNaN(result)

        # plot quantity
        clim = np.nanpercentile(result, clim_percs)
        clim = np.round(clim * 10) / 10

        plotHealpixShells(radPts, result, label=label, clim=clim, saveFilename=saveBase+'.pdf')

        # plot quantity relative to its average at that radius (subtract out radial profile)
        if field in ['shocks_dedt','temp','entr']:
            if np.isfinite(result[:,0]).sum() == 0: result[:,0] = 1.0 # avoid all nan slice
            rad_mean = np.nanmean( 10.0**result, axis=0 )
            rad_mean[np.where(rad_mean == 0)] = 1.0 # avoid division by zero (first, r=0 bin)
            result_norm = logZeroNaN(10.0**result / rad_mean)

        if field in ['vrad','shocks_mach']:
            rad_mean = np.nanmean(result,axis=0)
            rad_mean[np.where(rad_mean == 0)] = 1.0 # avoid division by zero (first, r=0 bin)
            result_norm = logZeroNaN(result / rad_mean)

        clim_val = np.abs(np.nanpercentile(result_norm, clim_percs)).min()
        clim_val = np.clip(clim_val, 0.001, np.inf)
        clim = np.array([-clim_val, clim_val])
        if clim[0] < 10.0: clim = np.round(clim * 10) / 10

        plotHealpixShells(radPts, result_norm, label=label2, clim=clim, saveFilename=saveBase+'_norm.pdf')

        # plot partial derivative of quantity with respect to radius
        radBinSize = sP.units.codeLengthToKpc(radBinSize * halo['Group_R_Crit200']) # pkpc

        # sign: negative if quantity decreases moving outwards (from r=0)
        result_deriv = np.gradient(result, radBinSize, axis=1)

        clim_val = np.abs(np.nanpercentile(result_deriv, clim_percs)).min()
        clim_val = np.clip(clim_val, 0.001, np.inf)
        clim = np.array([-clim_val, clim_val])
        if clim[0] < 10.0: clim = np.round(clim * 100) / 100

        plotHealpixShells(radPts, result_deriv, label=label3, clim=clim, ctName='curl', saveFilename=saveBase+'_deriv.pdf')

        # flag dquant/dr pixels below a threshold, and plot 1d histogram of their radii
        if field in ['temp','entr','vrad']:
            percs = [1,2,5,10] #[0.1,0.5,
            ineq = '<'
            vals = result_deriv

        if field in ['shocks_dedt']:
            percs = [80,85,90,95] #,98,99,99.5,99.9]
            ineq = '>'
            vals = result

        if field in ['shocks_mach']:
            percs = [85,90,95,98,99,99.5,99.9]
            ineq = '>'
            vals = result

        for perc in percs:
            # derive rshock and save debug plots
            base = saveBase+'_thresh%g' % perc

            rshock_vals = healpix_thresholded_radius(radPts, vals, perc, ineq, saveBase=base)

def visualizeHaloVirialShock(sP, haloID, conf=0):
    """ Driver for a single halo vis example, highlighting the virial shock structure. """
    run        = sP.run
    res        = sP.res
    redshift   = sP.redshift
    hInd       = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

    rVirFracs  = [1.0] #[1.0, 2.0, 3.0, 4.0]
    method     = 'sphMap_global'
    nPixels    = [1000,1000]
    axes       = [0,1]

    labelZ     = False
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True

    size       = 4.5 #9.0 # TODO CHANGE BACK
    sizeType   = 'rVirial'
    depthFac   = 1.0 # todo

    # panel
    partType   = 'gas'

    if conf == 0:
        partField  = 'shocks_dedt'
        valMinMax  = [34.0, 38.0]
    if conf == 1:
        partField  = 'temp'
        valMinMax  = [4.0, 7.2]
    if conf == 2:
        partField  = 'vrad'
        valMinMax  = [-350,350]
        depthFac   = 0.1 # todo
    if conf == 3:
        nPixels = [1920,1080]
        partField  = 'xray'
        valMinMax  = [30.0, 39.0]

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels
        colorbars    = False #True
        saveFilename = './vis_virshock_%s_%d_%s_h%d.pdf' % (sP.simName,sP.snap,partField,haloID)

    # render
    renderSingleHalo([{}], plotConfig, locals(), skipExisting=False)

def paperPlots():
    #TNG50_z0 = simParams(run='tng50-1', redshift=0.0)
    #TNG50_2_z0 = simParams(run='tng50-2', redshift=0.0)
    #TNG50_z2 = simParams(run='tng50-1', redshift=2.0)

    # figure 1: single-halo visualization of the virial shock
    if 0:
        sP = simParams(run='tng50-1',redshift=0.5)
        #sP = simParams(run='tng300-1',redshift=2.0)

        haloID = 0
        for conf in [0,1,2]:
            #for haloID in [0, 10,20,110,120,200,300]:    
            visualizeHaloVirialShock(sP, haloID=haloID, conf=conf)

    # figure 2: testing rshock detection
    if 1:
        haloID = 20 # 0,10,20,110,120,200,300
        sP = simParams(run='tng50-2', redshift=2.0)

        virialShockRadius(sP, haloID)

    # TODO: stars vrad and BH vrad = 'splashback'