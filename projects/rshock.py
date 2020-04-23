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
from util.helper import running_median, logZeroNaN, logZeroSafe, loadColorTable
from util.treeSearch import calcParticleIndices, buildFullTree, calcHsml, calcQuantReduction
from plot.config import *
from plot.general import plotStackedRadialProfiles1D, plotHistogram1D, plotPhaseSpace2D
from vis.halo import renderSingleHalo
from vis.box import renderBox

def plotHealpixShells(rad, data, label, clim=None, ctName='viridis', saveFilename='plot.pdf'):
    """ Plot a series of healpix shell samplings. """
    fig = plt.figure(figsize=figsize)

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

    # colorbar
    cax = make_axes_locatable(ax).append_axes('top', size='5%', pad=0.2)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.set_title(label)
    cax.xaxis.set_ticks_position('top')

    # finish
    fig.tight_layout()    
    fig.savefig(saveFilename)
    plt.close(fig)

def virialShockRadius():
    """ Given input gas cell set of required properties, extending at least beyond the 
    virial shock radius in extent (i.e. beyond fof-scope), derive rshock. 
    For now just some debugging plots. """
    sP = simParams(run='tng50-2', redshift=0.0)
    haloID = 0

    # load
    gas = local_gas_subset(sP, haloID=haloID)

    # config
    lim = [0.0, 4.0]
    nBins1D = 100
    binsize = (lim[1] - lim[0]) / nBins1D

    # (A) plot 1d histogram
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('R / R$_{\\rm 200}$')
    ax.set_ylabel('Total Shock Energy Dissipation [ log erg/s ]')

    yy, xx, _ = binned_statistic(gas['rad'], gas['shocks_dedt']/1e30, 'sum', bins=nBins1D, range=lim)

    yy = logZeroNaN(yy) + 30.0 # convert back to erg/s
    xx = xx[:-1] + binsize/2 # mid

    l, = ax.plot(xx, yy, '-', lw=lw, drawstyle='steps-mid')

    fig.tight_layout()
    fig.savefig('histo1d_shocks_dedt_%s_%d_h%d.pdf' % (sP.simName,sP.snap,haloID))
    plt.close(fig)

    # (B) let's look at 2D histogram of (r/r200,dedt) also vrad,temp,entr
    #  -- todo: generalize existing plot utils for global scope, with or without tree

    # (C) let's revisit the (angle,r/r200) colored by quantity plots
    #  -- let's redo the tophat/sph quantity sampler and use it, recreate tall skiny plot
    #ylabel = 'Angular Direction ($\\theta, \\phi$)'

    # (D) as above, but let's bin instead of re-sampling
    #  -- todo: how? simply choose a healpix resolution, map all (x,y,z)->(angular_px_id)
    #     and pass to binned_statistic_2d...

def healpix_shells_points(nRad, Nside):
    """ Return a set of spherical shell sample points as defined by healpix. """
    radMin = 0.0
    radMax = 4.9

    # generate one set sample positions on unit sphere
    nProj = healpy.nside2npix(Nside)
    projVecs = np.array(healpy.pix2vec(Nside, range(nProj), nest=True)).T # [nProj,3]

    # broadcast into nRad shells, radial coordinates in units of rvir
    samplePoints = np.repeat( projVecs[:,np.newaxis,:], nRad, axis=1 ) # [nProj,nRad,3]

    radPts = np.linspace(radMin, radMax, nRad, endpoint=False)
    
    # shift shells to radial distances
    pts = samplePoints * radPts[np.newaxis,:,np.newaxis]

    pts = np.reshape(pts, (nProj*nRad,3) ) # [N,3] for tree/search operations

    # bin sizes
    radBinSize = (radPts[1] - radPts[0]) # r/rvir
    thetaBinSize = np.sqrt(180**2 / (3*np.pi*Nside**2)) # deg

    return pts, nProj, radPts, radBinSize

def healpix_sample_and_plot():
    """ Test: healpix shell sampling of some gas quantities, and plots. """    

    # config
    haloID = 100
    sP = simParams(run='tng50-2', redshift=0.0)

    nRad = 400
    Nside = 8

    # get sample points, shift to box coords, halo centered (handle periodic boundaries)
    pts, nProj, radPts, radBinSize = healpix_shells_points(nRad=nRad, Nside=Nside)

    halo = sP.halo(haloID)
    pts *= halo['Group_R_Crit200']
    pts += halo['GroupPos'][np.newaxis, :]

    sP.correctPeriodicPosVecs(pts)

    # load a particle subset
    maxRadR200 = 5.0
    p = local_gas_subset(sP, haloID=haloID, maxRadR200=maxRadR200)

    # construct tree
    print('build tree...')
    tree = buildFullTree(p['pos'], boxSizeSim=sP.boxSize, treePrec='float32')

    # derive hsml (one per sample point)
    nNGB = 20
    nNGBDev = 1

    print('calc hsml...')
    hsml = calcHsml(p['pos'], sP.boxSize, posSearch=pts, nNGB=nNGB, nNGBDev=nNGBDev, tree=tree)

    # sample and plot different quantiites
    for field in ['shocks_dedt','temp','entr','vrad']:
        print('Total number of samples [%d], running [%s]...' % (pts.shape[0],field))

        saveBase = 'healpix_%s_h%d_ns%d_nr%d_%s' % (sP.simName,haloID,Nside,nRad,field)

        if field == 'shocks_dedt':
            op = 'kernel_mean' # 'mean' # kernel_mean
            quant = p[field] / 1e30
            label = 'Shock Energy Dissipation [ log $10^{30}$ erg/s ]'
            label2 = 'log( $\dot{E}_{\\rm shock}$ / <$\dot{E}_{\\rm shock}$> )'
            label3 = '$\partial$$\dot{E}_{\\rm shock}$/$\partial$r [ log $10^{30}$ erg/s kpc$^{-1}$ ]'
            clim = [2.0,7.0]
            clim2 = [-1.5, 1.5] # log
            clim3 =  None

        if field == 'temp':
            op = 'kernel_mean'
            quant = 10.0**p[field]
            label = 'Temperature [ log K ]'
            label2 = 'log( $\delta$T / <T> )'
            label3 = '$\partial$T/$\partial$r [ log K kpc$^{-1}$ ]'
            clim = [3.8,7.0]
            clim2 = [-0.6,0.6] # log
            clim3 = [-0.03, 0.03]

        if field == 'entr':
            op = 'kernel_mean'
            quant = 10.0**p[field]
            label = 'Entropy [ log K cm$^2$ ]'
            label2 = 'log( $\delta$S / <S> )'
            label3 = '$\partial$S/$\partial$r [ log K cm$^2$ kpc$^{-1}$ ]'
            clim = [8.0, 12.0]
            clim2 = [-0.6,0.6] # log
            clim3 = [-0.04, 0.04]

        if field == 'vrad':
            op = 'kernel_mean'
            quant = p[field]
            label = 'Radial Velocity [ km/s ]'
            label2 = 'log( v$_{\\rm rad}$ / <v$_{\\rm rad}$> )'
            label3 = '$\partial$v$_{\\rm rad}$/$\partial$r [ km/s kpc$^{-1}$ ]'
            clim = [-200, 200]
            clim2 = [-1.0, 1.0] # log
            clim3 = [-1, 1] # km/s/kpc

        result = calcQuantReduction(p['pos'], quant, hsml, op, sP.boxSize, posSearch=pts, tree=tree)
        result = np.reshape(result, (nProj,nRad))

        # unit conversions
        if field in ['shocks_dedt']:
            result = logZeroSafe(result, zeroVal=result[result>0].min())

        if field in ['temp','entr']:
            result = logZeroNaN(result)

        # plot
        plotHealpixShells(radPts, result, label=label, clim=clim, saveFilename=saveBase+'.pdf')

        # plot relative quantity
        if field in ['shocks_dedt','temp','entr']:
            rad_mean = np.nanmean( 10.0**result, axis=0 )
            result_norm = logZeroNaN(10.0**result / rad_mean)

        if field in ['vrad']:
            result_norm = logZeroNaN(result / np.nanmean(result,axis=0))

        plotHealpixShells(radPts, result_norm, label=label2, clim=clim2, saveFilename=saveBase+'_norm.pdf')

        # plot radial derivative
        radBinSize = sP.units.codeLengthToKpc(radBinSize * halo['Group_R_Crit200']) # pkpc

        result_deriv = np.gradient(result, radBinSize, axis=1)

        plotHealpixShells(radPts, result_deriv, label=label3, clim=clim3, ctName='curl', saveFilename=saveBase+'_deriv.pdf')

def local_gas_subset(sP, haloID=0, maxRadR200=5.0):
    """ Obtain and cache a set of gas cells in the vicinity of a halo.
    Temporary, move into auxCat with tree. """
    from util.helper import reportMemory

    gas_local = {}

    # cache
    cacheFilename = sP.derivPath + '/cache/rshock_subset_%d_h%d_r%d.hdf5' % (sP.snap,haloID,maxRadR200)

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

    # take local subset
    gas_local['inds'] = loc_inds
    gas_local['pos'] = pos[loc_inds]

    gas_local['rad'] = sP.periodicDists(haloPos, gas_local['pos'])
    gas_local['rad'] /= halo['Group_R_Crit200']

    gas_local['shocks_dedt'] = sP.snapshotSubsetP('gas', 'shocks_dedt')
    gas_local['shocks_dedt'] = sP.units.codeEnergyRateToErgPerSec(gas_local['shocks_dedt'][loc_inds])

    gas_local['temp'] = sP.snapshotSubsetP('gas', 'temp', inds=loc_inds)
    gas_local['entr'] = sP.snapshotSubsetP('gas', 'entr', inds=loc_inds)

    # vrad
    sub = sP.subhalo(halo['GroupFirstSub'])
    sP.refPos = sub['SubhaloPos']
    sP.refVel = sub['SubhaloVel']

    gas_local['vrad'] = sP.snapshotSubsetP('gas', 'vrad', inds=loc_inds)

    # save cache
    with h5py.File(cacheFilename,'w') as f:
        for key in gas_local:
            f[key] = gas_local[key]
    print('Saved: [%s]' % cacheFilename)

    # run algorithm
    return gas_local

def visualizeHaloVirialShock(sP, haloID, conf=0):
    """ Driver for a single halo vis example, highlighting the virial shock structure. """
    run        = sP.run
    res        = sP.res
    redshift   = sP.redshift
    hInd       = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

    rVirFracs  = [1.0, 2.0, 3.0, 4.0]
    method     = 'sphMap_global'
    nPixels    = [1000,1000]
    axes       = [0,1]

    labelZ     = False
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True

    size       = 9.0
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

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0]
        colorbars    = True
        saveFilename = './vis_virshock_%s_%d_%s_h%d.pdf' % (sP.simName,sP.snap,partField,haloID)

    # render
    renderSingleHalo([{}], plotConfig, locals(), skipExisting=False)

def paperPlots(haloID, conf):
    TNG50_z0 = simParams(run='tng50-1', redshift=0.0)
    TNG50_2_z0 = simParams(run='tng50-2', redshift=0.0)

    # fig X: single-halo visualization of the virial shock
    if 1:
        #haloID = 0
        #conf = 1
        visualizeHaloVirialShock(TNG50_z0, haloID=haloID, conf=conf)
