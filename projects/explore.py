"""
projects/explore.py
  Misc exploration plots.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import simParams
from util.helper import loadColorTable, logZeroNaN, running_median
from plot.config import *
from vis.halo import renderSingleHalo
from vis.box import renderBox

def singleHaloImage_CelineMuseProposal(conf=6):
    """ Metallicity distribution in CGM image for 2019 MUSE proposal. """
    run        = 'tng'
    res        = 2160
    redshift   = 0.5 # z=1.0 for paper figure
    rVirFracs  = [0.5, 1.0] # None
    method     = 'sphMap' # sphMap_global for paper figure
    #method     = 'histo'
    nPixels    = [100,100] #[800,800] # for celinemuse figure
    axes       = [0,1]
    labelZ     = True
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    rotation   = 'edge-on'

    size      = 300.0
    sizeType  = 'kpc'
    size = 30.0 # zoom-in

    # which halo?
    sP = simParams(res=res, run=run, redshift=redshift)

    for hInd in [440839]:
        panels = []
        haloID = sP.groupCatSingle(subhaloID=hInd)['SubhaloGrNr']

        if conf == 0:
            lines = ['H-alpha','H-beta','O--2-3728.81A','O--3-5006.84A','N--2-6583.45A','S--2-6730.82A']
            partField_loc = 'sb_%s_lum_kpc' % lines[0] # + '_sf0' to set SFR>0 cells to zero
            panels.append( {'partType':'gas', 'partField':partField_loc, 'valMinMax':[34,41]} )

        if conf == 4:
            panels.append( {'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-1.4,0.2]} )

        if conf == 5:
            panels.append( {'partType':'gas', 'partField':'MHIGK_popping', 'valMinMax':[16.0,22.0]} )

        if conf == 6:
            panels.append( {'partType':'gas', 'partField':'MH2GK_popping', 'valMinMax':[16.0,22.0]} )

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = 1000 #nPixels[0] 
            colorbars    = True
            #fontsize     = 42
            saveStr = panels[0]['partField'].replace("_lum","").replace("_kpc","")
            saveFilename = './%s.%d.%d.%s.%dkpc.pdf' % (sP.simName,sP.snap,hInd,saveStr,size)

        # get data for inspection
        #x, _ = renderSingleHalo(panels, plotConfig, locals(), skipExisting=True, returnData=True)
        #with h5py.File('save_%d.hdf5' % hInd,'w') as f:
        #    f['grid'] = x
        #print('Saved.')

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def celineMuseProposalMetallicityVsTheta():
    """ Use some projections to create the Z_gas vs. theta plot. """
    run        = 'tng'
    res        = 2160
    redshift   = 0.5 # z=1.0 for paper figure
    method     = 'sphMap' # sphMap_global for paper figure
    nPixels    = [1000,1000]
    axes       = [0,1]
    rotation   = 'edge-on'

    size      = 250.0
    sizeType  = 'kpc'

    #massBins = [ [8.50, 8.51]]
    #massBins = [ [9.00, 9.02] ]
    #massBins = [ [9.50, 9.54] ] # log mstar
    #massBins = [ [10.00, 10.05] ]
    #massBins = [ [10.5, 10.6] ]
    massBins = [ [11.0, 11.1] ]
    assert len(massBins) == 1 # otherwise generalize below

    distBins = [ [20,30], [45,55], [95,105] ]
    nThetaBins = 90

    # which halos?
    sP = simParams(res=res, run=run, redshift=redshift)

    gc = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log','central_flag'])
    subInds = []

    for massBin in massBins:
        with np.errstate(invalid='ignore'):
            w = np.where( (gc['mstar_30pkpc_log']>massBin[0])  & (gc['mstar_30pkpc_log']<massBin[1]) & gc['central_flag'] )
        subInds.append( w[0] )

        print('[%.2f - %.2f] Processing [%d] halos...' % (massBin[0],massBin[1],len(w[0])))
    
    # load
    for i, massBin in enumerate(massBins):

        dist_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds[i])), dtype='float32' )
        theta_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds[i])), dtype='float32' )
        grid_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds[i])), dtype='float32' )

        for j, hInd in enumerate(subInds[i]):
            #haloID = sP.groupCatSingle(subhaloID=hInd)['SubhaloGrNr']

            class plotConfig:
                saveFilename = 'dummy'

            panels = [{'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-1.4,0.2]}]
            grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

            # compute impact parameter and angle for every pixel
            pxSize = size / nPixels[0] # pkpc

            xx, yy = np.mgrid[0:nPixels[0], 0:nPixels[1]]
            xx = xx.astype('float64') - nPixels[0]/2
            yy = yy.astype('float64') - nPixels[1]/2
            dist = np.sqrt( xx**2 + yy**2 ) * pxSize
            theta = np.rad2deg(np.arctan2(xx,yy)) # 0 and +/- 180 is major axis, while +/- 90 is minor axis
            theta = np.abs(theta) # 0 -> 90 -> 180 is major -> minor -> major

            w = np.where(theta >= 90.0)
            theta[w] = 180.0 - theta[w] # 0 is major, 90 is minor

            # debug plots
            #from util.helper import plot2d
            #plot2d(grid, label='metallicity [log zsun]', filename='test_z.pdf')
            #plot2d(dist, label='distance [pkpc]', filename='test_dist.pdf')
            #plot2d(theta, label='theta [deg]', filename='test_theta.pdf')

            # bin
            dist_global[:,j] = dist.ravel()
            theta_global[:,j] = theta.ravel()
            grid_global[:,j] = grid.ravel()

    # flatten (ignore which halo each pixel came from)
    dist_global = dist_global.ravel()
    theta_global = theta_global.ravel()
    grid_global = grid_global.ravel()

    # start the plot
    figsize = np.array([14,10]) * 0.7
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Azimuthal Angle [deg, 0$^\circ$ = major axis, 90$^\circ$ = minor axis]')
    ax.set_xlim([-2,92])
    ax.set_ylim([-1.4,-0.2])
    ax.set_xticks([0,15,30,45,60,75,90])
    ax.set_ylabel('Median Gas Metallicity [log Z$_\odot$]')

    # bin on the global concatenated grids
    for distBin in distBins:
        w = np.where( (dist_global >= distBin[0]) & (dist_global < distBin[1]) )

        # median metallicity as a function of theta, 1 degree bins
        theta_vals, hist, hist_std = running_median(theta_global[w], grid_global[w], nBins=nThetaBins)

        ax.plot(theta_vals, hist, '-', lw=2.5, label='b = %d kpc' % np.mean(distBin))

    # finish and save plot
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig('z_vs_theta_Mstar=%.1f.pdf' % (massBins[0][0]))
    plt.close(fig)

def amyDIGzProfiles():
    """ Use some projections to create the SB(em lines) vs z plot. """
    run        = 'tng'
    res        = 2160
    redshift   = 0.1
    method     = 'sphMap'
    nPixels    = [100,100]
    axes       = [0,1]
    rotation   = 'edge-on'

    size      = 30.0
    sizeType  = 'kpc'

    massBin = [10.00,10.02] # log mstar
    maxXDistPkpc = 5.0 # select pixels within 5 kpc of disk center

    lines = ['H-alpha','H-beta','O--2-3728.81A','O--3-5006.84A','N--2-6583.45A','S--2-6730.82A']

    # which halos?
    sP = simParams(res=res, run=run, redshift=redshift)

    gc = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log','central_flag'])

    with np.errstate(invalid='ignore'):
        w = np.where( (gc['mstar_30pkpc_log']>massBin[0])  & (gc['mstar_30pkpc_log']<massBin[1]) & gc['central_flag'] )
    subInds = w[0]

    print('[%.2f - %.2f] Processing [%d] halos...' % (massBin[0],massBin[1],len(w[0])))

    # start the plot
    figsize = np.array([14,10]) * 0.9
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('z [pkpc]')
    ax.set_xlim([0, 10])
    ax.set_ylim([1, 600])
    ax.set_yscale('log')
    ax.set_ylabel('Luminosity Surface Density [ 10$^{36}$ erg s$^{-1}$ kpc$^{-2}$ ]')

    # loop over lines
    for line in lines:
        partField_loc = 'sb_%s_lum_kpc' % line # + '_sf0' to set SFR>0 cells to zero

        x_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )
        z_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )
        grid_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float64' )

        for j, hInd in enumerate(subInds):
            # project
            class plotConfig:
                saveFilename = 'dummy'

            panels = [{'partType':'gas', 'partField':partField_loc, 'valMinMax':[34,41]}]
            grid, conf = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

            # compute z-distance and x-distance for each pixel
            pxSize = size / nPixels[0] # pkpc

            xx, yy = np.mgrid[0:nPixels[0], 0:nPixels[1]]
            xx = xx.astype('float64') - nPixels[0]/2 # z-axis, i.e. perpendicular to disk
            yy = yy.astype('float64') - nPixels[1]/2 # x-axis, i.e. along the major axis

            zdist = np.abs(xx) * pxSize # symmetric (both above and below the disk)
            xdist = np.abs(yy) * pxSize

            # debug plots
            #from util.helper import plot2d
            #plot2d(grid, label='sb [log erg/s/kpc^2]', filename='test_grid.pdf')
            #plot2d(xdist, label='x distance[pkpc]', filename='test_xdist.pdf')
            #plot2d(zdist, label='z distance[pkpc]', filename='test_zdist.pdf')

            # save
            x_global[:,j] = xdist.ravel()
            z_global[:,j] = zdist.ravel()
            grid_global[:,j] = grid.ravel()

        # flatten and select in [x-bounds]
        x_global = x_global.ravel()
        z_global = z_global.ravel()
        grid_global = grid_global.ravel()

        w = np.where( x_global < maxXDistPkpc )

        with np.errstate(invalid='ignore'):
            grid_global = 10.0**grid_global # remove log

        # bin: median SB as a function of z
        nBins = int(nPixels[0]/2)
        z_vals, hist, hist_std = running_median(z_global[w], grid_global[w], nBins=nBins)

        hist /= 1e36 # units to match y-axis label

        # plot
        label = conf['label'].split(" Luminosity")[0]
        ax.plot(z_vals, hist, '-', lw=2.5, label=label)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('sb_vs_z_Mstar=%.1f.pdf' % (massBin[0]))
    plt.close(fig)

def martinSubboxProj3DGrid():
    """ Compute (i) 2D histo projection, (ii) 2D sphMap projection, (iii) 3D sphMap grid then projection (todo), of subbox gas. """
    run        = 'tng'
    redshift   = 8.0 # subbox snap 126
    variant    = 'subbox0'
    res        = 1080

    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)
    nPixels    = [128,128]

    partType   = 'gas'
    partField  = 'coldens_msunkpc2'
    valMinMax  = [5.5, 7.3]

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    class plotConfig:
        plotStyle  = 'open' # open, edged
        rasterPx   = 1000
        colorbars  = True
        saveBase   = ''

    # (A) 
    panels = [{}]
    method = 'sphMap'
    plotConfig.saveFilename = './boxImage_%s_%d_%s_%s.pdf' % (sP.simName,sP.snap,partType,method)

    renderBox(panels, plotConfig, locals())

    # (B)
    panels = [{}]
    method = 'histo'
    plotConfig.saveFilename = './boxImage_%s_%d_%s_%s.pdf' % (sP.simName,sP.snap,partType,method)

    renderBox(panels, plotConfig, locals())

    # (C) get data grids and compare histograms
    panels = [{}]
    method = 'sphMap'
    grid_sphmap, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    panels = [{}]
    method = 'histo'
    grid_histo, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    sphmap_total = np.sum(10.0**grid_sphmap)
    histo_total  = np.sum(10.0**grid_histo)
    frac         = np.sum(10.0**grid_sphmap)/np.sum(10.0**grid_histo)

    # start plot
    vmm = [5.0, 8.0]
    nBins = 120

    figsize = np.array([14,10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(conf['label'])
    ax.set_xlim([5.5,7.5])
    ax.set_yscale('log')
    ax.set_ylabel('Number of Pixels')
    ax.set_title('sphmap total = %g, histo total = %g, frac = %.4f ' % (sphmap_total, histo_total, frac))

    # histogram and plot
    yy, xx = np.histogram(grid_sphmap.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5*(vmm[1]-vmm[0])/nBins

    ax.plot(xx, yy, '-', drawstyle='steps', label='sphmap')

    yy, xx = np.histogram(grid_histo.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5*(vmm[1]-vmm[0])/nBins

    ax.plot(xx, yy, '-', drawstyle='steps', label='histo')

    # finish and save plot
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('px_comp.pdf')
    plt.close(fig)
