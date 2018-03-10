"""
cosmo/zooms.py
  Analysis and helpers specifically for zoom resimulations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import OrderedDict
from scipy.spatial import ConvexHull
from scipy.stats import binned_statistic

from cosmo.load import groupCatSingle, groupCat, snapshotHeader, snapshotSubset
from cosmo.util import periodicDists
from cosmo.perf import loadCpuTxt
from util.simParams import simParams
from util.helper import logZeroNaN
from vis.halo import renderSingleHalo

def calculate_contamination(sPzoom, rVirFacs=[1,2,3,4,5,10], verbose=False):
    """ Calculate number of low-res DM within each rVirFac*rVir distance, as well 
    as the minimum distance to any low-res DM particle, and a radial profile of 
    contaminating particles. """
    halo = groupCatSingle(sPzoom, haloID=0)
    r200 = halo['Group_R_Crit200']

    # load
    h = snapshotHeader(sPzoom)

    pos_hr  = snapshotSubset(sPzoom, 'dm', 'pos')
    pos_lr  = snapshotSubset(sPzoom, 2, 'pos')

    mass_lr = snapshotSubset(sPzoom, 2, 'mass')
    mass_hr = h['MassTable'][sPzoom.ptNum('dm')]

    dists_lr = periodicDists( halo['GroupPos'], pos_lr, sP=sPzoom )
    dists_hr = periodicDists( halo['GroupPos'], pos_hr, sP=sPzoom )

    min_dist_lr = dists_lr.min()
    if verbose:
        print('min dists from halo to closest low-res DM: ', min_dist_lr)

    # allocate
    counts    = np.zeros( len(rVirFacs), dtype='int32' )   # number low-res DM
    fracs     = np.zeros( len(rVirFacs), dtype='float32' ) # of low-res DM to HR DM
    massfracs = np.zeros( len(rVirFacs), dtype='float32' ) # of low-res DM mass to HR DM mass

    # calculate counts
    for i, rVirFac in enumerate(rVirFacs):
        w_lr = np.where(dists_lr < rVirFac*r200)
        w_hr = np.where(dists_hr < rVirFac*r200)

        counts[i] = len( w_lr[0] )
        fracs[i]  = counts[i] / float( len(w_hr[0]) )

        totmass_lr = mass_lr[w_lr].sum()
        totmass_hr = len(w_hr[0]) * mass_hr

        massfracs[i] = totmass_lr / totmass_hr

        if verbose:
            print('num within %2d rvir (%6.1f) = %6d (%5.2f%% of HR num) (%5.2f%% of HR mass)' % \
                (rVirFac,rVirFac*r200, counts[i], fracs[i]*100, massfracs[i]*100))

    # calculate radial profiles
    rlim  = [0.0, np.max(rVirFacs)*r200]
    nbins = 30

    r_count_hr, rr = np.histogram(dists_hr, bins=nbins, range=rlim)
    r_count_lr, _  = np.histogram(dists_lr, bins=nbins, range=rlim)

    r_mass_lr, _ = np.histogram(dists_lr, bins=nbins, range=rlim, weights=mass_lr)
    r_mass_hr    = r_count_hr * mass_hr

    r_frac     = r_count_lr / r_count_hr
    r_massfrac = r_mass_lr / r_mass_hr

    rr = rr[:-1] + (rlim[1]-rlim[0])/nbins # bin midpoints

    return min_dist_lr, rVirFacs, counts, fracs, massfracs, rr, r_frac, r_massfrac

def check_contamination():
    """ Check level of low-resolution contamination (DM particles) in zoom run. """
    hInd = 8
    zoomRes = 13
    variant = 'sf2'

    # load parent box: halo
    sP = simParams(res=2048,run='tng_dm',redshift=0.0)
    halo = groupCatSingle(sP, haloID=hInd)

    # load zoom: group catalog
    sPz = simParams(res=zoomRes, run='tng_zoom_dm', hInd=hInd, redshift=0.0, variant=variant)

    halo_zoom = groupCatSingle(sPz, haloID=0)
    halos_zoom = groupCat(sPz, fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
    subs_zoom = groupCat(sPz, fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

    print('parent halo pos: ', halo['GroupPos'])
    print('zoom halo cenrelpos: ', halo_zoom['GroupPos'] - sP.boxSize/2)
    print('parent halo mass: ',sP.units.codeMassToLogMsun([halo['Group_M_Crit200'],halo['GroupMass']]))
    print('zoom halo mass: ',sP.units.codeMassToLogMsun([halo_zoom['Group_M_Crit200'],halo_zoom['GroupMass']]))

    # print/load contamination statistics
    min_dist_lr, _, _, _, _, rr, r_frac, r_massfrac = calculate_contamination(sPz, verbose=True)

    # plot contamination profiles
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)
    lw = 2.5
    ylim = [-4.0, 0.0]

    ax.set_xlabel('Distance [ckpc/h]')
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.set_xlim([0.0, rr.max()])
    ax.set_ylim(ylim)

    ax.plot([0,rr[-1]], [-1.0, -1.0], '-', color='#888888', lw=lw-1, alpha=0.5, label='10%')
    ax.plot([0,rr[-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw-1, alpha=0.2, label='1%')
    ax.plot(rr, logZeroNaN(r_frac), '-', lw=lw, label='by number')
    ax.plot(rr, logZeroNaN(r_massfrac), '-', lw=lw, label='by mass')

    ax.plot([min_dist_lr,min_dist_lr], ylim, ':', color='#555555', lw=lw-1, alpha=0.5, label='closest LR' )

    ax2 = ax.twiny()
    ax2.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax2.set_xlim([0.0, rr.max()/halo_zoom['Group_R_Crit200']])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax.legend(loc='lower right')
    fig.tight_layout()    
    fig.savefig('contamination_profile_%s_%d.pdf' % (sPz.simName,sPz.snap))
    plt.close(fig)

    # time evo of convex hull volume
    if 0:
        for snap in range(sPz.snap,0,-1):
            sPz.setSnap(snap)
            x = snapshotSubset(sPz, 'dm', 'pos')
            hull = ConvexHull(x)
            print('[%3d] z = %5.2f high-res volume frac = %.3f%%' % (snap,sPz.redshift,hull.volume/sP.boxSize**3*100))

            # plot points scatter
            continue # disable
            fig = plt.figure(figsize=(16,16))
            ax = fig.add_subplot(111)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim([0,sP.boxSize])
            ax.set_ylim([0,sP.boxSize])
            ax.plot(x[:,0], x[:,1], '.')

            fig.tight_layout()    
            fig.savefig('check_zoom-%d.png' % snap)
            plt.close(fig)

def sizefacComparison():
    """ Compare SizeFac 2,3,4 runs (contamination and CPU times) in the testing set. """

    # config
    hInds    = [8,50,51,90]
    sizeFacs = [2,3,4]

    run      = 'tng_zoom_dm'
    zoomRes  = 13
    redshift = 0.0

    # load
    results = []

    for hInd in hInds:
        for sizeFac in sizeFacs:
            print('Load hInd=%2d sizeFac=%d' % (hInd,sizeFac))
            if hInd == 8 and sizeFac == 4:
                print(' skipping h8sf4 for now (not done).')
                continue

            variant = 'sf%d' % sizeFac
            sP = simParams(run=run, res=zoomRes, hInd=hInd, redshift=redshift, variant=variant)

            cpu = loadCpuTxt(sP.arepoPath, keys=['total'])
            cpuHours = cpu['total'][0,-1,2] / (60.0*60.0) * cpu['numCPUs']

            min_dist_lr, rVirFacs, counts, fracs, massfracs, _, _, _ = calculate_contamination(sP)

            halo = groupCatSingle(sP, haloID=0)
            haloMass = sP.units.codeMassToLogMsun( halo['Group_M_Crit200'] )
            haloRvir = halo['Group_R_Crit200']

            r = {'hInd':hInd, 'sizeFac':sizeFac, 'cpuHours':cpuHours, 'haloMass':haloMass, 'haloRvir':haloRvir, 
                 'contam_min':min_dist_lr, 'contam_rvirfacs':rVirFacs, 'contam_counts':counts}
            results.append(r)

    # start plot
    fig = plt.figure(figsize=(22,12))
    lw = 2.5

    for rowNum in [0,1]:
        xlabel = 'Halo ID' if rowNum == 0 else 'Halo Mass [log M$_{\\rm sun}$]'
        ax = fig.add_subplot(2,3,rowNum*3+1)

        # set up unique coloring by sizeFac
        colors = OrderedDict()

        for sizeFac in sizeFacs:
            c = ax._get_lines.prop_cycler.next()['color']
            colors[ sizeFac ] = c

        handles = [plt.Line2D((0,1), (0,0), color=colors[sf], marker='o', lw=lw) for sf in colors.keys()]

        # (A) contamination dist kpc
        ylim = [-4.0, 0.0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Min LR Dist [ckpc/h]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['sizeFac'] ]
            ax.plot( xx, result['contam_min'], 'o', color=color, label='')
        
        ax.legend(handles, ['sf=%d'%sf for sf in colors.keys()], loc='best')

        # (B) contamination rvir
        ax = fig.add_subplot(2,3,rowNum*3+2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Min LR Dist [r$_{\\rm vir}$]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['sizeFac'] ]
            ax.plot( xx, result['contam_min']/result['haloRvir'], 'o', color=color, label='')

        xlim = ax.get_xlim()
        for rVirFac in [5,2,1]:
            ax.plot( xlim, [rVirFac,rVirFac], '-', color='#bbbbbb', alpha=0.4 )

        ax.legend(handles, ['sf=%d'%sf for sf in colors.keys()], loc='best')

        # (C) cpu hours
        ax = fig.add_subplot(2,3,rowNum*3+3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('CPU Time [log kHours]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['sizeFac'] ]
            ax.plot( xx, np.log10(result['cpuHours']/1e3), 'o', color=color, label='')

        ax.legend(handles, ['sf=%d'%sf for sf in colors.keys()], loc='best')

    # finish
    fig.tight_layout()    
    fig.savefig('sizefac_comparison.pdf')
    plt.close(fig)

def parentBoxVisualComparison(haloID, variant='sf2', conf=0):
    """ Make a visual comparison (density projection images) between halos in the parent box and their zoom realizations. """
    sPz = simParams(run='tng_zoom_dm', res=13, hInd=haloID, redshift=0.0, variant=variant)

    # render config
    rVirFracs  = [0.5, 1.0] # None
    method     = 'sphMap_global' # sphMap
    nPixels    = [1920,1920]
    axes       = [0,1]
    labelZ     = True
    labelScale = True
    labelSim   = True
    labelHalo  = True
    relCoords  = True
    size       = 6000.0
    sizeType   = 'pkpc'

    # setup panels
    if conf == 0:
        # dm column density
        p = {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 9.5]}
    if conf == 2:
        # gas column density
        p = {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 8.0]}

    panel_zoom = p.copy()
    panel_parent = p.copy()

    parSubID = groupCatSingle(sPz.sP_parent, haloID=haloID)['GroupFirstSub']

    panel_zoom.update( {'run':sPz.run, 'res':sPz.res, 'redshift':sPz.redshift, 'variant':sPz.variant, 'hInd':haloID})
    panel_parent.update( {'run':sPz.sP_parent.run, 'res':sPz.sP_parent.res, 'redshift':sPz.sP_parent.redshift, 'hInd':parSubID})

    panels = [panel_zoom, panel_parent]

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = nPixels[0]
        colorbars    = True
        saveFilename = './zoomParentBoxVisualComparison_%s_z%.1f_%s.pdf' % (sPz.simName,sPz.redshift,p['partType'])

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)
