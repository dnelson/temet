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
from plot.config import *

def pick_halos():
    """ Testing. """
    from vis.halo import selectHalosFromMassBins
    sP = simParams(res=2048, run='tng_dm', redshift=0.0)
    #sP = simParams(res=2500, run='tng', redshift=0.0)

    # config
    bins = [ [x+0.0,x+0.1] for x in np.linspace(14.0,15.4,15) ]
    numPerBin = 10

    hInds = selectHalosFromMassBins(sP, bins, numPerBin, 'random')

    for i, bin in enumerate(bins):
        print(bin,hInds[i])

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

    min_dist_lr = dists_lr.min() # code units
    min_dist_lr = sPzoom.units.codeLengthToMpc(min_dist_lr)
    if verbose:
        print('min dists from halo to closest low-res DM [pMpc]: ', min_dist_lr)

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
    ylim = [-4.0, 0.0]

    ax.set_xlabel('Distance [ckpc/h]')
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.set_xlim([0.0, rr.max()])
    ax.set_ylim(ylim)

    ax.plot([0,rr[-1]], [-1.0, -1.0], '-', color='#888888', lw=lw, alpha=0.5, label='10%')
    ax.plot([0,rr[-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw, alpha=0.2, label='1%')
    ax.plot(rr, logZeroNaN(r_frac), '-', lw=lw, label='by number')
    ax.plot(rr, logZeroNaN(r_massfrac), '-', lw=lw, label='by mass')

    ax.plot([min_dist_lr,min_dist_lr], ylim, ':', color='#555555', lw=lw, alpha=0.5, label='closest LR' )

    ax2 = ax.twiny()
    ax2.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax2.set_xlim([0.0, rr.max()/halo_zoom['Group_R_Crit200']])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax.legend(loc='lower right')
    fig.tight_layout()    
    fig.savefig('contamination_profile_%s_%d.pdf' % (sPz.simName,sPz.snap))
    plt.close(fig)

def compare_contamination():
    """ Compare contamination radial profiles between runs. """
    zoomRes = 13
    hInds = [8,50,51,90]
    variants = ['sf2','sf3','sf4']

    # start plot
    fig = plt.figure(figsize=(figsize[0]*sfclean,figsize[1]*sfclean))
    ylim = [-4.0, 0.0]

    ax = fig.add_subplot(111)

    ax.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim(ylim)

    # loop over hInd/variant combination
    for hInd in hInds:
        c = ax._get_lines.prop_cycler.next()['color']

        for j, variant in enumerate(variants):
            # load zoom: group catalog
            print(hInd,variant)
            sPz = simParams(res=zoomRes, run='tng_zoom_dm', hInd=hInd, redshift=0.0, variant=variant)

            halo_zoom = groupCatSingle(sPz, haloID=0)
            halos_zoom = groupCat(sPz, fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
            subs_zoom = groupCat(sPz, fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

            # load contamination statistics and plot
            min_dist_lr, _, _, _, _, rr, r_frac, r_massfrac = calculate_contamination(sPz)
            rr /= halo_zoom['Group_R_Crit200']
            min_dist_lr /= halo_zoom['Group_R_Crit200']

            l, = ax.plot(rr, logZeroNaN(r_frac), linestyles[j], color=c, lw=lw, label='h%d_%s' % (hInd,variant))
            #l, = ax.plot(rr, logZeroNaN(r_massfrac), '--', lw=lw, color=c)

            ax.plot([min_dist_lr,min_dist_lr], [ylim[0],ylim[0]+0.5], linestyles[j], color=c, lw=lw, alpha=0.5)

    ax.plot([0,rr[-1]], [-1.0, -1.0], '-', color='#888888', lw=lw-1.0, alpha=0.4, label='10%')
    ax.plot([0,rr[-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw-1.0, alpha=0.4, label='1%')
    ax.plot([1.0,1.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)
    ax.plot([2.0,2.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)

    ax.legend(loc='upper left')
    fig.tight_layout()    
    fig.savefig('contamination_comparison_L%d_h%s_%s.pdf' % (zoomRes,'-'.join([str(h) for h in hInds]),'-'.join(variants)))
    plt.close(fig)

def sizefacComparison():
    """ Compare SizeFac 2,3,4 runs (contamination and CPU times) in the testing set. """

    # config
    #hInds    = [8,50,51,90]
    #variants = ['sf2','sf3','sf4']
    #run      = 'tng_zoom_dm'

    #hInds    = [50]
    #variants = ['sf2_n160s','sf2_n160s_mpc','sf2_n320s','sf2_n640s','sf3']

    hInds = [50,3232]
    variants = ['sf3']

    run      = 'tng_zoom'
    zoomRes  = 13
    redshift = 0.0

    # load
    results = []

    for hInd in hInds:
        for variant in variants:
            sP = simParams(run=run, res=zoomRes, hInd=hInd, redshift=redshift, variant=variant)

            cpu = loadCpuTxt(sP.arepoPath, keys=['total'])
            cpuHours = cpu['total'][0,-1,2] / (60.0*60.0) * cpu['numCPUs']

            min_dist_lr, rVirFacs, counts, fracs, massfracs, _, _, _ = calculate_contamination(sP)

            halo = groupCatSingle(sP, haloID=0)
            haloMass = sP.units.codeMassToLogMsun( halo['Group_M_Crit200'] )
            haloRvir = halo['Group_R_Crit200']

            print('Load hInd=%2d variant=%s minDist=%.2f' % (hInd,variant,min_dist_lr))

            r = {'hInd':hInd, 'variant':variant, 'cpuHours':cpuHours, 'haloMass':haloMass, 'haloRvir':haloRvir, 
                 'contam_min':min_dist_lr, 'contam_rvirfacs':rVirFacs, 'contam_counts':counts}
            results.append(r)

    # start plot
    fig = plt.figure(figsize=(22,12))

    for rowNum in [0,1]:
        xlabel = 'Halo ID' if rowNum == 0 else 'Halo Mass [log M$_{\\rm sun}$]'
        ax = fig.add_subplot(2,3,rowNum*3+1)

        # set up unique coloring by variant/sizeFac
        colors = OrderedDict()

        for variant in variants:
            c = ax._get_lines.prop_cycler.next()['color']
            colors[ variant ] = c

        handles = [plt.Line2D((0,1), (0,0), color=colors[sf], marker='o', lw=lw) for sf in colors.keys()]

        # (A) contamination dist kpc
        ylim = [-4.0, 0.0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Min LR Dist [pMpc]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['variant'] ]
            ax.plot( xx, result['contam_min'], 'o', color=color, label='')
        
        ax.legend(handles, ['%s' % variant for variant in colors.keys()], loc='best')

        # (B) contamination rvir
        ax = fig.add_subplot(2,3,rowNum*3+2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Min LR Dist [r$_{\\rm vir}$]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['variant'] ]
            ax.plot( xx, result['contam_min']/result['haloRvir'], 'o', color=color, label='')

        xlim = ax.get_xlim()
        for rVirFac in [5,2,1]:
            ax.plot( xlim, [rVirFac,rVirFac], '-', color='#bbbbbb', alpha=0.4 )

        ax.legend(handles, ['%s' % variant for variant in colors.keys()], loc='best')

        # (C) cpu hours
        ax = fig.add_subplot(2,3,rowNum*3+3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('CPU Time [log kHours]')

        for result in results:
            xx = result['hInd'] if rowNum == 0 else result['haloMass']
            color = colors[ result['variant'] ]
            ax.plot( xx, np.log10(result['cpuHours']/1e3), 'o', color=color, label='')

        ax.legend(handles, ['%s' % variant for variant in colors.keys()], loc='best')

    # finish
    fig.tight_layout()    
    fig.savefig('sizefac_comparison.pdf')
    plt.close(fig)

def parentBoxVisualComparison(haloID, variant='sf2', conf=0):
    """ Make a visual comparison (density projection images) between halos in the parent box and their zoom realizations. """
    sPz = simParams(run='tng_zoom', res=13, hInd=haloID, redshift=0.0, variant=variant)

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
