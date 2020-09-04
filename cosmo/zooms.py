"""
cosmo/zooms.py
  Analysis and helpers specifically for zoom resimulations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile, isdir, expanduser
from os import mkdir
from matplotlib.ticker import MultipleLocator
from collections import OrderedDict
from scipy.stats import binned_statistic
from numba import jit

from cosmo.perf import loadCpuTxt, getCpuTxtLastTimestep
from util.simParams import simParams
from util.helper import logZeroNaN
from vis.halo import renderSingleHalo
from vis.box import renderBox
from plot.config import *

def pick_halos():
    """ Testing. """
    from vis.halo import selectHalosFromMassBins
    sP = simParams(res=2048, run='tng_dm', redshift=0.0)
    #sP = simParams(res=2500, run='tng', redshift=0.0)

    # config
    bins = [ [x+0.0,x+0.1] for x in np.linspace(14.0,15.4,15) ]
    numPerBin = 30

    hInds = selectHalosFromMassBins(sP, bins, numPerBin, 'random')

    for i, bin in enumerate(bins):
        print(bin,hInds[i])

    # note: skipped h305 (IC gen failures, replaced with 443 in its mass bin)
    # note: skipped h1096 (IC gen failure, spans box edge, replaced with 799)
    # note: skipped h604 (corrupt GroupNsubs != Nsubgroups_Total in snap==53, replaced with 616)
    return hInds

def _halo_ids_run(onlyDone=False):
    """ Parse runs.txt and return the list of (all) halo IDs. """
    path = expanduser("~") + "/sims.TNG_zooms/"

    with open(path + 'runs.txt','r') as f:
        runs_txt = [line.strip() for line in f.readlines()]

    halo_inds = []
    for i, line in enumerate(runs_txt):
        if ' ' in line:
            line = line.split(' ')[0]
        if line.isdigit():
            halo_inds.append(int(line))

    if onlyDone:
        # restrict to completed runs
        halo_inds_done = [hInd for hInd in halo_inds if isdir(path+'L680n2048TNG_h%d_L13_sf3' % hInd)]
        return halo_inds_done

    return halo_inds

def mass_function():
    """ Plot halo mass function from the parent box (TNG300) and the zoom sample. """
    mass_range = [14.0, 15.5]
    binSize = 0.1
    redshift = 0.0
    
    sP_tng300 = simParams(res=2500,run='tng',redshift=redshift)
    sP_tng1 = simParams(res=2048, run='tng_dm', redshift=redshift)

    # load halos
    halo_inds = _halo_ids_run()

    print(len(halo_inds))
    print(halo_inds)

    # start figure
    fig = plt.figure(figsize=figsize)

    nBins = int((mass_range[1]-mass_range[0])/binSize)

    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(mass_range)
    ax.set_xticks(np.arange(mass_range[0],mass_range[1],0.1))
    ax.set_xlabel('Halo Mass M$_{\\rm 200,crit}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('Number of Halos [%.1f dex$^{-1}$]' % binSize)
    ax.set_yscale('log')
    ax.yaxis.set_ticks_position('both')

    yy_max = 1.0

    hh = []
    labels = []

    for sP in [sP_tng300,sP_tng1]:
        if sP == sP_tng300:
            # tng300
            gc = sP_tng300.groupCat(fieldsHalos=['Group_M_Crit200'])
            masses = sP_tng300.units.codeMassToLogMsun(gc)
            label = 'TNG300-1'
        elif sP == sP_tng1:
            # tng1 - achieved targets (from runs.txt)
            gc = sP_tng1.groupCat(fieldsHalos=['Group_M_Crit200'])
            masses = sP_tng1.units.codeMassToLogMsun(gc[halo_inds])
            label = 'TNG1-Cluster'
        else:
            # OLD: tng1 parent box for zooms (planned targets)
            gc = sP_tng1.groupCat(fieldsHalos=['Group_M_Crit200'])
            halo_inds = pick_halos()
            first_bin = 5 # >=14.5
            masses = [gc[inds] for inds in halo_inds[first_bin:]] # list of masses in each bin
            masses = np.hstack(masses)
            masses = sP_tng1.units.codeMassToLogMsun(masses)
            label = 'TNG1-Cluster'

        w = np.where(~np.isnan(masses))
        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
        yy_max = np.nanmax([yy_max,np.nanmax(yy)])

        print(xx,yy)

        hh.append(masses[w])
        labels.append(label)

    # 'bonus': halos above 14.0 in the high-res regions of more massive zoom targets
    if 0:
        cacheFile = 'cache_mass_function_bonus.hdf5'
        if isfile(cacheFile):
            with h5py.File(cacheFile,'r') as f:
                masses = f['masses'][()]
        else:
            masses = []
            for i, hInd in enumerate(halo_inds):
                # only runs with existing data
                if not isdir('sims.TNG_zooms/L680n2048TNG_h%d_L13_sf3' % hInd):
                    print('[%3d of %3d]  skip' % (i,len(halo_inds)))
                    continue

                # load FoF catalog, record clusters with zero contamination
                sP = simParams(res=13, run='tng_zoom', redshift=redshift, hInd=hInd, variant='sf3')
                loc_masses = sP.halos('Group_M_Crit200')
                loc_masses = sP.units.codeMassToLogMsun(loc_masses[1:]) # skip FoF 0 (assume is target)

                loc_length = sP.halos('GroupLenType')[1:,:]
                contam_frac = loc_length[:,sP.ptNum('dm_lowres')] / loc_length[:,sP.ptNum('dm')]

                w = np.where( (loc_masses >= mass_range[0]) & (contam_frac == 0) )

                if len(w[0]):
                    masses = np.hstack( (masses,loc_masses[w]) )
                print('[%3d of %3d] ' % (i,len(halo_inds)), hInd, len(w[0]), len(masses))
            with h5py.File(cacheFile,'w') as f:
                f['masses'] = masses

        hh.append(masses)
        labels.append('TNG1 Bonus')

    # plot
    ax.hist(hh,bins=nBins,range=mass_range,label=labels,histtype='bar',alpha=0.9,stacked=True)

    ax.set_ylim([0.8,100])
    ax.legend(loc='upper right')

    fig.savefig('mass_functions.pdf')
    plt.close(fig)

def calculate_contamination(sPzoom, rVirFacs=[1,2,3,4,5,10], verbose=False):
    """ Calculate number of low-res DM within each rVirFac*rVir distance, as well 
    as the minimum distance to any low-res DM particle, and a radial profile of 
    contaminating particles. """
    cacheFile = sPzoom.derivPath + 'contamination_stats.hdf5'

    # check for existence of cache
    fields = ['min_dist_lr','rVirFacs','counts','fracs','massfracs','rr','r_frac','r_massfrac']

    if isfile(cacheFile):
        r = {}
        with h5py.File(cacheFile,'r') as f:
            for key in fields:
                r[key] = f[key][()]
        return [r[key] for key in fields]

    # load and calculate now
    halo = sPzoom.groupCatSingle(haloID=0)
    r200 = halo['Group_R_Crit200']

    h = sPzoom.snapshotHeader()

    pos_hr  = sPzoom.snapshotSubset('dm', 'pos')
    pos_lr  = sPzoom.snapshotSubset(2, 'pos')

    mass_lr = sPzoom.snapshotSubset(2, 'mass')
    mass_hr = h['MassTable'][sPzoom.ptNum('dm')]

    dists_lr = sPzoom.periodicDists( halo['GroupPos'], pos_lr )
    dists_hr = sPzoom.periodicDists( halo['GroupPos'], pos_hr )

    min_dist_lr = dists_lr.min() # code units
    min_dist_lr = sPzoom.units.codeLengthToMpc(min_dist_lr) # pMpc
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

    # save cache
    with h5py.File(cacheFile,'w') as f:
        for key in fields:
            f[key] = locals()[key]
    print('Saved: [%s]' % cacheFile)

    return min_dist_lr, rVirFacs, counts, fracs, massfracs, rr, r_frac, r_massfrac

def check_contamination():
    """ Check level of low-resolution contamination (DM particles) in zoom run. """
    hInd = 23 #8
    zoomRes = 11 #13
    variant = None #'sf2'

    zoomRun = 'tng50_zoom_dm' #'tng_zoom_dm'
    redshift = 0.5 # 0.0
    #sP = simParams(res=2048,run='tng_dm',redshift=redshift) # parent box
    sP = simParams(run='tng50-1', redshift=redshift)

    # load parent box: halo
    halo = sP.groupCatSingle(haloID=hInd)

    # load zoom: group catalog
    sPz = simParams(res=zoomRes, run=zoomRun, hInd=hInd, redshift=redshift, variant=variant)

    halo_zoom = sPz.groupCatSingle(haloID=0)
    halos_zoom = sPz.groupCat(fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
    subs_zoom = sPz.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

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
    fig.savefig('contamination_profile_%s_%d.pdf' % (sPz.simName,sPz.snap))
    plt.close(fig)

def compare_contamination():
    """ Compare contamination radial profiles between runs. """
    zoomRes = 13
    hInds = [8,50,51,90]
    variants = ['sf2','sf3','sf4']

    # start plot
    fig = plt.figure(figsize=figsize)
    ylim = [-4.0, 0.0]

    ax = fig.add_subplot(111)

    ax.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim(ylim)

    # loop over hInd/variant combination
    for hInd in hInds:
        c = next(ax._get_lines.prop_cycler)['color']

        for j, variant in enumerate(variants):
            # load zoom: group catalog
            print(hInd,variant)
            sPz = simParams(res=zoomRes, run='tng_zoom_dm', hInd=hInd, redshift=0.0, variant=variant)

            halo_zoom = sPz.groupCatSingle(haloID=0)
            halos_zoom = sPz.groupCat(fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
            subs_zoom = sPz.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

            # load contamination statistics and plot
            min_dist_lr, _, _, _, _, rr, r_frac, r_massfrac = calculate_contamination(sPz)
            rr /= halo_zoom['Group_R_Crit200']
            min_dist_lr /= sPz.units.codeLengthToMpc( halo_zoom['Group_R_Crit200'] )

            l, = ax.plot(rr, logZeroNaN(r_frac), linestyles[j], color=c, lw=lw, label='h%d_%s' % (hInd,variant))
            #l, = ax.plot(rr, logZeroNaN(r_massfrac), '--', lw=lw, color=c)

            ax.plot([min_dist_lr,min_dist_lr], [ylim[0],ylim[0]+0.5], linestyles[j], color=c, lw=lw, alpha=0.5)

    ax.plot([0,rr[-1]], [-1.0, -1.0], '-', color='#888888', lw=lw-1.0, alpha=0.4, label='10%')
    ax.plot([0,rr[-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw-1.0, alpha=0.4, label='1%')
    ax.plot([1.0,1.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)
    ax.plot([2.0,2.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)

    ax.legend(loc='upper left')
    fig.savefig('contamination_comparison_L%d_h%s_%s.pdf' % (zoomRes,'-'.join([str(h) for h in hInds]),'-'.join(variants)))
    plt.close(fig)

def sizefacComparison():
    """ Compare SizeFac 2,3,4 runs (contamination and CPU times) in the testing set. """

    # config
    zoomRes  = 13
    redshift = 0.0

    if 0:
        # testing contam/CPU time scalings with size fac
        hInds    = [8,50,51,90]
        variants = ['sf2','sf3','sf4']
        run      = 'tng_zoom_dm'

    if 0:
        # testing CPU time scaling with core count and unit systems
        hInds    = [50]
        variants = ['sf2_n160s','sf2_n160s_mpc','sf2_n320s','sf2_n640s','sf3']

    if 1:
        # main TNG1-Cluster sample
        hInds    = _halo_ids_run(onlyDone=True)
        variants = ['sf3']
        run      = 'tng_zoom'

    # load
    results = []

    for hInd in hInds:
        for variant in variants:
            sP = simParams(run=run, res=zoomRes, hInd=hInd, redshift=redshift, variant=variant)

            #cpu = loadCpuTxt(sP.arepoPath, keys=['total'])
            #cpuHours = cpu['total'][0,-1,2] / (60.0*60.0) * cpu['numCPUs']
            _, _, _, cpuHours = getCpuTxtLastTimestep(sP.simPath + '/txt-files/cpu.txt')

            min_dist_lr, rVirFacs, counts, fracs, massfracs, _, _, _ = calculate_contamination(sP)

            halo = sP.groupCatSingle(haloID=0)
            haloMass = sP.units.codeMassToLogMsun( halo['Group_M_Crit200'] )
            haloRvir = sP.units.codeLengthToMpc( halo['Group_R_Crit200'] )

            print('Load hInd=%4d variant=%s minDist=%5.2f' % (hInd,variant,min_dist_lr))

            r = {'hInd':hInd, 'variant':variant, 'cpuHours':cpuHours, 'haloMass':haloMass, 'haloRvir':haloRvir, 
                 'contam_min':min_dist_lr, 'contam_rvirfacs':rVirFacs, 'contam_counts':counts}
            results.append(r)

    # print some stats
    print('Median contam [pMpc]: ', np.median([result['contam_min'] for result in results]))
    print('Median contam [rVir]: ', np.median([result['contam_min']/result['haloRvir'] for result in results]))
    print('Mean CPU hours: ', np.mean([result['cpuHours'] for result in results]))

    num_lowres = []
    for result in results:
        contam_rel = result['contam_min']/result['haloRvir']
        num = result['contam_counts'][0] # 0=1rvir, 1=2rvir

        if contam_rel > 1.0:
            continue

        num_lowres.append(num)
        print(' [h = %4d] min contamination = %.2f rvir, num inside rvir = %d' % (result['hInd'],contam_rel,num))

    print(' within 1rvir, numhalos, mean median number of low-res dm: ', len(num_lowres), np.mean(num_lowres), np.median(num_lowres))

    # start plot
    fig = plt.figure(figsize=(22,12))

    for rowNum in [0,1]:
        xlabel = 'Halo ID' if rowNum == 0 else 'Halo Mass [log M$_{\\rm sun}$]'
        ax = fig.add_subplot(2,3,rowNum*3+1)

        # set up unique coloring by variant/sizeFac
        colors = OrderedDict()

        for variant in variants:
            c = next(ax._get_lines.prop_cycler)['color']
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
    fig.savefig('sizefac_comparison.pdf')
    plt.close(fig)

def parentBoxVisualComparison(haloID, variant='sf3', conf=0, snap=99):
    """ Make a visual comparison (density projection images) between halos in the parent box and their zoom realizations. """

    #sPz = simParams(run='tng_zoom', res=13, hInd=haloID, redshift=0.0, variant=variant)
    #sPz = simParams(run='tng100_zoom_dm', res=11, hInd=haloID, snap=snap, variant='sf4')
    sPz = simParams(run='tng50_zoom_dm', res=11, hInd=haloID, snap=snap, variant=None)

    # render config
    rVirFracs  = [1.0] #[0.5, 1.0] # None
    method     = 'sphMap' # sphMap
    nPixels    = [800,800] #[1920,1920]
    axes       = [0,1]
    labelZ     = True
    labelScale = True
    labelSim   = True
    labelHalo  = True
    relCoords  = True

    #size       = 6000.0
    #sizeType   = 'kpc'
    size        = 3.0
    sizeType    = 'rVirial'

    # setup panels
    if conf == 0:
        # dm column density
        p = {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 9.5]}
    if conf == 1:
        # gas column density
        p = {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 8.0]}

    panel_zoom = p.copy()
    panel_parent = p.copy()

    # haloID refers to z=0?
    if 0:
        sP_parent_z0 = sPz.sP_parent.copy()
        sP_parent_z0.setRedshift(0.0)
    else:
        # haloID refers to sPz.snap
        sP_parent_z0 = sPz.sP_parent

    # load MPB of this halo
    haloMPB = sP_parent_z0.loadMPB( sP_parent_z0.groupCatSingle(haloID=haloID)['GroupFirstSub'] )
    assert sPz.snap in haloMPB['SnapNum']

    # locate subhaloID at requested snapshot (could be z=0 or z>0)
    parSubID = haloMPB['SubfindID'][ list(haloMPB['SnapNum']).index(sPz.snap) ]

    panel_zoom.update( {'run':sPz.run, 'res':sPz.res, 'redshift':sPz.redshift, 'variant':sPz.variant, 'hInd':haloID})
    panel_parent.update( {'run':sPz.sP_parent.run, 'res':sPz.sP_parent.res, 'redshift':sPz.sP_parent.redshift, 'hInd':parSubID})

    panels = [panel_zoom, panel_parent]

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = nPixels[0]
        colorbars    = True
        saveFilename = './zoomParentBoxVisualComparison_%s_z%.1f_%s_snap%d.pdf' % (sPz.simName,sPz.redshift,p['partType'],sPz.snap)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def zoomBoxVis(sPz=None, conf=0):
    """ Make a visualization of a zoom simulation, without using/requiring group catalog information. """

    if sPz is None:
        sPz = simParams(res=11,run='tng100_zoom',redshift=0.0,hInd=5405,variant='sf4')

    # render config
    method     = 'sphMap_global'
    nPixels    = [1000,1000] #[1920,1920]
    axes       = [1,2]
    labelZ     = True
    labelScale = True
    labelSim   = True
    plotHalos  = 20

    if not sPz.isZoom:
        # full box (fixed vis in box coordinates)
        sPz = simParams(res=1820,run='tng',snap=sPz.snap)
        zoomFac = 5000.0 / sPz.boxSize # show 1 cMpc/h size region around location
        relCenPos = None
        absCenPos = [3.64e4, 3.9e4, 0.0] # from Shy's movie
        absCenPos = [3.9e4, 0.0, 3.64e4] # axes = [1,2] order
        sliceFac = 5000.0 / sPz.boxSize # 1000 ckpc/h depth
        print('Centering on: ', absCenPos)

    if sPz.isZoom:
        # zoom 5405 (fixed vis in box coordinates around the object)
        zoomFac = 5000.0 / sPz.boxSize # show 1 cMpc/h size region around location
        relCenPos = None

        absCenPos = [3.6e4+1000, 75000/2-3000, 75000/2] # axes = [1,2] order
        sliceFac = 10000.0 / sPz.boxSize # 1000 ckpc/h depth
        print('Centering on: ', absCenPos)

    # setup panels
    if conf == 0:
        # dm column density
        p = {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[4.0, 8.0]}
    if conf == 1:
        # gas column density
        p = {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 8.0]}
    if conf == 2:
        # stars
        p = {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[4.8,7.8]}

    panel_zoom = p.copy()
    #panel_parent = p.copy()

    panel_zoom.update( {'run':sPz.run, 'res':sPz.res, 'redshift':sPz.redshift, 'variant':sPz.variant, 'hInd':sPz.hInd})
    #panel_parent.update( {'run':sPz.sP_parent.run, 'res':sPz.sP_parent.res, 'redshift':sPz.sP_parent.redshift, 'hInd':parSubID})

    panels = [panel_zoom] #[panel_zoom, panel_parent]

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = nPixels[0]
        colorbars    = True
        saveFilename = './zoomBoxVis_%s_%s_snap%d.pdf' % (sPz.simName,p['partType'],sPz.snap)

    renderBox(panels, plotConfig, locals(), skipExisting=False)

@jit(nopython=True, nogil=True, cache=True)
def _mark_mask(mask, pxsize, pos, value):
    """ Helper. """
    for i in range(pos.shape[0]):
        ix = int(np.floor(pos[i,0] / pxsize))
        iy = int(np.floor(pos[i,1] / pxsize))
        iz = int(np.floor(pos[i,2] / pxsize))

        cur_val = mask[ix,iy,iz]
        if cur_val != -1 and cur_val != value:
            print(cur_val,value)
        assert cur_val == -1 or cur_val == value

        mask[ix,iy,iz] = value

@jit(nopython=True, nogil=True, cache=True)
def _volumes_from_mask(mask):
    """ Compute the 'volume' (number of grid cells) per value in the mask. 
    Return volumes[i] is the volume of halo ID i. The last entry is the unoccupied space. """
    maxval = np.max(mask)

    volumes = np.zeros( maxval+2, dtype=np.int32 )

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                val = mask[i,j,k]
                volumes[val] += 1

    return volumes

def maskBoxRegionsByHalo():
    """ Testing spatial concatenation of zoom runs via discrete convex hull type approach. """
    # zoom config
    res = 13
    variant = 'sf3'
    run = 'tng_zoom'

    #hInds = [0,10,120,13,877,901,1041] # testing
    hInds = _halo_ids_run()[0:50] # 184 for all complete until now

    snap = 99

    # mask config
    nGrid = 1024 #2048

    sP = simParams(run=run, res=res, snap=snap, hInd=hInds[0], variant=variant)
    gridSize = sP.boxSize / nGrid # linear, code units
    gridVol  = sP.units.codeLengthToMpc(gridSize)**3 # volume, pMpc^3
    volumeTot = nGrid**3 * gridVol # equals sP.units.codeLengthToMpc(sP.boxSize)**3

    # allocate mask
    mask = np.zeros( (nGrid,nGrid,nGrid), dtype='int16' ) - 1

    numHalosTot = 0
    numSubhalosTot = 0

    numHalos14nocontam = 0
    numHalos12nocontam = 0

    # test
    for hInd in hInds:
        # load
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
        halos = sP.halos(['GroupPos','GroupLenType','Group_M_Crit200'])
        halo_contam = halos['GroupLenType'][:,2] / halos['GroupLenType'][:,1]
        halo_m200 = sP.units.codeMassToLogMsun(halos['Group_M_Crit200'])

        # offset
        halos['GroupPos'] -= sP.boxSize/2
        halos['GroupPos'] += sP.zoomShiftPhys

        print(sP.simName, sP.numHalos, sP.numSubhalos, halos['GroupPos'][0,:])

        _mark_mask(mask, gridSize, halos['GroupPos'], hInd)

        # diagnostics
        numHalosTot += sP.numHalos
        numSubhalosTot += sP.numSubhalos

        with np.errstate(invalid='ignore'):
            w14 = np.where( (halo_contam == 0) & (halo_m200 >= 14.0) )
            w12 = np.where( (halo_contam == 0) & (halo_m200 >= 12.0) )

        numHalos14nocontam += len( w14[0] )
        numHalos12nocontam += len( w12[0] )

    print('\nTotal: targeted halos = %d, nHalos = %d, nSubhalos = %d' % (len(hInds),numHalosTot,numSubhalosTot))
    print('Number of total halos without contamination: [%d] above 14.0, [%d] above 12.0' % (numHalos14nocontam,numHalos12nocontam))

    # compute volume occupied by each halo
    totOccupiedVolFrac = 0.0
    volumes = _volumes_from_mask(mask) * gridVol

    for hInd in hInds + [-1]:
        frac = volumes[hInd] / volumeTot * 100
        if hInd != -1: totOccupiedVolFrac += frac
        print('[%4d] vol = [%8.1f pMpc^3], frac of total = [%8.6f%%]' % (hInd, volumes[hInd], frac))

    assert np.abs( totOccupiedVolFrac - (100-frac) ) < 1e-6 # total occupied should equal 1-(total unoccupied)

def combineZoomRunsIntoVirtualParentBox(snap=99):
    """ Combine a set of individual zoom simulations into a 'virtual' parent 
    simulation, i.e. concatenate the output/group* and output/snap* of these runs. 
    Process a single snapshot, since all are independent. Note that we write exactly one
    output groupcat file per zoom halo, and exactly two output snapshot files. """
    outPath = '/u/dnelson/sims.TNG/L680n6144TNG/output/'

    # zoom config
    res = 13
    variant = 'sf3'
    run = 'tng_zoom'

    hInds = [0,8,36,50,51,93,125,171,231,330,470,877,901,1041,2191,3297,4274] # testing
    #hInds = _halo_ids_run(onlyDone=True)

    def _newpartid(old_ids, halo_ind, ptNum):
        """ Define convention to offset particle/cell/tracer IDs based on zoom run halo ID. 
        No zoom halo has more than 100M (halo 0 has ~85M) of any type. This requires conversion 
        to LONGIDS by definition. """
        new_ids = old_ids.astype('uint64')

        # shift to start at 1 instead of 1000000000 (for IC-split/spawned types)
        if ptNum in [0,3,4,5]:
            new_ids -= (1000000000-1)

        # offset (increase) by hInd*1e9
        new_ids += halo_ind * 1000000000
        return new_ids

    # --- groupcat ---

    savePath = outPath + 'groups_%03d/' % snap
    if not isdir(savePath):
        mkdir(savePath)

    # load total number of halos and subhalos
    lengths = {'Group'   : np.zeros(len(hInds), dtype='int32'),
               'Subhalo' : np.zeros(len(hInds), dtype='int32')}

    for i, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        lengths['Group'][i] = sP.numHalos
        lengths['Subhalo'][i] = sP.numSubhalos

        # verify
        if lengths['Subhalo'][i]:
            GroupNsubs = sP.groups('GroupNsubs')
            assert lengths['Subhalo'][i] == GroupNsubs.sum() #h604 fails snap==53

    numHalosTot    = np.sum(lengths['Group'], dtype='int32')
    numSubhalosTot = np.sum(lengths['Subhalo'], dtype='int32')

    print('\nSnapshot = [%2d], total [%d] halos, [%d] subhalos.' % (snap,numHalosTot,numSubhalosTot))

    GroupLenType_hInd = np.zeros( (len(hInds),6), dtype='int32' )

    offsets = {}
    offsets['Group']   = np.hstack( (0,np.cumsum(lengths['Group'])[:-1]) )
    offsets['Subhalo'] = np.hstack( (0,np.cumsum(lengths['Subhalo'])[:-1]) )

    numFiles = sP.groupCatHeader()['NumFiles']
    print('\nCombine [%d] zooms, re-writing group catalogs:' % len(hInds))

    # use first zoom run: load header-type groups
    headers = {}
    sP = simParams(run=run, res=res, snap=snap, hInd=hInds[0], variant=variant)

    with h5py.File(sP.gcPath(sP.snap,0), 'r') as f:
        for gName in ['Config','Header','Parameters']:
            headers[gName] = dict( f[gName].attrs.items() )

    # header adjustments
    fac = 1000.0

    if headers['Header']['Redshift'] < 1e-10:
        assert headers['Header']['Redshift'] > 0
        headers['Header']['Redshift'] = 0.0
        headers['Header']['Time'] = 1.0

    headers['Header']['BoxSize'] *= fac # Mpc -> kpc units
    headers['Header']['Nids_ThisFile'] = 0 # always unused
    headers['Header']['Nids_Total'] = 0 # always unused

    headers['Header']['Ngroups_Total'] = numHalosTot
    headers['Header']['Nsubgroups_Total'] = numSubhalosTot
    headers['Header']['NumFiles'] = np.int32(len(hInds))

    headers['Parameters']['BoxSize'] *= fac # Mpc -> kpc units
    headers['Parameters']['InitCondFile'] = 'various'
    headers['Parameters']['NumFilesPerSnapshot'] = np.int32(len(hInds)*2)
    headers['Parameters']['UnitLength_in_cm'] /= fac # kpc units
    headers['Config']['LONGIDS'] = "" # True

    # loop over all zoom runs: load full group cats
    for hCount, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # loop over original files, load
        data = {'Group' : {}, 'Subhalo' : {} }
        offsets_loc = {'Group' : 0, 'Subhalo' : 0}

        for i in range(numFiles):
            with h5py.File(sP.gcPath(sP.snap,i), 'r') as f:
                # loop over groups with datasets
                for gName in data.keys():
                    if len(f[gName]) == 0:
                        continue

                    start = offsets_loc[gName]
                    length = f[gName][ list(f[gName].keys())[0] ].shape[0]

                    for field in f[gName]:
                        if field in ['SubhaloBfldDisk','SubhaloBfldHalo']:
                            continue # do not save (not fixed)

                        if i == 0:
                            # allocate
                            shape = list(f[gName][field].shape)
                            shape[0] = lengths[gName][hCount] # override chunk length with global
                            data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)

                        # read chunk
                        data[gName][field][start:start+length] = f[gName][field][()]

                    offsets_loc[gName] += length

        
        if lengths['Group'][hCount]:
            # allocate fields to save originating zoom run IDs
            data['Group']['GroupOrigHaloID'] = np.zeros(lengths['Group'][hCount], dtype='int32')

            # allocate new meta-data fields
            data['Group']['GroupPrimaryZoomTarget'] = np.zeros(lengths['Group'][hCount], dtype='int32')
            data['Group']['GroupContaminationFracByMass'] = np.zeros(lengths['Group'][hCount], dtype='float32')
            data['Group']['GroupContaminationFracByNumPart'] = np.zeros(lengths['Group'][hCount], dtype='float32')

            w = np.where(data['Group']['GroupMassType'][:,1] > 0)
            data['Group']['GroupContaminationFracByMass'][w] = \
              data['Group']['GroupMassType'][w,2] / (data['Group']['GroupMassType'][w,1]+data['Group']['GroupMassType'][w,2])

            w = np.where(data['Group']['GroupLenType'][:,1] > 0)
            data['Group']['GroupContaminationFracByNumPart'][w] = \
              data['Group']['GroupLenType'][w,2] / (data['Group']['GroupLenType'][w,1]+data['Group']['GroupLenType'][w,2])

            w = np.where((data['Group']['GroupLenType'][:,2] > 0) & (data['Group']['GroupLenType'][:,2] == 0))
            data['Group']['GroupContaminationFracByMass'][w] = 1.0
            data['Group']['GroupContaminationFracByNumPart'][w] = 1.0

            # save originating zoom run halo ID
            data['Group']['GroupOrigHaloID'][:] = hInd
            data['Group']['GroupPrimaryZoomTarget'][0] = 1

            # make index adjustments
            data['Group']['GroupFirstSub'] += offsets['Subhalo'][hCount]

            # spatial offset adjustments: un-shift zoom center and periodic shift
            for field in ['GroupCM','GroupPos']:
                data['Group'][field] -= sP.boxSize/2
                data['Group'][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data['Group'][field])

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in ['Group_R_Crit200','Group_R_Crit500','Group_R_Mean200','Group_R_TopHat200',
                          'GroupCM','GroupPos']:
                data['Group'][field] *= fac

            data['Group']['GroupBHMdot'] *= fac**(-1) # UnitLength^-1

            # record fof-scope lengths by type
            GroupLenType_hInd[hCount,:] = np.sum(data['Group']['GroupLenType'], axis=0)

        if lengths['Subhalo'][hCount]:
            data['Subhalo']['SubhaloOrigHaloID'] = np.zeros(lengths['Subhalo'][hCount], dtype='int32')

            data['Subhalo']['SubhaloOrigHaloID'][:] = hInd

            # make index adjustments
            data['Subhalo']['SubhaloGrNr'] += offsets['Group'][hCount]

            # SubhaloIDMostbound could be any type, identify any of PT1/PT2 (with low IDs) and offset those
            # such that they are unchanged after _newpartid(ptNum=0)
            w = np.where(data['Subhalo']['SubhaloIDMostbound'] < 1000000000)
            data['Subhalo']['SubhaloIDMostbound'][w] += (1000000000-1)

            data['Subhalo']['SubhaloIDMostbound'] = _newpartid(data['Subhalo']['SubhaloIDMostbound'], hInd, ptNum=0)

            # spatial offset adjustments: un-shift zoom center and periodic shift
            for field in ['SubhaloCM','SubhaloPos']:
                data['Subhalo'][field] -= sP.boxSize/2
                data['Subhalo'][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data['Subhalo'][field])

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in ['SubhaloHalfmassRad','SubhaloHalfmassRadType','SubhaloSpin',
                          'SubhaloStellarPhotometricsRad','SubhaloVmaxRad','SubhaloCM','SubhaloPos']:
                data['Subhalo'][field] *= fac

            data['Subhalo']['SubhaloBHMdot'] *= fac**(-1)

            for field in ['SubhaloBfldDisk','SubhaloBfldHalo']:
                if field in data['Subhalo']:
                    data['Subhalo'][field] *= fac**(-1.5) # UnitLength^-1.5

        # per-halo header adjustments
        headers['Header']['Ngroups_ThisFile'] = lengths['Group'][hCount]
        headers['Header']['Nsubgroups_ThisFile'] = lengths['Subhalo'][hCount]

        # write this zoom halo into single file    
        outFile  = "fof_subhalo_tab_%03d.%d.hdf5" % (snap,hCount)

        with h5py.File(savePath + outFile, 'w') as f:
            # add header groups
            for gName in headers:
                f.create_group(gName)
                for at in headers[gName]:
                    f[gName].attrs[at] = headers[gName][at]

            # add datasets
            for gName in data:
                f.create_group(gName)
                for field in data[gName]:
                    f[gName][field] = data[gName][field]

        print(' [%3d] Wrote [%s] (hInd = %4d) (offsets = %8d %8d)' % \
            (hCount,outFile,hInd,offsets['Group'][hCount],offsets['Subhalo'][hCount]))

    # --- snapshot ---

    savePath = outPath + 'snapdir_%03d/' % snap
    if not isdir(savePath):
        mkdir(savePath)

    print('\nCombine [%d] zooms, re-writing snapshots:' % len(hInds))

    # load total number of particles
    NumPart_Total = np.zeros( (len(hInds),6), dtype='int64')

    for i, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # load snapshot header from first run (reuse Parameters and Config from groupcats)
        with h5py.File(sP.snapPath(sP.snap,0), 'r') as f:
            Header_snap = dict( f['Header'].attrs.items() )

        if i == 0:
            headers['Header'] = Header_snap
        else:
            assert np.allclose(headers['Header']['MassTable'], Header_snap['MassTable'])

        # count particles
        assert np.sum(Header_snap['NumPart_Total_HighWord']) == 0

        NumPart_Total[i,:] = Header_snap['NumPart_Total']

    NumPart_Total_Global = np.sum(NumPart_Total, axis=0)

    # determine sizes/split between two files per halo
    assert np.sum(GroupLenType_hInd[:,3]) == 0

    GroupLenType_hInd[:,3] = np.int32(NumPart_Total[:,3]/2) # place half of tracers in first file

    OuterFuzzLenType_hInd = NumPart_Total - GroupLenType_hInd

    # quick save of offsets
    saveFilename = 'lengths_hind_%03d.hdf5' % snap
    with h5py.File(outPath + saveFilename,'w') as f:
        # particle lengths in all fofs for this hInd (file 1)
        f['GroupLenType_hInd'] = GroupLenType_hInd 
        # particle lengths outside fofs for this hInd (file 2)
        f['OuterFuzzLenType_hInd'] = OuterFuzzLenType_hInd 

    print(' Saved [%s%s].' % (outPath,saveFilename))

    # header adjustments
    if headers['Header']['Redshift'] < 1e-10:
        assert headers['Header']['Redshift'] > 0
        headers['Header']['Redshift'] = 0.0
        headers['Header']['Time'] = 1.0

    headers['Header']['BoxSize'] *= fac # Mpc -> kpc units
    headers['Header']['NumFilesPerSnapshot'] = np.int32(len(hInds) * 2)
    headers['Header']['UnitLength_in_cm'] /= fac # kpc units

    headers['Header']['NumPart_Total'] = np.uint32(NumPart_Total_Global & 0xFFFFFFFF) # first 32 bits
    headers['Header']['NumPart_Total_HighWord'] = np.uint32(NumPart_Total_Global >> 32)

    # loop over all zoom runs: load full group cats
    for hCount, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # loop over original files, load
        data = {}
        attr = {}
        offsets_loc = {}

        for pt in [0,1,2,3,4,5]:
            data['PartType%d' % pt] = {}
            attr['PartType%d' % pt] = {}
            offsets_loc['PartType%d' % pt] = 0

        for i in range(numFiles):
            with h5py.File(sP.snapPath(sP.snap,i), 'r') as f:
                # loop over groups with datasets
                for gName in data.keys():
                    if gName not in f or len(f[gName]) == 0:
                        continue

                    start = offsets_loc[gName]
                    length = f[gName][ list(f[gName].keys())[0] ].shape[0]
                    length_global = f['Header'].attrs['NumPart_Total'][int(gName[-1])]

                    for field in f[gName]:
                        if field in ['TimeStep','TimebinHydro','HighResGasMass']:
                            continue # do not save

                        if field not in data[gName]:
                            # allocate (could be i>0)
                            shape = list(f[gName][field].shape)
                            shape[0] = length_global # override chunk length with global (for this hInd)
                            data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)
                            attr[gName][field] = dict(f[gName][field].attrs)

                        # read chunk
                        data[gName][field][start:start+length] = f[gName][field][()]

                    offsets_loc[gName] += length

        # loop over all particle types to apply corrections
        for gName in data.keys():
            # spatial offset adjustments: un-shift zoom center and periodic shift
            ptNum = int(gName[-1])

            for field in ['CenterOfMass','Coordinates','BirthPos']:
                if field not in data[gName]:
                    continue

                data[gName][field] -= sP.boxSize/2
                data[gName][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data[gName][field])

            # make ID index adjustments
            if 'ParticleIDs' in data[gName]:
                data[gName]['ParticleIDs'] = _newpartid(data[gName]['ParticleIDs'], hInd, ptNum)

            if gName == 'PartType3':
                data[gName]['TracerID'] = _newpartid(data[gName]['TracerID'], hInd, ptNum)
                data[gName]['ParentID'] = _newpartid(data[gName]['ParentID'], hInd, ptNum)

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in ['CenterOfMass','Coordinates','BirthPos','SubfindHsml','BH_Hsml']:
                if field in data[gName]: # UnitLength
                    data[gName][field] *= fac
                    attr[gName][field]['to_cgs'] /= fac

            for field in ['Density','SubfindDensity','SubfindDMDensity','BH_Density']:
                if field in data[gName]: # UnitLength^-3
                    data[gName][field] *= fac**(-3)
                    attr[gName][field]['to_cgs'] /= fac**(-3)

        if 'MagneticField' in data['PartType0']:
            # unit meta-data missing in TNG codebase, add now
            data['PartType0']['MagneticField'] *= fac**(-1.5) # UnitLength^-1.5
            attr['PartType0']['MagneticField']['a_scaling'] = -2.0
            attr['PartType0']['MagneticField']['h_scaling'] = 1.0
            attr['PartType0']['MagneticField']['length_scaling'] = -1.5
            attr['PartType0']['MagneticField']['mass_scaling'] = 0.5
            attr['PartType0']['MagneticField']['to_cgs'] = 2.60191e-06
            attr['PartType0']['MagneticField']['velocity_scaling'] = 1.0

            data['PartType0']['MagneticFieldDivergence'] *= fac**(-2.5) # UnitLength^-2.5
            attr['PartType0']['MagneticField']['a_scaling'] = -3.0
            attr['PartType0']['MagneticField']['h_scaling'] = 2.0
            attr['PartType0']['MagneticField']['length_scaling'] = -2.5
            attr['PartType0']['MagneticField']['mass_scaling'] = 0.5
            attr['PartType0']['MagneticFieldDivergence']['to_cgs'] = 8.43220e-28
            attr['PartType0']['MagneticField']['velocity_scaling'] = 1.0

        if 'EnergyDissipation' in data['PartType0']: # UnitLength^-1
            data['PartType0']['EnergyDissipation'] *= fac**(-1)
            #attr['PartType0']['EnergyDissipation']['to_cgs'] /= fac**(-1) # meta-data not present

        if len(data['PartType5']):
            data['PartType5']['BH_BPressure'] *= fac**(-3) # assume same as BH_Pressure (todo verify!)
            attr['PartType5']['BH_BPressure']['a_scaling'] = -4.0 # unit meta-data missing
            attr['PartType5']['BH_BPressure']['h_scaling'] = 2.0 # units are (MagneticField)^2
            attr['PartType5']['BH_BPressure']['length_scaling'] = -3.0
            attr['PartType5']['BH_BPressure']['mass_scaling'] = 1.0
            attr['PartType5']['BH_BPressure']['to_cgs'] = 6.76994e-12 #attr['PartType0']['MagneticField']['to_cgs']**2
            attr['PartType5']['BH_BPressure']['velocity_scaling'] = 2.0

            data['PartType5']['BH_Pressure'] *= fac**(-3) # Pressure = UnitLength^-3 (assuming wrong in io_fields.c)
            attr['PartType5']['BH_Pressure']['to_cgs'] /= fac**(-3)

            for field in ['BH_CumEgyInjection_RM','BH_CumEgyInjection_QM']:
                data['PartType5'][field] *= 1 # UnitLength^0
                attr['PartType5'][field]['to_cgs'] *= 1

            for field in ['BH_Mdot','BH_MdotBondi','BH_MdotEddington']: # UnitMass/UnitTime = UnitLength^-1
                data['PartType5'][field] *= fac**(-1) # (assuming wrong in io_fields.c)
                attr['PartType5'][field]['to_cgs'] /= fac**(-1)

        # write this zoom halo into two files: one for fof-particles, one for outside-fof-particles
        for fileNum in [0,1]:
            # per-halo header adjustments
            if fileNum == 0:
                headers['Header']['NumPart_ThisFile'] = GroupLenType_hInd[hCount,:]
                start = np.zeros( 6, dtype='int32' )
            else:
                headers['Header']['NumPart_ThisFile'] = OuterFuzzLenType_hInd[hCount,:]
                start = GroupLenType_hInd[hCount,:]
            
            outFile = "snap_%03d.%d.hdf5" % (snap,hCount+len(hInds)*fileNum)

            with h5py.File(savePath + outFile, 'w') as f:
                # add header groups
                for gName in headers:
                    f.create_group(gName)
                    for at in headers[gName]:
                        f[gName].attrs[at] = headers[gName][at]

                # add datasets
                for gName in data:
                    ptNum = int(gName[-1])
                    length_loc = headers['Header']['NumPart_ThisFile'][ptNum]

                    if length_loc == 0:
                        continue

                    f.create_group(gName)

                    for field in data[gName]:
                        # write
                        f[gName][field] = data[gName][field][start[ptNum] : start[ptNum] + length_loc]

                        # add unit meta-data
                        for at in attr[gName][field]:
                            f[gName][field].attrs[at] = attr[gName][field][at]

            print(' [%3d] hInd = %4d (gas %8d - %8d of %8d) Wrote: [%s]' % \
                (hCount,hInd,start[0],start[0]+headers['Header']['NumPart_ThisFile'][0],NumPart_Total[hCount,0],outFile))

    print('Done.')

    # TODO: add offsets from lengths_hind_%03d.hdf5 to offsets files during their construction

def testVirtualParentBoxGroupCat(snap=99):
    """ Compare all group cat fields (1d histograms) vs TNG300-1 to check unit conversions, etc. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    nBins = 50

    sP1 = simParams(run='tng-cluster', snap=snap)
    sP2 = simParams(run='tng300-1', snap=snap)

    # compare group catalogs: entire (un-contaminated) TNG-Cluster vs. ~same (first) N of TNG300-1
    contam = sP1.halos('GroupContaminationFracByMass')
    m200 = sP1.units.codeMassToLogMsun(sP1.halos('Group_M_Crit200'))

    haloIDs = np.where( (contam == 0) & (m200 > 14.0) )[0]

    # subhalos: all of these groups
    nSubs = sP1.halos('GroupNsubs')[haloIDs]
    firstSub = sP1.halos('GroupFirstSub')[haloIDs]

    subIDs = np.hstack( [np.arange(nSubs[i]) + firstSub[i] for i in range(haloIDs.size)] )

    for gName in ['Group','Subhalo']:
        # get list of halo/subhalo properties
        with h5py.File(sP2.gcPath(sP2.snap,0),'r') as f:
            fields = list(f[gName].keys())

        # start pdf book
        pdf = PdfPages('compare_%s_%s_%s_%d.pdf' % (gName,sP1.simName,sP2.simName,snap))

        for field in fields:
            # start plot
            print(field)
            if field in ['SubhaloFlag','SubhaloBfldDisk','SubhaloBfldHalo']: continue

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(field + ' [log]')
            ax.set_ylabel('log N')

            # load and histogram
            count = 0

            for sP in [sP1,sP2]:
                if gName == 'Group':
                    vals = sP.halos(field)
                    if count == 0: vals = vals[haloIDs] # sP1 uncontaminated
                if gName == 'Subhalo':
                    vals = sP.subhalos(field)
                    if count == 0: vals = vals[subIDs] # sP1 uncontaminated

                if count > 0: # slice on sP2
                    vals = vals[0:count]
                if count == 0: # set on sP1
                    count = vals.shape[0]

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel() # 1D for all multi-D

                if field not in ['GroupCM','GroupPos','SubhaloCM','SubhaloGrNr','SubhaloIDMostbound']:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, label=sP.simName)

            # finish plot
            ax.legend(loc='best')
            pdf.savefig()
            plt.close(fig)

        # finish
        pdf.close()

def testVirtualParentBoxSnapshot(snap=99):
    """ Compare all snapshot fields (1d histograms) vs TNG300-1 to check unit conversions, etc. """
    from matplotlib.backends.backend_pdf import PdfPages
    from util.helper import closest

    # config
    haloID = 4 # for particle comparison, indexing primary targets of TNG-Cluster
    nBins = 50

    sP1 = simParams(run='tng-cluster', snap=snap)
    sP2 = simParams(run='tng300-1', snap=snap)

    # compare particle fields: one halo
    pri_target = sP1.halos('GroupPrimaryZoomTarget')

    sP1_hInd = np.where(pri_target)[0][haloID]
    sP1_m200 = sP1.halo(sP1_hInd)['Group_M_Crit200']
    zoomOrigID = sP1.groupCatSingle(haloID=sP1_hInd)['GroupOrigHaloID']

    # locate close mass in sP2 to compare to
    sP2_m200, sP2_hInd = closest(sP2.halos('Group_M_Crit200'), sP1_m200)

    haloIDs = [sP1_hInd, sP2_hInd]

    print('Comparing hInd [%d (%d)] from TNG-Cluster to [%d] from TNG300-1 (%.1f vs %.1f log msun).' % \
        (sP1_hInd,zoomOrigID,sP2_hInd,sP1.units.codeMassToLogMsun(sP1_m200),sP2.units.codeMassToLogMsun(sP2_m200)))

    if 0:
        # debugging: load one field
        pt = 'dm'
        field = 'ParticleIDs'

        vals1 = sP1.snapshotSubset(pt, field, haloID=sP1_hInd)
        vals2 = sP2.snapshotSubset(pt, field, haloID=sP2_hInd)

        print(sP1.simName, ' min max mean: ', vals1.min(), vals1.max(), np.mean(vals1))
        print(sP2.simName, ' min max mean: ', vals2.min(), vals2.max(), np.mean(vals2))

        import pdb; pdb.set_trace()

    # loop over part types
    for ptNum in [0,1,4,5]: # skip low-res DM (2) and tracers (3)
        gName = 'PartType%d' % ptNum

        # get list of particle datasets
        with h5py.File(sP2.snapPath(sP2.snap,0),'r') as f:
            fields = list(f[gName].keys())

        # start pdf book
        pdf = PdfPages('compare_%s_%s_%s_h%d_%d.pdf' % (gName,sP1.simName,sP2.simName,haloID,snap))

        for field in fields:
            # start plot
            print(gName, field)
            if field in ['InternalEnergyOld','StellarHsml']:
                continue

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(field + ' [log]')
            ax.set_ylabel('N')
            ax.set_yscale('log')

            # load and histogram
            for i, sP in enumerate([sP1,sP2]):
                vals = sP.snapshotSubset(ptNum, field, haloID=haloIDs[i])

                if field == 'ParticleIDs' and i == 0: # verify ID spacing
                    offset = 1000000000*zoomOrigID
                    assert (vals-offset).min() > 0 and (vals-offset).max() < 1000000000

                if field == 'Potential': vals *= -1

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel() # 1D for all multi-D

                if field not in []:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, label=sP.simName)

            # finish plot
            ax.legend(loc='best')
            pdf.savefig()
            plt.close(fig)

        # finish
        pdf.close()

def vis_fullbox_virtual(conf=0):
    """ Visualize the entire virtual reconstructed box. """
    from vis.box import renderBox

    sP = simParams(run='tng-cluster', redshift=0.0)

    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    nPixels    = 2000

    # halo plotting
    plotHalos  = False

    if conf in [0,1,2,3,4]:
        pri = sP.groups('GroupPrimaryZoomTarget')
        plotHaloIDs = np.where(pri == 1)[0]

    # panel config
    if conf == 0:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.5,7.1]}]
    if conf == 1:
        method = 'sphMap' # is global, overlapping coarse cells
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.7,8.0]}]
    if conf == 2:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]
    if conf == 3:
        method = 'sphMap' # is global
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]
    if conf == 4:
        method = 'sphMap' # is global
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.2,6.8]}]
        numBufferLevels = 3 # 2 or 3, free parameter

        maxGasCellMass = sP.targetGasMass
        if numBufferLevels >= 1:
            # first buffer level is 27x mass, then 8x mass for each subsequent level
            maxGasCellMass *= 27 * np.power(8,numBufferLevels-1)
            # add padding for x2 Gaussian distribution
            maxGasCellMass *= 3

        ptRestrictions = {'Masses':['lt',maxGasCellMass]}

    if conf == 5:
        sP = simParams(run='tng_dm',res=2048,redshift=0.0) # parent box
        method = 'sphMap'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[7.0,8.4]}]

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = [nPixels,nPixels]
        colorbars  = True

        saveFilename = './boxImage_%s_%s-%s_%s_conf%d.pdf' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],sP.snap,conf)

    renderBox(panels, plotConfig, locals(), skipExisting=False)

def test_variant_box():
    """ Test if a variant box is responsible for a noted difference. """
    from matplotlib.backends.backend_pdf import PdfPages

    sP1 = simParams(run='tng',res=512,redshift=0.0,variant='0000')
    sP2 = simParams(run='tng',res=512,redshift=0.0,variant='5008')

    # 4503 = sh03: STEEPER_SFR_FOR_STARBURST (1.0)
    # 5001 = 17Oct2016 gfm_metal_cooling bugfix
    # 5002 = 17Nov2016 benergy_wind_spawning fix
    # 5003 = 8May2017 z>6 UVB fix
    # 5004 = 6Apr2018 velocity gradients sign/limiter fixes (see StarFormationRate!)
    # 5005 = 7May2018 vel slope limiter fix
    # --- not included in TNG-Cluster -- 5006 = 7May2018 wind-recoupling/stellar-evo energy injection fix
    # --- not included in TNG-Cluster -- 5007 = 13May2018 gas total energy kick fix
    # 5008 = 3Oct2018 mhd missing scalefactor in powell source term fix
    # --- not included in TNG-Cluster -- 5011 = subfind phase2 pot fix (see StarFormationRate!)

    # TODO: understand PartType0: ElectronAbundance, GFM_CoolingRate, StarFormationRate differences

    # TODO: verify PartType0: NeutralHydrogenAbundance, InternalEnergy

    pt = 'gas'
    fields = ['ElectronAbundance','GFM_CoolingRate','StarFormationRate','NeutralHydrogenAbundance','InternalEnergy']
    haloID = 0

    pdf = PdfPages('compare_%s_h%d_%s.pdf' % (sP2.simName,haloID,pt))

    for field in fields:
        # start plot
        print(field)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlabel(field + ' [log]')
        ax.set_ylabel('N')
        ax.set_yscale('log')

        # load and histogram
        for i, sP in enumerate([sP1,sP2]):
            vals = sP.snapshotSubset(pt, field, haloID=haloID)

            vals = vals[np.isfinite(vals) & (vals > 0)]
            vals = vals.ravel() # 1D for all multi-D

            vals = np.log10(vals)

            ax.hist(vals, bins=50, alpha=0.6, label=sP.simName)

        # finish plot
        ax.legend(loc='best')
        pdf.savefig()
        plt.close(fig)

    # finish
    pdf.close()

def check_all():
    """ TEMP CHECK NSUBS. """
    # zoom config
    res = 13
    variant = 'sf3'
    run = 'tng_zoom'

    hInds = [3425] #_halo_ids_run(onlyDone=True)

    # load total number of halos and subhalos
    lengths = {'Group'   : np.zeros(len(hInds), dtype='int32'),
               'Subhalo' : np.zeros(len(hInds), dtype='int32')}

    for i, hInd in enumerate(hInds):
        print(i,hInd)
        if hInd in [5,701,1039,1067,2766]:
            print(' skip')
            continue

        for snap in range(100):
            print(' snap: ',snap)
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

            lengths['Group'][i] = sP.numHalos
            lengths['Subhalo'][i] = sP.numSubhalos

            if lengths['Subhalo'][i]:
                assert lengths['Subhalo'][i] == sP.groups('GroupNsubs').sum()
