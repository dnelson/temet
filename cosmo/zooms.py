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
    # note: skipped h1096 (IC failure, spans box edge, replaced with 799)
    # note: run down to 14.5 mass bin with 10 per bin, then:
    #  increase to 20 for 14.9-15, 50 for 14.8-14.9, 40 each for 14.6-14.8
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
    #ax.yaxis.tick_right()
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
    """ Testing discrete convex hull type approach. """
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

def combineZoomGroupCatsIntoVirtualParentBox():
    """ Add a zoom simulation, at the group catalog level, into a 'virtual' parent 
    simulation, i.e. concatenate the output/group* of these runs."""
    outPath = '/u/dnelson/sims.TNG/L680n8192TNG/output/'

    # zoom config
    res = 13
    variant = 'sf3'
    run = 'tng_zoom'

    hInds = [877,901,1041] # testing

    nSnaps = 99

    def _newpartid(old_ids, halo_ind):
        """ Define convention to offset particle/cell/tracer IDs based on zoom run halo ID. """
        return halo_ind * 100000000 + old_ids # no halo has more than 100M

    # loop over snapshots
    for snap in [99]: #range(nSnaps):
        # load total number of halos and subhalos
        numHalosTot = 0
        numSubhalosTot = 0

        for hInd in hInds:
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
            print(sP.simName, sP.numHalos, sP.numSubhalos)
            numHalosTot += sP.numHalos
            numSubhalosTot += sP.numSubhalos

        numFiles = sP.groupCatHeader()['NumFiles']
        print('[snap = %2d] Total [%d] halos, [%d] subhalos.' % (snap,numHalosTot,numSubhalosTot))

        # TODO: have to choose which... FoFs are in entire box, not just high res region
        # 877: 10k FoFs, all have high-rs DM, 8k have no low-res DM, dists out to 17 Mpc
        #  -- mark targeted zoom halos (not all second-most-massive halos are <14.3)
        #  -- compute contamination fraction for all halos/subhalos
        import pdb; pdb.set_trace()

        # use first zoom run: allocate
        data = {}
        offsets = {}
        headers = {}

        sP = simParams(run=run, res=res, snap=snap, hInd=hInds[0], variant=variant)

        with h5py.File(sP.gcPath(0), 'r') as f:
            for gName in f.keys():
                if len(f[gName]):
                    # group with datasets, e.g. Group, Subhalo
                    data[gName] = {}
                    offsets[gName] = 0
                else:
                    # group with no datasets, i.e. only attributes, e.g. Header, Config, Parameters
                    headers[gName] = dict( f[gName].attrs.items() )

                for field in f[gName].keys():
                    shape = list(f[gName][field].shape)

                    # replace first dim with total length
                    if gName == 'Group':
                        shape[0] = numHalosTot
                    elif gName == 'Subhalo':
                        shape[0] = numSubhalosTot
                    else:
                        assert 0 # handle
                    
                    # allocate
                    data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)

        # add field to save originating zoom run
        data['Group']['GroupOrigHaloID'] = np.zeros(numHalosTot, dtype='int32')
        data['Subhalo']['SubhaloOrigHaloID'] = np.zeros(numSubhalosTot, dtype='int32')

        # loop over all zoom runs: load full group cats
        import pdb; pdb.set_trace()

        for hInd in hInds:
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

            for i in range(numFiles):
                print(' ',sP.hInd,i,offsets['Group'],offsets['Subhalo'])

                startOffHalo = offsets['Group'] # starting halo index for this zoom run
                startOffSub  = offsets['Subhalo'] # starting subhalo index for this zoom run

                with h5py.File(sP.gcPath(i), 'r') as f:
                    for gName in f.keys():
                        if len(f[gName]) == 0:
                            continue

                        offset = offsets[gName]

                        # load and stamp
                        for field in f[gName]:
                            length = f[gName][field].shape[0]

                            data[gName][field][offset:offset+length,...] = f[gName][field][()]

                        # save originating zoom run halo ID, and make index adjustments
                        if gName == 'Group':
                            data['Group']['GroupOrigHaloID'][offset:offset+length] = hInd
                            data['Group']['GroupFirstSub'][offset:offset+length] += startOffSub
                        if gName == 'Subhalo':
                            data['Subhalo']['SubhaloOrigHaloID'][offset:offset+length] = hInd
                            data['Subhalo']['SubhaloGrNr'][offset:offset+length] += startOffHalo
                            data['Subhalo']['SubhaloIDMostbound'][offset:offset+length] = \
                              _newpartid(data['Subhalo']['SubhaloIDMostbound'][offset:offset+length], hInd)

                        # TODO: spatial offset adjustments: GroupPos, SubhaloPos, ...

                        offsets[gName] += length

        # header adjustments
        headers['Header']['Ngroups_Total'] = numHalosTot
        headers['Header']['Nsubgroups_Total'] = numSubhalosTot

        import pdb; pdb.set_trace()

        # write into single file
        savePath = outPath + 'groups_%03d/' % snap
        outFile  = "fof_subhalo_tab_%03d.hdf5" % snap

        if not path.isdir(savePath):
            mkdir(savePath)

        with h5py.File(savePath + outFile, 'w') as f:
            # add header groups
            for gName in headers:
                f.create_group(gName)
                for attr in headers[gName]:
                    f[gName].attrs[attr] = headers[gName][attr]

            f['Header'].attrs['Ngroups_ThisFile'] = f['Header'].attrs['Ngroups_Total']
            f['Header'].attrs['Nsubgroups_ThisFile'] = f['Header'].attrs['Nsubgroups_Total']
            f['Header'].attrs['NumFiles'] = 1

            # add datasets
            for gName in data:
                f.create_group(gName)
                for field in data[gName]:
                    f[gName][field] = data[gName][field]
                    assert data[gName][field].shape[0] == offsets[gName]

        print(' Wrote [%s].' % outFile)

    print('Done.')
