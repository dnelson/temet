"""
Analysis and helpers specifically for zoom resimulations in cosmological volumes.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from collections import OrderedDict
from glob import glob

from ..cosmo.perf import getCpuTxtLastTimestep
from ..util.simParams import simParams
from ..util.helper import logZeroNaN, running_median
from ..vis.halo import renderSingleHalo
from ..vis.box import renderBox
from ..plot.config import *

def pick_halos():
    """ Testing. """
    from ..vis.halo import selectHalosFromMassBins
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

def _halo_ids_run(res=14, onlyDone=False):
    """ Parse runs.txt and return the list of (all) halo IDs. """
    path = "/virgotng/mpia/TNG-Cluster/individual/"
    path2 = "/virgotng/mpia/TNG-Cluster/inprogress/"

    # runs.txt file no longer relevant, use directories which exist
    dirs = glob(path + 'L680n2048TNG_h*_L%d_sf3' % res)
    halo_inds = sorted([int(folder.split('_')[-3][1:]) for folder in dirs])

    if onlyDone:
        return halo_inds

    dirs_inprogress = glob(path2 + 'L680n2048TNG_h*_L%d_sf3' % res)
    halo_inds2 = sorted([int(folder.split('_')[-3][1:]) for folder in dirs_inprogress])

    return sorted(halo_inds + halo_inds2)

def calculate_contamination(sPzoom, rVirFacs=[1,2,3,4,5,10], verbose=False):
    """ Calculate number of low-res DM within each rVirFac*rVir distance, as well 
    as the minimum distance to any low-res DM particle, and a radial profile of 
    contaminating particles. """
    cacheFile = sPzoom.derivPath + 'contamination_stats.hdf5'

    # check for existence of cache
    if isfile(cacheFile):
        r = {}
        with h5py.File(cacheFile,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

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
    nbins = 50

    r_count_hr, rr = np.histogram(dists_hr, bins=nbins, range=rlim)
    r_count_lr, _  = np.histogram(dists_lr, bins=nbins, range=rlim)

    r_mass_lr, _ = np.histogram(dists_lr, bins=nbins, range=rlim, weights=mass_lr)
    r_mass_hr    = r_count_hr * mass_hr

    r_frac     = r_count_lr / r_count_hr
    r_massfrac = r_mass_lr / r_mass_hr

    r_frac_cum  = np.cumsum(r_count_lr) / np.cumsum(r_count_hr)
    r_massfrac_cum = np.cumsum(r_mass_lr) / np.cumsum(r_mass_hr)

    rr = rr[:-1] + (rlim[1]-rlim[0])/nbins # bin midpoints

    # save cache
    r = {'min_dist_lr':min_dist_lr, 'rVirFacs':rVirFacs, 'counts':counts, 'fracs':fracs, 
         'massfracs':massfracs, 'rr':rr, 'r_frac':r_frac, 'r_massfrac':r_massfrac,
         'r_massfrac_cum':r_massfrac_cum, 'r_frac_cum':r_frac_cum}

    with h5py.File(cacheFile,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved: [%s]' % cacheFile)

    return r

def contamination_profile():
    """ Check level of low-resolution contamination (DM particles) in zoom run. Plot radial profile. """
    # config
    hInd = 0 #31619 # 10677
    zoomRes = 14
    variant = 'sf3' # TNG

    zoomRun = 'tng_zoom' # 'tng50_zoom_dm', 'structures'
    redshift = 0.0 # 3.0

    # load zoom: group catalog
    #sPz = simParams(res=zoomRes, run=zoomRun, hInd=hInd, redshift=redshift, variant=variant)
    sPz = simParams(run='tng50_zoom', hInd=11, res=11, variant='sf8', redshift=6.0)

    halo_zoom = sPz.groupCatSingle(haloID=0)
    halos_zoom = sPz.groupCat(fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
    subs_zoom = sPz.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

    # load parent box
    sP = sPz.sP_parent
    halo = sP.groupCatSingle(haloID=hInd)

    print('parent halo pos: ', halo['GroupPos'])
    print('zoom halo cenrelpos: ', halo_zoom['GroupPos'] - sP.boxSize/2)
    print('parent halo mass: ',sP.units.codeMassToLogMsun([halo['Group_M_Crit200'],halo['GroupMass']]))
    print('zoom halo mass: ',sP.units.codeMassToLogMsun([halo_zoom['Group_M_Crit200'],halo_zoom['GroupMass']]))

    # print/load contamination statistics
    contam = calculate_contamination(sPz, verbose=True)
    min_dist_lr = contam['min_dist_lr'] * sPz.HubbleParam

    # plot contamination profiles
    fig, ax = plt.subplots(figsize=figsize)
    ylim = [-5.0, 0.0]

    ax.set_xlabel('Distance [%s]' % sPz.units.UnitLength_str)
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.set_xlim([0.0, contam['rr'].max()])
    ax.set_ylim(ylim)

    ax.plot([0,contam['rr'][-1]], [-1.0, -1.0], '-', color='#888888', lw=lw, alpha=0.5, label='10%')
    ax.plot([0,contam['rr'][-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw, alpha=0.2, label='1%')
    ax.plot(contam['rr'], logZeroNaN(contam['r_frac']), '-', lw=lw, label='by number')
    ax.plot(contam['rr'], logZeroNaN(contam['r_massfrac']), '-', lw=lw, label='by mass')

    ax.plot([min_dist_lr,min_dist_lr], ylim, ':', color='#555555', lw=lw, alpha=0.5, label='closest LR' )

    ax2 = ax.twiny()
    ax2.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax2.set_xlim([0.0, contam['rr'].max()/halo_zoom['Group_R_Crit200']])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax.legend(loc='lower right')
    fig.savefig('contamination_profile_%s_%d.pdf' % (sPz.simName,sPz.snap))
    plt.close(fig)

def contamination_compare_profiles():
    """ Compare contamination radial profiles between runs. """
    zoomRes = 14
    hInds = [0,50,79,84,102,107,1335,1919,3232,3693]
    variants = ['sf3'] #['sf2','sf3','sf4']
    run = 'tng_zoom'

    # start plot
    fig, ax = plt.subplots(figsize=figsize)
    ylim = [-6.0, 0.0]

    ax.set_xlabel('Distance [$R_{\\rm 200,crit}$]')
    ax.set_ylabel('Low-res DM Contamination Fraction [log]')
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim(ylim)

    # load: loop over hInd/variant combination
    for hInd in hInds:
        c = next(ax._get_lines.prop_cycler)['color']

        for j, variant in enumerate(variants):
            # load zoom: group catalog
            sPz = simParams(res=zoomRes, run=run, hInd=hInd, redshift=0.0, variant=variant)

            halo_zoom = sPz.groupCatSingle(haloID=0)
            halos_zoom = sPz.groupCat(fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
            subs_zoom = sPz.groupCat(fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

            # load contamination statistics and plot
            contam = calculate_contamination(sPz, verbose=True)
            rr = contam['rr'] / halo_zoom['Group_R_Crit200']
            min_dist_lr = contam['min_dist_lr'] / sPz.units.codeLengthToMpc( halo_zoom['Group_R_Crit200'] )

            l, = ax.plot(rr, logZeroNaN(contam['r_frac']), linestyles[j], color=c, lw=lw, label='h%d_%s' % (hInd,variant))
            #l, = ax.plot(rr, logZeroNaN(contam['r_massfrac']), '--', lw=lw, color=c)

            ax.plot([min_dist_lr,min_dist_lr], [ylim[1]-0.3,ylim[1]], linestyles[j], color=c, lw=lw, alpha=0.5)
            print(hInd,variant,min_dist_lr)

    ax.plot([0,rr[-1]], [-1.0, -1.0], '-', color='#888888', lw=lw-1.0, alpha=0.4, label='10%')
    ax.plot([0,rr[-1]], [-2.0, -2.0], '-', color='#bbbbbb', lw=lw-1.0, alpha=0.4, label='1%')
    ax.plot([1.0,1.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)
    ax.plot([2.0,2.0], ylim, '-', color='#bbbbbb', lw=lw-1.0, alpha=0.2)

    ax.legend(loc='upper left')
    fig.savefig('contamination_profiles_L%d_hN%d_%s.pdf' % (zoomRes,len(hInds),'-'.join(variants)))
    plt.close(fig)

def contamination_mindist():
    """ Plot distribution of contamination minimum distances, and trend with halo mass. """
    # config
    zoomRes = 14
    hInds = _halo_ids_run(onlyDone=True)
    variant = 'sf3'
    redshift = 0.0
    run = 'tng_zoom'

    frac_thresh = 1e-6

    # load data
    halo_mass = np.zeros(len(hInds), dtype='float32')
    min_dists = np.zeros(len(hInds), dtype='float32')
    min_dists_thresh = np.zeros(len(hInds), dtype='float32')

    for i, hInd in enumerate(hInds):
        if i % len(hInds)//10 == 0: print(f'{i*10:2d}% ')

        sPz = simParams(res=zoomRes, run=run, hInd=hInd, redshift=redshift, variant=variant)
        halo_zoom = sPz.groupCatSingle(haloID=0)
        halo_mass[i] = sPz.units.codeMassToLogMsun(halo_zoom['Group_M_Crit200'])

        #min_dist_lr, _, _, _, _, rr, r_frac, r_massfrac = calculate_contamination(sPz)
        contam = calculate_contamination(sPz)

        # minimum distance to first LR particle
        min_dist_lr = contam['min_dist_lr'] / sPz.units.codeLengthToMpc(halo_zoom['Group_R_Crit200'])
        min_dists[i] = min_dist_lr

        # distance at which cumulative fraction of LR/HR particles exceeds a threshold (linear interp)
        #min_ind = np.where(contam['r_frac_cum'] > frac_thresh)[0].min()
        #min_dists_thresh[i] = contam['rr'][min_ind] / halo_zoom['Group_R_Crit200']
        min_dists_thresh[i] = np.interp(frac_thresh, contam['r_frac_cum'], contam['rr']) / halo_zoom['Group_R_Crit200']   
    
    # plot distribution
    xlim = [0.0, 6.0]
    nbins = 60

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Minimum Contamination Distance [$R_{\\rm 200,crit}$]')
    ax.set_ylabel('Number of Halos')
    ax.set_xlim(xlim)

    label1 = 'Single Closest LR Particle'
    label2 = 'Low-Resolution Fraction $f_{\\rm LR} > 10^{%d}$' % np.log10(frac_thresh)
    ax.hist(min_dists, bins=nbins, range=xlim, lw=lw, alpha=0.7, label=label1)
    ax.hist(min_dists_thresh, bins=nbins, range=xlim, lw=lw, alpha=0.7, label=label2)
    ax.legend(loc='upper right')

    ax.plot([1,1], ax.get_ylim(), '-', color='#bbbbbb', lw=lw, alpha=0.2)

    fig.savefig('contamination_mindist_L%d_hN%d_%s.pdf' % (zoomRes,len(hInds),variant))
    plt.close(fig)

    # plot min dist vs mass trend
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Halo Mass M$_{\\rm 200c}$ [ log $M_{\\odot}$ ]')
    ax.set_ylabel('Minimum Contamination Distance [ $R_{\\rm 200,crit}$ ]')
    ax.set_xlim([14.25, 15.4])
    ax.set_ylim([0.0, 8.0])

    for rr in np.arange(1,7):
        ax.plot(ax.get_xlim(), [rr,rr], '-', color='#bbbbbb', lw=lw, alpha=0.2)
    ax.plot(halo_mass, min_dists, 'o', label=label1)
    ax.plot(halo_mass, min_dists_thresh, 'o', label=label2)

    xm, ym, _ = running_median(halo_mass, min_dists, binSize=0.1)
    xm2, ym2, _ = running_median(halo_mass, min_dists_thresh, binSize=0.1)
    ym = savgol_filter(ym, sKn, sKo)
    ax.plot(xm, ym, '--', lw=lw*2, color='black', alpha=0.7, label='Median (Closest LR particle)')
    ax.plot(xm2, ym2, '-', lw=lw*2, color='black', label='Median ($f_{\\rm LR} > 10^{%d}$)' % np.log10(frac_thresh))

    ax.legend(loc='upper left')
    fig.savefig('contamination_mindist_vs_mass_L%d_hN%d_%s.pdf' % (zoomRes,len(hInds),variant))
    plt.close(fig)

def sizefacComparison():
    """ Compare SizeFac 2,3,4 runs (contamination and CPU times) in the testing set. """

    # config
    zoomRes  = 14
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
        # main TNG-Cluster sample
        hInds    = _halo_ids_run(onlyDone=True)
        variants = ['sf3']
        run      = 'tng_zoom'

    # load
    results = []

    for hInd in hInds:
        for variant in variants:
            sP = simParams(run=run, res=zoomRes, hInd=hInd, redshift=redshift, variant=variant)

            _, _, _, cpuHours = getCpuTxtLastTimestep(sP.simPath + '/txt-files/cpu.txt')

            contam = calculate_contamination(sP, verbose=True)

            halo = sP.groupCatSingle(haloID=0)
            haloMass = sP.units.codeMassToLogMsun( halo['Group_M_Crit200'] )
            haloRvir = sP.units.codeLengthToMpc( halo['Group_R_Crit200'] )

            print('Load hInd=%4d variant=%s minDist=%5.2f' % (hInd,variant,contam['min_dist_lr']))

            r = {'hInd':hInd, 'variant':variant, 'cpuHours':cpuHours, 'haloMass':haloMass, 'haloRvir':haloRvir, 
                 'contam_min':contam['min_dist_lr'], 'contam_rvirfacs':contam['rVirFacs'], 
                 'contam_counts':contam['counts']}
            results.append(r)

    # print some stats
    print('Median contam [pMpc]: ', np.median([result['contam_min'] for result in results]))
    print('Median contam [rVir]: ', np.median([result['contam_min']/result['haloRvir'] for result in results]))
    print('Mean CPU hours: ', np.mean([result['cpuHours'] for result in results]))

    num_lowres = []
    for result in results:
        contam_rel = result['contam_min']*sP.HubbleParam/result['haloRvir']
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
            ax.plot( xx, result['contam_min']*sP.HubbleParam/result['haloRvir'], 'o', color=color, label='')

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

def parentBoxVisualComparison(haloID, conf=0):
    """ Make a visual comparison (density projection images) between halos in the parent box 
    and their zoom realizations.
    
    Args:
      haloID (int): the zoom halo ID, at the final redshift (z=0).
      variant (str): the zoom variant.
      conf (int): the plotting configuration.
      snap (int): if not the final snapshot, plot at some redshift other than z=0.
    """
    sPz = simParams(run='tng50_zoom', res=11, hInd=haloID, redshift=6.0, variant='sf8')

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

    #size       = 500.0
    #sizeType   = 'kpc'
    size        = 4.0
    sizeType    = 'rVirial'

    # setup panels
    if conf == 0:
        # dm column density
        p = {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 9.5]}
    if conf == 1:
        # gas column density
        p = {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.5, 8.0]}
    if conf == 2:
        # stellar density
        p = {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0, 8.0]}

    panel_zoom = p.copy()
    panel_parent = p.copy()

    # sPz at a different redshift than the parent volume?
    if np.abs(sPz.redshift - sPz.sP_parent.redshift) > 0.1:
        # load MPB of this halo
        haloMPB = sPz.sP_parent.loadMPB( sPz.sP_parent.groupCatSingle(haloID=haloID)['GroupFirstSub'] )
        assert sPz.snap in haloMPB['SnapNum']

        # locate subhaloID at requested snapshot (could be z=0 or z>0)
        parSubID = haloMPB['SubfindID'][ list(haloMPB['SnapNum']).index(sPz.snap) ]
    else:
        # same redshift
        parSubID = sPz.sP_parent.halo(haloID)['GroupFirstSub']

    panel_zoom.update( {'sP':sPz})
    panel_parent.update( {'sP':sPz.sP_parent, 'subhaloInd':parSubID})

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
        zoomFac = 3000.0 / sPz.boxSize # show 1 cMpc/h size region around location
        relCenPos = [0.5,0.5,0.5]

        sliceFac = 10000.0 / sPz.boxSize # 1000 ckpc/h depth
        #absCenPos = [3.6e4+1000, 75000/2-3000, 75000/2] # axes = [1,2] order
        absCenPos = None
        #print('Centering on: ', absCenPos)

    # setup panels
    if conf == 0:
        # dm column density
        p = {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[5.0, 9.0]}
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

def plot_timeevo():
    """ Diagnostic plots: group catalog properties for one halo vs time (no merger trees). """
    # config simulations and field
    sims = []
    sims.append( simParams(run='tng_zoom', res=14, hInd=1335, variant='sf3'))
    sims.append( simParams(run='tng_zoom', res=14, hInd=1335, variant='sf3_s'))
    sims.append( simParams(run='tng_zoom', res=14, hInd=1335, variant='sf3_kpc'))
    sims.append( simParams(run='tng_zoom', res=14, hInd=1919, variant='sf3'))
    sims.append( simParams(run='tng_zoom', res=14, hInd=1919, variant='sf3_s'))
    sims.append( simParams(run='tng_zoom', res=14, hInd=1919, variant='sf3_kpc'))

    # SubhaloSFR, SubhaloBHMass, SubhaloMassInRadType, SubhaloHalfmassRadType
    # SubhaloGasMetallicity, SubhaloVelDisp, SubaloVmaxRad
    field = 'SubhaloMassInRadType' #VmaxRad' 
    fieldIndex = 4 #-1 # -1 for scalar fields, otherwise >=0 index to use
    subhaloID = 0

    # quick check
    for sim in sims:
        sim.setSnap(33)
        subhalo = sim.subhalo(subhaloID)
        bhmass = sim.units.codeMassToLogMsun(subhalo['SubhaloBHMass'])[0]
        mstar = sim.units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][4])[0]

        print(f'{sim = } {bhmass = :.3f} {mstar = :.3f}')
    #return

    # load
    data = []

    for sim in sims:
        cache_file = 'cache_%s_%s_%d.hdf5' % (sim.simName,field,fieldIndex)

        if isfile(cache_file):
            with h5py.File(cache_file,'r') as f:
                data.append( {'sim':sim, 'result':f['result'][()], 'z':f['z'][()]} )
            print('Loaded: [%s]' % cache_file)
            continue

        # load
        snaps = sim.validSnapList()
        z = sim.snapNumToRedshift(snaps)

        result = np.zeros(snaps.size, dtype='float32')
        result.fill(np.nan)

        for i, snap in enumerate(snaps):
            print(sim.simName, snap)
            # set snap and load single subhalo from group catalog
            sim.setSnap(snap)
            subhalo = sim.subhalo(subhaloID)

            if field not in subhalo:
                print(' skip')
                continue

            # store result
            if subhalo[field].size > 1: assert fieldIndex >= 0
            if subhalo[field].size == 1: assert fieldIndex == -1

            result[i] = subhalo[field] if subhalo[field].size == 1 else subhalo[field][fieldIndex]

        assert len(result) == z.size

        # save cache
        with h5py.File(cache_file,'w') as f:
            f['result'] = result
            f['z'] = z
        print('Saved: [%s]' % cache_file)

        data.append( {'sim':sim, 'result':result, 'z':z} )

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ylabel = field if fieldIndex == -1 else field + '-%d' % fieldIndex
    if 'Mass' in field: ylabel += ' [ log M$_{\\rm sun}$ ]'
    if 'SFR' in field: ylabel += ' [ M$_{\\rm sun}$/yr ]'
    if 'HalfmassRad' in field: ylabel += ' [ log ckpc/h ]'

    ax.set_xscale('symlog')
    ax.set_xlabel('Redshift')
    ax.set_xticks([0,1,2,4,6,10])
    ax.set_ylabel(ylabel)
    #ax.set_yscale('log')

    # loop over runs
    for i, sim in enumerate(sims):
        # load
        x = data[i]['z']
        y = data[i]['result']

        # unit conversions?
        if 'Mass' in field:
            y = sim.units.codeMassToLogMsun(y)
        if 'HalfmassRad' in field:
            y = np.log10(y)

        ax.plot(x, y, label=sim.simName)

    # finish plot
    ax.legend(loc='best')
    fig.savefig('time_evo_sh%d_%s_%d.pdf' % (subhaloID,field,fieldIndex))
    plt.close(fig)

def mstarVsMhaloEvo():
    """ Plot galaxy stellar mass vs halo mass evolution/relation, comparison between runs. 
    """
    # config simulations and field
    variants = ['SN','SNPIPE','TNG','ST']
    res = [11, 12, 14, 18, 26]
    hInd = 1242 #10677 #31619
    redshift = 3.0
    mpbBased = True # if True, use SubLink MPB for SFH, else use histogram of stellar zform

    # add all simulations which exist, skipping those which do not
    sims = []
    for variant in variants:
        for r in res:
            try:
                sim = simParams(run='structures', res=r, hInd=hInd, variant=variant, redshift=redshift)
                sims.append(sim)
                print(sim)
            except:
                pass

    # load
    z = []
    mhalo = []
    mstar = []

    mhalo_evo = []
    mstar_evo = []
    z_evo = []
    stellar_zform = []
    stellar_mass = []

    for sim in sims:
        sub = sim.subhalo(sim.zoomSubhaloID)
        halo = sim.halo(sub['SubhaloGrNr'])

        mhalo.append( halo['Group_M_Crit200'] )
        mstar.append( sub['SubhaloMassType'][sim.ptNum('stars')] )
        z.append(sim.redshift)

        # use merger tree MPB for mhalo and mstar versus redshift
        try:
            mpb = sim.loadMPB(sim.zoomSubhaloID)
            mhalo_evo.append( sim.units.codeMassToLogMsun(mpb['Group_M_Crit200']) )
            mstar_evo.append( sim.units.codeMassToLogMsun(mpb['SubhaloMassType'][:,sim.ptNum('stars')]) )
            z_evo.append( sim.snapNumToRedshift(mpb['SnapNum']) )
        except:
            print('No merger tree and/or offsets for [%s], skipping.' % sim)
            mhalo_evo.append(None)
            mstar_evo.append(None)
            z_evo.append(None)

        # load all stellar masses and formation times for alternative (non-tree) SFH
        star_zform = sim.snapshotSubset('stars', 'z_form', subhaloID=sim.zoomSubhaloID)
        star_mass = sim.snapshotSubset('stars', 'mass', subhaloID=sim.zoomSubhaloID)
        sort_inds = np.argsort(star_zform)[::-1]
        star_zform = star_zform[sort_inds]
        star_mass = sim.units.codeMassToLogMsun(np.cumsum(star_mass[sort_inds]))

        stellar_zform.append(star_zform)
        stellar_mass.append(star_mass)

    mstar = sim.units.codeMassToLogMsun(mstar)
    mhalo = sim.units.codeMassToLogMsun(mhalo)

    # load parent box relation
    tng100 = simParams(run='tng100-1', redshift=redshift)
    tng100_cen = tng100.subhalos('cen_flag')

    tng100_mstar = tng100.subhalos('mstar2_log')[tng100_cen == True]
    tng100_mhalo = tng100.subhalos('m200_log')[tng100_cen == True]
    
    xm, ym, _, pm = running_median(tng100_mhalo,tng100_mstar,binSize=0.05,percs=[5,16,50,84,95])

    # plot 1
    fig, ax = plt.subplots()

    ax.set_xlabel('Halo Mass $\\rm{M_{200,crit}}$ [ $\\rm{M_\odot}$ ]')
    ax.set_ylabel('Stellar Mass [ $\\rm{M_\odot}$ ]')
    if hInd == 10677:
        ax.set_xlim([9.4,10.2])
        ax.set_ylim([6.2,7.8])
    if hInd == 31619:
        ax.set_xlim([9.1,9.6])
        ax.set_ylim([5.5,7.5])

    # parent box relation
    pm = savgol_filter(pm,sKn,sKo,axis=1)
    ax.fill_between(xm, pm[0,:], pm[-1,:], color='#ccc', alpha=0.4)
    ax.fill_between(xm, pm[1,:], pm[-2,:], color='#ccc', alpha=0.6)
    ax.plot(xm, ym, color='#ccc', lw=lw*2, alpha=0.9, label=tng100)

    # halo from parent box
    sim = sims[0]
    halo_parentbox = sim.sP_parent.halo(hInd)
    subhalo_parentbox = sim.sP_parent.subhalo(halo_parentbox['GroupFirstSub'])

    mhalo_parentbox = sim.units.codeMassToLogMsun(halo_parentbox['Group_M_Crit200'])
    mstar_parentbox = sim.units.codeMassToLogMsun(subhalo_parentbox['SubhaloMassType'][sim.ptNum('stars')])

    label = 'h%d in %s' % (hInd,sim.sP_parent.simName)
    l, = ax.plot(mhalo_parentbox, mstar_parentbox, markers[3], color='#555', label=label)

    mpb = sim.sP_parent.loadMPB(halo_parentbox['GroupFirstSub'])
    mpb_mhalo = sim.units.codeMassToLogMsun(mpb['Group_M_Crit200'])
    mpb_mstar = sim.units.codeMassToLogMsun(mpb['SubhaloMassType'][:,sim.ptNum('stars')])

    star_zform = sim.sP_parent.snapshotSubset('stars', 'z_form', subhaloID=halo_parentbox['GroupFirstSub'])
    star_mass = sim.sP_parent.snapshotSubset('stars', 'mass', subhaloID=halo_parentbox['GroupFirstSub'])
    sort_inds = np.argsort(star_zform)[::-1]
    star_zform = star_zform[sort_inds]
    star_mass = sim.sP_parent.units.codeMassToLogMsun(np.cumsum(star_mass[sort_inds]))

    ax.plot(mpb_mhalo, mpb_mstar, '-', lw=1.5, color=l.get_color(), alpha=0.2)

    # individual zoom runs
    for i, sim in enumerate(sims):
        if sim.variant == 'TNG': marker = markers[0]
        if sim.variant == 'SN': marker = markers[1]
        if sim.variant == 'ST': marker = markers[2]

        l, = ax.plot(mhalo[i], mstar[i], marker, label=sim.simName)

        if mhalo_evo[i] is not None:
            ax.plot(mhalo_evo[i], mstar_evo[i], '-', lw=1.5, color=l.get_color(), alpha=0.5)
        #print(sim.simName, mhalo[i], mstar[i])

    # finish and save plot
    ax.legend(loc='upper left', ncols=2)   
    fig.savefig('smhm_structures_h%d_comp_%d.png' % (hInd,len(sims)))
    plt.close(fig)

    # plot 2 - vs redshift time axis
    for mpbIter in [0,1]:
        fig, ax = plt.subplots()

        ax.set_xlabel('Redshift')
        ax.set_ylabel('Stellar Mass [ $\\rm{M_\odot}$ ]')
        
        ax.set_xlim([10.0, 2.9])
        
        # galaxy from parent box
        label = 'h%d in %s' % (hInd,sim.sP_parent.simName)
        mpb_z = sim.sP_parent.snapNumToRedshift(mpb['SnapNum'])
        l, = ax.plot(sim.sP_parent.redshift, mstar_parentbox, markers[3], color='#555', label=label)

        if mpbIter:
            ax.plot(mpb_z, mpb_mstar, '-', lw=1.5, color=l.get_color(), alpha=0.7)
        else:
            w = np.where( (star_zform >= 0.0) & (star_zform < ax.get_xlim()[0]) )
            ax.plot(star_zform[w], star_mass[w], '-', lw=1.5, color=l.get_color(), alpha=0.7)

        # individual zoom runs
        for i, sim in enumerate(sims):
            if sim.variant == 'TNG': marker = markers[0]
            if sim.variant == 'SN': marker = markers[1]
            if sim.variant == 'ST': marker = markers[2]

            l, = ax.plot(z[i], mstar[i], marker, label=sim.simName)

            linestyle = ':' if sim.variant == 'SN' else '-'

            if mpbIter:
                if z_evo[i] is not None:
                    ax.plot(z_evo[i], mstar_evo[i], linestyle, lw=1.5, color=l.get_color(), alpha=0.7)
            else:
                w = np.where(stellar_zform[i] < ax.get_xlim()[0])
                ax.plot(stellar_zform[i][w], stellar_mass[i][w], linestyle, lw=1.5, color=l.get_color(), alpha=0.7)

        ax.set_ylim(np.max([4.8,ax.get_ylim()[0]]), ax.get_ylim()[1])

        # finish and save plot
        ax.legend(loc='upper left', ncols=2)   
        fig.savefig('mstarz_structures_h%d_comp_%d_mpb%d.png' % (hInd,len(sims),mpbIter))
        plt.close(fig)
