"""
Analysis and helpers specifically for zoom resimulations in cosmological volumes.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile, isdir, expanduser
from os import mkdir
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from collections import OrderedDict
from numba import jit

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
    path = "/virgotng/mpia/TNG-Cluster/"

    if res == 14 and onlyDone:
        # runs.txt file no longer relevant, use directories which exist
        from glob import glob
        dirs = glob(path + 'L680n2048TNG_h*_L%d_sf3' % res)
        halo_inds = sorted([int(folder.split('_')[-3][1:]) for folder in dirs])

        return halo_inds

    with open(path + 'runs.txt','r') as f:
        runs_txt = [line.strip() for line in f.readlines()]

    halo_inds = []
    for i, line in enumerate(runs_txt):
        if ' ' in line:
            line = line.split(' ')[0]
        if line.isdigit():
            halo_inds.append(int(line))
        if 'running:' in line and onlyDone:
            break
        if 'OLD:' in line:
            break

    if onlyDone:
        # restrict to completed runs
        halo_inds_done = [hInd for hInd in halo_inds if isdir(path+'L680n2048TNG_h%d_L%d_sf3' % (hInd,res))]
        return halo_inds_done

    return halo_inds

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
    hInd = 31619 # 10677 # 
    zoomRes = 12
    variant = 'TNG' # sf3

    zoomRun = 'structures' #'tng_zoom' #'tng50_zoom_dm'
    redshift = 3.0 # 0.0

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
    zoomRes = 14
    hInds = [50,79,84,102,107,136,155,156,179,202,205,210,217,224,239,280,282,361,390,1335,1919,3232,3693]
    variants = ['sf3'] #['sf2','sf3','sf4']
    run = 'tng_zoom'

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
            sPz = simParams(res=zoomRes, run=run, hInd=hInd, redshift=0.0, variant=variant)

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
            print(hInd,variant,min_dist_lr)

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
    """ Make a visual comparison (density projection images) between halos in the parent box and their zoom realizations.
    
    Args:
      haloID (int): the zoom halo ID, at the final redshift (z=0).
      variant (str): the zoom variant.
      conf (int): the plotting configuration.
      snap (int): if not the final snapshot, plot at some redshift other than z=0.
    """

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

    # load MPB of this halo
    haloMPB = sPz.sP_parent.loadMPB( sPz.sP_parent.groupCatSingle(haloID=haloID)['GroupFirstSub'] )
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
    """ Compute spatial volume fractions of zoom runs via discrete convex hull type approach. """
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
    from ..tracer.tracerMC import globalTracerChildren, globalTracerLength, match3

    outPath = '/u/dnelson/sims.TNG/L680n8192TNG/output/'
    parent_sim = simParams('tng-cluster')

    # zoom config
    res = 14
    variant = 'sf3'
    run = 'tng_zoom'

    hInds = _halo_ids_run(res=res, onlyDone=True)

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

    # --- tracers ---
    if snap == 99:
        # tracers: is final snapshot? then decide tracer ordering and save for use on all snapshots
        GroupLenTypeTracers = np.zeros(len(hInds), dtype='int32')

        for i, hInd in enumerate(hInds):
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
            print('[%4d] z=0 tracers.' % hInd)

            # cache
            saveFilename = outPath + 'tracers_%d.hdf5' % hInd

            if isfile(saveFilename):
                with h5py.File(saveFilename,'r') as f:
                    for key in f['TracerLength_Halo']:
                        GroupLenTypeTracers[i] += np.sum(f['TracerLength_Halo'][key][()])

                print(' skip')
                continue

            # get child tracers of all particle types in all FoFs
            # (ordered first by parent type: gas->stars->BHs, then by halo/subhalo membership)
            trIDs = globalTracerChildren(sP, halos=True)

            # TracerLength and TracerOffset by halo and subhalo
            trCounts_halo, trOffsets_halo = globalTracerLength(sP, halos=True)

            trCounts_sub, trOffsets_sub = globalTracerLength(sP, subhalos=True, haloTracerOffsets=trOffsets_halo)

            # load all TracerIDs, get those not in FoFs
            TracerID = sP.snapshotSubsetP('tracer', 'TracerID')

            trInds,_ = match3(TracerID,trIDs)
            assert trInds.size == trIDs.size

            mask = np.zeros(TracerID.size, dtype='int8')
            mask[trInds] = 1

            trInds_outside_halos = np.where(mask == 0)[0]

            trIDs_outside_halos = TracerID[trInds_outside_halos]

            # make final, z=0 ordered, list of tracerIDs
            trInds_final = np.hstack((trInds,trInds_outside_halos))

            if 1:
                # debug verify
                mask2 = np.zeros(TracerID.size, dtype='int16')
                mask2[trInds_final] += 1
                assert mask2.min() == 1
                assert mask2.max() == 1

            TracerID = TracerID[trInds_final]

            # save            
            with h5py.File(saveFilename,'w') as f:
                # lengths and offsets
                for key in trCounts_halo.keys():
                    f['TracerLength_Halo/%s' % key] = trCounts_halo[key]
                    f['TracerOffset_Halo/%s' % key] = trOffsets_halo[key]

                    f['TracerLength_Subhalo/%s' % key] = trCounts_sub[key]
                    f['TracerOffset_Subhalo/%s' % key] = trOffsets_sub[key]

                # z=0 ordered TracerIDs
                f['TracerID'] = TracerID

            # save total length of tracers in FoF halos, for each hInd (sum over all parent types)
            for key in trCounts_halo.keys():
                GroupLenTypeTracers[i] += np.sum(trCounts_halo[key])

        with h5py.File(outPath + 'tracers_halolengths.hdf5','w') as f:
            f['GroupLenTypeTracers'] = GroupLenTypeTracers

    with h5py.File(outPath + 'tracers_halolengths.hdf5','r') as f:
        GroupLenTypeTracers = f['GroupLenTypeTracers'][()]

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
    offsets['Tracers'] = np.hstack( (0,np.cumsum(GroupLenTypeTracers)[:-1]) )

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
            w = np.where(data['Group']['GroupFirstSub'] != -1)
            data['Group']['GroupFirstSub'][w] += offsets['Subhalo'][hCount]

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

            # tracers: add Tracer{Length,Offset}Type at z=0
            if snap == 99:
                data['Group']['TracerLengthType'] = np.zeros((lengths['Group'][hCount],6), dtype='int32')
                data['Group']['TracerOffsetType'] = np.zeros((lengths['Group'][hCount],6), dtype='int32')

                with h5py.File(outPath + 'tracers_%d.hdf5' % hInd,'r') as f:
                    for key in f['TracerLength_Halo'].keys():
                        data['Group']['TracerLengthType'][:,sP.ptNum(key)] = f['TracerLength_Halo'][key][()]
                        data['Group']['TracerOffsetType'][:,sP.ptNum(key)] = f['TracerOffset_Halo'][key][()] + \
                                                                             offsets['Tracers'][hCount]

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

            # tracers: add Tracer{Length,Offset}Type at z=0
            if snap == 99:
                data['Subhalo']['TracerLengthType'] = np.zeros((lengths['Subhalo'][hCount],6), dtype='int32')
                data['Subhalo']['TracerOffsetType'] = np.zeros((lengths['Subhalo'][hCount],6), dtype='int32')

                with h5py.File(outPath + 'tracers_%d.hdf5' % hInd,'r') as f:
                    for key in f['TracerLength_Subhalo'].keys():
                        data['Subhalo']['TracerLengthType'][:,sP.ptNum(key)] = f['TracerLength_Subhalo'][key][()]
                        data['Subhalo']['TracerOffsetType'][:,sP.ptNum(key)] = f['TracerOffset_Subhalo'][key][()] + \
                                                                             offsets['Tracers'][hCount]

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

    # load total number of tracers in halos and overwrite GroupLenType[3]
    assert np.sum(GroupLenType_hInd[:,3]) == 0
    assert GroupLenTypeTracers.size == GroupLenType_hInd.shape[0]

    GroupLenType_hInd[:,3] = GroupLenTypeTracers

    # determine sizes/split between two files per halo
    OuterFuzzLenType_hInd = NumPart_Total - GroupLenType_hInd

    # quick save of offsets
    saveFilename = 'lengths_hind_%03d.hdf5' % snap
    with h5py.File(outPath + saveFilename,'w') as f:
        # particle lengths in all fofs for this hInd (file 1)
        f['GroupLenType_hInd'] = GroupLenType_hInd
        # particle lengths outside fofs for this hInd (file 2)
        f['OuterFuzzLenType_hInd'] = OuterFuzzLenType_hInd
        # halo IDs of original zooms
        f['HaloIDs'] = np.array(hInds, dtype='int32')

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

                        if gName == 'PartType2' and field in ['SubfindDMDensity','SubfindDensity','SubfindVelDisp']:
                            continue # all zero for low-res DM, do not save

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

            # reorder tracers into z=0 order
            if gName == 'PartType3':
                with h5py.File(outPath + 'tracers_%d.hdf5' % hInd,'r') as f:
                    # z=0 ordered TracerIDs
                    TracerID_z0 = f['TracerID'][()]

                inds_snap, inds_z0 = match3(data[gName]['TracerID'], TracerID_z0)
                assert data[gName]['TracerID'].size == TracerID_z0.size
                assert inds_z0.size == TracerID_z0.size

                for trField in data[gName].keys():
                    data[gName][trField] = data[gName][trField][inds_snap]

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
                headers['Header']['NumPart_ThisFile'] = np.int32(GroupLenType_hInd[hCount,:])
                start = np.zeros( 6, dtype='int32' )
            else:
                headers['Header']['NumPart_ThisFile'] = np.int32(OuterFuzzLenType_hInd[hCount,:])
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


    # compute offsets and insert them (e.g. new/MTNG convention)
    parent_sim.snap = snap # avoid setSnap() since our snap<->redshift mapping file is incomplete
    
    offsets = parent_sim.groupCatOffsetListIntoSnap()

    w_offset_halos = 0
    w_offset_subs = 0

    for hCount, hInd in enumerate(hInds):
        outFile  = outPath + "groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap,snap,hCount)

        with h5py.File(outFile, 'r+') as f:
            Nsubhalos = f['Header'].attrs['Nsubgroups_ThisFile']
            Nhalos = f['Header'].attrs['Ngroups_ThisFile']

            f['Group']['GroupOffsetType'] = offsets['snapOffsetsGroup'][w_offset_halos:w_offset_halos+Nhalos]
            f['Subhalo']['SubhaloOffsetType'] = offsets['snapOffsetsSubhalo'][w_offset_subs:w_offset_subs+Nsubhalos]

            w_offset_halos += Nhalos
            w_offset_subs += Nsubhalos

    print('Done.')

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
    m200_sP2 = sP2.units.codeMassToLogMsun(sP2.halos('Group_M_Crit200'))

    haloIDs_1 = np.where( (contam < 0.01) & (m200 > 14.0) )[0]
    haloIDs_2 = np.where( (m200_sP2 > 14.0) )[0]

    haloIDs = [haloIDs_1, haloIDs_2]

    # subhalos: all of these groups
    subIDs = []

    for i, sP in enumerate([sP1, sP2]):
        nSubs = sP.halos('GroupNsubs')[haloIDs[i]]
        firstSub = sP.halos('GroupFirstSub')[haloIDs[i]]

        subIDs_loc = np.hstack( [np.arange(nSubs[i]) + firstSub[i] for i in range(haloIDs[i].size)] )
        subIDs.append(subIDs_loc)

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
            for i, sP in enumerate([sP1,sP2]):
                if gName == 'Group':
                    vals = sP.halos(field)
                    vals = vals[haloIDs[i]]

                if gName == 'Subhalo':
                    vals = sP.subhalos(field)
                    vals = vals[subIDs[i]]

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel() # 1D for all multi-D

                if field not in ['GroupCM','GroupPos','SubhaloCM','SubhaloGrNr','SubhaloIDMostbound']:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, density=True, label=sP.simName)

            # finish plot
            ax.legend(loc='best')
            pdf.savefig()
            plt.close(fig)

        # finish
        pdf.close()

def testVirtualParentBoxSnapshot(snap=99):
    """ Compare all snapshot fields (1d histograms) vs TNG300-1 to check unit conversions, etc. """
    from matplotlib.backends.backend_pdf import PdfPages
    from ..util.helper import closest

    # config
    haloID = 0 # for particle comparison, indexing primary targets of TNG-Cluster
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

def check_groupcat_property():
    """ Compare TNG300 vs TNG-Cluster property. """
    xprop = 'Group_M_Crit200'
    yprop = 'SubhaloBHMass' #'GroupMassType', 'GroupSFR', 'GroupNsubs', 'GroupWindMass', 'GroupBHMass'
    ypropind = None

    snap = 99

    halo_inds = _halo_ids_run(onlyDone=True)
    halo_inds.pop(-1) # remove h4

    # load
    cache = 'cache_%s_%s.hdf5' % (yprop,ypropind)
    if isfile(cache):
        with h5py.File(cache,'r') as f:
            x = f['x'][()]
            y = f['y'][()]

    else:
        # compute now
        res = 13
        variant = 'sf3'
        run = 'tng_zoom'
        haloID = 0 # always use first fof

        # allocate
        x = np.zeros( len(halo_inds), dtype='float32' )
        y = np.zeros( len(halo_inds), dtype='float32')

        # loop over halos
        for i, hInd in enumerate(halo_inds):
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
            halo = sP.halo(haloID)
            subh = sP.subhalo(halo['GroupFirstSub'])

            x[i] = halo[xprop]
            if 'Group' in yprop:
                y[i] = halo[yprop][ypropind] if ypropind is not None else halo[yprop]
            else:
                y[i] = subh[yprop][ypropind] if ypropind is not None else subh[yprop]

            print(i,hInd)

        with h5py.File(cache,'w') as f:
            f['x'] = x
            f['y'] = y
        print('Saved: [%s]' % cache)

    # TNG-Cluster unit conversions
    sP = simParams(run='tng-cluster')
    x = sP.units.codeMassToLogMsun(x)
    if 'Mass' in yprop:
        y = sP.units.codeMassToLogMsun(y)
    else:
        y = np.log10(y)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xprop)
    ax.set_ylabel(yprop + (' ['+str(ypropind)+']' if ypropind is not None else ''))

    ax.set_xlim([14.0,15.5])

    ax.scatter(x, y, marker='o', label='TNG-Cluster')

    for run in ['tng300-1','tng300-2','tng300-3']:
        # load TNG300-x
        sP = simParams(run=run, snap=snap)

        x2 = sP.halos(xprop)

        if 'Group' in yprop:
            y2 = sP.halos(yprop)[:,ypropind] if ypropind is not None else sP.halos(yprop)
        else:
            GroupFirstSub = sP.halos('GroupFirstSub')
            #y2 = np.zeros( sP.numHalos, dtype='float32' )
            #y2.fill(np.nan)
            y2 = sP.subhalos(yprop)[GroupFirstSub,ypropind] if ypropind is not None else sP.subhalos(yprop)[GroupFirstSub]

        x2 = sP.units.codeMassToLogMsun(x2)
        if 'Mass' in yprop:
            y2 = sP.units.codeMassToLogMsun(y2)
        else:
            y2 = np.log10(y2)

        w = np.where(x2 >= 14.0)
        x2 = x2[w]
        y2 = y2[w]

        # plot
        ax.scatter(x2, y2, marker='s', label=sP.simName)

    ax.legend(loc='best')
    fig.savefig('check_%s_%s_%s.pdf' % (xprop,yprop,ypropind))
    plt.close(fig)

def check_particle_property():
    snap = 99
    pt = 'bh'
    prop = 'BH_Mass'

    # load zoom
    hInd = 0 
    haloID = 0  # always use first fof

    sP = simParams(run='tng_zoom', res=13, snap=snap, hInd=hInd, variant='sf3')

    x = sP.snapshotSubset(pt, prop, haloID=haloID)
    x = sP.units.codeMassToLogMsun(x)

    # load box
    sP = simParams(run='tng300-1', snap=snap)

    y = sP.snapshotSubset(pt, prop, haloID=haloID)
    y = sP.units.codeMassToLogMsun(y)

    print( 10.0**y.mean() / 10.0**x.mean() )
    print( 10.0**x.mean() / 10.0**y.mean() )

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('%s %s' % (pt,prop))
    ax.set_ylabel('PDF')

    ax.hist(x, bins=40, label='TNG-Cluster')
    ax.hist(y, bins=40, label='TNG300')

    ax.legend(loc='best')
    fig.savefig('check_%s_%s.pdf' % (pt,prop))
    plt.close(fig)

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
