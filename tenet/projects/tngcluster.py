"""
TNG-Cluster: introduction paper.
(in prep)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from os.path import isfile

from ..cosmo.zooms import contamination_mindist
from ..plot.cosmoGeneral import quantMedianVsSecondQuant
from ..plot.cosmoMisc import simClustersComparison
from ..plot.config import *
from ..util.helper import logZeroNaN, running_median
from ..projects.azimuthalAngleCGM import _get_dist_theta_grid
from ..tracer.tracerMC import match3

def vis_fullbox_virtual(sP, conf=0):
    """ Visualize the entire virtual reconstructed box. """
    from ..vis.box import renderBox

    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    nPixels    = 2000

    # halo plotting
    plotHalos  = False

    if conf in [0,1,2,3,4,5]:
        pri = sP.groups('GroupPrimaryZoomTarget')
        plotHaloIDs = np.where(pri == 1)[0]

    # panel config
    if conf == 0:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.5,7.1]}]

    if conf == 1:
        method = 'sphMap' # is global, overlapping coarse cells
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[9.1,9.6]}]

    if conf == 2:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]

    if conf == 3:
        method = 'sphMap' # is global
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]

    if conf in [4,5]:
        method = 'sphMap' # is global
        
        if conf == 4:
            panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.8,7.2]}]
            numBufferLevels = 3 # 2 or 3, free parameter
        if conf == 5:
            panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.2,7.6]}]
            numBufferLevels = 4
            
        maxGasCellMass = sP.targetGasMass
        if numBufferLevels >= 1:
            # first buffer level is 27x mass (TODO CHECK STILL?), then 8x mass for each subsequent level
            #maxGasCellMass *= 27 * np.power(8,numBufferLevels-1)
            # TEST:
            maxGasCellMass *= np.power(8,numBufferLevels)
            # add padding for x2 Gaussian distribution
            maxGasCellMass *= 3

        ptRestrictions = {'Masses':['lt',maxGasCellMass]}

    if conf == 6:
        from ..util.simParams import simParams
        sP = simParams(run='tng_dm',res=2048,redshift=0.0) # parent box
        method = 'sphMap'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[7.0,8.4]}]

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = [1400,1400]
        colorbars  = True
        fontsize   = 22

        saveFilename = './boxImage_%s_%s-%s_%s_conf%d.pdf' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],sP.snap,conf)

    renderBox(panels, plotConfig, locals(), skipExisting=False)

def vis_gallery(sP, conf=0, num=20):
    """ Visualize the entire virtual reconstructed box. """
    from ..vis.halo import renderSingleHalo

    rVirFracs  = [1.0]
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = 'physical' if conf == 1 and num == 1 else False
    labelHalo  = 'mhalo,id' if conf == 1 and num == 1 else False
    rotation   = None # random
    nPixels    = 600
    size       = 3.0
    sizeType   = 'rVirial'

    if num == 1:
        # for single halo showcase image
        nPixels = [1920, 1080] if conf == 1 else [960,540]
        size = 4.0

    method = 'sphMap_globalZoomOrig' # all particles of original zoom run only

    # panel
    partType = 'gas'
    valMinMax = None # auto

    if conf == 0:
        partField = 'coldens_msunkpc2'
        valMinMax = [5.8, 7.8]
    if conf == 1:
        # main panel
        partField = 'xray_lum_0.5-2.0kev' #'xray'
        valMinMax = [34.0, 39.5]
    if conf == 2:
        partType = 'dm'
        partField = 'coldens_msunkpc2'
        valMinMax = [5.5, 9.3]
    if conf == 3:
        partType = 'stars'
        partField = 'coldens_msunkpc2'
        valMinMax = [4.0, 7.0]
    if conf == 4:
        partField = 'sz_yparam'
    if conf == 5:
        partField = 'bmag_uG'
    if conf == 6:
        partField = 'Z_solar'
    if conf == 7:
        partField = 'HI_segmented'
        valMinMax = [12.0, 21.0]
        plotSubhalos = 100
    if conf == 8:
        partField = 'temp'
        valMinMax = [7.0, 7.8]
    if conf == 9:
        partField = 'vrad'
        valMinMax = [-1200,1200]
    if conf == 10:
        partField = 'velsigma_los'

    # targets
    pri_target = sP.groups('GroupPrimaryZoomTarget')
    subIDs = sP.groups('GroupFirstSub')[np.where(pri_target == 1)]

    if num == 1:
        subIDs = [subIDs[2]]
    else:
        subIDs = subIDs[0:num]

    # panels
    panels = []

    for subID in subIDs:
        panels.append( {'subhaloInd':subID} )

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = nPixels
        colorbars  = True if num == 1 else False
        fontsize   = 24 if conf == 1 else 26
        nCols      = int(np.floor(np.sqrt(num)))
        nRows      = int(np.ceil(num/nCols))

        saveFilename = './gallery_%s_%d_%s-%s_n%d.pdf' % (sP.simName,sP.snap,partType,partField,num)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def mass_function(secondaries=False):
    """ Plot halo mass function from the parent box (TNG300) and the zoom sample. 
    If secondaries == True, then also include non-targeted halos with no/little contamination. """
    from ..util import simParams
    from ..cosmo.zooms import _halo_ids_run
    
    mass_range = [14.0, 15.5] #if not secondaries else [13.0, 15.5]
    binSize = 0.1
    redshift = 0.0
    
    sP_tng300 = simParams(res=2500,run='tng',redshift=redshift)
    sP_tngc = simParams(res=2048, run='tng_dm', redshift=redshift)

    # load halos
    halo_inds = _halo_ids_run(onlyDone=True)
    #print(len(halo_inds), halo_inds)

    # start figure
    fig = plt.figure(figsize=figsize)

    nBins = int((mass_range[1]-mass_range[0])/binSize)

    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(mass_range)
    ax.set_xticks(np.arange(mass_range[0],mass_range[1],0.1))
    ax.set_xlabel('Halo Mass M$_{\\rm 200c}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('Number of Halos [%.1f dex$^{-1}$]' % binSize)
    ax.set_yscale('log')
    ax.yaxis.set_ticks_position('both')

    yy_max = 1.0

    hh = []
    labels = []

    for sP in [sP_tng300,sP_tngc]:
        if sP == sP_tng300:
            # tng300
            gc = sP_tng300.halos('Group_M_Crit200')
            masses = sP_tng300.units.codeMassToLogMsun(gc)
            label = 'TNG300-1'
        elif sP == sP_tngc:
            # tng-cluster
            gc = sP_tngc.halos('Group_M_Crit200')
            masses = sP_tngc.units.codeMassToLogMsun(gc[halo_inds])
            label = 'TNG-Cluster'

        w = np.where(~np.isnan(masses))
        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
        yy_max = np.nanmax([yy_max,np.nanmax(yy)])

        hh.append(masses[w])
        labels.append(label)

    # 'bonus': halos above 14.0 in the high-res regions of more massive zoom targets
    if secondaries:
        sP = simParams('tng-cluster', redshift=redshift)
        masses = sP.units.codeMassToLogMsun(sP.halos('Group_M_Crit200'))

        # zero contamination
        contam_frac = sP.halos('GroupContaminationFracByMass')
        w = np.where((masses > ax.get_xlim()[0]) & (contam_frac == 0))

        w = np.array(list(set(w[0]) - set(halo_inds))) # exclude targeted halos

        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
        yy_max = np.nanmax([yy_max,np.nanmax(yy)])

        hh.append(masses[w])
        labels.append('TNG-Cluster Bonus (no contamination)')

        # small contamination
        contam_thresh = 1e-2
        w = np.where((masses > ax.get_xlim()[0]) & (contam_frac < contam_thresh) & (contam_frac != 0))

        w = np.array(list(set(w[0]) - set(halo_inds))) # exclude targeted halos

        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
        yy_max = np.nanmax([yy_max,np.nanmax(yy)])

        hh.append(masses[w])
        labels.append('TNG-Cluster Bonus ($f_{\\rm contam} < %.1e$)' % contam_thresh)

    # plot
    ax.hist(hh,bins=nBins,range=mass_range,label=labels,histtype='bar',alpha=0.9,stacked=True)

    ax.set_ylim([0.8,100])
    ax.legend(loc='upper right' if not secondaries else 'lower left')

    fig.savefig('mass_functions.pdf')
    plt.close(fig)

    # plot histogram of contamination fraction
    if secondaries:
        bad_val = -6.0
        w = np.where(contam_frac == 0)
        contam_frac = logZeroNaN(contam_frac)
        contam_frac[w] = bad_val

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(contam_frac, bins=100, range=[-7.0, 0.0])
        ax.text(bad_val+0.1, 1e7, 'Zero', ha='left', va='bottom')
        ax.set_xlabel('Contamination Fraction [by mass]')
        ax.set_ylabel('Number of Halos')
        ax.set_yscale('log')
        ax.set_xlim([bad_val-0.2, 0.0])

        fig.savefig('contamination_fraction.pdf')
        plt.close(fig)

def sample_halomasses_vs_redshift(sPs):
    """ Compare simulation vs observed cluster samples as a function of (redshift,mass). """
    from ..load.data import rossetti17planck, pintoscastro19, hilton21act, adami18xxl
    from ..load.data import bleem20spt, piffaretti11rosat

    zrange = [0.0, 0.8]
    alpha = 1.0 # for data
    msize = 30 # scatter() default is 20

    # start plot
    fig = plt.figure(figsize=(figsize[0]*1.2, figsize[1]*1.2))
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1) # elements below z=1 are rasterized

    ax.set_xlabel('Redshift')
    ax.set_ylabel('Halo Mass M$_{\\rm 500c}$ [log M$_{\\rm sun}$]')

    ax.set_xlim(zrange)
    ax.set_ylim([13.97, 15.5])

    rng = np.random.default_rng()

    # load simulations and plot
    for i, sP in enumerate(sPs):
        # cache file
        cache_file = sP.derivPath + 'cache/clusters_m500_evo.hdf5'

        if isfile(cache_file):
            print('Loading [%s]' % cache_file)
            with h5py.File(cache_file,'r') as f:
                z = f['z'][()]
                m500 = f['m500'][()]
                subid = f['subid'][()]
        else:
            # TNG300: all halos with M200c > 14.0, TNG-Cluster: all primary zoom targets
            subhaloIDs = sP.cenSatSubhaloIndices(cenSatSelect='cen')
            m200c = sP.subhalos('mhalo_200_log')[subhaloIDs]

            if sP.simName != 'TNG-Cluster':
                # include low-mass progenitors at high redshift
                w = np.where(m200c > 14.0)[0]
                subhaloIDs = subhaloIDs[w]

            # allocate
            snaps = sP.validSnapList()
            nsnaps = len(snaps)

            z = np.zeros(nsnaps, dtype='float32')
            subid = np.zeros((subhaloIDs.size,nsnaps), dtype='int32')
            m500 = np.zeros((subhaloIDs.size,nsnaps), dtype='float32')

            subid.fill(-1)
            m500.fill(np.nan)

            z = sP.snapNumToRedshift(snaps)

            # loop over each cluster
            for j in range(len(subhaloIDs)):
                # load the MPB
                mpb = sP.loadMPB(subhaloIDs[j], fields=['SubfindID','SnapNum'])

                # match to master snapshot list
                inds, _ = match3(snaps, mpb['SnapNum'])
                subid[j,inds] = mpb['SubfindID']

            # loop over each snapshot
            for j, snap in enumerate(snaps):
                # load m500 values
                sP.setSnap(snap)
                m500_loc = sP.subhalos('m500_log') # log msun

                # index to our subset of interest and stamp
                w = np.where(subid[:,j] >= 0)[0]
                m500[w,j] = m500_loc[subid[w,j]]

            # save
            with h5py.File(cache_file,'w') as f:
                f['z'] = z
                f['m500'] = m500
                f['subid'] = subid

        # draw evolution lines
        color = next(ax._get_lines.prop_cycler)['color']

        for j in range(subid.shape[0]):
            label = sP.simName if j == 0 else ''

            # interpolate to fill any nan values for display
            m500_loc = m500[j,:]

            w_nan = np.where(np.isnan(m500_loc))[0]
            if len(w_nan) > 0:
                w_finite = np.where(~np.isnan(m500_loc))[0]
                new_m500_values = np.interp(z[w_nan], z[w_finite][::-1], m500_loc[w_finite][::-1])
                m500_loc[w_nan] = new_m500_values

            # decide an alpha which decreases with decreasing z=0 mass
            if sP.simName == 'TNG-Cluster':
                alpha_loc = 0.2 + 1.0*(m500_loc[-1] - 14.2)
                zorder = rng.integers(-500, high=-2, size=1) #-2
            else:
                alpha_loc = 0.2 + 0.5*(m500_loc[-1] - 13.4)
                zorder = rng.integers(-500, high=-2, size=1) #-3
            alpha_loc = np.clip(alpha_loc, 0.2, 0.9)

            ax.plot(z, m500_loc, lw=lw, c=color, alpha=alpha_loc, zorder=zorder, label=label)
            #ax.plot(z[-1], m500_loc[-1], 'o', c=color, alpha=alpha_loc, zorder=zorder, ms=4)

    # first legend
    legend1 = ax.legend(loc='upper left')
    ax.add_artist(legend1)

    # plot obs samples
    r17 = rossetti17planck()
    pc19 = pintoscastro19(sPs[0])
    h21 = hilton21act()
    a18 = adami18xxl()
    b20 = bleem20spt(sPs[0])
    p11 = piffaretti11rosat()

    d1 = ax.scatter(r17['z'], r17['m500'], s=msize+8, c='#000000', marker='s', alpha=alpha, label=r17['label'], zorder=0)
    d2 = ax.scatter(pc19['z'], pc19['m500'], s=msize+8, c='#222222', marker='*', alpha=alpha, label=pc19['label'], zorder=0)
    d3 = ax.scatter(h21['z'], h21['m500'], s=msize-9, c='#222222', marker='p', alpha=alpha-0.1, label=h21['label'], zorder=0)
    d4 = ax.scatter(a18['z'], a18['m500'], s=msize+8, c='#222222', marker='D', alpha=alpha, label=a18['label'], zorder=0)
    d5 = ax.scatter(b20['z'], b20['m500'], s=msize+8, c='#222222', marker='X', alpha=alpha, label=b20['label'], zorder=0)
    d6 = ax.scatter(p11['z'], p11['m500'], s=msize-4, c='#222222', marker='h', alpha=alpha-0.1, label=p11['label'], zorder=0)

    # add first legend
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles[len(sPs):], labels[len(sPs):], loc='upper right', frameon=True)
    legend2.get_frame().set_edgecolor('#bbbbbb')
    legend2.get_frame().set_linewidth(1.0)
    ax.add_artist(legend2)

    # plot coma cluster
    def _plot_single_cluster(m500_msun, m500_err_up, m500_err_down, redshift, name):
        """ Helper. Input in linear msun. """
        m500 = np.log10([m500_msun, m500_err_up, m500_err_down])

        error_lower = m500[0] - m500[2]
        error_upper = m500[1] - m500[0]
        yerr = np.reshape( [error_lower, error_upper], (2,1) )

        color = '#ffffff' if name not in ['Bullet','El Gordo'] else '#000000'
        zoff = 0.01 if name != 'El Gordo' else -0.01
        ha = 'left' if name != 'El Gordo' else 'right'
        d4 = ax.errorbar(redshift, m500[0], yerr=yerr, color=color, marker='H')
        t = ax.text(redshift+zoff, m500[0], name, fontsize=14, va='center', ha=ha, color='#ffffff')
        t.set_bbox(dict(facecolor='#000000', alpha=0.2 if color == '#ffffff' else 0.4, linewidth=0))

    # plot coma cluster (Okabe+2014 Table 8, g+ profile)
    coma_z = 0.0231
    coma_m500 = np.array([3.89, 3.89+1.04, 3.89-0.76]) * 1e14 / sP.HubbleParam
    _plot_single_cluster(coma_m500[0], coma_m500[1], coma_m500[2], coma_z, 'Coma')

    # plot pheonix cluster
    pheonix_z = 0.597 # currently off edge of plot
    pheonix_m500 = 2.34e15 # msun, Tozzi+15 (Section 3)
    m500_err = 0.71e15 # msun
    _plot_single_cluster(pheonix_m500, pheonix_m500+m500_err, pheonix_m500-m500_err, pheonix_z, 'Pheonix')

    # plot el gordo cluster
    elgordo_z = 0.795 # true z = 0.870, moved for visibility
    elgordo_m500 = 8.8e14 # msun (Botteon+16)
    elgordo_m500_err = 1.2e14 # msun
    _plot_single_cluster(elgordo_m500, elgordo_m500+elgordo_m500_err, elgordo_m500-elgordo_m500_err, elgordo_z, 'El Gordo')

    # plot bullet cluster
    bullet_z = 0.296
    bullet_m500 = 1.1e15 # msun, Clowe+2006
    m500_err = 0.2e15 # msun
    _plot_single_cluster(bullet_m500, bullet_m500+m500_err, bullet_m500-m500_err, bullet_z, 'Bullet')

    # plot perseus cluster
    perseus_z = 0.0183
    perseus_m500 = sP.units.m200_to_m500(6.65e14) # Simionescu+2011
    perseus_m500_errup = sP.units.m200_to_m500(6.65e14 + 0.43e14)
    perseus_m500_errdown = sP.units.m200_to_m500(6.65e14 - 0.46e14)
    _plot_single_cluster(perseus_m500, perseus_m500_errup, perseus_m500_errdown, perseus_z, 'Perseus')

    # plot virgo cluster (note: fornax m500<1e14)
    virgo_z = 0.01
    virgo_m500 = 1.01e14 # 0.8e14 msun is true value (Simionescu+2017), moved up for visibility
    virgo_m500_err = 0.05e14 # msun
    _plot_single_cluster(virgo_m500, virgo_m500+virgo_m500_err, virgo_m500-virgo_m500_err, virgo_z, 'Virgo')

    # plot eROSITA completeness goal
    erosita_minhalo = [0.20,0.32,0.47,0.65,0.86,1.12,1.44,1.87,2.33,2.91,3.46,4.19,4.86,5.80,6.68,7.33,7.79]
    erosita_z       = [0.05,0.08,0.11,0.14,0.17,0.21,0.25,0.32,0.38,0.47,0.56,0.69,0.82,1.03,1.30,1.60,1.92]

    erosita_minhalo = np.log10(sP.units.m200_to_m500(np.array(erosita_minhalo) * 1e14)) # log msun

    l, = ax.plot(erosita_z, erosita_minhalo, '-', lw=lw, alpha=alpha, color='#ffffff')
    #ax.arrow(erosita_z[6], erosita_minhalo[6]+0.02, 0.0, 0.1, lw=lw, head_length=0.008, color=l.get_color())
    t = ax.text(erosita_z[8]-0.04, 14.22, 'eROSITA All-Sky Complete', color=l.get_color(), fontsize=14, rotation=21)
    t.set_bbox(dict(facecolor='#000000', alpha=0.3, linewidth=0))

    fig.savefig('sample_halomass_vs_redshift.pdf')
    plt.close(fig)

def bfield_strength_vs_halomass(sPs, redshifts):
    """ Driver for quantMedianVsSecondQuant. """
    sPs_in = []
    for redshift in redshifts:
        for sP in sPs:
            sPloc = sP.copy()
            sPloc.setRedshift(redshift)
            sPs_in.append( sPloc )

    xQuant = 'mhalo_200_log'
    yQuant = 'bmag_halfr500_volwt'
    scatterColor = 'redshift'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.4]
    ylim = [-0.65, 0.85]
    clim = [0.0, 2.0]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    def _draw_data(ax):
        """ Draw data constraints on figure. """

        # Di Gennaro+2020 (https://arxiv.org/abs/2011.01628)
        #  -- measurements based on integrated flux within <~ 0.5*r500
        bmin = np.log10(1.0) # uG
        bmax = np.log10(3.0) # uG
        mass_range = sPs[0].units.m500_to_m200(np.array([5e14, 9e14])) # msun

        ax.fill_between(np.log10(mass_range), y1=[bmin,bmin], y2=[bmax,bmax], edgecolor='#cccccc', 
            facecolor='#cccccc', alpha=0.7,lw=lw, label='Di Gennaro+20 ($z \sim 0.8$)', zorder=-1)

        # Boehringer+2016 (https://arxiv.org/abs/1610.02887)
        # -- about ~90 measurements have mean r/r500 = 0.32, median r/r500 = 0.25
        bmin = np.log10(2.0) # uG
        bmax = np.log10(6.0) # uG
        mass_range = [2e14,4e14] # m200 msun

        ax.fill_between(np.log10(mass_range), y1=[bmin,bmin], y2=[bmax,bmax], edgecolor='#eeeeee', 
            facecolor='#eeeeee', lw=lw, label='B$\\rm\\"{o}$hringer+16 ($z \sim 0.1$)', zorder=-1)

    def _draw_data2(ax):
        """ Draw additional data constraints on figure, individual halos. """
        b = np.log10(2.0) # uG, Bonafede+10 https://arxiv.org/abs/1002.0594
        yerr = np.reshape( [0.34, 0.34], (2,1) ) # 1-4.5 uG (center vs 1 Mpc), Bonafede+10
        m200 = 14.88 # Okabe+14 m500->m200
        xerr = np.reshape( [0.1, 0.1], (2,1) ) # Okabe+14

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color='#000000', marker='D', label='Bonafede+10 (Coma)')

        b = np.log10( (1.5+0.3)/2 ) # average of <B0>=1.5 uG and ~0.3 uG (volume average within 1 Mpc, ~1r500)
        yerr = np.reshape( [0.47, 0.22], (2,1) )
        m200 = np.log10(1e14 / 0.6774) # Govoni+17 Sec 4.2
        xerr = np.reshape( [0.1, 0.1], (2,1) ) # assumed, e.g. minimum of ~30% uncertainty

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color='#000000', marker='H', label='Govoni+17 (Abell 194)')

        # TODO: Stuardi, C.+2021, Abell 2345
        # |B| = 2.8 +/- 0.1 uG (within 200 kpc)
        # M_500,SZ = 5.91e14 Msun

        # TODO: Mernier+2022 https://arxiv.org/abs/2207.10092
        # |B| = 1.9 +/- 0.3 uG (volume-averaged) (z=0.1) (aperture? mass?)

        # second legend
        handles = [ plt.Line2D( (0,1), (0,0), color='black', lw=0, marker=['o','s'][i]) for i in range(len(sPs)) ]
        legend2 = ax.legend(handles, [sP.simName for sP in sPs], borderpad=0.4, loc='upper right')
        ax.add_artist(legend2)

    quantMedianVsSecondQuant(sPs_in, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, f_post=_draw_data2, legendLoc='lower right', labelSims=False, pdf=None)

def stellar_mass_vs_halomass(sPs, conf=0):
    """ Plot various stellar mass quantities vs halo mass. """
    from ..load.data import behrooziSMHM, mosterSMHM, kravtsovSMHM

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [13.8, 15.4]
    clim = [-0.4, 0.0] # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    if conf == 0:
        yQuant = 'mstar_30pkpc'
        ylabel = 'BCG Stellar Mass [ log M$_{\\rm sun}$ ]'
        ylim = [10.9, 12.8]
        scatterColor = None #'massfrac_exsitu2' (TODO: can re-enable)

        def _draw_data(ax):
            # Kravtsov+ 2018 (Table 1 for M500crit + Table 4)
            label = 'Kravtsov+18'
            
            m500c = np.log10(np.array([15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]) * 1e14)
            mstar_30pkpc = np.log10(np.array([5.18, 10.44, 7.12, 3.85, 3.67, 4.35, 4.71, 4.59, 6.76]) * 1e11)

            ax.scatter(m500c, mstar_30pkpc, s=markersize+20, c='#000000', marker='D', alpha=1.0, label=label)

            # empirical SMHM relations
            b = behrooziSMHM(sPs[0], redshift=0.0)
            m = mosterSMHM(sPs[0], redshift=0.0)
            k = kravtsovSMHM(sPs[0])

            ax.plot(b['m500c'], b['mstar_mid'], color='#333333', label='Behroozi+ (2013)')
            ax.fill_between(b['m500c'], b['mstar_low'], b['mstar_high'], color='#333333', interpolate=True, alpha=0.3)

            ax.plot(m['m500c'], m['mstar_mid'], color='#dddddd', label='Moster+ (2013)')
            ax.fill_between(m['m500c'], m['mstar_low'], m['mstar_high'], color='#dddddd', interpolate=True, alpha=0.3)

            ax.plot(k['m500c'], k['mstar_mid'], color='#888888', label='Kravtsov+ (2014)')

    if conf == 1:
        yQuant = 'mstar_r500'
        ylabel = 'Total Halo Stellar Mass [ log M$_{\\rm sun}$ ]' # BCG+SAT+ICL (e.g. <r500c)
        ylim = [11.9, 13.4]
        scatterColor = None #'massfrac_exsitu' (TODO: can re-enable)

        def _draw_data(ax):
            # Kravtsov+ 2018 (Figure 7 for r500c, Figure 8 for satellites within r500c)
            label = 'Kravtsov+18'
            m500c       = np.log10([5.31e13,5.68e13,1.29e14,1.79e14,2.02e14,5.40e14,5.87e14,8.59e14,1.19e15])
            mstar_r500c = np.log10([1.47e12,1.45e12,2.28e12,2.80e12,2.42e12,4.36e12,6.58e12,1.01e13,1.33e13])
            #mstar_sats  = np.log10([7.97e11,3.89e11,1.45e12,1.78e12,1.83e12,3.27e12,4.39e12,6.61e12,1.07e13])

            ax.scatter(m500c, mstar_r500c, s=markersize+20, c='#000000', marker='s', alpha=1.0, label=label)

            # Gonzalez+13 (Figure 7, mstar is <r500c, and msats is satellites within r500c)
            label = 'Gonzalez+13'
            m500c = [9.55e13,9.84e13,9.54e13,1.45e14,3.66e14,3.52e14,3.23e14,5.35e14,2.28e14,2.44e14,2.42e14,2.26e14]
            mstar = [2.82e12,3.21e12,4.18e12,3.06e12,4.99e12,6.07e12,7.53e12,7.04e12,5.95e12,5.95e12,5.56e12,5.50e12]
            #msats = [1.96e12,1.75e12,1.55e12,1.51e12,2.65e12,4.60e12,4.61e12,4.94e12,3.48e12,3.56e12,3.65e12,3.87e12]

            ax.scatter(np.log10(m500c), np.log10(mstar), s=markersize+20, c='#000000', marker='D', alpha=1.0, label=label)

            # Leauthaud+12 (obtained from Kravtsov+18 Fig 7)
            label = 'Leauthaud+12'
            m500c = [3e13, 4.26e14]
            mstar_r500c = [5.8e11, 5.6e12]

            ax.plot(np.log10(m500c), np.log10(mstar_r500c), '-', color='#000000', alpha=1.0, label=label)

            # Bahe+17 (Hydrangea sims, Fig 4 left) (arXiv:1703.10610)
            label = 'Bahe+17 (sims)'
            m500c = [13.83,13.92,13.88,13.97,14.04,14.07,14.29,14.31,14.35,14.40,14.42,14.48,14.55,14.58,
                     14.64,14.79,14.81,14.84,14.90,14.90,15.04,15.07] # 14.69,
            mstar = [12.02,12.14,12.21,12.25,12.32,12.29,12.47,12.53,12.49,12.60,12.66,12.69,12.71,12.76,
                     12.81,13.00,12.97,12.99,13.05,13.08,13.17,13.23] # 12.38,

            ax.scatter(m500c, mstar, s=markersize+20, c='#000000', marker='*', alpha=1.0, label=label)

    if conf == 2:
        yQuant = 'TODO'
        ylabel = 'Stellar Mass <100 pkpc)[ log M$_{\\rm sun}$ ]'

        def _draw_data(ax):
            pass

    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_post=_draw_data, ylabel=ylabel, legendLoc='lower right', pdf=None)

def gas_fraction_vs_halomass(sPs):
    """ Plot f_gas vs halo mass. """
    from ..load.data import giodini2009, lovisari2015

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [13.9, 15.3]
    clim = [-0.4, 0.0] # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    yQuant = 'fgas_r500'
    ylim = [0.05, 0.18]
    scatterColor = None #'massfrac_exsitu2'

    def _draw_data(ax):
        # observational points
        g = giodini2009(sPs[0])
        l = lovisari2015(sPs[0])

        ax.errorbar(g['m500_logMsun'], g['fGas500'], yerr=g['fGas500Err'],
                             color='#999999', alpha=0.9, 
                             fmt=markers[0]+linestyles[0],label=g['label'])
        ax.errorbar(l['m500_logMsun'], l['fGas500'], #yerr=l['fGas500Err'],
                             color='#555555', alpha=0.9, 
                             marker=markers[0],linestyle='',label=l['label'])

        # Tanimura+2020 (https://arxiv.org/abs/2007.02952) (xerr assumed)
        ax.errorbar( np.log10(0.9e14/0.6774), 0.13, xerr=0.1, yerr=0.03, marker='D', alpha=0.9, 
                     color='#333333', label='Tanimura+ (2020) z~0.4')

        # universal baryon fraction line
        OmegaU = sPs[0].omega_b / sPs[0].omega_m
        ax.plot( xlim, [OmegaU,OmegaU], '--', lw=1.0, color='#444444', alpha=0.3)
        ax.text( xlim[1]-0.2, OmegaU+0.003, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', size='large', alpha=0.3)

        # second legend
        plt.gca().set_prop_cycle(None) # reset color cycle
        colors = [next(ax._get_lines.prop_cycler)['color'] for i in range(len(sPs))]
        handles = [ plt.Line2D( (0,1), (0,0), color=colors[i], lw=0, marker='o') for i in range(len(sPs)) ]
        legend2 = ax.legend(handles, [sP.simName for sP in sPs], loc='upper left')
        ax.add_artist(legend2)

    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_post=_draw_data, legendLoc='lower right', pdf=None)

def sfr_vs_halomass(sPs):
    """ Plot star formation rate vs halo mass. """
    #from ..load.data import giodini2009, lovisari2015
    xQuant = 'mhalo_200_log'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.4]
    clim = [-0.4, 0.0] # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    yQuant = 'sfr_30pkpc'
    ylim = [-3.5, 4.0]
    scatterColor = None

    def _draw_data(ax):
        # observational points


        # 'quenched' indicators i.e. threshold line
        mhalo = sPs[0].subhalos(xQuant)
        mstar = sPs[0].subhalos('mstar_30pkpc_log')
        inds = sPs[0].cenSatSubhaloIndices(cenSatSelect='cen')
        mhalo = mhalo[inds]
        mstar = mstar[inds]

        xx, yy, _ = running_median(mhalo, mstar, binSize=0.1) # determine mstar/mhalo relation

        xx_mhalo = xlim #np.linspace(xlim[0], xlim[1], 10) # mhalo
        f_interp = interp1d(xx, yy, kind='linear', fill_value='extrapolate') # interpolate mstar to requested mhalo values
        xx_mstar = f_interp(xx_mhalo)

        ssfr_thresh = 1e-11 # 1/yr
        sfr = np.log10(10.0**xx_mstar * ssfr_thresh) # log(1/yr)
        label = 'Quiescent\n(sSFR < %g yr$^{-1}$)' % ssfr_thresh
        ax.plot(xx_mhalo, sfr, '-', lw=lw, color='#000', alpha=0.5)
        ax.fill_between(xx_mhalo, ylim[0], sfr, color='#000', alpha=0.1)
        ax.text(xlim[1]-0.05, ylim[0]+2.0, label, color='#000', alpha=0.5, fontsize=20, ha='right')

    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, legendLoc='upper left', pdf=None)

def mhi_vs_halomass(sPs):
    """ Plot cold gas mass (M_HI) vs halo mass. """
    from ..load.data import obuljen2019
    from scipy.signal import savgol_filter

    xQuant = 'mhalo_200_log'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.4]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    # TODO: we should perhaps do 2d projection within r200, over a pretty good depth (to match ALFALFA beam)
    yQuant = 'mhi_halo'
    ylim = [9.9, 12.0]
    scatterColor = None #'xray_peak_offset_2d_r500'
    clim = [-2.0, -1.0] # log fraction

    def _draw_data(ax):
        # observational points
        o18 = obuljen2019()

        color = '#222222'
        l1, = ax.plot(o18['Mhalo'], o18['mHI'], color=color)
        ax.fill_between(o18['Mhalo'], savgol_filter(o18['mHI_low'],sKn,sKo), savgol_filter(o18['mHI_high'],sKn,sKo), color=color, alpha=0.2)

        ax.errorbar(o18['pts_M_halo'], o18['pts_MHI'], yerr=[o18['pts_MHI_errdown'],o18['pts_MHI_errup']], 
                    fmt='D', zorder=2, color='#000000', alpha=0.7, label=o18['label']) # ms=6

        # TODO: cold gas mass (https://arxiv.org/abs/2305.12750 Fig 8)

        # TODO: add virgo
        # https://arxiv.org/pdf/2209.07691.pdf (Fig 8)


    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, legendLoc='upper left', pdf=None)                

def generateProjections(sP, partType='gas', partField='coldens_msunkpc2', conf=0, saveImages=False):
    """ Generate projections for a given configuration, save results into a single postprocessing file. """
    from ..vis.halo import renderSingleHalo

    axes_set   = [[0,1],[0,2],[1,2]]
    nPixels    = 2000
    depthType  = 'rVirial' # r200c

    if conf == 0:
        sizeType  = 'rVirial' # r200c
        size      = 4.0 # +/- 2 rvir in extent
        depth     = 2.0 # +/- 1 rvir in depth
        confStr   = '2r200_d=r200'
    if conf == 1:
        sizeType  = 'r500'
        size      = 1.0 # +/- 0.5 r500 in extent
        depth     = 2.0 # +/- 1 rvir in depth
        confStr   = '0.5r500_d=r200'
    if conf == 2:
        sizeType  = 'r500'
        size      = 1.0 # +/- 0.5 r500 in extent
        depth     = 6.0 # +/- 3 rvir in depth
        confStr   = '0.5r500_d=3r200'
    if conf == 3:
        sizeType  = 'r500'
        depthType = 'r500'
        size      = 2.0 # +/- r500 in extent
        depth     = 2.0 # +/- r500 in depth
        confStr   = 'r500_d=r500'

    method = 'sphMap_globalZoomOrig' # all particles of original zoom run only

    # TNG300: all halos with M200c > 14.0, TNG-Cluster: all primary zoom targets
    subhaloIDs = sP.cenSatSubhaloIndices(cenSatSelect='cen')
    m200c = sP.subhalos('mhalo_200_log')[subhaloIDs]

    if sP.simName != 'TNG-Cluster':
        # include low-mass progenitors at high redshift
        w = np.where(m200c > 14.0)[0]
        subhaloIDs = subhaloIDs[w]

    haloIDs = sP.subhalos('SubhaloGrNr')[subhaloIDs]

    r200c = sP.subhalos('r200')[subhaloIDs]
    r500c = sP.subhalos('r500')[subhaloIDs]

    # render images?
    if saveImages:
        # plot config
        class plotConfig:
            plotStyle  = 'edged' # open, edged
            rasterPx   = [800,800]
            colorbars  = True
            fontsize   = 14

        labelHalo = 'mhalo,haloidorig'
        labelSim = True
        labelZ = True
        labelScale = True

        # loop over all halos
        for subhaloInd, haloID in zip(subhaloIDs, haloIDs):
            # loop over all projection directions
            for axes in axes_set:
                # render and stamp
                panels = [{}]

                projAxis = ['x','y','z'][3-np.sum(axes)]
                saveStr = '%s-%s_%s_%s' % (partType,partField,confStr,projAxis)
                labelCustom = [confStr.replace('_',' ') + ' ($\hat{%s}$)' % projAxis]
                plotConfig.saveFilename = '%s.%d.%08d.%s.png' % (sP.simName,sP.snap,subhaloInd,saveStr)
                
                renderSingleHalo(panels, plotConfig, locals()) #, skipExisting=False)

        print('Done.')
        return

    # save projections (instead of rendering images): start save file
    savePath = sP.postPath + 'projections/'
    saveFilename = savePath + '%s-%s_%03d_%s.hdf5' % (partType, partField, sP.snap, confStr)

    with h5py.File(saveFilename,'a') as f:
        f.attrs['axes_set'] = axes_set
        f.attrs['nPixels'] = nPixels
        f.attrs['size'] = size
        f.attrs['sizeType'] = sizeType
        f.attrs['method'] = method
        f.attrs['partType'] = partType
        f.attrs['partField'] = partField
        f.attrs['simName'] = sP.simName
        f.attrs['snap'] = sP.snap
        f.attrs['depth'] = depth
        f.attrs['depthType'] = depthType

        if 'HaloIDs' not in f:
            f['HaloIDs'] = haloIDs
            f['SubhaloIDs'] = subhaloIDs
            f['r200c'] = r200c
            f['r500c'] = r500c

            dist, theta = _get_dist_theta_grid(size, nPixels)

            f['grid_dist'] = dist
            f['grid_angle'] = theta

    # loop over all halos
    for i, haloID in enumerate(haloIDs):
        # check for existence
        subhaloInd = subhaloIDs[i]
        gName = f'Halo_{haloID}'

        print(f'[{i:03d} of {len(haloIDs):03d}] Halo ID = {haloID}')

        with h5py.File(saveFilename,'r') as f:
            if gName in f:
                print(' skip')
                continue

        # loop over all orientations
        grids = np.zeros((nPixels,nPixels,len(axes_set)), dtype='float32')

        for j, axes in enumerate(axes_set):
            # render and stamp
            panels = [{}]
            plotConfig = lambda: None
            
            grid_loc, config = renderSingleHalo(panels, plotConfig, locals(), returnData=True)
            grids[:,:,j] = grid_loc

        # save
        with h5py.File(saveFilename,'a') as f:
            f.create_dataset(gName, data=grids)

            f.attrs['name'] = config['label']

            f[gName].attrs['minmax_guess'] = config['vMM_guess']
            f[gName].attrs['box_center'] = config['boxCenter'] # code
            f[gName].attrs['box_size'] = config['boxSizeImg'] # code, multiply by grid_dist to have px dists in code units for this halo
    
    print('Done.')

def _find_peak(grid, method='center_of_light'):
    """ Find peak location in a 2D grid using different methods. Return is in pixel coordinates. """
    x = np.arange(grid.shape[0])
    y = np.arange(grid.shape[1])

    grid_norm = grid / grid.max()

    xx, yy = np.meshgrid(x, y, indexing='xy')

    if method == 'max_pixel':
        # single maximum valued pixel
        ind = np.argmax(grid)
        xy_cen = np.unravel_index(ind, grid.shape)[::-1]

    if method == 'center_of_light':
        # center of light
        xy_cen = np.average(xx, weights=grid_norm), np.average(yy, weights=grid_norm)

    if method == 'shrinking_circle':
        # iterative/shrinking circle
        npx = grid.shape[0] * grid.shape[1]
        rad = grid.shape[0] / 2
        xy_cen = np.array([grid.shape[0]/2, grid.shape[1]/2], dtype='float32')
        iter_count = 0

        circ_cens = []
        circ_rads = []

        while npx > 100 and iter_count < 500:
            dx = xx - xy_cen[0]
            dy = yy - xy_cen[1]
            rr = np.sqrt(dx**2 + dy**2)

            ind = np.where(rr < rad)
            npx = len(ind[0])

            xy_cen = np.average(xx[ind], weights=grid_norm[ind]), np.average(yy[ind], weights=grid_norm[ind])
            rad *= 0.95

            iter_count += 1

            #print(iter_count, npx, xy_cen, rad)
            circ_cens.append(xy_cen)
            circ_rads.append(rad)

        return xy_cen, circ_cens, circ_rads

    return xy_cen

def summarize_projection_2d(sim, pSplit=None, quantity='sz_yparam', projConf='2r200_d=r200', 
                            op='sum', aperture='r500'):
    """ Calculate summary statistic(s) from existing projections in 2D, e.g. Y_{r500,2D} for SZ. """
    # config
    assert pSplit is None
    assert op in ['sum','peak_offset']

    path = sim.postPath + 'projections/'
    filename = 'gas-%s_%03d_%s.hdf5' % (quantity,sim.snap,projConf)

    # load list of halos
    with h5py.File(path + filename,'r') as f:
        haloIDs = f['HaloIDs'][()]
        subhaloIDs = f['SubhaloIDs'][()]
        nproj = f['Halo_%d' % haloIDs[0]].shape[2]
        name = f.attrs['name']

    # allocate
    rr = np.zeros((len(haloIDs), nproj), dtype='float32')

    # load distance grid
    with h5py.File(path + filename,'r') as f:
        dist = f['grid_dist'][()]

    # loop over all halos
    for haloInd, haloID in enumerate(haloIDs):
        # status
        if haloInd % 10 == 0: print(' %4.1f%%' % (float(haloInd+1)*100.0/len(haloIDs)))

        # load three projections
        with h5py.File(path + filename,'r') as f:
            proj = f['Halo_%d' % haloID][()]
            box_size = f['Halo_%d' % haloID].attrs['box_size']

        # halo-dependent unit conversions
        assert box_size[0] == box_size[1] and proj.shape[0] == proj.shape[1]
        dists_code_loc = dist * (box_size[0]/2)

        pxSize_code = box_size[0] / proj.shape[0]
        pxSize_Kpc = sim.units.codeLengthToKpc(pxSize_code)
        pxArea_Kpc = pxSize_Kpc**2

        if op == 'sum':
            # halo-specific aperture [code units]
            if aperture == 'r500':
                aperture_rad = sim.halo(haloID)['Group_R_Crit500']
            if aperture == 'r200':
                aperture_rad = sim.halo(haloID)['Group_R_Crit200']

            assert aperture_rad < box_size[0] / 2.0 + 1e-6

            # select spatial region
            w_px_spatial = np.where(dists_code_loc < aperture_rad)

        # loop over each projection direction
        for i in range(nproj):
            # linear map [e.g. dimensionless for SZY, erg/s/kpc^2 for LX]
            proj_loc = 10.0**(proj[:,:,i]).astype('float64')

            if op == 'sum':
                # integrate within aperture
                quant_sum = np.nansum(proj_loc[w_px_spatial])
                
                # multiply by total area in [pKpc^2]
                # e.g. for SZ, this is [dimensionless] -> [pKpc^2]
                # e.g. for LX, this is [erg/s/kpc^2] -> [erg/s]
                quant_sum *= pxArea_Kpc

                rr[haloInd,i] = np.log10(quant_sum)

            if op == 'peak_offset':
                # mark galaxy position (SubhaloPos at center by definition)
                xy_cen0 = proj.shape[0] / 2.0, proj.shape[1] / 2.0

                xy_cen2, circ_cens, circ_rads = _find_peak(proj_loc, method='shrinking_circle')
                offsets_px = np.sqrt((xy_cen0[0]-xy_cen2[0])**2 + (xy_cen0[1]-xy_cen2[1])**2)
                offsets_code = offsets_px * pxSize_code

                rr[haloInd,i] = offsets_code

    # return quantities for save, as expected by load.auxCat()
    if op == 'sum':
        units = name.split('[')[-1].split(']')[0] + ' pkpc^2'
        desc = 'Sum of [%s] in 2D projection (%s) [%s].' % (quantity,projConf,units)
    if op == 'peak_offset':
        units = 'code_length'
        desc = 'Spatial offset of peak [%s] in 2D projection (%s) [%s].' % (quantity,projConf,units)

    select = 'TNG-Cluster primary zoom targets only.'

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : 'gas'.encode('ascii'),
             'subhaloIDs'  : subhaloIDs}

    return rr, attrs

def szy_vs_halomass(sPs):
    """ Plot SZ y-parameter vs halo mass. """
    from ..load.data import hilton21act, planck13xx, bleem15spt, nagarajan18

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.3]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    yQuant = 'szy_r500c_3d' # 2D: 'szy_r500c_2d' (only for TNG-Cluster)
    ylim = [-5.8, -3.5]
    scatterColor = None

    def _draw_data_pre(ax):
        pass

    def _draw_data(ax):
        # create second ghost axis (to hold a second legend)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_axis_off()

        # (1) observational shaded bands

        # lines from Jimeno+18 (Fig 7) for Planck (red)
        xx = np.log10([8.012e13, 3.0e15]) # 1.655e15
        y_lower = np.log10([7.818e-7, 0.00052325]) # 1.798e-4
        y_upper = np.log10([1.578e-6, 0.00106515]) # 3.655e-4

        ax2.fill_between(xx, y_lower, y_upper, color='#000000', alpha=0.2, zorder=-4, label='Planck XX (2013)')

        # lines from Jimeno+18 (Fig 7) for that work (blue)
        xx = np.log10([8.012e13, 3.0e15])
        y_lower = np.log10([8.244e-7, 0.000380666])
        y_upper = np.log10([1.447e-6, 0.000658573])

        # Jimeno+ (2018) https://arxiv.org/abs/1706.00395 (Eqn. 30)
        #M500 = 10.0**np.array(ax.get_xlim()) # msun
        #mass_bias = 0.82
        #Y500 = 10.0**(-0.35) * (mass_bias*M500 / 6e14)**(1.70)
        #Y500 = np.log10(Y500 * 1e-4) # convert to pMpc^2
        #ax.plot(np.log10(M500), Y500, '-', color='#000000', alpha=0.7, label='Jimeno+ (2018) Planck')

        ax2.fill_between(xx, y_lower, y_upper, color='#000000', alpha=0.3, zorder=-3, label='Jimeno+ (2018) Planck')

        # Hill+ 2018
        #M500 = np.array([1.0e12, 1.0e13, 6.63e13, 2.44e14, 5.59e14, 1.0e15])
        #Y500 = np.array([2.845e-9, 5.687e-7, 5.381e-5, 5.179e-4, 2.218e-3, 6.074e-3]) # arcmin^2 at z=0.15
        #Y500 *= 3600 # arcsec^2
        #Y500 *= sPs[0].units.arcsecToAngSizeKpcAtRedshift(1.0, 0.15)**2 / 1e6 # convert to pMpc^2

        #ax2.plot(np.log10(M500), np.log10(Y500), ':', color='#000000', alpha=0.7, label='Hill+ (2018) CB')

        # self-similar slope
        M500 = np.array([3e14, 9e14])
        Y500 = np.log10(1e-5 * (M500/5e14)**(5/3))

        ax2.plot(np.log10(M500), Y500, '--', color='#000000', alpha=0.7, label='$\\rm{Y_{500} \propto M_{500}^{5/3}}$')

        # (2) observational pointsets
        #b15 = bleem15spt(sPs[0])
        #h21 = hilton21act()
        #p13 = planck13xx()
        n18 = nagarajan18()

        #ax.plot(h21['m500'], h21['sz_y'], 'p', color='#000000', ms=6, alpha=0.7)
        #ax.plot(p13['M500'], p13['Y500'], '*', color='#000000', ms=6, alpha=0.7)
        #ax.plot(b15['M500'], b15['Y'], 's', color='#000000', ms=6, alpha=0.7)

        xerr = np.vstack((n18['M500_errdown'],n18['M500_errup']))
        yerr = np.vstack((n18['Y_errup'],n18['Y_errdown']))
        ax2.errorbar(n18['M500'], n18['Y'], xerr=xerr, yerr=yerr, fmt='D', zorder=-2, 
                     color='#aaaaaa', ms=6, alpha=0.7, label=n18['label'])

        # second legend
        ax2.legend(loc='lower right')

    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data_pre, f_post=_draw_data, legendLoc='upper left', pdf=None)

def XrayLum_vs_halomass(sPs):
    """ Plot X-ray luminosity vs halo mass. """
    from ..load.data import pratt09, vikhlinin09, mantz16, bulbul19

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.3]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    yQuant = 'xray_0.5-2.0kev_r500_halo' # 2D: 'xraylum_r500c_2d' (only for TNG-Cluster)
    ylim = [42.8, 46.0]
    scatterColor = None

    def _draw_data(ax):
        # observational points
        p09 = pratt09()
        v09 = vikhlinin09()
        m16 = mantz16()
        b19 = bulbul19()

        markers = ['p','D','*','s']

        ax.plot(p09['M500'], p09['L05_2'], markers[0], color='#000000', ms=6, alpha=0.7)
        ax.plot(v09['M500_Y'], v09['LX'], markers[1], color='#000000', ms=6, alpha=0.7)
        ax.plot(m16['M500'], m16['LX'], markers[2], color='#000000', ms=9, alpha=0.7)
        ax.plot(b19['M500'], b19['LX'], markers[3], color='#000000', ms=6, alpha=0.7)

        labels = [p09['label'],v09['label'],m16['label'],b19['label']]

        # second legend
        handles = [plt.Line2D((0,1), (0,0), color='black', lw=0, marker=m) for m in markers]
        legend2 = ax.legend(handles, labels, borderpad=0.4, loc='lower right')
        ax.add_artist(legend2)

    quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_post=_draw_data, legendLoc='upper left', pdf=None)

def paperPlots():
    """ Plots for TNG-Cluster intro paper. """
    from ..util.simParams import simParams

    # all analysis at z=0 unless changed below
    TNG300 = simParams(run='tng300-1', redshift=0.0)
    TNG_C  = simParams(run='tng-cluster', redshift=0.0)

    sPs = [TNG300, TNG_C]

    # figure 1 - mass function
    if 0:
        mass_function()
        #mass_function(secondaries=True)

    # figure 2 - samples
    if 0:
        sample_halomasses_vs_redshift(sPs)

    # figure 3 - simulation meta-comparison
    if 0:
        simClustersComparison()

    # figures 4,5 - individual halo/gallery vis (x-ray)
    if 1:
        #for conf in range(10):
        #    vis_gallery(TNG_C, conf=conf, num=1) # single
        vis_gallery(TNG_C, conf=10, num=1) # single
        #vis_gallery(TNG_C, conf=1, num=72) # gallery

    # figure 6 - gas fractions
    if 0:
        gas_fraction_vs_halomass(sPs)

    # figure 7 - magnetic fields
    if 0:
        redshifts = [0.0, 1.0, 2.0]
        bfield_strength_vs_halomass(sPs, redshifts)

    # figure 8 - halo synchrotron power
    if 0:
        from ..plot.globalComp import haloSynchrotronPower
        haloSynchrotronPower(sPs, xlim=[14.0,15.3], ylim=[21.5,28.5])

    # figure 9 - X-ray scaling relations
    if 0:
        from ..plot.globalComp import haloXrayLum
        sPs_loc = [sP.copy() for sP in sPs]
        for sP in sPs_loc: sP.setRedshift(0.3) # median of data at high-mass end
        haloXrayLum(sPs_loc, xlim=[11.4,12.7], ylim=[41.6,45.7])

    # figure 10 - SZ-y and X-ray vs mass scaling relations
    if 0:
        szy_vs_halomass(sPs)
        XrayLum_vs_halomass(sPs)

    # figure 11 - sfr/cold gas mass
    if 0:
        #sfr_vs_halomass(sPs)
        mhi_vs_halomass([TNG_C])
        # todo: quenched fraction

    # figure 12 - stellar mass contents
    if 0:
        stellar_mass_vs_halomass(sPs, conf=0)
        stellar_mass_vs_halomass(sPs, conf=1)

    # figure 14 - black hole mass scaling relation
    if 0:
        from ..plot.globalComp import blackholeVsStellarMass

        pdf = PdfPages('blackhole_masses_vs_mstar_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        blackholeVsStellarMass(sPs, pdf, twiceR=True, xlim=[11,13.0], ylim=[7.5,11], sizefac=0.8)
        pdf.close()

        pdf = PdfPages('blackhole_masses_vs_mbulge_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        blackholeVsStellarMass(sPs, pdf, vsBulgeMass=True, xlim=[11.5,13.0], ylim=[8.5,11], sizefac=0.8)
        pdf.close()

        pdf = PdfPages('blackhole_masses_vs_mhalo_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        blackholeVsStellarMass(sPs, pdf, vsHaloMass=True, xlim=[13.9,15.4], ylim=[8.5,11], sizefac=0.8)
        pdf.close()

        # todo: add vs sigma
        # https://arxiv.org/abs/2308.01800

    # appendix - virtual full box vis
    if 0:
        for conf in [0,1,2,3,4,5,6]:
            vis_fullbox_virtual(TNG_C, conf=conf)

    # appendix - contamination
    if 0:
        contamination_mindist()

    # satellite smhm
    # satellite radial number density (https://arxiv.org/abs/2305.09629 Fig 11)
    # richness (N_sat), see Pakmor MTNG paper (also https://arxiv.org/abs/2307.08749)
    # SPA relaxedness measures from X-ray SB maps (https://arxiv.org/abs/2303.10185) (https://arxiv.org/pdf/2006.10752.pdf)
    # members: L_X vs velocity dispersion relation (Kirkpatrick,C.+2020 SPIDERS), 'cosmo constraints with the velocity dispersion function'
    # TODO: https://arxiv.org/abs/2111.13071 (M_SZ vs M_dyn from PSZ2)

    # ex-situ fractions, or radial profiles of ex-situ stacked in mass (or similar)

    # figure todo - kinematics overview (vel disp cold,warm,hot) (https://arxiv.org/abs/2304.08810)
