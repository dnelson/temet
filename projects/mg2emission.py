"""
projects/mg2emission.py
  TNG50: MgII CGM emission paper.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from os.path import isfile
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

from vis.halo import renderSingleHalo
from util.helper import running_median, sampleColorTable, loadColorTable
from plot.cosmoGeneral import quantMedianVsSecondQuant, quantHisto2D
from plot.config import *
from projects.azimuthalAngleCGM import _get_dist_theta_grid

def singleHaloImageMGII(sP, hInd, conf=1):
    """ MgII emission image test.. """
    rVirFracs  = [0.25]
    method     = 'sphMap' # note: fof-scope
    nPixels    = [800,800]
    axes       = [0,1]
    labelZ     = True
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    rotation   = 'edge-on'

    sizeType   = 'kpc'
    size       = 100

    # smooth? MUSE UDF has a PSF FWMH ~ 0.7 arcsec
    #smoothFWHM = sP.units.arcsecToAngSizeKpcAtRedshift(0.7)

    # contour test
    contour = ['stars','coldens_msunkpc2']
    contourLevels = [7.0,7.5,8.0] # msun/kpc^2
    contourOpts = {'colors':'white', 'alpha':0.8}

    panels = []

    if conf == 0:
        panels.append( {'partType':'gas', 'partField':'Mg II', 'valMinMax':[10.0,16.0]} )
    if conf == 1:
        panels.append( {'partType':'gas', 'partField':'sb_MgII_ergs_dustdeplete', 'valMinMax':[-20.5, -17.0]} )
    if conf == 2:
        panels.append( {'partType':'gas', 'partField':'vrad', 'valMinMax':[-180,180]} )
    if conf == 3:
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,7.5]} )
    if conf == 4:
        panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,7.5]} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0] 
        colorbars    = True
        fontsize     = 16
        saveFilename = '%s_%d_%d_%s.pdf' % (sP.simName,sP.snap,hInd,'-'.join([p['partField'] for p in panels]))

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def visMg2Gallery(sP, mStarBin, num, size=100, colorbar=True):
    """ MgII emission: image gallery of example stamps in a given stellar mass bin. """
    rVirFracs  = [5.0]
    fracsType  = 'rHalfMassStars'
    method     = 'sphMap' # note: fof-scope
    nPixels    = [800,800]
    axes       = [0,1]
    labelZ     = False
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = 'mstar'
    rotation   = None # random
    sizeType   = 'kpc'

    partType  = 'gas'
    partField = 'sb_MgII_ergs_dustdeplete'
    valMinMax = [-20.5, -17.0]

    # targets
    mstar = sP.subhalos('mstar_30pkpc_log')
    cen_flag = sP.subhalos('central_flag')
    with np.errstate(invalid='ignore'):
        w = np.where( (mstar>mStarBin[0]) & (mstar<mStarBin[1]) & cen_flag )[0]

    np.random.seed(424242)
    np.random.shuffle(w)
    hInds = w[0:num]

    panels = []

    for hInd_loc in hInds:
        panels.append( {'hInd':hInd_loc} )

    panels[-1]['labelZ'] = True

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0]/2
        colorbars    = colorbar
        fontsize     = 22
        nCols        = int(np.floor(np.sqrt(num)))
        nRows        = int(np.ceil(num/nCols))
        saveFilename = ('%s_%d_mstar-%.1f-%.1f_gallery-%d' % \
          (sP.simName,sP.snap,mStarBin[0],mStarBin[1],num)).replace(".","p") + '.pdf'

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def visDustDepletionImpact(sP, hInd):
    """ MgII emission image test.. """
    
    rVirFracs  = [0.25]
    method     = 'sphMap'
    nPixels    = [800,800]
    axes       = [0,1]
    labelZ     = True
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = False
    relCoords  = True
    rotation   = 'edge-on'

    sizeType   = 'kpc'
    size       = 100

    # panels
    panels = []

    partType = 'gas'
    valMinMax = [-20.5, -17.0]

    panels.append( {'partField':'sb_MgII_ergs', 'labelZ':False, 'labelCustom':['no dust depletion']} )
    panels.append( {'partField':'sb_MgII_ergs_dustdeplete', 'labelScale':False, 'labelCustom':['dust-depleted']} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0] 
        colorbars    = True
        fontsize     = 32
        saveFilename = '%s_%d_%d_vs_nodustdepletion.pdf' % (sP.simName,sP.snap,hInd)

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def radialSBProfiles(sP, massBins, minRedshift=None, psf=False, indiv=False, xlim=None):
    """ Use grids to produce individual and stacked radial surface brightness profiles. """
    method     = 'sphMap' # note: fof-scope
    nPixels    = [800,800]
    axes       = [0,1]
    rotation   = None # random
    sizeType   = 'kpc'
    size       = 100

    partType   = 'gas'
    partField  = 'sb_MgII_ergs_dustdeplete'

    # binning config
    nRadBins  = 100
    radMinMax = [0.0, 70] # ~ sqrt((size/2)**2 + (size/2)**2)
    radBins   = np.linspace(radMinMax[0], radMinMax[1], nRadBins+1)
    radMidPts = radBins[:-1] + (radMinMax[1]-radMinMax[0])/nRadBins/2
    percs     = [16,50,84] # +/- 1 sigma (50 must be in the middle)

    # load catalog
    dist, _ = _get_dist_theta_grid(size, nPixels)

    mstar = sP.subhalos('mstar_30pkpc_log')
    cen_flag = sP.subhalos('central_flag')

    # MUSE UDF has a PSF FWMH ~ 0.7 arcsec
    pxScale = size / nPixels[0] # pkpc/px
    psfFWHM_px = sP.units.arcsecToAngSizeKpcAtRedshift(0.7) / pxScale
    psfSigma_px = psfFWHM_px / 2.3548

    def _rad_profile(rad_pts, sb_pts):
        """ Profile helper. """
        w = np.where(np.isfinite(sb_pts))
        rad_pts = rad_pts[w]
        sb_pts = sb_pts[w] # remove a negligible number of nan pixels

        # make profile: adaptive running median
        #rr, sb, sb_std, sb_percs = running_median(rad_pts, sb_pts, nBins=nRadBins, percs=percs)

        # make profile: mean of pixels in each annulus
        yy = 1e20 * (10.0 ** sb_pts.astype('float64'))
        sb2, rr_edges = np.histogram(rad_pts, bins=nRadBins, range=radMinMax, weights=yy)
        npx, _ = np.histogram(rad_pts, bins=nRadBins, range=radMinMax)
        sb2 = np.log10(sb2/npx/1e20).astype('float32')

        # make profile: median (and percentiles) of pixels in each annulus
        sb3 = np.zeros( (nRadBins,len(percs)), dtype='float32' )
        for j in range(nRadBins):
            w = np.where( (rad_pts >= radBins[j]) & (rad_pts < radBins[j+1]) )
            sb3[j,:] = np.nanpercentile(sb_pts[w], percs)

        return sb2, sb3

    # start figure
    sizefac = 0.8 # for single column figure
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_ylabel('MgII SB [log erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
    ax.set_xlim(radMinMax if xlim is None else xlim)
    ax.set_ylim([-22.0,-16.5])

    def _cachefile(sP):
        return sP.derivPath + 'cache/mg2emission_%d_sbr_%d_%d_%s_%d.hdf5' % (sP.snap,massBin[0]*10,massBin[1]*10,rotation,size)

    # not at lowest redshift? add low redshift line for comparison
    if not indiv and sP.redshift > minRedshift+0.1:
        sP_lowz = sP.copy()
        sP_lowz.setRedshift(minRedshift)
        massBin = massBins[-1] # highest mass

        if isfile(_cachefile(sP_lowz)):
            with h5py.File(_cachefile(sP_lowz),'r') as f:
                sbr_indiv_mean = f['sbr_indiv_mean'][()]
                sbr_indiv_mean_psf = f['sbr_indiv_mean_psf'][()]

        if psf:
            yy = np.nanpercentile(sbr_indiv_mean_psf, percs, axis=0)
        else:
            yy = np.nanpercentile(sbr_indiv_mean, percs, axis=0)
        l, = ax.plot(radMidPts, yy[1,:], lw=lw, linestyle='-', color='#bbbbbb', alpha=0.4)

    if indiv:
        colors = sampleColorTable('rainbow', indiv, bounds=[0.0,1.0])
        #cmap = loadColorTable('rainbow')
        cmap = loadColorTable('terrain', fracSubset=[0.0,0.8])
    else:
        colors = sampleColorTable('plasma', len(massBins), bounds=[0.1,0.7])

    # loop over halos in each mass bin
    for i, massBin in enumerate(massBins):

        with np.errstate(invalid='ignore'):
            subInds = np.where( (mstar>massBin[0]) & (mstar<massBin[1]) & cen_flag )[0]

        print('[%.2f - %.2f] Processing [%d] halos...' % (massBin[0],massBin[1],len(subInds)))

        # check for existence of cache
        if isfile(_cachefile(sP)):
            # load cached result
            with h5py.File(_cachefile(sP),'r') as f:
                sbr_indiv_mean = f['sbr_indiv_mean'][()]
                sbr_indiv_med = f['sbr_indiv_med'][()]
                sbr_indiv_mean_psf = f['sbr_indiv_mean_psf'][()]
                sbr_indiv_med_psf = f['sbr_indiv_med_psf'][()]
                sbr_stack_mean = f['sbr_stack_mean'][()]
                sbr_stack_med = f['sbr_stack_med'][()]
                sbr_stack_mean_psf = f['sbr_stack_mean_psf'][()]
                sbr_stack_med_psf = f['sbr_stack_med_psf'][()]
                #grid_global = f['grid_global'][()]
            print('Loaded: [%s]' % _cachefile(sP))
        else:
            # compute now
            sbr_indiv_mean = np.zeros( (len(subInds), nRadBins), dtype='float32' )
            sbr_indiv_med = np.zeros( (len(subInds), nRadBins, len(percs)), dtype='float32' )
            sbr_indiv_mean_psf = np.zeros( (len(subInds), nRadBins), dtype='float32' )
            sbr_indiv_med_psf = np.zeros( (len(subInds), nRadBins, len(percs)), dtype='float32' )
            grid_global = np.zeros( (len(subInds), nPixels[0], nPixels[1]), dtype='float32' )

            sbr_indiv_mean.fill(np.nan)
            sbr_indiv_med.fill(np.nan)
            sbr_indiv_mean_psf.fill(np.nan)
            sbr_indiv_med_psf.fill(np.nan)

            for j, hInd in enumerate(subInds):
                class plotConfig:
                    saveFilename = 'dummy'

                grid, _ = renderSingleHalo([{}], plotConfig, locals(), returnData=True)

                grid_global[j,:,:] = grid
                sbr_indiv_mean[j,:], sbr_indiv_med[j,:,:] = _rad_profile(dist.ravel(), grid.ravel())

                # psf smooth and recompute
                grid2 = gaussian_filter(grid, psfSigma_px, mode='reflect', truncate=5.0)
                sbr_indiv_mean_psf[j,:], sbr_indiv_med_psf[j,:,:] = _rad_profile(dist.ravel(), grid2.ravel())

            # create dist in same shape as grid_global
            dist_global = np.zeros( (len(subInds), nPixels[0]*nPixels[1]), dtype='float32' )

            for j, hInd in enumerate(subInds):
                dist_global[j,:] = dist.ravel()

            # compute profile on 'stacked images'
            sbr_stack_mean, sbr_stack_med = _rad_profile(dist_global.ravel(), grid_global.ravel())

            # psf smooth and recompute
            grid_global2 = gaussian_filter(grid_global, psfSigma_px, mode='reflect', truncate=5.0)
            sbr_stack_mean_psf, sbr_stack_med_psf = _rad_profile(dist_global.ravel(), grid_global2.ravel())

            # save cache
            with h5py.File(_cachefile(sP),'w') as f:
                f['grid_global'] = grid_global
                f['sbr_indiv_mean'] = sbr_indiv_mean
                f['sbr_indiv_med'] = sbr_indiv_med
                f['sbr_indiv_mean_psf'] = sbr_indiv_mean_psf
                f['sbr_indiv_med_psf'] = sbr_indiv_med_psf

                f['sbr_stack_mean'] = sbr_stack_mean
                f['sbr_stack_med'] = sbr_stack_med
                f['sbr_stack_mean_psf'] = sbr_stack_mean_psf
                f['sbr_stack_med_psf'] = sbr_stack_med_psf

            print('Saved: [%s]' % _cachefile(sP))

        # use psf smoothed results? just replace all arrays now
        if psf:
            sbr_indiv_mean = sbr_indiv_mean_psf
            sbr_indiv_med = sbr_indiv_med_psf
            sbr_stack_mean = sbr_stack_mean_psf
            sbr_stack_med = sbr_stack_med_psf

        # individual profiles or stack?
        if indiv:
            # plot stack under
            yy = np.nanpercentile(sbr_indiv_med[:,:,1], percs, axis=0)
            yy = savgol_filter(yy, sKn, sKo, axis=1)
            #l, = ax.plot(radMidPts, yy[1,:], linestyle='-', lw=lw*4, color='#000000', alpha=0.4)
            ax.fill_between(radMidPts, yy[0,:], yy[2,:], color='#000000', alpha=0.2)

            norm = Normalize(vmin=massBin[0], vmax=massBin[1]) # color on mstar

            # plot individual profiles
            for j in range( np.min([indiv,sbr_indiv_med.shape[0]]) ):
                yy = sbr_indiv_med[j,:,1]
                if 0:
                    c = colors[j] # sampled uniformly from colormap in order of subhalo IDs
                if 1:
                    c = cmap(norm(mstar[subInds[j]]))

                ax.plot(radMidPts, savgol_filter(yy,sKn,sKo,axis=0), '-', color=c)
        else:
            # compute stack of 'per halo profiles' and plot
            label = 'M$_{\star}$ = %.1f' % (massBin[0])

            yy = np.nanpercentile(sbr_indiv_med[:,:,1], percs, axis=0)
            l, = ax.plot(radMidPts, yy[1,:], linestyle=':', lw=lw, color=colors[i])
            #ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=colors[i], alpha=0.15)

            yy = np.nanpercentile(sbr_indiv_mean, percs, axis=0)
            l, = ax.plot(radMidPts, yy[1,:], lw=lw, linestyle='-', color=colors[i], label=label)
            ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.15)

            # plot 'image stack' and shaded band
            #l, = ax.plot(radMidPts, sbr_stack_med[:,1], lw=lw, label=label + ' (stack then prof, median)')
            #ax.fill_between(radMidPts, sbr_stack_med[:,0], sbr_stack_med[:,2], color=l.get_color(), alpha=0.1)

            #l, = ax.plot(radMidPts, sbr_stack_mean, lw=lw, label=label + ' (stack then prof, mean)')

    # finish and save plot
    if indiv:
        cax = fig.add_axes([0.5,0.84,0.4,0.04])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.ax.set_title('Galaxy Stellar Mass [log M$_{\\rm sun}$]')
    else:
        ax.text(ax.get_xlim()[1]-0.22*(ax.get_xlim()[1]-ax.get_xlim()[0]), -18.9, 'z = %.1f' % sP.redshift, fontsize=24)
        ax.legend(loc='best')

    fig.savefig('SB_profiles_%s_%d_%s%s%s.pdf' % \
        (sP.simName,sP.snap,partField,'_indiv' if indiv else '','_psf' if psf else ''))
    plt.close(fig)

def mg2lum_vs_mass(sP, redshifts=None):
    """ Driver for quantMedianVsSecondQuant. """
    sPs = [sP]
    if redshifts is not None:
        sPs = []
        for redshift in redshifts:
            sPloc = sP.copy()
            sPloc.setRedshift(redshift)
            sPs.append( sPloc )

    xQuant = 'mstar_30pkpc_log'
    yQuant = 'mg2_lum'
    cenSatSelect = 'cen'

    scatterColor = 'redshift'
    clim = [0.3, 2.2]
    xlim = [8.0, 11.5]
    ylim = [37, 43]
    scatterPoints = True
    drawMedian = False
    markersize = 30.0
    alpha = 0.7
    maxPointsPerDex = 500
    colorbarInside = True
    sizefac = 0.8 # for single column figure

    def _draw_data(ax):
        """ Draw data constraints on figure. """
        # Johannes Zahl+ (in prep, MUSE UDF) (BlueMUSE slides)
        mstar = 10.0 # log msun
        mstar_err = 0.1 # assumed

        lum = np.log10(9e40) # erg/s
        lum_err = 0.2 # assumed

        ax.errorbar(mstar, lum, xerr=mstar_err, yerr=lum_err, color='#000000', marker='D', label='Zabl+21 (MEGAFLOW)')

    quantMedianVsSecondQuant(sPs, pdf=None, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, alpha=alpha, maxPointsPerDex=maxPointsPerDex, 
                             colorbarInside=colorbarInside, legendLoc='upper left')

def paperPlots():
    """ Plots for the TNG50 MgII CGM emission paper. """
    from util.simParams import simParams

    redshifts = [0.3, 0.7, 1.0, 2.0]

    sP = simParams(run='tng50-1', redshift=0.7) # default unless changed

    # example: TNG50-1 z=0.7 np.where( (mstar_30pkpc>10.0) & (mstar_30pkpc<10.1) & cen_flag )[1]
    subhaloID = 396565

    # figure 1 - single halo example
    if 0:
        singleHaloImageMGII(sP, subhaloID, conf=1)

    # figure 2 - vis gallery
    if 0:
        visMg2Gallery(sP, mStarBin=[10.0, 10.1], num=20, size=70)
        #visMg2Gallery(sP, mStarBin=[10.0, 10.1], num=72, colorbar=False)

    # figure 3 - stacked SB profiles
    if 0:
        mStarBins = [ [9.0, 9.05], [9.5, 9.6], [10.0,10.1], [10.4,10.45] ]
        for redshift in redshifts:
            sP.setRedshift(redshift)
            radialSBProfiles(sP, mStarBins, minRedshift=redshifts[0], xlim=[0,50], psf=True)

    # figure 4 - individual SB profiles
    if 0:
        mStarBins = [ [10.0,10.1] ]
        radialSBProfiles(sP, mStarBins, indiv=70, xlim=[0,30], psf=False)

    # figure 5 - L_MgII vs M*, multiple redshifts overplotted
    if 0:
        mg2lum_vs_mass(sP, redshifts)

    # figure 6 - L_MgII correlation with SFR
    if 0:
        quantMedianVsSecondQuant([sP], pdf=None, yQuants=['ssfr'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                 xlim=[8.0,11.5], ylim=[-2.6,0.6], clim=[37,42], drawMedian=False, markersize=30,
                                 scatterPoints=True, scatterColor='mg2_lum', sizefac=0.8, 
                                 alpha=0.7, maxPointsPerDex=None, colorbarInside=False)
        quantMedianVsSecondQuant([sP], pdf=None, yQuants=['ssfr'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                 xlim=[8.0,11.5], ylim=[-2.6,0.6], clim=[1,10], drawMedian=False, markersize=30,
                                 scatterPoints=True, scatterColor='mg2_lumsize', sizefac=0.8, 
                                 alpha=0.7, maxPointsPerDex=None, colorbarInside=False)
        #quantHisto2D(sP, pdf=None, yQuant='ssfr', xQuant='mstar_30pkpc_log', cenSatSelect='cen', cQuant='mg2_lumsize', 
        #             xlim=[8.0,11.5], ylim=[-2.6,0.6], minCount=None, cRel=[0.5,2.0,False], nBins=50, 
        #             filterFlag=True, medianLine=True, sizeFac=0.8, ctName=None)

    # figure X - impact of dust depletion
    if 0:
        visDustDepletionImpact(sP, subhaloID)

    # todo: psf vs no psf
    # todo: correlation with vrad: inflow vs outflow
    # todo: lumsize vs mstar, compare with other sizes
    # todo: resolution convergence
