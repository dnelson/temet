"""
projects/mg2emission.py
  TNG50: MgII CGM emission paper.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

from vis.halo import renderSingleHalo
from util.helper import running_median
from plot.cosmoGeneral import quantMedianVsSecondQuant
from plot.config import *
from projects.azimuthalAngleCGM import _get_dist_theta_grid

def singleHaloImageMGII(sP, hInd, conf=0):
    """ MgII emission image test.. """
    rVirFracs  = [0.25]
    method     = 'sphMap'
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
        fontsize     = 24
        saveFilename = '%s_%d_%d_%s.pdf' % (sP.simName,sP.snap,hInd,'-'.join([p['partField'] for p in panels]))

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def visMg2Gallery(sP, mStarBin, num, size=100, colorbar=True):
    """ MgII emission: image gallery of example stamps in a given stellar mass bin. """
    rVirFracs  = [5.0]
    fracsType  = 'rHalfMassStars'
    method     = 'sphMap'
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

def radialSBProfiles(sP, massBins):
    """ Use grids to produce individual and stacked radial surface brightness profiles. """
    method     = 'sphMap'
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
    percs     = [38,50,62] # 50 must be in the middle

    # load catalog
    dist, _ = _get_dist_theta_grid(size, nPixels)

    mstar = sP.subhalos('mstar_30pkpc_log')
    cen_flag = sP.subhalos('central_flag')

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
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_ylabel('MgII Emission [log erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
    ax.set_xlim(radMinMax)

    # loop over halos in each mass bin
    for i, massBin in enumerate(massBins):

        with np.errstate(invalid='ignore'):
            subInds = np.where( (mstar>massBin[0]) & (mstar<massBin[1]) & cen_flag )[0]

        print('[%.2f - %.2f] Processing [%d] halos...' % (massBin[0],massBin[1],len(subInds)))

        # check for existence of cache
        cacheFile = sP.derivPath + 'cache/mg2emission_sbr_%d_%d_%s_%d.hdf5' % (massBin[0]*10,massBin[1]*10,rotation,size)

        if isfile(cacheFile):
            # load cached result
            with h5py.File(cacheFile,'r') as f:
                sbr_indiv_mean = f['sbr_indiv_mean'][()]
                sbr_indiv_med = f['sbr_indiv_med'][()]
                sbr_stack_mean = f['sbr_stack_mean'][()]
                sbr_stack_med = f['sbr_stack_med'][()]
                #grid_global = f['grid_global'][()]
            print('Loaded: [%s]' % cacheFile)
        else:
            # compute now
            sbr_indiv_mean = np.zeros( (len(subInds), nRadBins), dtype='float32' )
            sbr_indiv_med = np.zeros( (len(subInds), nRadBins, len(percs)), dtype='float32' )
            grid_global = np.zeros( (len(subInds), nPixels[0], nPixels[1]), dtype='float32' )

            sbr_indiv_mean.fill(np.nan)
            sbr_indiv_med.fill(np.nan)

            for i, hInd in enumerate(subInds):
                class plotConfig:
                    saveFilename = 'dummy'

                grid, _ = renderSingleHalo([{}], plotConfig, locals(), returnData=True)

                grid_global[i,:,:] = grid
                sbr_indiv_mean[i,:], sbr_indiv_med[i,:,:] = _rad_profile(dist.ravel(), grid.ravel())

            # create dist in same shape as grid_global
            dist_global = np.zeros( (len(subInds), nPixels[0]*nPixels[1]), dtype='float32' )

            for i, hInd in enumerate(subInds):
                dist_global[i,:] = dist.ravel()

            # compute profile on 'stacked images'
            sbr_stack_mean, sbr_stack_med = _rad_profile(dist_global.ravel(), grid_global.ravel())

            # save cache
            with h5py.File(cacheFile,'w') as f:
                f['grid_global'] = grid_global
                f['sbr_indiv_mean'] = sbr_indiv_mean
                f['sbr_indiv_med'] = sbr_indiv_med
                f['sbr_stack_mean'] = sbr_stack_mean
                f['sbr_stack_med'] = sbr_stack_med

            print('Saved: [%s]' % cacheFile)

        # compute stack of 'per halo profiles' and plot
        label = 'M$_{\star}$ = %.1f' % (massBin[0])

        yy = np.nanpercentile(sbr_indiv_med[:,:,1], percs, axis=0)
        l, = ax.plot(radMidPts, yy[1,:], lw=lw, label=label + ' (prof then stack, med)')
        ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.1)

        yy = np.nanpercentile(sbr_indiv_mean, percs, axis=0)
        l, = ax.plot(radMidPts, yy[1,:], lw=lw, label=label + ' (prof then stack, mean)')
        ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.1)

        # plot 'image stack' and shaded band
        l, = ax.plot(radMidPts, sbr_stack_med[:,1], lw=lw, label=label + ' (stack then prof, median)')
        ax.fill_between(radMidPts, sbr_stack_med[:,0], sbr_stack_med[:,2], color=l.get_color(), alpha=0.1)

        l, = ax.plot(radMidPts, sbr_stack_mean, lw=lw, label=label + ' (stack then prof, mean)')

    # finish and save plot
    ax.legend(loc='best')
    fig.savefig('SB_profiles_%s_%d_%s.pdf' % (sP.simName,sP.snap,partField))
    plt.close(fig)

def paperPlots():
    """ Plots for the TNG50 MgII CGM emission paper. """
    from util.simParams import simParams

    redshifts = [0.7, 1.0, 1.5, 2.0]

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

    # figure 3 - individual/stacked SB profiles
    if 1:
        mStarBins = [ [10.0,10.1] ]
        radialSBProfiles(sP, mStarBins)

    # figure X - impact of dust depletion
    if 0:
        visDustDepletionImpact(sP, subhaloID)

    # todo: L_MgII vs M*
    # todo: correlation with SFR
    # todo: correlation with vrad: inflow vs outflow
