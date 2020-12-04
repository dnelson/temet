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
from util.helper import running_median, sampleColorTable, loadColorTable, logZeroNaN
from plot.cosmoGeneral import quantMedianVsSecondQuant, quantHisto2D
from plot.general import plotParticleMedianVsSecondQuant, plotPhaseSpace2D
from plot.config import *
from projects.azimuthalAngleCGM import _get_dist_theta_grid

def singleHaloImageMGII(sP, hInd, conf=1, size=100, rotation='edge-on', labelCustom=None,
                        rVirFracs=[0.25], fracsType='rVirial', font=16, cbars=True):
    """ MgII emission image test.. """
    method     = 'sphMap' # note: fof-scope
    nPixels    = [800,800]
    axes       = [0,1]
    labelZ     = True
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    sizeType   = 'kpc'

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
    if conf == 5:
        panels.append( {'partType':'gas', 'partField':'tau0_MgII2796', 'valMinMax':[-2.0,2.0], 'ctName':'tarn0'} )
    if conf == 6:
        # equirectangular
        contour = None
        rVirFracs = False
        panels.append( {'partType':'gas', 'partField':'tau0_MgII2796', 'valMinMax':[-2.0,2.0], 'ctName':'tarn0'} )
        #panels.append( {'partType':'gas', 'partField':'tau0_LyA', 'valMinMax':[4.0,8.0]} )

        projType   = 'equirectangular'
        projParams = {'fov':360.0}
        nPixels    = [1200,600]
        axesUnits  = 'rad_pi'

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels
        colorbars    = cbars
        fontsize     = font
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

def radialSBProfiles(sPs, massBins, minRedshift=None, psf=False, indiv=False, xlim=None, ylim=None):
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

    # MUSE UDF has a PSF FWMH ~ 0.7 arcsec
    pxScale = size / nPixels[0] # pkpc/px
    psfFWHM_px = sPs[0].units.arcsecToAngSizeKpcAtRedshift(0.7) / pxScale
    psfSigma_px = psfFWHM_px / 2.3548

    def _cachefile(sP):
        return sP.derivPath + 'cache/mg2emission_%d_sbr_%d_%d_%s_%d.hdf5' % (sP.snap,massBin[0]*10,massBin[1]*10,rotation,size)

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
    fig = plt.figure(figsize=figsize_sm)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_ylabel('MgII SB [log erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
    ax.set_xlim(radMinMax if xlim is None else xlim)
    ax.set_ylim([-22.0,-16.5] if ylim is None else ylim)

    # loop over runs
    for k, sP in enumerate(sPs):
        # load catalog
        dist, _ = _get_dist_theta_grid(size, nPixels)

        mstar = sP.subhalos('mstar_30pkpc_log')
        cen_flag = sP.subhalos('central_flag')

        # not at lowest redshift? add low redshift line for comparison
        if not indiv and minRedshift is not None and sP.redshift > minRedshift+0.1:
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
            colorQuant = 'sfr' #'mstar'
            sfr = logZeroNaN(sP.subhalos('SubhaloSFRinRad'))
            #cmap = loadColorTable('terrain', fracSubset=[0.0,0.8])
            cmap = loadColorTable('turbo', fracSubset=[0.1,1.0])
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
            if psf == True:
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

                if colorQuant == 'mstar':
                    norm = Normalize(vmin=massBin[0], vmax=massBin[1]) # color on mstar
                if colorQuant == 'sfr':
                    norm = Normalize(vmin=-0.2, vmax=0.8)
                    #norm = Normalize(vmin=np.percentile(sfr[subInds],5), vmax=np.percentile(sfr[subInds],95))

                # plot individual profiles
                for j in range( np.min([indiv,sbr_indiv_med.shape[0]]) ):
                    yy = sbr_indiv_med[j,:,1]
                    if 0:
                        c = colors[j] # sampled uniformly from colormap in order of subhalo IDs
                    if 1:
                        if colorQuant == 'mstar': c = cmap(norm(mstar[subInds[j]]))
                        if colorQuant == 'sfr': c = cmap(norm(sfr[subInds[j]]))

                    ax.plot(radMidPts, savgol_filter(yy,sKn,sKo,axis=0), '-', color=c)
            else:
                # compute stack of 'per halo profiles' and plot
                label = 'M$_{\star}$ = %.1f' % (massBin[0])
                if psf == 'both': label += ' (no PSF)'

                if psf != 'both' and len(sPs) == 1:
                    yy = np.nanpercentile(sbr_indiv_med[:,:,1], percs, axis=0)
                    l, = ax.plot(radMidPts, yy[1,:], linestyle=':', lw=lw, color=colors[i])
                    #ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=colors[i], alpha=0.15)

                if len(sPs) == 1:
                    # single run
                    yy = np.nanpercentile(sbr_indiv_mean, percs, axis=0)
                    l, = ax.plot(radMidPts, yy[1,:], lw=lw, linestyle='-', color=colors[i], label=label)
                    ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.15)
                else:
                    # multiple runs (resolution convergence plot)
                    yy = np.nanpercentile(sbr_indiv_mean, percs, axis=0)
                    if k > 0: label = ''
                    l, = ax.plot(radMidPts, yy[1,:], lw=lw, linestyle=linestyles[k], color=colors[i], label=label)
                    #ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.15)

                if psf == 'both':
                    label = 'M$_{\star}$ = %.1f (w/ PSF)' % (massBin[0])
                    yy = np.nanpercentile(sbr_indiv_mean_psf, percs, axis=0)
                    l, = ax.plot(radMidPts, yy[1,:], lw=lw, linestyle=':', color=colors[i], label=label)
                    ax.fill_between(radMidPts, yy[0,:], yy[2,:], color=l.get_color(), alpha=0.15)

                # plot 'image stack' and shaded band
                #l, = ax.plot(radMidPts, sbr_stack_med[:,1], lw=lw, label=label + ' (stack then prof, median)')
                #ax.fill_between(radMidPts, sbr_stack_med[:,0], sbr_stack_med[:,2], color=l.get_color(), alpha=0.1)

                #l, = ax.plot(radMidPts, sbr_stack_mean, lw=lw, label=label + ' (stack then prof, mean)')

    # observational data
    if sP.redshift == 0.7 and not indiv:
        # (Rickards Vaught+ 2019)
        dist = 14.5 # kpc
        dist_err = 6.5 # kpc

        sb_lim = np.log10(6.51e-19) # erg/s/cm^2/arcsec^2 (5sigma upper limit)
        #mstar_gals = [9.9, 11.0] # log msun, 5 galaxies, 4<SFR<40 msun/yr

        ax.errorbar(dist, sb_lim, xerr=dist_err, yerr=0.4, uplims=True, color='#000000', marker='D', alpha=0.6)
        ax.text(dist, sb_lim+0.15, 'RV+2019', ha='center', fontsize=20)

        # Burchett+2020
        dist     = np.array([0.33, 1.03, 1.73, 2.44, 3.15, 3.85, 4.55]) # arcsec
        dist_kpc = sP.units.arcsecToAngSizeKpcAtRedshift(dist)
        sb       = np.log10([1.70e-17, 1.01e-17, 4.36e-18, 2.10e-18, 9.48e-19, 5.26e-19, 1.03e-19]) # erg/s/cm^2/arcsec^2
        sb_up    = np.log10([1.70e-17, 1.01e-17, 4.36e-18, 2.10e-18, 1.04e-18, 6.26e-19, 1.89e-19]) - sb
        sb_down  = sb - np.log10([1.70e-17, 1.01e-17, 4.36e-18, 2.10e-18, 8.60e-19, 4.53e-19, 4.99e-20])

        ax.errorbar(dist_kpc, sb, yerr=[sb_down,sb_up], color='#000000', marker='s', alpha=0.6)
        if indiv:
            yy = sb[2] - 0.5
        else:
            yy = sb[2] + 0.2
        ax.text(dist_kpc[2], sb[2]+0.2, 'Burchett+2020', ha='left', fontsize=20)

    if sP.redshift == 0.3 and not indiv:
        # Rupke+ 2019 Makani
        from load.data import rupke19
        r19 = rupke19()

        ax.errorbar(r19['rad_kpc'], r19['sb'], yerr=[r19['sb_down'],r19['sb_up']], color='#000000', marker='s', alpha=0.6)
        ax.text(r19['rad_kpc'][14], r19['sb'][14]+0.3, r19['label'], ha='left', fontsize=20)

    # finish and save plot
    if len(sPs) > 1:
        handles = []
        labels = []

        for k, sP in enumerate(sPs):
            handles += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[k],marker='')]
            labels += [sP.simName]

        legend2 = ax.legend(handles, labels, loc='lower left')
        ax.add_artist(legend2)

    if indiv:
        fig.tight_layout()
        fig.set_tight_layout(False)
        cax = fig.add_axes([0.5,0.84,0.4,0.04])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        if colorQuant == 'mstar': ctitle = 'Galaxy Stellar Mass [log M$_{\\rm sun}$]'
        if colorQuant == 'sfr': ctitle = 'Galaxy SFR [log M$_{\\rm sun}$ yr$^{-1}$]'
        cb.ax.set_title(ctitle)
    else:
        if psf != 'both' and len(sPs) == 1:
            xx = ax.get_xlim()[1]-0.22*(ax.get_xlim()[1]-ax.get_xlim()[0])
            yy = -18.9
            ax.text(xx, yy, 'z = %.1f' % sP.redshift, fontsize=24)
        ax.legend(loc='best')

    fig.savefig('SB_profiles_%s_%d_%s%s%s%s.pdf' % \
        (sP.simName,sP.snap,partField,'_indiv' if indiv else '','_psf=%s' % psf,
            '_sP%d' % len(sPs) if len(sPs)>1 else ''))
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
    ylim = [37, 42]
    scatterPoints = True
    drawMedian = False
    markersize = 30.0
    alpha = 0.7
    maxPointsPerDex = 500
    colorbarInside = True

    def _draw_data(ax):
        """ Draw data constraints on figure. """
        # Rubin+ (2011), z=0.7, TKRS4389, SFR=50 or 60 msun/yr (same object as Burchett+2020)
        mstar = 9.9
        mstar_err = 0.1 # assumed
        flux = 1.920e-17 + 2.103e-17 # 2796+2803, Table 1, [erg/s/cm^2]
        lum = np.log10(sP.units.fluxToLuminosity(flux)) # log(erg/s)
        lum_err = 0.4 # assumed

        label = 'Rubin+11 (z=0.7 LRIS/TKRS4389)'
        ax.errorbar(mstar, lum, xerr=mstar_err, yerr=lum_err, color='#000000', marker='H', label=label)

        # Martin+ (2013) z=1.0, 32016857, SFR=80 msun/yr
        mstar = 9.8
        mstar_err = 0.15 # factor of two uncertainty
        flux = 1.5e-17 + 1.0e-17 # 2796+2803 [erg/s/cm^2]
        lum = np.log10(sP.units.fluxToLuminosity(flux)) # log(erg/s)
        lum_err = 0.4 # assumed

        label = 'Martin+13 (z=1 LRIS/32016857)'
        ax.errorbar(mstar, lum, xerr=mstar_err, yerr=lum_err, color='#000000', marker='s', label=label)

        # Johannes Zahl+ (in prep, MUSE UDF) (BlueMUSE slides)
        mstar = 10.0 # log msun
        mstar_err = 0.1 # assumed

        lum = np.log10(9e40) # erg/s
        lum_err = 0.2 # assumed

        label = 'Zabl+21 (z=0.7 MEGAFLOW)'
        ax.errorbar(mstar, lum, xerr=mstar_err, yerr=lum_err, color='#000000', marker='D', label=label)

        # Rupke+ (2019) (KCWI 'Makani', OII nebula)
        redshift = 0.459
        mstar = 11.1 # log msun
        mstar_err = 0.2 # assumed

        flux = 4.5e-16 # erg/s/cm^2 (David's email, "closer to 6-7e16 than 2-3e16")
        lum = np.log10(sP.units.fluxToLuminosity(flux, redshift=redshift))
        lum_err = 0.2 # assumed

        label = 'Rupke+19 (z=0.5 KCWI/Makani)'
        ax.errorbar(mstar, lum, xerr=mstar_err, yerr=lum_err, color='#000000', marker='p', label=label)

    quantMedianVsSecondQuant(sPs, pdf=None, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, alpha=alpha, maxPointsPerDex=maxPointsPerDex, 
                             colorbarInside=colorbarInside, legendLoc='lower right')

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
            radialSBProfiles([sP], mStarBins, minRedshift=redshifts[0], xlim=[0,50], psf=True)

    # figure 4 - individual SB profiles
    if 0:
        mStarBins = [ [10.0,10.1] ]
        radialSBProfiles([sP], mStarBins, indiv=70, xlim=[0,30], psf=True)

    # figure 5 - L_MgII vs M*, multiple redshifts overplotted
    if 0:
        mg2lum_vs_mass(sP, redshifts)

    # figure 6 - L_MgII and r_1/2,MgII correlation with SFR
    if 0:
        quantMedianVsSecondQuant([sP], pdf=None, yQuants=['ssfr'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                 xlim=[8.0,11.5], ylim=[-2.6,0.6], clim=[38,41], drawMedian=False, markersize=40,
                                 scatterPoints=True, scatterColor='mg2_lum', sizefac=sizefac, cbarticks=[38,39,40,41],
                                 alpha=0.7, maxPointsPerDex=1000, colorbarInside=False, lowessSmooth=True)
        quantMedianVsSecondQuant([sP], pdf=None, yQuants=['ssfr'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                 xlim=[8.0,11.5], ylim=[-2.6,0.6], clim=[2,10], drawMedian=False, markersize=40,
                                 scatterPoints=True, scatterColor='mg2_lumsize', sizefac=sizefac, 
                                 alpha=0.7, maxPointsPerDex=1000, colorbarInside=False, lowessSmooth=True)

    # figure 7 - lumsize vs mstar, compare with other sizes
    if 0:
        # find three particular subhalos of interest on this plot
        mg2_lumsize,_,_,_ = sP.simSubhaloQuantity('mg2_lumsize')
        subIDs = [435554,445695,460949] # high, medium, low

        if 0:
            xx = sP.subhalos('mstar_30pkpc_log')
            cen_flag = sP.subhalos('cen_flag')
            ww = np.where( (xx>9.8) & (xx < 9.9) & (np.abs(mg2_lumsize-17.5) < 0.2) & cen_flag ) # 2.4,6.0,17.5

        quantMedianVsSecondQuant([sP], pdf=None, yQuants=['mg2_lumsize'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                 xlim=[8.0,11.5], ylim=[0,20], clim=[37,42], drawMedian=True, medianLabel='r$_{\\rm 1/2,L(MgII)}$', 
                                 markersize=30, scatterPoints=True, scatterColor='mg2_lum', sizefac=sizefac, markSubhaloIDs=subIDs,
                                 extraMedians=['size_stars','size_gas'], alpha=0.7, maxPointsPerDex=None, colorbarInside=False)

        for subID in subIDs:
            subh = sP.subhalo(subID)
            label = 'SFR = %.1f M$_{\\odot}$ yr$^{-1}$' % subh['SubhaloSFR']
            label += '\nr$_{\\rm 1/2,L(MgII)}$ = %.1f kpc' % mg2_lumsize[subID]
            label += '\nID #%d' % subID
            singleHaloImageMGII(sP, subID, conf=1, size=70, rotation=None, labelCustom=[label],
                                rVirFracs=[2*mg2_lumsize[subID]], fracsType='kpc', font=26, cbars=False)

    # figure 8 - L_MgII and r_1/2,MgII correlation with SFR
    if 0:
        for cQuant in ['mg2_lum','mg2_lumsize']:
            quantMedianVsSecondQuant([sP], pdf=None, yQuants=['sfr2_surfdens'], xQuant='mstar_30pkpc_log', cenSatSelect='cen', 
                                     xlim=[8.0,11.5], ylim=[-4.2,-0.4], clim=[37,41.5], drawMedian=False, markersize=40,
                                     scatterPoints=True, scatterColor=cQuant, sizefac=sizefac, cRel=[-0.2,0.2,True], #[0.5,1.5,False],
                                     alpha=0.7, maxPointsPerDex=1000, colorbarInside=False, lowessSmooth=True)

    # figure 9 - relation of MgII flux and vrad (inflow vs outflow)
    if 0:
        SubhaloGrNr = sP.subhalos('SubhaloGrNr')
        mstar = sP.subhalos('mstar_30pkpc_log')
        cen_flag = sP.subhalos('cen_flag')

        with np.errstate(invalid='ignore'):
            subIDs = np.where( (mstar>9.99) & (mstar < 10.01) & cen_flag )
            haloIDs = SubhaloGrNr[subIDs]

        def _f_post(ax):
            ax.text(0.97, 0.97, 'M$_{\\star} = 10^{10}\,$M$_{\odot}$', transform=ax.transAxes, 
                    color='#ffffff', fontsize=22, ha='right', va='top')

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', yQuant='vrad', weights=['MgII lum_dustdepleted'], meancolors=None, 
                         xlim=[0, 30], ylim=[-160,160], clim=[-5,-3], contours=None, contourQuant=None, normColMax=False, hideBelow=False, 
                         ctName='thermal', colorEmpty=True, smoothSigma=0.0, nBins=100, qRestrictions=None, median=False, 
                         normContourQuantColMax=False, haloIDs=haloIDs, f_post=_f_post, addHistY=50)

    # TODO: What is the contribution of winds (vrad>0) vs ambient CGM to MgII lum (as a function of radius)

    # TODO: impact of orientation (face-on vs edge-on) on MgII lum/morphology

    # TODO: morphology in general: isotropic or not? shape a/b axis ratio of different isophotes?

    # figure X - line-center optical depth for MgII all-sky map, show low values
    if 0:
        singleHaloImageMGII(sP, subhaloID, conf=6)

    # figure A1 - impact of dust depletion
    if 0:
        visDustDepletionImpact(sP, subhaloID)

    # figure A2 - psf vs no psf
    if 0:
        mStarBins = [ [9.0, 9.05], [9.5, 9.6], [10.0,10.1] ]
        radialSBProfiles([sP], mStarBins, xlim=[0,50], ylim=[-23.0,-16.5], psf='both')

    # figure A3 - resolution convergence
    if 0:
        runs = ['tng50-1','tng50-2','tng50-3']
        mStarBins = [ [9.0, 9.05], [9.5, 9.6], [10.0,10.1] ]
        sPs = [simParams(run=run, redshift=0.7) for run in runs]
        radialSBProfiles(sPs, mStarBins, xlim=[0,50], ylim=[-23.0,-16.5])
