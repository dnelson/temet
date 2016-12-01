"""
galaxyColor.py
  Plots for TNG flagship paper: galaxy colors, color bimodality.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, colorConverter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d, gaussian_kde

from util import simParams
from util.helper import loadColorTable, running_median, contourf, logZeroSafe, logZeroNaN
from cosmo.stellarPop import stellarPhotToSDSSColor, calcSDSSColors, loadSimGalColors
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, groupCatSingle
from plot.general import simSubhaloQuantity

# global configuration (remove duplication with globalComp.py)
sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines
figsize = (14,10) # (8,6)
clean   = False  # make visually clean plots with less information

linestyles = ['-',':','--','-.']       # typically for analysis variations per run

def _bandMagRange(bands, tight=False):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'u' and bands[1] == 'r': mag_range = [0.5,3.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    if bands[0] == 'r' and bands[1] == 'z': mag_range = [0.0,0.8]

    if tight:
        # alternative set
        if bands == ['u','i']: mag_range = [0.5,4.0]
        if bands == ['u','i']: mag_range = [0.5,3.5]
        if bands == ['g','r']: mag_range = [0.15,0.85]
        if bands == ['r','i']: mag_range = [0.0,0.6]
        if bands == ['i','z']: mag_range = [0.0,0.4]
        if bands == ['i','z']: mag_range = [0.0,0.8]
    return mag_range

def calcMstarColor2dKDE(bands, gal_Mstar, gal_color, Mstar_range, mag_range, sP=None, simColorsModel=None):
    """ Quick caching of (slow) 2D KDE calculation of (Mstar,color) plane for SDSS z<0.1 points 
    if sP is None, otherwise for simulation (Mstar,color) points if sP is specified. """
    from os.path import isfile, isdir, expanduser
    from os import mkdir

    if sP is None:
        saveFilename = expanduser("~") + "/obs/sdss_2dkde_%s_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10)
        dName = 'kde_obs'
    else:
        assert simColorsModel is not None
        savePath = sP.derivPath + "/galMstarColor/"

        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + "galMstarColor_2dkde_%s_%s_%d_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),simColorsModel,sP.snap,
            Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10)
        dName = 'kde_sim'

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            xx = f['xx'][()]
            yy = f['yy'][()]
            kde_obs = f[dName][()]

        return xx, yy, kde_obs

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    vv = np.vstack( [gal_Mstar, gal_color] )
    kde = gaussian_kde(vv)

    xx, yy = np.mgrid[Mstar_range[0]:Mstar_range[1]:200j, mag_range[0]:mag_range[1]:400j]
    xy = np.vstack( [xx.ravel(), yy.ravel()] )
    kde2d = np.reshape( np.transpose(kde(xy)), xx.shape)

    # save
    with h5py.File(saveFilename,'w') as f:
        f['xx'] = xx
        f['yy'] = yy
        f[dName] = kde2d
    print('Saved: [%s]' % saveFilename)

    return xx, yy, kde2d

def histo2D(sP, pdf, bands, xQuant='mstar2_log', cenSatSelect='cen', cQuant=None, cStatistic=None, 
            minCount=None, simColorsModel='p07c_cf00dust'):
    """ Make a 2D histogram of subhalos with some color on the y-axis, a property on the x-axis, 
    and optionally a third property as the colormap per bin. minCount specifies the minimum number of 
    points a bin must contain to show it as non-white. If '_nan' is not in cStatistic, then by default, 
    empty bins are white, and bins whose cStatistic is NaN (e.g. any NaNs in bin) are gray. Or, if 
    '_nan' is in cStatistic, then empty bins remain white, while the cStatistic for bins with any 
    non-NaN values is computed ignoring NaNs (e.g. np.nanmean() instead of np.mean()), and bins 
    which are non-empty but contain only NaN values are gray. """
    assert cenSatSelect in ['all', 'cen', 'sat']
    assert cStatistic in [None,'mean','median','count','sum','median_nan'] # or any user function
    assert simColorsModel in ['snap','p07c_nodust','p07c_bc00dust','p07c_cf00dust']

    # hard-coded config
    nBins = 80

    cmap = loadColorTable('viridis') # plasma
    mag_range = _bandMagRange(bands, tight=True)

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    if 'sim_xvals' in sP.data:
        sim_xvals, xlabel, xMinMax = sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax']
    else:
        sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant, clean)
        if xMinMax[0] > xMinMax[1]: xMinMax = xMinMax[::-1] # reverse
        sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax'] = sim_xvals, xlabel, xMinMax

    # y-axis: load/calculate simulation colors, cached in sP.data
    if 'sim_colors' in sP.data:
        sim_colors = sP.data['sim_colors']
    else:
        sim_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands)
        sP.data['sim_colors'] = sim_colors

    ylabel = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
    if not clean: ylabel += ' %s' % simColorsModel

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )
        assert cStatistic in [None,'count']

        # overrides for density distribution
        cStatistic = 'count'
        cmap = loadColorTable('gray_r')

        clabel = 'log N$_{\\rm gal}$'
        cMinMax = [0.0,2.0]
        if sP.boxSize > 100000: cMinMax = [0.0,2.5]
    else:
        sim_cvals, clabel, cMinMax, cLog = simSubhaloQuantity(sP, cQuant, clean)

    # central/satellite selection?
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_colors = sim_colors[wSelect]

    # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
    wFiniteColor = np.isfinite(sim_colors) #& np.isfinite(sim_xvals)
    sim_colors = sim_colors[wFiniteColor]
    sim_cvals  = sim_cvals[wFiniteColor]
    sim_xvals  = sim_xvals[wFiniteColor]

    # _nan cStatistic? separate points into two sets
    nanFlag = False
    if '_nan' in cStatistic:
        nanFlag = True

        wFiniteCval = np.isfinite(sim_cvals)
        wNaNCval = np.isnan(sim_cvals)

        assert np.count_nonzero(wFiniteCval) + np.count_nonzero(wNaNCval) == sim_cvals.size

        # save points with NaN cvals
        sim_colors_nan = sim_colors[wNaNCval]
        sim_cvals_nan  = sim_cvals[wNaNCval]
        sim_xvals_nan  = sim_xvals[wNaNCval]

        # override default binning to only points with finite cvals
        sim_colors = sim_colors[wFiniteCval]
        sim_cvals  = sim_cvals[wFiniteCval]
        sim_xvals  = sim_xvals[wFiniteCval]

        # replace cStatistic string
        cStatistic = cStatistic.split("_nan")[0]

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not clean:
        print(' ','-'.join(bands),simColorsModel,xQuant,cenSatSelect,cQuant,cStatistic,minCount)
        ax.set_title('stat=%s select=%s mincount=%s' % (cStatistic,cenSatSelect,minCount))

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    if '_log' not in xQuant: sim_xvals = logZeroSafe(sim_xvals) # xMinMax always corresponds to log values

    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, cStatistic, 
                                                 bins=nBins2D, range=[xMinMax,mag_range])

    cc = cc.T # imshow convention

    # only show bins with a minimum number of points?
    nn, _, _, _ = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, 'count', 
                                      bins=nBins2D, range=[xMinMax,mag_range])
    nn = nn.T

    if minCount is not None:
        cc[nn < minCount] = np.nan

    # for now: log on density and all color quantities
    cc2d = cc
    if cQuant is None or cLog is True:
        cc2d = logZeroNaN(cc)

    # normalize and color map
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap(norm(cc2d))

    # mask bins with median==0 and map to special color, which right now have been set to log10(0)=NaN
    if cQuant is not None:
        cc2d_rgb[cc == 0.0,:] = colorConverter.to_rgba('#eeeeee')

    if nanFlag:
        # bin NaN point set counts
        nn_nan, _, _, _ = binned_statistic_2d(sim_xvals_nan, sim_colors_nan, sim_cvals_nan, 'count', 
                                              bins=nBins2D, range=[xMinMax,mag_range])
        nn_nan = nn_nan.T        

        # flag bins with nn_nan>0 and nn==0 (only NaNs in bin) as second gray color
        cc2d_rgb[ ((nn_nan > 0) & (nn == 0)), :] = colorConverter.to_rgba('#dddddd')

        nn += nn_nan # accumulate total counts
    else:
        # mask bins with median==NaN (nonzero number of NaNs in bin) to gray
        cc2d_rgb[~np.isfinite(cc),:] = colorConverter.to_rgba('#dddddd')

    # mask empty bins to whity
    cc2d_rgb[(nn == 0),:] = colorConverter.to_rgba('#ffffff')

    # plot
    plt.imshow(cc2d_rgb, extent=extent, origin='lower', interpolation='nearest', aspect='auto', 
               cmap=cmap, norm=norm)

    # method (B) unused
    #reduceMap = {'mean':np.mean, 'median':np.median, 'count':np.size, 'sum':np.sum}
    #reduceFunc = reduceMap[cStatistic] if cStatistic in reduceMap else cStatistic
    #plt.hexbin(sim_xvals, sim_colors, C=None, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False)
    #plt.hexbin(sim_xvals, sim_colors, C=sim_cvals, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False, reduce_C_function=reduceFunc)

    # median line?
    if cQuant is None:
        binSizeMed = (xMinMax[1]-xMinMax[0]) / nBins * 2
        colorMed = 'orange'
        lwMed = 2.0

        xm, ym, sm, pm = running_median(sim_xvals,sim_colors,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        ym2 = savgol_filter(ym,sKn,sKo)
        sm2 = savgol_filter(sm,sKn,sKo)
        pm2 = savgol_filter(pm,sKn,sKo,axis=1)

        ax.plot(xm[:-1], ym2[:-1], '-', color=colorMed, lw=lwMed, label='median')

        ax.plot(xm[:-1], pm2[1,:-1], ':', color=colorMed, lw=lwMed, label='P[10,90]')
        ax.plot(xm[:-1], pm2[-2,:-1], ':', color=colorMed, lw=lwMed)

        if not clean:
            #ax.plot(xm[:-1], pm2[0,:-1], ':', color='red', lw=lwMed, label='P[5,95]')
            #ax.plot(xm[:-1], pm2[-1,:-1], ':', color='red', lw=lwMed)
            #ax.plot(xm[:-1], pm2[2,:-1], ':', color='green', lw=lwMed, label='IQR[25,75]')
            #ax.plot(xm[:-1], pm2[-3,:-1], ':', color='green', lw=lwMed)
            #ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='yellow', lw=lwMed, label='$\pm 1\sigma$')
            #ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='yellow', lw=lwMed)
            ax.legend(loc='lower right')

    # colorbar
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)

    # finish plot and save
    fig.tight_layout()
    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig('histo2d_%s_%s_%s_%s_%s_%s_%s.pdf' % \
            ('-'.join(bands),simColorsModel,xQuant,cQuant,cStatistic,cenSatSelect,minCount))
    plt.close(fig)

def galaxyColorPDF(sPs, pdf, bands=['u','i'], simColorsModels=['p07c_cf00dust'], 
                   simRedshift=0.0, splitCenSat=False, cenOnly=False, stellarMassBins=None):
    """ PDF of galaxy colors (by default: (u-i)), with no dust corrections. (Vog 14b Fig 13) """
    from util import simParams

    if cenOnly: assert splitCenSat is False
    allOnly = True if (splitCenSat is False and cenOnly is False) else False
    assert not isinstance(simColorsModels,basestring) # should be iterable
    assert len(sPs) == 1 or len(simColorsModels) == 1
    
    # config
    if stellarMassBins is None:
        # default, 2 cols 3 rows
        stellarMassBins = ( [9.0,9.5],   [9.5,10.0],  [10.0,10.5], 
                            [10.5,11.0], [11.0,11.5], [11.5,12.0] )
    obs_color = '#333333'

    eCorrect = True # True, False
    kCorrect = True # True, False

    # start plot
    sizefac = 1.5
    if clean: sizefac = 1.3
    if len(stellarMassBins) >= 4: figsize = (16,9)
    else: figsize = (5.3, 13.5)

    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
    axes = []

    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.0,4.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [-0.2,1.2]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.6]

    # loop over each mass bin
    for i, stellarMassBin in enumerate(stellarMassBins):

        # panel setup
        iLeg = 2 # upper right (2x3), or bottom (3x1)
        if len(stellarMassBins) >= 4: # 2 rows, N columns
            ax = fig.add_subplot(2,len(stellarMassBins)/2,i+1)
        else: # N rows, 1 column
            ax = fig.add_subplot(len(stellarMassBins), 1, i+1)

        axes.append(ax)
        
        ax.set_xlim(mag_range)
        xlabel = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
        ylabel = 'PDF' 
        Mlabel = '%.1f < M$_{\\rm \star}$ < %.1f'

        if not clean:
            obsMagStr = 'obs=modelMag%s%s' % ('-E' if eCorrect else '','+K' if kCorrect else '')
            cenSatStr = '' if splitCenSat else ', cen+sat'
            Mlabel = '%.1f < M$_{\\rm \star}(<2r_{\star,1/2})$ < %.1f'

            xlabel += ' [ %s ]' % obsMagStr
            ylabel += ' [ sim=%s%s ]' % (simColorsModels[0],cenSatStr)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add stellar mass bin legend
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=0.0,marker='',linestyle=linestyles[0])]
        lExtra = [Mlabel % (stellarMassBin[0],stellarMassBin[1])]

        legend1 = ax.legend(sExtra, lExtra, loc='upper right')
        ax.add_artist(legend1)

    # load observational points, restrict colors to mag_range as done for sims (for correct normalization)
    sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=eCorrect, kCorrect=kCorrect)

    w = np.where( (sdss_color >= mag_range[0]) & (sdss_color <= mag_range[1]) )
    sdss_color = sdss_color[w]
    sdss_Mstar = sdss_Mstar[w]

    # loop over each fullbox run
    pMaxVals = np.zeros( len(stellarMassBins), dtype='float32' )
    pNum = 0

    for sP in sPs:
        if sP.isZoom:
            continue

        # loop over dustModels, for model comparison plot
        for simColorsModel in simColorsModels:

            print('Color PDF [%s] [%s]: %s' % ('-'.join(bands),simColorsModel,sP.simName))
            sP.setRedshift(simRedshift)

            # load fullbox stellar masses
            gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
            gc_masses = sP.units.codeMassToLogMsun( gc['subhalos'][:,sP.ptNum('stars')] )
            
            # galaxy selection
            w_cen, w_all, w_sat = cenSatSubhaloIndices(sP)

            # determine unique color
            c = ax._get_lines.prop_cycler.next()['color']
            # skip second color (pNum=1), reserved for L205n2500, if we do a res or dustModel comparison
            if pNum == 1 and (len(sPs) >=3 or len(simColorsModels) >= 3):
                c = ax._get_lines.prop_cycler.next()['color']
            pNum += 1

            # load simulation colors
            if simColorsModel[-4:] == '_all':
                # request all 12*Nside^2 projections per subhalo, flatten into 1D array (sh0p0,sh0p1,...)
                gc_colors, _ = loadSimGalColors(sP, simColorsModel[:-4], bands=bands, projs='all')
                gc_colors = np.reshape( gc_colors, gc_colors.shape[0]*gc_colors.shape[1] )

                # replicate stellar masses
                from re import findall # could replace with actual Nside return from loadSimGalColors()
                Nside = np.int32( findall(r'ns\d+',simColorsModel)[0][2:] )
                assert Nside == 1

                gc_inds = np.arange( gc_masses.size, dtype='int32' )
                gc_inds = np.repeat( gc_inds, 12*Nside**2 )
                gc_masses = gc_masses[gc_inds]

                # replicate galaxy selections by crossmatching original selection indices with replicated list
                from tracer.tracerMC import match3
                origSatSize = w_sat.size
                origCenSize = w_cen.size
                _, w_cen = match3(w_cen, gc_inds)
                _, w_sat = match3(w_sat, gc_inds)
                _, w_all = match3(w_all, gc_inds)
                assert w_sat.size == 12*Nside**2 * origSatSize
                assert w_cen.size == 12*Nside**2 * origCenSize
            else:
                # request a single random color per subhalo (for "_res" models), and/or for simple models
                # without multiple projections even saved
                gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands)
            
            assert gc_colors.size == gc_masses.size
            assert w_all.size == gc_masses.size

            # selection:
            normFacs = np.zeros( len(stellarMassBins) )
            binSize  = np.zeros( len(stellarMassBins) )
            nBins    = np.zeros( len(stellarMassBins), dtype='int32' )

            if allOnly:     loopInds = [0,1] # total only, except we add centrals for the first mass bin only
            if splitCenSat: loopInds = [0,1,2] # show total, and cen/sat decomposition all at once
            if cenOnly:     loopInds = [1] # centrals only

            for j in loopInds:
                if j == 0: w = w_all
                if j == 1: w = w_cen
                if j == 2: w = w_sat

                # galaxy mass definition and color
                stellar_mass = gc_masses[w]
                galaxy_color = gc_colors[w]

                wNotNan = np.isfinite(galaxy_color) # filter out subhalos with e.g. no stars
                galaxy_color = galaxy_color[wNotNan]
                stellar_mass = stellar_mass[wNotNan]

                # loop over each mass bin
                for i, stellarMassBin in enumerate(stellarMassBins):
                    if allOnly and j == 1 and stellarMassBin[0] > 9.0:
                        continue # add centrals for first mass bin only, if showing total only

                    wBin = np.where( (stellar_mass >= stellarMassBin[0]) & (stellar_mass < stellarMassBin[1]) & \
                                     (galaxy_color >= mag_range[0]) & (galaxy_color < mag_range[1]) )

                    if j == 0 or (cenOnly and j == loopInds[0]):
                        # set normalization (such that integral of PDF is one) based on 'all galaxies'
                        nBins[i] = np.max( [16, np.int( np.sqrt( len(wBin[0] )) * 1.4)] ) # adaptive
                        binSize[i] = (mag_range[1]-mag_range[0]) / nBins[i]
                        normFacs[i] = 1.0 / (binSize[i] * len(wBin[0]))

                    # plot panel config
                    label = sP.simName if i == iLeg and j == loopInds[0] and splitCenSat else ''
                    alpha = 1.0 if j == loopInds[0] else 0.7
                    if not splitCenSat: alpha = 0.1

                    # obs histogram
                    wObs = np.where((sdss_Mstar >= stellarMassBin[0]) & (sdss_Mstar < stellarMassBin[1]))
                    yy, xx = np.histogram(sdss_color[wObs], bins=nBins[i], range=mag_range, density=True)
                    xx = xx[:-1] + 0.5*binSize[i]

                    if not clean:
                        axes[i].plot(xx, yy, '-', color=obs_color, alpha=alpha, lw=3.0)

                    # obs kde
                    xx = np.linspace(mag_range[0], mag_range[1], 200)
                    bw_scotthalf = sdss_color[wObs].size**(-1.0/(sdss_color.ndim+4.0)) * 0.5
                    kde2 = gaussian_kde(sdss_color[wObs], bw_method='scott')
                    yy_obs = kde2(xx)
                    axes[i].plot(xx, yy_obs, '-', color=obs_color, alpha=1.0, lw=3.0)
                    axes[i].fill_between(xx, 0.0, yy_obs, facecolor=obs_color, alpha=0.1, interpolate=True)\

                    if len(wBin[0]) <= 1:
                        print(' skip sim kde no data: ',sP.simName,i)
                        continue

                    # sim histogram
                    yy, xx = np.histogram(galaxy_color[wBin], bins=nBins[i], range=mag_range)
                    yy2 = yy.astype('float32') * normFacs[i]
                    xx = xx[:-1] + 0.5*binSize[i]

                    if not clean:
                        axes[i].plot(xx, yy2, linestyles[j], label=label, color=c, alpha=alpha, lw=3.0)

                    # sim kde
                    if not splitCenSat:
                        xx = np.linspace(mag_range[0], mag_range[1], 200)
                        bw_scotthalf = galaxy_color[wBin].size**(-1.0/(galaxy_color.ndim+4.0)) * 0.5
                        kde1 = gaussian_kde(galaxy_color[wBin], bw_method='scott') # scott, silvermann, or scalar
                        yy_sim = kde1(xx)

                        if len(simColorsModels) == 1:
                            # label by simulation
                            label = sP.simName if i == iLeg and j == loopInds[0] else ''
                        else:
                            # label by dust model
                            label = simColorsModel if i == iLeg and j == loopInds[0] else ''
                        if clean:
                            # replace dust model labels by paper versions
                            label = label.replace("p07c_cf00dust_res_conv_ns1_all", "Model C (all)")
                            label = label.replace("p07c_cf00dust_res_conv_ns1", "Model C")
                            label = label.replace("p07c_cf00dust", "Model B")
                            label = label.replace("p07c_nodust", "Model A")

                        axes[i].plot(xx, yy_sim, linestyles[j], label=label, color=c, alpha=1.0, lw=3.0)
                        if j == 0:
                            axes[i].fill_between(xx, 0.0, yy_sim, color=c, alpha=0.1, interpolate=True)

                    pMaxVals[i] = np.max( [pMaxVals[i], np.max([yy_obs.max(),yy_sim.max()]) ])

    for i, stellarMassBin in enumerate(stellarMassBins):
        axes[i].set_ylim([0, 1.2 * pMaxVals[i]])

    # legend (simulations) (obs)
    handles, labels = axes[iLeg].get_legend_handles_labels()
    handlesO = [plt.Line2D( (0,1),(0,0),color=obs_color,lw=3.0,marker='',linestyle='-')]
    labelsO  = ['SDSS z<0.1'] # DR12 fspsGranWideDust

    legend2 = axes[iLeg].legend(handles+handlesO, labels+labelsO, loc='upper left')

    # legend (central/satellite split)
    if splitCenSat:
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
        lExtra = ['all galaxies','centrals','satellites']

        handles, labels = axes[iLeg+1].get_legend_handles_labels()
        legend3 = axes[iLeg+1].legend(handles+sExtra, labels+lExtra, loc='upper left')

    if allOnly and len(stellarMassBins) > 3:
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles[0:2]]
        lExtra = ['all galaxies','centrals only']

        handles, labels = axes[0].get_legend_handles_labels()
        legend3 = axes[0].legend(handles+sExtra, labels+lExtra, loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def galaxyColor2DPDFs(sPs, pdf, simColorsModel='p07c_cf00dust', splitCenSat=False, simRedshift=0.0):
    """ 2D contours of galaxy colors/Mstar plane, multiple bands. """
    from util import simParams
    
    # config
    obs_color = '#000000'
    Mstar_range = [9.0, 12.0]
    bandCombos = [ ['u','i'], ['g','r'], ['r','i'], ['i','z'] ] # use multiple of 2
    #bandCombos = [ ['u','i'], ['g','r'], ['r','i'], ['u','r'] ] # use multiple of 2

    eCorrect = True # True, False (for sdss points)
    kCorrect = True # True, False (for sdss points)

    sizefac = 1.3 if clean else 1.5
    figsize = (16,9)

    def _discreteReSampleMatched(obs_1dhist, sim_1dhist, nBinsDS):
        """ Apply a quasi inverse transform sampling method to draw a Mstar distribution
        from the simulation matching that from SDSS, so we can plot a fair comparison of 
        the full 1D histogram of a color over the Mstar_range. """
        obsMstarHist = obs_1dhist / obs_1dhist.sum()
        simInds = np.zeros( sim_1dhist.size+1, dtype='int32' )
        binSize = (Mstar_range[1] - Mstar_range[0]) / nBinsDS

        numAdded = 0

        for k in range(nBinsDS):
            binMin = Mstar_range[0] + k*binSize
            binMax = Mstar_range[0] + (k+1)*binSize
            simIndsBin = np.where((sim_1dhist >= binMin) & (sim_1dhist < binMax))

            nWantedToMatchObs = np.int32(obsMstarHist[k] * sim_1dhist.size)

            if len(simIndsBin[0]) == 0 and nWantedToMatchObs > 0:
                continue # failure in small boxes to have massive halos, or low res to have small halos

            #print(k,'sim size: ',simIndsBin[0].size,'wanted: ',nWantedToMatchObs)
            indsToAdd = np.random.choice(simIndsBin[0], size=nWantedToMatchObs, replace=True)
            simInds[numAdded : numAdded+indsToAdd.size] = indsToAdd

            numAdded += indsToAdd.size

        return simInds[0:numAdded]

    # create an entire plot PER run, only one 2D sim contour set each
    for sP_target in sPs:
        # start plot
        fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
        axes  = []
        axes2 = []

        # loop over each requested color
        obs1DHistos = {}

        for i, bands in enumerate(bandCombos):
            print('Color 2D PDFs [%s-%s]: obs' % (bands[0],bands[1]))

            # panel setup
            ax = fig.add_subplot(2,len(bandCombos)/2,i+1)
            axes.append(ax)
            mag_range = _bandMagRange(bands)
            
            if not clean and 0:
                obsMagStr = 'modelMag%s%s' % ('-E' if eCorrect else '','+K' if kCorrect else '')
                cenSatStr = '' if splitCenSat else ', cen+sat'
                titleStr = '[ obs=%s, sim=%s%s ]' % (obsMagStr,simColorsModel,cenSatStr)
                ax.set_title(titleStr)

            xlabel2 = '' if clean else '(<2r_{\star,1/2})'
            ax.set_xlim(Mstar_range)
            ax.set_ylim(mag_range)
            ax.set_xlabel('M$_{\\rm \star}%s$ [ log M$_{\\rm sun}$ ]' % xlabel2)
            ax.set_ylabel('(%s-%s) color [ mag ]' % (bands[0],bands[1]))

            # load observational points, restrict colors to mag_range as done for sims (for correct normalization)
            sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=eCorrect, kCorrect=kCorrect)

            w = np.where( (sdss_color >= mag_range[0]) & (sdss_color <= mag_range[1]) & \
                          (sdss_Mstar >= Mstar_range[0]) & (sdss_Mstar <= Mstar_range[1]) )

            sdss_color = sdss_color[w]
            sdss_Mstar = sdss_Mstar[w]

            # config
            extent = [Mstar_range[0],Mstar_range[1],mag_range[0],mag_range[1]]
            cLevels = [0.2,0.5,0.75,0.98]
            cAlphas = [0.05,0.2,0.5,1.0]
            nKDE1D  = 200
            nBins1D = 100
            nBinsDS = 40 # discrete re-sampling
            nBins2D = [50,100]

            # (A) create kde of observations
            xx, yy, kde_obs = calcMstarColor2dKDE(bands, sdss_Mstar, sdss_color, Mstar_range, mag_range)

            for k in range(kde_obs.shape[0]):
                kde_obs[k,:] /= kde_obs[k,:].max() # by column normalization

            for k, cLevel in enumerate(cLevels):
                ax.contour(xx, yy, kde_obs, [cLevel], 
                           colors=[obs_color], alpha=cAlphas[k], linewidths=3.0, extent=extent)

            # (B) hist approach
            #cc, xBins, yBins = np.histogram2d(sdss_Mstar, sdss_color, bins=nBins2D, range=[Mstar_range,mag_range])
            #for k in range(c.shape[0]):
            #    cc[k,:] /= cc[k,:].max() # by column normalization
            #ax.contour(xBins[:-1], yBins[:-1], cc.T, cLevels, extent=extent)

            # vertical 1D histogram on the right side
            ax2 = make_axes_locatable(ax).append_axes('right', size='20%', pad=0.1)
            axes2.append(ax2)

            yy, xx = np.histogram(sdss_color, bins=nBins1D, range=mag_range, density=True)
            xx = xx[:-1] + 0.5*(mag_range[1]-mag_range[0])/nBins1D
            ax2.plot(yy, xx, '-', color=obs_color, alpha=0.2, lw=3.0)

            obs1DHistos[''.join(bands)], _ = np.histogram(sdss_Mstar, bins=nBinsDS, range=Mstar_range)

            # obs 1D kde
            xx = np.linspace(mag_range[0], mag_range[1], nKDE1D)
            kde2 = gaussian_kde(sdss_color, bw_method='scott')
            ax2.plot(kde2(xx), xx, '-', color=obs_color, alpha=0.9, lw=3.0, label='SDSS z<0.1')
            ax2.fill_betweenx(xx, 0.0, kde2(xx), facecolor=obs_color, alpha=0.05)

            ax2.set_ylim(mag_range)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

        # loop over each fullbox run
        spColors = []

        for sP in sPs:
            if sP.isZoom:
                continue

            print('Color 2D PDFs [%s] [%s]: %s' % ('-'.join(bands),simColorsModel,sP.simName))
            sP.setRedshift(simRedshift)

            c = ax._get_lines.prop_cycler.next()['color']
            spColors.append(c)

            # load fullbox stellar masses and photometrics
            gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
            gc_masses = sP.units.codeMassToLogMsun( gc['subhalos'][:,sP.ptNum('stars')] )
            
            # load simulation colors
            colorData = loadSimGalColors(sP, simColorsModel)

            # galaxy selection
            w_cen, w_all, w_sat = cenSatSubhaloIndices(sP)

            # selection:
            normFacs = np.zeros( len(bandCombos) )
            binSize  = np.zeros( len(bandCombos) )
            nBins    = np.zeros( len(bandCombos), dtype='int32' )

            # loop over each requested color
            for i, bands in enumerate(bandCombos):

                # calculate simulation colors
                gc_colors, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=bands)

                # config for this band
                mag_range = _bandMagRange(bands)
                extent = [Mstar_range[0],Mstar_range[1],mag_range[0],mag_range[1]]

                loopInds = range(1) # total only
                if splitCenSat: loopInds = range(3)

                for j in loopInds:
                    if j == 0: w = w_all
                    if j == 1: w = w_cen
                    if j == 2: w = w_sat

                    # galaxy mass definition and color
                    stellar_mass = gc_masses[w]
                    galaxy_color = gc_colors[w]

                    wNotNan = np.isfinite(galaxy_color) # filter out subhalos with e.g. no stars
                    galaxy_color = galaxy_color[wNotNan]
                    stellar_mass = stellar_mass[wNotNan]

                    # select in bounds
                    wBin = np.where( (stellar_mass >= Mstar_range[0]) & (stellar_mass < Mstar_range[1]) & \
                                     (galaxy_color >= mag_range[0]) & (galaxy_color < mag_range[1]) )

                    # 1d: resample simulated Mstar distribution to roughly matched SDSS Mstar distribution
                    binInds = _discreteReSampleMatched(obs1DHistos[''.join(bands)], stellar_mass[wBin], nBinsDS)
                    simInds = wBin[0][binInds]

                    # sim 1D histogram on the side
                    #yy, xx = np.histogram(galaxy_color[simInds], bins=nBins1D, range=mag_range, density=True)
                    #xx = xx[:-1] + 0.5*(mag_range[1]-mag_range[0])/nBins1D
                    #axes2[i].plot(yy, xx, '-', color=c, alpha=0.2, lw=3.0)

                    # sim 1D KDE on the side
                    xx = np.linspace(mag_range[0], mag_range[1], nKDE1D)
                    kde = gaussian_kde(galaxy_color[simInds], bw_method='scott')

                    label = sP.simName if j == 0 else ''
                    axes2[i].plot(kde(xx), xx, '-', color=c, alpha=1.0, lw=3.0, label=label)
                    axes2[i].fill_betweenx(xx, 0.0, kde(xx), facecolor=c, alpha=0.1)

                    # (only one 2D contour set per plot)
                    if sP != sP_target:
                        continue

                    # (A) sim 2D kde approach
                    xx, yy, kde_sim = calcMstarColor2dKDE(bands, stellar_mass[wBin], galaxy_color[wBin], 
                                                          Mstar_range, mag_range, 
                                                          sP=sP, simColorsModel=simColorsModel)

                    for k in range(kde_sim.shape[0]):
                        kde_sim[k,:] /= kde_sim[k,:].max() # by column normalization

                    for k, cLevel in enumerate(cLevels):
                        axes[i].contour(xx, yy, kde_sim, [cLevel], 
                                       colors=[c], alpha=cAlphas[k], linewidths=3.0, extent=extent)

                    # (B) sim 2D histogram approach
                    #cc, xBins, yBins = np.histogram2d(stellar_mass[wBin], galaxy_color[wBin], bins=nBins2D, \
                    #                                 range=[Mstar_range,mag_range])
                    #for k in range(cc.shape[0]):
                    #    cc[k,:] /= cc[k,:].max() # by column normalization
                    #for k, cLevel in enumerate(cLevels):
                    #    axes[i].contour(xBins[:-1], yBins[:-1], cc.T, [cLevel], 
                    #               colors=[c], alpha=cAlphas[k], linewidths=3.0, extent=extent)

        # legend (simulations) (obs)
        hExtra = []#[plt.Line2D( (0,1),(0,0),color=obs_color,lw=3.0,marker='',linestyle='-')]
        lExtra = []#['SDSS z<0.1']

        handles, labels = axes2[0].get_legend_handles_labels()
        legend = axes[0].legend(handles+hExtra, labels+lExtra, loc='upper left')

        # legend (central/satellite split)
        if splitCenSat:
            sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
            lExtra = ['all galaxies','centrals','satellites']

            legend2 = axes[1].legend(sExtra, lExtra, loc='upper left')

        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def viewingAngleVariation():
    """ Demo of Nside>>1 results for variation of (one or two) galaxy colors as a function of 
    viewing angle. 1D Histogram. """

    # config
    nBins = 250

    sP = simParams(res=1820, run='tng', redshift=0.0)

    ac_modelA = 'p07c_nodust'
    ac_modelB = 'p07c_cf00dust'
    ac_modelD = 'p07c_bc00dust' # debugging
    ac_modelC_demos  = ['p07c_ns4_demo', 'p07c_ns8_demo']

    bands = ['g','r']

    # load
    modelA_colors, _ = loadSimGalColors(sP, ac_modelA, bands=bands)
    modelB_colors, _ = loadSimGalColors(sP, ac_modelB, bands=bands)
    modelD_colors, _ = loadSimGalColors(sP, ac_modelD, bands=bands)
    modelC_colors = {}
    modelC_ids = {}

    for ac_demo in ac_modelC_demos:
        modelC_colors[ac_demo], modelC_ids[ac_demo] = loadSimGalColors(sP, ac_demo, bands=bands, projs='all')

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    mag_range  = [0.3,0.8] #_bandMagRange(bands, tight=True)

    #mag_range = [0.7,0.75]
    #nBins = 100

    markers    = ['o','s','D']
    linestyles = [':','-']
    binSize    = (mag_range[1]-mag_range[0])/nBins

    ax.set_xlim(mag_range)
    ax.set_yscale('log')
    ax.set_ylim([2,500])
    ax.set_xlabel('(%s-%s) color [ mag ]' % (bands[0],bands[1]))
    ax.set_ylabel('PDF $\int=1$')

    # loop over multiple Nside demos
    lineColors = {}

    for i, ac_demo in enumerate(ac_modelC_demos):
        # loop over each subhalo included in the demo
        for j in range(modelC_colors[ac_demo].shape[0]):
            # histogram demo color distribution
            colors = modelC_colors[ac_demo][j,:]
            sub_id = modelC_ids[ac_demo][j]

            yy, xx = np.histogram(colors, bins=nBins, range=mag_range, density=True)
            xx = xx[:-1] + 0.5 * binSize

            label = ''
            subhalo = groupCatSingle(sP, subhaloID=sub_id)
            mstar = sP.units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][sP.ptNum('star')])
            sSFR = np.log10(subhalo['SubhaloSFR'] / 10.0**mstar)
            label = '[%d] M$_\star$=10$^{%.1f}$ sSFR=%.1f' % (sub_id,mstar,sSFR)

            # keep same color per subhalo, across different Ns demos
            if i == 0:
                l, = ax.plot(xx, yy, linestyle=linestyles[i], lw=2.5)
                lineColors[j] = l.get_color()
            else:
                l, = ax.plot(xx, yy, linestyle=linestyles[i], label=label, lw=2.5, color=lineColors[j])

            # plot model A and model B values for this subhalo
            if i == 0:
                color_A = modelA_colors[sub_id]
                color_B = modelB_colors[sub_id]
                color_D = modelD_colors[sub_id]
                print(sub_id,color_A,color_B,color_D,colors.mean())

                ax.plot([color_A], [10], color=lineColors[j], marker=markers[0], lw=2.5)
                ax.plot([color_B], [12], color=lineColors[j], marker=markers[1], lw=2.5)
                #ax.plot([color_D], [16], color=lineColors[j], marker=markers[2], lw=2.5)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker=markers[0], lw=0.0),
              plt.Line2D( (0,1), (0,0), color='black', marker=markers[1], lw=0.0),
              #plt.Line2D( (0,1), (0,0), color='black', marker=markers[2], lw=0.0),
              plt.Line2D( (0,1), (0,0), color='black', linestyle=linestyles[0], lw=2.5),
              plt.Line2D( (0,1), (0,0), color='black', linestyle=linestyles[1], lw=2.5)]
    lExtra = ['Model A','Model B','Model C, N$_{\\rm side}$ = 4','Model C, N$_{\\rm side}$ = 8'] #'Model D',

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    # finish plot and save
    fig.tight_layout()
    fig.savefig('figure_appendix1_viewing_angle_variation.pdf')
    plt.close(fig)


def plots():
    """ Driver. """
    sP = simParams(res=1820, run='tng', redshift=0.0)

    # debug:
    #pdf = PdfPages('galaxyColor_2dhistos.pdf')
    #bands = ['g','r']
    #xQuant = 'mstar2_log'
    #cs = 'median'
    #css = 'cen'
    #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='ssfr', cStatistic=cs)
    #pdf.close()
    #import pdb; pdb.set_trace()

    # plots
    #for cs in ['median','mean']:
    for css in ['cen']: #['cen','sat','all']:
        bands = ['g','r']
        xQuant = 'mstar2_log'
        cs = 'median_nan'
        #css = 'cen'

        pdf = PdfPages('galaxyColor_2dhistos_%s_%s_%s_%s_%s.pdf' % (sP.simName,''.join(bands),xQuant,cs,css))

        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant=None)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='ssfr', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_stars', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_gas', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='size_stars', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='size_gas', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fgas1', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fgas2', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='mgas2', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='mgas1', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='stellarage', cStatistic=cs)

        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_mm5', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_ma5', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_poly7', cStatistic=cs)

        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_all_eps07o', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_all_eps07m', cStatistic=cs)
        ###histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_10re_eps07o', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_10re_eps07m', cStatistic=cs)

        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_sfrgt0_masswt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_sfrgt0_volwt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_2rhalf_masswt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_2rhalf_volwt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_halo_masswt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='bmag_halo_volwt', cStatistic=cs)

        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='pratio_halo_masswt', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='pratio_halo_volwt', cStatistic=cs)

        #histo2D(sP, pdf, bands, xQuant='ssfr', cenSatSelect=css, cQuant=None)
        #histo2D(sP, pdf, bands, xQuant='ssfr', cenSatSelect=css, cQuant='fgas2', cStatistic=cs)

        pdf.close()
    
def paperPlots():
    """ Construct all the plots for the paper. """
    global clean
    clean = True

    L75  = simParams(res=1820,run='tng',redshift=0.0)
    L205 = simParams(res=2500,run='tng',redshift=0.0)

    dust_A = 'p07c_nodust'
    dust_B = 'p07c_cf00dust'
    dust_C = 'p07c_cf00dust_res_conv_ns1' # one random projection per subhalo
    dust_C_all = 'p07c_cf00dust_res_conv_ns1_all' # all projections shown

    # testing
    if 0:
        sPs = [L205]
        dusts = [dust_B,dust_B+'_rad30pkpc']

        pdf = PdfPages('figure_test.pdf')
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=dusts)
        pdf.close()

    # figure 1
    if 0:
        sPs = [L75, L205]
        dust = dust_B # dust_C_all # need L205

        pdf = PdfPages('figure1_%s.pdf' % dust)
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=[dust])
        pdf.close()

    # figure 2
    if 0:
        pass

    # appendix figure 1, viewing angle variation
    if 0:
        viewingAngleVariation()

    # appendix figure 2, dust model dependence (3 1D histos)
    if 0:
        sPs = [L75]
        dusts = [dust_C_all, dust_C, dust_B, dust_A]
        massBins = ( [9.5,10.0], [10.0,10.5], [10.5,11.0] )

        pdf = PdfPages('figure_appendix2.pdf')
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=dusts, stellarMassBins=massBins)
        pdf.close()

    # appendix figure 3, resolution convergence (3 1D histos)
    if 0:
        L75n910 = simParams(res=910,run='tng',redshift=0.0)
        L75n455 = simParams(res=455,run='tng',redshift=0.0)
        sPs = [L75, L75n910, L75n455]
        dust = dust_C_all
        massBins = ( [9.5,10.0], [10.0,10.5], [10.5,11.0] )

        pdf = PdfPages('figure_appendix3_%s.pdf' % dust)
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=[dust], stellarMassBins=massBins)
        pdf.close()

    # appendix figure 4, 2x2 grid of different colors
    if 1:
        sPs = [L75]
        dust = dust_B

        pdf = PdfPages('figure_appendix4_%s.pdf' % dust)
        galaxyColor2DPDFs(sPs, pdf, simColorsModel=dust)
        pdf.close()
