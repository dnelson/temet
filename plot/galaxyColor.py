"""
galaxyColor.py
  Plots for TNG flagship paper: galaxy colors, color bimodality.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d, gaussian_kde

from util import simParams
from util.helper import running_median, contourf, logZeroSafe, logZeroNaN, closest, loadColorTable
from tracer.tracerMC import match3
from cosmo.galaxyColor import calcMstarColor2dKDE, calcColorEvoTracks, characterizeColorMassPlane, \
   loadSimGalColors, stellarPhotToSDSSColor, calcSDSSColors
from cosmo.util import cenSatSubhaloIndices, snapNumToRedshift
from cosmo.load import groupCat, groupCatSingle, groupCatHeader
from plot.general import simSubhaloQuantity, getWhiteBlackColors, bandMagRange, quantList
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from vis.common import setAxisColors
from plot.config import *

def galaxyColorPDF(sPs, pdf, bands=['u','i'], simColorsModels=[defSimColorModel], 
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
    if len(stellarMassBins) >= 4:
        figsize = (16,9) # 2 rows, N columns
        #figsize = (9,16) # 3 rows, N (2) columns
    else:
        figsize = (5.3, 13.5)

    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
    axes = []

    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.0,4.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [-0.2,1.2]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.6]

    # loop over each mass bin
    for i, stellarMassBin in enumerate(stellarMassBins):

        # panel setup
        if len(stellarMassBins) >= 4:
            iLeg = 5 #2 # lower right (2x3)
            ax = fig.add_subplot(2,len(stellarMassBins)/2,i+1) #2 rows, N columns
            #ax = fig.add_subplot(3,len(stellarMassBins)/3,i+1) #3 rows, N columns
        else: # N rows, 1 column
            iLeg = 2 # bottom (3x1)
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

        legendPos = 'upper right'
        if len(stellarMassBins) >= 4: # 2 rows, N columns
            legendPos = 'upper left'

        legend1 = ax.legend(sExtra, lExtra, loc=legendPos, handlelength=0, handletextpad=0)
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
            # skip second color (pNum=1), reserved for L205n2500, if we do a res (x3), dustModel (x3), 
            # or TNG vs Illlustris (x2) comparison
            if pNum == 1 and (len(sPs) >=3 or len(simColorsModels) >= 3):
                c = ax._get_lines.prop_cycler.next()['color']
            pNum += 1
            if (len(sPs) == 2 and sPs[1].simName == 'TNG100-1' and sPs[0].simName == 'Illustris-1'):
                #if sP.simName == 'Illustris-1': c = '#9467BD' # tableau10 fifth (purple) for Illustris-1
                #if sP.simName == 'Illustris-1': c = '#8C564B' # tableau10 sixth (brown) for Illustris-1
                if sP.simName == 'Illustris-1': c = '#D62728' # tableau10 fourth (red) for Illustris-1
                if sP.simName == 'TNG100-1': c = '#1F77B4' # tableau10 first (blue) for TNG100-1

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
                    #axes[i].fill_between(xx, 0.0, yy_obs, facecolor=obs_color, alpha=0.1, interpolate=True)

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
                            label = label.replace("p07c_cf00dust_res_conv_ns1_rad30pkpc_all", "Model C (all)")
                            label = label.replace("p07c_cf00dust_res_conv_ns1_rad30pkpc", "Model C")
                            label = label.replace("p07c_cf00dust", "Model B")
                            label = label.replace("p07c_nodust", "Model A")

                        alpha = 0.85 if 'Illustris' in sP.simName else 1.0
                        axes[i].plot(xx, yy_sim, linestyles[j], label=label, color=c, alpha=alpha, lw=3.0)
                        if j == 0:
                            axes[i].fill_between(xx, 0.0, yy_sim, color=c, alpha=0.1, interpolate=True)

                    pMaxVals[i] = np.max( [pMaxVals[i], np.max([yy_obs.max(),yy_sim.max()]) ])

    # y-ranges
    for i, stellarMassBin in enumerate(stellarMassBins):
        if clean:  
            # fix y-axis limits for talk series
            y_max = [8.0, 6.0, 6.0, 8.0, 10.0, 10.0][i]
            axes[i].set_ylim([0.0,y_max])
        else:
            axes[i].set_ylim([0, 1.2 * pMaxVals[i]])

    # legend (simulations) (obs)
    legendPos = 'upper left'
    if len(stellarMassBins) >= 4: # 2 rows, N columns
        legendPos = 'lower left'

    handles, labels = axes[iLeg].get_legend_handles_labels()
    handlesO = [plt.Line2D( (0,1),(0,0),color=obs_color,lw=3.0,marker='',linestyle='-')]
    labelsO  = ['SDSS z<0.1'] # DR12 fspsGranWideDust

    legend2 = axes[iLeg].legend(handlesO+handles, labelsO+labels, loc=legendPos)

    # legend (central/satellite split)
    if splitCenSat:
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
        lExtra = ['all galaxies','centrals','satellites']

        handles, labels = axes[iLeg+1].get_legend_handles_labels()
        legend3 = axes[iLeg+1].legend(handles+sExtra, labels+lExtra, loc='upper right')

    if allOnly and len(stellarMassBins) > 3:
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles[0:2]]
        lExtra = ['all galaxies','centrals only']

        handles, labels = axes[0].get_legend_handles_labels()
        legend3 = axes[0].legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def galaxyColor2DPDFs(sPs, pdf, simColorsModel=defSimColorModel, splitCenSat=False, simRedshift=0.0):
    """ 2D contours of galaxy colors/Mstar plane, multiple bands. """
    from util import simParams
    
    # config
    obs_color = '#000000'
    Mstar_range = [9.0, 12.0]
    #bandCombos = [ ['u','i'], ['g','r'], ['r','i'], ['i','z'] ] # use multiple of 2
    bandCombos = [ ['u','i'], ['g','r'], ['r','i'], ['u','r'] ]
    #bandCombos = [ ['u','i'], ['g','r'], ['r','z'], ['u','r'] ] 

    eCorrect = True # True, False (for sdss points)
    kCorrect = True # True, False (for sdss points)

    sizefac = 1.0/sfclean if clean else 1.5

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

            if len(simIndsBin[0]) == 0 or nWantedToMatchObs == 0:
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
            mag_range = bandMagRange(bands)
            
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
                mag_range = bandMagRange(bands)
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
    ac_modelD = 'p07c_bc00dust' # debugging, not shown
    ac_modelC_demos  = ['p07c_ns4_demo_rad30pkpc', 'p07c_ns8_demo_rad30pkpc']

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
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
    ax = fig.add_subplot(111)

    mag_range  = [0.3,0.85] #bandMagRange(bands, tight=True)
    markers    = ['o','s','D']
    linestyles = [':','-']
    binSize    = (mag_range[1]-mag_range[0])/nBins

    ax.set_xlim(mag_range)
    ax.set_yscale('log')
    ax.set_ylim([4,800])
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
            label = 'M$_\star$=10$^{%.1f}$ sSFR=%.1f' % (mstar,sSFR)
            if not clean: label = '[%d] %s' % (sub_id,label)

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
    fig.savefig('appendix1_viewing_angle_variation.pdf')
    plt.close(fig)

def colorFluxArrows2DEvo(sP, pdf, bands, toRedshift, cenSatSelect='cen', minCount=None, 
      simColorsModel=defSimColorModel, arrowMethod='arrow', fig_subplot=[None,None], pStyle='white'):
    """ Plot 'flux' arrows in the (color,Mstar) 2D plane showing the median evolution of all 
    galaxies in each bin over X Gyrs of time. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    # hard-coded config
    xQuant = 'mstar2_log'
    contourColor = '#555555' #'orange'
    arrowColor = 'black'
    arrowAlpha = 0.8
    contourLw = 2.0

    nBins = 12 #or 20
    rndProjInd = 0

    if arrowMethod in ['stream']:
        nBins = 30

    mag_range = bandMagRange(bands, tight=True)

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    if 'sim_xvals' in sP.data:
        sim_xvals, xlabel, xMinMax = sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax']
    else:
        sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant, clean)
        sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax'] = sim_xvals, xlabel, xMinMax

    # y-axis: load/calculate evolution of simulation colors, cached in sP.data
    if 'sim_colors_evo' in sP.data:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = \
          sP.data['sim_colors_evo'], sP.data['shID_evo'], sP.data['subhalo_ids'], sP.data['snaps']
    else:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = \
          calcColorEvoTracks(sP, bands=bands, simColorsModel=simColorsModel)
        sP.data['sim_colors_evo'], sP.data['shID_evo'], sP.data['subhalo_ids'], sP.data['snaps'] = \
          sim_colors_evo, shID_evo, subhalo_ids, snaps

    ylabel = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
    if not clean: ylabel += ' %s' % simColorsModel

    # pick initial and final color corresponding to timewindow
    savedRedshifts = snapNumToRedshift(sP, snap=snaps)
    _, zIndTo = closest(savedRedshifts, toRedshift)
    _, zIndFrom = closest(savedRedshifts, sP.redshift)

    assert zIndTo > 0 and zIndTo < snaps.size
    assert snaps[zIndFrom] == sP.snap

    sim_colors_from = np.squeeze( sim_colors_evo[:,rndProjInd,zIndFrom] )
    sim_colors_to   = np.squeeze( sim_colors_evo[:,rndProjInd,zIndTo] )

    # load x-axis quantity at final (to) time
    origSnap = sP.snap
    sP.setSnap( snaps[zIndTo] )
    sim_xvals_to, _, _, _ = simSubhaloQuantity(sP, xQuant, clean)
    ageTo = sP.tage

    sP.setSnap( origSnap )

    subhalo_ids_to = np.squeeze( shID_evo[:,zIndTo] )
    sim_xvals_to = sim_xvals_to[subhalo_ids_to]

    # restrict xvals to the subhaloIDs for which we have saved colors
    sim_xvals_from = sim_xvals[subhalo_ids]

    # central/satellite selection?
    wSelect_orig = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
    wSelect, _ = match3(subhalo_ids, wSelect_orig)

    frac_global = float(wSelect_orig.size) / sim_xvals.size * 100
    frac_local  = float(wSelect.size) / subhalo_ids.size * 100
    print(sP.simName,'-'.join(bands),simColorsModel,xQuant,cenSatSelect,minCount)
    print(' time interval [%.2f] Gyr (z=%.1f to z=%.1f)' % (sP.tage-ageTo,sP.redshift,toRedshift))
    print(' css (%s): [%d] of global [%d] = %.1f%% (reduced to [%d] of the [%d] in colorEvo = %.1f%%)' % \
        (cenSatSelect,wSelect_orig.size,sim_xvals.size,frac_global,wSelect.size,subhalo_ids.size,frac_local))

    sim_colors_from = sim_colors_from[wSelect]
    sim_colors_to   = sim_colors_to[wSelect]
    sim_xvals_from  = sim_xvals_from[wSelect]
    sim_xvals_to    = sim_xvals_to[wSelect]

    # reduce to the subset with non-NaN colors at both ends of the time interval
    wFiniteColor = np.isfinite(sim_colors_from) & np.isfinite(sim_colors_to)

    sim_colors_from = sim_colors_from[wFiniteColor]
    sim_colors_to   = sim_colors_to[wFiniteColor]
    sim_xvals_from  = sim_xvals_from[wFiniteColor]
    sim_xvals_to    = sim_xvals_to[wFiniteColor]

    # start plot
    if fig_subplot[0] is None:
        sizefac = sfclean if clean else 1.0
        fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
        ax = fig.add_subplot(111, axisbg=color1)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1], axisbg=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not clean:
        ax.set_title('select=%s mincount=%s' % (cenSatSelect,minCount))

    # 2d bin configuration
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]]

    binSize_x = (xMinMax[1] - xMinMax[0]) / nBins2D[0]
    binSize_y = (mag_range[1] - mag_range[0]) / nBins2D[1]
    print(' nBins2D: ',nBins2D,'binSizes (x,y): ', binSize_x, binSize_y)

    # calculate arrows ('velocity') for each bin (across whole grid)
    arrows_start_x = np.zeros( nBins2D, dtype='float32' )
    arrows_start_y = np.zeros( nBins2D, dtype='float32' )
    arrows_end_x = np.zeros( nBins2D, dtype='float32' )
    arrows_end_y = np.zeros( nBins2D, dtype='float32' )
    counts = np.zeros( nBins2D, dtype='int32' )

    arrows_end_x.fill(np.nan)
    arrows_end_y.fill(np.nan)

    xx = np.zeros( nBins2D[0], dtype='float32' )
    yy = np.zeros( nBins2D[1], dtype='float32' )

    for i in range(nBins2D[0]):
        for j in range(nBins2D[1]):
            x0 = xMinMax[0] + i*binSize_x
            x1 = xMinMax[0] + (i+1)*binSize_x
            y0 = mag_range[0] + j*binSize_y
            y1 = mag_range[0] + (j+1)*binSize_y

            xx[i] = 0.5*(x0+x1)
            yy[j] = 0.5*(y0+y1)

            # select in bin (at sP.redshift, e.g. z=0)
            w = np.where( (sim_xvals_from > x0) & (sim_xvals_from <= x1) & 
                          (sim_colors_from > y0) & (sim_colors_from <= y1) )

            counts[i,j] = len(w[0])
            if counts[i,j] == 0:
                continue

            # arrow end points are 2d bin centers
            arrows_end_x[i,j] = 0.5*(x0+x1)
            arrows_end_y[i,j] = 0.5*(y0+y1)

            # arrow start points are median ending (Mstar,color) values of members of this bin
            # at toRedshift, e.g. where did the z=0 occupants of this bin come from?
            arrows_start_x[i,j] = np.median(sim_xvals_to[w])
            arrows_start_y[i,j] = np.median(sim_colors_to[w])

    # delta vectors
    delta_x = arrows_end_x - arrows_start_x
    delta_y = arrows_end_y - arrows_start_y

    # smoothing? interpolation? outlier exclusion?
    for iter in range(3):
        for i in range(1,nBins2D[0]-1):
            for j in range(1,nBins2D[1]-1):
                # are all neighbors missing? then set counts such that -arrows- are skipped
                dx_ngb = [delta_x[i-1,j-1],delta_x[i-1,j],delta_x[i-1,j+1],
                          delta_x[i+1,j-1],delta_x[i+1,j],delta_x[i+1,j+1],
                          delta_x[i,j-1],delta_x[i,j+1]]
                dy_ngb = [delta_y[i-1,j-1],delta_y[i-1,j],delta_y[i-1,j+1],
                          delta_y[i+1,j-1],delta_y[i+1,j],delta_y[i+1,j+1],
                          delta_y[i,j-1],delta_y[i,j+1]]

                dx_ngood = np.count_nonzero( np.isfinite(dx_ngb) )
                dy_ngood = np.count_nonzero( np.isfinite(dy_ngb) )
                if dx_ngood == 0 and dy_ngood == 0 and counts[i,j] >= 0:
                    counts[i,j] = -1

    for iter in range(10):
        for i in range(1,nBins2D[0]-1):
            for j in range(1,nBins2D[1]-1):
                # is the current pixel missing (nan)? if so, make a bilinear interpolation from the
                # four immediate neighbors, so long as >=3 are non-nan, to fill in this missing pt
                if np.isnan(delta_x[i,j]):
                    ngb = [delta_x[i-1,j],delta_x[i+1,j],delta_x[i,j-1],delta_x[i,j+1]]
                    ngood = np.count_nonzero( np.isfinite(ngb) )
                    if ngood >= 3:
                        delta_x[i,j] = np.nanmean(ngb)

                if np.isnan(delta_y[i,j]):
                    ngb = [delta_y[i-1,j],delta_y[i+1,j],delta_y[i,j-1],delta_y[i,j+1]]
                    ngood = np.count_nonzero( np.isfinite(ngb) )
                    if ngood >= 3:
                        delta_y[i,j] = np.nanmean(ngb)

    # (A): draw individual arrows
    if arrowMethod in ['arrow','comp']:
        for i in range(nBins2D[0]):
            for j in range(nBins2D[1]):
                if counts[i,j] < minCount:
                    continue

                # http://matplotlib.org/examples/pylab_examples/fancyarrow_demo.html
                # http://matplotlib.org/devdocs/api/patches_api.html#matplotlib.patches.FancyArrowPatch
                posA = [arrows_start_x[i,j], arrows_start_y[i,j]]
                posB = [arrows_start_x[i,j]+delta_x[i,j], arrows_start_y[i,j]+delta_y[i,j]]
                arrowstyle = 'fancy, head_width=12, head_length=12, tail_width=20'
                #arrowstyle = 'simple, head_width=12, head_length=12, tail_width=4'

                # alternating color by row (in color)
                c = '#555555' if j % 2 == 0 else '#68af5a'
                p = FancyArrowPatch(posA=posA, posB=posB, arrowstyle=arrowstyle,
                                     alpha=arrowAlpha, color=c)
                ax.add_artist(p)

    # (B): quiver
    if arrowMethod == 'quiver':
        q = ax.quiver(arrows_start_x, arrows_start_y, delta_x, delta_y, 
                      color='black', angles='xy', pivot='tail') # mid,head,tail

    # (C): draw streamlines using 2d vector field
    if arrowMethod in ['stream','comp']:

        # image gives stellar mass growth rate (e.g. dex/Gyr) or color change (e.g. mag/Gyr)
        #delta_mstar_dex_per_Gyr = delta_x.T / (sP.tage-ageTo)
        #delta_diag_per_Gyr = np.sqrt( delta_x**2.0 + delta_y**2.0 ).T / (sP.tage-ageTo)
        delta_mag_per_Gyr = delta_y.T / (sP.tage-ageTo)

        cmap = loadColorTable('jet', plawScale=1.0, fracSubset=[0.15,0.95])
        img = ax.imshow(delta_mag_per_Gyr, extent=[xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]], 
                        alpha=1.0, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap,
                        vmin=-0.06, vmax=0.1)

        # colorbar
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(img, cax=cax, drawedges=False)

        color2 = 'black'
        clabel = 'Rate of (%s-%s) Evolution [ mag / Gyr ]' % (bands[0],bands[1])
        cb.ax.set_ylabel(clabel, color=color2)
        cb.outline.set_edgecolor(color2)
        cb.ax.yaxis.set_tick_params(color=color2)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color2)

        # bugfix for alpha<1 striping in colorbar
        #cb.solids.set_rasterized(True)
        #cb.solids.set_edgecolor("face")

        # do stream plot
        ax.streamplot(xx, yy, delta_x.T, delta_y.T, 
            density=[1.0,1.0], linewidth=2.0, arrowsize=1.4, color='black')

    # single box showing grid size
    if 0:
        box_x0 = xx[-3] - binSize_x/2
        box_x1 = xx[-2] - binSize_x/2
        box_y0 = yy[-2] - binSize_y/2
        box_y1 = yy[-1] - binSize_y/2

        ax.plot( [box_x0,box_x1,box_x1,box_x0,box_x0], [box_y0,box_y0,box_y1,box_y1,box_y0], ':', 
            color='#000000', alpha=0.6)

    # full box grid
    for i in range(nBins2D[0]):
        box_x0 = xx[i] - binSize_x/2
        box_x1 = box_x0 + binSize_x

        for j in range(nBins2D[1]):
            box_y0 = yy[j] - binSize_y/2
            box_y1 = box_y0 + binSize_y

            ax.plot( [box_x0,box_x1,box_x1,box_x0,box_x0], [box_y0,box_y0,box_y1,box_y1,box_y0], '-', 
                color='#000000', alpha=0.05, linewidth=0.5)

    # contours
    extent = [xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]]
    cLevels = [0.25,0.5,0.90]
    cAlphas = [0.1,0.25,0.4]

    sim_colors_1d = np.squeeze( sim_colors_evo[:,rndProjInd,zIndFrom] )
    xx, yy, kde_sim = calcMstarColor2dKDE(bands, sim_xvals, sim_colors_1d, xMinMax, mag_range, 
                                          sP=sP, simColorsModel=simColorsModel)

    for k in range(kde_sim.shape[0]):
        kde_sim[k,:] /= kde_sim[k,:].max() # by column normalization

    for k, cLevel in enumerate(cLevels):
        ax.contour(xx, yy, kde_sim, [cLevel], colors=[contourColor], 
                   alpha=cAlphas[k], linewidths=contourLw, extent=extent)

    # finish plot and save
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(facecolor=fig.get_facecolor())
        else:
            fig.savefig('arrows2d_z=%.1f_%s_%s_%s_%s_%s.pdf' % \
                (toRedshift,'-'.join(bands),simColorsModel,xQuant,cenSatSelect,minCount), 
                facecolor=fig.get_facecolor())
        plt.close(fig)

def colorTransitionTimescale(sP, cenSatSelect='cen', simColorsModel=defSimColorModel, pStyle='white'):
    """ Plot the distribution of 'color transition' timescales (e.g. Delta_t_green). """
    assert cenSatSelect in ['all', 'cen', 'sat']

    # hard-coded config
    bands = ['g','r']
    mag_range = bandMagRange(bands, tight=True)

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    xMinMax = [0,10] # Gyr
    yMinMax = [0,1] # histo todo
    xlabel = '$\Delta t$ [ Gyr ]'
    ylabel = 'PDF $\int=1$'

    # load/calculate evolution of simulation colors, cached in sP.data
    if 'sim_colors_evo' in sP.data:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = \
          sP.data['sim_colors_evo'], sP.data['shID_evo'], sP.data['subhalo_ids'], sP.data['snaps']
    else:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = \
          calcColorEvoTracks(sP, bands=bands, simColorsModel=simColorsModel)
        sP.data['sim_colors_evo'], sP.data['shID_evo'], sP.data['subhalo_ids'], sP.data['snaps'] = \
          sim_colors_evo, shID_evo, subhalo_ids, snaps

    redshifts = snapNumToRedshift(sP, snaps)
    ages = sP.units.redshiftToAgeFlat(redshifts)

    # processing at every snapshot we have calculated colors
    for snapInd, snap in enumerate(snaps):
        print(snapInd, snap, redshifts[snapInd])

        # load the corresponding mstar values at this snapshot
        sP.setSnap(snap)
        mstar2_log, _, xMinMax, _ = simSubhaloQuantity(sP, 'mstar2_log', clean)

        # subhaloIDs in the color evo tracks (can index subhalos in groupcat at this snap)
        colorSHIDs_thisSnap = shID_evo[:,snapInd]

        # (A) two constant cuts for edges of red and blue populations

        # (B) two constant cuts w/ redshift evolution

        # (C) non-constant (e.g. linear or curvy) cut from literature?

        # (D) edges of double-gaussian fits at this snapshot

        import pdb; pdb.set_trace()

    # two timescales:
    #  (1) actual physical crossing from blue -> red
    #  (2) statistical!!, e.g. frequency/fraction of galaxies which leave blue at each time
    # for each, to define the measurement method requires the cut in the color-mass plane
    #  #1 requires two cuts, upper and lower, while #2 requires one cut
    # each cut should be a [optionally Mstar-dependent] (g-r) color value?
    #  [redshift-dependent]!?
    # check Schwinski, Trayford
    # could use any obs (e.g. Baldry) cuts if they exist in (g-r)
    # otherwise, adopt the general philosophy and define our own cuts by likewise fitting to the 
    #  color distributions as a function of Mstar/redshift

    import pdb; pdb.set_trace()

    # start plot
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
    ax = fig.add_subplot(111, axisbg=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # todo

    # finish plot and save
    fig.tight_layout()
    fig.savefig('figure9_%s_%s_%s.pdf' % (sP.simName,cenSatSelect,simColorsModel))
    plt.close(fig)

def colorMassPlaneFits(sP, bands=['g','r'], cenSatSelect='all', simColorsModel=defSimColorModel, pStyle='white'):
    """ Plot double gaussian fits in the color-mass plane with different methods. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    mag_range = [-0.5, 1.5] #bandMagRange(bands, tight=False)
    yMinMax   = [0.0, 4.0]
    mMinMax   = [9.0, 12.0] # log Mstar
    sMinMax   = [0.0, (mag_range[1]-mag_range[0])/10] # sigma

    lw = 2.5
    xlabel = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
    ylabel = 'PDF' 
    mlabel = '%.2f < M$_{\\rm \star}$ < %.2f'
    sizefac = 1.0 if not clean else sfclean

    # load
    #fits_obs = characterizeColorMassPlane(None, bands=bands, cenSatSelect=cenSatSelect, 
    #                                      simColorsModel=simColorsModel)
    #import pdb; pdb.set_trace()
    fits_obs = None
    fits = characterizeColorMassPlane(sP, bands=bands, cenSatSelect=cenSatSelect, 
                                      simColorsModel=simColorsModel)

    masses = fits['mStar'] # bin centers
    methods = ['A','B','C'] # plot each

    # (A) start plot, debugging double gaussians (two plots of 10 panels each to cover 20 bins)
    nCols = 2 * 0.6
    nRows = 5 * 0.6 # 0.6=visual adjust fac

    for iterNum in [0,1]:
        fig = plt.figure(figsize=(figsize[0]*nCols*sizefac,figsize[1]*nRows*sizefac),facecolor=color1)

        xx = np.linspace(mag_range[0], mag_range[1], 100)

        # loop over half the mass bins, with a stride of two (one panel per mass bin)
        for i, mass in enumerate(masses[iterNum::2]):
            # get index of this mass bin in params
            data_index = np.where(masses == mass)[0][0]
            print(iterNum,i,data_index,mass)

            # start plot
            ax = fig.add_subplot(5,2,i+1, axisbg=color1)

            setAxisColors(ax, color2)
            ax.set_xlim(mag_range)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # load results for a particular method, including the model
            for j, method in enumerate(methods):
                params = fits['%s_params' % method]

                (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,data_index]

                y1 = A1 * np.exp( - (xx - mu1)**2.0 / (2.0 * sigma1**2.0) ) # blue
                y2 = A2 * np.exp( - (xx - mu2)**2.0 / (2.0 * sigma2**2.0) ) # red

                ax.plot(xx,y1,linestyles[j],color='blue',alpha=0.8,lw=lw)
                ax.plot(xx,y2,linestyles[j],color='red',alpha=0.8,lw=lw)

                # obs
                if fits_obs is not None:
                    params = fits_obs['%s_params' % method]

                    (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,data_index]

                    y1 = A1 * np.exp( - (xx - mu1)**2.0 / (2.0 * sigma1**2.0) ) # blue
                    y2 = A2 * np.exp( - (xx - mu2)**2.0 / (2.0 * sigma2**2.0) ) # red

                    ax.plot(xx,y1,linestyles[j],color='black',alpha=0.8,lw=lw)
                    ax.plot(xx,y2,linestyles[j],color='gray',alpha=0.8,lw=lw)
            
            # make legend
            sExtra = []
            lExtra = []

            for j, method in enumerate(methods):
                sExtra.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[j]) )
                lExtra.append( 'Method (%s)' % method )

            sExtra.append(plt.Line2D( (0,1), (0,0), color='white', marker='', linestyle=linestyles[j]))
            lExtra.append(mlabel % (mass,mass+fits['binSizeMass']))

            legend2 = ax.legend(sExtra, lExtra, loc='best')

        # finish plot and save
        fig.tight_layout()
        fig.savefig('colorMassPlaneFits%d_%s_%s_%s_%s.pdf' % \
            (iterNum,sP.simName,'-'.join(bands),cenSatSelect,simColorsModel))
        plt.close(fig)

    # (B,C,D) start plot, sigma/mu/A vs Mstar
    for iterNum in [0,1,2]:
        fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)

        ax = fig.add_subplot(111, axisbg=color1)

        setAxisColors(ax, color2)
        ax.set_xlim(mMinMax)
        ax.set_xlabel('M$_{\star}$ [ log M$_{\\rm sun}$ ]')

        if iterNum == 0:
            ax.set_ylabel('$\sigma$ [standard deviation (%s-%s)]' % (bands[0],bands[1]))
            saveStr = 'Sigma'
            ax.set_ylim([0.0,0.4])
        if iterNum == 1:
            ax.set_ylabel('$\mu$ [mean (%s-%s) peak location]' % (bands[0],bands[1]))
            saveStr = 'Mu'
            ax.set_ylim(mag_range)
        if iterNum == 2:
            ax.set_ylabel('$A$ [peak amplitude (%s-%s)]' % (bands[0],bands[1]))
            saveStr = 'A'
            ax.set_ylim([0.0,3.0])

        val_red  = np.zeros( masses.size, dtype='float32' )
        val_blue = np.zeros( masses.size, dtype='float32' )

        # load results for a particular method, including the model
        for j, method in enumerate(methods):
            params = fits['%s_params' % method]

            for i in range(len(masses)):
                (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,i]

                if iterNum == 0:
                    val_blue[i] = sigma1
                    val_red[i] = sigma2
                if iterNum == 1:
                    val_blue[i] = mu1
                    val_red[i] = mu2
                if iterNum == 2:
                    val_blue[i] = A1
                    val_red[i] = A2

            ax.plot(masses,val_blue,'o'+linestyles[j],color='blue',alpha=0.8,lw=lw)
            ax.plot(masses,val_red,'o'+linestyles[j],color='red',alpha=0.8,lw=lw)

            if fits_obs is not None:
                params = fits_obs['%s_params' % method]

                for i in range(len(masses)):
                    (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,i]

                    if iterNum == 0:
                        val_blue[i] = sigma1
                        val_red[i] = sigma2
                    if iterNum == 1:
                        val_blue[i] = mu1
                        val_red[i] = mu2
                    if iterNum == 2:
                        val_blue[i] = A1
                        val_red[i] = A2

                ax.plot(masses,val_blue,'o'+linestyles[j],color='black',alpha=0.8,lw=lw)
                ax.plot(masses,val_red,'o'+linestyles[j],color='gray',alpha=0.8,lw=lw) 
            
        # make legend
        sExtra = [ plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[j]) \
                   for j in range(len(methods)) ]
        lExtra = [ 'Method (%s)' % m for m in methods ]
        legend2 = ax.legend(sExtra, lExtra, loc='best')

        # finish plot and save
        fig.tight_layout()
        fig.savefig('colorMassPlane-%s_%s_%s_%s_%s.pdf' % \
            (saveStr,sP.simName,'-'.join(bands),cenSatSelect,simColorsModel))
        plt.close(fig)

    # (E) start plot, red fraction (ratio of areas/heights?)
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)

    ax = fig.add_subplot(111, axisbg=color1)

    setAxisColors(ax, color2)
    ax.set_xlim(mMinMax)
    ax.set_xlabel('M$_{\star}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('Red Fraction [in (%s-%s) double Gaussian fit]' % (bands[0],bands[1]))
    ax.set_ylim([0.0,1.0])

    fraction_red = np.zeros( masses.size, dtype='float32' )

    # load results for a particular method, including the model
    for j, method in enumerate(methods):
        params = fits['%s_params' % method]

        for i in range(len(masses)):
            (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,i]
            integral_1 = A1 * sigma1 * np.sqrt(2*np.pi)
            integral_2 = A2 * sigma2 * np.sqrt(2*np.pi)
            print(method,i,masses[i],integral_1+integral_2)
            fraction_red[i] = (A2*sigma2) / (A1*sigma1+A2*sigma2) # area = A*sigma*sqrt(2pi)

        ax.plot(masses,fraction_red,linestyles[j],color='black',alpha=0.8,lw=lw)

        if fits_obs is not None:
            params = fits_obs['%s_params' % method]

            for i in range(len(masses)):
                (A1, mu1, sigma1, A2, mu2, sigma2) = params[:,i]
                integral_1 = A1 * sigma1 * np.sqrt(2*np.pi)
                integral_2 = A2 * sigma2 * np.sqrt(2*np.pi)
                print(method,i,masses[i],integral_1+integral_2)
                fraction_red[i] = (A2*sigma2) / (A1*sigma1+A2*sigma2) # area = A*sigma*sqrt(2pi)

            ax.plot(masses,fraction_red,linestyles[j],color='green',alpha=0.8,lw=lw)
        
    # make legend
    sExtra = [ plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[j]) \
               for j in range(len(methods)) ]
    lExtra = [ 'Method (%s)' % m for m in methods ]
    legend2 = ax.legend(sExtra, lExtra, loc='best')

    # finish plot and save
    fig.tight_layout()
    fig.savefig('colorMassPlane-RedFrac_%s_%s_%s_%s.pdf' % \
        (sP.simName,'-'.join(bands),cenSatSelect,simColorsModel))
    plt.close(fig)

def plots():
    """ Driver (exploration 2D histograms). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = 'color'
    ySpec  = [ ['g','r'], defSimColorModel ] # bands, simColorModel
    xQuant = 'mstar2_log' # mstar2_log, ssfr, M_BH_actual
    cs     = 'median_nan'
    cenSatSelects = ['cen'] #['cen','sat','all']

    quants = quantList()

    for sP in sPs:
        for css in cenSatSelects:

            pdf = PdfPages('galaxyColor_2dhistos_%s_%s_%s_%s_%s.pdf' % (sP.simName,yQuant,xQuant,cs,css))

            for cQuant in quants:
                quantHisto2D(sP, pdf, yQuant=yQuant, ySpec=ySpec, xQuant=xQuant, 
                             cenSatSelect=css, cQuant=cQuant, cStatistic=cs)

            pdf.close()

def plots2():
    """ Driver (exploration 1D slices). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    xQuant = 'color'
    xSpec  = [ ['g','r'], defSimColorModel ] # bands, simColorModel
    sQuant = 'mstar2_log'
    sRange = [10.4,10.6]
    cenSatSelects = ['cen']

    quants = quantList(wCounts=False,wTr=False)
    quantsTr = quantList(wCounts=False,onlyTr=True)

    for css in cenSatSelects:
        pdf = PdfPages('galaxyColor_1Dslices_%s_%s_%s-%.1f-%.1f_%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))

        # all quantities on one multi-panel page:
        quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=quants, sQuant=sQuant, 
                     sRange=sRange, cenSatSelect=css, )
        quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=quantsTr, sQuant=sQuant, 
                     sRange=sRange, cenSatSelect=css)

        # one page per quantity:
        for yQuant in quants + quantsTr:
            quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=[yQuant], sQuant=sQuant, 
                         sRange=sRange, cenSatSelect=css)

        pdf.close()

def plots3():
    """ Driver (exploration median trends). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    #sPs.append( simParams(res=1820, run='illustris', redshift=0.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    bands = ['g','r']
    xQuant = 'mstar1_log'
    cenSatSelects = ['cen']

    quants = quantList(onlyBH=True)

    # make plots
    for css in cenSatSelects:
        pdf = PdfPages('medianQuants_%s_%s_%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,css))

        # all quantities on one multi-panel page:
        quantMedianVsSecondQuant(sPs, pdf, yQuants=quants, xQuant=xQuant, cenSatSelect=css)

        # one page per quantity:
        for yQuant in quants:
            quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css)

        pdf.close()
    
def paperPlots():
    """ Construct all the final plots for the paper. """
    import plot.globalComp
    plot.globalComp.clean = True
    import plot.config
    plot.config.clean = True

    L75   = simParams(res=1820,run='tng',redshift=0.0)
    L205  = simParams(res=2500,run='tng',redshift=0.0)
    L75FP = simParams(res=1820,run='illustris',redshift=0.0)

    dust_A = 'p07c_nodust'
    dust_B = 'p07c_cf00dust'
    dust_C = 'p07c_cf00dust_res_conv_ns1_rad30pkpc' # one random projection per subhalo
    dust_C_all = 'p07c_cf00dust_res_conv_ns1_rad30pkpc_all' # all projections shown

    # figure 1, (g-r) 1D color PDFs in six mstar bins (3x2) Illustris vs TNG100 vs SDSS
    if 0:
        sPs = [L75FP, L75] # order reversed to put TNG100 on top, colors hardcoded
        dust = dust_C_all

        pdf = PdfPages('figure1_%s_%s.pdf' % ('_'.join([sP.simName for sP in sPs]),dust))
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=[dust])
        pdf.close()

    # figure 2, 2x2 grid of different 2D color PDFs, TNG100 vs SDSS
    if 0:
        sPs = [L75]
        dust = dust_C

        pdf = PdfPages('figure2_%s_%s.pdf' % (sPs[0].simName,dust))
        galaxyColor2DPDFs(sPs, pdf, simColorsModel=dust)
        pdf.close()

    # figure 3, stellar ages and metallicities vs mstar (2x1 in a row)
    if 1:
        sPs = [L75, L205] # L75FP
        simRedshift = 0.1

        pdf = PdfPages('figure3a_stellarAges_%s.pdf' % '_'.join([sP.simName for sP in sPs]))
        plot.globalComp.stellarAges(sPs, pdf, simRedshift=simRedshift, centralsOnly=True)
        pdf.close()
        pdf = PdfPages('figure3b_massMetallicityStars_%s.pdf' % '_'.join([sP.simName for sP in sPs]))
        plot.globalComp.massMetallicityStars(sPs, pdf, simRedshift=simRedshift)
        pdf.close()

    # figure 4: fullbox demonstratrion projections
    if 0:
        # render each fullbox image used in the composite
        for part in [0,1,2,3,4]:
            vis.boxDrivers.TNG_colorFlagshipBoxImage(part=part)

    # figure 5, grid of L205_cen 2d color histos vs. several properties (2x3)
    if 0:
        sP = L205
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        params = {'bands':['g','r'], 'cenSatSelect':'cen', 'cStatistic':'median_nan'}

        pdf = PdfPages('figure5_%s.pdf' % sP.simName)
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, pdf, cQuant='ssfr', fig_subplot=[fig,321], **params)
        quantHisto2D(sP, pdf, cQuant='Z_gas', fig_subplot=[fig,322], **params)
        quantHisto2D(sP, pdf, cQuant='fgas2', fig_subplot=[fig,323], **params)
        quantHisto2D(sP, pdf, cQuant='stellarage', fig_subplot=[fig,324], **params)
        quantHisto2D(sP, pdf, cQuant='bmag_2rhalf_masswt', fig_subplot=[fig,325], **params)
        quantHisto2D(sP, pdf, cQuant='pratio_halo_masswt', fig_subplot=[fig,326], **params)
        pdf.close()

    # figure 6: slice through 2d histo (one property)
    if 0:
        sPs = [L75, L205]
        xQuant = 'color'
        xSpec  = [ ['g','r'], defSimColorModel ] # bands, simColorModel
        sQuant = 'mstar2_log'
        sRange = [10.4,10.6]
        css = 'cen'
        quant = 'pratio_halo_masswt'

        pdf = PdfPages('figure6_%s_slice_%s_%s-%.1f-%.1f_%s.pdf' % \
            ('_'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))
        quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=[quant], sQuant=sQuant, 
                     sRange=sRange, cenSatSelect=css)
        pdf.close()

    # figure 7: BH cumegy vs mstar, model line on top (eddington transition to low-state?)
    if 0:
        sPs = [L75, L205]
        xQuant = 'mstar2_log'
        yQuant = 'BH_CumEgy_ratio'
        css = 'cen'

        pdf = PdfPages('figure7_medianTrend_%s_%s-%s_%s.pdf' % \
            ('_'.join([sP.simName for sP in sPs]),xQuant,yQuant,css))
        quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css)
        pdf.close()    

    # figure 8: flux arrows in color-mass plane
    if 0:
        sP = L75
        dust = dust_C
        css = 'cen'
        minCount = 1

        toRedshift = 0.5
        arrowMethod = 'arrow'
        pdf = PdfPages('figure8a_%s_toz-%.1f_%s_%s_min-%d_%s.pdf' % \
            (sP.simName,toRedshift,css,dust,minCount,arrowMethod))
        colorFluxArrows2DEvo(sP, pdf, bands=['g','r'], toRedshift=toRedshift, cenSatSelect=css, 
                             minCount=minCount, simColorsModel=dust, arrowMethod=arrowMethod)
        pdf.close()

        toRedshift = 0.3
        arrowMethod = 'stream'
        pdf = PdfPages('figure8b_%s_toz-%.1f_%s_%s_min-%d_%s.pdf' % \
            (sP.simName,toRedshift,css,dust,minCount,arrowMethod))
        colorFluxArrows2DEvo(sP, pdf, bands=['g','r'], toRedshift=toRedshift, cenSatSelect=css, 
                             minCount=minCount, simColorsModel=dust, arrowMethod=arrowMethod)
        pdf.close()

    # figure 9: timescale histogram for color transition
    if 0:
        sP = L75
        css = 'cen'
        dust = dust_C

        colorTransitionTimescale(sP, cenSatSelect=css, simColorsModel=dust)

    # figure 10: few N characteristic evolutionary tracks through color-mass 2d plane

    # figure 11: stellar image stamps of galaxies (time evolution of above tracks)

    # figure Fig 12: double gaussian fits, [peak/scatter vs Mstar], red fraction (e.g. Baldry Figs. 5, 6, 8)

    # figure Fig 13: distribution of initial M* when entering red sequence (crossing color cut) (Q1)

    # figure Fig 14: as a function of M*ini, the Delta_M* from t_{red,ini} to z=0 (Q2)

    # figure Fig 15: as a function of M*(z=0), the t_{red,ini} PDF (Q3)

    # appendix figure 1, viewing angle variation (1 panel)
    if 0:
        viewingAngleVariation()

    # appendix figure 2, dust model dependence (1x3 1D histos in a column)
    if 0:
        sPs = [L75]
        dusts = [dust_C_all, dust_C, dust_B, dust_A]
        massBins = ( [9.5,10.0], [10.0,10.5], [10.5,11.0] )

        pdf = PdfPages('appendix2.pdf')
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=dusts, stellarMassBins=massBins)
        pdf.close()

    # appendix figure 3, resolution convergence (1x3 1D histos in a column)
    if 0:
        L75n910 = simParams(res=910,run='tng',redshift=0.0)
        L75n455 = simParams(res=455,run='tng',redshift=0.0)
        sPs = [L75, L75n910, L75n455]
        dust = dust_C_all
        massBins = ( [9.5,10.0], [10.0,10.5], [10.5,11.0] )

        pdf = PdfPages('appendix3_%s.pdf' % dust)
        galaxyColorPDF(sPs, pdf, bands=['g','r'], simColorsModels=[dust], stellarMassBins=massBins)
        pdf.close()

    # appendix figure 4, 2d density histos (3x1 in a row) all_L75, cen_L75, cen_L205
    if 0:
        figsize_loc = [figsize[0]*3*0.7, figsize[1]*1*0.75]

        pdf = PdfPages('appendix4.pdf')
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(L75, pdf, ['g','r'], cenSatSelect='all', cQuant=None, fig_subplot=[fig,131])
        quantHisto2D(L75, pdf, ['g','r'], cenSatSelect='cen', cQuant=None, fig_subplot=[fig,132])
        quantHisto2D(L205, pdf, ['g','r'], cenSatSelect='cen', cQuant=None, fig_subplot=[fig,133])
        pdf.close()

    # testing
    if 0:
        params = {'bands':['g','r'], 'cenSatSelect':'cen', 'cStatistic':'median_nan'}

        pdf = PdfPages('figure_test.pdf')
        fig = plt.figure(figsize=figsize)
        quantHisto2D(L75, pdf, cQuant='ssfr', fig_subplot=[fig,111], **params)
        pdf.close()
