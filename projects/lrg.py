"""
projects/lrg.py
  Plots: LRG CGM paper (TNG50).
  in prep.
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, colorConverter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d
from os.path import isfile

from util import simParams
from util.helper import running_median, logZeroNaN, loadColorTable
from util.voronoi import voronoiThresholdSegmentation
from cosmo.util import subboxSubhaloCat
from plot.config import *
from plot.general import plotStackedRadialProfiles1D, plotHistogram1D, plotPhaseSpace2D
from tracer.tracerMC import match3, globalAllTracersTimeEvo
from vis.halo import renderSingleHalo
from projects.oxygen import obsSimMatchedGalaxySamples, obsColumnsDataPlot, obsColumnsDataPlotExtended, \
                            ionTwoPointCorrelation, totalIonMassVsHaloMass, stackedRadialProfiles

def radialResolutionProfiles(sPs, saveName, redshift=0.3, cenSatSelect='cen', 
                             radRelToVirRad=False, haloMassBins=None, stellarMassBins=None):
    """ Plot average/stacked radial gas cellsize profiles in stellar mass bins. Specify one of 
    haloMassBins or stellarMassBins. If radRelToVirRad, then [r/rvir] instead of [pkpc]. """
    from tracer.tracerMC import match3

    # config
    percs = [10,90]
    # # 'Mean' or 'Median' or 'Min' or 'p10', 'Gas_SFReq0' or 'Gas'
    fieldNames = ['Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Mean',
                  'Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_p10']

    # plot setup
    lw = 3.0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    if radRelToVirRad:
        ax.set_xlim([-2.0, 0.0])
        ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
        ax.set_xlabel('Radius / Virial Radius [ log ]')
    else:
        ax.set_xlim([0.0, 3.2])
        ax.set_xlabel('Radius [ log pkpc ]')

    ax.set_ylim([-0.9,1.0])
    ax.set_ylabel('Gas Resolution $r_{\\rm cell}$ [ log kpc ]')

    # init
    colors = []
    rvirs = []

    if haloMassBins is not None:
        massField = 'mhalo_200_log'
        massBins = haloMassBins
    else:
        massField = 'mstar_30pkpc_log'
        massBins = stellarMassBins

    # mark 1 and 2 pkpc
    ax.plot( ax.get_xlim(), np.log10([2.0,2.0]), lw=lw-0.5, color='#cccccc', alpha=0.6 )
    ax.plot( ax.get_xlim(), np.log10([1.0,1.0]), lw=lw-0.5, color='#cccccc', alpha=0.2 )
    ax.plot( ax.get_xlim(), np.log10([0.5,0.5]), lw=lw-0.5, color='#cccccc', alpha=0.8 )
    ax.text( -1.4, np.log10(2.0)-0.025, "2 pkpc", color='#cccccc', va='top', ha='left', 
                            fontsize=20.0, alpha=0.6)
    ax.text( -0.5, np.log10(0.5)+0.02, "500 parsecs", color='#cccccc', va='bottom', ha='left', 
                            fontsize=20.0, alpha=0.8)

    # loop over each fullbox run
    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        sP.setRedshift(redshift)
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii
        rad = sP.groupCat(fieldsSubhalos=['rhalo_200_code'])
        rad = rad[cssInds]

        # load and apply CSS
        for j, fieldName in enumerate(fieldNames):
            if 'p10' in fieldName and sP.res != 2160:
                continue # only add 10th percentile curves for TNG50-1

            ac = sP.auxCat(fields=[fieldName])
            if ac[fieldName] is None: continue
            yy = ac[fieldName]

            # crossmatch 'subhaloIDs' to cssInds
            ac_inds, css_inds = match3( ac['subhaloIDs'], cssInds )
            ac[fieldName] = ac[fieldName][ac_inds,:]
            masses_loc = masses[css_inds]
            rad_loc = rad[css_inds]            

            # loop over mass bins
            for k, massBin in enumerate(massBins):
                if k in [0,1] and sP.res in [540,1080]:
                    continue # only add all 3 massbins for TNG50-1

                # select
                w = np.where( (masses_loc >= massBin[0]) & (masses_loc < massBin[1]) )

                print('%s [%d] %.1f - %.1f : %d' % (sP.simName,k,massBin[0],massBin[1],len(w[0])))
                assert len(w[0])

                # radial bins: normalize to rvir if requested
                avg_rvir_code = np.nanmedian( rad_loc[w] )
                if i == 0: rvirs.append( avg_rvir_code )

                # y-quantity
                yy_local = np.squeeze( yy[w,:] ) # check

                # x-quantity
                if radRelToVirRad:
                    rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rvir_code
                else:
                    rr = ac[fieldName+'_attrs']['rad_bins_pkpc']

                # for low res runs, combine the inner bins which are poorly sampled
                if 0 and sP.res in [540,1080]:
                    nInner = int( 20 / (sP.res/540) )
                    rInner = np.mean( rr[0:nInner] )

                    for dim in range(yy_local.shape[0]):
                        yy_local[dim,nInner-1] = np.nanmedian( yy_local[dim,0:nInner] )
                    yy_local = yy_local[:,nInner-1:]
                    rr = np.hstack( [rInner,rr[nInner:]] )

                # replace zeros by nan so they are not included in percentiles
                yy_local[yy_local == 0.0] = np.nan

                # calculate mean profile and scatter
                if yy_local.ndim > 1:
                    yy_mean = np.nansum( yy_local, axis=0 ) / len(w[0])
                    yp = np.nanpercentile( yy_local, percs, axis=0 )
                else:
                    yy_mean = yy_local # single profile
                    yp = np.vstack( (yy_local,yy_local) ) # no scatter

                # log both axes
                yy_mean = logZeroNaN(yy_mean)
                yp = logZeroNaN(yp)
                rr = np.log10(rr)

                if rr.size > sKn:
                    sKn_loc = sKn+8 if j == 0 else sKn+16
                    yy_mean = savgol_filter(yy_mean,sKn_loc,sKo+1)
                    yp = savgol_filter(yp,sKn_loc,sKo+1,axis=1) # P[10,90]

                # determine color
                if i == 0 and j == 0:
                    c = next(ax._get_lines.prop_cycler)['color']
                    colors.append(c)
                else:
                    c = colors[k]

                # plot median line
                label = '%.1f < $M_{\\rm halo}$ < %.1f' % (massBin[0],massBin[1]) if (i == 0 and j == 0) else ''
                label = '$M_{\\rm halo}$ = %.1f' % (0.5*(massBin[0]+massBin[1])) if (i == 0 and j == 0) else ''
                alpha = 1.0 if j == 0 else 0.3
                linewidth = lw if j == 0 else lw-1
                ax.plot(rr, yy_mean, lw=linewidth, color=c, linestyle=linestyles[i], label=label, alpha=alpha)

                # draw rvir lines (or 300pkpc lines if x-axis is already relative to rvir)
                yrvir = ax.get_ylim()
                yrvir = np.array([ yrvir[0], yrvir[0] + (yrvir[1]-yrvir[0])*0.15]) + 0.05

                if not radRelToVirRad:
                    xrvir = np.log10( [avg_rvir_code, avg_rvir_code] )
                    textStr = 'R$_{\\rm vir}$'
                    yrvir[1] += 0.0 * k
                else:
                    rvir_150pkpc_ratio = sP.units.physicalKpcToCodeLength(150.0) / avg_rvir_code
                    xrvir = np.log10( [rvir_150pkpc_ratio, rvir_150pkpc_ratio] )
                    textStr = '150 kpc'
                    yrvir[1] += 0.0 * (len(massBins)-k)

                if i == 0 and j == 0:
                    ax.plot(xrvir, yrvir, lw=lw*1.5, color=c, alpha=0.1)
                    ax.text(xrvir[0]-0.02, yrvir[0], textStr, color=c, va='bottom', ha='right', 
                            fontsize=20.0, alpha=0.1, rotation=90)

                    # show percentile scatter only for first run
                    ax.fill_between(rr, yp[0,:], yp[-1,:], color=c, interpolate=True, alpha=0.2)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['%s' % sP.simName]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, ncol=2, loc='upper left')

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def _getStackedGrids(sP, ion, haloMassBin, fullDepth, radRelToVirRad, ConfigLan=False, indiv=False, axesSets=[[0,1]]):
    """ Return (and cache) a concatenated {N_ion,dist} set of pixels for all halos in the given mass bin. 
    Helper for the following two functions. """

    # grid config
    run        = sP.run
    res        = sP.res
    redshift   = sP.redshift
    method     = 'sphMap'
    nPixels    = [1000,1000]
    axes       = [0,1]
    rotation   = 'edge-on'
    size       = 400.0
    sizeType   = 'kpc'

    if fullDepth:
        # global accumulation with appropriate depth along the projection direction
        method = 'sphMap_global'
        dv = 500.0 # +/- km/s (Zahedy / COS-LRG config) or +/- 600 km/s (Werk / COS-Halos config)
        depth_code_units = (2*dv) / sP.units.H_of_a # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

    eStr = ''
    if ConfigLan:
        size = 2000.0
        nPixels = [2000,2000]
        rotation = None # random

        dv = 1000.0 # tbd to match the stacking procedure
        depth_code_units = (2*dv) / sP.units.H_of_a # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

        eStr = '_2k'

    # quick caching
    cacheSaveFile = sP.derivPath + 'cache/ionColumnsVsImpact2D_%s_%d_%s_%.1f-%.1f_rvir=%s_fd=%s%s_a%d.hdf5' % \
      (sP.simName,sP.snap,ion,haloMassBin[0],haloMassBin[1],radRelToVirRad,fullDepth,eStr,len(axesSets))

    if isfile(cacheSaveFile):
        # load previous result
        with h5py.File(cacheSaveFile,'r') as f:
            dist_global = f['dist_global'][()]
            grid_global = f['grid_global'][()]
        print('Loaded: [%s]' % cacheSaveFile)
    else:
        # get halo IDs in mass bin (centrals only by definition)
        gc = sP.groupCat(fieldsSubhalos=['mhalo_200_log','rhalo_200_code'])

        with np.errstate(invalid='ignore'):
            subInds = np.where( (gc['mhalo_200_log'] > haloMassBin[0]) & (gc['mhalo_200_log'] < haloMassBin[1]) )[0]

        # load grids
        dist_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)*len(axesSets)), dtype='float32' )
        grid_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)*len(axesSets)), dtype='float32' )

        for i, hInd in enumerate(subInds):
            print(i, len(subInds), hInd)
            class plotConfig:
                saveFilename = 'dummy'

            # compute impact parameter for every pixel
            pxSize = size / nPixels[0] # pkpc

            xx, yy = np.mgrid[0:nPixels[0], 0:nPixels[1]]
            xx = xx.astype('float64') - nPixels[0]/2
            yy = yy.astype('float64') - nPixels[1]/2
            dist = np.sqrt( xx**2 + yy**2 ) * pxSize

            if radRelToVirRad:
                dist /= gc['rhalo_200_code'][subInds[i]]

            for j, axes in enumerate(axesSets):
                # loop over projection directions and render
                panels = [{'partType':'gas', 'partField':ion, 'valMinMax':[-1.4,0.2]}]
                grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                # flatten and stamp
                dist_global[:,i*len(axesSets)+j] = dist.ravel()
                grid_global[:,i*len(axesSets)+j] = grid.ravel()

        # save cache
        with h5py.File(cacheSaveFile,'w') as f:
            f['dist_global'] = dist_global
            f['grid_global'] = grid_global

        print('Saved: [%s]' % cacheSaveFile)

    if not indiv:
        # flatten
        dist_global = dist_global.ravel()
        grid_global = grid_global.ravel()

    return dist_global, grid_global

def ionColumnsVsImpact2D(sP, haloMassBin, ion, radRelToVirRad=False, ycum=False, fullDepth=False):
    """ Use gridded N_ion maps to plot a 2D pixel histogram of (N_ion vs impact parameter). """

    ylim = [11.0, 17.0] # N_ion
    if 'MHI' in ion:
        ylim = [16.0, 22.0]

    xlog       = False
    nBins      = 100
    ctName     = 'viridis'
    medianLine = True
    colorMed   = 'black'

    if ycum:
        cMinMax = [-2.0, 0.0] # log fraction
        clabel  = "Fraction of Sightlines (at this b) with >= N"
    else:
        cMinMax = [-3.0, -1.8] # log fraction
        #if 'MHI' in ion: cMinMax = [-3.0, -1.4]
        clabel  = 'Conditional Covering Fraction = N [log]'

    # load projected column density grids of all halos
    dist_global, grid_global = _getStackedGrids(sP, ion, haloMassBin, fullDepth, radRelToVirRad)

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    if xlog:
        if radRelToVirRad:
            ax.set_xlim([-2.0, 0.0])
            ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
            ax.set_xlabel('Impact Parameter / Virial Radius [ log ]')
        else:
            ax.set_xlim([0.5, 2.5])
            ax.set_xlabel('Impact Parameter [ log pkpc ]')
    else:
        if radRelToVirRad:
            ax.set_xlim([0.0, 1.0])
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel('Impact Parameter / Virial Radius')
        else:
            ax.set_xlim([0, 200])
            ax.set_xlabel('Impact Parameter [ pkpc ]')

    ax.set_ylim(ylim)
    ax.set_ylabel('N$_{\\rm %s}$ [ log cm$^{-2}$ ]' % ion)

    # plot
    w = np.where( (dist_global > 0) & np.isfinite(grid_global) )

    dist_global = dist_global[w] # pkpc or r/rvir
    grid_global = grid_global[w] # log cm^2

    if xlog:
        dist_global = np.log10(dist_global)

    sim_cvals = np.zeros( dist_global.size, dtype='float32' ) # unused currently

    # histogram 2d
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()

    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xlim[0],xlim[1],ylim[0],ylim[1]]

    cc, xBins, yBins, inds = binned_statistic_2d(dist_global, grid_global, sim_cvals, 'count', 
                                                 bins=nBins2D, range=[xlim,ylim])

    cc = cc.T # imshow convention

    # histogram again, this time extending the y-axis bounds over all values, such that every pixel is counted
    # required for proper normalizations
    nn, _, _, _ = binned_statistic_2d(dist_global, grid_global, sim_cvals, 'count', 
                                                 bins=nBins2D, range=[xlim,[grid_global.min(),grid_global.max()]])
    nn = nn.T

    # normalize each column separately: cc value becomes [fraction of sightlines, at this impact parameter, with this column]
    with np.errstate(invalid='ignore'):
        totVals = np.nansum(nn, axis=0)
        totVals[totVals == 0] = 1
        cc /= totVals[np.newaxis, :]

    # cumulative y? i.e. each cc value becomes [fraction of sightlines, at this impact parameter, with >= this column]
    if ycum:
        cc = np.cumsum(cc[::-1,:], axis=0)[::-1,:] # flips give >= this column, otherwise is actually <= this column

    # units and colormap
    if not ycum:
        cc2d = logZeroNaN(cc)
    else:
        # linear fraction for cumulative version
        cc2d = cc
        cMinMax = [0,0.8]

    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)

    cmap = loadColorTable(ctName, numColors=8 if ycum else None)
    cc2d_rgb = cmap(norm(cc2d))

    # mask empty bins to white
    #cc2d_rgb[(cc == 0),:] = colorConverter.to_rgba('white')

    plt.imshow(cc2d_rgb, extent=extent, origin='lower', interpolation='nearest', aspect='auto', 
               cmap=cmap, norm=norm)

    if medianLine:
        binSizeMed = (xlim[1]-xlim[0]) / nBins * 2

        xm, ym, sm, pm = running_median(dist_global,grid_global,binSize=binSizeMed,percs=[16,50,84])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)
            sm = savgol_filter(sm,sKn,sKo)
            pm = savgol_filter(pm,sKn,sKo,axis=1)

        ax.plot(xm, ym, '-', color=colorMed, lw=lw, label='median')
        ax.plot(xm, pm[0,:], ':', color=colorMed, lw=lw, label='P[10,90]')
        ax.plot(xm, pm[-1,:], ':', color=colorMed, lw=lw)

    # add obs data points
    if 'Mg II' in ion:
        zahedy_lower_d = [18.13, 83.17, 91.08, 91.08]
        zahedy_lower_N = [13.61, 13.95, 14.07, 13.73]
        zahedy_upper_d = [18.82, 31.79, 76.86, 78.00, 115.88, 148.98]
        zahedy_upper_N = [12.38, 11.88, 12.00, 12.24, 11.97,  12.50]
        zahedy_detection_d = [45.89, 102.03, 134.85]
        zahedy_detection_N = [12.38, 12.71, 13.01]

        coshalos_upper_d = [78.5, 125.8]
        coshalos_upper_N = [11.84, 11.70]
        coshalos_detection_d = [83.6, 93.6, 98.6, 108.5, 106.6, 140.0, 156.0, 160.1]
        coshalos_detection_N = [13.97, 13.80, 12.39, 12.43, 12.70, 12.85, 12.22, 12.60]
        coshalos_bar_d = [47.1, 47.1]
        coshalos_bar_N = [13.45, 15.92]

        markerColor  = 'white'
        markerColor2 = 'orange'
        markersize   = 6.0

        ax.plot(zahedy_detection_d, zahedy_detection_N, 'o', color=markerColor, markersize=markersize)
        ax.plot(zahedy_upper_d, zahedy_upper_N, 'v', color=markerColor, markersize=markersize)
        ax.plot(zahedy_lower_d, zahedy_lower_N, '^', color=markerColor, markersize=markersize)

        ax.plot(coshalos_detection_d, coshalos_detection_N, 'o', color=markerColor2, markersize=markersize)
        ax.plot(coshalos_upper_d, coshalos_upper_N, 'v', color=markerColor2, markersize=markersize)
        ax.plot(coshalos_bar_d, coshalos_bar_N, '-', color=markerColor2, lw=lw-1, alpha=0.3)

        # legend
        handles = [plt.Line2D( (0,0),(0,0),color=markerColor,lw=lw-1,linestyle='-',marker='o'),
                   plt.Line2D( (0,0),(0,0),color=markerColor2,lw=lw-1,linestyle='-',marker='o'),]
        labels  = ['COS-LRG (Zahedy+ 2019)', 'COS-Halos (Werk+ 2013)']
        legend = ax.legend(handles, labels, loc='upper right', handlelength=0)
        legend.get_texts()[0].set_color(markerColor)
        legend.get_texts()[1].set_color(markerColor2)

    # colorbar and finish plot
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)
    #ax.set_title('%.1f < M$_{\\rm halo}$ [log M$_\odot$] < %.1f' % (haloMassBin[0],haloMassBin[1]))

    fig.tight_layout()
    fig.savefig('ionColumnsVsImpact2D_%s_%s_%.1f-%.1f_rvir=%d_xlog=%d_ycum=%d_fd=%d.pdf' % \
        (sP.simName,ion,haloMassBin[0],haloMassBin[1],radRelToVirRad,xlog,ycum,fullDepth))
    plt.close(fig)

def ionCoveringFractionVsImpact2D(sPs, haloMassBin, ion, Nthresh, sPs2=None, radRelToVirRad=False, fullDepth=False):
    """ Use gridded N_ion maps to plot covering fraction f(N_ion>N_thresh) vs impact parameter. """

    nBins = 50
    xlim  = [1, 3.1] # log pkpc
    ylim  = [1e-3,1] # Lan MgII LRGs
    ylog  = True

    def _get_grids(sPs):
        """ Helper to load grids from a set of runs. """
        for i, sP in enumerate(sPs):
            dist_loc, grid_loc = _getStackedGrids(sP, ion, haloMassBin, fullDepth, radRelToVirRad, 
                ConfigLan=True, indiv=True, axesSets=[[0,1],[0,2],[1,2]])

            if i == 0:
                dist_global = dist_loc
                grid_global = grid_loc
            else:
                dist_global = np.hstack( (dist_global, dist_loc))
                grid_global = np.hstack( (grid_global, grid_loc))

        dist_global[dist_global == 0] = dist_global[dist_global > 0].min() / 2

        dist_global = np.log10(dist_global)

        return dist_global, grid_global

    def _covering_fracs(dist, grid, col_N):
        """ Helper, compute fc for all halos individually, and also the stack. """
        numHalos = dist.shape[1]

        fc = np.zeros( (nBins,numHalos+1), dtype='float32' )

        for i in range(numHalos+1):
            # derive covering fraction
            loc_dist = dist[:,i] if i < numHalos else dist.ravel()
            loc_grid = grid[:,i] if i < numHalos else grid.ravel()

            mask = np.zeros( loc_dist.size, dtype='int16' ) # 0 = below, 1 = above
            w = np.where(loc_grid >= col_N)
            mask[w] = 1

            count_above, _, _ = binned_statistic(loc_dist, mask, 'sum', bins=nBins, range=[xlim])
            count_total, bin_edges, _ = binned_statistic(loc_dist, mask, 'count', bins=nBins, range=[xlim])

            # plot
            xx = bin_edges[:-1] + (xlim[1]-xlim[0])/nBins/2
            with np.errstate(invalid='ignore'):
                fc[:,i] = count_above / count_total

            fc[:,i] = savgol_filter(fc[:,i], sKn, sKo)

        return xx, fc

    # load projected column density grids of all halos (possibly across more than one run/redshift)
    dist_global, grid_global = _get_grids(sPs)

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim([12,1200])
    ax.set_xscale('log')
    ax.set_xlabel('Impact Parameter [ log pkpc ]')

    ax.set_ylim(ylim)
    if ylog: ax.set_yscale('log')
    ax.set_ylabel('%s Covering Fraction' % (ion))

    # loop over requested column density limits
    for N in Nthresh:
        print(N)
        # loop over individual halos, and one extra iter for all
        c = next(ax._get_lines.prop_cycler)['color']

        xx, fc = _covering_fracs(dist_global, grid_global, N)

        # fill band (1 sigma halo-to-halo variation) for first column threshold only
        if N == Nthresh[0]:
            fc_percs = np.percentile(fc[:,:-1], [16,84], axis=1)
            fc_percs = savgol_filter(fc_percs, sKn, sKo, axis=1)

            ax.fill_between(10**xx, fc_percs[0,:], fc_percs[1,:], color=c, alpha=0.1, interpolate=True)

        # plot global
        label = 'N$_{\\rm %s}$ > 10$^{%.1f}$ cm$^{-2}$' % (ion,N)
        ax.plot(10**xx, fc[:,-1], '-', lw=3.0, alpha=1.0, color=c, label=label)

    # second sim set? only do for last column threshold
    if sPs2 is not None:
        lines = []
        for i, sPset in enumerate(sPs2):
            print(i, sPset[0].simName)

            # load grid
            dist_global2, grid_global2 = _get_grids(sPset)

            # covering fraction calculation and plot
            xx, fc = _covering_fracs(dist_global2, grid_global2, Nthresh[-1])
            l, = ax.plot(10**xx, fc[:,-1], linestyles[i+1], lw=3.0, alpha=1.0, color=c)
            lines.append(l)

    # observations: Bowen+11: LRGs
    b14_label = 'Bowen+ (2011) LRGs'
    b14_rp    = [25, 75, 125, 175]
    b14_fc    = np.array([0.093, 0.089, 1e-5, 1e-5])
    b14_up    = np.array([0.154, 0.130, 0.02, 0.02])
    b14_low   = np.array([0.035, 0.050, 1e-6, 1e-6])

    ax.errorbar(b14_rp, b14_fc, yerr=[b14_fc-b14_low,b14_up-b14_fc], markerSize=8, 
         color='black', ecolor='black', alpha=0.4, capsize=0.0, fmt='D', label=b14_label)

    # observations: Lan+14 Fig 8: fc (W > 1 Ang, "passive" i < 20.6)
    lan14_label = "Lan+ (2014) W$_{\\rm 0}^{\\rm MgII}$ > 1$\AA$"
    lan14_rp  = [25.0, 36.0, 50.0, 70.0, 100, 150, 210, 300, 430]
    lan14_fc  = np.array([0.1, 0.147, 0.093, 0.048, 0.034, 0.019, 0.013, 0.0088, 0.0071])
    lan14_up  = np.array([0.149, 0.189, 0.118, 0.064, 0.044, 0.026, 0.018, 0.012, 0.010])
    lan14_low = np.array([0.0508, 0.104, 0.068, 0.033, 0.024, 0.012, 0.008, 0.0052, 0.0042])

    ax.errorbar(lan14_rp, lan14_fc, yerr=[lan14_fc-lan14_low,lan14_up-lan14_fc], markerSize=8, 
         color='black', ecolor='black', alpha=0.6, capsize=0.0, fmt='s', label=lan14_label)

    # observations: Lan+18: fc (W > 0.4 Ang, LRGs, DR14)
    lan18_label = "Lan+ (2018) W$_{\\rm 0}^{\\rm MgII}$ > 0.4$\AA$"
    lan18_rp   = [23, 35, 48, 68, 95, 135, 189, 265, 380, 525, 740, 1.0e3]
    lan18_fc   = np.array([0.205, 0.187, 0.291, 0.142, 0.196, 0.068, 0.061, 0.047, 0.027, 0.026, 0.014, 0.007])
    lan18_up   = np.array([0.371, 0.417, 0.383, 0.203, 0.250, 0.097, 0.089, 0.061, 0.038, 0.034, 0.020, 0.011])
    lan18_down = np.array([0.037, 0.037,  0.201, 0.086, 0.142, 0.039, 0.033, 0.032, 0.016, 0.017, 0.009, 0.004])

    ax.errorbar(lan18_rp, lan18_fc, yerr=[lan18_fc-lan18_down,lan18_up-lan18_fc], markerSize=8, 
         color='black', ecolor='black', alpha=0.9, capsize=0.0, fmt='p', label=lan18_label)

    # finish plot
    if sPs2 is not None:
        legend2 = ax.legend(lines, [sPset[0].simName for sPset in sPs2], loc='upper right')
        ax.add_artist(legend2)

    ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig('ionCoveringFracVsImpact2D_%s_%s_%.1f-%.1f_rvir=%d_fd=%d.pdf' % \
        (sPs[0].simName,ion.replace(" ",""),haloMassBin[0],haloMassBin[1],radRelToVirRad,fullDepth))
    plt.close(fig)

def lrgHaloVisualization(sP, haloIDs, conf=3, gallery=False, globalDepth=True, testClumpRemoval=False):
    """ Configure single halo and multi-halo gallery visualizations. """
    run        = sP.run
    res        = sP.res
    redshift   = sP.redshift

    rVirFracs  = [0.25]
    method     = 'sphMap'
    nPixels    = [1000,1000]
    axes       = [0,1]
    labelZ     = True
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True
    relCoords  = False
    rotation   = 'edge-on'

    size       = 400.0
    sizeType   = 'kpc'

    # global with ~appropriate depth (same as in ionColumnsVsImpact2D)
    if globalDepth:
        method = 'sphMap_global'
        dv = 500.0 # +/- km/s (Zahedy), or +/- 1000 km/s (Berg)
        depth_code_units = (2*dv) / sP.units.H_of_a # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

    # which conf?
    if conf == 1:
        panel = {'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-1.4,0.2]}
    if conf == 2:
        panel = {'partType':'gas', 'partField':'MHIGK_popping', 'valMinMax':[15.0,21.0]}
    if conf == 3:
        panel = {'partType':'gas', 'partField':'Mg II', 'valMinMax':[12.0,16.5]}
    if conf == 4:
        panel = {'partType':'stars', 'partField':'stellarComp'}

    if conf == 5:
        # NARROW SLICE! h1
        haloIDSets = [1]

        nPixels   = [2000,1500]
        method    = 'sphMap'
        rVirFracs = [0.01,0.02,0.05]
        depthFac  = 0.15
        partType  = 'gas'
        size      = 50.0
        cenShift  = [size/2+4,size/2*(nPixels[1]/nPixels[0])+4,0] # center on upper-right quadrant

        labelZ     = False
        labelHalo  = False

        panel = []
        panel.append( {'partField':'cellsize_kpc', 'valMinMax':[-1.0,-0.3]} )
        panel.append( {'partField':'P_gas', 'valMinMax':[4.4,6.0]} )
        panel.append( {'partField':'P_B', 'valMinMax':[2.8,5.2]} )
        panel.append( {'partField':'pressure_ratio', 'valMinMax':[-1.5,1.5]} )

    if conf == 6:
        # NARROW SLICE! h19
        haloIDs = [19]

        nPixels   = [2000,1500]
        method    = 'sphMap'
        rVirFracs = [0.05,0.1]
        depthFac  = 0.2
        partType  = 'gas'
        size      = 80.0
        cenShift  = [-size/2,-size/2,0] # center on lower-left

        labelZ     = False
        labelHalo  = False

        panel = []
        panel.append( {'partField':'cellsize_kpc', 'valMinMax':[-0.7,0.0]} )
        panel.append( {'partField':'P_gas', 'valMinMax':[3.2,4.6]} )
        panel.append( {'partField':'P_B', 'valMinMax':[2.0,3.8]} )
        panel.append( {'partField':'pressure_ratio', 'valMinMax':[-1.0,1.0]} )

    if gallery:
        # multi-panel
        panels = []
        labelZ = False

        for haloID in haloIDs[0:12]:
            panel_loc = dict(panel)
            panel_loc['hInd'] = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

            panels.append(panel_loc)

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels[0] * 2
            nRows        = 3 # 3x4
            colorbars    = True
            saveFilename = './vis_%s_%d_%s.pdf' % (sP.simName,sP.snap,panels[0]['partField'].replace(" ","_"))

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    else:
        # single image
        for haloID in haloIDs:
            hInd = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

            panels = [panel] if not isinstance(panel,list) else panel

        # test: remove single clump visually
        if len(haloIDs) == 1 and testClumpRemoval:
            th = {'propName':'Mg II numdens', 'propThreshComp':'gt', 'propThresh':1e-8}
            objs, props = voronoiThresholdSegmentation(sP, haloID=haloIDs[0], 
              propName=th['propName'], propThresh=th['propThresh'], propThreshComp=th['propThreshComp'])

            clumpID = np.where(objs['lengths'] == 100)[0][8]
            print('Testing clump removal [ID = %d].' % clumpID)

            # halo-local indices of member cells
            def _getClumpInds(clumpID):
                offset = objs['offsets'][clumpID]
                length = objs['lengths'][clumpID]
                inds = objs['cell_inds'][offset : offset + length]
                return inds

            inds1 = _getClumpInds(clumpID)
            inds2 = _getClumpInds(3416) # close in space
            inds3 = _getClumpInds(1147) # close in space

            skipCellIndices = np.hstack( (inds1,inds2,inds3) )

            assert method == 'sphMap' # cell_inds to remove must index fof-scope indRange

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels
            colorbars    = True
            saveFilename = saveFilename = './vis_%s_%d_h%d_%s.pdf' % (sP.simName,sP.snap,haloID,panels[0]['partField'].replace(" ","_"))

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def lrgHaloVisResolution(sP, haloIDs, sPs_other):
    """ Visualization: one halo, for four different resolution runs. """

    # cross match
    from cosmo.util import crossMatchSubhalosBetweenRuns

    subIDs = [sP.halos('GroupFirstSub')[haloIDs]]

    for sPo in sPs_other:
        subIDs.append( crossMatchSubhalosBetweenRuns(sP, sPo, subIDs[0], method='Positional') )

    # panel config
    rVirFracs  = [0.05, 0.1, 0.25]
    method     = 'sphMap'
    nPixels    = [800,800]
    axes       = [1,0] #[0,1]
    labelZ     = False
    labelScale = False
    labelSim   = True
    labelHalo  = False
    relCoords  = False

    size       = 200.0
    sizeType   = 'kpc'
    cenShift   = [-size/2,+size/2,0] # center on left (half)

    partType   = 'gas'
    partField  = 'Mg II'
    valMinMax  = [12.0, 16.5]


    # single image per halo
    for i, haloID in enumerate(haloIDs):
        panels = []

        for j, sPo in enumerate([sP] + sPs_other):
            panels.append( {'run':sPo.run, 'res':sPo.res, 'redshift':sPo.redshift, 'hInd':subIDs[j][i]} )

        panels[0]['labelScale'] = 'physical'

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels
            colorbars    = True
            nRows        = 1
            saveFilename = saveFilename = './vis_%s_%d_res_h%d.pdf' % (sP.simName,sP.snap,haloID)

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

# clump plot config
lims = {'size'     : [0, 3.0],    # linear pkpc
        'mass'     : [4.5, 8.5],  # log msun
        'ncells'   : [0, 40],     # linear
        'dist'     : [0, 500],    # linear pkpc
        'dens'     : [-3.0, 2.5], # log cc
        'temp'     : [3.5,5.0],   # log K
        'bmag'     : [-1.0,2.0],  # log G
        'beta'     : [-2.0, 1.0], # log
        'sfr'      : [0, 0.01],    # linear msun/yr
        'metal'    : [-1.4, 0.4], # log solar
        'rcell1'   : [0, 800],    # linear parsec
        'rcell2'   : [0, 800],    # linear parsec
        'mg2_mass' : [0.0, 5.0],  # log msun
        'hi_mass'  : [2.0, 7.0]}  # log msun

labels = {'size'     : 'Clump Radius [ kpc ]',
          'mass'     : 'Clump Total Mass [ log M$_{\\rm sun}$ ]',
          'ncells'   : 'Number of Gas Cells [ linear ]',
          'dist'     : 'Halocentric Distance [ kpc ]',
          'dens'     : 'Mean Hydrogen Number Density [ log cm$^{-3}$ ]',
          'temp'     : 'Mean Clump Temperature [ log K ]',
          'bmag'     : 'Mean Clump Magnetic Field Strength [ log $\mu$G ]',
          'beta'     : 'Mean $\\beta = \\rm{P}_{\\rm gas} / \\rm{P}_{\\rm B}$ [ log ]',
          'sfr'      : 'Total Clump Star Formation Rate [ M$_{\\rm sun}$ / yr ]',
          'metal'    : 'Mean Clump Gas Metallicity [ log Z$_{\\rm sun}$ ]',
          'rcell1'   : 'Average Member Gas r$_{\\rm cell}$ [ parsec ]',
          'rcell2'   : 'Smallest Member Gas r$_{\\rm cell}$ [ parsec ]',
          'mg2_mass' : 'Total MgII Mass [ log M$_{\\rm sun}$ ]',
          'hi_mass'  : 'Total Neutral HI Mass [ log M$_{\\rm sun}$ ]'}

def _clump_values(sP, objs, props):
    """ Helper: some common unit conversions. """
    values = {}
    values['size']   = sP.units.codeLengthToKpc(props['radius'])
    values['mass']   = sP.units.codeMassToLogMsun(props['mass'])
    values['ncells'] = objs['lengths']
    values['dist']   = sP.units.codeLengthToKpc(props['distance'])
    values['dens']   = np.log10(props['dens_mean'])
    values['temp']   = np.log10(props['temp_mean'])
    values['bmag']   = np.log10(props['bmag_mean'] * 1e6)
    values['beta']   = np.log10(props['beta_mean'])
    values['sfr']    = props['sfr_tot']
    values['metal']  = np.log10(props['metal_mean'])
    values['rcell1'] = sP.units.codeLengthToKpc(props['rcell_mean']) * 1000
    values['rcell2'] = sP.units.codeLengthToKpc(props['rcell_min']) * 1000
    values['mg2_mass'] = sP.units.codeMassToLogMsun(props['mg2_mass'])
    values['hi_mass']  = sP.units.codeMassToLogMsun(props['hi_mass'])

    return values

def clumpDemographics(sP, haloID):
    """ Plot demographics of clump population for a single halo. """

    # config
    threshSets = []

    for val in [1e-6,1e-7,1e-8,1e-9,1e-10]: #[1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-15]:
        label = "n$_{\\rm Mg II}$ > %s cm$^{-3}$" % val
        threshSets.append( {'propName':'Mg II numdens', 'propThreshComp':'gt', 'propThresh':val, 'label':label})

    #for val in [4.0,4.2,4.4,4.8]:
    #    label = "log(T) < %.1f K" % val
    #    threshSets.append( {'propName':'temp_sfcold', 'propThreshComp':'lt', 'propThresh':val, 'label':label})

    #threshSets.append( {'propName':'sfr', 'propThreshComp':'gt', 'propThresh':1e-10, 'label':'SFR > 0'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.02, 'label':'n$_{\\rm H}$ > 0.02 cm$^{-3}$'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.1, 'label':'n$_{\\rm H}$ > 0.1 cm$^{-3}$'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.5, 'label':'n$_{\\rm H}$ > 0.5 cm$^{-3}$'})

    nBins1D = 100 # 1d histograms

    configs_2d = ['size-mass','ncells-size','size-dist','ncells-dist','dens-size',
                  'temp-size','bmag-size','beta-size','sfr-size','metal-size','rcell1-size','rcell2-size',
                  'mg2_mass-size','hi_mass-size','metal-dist']

    # load
    data = []

    for i, th in enumerate(threshSets):
        objs, props = voronoiThresholdSegmentation(sP, haloID=haloID, 
            propName=th['propName'], propThresh=th['propThresh'], propThreshComp=th['propThreshComp'])

        # some common unit conversions
        values = _clump_values(sP, objs, props)

        data.append( [objs,props,values] )
        print(i, 'prop = ', th['propName'], ' ', th['propThreshComp'], ' ', th['propThresh'], ' tot objs = ', objs['count'])

    # A: 1D histograms of all properties
    for config in lims.keys():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        lim = lims[config]
        ax.set_xlabel(labels[config])
        ax.set_xlim(lim)
        ax.set_ylabel('Number of Clumps')

        nBins = nBins1D if config != 'ncells' else lim[1]

        binsize = (lim[1] - lim[0]) / nBins
        bins = np.linspace( lim[0]-binsize, lim[1]+binsize, nBins+3 )

        for i, th in enumerate(threshSets):
            # load
            objs, props, values = data[i]

            vals = values[config]

            # histogram
            yy, xx = np.histogram(vals, bins=bins)
            xx = xx[:-1] + binsize/2 # mid

            #label = '%s %s %s' % (th['dispName'], {'gt':'>','lt':'<'}[propThreshComp], propThresh)
            l, = ax.plot(xx, yy, '-', lw=lw, drawstyle='steps-mid', label=th['label'])
            ax.fill_between(xx, np.zeros(yy.size), yy, step='mid', color=l.get_color(), alpha=0.05)

        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.legend()

        fig.tight_layout()
        fig.savefig('clumpDemographics_%s_h%d_%s.pdf' % (sP.simName,haloID,config))
        plt.close(fig)

    # B: 2d xquant vs yquant plots
    for config in configs_2d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        yname, xname = config.split('-')

        ax.set_xlim(lims[xname])
        ax.set_ylim(lims[yname])
        ax.set_xlabel(labels[xname])
        ax.set_ylabel(labels[yname])

        for i, th in enumerate(threshSets):
            # load
            objs, props, values = data[i]

            xvals = values[xname]
            yvals = values[yname]

            if xvals.size == 1:
                ax.plot(xvals,yvals,'o',label=th['label'])
                continue

            # running median
            binSize = (lims[xname][1] - lims[xname][0]) / nBins1D
            if xname == 'ncells': binSize = 1
            xm, ym, sm, pm = running_median(xvals, yvals, binSize=binSize, percs=[16,50,84])

            l, = ax.plot(xm, ym, '-', lw=lw, alpha=0.8, label=th['label'])
            if i in [0,len(threshSets)-1]:
                ax.fill_between(xm, pm[0,:], pm[-1,:], facecolor=l.get_color(), alpha=0.2, interpolate=True)

        ax.legend(loc='lower right')

        fig.tight_layout()
        fig.savefig('clumpDemographics_%s_h%d_%s.pdf' % (sP.simName,haloID,config))
        plt.close(fig)

def clumpTracerTracksLoad(sP, haloID, clumpID):
    """ Load subbox time evolution tracks and make analysis for the time evolution of 
    clump cell/integral properties vs time. Helper for the plot function below. """

    saveFilename = 'cache_clump_%s_%d-%d.hdf5' % (sP.simName,haloID,clumpID)

    # check for cache existence
    if isfile(saveFilename):
        print('Loading [%s]...' % saveFilename)
        data = {}
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                data[key] = f[key][()]

        return data

    # config
    sbNum = 2

    thPropName = 'Mg II numdens'
    thPropThresh = 1e-8
    thPropThreshComp = 'gt'

    # load
    GroupFirstSub = sP.groupCat(fieldsHalos=['GroupFirstSub'])

    for sbNum in []: #[0,1,2]: # disabled (find halo-subbox intersections)
        # load catalog
        cat = subboxSubhaloCat(sP, sbNum)

        masses = sP.subhalos('mhalo_200_log')[cat['SubhaloIDs']]
        print(sbNum, np.nanmax(masses))

        # intersect with all haloIDs sets
        for i, haloIDset in enumerate(haloIDs):
            subIDset = GroupFirstSub[haloIDset]
            inds_cat, inds_sample = match3(cat['SubhaloIDs'], subIDset)

            num_matched = inds_cat.size if inds_cat is not None else 0

            print('sb = %d haloIDset = %d, num_matched = %d' % (sbNum,i,num_matched), inds_cat, inds_sample)

    # load subbox catalog at z=0, and locate the target subhalo via its descendant tree
    sP_z0 = sP.copy()
    sP_z0.setRedshift(0.0)

    sbCat = subboxSubhaloCat(sP_z0, sbNum)

    subhaloID = sP.halo(haloID)['GroupFirstSub']
    subMDB = sP.loadMDB(subhaloID)

    subhaloInd_z0 = np.where(subMDB['SnapNum'] == sP_z0.snap)[0]
    subhaloID_z0 = subMDB['SubfindID'][subhaloInd_z0][0]

    sbCatInd = np.where(sbCat['SubhaloIDs'] == subhaloID_z0)[0]
    subhaloCen = np.squeeze(sbCat['SubhaloPos'][sbCatInd,:,:])

    # load segmentation
    objs, props = voronoiThresholdSegmentation(sP, haloID=haloID, 
        propName=thPropName, propThresh=thPropThresh, propThreshComp=thPropThreshComp)

    print('Selected clump [%d], properties:' % clumpID)

    for prop in props:
        print(' ', prop, props[prop][clumpID])

    offset = objs['offsets'][clumpID]
    length = objs['lengths'][clumpID]
    cell_inds = objs['cell_inds'][offset : offset + length]

    # load cell IDs
    cell_ids = sP.snapshotSubset('gas', 'ids', haloID=haloID)[cell_inds]

    # load tracer tracks meta, starting at this subbox
    sPsub = sP.subboxSim(sbNum)

    print('Load tracer meta and cross-matching...')
    tr_meta = globalAllTracersTimeEvo(sPsub, 'meta')

    # cross-match IDs, get tracer catalog indices of the clump member gas cells
    _, inds_cat = match3(cell_ids, tr_meta['ParentIDs'])
    indRange = [inds_cat.min(), inds_cat.max()+1]
    inds_cat -= inds_cat.min()

    loadSize = (indRange[1]-indRange[0]) * 200 * 3 * 8 / 1024**3 # approx GB
    if loadSize > 10:
        print(' Load size [%.1f GB], skipping...' % loadSize)
        return None

    print('Found [%d] tracers in [%d] parent cells.' % (inds_cat.size, cell_ids.size))

    # load other tracer property tracks
    trProps = ['hdens','temp','pos','metal','beta','netcoolrate']

    data = {}

    # loop over properties to load
    for prop in trProps:
        # load: backwards in time
        data_loc = globalAllTracersTimeEvo(sPsub, prop, indRange=indRange)

        if prop == trProps[0]:
            # save snapshots/redshifts
            data['redshifts'] = data_loc['redshifts']
            data['snaps'] = data_loc['snaps']
        else:
            # should be the same for all properties
            assert np.array_equal(data['snaps'], data_loc['snaps'])

        w_notdone = np.where(data_loc['done'] == 0)
        data_loc[prop][w_notdone] = np.nan

        data[prop] = data_loc[prop][:,inds_cat] # subset within indRange

        # load: forwards in time
        data_loc = globalAllTracersTimeEvo(sPsub, prop, indRange=indRange, toRedshift=sP.redshift-0.1)

        if prop == trProps[0]:
            # append forward times
            data['redshifts_full'] = np.hstack( (data['redshifts'], data_loc['redshifts']) )
            data['snaps_full'] = np.hstack( (data['snaps'], data_loc['snaps']) )

        if data_loc is not None:
            w_notdone = np.where(data_loc['done'] == 0)
            data_loc[prop][w_notdone] = np.nan

            data[prop] = np.vstack( (data[prop], data_loc[prop][:,inds_cat]) )

            nSnaps = data[prop].shape[0]
            nTr = data[prop].shape[1]

        if prop == 'temp':
            data[prop] = 10.0**data[prop] # remove log

    # data manipulation
    for prop in trProps:
        # time averages across all member cells
        data[prop+'_avg'] = np.nanmean(data[prop], axis=1)

    data['tage'] = sP.units.redshiftToAgeFlat(data['redshifts_full']) * 1e3 # Gyr -> Myr
    data['dt'] = data['tage'] - data['tage'][0]
    data['dt'][0] -= 1e-6 # place starting time negative

    # derive center of clump and clump extent(s) from pos
    data['pos_avg'] = np.mean(data['pos'], axis=1)

    # derive distance of each tracer to center of clump, and clump 'size' as ~half mass radius
    data['rad']  = np.zeros( (nSnaps,nTr), dtype='float32' )
    data['dist'] = np.zeros( (nSnaps,nTr), dtype='float32' )
    data['pos_rel'] = np.zeros( (nSnaps,nTr,3), dtype='float32' )
    data['size_maxseparation'] = np.zeros( nSnaps, dtype='float32' )

    maxTempsCold = [3e4, 1e5]
    data['size_maxsep_cold'] = np.zeros( (nSnaps,len(maxTempsCold)), dtype='float32' )

    SubhaloPos_trSnaps = subhaloCen[data['snaps_full'],:]

    for i in range(nSnaps):
        # distance of each tracer to center of clump
        data['rad'][i,:] = np.sqrt((data['pos'][i,:,0] - data['pos_avg'][i,0])**2 + \
                                   (data['pos'][i,:,1] - data['pos_avg'][i,1])**2 + \
                                   (data['pos'][i,:,2] - data['pos_avg'][i,2])**2 )

        # distance of each tracer to center of halo
        data['pos_rel'][i,:,0] = data['pos'][i,:,0] - SubhaloPos_trSnaps[i,0]
        data['pos_rel'][i,:,1] = data['pos'][i,:,1] - SubhaloPos_trSnaps[i,1]
        data['pos_rel'][i,:,2] = data['pos'][i,:,2] - SubhaloPos_trSnaps[i,2]

        data['dist'][i,:] = np.sqrt(data['pos_rel'][i,:,0]**2 + \
                                    data['pos_rel'][i,:,1]**2 + \
                                    data['pos_rel'][i,:,2]**2 )

        # maximum pairwise distance between clump members
        data['size_maxseparation'][i] = sP.periodicPairwiseDists(data['pos'][i,:,:]).max()
        for j, maxTempCold in enumerate(maxTempsCold):
            w_cold = np.where(data['temp'][i,:] < maxTempsCold[j])
            data['size_maxsep_cold'][i,j] = sP.periodicPairwiseDists(np.squeeze(data['pos'][i,w_cold,:])).max()

    data['size_halfmassrad'] = np.median(data['rad'], axis=1)
    data['dist_rvir'] = data['dist'] / sP.halo(haloID)['Group_R_Crit200'] # take constant
    data['pos_rel'] = sP.units.codeLengthToKpc(data['pos_rel'])

    # convert dist to dist_avg, i.e. radial distance of clump from interpolated halo center position
    data['dist_avg'] = np.mean(data['dist'], axis=1)
    data['pos_rel_avg'] = np.mean(data['pos_rel'], axis=1)
    data['dist_rvir_avg'] = np.mean(data['dist_rvir'], axis=1)

    # calculate medians as an alternative to means
    for key in list(data.keys()):
        if key+'_avg' in data:
            data[key+'_median'] = np.nanmedian(data[key], axis=1)

    # save cache
    with h5py.File(saveFilename,'w') as f:
        for key in data:
            f[key] = data[key]
    print('Saved [%s].' % saveFilename)

    return data

def clumpTracerTracks(sP, clumpID):
    """ Intersect the LRG halo sample with the subbox catalogs, find which halos are available for high time resolution 
    tracking, and then make our analysis and plots of the time evolution of clump cell/integral properties vs time. """

    haloID = 0 # only >10^13 halo which intersects with subboxes

    time_xlim = [-500,300]
    lineAlpha = 0.05 # for individual tracers
    lineW = 1 # for individual tracers

    labels = {'size_halfmassrad'   : 'Clump Half-mass Radius [ kpc ]',
              'size_maxseparation' : 'Clump Size: Max Pairwise Separation [ kpc ]',
              'dist'               : 'Halocentric Distance [ kpc ]',
              'dist_rvir'          : 'Halocentric Distance / r$_{\\rm vir}$',
              'hdens'              : 'Hydrogen Number Density [ log cm$^{-3}$ ]',
              'temp'               : 'Temperature [ log K ]',
              'metal'              : 'Metallicity [ log (not solar) ]',
              'netcoolrate'        : 'Net Cooling Rate [ erg/s/g ]',
              'beta'               : '$\\beta = \\rm{P}_{\\rm gas} / \\rm{P}_{\\rm B}$ [ log ]'}

    lims = {'temp'  : [3.8, 8.2],
            'hdens' : [-3.5, 2.0],
            'dist'  : [-20, 300]}

    noForwardData = ['metal','netcoolrate'] # fields without tracer_tracks into the future

    circOpts = {'markeredgecolor':'white', 'markerfacecolor':'None', 'markersize':10, 'markeredgewidth':2} # marking t=0

    # load
    data = clumpTracerTracksLoad(sP, haloID, clumpID)

    if data is None:
        return

    # plot (A) - time series
    xx = data['dt']

    w_back = np.where(xx < 0)
    w_forward = np.where(xx >= 0)

    for prop in labels.keys():
        print(' plot ', prop)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlabel('Time since $z=0.5$ [Myr]')
        ax.set_ylabel(labels[prop])
        ax.set_xlim(time_xlim)
        if prop in lims.keys(): ax.set_ylim(lims[prop])
        ax.set_rasterization_zorder(1) # elements below z=1 are rasterized

        logf = np.log10 if ('dist' not in prop and 'coolrate' not in prop) else lambda x: x # identity

        if prop+'_avg' in data:
            for i in range(data[prop].shape[1]): # individuals
                yy = logf(data[prop][:,i])
                ax.plot(xx[w_back], yy[w_back], '-', lw=lineW, color='black', alpha=lineAlpha, zorder=0)
                if prop not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], '-', lw=lineW, color='black', alpha=lineAlpha, zorder=0)

            yy = logf(data[prop+'_avg']) # mean across member cells
            if prop not in noForwardData:
                ax.plot(xx[w_forward], yy[w_forward], 'o-', lw=lw, label='Clump Mean ($t>0$)')
            ax.plot(xx[w_back], yy[w_back], 'o-', lw=lw, label='Clump Mean ($t<0$)')
            if prop not in noForwardData:
                ax.plot(xx[w_forward][0], yy[w_forward][0], 'o', **circOpts)

        else:
            yy = logf(data[prop]) # quantity is 1 number per snapshot
            if prop not in noForwardData:
                l2, = ax.plot(xx[w_forward], yy[w_forward], 'o-', lw=lw, label='($t>0$)')
            l1, = ax.plot(xx[w_back], yy[w_back], 'o-', lw=lw, label='($t<0)$')

            if prop == 'size_maxseparation': # add _cold
                #yy = logf(data['size_maxsep_cold'][:,0])
                #ax.plot(xx[w_forward], yy[w_forward], ':', lw=lw, color=l2.get_color(), label='log(T) < 5.0 ($t>0$)')
                #ax.plot(xx[w_back], yy[w_back], ':', lw=lw, color=l1.get_color(), label='log(T) < 5.0 ($t<0$)')
                yy = logf(data['size_maxsep_cold'][:,1])
                ax.plot(xx[w_forward], yy[w_forward], '--', lw=lw, color=l2.get_color(), label='T < 30,000K ($t>0$)')
                ax.plot(xx[w_back], yy[w_back], '--', lw=lw, color=l1.get_color(), label='T < 30,000K ($t<0$)')

        ax.legend(loc='upper left' if prop == 'temp' else 'best')
        fig.tight_layout()
        fig.savefig('clumpEvo_%s_clumpID=%d_%s.pdf' % (sP.simName,clumpID,prop))
        plt.close(fig)

    # plot (B) - phase diagram with time track
    for xval in ['hdens','temp','dist_rvir']:
        for yval in labels.keys():
            if xval == yval: continue
            print(' plot ', xval, yval)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(labels[xval])
            ax.set_ylabel(labels[yval])

            logx = np.log10 if (('dist' not in xval) and ('coolrate' not in xval)) else lambda x: x # identity
            logy = np.log10 if (('dist' not in yval) and ('coolrate' not in yval)) else lambda x: x # identity

            if yval+'_avg' in data:
                for i in range(data[xval].shape[1]): # individuals
                    xx = logx(data[xval][:,i])
                    yy = logy(data[yval][:,i])
                    ax.plot(xx[w_back], yy[w_back], '-', lw=lineW, color='black', alpha=lineAlpha)
                    if xval not in noForwardData and yval not in noForwardData:
                        ax.plot(xx[w_forward], yy[w_forward], '-', lw=lineW, color='black', alpha=lineAlpha)

                xx = logx(data[xval+'_median']) # mean across member cells
                yy = logy(data[yval+'_median'])
                if xval not in noForwardData and yval not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], 'o-', lw=lw, label='Clump Mean ($t>0$)')
                ax.plot(xx[w_back], yy[w_back], 'o-', lw=lw, label='Clump Mean ($t<0$)')
                if xval not in noForwardData and yval not in noForwardData:
                    ax.plot(xx[w_forward][0], yy[w_forward][0], 'o', **circOpts)
                ax.legend()
            else:
                xx = logx(data[xval]) if data[xval].ndim == 1 else logx(data[xval+'_avg']) # quantity is 1 number per snapshot
                yy = logy(data[yval])
                ax.plot(xx[w_back], yy[w_back], 'o-', lw=lw)
                if xval not in noForwardData and yval not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], 'o-', lw=lw)

            fig.tight_layout()
            fig.savefig('clumpEvo_%s_clumpID=%d_x=%s_y=%s.pdf' % (sP.simName,clumpID,xval,yval))
            plt.close(fig)

    # plot (C) - spatial tracks
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1) # elements below z=1 are rasterized
    aspect = ax.get_window_extent().height / ax.get_window_extent().width

    axes = [0,1]
    prop = 'pos_rel'

    ax.set_xlabel('$\Delta$ %s [kpc]' % ['x','y','z'][axes[0]])
    ax.set_ylabel('$\Delta$ %s [kpc]' % ['x','y','z'][axes[1]])

    xylim = np.array([data[prop][:,:,axes].min(), data[prop][:,:,axes].max()]) * 0.8
    ax.set_xlim(xylim)
    ax.set_ylim(xylim*aspect) # ax is non-square, so make limits reflect the correct aspect ratio

    for i in range(data[prop].shape[1]): # individuals
        xx = data[prop][:,i,axes[0]]
        yy = data[prop][:,i,axes[1]]
        l1, = ax.plot(xx[w_back], yy[w_back], '-', color=l1.get_color() if i>0 else None, lw=lineW, alpha=lineAlpha, zorder=0)
        l2, = ax.plot(xx[w_forward], yy[w_forward], '-', color=l2.get_color() if i>0 else None, lw=lineW, alpha=lineAlpha, zorder=0)

    xx = data[prop+'_avg'][:,axes[0]]
    yy = data[prop+'_avg'][:,axes[1]] # mean across member cells
    #ax.plot(xx[w_back], yy[w_back], '-', color=l1.get_color(), lw=lw, label='Clump Mean ($t<0$)')
    #ax.plot(xx[w_forward], yy[w_forward], '-', color=l2.get_color(), lw=lw, label='Clump Mean ($t>0$)')
    ax.plot([0.0,0.0], 'o', color='black')
    ax.legend()

    fig.tight_layout()
    fig.savefig('clumpEvo_%s_clumpID=%d_%s.pdf' % (sP.simName,clumpID,prop))
    plt.close(fig)

def clumpAbundanceVsHaloMass(sP):
    """ Run segmentation on a flat mass-selection of halos, plot number of identified clumps vs halo mass. """
    from vis.halo import selectHalosFromMassBins

    # config
    minMaxHaloMass = [11.0, 14.0]
    numPerBin = 10
    xQuant = 'mhalo_200_log'

    thPropName = 'Mg II numdens'
    thPropThresh = 1e-8
    thPropThreshComp = 'gt'

    minCellsPerClump = 10

    # make halo selection
    binSize = 0.1
    numMassBins = int((minMaxHaloMass[1] - minMaxHaloMass[0]) / binSize) + 1
    bins = [ [x+0.0,x+binSize] for x in np.linspace(minMaxHaloMass[0],minMaxHaloMass[1],numMassBins) ]

    hInds = selectHalosFromMassBins(sP, bins, numPerBin, 'random')
    hInds = np.hstack( [h for h in hInds] ).astype('int32')

    # allocate
    clumpProps = {}
    for prop in labels.keys():
        clumpProps[prop] = np.zeros( hInds.size, dtype='float32' )
    clumpProps['number'] = np.zeros( hInds.size, dtype='int32' )

    # load/create segmentations, and accumulate mean properties per halo
    for i, haloID in enumerate(hInds):
        objs, props = voronoiThresholdSegmentation(sP, haloID=haloID, 
            propName=thPropName, propThresh=thPropThresh, propThreshComp=thPropThreshComp)
        values = _clump_values(sP, objs, props)

        w = np.where(objs['lengths'] >= minCellsPerClump)[0]
        clumpProps['number'][i] = len(w)

        for prop in labels.keys():
            clumpProps[prop][i] = np.median( values[prop][w] )

    # load x-quant and make median
    x_vals, x_label, minMax, takeLog = sP.simSubhaloQuantity(xQuant)
    if takeLog: x_vals = np.log10(x_vals)

    x_vals = x_vals[sP.halos('GroupFirstSub')[hInds]]

    # loop over clump properties to plot
    for prop in ['number'] + list(labels.keys()):
        # make median
        print(prop)

        xm, ym, sm, pm = running_median(x_vals,clumpProps[prop],binSize=binSize*2,percs=[16,50,84],minNumPerBin=3)
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)
            sm = savgol_filter(sm,sKn,sKo)
            pm = savgol_filter(pm,sKn,sKo,axis=1)

        # plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ylabel = 'Number of Clouds (n$_{\\rm MgII} > 10^{-8}$ cm$^{-3}$)' if prop == 'number' else labels[prop]

        ax.set_xlabel("Halo Mass [ log M$_{\\rm sun}$ ]")
        ax.set_ylabel(ylabel)
        ax.set_xlim([10.95, 14.05])
        if prop == 'number': ax.set_yscale('log')

        l, = ax.plot(x_vals, clumpProps[prop], marker='o', color='tab:green', alpha=0.8, linestyle='None')
        ax.fill_between(xm, pm[0,:], pm[-1,:], color='#444444', alpha=0.2)
        ax.plot(xm, ym, '-', color='#444444', lw=lw, label='median ($z=0.5$)')

        # finish plot
        ax.legend()

        fig.tight_layout()
        fig.savefig('clumps_%s_vs_%s_%s_%d_min%d.pdf' % (prop,xQuant,sP.simName,sP.snap,minCellsPerClump))
        plt.close(fig)

def paperPlots():
    """ Produce all papers for the LRG-MgII (small-scale CGM structure) TNG50 paper. """
    haloMassBins = [[12.3,12.7], [12.8, 13.2], [13.2, 14.0]]
    redshift = 0.5 # default for analysis

    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)
    TNG50_4 = simParams(res=270,run='tng',redshift=redshift)

    def _get_halo_ids(sP_loc, bin_inds=None, subhaloIDs=False):
        """ Load and return the set of halo IDs in each haloMassBin. 
        if subhaloIDs, then return these rather than haloIDs. """
        mhalo = sP_loc.groupCat(fieldsSubhalos=['mhalo_200_log'])

        if subhaloIDs:
            grnr = np.arange(sP_loc.numHalos) # identity
        else:
            grnr  = sP_loc.groupCat(fieldsSubhalos=['SubhaloGrNr'])

        haloIDs = []

        if bin_inds is None:
            # return a list, one entry per haloMassBin
            for haloMassBin in haloMassBins:
                with np.errstate(invalid='ignore'):
                    inds = np.where( (mhalo > haloMassBin[0]) & (mhalo < haloMassBin[1]) )[0]

                haloIDs.append(grnr[inds])
        else:
            # return a single array, all haloIDs in the specified haloMassBin(s)
            for haloMassBin in [haloMassBins[i] for i in bin_inds]:
                with np.errstate(invalid='ignore'):
                    inds = np.where( (mhalo > haloMassBin[0]) & (mhalo < haloMassBin[1]) )[0]
                haloIDs += list(grnr[inds])
            haloIDs = np.array(haloIDs, dtype='int32')

        return haloIDs

    # --- halo scale ---

    # figure 1 - cgm resolution
    if 0:
        sPs = [TNG50, TNG50_2, TNG50_3]
        cenSatSelect = 'cen'

        simNames = '_'.join([sP.simName for sP in sPs])
        
        for radRelToVirRad in [True]: #,False]:
            saveName = 'resolution_profiles_%s_z%02d_%s_rvir%d.pdf' % (simNames,redshift*10,cenSatSelect,radRelToVirRad)

            radialResolutionProfiles(sPs, saveName, redshift=redshift, radRelToVirRad=radRelToVirRad, 
                                     cenSatSelect='cen', haloMassBins=haloMassBins)

    # figs 2, 3, 4 - vis (single halo MgII, single halo HI, 3x3 gallery MgII)
    if 0:
        sP = TNG50
        haloIDs = _get_halo_ids(sP)[2]

        # gas metallicity, N_MgII, N_HI, stellar light
        for conf in [1,2,3,4]:
            lrgHaloVisualization(sP, haloIDs, conf=conf, gallery=True)
            lrgHaloVisualization(sP, haloIDs, conf=conf, gallery=False)

    # figure 5a - cgm gas density/temp/pressure 1D PDFs
    if 0:
        sP = TNG50
        qRestrictions = [['rad_rvir',0.1,1.0]] # 0.15<r/rvir<1
        nBins = 200
        medianPDF = True

        if 0:
            ptProperty = 'nh'
            xlim = [-6.0, 0.0]
        if 1:
            ptProperty = 'temp_sfcold'
            xlim = [3.8,7.6]
        if 0:
            ptProperty = 'gas_pres'
            qRestrictions.append( ['sfr',0.0,0.0] ) # non-eEOS
            xlim = [0.0, 5.0]

        # all halos in two most massive bins
        subhaloIDs = _get_halo_ids(sP, bin_inds=[1,2], subhaloIDs=True)

        # create density PDF of gas in radRangeRvir
        plotHistogram1D([sP], ptType='gas', ptProperty=ptProperty, xlim=xlim, nBins=nBins, medianPDF=medianPDF, 
                        qRestrictions=qRestrictions, subhaloIDs=[subhaloIDs], ctName='plasma', ctProp='mhalo_200_log', 
                        legend=False, colorbar=True)

    # fig 5b - cgm gas (n,T) 2D phase diagrams
    if 0:
        sP = TNG50
        qRestrictions = [['rad_rvir',0.1,1.0]]

        xQuant = 'nh'
        yQuant = 'temp'
        xlim = [-5.5, -1.0]
        ylim = [3.8, 7.8]

        weights = ['mass']
        meancolors = None
        clim = [-4.0,0.0]
        contourQuant = 'gas_pres'
        contours = [2.5,3.0,3.5] # log K/cm^3

        saveStr = weights[0] if weights is not None else meancolors[0]

        haloIDs = [8] # single example
        #haloIDs = _get_halo_ids(sP)[1] # all the halos in a mass bin

        pdf = PdfPages('phase2d_%s_%d_h%d_%s_%s_%s.pdf' % (sP.simName,sP.snap,haloIDs[0],xQuant,yQuant,weights[0]))

        for haloID in haloIDs:
            plotPhaseSpace2D(sP, partType='gas', xQuant=xQuant, yQuant=yQuant, weights=weights, meancolors=meancolors, 
                         haloID=haloID, pdf=pdf, xlim=xlim, ylim=ylim, clim=clim, nBins=None, 
                         contours=contours, contourQuant=contourQuant, qRestrictions=qRestrictions)
        pdf.close()

    # fig 6a: bound ion mass as a function of halo mass
    if 0:
        TNG50_z0 = simParams(run='tng50-1', redshift=0.0)
        TNG50_z1 = simParams(run='tng50-1', redshift=1.0)
        sPs = [TNG50,TNG50_z0,TNG50_z1]
        cenSatSelect = 'cen'
        ions = ['HIGK_popping','AllGas_Metal','AllGas_Mg','MgII']

        for vsHaloMass in [True,False]:
            massStr = '%smass' % ['stellar','halo'][vsHaloMass]

            saveName = 'ions_masses_vs_%s_%s_%d_%s.pdf' % \
                (massStr,cenSatSelect,sPs[0].snap,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, cenSatSelect=cenSatSelect, 
                vsHaloMass=vsHaloMass, colorOff=0) # , secondTopAxis=True

    # fig 6b: radial profiles
    if 0:
        sPs = [TNG50]
        ions = ['MgII']#,'HIGK_popping']
        cenSatSelect = 'cen'
        haloMassBins = [[11.4,11.6], [11.9,12.1], [12.4,12.6], [12.8, 13.2], [13.2, 13.8]]
        projSpecs = ['2Dz_6Mpc','3D']
        combine2Halo = True

        simNames = '_'.join([sP.simName for sP in sPs])

        for massDensity in [False]: #[True,False]:
            for radRelToVirRad in [False]: #[True,False]:
                for projDim in projSpecs:

                    saveName = 'radprofiles_%s_%s_%s_%d_%s_rho%d_rvir%d.pdf' % \
                      (projDim,'-'.join(ions),simNames,sPs[0].snap,cenSatSelect,massDensity,radRelToVirRad)
                    stackedRadialProfiles(sPs, saveName, redshift=sPs[0].redshift, ions=ions, massDensity=massDensity,
                                          radRelToVirRad=radRelToVirRad, cenSatSelect='cen', projDim=projDim, 
                                          haloMassBins=haloMassBins, combine2Halo=combine2Halo, median=True)

    # fig X: 2pcf
    if 0:
        sPs = [TNG50] #[TNG100, TNG300]
        ions = ['MgII'] #,'Mg','gas']

        # compute time for one split:
        # TNG100 [days] = (1820^3/256^3)^2 * (1148/60/60/60) * (8*100/nSplits) * (16/nThreads)
        # for nSplits=200000, should finish each in 1.5 days (nThreads=32) (each has 60,000 cells)
        # for TNG300, nSplits=500000, should finish each in 4 days (nThreads=32)
        for order in [0,1,2]:
            saveName = 'tpcf_order%d_%s_%s_z%02d.pdf' % \
              (order,'-'.join(ions),'_'.join([sP.simName for sP in sPs]),redshift)

            ionTwoPointCorrelation(sPs, saveName, ions=ions, redshift=redshift, order=order, colorOff=2)

        # todo: tpcf of single halo (0.15 < r/rvir < 1.0)

    # --- clump analysis ---

    # fig 7: zoomed in (high res) visualizations of multiple properties of clumps
    if 0:
        sP = TNG50
        haloIDs = _get_halo_ids(sP)[2]

        for conf in [5,6]:
            lrgHaloVisualization(sP, None, conf=conf, gallery=False)

        # vis test - verify a ~1.5kpc clump by weighting those indices to zero in vis()
        #haloIDs = [0]
        #lrgHaloVisualization(sP, haloIDs, conf=3, gallery=False, globalDepth=False, testClumpRemoval=True)

    # fig 8: clump demographics: size distribution, total mass, average numdens, etc
    if 0:
        haloID = 0 # 0, 19
        clumpDemographics(TNG50, haloID=haloID)

    # fig 9: time tracks via tracers
    if 0:
        # pick a single clump (from np.where(objs['lengths'] == 100))
        clumpID = 1592 # 3416, 3851

        clumpTracerTracks(TNG50, clumpID=clumpID)

        # helper: loop above over many clumps
        if 0:
            # load segmentation and select
            objs, props = voronoiThresholdSegmentation(sP, haloID=0, 
                propName='Mg II numdens', propThresh=1e-8, propThreshComp='gt')

            #clumpIDs = np.where(objs['lengths'] == 100)[0][0:10]
            clumpIDs = np.where( (objs['lengths'] >= 400) & (objs['lengths'] < 405) )[0]
            #clumpIDs = np.where(objs['lengths'] == 40)[0][0:10]
            print('Processing [%d] clumps...' % clumpIDs.size)

            # load tracer tracks, cache, make plots
            for clumpID in clumpIDs:
                clumpTracerTracks(sP, clumpID=clumpID)

    # fig 10: individual (or stacked) clump radial profiles, including pressure/cellsize
    # TODO

    # fig 10: N_clumps vs halo mass, to show they don't exist at below some threshold halo mass
    if 0:
        clumpAbundanceVsHaloMass(TNG50)

    # fig 11: resolution convergence, visual (matched halo)
    if 0:
        haloIDs = _get_halo_ids(TNG50)[2]
        lrgHaloVisResolution(TNG50, haloIDs, [TNG50_2, TNG50_3, TNG50_4])

    # fig 12: resolution convergence, quantitative (?)

    # --- observational comparison ---

    # fig 13 - N_MgII or N_HI vs. b (map-derived): 2D histo of N_px/N_px_tot_annuli (normalized independently by column)
    if 0:
        sP = TNG50
        haloMassBin = haloMassBins[1]
        #ion = 'Mg II'
        ion = 'MHIGK_popping'
        radRelToVirRad = False

        for ycum in [True,False]:
            for fullDepth in [True,False]:
                ionColumnsVsImpact2D(sP, haloMassBin, ion=ion, radRelToVirRad=radRelToVirRad, ycum=ycum, fullDepth=fullDepth)

    # fig X: obs matched samples for COS-LRG and LRG-RDR surveys
    if 0:
        sPs = [TNG50, TNG100]

        simNames = '_'.join([sP.simName for sP in sPs])
        obsSimMatchedGalaxySamples(sPs, 'sample_lrg_rdr_%s.pdf' % simNames, config='LRG-RDR')
        obsSimMatchedGalaxySamples(sPs, 'sample_cos_lrg_%s.pdf' % simNames, config='COS-LRG')

    # fig 14: run old OVI machinery to derive goodness of fit parameters and associated plots
    if 0:
        for sP in [TNG50]: #[TNG50, TNG100]:
            obsColumnsDataPlotExtended(sP, saveName='obscomp_lrg_rdr_hi_%s_ext.pdf' % sP.simName, config='LRG-RDR')
            obsColumnsDataPlotExtended(sP, saveName='obscomp_cos_lrg_hi_%s_ext.pdf' % sP.simName, config='COS-LRG HI')
            obsColumnsDataPlotExtended(sP, saveName='obscomp_cos_lrg_mgii_%s_ext.pdf' % sP.simName, config='COS-LRG MgII')

    # fig 15: covering fraction comparison
    if 1:
        haloMassBin = haloMassBins[2]
        ion = 'Mg II'
        Nthreshs = [15.0, 15.5, 16.0]
        sPs  = [ TNG50, simParams(run='tng50-1', redshift=0.4), simParams(run='tng50-1', redshift=0.6) ]
        sPs2 = [ [TNG50_2,simParams(run='tng50-2', redshift=0.4),simParams(run='tng50-2', redshift=0.6)], 
                 [TNG50_3,simParams(run='tng50-3', redshift=0.4),simParams(run='tng50-3', redshift=0.6)], 
                 [TNG50_4,simParams(run='tng50-4', redshift=0.4),simParams(run='tng50-4', redshift=0.6)] ]

        ionCoveringFractionVsImpact2D(sPs, haloMassBin, ion, Nthreshs, sPs2=sPs2, radRelToVirRad=False, fullDepth=True)

    # fig X: curve of growth for MgII
    if 0:
        from plot.cloudy import curveOfGrowth
        curveOfGrowth(lineName='MgII2803')
