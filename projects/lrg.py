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
from scipy.stats import binned_statistic_2d
from os.path import isfile

from util import simParams
from util.helper import running_median, logZeroNaN, loadColorTable
from util.voronoi import voronoiThresholdSegmentation
from cosmo.util import subboxSubhaloCat
from plot.config import *
from plot.general import plotStackedRadialProfiles1D, plotHistogram1D, plotPhaseSpace2D
from tracer.tracerMC import match3
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
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
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

    # quick caching
    cacheSaveFile = sP.derivPath + 'cache/ionColumnsVsImpact2D_%s_%s_%.1f-%.1f_rvir=%s_fd=%s.hdf5' % \
      (sP.simName,ion,haloMassBin[0],haloMassBin[1],radRelToVirRad,fullDepth)

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
        dist_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )
        grid_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )

        for i, hInd in enumerate(subInds):
            class plotConfig:
                saveFilename = 'dummy'

            panels = [{'partType':'gas', 'partField':ion, 'valMinMax':[-1.4,0.2]}]
            grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

            # compute impact parameter for every pixel
            pxSize = size / nPixels[0] # pkpc

            xx, yy = np.mgrid[0:nPixels[0], 0:nPixels[1]]
            xx = xx.astype('float64') - nPixels[0]/2
            yy = yy.astype('float64') - nPixels[1]/2
            dist = np.sqrt( xx**2 + yy**2 ) * pxSize

            if radRelToVirRad:
                dist /= gc['rhalo_200_code'][subInds[i]]

            # bin
            dist_global[:,i] = dist.ravel()
            grid_global[:,i] = grid.ravel()

        # flatten
        dist_global = dist_global.ravel()
        grid_global = grid_global.ravel()

        # save cache
        with h5py.File(cacheSaveFile,'w') as f:
            f['dist_global'] = dist_global
            f['grid_global'] = grid_global

        print('Saved: [%s]' % cacheSaveFile)

    # start plot
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
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

def lrgHaloVisualization(sP, haloIDSets, conf=3, gallery=False):
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
    if 1:
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
        haloIDSets = [[],[1]]

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
        haloIDSets = [[],[19]]

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
        setInd = 2 # just choose these manually
        labelZ = False

        for haloID in haloIDSets[setInd][0:12]:
            panel_loc = dict(panel)
            panel_loc['hInd'] = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

            panels.append(panel_loc)

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels[0] * 2
            nRows        = 3 # 3x4
            colorbars    = True
            saveFilename = './vis_%s_%d_gal%d_%s.pdf' % (sP.simName,sP.snap,setInd,panels[0]['partField'].replace(" ","_"))

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    else:
        # single image
        for haloID in [haloIDSets[2][0]]: # list(haloIDSets[1]) + list(haloIDSets[2]):
            hInd = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

            panels = [panel] if not isinstance(panel,list) else panel

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels
            colorbars    = True
            saveFilename = saveFilename = './vis_%s_%d_h%d_%s.pdf' % (sP.simName,sP.snap,haloID,panels[0]['partField'].replace(" ","_"))

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def clumpDemographics(sP):
    """ Plot demographics of clump population for a single halo or stacked halos. """

    # TODO: visually verify a ~1.5kpc clump by weighting those indices to zero in vis()

    # config
    haloID = 19

    threshSets = []

    for val in [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-15]:
        label = "n$_{\\rm Mg II}$ > %s cm$^{-3}$" % val
        threshSets.append( {'propName':'Mg II numdens', 'propThreshComp':'gt', 'propThresh':val, 'label':label})

    #for val in [4.0,4.2,4.4,4.8]:
    #    label = "log(T) < %.1f K" % val
    #    threshSets.append( {'propName':'temp_sfcold', 'propThreshComp':'lt', 'propThresh':val, 'label':label})

    #threshSets.append( {'propName':'sfr', 'propThreshComp':'gt', 'propThresh':1e-10, 'label':'SFR > 0'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.02, 'label':'n$_{\\rm H}$ > 0.02 cm$^{-3}$'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.1, 'label':'n$_{\\rm H}$ > 0.1 cm$^{-3}$'})
    #threshSets.append( {'propName':'nh', 'propThreshComp':'gt', 'propThresh':0.5, 'label':'n$_{\\rm H}$ > 0.5 cm$^{-3}$'})

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

    nBins1D = 100 # 1d histograms

    configs_2d = ['size-mass','ncells-size','size-dist','ncells-dist','dens-size',
                  'temp-size','bmag-size','beta-size','sfr-size','metal-size','rcell1-size','rcell2-size',
                  'mg2_mass-size','hi_mass-size','metal-dist']

    sizefac = 1.0 if not clean else sfclean

    # load
    data = []

    for i, th in enumerate(threshSets):
        objs, props = voronoiThresholdSegmentation(sP, haloID=haloID, 
            propName=th['propName'], propThresh=th['propThresh'], propThreshComp=th['propThreshComp'])

        # some common unit conversions
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

        data.append( [objs,props,values] )
        print(i, 'prop = ', th['propName'], ' ', th['propThreshComp'], ' ', th['propThresh'], ' tot objs = ', objs['count'])

    # A: 1D histograms of all properties
    for config in lims.keys():
        fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
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
        fig.savefig('clumpDemographics_%s_%s.pdf' % (sP.simName,config))
        plt.close(fig)

    # B: size vs mass
    for config in configs_2d:
        fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
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

            #if xm.size > sKn:
            #    ym2 = savgol_filter(ym,sKn,sKo)
            #    sm2 = savgol_filter(sm,sKn,sKo)
            #    pm2 = savgol_filter(pm,sKn,sKo,axis=1)

            #if yname == 'ncells' and xname == 'size':
            #    import pdb; pdb.set_trace()

            l, = ax.plot(xm, ym, '-', lw=lw, alpha=0.8, label=th['label'])
            if i in [0,len(threshSets)-1]:
                ax.fill_between(xm, pm[0,:], pm[-1,:], facecolor=l.get_color(), alpha=0.2, interpolate=True)

        ax.legend(loc='lower right')

        fig.tight_layout()
        fig.savefig('clumpDemographics_%s_%s.pdf' % (sP.simName,config))
        plt.close(fig)

    # C: 

def clumpTracerTracks(sP, haloIDs):
    """ Intersect the LRG halo sample with the subbox catalogs, find which halos are available for high time resolution 
    tracking, and then make our analysis and plots of the time evolution of clump cell/integral properties vs time. """

    GroupFirstSub = sP.groupCat(fieldsHalos=['GroupFirstSub'])

    for sbNum in [0,1,2]:
        # load catalog
        cat = subboxSubhaloCat(sP, sbNum)

        # intersect with all haloIDs sets
        for i, haloIDset in enumerate(haloIDs):
            subIDset = GroupFirstSub[haloIDset]
            inds_cat, inds_sample = match3(cat['SubhaloIDs'], subIDset)

            num_matched = inds_cat.size if inds_cat is not None else 0

            print('sb = %d haloIDset = %d, num_matched = %d' % (sbNum,i,num_matched))

            masses = sP.subhalos('mhalo_200_log')[cat['SubhaloIDs']]
            print( np.nanmax(masses) )

    import pdb; pdb.set_trace()

def paperPlots():
    """ Testing. """
    TNG100  = simParams(res=1820,run='tng',redshift=0.0)
    TNG50   = simParams(res=2160,run='tng',redshift=0.0)
    TNG50_2 = simParams(res=1080,run='tng',redshift=0.0)
    TNG50_3 = simParams(res=540,run='tng',redshift=0.0)

    haloMassBins = [[12.3,12.7], [12.8, 13.2], [13.2, 14.0]]
    redshift = 0.5 # default for analysis

    def _get_halo_ids(sP_loc):
        """ Load and return the set of halo IDs in each haloMassBin. """
        mhalo = sP_loc.groupCat(fieldsSubhalos=['mhalo_200_log'])
        grnr  = sP_loc.groupCat(fieldsSubhalos=['SubhaloGrNr'])
        haloIDs = []

        for haloMassBin in haloMassBins:
            with np.errstate(invalid='ignore'):
                inds = np.where( (mhalo > haloMassBin[0]) & (mhalo < haloMassBin[1]) )[0]
            haloIDs.append(grnr[inds])

        return haloIDs

    # figure 1 - cgm resolution
    if 0:
        sPs = [TNG50, TNG50_2, TNG50_3]
        cenSatSelect = 'cen'

        simNames = '_'.join([sP.simName for sP in sPs])
        
        for radRelToVirRad in [True]: #,False]:
            saveName = 'resolution_profiles_%s_z%02d_%s_rvir%d.pdf' % (simNames,redshift*10,cenSatSelect,radRelToVirRad)

            radialResolutionProfiles(sPs, saveName, redshift=redshift, radRelToVirRad=radRelToVirRad, 
                                     cenSatSelect='cen', haloMassBins=haloMassBins)

    # figure 2 - cgm gas density/temp/pressure 1D PDFs
    if 0:
        sP = simParams(res=2160,run='tng',redshift=redshift)
        qRestrictions = [['rad_rvir',0.15,0.75]] # 0.15<r/rvir<0.5
        nBins = 150
        medianPDF = True

        if 1:
            ptProperty = 'nh'
            xlim = [-6.0, 0.0]
        if 0:
            ptProperty = 'temp'
            qRestrictions.append( ['sfr',0.0,0.0] ) # non-eEOS
            xlim = [3.5,7.5]
        if 0:
            ptProperty = 'gas_pres'
            qRestrictions.append( ['sfr',0.0,0.0] ) # non-eEOS
            xlim = [0.0, 5.0]

        # testing: one mass bin
        haloIDs = _get_halo_ids(sP)[1]

        # create density PDF of gas in radRangeRvir
        plotHistogram1D([sP], ptType='gas', ptProperty=ptProperty, xlim=xlim, nBins=nBins, medianPDF=medianPDF, 
                        qRestrictions=qRestrictions, haloIDs=[haloIDs])

    # fig 3 - cgm gas (n,T) 2D phase diagrams
    if 0:
        sP = simParams(res=2160,run='tng',redshift=redshift)
        qRestrictions = [['rad_rvir',0.15,0.75]] # 0.15<r/rvir<0.5

        xQuant = 'nh'
        yQuant = 'temp'
        xlim = [-5.5, -1.0]
        ylim = [3.8, 7.8]

        if 0:
            weights = ['mass']
            meancolors = None
            clim = [-4.0,0.0]
            contours = None
        if 1:
            # color by pressure
            weights = None
            meancolors = ['gas_pres']
            clim = [1.0,4.0]
            contours = [2.5,3.0,3.5]

        saveStr = weights[0] if weights is not None else meancolors[0]

        # testing: one mass bin, just one halo
        massBinInd = 1
        haloIDs = _get_halo_ids(sP)[massBinInd]

        pdf = PdfPages('%s_z-%.1f_%s_massbin%d.pdf' % (sP.simName,sP.redshift,saveStr,massBinInd))

        for haloID in haloIDs:
            plotPhaseSpace2D(sP, partType='gas', xQuant=xQuant, yQuant=yQuant, weights=weights, meancolors=meancolors, 
                         haloID=haloID, pdf=pdf, xlim=xlim, ylim=ylim, clim=clim, nBins=None, 
                         contours=contours, qRestrictions=qRestrictions)
        pdf.close()

    # fig 4 - vis (single halo large)
    if 0:
        sP = simParams(run='tng50-1', redshift=redshift)
        haloIDSets = _get_halo_ids(sP)

        # gas metallicity, N_MgII, N_HI, stellar light
        for conf in [1,2,3,4]:
            lrgHaloVisualization(sP, haloIDSets, conf=conf, gallery=True)
            #lrgHaloVisualization(sP, haloIDSets, conf=conf, gallery=False)

        # zoomed in (high res) visualizations of multiple properties of clumps
        for conf in [5,6]:
            lrgHaloVisualization(sP, None, conf=conf, gallery=False)

    # fig 5 - N_MgII or N_HI vs. b (map-derived): 2D histo of N_px/N_px_tot_annuli (normalized independently by column)
    if 0:
        sP = simParams(res=2160, run='tng', redshift=redshift)
        haloMassBin = haloMassBins[1]
        #ion = 'Mg II'
        ion = 'MHIGK_popping'
        radRelToVirRad = False

        for ycum in [True,False]:
            for fullDepth in [True,False]:
                ionColumnsVsImpact2D(sP, haloMassBin, ion=ion, radRelToVirRad=radRelToVirRad, ycum=ycum, fullDepth=fullDepth)

    # fig 6: obs matched samples for COS-LRG and LRG-RDR surveys
    if 0:
        sPs = [TNG50, TNG100]

        simNames = '_'.join([sP.simName for sP in sPs])
        obsSimMatchedGalaxySamples(sPs, 'sample_lrg_rdr_%s.pdf' % simNames, config='LRG-RDR')
        obsSimMatchedGalaxySamples(sPs, 'sample_cos_lrg_%s.pdf' % simNames, config='COS-LRG')

    # fig 7: run old OVI machinery to derive goodness of fit parameters and associated plots
    if 0:
        for sP in [TNG50]: #[TNG50, TNG100]:
            obsColumnsDataPlotExtended(sP, saveName='obscomp_lrg_rdr_hi_%s_ext.pdf' % sP.simName, config='LRG-RDR')
            obsColumnsDataPlotExtended(sP, saveName='obscomp_cos_lrg_hi_%s_ext.pdf' % sP.simName, config='COS-LRG HI')
            obsColumnsDataPlotExtended(sP, saveName='obscomp_cos_lrg_mgii_%s_ext.pdf' % sP.simName, config='COS-LRG MgII')

    # fig 8: 2pcf
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

    # fig 9: bound ion mass as a function of halo mass
    if 0:
        sPs = [TNG50] #[TNG100,TNG50] #,TNG50_2,TNG50_3] #[TNG300]
        cenSatSelect = 'cen'
        ions = ['AllGas_Mg','MgII','HIGK_popping']

        for vsHaloMass in [True,False]:
            massStr = '%smass' % ['stellar','halo'][vsHaloMass]

            saveName = 'ions_masses_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, cenSatSelect=cenSatSelect, 
                redshift=redshift, vsHaloMass=vsHaloMass) # , secondTopAxis=True

            saveName = 'ions_avgcoldens_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, cenSatSelect=cenSatSelect, 
                redshift=redshift, vsHaloMass=vsHaloMass, toAvgColDens=True)

    # fig 10: radial profiles
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

                    saveName = 'radprofiles_%s_%s_%s_z%02d_%s_rho%d_rvir%d.pdf' % \
                      (projDim,'-'.join(ions),simNames,redshift,cenSatSelect,massDensity,radRelToVirRad)
                    stackedRadialProfiles(sPs, saveName, ions=ions, redshift=redshift, massDensity=massDensity,
                                          radRelToVirRad=radRelToVirRad, cenSatSelect='cen', projDim=projDim, 
                                          haloMassBins=haloMassBins, combine2Halo=combine2Halo, median=True)


    # fig todo: tpcf of single halo (0.15 < r/rvir < 1.0)

    # fig 11: clump demographics: size distribution, total mass, average numdens, etc
    if 0:
        sP = simParams(run='tng50-1', redshift=redshift)
        clumpDemographics(sP)

    # fig 12: time tracks via tracers
    if 1:
        sP = simParams(run='tng50-1', redshift=redshift)
        clumpTracerTracks(sP, haloIDs=_get_halo_ids(sP))

    # fig todo: individual (or stacked) 'clump' properties, i.e. radial profiles, including pressure/cellsize


