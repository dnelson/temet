"""
projects/lrg.py
  Plots: LRG CGM paper (TNG50).
  in prep.
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from util import simParams
from util.helper import running_median, logZeroNaN, loadColorTable
from plot.config import *
from plot.general import plotStackedRadialProfiles1D, plotHistogram1D, plotPhaseSpace2D
from tracer.tracerMC import match3

def stackedRadialProfiles(sPs, saveName, redshift=0.3, cenSatSelect='cen', 
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

def paperPlots():
    """ Testing. """
    TNG100  = simParams(res=1820,run='tng',redshift=0.0)
    TNG50   = simParams(res=2160,run='tng',redshift=0.2)
    TNG50_2 = simParams(res=1080,run='tng',redshift=0.0)
    TNG50_3 = simParams(res=540,run='tng',redshift=0.0)

    haloMassBins = [[12.3,12.7], [12.8, 13.2], [13.2, 13.8]]

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
        redshift = 0.3
        sPs = [TNG50, TNG50_2, TNG50_3]
        cenSatSelect = 'cen'

        simNames = '_'.join([sP.simName for sP in sPs])
        
        for radRelToVirRad in [True]: #,False]:
            saveName = 'resolution_profiles_%s_z%02d_%s_rvir%d.pdf' % (simNames,redshift*10,cenSatSelect,radRelToVirRad)

            stackedRadialProfiles(sPs, saveName, redshift=redshift, radRelToVirRad=radRelToVirRad, 
                                  cenSatSelect='cen', haloMassBins=haloMassBins)

    # figure 2 - cgm gas density/temp/pressure 1D PDFs
    if 0:
        sP = simParams(res=2160,run='tng',redshift=0.5)
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
        sP = simParams(res=2160,run='tng',redshift=0.5)
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

    # fig 4 - vis (single halo large) gas density, metallicity, MgII
    # TODO: check MgII for eEOS
    if 1:
        from vis.halo import renderSingleHalo

        run        = 'tng'
        res        = 2160
        redshift   = 0.2
        rVirFracs  = [0.25]
        method     = 'sphMap' # sphMap_global for paper figure
        nPixels    = [1000,1000]
        axes       = [0,1]
        labelZ     = True
        labelScale = 'physical'
        labelSim   = False
        labelHalo  = True
        relCoords  = True
        rotation   = 'edge-on'

        size       = 400.0
        sizeType   = 'kpc'

        # which halo?
        sP = simParams(res=res, run=run, redshift=redshift)
        haloIDs = _get_halo_ids(sP)[1]

        haloID = haloIDs[0] # testing
        conf = 4 # testing

        # config
        if conf == 0:
            lines = ['H-alpha','H-beta','O--2-3728.81A','O--3-5006.84A','N--2-6583.45A','S--2-6730.82A']
            partField_loc = 'sb_%s_lum_kpc' % lines[0] # + '_sf0' to set SFR>0 cells to zero
            panels = [{'partType':'gas', 'partField':partField_loc, 'valMinMax':[34,41]}]

        if conf == 4:
            panels = [{'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-1.4,0.2]}]

        if conf == 5:
            panels = [{'partType':'gas', 'partField':'MHIGK_popping', 'valMinMax':[16.0,22.0]}]

        if conf == 6:
            panels = [{'partType':'gas', 'partField':'MH2GK_popping', 'valMinMax':[16.0,22.0]}]

        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels[0]
            colorbars    = True
            saveFilename = './vis_%s_%d_h%d_%s.pdf' % (sP.simName,sP.snap,haloID,panels[0]['partField'])

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    # fig 5 - N_MgII vs. b (derive from map) 2D histo of N_px/N_px_tot_annuli (normalized independently by column)
    # add data points

    # fig 6 - N_HI vs. b (same as above)
