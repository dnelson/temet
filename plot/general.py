"""
general.py
  General exploratory/diagnostic plots of particle-level data of single halos or entire boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
import warnings

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d

from util import simParams
from util.helper import loadColorTable, sampleColorTable, getWhiteBlackColors, running_median, \
  logZeroNaN, iterable, evenlySample, gaussian_filter_nan
from plot.quantities import quantList, simSubhaloQuantity, simParticleQuantity
from plot.config import *

def plotHistogram1D(sPs, ptType='gas', ptProperty='temp_linear', ptWeight=None, subhaloIDs=None, haloIDs=None, 
      ylog=True, ylim=None, xlim=None, qRestrictions=None, nBins=400, medianPDF=False, 
      legend=True, ctName=None, ctProp=None, colorbar=False, pdf=None):
    """ Simple 1D histogram/PDF of some quantity ptProperty of ptType, either of the whole box (subhaloIDs and haloIDs 
    both None), or of one or more halos/subhalos, where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    Alternatively, subhaloIDs/haloIDs can be a list of lists, each sub-list of IDs of objects for that run, which are overplotted.
    If ptWeight is None, uniform weighting, otherwise weight by this quantity.
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If ylim is None, then hard-coded typical limits for PDFs. If 'auto', then autoscale. Otherwise, 2-tuple to use as limits. 
    If medianPDF is True, then add this mean (per sP) on top. If meanPDF is 'only', then skip the individual objects. 
    If ctName is not None, sample from this colormap to choose line color per object. Assign based on the property ctProp.
    If colorbar is not False, then use this field (string) to display a colorbar mapping. """

    # config
    if ylog:
        ylabel = 'PDF [ log ]'
        if ylim is None:
            ylim = [-3.0, 0.5]
        else:
            if ylim == 'auto': ylim = None
    else:
        ylim = [0.0, 1.0]
        ylabel = 'PDF'

    # inputs
    oneObjPerRun = False

    assert np.sum(e is not None for e in [haloIDs,subhaloIDs]) in [0,1] # pick one, or neither
    if subhaloIDs is not None:
        assert (len(subhaloIDs) == len(sPs)) or len(sPs) == 1 # one subhalo ID per sP, or one sP
        assert isinstance(subhaloIDs, (list,np.ndarray))
        if not isinstance(subhaloIDs[0], (list,np.ndarray)):
            assert len(subhaloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = subhaloIDs

    if haloIDs is not None:
        assert (len(haloIDs) == len(sPs)) or len(sPs) == 1 # one subhalo ID per sP, or one sP
        assert isinstance(haloIDs, (list,np.ndarray))
        if not isinstance(haloIDs[0], (list,np.ndarray)):
            assert len(haloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = haloIDs

    # load
    haloLims = (subhaloIDs is not None or haloIDs is not None)
    xlabel, xlim_quant, xlog = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean, haloLims=haloLims)
    if xlim is None: xlim = xlim_quant

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # loop over halo/subhalo IDs
        if subhaloIDs is not None or haloIDs is not None:
            if not oneObjPerRun:
                sP_objIDs = objIDs[i] # list
            else:
                sP_objIDs = [objIDs[i]]

        if ctName is not None:
            #colors = sampleColorTable(ctName, len(sP_objIDs), bounds=[0.1,0.9])
            if haloIDs is not None: cmap_props = sP.halos(ctProp)[sP_objIDs]
            if subhaloIDs is not None: cmap_props = sP.subhalos(ctProp)[sP_objIDs]
            cmap = loadColorTable(ctName, fracSubset=[0.2,0.9])
            cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=cmap_props.min(), vmax=cmap_props.max()), cmap=cmap)

        yy_save = np.zeros( (nBins,len(sP_objIDs)), dtype='float32' )

        for j, objID in enumerate(sP_objIDs):
            # load
            load_haloID = objID if haloIDs is not None else None
            load_subID = objID if subhaloIDs is not None else None

            vals = sP.snapshotSubset(ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)
            if xlog: vals = np.log10(vals)

            # weights
            if ptWeight is None:
                weights = np.zeros( vals.size, dtype='float32' ) + 1.0
            else:
                weights = sP.snapshotSubset(ptType, ptWeight, haloID=load_haloID, subhaloID=load_subID)

            # arbitrary property restriction(s)?
            if qRestrictions is not None:
                mask = np.zeros( vals.size, dtype='int16' )
                for rFieldName, rFieldMin, rFieldMax in qRestrictions:
                    # load and update mask
                    r_vals = sP.snapshotSubset(ptType, rFieldName, haloID=load_haloID, subhaloID=load_subID)

                    wRestrict = np.where( (r_vals < rFieldMin) | (r_vals > rFieldMax) )
                    mask[wRestrict] = 1
                    print('[%d] restrict [%s] eliminated [%d] of [%d] = %.2f%%' % \
                        (objID,rFieldName,len(wRestrict[0]),mask.size,len(wRestrict[0])/mask.size*100))

                # apply mask
                wRestrict = np.where(mask == 0)
                vals = vals[wRestrict]
                weights = weights[wRestrict]

            #print(sP.simName,', min: ', np.nanmin(vals), ' max: ', np.nanmax(vals), ' median: ', np.nanmedian(vals))
            #print(sP.simName, ', 5,10,90,95 percentiles: ',np.nanpercentile(vals, [5,10,90,95]))

            # histogram (all equivalent methods)
            bins = np.linspace( xlim[0], xlim[1], nBins+1 )
            xx = bins[:-1] + (bins[1]-bins[0])/2
            binsize = bins[1]-bins[0]

            #yy4 = np.zeros( nBins, dtype='float32' )
            #for j in range(nBins):
            #    w = np.where( (vals >= bins[j]) & (vals < bins[j+1]) )
            #    yy4[j] = np.sum(weights[w])
            #yy4 /= (np.sum(weights)*binsize)
            
            #yy2, xx2, _ = binned_statistic(vals, weights, statistic='sum', bins=bins)
            #yy2 /= (vals.size*binsize)
            #yy3, xx3 = np.histogram(vals, bins=bins, weights=weights, density=False)
            #yy3 /= (vals.size*binsize)
            yy, xx4 = np.histogram(vals, bins=bins, weights=weights, density=True)

            if ylog: yy = logZeroNaN(yy)

            yy_save[:,j] = yy

            # plot
            if xx.size > sKn:
                yy = savgol_filter(yy,sKn,sKo)

            label = '%s [%d]' % (sP.simName,objID) if len(sPs) > 1 else str(objID)
            ls = ':' if sP.simName == 'L11 (Primordial Only)' else '-'
            lw_loc = lw-1 if len(sP_objIDs) < 10 else 1
            color = None if len(sP_objIDs) < 10 else (None if j == 0 else l.get_color())
            if ctName is not None: color = cmap.to_rgba(cmap_props[j]) #color = colors[j]

            if medianPDF != 'only':
                l, = ax.plot(xx, yy, linestyle=ls, lw=lw_loc, color=color, label=label)

        # add mean?
        if len(sP_objIDs) > 1 and medianPDF:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # 'mean of empty slice'
                yy1 = np.nanmean(yy_save, axis=1)
                yy2 = np.nanmedian(yy_save, axis=1)
                yy3 = yy_save[:,int(yy_save.shape[1]/2)]

            if xx.size > sKn:
                yy1 = savgol_filter(yy1,sKn,sKo)
                yy2 = savgol_filter(yy2,sKn+4,sKo)
                yy3 = savgol_filter(yy3,sKn,sKo)

            #ax.plot(xx, yy1, linestyle='-', color='black', lw=lw, alpha=0.8, label='mean')
            #ax.plot(xx, yy3, linestyle='-', color='black', lw=lw, alpha=0.8, label='middle')
            ax.plot(xx, yy2, linestyle='-', color='black', lw=lw, alpha=1.0, label='median')

    # finish plot
    if legend:
        ax.legend(loc='best', ncol=3)

    if colorbar:
        #cb_axes = inset_locator.inset_axes(ax, width='40%', height='4%', loc=[0.2,0.8])
        cb_axes = fig.add_axes([0.2,0.9,0.4,0.04])
        _, label, _, _ = sP.simSubhaloQuantity(ctProp)
        plt.colorbar(cmap, label=label, cax=cb_axes, orientation='horizontal')

    # save plot
    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    hStr = 'global'
    if haloIDs is not None: hStr = 'haloIDs-n%d' % len(haloIDs)
    elif subhaloIDs is not None: hStr = 'subhIDs-n%d' % len(subhaloIDs)

    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('histo1D_%s_%s_%s_wt-%s_%s.pdf' % (sPstr,ptType,ptProperty,ptWeight,hStr))
    plt.close(fig)

def _draw_special_lines(sP, ax, ptProperty):
    """ Helper. Draw some common overlays. """
    if ptProperty in ['tcool','tff']:
        tage = np.log10( sP.units.redshiftToAgeFlat(0.0) )
        ax.plot(ax.get_xlim(), [tage,tage], ':', lw=lw, alpha=0.3, color='#ffffff')
        
    if ptProperty in ['tcool_tff']:
        ax.plot(ax.get_xlim(), [0.0, 0.0], ':', lw=lw, alpha=0.3, color='#ffffff')
        ax.plot(ax.get_xlim(), [1.0, 1.0], ':', lw=lw, alpha=0.3, color='#ffffff')

def plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', weights=['mass'], meancolors=None, haloIDs=None, 
                     xlim=None, ylim=None, clim=None, contours=None, contourQuant=None, normColMax=False, hideBelow=False, 
                     ctName='viridis', colorEmpty=False, smoothSigma=0.0, nBins=None, qRestrictions=None, median=False, 
                     normContourQuantColMax=False, addHistX=False, addHistY=False, colorbar=True, f_pre=None, f_post=None, pdf=None):
    """ Plot a 2D phase space plot (arbitrary values on x/y axes), for a list of halos, or for an entire box 
    (if haloIDs is None). weights is a list of the gas properties to weight the 2D histogram by, 
    if more than one, a horizontal multi-panel plot will be made with a single colorbar. Or, if meancolors is 
    not None, then show the mean value per pixel of these quantities, instead of weighted histograms.
    If xlim,ylim,clim specified, then use these bounds, otherwise use default/automatic bounds.
    If contours is not None, draw solid contours at these levels on top of the 2D histogram image. If contourQuant is None, 
    then the histogram itself (or meancolors) is used, otherwise this quantity is used.
    if normColMax, then normalize every column to its maximum (i.e. conditional 2D PDF).
    If normContourQuantColMax, same but for a specified contourQuant.
    If f_pre, f_post are not None, then these are 'custom' functions accepting the axis as a single argument, which 
    are called before and after the rest of plotting, respectively.
    If addHistX and/or addHistY, then specifies the number of bins to add marginalized 1D histogram(s).
    If hideBelow, then pixel values below clim[0] are left pure white. 
    If colorEmpty, then empty/unoccupied pixels are colored at the bottom of the cmap.
    If smoothSigma is not zero, gaussian smooth contours at this level. 
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If median, add a median line of the yQuant as a function of the xQuant. """

    # config
    nBins2D = None

    if nBins is None:
        # automatic (2d binning set below based on aspect ratio)
        nBins = 200
        if sP.isZoom:
            nBins = 150
    else:
        if isinstance(nBins,(list,np.ndarray)):
            # fully specified
            nBins2D = nBins
        else:
            # one-dim specified (2d binning set below based on aspect ratio)
            nBins = nBins

    sizefac = 0.9
    clim_default = [-6.0,0.0]

    # binned_statistic_2d instead of histogram2d?
    binnedStat = False
    if meancolors is not None:
        assert weights is None # one or the other
        binnedStat = True
        weights = iterable(meancolors) # loop over these instead
    if weights is None:
        # one or the other
        assert meancolors is not None

    contoursColor = 'k' # black

    def _load_all_halos(sP, partType, partField, haloIDs, GroupLenType=None):
        # global box load?
        if haloIDs is None:
            return sP.snapshotSubsetP(partType, partField)

        # set of halos: get total load size
        if GroupLenType is None:
            GroupLenType = sP.halos('GroupLenType')[:,sP.ptNum(partType)]
        loadSize = np.sum(GroupLenType[haloIDs])

        # allocate
        vals = np.zeros(loadSize, dtype='float32')

        offset = 0

        # load each
        for haloID in haloIDs:
            vals[offset:offset+GroupLenType[haloID]] = sP.snapshotSubset(partType, partField, haloID=haloID)
            offset += GroupLenType[haloID]

        return vals

    # load: x-axis
    xlabel, xlim_quant, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloIDs is not None))
    if xlim is None: xlim = xlim_quant
    xvals = _load_all_halos(sP, partType, xQuant, haloIDs)

    if xlog: xvals = logZeroNaN(xvals)

    # load: y-axis
    ylabel, ylim_quant, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloIDs is not None))
    if ylim is None: ylim = ylim_quant
    yvals = _load_all_halos(sP, partType, yQuant, haloIDs)

    if ylog: yvals = logZeroNaN(yvals)

    # arbitrary property restriction(s)?
    if qRestrictions is not None:
        mask = np.zeros( xvals.size, dtype='int16' )
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # load and update mask
            r_vals = _load_all_halos(sP, partType, rFieldName, haloIDs)

            wRestrict = np.where( (r_vals < rFieldMin) | (r_vals > rFieldMax) )
            mask[wRestrict] = 1
            print(' restrict [%s] eliminated [%d] of [%d] = %.2f%%' % \
                (rFieldName,len(wRestrict[0]),mask.size,len(wRestrict[0])/mask.size*100))

        # apply mask
        wRestrict = np.where(mask == 0)
        xvals = xvals[wRestrict]
        yvals = yvals[wRestrict]

    # start figure
    fig = plt.figure(figsize=figsize)

    # loop over each weight requested
    for i, wtProp in enumerate(weights):
        # load: weights
        weight = _load_all_halos(sP, partType, wtProp, haloIDs)

        if qRestrictions is not None:
            weight = weight[wRestrict]

        # add panel
        ax = fig.add_subplot(1,len(weights),i+1)

        if f_pre is not None:
            f_pre(ax)

        if len(weights) == 1: # title
            hStr = 'fullbox' if haloIDs is None else 'halos%s' % '-'.join([str(h) for h in haloIDs])
            wtStr = partType.capitalize() + ' ' + wtProp.capitalize()
            #ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if nBins2D is None:
            bbox = ax.get_window_extent()
            nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])

        if binnedStat:
            # remove NaN weight points prior to binning (default op is mean, not nanmean)
            assert not normColMax

            w_fin = np.where(np.isfinite(weight))
            xvals = xvals[w_fin]
            yvals = yvals[w_fin]
            weight = weight[w_fin]
                
            # plot 2D image, each pixel colored by the mean value of a third quantity
            clabel, clim_quant, clog = simParticleQuantity(sP, partType, wtProp, clean=clean, haloLims=(haloIDs is not None))
            wtStr = clabel # 'Mean ' + clabel
            zz, _, _, _ = binned_statistic_2d(xvals, yvals, weight, 'mean', # median unfortunately too slow
                                              bins=nBins2D, range=[xlim,ylim])
            zz = zz.T
            if clog: zz = logZeroNaN(zz)

            if clim is None:
                clim = clim_quant # colorbar limits
        else:
            # plot 2D histogram image, optionally weighted
            zz, _, _ = np.histogram2d(xvals, yvals, bins=nBins2D, range=[xlim,ylim], 
                                      normed=True, weights=weight)
            zz = zz.T

            if normColMax:
                colMax = np.nanmax(zz, axis=0)
                w = np.where(colMax == 0)
                colMax[w] = 1.0 # entire column is zero, will be log->nan anyways then not shown
                zz /= colMax[np.newaxis,:]

            zz = logZeroNaN(zz)

        if clim is None:
            clim = clim_default

        if hideBelow:
            w = np.where(zz < clim[0])
            zz[w] = np.nan
        if colorEmpty:
            w = np.where(np.isnan(zz))
            zz[w] = clim[0]

        cmap = loadColorTable(ctName)
        norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

        _draw_special_lines(sP, ax, yQuant)

        # plot contours?
        if contours is not None:

            if contourQuant is not None:
                # load a different quantity for the contouring
                contourq = _load_all_halos(sP, partType, contourQuant, haloIDs)

                if qRestrictions is not None: contourq = contourq[wRestrict]
                if binnedStat: contourq = contourq[w_fin]

                if contourQuant == 'mass':
                    zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBins2D[0]/2, nBins2D[1]/2], range=[xlim,ylim], 
                                              normed=True, weights=contourq)
                else:
                    zz, xc, yc, _ = binned_statistic_2d(xvals, yvals, contourq, 'mean',
                                                        bins=[nBins2D[0]/4, nBins2D[1]/4], range=[xlim,ylim])

                _, _, qlog = simParticleQuantity(sP, partType, contourQuant)

                if normContourQuantColMax:
                    assert contourQuant == 'mass' # otherwise does it make sense?
                    colMax = np.nanmax(zz, axis=0)
                    w = np.where(colMax == 0)
                    colMax[w] = 1.0 # entire column is zero, will be log->nan anyways then not shown
                    zz /= colMax[np.newaxis,:]

                if qlog: zz = logZeroNaN(zz)
            else:
                # contour the same quantity
                if binnedStat:
                    zz, xc, yc, _ = binned_statistic_2d(xvals, yvals, weight, 'mean',
                                                        bins=[nBins2D[0]/4, nBins2D[1]/4], range=[xlim,ylim])
                    if clog: zz = logZeroNaN(zz)
                else:
                    zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBins2D[0]/4, nBins2D[1]/4], range=[xlim,ylim], 
                                                normed=True, weights=weight)
                    zz = logZeroNaN(zz)

            XX, YY = np.meshgrid(xc[:-1], yc[:-1], indexing='ij')

            # smooth, ignoring NaNs
            zz = gaussian_filter_nan(zz, smoothSigma)

            c = plt.contour(XX, YY, zz, contours, colors=contoursColor, linestyles='solid', alpha=0.6)

        if len(weights) > 1: # text label inside panel
            wtStr = 'Gas Oxygen Ion Mass'
            labelText = wtProp.replace(" mass","").replace(" ","")
            ax.text(xlim[0]+0.3, yMinMax[-1]-0.3, labelText, 
                va='top', ha='left', color='black', fontsize='40')

        # median/percentiles line(s)?
        if median:
            binSize = (xlim[1] - xlim[0]) / nBins2D[0] * 5
            xm, ym, sm, pm = running_median(xvals, yvals, binSize=binSize, percs=[16,50,84])
            ax.plot(xm, ym, '-', lw=lw, color='black', alpha=0.5)

        # special behaviors
        if haloIDs is not None and xQuant in ['rad','rad_kpc','rad_kpc_linear']:
            # mark virial radius
            textOpts = {'rotation':90.0, 'ha':'right', 'va':'bottom', 'fontsize':18, 'color':'#ffffff'}
            rvir = sP.groupCatSingle(haloID=haloIDs[0])['Group_R_Crit200']
            if '_kpc' in xQuant: rvir = sP.units.codeLengthToKpc(rvir)

            for fac in [1,2,4,10]: #,100]:
                xx = rvir/fac if '_linear' in xQuant else np.log10(rvir/fac)
                dy = ax.get_ylim()[1] - ax.get_ylim()[0]
                yy = [ax.get_ylim()[0]+dy*0.1, ax.get_ylim()[0]+dy*0.2] # [1]*0.8, [1]*0.98
                xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                if xx >= ax.get_xlim()[1]: continue

                ax.plot( [xx,xx], yy, '-', lw=lw, color=textOpts['color'])
                ax.text( xx - xoff, yy[0], '$r_{\\rm vir}$/%d'%fac if fac != 1 else '$r_{\\rm vir}$', **textOpts)
        
        if yQuant == 'vrad' and ax.get_ylim()[1] > 0 and ax.get_ylim()[0] < 0:
            # mark inflow-outflow boundary
            ax.plot(ax.get_xlim(), [0,0], '-', lw=lw, color='#000000', alpha=0.5)

        if xQuant == 'density' and yQuant == 'temp':
            # add Torrey+12 'ISM cut' line
            xx = np.array(ax.get_xlim())
            yy = 6.0 + 0.25*xx
            ax.plot(xx, yy, '-', lw=lw, color='#000000', alpha=0.7, label='Torrey+12 ISM cut')

        if f_post is not None:
            f_post(ax)

    # marginalized 1D distributions
    aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]

    height = 0.12
    hpad = 0.004
    width = 0.12 #* aspect
    wpad = 0.004 #* aspect
    color = '#555555'

    if addHistX:
        # horizontal histogram on the top
        fig.tight_layout()
        fig.set_tight_layout(False)


        rect = ax.get_position().bounds # [left,bottom,width,height]
        ax.set_position([rect[0],rect[1],rect[2],rect[3]-height-hpad*2])
        
        rect_new = [rect[0], rect[1]+rect[3]-height+hpad, rect[2], height]
        if addHistY: rect_new[2] -= (width+wpad*2) # pre-emptively adjust width
        ax_histx = fig.add_axes(rect_new)
        ax_histx.tick_params(direction='in', labelbottom=False, left=False, labelleft=False)
        ax_histx.set_xlim(ax.get_xlim())
        ax_histx.hist(xvals, bins=addHistX, range=ax.get_xlim(), weights=weight, 
                      color=color, log=False, alpha=0.7)
        assert len(weights) == 1 # otherwise do multiple histograms
        colorbar = False # disable colorbar

    if addHistY:
        # vertical histogram on the right
        if not addHistX: fig.tight_layout()
        fig.set_tight_layout(False)

        rect = ax.get_position().bounds # [left,bottom,width,height]
        ax.set_position([rect[0],rect[1],rect[2]-width-wpad*2,rect[3]])

        ax_histy = fig.add_axes([rect[0]+rect[2]-width+wpad, rect[1], width, rect[3]])
        ax_histy.tick_params(direction='in', labelleft=False, bottom=False, labelbottom=False)
        ax_histy.set_ylim(ax.get_ylim())
        hist, bins, patches = ax_histy.hist(yvals, bins=addHistY, range=ax.get_ylim(), weights=weight, 
                                            color=color, log=False, alpha=0.7, orientation='horizontal')

        if yQuant == 'vrad':
            # special coloring: red for negative (inflow), blue for positive (outflow)
            colors = [sampleColorTable('tableau10','red'), sampleColorTable('tableau10','blue')]
            binsize = bins[1] - bins[0]
            w_neg = np.where(bins[:-1]+binsize/2 < 0.0)[0]
            w_pos = np.where(bins[:-1]+binsize/2 > 0.0)[0]
            for ind in w_neg: patches[ind].set_facecolor(colors[0])
            for ind in w_pos: patches[ind].set_facecolor(colors[1])

            # write fractions
            w_neg = np.where( yvals < 0 )
            w_pos = np.where( yvals > 0 )
            textOpts = {'ha':'center', 'fontsize':18, 'transform':ax_histy.transAxes}
            ax_histy.text(0.5, 0.06, 'inflow\n%.2f' % (weight[w_neg].sum()/weight.sum()), va='bottom', color=colors[0], **textOpts)
            ax_histy.text(0.5, 0.94, 'outflow\n%.2f' % (weight[w_pos].sum()/weight.sum()), va='top', color=colors[1], **textOpts)

        if yQuant == 'vrad' and 0:
            # second histogram: only data >vcirc or <vcirc (i.e. exclude galaxy itself)
            vcirc = sP.subhalos('SubhaloVmax')[sP.halos('GroupFirstSub')[haloIDs]] # km/s
            vel_thresh = np.median(vcirc) / 3

            w_nondisk = np.where( (yvals > vel_thresh) | (yvals < -vel_thresh) )

            hist, bins, patches = ax_histy.hist(yvals[w_nondisk], bins=addHistY, range=ax.get_ylim(), weights=weight[w_nondisk], 
                                                color=color, log=False, alpha=0.7, orientation='horizontal', linestyle=':', lw=lw)

        assert len(weights) == 1 # otherwise do multiple histograms
        colorbar = False # disable colorbar

    # colorbar
    if colorbar:
        if xQuant == 'hdens' and yQuant == 'temp' and len(weights) == 3:
            # TNG colors paper
            fig.subplots_adjust(right=0.93)
            cbar_ax = fig.add_axes([0.94, 0.131, 0.02, 0.831]) 
        else:
            # more general
            fig.subplots_adjust(right=0.89)
            cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)

        cb = plt.colorbar(im, cax=cbar_ax)
        if not binnedStat: wtStr = 'Relative ' + wtStr + ' [ log ]'
        cb.ax.set_ylabel(wtStr)

    # save
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('phase2d_%s_%d_%s_x-%s_y-%s_wt-%s_%s.pdf' % \
            (sP.simName,sP.snap,partType,xQuant,yQuant,
            "-".join([w.replace(" ","") for w in weights]),
            "nh%d" % len(haloIDs) if haloIDs is not None else 'fullbox') )
    plt.close(fig)

def plotParticleMedianVsSecondQuant(sPs, partType='gas', xQuant='hdens', yQuant='Si_H_numratio', 
                                    haloID=None, radMinKpc=None, radMaxKpc=None):
    """ Plot the (median) relation between two particle properties for a single halo (if haloID is not None), 
    or the whole box (if haloID is None). If a halo is specified, optionally restrict to a given 
    [radMinKpc,radMaxKpc] radial range, specified in physical kpc."""

    # config
    nBins = 50
    lw = 3.0

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    sStr = '%s z=%.1f' % (sPs[0].simName, sPs[0].redshift) if len(sPs) == 1 else ''
    ax.set_title('%s %s' % (sStr,hStr))

    for i, sP in enumerate(sPs):
        # load
        xlabel, xlim, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
        sim_xvals = sP.snapshotSubset(partType, xQuant, haloID=haloID)
        if xlog: sim_xvals = logZeroNaN(sim_xvals)

        ylabel, ylim, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
        sim_yvals = sP.snapshotSubset(partType, yQuant, haloID=haloID)
        if ylog: sim_yvals = logZeroNaN(sim_yvals)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # radial restriction
        if radMaxKpc is not None or radMinKpc is not None:
            assert haloID is not None
            rad = sP.snapshotSubset(partType, 'rad_kpc', haloID=haloID)
            
            if radMinKpc is None:
                w = np.where( (rad <= radMaxKpc) )
            elif radMaxKpc is None:
                w = np.where( (rad > radMinKpc) )
            else:
                w = np.where( (rad > radMinKpc) & (rad <= radMaxKpc) )

            sim_xvals = sim_xvals[w]
            sim_yvals = sim_yvals[w]

            if radMinKpc is not None:
                hStr += '_rad_gt_%.1fkpc' % radMinKpc
            if radMaxKpc is not None:
                hStr += '_rad_lt_%.1fkpc' % radMaxKpc

        # median and 10/90th percentile lines
        binSize = (xlim[1]-xlim[0]) / nBins

        xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
        xm = xm[1:-1]
        ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
        sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
        pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

        c = next(ax._get_lines.prop_cycler)['color']
        ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

        # percentile:
        if len(sPs) <= 3 or (len(sPs) > 3 and i == 0):
            ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    ax.legend(loc='best')

    # finish plot
    sStr = '%s_z-%.1f' % (sPs[0].simName,sPs[0].redshift) if len(sPs) == 1 else 'sPn%d' % len(sPs)
    fig.savefig('particleMedian_%s_%s-vs-%s_%s_%s.pdf' % (partType,xQuant,yQuant,sStr,hStr))
    plt.close(fig)

def plotStackedRadialProfiles1D(sPs, subhaloIDs=None, haloIDs=None, ptType='gas', ptProperty='temp_linear', op='mean', weighting=None, 
                                ptRestrictions=None, proj2D=None, xlim=None, ylim=None, plotMedian=True, plotIndiv=False, 
                                ctName=None, ctProp=None, colorbar=False, figsize=figsize):
    """ Radial profile(s) of some quantity ptProperty of ptType vs. radius from halo centers 
    (parent FoF particle restricted, using non-caching auxCat functionality). 
    subhaloIDs is a list, one entry per sPs entry. For each entry of subhaloIDs:
    If subhaloIDs[i] is a single subhalo ID number, then one halo only. If a list, then median stack.
    If a dict, then k:v pairs where keys are a string description, and values are subhaloID lists, which 
    are then overplotted. sPs supports one or multiple runs to be overplotted. 
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs. 
    ptType and ptProperty specify the quantity to bin, and op (mean, sum, min, max) the operation to apply in each bin.
    If ptRestrictions, then a dictionary containing k:v pairs where k is fieldName, v is a 2-tuple [min,max], 
      to restrict all cells/particles by, e.g. sfrgt0 = {'StarFormationRate':['gt',0.0]}, sfreq0 = {'StarFormationRate':['eq',0.0]}.
    if proj2D is not None, then a 2-tuple as input to subhaloRadialProfile().
    If plotMedian is False, then skip the average profile.
    if plotIndiv, then show individual profiles, and in this case:
    If ctName is not None, sample from this colormap to choose line color per object. Assign based on the property ctProp.
    If colorbar is not False, then use this field (string) to display a colorbar mapping. """
    from cosmo.auxcatalog import subhaloRadialProfile
    from tracer.tracerMC import match3

    # config
    if xlim is None: xlim = [0.0,3.0] # for plot only [loc pkpc]
    percs = [16,84]
    scope = 'fof' # fof, subfind

    # sanity checks
    assert subhaloIDs is not None or haloIDs is not None # pick one
    if subhaloIDs is None: subhaloIDs = haloIDs # use halo ids
    if isinstance(subhaloIDs,int) and len(sPs) == 1: subhaloIDs = [subhaloIDs] # single number to list (one sP case)
    assert (len(subhaloIDs) == len(sPs)) # one subhalo ID list per sP

    ylabel, ylim2, ylog = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean)
    if ylim is None: ylim = ylim2

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Radius [ log pkpc ]')
    ax.set_ylabel(ylabel)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    _draw_special_lines(sPs[0], ax, ptProperty)

    # loop over simulations
    for i, sP in enumerate(sPs):
        objIDs = subhaloIDs[i] # for this run

        # subhalo is a single number or dict? make a concatenated list
        if isinstance(objIDs,int):
            objIDs = [objIDs]
        if isinstance(objIDs,dict):
            objIDs = np.hstack( [objIDs[key] for key in objIDs.keys()])

        if haloIDs is not None:
            # transform fof ids to subhalo ids
            firstsub = sP.groupCat(fieldsHalos=['GroupFirstSub'])
            objIDs = firstsub[objIDs]

        if ctName is not None:
            #colors = sampleColorTable(ctName, len(sP_objIDs), bounds=[0.1,0.9])
            cmap_props = sP.subhalos(ctProp)[objIDs]
            cmap = loadColorTable(ctName, fracSubset=[0.2,0.9])
            cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=cmap_props.min(), vmax=cmap_props.max()), cmap=cmap)

        # load
        data, attrs = subhaloRadialProfile(sP, pSplit=None, ptType=ptType, ptProperty=ptProperty, op=op, 
                                          scope=scope, weighting=weighting, subhaloIDsTodo=objIDs, proj2D=proj2D, 
                                          ptRestrictions=ptRestrictions)
        assert data.shape[0] == len(objIDs)

        nSamples = 1 if not isinstance(subhaloIDs[i],dict) else len(subhaloIDs[i].keys())

        for j in range(nSamples):
            # crossmatch attrs['objIDs'] with subhalo[key] sub-list if needed
            subIDsLoc = subhaloIDs[i][list(subhaloIDs[i].keys())[j]] if isinstance(subhaloIDs[i],dict) else objIDs
            w, _ = match3( attrs['subhaloIDs'], subIDsLoc )
            assert len(w) == len(subIDsLoc)

            # calculate median radial profile and scatter
            yy_indiv = data[w,:]
            yy_mean = np.nanmean( data[w,:], axis=0 )
            yy_median = np.nanmedian( data[w,:], axis=0 )
            yp = np.nanpercentile( data[w,:], percs, axis=0 )

            if proj2D is not None:
                print('Normalizing to column density (could use generalization, i.e. mass fields only).')
                # [code mass] -> [code mass / code length^2]
                yy_indiv /= attrs['bin_areas_code']
                yy_mean /= attrs['bin_areas_code']
                yy_median /= attrs['bin_areas_code']
                yp /= attrs['bin_areas_code']

                if 0:
                    # [code mass / code length^2] -> [H atoms/cm^2], celineHIH2Profiles
                    cgs      = True #False
                    numDens  = True #False
                    msunKpc2 = False #True

                    ax.set_ylabel('Column Density [H atoms / cm$^2$]')
                    ax.set_ylim([14,22])
                    ax.set_xlim([0.5,2.5])
                if 1:
                    # [code mass / code length^2] -> DM Surface Density [Msun/kpc^2], burkert
                    cgs      = False
                    numDens  = False
                    msunKpc2 = True

                    ax.set_ylabel('DM Surface Density [ log M$_{\\rm sun}$ kpc$^{-2}$ ]')
                    ax.set_ylim([7,9.5])
                    ax.set_xlim([0.2,1.2])

                yy_indiv  = sP.units.codeColDensToPhys(yy_indiv, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yy_mean   = sP.units.codeColDensToPhys(yy_mean, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yy_median = sP.units.codeColDensToPhys(yy_median, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yp        = sP.units.codeColDensToPhys(yp, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)

            if ylog:
                yy_indiv  = logZeroNaN(yy_indiv)
                yy_median = logZeroNaN(yy_median)
                yy_mean   = logZeroNaN(yy_mean)
                yp = logZeroNaN(yp)

            rr = logZeroNaN(attrs['rad_bins_pkpc'])

            if rr.size > sKn:
                yy_indiv  = savgol_filter(yy_indiv,sKn,sKo,axis=1)
                yy_mean   = savgol_filter(yy_mean,sKn,sKo)
                yy_median = savgol_filter(yy_median,sKn,sKo)
                yp = savgol_filter(yp,sKn,sKo,axis=1) # P[10,90]

            # plot median scatter band?
            if plotMedian and len(sPs) == 1 and objIDs.size > 1:
                colorMed = None if not plotIndiv else 'black'
                w = np.where(np.isfinite(yp[0,:]) & np.isfinite(yp[-1,:]))[0]
                ax.fill_between(rr[w], yp[0,w], yp[-1,w], color=colorMed, interpolate=True, alpha=0.1)

            # plot individual?
            if plotIndiv:
                for k in range(yy_indiv.shape[0]):
                    color = 'black'
                    if ctName is not None: color = cmap.to_rgba(cmap_props[k]) #color = colors[j]
                    ax.plot(rr, yy_indiv[k,:], '-', lw=lw, color=color, alpha=0.6)

            # plot stack
            if plotMedian:
                sampleDesc = '' if nSamples == 1 else list(subhaloIDs[i].keys())[j]
                label = '%s %s' % (sP.simName,sampleDesc) if len(sPs) > 1 else sampleDesc
                ax.plot(rr, yy_median, '-', lw=lw, color=colorMed, label=label.strip())

            # save to text file (not generalized)
            if 0:
                filename = 'radprof_stacked_%s.txt' % ptProperty
                out = '# %s z=%.1f %s %s %s %s %s\n' % (sP.simName, sP.redshift, ptType, ptProperty, op, weighting, proj2D)
                out += '# r [log pkpc], N [log cm^-2], p%d, p%d\n' % (percs[0],percs[1])
                for k in range(rr.size):
                    out += '%8.5f %6.3f %6.3f %6.3f\n' % (rr[k], yy_median[k], yp[0,k], yp[-1,k])
                with open(filename, 'w') as f:
                    f.write(out)

    # finish plot
    ax.legend(loc='best')

    if colorbar:
        #cb_axes = inset_locator.inset_axes(ax, width='40%', height='4%', loc=[0.2,0.8])
        #cb_axes = fig.add_axes([0.48,0.2,0.35,0.04]) # x0, y0, width, height
        cb_axes = fig.add_axes([0.3,0.3,0.4,0.04]) # x0, y0, width, height
        _, label, _, _ = sP.simSubhaloQuantity(ctProp)
        plt.colorbar(cmap, label=label, cax=cb_axes, orientation='horizontal')

    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    wtStr = '_wt-'+weighting if weighting is not None else ''
    fig.savefig('radProfilesStack_%s_%s_%s_Ns-%d_Nh-%d_scope-%s%s.pdf' % \
        (sPstr,ptType,ptProperty,nSamples,len(objIDs),scope,wtStr))
    plt.close(fig)

def plotSingleRadialProfile(sPs, ptType='gas', ptProperty='temp_linear', subhaloIDs=None, haloIDs=None, 
    xlog=True, xlim=None, ylog=None, ylim=None, sfreq0=False, colorOffs=None, scope='fof'):
    """ Radial profile of some quantity ptProperty of ptType vs. radius from halo center,
    where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs. 
    Scope can be: global, fof, subfind. """

    # config
    if xlog:
        if xlim is None: xlim = [-0.5,3.0]
        xlabel = 'Radius [ log pkpc ]'
    else:
        if xlim is None: xlim = [0.0,500.0]
        xlabel = 'Radius [ pkpc ]'

    percs = [10,25,75,90]
    nRadBins = 40
    lw = 2.0

    assert np.sum(e is not None for e in [haloIDs,subhaloIDs]) == 1 # pick one
    if subhaloIDs is not None: assert (len(subhaloIDs) == len(sPs)) # one subhalo ID per sP
    if haloIDs is not None: assert (len(haloIDs) == len(sPs)) # one subhalo ID per sP

    ylabel, ylim_q, ylog_q = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean, haloLims=True)
    if ylim is None: ylim = ylim_q
    if ylog is None: ylog = ylog_q

    # start plot
    fig = plt.figure(figsize=(11.2,8.0)) #(11.2,8.0) #(14,10)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Galactocentric Radius [ log pkpc ]')
    ax.set_ylabel(ylabel)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # get halo and subhalo IDs
        if subhaloIDs is not None:
            subhaloID = subhaloIDs[i]
            haloID = sP.groupCatSingle(subhaloID=subhaloID)['SubhaloGrNr']
        else:
            haloID = haloIDs[i]
            subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

        # load
        load_haloID = haloID if scope == 'fof' else None
        load_subID = subhaloID if scope == 'subfind' else None
        if load_haloID is None and load_subID is None: assert scope == 'global'

        rad  = sP.snapshotSubset(ptType, 'rad_kpc', haloID=load_haloID, subhaloID=load_subID)
        vals = sP.snapshotSubset(ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)

        if sfreq0:
            # restrict to non eEOS cells
            sfr = sP.snapshotSubset(ptType, 'sfr', haloID=load_haloID, subhaloID=load_subID)
            w = np.where(sfr == 0.0)
            rad = rad[w]
            vals = vals[w]

        # radial bin
        rad_bins = np.linspace( xlim[0], xlim[1], nRadBins+1 )
        rr = rad_bins[:-1] + (rad_bins[1]-rad_bins[0])/2
        if xlog: rad = np.log10(rad)

        yy_mean = np.zeros( nRadBins, dtype='float32' )
        yy_median = np.zeros( nRadBins, dtype='float32' )
        yy_perc = np.zeros( (len(percs),nRadBins), dtype='float32' )

        for j in range(nRadBins):
            # calculate median radial profile and scatter
            w = np.where( (rad >= rad_bins[j]) & (rad < rad_bins[j+1]) )
            if len(w[0]) == 0:
                continue

            yy_mean[j] = np.nanmean( vals[w] )
            yy_median[j] = np.nanmedian( vals[w] )
            yy_perc[:,j] = np.nanpercentile( vals[w], percs )

        if ylog:
            yy_mean = logZeroNaN(yy_mean)
            yy_median = logZeroNaN(yy_median)
            yy_perc = logZeroNaN(yy_perc)

        if rr.size > sKn:
            yy_mean = savgol_filter(yy_mean,sKn,sKo)
            yy_median = savgol_filter(yy_median,sKn,sKo)
            yy_perc = savgol_filter(yy_perc,sKn,sKo,axis=1) # P[10,90]

        # plot lines
        if colorOffs is not None:
            for _ in range(colorOffs[i]):
                _ = next(ax._get_lines.prop_cycler)['color']

        label = '%s haloID=%d [%s]' % (sP.simName,haloID,scope) if not clean else sP.simName
        l, = ax.plot(rr, yy_median, '-', lw=lw, label=label)
        ax.plot(rr, yy_mean, '--', lw=lw, color=l.get_color())

        if len(sPs) <= 2:
            for j in range(int(yy_perc.shape[0]/2)):
                ax.fill_between(rr, yy_perc[0+j,:], yy_perc[-(j+1),:], color=l.get_color(), interpolate=True, alpha=0.15*(j+1))

    # special behavior
    if 'L11_12' in sPs[0].simName:
        rad_cgm_zoom = {'r$_{\\rm CGM,min}$':10,'r$_{\\rm CGM,max}$':300,'r$_{\\rm IGM}$':500} # pkpc
        ylim_p = [ylim[0] + (ylim[1]-ylim[0])/15, ylim[1] - (ylim[1]-ylim[0])/15]
        alpha = 1.0 if ptProperty == 'mass_msun' else 0.2
        for label,rad in rad_cgm_zoom.items():
            off = {10:0.0, 300:-0.04, 500:+0.04}[rad]
            if xlog: rad = np.log10(rad)
            ax.plot( [rad,rad], ylim_p, '-', lw=lw, color='black', alpha=0.1 )
            ax.text( rad+off, ylim_p[0], label, color='black', fontsize=20, alpha=alpha, verticalalignment='top', horizontalalignment='center')

    if ptProperty == 'cellsize_kpc':
        xlim_p = [xlim[0] + (xlim[1]-xlim[0])/40, xlim[1] - (xlim[1]-xlim[0])/40]
        notable_sizes = {'100pc':-1.0, '1kpc':0.0}
        for label,val in notable_sizes.items():
            ax.plot( xlim_p, [val,val], ':', lw=lw, color='black', alpha=0.05 )
            ax.text( xlim_p[0]+0.1, val+0.03, label, color='black', fontsize=20, alpha=0.1, verticalalignment='bottom', horizontalalignment='center' )

    # finish plot
    ax.legend(loc='best')

    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    if haloIDs is not None:
        hStr = 'haloID-%d' % haloIDs[0] if len(haloIDs) == 1 else 'nH-%d' % len(haloIDs)
    else:
        hStr = 'subhID-%d' % subhaloIDs[0] if len(subhaloIDs) == 1 else 'nSH-%d' % len(subhaloIDs)

    fig.savefig('radProfile_%s_%s_%s_%s_scope-%s.pdf' % (sPstr,ptType,ptProperty,hStr,scope))
    plt.close(fig)

def plot2DStackedRadialProfiles(sPs, ptType='gas', ptProperty='temp', ylim=[0.0, 2.0], 
                                cLog=True, clim=None, cNormQuant=None, smoothSigma=0.0, 
                                xQuant='mhalo_200_log', xlim=None, xbinsize=None, ctName='viridis'):
    """ 2D Stacked radial profile(s) of some quantity ptProperty of ptType vs. radius from (all) halo centers 
    (spatial_global based, using caching auxCat functionality, restricted to >1000 dm particle limit). 
    Note: Combination of {ptType,ptProperty} must already exist in auxCat mapping.
    xQuant and xlim specify x-axis (per subhalo) property, by default halo mass, binned by xbinsize.
    ylim specifies radial range, in linear rvir units.
    cLog specifies whether to log the color quantity, while clim (optionally) gives the colorbar bounds.
    If cNormQuant is not None, then normalize the profile values -per halo- by this subhalo quantity, 
    e.g. 'tvir' in the case of ptProperty=='temp'.
    If smoothSigma > 0 and cNormQuant is not None, use this smoothing to contour unity values. """

    # config
    scope = 'Global'

    acName = 'Subhalo_RadProfile3D_%s_%s_%s' % (scope,ptType.capitalize(), ptProperty.capitalize())

    # try to get automatic label/limits
    clabel, clim2, clog2 = simParticleQuantity(sPs[0], ptType, ptProperty, haloLims=True)
    if clim is None:
        clim = clim2

    # get x-axis and y-axis data/config from first sP
    xvals, xlabel, xlim2, xlog = sPs[0].simSubhaloQuantity(xQuant)

    if xlim is None: xlim = xlim2

    ac = sPs[0].auxCat(acName)

    # radial bins
    radBinEdges = ac[acName+'_attrs']['rad_bin_edges']
    radBinCen = (radBinEdges[1:] + radBinEdges[:-1]) / 2
    radBinInds = np.where( (radBinCen > ylim[0]) & (radBinCen <= ylim[1]) )[0]

    nRadBins = radBinInds.size
    radBinCen = radBinCen[radBinInds]

    # start figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if xbinsize is None:
        # automatically determine for ~square pixels
        bbox = ax.get_window_extent()
        nXBins = int(nRadBins*(bbox.height/bbox.width/0.6)) # hack: don't have axis labels/cbar yet...
        xbinsize = (xlim[1] - xlim[0]) / nXBins

    # sanity check that y-axis labels (radial range) will be close enough to binned profile values
    err_left = np.abs(radBinCen[radBinInds][0] - ylim[0])
    err_right = np.abs(radBinCen[radBinInds][-1] - ylim[1])
    err_cen = np.abs(radBinCen[radBinInds][int(nRadBins/2)] - (ylim[1]-ylim[0])/2)

    assert err_left < (ylim[1]-ylim[0])/20
    assert err_right < (ylim[1]-ylim[0])/20
    assert err_cen < (ylim[1]-ylim[0])/20 

    # x-axis bins, and allocate
    bin_edges = np.arange(xlim[0], xlim[1]+xbinsize, xbinsize)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    nbins = bin_centers.size

    binned_count = np.zeros( (nbins,nRadBins), dtype='int32' )
    binned_quant = np.zeros( (nbins,nRadBins), dtype='float32' )

    # loop over simulations
    for i, sP in enumerate(sPs):
        print(sP.simName)

        # load (except for first, which is already available above)
        if i > 0:
            xvals, xlabel, xlim2, xlog = sP.simSubhaloQuantity(xQuant)
        
            # load auxCat profiles
            ac = sP.auxCat(acName)

        # take subset for radial bins of interest, and restrict xvals/cnorm_vals to available subhalos
        yvals = ac[acName][:,radBinInds]
        subhaloIDs = ac['subhaloIDs']
        xvals = xvals[subhaloIDs]

        if cNormQuant is not None:
            cnorm_vals, cnorm_label, _, _ = sP.simSubhaloQuantity(cNormQuant)
            cnorm_vals = cnorm_vals[subhaloIDs]

        if i == 0:
            # try units verification
            unit1 = clabel.split("[")[1].split("]")[0].strip()
            unit2 = cnorm_label.split("[")[1].split("]")[0].strip()
            assert unit1 == unit2 # can generalize further

            # update colorbar label
            clabel = clabel.split("[")[0] + "/ " + cnorm_label.split("[")[0]
            if cLog: clabel += " [ log ]"

        # assign into bins
        for j in range(nbins):
            bin_start = bin_edges[j]
            bin_stop = bin_edges[j+1]

            w = np.where( (xvals > bin_start) & (xvals <= bin_stop) )[0]

            #print(bin_start, bin_stop, len(w), xvals[w].mean())

            if len(w) == 0:
                continue

            # save sum and counts per bin
            if cNormQuant is not None:
                yprof_loc = np.nansum( yvals[w,:] / cnorm_vals[w,np.newaxis], axis=0 )
            else:
                yprof_loc = np.nansum( yvals[w,:], axis=0 )

            count_loc = np.count_nonzero( np.isfinite( yvals[w,:] ), axis=0 )

            binned_quant[j,:] += yprof_loc
            binned_count[j,:] += count_loc

    # compute mean
    w_zero = np.where(binned_count == 0)
    assert np.sum(binned_quant[w_zero]) == 0

    w = np.where(binned_count > 0)
    binned_quant[w] /= binned_count[w]
    if cLog:
        binned_quant = logZeroNaN(binned_quant)

    binned_quant[w_zero] = np.nan # leave white

    # start plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Radius / R$_{\\rm vir}$')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)        

    _draw_special_lines(sPs[0], ax, ptProperty)

    # add image
    cmap = loadColorTable(ctName)
    norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
    im = plt.imshow(binned_quant.T, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
               cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

    # if we are taking a normalization, contour lines equal to unity in the color quantity
    if cNormQuant is not None:
        searchVal = 0.0 if cLog else 1.0
        XX, YY = np.meshgrid(bin_centers, radBinCen, indexing='ij')

        # smooth, ignoring NaNs
        binned_quant = gaussian_filter_nan(binned_quant, smoothSigma)

        c = plt.contour(XX, YY, binned_quant, [searchVal], colors='white', linestyles='solid', alpha=0.6)

    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.2)
    cb = fig.colorbar(im, cax=cax, label=clabel)

    # finish plot
    sPstr = '-'.join([sP.simName for sP in sPs])
    fig.savefig('radProfiles2DStack_%s_%d_%s_%s_vs_%s.pdf' % (sPstr,sPs[0].snap,ptType,ptProperty,xQuant))
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def compareRuns_PhaseDiagram():
    """ Driver. Compare a series of runs in a PDF booklet of phase diagrams. """
    import glob
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    redshift = 0.0
    yQuant = 'temp'
    xQuant = 'numdens'

    # get list of all 512 method runs via filesystem search
    sP = simParams(res=512,run='tng',redshift=redshift,variant='0000')
    dirs = glob.glob(sP.arepoPath + '../L25n512_*')
    variants = sorted([d.rsplit("_",1)[1] for d in dirs])
    variants = ['0000','1006']

    # start PDF, add one page per run
    pdf = PdfPages('compareRunsPhaseDiagram.pdf')

    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=redshift,variant=variant)
        if sP.simName == 'DM only': continue
        print(variant,sP.simName)
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, pdf=pdf)
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, meancolors=['coolrate_ratio'], weights=None, pdf=pdf)

    pdf.close()

def compareVariants_NO_OH_stellar():
    """ Driver. Igor. """
    import glob
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    partType = 'stars'
    yQuant = 'N_O_massratio'
    ylim   = [-3.5,0.0]
    xQuant = 'O_H_massratio'
    xlim   = [-5.5,-0.5]
    redshift = 0.0

    # variants
    sP = simParams(res=512,run='tng',redshift=redshift,variant='0000')
    dirs = glob.glob(sP.arepoPath + '../L25n512_*')
    variants = sorted([d.rsplit("_",1)[1] for d in dirs])

    # start PDF, add one page per run
    pdf = PdfPages('compareRuns_x=%s_y=%s_%s.pdf' % (xQuant,yQuant,partType))

    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=redshift,variant=variant)
        if sP.simName in ['L25n512_0020','L25n512_0030']: continue
        print(variant,sP.simName)
        plotPhaseSpace2D(sP, partType=partType, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, pdf=pdf)

    pdf.close()

def oneRun_PhaseDiagram(snaps=None):
    """ Driver. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    sP = simParams(res=1080,run='tng')
    yQuant = 'temp'
    xQuant = 'nh'

    zoom = False

    if zoom:
        xlim = [-6.0, -0.5]
        ylim = [2.0, 4.5]
    else:
        xlim = [-9.0, 2.0]
        ylim = [2.0, 8.5]
    clim   = [-6.0,-0.2]
    
    if snaps is None:
        #snaps = sP.validSnapList()[::2] # [99]
        snaps = [0,10,17,18,19,20,30,40,50,60,70,80,90,99]

    # start PDF, add one page per snapshot
    for snap in snaps:
        sP.setSnap(snap)
        print(snap)

        pdf = PdfPages('phaseDiagram_%s_%s%s_%d.pdf' % (yQuant,sP.simName,'_zoom' if zoom else '',snap))

        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, #meancolors=[cQuant], weights=weights, 
            xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, pdf=pdf)

        pdf.close()

def oneRun_tempcheck():
    """ Driver. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    sP = simParams(run='tng50-1')
    xQuant = 'nh'

    zoom = False

    xlim = [-9.0, 3.0]
    ylim = [1.0, 8.5]
    clim   = [-6.0,-0.2]

    snaps = sP.validSnapList()

    # start PDF, add one page for temp, one for old_temp
    for snap in snaps:
        sP.setSnap(snap)
        print(snap)

        pdf = PdfPages('phaseCheck_%s_%d.pdf' % (sP.simName,snap))

        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant='temp', 
            xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, pdf=pdf)
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant='temp_old', 
            xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, pdf=pdf)

        pdf.close()


def compareRuns_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two runs. """
    #from projects.oxygen import variantsMain as variants
    variants = ['0000','0010']

    sPs = []
    subhaloIDs = []

    for variant in variants:
        sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

        mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w = np.where( (mhalo > 11.5) & (mhalo < 12.5) )

        subhaloIDs.append( w[0] )

    for field in ['temp_linear']: #,'dens','temp_linear','P_gas_linear','z_solar']:
        plotStackedRadialProfiles1D(sPs, subhaloIDs=subhaloIDs, ptType='gas', ptProperty=field, weighting='O VI mass')

def compareHaloSets_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two different 
    types of halos. One run. """
    sPs = []
    sPs.append( simParams(res=1820,run='tng',redshift=2.0) )
    #sPs.append( simParams(res=1820,run='tng',redshift=2.0) )
    #sPs.append( simParams(res=1820,run='tng',redshift=2.0) )

    # select subhalos
    mhalo = sPs[0].groupCat(fieldsSubhalos=['mhalo_200_log'])

    if 0:
        gr,_,_,_ = simSubhaloQuantity(sPs[0], 'color_B_gr')

        with np.errstate(invalid='ignore'):
            w1 = np.where( (mhalo > 11.8) & (mhalo < 12.2) & (gr < 0.35) )
            w2 = np.where( (mhalo > 11.8) & (mhalo < 12.2) & (gr > 0.65) )

        print( len(w1[0]), len(w2[0]) )

        subhaloIDs = [{'11.8 < M$_{\\rm halo}$ < 12.2, (g-r) < 0.35':w1[0], 
                       '11.8 < M$_{\\rm halo}$ < 12.2, (g-r) > 0.65':w2[0]}]

    if 1:
        with np.errstate(invalid='ignore'):
            w0 = np.where( (mhalo > 11.3) & (mhalo < 13.4) )
            w1 = np.where( (mhalo > 11.9) & (mhalo < 12.1) )
            w2 = np.where( (mhalo > 12.2) & (mhalo < 12.4) )
            w3 = np.where( (mhalo > 12.5) & (mhalo < 12.7) )

        #subhaloIDs = [{'M$_{\\rm halo}$ = 12.0':w1[0]},
        #            {'M$_{\\rm halo}$ = 12.3':w2[0]},
        #            {'M$_{\\rm halo}$ = 12.6':w3[0]}]
        subhaloIDs = [{'M$_{\\rm halo}$ = 12.0':w1[0]}]
        #subhaloIDs = [{'M$_{\\rm halo}$ broad':w0[0]}]

    # select properties
    ptType = 'dm' # 'gas'
    #fields = ['tcool'] #['metaldens','dens','temp_linear','P_gas_linear','z_solar']
    fields = ['mass']
    weighting = None #'O VI mass'
    op = 'sum'
    plotIndiv = True

    #proj2D = [2, None] # z-axis, no depth restriction
    proj2D = [2, 10.0] # z-axis, 10 code units depth = 10 pkpc at z=2

    for field in fields:
        plotStackedRadialProfiles1D(sPs, subhaloIDs=subhaloIDs, ptType=ptType, ptProperty=field, op=op, 
                                    weighting=weighting, proj2D=proj2D, plotIndiv=plotIndiv)

def compareHaloSets_1DHists():
    """ Driver. Compare 1D histograms of a quantity, overplotting several halos. One run. """
    sPs = []

    sPs.append( simParams(res=910,run='tng',redshift=0.0) )
    mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w1 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
    subhaloIDs = [w1[0][0:5]]

    if 0:
        # add a second run
        sPs.append( simParams(res=455,run='tng',redshift=0.0) )
        mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w2 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
        subhaloIDs.append( w2[0][0:5] )

    for field in ['temp']: #['tcool','vrad']:
        plotHistogram1D(sPs, subhaloIDs=subhaloIDs, ptType='gas', ptProperty=field)

def singleHaloProperties():
    """ Driver. Several phase/radial profile plots for a single halo. """
    sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')

    partType = 'gas'
    xQuant = 'coolrate'
    yQuant = 'coolrate_ratio'

    # pick a MW
    #gc = sP.groupCat(fieldsHalos=['Group_M_Crit200','GroupPos'])
    #haloMasses = sP.units.codeMassToLogMsun(gc['Group_M_Crit200'])

    #haloIDs = np.where( (haloMasses > 12.02) & (haloMasses < 12.03) )[0]
    #haloID = haloIDs[6] # random: 3, 4, 5, 6

    haloID = None
    rMin = None
    rMax = None

    plotParticleMedianVsSecondQuant([sP], partType=partType, xQuant=xQuant, yQuant=yQuant, haloID=haloID, 
                                   radMinKpc=rMin, radMaxKpc=rMax)

    #for prop in ['hdens','temp_linear','cellsize_kpc','radvel','temp']:
    #    plotStackedRadialProfiles1D([sP], haloIDs=haloID, ptType='gas', ptProperty=prop)
    #    plotPhaseSpace2D(sP, partType='gas', xQuant='hdens', yQuant=prop, haloIDs=[haloID])

def compareRuns_particleQuant():
    """ Driver. Compare a series of runs in a single panel plot of a particle median quantity vs another. """

    # config
    yQuant = 'tcool'
    xQuant = 'dens_critb'
    ptType = 'gas'
    variants = ['0000'] #,'2102','2202','2302']

    # start PDF, add one page per run
    sPs = []
    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=0.0,variant=variant)
        if sP.simName == 'DM only': continue
        sPs.append(sP)

    plotParticleMedianVsSecondQuant(sPs, partType=ptType, xQuant=xQuant, yQuant=yQuant)

def coolingPhase():
    """ Driver. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    yQuant = 'temp'
    xQuant = 'numdens'
    cQuants = ['coolrate','heatrate','coolrate_powell','coolrate_ratio']
    xlim   = [-9.0, 2.0]
    ylim   = [2.0, 8.0]

    #sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')
    sP = simParams(res=1820,run='tng',redshift=0.0)

    # start PDF, add one page per run
    pdf = PdfPages('phaseDiagram_B_%s_%d.pdf' % (sP.simName,sP.snap))

    for cQuant in cQuants:
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, meancolors=[cQuant], xlim=xlim, ylim=ylim, 
                         weights=None, hideBelow=False, haloIDs=None, pdf=pdf)

    pdf.close()
